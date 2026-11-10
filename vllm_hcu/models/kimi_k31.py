# SPDX-License-Identifier: Apache-2.0
"""Text-only Kimi-K3.1 wrapper and checkpoint-backed DSA indexer modules."""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


class KimiK31IndexerBackend(DeepseekV32IndexerBackend):
    """Layer-compact indexer pages for HCU 64-token kernels."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        # gfx938's HCU preshuffled indexer kernels accept 64-token pages.
        return [64]

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def writes_kv_cache_from_block_table(cls):
        return True

    @staticmethod
    def get_builder_cls():
        return KimiK31IndexerMetadataBuilder


class KimiK31IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Derive indexer slots from its own 64-token block table."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slot_mapping_buffer = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from vllm.v1.attention.backends.mla.compressor_utils import (
            get_compressed_slot_mapping,
        )
        from dataclasses import replace

        slot_mapping = get_compressed_slot_mapping(
            common_attn_metadata.num_actual_tokens,
            common_attn_metadata.query_start_loc,
            common_attn_metadata.seq_lens,
            common_attn_metadata.block_table_tensor,
            self.kv_cache_spec.block_size,
            compress_ratio=1,
            out=self.slot_mapping_buffer,
        )
        metadata = super().build(
            common_prefix_len,
            replace(common_attn_metadata, slot_mapping=slot_mapping),
            fast_build,
        )
        # This HCU runtime's native FP8 writer emits token-major key pages.
        # Select its matching reader only for this model's uint8 indexer cache.
        if self.kv_cache_spec.dtype == torch.uint8:
            metadata.hcu_indexer_kcache_layout = "normal"
        return metadata


class KimiK31IndexerCache(torch.nn.Module, AttentionLayerBase):
    """Independent 64-token indexer pages with allocator-owned page padding."""

    def __init__(self, head_dim, dtype, prefix, cache_config):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache):
        # Preserve the padded block stride while exposing explicit kernel pages.
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=self.head_dim,
            head_size_v=0,
            dtype=self.dtype,
        )

    def forward(self):
        ...

    def get_attn_backend(self):
        return KimiK31IndexerBackend


class KimiK31DSAIndexerWeights(nn.Module):
    """Checkpoint-backed indexer projections and optional sparse DSA runtime."""

    def __init__(
        self,
        config,
        prefix: str,
        vllm_config: VllmConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        dsa = config.dsa_config
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank or config.hidden_size,
            dsa.index_num_heads * dsa.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wq_b",
            disable_tp=True,
        )
        self.wk = ReplicatedLinear(
            config.hidden_size,
            dsa.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk",
            disable_tp=True,
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            dsa.index_num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            disable_tp=True,
        )
        if dsa.index_k_norm_type == "layernorm":
            self.k_norm = KimiK31DSAIndexerKNorm(
                dsa.index_head_dim, eps=config.rms_norm_eps
            )
        elif dsa.index_k_norm_type == "rmsnorm":
            from vllm.model_executor.layers.layernorm import RMSNorm

            self.k_norm = RMSNorm(dsa.index_head_dim, eps=config.rms_norm_eps)
        else:
            raise ValueError(
                "unsupported Kimi-K3.1 DSA index_k_norm_type: "
                f"{dsa.index_k_norm_type!r}"
            )

        self.topk_tokens = dsa.index_topk
        self.vllm_config = vllm_config
        self.topk_indices_buffer = topk_indices_buffer
        self.indexer_op = None
        self.k_cache = None
        self.max_model_len = 0
        self.max_total_seq_len = 0
        self.index_head_dim = dsa.index_head_dim
        self.index_num_heads = dsa.index_num_heads
        self.quant_block_size = dsa.index_head_dim
        self.scale_fmt = "ue8m0"
        if vllm_config is not None:
            if topk_indices_buffer is None:
                raise ValueError("sparse Kimi-K3.1 indexer needs shared top-k buffer")
            self._init_sparse_runtime(prefix, vllm_config, topk_indices_buffer)

    def _init_sparse_runtime(self, prefix, vllm_config, topk_indices_buffer):
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
        from vllm_hcu.model_executor.layers.sparse_attn_indexer import (
            V32SparseAttnIndexer,
        )
        from vllm_hcu.platforms.hcu import on_gfx938

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.k_cache = KimiK31IndexerCache(
            head_dim=self.index_head_dim + (4 if on_gfx938() else 0),
            dtype=torch.uint8 if on_gfx938() else torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=vllm_config.cache_config,
        )
        self.indexer_op = V32SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.index_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            topk_indices_buffer,
        )

    def forward(self, hidden_states, index_q_input, positions=None, rotary_emb=None):
        if self.indexer_op is None:
            raise RuntimeError("Kimi-K3.1 sparse indexer was not initialized")
        if index_q_input is None:
            index_q_input = hidden_states
        index_q = self.wq_b(index_q_input)[0].view(
            -1, self.index_num_heads, self.index_head_dim
        )
        index_k = self.k_norm(self.wk(hidden_states)[0])
        index_weights = self.weights_proj(hidden_states)[0]
        q_flat = index_q.reshape(-1, self.index_head_dim).contiguous()
        q_fp8, q_scale = per_token_group_quant_fp8(
            q_flat,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=True,
        )
        q_fp8 = q_fp8.view(index_q.shape)
        q_scale = q_scale.view(-1, self.index_num_heads)
        weights = torch.empty(
            (index_q.shape[0], self.index_num_heads),
            dtype=torch.float32,
            device=index_q.device,
        )
        from triton_tmp_dir.kimi_k31_indexer import scale_indexer_weights

        scale_indexer_weights(
            index_weights,
            q_scale,
            weights,
            self.index_num_heads**-0.5,
            self.index_head_dim**-0.5,
        )
        return self.indexer_op.forward_hip(hidden_states, q_fp8, index_k, weights)


class KimiK31DSAIndexerKNorm(nn.Module):
    """FP32 affine LayerNorm parameters used by Kimi-K3.1 DSA checkpoints."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("weight_bf16", None, persistent=False)
        self.register_buffer("bias_bf16", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.weight_bf16, self.bias_bf16
        if weight is None or weight.dtype != x.dtype:
            weight = self.weight.to(x.dtype)
            bias = self.bias.to(x.dtype)
            if not torch.compiler.is_compiling():
                self.weight_bf16, self.bias_bf16 = weight, bias
        return torch.nn.functional.layer_norm(
            x, self.normalized_shape, weight, bias, self.eps
        )


class KimiK31ForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsQuant,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
):
    """Text-only K3.1 model wrapper around installed AMD KimiLinear modules.

    The existing KimiLinear decoder supplies embeddings, MoE, KDA, MLA,
    residual blocks, PP partitioning, recurrent state, and weight loading.
    DSA indexer parameters are attached to local DSA layers by their global
    layer index. Contexts no longer than index_topk use the equivalent causal
    dense NoPE MLA path; longer contexts wire the checkpoint-backed indexer to
    the existing HCU sparse-MLA and sparse-indexer implementations.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.text_config
        # vLLM sparse-MLA metadata builders read index_topk from the root HF config.
        self.config.index_topk = self.text_config.dsa_config.index_topk
        self.hidden_size = self.text_config.hidden_size
        self.quant_config = vllm_config.quant_config
        self.dsa_layer_indices = tuple(self.text_config.dsa_config.dsa_layers)
        self._uses_sparse_dsa = (
            vllm_config.model_config.max_model_len
            > self.text_config.dsa_config.index_topk
        )
        self.topk_indices_buffer = None
        if self._uses_sparse_dsa:
            self._validate_sparse_dsa_config()
            rows = vllm_config.scheduler_config.max_num_batched_tokens
            self.topk_indices_buffer = torch.empty(
                (rows, self.text_config.dsa_config.index_topk),
                dtype=torch.int32,
                device="cuda",
            )

        language_prefix = maybe_prefix(prefix, "language_model")
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.text_config,
            prefix=language_prefix,
            architectures=["KimiLinearForCausalLM"],
        )
        self._attach_dsa_indexer_weights(language_prefix)
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _validate_sparse_dsa_config(self) -> None:
        dsa = self.text_config.dsa_config
        rope = getattr(self.text_config, "rope_parameters", None) or {}
        rope_type = rope.get("rope_type", "default")
        supported = (
            self.text_config.mla_use_nope is True
            and dsa.index_topk_freq == 1
            and dsa.index_skip_topk_offset == 1
            and dsa.index_rope_dim == 0
            and rope_type in (None, "default")
        )
        if not supported:
            raise NotImplementedError(
                "Kimi-K3.1 sparse DSA currently requires NoPE, "
                "index_rope_dim=0, index_topk_freq=1, "
                "index_skip_topk_offset=1 and default rope scaling"
            )

    def _replace_with_sparse_dsa(self, attention, indexer) -> None:
        from vllm.model_executor.layers.mla import MLAModules

        old = attention.mla_attn
        old_mla = old.mla_attn
        static_context = self.vllm_config.compilation_config.static_forward_context
        registered = static_context.pop(old_mla.layer_name, None)
        if registered is not old_mla:
            raise RuntimeError(
                f"could not replace dense MLA registration {old_mla.layer_name!r}"
            )

        modules = MLAModules(
            kv_a_layernorm=old.kv_a_layernorm,
            kv_b_proj=old.kv_b_proj,
            rotary_emb=old.rotary_emb,
            o_proj=old.o_proj,
            fused_qkv_a_proj=old.fused_qkv_a_proj,
            kv_a_proj_with_mqa=old.kv_a_proj_with_mqa,
            q_a_layernorm=old.q_a_layernorm,
            q_b_proj=old.q_b_proj,
            q_proj=old.q_proj,
            indexer=indexer,
            indexer_rotary_emb=old.rotary_emb,
            is_sparse=True,
            topk_indices_buffer=self.topk_indices_buffer,
            g_proj=old.g_proj,
        )
        sparse = type(old)(
            old.hidden_size,
            old.num_heads,
            old_mla.scale,
            old.qk_nope_head_dim,
            old.qk_rope_head_dim,
            old.v_head_dim,
            old.q_lora_rank,
            old.kv_lora_rank,
            modules,
            self.vllm_config.cache_config,
            self.quant_config,
            old.prefix,
        )
        if getattr(old, "_hcu_fused_output_gate", False):
            sparse._hcu_fused_output_gate = True
        attention.mla_attn = sparse

    def _attach_dsa_indexer_weights(self, language_prefix: str) -> None:
        model = self.language_model.model
        model_prefix = maybe_prefix(language_prefix, "model")
        for layer in model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            layer_idx = getattr(layer, "layer_idx", None)
            if layer_idx is None or not self.text_config.is_dsa_layer(layer_idx):
                continue
            attention = layer.self_attn
            if hasattr(attention, "indexer"):
                raise RuntimeError(
                    f"DSA indexer already exists for layer {layer_idx}"
                )
            indexer_prefix = (
                f"{model_prefix}.layers.{layer_idx}.self_attn.indexer"
            )
            indexer = KimiK31DSAIndexerWeights(
                self.text_config,
                prefix=indexer_prefix,
                vllm_config=self.vllm_config if self._uses_sparse_dsa else None,
                topk_indices_buffer=self.topk_indices_buffer,
            )
            attention.indexer = indexer
            if self._uses_sparse_dsa:
                self._replace_with_sparse_dsa(attention, indexer)

    def _short_context_dense_dsa_supported(self) -> bool:
        dsa = self.text_config.dsa_config
        if not dsa or not self.dsa_layer_indices:
            return True
        model_config = self.vllm_config.model_config
        max_model_len = model_config.max_model_len
        rope = getattr(self.text_config, "rope_parameters", None) or {}
        rope_type = rope.get("rope_type", "default")
        return self._uses_sparse_dsa or (
            self.text_config.mla_use_nope is True
            and dsa.index_topk_freq == 1
            and dsa.index_skip_topk_offset == 1
            and dsa.index_rope_dim == 0
            and rope_type in (None, "default")
            and max_model_len <= dsa.index_topk
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if not self._short_context_dense_dsa_supported():
            dsa = self.text_config.dsa_config
            raise NotImplementedError(
                "Kimi-K3.1 DSA dense fallback is restricted to NoPE models "
                "with index_rope_dim=0, index_topk_freq=1, "
                "index_skip_topk_offset=1, default rope scaling, and "
                "max_model_len <= index_topk; got "
                f"max_model_len={self.vllm_config.model_config.max_model_len}, "
                f"index_topk={dsa.index_topk}, "
                f"index_topk_freq={dsa.index_topk_freq}, "
                f"index_skip_topk_offset={dsa.index_skip_topk_offset}, "
                f"index_rope_dim={dsa.index_rope_dim}, "
                f"mla_use_nope={self.text_config.mla_use_nope}"
            )
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs):
        return self.language_model.compute_logits(hidden_states, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor):
        return self.language_model.embed_input_ids(input_ids)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["vision_tower.", "mm_projector."],
        )
        return loader.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM

        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM

        text_config = vllm_config.model_config.hf_config.text_config
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.models.kimi_k3.amd.linear import KimiLinearForCausalLM

        return KimiLinearForCausalLM.get_mamba_state_copy_func()
