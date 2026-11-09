# SPDX-License-Identifier: Apache-2.0
"""Text-only Kimi-K3.1 wrapper and checkpoint-backed DSA indexer modules."""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
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


class KimiK31DSAIndexerWeights(nn.Module):
    """Parameter-only indexer shell matching the converted K3.1 checkpoint."""

    def __init__(self, config, prefix: str):
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Kimi-K3.1 DSA indexer forward is not implemented on HCU yet"
        )


class KimiK31DSAIndexerKNorm(nn.Module):
    """FP32 affine LayerNorm parameters used by Kimi-K3.1 DSA checkpoints."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Kimi-K3.1 DSA indexer norm forward is not implemented on HCU yet"
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
    layer index. For the checkpoint's bounded short-context configuration,
    the existing causal dense NoPE MLA path is equivalent while every valid
    causal key fits inside index_topk. Longer contexts remain fail-closed.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.text_config
        self.hidden_size = self.text_config.hidden_size
        self.quant_config = vllm_config.quant_config
        self.dsa_layer_indices = tuple(self.text_config.dsa_config.dsa_layers)

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
            attention.indexer = KimiK31DSAIndexerWeights(
                self.text_config, prefix=indexer_prefix
            )

    def _short_context_dense_dsa_supported(self) -> bool:
        dsa = self.text_config.dsa_config
        if not dsa or not self.dsa_layer_indices:
            return True
        model_config = self.vllm_config.model_config
        max_model_len = model_config.max_model_len
        rope = getattr(self.text_config, "rope_parameters", None) or {}
        rope_type = rope.get("rope_type", "default")
        return (
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
