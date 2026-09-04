# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for Kimi-K3's wider Kimi-K2.5 vision QKV layout."""

from __future__ import annotations

from types import ModuleType

import torch
from torch import nn


def _make_norm(module: ModuleType, norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    raise NotImplementedError(f"Unsupported Kimi vision norm type: {norm_type}")


def install_kimi_k25_qkv_layout_compat(module: ModuleType) -> None:
    """Use ``qkv_hidden_size`` when constructing Kimi-K3's vision tower."""

    if getattr(module, "_hcu_kimi_k25_qkv_layout_patch_applied", False):
        return

    base_layer = module.MoonViTEncoderLayer
    base_encoder = module.MoonViT3dEncoder

    class HcuMoonViTEncoderLayer(base_layer):
        def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            quant_config=None,
            prefix: str = "",
            *,
            activation=module.F.gelu,
            attn_bias: bool = False,
            qkv_hidden_size: int | None = None,
            norm_type: str = "layernorm",
            mlp_type: str = "mlp2",
            linear_bias: bool = True,
        ) -> None:
            nn.Module.__init__(self)
            self.use_data_parallel = module.is_vit_use_data_parallel()
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.qkv_hidden_size = qkv_hidden_size or hidden_dim
            self.hidden_size_per_attention_head = self.qkv_hidden_size // num_heads
            self.tp_size = (
                1
                if self.use_data_parallel
                else module.get_tensor_model_parallel_world_size()
            )
            self.num_attention_heads_per_partition = module.divide(
                num_heads, self.tp_size
            )
            self.norm0 = _make_norm(module, norm_type, hidden_dim)
            self.norm1 = _make_norm(module, norm_type, hidden_dim)
            if mlp_type != "mlp2":
                raise NotImplementedError(f"Unsupported Kimi vision MLP type: {mlp_type}")
            self.mlp = module.MLP2(
                [hidden_dim, mlp_dim, hidden_dim],
                activation,
                bias=linear_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                use_data_parallel=self.use_data_parallel,
            )
            self.wqkv = module.QKVParallelLinear(
                hidden_size=hidden_dim,
                head_size=self.hidden_size_per_attention_head,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.wqkv",
                disable_tp=self.use_data_parallel,
            )
            self.wo = module.RowParallelLinear(
                self.qkv_hidden_size,
                hidden_dim,
                bias=attn_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.wo",
                disable_tp=self.use_data_parallel,
            )
            self.attn = module.MMEncoderAttention(
                num_heads=self.num_attention_heads_per_partition,
                head_size=self.hidden_size_per_attention_head,
                scale=self.hidden_size_per_attention_head**-0.5,
                prefix=f"{prefix}.attn",
            )

    class HcuMoonViT3dEncoder(base_encoder):
        def __init__(
            self,
            hidden_dim: int,
            num_layers: int,
            block_cfg: dict,
            video_attn_type: str = "spatial_temporal",
            quant_config=None,
            prefix: str = "",
        ) -> None:
            nn.Module.__init__(self)
            if video_attn_type != "spatial_temporal":
                raise AssertionError(
                    f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
                )
            self.video_attn_type = video_attn_type
            qkv_hidden_size = block_cfg.get("qkv_hidden_size")
            if qkv_hidden_size is None:
                # vLLM 0.25.1's K25 wrapper omits this K3-only field while
                # constructing block_cfg.  K3 keeps a 1536-wide QKV path
                # over a 1024-wide token embedding (12 heads, 128 per head).
                if block_cfg.get("hidden_dim") == 1024 and block_cfg.get("num_heads") == 12:
                    qkv_hidden_size = 1536
                else:
                    qkv_hidden_size = block_cfg["hidden_dim"]
            self.rope_2d = module.Rope2DPosEmbRepeated(
                qkv_hidden_size // block_cfg["num_heads"], 512, 512
            )
            self.blocks = nn.ModuleList(
                [
                    HcuMoonViTEncoderLayer(
                        **block_cfg,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{layer_idx}",
                    )
                    for layer_idx in range(num_layers)
                ]
            )
            self.final_layernorm = _make_norm(
                module, block_cfg.get("norm_type", "layernorm"), hidden_dim
            )

    module.MoonViTEncoderLayer = HcuMoonViTEncoderLayer
    module.MoonViT3dEncoder = HcuMoonViT3dEncoder
    module._hcu_kimi_k25_qkv_layout_patch_applied = True


__all__ = ["install_kimi_k25_qkv_layout_compat"]
