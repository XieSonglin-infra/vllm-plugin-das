# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 configuration that preserves its root model identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


KIMI_K31_W8A8_INCLUDE = [
    "language_model.model.layers.*.block_sparse_moe.routed_expert_down_proj",
    "language_model.model.layers.*.block_sparse_moe.routed_expert_up_proj",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj",
    "language_model.model.layers.*.self_attn.k_proj",
    "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa",
    "language_model.model.layers.*.self_attn.q_a_proj",
    "language_model.model.layers.*.self_attn.q_b_proj",
    "language_model.model.layers.*.self_attn.q_proj",
    "language_model.model.layers.*.self_attn.v_proj",
]


@dataclass
class KimiK31DSAConfig:
    """Nested DSA layer/indexer metadata from a Kimi-K3.1 checkpoint."""

    dsa_layers: list[int]
    index_topk: int = 2048
    index_head_dim: int = 64
    index_num_heads: int = 16
    index_rope_dim: int = 0
    index_k_norm_type: str = "rmsnorm"
    indexer_use_fp8: bool = False
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 1

    @classmethod
    def from_value(cls, value: dict[str, Any] | "KimiK31DSAConfig" | None):
        if value is None or isinstance(value, cls):
            return value
        return cls(**dict(value))


class KimiK31TextConfig(KimiLinearConfig):
    """KimiLinear config with typed DSA selectors and indexer dimensions."""

    def __init__(self, dsa_config=None, **kwargs):
        super().__init__(dsa_config=dsa_config, **kwargs)
        self.dsa_config = KimiK31DSAConfig.from_value(dsa_config)
        if self.dsa_config is not None:
            self.index_topk = self.dsa_config.index_topk

    def is_dsa_layer(self, layer_idx: int) -> bool:
        return (
            self.dsa_config is not None
            and (layer_idx + 1) in self.dsa_config.dsa_layers
        )

    def to_dict(self):
        ret = super().to_dict()
        if isinstance(ret.get("dsa_config"), KimiK31DSAConfig):
            ret["dsa_config"] = asdict(ret["dsa_config"])
        return ret


class KimiK31Config(KimiK3Config):
    """K3.1 root config with KimiLinear text config and DSA contract."""

    model_type = "kimi_k31"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensure_text_config_object()
        self._apply_k31_slimquant_policy()

    def _ensure_text_config_object(self):
        text_config = getattr(self, "text_config", None)
        if isinstance(text_config, dict):
            self.text_config = KimiK31TextConfig(**text_config)
        elif isinstance(text_config, KimiLinearConfig) and not isinstance(
            text_config, KimiK31TextConfig
        ):
            self.text_config = KimiK31TextConfig(**text_config.to_dict())
        return self.text_config


    @property
    def quantization_config(self):
        return self.__dict__.get("quantization_config")

    @quantization_config.setter
    def quantization_config(self, value):
        self.__dict__["quantization_config"] = value
        self._apply_k31_slimquant_policy()

    def _apply_k31_slimquant_policy(self):
        """Translate checkpoint mixed-precision metadata for the HCU backend."""
        raw = self.__dict__.get("quantization_config")
        if not isinstance(raw, dict) or raw.get("quant_method") != "slimquant_w4a8":
            return
        quant = deepcopy(raw)
        excluded = list(quant.get("modules_to_not_convert") or [])
        ignored = list(quant.get("ignore") or [])
        for module in excluded:
            pattern = f"*.{module}"
            if pattern not in ignored:
                ignored.append(pattern)
        quant["ignore"] = ignored
        mixed = dict(quant.get("mixed_precision") or {})
        if "w8a8_include" not in mixed:
            mixed["w8a8_include"] = {
                "format": "int8",
                "weight_bits": 8,
                "modules": list(KIMI_K31_W8A8_INCLUDE),
            }
        quant["mixed_precision"] = mixed
        self.__dict__["quantization_config"] = quant

    def update(self, config_dict):
        """Merge nested text overrides without dropping DSA layer metadata."""
        config_dict = dict(config_dict)
        text_overrides = config_dict.pop("text_config", None)
        super().update(config_dict)
        text_config = self._ensure_text_config_object()
        self._apply_k31_slimquant_policy()
        if text_overrides is not None:
            if not isinstance(text_overrides, dict):
                raise TypeError("Kimi-K3.1 text_config override must be a dictionary")
            for name, value in text_overrides.items():
                if name == "dsa_config":
                    text_config.dsa_config = KimiK31DSAConfig.from_value(value)
                else:
                    setattr(text_config, name, value)
            if text_config.dsa_config is not None:
                text_config.index_topk = text_config.dsa_config.index_topk
        return self
