# SPDX-License-Identifier: Apache-2.0
"""Register the Kimi-K3.1 root config without rewriting nested DSA metadata."""

from __future__ import annotations

from types import ModuleType

from vllm_hcu.runtime_compat.step5_patch_contract import (
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module

TARGET_MODULE = "vllm.transformers_utils.config"
PATCH_ID = "platform.core_fix.kimi_k31_config_registry"
TARGETS = ("vllm.transformers_utils.config._CONFIG_REGISTRY",)
_MARKER = "_vllm_hcu_kimi_k31_config_registry_applied"
SIGNATURES = ()
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    mapping = getattr(target, "_CONFIG_REGISTRY", None)
    if not isinstance(mapping, dict):
        raise TypeError("vLLM model config registry must be a dictionary")
    from vllm_hcu.transformers_utils.configs.kimi_k31 import KimiK31Config

    mapping["kimi_k31"] = KimiK31Config
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
