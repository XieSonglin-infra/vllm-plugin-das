# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Validate target-owned Channel-FP8 MoE backend routing."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w8a8_fp8"
TARGETS = (
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.__init__",
    f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod.process_weights_after_loading",
)
_CLASS_MARKER = "_vllm_hcu_moe_w8a8_fp8_applied"
_WRAPPER_MARKER = "_vllm_hcu_moe_w8a8_fp8_wrapper"
_EXPLICIT_BACKENDS = {
    "aiter": "AITER",
    "deep_gemm": "HCU_DEEPGEMM",
    "triton": "TRITON",
}


def _selected_backend_name(method) -> str | None:
    backend = getattr(method, "fp8_backend", None)
    value = getattr(backend, "value", backend)
    return value if isinstance(value, str) else None


def _is_channel_token_route(fp8_moe_module, weight_quant, input_quant) -> bool:
    strategy = getattr(fp8_moe_module, "QuantizationStrategy", None)
    try:
        return bool(
            weight_quant.strategy == strategy.CHANNEL
            and input_quant.strategy == strategy.TOKEN
        )
    except AttributeError as exc:
        raise PatchCompatibilityError(
            "vLLM compressed-tensors FP8 MoE quantization strategy contract "
            "is unavailable"
        ) from exc


def _requested_selected_backend(moe) -> str | None:
    requested_backend = getattr(moe, "moe_backend", None)
    if requested_backend == "auto":
        return None
    selected_backend = _EXPLICIT_BACKENDS.get(requested_backend)
    if selected_backend is None:
        raise RuntimeError(
            "Channel-FP8 MoE backend must be auto, aiter, deep_gemm, or triton"
        )
    return selected_backend


def _restore_preprocessed_channel_fp8(method, layer, fp8_moe_module):
    """Restore only the bounded contiguous layout verified by the HCU probe."""
    import torch

    if (not _is_channel_token_route(fp8_moe_module, method.weight_quant,
                                    method.input_quant)
        or _selected_backend_name(method) != "HCU_DEEPGEMM"
        or method.static_input_scales):
        raise RuntimeError("unsupported preprocessed Channel-FP8 MoE scheme")
    parallel = getattr(method.moe, "moe_parallel_config", None)
    if parallel is None:
        raise RuntimeError("unsupported preprocessed Channel-FP8 MoE topology")
    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    common_sizes = (getattr(parallel, "tp_size", None),
                    getattr(parallel, "pcp_size", None),
                    getattr(parallel, "sp_size", None))
    single = (common_sizes == (1, 1, 1)
              and getattr(parallel, "dp_size", None) == 1
              and getattr(parallel, "ep_size", None) == 1
              and layer.local_num_experts == 1
              and getattr(w13, "_hcu_dsv4_deepgemm_layout", None) in (None, "contiguous"))
    shared_auto = (common_sizes == (1, 1, 1)
                   and getattr(parallel, "dp_size", None) == 8
                   and getattr(parallel, "ep_size", None) == 8
                   and getattr(parallel, "all2all_backend", None) == "deepep_auto"
                   and layer.local_num_experts == 32
                   and getattr(w13, "_hcu_dsv4_deepgemm_layout", None) == "contiguous=masked")
    if not (single or shared_auto):
        raise RuntimeError("unsupported preprocessed Channel-FP8 MoE topology or layout")
    count = layer.local_num_experts
    if (tuple(w13.shape) != (count, 64, 256, 4, 16, 16)
        or tuple(w2.shape) != (count, 32, 256, 4, 16, 16)
        or w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn
        or not w13.is_contiguous() or not w2.is_contiguous()
        or tuple(s13.shape) != (count, 4096, 1)
        or tuple(s2.shape) != (count, 4096, 1)
        or s13.dtype != torch.float32 or s2.dtype != torch.float32
        or not s13.is_contiguous() or not s2.is_contiguous()):
        raise RuntimeError("unsupported preprocessed HCU Channel-FP8 MoE layout")
    layer._dsv4_channel_deepgemm_layout = (
        "contiguous=masked" if shared_auto else "contiguous"
    )


def apply_to_module(module: ModuleType) -> bool:
    fp8_moe_module = load_exact_module(TARGET_MODULE, module)
    method_class = require_class(
        fp8_moe_module,
        "CompressedTensorsW8A8Fp8MoEMethod",
        f"{TARGET_MODULE}.CompressedTensorsW8A8Fp8MoEMethod",
    )
    wrapped = (
        (method_class, "__init__", TARGETS[0], _WRAPPER_MARKER),
        (
            method_class,
            "process_weights_after_loading",
            TARGETS[1],
            _WRAPPER_MARKER,
        ),
    )
    if already_applied(method_class, _CLASS_MARKER, wrapped):
        return False

    original_init = require_callable(method_class, "__init__", TARGETS[0])
    original_process = require_callable(
        method_class,
        "process_weights_after_loading",
        TARGETS[1],
    )
    require_exact_signature(
        original_init,
        TARGETS[0],
        positional=("self", "weight_quant", "input_quant", "moe", "layer_name"),
        defaults={"layer_name": None},
    )
    require_exact_signature(
        original_process,
        TARGETS[1],
        positional=("self", "layer"),
        defaults={},
    )

    @functools.wraps(original_init)
    def hcu_init(self, weight_quant, input_quant, moe, layer_name=None):
        channel_token = _is_channel_token_route(
            fp8_moe_module, weight_quant, input_quant
        )
        requested_backend = None
        if channel_token:
            requested_backend = _requested_selected_backend(moe)
        original_init(self, weight_quant, input_quant, moe, layer_name)
        if channel_token:
            if (_selected_backend_name(self) == "HCU_DEEPGEMM"
                and not getattr(self, "static_input_scales", True)):
                self.supports_pre_processed_weights = True
            selected_backend = _selected_backend_name(self)
            if selected_backend is None:
                raise RuntimeError(
                    "vLLM v0.25.1 did not select a Channel-FP8 MoE backend"
                )
            if (
                requested_backend is not None
                and selected_backend != requested_backend
            ):
                raise RuntimeError(
                    "vLLM v0.25.1 selected a Channel-FP8 MoE backend "
                    "different from the explicit request "
                    f"(expected={requested_backend!r}, "
                    f"selected={selected_backend!r})"
                )

    @functools.wraps(original_process)
    def hcu_process_weights_after_loading(self, layer):
        from vllm.model_executor.utils import is_weights_pre_processed
        if is_weights_pre_processed():
            _restore_preprocessed_channel_fp8(self, layer, fp8_moe_module)
        original_process(self, layer)
        selected_backend = _selected_backend_name(self)
        if selected_backend == "AITER":
            from vllm_hcu.model_executor.layers.quantization import (
                compressed_tensors_moe_runtime as hcu_runtime,
            )

            quant_config = getattr(self, "moe_quant_config", None)
            if quant_config is None:
                raise RuntimeError(
                    "AITER FP8 MoE did not initialize its quantization config"
                )
            hcu_runtime.prewarm_aiter_quantized_moe(
                layer,
                self.moe,
                quant_config,
            )
            return
        if selected_backend != "HCU_DEEPGEMM":
            return
        moe_kernel = getattr(self, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        experts = getattr(fused_experts, "experts", fused_experts)
        process = getattr(experts, "process_weights_after_loading", None)
        if not callable(process):
            raise RuntimeError(
                "HCU_DEEPGEMM FP8 backend did not construct modular experts "
                "before weight postprocessing"
            )
        process(layer)

    setattr(hcu_init, _WRAPPER_MARKER, True)
    setattr(hcu_process_weights_after_loading, _WRAPPER_MARKER, True)
    setattr(method_class, "_vllm_hcu_original_init", original_init)
    setattr(
        method_class,
        "_vllm_hcu_original_process_weights_after_loading",
        original_process,
    )
    setattr(method_class, "__init__", hcu_init)
    setattr(
        method_class,
        "process_weights_after_loading",
        hcu_process_weights_after_loading,
    )
    setattr(method_class, _CLASS_MARKER, True)
    setattr(method_class, "_vllm_hcu_fp8_moe_owner", "target-owned")
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
]
