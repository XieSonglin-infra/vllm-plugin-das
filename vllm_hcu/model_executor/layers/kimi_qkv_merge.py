# SPDX-License-Identifier: Apache-2.0
"""Inference-only derived QKV cache; canonical IPC parameters stay untouched."""
from __future__ import annotations

import torch


def _signature(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = None  # Frozen inference tensors have no version counter.
    return (id(tensor), tensor.data_ptr(), version, tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor.device)


def merged_qkv(owner, projections, x, quantized):
    """Return one packed output, or None for a configuration we cannot audit."""
    if (torch.is_grad_enabled() or x.device.type != "cuda"
            or x.dtype not in (torch.bfloat16, torch.float16)
            or x.ndim < 2 or x.numel() == 0
            or torch.cuda.is_current_stream_capturing()):
        return None
    weights, scales, dtypes = [], [], []
    for layer in projections:
        scheme = getattr(layer, "scheme", None)
        if not getattr(type(scheme), "_vllm_hcu_w8a8_int8_applied", False):
            return None
        if (getattr(layer, "bias", None) is not None
                or getattr(layer, "gather_output", False)):
            return None
        weight, scale = getattr(layer, "weight", None), getattr(layer, "weight_scale", None)
        dtype = getattr(layer, "params_dtype", None)
        if (not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor)
                or weight.ndim != 2 or weight.dtype != torch.int8
                or weight.device != x.device or scale.device != x.device
                or weight.shape[1] != x.shape[-1] or not weight.is_contiguous()
                or scale.dtype != torch.float32
                or tuple(scale.shape) not in ((weight.shape[0],), (weight.shape[0], 1))
                or dtype != x.dtype):
            return None
        weights.append(weight)
        scales.append(scale)
        dtypes.append(dtype)
    if (len(weights) != 3 or len({tuple(w.shape) for w in weights}) != 1
            or len({tuple(s.shape) for s in scales}) != 1
            or weights[0].shape[0] % 16 or weights[0].shape[1] % 64):
        return None
    key = tuple(_signature(t) for t in (*weights, *scales))
    cache = getattr(owner, "_hcu_merged_qkv_cache", None)
    if cache is None or cache[0] != key:
        # Ordinary attributes, never Parameters/buffers: no new IPC/state_dict keys.
        packed_weight = torch.cat(weights, dim=0)
        packed_scale = torch.cat(scales, dim=0)
        cache = (key, packed_weight, packed_scale)
        owner._hcu_merged_qkv_cache = cache
        owner._hcu_qkv_cache_bytes = sum(t.numel() * t.element_size() for t in cache[1:])
        owner._hcu_qkv_cache_builds = getattr(owner, "_hcu_qkv_cache_builds", 0) + 1
    from vllm_hcu.model_executor.layers.quantization.int8_runtime import apply_int8_linear

    with torch.profiler.record_function("hcu.kimi_kda.merged_qkv"):
        return apply_int8_linear(x, cache[1], cache[2], dtypes[0],
                                 x_and_scale_quanted=quantized)
