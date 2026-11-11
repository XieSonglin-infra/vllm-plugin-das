# SPDX-License-Identifier: Apache-2.0
"""Opt-in HIP sparse MLA. Build explicitly before launching model workers."""
from __future__ import annotations
import ctypes
import os
import math
from pathlib import Path
import torch

_LIB = None

def load_library():
    global _LIB
    if _LIB is None:
        path = os.environ.get("VLLM_HCU_KIMI_FUSED_HIP_LIBRARY")
        if not path:
            raise RuntimeError("Set VLLM_HCU_KIMI_FUSED_HIP_LIBRARY to the incrementally built .so")
        handle = ctypes.CDLL(str(Path(path).resolve()))
        handle.kimi_fused_sparse.argtypes = ([ctypes.c_void_p] * 6 +
            [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.POINTER(ctypes.c_long), ctypes.c_void_p])
        handle.kimi_fused_sparse.restype = ctypes.c_int
        _LIB = handle
    return _LIB

def fused_sparse_attention(q, key, value, indices, scale, *, splits=1, rowwise=False):
    """BF16 Q[N,H,192], K[H,N,192], V[H,N,128]; unique logical indices.

    Arbitrary positive strides are supported; invalid (<0, >=N), future keys
    and empty rows are ignored. Duplicate valid indices are outside the contract.
    All inputs must be finite. FP32 dot, online softmax and PV accumulators.
    """
    n,h,d=q.shape
    if rowwise:
        if indices.shape[-1]>2048:raise ValueError("rowwise topk exceeds 2048")
        splits=0
    if (n < 1 or h < 1 or not math.isfinite(float(scale)) or d != 192 or key.shape != (h,n,192) or value.shape != (h,n,128)
        or indices.ndim != 2 or indices.shape[0] != n or indices.shape[1] < 1
        or indices.dtype != torch.int32 or (splits < 1 and not rowwise) or splits > 32
        or not all(x.is_cuda and x.device == q.device for x in (q,key,value,indices))
        or not all(x.dtype == torch.bfloat16 for x in (q,key,value))
        or any(s <= 0 for x in (q,key,value,indices) for s in x.stride())
):
        raise ValueError("unsupported fused sparse HIP input contract")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("fused sparse HIP does not support graph capture")
        tmp=torch.empty((n,h,splits,130) if splits>1 else (0,),device=q.device,dtype=torch.float32)
        out=torch.empty((n,h,128),device=q.device,dtype=q.dtype)
        strides=(ctypes.c_long*11)(*(q.stride()+key.stride()+value.stride()+indices.stride()))
        status=load_library().kimi_fused_sparse(
            *(ctypes.c_void_p(x.data_ptr()) for x in (q,key,value,indices,tmp,out)),
            n,h,indices.shape[1],splits,float(scale),strides,
            ctypes.c_void_p(torch.cuda.current_stream(q.device).cuda_stream))
        if status:
            raise RuntimeError(f"kimi fused HIP launch failed: {status}")
        return out.reshape(n,h*128)

def fused_sparse_mla(q,c,pe,w_uk_t,w_uv,indices,scale):
    n,h,_=q.shape
    with torch.profiler.record_function("hcu.kimi_sparse_mla.fused_hip"):
        latent=c.unsqueeze(0).expand(h,-1,-1)
        k_nope=torch.bmm(latent,w_uk_t.transpose(1,2))
        value=torch.bmm(latent,w_uv)
        key=torch.cat((k_nope,pe.reshape(1,n,64).expand(h,-1,-1)),dim=-1)
        return fused_sparse_attention(q,key,value,indices,scale)

def supports_fused_prefill(layer,q,c,pe,metadata,output):
    from .kimi_expanded_mla import supports_expanded_prefill
    return (getattr(layer.impl, "kv_cache_dtype", "auto") in {"auto", "bfloat16"}
        and layer.impl.topk_indices_buffer is not None
        and layer.impl.topk_indices_buffer.dtype == torch.int32
        and supports_expanded_prefill(layer,q,c,pe,metadata,output,backend="fused_hip"))
