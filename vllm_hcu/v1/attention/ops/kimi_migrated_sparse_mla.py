# SPDX-License-Identifier: Apache-2.0
"""Opt-in gfx938 BF16 migration of AMD's query/all-head sparse MLA dataflow."""
from __future__ import annotations
import ctypes
import math
import os
from pathlib import Path
import torch
_LIBRARY=None
def load_library():
    global _LIBRARY
    if _LIBRARY is None:
        path=os.environ.get("VLLM_HCU_MIGRATED_MLA_LIBRARY")
        if not path:raise RuntimeError("Build csrc/kimi_migrated/build.py and set VLLM_HCU_MIGRATED_MLA_LIBRARY")
        handle=ctypes.CDLL(str(Path(path).resolve()))
        handle.migrated_sparse_mla.argtypes=([ctypes.c_void_p]*5+[ctypes.c_int]*6+
            [ctypes.POINTER(ctypes.c_int64),ctypes.c_float,ctypes.c_void_p])
        handle.migrated_sparse_mla.restype=ctypes.c_int
        handle.migrated_layout_probe.argtypes=[ctypes.c_void_p]*4
        handle.migrated_layout_probe.restype=ctypes.c_int
        handle.migrated_device_info.argtypes=[ctypes.POINTER(ctypes.c_int)]
        handle.migrated_device_info.restype=ctypes.c_int
        _LIBRARY=handle
    return _LIBRARY

def migrated_latent_attention(q,cache,indices,scale,*,topk_length=None,tile=32):
    """Q[N,H<=16,576], BF16 cache[slots,576] or [blocks,page,576].

    Indices are unique PHYSICAL/global cache slots: no slot<=query comparison.
    Caller/indexer owns causal selection. Invalid slots/holes are ignored;
    empty rows return zero. Positive strides and padded pages are supported
    without copying the cache. topk_length bounds consumed matrix columns.
    BF16 MMAC operands/P, FP32 dot/softmax/PV accumulators; output BF16[N,H,512].
    """
    if q.ndim!=3 or cache.ndim not in (2,3) or indices.ndim!=2:
        raise ValueError("unsupported migrated latent tensor ranks")
    n,h,d=q.shape
    if (n<1 or not 1<=h<=16 or d!=576 or cache.shape[-1]!=576
        or cache.numel()==0 or indices.shape[0]!=n or tile not in (16,32)
        or indices.dtype!=torch.int32 or not math.isfinite(float(scale))
        or not all(x.is_cuda and x.device==q.device for x in (q,cache,indices))
        or q.dtype!=torch.bfloat16 or cache.dtype!=q.dtype
        or any(s<=0 for x in (q,cache,indices) for s in x.stride())):
        raise ValueError("unsupported migrated latent contract")
    if topk_length is not None and (
        topk_length.shape!=(n,) or topk_length.dtype!=torch.int32
        or topk_length.device!=q.device or topk_length.stride(0)<=0):
        raise ValueError("topk_length must be int32[N] on the query device")
    if cache.ndim==2:
        rows=cache.shape[0];page=1;bs=cache.stride(0);ts=0;ds=cache.stride(1)
    else:
        rows=cache.shape[0]*cache.shape[1];page=cache.shape[1]
        bs,ts,ds=cache.stride()
    if rows>2**31-1 or indices.shape[1]>2**31-1:raise ValueError("int32 slot/count limit exceeded")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():raise ValueError("graph capture unsupported")
        stream=torch.cuda.current_stream(q.device)
        out=torch.empty((n,h,512),device=q.device,dtype=q.dtype)
        strides=(ctypes.c_int64*9)(*q.stride(),bs,ts,ds,*indices.stride(),
                                   topk_length.stride(0) if topk_length is not None else 0)
        status=load_library().migrated_sparse_mla(
            ctypes.c_void_p(q.data_ptr()),ctypes.c_void_p(cache.data_ptr()),
            ctypes.c_void_p(indices.data_ptr()),
            ctypes.c_void_p(topk_length.data_ptr() if topk_length is not None else 0),
            ctypes.c_void_p(out.data_ptr()),n,h,rows,indices.shape[1],page,tile,
            strides,float(scale),ctypes.c_void_p(stream.cuda_stream))
        if status:raise RuntimeError(f"migrated sparse MLA launch failed: {status}")
        for x in (q,cache,indices,topk_length):
            if x is not None:x.record_stream(stream)
        return out

def migrated_sparse_mla(q,c,pe,w_uk_t,w_uv,indices,scale,*,tile=32):
    """Fresh-prefill adapter; retains original BF16 Q absorption/V-up GEMMs."""
    n,h,_=q.shape
    with torch.profiler.record_function("hcu.kimi_sparse_mla.migrated_hip"):
        q_latent=torch.bmm(q[:,:,:128].transpose(0,1),w_uk_t).transpose(0,1)
        query=torch.cat((q_latent,q[:,:,128:]),dim=-1)
        # Only this fresh request's input rows; no checkpoint/IPC/cache rewrite.
        cache=torch.cat((c,pe.reshape(n,64)),dim=-1)
        result=migrated_latent_attention(query,cache,indices,scale,tile=tile)
        return torch.bmm(result.transpose(0,1),w_uv).transpose(0,1).contiguous().view(n,h*128)

def supports_migrated_prefill(layer,q,c,pe,metadata,output):
    from .kimi_expanded_mla import supports_expanded_prefill
    if q.is_cuda:
        with torch.cuda.device(q.device):
            if torch.cuda.is_current_stream_capturing():return False
    return (getattr(layer.impl,"kv_cache_dtype","auto") in {"auto","bfloat16"}
        and layer.impl.topk_indices_buffer is not None
        and layer.impl.topk_indices_buffer.dtype==torch.int32
        and supports_expanded_prefill(layer,q,c,pe,metadata,output,backend="migrated_hip"))
