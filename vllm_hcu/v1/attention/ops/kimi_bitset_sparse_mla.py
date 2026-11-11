# SPDX-License-Identifier: Apache-2.0
"""Experimental compressed-bitset CK Tile prefill; no dense mask/scores/P."""
import ctypes
import math
import os
import torch
_CK=None
def load_ck_library():
    global _CK
    if _CK is None:
        path=os.environ.get("VLLM_HCU_KIMI_CK_HIP_LIBRARY")
        if not path:raise RuntimeError("Set VLLM_HCU_KIMI_CK_HIP_LIBRARY to the isolated CK build")
        handle=ctypes.CDLL(path)
        handle.kimi_ck_pack.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*4+[ctypes.c_void_p]
        handle.kimi_ck_pack.restype=ctypes.c_int
        handle.kimi_ck_attention.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_void_p]
        handle.kimi_ck_attention.restype=ctypes.c_int
        _CK=handle
    return _CK
def build_sparse_bitset(indices,n):
    if indices.ndim!=2 or indices.shape[0]!=n or indices.dtype!=torch.int32 or not indices.is_cuda:
        raise ValueError("bitset requires int32[N,TopK] GPU logical indices")
    with torch.cuda.device(indices.device):
        if torch.cuda.is_current_stream_capturing():raise ValueError("capture unsupported")
        bits=torch.empty((n,(n+31)//32),device=indices.device,dtype=torch.int32)
        status=load_ck_library().kimi_ck_pack(ctypes.c_void_p(indices.data_ptr()),ctypes.c_void_p(bits.data_ptr()),
            n,indices.shape[1],*indices.stride(),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if status:raise RuntimeError(f"CK bitset launch failed: {status}")
        return bits
def bitset_sparse_attention(q,key,value,indices,scale,*,bits=None):
    n,h,d=q.shape
    if (n<1 or d!=192 or key.shape!=(h,n,192) or value.shape!=(h,n,128)
        or not math.isfinite(float(scale))
        or not all(x.is_cuda and x.device==q.device and x.dtype==torch.bfloat16 for x in (q,key,value))
        or indices.device!=q.device):
        raise ValueError("unsupported CK sparse attention contract")
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():raise ValueError("capture unsupported")
        # The standalone MMAC instantiation accepts contiguous per-head layouts.
        q,key,value=(x.contiguous() for x in (q,key,value))
        if bits is None:bits=build_sparse_bitset(indices,n)
        if bits.shape!=(n,(n+31)//32) or bits.dtype!=torch.int32 or bits.device!=q.device or not bits.is_contiguous():
            raise ValueError("invalid packed bitset")
        out=torch.empty((n,h,128),device=q.device,dtype=q.dtype)
        status=load_ck_library().kimi_ck_attention(
            *(ctypes.c_void_p(x.data_ptr()) for x in (q,key,value,bits,out)),
            n,h,float(scale),ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if status:raise RuntimeError(f"CK attention launch failed: {status}")
        return out.reshape(n,h*128)
def bitset_sparse_mla(q,c,pe,w_uk_t,w_uv,indices,scale):
    n,h,_=q.shape
    with torch.profiler.record_function("hcu.kimi_sparse_mla.bitset_ck"):
        latent=c.unsqueeze(0).expand(h,-1,-1)
        kn=torch.bmm(latent,w_uk_t.transpose(1,2));value=torch.bmm(latent,w_uv)
        key=torch.cat((kn,pe.reshape(1,n,64).expand(h,-1,-1)),dim=-1)
        return bitset_sparse_attention(q,key,value,indices,scale)
