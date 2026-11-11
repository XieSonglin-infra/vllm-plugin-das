# SPDX-License-Identifier: Apache-2.0
"""BF16 Kimi sparse MLA using AITER TileLang without 64-head padding."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compact_sparse_indices_kernel(
    source, lengths, output, stride_row: tl.constexpr, stride_col: tl.constexpr,
    stride_length: tl.constexpr, KV_ROWS: tl.constexpr, TOPK: tl.constexpr,
    HAS_LENGTHS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    index = tl.load(source + row * stride_row + col * stride_col,
                    mask=col < TOPK, other=-1)
    valid = (col < TOPK) & (index >= 0) & (index < KV_ROWS)
    if HAS_LENGTHS:
        length = tl.load(lengths + row * stride_length)
        valid = valid & (col < length)
    positions = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    # Tail and compacted prefix are disjoint; no global store race/barrier.
    tl.store(output + row * TOPK + col, -1, mask=(col < TOPK) & (col >= count))
    tl.store(output + row * TOPK + positions, index, mask=valid)


def compact_sparse_indices(indices, kv_rows, topk_length=None):
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("sparse indices must be int32/int64 [tokens, topk]")
    rows, topk = indices.shape
    if kv_rows >= 2**31:
        raise ValueError("AITER sparse MLA requires int32-addressable KV rows")
    if topk_length is not None:
        if (topk_length.shape != (rows,) or topk_length.device != indices.device
                or topk_length.dtype not in (torch.int32, torch.int64)):
            raise ValueError("topk_length must be an integer vector on the index device")
    output = torch.empty((rows, 1, topk), device=indices.device, dtype=torch.int32)
    if rows:
        _compact_sparse_indices_kernel[(rows,)](
            indices, topk_length, output, indices.stride(0), indices.stride(1),
            0 if topk_length is None else topk_length.stride(0),
            KV_ROWS=kv_rows, TOPK=topk, HAS_LENGTHS=topk_length is not None,
            BLOCK=triton.next_power_of_2(topk), num_warps=4,
        )
    return output


def kimi_sparse_mla_aiter(q, kv, indices, sm_scale, topk_length=None):
    """The caller selects the validated BF16/12-head prefill path."""
    if (q.ndim != 3 or q.shape[1:] != (12, 576)
            or q.dtype != torch.bfloat16 or kv.dtype != q.dtype
            or kv.ndim != 3 or kv.shape[1:] != (1, 576)
            or q.device != kv.device or indices.device != q.device
            or not kv.is_contiguous()):
        raise ValueError("AITER Kimi sparse MLA requires BF16 Q[T,12,576], KV[S,1,576]")
    if indices.shape[0] != q.shape[0] or indices.shape[-1] != 2048:
        raise ValueError("AITER Kimi sparse MLA requires topk=2048 for each query")
    from aiter.ops.tilelang.sparse_mla_fwd import tilelang_sparse_fwd

    packed = compact_sparse_indices(indices, kv.shape[0], topk_length)
    with torch.profiler.record_function("hcu.kimi_sparse_mla.aiter_tilelang"):
        out = tilelang_sparse_fwd(q.contiguous(), kv, packed, sm_scale)
    return out.squeeze(0)


@triton.jit
def _masked_softmax_kernel(S, M, P, N: tl.constexpr, MASK_STRIDE: tl.constexpr,
                           SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = tl.load(M + (row % N) * MASK_STRIDE + col, mask=col < N, other=0)
    score = tl.load(S + row * N + col, mask=col < N, other=-float("inf")) * SCALE
    score = tl.where(valid, score, -float("inf"))
    maximum = tl.max(score, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp(score - maximum)
    denominator = tl.sum(weight, 0)
    probability = weight / tl.where(denominator > 0, denominator, 1.0)
    tl.store(P + row * N + col, probability, mask=col < N)


def kimi_sparse_mla_dense_masked(q, kv, indices, scale):
    """Exact indexer mask, dense GEMMs; one fresh 4096-token BF16 request.

    indices are unique logical positions from the Kimi top-k indexer.
    kv is gathered in logical token order, never the entire paged cache.
    FP32 QK accumulation/output and softmax; BF16 probabilities for PV.
    Peak temporary tensors are about 1.3 GiB per rank for this fixed shape.
    """
    if (q.shape != (4096, 12, 576) or kv.shape != (4096, 1, 576)
            or q.dtype != torch.bfloat16 or kv.dtype != q.dtype
            or indices.shape != (4096, 2048)):
        raise ValueError("dense masked MLA requires the validated 4096-token TP8 geometry")
    n, h, _ = q.shape
    # Last mask column is a discard slot for invalid indices, including holes.
    mask = torch.zeros((n, n + 1), device=q.device, dtype=torch.bool)
    locations = torch.where((indices >= 0) & (indices < n), indices, n).long()
    mask.scatter_(1, locations, True)
    with torch.profiler.record_function("hcu.kimi_sparse_mla.dense_masked"):
        query = q.transpose(0, 1).contiguous()
        key = kv[:, 0].T.unsqueeze(0).expand(h, -1, -1)
        value = kv[:, 0, :512].unsqueeze(0).expand(h, -1, -1)
        scores = torch.bmm(query, key, out_dtype=torch.float32)
        probabilities = torch.empty_like(scores, dtype=q.dtype)
        _masked_softmax_kernel[(n * h,)](
            scores, mask, probabilities, N=n, MASK_STRIDE=n + 1,
            SCALE=scale, BLOCK=n, num_warps=4,
        )
        output = torch.bmm(probabilities, value)
    return output.transpose(0, 1).contiguous()
