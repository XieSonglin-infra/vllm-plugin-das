# SPDX-License-Identifier: Apache-2.0
"""Opt-in Kimi sparse prefill in the unabsorbed 192/128-dimensional space."""
from __future__ import annotations
import torch
from vllm_hcu.v1.attention.ops.kimi_sparse_mla import _masked_softmax_kernel


def supports_expanded_prefill(layer, q, c, pe, metadata, output, *, backend="expanded_masked"):
    impl = layer.impl
    return (
        getattr(impl, "_kimi_sparse_backend", "flashmla") == backend
        and metadata is not None and impl.is_sparse
        and impl.dcp_world_size == 1 and not impl.need_to_return_lse_for_decode
        and metadata.num_reqs == 1 and metadata.max_query_len == 4096
        and metadata.max_seq_len == 4096 and metadata.num_actual_tokens == 4096
        and q.shape == (4096, 12, 192) and c.shape == (4096, 512)
        and pe.numel() == 4096 * 64
        and q.dtype == c.dtype == pe.dtype == output.dtype == torch.bfloat16
        and layer.W_UK_T is not None and layer.W_UV is not None
        and layer.W_UK_T.shape == (12, 128, 512)
        and layer.W_UV.shape == (12, 512, 128)
        and layer.W_UK_T.dtype == layer.W_UV.dtype == q.dtype
        and layer.impl.topk_indices_buffer is not None
        and layer.impl.topk_indices_buffer.shape[-1] == 2048
        and not torch.cuda.is_current_stream_capturing()
    )


def expanded_sparse_mla(q, c, pe, w_uk_t, w_uv, indices, scale):
    """Include both KV projections, original top-k mask, QK/softmax/PV.

    No KV-cache layout/weight change. Inputs cover a single fresh request.
    Unique logical indices come directly from the unchanged Kimi indexer.
    """
    n, h, _ = q.shape
    if (n, h, q.shape[-1]) != (4096, 12, 192):
        raise ValueError("expanded MLA currently validates only 4096-token TP8 prefill")
    with torch.profiler.record_function("hcu.kimi_sparse_mla.expanded_masked"):
        latent = c.unsqueeze(0).expand(h, -1, -1)
        k_nope = torch.bmm(latent, w_uk_t.transpose(1, 2))
        value = torch.bmm(latent, w_uv)
        key = torch.cat((k_nope, pe.reshape(1, n, 64).expand(h, -1, -1)), dim=-1)
        query = q.transpose(0, 1).contiguous()
        mask = torch.zeros((n, n + 1), device=q.device, dtype=torch.bool)
        locations = torch.where((indices >= 0) & (indices < n), indices, n).long()
        mask.scatter_(1, locations, True)
        scores = torch.bmm(query, key.transpose(1, 2), out_dtype=torch.float32)
        probabilities = torch.empty_like(scores, dtype=q.dtype)
        _masked_softmax_kernel[(n * h,)](
            scores, mask, probabilities, N=n, MASK_STRIDE=n + 1,
            SCALE=scale, BLOCK=n, num_warps=4,
        )
        output = torch.bmm(probabilities, value)
        return output.transpose(0, 1).contiguous().view(n, h * 128)
