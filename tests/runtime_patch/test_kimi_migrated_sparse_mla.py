# SPDX-License-Identifier: Apache-2.0
"""Expanded prefill must defer unsupported workloads to the existing runtime."""
from types import SimpleNamespace as NS
import pytest
import torch
from vllm_hcu.v1.attention.ops.kimi_migrated_sparse_mla import supports_migrated_prefill as supports_expanded_prefill

@pytest.mark.parametrize("case", ["supported", "default", "metadata_none", "multi_request", "context", "decode", "dcp", "lse", "fp32", "quantized_weight", "capture", "topk", "fp8", "index_dtype"])
def test_expanded_prefill_selection(monkeypatch, case):
    def tensor(shape,dtype=torch.bfloat16):return torch.empty(shape,device="meta",dtype=dtype)
    impl=NS(_kimi_sparse_backend="flashmla" if case=="default" else "migrated_hip",is_sparse=True,
        dcp_world_size=2 if case=="dcp" else 1,need_to_return_lse_for_decode=case=="lse",
        topk_indices_buffer=tensor((4096,1024 if case=="topk" else 2048),torch.int64 if case=="index_dtype" else torch.int32), kv_cache_dtype="fp8_ds_mla" if case=="fp8" else "auto")
    layer=NS(impl=impl,W_UK_T=tensor((12,128,512),torch.int8 if case=="quantized_weight" else torch.bfloat16),W_UV=tensor((12,512,128)))
    metadata=NS(num_reqs=2 if case=="multi_request" else 1,max_query_len=1 if case=="decode" else 4096,
        max_seq_len=8192 if case=="context" else 4096,num_actual_tokens=4096)
    if case=="metadata_none":metadata=None
    q=tensor((4096,12,192),torch.float32 if case=="fp32" else torch.bfloat16)
    c=tensor((4096,512));pe=tensor((4096,1,64));output=tensor((4096,1536))
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:case=="capture")
    assert supports_expanded_prefill(layer,q,c,pe,metadata,output)==(case=="supported")
