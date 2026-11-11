"""Provider selection contracts; GPU numerical checks live in task scripts."""
import pytest
import torch
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import HcuFlashMLASparseImpl
from vllm_hcu.v1.attention.ops import kimi_sparse_mla

@pytest.mark.parametrize("case", ["supported", "default", "lse", "dcp", "stride", "decode", "dtype", "capture"])
def test_provider_selection(monkeypatch, case):
    impl = object.__new__(HcuFlashMLASparseImpl)
    impl._kimi_sparse_backend = "flashmla" if case == "default" else "aiter_tilelang"
    impl.dcp_world_size = 2 if case == "dcp" else 1
    impl.need_to_return_lse_for_decode = case == "lse"
    impl.softmax_scale = 0.1
    q = torch.empty((1 if case == "decode" else 512, 12, 576), dtype=torch.float32 if case == "dtype" else torch.bfloat16)
    kv = torch.empty((32, 1, 1152 if case == "stride" else 576), dtype=q.dtype)[..., :576]
    indices = torch.empty((q.shape[0], 2048), dtype=torch.int32)
    lengths = torch.empty(q.shape[0], dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: case == "capture")
    monkeypatch.delenv("VLLM_HCU_KIMI_SPARSE_MLA_VALIDATE", raising=False)
    seen = []
    def native(self, *args):
        seen.append(("native", args))
        return "native", "lse"
    def aiter(*args):
        seen.append(("aiter", args))
        return "aiter"
    monkeypatch.setattr(FlashMLASparseImpl, "_bf16_flash_mla_kernel", native)
    monkeypatch.setattr(kimi_sparse_mla, "kimi_sparse_mla_aiter", aiter)
    result = impl._bf16_flash_mla_kernel(q, kv, indices, lengths)
    if case == "supported":
        assert result == ("aiter", None)
        assert seen[0][1][0].shape[1] == 12
        assert seen[0][1][-1] is lengths
        assert seen[0][1][1].data_ptr() == kv.data_ptr()
    else:
        assert result == ("native", "lse")
    assert len(seen) == 1


@pytest.mark.parametrize("case", ["supported", "default", "multi_request", "context", "chunk", "dcp", "lse", "capture"])
def test_dense_masked_selection(monkeypatch, case):
    from types import SimpleNamespace
    import vllm.v1.attention.backends.mla.flashmla_sparse as native_module
    impl = object.__new__(HcuFlashMLASparseImpl)
    impl._kimi_sparse_backend = "flashmla" if case == "default" else "dense_masked"
    impl.dcp_world_size = 2 if case == "dcp" else 1
    impl.need_to_return_lse_for_decode = case == "lse"
    impl.softmax_scale = .1
    q = torch.empty((4096, 64, 576), device="meta", dtype=torch.bfloat16)
    cache = torch.empty((4096, 576), device="meta", dtype=q.dtype)
    indices = torch.empty((4096, 2048), device="meta", dtype=torch.int32)
    metadata = SimpleNamespace(num_reqs=2 if case == "multi_request" else 1,
        max_query_len=2048 if case == "chunk" else 4096,
        max_seq_len=8192 if case == "context" else 4096,
        num_actual_tokens=4096, block_size=64,
        block_table=torch.empty((1,64),device="meta",dtype=torch.int32))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: case == "capture")
    monkeypatch.delenv("VLLM_HCU_KIMI_SPARSE_MLA_VALIDATE", raising=False)
    monkeypatch.setattr(native_module, "flat_kv_row_view", lambda *args: (cache,64))
    seen=[]
    def native(self,*args):
        seen.append("native")
        return "native","lse"
    def dense(q,kv,idx,scale):
        assert q.shape == (4096,12,576)
        assert kv.shape == (4096,1,576)
        assert idx is indices
        seen.append("dense")
        return "dense"
    monkeypatch.setattr(FlashMLASparseImpl,"_forward_bf16_kv",native)
    monkeypatch.setattr(kimi_sparse_mla,"kimi_sparse_mla_dense_masked",dense)
    result=impl._forward_bf16_kv(q,cache,indices,metadata,12)
    assert result == (("dense",None) if case == "supported" else ("native","lse"))
    assert len(seen)==1
