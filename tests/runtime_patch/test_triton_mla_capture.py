"""Disabled diagnostic capture must add no tensor work to MLA decode."""
from types import SimpleNamespace


def test_disabled_capture_does_not_concatenate_query(monkeypatch):
    from vllm_hcu.v1.attention.backends.mla import triton_mla

    monkeypatch.setattr(triton_mla, "_TRITON_CAPTURE_DIR", None)
    query = (object(), object())
    output = (object(), object())
    seen = []

    def forward(self, q, cache, metadata, layer):
        seen.append(q)
        return output

    def forbidden_cat(*args, **kwargs):
        raise AssertionError("disabled capture concatenated query")

    monkeypatch.setattr(triton_mla.TritonMLAImpl, "forward_mqa", forward)
    monkeypatch.setattr(triton_mla.torch, "cat", forbidden_cat)
    instance = object.__new__(triton_mla.HcuTritonMLAImpl)
    metadata = SimpleNamespace(decode=SimpleNamespace(block_table=None, seq_lens=None))
    assert instance.forward_mqa(query, None, metadata, None) == output
    assert seen == [query]
