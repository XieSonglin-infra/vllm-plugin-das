import torch

from vllm_hcu.preload.audit import model_audit


def test_audit_detects_value_changes_and_preserves_alias_information():
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(12.0).reshape(3, 4)))
    model.register_buffer("alias", model.weight.detach()[:, :2])
    before = model_audit(model)
    assert before["weight"]["storage"] == before["alias"]["storage"]
    assert before["alias"]["stride"] == [4, 1]
    model.weight.data[0, 0] += 1
    after = model_audit(model)
    assert before["weight"]["sha256"] != after["weight"]["sha256"]
    assert before["alias"]["sha256"] != after["alias"]["sha256"]


def test_audit_accepts_bfloat16_scalar_and_empty_buffers():
    model = torch.nn.Module()
    model.register_buffer("scalar", torch.tensor(3.0, dtype=torch.bfloat16))
    model.register_buffer("empty", torch.empty(0))
    record = model_audit(model)
    assert record["scalar"]["shape"] == []
    assert record["scalar"]["dtype"] == "torch.bfloat16"
    assert record["scalar"]["sha256"] != record["empty"]["sha256"]


def test_audit_records_cross_role_shared_storage(tmp_path):
    import json
    from types import SimpleNamespace
    from vllm_hcu.preload.audit import write_model_audits

    target = torch.nn.Linear(3, 2)
    draft = torch.nn.Linear(3, 2)
    draft.weight = target.weight
    runner = SimpleNamespace(model=target, get_draft_model=lambda: draft)
    write_model_audits(runner, tmp_path, 0)
    aliases = json.loads((tmp_path / "dp0_shared_storages.json").read_text())
    assert ["draft:weight", "target:weight"] in aliases
    assert all("draft:bias" not in group for group in aliases)
