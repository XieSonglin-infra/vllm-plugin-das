from __future__ import annotations

import json
import hashlib

import torch

from vllm_hcu.models.kimi_k3.activation_trace import KimiActivationTrace


def test_kimi_activation_trace_filters_layers_and_writes_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_HCU_KIMI_TRACE_LAYERS", "0,2")
    monkeypatch.setenv("VLLM_HCU_KIMI_TRACE_MAX_COLS", "3")
    trace = KimiActivationTrace(str(tmp_path), prefix="language_model.model")

    trace.begin(torch.tensor([[11, 12]]), torch.tensor([0, 1]))
    trace.capture("embedding", torch.arange(8, dtype=torch.float32).reshape(2, 4))
    trace.capture("layer_0_output", torch.ones(2, 4), layer_idx=0)
    trace.capture("layer_1_output", torch.ones(2, 4), layer_idx=1)
    trace.capture("layer_2_output", torch.full((2, 4), 2.0), layer_idx=2)

    metadata = next(tmp_path.glob("*.jsonl"))
    records = [json.loads(line) for line in metadata.read_text().splitlines()]
    names = [record.get("name") for record in records if record["kind"] == "activation"]
    assert names == ["embedding", "layer_0_output", "layer_2_output"]
    snapshot = torch.load(
        tmp_path / f"{metadata.stem}.layer_0_output.layer0.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert tuple(snapshot.shape) == (1, 3)
    assert records[0]["kind"] == "input"


def test_kimi_activation_trace_filters_by_input_digest(tmp_path, monkeypatch):
    wanted = torch.tensor([11, 12])
    digest = hashlib.sha256(b"11,12").hexdigest()
    monkeypatch.setenv("VLLM_HCU_KIMI_TRACE_INPUT_SHA256", digest)
    trace = KimiActivationTrace(str(tmp_path))

    trace.begin(torch.tensor([1, 2]), torch.tensor([0, 1]))
    trace.capture("embedding", torch.ones(1, 2))
    assert not list(tmp_path.glob("*.jsonl"))

    trace.begin(wanted, torch.tensor([0, 1]))
    trace.capture("embedding", torch.ones(1, 2))
    metadata = next(tmp_path.glob("*.jsonl"))
    record = json.loads(metadata.read_text().splitlines()[0])
    assert record["input_sha256"] == digest
