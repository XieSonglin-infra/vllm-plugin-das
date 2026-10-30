# SPDX-License-Identifier: Apache-2.0
"""Explicit, read-only full-weight audit for disk versus IPC acceptance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def model_audit(model):
    """Hash logical tensor bytes; also record layouts and storage aliases.

    Copies one tensor at a time to CPU. This intentionally expensive check is
    only enabled by the acceptance operator, never by the ordinary load path.
    """
    records = {}
    storages = {}
    for name, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        if tensor.is_meta:
            raise RuntimeError(f"cannot audit meta tensor {name}")
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr())
        storage_index = storages.setdefault(identity, len(storages))
        raw = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
        records[name] = {
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "sha256": hashlib.sha256(raw).hexdigest(),
            "storage": storage_index, "storage_offset": tensor.storage_offset(),
        }
        del raw
    return records


def write_model_audits(model_runner, output_dir, dp_rank):
    models = {"target": model_runner.model}
    get_draft = getattr(model_runner, "get_draft_model", None)
    if callable(get_draft):
        draft = get_draft()
        if draft is not None:
            models["draft"] = draft
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for role, model in models.items():
        record = {"role": role, "dp_rank": dp_rank, "pid": os.getpid(),
                  "tensors": model_audit(model)}
        path = output / f"dp{dp_rank}_{role}.json"
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(record, sort_keys=True))
        temporary.replace(path)
    # Per-model storage IDs above cannot express ties between target/draft.
    shared = {}
    for role, model in models.items():
        for name, tensor in list(model.named_parameters(remove_duplicate=False)) + list(
            model.named_buffers(remove_duplicate=False)
        ):
            identity = (str(tensor.device), tensor.untyped_storage().data_ptr())
            shared.setdefault(identity, []).append(f"{role}:{name}")
    aliases = sorted(sorted(names) for names in shared.values()
                     if len({name.split(":", 1)[0] for name in names}) > 1)
    (output / f"dp{dp_rank}_shared_storages.json").write_text(json.dumps(aliases, indent=2))
