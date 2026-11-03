# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the daemon footprint manifest and engine budget fix.

Zero-copy IPC weights are allocated before the engine takes its startup memory
snapshot, so the engine must count the daemon-resident bytes itself. These tests
pin the manifest format, the per-role/dedup accounting and the fail-closed cases.
"""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.runtime_compat import weight_cache
from vllm_hcu.runtime_compat.weight_cache import (
    ipc_cache_resident_memory,
    write_cache_manifest,
)


@pytest.fixture(autouse=True)
def fake_device_usage(monkeypatch):
    # The manifest records device-wide usage; no GPU is available in CPU tests.
    monkeypatch.setattr(weight_cache, "_device_memory_usage", lambda: (4096, 8192))


class FakeDaemon:
    def __init__(self, socket_path, model, role="target", uuid="A"):
        self._path = socket_path
        self.model = model
        self.cache_role = None if role == "target" else role
        self._uuid = uuid
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=3)
        )
        self.cache_config = SimpleNamespace(checkpoint="hcu-config-v3:abc")

    def _socket_path(self):
        return self._path

    def _gpu_uuid(self):
        return self._uuid


def make_model(shared=True):
    model = torch.nn.Module()
    first = torch.nn.Parameter(torch.zeros(16))
    model.register_parameter("first", first)
    second_source = first.data[:8] if shared else torch.zeros(8)
    model.register_parameter("second", torch.nn.Parameter(second_source))
    model.register_buffer("buffer", torch.ones(4, dtype=torch.float32))
    return model


def socket_base(tmp_path, identifier="uuid-A"):
    from vllm.model_executor.model_loader.weight_cache import protocol

    return protocol.get_socket_path(identifier, str(tmp_path))


def test_manifest_records_unique_storage_footprint(tmp_path):
    model = make_model(shared=True)
    daemon = FakeDaemon(socket_base(tmp_path), model)
    path = write_cache_manifest(daemon)
    with open(path) as handle:
        record = json.load(handle)
    assert record["role"] == "target"
    assert record["dp_rank"] == 3
    assert record["gpu_uuid"] == "A"
    assert record["checkpoint_fingerprint"] == "hcu-config-v3:abc"
    # first and second share one storage; the buffer adds its own.
    expected = model.first.untyped_storage().nbytes() + model.buffer.untyped_storage().nbytes()
    assert record["model_footprint_bytes"] == expected
    assert len(record["storages"]) == 2
    assert record["device_used_bytes"] == 4096
    assert record["device_total_bytes"] == 8192
    assert os.stat(path).st_mode & 0o777 == 0o600


def _config(tmp_path, speculative=False, load_format="ipc_cache", mode=None):
    extra = {"socket_dir": str(tmp_path), "fallback": False}
    if mode is not None:
        extra["mode"] = mode
    return SimpleNamespace(
        load_config=SimpleNamespace(
            load_format=load_format, model_loader_extra_config=extra
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, pipeline_parallel_rank=0
        ),
        speculative_config=(
            SimpleNamespace(method="dspark") if speculative else None
        ),
    )


def test_resident_memory_counts_target_only(tmp_path):
    write_cache_manifest(FakeDaemon(socket_base(tmp_path), make_model()))
    assert ipc_cache_resident_memory(_config(tmp_path), gpu_uuid="A") == 4096


def test_resident_memory_takes_max_across_roles_of_one_device(tmp_path, monkeypatch):
    base = socket_base(tmp_path)
    write_cache_manifest(FakeDaemon(base, make_model()))
    # The draft role is served by the same rank process after the target;
    # device usage only grew, and must not be summed with the target's.
    monkeypatch.setattr(weight_cache, "_device_memory_usage", lambda: (5000, 8192))
    write_cache_manifest(FakeDaemon(base + ".draft", make_model(), role="draft"))
    assert ipc_cache_resident_memory(
        _config(tmp_path, speculative=True), gpu_uuid="A"
    ) == 5000


def test_missing_draft_manifest_fails_closed(tmp_path):
    write_cache_manifest(FakeDaemon(socket_base(tmp_path), make_model()))
    with pytest.raises(RuntimeError, match="manifest .* is missing"):
        ipc_cache_resident_memory(_config(tmp_path, speculative=True), gpu_uuid="A")


def test_wrong_uuid_or_role_fails_closed(tmp_path):
    # A manifest that exists at this device's socket path but names another
    # device must not be trusted.
    base = socket_base(tmp_path, identifier="uuid-B")
    write_cache_manifest(FakeDaemon(base, make_model(), uuid="A"))
    with pytest.raises(RuntimeError, match="belongs to device"):
        ipc_cache_resident_memory(_config(tmp_path), gpu_uuid="B")

    base = socket_base(tmp_path, identifier="uuid-C")
    write_cache_manifest(FakeDaemon(base, make_model(), role="draft", uuid="C"))
    with pytest.raises(RuntimeError, match="role"):
        ipc_cache_resident_memory(_config(tmp_path), gpu_uuid="C")


def test_dead_daemon_manifest_fails_closed(tmp_path):
    daemon = FakeDaemon(socket_base(tmp_path), make_model())
    path = write_cache_manifest(daemon)
    with open(path) as handle:
        record = json.load(handle)
    record["pid"] = 999999999
    with open(path, "w") as handle:
        json.dump(record, handle)
    with pytest.raises(RuntimeError, match="no live daemon"):
        ipc_cache_resident_memory(_config(tmp_path), gpu_uuid="A")


def test_reused_pid_manifest_fails_closed(tmp_path):
    path = write_cache_manifest(FakeDaemon(socket_base(tmp_path), make_model()))
    with open(path) as handle:
        record = json.load(handle)
    record["pid_start_time"] -= 1
    with open(path, "w") as handle:
        json.dump(record, handle)
    with pytest.raises(RuntimeError, match="no live daemon"):
        ipc_cache_resident_memory(_config(tmp_path), gpu_uuid="A")


def test_zombie_daemon_is_not_live(tmp_path):
    import subprocess

    child = subprocess.Popen(["true"])
    try:
        # waitid(WNOWAIT) observes exit but deliberately leaves a zombie.
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        with pytest.raises(ProcessLookupError):
            weight_cache._live_process_start_time(child.pid)
    finally:
        child.wait(timeout=5)


def test_budget_snapshot_keeps_the_adjustment():
    import dataclasses

    @dataclasses.dataclass
    class MeasuringSnapshot:
        # Mirrors MemorySnapshot: __post_init__ re-measures and would discard
        # any free_memory passed through dataclasses.replace.
        free_memory: int = 0
        auto_measure: bool = True

        def __post_init__(self):
            if self.auto_measure:
                self.free_memory = 111

    measured = MeasuringSnapshot()
    assert measured.free_memory == 111
    adjusted = weight_cache.budget_snapshot_with_resident(measured, 50)
    assert adjusted.free_memory == 161
    assert adjusted.auto_measure is False


def test_disk_loading_and_copy_mode_need_no_manifest(tmp_path):
    assert ipc_cache_resident_memory(
        _config(tmp_path, load_format="auto"), gpu_uuid="A"
    ) == 0
    assert ipc_cache_resident_memory(
        _config(tmp_path, mode="copy"), gpu_uuid="A"
    ) == 0


def test_release_all2all_buffers_destroys_each_manager(monkeypatch):
    import torch
    from vllm_hcu.runtime_compat import weight_cache

    calls = []
    fake_type = type(
        "DeepEPFakeAll2AllManager",
        (),
        {"__module__": "vllm.fake", "destroy": lambda self: calls.append(self)},
    )
    first = fake_type()
    second = fake_type()
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda *a, **k: None)
    released = weight_cache.release_all2all_buffers()
    assert released.count("DeepEPFakeAll2AllManager") >= 2
    assert first in calls and second in calls
    assert calls.count(first) == 1


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_pp_dspark_only_last_stage_requires_draft_manifest(tmp_path, monkeypatch, pp_rank):
    monkeypatch.setattr(weight_cache, "_pipeline_context", lambda: (2, pp_rank))
    base = socket_base(tmp_path, identifier=f"uuid-A-pp{pp_rank}")
    write_cache_manifest(FakeDaemon(base, make_model()))
    config = _config(tmp_path, speculative=True)
    # Real ParallelConfig has no pipeline_parallel_rank attribute: obtain it
    # from the initialized PP group rather than silently selecting stage 0.
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    if pp_rank == 1:
        with pytest.raises(RuntimeError, match="manifest .* is missing"):
            ipc_cache_resident_memory(config, gpu_uuid="A")
        write_cache_manifest(FakeDaemon(base + ".draft", make_model(), role="draft"))
    assert ipc_cache_resident_memory(config, gpu_uuid="A") == 4096
