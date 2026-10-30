# SPDX-License-Identifier: Apache-2.0
import time
from types import SimpleNamespace

import pytest


def fingerprint(key, config, tp_size, tp_rank):
    from vllm_hcu.runtime_compat.weight_cache import _weight_cache_config_scope

    vllm_config = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="auto"))
    with _weight_cache_config_scope(vllm_config):
        return key.from_model_config(config, tp_size, tp_rank)

from vllm_hcu.runtime_compat.weight_cache import (
    install_loader_compat, install_uuid_daemon,
)


class Unavailable(Exception):
    pass


class Mismatch(Exception):
    pass


def module_with_loader():
    class Loader:
        @staticmethod
        def _check_supported(vllm_config, model_config):
            return "checked"
        def _resolve_socket_path(self):
            raise AssertionError("old resolution")
        def _connect(self, timeout):
            self.connect_timeouts.append(timeout)
            return None
        def _apply_entries(self, model, entries, aliases, device_index):
            return "applied"
        def _fetch_entries(self, config):
            self._connect(self.state_timeout_s)
            self.calls += 1
            if self.calls <= self.failures:
                raise self.error("not ready")
            return "ok"

    torch = SimpleNamespace(
        accelerator=SimpleNamespace(current_device_index=lambda: 0),
        cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(uuid="GPU-123")),
    )
    return SimpleNamespace(IpcModelLoader=Loader, torch=torch,
                           get_socket_path=lambda gpu, directory: f"{directory}/{gpu}",
                           WeightCacheUnavailableError=Unavailable,
                           CacheConfigMismatchError=Mismatch,
                           _materialize_remaining_meta_tensors=lambda model, device: "materialized")


def test_uuid_resolution_and_explicit_socket():
    module = module_with_loader()
    install_loader_compat(module)
    loader = module.IpcModelLoader()
    loader.socket_path, loader.socket_dir = None, "/cache"
    assert loader._resolve_socket_path() == "/cache/uuid-GPU-123"
    loader.socket_path = "/manual/socket"
    assert loader._resolve_socket_path() == "/manual/socket"
    loader.socket_path = None
    module.torch.cuda.get_device_properties = lambda index: SimpleNamespace(uuid="../bad")
    with pytest.raises(Unavailable, match="UUID"):
        loader._resolve_socket_path()


def test_wait_only_for_unavailable_strict_loader(monkeypatch):
    module = module_with_loader()
    install_loader_compat(module)
    ticks = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds))
    loader = module.IpcModelLoader()
    loader.calls, loader.failures, loader.error = 0, 2, Unavailable
    loader.connect_timeouts = []
    loader.fallback, loader.state_timeout_s = False, 2
    assert loader._fetch_entries(None) == "ok"
    assert loader.calls == 3
    assert loader.connect_timeouts == [2.0, 1.5, 1.0]
    loader.calls, loader.failures = 0, 99
    with pytest.raises(Unavailable, match="did not become ready"):
        loader._fetch_entries(None)
    loader.calls, loader.error = 0, Mismatch
    with pytest.raises(Mismatch):
        loader._fetch_entries(None)
    loader.calls, loader.error, loader.fallback = 0, Unavailable, True
    with pytest.raises(Unavailable, match="not ready"):
        loader._fetch_entries(None)
    assert loader.calls == 1


def test_strict_fetch_clears_deadline_after_failure(monkeypatch):
    module = module_with_loader()
    install_loader_compat(module)
    ticks = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds))
    loader = module.IpcModelLoader()
    loader.calls, loader.failures, loader.error = 0, 99, Unavailable
    loader.connect_timeouts = []
    loader.fallback, loader.state_timeout_s = False, 1
    with pytest.raises(Unavailable, match="did not become ready"):
        loader._fetch_entries(None)
    assert not hasattr(loader, "_hcu_fetch_deadline")
    loader.fallback = True
    loader.error = Unavailable
    loader.calls = 0
    loader.failures = 0
    loader._fetch_entries(None)
    assert loader.connect_timeouts[-1] == 1


def test_daemon_uuid_matches_loader():
    module = module_with_loader()
    class Daemon:
        def _socket_path(self):
            pass
    module.WeightCacheDaemon = Daemon
    install_uuid_daemon(module)
    daemon = Daemon()
    daemon.socket_dir = "/cache"
    assert daemon._socket_path() == "/cache/uuid-GPU-123"


def test_full_model_launcher_rejects_without_explicit_opt_in(monkeypatch):
    import sys
    from vllm_hcu.preload import daemon
    monkeypatch.setattr(sys, "argv", ["vllm-hcu-preload", "--model", "example"])
    monkeypatch.setattr(daemon, "_prepare", lambda: pytest.fail("must reject before imports"))
    with pytest.raises(SystemExit, match="FULL model"):
        daemon.main()


def test_checkpoint_fingerprint_changes_with_shard(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str
        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header-hash", "vllm")

    path = tmp_path / "part.safetensors"
    path.write_bytes(b"one")
    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config = SimpleNamespace(model=str(tmp_path))
    first = fingerprint(Key, config, 1, 0)
    path.write_bytes(b"new data")
    second = fingerprint(Key, config, 1, 0)
    assert first.checkpoint != second.checkpoint
    assert "|hcu:" in first.vllm_version


def test_launcher_installs_spawn_target_before_main(monkeypatch):
    import sys
    from vllm_hcu.preload import daemon as launcher
    from vllm_hcu.preload import pp_daemon
    called = []
    upstream = SimpleNamespace(_run_daemon=lambda *args: None)
    upstream.main = lambda: called.append(upstream._run_daemon)
    monkeypatch.setattr(pp_daemon, "main", lambda daemon: daemon.main())
    monkeypatch.setattr(launcher, "_prepare", lambda: upstream)
    monkeypatch.setattr(sys, "argv", ["vllm-hcu-preload", "--enable-full-model-preload", "--model", "x"])
    launcher.main()
    assert called == [launcher.run_daemon]
    assert "--enable-full-model-preload" not in sys.argv


def test_spawn_target_prepares_fresh_worker(monkeypatch):
    from vllm_hcu.preload import daemon as launcher
    calls = []
    upstream = SimpleNamespace(_run_daemon=lambda *args: calls.append(args))
    monkeypatch.setattr(launcher, "_prepare", lambda: upstream)
    launcher.run_daemon(0, "config")
    assert calls == [(0, "config")]


def test_checkpoint_fingerprint_captures_nested_config_and_overrides(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str
        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("same-header", "vllm")

    class HFConfig:
        def __init__(self):
            self.weight_bits = 4
        def to_dict(self):
            return {"text_config": {"compression_config": {"bits": self.weight_bits}}}

    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config = SimpleNamespace(model=str(tmp_path), hf_config=HFConfig(), model_class_overrides={})
    first = fingerprint(Key, config, 1, 0)
    config.hf_config.weight_bits = 8
    second = fingerprint(Key, config, 1, 0)
    config.hf_config.weight_bits = 4
    config.model_class_overrides = {"architecture": "OtherModel"}
    third = fingerprint(Key, config, 1, 0)
    assert len({first.checkpoint, second.checkpoint, third.checkpoint}) == 3


def test_checkpoint_fingerprint_captures_effective_aiter_shuffle(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from vllm_hcu.platforms.envs import resolve_aiter_moe_shuffle
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("same-header", "vllm")

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    config = SimpleNamespace(model=str(tmp_path))
    try:
        with monkeypatch.context() as patch:
            patch.delenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", raising=False)
            resolve_aiter_moe_shuffle.cache_clear()
            default = fingerprint(Key, config, 1, 0)
            patch.setenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", "1")
            resolve_aiter_moe_shuffle.cache_clear()
            enabled = fingerprint(Key, config, 1, 0)
            patch.setenv("VLLM_HCU_USE_AITER_MOE_SHUFFLE", "0")
            resolve_aiter_moe_shuffle.cache_clear()
            disabled = fingerprint(Key, config, 1, 0)
            assert default.checkpoint == enabled.checkpoint
            assert enabled.checkpoint != disabled.checkpoint
    finally:
        resolve_aiter_moe_shuffle.cache_clear()


def test_checkpoint_fingerprint_uses_effective_nn_setting(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header", "vllm")

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    config = SimpleNamespace(model=str(tmp_path))
    monkeypatch.delenv("VLLM_USE_NN", raising=False)
    default = fingerprint(Key, config, 1, 0)
    monkeypatch.setenv("VLLM_USE_NN", "true")
    enabled = fingerprint(Key, config, 1, 0)
    monkeypatch.setenv("VLLM_USE_NN", "1")
    assert fingerprint(Key, config, 1, 0) == enabled == default
    monkeypatch.setenv("VLLM_USE_NN", "0")
    disabled = fingerprint(Key, config, 1, 0)
    monkeypatch.setenv("VLLM_USE_NN", "false")
    assert fingerprint(Key, config, 1, 0) == disabled
    assert disabled != enabled


def test_daemon_and_loader_fingerprint_moe_backend_and_restore_scope(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header", "vllm")

    model_config = SimpleNamespace(model=str(tmp_path))

    class Daemon:
        def __init__(self, vllm_config):
            self.key = Key.from_model_config(model_config, 1, 0)

    class PipelineDaemon(Daemon):
        pass

    class Loader:
        def load_model(self, vllm_config):
            if getattr(vllm_config, "fail", False):
                raise ValueError("load failed")
            return Key.from_model_config(model_config, 1, 0)

        def load_weights(self):
            return Key.from_model_config(model_config, 1, 0)

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key, WeightCacheDaemon=Daemon))
    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key, IpcModelLoader=Loader))
    aiter = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="aiter"))
    triton = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="triton"))
    daemon_key = PipelineDaemon(aiter).key
    assert daemon_key == Loader().load_model(aiter)
    assert daemon_key != Loader().load_model(triton)
    with pytest.raises(RuntimeError, match="requires vLLM kernel configuration"):
        Loader().load_weights()
    with pytest.raises(RuntimeError, match="kernel_config.moe_backend"):
        Loader().load_model(SimpleNamespace())
    aiter.fail = True
    with pytest.raises(ValueError, match="load failed"):
        Loader().load_model(aiter)
    with pytest.raises(RuntimeError, match="requires vLLM kernel configuration"):
        Key.from_model_config(model_config, 1, 0)


def test_expert_parallel_cache_key_distinguishes_dp_shards(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import (
        _weight_cache_config_scope, install_key_fingerprint,
    )

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header", "vllm")

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    model = SimpleNamespace(model=str(tmp_path))
    parallel = SimpleNamespace(enable_expert_parallel=True,
                               data_parallel_size=8, data_parallel_rank=0)
    config = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="deep_gemm"),
                             parallel_config=parallel)
    with _weight_cache_config_scope(config):
        rank0 = Key.from_model_config(model, 1, 0)
        parallel.data_parallel_rank = 1
        rank1 = Key.from_model_config(model, 1, 0)
        parallel.enable_expert_parallel = False
        non_ep = Key.from_model_config(model, 1, 0)
        parallel.enable_expert_parallel = True
        parallel.data_parallel_rank = 8
        with pytest.raises(RuntimeError, match="invalid data-parallel rank"):
            Key.from_model_config(model, 1, 0)
    assert rank0 != rank1
    assert rank0 != non_ep


def test_expert_parallel_cache_key_distinguishes_parallel_topology(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import (
        _weight_cache_config_scope, install_key_fingerprint,
    )

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header", "vllm")

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    model = SimpleNamespace(model=str(tmp_path))
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        data_parallel_size=8,
        data_parallel_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size_local=8,
        expert_parallel_size=8,
        all2all_backend="deepep",
        expert_placement_strategy="linear",
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm"),
        parallel_config=parallel,
    )
    with _weight_cache_config_scope(config):
        baseline = Key.from_model_config(model, 1, 0)
        parallel.expert_placement_strategy = "round_robin"
        changed_placement = Key.from_model_config(model, 1, 0)
        parallel.expert_placement_strategy = "linear"
        parallel.decode_context_parallel_size = 2
        changed_dcp = Key.from_model_config(model, 1, 0)
    assert baseline != changed_placement
    assert baseline != changed_dcp


def test_incomplete_meta_parameters_fail_closed():
    import torch
    module = module_with_loader()
    install_loader_compat(module)
    with torch.device("meta"):
        layer = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="missing model tensors"):
        module._materialize_remaining_meta_tensors(layer, torch.device("cpu"))
    layer = torch.nn.Linear(2, 2)
    assert module._materialize_remaining_meta_tensors(layer, torch.device("cpu")) == "materialized"


def test_loader_rejects_non_cuda_platform(monkeypatch):
    import vllm.platforms
    module = module_with_loader()
    install_loader_compat(module)
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace(is_cuda_alike=lambda: False))
    with pytest.raises(RuntimeError, match="CUDA/ROCm"):
        module.IpcModelLoader._check_supported(None, None)
    monkeypatch.setattr(vllm.platforms, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True))
    assert module.IpcModelLoader._check_supported(None, None) == "checked"


def test_incomplete_meta_buffers_fail_closed():
    import torch
    module = module_with_loader()
    install_loader_compat(module)
    layer = torch.nn.Module()
    layer.register_buffer("cache", torch.empty(3, device="meta"), persistent=False)
    with pytest.raises(RuntimeError, match="missing model tensors: cache"):
        module._materialize_remaining_meta_tensors(layer, torch.device("cpu"))


def test_checkpoint_fingerprint_includes_deepep_shared_weight_layout(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from vllm_hcu.platforms import envs
    from vllm_hcu.runtime_compat.weight_cache import install_key_fingerprint

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("same-header", "vllm")

    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    config = SimpleNamespace(model=str(tmp_path))
    monkeypatch.setattr(envs, "optional_custom_op_enabled", lambda value: bool(value))
    monkeypatch.setattr(envs, "VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS", True)
    shared = fingerprint(Key, config, 1, 0)
    monkeypatch.setattr(envs, "VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS", False)
    separate = fingerprint(Key, config, 1, 0)
    assert shared.checkpoint != separate.checkpoint


def test_dspark_role_identity_and_scope_restoration():
    from vllm_hcu.runtime_compat.weight_cache import (
        _cache_model_role, _weight_cache_config_scope,
    )
    target, draft = object(), object()
    cfg = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="deep_gemm"),
                          model_config=target,
                          speculative_config=SimpleNamespace(method="dspark", draft_model_config=draft))
    with _weight_cache_config_scope(cfg, target):
        assert _cache_model_role.get() == "target"
        with _weight_cache_config_scope(cfg, draft):
            assert _cache_model_role.get() == "draft"
        assert _cache_model_role.get() == "target"
        with pytest.raises(RuntimeError, match="cannot identify DSpark"):
            with _weight_cache_config_scope(cfg, object()):
                pass
        assert _cache_model_role.get() == "target"
    assert _cache_model_role.get() is None


def test_dspark_roles_separate_socket_and_fingerprint(tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import (
        _weight_cache_config_scope, install_key_fingerprint,
    )

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("same-header", "vllm")

    module = module_with_loader()
    install_loader_compat(module)
    install_key_fingerprint(SimpleNamespace(WeightCacheKey=Key))
    loader = module.IpcModelLoader()
    loader.socket_path, loader.socket_dir = None, "/cache"
    target, draft = SimpleNamespace(model=str(tmp_path)), SimpleNamespace(model=str(tmp_path))
    config = SimpleNamespace(kernel_config=SimpleNamespace(moe_backend="deep_gemm"),
                             model_config=target,
                             speculative_config=SimpleNamespace(method="dspark", draft_model_config=draft))
    with _weight_cache_config_scope(config, target):
        target_key = Key.from_model_config(target, 1, 0)
        assert loader._resolve_socket_path() == "/cache/uuid-GPU-123"
    with _weight_cache_config_scope(config, draft):
        draft_key = Key.from_model_config(draft, 1, 0)
        assert loader._resolve_socket_path() == "/cache/uuid-GPU-123.draft"
        loader.socket_path = "/manual/socket"
        assert loader._resolve_socket_path() == "/manual/socket.draft"
    assert target_key.checkpoint != draft_key.checkpoint


def test_expert_maps_are_real_only_during_ipc_meta_construction():
    import torch
    from vllm_hcu.runtime_compat.weight_cache import (
        _cache_meta_construction, install_meta_expert_map_compat,
    )
    class Manager:
        def __init__(self):
            self.mapping = torch.arange(4)
    install_meta_expert_map_compat(SimpleNamespace(ExpertMapManager=Manager))
    with torch.device("meta"):
        assert Manager().mapping.is_meta
        token = _cache_meta_construction.set(True)
        try:
            assert Manager().mapping.device.type == "cpu"
        finally:
            _cache_meta_construction.reset(token)
        assert Manager().mapping.is_meta


def test_ipc_expert_manager_aliases_rebind_to_loaded_buffers():
    import torch
    from vllm_hcu.runtime_compat.weight_cache import _rebind_expert_map_managers
    model = torch.nn.Module()
    model.register_buffer("_expert_map", torch.tensor([0, -1]))
    model.register_buffer("expert_mask", torch.tensor([1, 0]))
    model.expert_map_manager = SimpleNamespace(_expert_map=None, _expert_mask=None,
                                                _routing_tables=None)
    _rebind_expert_map_managers(model)
    assert model.expert_map_manager._expert_map is model._expert_map
    assert model.expert_map_manager._expert_mask is model.expert_mask


def _dspark_config(tmp_path):
    target = SimpleNamespace(model=str(tmp_path), mutated=False)
    draft = SimpleNamespace(model=str(tmp_path), mutated=False)
    draft_parallel = SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1,
        data_parallel_size=1, enable_expert_parallel=False,
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"),
        model_config=target,
        parallel_config=SimpleNamespace(tensor_parallel_size=1, tensor_parallel_rank=0),
        speculative_config=SimpleNamespace(
            method="dspark", draft_model_config=draft,
            draft_parallel_config=draft_parallel,
        ),
    )
    return config, target, draft, draft_parallel


def test_target_load_precomputes_draft_fingerprint(monkeypatch, tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat import weight_cache

    monkeypatch.setattr(weight_cache, "_precomputed_cache_keys", {})

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str = "v"

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls(f"key:{model_config.mutated}")

    config, target, draft, _ = _dspark_config(tmp_path)

    class Daemon:
        def __init__(self, vllm_config):
            pass

    class Loader:
        def load_model(self, vllm_config, model_config, *args, **kwargs):
            # The draft fingerprint must exist before the target body runs.
            assert (id(draft), "draft") in weight_cache._precomputed_cache_keys
            return model_config is target

    module = SimpleNamespace(
        WeightCacheKey=Key, WeightCacheDaemon=Daemon, IpcModelLoader=Loader
    )
    weight_cache._install_cache_config_scope(module)
    assert Loader().load_model(config, target) is True
    assert weight_cache._precomputed_cache_keys[(id(draft), "draft")].checkpoint == "key:False"


def test_cached_draft_fingerprint_survives_target_postprocessing(monkeypatch, tmp_path):
    from dataclasses import dataclass
    from vllm_hcu.runtime_compat.weight_cache import (
        _precompute_draft_key, _weight_cache_config_scope, install_key_fingerprint,
    )

    monkeypatch.setattr(
        "vllm_hcu.runtime_compat.weight_cache._precomputed_cache_keys", {}
    )

    @dataclass(frozen=True)
    class Key:
        checkpoint: str
        vllm_version: str = "v"

        @classmethod
        def from_model_config(cls, model_config, tp_size, tp_rank):
            return cls("header", "vllm")

    module = SimpleNamespace(WeightCacheKey=Key)
    install_key_fingerprint(module)
    config, _, draft, draft_parallel = _dspark_config(tmp_path)
    _precompute_draft_key(module, config)
    with _weight_cache_config_scope(
        config, draft, model_role="draft", parallel_config=draft_parallel
    ):
        cached = Key.from_model_config(draft, 1, 0)
    # A later call under the draft scope returns the precomputed key even
    # after target post-processing changed fingerprint inputs.
    draft.quantization_config = {"changed": True}
    with _weight_cache_config_scope(
        config, draft, model_role="draft", parallel_config=draft_parallel
    ):
        assert Key.from_model_config(draft, 1, 0) is cached


def test_router_hash_table_rebinds_to_loaded_gate_parameter():
    import torch
    from vllm_hcu.runtime_compat.weight_cache import reconcile_router_state_devices

    class Gate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tid2eid = torch.nn.Parameter(
                torch.zeros(4, 2, dtype=torch.int32), requires_grad=False
            )

    class Experts(torch.nn.Module):
        def __init__(self, gate):
            super().__init__()
            # FusedMoERouter is a plain object that captured the constructor's
            # Parameter; IPC loading replaces the gate's registration.
            self.router = SimpleNamespace(
                _hash_indices_table=gate.tid2eid,
                e_score_correction_bias=None,
            )

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate()
            self.experts = Experts(self.gate)

    model = torch.nn.Module()
    model.mlp = Mlp()
    loaded = torch.nn.Parameter(torch.ones(4, 2, dtype=torch.int32), requires_grad=False)
    model.mlp.gate._parameters["tid2eid"] = loaded

    moved = reconcile_router_state_devices(model, torch.device("cpu"))
    assert model.mlp.experts.router._hash_indices_table is loaded
    assert "mlp.experts.router._hash_indices_table" in moved
