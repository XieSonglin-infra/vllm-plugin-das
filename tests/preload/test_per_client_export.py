# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm_hcu.runtime_compat.weight_cache import install_per_client_export


def test_export_is_per_matching_client_and_release_stops_exports():
    messages = []
    exports = []

    class Key:
        def __init__(self, value):
            self.value = value
        def mismatched_fields(self, other):
            return [] if self.value == other.value else ["checkpoint"]

    class Daemon:
        def _export_entries(self):
            raise AssertionError("old eager export must not be used")
        def _handle_get_state(self, conn, request):
            raise AssertionError("old handler must not be used")
        def _handle_release(self):
            self.model = None
            self.entries.clear()
        def _gpu_uuid(self):
            return "uuid"

    def export(model):
        exports.append(model)
        return {"weight": object()}, {}

    module = SimpleNamespace(WeightCacheDaemon=Daemon, WeightCacheKey=Key,
                             export_entries=export,
                             send_msg=lambda conn, msg: messages.append(msg))
    install_per_client_export(module)
    install_per_client_export(module)
    daemon = Daemon()
    daemon.model = SimpleNamespace(named_parameters=lambda: [("weight", object())],
                                   named_buffers=lambda: [])
    daemon.cache_config = Key(1)
    daemon._export_entries()
    assert not exports
    daemon._handle_get_state(None, {"cache_config": Key(2)})
    assert messages[-1]["status"] == "mismatch"
    assert not exports
    daemon._handle_get_state(None, {"cache_config": Key(1)})
    daemon._handle_get_state(None, {"cache_config": Key(1)})
    assert len(exports) == 2
    assert messages[-1]["entries"]["weight"] is not messages[-2]["entries"]["weight"]
    daemon._handle_release()
    daemon._handle_get_state(None, {"cache_config": Key(1)})
    assert messages[-1]["status"] == "error"
    assert len(exports) == 2


def test_unsupported_deepgemm_layout_is_rejected_before_ready():
    import pytest
    import torch
    from vllm_hcu.runtime_compat.weight_cache import _validate_expert_cache_layouts
    model = torch.nn.Module()
    model.register_parameter("w13_weight", torch.nn.Parameter(torch.ones(1)))
    model._dsv4_channel_deepgemm_layout = "contiguous+masked"
    with pytest.raises(RuntimeError, match="unsupported DeepGEMM IPC layout"):
        _validate_expert_cache_layouts(model)


def test_export_failure_returns_explicit_error_to_client():
    messages = []

    class Key:
        def mismatched_fields(self, other):
            return []

    class Daemon:
        def _export_entries(self): pass
        def _handle_get_state(self, conn, request): pass
        def _handle_release(self): pass
        def _gpu_uuid(self): return "uuid"

    def failing_export(model):
        raise RuntimeError("invalid exported layout")

    module = SimpleNamespace(WeightCacheDaemon=Daemon, WeightCacheKey=Key,
                             export_entries=failing_export,
                             send_msg=lambda conn, msg: messages.append(msg))
    install_per_client_export(module)
    daemon = Daemon()
    daemon.model, daemon.cache_config = object(), Key()
    daemon._handle_get_state(None, {"cache_config": Key()})
    assert messages == [{"status": "error", "message": "cache export failed: invalid exported layout"}]
