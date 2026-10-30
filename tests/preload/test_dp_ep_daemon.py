from types import SimpleNamespace

import pytest

from vllm_hcu.preload.dp_ep_daemon import (
    _start_rank_processes,
    plan_local_dp_ep_ranks,
    run_dp_ep_daemon,
)


def test_plan_local_dp_ep_ranks_maps_one_device_per_rank():
    assert plan_local_dp_ep_ranks(8, 1, 8) == tuple((rank, rank) for rank in range(8))


@pytest.mark.parametrize(
    ("dp_size", "tp_size", "local_world_size"),
    [(0, 1, 1), (1, 0, 1), (8, 1, 7), (4, 2, 4)],
)
def test_plan_local_dp_ep_ranks_rejects_incomplete_local_groups(
    dp_size, tp_size, local_world_size
):
    with pytest.raises(ValueError):
        plan_local_dp_ep_ranks(dp_size, tp_size, local_world_size)


def _config(**parallel_overrides):
    parallel = dict(
        data_parallel_size=8,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        nnodes=1,
        node_rank=0,
    )
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        load_config=SimpleNamespace(load_format="safetensors"),
        model_config=SimpleNamespace(),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"enable_expert_parallel": False}, "requires --enable-expert-parallel"),
        ({"tensor_parallel_size": 2}, "TP=1 and PP=1"),
        ({"pipeline_parallel_size": 2}, "TP=1 and PP=1"),
        ({"nnodes": 2}, "one node only"),
        ({"node_rank": 1}, "one node only"),
        ({"data_parallel_size": 1}, "data_parallel_size=8 only"),
        ({"data_parallel_size": 2}, "data_parallel_size=8 only"),
    ],
)
def test_run_dp_ep_daemon_rejects_unsupported_topology(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_dp_ep_daemon(SimpleNamespace(), SimpleNamespace(), _config(**overrides))


def test_run_dp_ep_daemon_rejects_ipc_cache_as_producer():
    config = _config()
    config.load_config.load_format = "ipc_cache"
    with pytest.raises(ValueError, match="must load weights from the checkpoint"):
        run_dp_ep_daemon(SimpleNamespace(), SimpleNamespace(), config)


def test_rank_start_failure_cleans_started_and_failed_children():
    class Process:
        def __init__(self, name, fail_start=False):
            self.name = name
            self.fail_start = fail_start
            self.started = False
            self.terminated = False
            self.joined = False

        def start(self):
            if self.fail_start:
                raise RuntimeError("synthetic child start failure")
            self.started = True

        def is_alive(self):
            if not self.started:
                raise AssertionError("process was never started")
            return not self.terminated

        def terminate(self):
            self.terminated = True

        def join(self, timeout=None):
            if not self.started:
                raise AssertionError("process was never started")
            self.joined = True

    class Context:
        def __init__(self):
            self.processes = []

        def Process(self, *, name, **_kwargs):
            process = Process(name, fail_start=len(self.processes) == 1)
            self.processes.append(process)
            return process

    context = Context()
    with pytest.raises(RuntimeError, match="synthetic child start failure"):
        _start_rank_processes(
            context,
            ((0, 0), (1, 1)),
            SimpleNamespace(),
            "tcp://127.0.0.1:1",
            None,
            object(),
        )
    assert context.processes[0].terminated
    assert context.processes[0].joined
    assert not context.processes[1].terminated


def test_ready_requires_every_distinct_expected_live_rank():
    import queue
    from vllm_hcu.preload.dp_ep_daemon import _wait_for_ready

    messages = queue.Queue()
    messages.put(1)
    messages.put(0)
    processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=None)]
    _wait_for_ready(processes, messages, {0, 1}, lambda: False)


@pytest.mark.parametrize("notifications", [(0, 0), (2,)])
def test_ready_rejects_duplicate_or_unknown_rank(notifications):
    import queue
    from vllm_hcu.preload.dp_ep_daemon import _wait_for_ready

    messages = queue.Queue()
    for rank in notifications:
        messages.put(rank)
    with pytest.raises(RuntimeError, match="invalid or duplicate ready rank"):
        _wait_for_ready([SimpleNamespace(exitcode=None)], messages, {0, 1}, lambda: False)


def test_ready_rejects_rank_that_exited_after_reporting_ready():
    import queue
    from vllm_hcu.preload.dp_ep_daemon import _wait_for_ready

    messages = queue.Queue()
    messages.put(0)
    with pytest.raises(SystemExit, match="rank0 exit=1"):
        _wait_for_ready([SimpleNamespace(name="rank0", exitcode=1)], messages,
                        {0}, lambda: False)


def test_stop_all_kills_child_that_ignores_terminate():
    from vllm_hcu.preload.dp_ep_daemon import _stop_all

    class StubbornProcess:
        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False
            self.joins = []

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout=None):
            self.joins.append(timeout)

    child = StubbornProcess()
    _stop_all([child])
    assert child.terminated and child.killed
    assert child.joins == [10, 10]


def test_two_roles_report_ready_only_after_both_are_listening(monkeypatch):
    import queue
    import threading
    import torch
    from vllm_hcu.preload.dp_ep_daemon import _serve_roles

    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda index: None)
    release_second = threading.Event()
    release_first = threading.Event()
    ready = queue.Queue()
    failures = []

    class Role:
        def __init__(self, gate, release):
            self.gate, self.release = gate, release

        def serve_forever(self, ready_callback):
            if self.gate is not None:
                self.gate.wait(timeout=2)
            ready_callback()
            self.release.wait(timeout=2)

    def run():
        try:
            _serve_roles((Role(None, release_first), Role(release_second, release_first)),
                         0, 5, ready)
        except RuntimeError as exc:
            failures.append(str(exc))

    worker = threading.Thread(target=run)
    worker.start()
    with pytest.raises(queue.Empty):
        ready.get(timeout=0.2)
    release_second.set()
    assert ready.get(timeout=2) == 5
    release_first.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert failures and "role server failed" in failures[0]


def test_rejects_unsupported_draft_topology_before_quant_or_gpu():
    config = _config()
    draft_parallel = _config(data_parallel_size=4).parallel_config
    config.speculative_config = SimpleNamespace(method="dspark", draft_model_config=object(),
                                                 draft_parallel_config=draft_parallel)
    with pytest.raises(ValueError, match="draft preload requires TP1/PP1/DP1"):
        run_dp_ep_daemon(SimpleNamespace(), SimpleNamespace(), config)
