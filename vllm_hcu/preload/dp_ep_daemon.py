# SPDX-License-Identifier: Apache-2.0
"""Single-node DP/EP weight-cache process group for TP=1 deployments.

The daemon must initialize the same DP world as the engine. With expert
parallel enabled, each DP rank owns a different expert shard; UUID-only socket
selection is safe only because the cache key also includes the DP/EP identity.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait
import os
import queue
import signal


def plan_local_dp_ep_ranks(dp_size: int, tp_size: int, local_world_size: int):
    """Return ``(global_rank, local_rank)`` pairs for the supported single node."""
    if dp_size < 1 or tp_size < 1:
        raise ValueError("DP and TP sizes must be positive")
    world_size = dp_size * tp_size
    if local_world_size != world_size:
        raise ValueError(
            "single-node DP/EP preload requires every DP*TP rank locally; "
            f"got world size {world_size} and local world size {local_world_size}"
        )
    return tuple((rank, rank) for rank in range(world_size))


def _run_rank(
    rank: int,
    local_rank: int,
    vllm_config,
    init_method: str,
    socket_dir: str | None,
    ready_queue,
) -> None:
    """Load one rank's normal model shard and serve its UUID socket."""
    os.environ["VLLM_HCU_PRELOAD_DP_RANK"] = str(rank)
    parallel = vllm_config.parallel_config
    parallel.data_parallel_rank = rank // parallel.tensor_parallel_size

    from vllm_hcu.preload.daemon import _prepare

    daemon_module = _prepare()
    # Match the serving worker: bind serialized HCU feature flags before
    # importing model/communicator code or constructing any weights.
    from vllm_hcu.patch.worker import apply_worker_patches
    apply_worker_patches(vllm_config)
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.model_loader import get_model
    from vllm.platforms import current_platform

    class DpEpWeightCacheDaemon(daemon_module.WeightCacheDaemon):
        def __init__(self):
            # TP=1 is currently required, but pass the true TP coordinate so
            # the base protocol retains its usual TP fields and diagnostics.
            super().__init__(
                vllm_config,
                rank % parallel.tensor_parallel_size,
                init_method,
                socket_dir,
            )
            self.global_rank = rank
            self.local_rank = local_rank

        def load_model(self) -> None:
            torch.accelerator.set_device_index(self.local_rank)
            init_distributed_environment(
                world_size=parallel.data_parallel_size
                * parallel.tensor_parallel_size,
                rank=self.global_rank,
                distributed_init_method=init_method,
                local_rank=self.local_rank,
                backend=current_platform.dist_backend,
            )
            with set_current_vllm_config(vllm_config):
                ensure_model_parallel_initialized(
                    parallel.tensor_parallel_size,
                    parallel.pipeline_parallel_size,
                    parallel.prefill_context_parallel_size,
                    parallel.decode_context_parallel_size,
                    backend=current_platform.dist_backend,
                )
                self.model = get_model(vllm_config=vllm_config)
            self._export_entries()

    daemon = DpEpWeightCacheDaemon()
    spec = getattr(vllm_config, "speculative_config", None)
    draft_daemon = None
    if spec is not None:
        if getattr(spec, "method", None) != "dspark":
            raise ValueError("DP/EP preload supports DSpark as its only draft role")
        draft_config = getattr(spec, "draft_model_config", None)
        if draft_config is None:
            raise RuntimeError("DSpark draft model configuration is missing")
        from vllm_hcu.runtime_compat.weight_cache import _weight_cache_config_scope
        # Fingerprint both roles before target loading can mutate HF config.
        draft_daemon = DpEpWeightCacheDaemon()
        draft_daemon.cache_role = "draft"
        with _weight_cache_config_scope(
            vllm_config, draft_config, parallel_config=spec.draft_parallel_config
        ):
            draft_daemon.cache_config = daemon_module.WeightCacheKey.from_model_config(
                draft_config, tp_size=parallel.tensor_parallel_size,
                tp_rank=rank % parallel.tensor_parallel_size,
            )
    daemon.load_model()
    if draft_daemon is not None:
        from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model
        with set_current_vllm_config(vllm_config):
            draft_daemon.model = load_dspark_model(daemon.model, vllm_config)
        draft_daemon._export_entries()
    from vllm.distributed import get_ep_group

    # The daemon never runs a forward: free the all2all/DeepEP workspaces built
    # during construction so the engine's own communicators do not stack on
    # top of a duplicate copy when computing the KV-cache budget.
    from vllm_hcu.runtime_compat.weight_cache import release_all2all_buffers

    released = release_all2all_buffers()
    print(
        f"DP/EP cache rank released {len(released)} all2all manager(s) before "
        f"serving: {', '.join(released) if released else 'none'}",
        flush=True,
    )

    ep_group = get_ep_group()
    print(
        "DP/EP cache rank ready-to-serve: "
        f"dp_rank={parallel.data_parallel_rank} "
        f"tp_rank={rank % parallel.tensor_parallel_size} "
        f"ep_rank={ep_group.rank_in_group}/{ep_group.world_size} "
        f"local_device={local_rank} gpu_uuid={daemon._gpu_uuid()} "
        f"model_role=target checkpoint_fingerprint={daemon.cache_config.checkpoint} "
        f"tensor_names={len(daemon.entries)}",
        flush=True,
    )
    if draft_daemon is None:
        daemon.serve_forever(ready_callback=lambda: ready_queue.put(rank))
    else:
        print(
            "DP/EP cache draft ready-to-serve: "
            f"dp_rank={parallel.data_parallel_rank} local_device={local_rank} "
            f"gpu_uuid={draft_daemon._gpu_uuid()} model_role=draft "
            f"checkpoint_fingerprint={draft_daemon.cache_config.checkpoint} "
            f"tensor_names={len(draft_daemon.entries)}",
            flush=True,
        )
        _serve_roles((daemon, draft_daemon), local_rank, rank, ready_queue)


def _serve_roles(daemons, local_rank, rank, ready_queue):
    """Serve both role sockets in one GPU owner; report ready only together."""
    import threading
    import torch

    errors = queue.Queue()
    ready = [threading.Event() for _ in daemons]

    def serve(daemon, event):
        try:
            # Device selection is thread-local in CUDA/HIP.
            torch.accelerator.set_device_index(local_rank)
            daemon.serve_forever(ready_callback=event.set)
            errors.put(RuntimeError("weight cache role server exited"))
        except BaseException as exc:
            errors.put(exc)

    threads = [threading.Thread(target=serve, args=(daemon, event), daemon=True)
               for daemon, event in zip(daemons, ready)]
    for thread in threads:
        thread.start()
    reported = False
    while True:
        try:
            error = errors.get(timeout=0.1)
        except queue.Empty:
            if not reported and all(event.is_set() for event in ready):
                ready_queue.put(rank)
                reported = True
            continue
        raise RuntimeError(f"weight cache role server failed: {error}") from error


def run_dp_ep_daemon(daemon_module, args, vllm_config) -> None:
    """Start every local DP/EP rank and fail the group on any rank failure."""
    parallel = vllm_config.parallel_config
    dp_size = int(parallel.data_parallel_size)
    tp_size = int(parallel.tensor_parallel_size)
    pp_size = int(parallel.pipeline_parallel_size)
    if not parallel.enable_expert_parallel:
        raise ValueError("DP weight-cache daemon requires --enable-expert-parallel")
    if tp_size != 1 or pp_size != 1:
        raise ValueError(
            "DP/EP preload currently supports TP=1 and PP=1; "
            f"got TP={tp_size}, PP={pp_size}"
        )
    if int(parallel.nnodes) != 1 or int(parallel.node_rank) != 0:
        raise ValueError("DP/EP preload currently supports one node only")
    if dp_size != 8:
        raise ValueError("DP/EP preload currently supports data_parallel_size=8 only")
    if vllm_config.load_config.load_format == "ipc_cache":
        raise ValueError("the preload daemon must load weights from the checkpoint")

    spec = getattr(vllm_config, "speculative_config", None)
    if spec is not None:
        if getattr(spec, "method", None) != "dspark":
            raise ValueError("DP/EP preload supports DSpark as its only draft role")
        if getattr(spec, "draft_model_config", None) is None:
            raise ValueError("DSpark draft model configuration is missing")
        draft_parallel = getattr(spec, "draft_parallel_config", None)
        if (draft_parallel is None
            or int(draft_parallel.tensor_parallel_size) != 1
            or int(draft_parallel.pipeline_parallel_size) != 1
            or int(draft_parallel.data_parallel_size) != 1
            or bool(draft_parallel.enable_expert_parallel)):
            raise ValueError("DSpark draft preload requires TP1/PP1/DP1 without EP")

    # The quantization gate stays fail-closed: only the channel-weight /
    # token-activation FP8 scheme validated by the HCU probes is allowed, and
    # the quantization patches still reject any kernel backend, packed expert
    # layout or topology that was not verified.
    daemon_module.check_ipc_quant_support(vllm_config.model_config, where="daemon")
    if spec is not None:
        daemon_module.check_ipc_quant_support(spec.draft_model_config, where="daemon draft")

    from vllm.utils.network_utils import get_distributed_init_method, get_open_port
    import torch

    expected = plan_local_dp_ep_ranks(dp_size, tp_size, torch.cuda.device_count())
    master_port = args.weight_cache_master_port or get_open_port()
    init_method = get_distributed_init_method("127.0.0.1", master_port)
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    processes = _start_rank_processes(
        context,
        expected,
        vllm_config,
        init_method,
        args.weight_cache_socket_dir,
        ready_queue,
    )

    stopping = False

    def shutdown(_signum, _frame):
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.is_alive():
                process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        _wait_for_ready(processes, ready_queue, {rank for rank, _ in expected},
                        lambda: stopping)
        if stopping:
            return
        daemon_module._report_ready(
            f"===== DP/EP weight cache READY: {len(expected)} local ranks, "
            f"DP={dp_size}, TP={tp_size}, EP=enabled ====="
        )
        completed = wait([process.sentinel for process in processes])
        failed = [process for process in processes if process.sentinel in completed]
        for process in failed:
            process.join()
        if not stopping:
            raise SystemExit(
                "DP/EP weight-cache rank exited unexpectedly: "
                + ", ".join(
                    f"{process.name} exit={process.exitcode}" for process in failed
                )
            )
    finally:
        _stop_all(processes)
        ready_queue.close()


def _wait_for_ready(processes, ready_queue, expected, stopping):
    """Accept exactly one readiness notification per expected live rank."""
    ready = set()
    while ready != expected and not stopping():
        try:
            rank = ready_queue.get(timeout=1.0)
            if rank not in expected or rank in ready:
                raise RuntimeError(f"invalid or duplicate ready rank: {rank!r}")
            ready.add(rank)
        except queue.Empty:
            pass
        dead = [process for process in processes if process.exitcode is not None]
        if dead and not stopping():
            raise SystemExit(
                "DP/EP weight-cache startup failed: "
                + ", ".join(
                    f"{process.name} exit={process.exitcode}" for process in dead
                )
            )


def _stop_all(processes) -> None:
    for process in processes:
        try:
            alive = process.is_alive()
        except AssertionError:
            # A Process whose start() failed may not have a pid yet.
            alive = False
        if alive:
            process.terminate()
    for process in processes:
        try:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        except AssertionError:
            # Keep cleanup best-effort when a child failed before start().
            pass


def _start_rank_processes(
    context,
    expected,
    vllm_config,
    init_method,
    socket_dir,
    ready_queue,
):
    """Start all rank children, cleaning up if any ``start`` call fails."""
    processes = []
    for rank, local_rank in expected:
        process = context.Process(
            target=_run_rank,
            args=(
                rank,
                local_rank,
                vllm_config,
                init_method,
                socket_dir,
                ready_queue,
            ),
            name=f"vllm-weight-cache-daemon-dp{rank}",
        )
        try:
            process.start()
        except BaseException:
            # Include the failing Process: start() can fail after creating a
            # child, and that child must not survive an otherwise visible
            # launcher error.
            _stop_all([process, *processes])
            raise
        processes.append(process)
    return processes
