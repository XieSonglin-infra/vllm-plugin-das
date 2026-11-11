# SPDX-License-Identifier: Apache-2.0
"""Pipeline-aware weight-cache daemon launcher.

The vLLM 0.28 daemon predates PP support and starts a TP-only process group.
This module keeps the installed protocol and model loader intact while
launching one cache process for every global TP/PP rank.  Each process loads
the stage selected by vLLM's normal PP group and serves a UUID+stage socket.
"""
from __future__ import annotations

import multiprocessing
from multiprocessing.connection import wait
import os
import queue
import signal


def _run_pipeline_daemon(
    global_rank: int,
    local_rank: int,
    vllm_config,
    distributed_init_method: str,
    socket_dir: str | None,
    ready_queue,
    tp_size: int,
    pp_size: int,
) -> None:
    """Load one PP/TP stage and keep its CUDA tensors available."""
    os.environ["VLLM_HCU_PRELOAD_PP_SIZE"] = str(pp_size)
    os.environ["VLLM_HCU_PRELOAD_PP_RANK"] = str(global_rank // tp_size)

    # Each spawned process must arm the same plugin callbacks as an engine
    # worker before importing the installed daemon/model loader modules.
    from vllm_hcu.preload.daemon import _prepare

    daemon_module = _prepare()
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

    class PipelineWeightCacheDaemon(daemon_module.WeightCacheDaemon):
        def __init__(self):
            super().__init__(
                vllm_config,
                global_rank % tp_size,
                distributed_init_method,
                socket_dir,
            )
            self.global_rank = global_rank
            self.local_rank = local_rank

        def load_model(self) -> None:
            torch.accelerator.set_device_index(self.local_rank)
            # Match the TP daemon before constructing sparse MLA workspaces.
            from vllm.v1.worker.workspace import (
                init_workspace_manager, is_workspace_manager_initialized,
            )

            if not is_workspace_manager_initialized():
                parallel = self.vllm_config.parallel_config
                spec = getattr(self.vllm_config, "speculative_config", None)
                init_workspace_manager(
                    torch.device("cuda", self.local_rank),
                    num_ubatches=2 if getattr(parallel, "enable_dbo", False) else 1,
                    num_lanes=2 if getattr(spec, "method", None) == "dspark" else 1,
                )
            init_distributed_environment(
                world_size=tp_size * pp_size,
                rank=self.global_rank,
                distributed_init_method=distributed_init_method,
                local_rank=self.local_rank,
                backend=current_platform.dist_backend,
            )
            with set_current_vllm_config(vllm_config):
                ensure_model_parallel_initialized(
                    tp_size,
                    pp_size,
                    backend=current_platform.dist_backend,
                )
                self.model = get_model(vllm_config=vllm_config)
            torch.accelerator.empty_cache()
            self._export_entries()

    daemon = PipelineWeightCacheDaemon()

    # V2 instantiates the independent DSpark draft only on the last target
    # pipeline stage. Export a role-specific socket on those GPUs alone.
    spec = getattr(vllm_config, "speculative_config", None)
    draft_daemon = None
    if spec is not None and global_rank // tp_size == pp_size - 1:
        _validate_pipeline_spec(spec, tp_size, pp_size)
        from vllm_hcu.runtime_compat.weight_cache import _weight_cache_config_scope

        draft_config = spec.draft_model_config
        draft_parallel = spec.draft_parallel_config
        draft_daemon = PipelineWeightCacheDaemon()
        draft_daemon.cache_role = "draft"
        with _weight_cache_config_scope(
            vllm_config,
            draft_config,
            model_role="draft",
            parallel_config=draft_parallel,
        ):
            draft_daemon.cache_config = daemon_module.WeightCacheKey.from_model_config(
                draft_config,
                tp_size=tp_size,
                tp_rank=global_rank % tp_size,
            )
    daemon.load_model()
    if draft_daemon is not None:
        from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

        # Only the last PP stage participates in draft loading. The fast
        # loader uses the full WORLD group and would wait for earlier stages.
        from vllm.config import replace

        draft_load_config = replace(
            vllm_config,
            load_config=replace(vllm_config.load_config, load_format="safetensors"),
        )
        with set_current_vllm_config(draft_load_config):
            draft_daemon.model = load_dspark_model(daemon.model, draft_load_config)
        torch.accelerator.empty_cache()
        draft_daemon._export_entries()

    if draft_daemon is None:
        daemon.serve_forever(ready_callback=lambda: ready_queue.put(global_rank))
    else:
        print(
            f"PP/TP cache rank ready-to-serve: global_rank={global_rank} "
            f"pp_rank={global_rank // tp_size} tp_rank={global_rank % tp_size} "
            f"model_role=target tensor_names={len(daemon.entries)} "
            f"draft_tensor_names={len(draft_daemon.entries)}",
            flush=True,
        )
        from vllm_hcu.preload.dp_ep_daemon import _serve_roles

        _serve_roles((daemon, draft_daemon), local_rank, global_rank, ready_queue)


def _validate_pipeline_spec(spec, tp_size: int, pp_size: int) -> None:
    """Validate the only draft topology supported by the PP cache launcher."""
    if getattr(spec, "method", None) != "dspark":
        raise ValueError(
            "pipeline weight-cache preload currently supports DSpark as its "
            "only draft role"
        )
    draft_config = getattr(spec, "draft_model_config", None)
    draft_parallel = getattr(spec, "draft_parallel_config", None)
    if draft_config is None or draft_parallel is None:
        raise ValueError("DSpark draft model and parallel configurations are required")
    # vLLM creates DSpark's draft parallel config with the target TP but a
    # single pipeline stage. The draft model is still served by the target PP
    # rank that owns its target-layer state; requiring draft PP=2 would reject
    # every valid PP2 DSpark engine configuration.
    fields = {
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    for name, expected in fields.items():
        actual = int(getattr(draft_parallel, name, 1) or 1)
        if actual != expected:
            raise ValueError(
                "PP/TP DSpark preload requires draft "
                f"{name}={expected}, got {actual}"
            )
    if bool(getattr(draft_parallel, "enable_expert_parallel", False)):
        raise ValueError("PP/TP DSpark preload does not support draft expert parallelism")


def main(daemon_module) -> None:
    """Parse full-model daemon arguments and launch a PP-aware local group."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.utils.network_utils import get_distributed_init_method, get_open_port

    parser = FlexibleArgumentParser(
        description="Launch pipeline-aware weight cache daemons (one per TP/PP rank)."
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--weight-cache-socket-dir", type=str, default=None)
    parser.add_argument(
        "--weight-cache-master-port",
        type=int,
        default=None,
        help="Rendezvous port for the daemon TP/PP group; use a port distinct "
        "from the engine master port.",
    )
    args = parser.parse_args()
    from vllm_hcu.runtime_compat.weight_cache import validate_preload_socket_dir

    validate_preload_socket_dir(args.weight_cache_socket_dir)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    parallel = vllm_config.parallel_config
    tp_size = int(parallel.tensor_parallel_size)
    pp_size = int(parallel.pipeline_parallel_size)
    if pp_size <= 1:
        if (
            int(getattr(parallel, "data_parallel_size", 1)) > 1
            or bool(getattr(parallel, "enable_expert_parallel", False))
        ):
            from vllm_hcu.preload.dp_ep_daemon import run_dp_ep_daemon

            return run_dp_ep_daemon(daemon_module, args, vllm_config)
        # This path is retained for the existing, installed daemon behavior.
        return daemon_module.main()
    if vllm_config.load_config.load_format == "ipc_cache":
        raise ValueError(
            "the weight cache daemon must load from disk; do not pass "
            "--load-format ipc_cache to the daemon"
        )
    if parallel.data_parallel_size != 1:
        raise ValueError("pipeline weight cache currently requires data_parallel_size=1")
    if parallel.enable_expert_parallel:
        raise ValueError("pipeline weight cache currently does not support expert parallelism")

    spec = getattr(vllm_config, "speculative_config", None)
    if spec is not None:
        _validate_pipeline_spec(spec, tp_size, pp_size)

    daemon_module.check_ipc_quant_support(vllm_config.model_config, where="daemon")
    if spec is not None:
        daemon_module.check_ipc_quant_support(
            spec.draft_model_config, where="daemon draft"
        )
    nnodes = int(parallel.nnodes)
    node_rank = int(parallel.node_rank)
    world_size = tp_size * pp_size
    if not 0 <= node_rank < nnodes:
        raise ValueError(f"--node-rank must be in [0, {nnodes}), got {node_rank}")
    if world_size % nnodes:
        raise ValueError(
            f"TP*PP world size ({world_size}) must be divisible by --nnodes ({nnodes})"
        )
    local_world_size = world_size // nnodes
    if parallel.master_addr is None:
        raise ValueError("--master-addr is required for pipeline weight cache")
    if nnodes > 1 and args.weight_cache_master_port is None:
        raise ValueError("--weight-cache-master-port is required when --nnodes > 1")
    master_port = args.weight_cache_master_port or get_open_port()
    init_method = get_distributed_init_method(parallel.master_addr, master_port)

    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    processes = []
    for global_rank, local_rank, _, _ in plan_local_pipeline_ranks(
        tp_size, pp_size, nnodes, node_rank
    ):
        process = context.Process(
            target=_run_pipeline_daemon,
            args=(
                global_rank,
                local_rank,
                vllm_config,
                init_method,
                args.weight_cache_socket_dir,
                ready_queue,
                tp_size,
                pp_size,
            ),
            name=f"vllm-weight-cache-daemon-pp{global_rank}",
        )
        process.start()
        processes.append(process)

    stopping = False

    def shutdown(_signum, _frame):
        nonlocal stopping
        stopping = True
        for process in processes:
            process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if not _wait_until_ready(processes, ready_queue, local_world_size, lambda: stopping):
        return
    print(
        f"Weight cache daemon READY: node {node_rank}/{nnodes}, "
        f"{local_world_size} local TP/PP ranks, PP={pp_size}",
        flush=True,
    )
    _monitor_processes(processes, lambda: stopping)


def _wait_until_ready(processes, ready_queue, local_world_size, is_stopping):
    ready = set()
    while len(ready) < local_world_size:
        if is_stopping():
            break
        try:
            ready.add(ready_queue.get(timeout=1.0))
        except queue.Empty:
            if is_stopping():
                break
            dead = [process for process in processes if process.exitcode is not None]
            if dead:
                for process in processes:
                    process.terminate()
                for process in processes:
                    process.join()
                raise SystemExit(max(process.exitcode or 1 for process in dead))
    if is_stopping():
        for process in processes:
            process.join()
        return False
    return True


def _monitor_processes(processes, is_stopping):
    """Stop the local group when any indefinitely serving child exits."""
    ready = set(wait([process.sentinel for process in processes]))
    failed = [process for process in processes if process.sentinel in ready]
    for process in failed:
        process.join()
    for process in processes:
        if process.exitcode is None:
            process.terminate()
    for process in processes:
        process.join()
    if not is_stopping():
        names = ", ".join(f"{p.name} (exit={p.exitcode})" for p in failed)
        raise SystemExit(f"Weight cache daemon exited unexpectedly: {names}")


def plan_local_pipeline_ranks(
    tp_size: int,
    pp_size: int,
    nnodes: int,
    node_rank: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(global, local, pp, tp)`` ranks for one node."""
    if tp_size < 1 or pp_size < 1 or nnodes < 1:
        raise ValueError("TP, PP and node count must be positive")
    world_size = tp_size * pp_size
    if not 0 <= node_rank < nnodes:
        raise ValueError(f"node rank {node_rank} is outside [0, {nnodes})")
    if world_size % nnodes:
        raise ValueError("TP*PP world size must be divisible by node count")
    local_world_size = world_size // nnodes
    first_global_rank = node_rank * local_world_size
    return [
        (
            global_rank,
            local_rank,
            global_rank // tp_size,
            global_rank % tp_size,
        )
        for local_rank, global_rank in enumerate(
            range(first_global_rank, first_global_rank + local_world_size)
        )
    ]
