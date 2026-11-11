# SPDX-License-Identifier: Apache-2.0
"""Compatibility fixes for the vLLM 0.28 weight cache daemon."""
from __future__ import annotations

from types import ModuleType
from contextlib import contextmanager
from contextvars import ContextVar

_cache_moe_backend = ContextVar("hcu_weight_cache_moe_backend", default=None)
_cache_parallel_config = ContextVar("hcu_weight_cache_parallel_config", default=None)
_cache_model_role = ContextVar("hcu_weight_cache_model_role", default=None)
_cache_meta_construction = ContextVar("hcu_weight_cache_meta_construction", default=False)


def _dspark_cache_role(vllm_config, model_config=None):
    """Distinguish DSpark target and draft without guessing from a path."""
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "dspark":
        return None
    target = getattr(vllm_config, "model_config", None)
    draft = getattr(spec, "draft_model_config", None)
    if target is not None and draft is target:
        raise RuntimeError("DSpark target and draft model configurations must be distinct")
    if model_config is None or model_config is target:
        return "target"
    if draft is not None and model_config is draft:
        return "draft"
    raise RuntimeError("cannot identify DSpark weight cache model role")


@contextmanager
def _weight_cache_config_scope(vllm_config, model_config=None, model_role=None,
                               parallel_config=None):
    kernel_config = getattr(vllm_config, "kernel_config", None)
    backend = getattr(kernel_config, "moe_backend", None)
    if backend is None:
        raise RuntimeError("weight cache requires kernel_config.moe_backend")
    role = model_role if model_role is not None else _dspark_cache_role(vllm_config, model_config)
    token = _cache_moe_backend.set(str(backend))
    parallel_token = _cache_parallel_config.set(
        parallel_config if parallel_config is not None
        else getattr(vllm_config, "parallel_config", None)
    )
    role_token = _cache_model_role.set(role)
    try:
        yield
    finally:
        _cache_model_role.reset(role_token)
        _cache_parallel_config.reset(parallel_token)
        _cache_moe_backend.reset(token)


# Fingerprints computed before the target model is post-processed, keyed by
# ``(id(model_config), role)``. Target loading mutates shared HF/quantization
# state that the draft fingerprint reads, so the draft key must be captured
# first -- exactly as the daemon does.
_precomputed_cache_keys = {}


def _precompute_draft_key(module, vllm_config):
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or getattr(spec, "method", None) != "dspark":
        return
    draft_config = getattr(spec, "draft_model_config", None)
    draft_parallel = getattr(spec, "draft_parallel_config", None)
    if draft_config is None or draft_parallel is None:
        return
    if (id(draft_config), "draft") in _precomputed_cache_keys:
        return
    parallel = getattr(vllm_config, "parallel_config", None)
    tp_size = int(getattr(parallel, "tensor_parallel_size", 1) or 1)
    tp_rank = int(getattr(parallel, "tensor_parallel_rank", 0) or 0)
    from vllm.distributed import (
        get_tensor_model_parallel_rank, model_parallel_is_initialized,
    )
    if model_parallel_is_initialized():
        tp_rank = get_tensor_model_parallel_rank()
    with _weight_cache_config_scope(
        vllm_config, draft_config, model_role="draft", parallel_config=draft_parallel
    ):
        key = module.WeightCacheKey.from_model_config(
            draft_config, tp_size=tp_size, tp_rank=tp_rank
        )
    _precomputed_cache_keys[(id(draft_config), "draft")] = key


def _install_cache_config_scope(module):
    import functools

    for class_name, method_name in (("WeightCacheDaemon", "__init__"),
                                    ("IpcModelLoader", "load_model")):
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        method = getattr(cls, method_name, None)
        if not callable(method):
            raise RuntimeError(f"Incompatible weight cache: missing {class_name}.{method_name}")
        if getattr(method, "_hcu_cache_config_scope", False):
            continue

        def wrap(original, is_loader):
            @functools.wraps(original)
            def scoped(self, vllm_config, *args, **kwargs):
                model_config = None
                if is_loader:
                    model_config = kwargs.get("model_config", args[0] if args else None)
                role = _dspark_cache_role(vllm_config, model_config)
                # The target loads first in the engine; fingerprint its draft
                # before target post-processing can change the inputs.
                if is_loader and role == "target":
                    _precompute_draft_key(module, vllm_config)
                with _weight_cache_config_scope(vllm_config, model_config,
                                                model_role=role):
                    return original(self, vllm_config, *args, **kwargs)

            scoped._hcu_cache_config_scope = True
            return scoped

        setattr(cls, method_name, wrap(method, class_name == "IpcModelLoader"))


def reconcile_router_state_devices(model, target_device):
    """Move router state tensors that are not registered module state.

    Some Kimi/AMD router implementations replace the bias with a plain Tensor
    attribute (or reassign it after construction). ``Module.to`` only visits
    registered parameters and buffers, so an IPC client can otherwise invoke
    ``moe_fused_gate(router_logits, bias)`` with the activation on its local
    GPU and the bias left on CPU/meta.
    """
    import torch

    target = torch.device(target_device)
    # attribute name -> owning gate attribute when the names differ
    names = {
        "e_score_correction_bias": "e_score_correction_bias",
        "expert_bias": "expert_bias",
        "router_bias": "router_bias",
        # DeepSeek V4 hash-MoE layers capture gate.tid2eid at construction;
        # IPC loading rebinds the gate Parameter, leaving the router on meta.
        "_hash_indices_table": "tid2eid",
    }
    # vLLM's FusedMoERouter is a plain Python object owned by MoERunner, not an
    # nn.Module. Its bias reference is therefore invisible to named_modules()
    # and can remain bound to the constructor's stale/default-device tensor
    # after the gate Parameter has been loaded onto this rank.
    modules = dict(model.named_modules())
    owners = []
    for module_name, module in modules.items():
        owners.append((module_name, module))
        router = getattr(module, "router", None)
        if router is not None and not isinstance(router, torch.nn.Module):
            owners.append((f"{module_name}.router" if module_name else "router", router))

    moved = []
    for module_name, module in owners:
        for name, gate_name in names.items():
            if not hasattr(module, name):
                continue
            value = getattr(module, name)
            if not isinstance(value, torch.Tensor):
                continue

            # A FusedMoERouter is held by the sibling MoERunner (usually the
            # ``...experts`` module), but is not itself an nn.Module. IPC
            # loading replaces the registered gate Parameter, leaving this
            # plain router object with the original meta Parameter reference.
            # Rebind it to the loaded sibling gate tensor before device checks.
            if not isinstance(module, torch.nn.Module) and module_name.endswith(
                ".router"
            ):
                runner_name = module_name[: -len(".router")]
                parent_name = runner_name.rpartition(".")[0]
                gate = modules.get(f"{parent_name}.gate")
                gate_value = getattr(gate, gate_name, None) if gate is not None else None
                if isinstance(gate_value, torch.Tensor):
                    value = gate_value
                    setattr(module, name, value)
                    moved.append(f"{module_name}.{name}")

            if value.device == target:
                continue
            if value.is_meta:
                raise RuntimeError(
                    f"router state {module_name}.{name} is still meta and has no "
                    "loaded sibling gate tensor"
                )
            replacement = value.to(target)
            # Preserve Parameter registration and object semantics where
            # possible; plain Tensor attributes are simply replaced.
            if isinstance(value, torch.nn.Parameter):
                if value.is_meta or target.type == "meta":
                    # Meta Parameters have no storage to rebind.
                    setattr(module, name, torch.nn.Parameter(
                        replacement, requires_grad=value.requires_grad
                    ))
                else:
                    # Keep aliases held by router objects and the owning gate
                    # attached to the same Parameter after local placement.
                    value.data = replacement
            else:
                setattr(module, name, replacement)
            moved.append(f"{module_name}.{name}" if module_name else name)
    return tuple(moved)


def _pipeline_context() -> tuple[int, int]:
    """Return ``(pipeline_size, pipeline_rank)`` without forcing PP init.

    The daemon launcher sets the environment values before constructing its
    cache object.  Engine workers resolve the same values from vLLM's model
    parallel group after distributed initialization.  Keeping this helper
    best-effort is important because the weight-cache modules are imported
    during plugin bootstrap, before model-parallel groups exist.
    """
    import os

    size = int(os.environ.get("VLLM_HCU_PRELOAD_PP_SIZE", "1"))
    rank = int(os.environ.get("VLLM_HCU_PRELOAD_PP_RANK", "0"))
    try:
        from vllm.distributed import get_pp_group

        group = get_pp_group()
        size = int(getattr(group, "world_size", size))
        rank = int(getattr(group, "rank_in_group", rank))
    except Exception:
        # The group is legitimately unavailable while the daemon constructs
        # its cache key and while the protocol modules are imported.
        pass
    if size < 1 or not 0 <= rank < size:
        raise RuntimeError(f"invalid pipeline cache context size={size} rank={rank}")
    return size, rank


def _pp_socket_identifier(identifier: str) -> str:
    """Add the PP stage to a socket identifier when PP is enabled."""
    size, rank = _pipeline_context()
    return f"{identifier}-pp{rank}" if size > 1 else identifier


def _validate_expert_cache_layouts(model):
    """Reject unsupported postprocessed layouts before reporting cache ready."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    for module_name, owner in named_modules(remove_duplicate=False):
        layout = getattr(owner, "_dsv4_channel_deepgemm_layout", None)
        if layout is None:
            continue
        if layout not in ("contiguous", "contiguous=masked"):
            raise RuntimeError(f"unsupported DeepGEMM IPC layout at {module_name}: {layout}")
        if getattr(owner, "w13_weight", None) is None:
            raise RuntimeError(f"DeepGEMM IPC layout missing weight at {module_name}")


def install_per_client_export(module: ModuleType) -> None:
    """Export fresh IPC reductions for each successful client request.

    Each reduction carries a PyTorch IPC reference. Sharing one reduction
    between independent clients can free it while another client still uses it.
    Keep the model itself alive and create handles only when serving a client.
    """
    cls = module.WeightCacheDaemon
    if cls.__dict__.get("_hcu_per_client_export", False):
        return
    for name in ("_export_entries", "_handle_get_state", "_handle_release"):
        if not callable(getattr(cls, name, None)):
            raise RuntimeError(f"Incompatible weight cache daemon: missing {name}")
    for name in ("WeightCacheKey", "export_entries", "send_msg"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"Incompatible weight cache daemon module: missing {name}")

    def mark_loaded(self):
        if self.model is None:
            raise RuntimeError("weight cache model was not loaded")
        # Preserve the original loader's tensor-count log without creating
        # IPC reductions until a matching client requests the state.
        _validate_expert_cache_layouts(self.model)
        names = {name for name, _ in self.model.named_parameters()}
        names.update(name for name, _ in self.model.named_buffers())
        self.entries = dict.fromkeys(names)
        self.aliases = {}

    def handle_get_state(self, conn, request):
        client_config = request.get("cache_config")
        if not isinstance(client_config, module.WeightCacheKey):
            module.send_msg(conn, {"status": "error", "message": "Missing cache_config"})
            return
        mismatched = self.cache_config.mismatched_fields(client_config)
        if mismatched:
            module.send_msg(conn, {"status": "mismatch", "fields": mismatched})
            return
        if self.model is None:
            module.send_msg(conn, {"status": "error", "message": "Weights were released"})
            return
        try:
            entries, aliases = module.export_entries(self.model)
        except Exception as exc:
            module.send_msg(conn, {"status": "error", "message": f"cache export failed: {exc}"})
            return
        module.send_msg(conn, {
            "status": "ok", "entries": entries, "aliases": aliases,
            "gpu_uuid": self._gpu_uuid(),
        })

    cls._export_entries = mark_loaded
    cls._handle_get_state = handle_get_state
    cls._hcu_per_client_export = True


def install_expert_export_metadata(module: ModuleType) -> None:
    """Preserve the explicit unpacked MoE layout marker in exported entries."""
    original = module.export_entries
    if getattr(original, "_hcu_expert_metadata", False):
        return

    def export_entries(model):
        entries, aliases = original(model)
        metadata_names = (
            "_hcu_vllm_w4a8_unpacked",
            "_hcu_aiter_moe_solution_type",
            "_hcu_aiter_moe_weight_layout",
            "_hcu_aiter_moe_logical_shape",
            "_hcu_aiter_moe_scale_layout",
            "is_shuffled",
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            if name not in entries:
                continue
            for attr in metadata_names:
                value = getattr(param, attr, None)
                if value is not None:
                    setattr(entries[name], attr, value)
        # The packed auto backend keeps its layout marker on the module, not
        # on registered tensors. Attach it to the exported w13 entry so the
        # client can restore the backend state before postprocessing.
        for module_name, owner in model.named_modules(remove_duplicate=False):
            layout = getattr(owner, "_dsv4_channel_deepgemm_layout", None)
            if layout is None:
                continue
            if layout not in ("contiguous", "contiguous=masked"):
                raise RuntimeError(f"unsupported DeepGEMM IPC layout: {layout}")
            name = f"{module_name}.w13_weight" if module_name else "w13_weight"
            canonical = aliases.get(name, name)
            entry = entries.get(canonical)
            if entry is None:
                raise RuntimeError(f"DeepGEMM IPC layout missing tensor {name}")
            existing = getattr(entry, "_hcu_dsv4_deepgemm_layout", layout)
            if existing != layout:
                raise RuntimeError(f"conflicting DeepGEMM IPC layouts for {name}")
            entry._hcu_dsv4_deepgemm_layout = layout
        return entries, aliases

    export_entries._hcu_expert_metadata = True
    module.export_entries = export_entries


def _is_channel_fp8_quant_config(quant_config) -> bool:
    """Match only the channel-weight/token-activation FP8 scheme validated on HCU.

    The allowlist predicate only decides whether the *scheme* is one whose
    post-load effect can travel through tensor export. Per-model details that
    tensor export cannot carry (kernel backend, packed expert layout, topology)
    stay fail-closed in the quantization patches, which re-check them from the
    live module state and raise instead of silently double-processing.
    """
    if not isinstance(quant_config, dict):
        return False
    if quant_config.get("format") != "float-quantized":
        return False
    if "weight_block_size" in quant_config:
        return False
    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        return False
    group = next(iter(groups.values()))
    if not isinstance(group, dict) or group.get("targets") != ["Linear"]:
        return False
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, dict) or not isinstance(activations, dict):
        return False
    return (
        weights.get("num_bits") == 8
        and weights.get("type") == "float"
        and weights.get("strategy") == "channel"
        and weights.get("symmetric") is True
        and weights.get("group_size") is None
        and weights.get("dynamic") is False
        and activations.get("num_bits") == 8
        and activations.get("type") == "float"
        and activations.get("strategy") == "token"
        and activations.get("symmetric") is True
        and activations.get("group_size") is None
        and activations.get("dynamic") is True
    )


def install_ipc_quant_support(module: ModuleType) -> None:
    """Allow the verified SlimQuant W4A8 and channel-FP8 schemes through IPC."""
    allowlist = getattr(module, "IPC_QUANT_ALLOWLIST", None)
    if not isinstance(allowlist, dict):
        raise RuntimeError("Incompatible weight cache protocol: missing IPC allowlist")
    # Install each entry independently so a partially applied install (or a
    # newer entry added later) never masks the other one.
    existing = allowlist.get("slimquant_w4a8")
    if not getattr(existing, "_hcu_slimquant_ipc", False):
        def slimquant_w4a8_ipc_enabled(_quant_config):
            return True

        slimquant_w4a8_ipc_enabled._hcu_slimquant_ipc = True
        allowlist["slimquant_w4a8"] = slimquant_w4a8_ipc_enabled

    existing = allowlist.get("compressed-tensors")
    if not getattr(existing, "_hcu_compressed_tensors_ipc", False):
        def compressed_tensors_ipc_enabled(quant_config):
            return _is_channel_fp8_quant_config(quant_config)

        compressed_tensors_ipc_enabled._hcu_compressed_tensors_ipc = True
        allowlist["compressed-tensors"] = compressed_tensors_ipc_enabled


def _bounded_socket_path(get_socket_path, identifier, socket_dir):
    """Keep UUID/PP identity while reserving AF_UNIX space for the draft role."""
    import hashlib
    import os

    path = get_socket_path(identifier, socket_dir)
    limit = 107 - len(".draft")
    if len(os.fsencode(path)) <= limit:
        return path
    directory = os.path.dirname(path)
    # The directory is operator-selected; never silently move the socket.
    available = limit - len(os.fsencode(directory)) - 1 - len("wc-.sock")
    if available < 16:
        raise ValueError(
            f"Weight cache socket directory is too long: {directory!r}; "
            "choose a shorter directory (at least 64 bits of socket identity required)"
        )
    digest = hashlib.sha256(os.fsencode(os.path.basename(path))).hexdigest()
    return os.path.join(directory, f"wc-{digest[:min(32, available)]}.sock")


def validate_preload_socket_dir(socket_dir):
    """Check ownership, AF_UNIX length and filesystem support before loading."""
    import os
    import secrets
    import socket
    from vllm.model_executor.model_loader.weight_cache import protocol

    directory = protocol.get_socket_dir(socket_dir)
    protocol.ensure_private_socket_dir(directory)
    identifier = "preflight-" + secrets.token_hex(24)
    path = _bounded_socket_path(protocol.get_socket_path, identifier, directory) + ".draft"
    sock = socket.socket(socket.AF_UNIX)
    bound = False
    try:
        sock.bind(path)
        bound = True
    finally:
        sock.close()
        if bound:
            os.unlink(path)
    return directory


def _uuid_socket_path(torch_module, get_socket_path, socket_dir):
    """Use the UUID of the current logical device, independent of visibility order."""
    import re

    index = torch_module.accelerator.current_device_index()
    props = torch_module.cuda.get_device_properties(index)
    value = getattr(props, "uuid", None)
    if value is None or not re.fullmatch(r"[A-Za-z0-9_-]+", str(value)):
        raise RuntimeError("weight cache requires a stable, filesystem-safe GPU UUID")
    return _bounded_socket_path(
        get_socket_path, _pp_socket_identifier("uuid-" + str(value)), socket_dir
    )


def install_daemon_workspace(module: ModuleType) -> None:
    """Match engine workspace initialization before TP daemon model construction."""
    import functools

    cls = module.WeightCacheDaemon
    original = cls.load_model
    if getattr(original, "_hcu_preload_workspace", False):
        return

    @functools.wraps(original)
    def load_model(self):
        from vllm.v1.worker.workspace import (
            init_workspace_manager, is_workspace_manager_initialized,
        )

        module.torch.accelerator.set_device_index(self.tp_rank)
        if not is_workspace_manager_initialized():
            parallel = self.vllm_config.parallel_config
            spec = getattr(self.vllm_config, "speculative_config", None)
            init_workspace_manager(
                module.torch.device("cuda", self.tp_rank),
                num_ubatches=2 if getattr(parallel, "enable_dbo", False) else 1,
                num_lanes=2 if getattr(spec, "method", None) == "dspark" else 1,
            )
        return original(self)

    load_model._hcu_preload_workspace = True
    cls.load_model = load_model


def install_uuid_daemon(module: ModuleType) -> None:
    cls = module.WeightCacheDaemon
    if cls.__dict__.get("_hcu_uuid_socket", False):
        return
    if not callable(getattr(cls, "_socket_path", None)):
        raise RuntimeError("Incompatible weight cache daemon socket interface")

    def socket_path(self):
        path = _uuid_socket_path(module.torch, module.get_socket_path, self.socket_dir)
        return path + (".draft" if getattr(self, "cache_role", None) == "draft" else "")

    cls._socket_path = socket_path
    cls._hcu_uuid_socket = True


_MANIFEST_SUFFIX = ".meta.json"


def _live_process_start_time(pid):
    """Return Linux birth ticks, rejecting exited and zombie cache owners."""
    from pathlib import Path

    fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X"):
        raise ProcessLookupError(f"cache owner {pid} is {fields[0]}")
    return int(fields[19])


def _device_memory_usage():
    """Return ``(used_bytes, total_bytes)`` for the current accelerator device."""
    import torch

    device = torch.accelerator.current_device_index()
    free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
    return int(total_bytes - free_bytes), int(total_bytes)


def release_all2all_buffers():
    """Destroy every all2all (DeepEP) buffer owned by this process.

    A preload daemon only exports tensors; it never runs a forward, so the
    symmetric/HT/LL communication buffers built while the model was
    constructed are dead weight. They are created with ``explicitly_destroy``
    support by vLLM, so releasing them before serving keeps the resident
    footprint close to the exported weights instead of duplicating the
    serving engine's communicator workspaces.
    """
    import gc

    import torch

    released = []
    seen = set()
    for obj in list(gc.get_objects()):
        cls = type(obj)
        if id(obj) in seen:
            continue
        if not cls.__name__.endswith("All2AllManager"):
            continue
        if not str(getattr(cls, "__module__", "")).startswith("vllm"):
            continue
        destroy = getattr(obj, "destroy", None)
        if not callable(destroy):
            continue
        seen.add(id(obj))
        destroy()
        released.append(cls.__name__)
    gc.collect()
    torch.accelerator.empty_cache()
    return tuple(released)


def _model_storages(model):
    """Return unique ``(storage_data_ptr, nbytes)`` pairs held by the model."""
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.is_meta:
            continue
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer == 0:
            continue
        # Shared storages (tied embeddings, aliased draft weights) count once.
        storages[pointer] = storage.nbytes()
    return sorted(storages.items())


def write_cache_manifest(daemon) -> str:
    """Record the daemon's resident footprint next to its socket.

    The engine maps these tensors zero-copy, so they occupy device memory
    before the engine takes its startup snapshot. Without this manifest the
    engine's ``gpu-memory-utilization`` budget would ignore them and either
    refuse to start or over-allocate KV cache.
    """
    import gc
    import json
    import os

    import torch

    model = getattr(daemon, "model", None)
    if model is None:
        raise RuntimeError("cannot cache the manifest of a released daemon")
    socket_path = daemon._socket_path()
    storages = _model_storages(model)
    # Loading and post-processing leave allocator-cached blocks that the
    # engine can never share. Return them before measuring and serving.
    gc.collect()
    torch.accelerator.empty_cache()
    used_bytes, total_bytes = _device_memory_usage()
    record = {
        "version": 1,
        "pid": os.getpid(),
        "pid_start_time": _live_process_start_time(os.getpid()),
        "gpu_uuid": daemon._gpu_uuid(),
        "role": "draft" if getattr(daemon, "cache_role", None) == "draft" else "target",
        "dp_rank": int(getattr(daemon.vllm_config.parallel_config,
                               "data_parallel_rank", 0) or 0),
        "checkpoint_fingerprint": daemon.cache_config.checkpoint,
        # Device memory this daemon process group already holds, including
        # allocator caches and communicator buffers. Roles served by one rank
        # process share the device, so the engine takes the maximum instead of
        # summing them.
        "device_used_bytes": int(used_bytes),
        "device_total_bytes": int(total_bytes),
        "storages": [[pointer, nbytes] for pointer, nbytes in storages],
        "model_footprint_bytes": sum(nbytes for _, nbytes in storages),
    }
    path = socket_path + _MANIFEST_SUFFIX
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w") as handle:
        json.dump(record, handle)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return path


def install_cache_manifest(module: ModuleType) -> None:
    """Write the resident footprint manifest whenever a daemon starts serving."""
    import functools

    cls = module.WeightCacheDaemon
    original = cls.serve_forever
    if getattr(original, "_hcu_cache_manifest", False):
        return

    @functools.wraps(original)
    def serve_forever(self, ready_callback=None):
        write_cache_manifest(self)
        return original(self, ready_callback)

    serve_forever._hcu_cache_manifest = True
    cls.serve_forever = serve_forever


def _read_cache_manifest(path, gpu_uuid, role):
    import json
    import os

    try:
        with open(path) as handle:
            record = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"weight cache manifest {path} is missing; restart the preload "
            "daemon with the current plugin before starting an ipc_cache engine"
        ) from exc
    if record.get("version") != 1:
        raise RuntimeError(f"unsupported weight cache manifest version in {path}")
    if record.get("gpu_uuid") != gpu_uuid:
        raise RuntimeError(
            f"weight cache manifest {path} belongs to device "
            f"{record.get('gpu_uuid')!r}, not {gpu_uuid!r}"
        )
    if record.get("role") != role:
        raise RuntimeError(
            f"weight cache manifest {path} is for role {record.get('role')!r}, "
            f"not {role!r}"
        )
    try:
        os.kill(int(record["pid"]), 0)
        started = _live_process_start_time(record["pid"])
        if record.get("pid_start_time") != started:
            raise ProcessLookupError("cache owner PID was reused or manifest is stale")
    except (KeyError, ValueError, OSError, IndexError) as exc:
        raise RuntimeError(f"weight cache manifest {path} has no live daemon") from exc
    storages = record.get("storages")
    if not isinstance(storages, list):
        raise RuntimeError(f"weight cache manifest {path} has no tensor list")
    return record


def budget_snapshot_with_resident(snapshot, resident_bytes):
    """Copy the startup snapshot with daemon-resident memory credited as free.

    ``MemorySnapshot.__post_init__`` re-measures when ``auto_measure`` is true,
    so the replacement must switch measurement off or it silently discards the
    adjustment.
    """
    import dataclasses

    return dataclasses.replace(
        snapshot,
        free_memory=snapshot.free_memory + resident_bytes,
        auto_measure=False,
    )


def ipc_cache_resident_memory(vllm_config, gpu_uuid=None, socket_dir=None):
    """Bytes of model weights the weight-cache daemons already hold on this device.

    Returns 0 for disk loading and copy mode, where the engine allocates its
    own weights after the startup snapshot.
    """
    load_config = getattr(vllm_config, "load_config", None)
    if getattr(load_config, "load_format", None) != "ipc_cache":
        return 0
    extra = getattr(load_config, "model_loader_extra_config", None) or {}
    if extra.get("mode", "zero_copy") != "zero_copy":
        return 0

    import torch
    from vllm.model_executor.model_loader.weight_cache import protocol

    if gpu_uuid is None:
        index = torch.accelerator.current_device_index()
        gpu_uuid = str(torch.cuda.get_device_properties(index).uuid)
    if socket_dir is None:
        socket_dir = extra.get("socket_dir")
    parallel = getattr(vllm_config, "parallel_config", None)
    pp_size = int(getattr(parallel, "pipeline_parallel_size", 1) or 1)
    pp_rank = int(getattr(parallel, "pipeline_parallel_rank", 0) or 0)
    group_size, group_rank = _pipeline_context()
    if group_size > 1:
        pp_size, pp_rank = group_size, group_rank
    identifier = "uuid-" + str(gpu_uuid)
    if pp_size > 1:
        identifier = f"{identifier}-pp{pp_rank}"
    base = _bounded_socket_path(protocol.get_socket_path, identifier, socket_dir)

    roles = [("target", base)]
    spec = getattr(vllm_config, "speculative_config", None)
    if (spec is not None and getattr(spec, "method", None) == "dspark"
            and pp_rank == pp_size - 1):
        roles.append(("draft", base + ".draft"))

    resident = 0
    for role, role_base in roles:
        record = _read_cache_manifest(
            role_base + _MANIFEST_SUFFIX, str(gpu_uuid), role
        )
        # Both roles of one rank are served by the same process on the same
        # device, so their device footprints overlap: take the maximum.
        resident = max(resident, int(record.get("device_used_bytes") or 0))
    if resident <= 0:
        raise RuntimeError(
            "weight cache manifest reports no resident device memory; "
            "restart the preload daemon with the current plugin"
        )
    return resident


def install_meta_expert_map_compat(module: ModuleType) -> None:
    """Build small data-dependent expert maps on CPU during IPC meta init."""
    import functools
    import torch

    cls = module.ExpertMapManager
    original = cls.__init__
    if getattr(original, "_hcu_ipc_meta_expert_map", False):
        return

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        if _cache_meta_construction.get():
            with torch.device("cpu"):
                return original(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    init._hcu_ipc_meta_expert_map = True
    cls.__init__ = init


def _rebind_expert_map_managers(model):
    """Rebind manager aliases to IPC-rebuilt registered routing buffers."""
    for owner in model.modules():
        manager = getattr(owner, "expert_map_manager", None)
        if manager is None or not hasattr(owner, "_expert_map"):
            continue
        manager._expert_map = owner._expert_map
        manager._expert_mask = getattr(owner, "expert_mask", None)
        names = ("expert_global_to_physical", "expert_physical_to_global", "expert_local_to_global")
        if getattr(manager, "_routing_tables", None) is not None:
            tables = tuple(getattr(owner, name, None) for name in names)
            if any(table is None for table in tables):
                raise RuntimeError("weight cache is missing expert routing tables")
            manager._routing_tables = tables


def _draft_entries_before_target_sharing(model, entries, aliases):
    """Skip target-owned placeholders that DSpark binds after draft loading.

    K3 exports the shared embedding/head as children in the daemon, but its
    constructor intentionally leaves them None. They are already present in
    the target IPC model and load_dspark_model will bind those same modules.
    """
    prefixes = []
    inner = getattr(model, "model", None)
    if (getattr(model, "has_own_embed_tokens", True) is False
            and inner is not None and getattr(inner, "embed_tokens", False) is None):
        prefixes.append("model.embed_tokens.")
    if (getattr(model, "has_own_lm_head", True) is False
            and getattr(model, "lm_head", False) is None):
        prefixes.append("lm_head.")
    if not prefixes:
        return entries, aliases
    prefixes = tuple(prefixes)
    entries = {name: value for name, value in entries.items()
               if not name.startswith(prefixes)}
    remaining_aliases = {}
    for name, canonical in aliases.items():
        if name.startswith(prefixes):
            continue
        if canonical.startswith(prefixes):
            raise RuntimeError(f"Draft-owned alias {name} refers to target-owned {canonical}")
        remaining_aliases[name] = canonical
    return entries, remaining_aliases


def install_loader_compat(module: ModuleType) -> None:
    """Use UUID sockets and bounded startup waiting for strict cache loads."""
    import functools
    import time

    cls = module.IpcModelLoader
    if cls.__dict__.get("_hcu_weight_cache_loader", False):
        return
    original_build = getattr(cls, "_build_model", None)
    original_fetch = getattr(cls, "_fetch_entries", None)
    original_connect = getattr(cls, "_connect", None)
    original_apply = getattr(cls, "_apply_entries", None)
    original_check = getattr(cls, "_check_supported", None)
    original_materialize = getattr(module, "_materialize_remaining_meta_tensors", None)
    if (not callable(original_fetch)
            or not callable(original_connect)
            or not callable(original_apply)
            or not callable(original_check)
            or not callable(getattr(cls, "_resolve_socket_path", None))
            or not callable(original_materialize)):
        raise RuntimeError("Incompatible weight cache loader interface")

    def materialize_remaining(model, target_device):
        # The daemon exports parameters and buffers, including non-persistent
        # buffers. A remaining meta tensor can otherwise become uninitialized
        # storage and silently corrupt model outputs.
        missing = [name for name, tensor in model.named_parameters(remove_duplicate=False)
                   if tensor.is_meta]
        missing += [name for name, tensor in model.named_buffers(remove_duplicate=False)
                    if tensor.is_meta]
        if missing:
            # A generic failure routes copy-mode loads through the original
            # loader's release path before any disk fallback.
            raise RuntimeError(
                "Weight cache is missing model tensors: " + ", ".join(missing[:8])
            )
        return original_materialize(model, target_device)

    def apply_entries(self, model, entries, aliases, device_index):
        if _cache_model_role.get() == "draft":
            entries, aliases = _draft_entries_before_target_sharing(model, entries, aliases)
        result = original_apply(self, model, entries, aliases, device_index)
        params = dict(model.named_parameters(remove_duplicate=False))
        for name, entry in entries.items():
            for attr in (
                "_hcu_vllm_w4a8_unpacked",
                "_hcu_aiter_moe_solution_type",
                "_hcu_aiter_moe_weight_layout",
                "_hcu_aiter_moe_logical_shape",
                "_hcu_aiter_moe_scale_layout",
                "_hcu_dsv4_deepgemm_layout",
                "is_shuffled",
            ):
                value = getattr(entry, attr, None)
                if value is not None:
                    param = params.get(name)
                    if param is None:
                        raise RuntimeError(f"Expert parameter {name!r} is missing")
                    setattr(param, attr, value)
            if getattr(entry, "_hcu_vllm_w4a8_unpacked", False) is True:
                param = params.get(name)
                if param is None:
                    raise RuntimeError(f"Unpacked expert parameter {name!r} is missing")
                param._hcu_vllm_w4a8_unpacked = True
        _rebind_expert_map_managers(model)
        return result

    def check_supported(vllm_config, model_config):
        from vllm.platforms import current_platform
        if not current_platform.is_cuda_alike():
            raise RuntimeError("weight-cache IPC requires a CUDA/ROCm-compatible platform")
        return original_check(vllm_config, model_config)

    def resolve_socket_path(self):
        role = _cache_model_role.get()
        if self.socket_path is not None:
            return self.socket_path + (".draft" if role == "draft" else "")
        try:
            path = _uuid_socket_path(module.torch, module.get_socket_path, self.socket_dir)
            return path + (".draft" if role == "draft" else "")
        except (RuntimeError, AttributeError) as exc:
            raise module.WeightCacheUnavailableError(
                f"Cannot identify weight cache GPU UUID: {exc}"
            ) from exc

    @functools.wraps(original_connect)
    def connect(self, timeout):
        # _request_state passes state_timeout_s to _connect. In strict mode,
        # cap that socket timeout to the remaining outer deadline so one
        # unavailable daemon cannot overrun the advertised startup timeout.
        deadline = getattr(self, "_hcu_fetch_deadline", None)
        if deadline is not None:
            timeout = min(timeout, max(0.0, deadline - time.monotonic()))
        return original_connect(self, timeout)

    @functools.wraps(original_fetch)
    def fetch_entries(self, model_config):
        if self.fallback:
            return original_fetch(self, model_config)
        deadline = time.monotonic() + max(0.0, self.state_timeout_s)
        self._hcu_fetch_deadline = deadline
        try:
            while True:
                try:
                    return original_fetch(self, model_config)
                except module.WeightCacheUnavailableError as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise module.WeightCacheUnavailableError(
                            "Weight cache daemon did not become ready within "
                            f"{self.state_timeout_s:.1f}s"
                        ) from exc
                    time.sleep(min(0.5, remaining))
        finally:
            delattr(self, "_hcu_fetch_deadline")

    if callable(original_build):
        @functools.wraps(original_build)
        def build_model(self, *args, **kwargs):
            from vllm.model_executor.layers.fused_moe import expert_map_manager
            install_meta_expert_map_compat(expert_map_manager)
            config = args[0] if args else kwargs.get("vllm_config")
            parallel = getattr(config, "parallel_config", None)
            use_cpu_maps = bool(
                parallel is not None
                and getattr(parallel, "enable_expert_parallel", False)
                and getattr(parallel, "data_parallel_size", 1) > 1
            )
            token = _cache_meta_construction.set(use_cpu_maps)
            try:
                return original_build(self, *args, **kwargs)
            finally:
                _cache_meta_construction.reset(token)

        cls._build_model = build_model

    module._materialize_remaining_meta_tensors = materialize_remaining
    cls._apply_entries = apply_entries
    cls._check_supported = staticmethod(check_supported)
    cls._connect = connect
    cls._resolve_socket_path = resolve_socket_path
    cls._fetch_entries = fetch_entries
    cls._hcu_weight_cache_loader = True


def install_key_fingerprint(module: ModuleType) -> None:
    """Invalidate local checkpoint keys when shard metadata or plugin version changes.

    No checkpoint tensor is read here. Stat timestamps are a conservative
    change detector, not a cryptographic hash of checkpoint contents.
    """
    import dataclasses
    import glob
    import hashlib
    import json
    import os

    _install_cache_config_scope(module)
    cls = module.WeightCacheKey
    if cls.__dict__.get("_hcu_fingerprint", False):
        return
    original = cls.from_model_config
    if not callable(original):
        raise RuntimeError("Incompatible weight cache key interface")

    @classmethod
    def from_model_config(key_cls, model_config, tp_size, tp_rank):
        from vllm_hcu.version import __hcu_version__

        role = _cache_model_role.get()
        if role is not None:
            cached = _precomputed_cache_keys.get((id(model_config), role))
            if cached is not None:
                return cached
        moe_backend = _cache_moe_backend.get()
        if moe_backend is None:
            raise RuntimeError("weight cache key requires vLLM kernel configuration")
        key = original(model_config, tp_size=tp_size, tp_rank=tp_rank)
        pp_size, pp_rank = _pipeline_context()
        model_path = model_config.model
        local_shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors"))) if os.path.isdir(model_path) else []
        hasher = hashlib.sha256()
        hasher.update(key.checkpoint.encode())
        for path in local_shards:
            info = os.stat(path)
            record = [os.path.basename(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns]
            hasher.update(json.dumps(record, separators=(",", ":")).encode())

        # Header hashes alone do not encode model overrides or nested Kimi
        # compression settings. Include the resolved HF config and the model
        # options that affect weight interpretation, before post-load can
        # mutate the quantization config. Runtime capacity settings such as
        # max_model_len must not participate: the preload daemon and serving
        # engine can use different KV-cache budgets for the same weights.
        hf_config = getattr(model_config, "hf_config", None)
        hf_dict = hf_config.to_dict() if hf_config is not None else None
        options = {
            name: getattr(model_config, name, None)
            for name in (
                "code_revision", "model_weights", "model_impl", "convert",
                "model_class_overrides", "quantization_config",
            )
        }
        # HCU switches alter post-processed layouts and must match in both
        # daemon and engine. Record effective values rather than raw spelling.
        from vllm_hcu.platforms import envs as henvs
        from vllm_hcu.platforms.envs import (
            custom_ops_enabled,
            custom_quantization_gemm_enabled,
            resolve_aiter_moe_shuffle,
        )
        role = _cache_model_role.get()
        if role is not None:
            options["model_role"] = role
        options["hcu_runtime"] = {
            "custom_ops": custom_ops_enabled(),
            "custom_quantization_gemm": custom_quantization_gemm_enabled(),
            "aiter_moe_shuffle": resolve_aiter_moe_shuffle(),
            "moe_backend": moe_backend,
            "use_nn": bool(henvs.VLLM_USE_NN),
            "deepep_auto_shared_weights": henvs.optional_custom_op_enabled(
                henvs.VLLM_HCU_DEEPEP_AUTO_SHARED_WEIGHTS
            ),
        }
        parallel = _cache_parallel_config.get()
        if parallel is not None and getattr(parallel, "enable_expert_parallel", False):
            # Different DP ranks own different expert shards even when TP=1.
            # UUID identifies a device, not the contents of an older cache on it.
            # Until the full DP/EP cache path is validated, this also prevents
            # a strict EP engine from accepting an existing non-EP cache by
            # accident.
            dp_size = int(parallel.data_parallel_size)
            dp_rank = int(parallel.data_parallel_rank)
            if dp_size < 1 or not 0 <= dp_rank < dp_size:
                raise RuntimeError("invalid data-parallel rank for expert cache")
            def parallel_int(name, default):
                value = getattr(parallel, name, default)
                return int(default if value is None else value)

            options["expert_parallel"] = {
                "data_parallel_size": dp_size,
                "data_parallel_rank": dp_rank,
                # Keep the full topology identity in the key.  The daemon
                # currently accepts TP=1/PP=1 only, but recording these fields
                # prevents a future wider launcher from reusing a shard made
                # for a different tensor/pipeline layout.
                "tensor_parallel_size": parallel_int("tensor_parallel_size", 1),
                "pipeline_parallel_size": parallel_int("pipeline_parallel_size", 1),
                "prefill_context_parallel_size": parallel_int(
                    "prefill_context_parallel_size", 1
                ),
                "decode_context_parallel_size": parallel_int(
                    "decode_context_parallel_size", 1
                ),
                "data_parallel_size_local": parallel_int(
                    "data_parallel_size_local", dp_size
                ),
                "expert_parallel_size": parallel_int(
                    "expert_parallel_size", dp_size
                ),
                "all2all_backend": getattr(parallel, "all2all_backend", None),
                "expert_placement_strategy": getattr(
                    parallel, "expert_placement_strategy", None
                ),
            }
        try:
            config_bytes = json.dumps(
                {"hf_config": hf_dict, "model_options": options},
                sort_keys=True, separators=(",", ":"), default=str,
            ).encode()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Cannot fingerprint model configuration") from exc
        hasher.update(config_bytes)
        scope = f"|hcu-pp:{pp_size}:{pp_rank}" if pp_size > 1 else ""
        key = dataclasses.replace(
            key, checkpoint="hcu-config-v3:" + hasher.hexdigest() + scope
        )
        return dataclasses.replace(key, vllm_version=f"{key.vllm_version}|hcu:{__hcu_version__}")

    cls.from_model_config = from_model_config
    cls._hcu_fingerprint = True
