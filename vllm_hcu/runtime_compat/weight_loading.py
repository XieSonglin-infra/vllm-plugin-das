# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opt-in debug weight loading behavior owned by the HCU plugin."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from collections.abc import Generator
from types import ModuleType

import torch
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

_WEIGHT_UTILS_MODULE = "vllm.model_executor.model_loader.weight_utils"
_DEFAULT_LOADER_MODULE = "vllm.model_executor.model_loader.default_loader"
_KIMI_FAKE_WEIGHT_LAYERS_ENV = "VLLM_HCU_KIMI_FAKE_WEIGHT_LAYERS"
_LAYER_KEY_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _natural_sort_key(filepath: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", os.path.basename(filepath))
    ]


def _require_exact_module(module: object, expected_name: str) -> ModuleType:
    if not isinstance(module, ModuleType) or module.__name__ != expected_name:
        actual_name = getattr(module, "__name__", None)
        raise TypeError(
            f"expected module {expected_name!r}, got {actual_name!r}"
        )
    return module


def _resolve_loader_modules(
    weight_utils: ModuleType | None,
    default_loader: ModuleType | None,
) -> tuple[ModuleType, ModuleType]:
    """Resolve direct-call inputs without importing from a partial package.

    Runtime callbacks always pass both completed modules explicitly.  The
    zero-argument form remains a supported direct/idempotent API: importing
    default_loader first lets vLLM finish its normal base_loader -> reload ->
    weight_utils chain before either module is patched.
    """

    if (weight_utils is None) != (default_loader is None):
        raise TypeError("weight_utils and default_loader must be provided together")
    if default_loader is None:
        default_loader = importlib.import_module(_DEFAULT_LOADER_MODULE)
        loaded_weight_utils = sys.modules.get(_WEIGHT_UTILS_MODULE)
        if not isinstance(loaded_weight_utils, ModuleType):
            raise RuntimeError(
                f"{_DEFAULT_LOADER_MODULE} loaded without {_WEIGHT_UTILS_MODULE}"
            )
        weight_utils = loaded_weight_utils

    return (
        _require_exact_module(weight_utils, _WEIGHT_UTILS_MODULE),
        _require_exact_module(default_loader, _DEFAULT_LOADER_MODULE),
    )


def _resolve_fake_layer_limit() -> int | None:
    """Return the optional Kimi decoder layer count for fake-weight smoke."""

    raw_limit = os.environ.get(_KIMI_FAKE_WEIGHT_LAYERS_ENV)
    if raw_limit is None:
        return None
    try:
        layer_limit = int(raw_limit)
    except ValueError as error:
        raise ValueError(
            f"{_KIMI_FAKE_WEIGHT_LAYERS_ENV} must be a positive integer, "
            f"got {raw_limit!r}."
        ) from error
    if layer_limit <= 0:
        raise ValueError(
            f"{_KIMI_FAKE_WEIGHT_LAYERS_ENV} must be a positive integer, "
            f"got {raw_limit!r}."
        )
    return layer_limit


def _is_excluded_fake_layer(name: str, layer_limit: int | None) -> bool:
    if layer_limit is None:
        return False
    match = _LAYER_KEY_PATTERN.search(name)
    return match is not None and int(match.group(1)) >= layer_limit


def _fake_tensor_for_metadata(
    name: str,
    shape: list[int],
    safetensors_dtype: str,
) -> torch.Tensor:
    """Build a deterministic, numerically safe CPU tensor from tensor metadata."""

    dtype = _SAFETENSORS_DTYPE_TO_TORCH.get(safetensors_dtype)
    if dtype is None:
        raise ValueError(
            "Kimi fake-weight mode does not support safetensors dtype "
            f"{safetensors_dtype!r} for {name!r}."
        )

    if name.endswith("weight_scale"):
        return torch.full(shape, 0.01, dtype=dtype)
    if name.endswith("norm.weight"):
        return torch.ones(shape, dtype=dtype)
    if dtype == torch.int8:
        # 0x11 encodes two valid positive INT4 nibbles and reaches the W4A8 path.
        return torch.full(shape, 0x11, dtype=dtype)
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        return torch.zeros(shape, dtype=dtype)

    seed = int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
    )
    # A deterministic per-key scalar avoids the very high CPU cost of filling
    # every full expert tensor with a random distribution during full fake.
    scalar = ((seed % 2001) - 1000) / 1e6
    return torch.full(shape, scalar, dtype=dtype)


def _fake_safetensors_weights_iterator(
    hf_weights_files: list[str],
    local_expert_ids: set[int] | None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield synthetic weights from safetensors headers without reading payloads."""

    should_skip_weight = None
    if local_expert_ids is not None:
        from vllm.model_executor.model_loader.ep_weight_filter import (
            should_skip_weight as _should_skip_weight,
        )

        should_skip_weight = _should_skip_weight

    layer_limit = _resolve_fake_layer_limit()
    for st_file in sorted(hf_weights_files, key=_natural_sort_key):
        with safe_open(st_file, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if should_skip_weight is not None and should_skip_weight(
                    name, local_expert_ids
                ):
                    continue
                if _is_excluded_fake_layer(name, layer_limit):
                    continue
                tensor_slice = checkpoint.get_slice(name)
                yield name, _fake_tensor_for_metadata(
                    name,
                    tensor_slice.get_shape(),
                    tensor_slice.get_dtype(),
                )


def install_weight_debug_skip_compat(
    weight_utils: ModuleType | None = None,
    default_loader: ModuleType | None = None,
) -> None:
    """Install the opt-in HCU debug weight-skipping iterator.

    Supplying both modules is the import-callback path and performs no vLLM
    import.  Calling without arguments remains supported for explicit callers.
    """

    weight_utils, default_loader = _resolve_loader_modules(
        weight_utils,
        default_loader,
    )

    weight_iterator = getattr(weight_utils, "safetensors_weights_iterator", None)
    default_iterator = getattr(default_loader, "safetensors_weights_iterator", None)
    if not callable(weight_iterator) or not callable(default_iterator):
        raise RuntimeError("vLLM safetensors iterator contract is unavailable")

    if getattr(weight_utils, "_hcu_skip_weight_patch_applied", False):
        if weight_iterator is not default_iterator:
            raise RuntimeError(
                "weight-debug marker is set but loader iterator bindings diverged"
            )
        return

    if weight_iterator is not default_iterator:
        raise RuntimeError(
            "weight_utils and default_loader iterator bindings differ before patch"
        )

    original_iterator = weight_iterator
    skip_logged = False
    fake_logged = False

    DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS = 8
    DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024

    def wrapped_safetensors_weights_iterator(
        hf_weights_files: list[str],
        use_tqdm_on_load: bool,
        safetensors_load_strategy: str = "lazy",
        local_expert_ids: set[int] | None = None,
        *,
        safetensors_prefetch_num_threads: int = (
            DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS
        ),
        safetensors_prefetch_block_size: int = DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE,
    ) -> Generator[tuple[str, object], None, None]:
        import vllm_hcu.platforms.envs as henvs

        skip_weight_debug_enabled = henvs.VLLM_HCU_USE_SKIP_WEIGHT_DEBUG
        fake_weight_debug_enabled = henvs.VLLM_HCU_KIMI_FAKE_WEIGHT_DEBUG
        if fake_weight_debug_enabled:
            if safetensors_load_strategy not in (None, "lazy"):
                raise ValueError(
                    "Kimi fake-weight mode requires the default lazy "
                    "safetensors loader."
                )
            nonlocal fake_logged
            if not fake_logged:
                logger.warning(
                    "Kimi fake-weight debug mode is enabled: reading "
                    "safetensors metadata only; checkpoint payloads are not read."
                )
                fake_logged = True
            yield from _fake_safetensors_weights_iterator(
                hf_weights_files,
                local_expert_ids,
            )
            return
        for name, param in original_iterator(
            hf_weights_files=hf_weights_files,
            use_tqdm_on_load=use_tqdm_on_load,
            safetensors_load_strategy=safetensors_load_strategy,
            local_expert_ids=local_expert_ids,
        ):
            if skip_weight_debug_enabled and "layers." in name:
                try:
                    layer_id = int(name.split("layers.")[1].split(".")[0])
                    if layer_id >= 5:
                        nonlocal skip_logged
                        if not skip_logged:
                            logger.warning(
                                "VLLM_HCU_USE_SKIP_WEIGHT_DEBUG is enabled "
                                "skipping weights from layer %s starting with %s",
                                layer_id,
                                name,
                            )
                            skip_logged = True
                        continue
                except Exception:
                    pass
            yield name, param

    weight_utils.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    default_loader.safetensors_weights_iterator = wrapped_safetensors_weights_iterator
    if (
        weight_utils.safetensors_weights_iterator
        is not default_loader.safetensors_weights_iterator
    ):
        raise RuntimeError("weight-debug iterator bindings diverged after patch")
    # Publish the marker only after both import-time copies are coherent.
    weight_utils._hcu_skip_weight_patch_applied = True
