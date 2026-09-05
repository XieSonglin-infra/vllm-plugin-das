"""Optional, rank-local activation snapshots for Kimi-K3 precision diagnosis."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import torch


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _parse_layers(value: str | None) -> set[int] | None:
    if not value or value.strip().lower() in {"all", "*"}:
        return None
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.add(int(item))
        except ValueError:
            raise ValueError(
                "VLLM_HCU_KIMI_TRACE_LAYERS must contain comma-separated integers"
            ) from None
    return layers


class KimiActivationTrace:
    """Write compact JSONL metadata and first-row tensors for selected layers.

    The trace is disabled unless ``VLLM_HCU_KIMI_TRACE_DIR`` is set. Each TP
    rank writes its own files, which makes the same trace usable for TP2 and
    TP16 comparisons without introducing collective communication.
    """

    def __init__(self, directory: str, prefix: str = "") -> None:
        self.directory = Path(directory)
        self.prefix = prefix.replace(".", "_") or "model"
        self.layers = _parse_layers(os.environ.get("VLLM_HCU_KIMI_TRACE_LAYERS"))
        self.max_calls = max(1, _int_env("VLLM_HCU_KIMI_TRACE_MAX_CALLS", 2))
        self.max_rows = max(1, _int_env("VLLM_HCU_KIMI_TRACE_MAX_ROWS", 1))
        self.max_cols = _int_env("VLLM_HCU_KIMI_TRACE_MAX_COLS", 0)
        self.input_sha256 = os.environ.get("VLLM_HCU_KIMI_TRACE_INPUT_SHA256")
        self.rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
        if self.rank is None:
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                self.rank = str(get_tensor_model_parallel_rank())
            except Exception:
                self.rank = os.environ.get("HIP_VISIBLE_DEVICES", "0")
        self.call_index = 0
        self._metadata = None

    @classmethod
    def from_env(cls, prefix: str = "") -> "KimiActivationTrace | None":
        directory = os.environ.get("VLLM_HCU_KIMI_TRACE_DIR")
        if not directory:
            return None
        return cls(directory, prefix)

    def begin(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> None:
        input_digest = None
        if input_ids is not None:
            input_values = input_ids.detach().reshape(-1).cpu().tolist()
            input_digest = hashlib.sha256(
                ",".join(str(value) for value in input_values).encode("ascii")
            ).hexdigest()
        if self.input_sha256 and input_digest != self.input_sha256:
            self._metadata = None
            return
        self.call_index += 1
        if self.call_index > self.max_calls:
            self._metadata = None
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{self.prefix}.rank{self.rank}.call{self.call_index:04d}"
        )
        self._metadata = self.directory / f"{stem}.jsonl"
        self._write(
            {
                "kind": "input",
                "shape": list(input_ids.shape) if input_ids is not None else None,
                "input_sha256": input_digest,
                "input_ids": input_ids.detach().cpu().tolist()
                if input_ids is not None and input_ids.numel() <= 4096
                else None,
                "positions": positions.detach().cpu().tolist()
                if positions is not None and positions.numel() <= 4096
                else None,
            }
        )

    def capture(self, name: str, tensor: torch.Tensor, layer_idx: int | None = None) -> None:
        if self._metadata is None:
            return
        if layer_idx is not None and self.layers is not None and layer_idx not in self.layers:
            return
        if not isinstance(tensor, torch.Tensor):
            return
        value = tensor.detach()
        flat = value.float().reshape(-1)
        finite = bool(torch.isfinite(flat).all().item()) if flat.numel() else True
        sample = flat[: min(flat.numel(), 16)].cpu().tolist()
        stats: dict[str, Any] = {
            "kind": "activation",
            "name": name,
            "layer": layer_idx,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": finite,
            "min": float(flat.min().item()) if flat.numel() else 0.0,
            "max": float(flat.max().item()) if flat.numel() else 0.0,
            "mean": float(flat.mean().item()) if flat.numel() else 0.0,
            "l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
            "sample": sample,
        }
        safe_name = name.replace(".", "_")
        rows = min(self.max_rows, value.shape[0]) if value.ndim else 1
        if value.ndim:
            cols = self.max_cols or value.shape[-1]
            snapshot = value[:rows, :cols].cpu() if value.ndim >= 2 else value[:cols].cpu()
        else:
            snapshot = value.cpu()
        layer_suffix = f".layer{layer_idx}" if layer_idx is not None else ""
        tensor_path = self._metadata.with_name(
            f"{self._metadata.stem}.{safe_name}{layer_suffix}.pt"
        )
        torch.save(snapshot, tensor_path)
        stats["tensor_file"] = tensor_path.name
        self._write(stats)

    def _write(self, value: dict[str, Any]) -> None:
        if self._metadata is None:
            return
        with self._metadata.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=True) + "\n")


__all__ = ["KimiActivationTrace"]
