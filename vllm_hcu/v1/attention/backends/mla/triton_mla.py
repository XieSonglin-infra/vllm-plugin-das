# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

import os
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
)

# ---- temporary triton decode-input capture (precision diagnosis, env-gated) ----
_TRITON_CAPTURE_DIR = os.environ.get("VLLM_HCU_TRITON_CAPTURE_DIR", "") or None
_TRITON_CAPTURE_N = [0]
_TRITON_CAPTURE_MAX = 300


def _triton_capture(payload: dict) -> None:
    """Save one triton decode call's inputs + output under the capture dir."""
    if not _TRITON_CAPTURE_DIR:
        return
    try:
        _TRITON_CAPTURE_N[0] += 1
        n = _TRITON_CAPTURE_N[0]
        if n > _TRITON_CAPTURE_MAX:
            return
        rank = "0"
        try:
            import torch.distributed as _dist

            if _dist.is_available() and _dist.is_initialized():
                rank = str(_dist.get_rank())
        except Exception:
            pass
        path = os.path.join(
            _TRITON_CAPTURE_DIR, f"triton_decode.r{rank}.call{n:04d}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
    except Exception:  # never break inference for a diagnostic capture
        pass


class HcuTritonMLAImpl(TritonMLAImpl):
    """Triton MLA decode impl with an optional per-call capture (env-gated),
    matching the FlashMLA decode-input capture so the two backends can be
    compared layer-by-layer at the same decode step."""

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer,
    ):
        dm = attn_metadata.decode
        bt = dm.block_table
        sl = dm.seq_lens
        cap_payload = None
        if _TRITON_CAPTURE_DIR:
            try:
                qq = torch.cat(q, dim=-1) if isinstance(q, tuple) else q
                Bq = qq.shape[0]
                rows = []
                for i in range(Bq):
                    npg = (int(sl[i]) + 63) // 64
                    for b in bt[i, :npg].tolist():
                        rows.append(kv_c_and_k_pe_cache[b].detach())
                crows = torch.stack(rows) if rows else kv_c_and_k_pe_cache[:0]
                cap_payload = {
                    "q": qq.detach(),  # (B, H, 576)
                    "cache_rows": crows,  # (used_blocks, 64, 576) bf16
                    "block_table": bt.detach(),
                    "seq_lens": sl.detach().tolist(),
                    "num_decodes": attn_metadata.num_decodes,
                    "cache_shape": tuple(kv_c_and_k_pe_cache.shape),
                    "head_dim_v": self.kv_lora_rank,
                    "scale": float(self.scale),
                }
            except Exception:
                cap_payload = None
        o, lse = super().forward_mqa(
            q, kv_c_and_k_pe_cache, attn_metadata, layer
        )
        if cap_payload is not None:
            cap_payload["out"] = o.detach()
            cap_payload["lse"] = lse.detach() if lse is not None else None
            _triton_capture(cap_payload)
        return o, lse


class HcuTritonMLABackend(TritonMLABackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type[TritonMLAImpl]:
        return HcuTritonMLAImpl
