"""Opt-in profiler ranges for HCU-owned tensor-parallel all-reduces."""

import os
from contextlib import AbstractContextManager, nullcontext

import torch


def tp_all_reduce_trace(label: str) -> AbstractContextManager[object]:
    """Return a profiler range only while TP all-reduce tracing is requested."""

    if os.environ.get("VLLM_HCU_TP_ALLREDUCE_TRACE") != "1":
        return nullcontext()
    return torch.autograd.profiler.record_function(f"hcu_tp_all_reduce:{label}")
