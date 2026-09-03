"""HCU-owned SiTU activation used by Kimi-K3 routed experts."""

from __future__ import annotations

import torch
from torch import nn


class SituAndMul(nn.Module):
    """Apply Kimi's gated SiTU activation to a fused gate/up tensor."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        if beta <= 0 or (linear_beta is not None and linear_beta <= 0):
            raise ValueError("SiTU beta values must be positive")
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return gate * up


__all__ = ["SituAndMul"]
