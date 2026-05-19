"""LayerNorm — mirrors `lib/core/layers/layer_norm.dart`."""

from __future__ import annotations

from typing import List

from ..tensor.gpu_tensor import Tensor
from .nn import Module


class LayerNorm(Module):
    def __init__(self, dim: int, *, eps: float = 1e-5):
        self.gamma = Tensor.fill([1, dim], 1.0)
        self.beta = Tensor.fill([1, dim], 0.0)
        self.eps = eps

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        out = Tensor.layer_norm(x, self.gamma, self.beta, self.eps)
        tracker.append(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]
