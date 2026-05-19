"""FeedForward MLP block — mirrors `lib/core/layers/feed_forward.dart`."""

from __future__ import annotations

from typing import List

from ..tensor.gpu_tensor import Tensor
from .nn import Layer, Module


class FeedForward(Module):
    def __init__(self, dim: int):
        self.w1 = Layer(dim, dim * 4, useGelu=True)
        self.w2 = Layer(dim * 4, dim, useGelu=False)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        hidden = self.w1.forward(x, tracker)
        return self.w2.forward(hidden, tracker)

    def parameters(self) -> List[Tensor]:
        return [*self.w1.parameters(), *self.w2.parameters()]
