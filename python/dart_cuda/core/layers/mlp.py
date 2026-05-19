"""MLP — mirrors `lib/core/layers/mlp.dart`."""

from __future__ import annotations

from typing import List, Sequence

from ..tensor.gpu_tensor import Tensor
from .nn import Layer, Module


class MLP(Module):
    def __init__(self, nin: int, nouts: Sequence[int]):
        dims = [nin, *nouts]
        self.layers: List[Layer] = []
        for i in range(len(nouts)):
            self.layers.append(
                Layer(dims[i], dims[i + 1], useGelu=(i != len(nouts) - 1))
            )

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        cur = x
        for layer in self.layers:
            cur = layer.forward(cur, tracker)
        return cur

    def parameters(self) -> List[Tensor]:
        out: List[Tensor] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out
