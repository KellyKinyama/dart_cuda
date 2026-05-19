"""Base Module + Layer, mirrors `lib/core/layers/nn.dart`."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import List

from ..tensor.gpu_tensor import Tensor


class Module(ABC):
    @abstractmethod
    def parameters(self) -> List[Tensor]:
        ...

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    zeroGrad = zero_grad

    def step(self, lr: float) -> None:
        for p in self.parameters():
            p.step(lr)


class Layer(Module):
    """Linear `nin -> nout` (+ optional GELU). Matches Dart `Layer`."""

    def __init__(self, nin: int, nout: int, *, useGelu: bool):
        scale = math.sqrt(2.0 / (nin + nout))
        rng = random.Random()
        w_vals = [(rng.random() * 2 - 1) * scale for _ in range(nin * nout)]
        self.w = Tensor.from_list([nin, nout], w_vals)
        self.b = Tensor.fill([1, nout], 0.01)
        self.use_gelu = useGelu

    # Dart-side name
    @property
    def useGelu(self) -> bool:
        return self.use_gelu

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        xw = x.matmul(self.w)
        out = xw + self.b
        tracker.extend([xw, out])
        if self.use_gelu:
            act = out.gelu()
            tracker.append(act)
            return act
        return out

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]
