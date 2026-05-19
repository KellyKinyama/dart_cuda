"""SGD optimizer — mirrors `lib/core/optimizers/stochastic_grad_desc.dart`.

(Despite the name, the native `sdg_step` kernel is invoked just like Adam.)
"""

from __future__ import annotations

import math
from typing import List

from ..tensor.gpu_tensor import Tensor, engine


class SDG:
    def __init__(
        self,
        params: List[Tensor],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        gradClip: float = 1.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = gradClip
        self.t = 0
        self.m: List[Tensor] = [Tensor.zeros(p.shape) for p in self.params]
        self.v: List[Tensor] = [Tensor.zeros(p.shape) for p in self.params]
        self.tracker: List[Tensor] = []

    def zero_grad(self) -> None:
        for p in self.params:
            engine.zeroGrad(p.handle)

    zeroGrad = zero_grad

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            engine.sdgStep(
                p.handle,
                self.m[i].handle,
                self.v[i].handle,
                self.t,
                float(self.lr),
                float(self.beta1),
                float(self.beta2),
                float(self.eps),
            )
            sample = p.fetch_data()
            if sample and math.isnan(sample[0]):
                self.dispose()
                raise RuntimeError(f"parameter[{i}] became NaN")
            self.tracker.extend([self.m[i], self.v[i]])

    def dispose(self) -> None:
        for t in self.m:
            t.dispose()
        for t in self.v:
            t.dispose()
        self.m.clear()
        self.v.clear()
