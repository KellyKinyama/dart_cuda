"""RMSNorm — port of `lib/transformer/rms_norm.dart` (dart-torch).

Root-mean-square layer norm:

    y = x * rsqrt(mean(x^2) + eps) * gamma

* No mean centering, no bias — just a per-feature scale `gamma`.
* Used by LLaMA / DeepSeek / Mistral / Qwen / etc.

Composed from existing dart_cuda tensor ops (pow, mean, sum, scalar-mul, etc.)
plus a sqrt built on `pow(x, 0.5)`. The mean is taken across all elements,
matching the dart-torch behaviour where `x.squared().mean()` reduces the
whole vector.
"""

from __future__ import annotations

from typing import List

from ..tensor.gpu_tensor import Tensor
from .nn import Module


class RMSNorm(Module):
    def __init__(self, dim: int, *, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        # Per-feature scale, initialised to 1.
        self.gamma = Tensor.fill([1, dim], 1.0)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        sq = x.pow(2.0)
        ms = sq.mean()                   # scalar Tensor (1x1)
        ms_eps = ms + float(self.eps)
        rms = ms_eps.pow(0.5)            # scalar Tensor
        # Divide each row element by the scalar rms.
        normed = x / rms
        out = normed * self.gamma        # broadcast over rows via row-bias style
        tracker.extend([sq, ms, ms_eps, rms, normed, out])
        return out

    def parameters(self) -> List[Tensor]:
        return [self.gamma]
