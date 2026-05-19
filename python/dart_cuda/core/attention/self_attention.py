"""Standard scaled-dot-product self-attention (single head).

Port of `lib/transformer/self_attention.dart` (dart-torch) and
`lib/transformer/attention.dart` to dart_cuda's GPU tensor pipeline.

Forward (T = sequence length, D_in = embed size, H = head size):
    Q = X @ Wq      [T, H]
    K = X @ Wk      [T, H]
    V = X @ Wv      [T, H]
    S = Q @ K^T / sqrt(H)            [T, T]
    if masked: add upper-triangular -1e9 (causal mask)
    P = softmax(S, axis=-1)           [T, T]
    out = P @ V                       [T, H]
"""

from __future__ import annotations

import math
from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor


def _causal_mask(T: int) -> Tensor:
    """Upper-triangular additive mask: -1e9 above the diagonal, 0 elsewhere."""
    vals = [0.0] * (T * T)
    for i in range(T):
        for j in range(i + 1, T):
            vals[i * T + j] = -1e9
    return Tensor.from_list([T, T], vals)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tracker: List[Tensor],
    *,
    masked: bool = False,
) -> Tensor:
    """Composable SDPA built from transpose + matmul + softmax."""
    H = q.shape[-1]
    scale = 1.0 / math.sqrt(float(H))
    kt = k.transpose()
    scores = q.matmul(kt)
    scaled = scores * scale
    tracker.extend([kt, scores, scaled])
    if masked:
        mask = _causal_mask(q.shape[0])
        scaled = scaled + mask
        tracker.extend([mask, scaled])
    weights = scaled.softmax()
    out = weights.matmul(v)
    tracker.extend([weights, out])
    return out


class SelfAttention(Module):
    """Single-head self-attention."""

    def __init__(self, embed_size: int, head_size: int, *, masked: bool = False):
        self.embed_size = embed_size
        self.head_size = head_size
        self.masked = masked
        self.key = Layer(embed_size, head_size, useGelu=False)
        self.query = Layer(embed_size, head_size, useGelu=False)
        self.value = Layer(embed_size, head_size, useGelu=False)

    # dart-torch aliases
    @property
    def headSize(self) -> int:
        return self.head_size

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        q = self.query.forward(x, tracker)
        k = self.key.forward(x, tracker)
        v = self.value.forward(x, tracker)
        return scaled_dot_product_attention(q, k, v, tracker, masked=self.masked)

    def parameters(self) -> List[Tensor]:
        return [
            *self.query.parameters(),
            *self.key.parameters(),
            *self.value.parameters(),
        ]
