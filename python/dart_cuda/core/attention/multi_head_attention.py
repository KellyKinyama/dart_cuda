"""Multi-head standard self-attention.

Port of `lib/transformer/multi_head_attention.dart` (dart-torch).
"""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor
from .self_attention import SelfAttention


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, embed_size: int, *, masked: bool = False):
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_size = embed_size // num_heads
        self.heads: List[SelfAttention] = [
            SelfAttention(embed_size, self.head_size, masked=masked)
            for _ in range(num_heads)
        ]
        self.proj = Layer(embed_size, embed_size, useGelu=False)

    @property
    def numHeads(self) -> int:
        return self.num_heads

    @property
    def headSize(self) -> int:
        return self.head_size

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        head_outs = [h.forward(x, tracker) for h in self.heads]
        concat = Tensor.concat(head_outs)
        tracker.append(concat)
        out = self.proj.forward(concat, tracker)
        return out

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for h in self.heads:
            ps.extend(h.parameters())
        ps.extend(self.proj.parameters())
        return ps
