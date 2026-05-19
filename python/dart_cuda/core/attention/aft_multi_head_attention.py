"""Multi-head AFT — mirrors `lib/core/attention/aft_multi_head_attention.dart`."""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor
from .aft import AFTAttention


class MultiHeadAFT(Module):
    def __init__(self, numHeads: int, embedSize: int, maxSeqLen: int, *, masked: bool = False):
        assert embedSize % numHeads == 0
        self.num_heads = numHeads
        self.head_size = embedSize // numHeads
        self.masked = masked
        self.heads = [
            AFTAttention(embedSize, self.head_size, maxSeqLen, masked=masked)
            for _ in range(numHeads)
        ]
        self.proj = Layer(embedSize, embedSize, useGelu=False)

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
        return self.proj.forward(concat, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for h in self.heads:
            ps.extend(h.parameters())
        ps.extend(self.proj.parameters())
        return ps
