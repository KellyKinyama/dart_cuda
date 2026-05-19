"""Multi-head AFT cross — mirrors `aft_multi_head_cross_attention.dart`."""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor
from .aft_cross_attention import AFTCrossAttention


class MultiHeadAFTCross(Module):
    def __init__(
        self,
        numHeads: int,
        decoderEmbedSize: int,
        encoderEmbedSize: int,
        maxTDec: int,
        maxTEnc: int,
    ):
        assert decoderEmbedSize % numHeads == 0
        self.num_heads = numHeads
        self.head_size = decoderEmbedSize // numHeads
        self.heads = [
            AFTCrossAttention(
                decoderEmbedSize,
                encoderEmbedSize,
                self.head_size,
                maxTDec,
                maxTEnc,
            )
            for _ in range(numHeads)
        ]
        self.proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu=False)

    @property
    def numHeads(self) -> int:
        return self.num_heads

    @property
    def headSize(self) -> int:
        return self.head_size

    def forward(self, x_dec: Tensor, x_enc: Tensor, tracker: List[Tensor]) -> Tensor:
        head_outs = [h.forward(x_dec, x_enc, tracker) for h in self.heads]
        concat = Tensor.concat(head_outs)
        tracker.append(concat)
        return self.proj.forward(concat, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        for h in self.heads:
            ps.extend(h.parameters())
        ps.extend(self.proj.parameters())
        return ps
