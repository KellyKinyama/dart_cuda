"""Text decoder block — mirrors `lib/core/transformers/aft/text_decoder_block.dart`."""

from __future__ import annotations

from typing import List

from ...attention.aft import AFTAttention
from ...layers.layer_norm import LayerNorm
from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor


class TextDecoderBlock(Module):
    def __init__(self, embedSize: int, numHeads: int, blockSize: int):
        self.self_attention = AFTAttention(embedSize, numHeads, blockSize, masked=True)
        self.norm1 = LayerNorm(embedSize)
        self.cross_attention = AFTAttention(embedSize, numHeads, blockSize)
        self.norm2 = LayerNorm(embedSize)
        self.ff1 = Layer(embedSize, embedSize * 4, useGelu=True)
        self.ff2 = Layer(embedSize * 4, embedSize, useGelu=False)
        self.norm3 = LayerNorm(embedSize)

    def forward(self, x: Tensor, encoder_output: Tensor, tracker: List[Tensor]) -> Tensor:
        attn1 = self.self_attention.forward(self.norm1.forward(x, tracker), tracker)
        x1 = x + attn1

        attn2 = self.cross_attention.forward(
            self.norm2.forward(x1, tracker), tracker, kv=encoder_output
        )
        x2 = x1 + attn2

        ff_out = self.ff2.forward(
            self.ff1.forward(self.norm3.forward(x2, tracker), tracker),
            tracker,
        )
        return x2 + ff_out

    def parameters(self) -> List[Tensor]:
        return [
            *self.self_attention.parameters(),
            *self.norm1.parameters(),
            *self.cross_attention.parameters(),
            *self.norm2.parameters(),
            *self.ff1.parameters(),
            *self.ff2.parameters(),
            *self.norm3.parameters(),
        ]
