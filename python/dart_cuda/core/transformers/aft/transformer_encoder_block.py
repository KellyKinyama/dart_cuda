"""Transformer encoder block — mirrors
`lib/core/transformers/aft/transformer_encoder_block.dart`."""

from __future__ import annotations

from typing import List

from ...attention.aft_multi_head_attention import MultiHeadAFT
from ...layers.layer_norm import LayerNorm
from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor


class TransformerEncoderBlock(Module):
    def __init__(self, embedSize: int, numHeads: int, maxSeqLen: int):
        self.embed_size = embedSize
        self.attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked=False)
        self.ffn = Layer(embedSize, embedSize, useGelu=True)
        self.ln1 = LayerNorm(embedSize)
        self.ln2 = LayerNorm(embedSize)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        x_norm1 = self.ln1.forward(x, tracker)
        attn_out = self.attention.forward(x_norm1, tracker)
        x_res1 = x + attn_out
        tracker.extend([attn_out, x_res1])

        x_norm2 = self.ln2.forward(x_res1, tracker)
        ffn_out = self.ffn.forward(x_norm2, tracker)
        out = x_res1 + ffn_out
        tracker.append(ffn_out)
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.attention.parameters(),
            *self.ffn.parameters(),
            *self.ln1.parameters(),
            *self.ln2.parameters(),
        ]
