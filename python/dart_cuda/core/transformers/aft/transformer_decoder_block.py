"""Transformer decoder block — mirrors
`lib/core/transformers/aft/transformer_decoder_block.dart`."""

from __future__ import annotations

import random
from typing import List

from ...attention.aft_multi_head_attention import MultiHeadAFT
from ...attention.aft_multi_head_cross_attention import MultiHeadAFTCross
from ...layers.feed_forward import FeedForward
from ...layers.layer_norm import LayerNorm
from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor


class TransformerDecoderBlock(Module):
    def __init__(self, embedSize: int, numHeads: int, encoderEmbedSize: int, maxSeqLen: int):
        self.embed_size = embedSize
        self.self_attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked=True)
        self.cross_attention = MultiHeadAFTCross(
            numHeads, embedSize, encoderEmbedSize, maxSeqLen, maxSeqLen
        )
        self.ffn = FeedForward(embedSize)
        self.ln1 = LayerNorm(embedSize)
        self.ln2 = LayerNorm(embedSize)
        self.ln3 = LayerNorm(embedSize)
        self._init_weights()

    def _init_weights(self) -> None:
        rng = random.Random()
        for p in self.parameters():
            data = p.fetch_data()
            for i in range(len(data)):
                data[i] = (rng.random() * 2 - 1) * 0.02
            p.data = data

    def forward(self, x_dec: Tensor, x_enc: Tensor, tracker: List[Tensor]) -> Tensor:
        x_norm1 = self.ln1.forward(x_dec, tracker)
        self_out = self.self_attention.forward(x_norm1, tracker)
        x_res1 = x_dec + self_out
        tracker.append(x_res1)

        x_norm2 = self.ln2.forward(x_res1, tracker)
        cross_out = self.cross_attention.forward(x_norm2, x_enc, tracker)
        x_res2 = x_res1 + cross_out
        tracker.append(x_res2)

        x_norm3 = self.ln3.forward(x_res2, tracker)
        ffn_out = self.ffn.forward(x_norm3, tracker)
        out = x_res2 + ffn_out
        tracker.append(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.self_attention.parameters(),
            *self.cross_attention.parameters(),
            *self.ffn.parameters(),
            *self.ln1.parameters(),
            *self.ln2.parameters(),
            *self.ln3.parameters(),
        ]
