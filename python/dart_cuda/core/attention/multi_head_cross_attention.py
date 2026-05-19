"""Multi-head standard cross-attention.

Port of `lib/transformer/multi_head_cross_attention.dart` (dart-torch).
"""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor
from .cross_attention import CrossAttention


class MultiHeadCrossAttention(Module):
    def __init__(
        self,
        num_heads: int,
        decoder_embed_size: int,
        encoder_embed_size: int,
    ):
        assert decoder_embed_size % num_heads == 0
        self.num_heads = num_heads
        self.decoder_embed_size = decoder_embed_size
        self.encoder_embed_size = encoder_embed_size
        self.head_size = decoder_embed_size // num_heads
        self.heads: List[CrossAttention] = [
            CrossAttention(decoder_embed_size, encoder_embed_size, self.head_size)
            for _ in range(num_heads)
        ]
        self.proj = Layer(decoder_embed_size, decoder_embed_size, useGelu=False)

    @property
    def numHeads(self) -> int:
        return self.num_heads

    @property
    def headSize(self) -> int:
        return self.head_size

    def forward(
        self,
        x_decoder: Tensor,
        x_encoder: Tensor,
        tracker: List[Tensor],
    ) -> Tensor:
        head_outs = [h.forward(x_decoder, x_encoder, tracker) for h in self.heads]
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
