"""Standard scaled-dot-product cross-attention (single head).

Port of `lib/transformer/cross_attention.dart` (dart-torch).

Queries come from the decoder side, keys/values from the encoder side.
No causal masking (encoder output is fully visible to the decoder).
"""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor
from .self_attention import scaled_dot_product_attention


class CrossAttention(Module):
    def __init__(
        self,
        decoder_embed_size: int,
        encoder_embed_size: int,
        head_size: int,
    ):
        self.decoder_embed_size = decoder_embed_size
        self.encoder_embed_size = encoder_embed_size
        self.head_size = head_size
        self.query = Layer(decoder_embed_size, head_size, useGelu=False)
        self.key = Layer(encoder_embed_size, head_size, useGelu=False)
        self.value = Layer(encoder_embed_size, head_size, useGelu=False)

    @property
    def headSize(self) -> int:
        return self.head_size

    def forward(
        self,
        x_decoder: Tensor,
        x_encoder: Tensor,
        tracker: List[Tensor],
    ) -> Tensor:
        q = self.query.forward(x_decoder, tracker)
        k = self.key.forward(x_encoder, tracker)
        v = self.value.forward(x_encoder, tracker)
        return scaled_dot_product_attention(q, k, v, tracker, masked=False)

    def parameters(self) -> List[Tensor]:
        return [
            *self.query.parameters(),
            *self.key.parameters(),
            *self.value.parameters(),
        ]
