"""AFT cross-attention — mirrors `lib/core/attention/aft_cross_attention.dart`."""

from __future__ import annotations

from typing import List

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor


class AFTCrossAttention(Module):
    def __init__(
        self,
        decoderEmbedSize: int,
        encoderEmbedSize: int,
        headSize: int,
        maxTDec: int,
        maxTEnc: int,
    ):
        self.key_layer = Layer(encoderEmbedSize, headSize, useGelu=False)
        self.query_layer = Layer(decoderEmbedSize, headSize, useGelu=False)
        self.value_layer = Layer(encoderEmbedSize, headSize, useGelu=False)
        self.pos_bias = Tensor.from_list(
            [maxTDec, maxTEnc],
            [0.01] * (maxTDec * maxTEnc),
        )

    @property
    def keyLayer(self) -> Layer:
        return self.key_layer

    @property
    def queryLayer(self) -> Layer:
        return self.query_layer

    @property
    def valueLayer(self) -> Layer:
        return self.value_layer

    @property
    def posBias(self) -> Tensor:
        return self.pos_bias

    def forward(self, x_dec: Tensor, x_enc: Tensor, tracker: List[Tensor]) -> Tensor:
        q = self.query_layer.forward(x_dec, tracker)
        k = self.key_layer.forward(x_enc, tracker)
        v = self.value_layer.forward(x_enc, tracker)
        out = Tensor.aft_cross(q, k, v, self.pos_bias)
        tracker.append(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.key_layer.parameters(),
            *self.query_layer.parameters(),
            *self.value_layer.parameters(),
            self.pos_bias,
        ]
