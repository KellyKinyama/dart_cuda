"""AFT attention — mirrors `lib/core/attention/aft.dart`."""

from __future__ import annotations

from typing import List, Optional

from ..layers.nn import Layer, Module
from ..tensor.gpu_tensor import Tensor


class AFTAttention(Module):
    def __init__(self, embedSize: int, headSize: int, maxSeqLen: int, *, masked: bool = False):
        self.query_layer = Layer(embedSize, headSize, useGelu=False)
        self.key_layer = Layer(embedSize, headSize, useGelu=False)
        self.value_layer = Layer(embedSize, headSize, useGelu=False)
        self.pos_bias = Tensor.from_list(
            [maxSeqLen, maxSeqLen],
            [0.01] * (maxSeqLen * maxSeqLen),
        )
        self.masked = masked

    # Dart aliases
    @property
    def queryLayer(self) -> Layer:
        return self.query_layer

    @property
    def keyLayer(self) -> Layer:
        return self.key_layer

    @property
    def valueLayer(self) -> Layer:
        return self.value_layer

    @property
    def posBias(self) -> Tensor:
        return self.pos_bias

    def forward(self, x: Tensor, tracker: List[Tensor], *, kv: Optional[Tensor] = None) -> Tensor:
        ctx = kv if kv is not None else x
        q = self.query_layer.forward(x, tracker)
        k = self.key_layer.forward(ctx, tracker)
        v = self.value_layer.forward(ctx, tracker)
        out = Tensor.aft(q, k, v, self.pos_bias, self.masked)
        tracker.append(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.query_layer.parameters(),
            *self.key_layer.parameters(),
            *self.value_layer.parameters(),
            self.pos_bias,
        ]
