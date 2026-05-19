"""Text transformer — mirrors `modalities/text_transformer.dart`."""

from __future__ import annotations

from typing import List, Sequence

from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor
from ..aft.transformer_encoder import TransformerEncoder


class TextTransformer(Module):
    def __init__(
        self,
        *,
        vocabSize: int,
        maxSeqLen: int,
        embedSize: int,
        numLayers: int = 4,
        numHeads: int = 4,
    ):
        self.embed_size = embedSize
        self.encoder = TransformerEncoder(
            vocabSize=vocabSize,
            embedSize=embedSize,
            blockSize=maxSeqLen,
            numLayers=numLayers,
            numHeads=numHeads,
        )

    @property
    def embedSize(self) -> int:
        return self.embed_size

    def forward(self, tokens: Sequence[int], tracker: List[Tensor]) -> Tensor:
        return self.encoder.forward(tokens, tracker)

    def parameters(self) -> List[Tensor]:
        return self.encoder.parameters()
