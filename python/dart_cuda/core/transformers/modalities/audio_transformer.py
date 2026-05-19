"""Audio transformer — mirrors `modalities/audio_transformer.dart`."""

from __future__ import annotations

from typing import List

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from ..aft.transformer_encoder import TransformerEncoder


class AudioTransformer(Module):
    def __init__(
        self,
        *,
        featureDim: int,
        maxSequenceLength: int,
        embedSize: int,
        numClasses: int,
        numLayers: int = 4,
        numHeads: int = 4,
    ):
        self.feature_dim = featureDim
        self.max_sequence_length = maxSequenceLength
        self.embed_size = embedSize
        self.num_classes = numClasses
        self.transformer_encoder = TransformerEncoder(
            vocabSize=0,
            embedSize=embedSize,
            blockSize=maxSequenceLength,
            numLayers=numLayers,
            numHeads=numHeads,
        )
        self.feature_projection = Layer(featureDim, embedSize, useGelu=True)
        self.pos_embeddings = Tensor.random([maxSequenceLength, embedSize], scale=0.02)
        self.classification_head = Layer(embedSize, numClasses, useGelu=False)

    @property
    def embedSize(self) -> int:
        return self.embed_size

    @property
    def featureProjection(self) -> Layer:
        return self.feature_projection

    @property
    def posEmbeddings(self) -> Tensor:
        return self.pos_embeddings

    @property
    def transformerEncoder(self) -> TransformerEncoder:
        return self.transformer_encoder

    def forward(self, audio_features: Tensor, tracker: List[Tensor]) -> Tensor:
        x_embed = self.feature_projection.forward(audio_features, tracker)
        actual_seq_len = audio_features.shape[0]
        cur_pos = self.pos_embeddings.slice(0, actual_seq_len)
        x = x_embed + cur_pos
        tracker.append(x)
        encoded = self.transformer_encoder.forward_embeddings(x, tracker)
        pooled = encoded.mean()
        tracker.append(pooled)
        return self.classification_head.forward(pooled, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.feature_projection.parameters(),
            self.pos_embeddings,
            *self.transformer_encoder.parameters(),
            *self.classification_head.parameters(),
        ]
