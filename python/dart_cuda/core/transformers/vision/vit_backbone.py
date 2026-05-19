"""ViT backbone — mirrors `lib/core/transformers/vision/vit_backbone.dart`."""

from __future__ import annotations

from typing import List

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from ..aft.transformer_encoder import TransformerEncoder


class ViTBackbone(Module):
    def __init__(
        self,
        *,
        imageSize: int,
        patchSize: int,
        numChannels: int = 3,
        embedSize: int,
        numLayers: int = 4,
        numHeads: int = 4,
    ):
        self.image_size = imageSize
        self.patch_size = patchSize
        self.num_channels = numChannels
        self.embed_size = embedSize

        num_patches = (imageSize // patchSize) * (imageSize // patchSize)
        self.transformer_encoder = TransformerEncoder(
            vocabSize=1,
            embedSize=embedSize,
            blockSize=num_patches + 1,
            numLayers=numLayers,
            numHeads=numHeads,
        )

        self.patch_projection = Layer(
            patchSize * patchSize * numChannels, embedSize, useGelu=False
        )
        self.cls_token = Tensor.random([1, embedSize], scale=0.02)
        self.pos_embeddings = Tensor.random([num_patches + 1, embedSize], scale=0.02)

    @property
    def embedSize(self) -> int:
        return self.embed_size

    @property
    def transformerEncoder(self) -> TransformerEncoder:
        return self.transformer_encoder

    @property
    def patchProjection(self) -> Layer:
        return self.patch_projection

    @property
    def clsToken(self) -> Tensor:
        return self.cls_token

    @property
    def posEmbeddings(self) -> Tensor:
        return self.pos_embeddings

    def forward(self, patchified_image: Tensor, tracker: List[Tensor]) -> Tensor:
        x_patches = self.patch_projection.forward(patchified_image, tracker)
        x_seq = Tensor.concat_axis0([self.cls_token, x_patches])
        tracker.append(x_seq)
        x_pos = x_seq + self.pos_embeddings
        tracker.append(x_pos)
        return self.transformer_encoder.forward_embeddings(x_pos, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.patch_projection.parameters(),
            self.cls_token,
            self.pos_embeddings,
            *self.transformer_encoder.parameters(),
        ]
