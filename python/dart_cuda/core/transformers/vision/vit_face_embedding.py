"""ViT face embedding head — mirrors `vit_face_embedding.dart`."""

from __future__ import annotations

from typing import List, Optional

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .vit_backbone import ViTBackbone


class ViTFaceEmbeddingGPU(Module):
    def __init__(
        self,
        *,
        imageSize: int,
        patchSize: int,
        embedSize: int,
        outputDim: int = 512,
        numLayers: int = 4,
    ):
        self.backbone = ViTBackbone(
            imageSize=imageSize,
            patchSize=patchSize,
            embedSize=embedSize,
            numLayers=numLayers,
        )
        self.final_projection: Optional[Layer]
        if embedSize != outputDim:
            self.final_projection = Layer(embedSize, outputDim, useGelu=False)
        else:
            self.final_projection = None

    def get_face_embedding(self, patchified_image: Tensor, tracker: List[Tensor]) -> Tensor:
        encoded = self.backbone.forward(patchified_image, tracker)
        tracker.append(encoded)
        cls_out = encoded.slice(0, 1)
        tracker.append(cls_out)
        face_vec = cls_out
        if self.final_projection is not None:
            face_vec = self.final_projection.forward(cls_out, tracker)
            tracker.append(face_vec)
        norm_vec = face_vec.normalize(eps=1e-10)
        tracker.append(norm_vec)
        return norm_vec

    getFaceEmbedding = get_face_embedding

    def parameters(self) -> List[Tensor]:
        ps = list(self.backbone.parameters())
        if self.final_projection is not None:
            ps.extend(self.final_projection.parameters())
        return ps
