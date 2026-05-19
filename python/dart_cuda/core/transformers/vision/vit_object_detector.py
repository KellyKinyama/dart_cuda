"""ViT object detector — mirrors `vit_object_detector.dart`."""

from __future__ import annotations

from typing import Dict, List

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .vit_backbone import ViTBackbone


class ViTObjectDetector(Module):
    def __init__(
        self,
        *,
        imageSize: int,
        patchSize: int,
        numChannels: int,
        embedSize: int,
        numLayers: int,
        numHeads: int,
        numClasses: int,
        numQueries: int,
    ):
        self.embed_size = embedSize
        self.num_classes = numClasses
        self.num_queries = numQueries
        self.backbone = ViTBackbone(
            imageSize=imageSize,
            patchSize=patchSize,
            numChannels=numChannels,
            embedSize=embedSize,
            numLayers=numLayers,
            numHeads=numHeads,
        )
        self.class_head = Layer(embedSize, numQueries * (numClasses + 1), useGelu=False)
        self.box_head = Layer(embedSize, numQueries * 4, useGelu=False)

    def forward(self, patchified_image: Tensor, tracker: List[Tensor]) -> Dict[str, Tensor]:
        encoded = self.backbone.forward(patchified_image, tracker)
        cls_feat = encoded.get_row(0)
        tracker.append(cls_feat)

        raw_logits = self.class_head.forward(cls_feat, tracker)
        logits = raw_logits.reshape([self.num_queries, self.num_classes + 1])

        raw_boxes = self.box_head.forward(cls_feat, tracker)
        boxes = raw_boxes.reshape([self.num_queries, 4]).sigmoid()
        tracker.append(boxes)

        return {"logits": logits, "boxes": boxes}

    def parameters(self) -> List[Tensor]:
        return [
            *self.backbone.parameters(),
            *self.class_head.parameters(),
            *self.box_head.parameters(),
        ]
