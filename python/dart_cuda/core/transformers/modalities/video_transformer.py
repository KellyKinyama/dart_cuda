"""Video transformer — mirrors `modalities/video_transformer.dart`."""

from __future__ import annotations

from typing import List, Optional

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from ..aft.transformer_encoder import TransformerEncoder


class VideoTransformer(Module):
    def __init__(
        self,
        *,
        frameEmbedDim: int,
        embedSize: int,
        maxVideoSequenceLength: int,
        numClasses: int,
        numLayers: int = 2,
        numHeads: int = 4,
    ):
        self.frame_embed_dim = frameEmbedDim
        self.embed_size = embedSize
        self.max_video_sequence_length = maxVideoSequenceLength
        self.num_classes = numClasses

        self.frame_projection: Optional[Layer]
        if frameEmbedDim != embedSize:
            self.frame_projection = Layer(frameEmbedDim, embedSize, useGelu=False)
        else:
            self.frame_projection = None
        self.pos_embeddings = Tensor.random([maxVideoSequenceLength, embedSize])
        self.transformer_encoder = TransformerEncoder(
            vocabSize=0,
            embedSize=embedSize,
            blockSize=maxVideoSequenceLength,
            numLayers=numLayers,
            numHeads=numHeads,
        )
        self.mlp_head = Layer(embedSize, numClasses, useGelu=False)

    @property
    def embedSize(self) -> int:
        return self.embed_size

    @property
    def frameProjection(self) -> Optional[Layer]:
        return self.frame_projection

    @property
    def posEmbeddings(self) -> Tensor:
        return self.pos_embeddings

    @property
    def transformerEncoder(self) -> TransformerEncoder:
        return self.transformer_encoder

    def forward(self, video_embeddings: Tensor, tracker: List[Tensor]) -> Tensor:
        num_frames = video_embeddings.shape[0]
        if num_frames > self.max_video_sequence_length:
            raise ValueError("Video too long for maxVideoSequenceLength")
        x = (
            self.frame_projection.forward(video_embeddings, tracker)
            if self.frame_projection is not None
            else video_embeddings
        )
        pos = self.pos_embeddings.slice(0, num_frames)
        x_with_pos = x + pos
        tracker.append(x_with_pos)
        encoded = self.transformer_encoder.forward_embeddings(x_with_pos, tracker)
        pooled = encoded.mean()
        tracker.append(pooled)
        return self.mlp_head.forward(pooled, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = []
        if self.frame_projection is not None:
            ps.extend(self.frame_projection.parameters())
        ps.append(self.pos_embeddings)
        ps.extend(self.transformer_encoder.parameters())
        ps.extend(self.mlp_head.parameters())
        return ps
