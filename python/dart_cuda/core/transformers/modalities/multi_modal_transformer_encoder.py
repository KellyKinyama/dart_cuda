"""Multi-modal encoder for downstream decoder — mirrors
`modalities/multi_modal_transformer_encoder.dart`."""

from __future__ import annotations

from typing import List, Sequence

from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor
from ..aft.transformer_encoder import TransformerEncoder
from .audio_transformer import AudioTransformer
from .text_transformer import TextTransformer
from .video_transformer import VideoTransformer


class MultimodalTransformerEncoder(Module):
    def __init__(
        self,
        *,
        audio_encoder: AudioTransformer,
        video_encoder: VideoTransformer,
        text_encoder: TextTransformer,
        jointEmbedSize: int,
        fusionLayers: int = 2,
        fusionHeads: int = 4,
        maxTotalSeqLen: int = 200,
    ):
        assert (
            audio_encoder.embed_size == jointEmbedSize
            and video_encoder.embed_size == jointEmbedSize
            and text_encoder.embed_size == jointEmbedSize
        ), "All encoder outputs must match jointEmbedSize for concatenation"
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.joint_embed_size = jointEmbedSize
        self.fusion_encoder = TransformerEncoder(
            vocabSize=0,
            embedSize=jointEmbedSize,
            blockSize=maxTotalSeqLen,
            numLayers=fusionLayers,
            numHeads=fusionHeads,
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
        input_text_tokens: Sequence[int],
        tracker: List[Tensor],
    ) -> Tensor:
        audio_seq = self.audio_encoder.forward(audio, tracker)
        video_seq = self.video_encoder.forward(video, tracker)
        text_seq = self.text_encoder.forward(input_text_tokens, tracker)
        combined = Tensor.concat([audio_seq, video_seq, text_seq])
        tracker.append(combined)
        return self.fusion_encoder.forward_embeddings(combined, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.audio_encoder.parameters(),
            *self.video_encoder.parameters(),
            *self.text_encoder.parameters(),
            *self.fusion_encoder.parameters(),
        ]
