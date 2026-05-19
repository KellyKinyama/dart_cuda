"""Multi-modal transformer (audio + video + text) — mirrors
`modalities/multi_modal_transformer2.dart`."""

from __future__ import annotations

from typing import List, Sequence

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .audio_transformer import AudioTransformer
from .text_transformer import TextTransformer
from .video_transformer import VideoTransformer


class MultimodalTransformer2(Module):
    def __init__(
        self,
        *,
        audio_model: AudioTransformer,
        video_model: VideoTransformer,
        text_model: TextTransformer,
        numClasses: int,
    ):
        self.audio_model = audio_model
        self.video_model = video_model
        self.text_model = text_model
        self.fusion_layer = Layer(
            audio_model.embed_size + video_model.embed_size + text_model.embed_size,
            numClasses,
            useGelu=False,
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
        text_tokens: Sequence[int],
        tracker: List[Tensor],
    ) -> Tensor:
        a = self.audio_model.forward(audio, tracker).mean()
        v = self.video_model.forward(video, tracker).mean()
        t = self.text_model.forward(text_tokens, tracker).mean()
        combined = Tensor.concat([a, v, t])
        tracker.append(combined)
        return self.fusion_layer.forward(combined, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.audio_model.parameters(),
            *self.video_model.parameters(),
            *self.text_model.parameters(),
            *self.fusion_layer.parameters(),
        ]
