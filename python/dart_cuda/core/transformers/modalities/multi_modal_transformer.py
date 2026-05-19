"""Multi-modal transformer (audio + video) — mirrors
`modalities/multi_modal_transformer.dart`."""

from __future__ import annotations

from typing import List

from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .audio_transformer import AudioTransformer
from .video_transformer import VideoTransformer


class MultimodalTransformer(Module):
    def __init__(
        self,
        *,
        audio_model: AudioTransformer,
        video_model: VideoTransformer,
        numClasses: int,
    ):
        self.audio_model = audio_model
        self.video_model = video_model
        self.num_classes = numClasses
        self.fusion_head = Layer(
            audio_model.embed_size + video_model.embed_size,
            numClasses,
            useGelu=False,
        )

    def forward(self, audio_in: Tensor, video_in: Tensor, tracker: List[Tensor]) -> Tensor:
        a_embed = self.audio_model.feature_projection.forward(audio_in, tracker)
        a_pos = self.audio_model.pos_embeddings.slice(0, audio_in.shape[0])
        a_combined = a_embed + a_pos
        tracker.append(a_combined)
        a_encoded = self.audio_model.transformer_encoder.forward_embeddings(a_combined, tracker)
        a_pooled = a_encoded.mean()
        tracker.append(a_pooled)

        if self.video_model.frame_projection is not None:
            v_proj = self.video_model.frame_projection.forward(video_in, tracker)
        else:
            v_proj = video_in
        v_pos = self.video_model.pos_embeddings.slice(0, video_in.shape[0])
        v_combined = v_proj + v_pos
        tracker.append(v_combined)
        v_encoded = self.video_model.transformer_encoder.forward_embeddings(v_combined, tracker)
        v_pooled = v_encoded.mean()
        tracker.append(v_pooled)

        fused = Tensor.concat([a_pooled, v_pooled])
        tracker.append(fused)
        return self.fusion_head.forward(fused, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.audio_model.parameters(),
            *self.video_model.parameters(),
            *self.fusion_head.parameters(),
        ]
