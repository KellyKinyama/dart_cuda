"""Decoder transformer block — port of
`lib/transformer/transformer_decoder_block.dart` (dart-torch).

Three sub-layers (pre-norm):
    1. Masked multi-head self-attention.
    2. Multi-head cross-attention over encoder output.
    3. Position-wise feed-forward network.
"""

from __future__ import annotations

from typing import List

from ...attention.multi_head_attention import MultiHeadAttention
from ...attention.multi_head_cross_attention import MultiHeadCrossAttention
from ...layers.feed_forward import FeedForward
from ...layers.layer_norm import LayerNorm
from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor


class TransformerDecoderBlock(Module):
    def __init__(self, embed_size: int, num_heads: int, encoder_embed_size: int):
        self.embed_size = embed_size
        self.encoder_embed_size = encoder_embed_size
        self.self_attention = MultiHeadAttention(num_heads, embed_size, masked=True)
        self.cross_attention = MultiHeadCrossAttention(
            num_heads, embed_size, encoder_embed_size
        )
        self.ffn = FeedForward(embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)
        self.ln3 = LayerNorm(embed_size)

    # dart-torch aliases
    @property
    def selfAttention(self) -> MultiHeadAttention: return self.self_attention
    @property
    def crossAttention(self) -> MultiHeadCrossAttention: return self.cross_attention

    def forward(
        self,
        x_decoder: Tensor,
        x_encoder: Tensor,
        tracker: List[Tensor],
    ) -> Tensor:
        # 1. Masked self-attention.
        x_norm1 = self.ln1.forward(x_decoder, tracker)
        self_out = self.self_attention.forward(x_norm1, tracker)
        x_res1 = x_decoder + self_out
        tracker.extend([self_out, x_res1])

        # 2. Cross-attention over encoder output.
        x_norm2 = self.ln2.forward(x_res1, tracker)
        cross_out = self.cross_attention.forward(x_norm2, x_encoder, tracker)
        x_res2 = x_res1 + cross_out
        tracker.extend([cross_out, x_res2])

        # 3. FFN.
        x_norm3 = self.ln3.forward(x_res2, tracker)
        ffn_out = self.ffn.forward(x_norm3, tracker)
        out = x_res2 + ffn_out
        tracker.extend([ffn_out, out])
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.self_attention.parameters(),
            *self.cross_attention.parameters(),
            *self.ffn.parameters(),
            *self.ln1.parameters(),
            *self.ln2.parameters(),
            *self.ln3.parameters(),
        ]
