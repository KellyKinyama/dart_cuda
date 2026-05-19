"""Decoder-only GPT-style transformer block.

Port of `lib/transformer/transformer_block.dart` (dart-torch).
Pre-norm: LN -> MHA -> residual -> LN -> FFN -> residual.
"""

from __future__ import annotations

from typing import List

from ...attention.multi_head_attention import MultiHeadAttention
from ...layers.feed_forward import FeedForward
from ...layers.layer_norm import LayerNorm
from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor


class TransformerBlock(Module):
    def __init__(self, embed_size: int, num_heads: int, *, masked: bool = False):
        self.embed_size = embed_size
        self.attention = MultiHeadAttention(num_heads, embed_size, masked=masked)
        self.ffn = FeedForward(embed_size)
        self.ln1 = LayerNorm(embed_size)
        self.ln2 = LayerNorm(embed_size)

    def forward(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        x_norm1 = self.ln1.forward(x, tracker)
        attn_out = self.attention.forward(x_norm1, tracker)
        x_res1 = x + attn_out
        tracker.extend([attn_out, x_res1])

        x_norm2 = self.ln2.forward(x_res1, tracker)
        ffn_out = self.ffn.forward(x_norm2, tracker)
        out = x_res1 + ffn_out
        tracker.extend([ffn_out, out])
        return out

    def parameters(self) -> List[Tensor]:
        return [
            *self.attention.parameters(),
            *self.ffn.parameters(),
            *self.ln1.parameters(),
            *self.ln2.parameters(),
        ]
