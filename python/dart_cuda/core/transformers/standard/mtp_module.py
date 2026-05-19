"""Multi-Token Prediction (MTP) module — adapted port of
`lib/transformer/mtp_module.dart` (dart-torch).

DeepSeek-V3 style: one MTP head at depth k predicts the token at position
(i + k + 1) using:
  * hidden state from previous depth at position i
  * embedding of the actual next token at depth k

Pipeline per token i:
  norm_h  = RMSNorm_h(prev_hidden_i)
  norm_e  = RMSNorm_e(shifted_embed_i)
  fused   = fusion_proj([norm_h ; norm_e])      # 2d -> d
  h_i^k   = TransformerDecoderBlock(fused)      # one decoder block
  out_i   = final_norm(h_i^k)                   # caller applies shared LM head

Note: the original dart-torch MTP module wraps a DeepSeek-SOTA decoder block
(RMSNorm + AFT + SwiGLU MoE). To keep this port self-contained and avoid
pulling in the full DeepSeek stack, we wrap the standard
`TransformerDecoderBlock` from this package. Build a custom MTP variant by
subclassing if you want a different inner block.
"""

from __future__ import annotations

from typing import List, Optional

from ...layers.nn import Layer, Module
from ...layers.rms_norm import RMSNorm
from ...tensor.gpu_tensor import Tensor
from .transformer_decoder_block import TransformerDecoderBlock


class MTPModule(Module):
    def __init__(
        self,
        embed_size: int,
        *,
        num_heads: int,
        encoder_embed_size: int,
    ):
        self.embed_size = embed_size
        self.norm_prev_hidden = RMSNorm(embed_size)
        self.norm_token_embed = RMSNorm(embed_size)
        # Fuse concat([norm_h, norm_e]) of feature dim 2*embed_size back to embed_size.
        self.fusion_proj = Layer(2 * embed_size, embed_size, useGelu=False)
        self.block = TransformerDecoderBlock(
            embed_size, num_heads, encoder_embed_size
        )
        self.final_norm = RMSNorm(embed_size)

    # dart-torch aliases
    @property
    def normPrevHidden(self) -> RMSNorm: return self.norm_prev_hidden
    @property
    def normTokenEmbed(self) -> RMSNorm: return self.norm_token_embed
    @property
    def fusionProj(self) -> Layer: return self.fusion_proj
    @property
    def finalNorm(self) -> RMSNorm: return self.final_norm

    def forward(
        self,
        prev_hidden: Tensor,
        shifted_embeds: Tensor,
        encoder_output: Tensor,
        tracker: List[Tensor],
    ) -> Tensor:
        """All three inputs share the same sequence length T.

        Shapes:
          prev_hidden     [T, embed_size]
          shifted_embeds  [T, embed_size]
          encoder_output  [T_enc, encoder_embed_size]
        Returns: [T, embed_size]
        """
        if prev_hidden.shape[0] != shifted_embeds.shape[0]:
            raise ValueError("prev_hidden and shifted_embeds must have the same length")

        nh = self.norm_prev_hidden.forward(prev_hidden, tracker)
        ne = self.norm_token_embed.forward(shifted_embeds, tracker)
        # Concat along feature dim: [T, 2*embed_size].
        fused_in = Tensor.concat([nh, ne])
        tracker.append(fused_in)
        fused = self.fusion_proj.forward(fused_in, tracker)
        y = self.block.forward(fused, encoder_output, tracker)
        return self.final_norm.forward(y, tracker)

    def parameters(self) -> List[Tensor]:
        return [
            *self.norm_prev_hidden.parameters(),
            *self.norm_token_embed.parameters(),
            *self.fusion_proj.parameters(),
            *self.block.parameters(),
            *self.final_norm.parameters(),
        ]
