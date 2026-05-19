"""Full encoder-decoder Transformer — port of
`lib/transformer/encoder_decoder_transformer.dart` (dart-torch).
"""

from __future__ import annotations

from typing import List, Sequence

from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder


class EncoderDecoderTransformer(Module):
    def __init__(
        self,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        embed_size: int,
        source_block_size: int,
        target_block_size: int,
        num_layers: int,
        num_heads: int,
    ):
        self.encoder = TransformerEncoder(
            vocab_size=source_vocab_size,
            embed_size=embed_size,
            block_size=source_block_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.decoder = TransformerDecoder(
            vocab_size=target_vocab_size,
            embed_size=embed_size,
            block_size=target_block_size,
            num_layers=num_layers,
            num_heads=num_heads,
            encoder_embed_size=embed_size,
        )

    def forward(
        self,
        source_idx: Sequence[int],
        target_idx: Sequence[int],
        tracker: List[Tensor],
    ) -> Tensor:
        encoder_output = self.encoder.forward(source_idx, tracker)
        logits = self.decoder.forward(target_idx, encoder_output, tracker)
        return logits

    def parameters(self) -> List[Tensor]:
        return [*self.encoder.parameters(), *self.decoder.parameters()]
