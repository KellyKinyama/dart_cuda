"""Transformer encoder — port of
`lib/transformer/transformer_encoder.dart` (dart-torch).

Takes a token-id sequence and returns contextualised embeddings
(no LM head; for that, use `Transformer`).
"""

from __future__ import annotations

from typing import List, Sequence

from ...layers.layer_norm import LayerNorm
from ...layers.nn import Module
from ...tensor.gpu_tensor import Tensor
from .transformer_encoder_block import TransformerEncoderBlock


class TransformerEncoder(Module):
    def __init__(
        self,
        *,
        vocab_size: int = 100,
        embed_size: int = 64,
        block_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        assert embed_size % num_heads == 0
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.wte = Tensor.random([max(vocab_size, 1), embed_size], scale=0.02)
        self.wpe = Tensor.random([block_size, embed_size], scale=0.02)
        self.blocks = [
            TransformerEncoderBlock(embed_size, num_heads)
            for _ in range(num_layers)
        ]
        self.final_layer_norm = LayerNorm(embed_size)

    # dart-torch aliases
    @property
    def vocabSize(self) -> int: return self.vocab_size
    @property
    def embedSize(self) -> int: return self.embed_size
    @property
    def blockSize(self) -> int: return self.block_size
    @property
    def numLayers(self) -> int: return self.num_layers
    @property
    def numHeads(self) -> int: return self.num_heads
    @property
    def finalLayerNorm(self) -> LayerNorm: return self.final_layer_norm

    def forward(self, idx: Sequence[int], tracker: List[Tensor]) -> Tensor:
        T = len(idx)
        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block size {self.block_size}"
            )
        x = Tensor.embeddings(idx, self.wte, self.wpe)
        tracker.append(x)
        for b in self.blocks:
            x = b.forward(x, tracker)
        return self.final_layer_norm.forward(x, tracker)

    def forward_embeddings(self, embedded: Tensor, tracker: List[Tensor]) -> Tensor:
        if embedded.shape[0] > self.block_size:
            raise ValueError("Input sequence length exceeds block size")
        x = embedded
        for b in self.blocks:
            x = b.forward(x, tracker)
        return self.final_layer_norm.forward(x, tracker)

    forwardEmbeddings = forward_embeddings

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = [self.wte, self.wpe]
        for b in self.blocks:
            ps.extend(b.parameters())
        ps.extend(self.final_layer_norm.parameters())
        return ps
