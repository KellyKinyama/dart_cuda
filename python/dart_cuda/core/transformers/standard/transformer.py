"""GPT-style decoder-only Transformer — port of
`lib/transformer/transformer.dart` (dart-torch).
"""

from __future__ import annotations

from typing import List, Sequence

from ...layers.layer_norm import LayerNorm
from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .transformer_block import TransformerBlock


class Transformer(Module):
    def __init__(
        self,
        *,
        vocab_size: int = 50,
        embed_size: int = 32,
        block_size: int = 8,
        num_layers: int = 4,
        num_heads: int = 4,
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
            TransformerBlock(embed_size, num_heads, masked=True)
            for _ in range(num_layers)
        ]
        self.final_layer_norm = LayerNorm(embed_size)
        self.lm_head = Layer(embed_size, vocab_size, useGelu=False)

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
    @property
    def lmHead(self) -> Layer: return self.lm_head

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
        x = self.final_layer_norm.forward(x, tracker)
        logits = self.lm_head.forward(x, tracker)
        return logits

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = [self.wte, self.wpe]
        for b in self.blocks:
            ps.extend(b.parameters())
        ps.extend(self.final_layer_norm.parameters())
        ps.extend(self.lm_head.parameters())
        return ps
