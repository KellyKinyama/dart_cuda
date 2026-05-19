"""Transformer encoder — mirrors `lib/core/transformers/aft/transformer_encoder.dart`."""

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
        vocabSize: int,
        embedSize: int,
        blockSize: int,
        numLayers: int = 6,
        numHeads: int = 8,
    ):
        assert embedSize % numHeads == 0, "embedSize must be divisible by numHeads"
        self.vocab_size = vocabSize
        self.embed_size = embedSize
        self.block_size = blockSize
        self.num_layers = numLayers
        self.num_heads = numHeads

        self.wte = Tensor.random([max(vocabSize, 1), embedSize], scale=0.02)
        self.wpe = Tensor.random([blockSize, embedSize], scale=0.02)
        self.blocks = [
            TransformerEncoderBlock(embedSize, numHeads, blockSize)
            for _ in range(numLayers)
        ]
        self.final_layer_norm = LayerNorm(embedSize)

    @property
    def vocabSize(self) -> int:
        return self.vocab_size

    @property
    def embedSize(self) -> int:
        return self.embed_size

    @property
    def blockSize(self) -> int:
        return self.block_size

    @property
    def numLayers(self) -> int:
        return self.num_layers

    @property
    def numHeads(self) -> int:
        return self.num_heads

    @property
    def finalLayerNorm(self) -> LayerNorm:
        return self.final_layer_norm

    def forward(self, idx: Sequence[int], tracker: List[Tensor]) -> Tensor:
        T = len(idx)
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}")
        x = Tensor.embeddings(idx, self.wte, self.wpe)
        tracker.append(x)
        return self._process(x, tracker)

    def forward_embeddings(self, embedded: Tensor, tracker: List[Tensor]) -> Tensor:
        if embedded.shape[0] > self.block_size:
            raise ValueError("Input sequence length exceeds block size")
        return self._process(embedded, tracker)

    forwardEmbeddings = forward_embeddings

    def _process(self, x: Tensor, tracker: List[Tensor]) -> Tensor:
        cur = x
        for block in self.blocks:
            cur = block.forward(cur, tracker)
        return self.final_layer_norm.forward(cur, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = [self.wte, self.wpe]
        for b in self.blocks:
            ps.extend(b.parameters())
        ps.extend(self.final_layer_norm.parameters())
        return ps
