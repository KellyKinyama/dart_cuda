"""Transformer decoder — mirrors `lib/core/transformers/aft/transformer_decoder.dart`."""

from __future__ import annotations

import random
from typing import List, Sequence

from ...layers.layer_norm import LayerNorm
from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from .transformer_decoder_block import TransformerDecoderBlock


class TransformerDecoder(Module):
    def __init__(
        self,
        *,
        vocabSize: int = 4098,
        embedSize: int = 128,
        blockSize: int = 16,
        numLayers: int = 4,
        numHeads: int = 4,
        encoderEmbedSize: int = 128,
    ):
        self.vocab_size = vocabSize
        self.embed_size = embedSize
        self.block_size = blockSize
        self.num_layers = numLayers
        self.num_heads = numHeads
        self.encoder_embed_size = encoderEmbedSize

        self.wte = Tensor.random([vocabSize, embedSize])
        self.wpe = Tensor.random([blockSize, embedSize])
        self.blocks = [
            TransformerDecoderBlock(embedSize, numHeads, encoderEmbedSize, blockSize)
            for _ in range(numLayers)
        ]
        self.final_layer_norm = LayerNorm(embedSize)
        self.lm_head = Layer(embedSize, vocabSize, useGelu=False)

        # Re-init wte to small Gaussian-ish range, zero out lm_head bias.
        rng = random.Random()
        raw = self.wte.fetch_data()
        for i in range(len(raw)):
            raw[i] = (rng.random() * 2 - 1) * 0.02
        self.wte.data = raw

        params = self.lm_head.parameters()
        if len(params) > 1:
            bias = params[1]
            bias.data = [0.0] * bias.length

    def forward(self, idx: Sequence[int], encoder_output: Tensor, tracker: List[Tensor]) -> Tensor:
        T = len(idx)
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds max block size {self.block_size}")
        x = Tensor.embeddings(idx, self.wte, self.wpe)
        tracker.append(x)
        for block in self.blocks:
            x = block.forward(x, encoder_output, tracker)
        x_norm = self.final_layer_norm.forward(x, tracker)
        return self.lm_head.forward(x_norm, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = [self.wte, self.wpe]
        for b in self.blocks:
            ps.extend(b.parameters())
        ps.extend(self.final_layer_norm.parameters())
        ps.extend(self.lm_head.parameters())
        return ps
