"""Text decoder — mirrors `modalities/text_decoder.dart`."""

from __future__ import annotations

from typing import List, Sequence

from ...layers.layer_norm import LayerNorm
from ...layers.nn import Layer, Module
from ...tensor.gpu_tensor import Tensor
from ..aft.text_decoder_block import TextDecoderBlock


class TextDecoder(Module):
    def __init__(
        self,
        *,
        vocabSize: int,
        maxSeqLen: int,
        embedSize: int,
        numLayers: int = 4,
        numHeads: int = 4,
    ):
        self.vocab_size = vocabSize
        self.max_seq_len = maxSeqLen
        self.embed_size = embedSize
        self.wte = Tensor.random([vocabSize, embedSize], scale=0.02)
        self.wpe = Tensor.random([maxSeqLen, embedSize], scale=0.02)
        self.blocks = [
            TextDecoderBlock(embedSize, numHeads, maxSeqLen)
            for _ in range(numLayers)
        ]
        self.final_layer_norm = LayerNorm(embedSize)
        self.lm_head = Layer(embedSize, vocabSize, useGelu=False)

    def forward(self, text_tokens: Sequence[int], encoder_output: Tensor, tracker: List[Tensor]) -> Tensor:
        T = len(text_tokens)
        if T > self.max_seq_len:
            raise ValueError("Text sequence length exceeds maxSeqLen")
        x = Tensor.embeddings(text_tokens, self.wte, self.wpe)
        tracker.append(x)
        for block in self.blocks:
            x = block.forward(x, encoder_output, tracker)
        normalized = self.final_layer_norm.forward(x, tracker)
        return self.lm_head.forward(normalized, tracker)

    def parameters(self) -> List[Tensor]:
        ps: List[Tensor] = [self.wte, self.wpe]
        for b in self.blocks:
            ps.extend(b.parameters())
        ps.extend(self.final_layer_norm.parameters())
        ps.extend(self.lm_head.parameters())
        return ps
