"""Standard (softmax-attention) Transformer modules."""

from __future__ import annotations

import pytest

from dart_cuda.core.tensor.gpu_tensor import Tensor
from dart_cuda.core.transformers.standard import (
    EncoderDecoderTransformer,
    MTPModule,
    Transformer,
    TransformerBlock,
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
)


class TestBlocks:
    def test_transformer_block_forward(self, tracker):
        blk = TransformerBlock(embed_size=8, num_heads=2, masked=False)
        x = Tensor.random([4, 8])
        tracker.append(x)
        out = blk.forward(x, tracker)
        assert out.shape == [4, 8]
        for p in blk.parameters():
            p.dispose()

    def test_encoder_block_forward(self, tracker):
        blk = TransformerEncoderBlock(embed_size=8, num_heads=4)
        x = Tensor.random([3, 8])
        tracker.append(x)
        out = blk.forward(x, tracker)
        assert out.shape == [3, 8]
        for p in blk.parameters():
            p.dispose()

    def test_decoder_block_forward(self, tracker):
        blk = TransformerDecoderBlock(
            embed_size=8, num_heads=2, encoder_embed_size=8
        )
        x = Tensor.random([3, 8])
        enc = Tensor.random([5, 8])
        tracker.extend([x, enc])
        out = blk.forward(x, enc, tracker)
        assert out.shape == [3, 8]
        for p in blk.parameters():
            p.dispose()

    def test_decoder_block_backward(self, tracker):
        blk = TransformerDecoderBlock(
            embed_size=4, num_heads=2, encoder_embed_size=4
        )
        x = Tensor.random([3, 4])
        enc = Tensor.random([4, 4])
        tracker.extend([x, enc])
        out = blk.forward(x, enc, tracker)
        loss = out.sum()
        tracker.append(loss)
        blk.zero_grad()
        loss.backward()
        for p in blk.parameters():
            assert len(p.grad) == p.length
        for p in blk.parameters():
            p.dispose()


class TestGPT:
    def test_forward_shape(self, tracker):
        m = Transformer(
            vocab_size=10, embed_size=8, block_size=6, num_layers=1, num_heads=2
        )
        logits = m.forward([0, 1, 2, 3, 4], tracker)
        assert logits.shape == [5, 10]
        for p in m.parameters():
            p.dispose()

    def test_sequence_too_long_raises(self, tracker):
        m = Transformer(
            vocab_size=4, embed_size=4, block_size=2, num_layers=1, num_heads=1
        )
        try:
            with pytest.raises(ValueError):
                m.forward([0, 1, 2], tracker)
        finally:
            for p in m.parameters():
                p.dispose()

    def test_backward_runs(self, tracker):
        m = Transformer(
            vocab_size=6, embed_size=4, block_size=4, num_layers=1, num_heads=2
        )
        logits = m.forward([0, 1, 2, 3], tracker)
        loss = logits.cross_entropy([1, 2, 3, 0])
        tracker.append(loss)
        m.zero_grad()
        loss.backward()
        # Embedding tables should accumulate grads.
        assert any(abs(g) > 0 for g in m.wte.grad)
        for p in m.parameters():
            p.dispose()


class TestEncoderDecoder:
    def test_forward_shape(self, tracker):
        m = EncoderDecoderTransformer(
            source_vocab_size=8,
            target_vocab_size=6,
            embed_size=8,
            source_block_size=4,
            target_block_size=4,
            num_layers=1,
            num_heads=2,
        )
        logits = m.forward([0, 1, 2, 3], [0, 1, 2], tracker)
        assert logits.shape == [3, 6]
        for p in m.parameters():
            p.dispose()


class TestEncoderOnly:
    def test_forward_shape(self, tracker):
        m = TransformerEncoder(
            vocab_size=8, embed_size=4, block_size=4, num_layers=1, num_heads=2
        )
        out = m.forward([0, 1, 2, 3], tracker)
        assert out.shape == [4, 4]
        for p in m.parameters():
            p.dispose()


class TestDecoderOnly:
    def test_forward_shape(self, tracker):
        m = TransformerDecoder(
            vocab_size=5,
            embed_size=4,
            block_size=3,
            num_layers=1,
            num_heads=2,
            encoder_embed_size=4,
        )
        enc = Tensor.random([4, 4])
        tracker.append(enc)
        out = m.forward([0, 1, 2], enc, tracker)
        assert out.shape == [3, 5]
        for p in m.parameters():
            p.dispose()


class TestMTPModule:
    def test_forward_shape(self, tracker):
        mtp = MTPModule(embed_size=4, num_heads=2, encoder_embed_size=4)
        prev_h = Tensor.random([3, 4])
        shifted = Tensor.random([3, 4])
        enc = Tensor.random([4, 4])
        tracker.extend([prev_h, shifted, enc])
        out = mtp.forward(prev_h, shifted, enc, tracker)
        assert out.shape == [3, 4]
        for p in mtp.parameters():
            p.dispose()
