"""Attention modules: SDPA, SelfAttention, CrossAttention, multi-head variants."""

from __future__ import annotations

import pytest

from dart_cuda.core.attention.cross_attention import CrossAttention
from dart_cuda.core.attention.multi_head_attention import MultiHeadAttention
from dart_cuda.core.attention.multi_head_cross_attention import MultiHeadCrossAttention
from dart_cuda.core.attention.self_attention import (
    SelfAttention,
    _causal_mask,
    scaled_dot_product_attention,
)
from dart_cuda.core.tensor.gpu_tensor import Tensor


class TestCausalMask:
    def test_mask_shape_and_values(self):
        m = _causal_mask(3)
        try:
            assert m.shape == [3, 3]
            d = m.data
            # diagonal & below = 0
            assert d[0] == 0 and d[3] == 0 and d[4] == 0 and d[6] == 0
            # upper triangle = -1e9
            assert d[1] == pytest.approx(-1e9)
            assert d[2] == pytest.approx(-1e9)
            assert d[5] == pytest.approx(-1e9)
        finally:
            m.dispose()


class TestSDPA:
    def test_sdpa_unmasked_shape(self, tracker):
        T, H = 3, 4
        q = Tensor.random([T, H])
        k = Tensor.random([T, H])
        v = Tensor.random([T, H])
        tracker.extend([q, k, v])
        out = scaled_dot_product_attention(q, k, v, tracker, masked=False)
        assert out.shape == [T, H]

    def test_sdpa_masked_shape(self, tracker):
        T, H = 4, 2
        q = Tensor.random([T, H])
        k = Tensor.random([T, H])
        v = Tensor.random([T, H])
        tracker.extend([q, k, v])
        out = scaled_dot_product_attention(q, k, v, tracker, masked=True)
        assert out.shape == [T, H]


class TestSelfAttention:
    def test_forward_shape(self, tracker):
        sa = SelfAttention(embed_size=8, head_size=4)
        x = Tensor.random([3, 8])
        tracker.append(x)
        out = sa.forward(x, tracker)
        assert out.shape == [3, 4]
        for p in sa.parameters():
            p.dispose()

    def test_backward_runs_masked(self, tracker):
        sa = SelfAttention(embed_size=4, head_size=4, masked=True)
        x = Tensor.random([4, 4])
        tracker.append(x)
        out = sa.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        sa.zero_grad()
        loss.backward()
        for p in sa.parameters():
            assert len(p.grad) == p.length
        for p in sa.parameters():
            p.dispose()


class TestCrossAttention:
    def test_forward_shape(self, tracker):
        ca = CrossAttention(decoder_embed_size=8, encoder_embed_size=6, head_size=4)
        dec_x = Tensor.random([3, 8])
        enc_x = Tensor.random([5, 6])
        tracker.extend([dec_x, enc_x])
        out = ca.forward(dec_x, enc_x, tracker)
        assert out.shape == [3, 4]
        for p in ca.parameters():
            p.dispose()


class TestMultiHeadAttention:
    def test_divides_evenly(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(num_heads=3, embed_size=8)

    def test_forward_shape(self, tracker):
        mha = MultiHeadAttention(num_heads=4, embed_size=16)
        x = Tensor.random([3, 16])
        tracker.append(x)
        out = mha.forward(x, tracker)
        assert out.shape == [3, 16]
        for p in mha.parameters():
            p.dispose()

    def test_masked_backward(self, tracker):
        mha = MultiHeadAttention(num_heads=2, embed_size=4, masked=True)
        x = Tensor.random([3, 4])
        tracker.append(x)
        out = mha.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        mha.zero_grad()
        loss.backward()
        for p in mha.parameters():
            assert len(p.grad) == p.length
        for p in mha.parameters():
            p.dispose()


class TestMultiHeadCrossAttention:
    def test_forward_shape(self, tracker):
        mhca = MultiHeadCrossAttention(
            num_heads=2, decoder_embed_size=8, encoder_embed_size=6
        )
        dec = Tensor.random([3, 8])
        enc = Tensor.random([5, 6])
        tracker.extend([dec, enc])
        out = mhca.forward(dec, enc, tracker)
        assert out.shape == [3, 8]
        for p in mhca.parameters():
            p.dispose()
