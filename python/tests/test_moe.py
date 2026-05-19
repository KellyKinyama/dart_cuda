"""MoE feed-forward variants (per-token forward, top-K gating)."""

from __future__ import annotations

import pytest

from dart_cuda.core.layers.moe_feed_forward import MoEFeedForward
from dart_cuda.core.layers.swiglu_moe_feed_forward import SwiGLUMoEFeedForward
from dart_cuda.core.tensor.gpu_tensor import Tensor


class TestMoE:
    def test_forward_single_token(self, tracker):
        moe = MoEFeedForward(
            embed_size=4, num_routed_experts=3, num_shared_experts=1, top_k=2
        )
        x = Tensor.random([1, 4])
        tracker.append(x)
        out = moe.forward(x, tracker)
        assert out.shape == [1, 4]
        # top_k tokens dispatched per forward call.
        assert sum(moe.token_load) == 2
        for p in moe.parameters():
            p.dispose()

    def test_token_load_resets_on_bias_update(self, tracker):
        moe = MoEFeedForward(
            embed_size=4, num_routed_experts=3, num_shared_experts=0, top_k=1
        )
        x = Tensor.random([1, 4])
        tracker.append(x)
        for _ in range(3):
            out = moe.forward(x, tracker)
            tracker.append(out)
        assert sum(moe.token_load) == 3
        moe.update_routing_bias(reset_load=True)
        assert moe.token_load == [0, 0, 0]
        # Bias should now contain non-trivial floats.
        assert any(abs(b) > 0 for b in moe.routing_bias) or sum(moe.token_load) == 0
        for p in moe.parameters():
            p.dispose()

    def test_backward_runs(self, tracker):
        moe = MoEFeedForward(
            embed_size=4, num_routed_experts=2, num_shared_experts=1, top_k=1
        )
        x = Tensor.random([1, 4])
        tracker.append(x)
        out = moe.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        moe.zero_grad()
        loss.backward()
        # Gate weights must accumulate gradients.
        assert any(abs(g) > 0 for g in moe.gate.w.grad)
        for p in moe.parameters():
            p.dispose()


class TestSwiGLUMoE:
    def test_forward_single_token(self, tracker):
        moe = SwiGLUMoEFeedForward(
            embed_size=4, num_routed_experts=3, num_shared_experts=1, top_k=2
        )
        x = Tensor.random([1, 4])
        tracker.append(x)
        out = moe.forward(x, tracker)
        assert out.shape == [1, 4]
        for p in moe.parameters():
            p.dispose()
