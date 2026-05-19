"""Layer / LayerNorm / RMSNorm / FeedForward / MLP tests."""

from __future__ import annotations

import pytest

from dart_cuda.core.layers.feed_forward import FeedForward
from dart_cuda.core.layers.layer_norm import LayerNorm
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.layers.nn import Layer
from dart_cuda.core.layers.rms_norm import RMSNorm
from dart_cuda.core.tensor.gpu_tensor import Tensor


class TestLinearLayer:
    def test_forward_shape(self, tracker):
        layer = Layer(4, 6, useGelu=False)
        x = Tensor.fill([3, 4], 1.0)
        tracker.append(x)
        out = layer.forward(x, tracker)
        assert out.shape == [3, 6]
        for p in layer.parameters():
            p.dispose()

    def test_parameters(self):
        layer = Layer(3, 5, useGelu=True)
        params = layer.parameters()
        try:
            assert len(params) == 2
            assert params[0].shape == [3, 5]
            assert params[1].shape == [1, 5]
        finally:
            for p in params:
                p.dispose()

    def test_backward_runs(self, tracker):
        layer = Layer(4, 2, useGelu=False)
        x = Tensor.from_list([1, 4], [0.1, 0.2, 0.3, 0.4])
        tracker.append(x)
        out = layer.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        layer.zero_grad()
        loss.backward()
        # Each parameter should have a gradient buffer with the right shape.
        for p in layer.parameters():
            g = p.grad
            assert len(g) == p.length
        for p in layer.parameters():
            p.dispose()


class TestLayerNorm:
    def test_forward_shape(self, tracker):
        ln = LayerNorm(8)
        x = Tensor.random([4, 8])
        tracker.append(x)
        out = ln.forward(x, tracker)
        assert out.shape == [4, 8]
        for p in ln.parameters():
            p.dispose()

    def test_backward_runs(self, tracker):
        ln = LayerNorm(4)
        x = Tensor.from_list([2, 4], [1, 2, 3, 4, 5, 6, 7, 8])
        tracker.append(x)
        out = ln.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        ln.zero_grad()
        loss.backward()
        # Trainable affine params should have grads.
        for p in ln.parameters():
            assert len(p.grad) == p.length
        for p in ln.parameters():
            p.dispose()


class TestRMSNorm:
    def test_gamma_is_ones(self):
        rms = RMSNorm(4)
        try:
            assert rms.gamma.data == pytest.approx([1, 1, 1, 1])
        finally:
            rms.gamma.dispose()

    def test_forward_normalises(self, tracker):
        # For x = [c, c, c, c] (all equal, nonzero), RMS(x) = |c|, so y ~ sign(c).
        rms = RMSNorm(4)
        x = Tensor.from_list([1, 4], [2.0, 2.0, 2.0, 2.0])
        tracker.append(x)
        out = rms.forward(x, tracker)
        data = out.data
        # All four entries should be ~1.0 (within eps slack).
        assert max(abs(v - 1.0) for v in data) < 1e-3
        for p in rms.parameters():
            p.dispose()

    def test_backward_runs(self, tracker):
        rms = RMSNorm(4)
        x = Tensor.from_list([2, 4], [1, 2, 3, 4, 0.5, 1.5, 2.5, 3.5])
        tracker.append(x)
        out = rms.forward(x, tracker)
        loss = out.sum()
        tracker.append(loss)
        rms.zero_grad()
        loss.backward()
        assert len(rms.gamma.grad) == rms.gamma.length
        for p in rms.parameters():
            p.dispose()


class TestFeedForward:
    def test_forward_preserves_dim(self, tracker):
        ff = FeedForward(8)
        x = Tensor.random([3, 8])
        tracker.append(x)
        out = ff.forward(x, tracker)
        assert out.shape == [3, 8]
        for p in ff.parameters():
            p.dispose()


class TestMLP:
    def test_forward_shape(self, tracker):
        mlp = MLP(3, [4, 2])
        x = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        tracker.append(x)
        out = mlp.forward(x, tracker)
        assert out.shape == [2, 2]
        for p in mlp.parameters():
            p.dispose()

    def test_overfit_tiny_regression(self, tracker):
        """End-to-end: MLP should reduce MSE on a tiny constant target."""
        mlp = MLP(2, [4, 1])
        x = Tensor.from_list([2, 2], [0.1, 0.2, 0.3, 0.4])
        y = Tensor.from_list([2, 1], [1.0, 0.0])
        tracker.extend([x, y])

        initial_loss = None
        final_loss = None
        for step in range(40):
            inner = []
            pred = mlp.forward(x, inner)
            loss = pred.mse_loss(y)
            inner.append(loss)
            mlp.zero_grad()
            loss.backward()
            mlp.step(0.1)
            val = float(loss.data[0])
            if step == 0:
                initial_loss = val
            final_loss = val
            for t in inner:
                t.dispose()

        assert initial_loss is not None and final_loss is not None
        assert final_loss < initial_loss
        for p in mlp.parameters():
            p.dispose()
