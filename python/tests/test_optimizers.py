"""Optimizer tests — Adam and SGD reduce loss on a tiny regression."""

from __future__ import annotations

import pytest

from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.optimizers.adam import Adam
from dart_cuda.core.optimizers.stochastic_grad_desc import SDG
from dart_cuda.core.tensor.gpu_tensor import Tensor


def _train_with(optimizer_cls, steps: int = 60) -> tuple[float, float]:
    mlp = MLP(2, [4, 1])
    x = Tensor.from_list([4, 2], [0, 0, 0, 1, 1, 0, 1, 1])
    y = Tensor.from_list([4, 1], [0, 1, 1, 0])
    opt = optimizer_cls(mlp.parameters(), lr=0.05)

    initial = None
    final = None
    for step in range(steps):
        tracker = []
        pred = mlp.forward(x, tracker)
        diff = pred - y
        loss = diff.pow(2.0).mean()
        tracker.extend([pred, diff, loss])

        opt.zero_grad()
        loss.backward()
        opt.step()
        val = float(loss.data[0])
        if step == 0:
            initial = val
        final = val
        for t in tracker:
            t.dispose()

    x.dispose()
    y.dispose()
    for p in mlp.parameters():
        p.dispose()
    opt.dispose()
    return initial, final


class TestAdam:
    def test_decreases_loss(self):
        initial, final = _train_with(Adam, steps=80)
        assert initial is not None and final is not None
        assert final < initial


class TestSGD:
    def test_decreases_loss(self):
        initial, final = _train_with(SDG, steps=80)
        assert initial is not None and final is not None
        assert final < initial


class TestOptimizerDispose:
    def test_dispose_clears_state(self):
        params = [Tensor.zeros([2, 2]) for _ in range(2)]
        opt = Adam(params, lr=1e-3)
        assert len(opt.m) == 2 and len(opt.v) == 2
        opt.dispose()
        assert opt.m == [] and opt.v == []
        for p in params:
            p.dispose()
