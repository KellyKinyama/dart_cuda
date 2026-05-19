"""Autograd verification — mirrors `test/scratch/test_tensor.dart`."""

from __future__ import annotations

import math

import pytest

from dart_cuda.core.tensor.gpu_tensor import Tensor


class TestReductionGrads:
    def test_sum_grad_is_ones(self):
        x = Tensor.from_list([2, 2], [1, 2, 3, 4])
        try:
            loss = x.sum()
            loss.backward()
            assert x.grad == pytest.approx([1, 1, 1, 1], abs=1e-4)
            loss.dispose()
        finally:
            x.dispose()

    def test_mean_grad(self):
        x = Tensor.from_list([1, 4], [1.0, 2.0, 3.0, 4.0])
        try:
            loss = x.mean()
            loss.backward()
            assert x.grad == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=1e-4)
            loss.dispose()
        finally:
            x.dispose()


class TestUnaryGrads:
    def test_sigmoid_zero(self):
        x = Tensor.from_list([1, 1], [0.0])
        try:
            s = x.sigmoid()
            loss = s.sum()
            loss.backward()
            assert x.grad[0] == pytest.approx(0.25, abs=1e-4)
            s.dispose()
            loss.dispose()
        finally:
            x.dispose()

    def test_pow(self):
        x = Tensor.from_list([1, 1], [3.0])
        try:
            loss = x.pow(2.0).sum()
            loss.backward()
            assert x.grad[0] == pytest.approx(6.0, abs=1e-3)
        finally:
            x.dispose()

    def test_log(self):
        x = Tensor.from_list([1, 1], [2.0])
        try:
            loss = x.log().sum()
            loss.backward()
            assert x.grad[0] == pytest.approx(0.5, abs=1e-4)
        finally:
            x.dispose()

    def test_relu_negative_zero_grad(self):
        x = Tensor.from_list([1, 2], [-1.0, 2.0])
        try:
            loss = x.relu().sum()
            loss.backward()
            # ReLU gradient is 0 for negative inputs, 1 for positive.
            assert x.grad[0] == pytest.approx(0.0, abs=1e-4)
            assert x.grad[1] == pytest.approx(1.0, abs=1e-4)
        finally:
            x.dispose()


class TestMatmul:
    def test_simple_matmul_grad(self):
        a = Tensor.from_list([1, 2], [2.0, 3.0])
        b = Tensor.from_list([2, 1], [4.0, 5.0])
        try:
            loss = a.matmul(b).sum()
            loss.backward()
            assert a.grad == pytest.approx([4.0, 5.0], abs=1e-4)
            assert b.grad == pytest.approx([2.0, 3.0], abs=1e-4)
        finally:
            a.dispose()
            b.dispose()

    def test_matmul_weight_update(self):
        X = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        W = Tensor.fill([3, 2], 0.5)
        try:
            out = X.matmul(W)
            loss = out.sum()
            loss.backward()
            # dL/dW[i,j] = sum over rows of X[r,i]
            assert W.grad[0] == pytest.approx(5.0, abs=1e-4)   # x[0,0]+x[1,0]=1+4
            assert W.grad[5] == pytest.approx(9.0, abs=1e-4)   # x[0,2]+x[1,2]=3+6
            out.dispose()
            loss.dispose()
        finally:
            X.dispose()
            W.dispose()


class TestSoftmax:
    def test_softmax_forward(self):
        x = Tensor.from_list([1, 3], [1.0, 2.0, 3.0])
        try:
            y = x.softmax()
            data = y.fetch_data()
            # Standard softmax values.
            assert data[0] == pytest.approx(0.0900, abs=1e-3)
            assert data[1] == pytest.approx(0.2447, abs=1e-3)
            assert data[2] == pytest.approx(0.6652, abs=1e-3)
            y.dispose()
        finally:
            x.dispose()

    def test_softmax_sum_grad_zero(self):
        """d/dx sum(softmax(x)) == 0 (a constant)."""
        x = Tensor.from_list([1, 3], [1.0, 2.0, 3.0])
        try:
            loss = x.softmax().sum()
            loss.backward()
            assert max(abs(g) for g in x.grad) < 1e-4
            loss.dispose()
        finally:
            x.dispose()

    def test_softmax_weighted_grad_analytic(self):
        """For L = sum_i w_i * y_i, dL/dx_i = y_i*(w_i - sum_j y_j w_j)."""
        x = Tensor.from_list([1, 3], [0.5, -0.5, 1.0])
        w_vals = [1.0, 2.0, 3.0]
        w = Tensor.from_list([1, 3], w_vals)
        try:
            y = x.softmax()
            y_data = y.fetch_data()
            wsum = sum(y_data[i] * w_vals[i] for i in range(3))
            expected = [y_data[i] * (w_vals[i] - wsum) for i in range(3)]
            loss = (y * w).sum()
            loss.backward()
            assert x.grad == pytest.approx(expected, abs=1e-4)
            y.dispose()
            loss.dispose()
        finally:
            x.dispose()
            w.dispose()


class TestTranspose:
    def test_transpose_forward(self):
        x = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        try:
            y = x.transpose()
            try:
                assert y.shape == [3, 2]
                assert y.data == pytest.approx([1, 4, 2, 5, 3, 6])
            finally:
                y.dispose()
        finally:
            x.dispose()

    def test_transpose_grad_is_ones(self):
        x = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        try:
            loss = x.transpose().sum()
            loss.backward()
            assert x.grad == pytest.approx([1, 1, 1, 1, 1, 1], abs=1e-4)
            loss.dispose()
        finally:
            x.dispose()


class TestLayerNormOp:
    def test_normalize_constant_input_zero_grad(self):
        x = Tensor.from_list([1, 4], [10.0, 2.0, 5.0, 3.0])
        try:
            loss = x.normalize(eps=1e-6).sum()
            loss.backward()
            assert max(abs(g) for g in x.grad) < 1e-3
            loss.dispose()
        finally:
            x.dispose()


class TestCrossEntropy:
    def test_cross_entropy_grad_sign(self):
        x = Tensor.from_list([1, 2], [0.0, 0.0])
        try:
            loss = x.cross_entropy([1])
            loss.backward()
            # With label-smoothing eps=0.1 the engine emits +/-0.45.
            assert x.grad[0] == pytest.approx(0.45, abs=1e-3)
            assert x.grad[1] == pytest.approx(-0.45, abs=1e-3)
            loss.dispose()
        finally:
            x.dispose()
