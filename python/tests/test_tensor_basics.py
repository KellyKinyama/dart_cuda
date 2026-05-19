"""Tensor construction, host<->device round-trip, and dispose semantics."""

from __future__ import annotations

import pytest

from dart_cuda.core.tensor.gpu_tensor import Tensor


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
class TestFactories:
    def test_from_list_roundtrip(self):
        x = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        try:
            data = x.data
            assert data == pytest.approx([1, 2, 3, 4, 5, 6], rel=1e-5)
            assert x.shape == [2, 3]
            assert x.length == 6
        finally:
            x.dispose()

    def test_fill(self):
        x = Tensor.fill([3, 4], 0.5)
        try:
            assert x.data == pytest.approx([0.5] * 12)
        finally:
            x.dispose()

    def test_zeros(self):
        x = Tensor.zeros([2, 2])
        try:
            assert x.data == pytest.approx([0.0] * 4)
        finally:
            x.dispose()

    def test_random_in_bounds(self):
        x = Tensor.random([4, 8])
        try:
            data = x.data
            assert len(data) == 32
            # Glorot init: values should be small.
            assert max(abs(v) for v in data) < 2.0
        finally:
            x.dispose()

    def test_data_setter_validates_length(self):
        x = Tensor.zeros([2, 2])
        try:
            with pytest.raises(ValueError):
                x.data = [1.0, 2.0]
        finally:
            x.dispose()

    def test_data_setter_writes(self):
        x = Tensor.zeros([2, 2])
        try:
            x.data = [9.0, 8.0, 7.0, 6.0]
            assert x.data == pytest.approx([9, 8, 7, 6])
        finally:
            x.dispose()


# ---------------------------------------------------------------------------
# fetch_data, fetch_row, grad
# ---------------------------------------------------------------------------
class TestFetch:
    def test_fetch_data_matches_data(self):
        x = Tensor.from_list([2, 2], [1.0, 2.0, 3.0, 4.0])
        try:
            assert x.fetch_data() == pytest.approx(x.data)
        finally:
            x.dispose()

    def test_fetch_row(self):
        x = Tensor.from_list([3, 2], [1, 2, 3, 4, 5, 6])
        try:
            assert x.fetch_row(0) == pytest.approx([1, 2])
            assert x.fetch_row(2) == pytest.approx([5, 6])
        finally:
            x.dispose()

    def test_zero_grad_resets(self):
        x = Tensor.from_list([1, 1], [3.0])
        try:
            loss = x.pow(2.0).sum()
            loss.backward()
            assert x.grad[0] == pytest.approx(6.0, abs=1e-3)
            x.zero_grad()
            assert x.grad[0] == pytest.approx(0.0, abs=1e-6)
            loss.dispose()
        finally:
            x.dispose()


# ---------------------------------------------------------------------------
# DISPOSE SEMANTICS — user's specific request:
# "dispose on the model deletes the tensor pointer"
# ---------------------------------------------------------------------------
class TestDispose:
    def test_dispose_marks_disposed(self):
        x = Tensor.zeros([2, 2])
        assert x._disposed is False
        assert x.handle.value is not None and x.handle.value != 0
        x.dispose()
        assert x._disposed is True

    def test_dispose_is_idempotent(self):
        x = Tensor.zeros([2, 2])
        x.dispose()
        # Calling again must be safe (matches Dart behaviour).
        x.dispose()
        x.dispose()
        assert x._disposed is True

    def test_dispose_does_not_corrupt_freshly_allocated(self):
        # After disposing one tensor, a new tensor should still work.
        a = Tensor.from_list([2, 2], [1, 2, 3, 4])
        a.dispose()
        b = Tensor.from_list([2, 2], [5, 6, 7, 8])
        try:
            assert b.data == pytest.approx([5, 6, 7, 8])
        finally:
            b.dispose()

    def test_view_dispose_is_noop_on_handle(self):
        """A reshape produces a view sharing the parent's handle.
        Disposing the view must NOT free the underlying memory, so the
        parent is still usable."""
        parent = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        view = parent.reshape([3, 2])
        try:
            assert view.is_view is True
            view.dispose()
            # Parent still works — read fresh data.
            assert parent.data == pytest.approx([1, 2, 3, 4, 5, 6])
        finally:
            parent.dispose()

    def test_detach_returns_independent_tensor(self):
        x = Tensor.from_list([1, 3], [1.0, 2.0, 3.0])
        y = x.detach()
        try:
            # Mutate the detached copy; original unchanged.
            y.data = [9.0, 9.0, 9.0]
            assert x.data == pytest.approx([1.0, 2.0, 3.0])
            assert y.data == pytest.approx([9.0, 9.0, 9.0])
        finally:
            x.dispose()
            y.dispose()
