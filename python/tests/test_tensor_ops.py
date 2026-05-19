"""Elementwise ops, broadcasting, concat, slicing, embeddings."""

from __future__ import annotations

import pytest

from dart_cuda.core.tensor.gpu_tensor import Tensor


class TestElementwise:
    def test_add_tensors(self):
        a = Tensor.from_list([1, 3], [1, 2, 3])
        b = Tensor.from_list([1, 3], [10, 20, 30])
        try:
            c = a + b
            try:
                assert c.data == pytest.approx([11, 22, 33])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()

    def test_sub_tensors(self):
        a = Tensor.from_list([1, 3], [10, 20, 30])
        b = Tensor.from_list([1, 3], [1, 2, 3])
        try:
            c = a - b
            try:
                assert c.data == pytest.approx([9, 18, 27])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()

    def test_mul_tensors(self):
        a = Tensor.from_list([1, 3], [1, 2, 3])
        b = Tensor.from_list([1, 3], [4, 5, 6])
        try:
            c = a * b
            try:
                assert c.data == pytest.approx([4, 10, 18])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()

    def test_div_tensors(self):
        a = Tensor.from_list([1, 3], [4, 9, 16])
        b = Tensor.from_list([1, 3], [2, 3, 4])
        try:
            c = a / b
            try:
                assert c.data == pytest.approx([2, 3, 4])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()

    def test_scalar_mul(self):
        a = Tensor.from_list([1, 3], [1, 2, 3])
        try:
            c = a * 2.5
            try:
                assert c.data == pytest.approx([2.5, 5.0, 7.5])
            finally:
                c.dispose()
        finally:
            a.dispose()

    def test_scalar_div(self):
        a = Tensor.from_list([1, 3], [2, 4, 8])
        try:
            c = a / 2.0
            try:
                assert c.data == pytest.approx([1, 2, 4])
            finally:
                c.dispose()
        finally:
            a.dispose()

    def test_neg(self):
        a = Tensor.from_list([1, 3], [1, -2, 3])
        try:
            c = -a
            try:
                assert c.data == pytest.approx([-1, 2, -3])
            finally:
                c.dispose()
        finally:
            a.dispose()

    def test_scalar_div_by_zero(self):
        a = Tensor.from_list([1, 1], [1.0])
        try:
            with pytest.raises(ZeroDivisionError):
                _ = a / 0
        finally:
            a.dispose()


class TestScalarBroadcast:
    def test_scalar_tensor_broadcast_mul(self):
        a = Tensor.from_list([1, 4], [1, 2, 3, 4])
        s = Tensor.from_list([1, 1], [3.0])
        try:
            c = a * s
            try:
                assert c.data == pytest.approx([3, 6, 9, 12])
            finally:
                c.dispose()
        finally:
            a.dispose()
            s.dispose()


class TestConcat:
    def test_concat_axis1(self):
        a = Tensor.from_list([2, 2], [1, 2, 3, 4])
        b = Tensor.from_list([2, 2], [5, 6, 7, 8])
        try:
            c = Tensor.concat([a, b])
            try:
                assert c.shape == [2, 4]
                # row-major concat along last axis:
                # row 0: [1, 2, 5, 6], row 1: [3, 4, 7, 8]
                assert c.data == pytest.approx([1, 2, 5, 6, 3, 4, 7, 8])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()

    def test_concat_axis0(self):
        a = Tensor.from_list([1, 3], [1, 2, 3])
        b = Tensor.from_list([2, 3], [4, 5, 6, 7, 8, 9])
        try:
            c = Tensor.concat_axis0([a, b])
            try:
                assert c.shape == [3, 3]
                assert c.data == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9])
            finally:
                c.dispose()
        finally:
            a.dispose()
            b.dispose()


class TestSlicing:
    def test_get_row(self):
        x = Tensor.from_list([3, 2], [1, 2, 3, 4, 5, 6])
        try:
            r = x.get_row(1)
            try:
                assert r.shape == [1, 2]
                assert r.data == pytest.approx([3, 4])
            finally:
                r.dispose()
        finally:
            x.dispose()

    def test_get_row_out_of_bounds(self):
        x = Tensor.from_list([2, 2], [1, 2, 3, 4])
        try:
            with pytest.raises(IndexError):
                x.get_row(2)
        finally:
            x.dispose()

    def test_slice(self):
        x = Tensor.from_list([4, 2], [1, 2, 3, 4, 5, 6, 7, 8])
        try:
            s = x.slice(1, 2)
            try:
                assert s.shape == [2, 2]
                assert s.data == pytest.approx([3, 4, 5, 6])
            finally:
                s.dispose()
        finally:
            x.dispose()

    def test_slice_out_of_bounds(self):
        x = Tensor.from_list([2, 2], [1, 2, 3, 4])
        try:
            with pytest.raises(IndexError):
                x.slice(1, 5)
        finally:
            x.dispose()


class TestReshape:
    def test_reshape_returns_view(self):
        x = Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
        try:
            v = x.reshape([3, 2])
            assert v.is_view is True
            assert v.shape == [3, 2]
            # underlying data is the same buffer
            assert v.data == pytest.approx([1, 2, 3, 4, 5, 6])
            # disposing the view must NOT free parent memory
            v.dispose()
            assert x.data == pytest.approx([1, 2, 3, 4, 5, 6])
        finally:
            x.dispose()


class TestEmbeddings:
    def test_token_plus_position_embedding(self):
        # wte: [V=3, D=2], wpe: [T_max=4, D=2]
        wte = Tensor.from_list([3, 2], [1, 2, 3, 4, 5, 6])
        wpe = Tensor.from_list([4, 2], [10, 20, 30, 40, 50, 60, 70, 80])
        try:
            out = Tensor.embeddings([0, 2, 1], wte, wpe)
            try:
                assert out.shape == [3, 2]
                # row 0: wte[0]+wpe[0] = [11, 22]
                # row 1: wte[2]+wpe[1] = [5+30, 6+40] = [35, 46]
                # row 2: wte[1]+wpe[2] = [3+50, 4+60] = [53, 64]
                assert out.data == pytest.approx([11, 22, 35, 46, 53, 64])
            finally:
                out.dispose()
        finally:
            wte.dispose()
            wpe.dispose()
