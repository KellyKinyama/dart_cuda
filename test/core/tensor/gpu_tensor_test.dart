// Tests for `Tensor` (lib/core/tensor/gpu_tensor.dart).
//
// These tests exercise the FFI path into the CUDA backend, so they require
// `native/lib/libmat_mul.so` to be present and a CUDA-capable GPU.
// Run from the repo root: `dart test test/core/tensor/`.

import 'dart:math' as math;

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

const double _tol = 1e-3;

Matcher closeToList(List<double> expected, [double tol = _tol]) =>
    pairwiseCompare<num, double>(
      expected,
      (e, a) => (a.toDouble() - e).abs() < tol,
      'each element within $tol',
    );

void main() {
  group('Tensor — FFI bridge', () {
    test('fromList round-trips through fetchData', () {
      final values = [1.0, 2.0, 3.0, 4.0];
      final x = Tensor.fromList([2, 2], values);
      addTearDown(x.dispose);

      expect(x.fetchData().toList(), closeToList(values));
    });

    test('shape and length are populated', () {
      final x = Tensor.fromList([2, 3], List.filled(6, 0.0));
      addTearDown(x.dispose);

      expect(x.shape, equals([2, 3]));
      expect(x.length, equals(6));
    });

    test('data setter pushes new values to GPU', () {
      final x = Tensor.fromList([1, 4], [0, 0, 0, 0]);
      addTearDown(x.dispose);

      x.data = [10, 20, 30, 40];
      expect(x.data, closeToList([10, 20, 30, 40]));
    });
  });

  group('Tensor — factories', () {
    test('Tensor.zeros yields all-zero data', () {
      final x = Tensor.zeros([2, 3]);
      addTearDown(x.dispose);

      expect(x.length, 6);
      expect(x.data, everyElement(closeTo(0.0, _tol)));
    });

    test('Tensor.fill yields the given constant', () {
      final x = Tensor.fill([2, 2], 7.5);
      addTearDown(x.dispose);

      expect(x.data, everyElement(closeTo(7.5, _tol)));
    });

    test('Tensor.random produces values inside the Xavier limit', () {
      final shape = [4, 8];
      final limit = math.sqrt(6.0 / (shape[0] + shape.last));
      final x = Tensor.random(shape);
      addTearDown(x.dispose);

      for (final v in x.data) {
        expect(v.abs(), lessThanOrEqualTo(limit + _tol));
      }
    });
  });

  group('Tensor — element-wise ops', () {
    test('addition of two tensors', () {
      final a = Tensor.fromList([1, 3], [1, 2, 3]);
      final b = Tensor.fromList([1, 3], [10, 20, 30]);
      final c = a + b;
      addTearDown(() {
        a.dispose();
        b.dispose();
        c.dispose();
      });

      expect(c.data, closeToList([11, 22, 33]));
    });

    test('subtraction and unary minus', () {
      final a = Tensor.fromList([1, 3], [5, 5, 5]);
      final b = Tensor.fromList([1, 3], [1, 2, 3]);
      final diff = a - b;
      final neg = -a;
      addTearDown(() {
        a.dispose();
        b.dispose();
        diff.dispose();
        neg.dispose();
      });

      expect(diff.data, closeToList([4, 3, 2]));
      expect(neg.data, closeToList([-5, -5, -5]));
    });

    test('hadamard product', () {
      final a = Tensor.fromList([1, 3], [2, 3, 4]);
      final b = Tensor.fromList([1, 3], [5, 6, 7]);
      final c = a * b;
      addTearDown(() {
        a.dispose();
        b.dispose();
        c.dispose();
      });

      expect(c.data, closeToList([10, 18, 28]));
    });

    test('elementwise division of two tensors', () {
      final a = Tensor.fromList([1, 3], [10, 20, 30]);
      final b = Tensor.fromList([1, 3], [2, 4, 5]);
      final c = a / b;
      addTearDown(() {
        a.dispose();
        b.dispose();
        c.dispose();
      });

      expect(c.data, closeToList([5, 5, 6]));
    });

    test(
      'division by scalar',
      () {
        final a = Tensor.fromList([1, 3], [10, 20, 30]);
        final c = a / 10.0;
        addTearDown(() {
          a.dispose();
          c.dispose();
        });

        expect(c.data, closeToList([1, 2, 3]));
      },
      // Known issue: scalar `/` lowers to `* Tensor.fill([1], 1/x)` which
      // the CUDA `mul_tensors` kernel does not broadcast across rank-2.
      skip: 'scalar broadcast not yet supported by mul_tensors kernel',
    );

    test('division by zero scalar throws', () {
      final a = Tensor.fromList([1, 1], [1.0]);
      addTearDown(a.dispose);

      expect(() => a / 0, throwsA(isA<UnsupportedError>()));
    });
  });

  group('Tensor — reductions and unary ops', () {
    test('sum reduces to a [1,1] scalar', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final s = x.sum();
      addTearDown(() {
        x.dispose();
        s.dispose();
      });

      expect(s.shape, equals([1, 1]));
      expect(s.data.first, closeTo(10.0, _tol));
    });

    test('mean reduces to a [1,1] scalar', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
      final m = x.mean();
      addTearDown(() {
        x.dispose();
        m.dispose();
      });

      expect(m.shape, equals([1, 1]));
      expect(m.data.first, closeTo(2.5, _tol));
    });

    test('relu zeros negatives', () {
      final x = Tensor.fromList([1, 4], [-1, 0.5, -2, 3]);
      final y = x.relu();
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      expect(y.data, closeToList([0.0, 0.5, 0.0, 3.0]));
    });

    test('abs is non-negative and matches |x|', () {
      final x = Tensor.fromList([1, 4], [-1.5, 0.0, 2.5, -4.0]);
      final y = x.abs();
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      expect(y.data, closeToList([1.5, 0.0, 2.5, 4.0]));
    });

    test('sigmoid matches numerical reference', () {
      final values = [-2.0, -0.5, 0.0, 0.5, 2.0];
      final x = Tensor.fromList([1, values.length], values);
      final y = x.sigmoid();
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      final expected =
          values.map((v) => 1.0 / (1.0 + math.exp(-v))).toList();
      expect(y.data, closeToList(expected));
    });

    test('tanh matches numerical reference', () {
      final values = [-1.0, 0.0, 1.0];
      final x = Tensor.fromList([1, values.length], values);
      final y = x.tanh();
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      expect(
        y.data,
        closeToList(values.map((v) {
          final ep = math.exp(v);
          final en = math.exp(-v);
          return (ep - en) / (ep + en);
        }).toList()),
      );
    });

    test('pow with exponent 2 squares each element', () {
      final x = Tensor.fromList([1, 3], [1, 2, 3]);
      final y = x.pow(2.0);
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      expect(y.data, closeToList([1.0, 4.0, 9.0]));
    });

    test('log on positive values', () {
      final values = [1.0, math.e, math.e * math.e];
      final x = Tensor.fromList([1, values.length], values);
      final y = x.log();
      addTearDown(() {
        x.dispose();
        y.dispose();
      });

      expect(y.data, closeToList([0.0, 1.0, 2.0]));
    });
  });

  group('Tensor — matrix multiplication', () {
    test('2x3 @ 3x2 = 2x2', () {
      // a = [[1,2,3],[4,5,6]]   b = [[7,8],[9,10],[11,12]]
      final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12]);
      final c = a.matmul(b);
      addTearDown(() {
        a.dispose();
        b.dispose();
        c.dispose();
      });

      expect(c.shape, equals([2, 2]));
      expect(c.data, closeToList([58, 64, 139, 154]));
    });

    test('identity matmul preserves the input', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final i = Tensor.fromList([2, 2], [1, 0, 0, 1]);
      final c = a.matmul(i);
      addTearDown(() {
        a.dispose();
        i.dispose();
        c.dispose();
      });

      expect(c.data, closeToList([1, 2, 3, 4]));
    });
  });

  group('Tensor — slicing and views', () {
    test('getRow returns the requested row', () {
      final x = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      final row1 = x.getRow(1);
      addTearDown(() {
        x.dispose();
        row1.dispose();
      });

      expect(row1.shape, equals([1, 2]));
      expect(row1.data, closeToList([3, 4]));
    });

    test('getRow rejects out-of-bounds indices', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      addTearDown(x.dispose);

      expect(() => x.getRow(-1), throwsRangeError);
      expect(() => x.getRow(2), throwsRangeError);
    });

    test('slice returns a contiguous block of rows', () {
      final x = Tensor.fromList([4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      final mid = x.slice(1, 2);
      addTearDown(() {
        x.dispose();
        mid.dispose();
      });

      expect(mid.shape, equals([2, 2]));
      expect(mid.data, closeToList([3, 4, 5, 6]));
    });

    test('slice rejects out-of-bounds indices', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      addTearDown(x.dispose);

      expect(() => x.slice(0, 3), throwsRangeError);
      expect(() => x.slice(-1, 1), throwsRangeError);
    });

    test('reshape produces a view that shares the handle', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final v = x.reshape([3, 2]);
      addTearDown(x.dispose); // v is a view; only x must be disposed.

      expect(v.shape, equals([3, 2]));
      expect(v.length, equals(6));
      expect(v.handle, equals(x.handle));
      expect(v.data, closeToList([1, 2, 3, 4, 5, 6]));
    });
  });

  group('Tensor — autograd', () {
    test('sum.backward() yields ones gradient', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final loss = x.sum();
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
      });

      expect(x.grad, closeToList([1.0, 1.0, 1.0, 1.0]));
    });

    test('mean.backward() yields 1/N gradient', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
      final loss = x.mean();
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
      });

      expect(x.grad, closeToList([0.25, 0.25, 0.25, 0.25]));
    });

    test('zeroGrad clears gradients', () {
      final x = Tensor.fromList([1, 2], [3.0, 4.0]);
      final loss = x.sum();
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
      });

      expect(x.grad, closeToList([1.0, 1.0]));
      x.zeroGrad();
      expect(x.grad, closeToList([0.0, 0.0]));
    });

    test('SGD step subtracts lr * grad from data', () {
      final x = Tensor.fromList([1, 2], [10.0, 20.0]);
      final loss = x.sum(); // grad = [1, 1]
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
      });

      x.step(0.5);
      expect(x.data, closeToList([9.5, 19.5]));
    });
  });

  group('Tensor — dispose', () {
    test('dispose is idempotent', () {
      final x = Tensor.fromList([1, 2], [1.0, 2.0]);
      x.dispose();
      expect(x.dispose, returnsNormally);
    });

    test('reshape view is not destroyed by dispose', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
      final v = x.reshape([2, 2]);
      v.dispose(); // no-op because v.isView == true
      // Underlying handle still valid:
      expect(x.data, closeToList([1, 2, 3, 4]));
      x.dispose();
    });
  });
}
