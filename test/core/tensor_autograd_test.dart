// Autograd verification — Dart twin of python/tests/test_tensor_autograd.py
// and test/scratch/test_tensor.dart, organised as proper package:test groups.

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void main() {
  group('reduction gradients', () {
    test('sum gradient is ones', () {
      final x = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      try {
        final loss = x.sum();
        loss.backward();
        for (final g in x.grad) {
          expect(g, closeTo(1.0, 1e-4));
        }
        loss.dispose();
      } finally {
        x.dispose();
      }
    });

    test('mean gradient is 1/N', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
      try {
        final loss = x.mean();
        loss.backward();
        for (final g in x.grad) {
          expect(g, closeTo(0.25, 1e-4));
        }
        loss.dispose();
      } finally {
        x.dispose();
      }
    });
  });

  group('unary activations', () {
    test('sigmoid gradient at zero is 0.25', () {
      final x = Tensor.fromList([1, 1], [0.0]);
      try {
        final s = x.sigmoid();
        final loss = s.sum();
        loss.backward();
        expect(x.grad[0], closeTo(0.25, 1e-4));
        s.dispose();
        loss.dispose();
      } finally {
        x.dispose();
      }
    });

    test('pow(2) at x=3 -> dy/dx = 6', () {
      final x = Tensor.fromList([1, 1], [3.0]);
      try {
        final loss = x.pow(2.0).sum();
        loss.backward();
        expect(x.grad[0], closeTo(6.0, 1e-3));
      } finally {
        x.dispose();
      }
    });

    test('log at x=2 -> dy/dx = 0.5', () {
      final x = Tensor.fromList([1, 1], [2.0]);
      try {
        final loss = x.log().sum();
        loss.backward();
        expect(x.grad[0], closeTo(0.5, 1e-4));
      } finally {
        x.dispose();
      }
    });

    test('relu: zero gradient on negative inputs', () {
      final x = Tensor.fromList([1, 2], [-1.0, 2.0]);
      try {
        final loss = x.relu().sum();
        loss.backward();
        expect(x.grad[0], closeTo(0.0, 1e-4));
        expect(x.grad[1], closeTo(1.0, 1e-4));
      } finally {
        x.dispose();
      }
    });
  });

  group('matmul gradients', () {
    test('simple vector matmul', () {
      final a = Tensor.fromList([1, 2], [2.0, 3.0]);
      final b = Tensor.fromList([2, 1], [4.0, 5.0]);
      try {
        final loss = a.matmul(b).sum();
        loss.backward();
        expect(a.grad[0], closeTo(4.0, 1e-4));
        expect(a.grad[1], closeTo(5.0, 1e-4));
        expect(b.grad[0], closeTo(2.0, 1e-4));
        expect(b.grad[1], closeTo(3.0, 1e-4));
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('weight update gradients', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final w = Tensor.fill([3, 2], 0.5);
      try {
        final out = x.matmul(w);
        final loss = out.sum();
        loss.backward();
        // dL/dW[i,j] = sum_r X[r,i]
        expect(w.grad[0], closeTo(5.0, 1e-4)); // X[0,0]+X[1,0]=1+4
        expect(w.grad[5], closeTo(9.0, 1e-4)); // X[0,2]+X[1,2]=3+6
        out.dispose();
        loss.dispose();
      } finally {
        x.dispose();
        w.dispose();
      }
    });
  });

  group('softmax', () {
    test('softmax forward values', () {
      final x = Tensor.fromList([1, 3], [1.0, 2.0, 3.0]);
      try {
        final y = x.softmax();
        final d = y.fetchData();
        expect(d[0], closeTo(0.0900, 1e-3));
        expect(d[1], closeTo(0.2447, 1e-3));
        expect(d[2], closeTo(0.6652, 1e-3));
        y.dispose();
      } finally {
        x.dispose();
      }
    });

    test('grad of sum(softmax(x)) is ~ zero', () {
      final x = Tensor.fromList([1, 3], [1.0, 2.0, 3.0]);
      try {
        final loss = x.softmax().sum();
        loss.backward();
        for (final g in x.grad) {
          expect(g.abs(), lessThan(1e-3));
        }
        loss.dispose();
      } finally {
        x.dispose();
      }
    });
  });

  group('cross-entropy', () {
    test('label-smoothed binary case', () {
      final x = Tensor.fromList([1, 2], [0.0, 0.0]);
      try {
        final loss = x.crossEntropy([1]);
        loss.backward();
        expect(x.grad[0], closeTo(0.45, 1e-3));
        expect(x.grad[1], closeTo(-0.45, 1e-3));
        loss.dispose();
      } finally {
        x.dispose();
      }
    });
  });

  group('layer norm op', () {
    test('grad of sum(normalize(x)) ~ zero', () {
      final x = Tensor.fromList([1, 4], [10.0, 2.0, 5.0, 3.0]);
      try {
        final y = x.normalize(eps: 1e-6);
        final loss = y.sum();
        loss.backward();
        for (final g in x.grad) {
          expect(g.abs(), lessThan(1e-3));
        }
        loss.dispose();
      } finally {
        x.dispose();
      }
    });
  });
}
