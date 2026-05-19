// Tensor construction, host<->device round-trip, and dispose semantics.
//
// Run from the repo root with:
//     dart test test/core/tensor_basics_test.dart

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor factories', () {
    test('fromList round-trip', () {
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      try {
        expect(x.shape, equals([2, 3]));
        expect(x.length, equals(6));
        final data = x.data;
        for (int i = 0; i < 6; i++) {
          expect(data[i], closeTo(i + 1.0, 1e-4));
        }
      } finally {
        x.dispose();
      }
    });

    test('fill', () {
      final x = Tensor.fill([3, 4], 0.5);
      try {
        for (final v in x.data) {
          expect(v, closeTo(0.5, 1e-4));
        }
      } finally {
        x.dispose();
      }
    });

    test('zeros', () {
      final x = Tensor.zeros([2, 2]);
      try {
        for (final v in x.data) {
          expect(v, closeTo(0.0, 1e-6));
        }
      } finally {
        x.dispose();
      }
    });

    test('fetchData matches data', () {
      final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
      try {
        final a = x.fetchData();
        final b = x.data;
        for (int i = 0; i < 4; i++) {
          expect(a[i], closeTo(b[i], 1e-6));
        }
      } finally {
        x.dispose();
      }
    });

    test('fetchRow', () {
      final x = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      try {
        expect(x.fetchRow(0), equals([1.0, 2.0]));
        expect(x.fetchRow(2), equals([5.0, 6.0]));
      } finally {
        x.dispose();
      }
    });
  });

  group('Tensor.dispose() semantics', () {
    test('marks tensor as disposed', () {
      final x = Tensor.zeros([2, 2]);
      x.dispose();
      // Disposing a second time must be a no-op (idempotent).
      x.dispose();
      x.dispose();
    });

    test('new tensors still allocate after a dispose', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      a.dispose();
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8]);
      try {
        expect(b.data, equals([5.0, 6.0, 7.0, 8.0]));
      } finally {
        b.dispose();
      }
    });

    test('reshape returns a view; disposing the view does not free parent', () {
      final parent = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      try {
        final view = parent.reshape([3, 2]);
        expect(view.isView, isTrue);
        expect(view.shape, equals([3, 2]));
        view.dispose(); // no-op
        // Parent is still alive and readable.
        expect(parent.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
      } finally {
        parent.dispose();
      }
    });
  });

  group('Tensor.getRow / slice', () {
    test('getRow returns single-row tensor', () {
      final x = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      try {
        final r = x.getRow(1);
        try {
          expect(r.shape, equals([1, 2]));
          expect(r.data, equals([3.0, 4.0]));
        } finally {
          r.dispose();
        }
      } finally {
        x.dispose();
      }
    });

    test('slice returns sub-block', () {
      final x = Tensor.fromList([4, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
      try {
        final s = x.slice(1, 2);
        try {
          expect(s.shape, equals([2, 2]));
          expect(s.data, equals([3.0, 4.0, 5.0, 6.0]));
        } finally {
          s.dispose();
        }
      } finally {
        x.dispose();
      }
    });
  });
}
