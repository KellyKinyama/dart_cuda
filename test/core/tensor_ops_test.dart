// Element-wise tensor operations.

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void main() {
  group('elementwise binary', () {
    test('add', () {
      final a = Tensor.fromList([1, 3], [1, 2, 3]);
      final b = Tensor.fromList([1, 3], [10, 20, 30]);
      try {
        final c = a + b;
        try {
          expect(c.data, equals([11.0, 22.0, 33.0]));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('sub', () {
      final a = Tensor.fromList([1, 3], [10, 20, 30]);
      final b = Tensor.fromList([1, 3], [1, 2, 3]);
      try {
        final c = a - b;
        try {
          expect(c.data, equals([9.0, 18.0, 27.0]));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('mul (hadamard)', () {
      final a = Tensor.fromList([1, 3], [1, 2, 3]);
      final b = Tensor.fromList([1, 3], [4, 5, 6]);
      try {
        final c = a * b;
        try {
          expect(c.data, equals([4.0, 10.0, 18.0]));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('div', () {
      final a = Tensor.fromList([1, 3], [4, 9, 16]);
      final b = Tensor.fromList([1, 3], [2, 3, 4]);
      try {
        final c = a / b;
        try {
          expect(c.data[0], closeTo(2.0, 1e-4));
          expect(c.data[1], closeTo(3.0, 1e-4));
          expect(c.data[2], closeTo(4.0, 1e-4));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });
  });

  group('scalar broadcast', () {
    test('mul by scalar', () {
      final a = Tensor.fromList([1, 3], [1, 2, 3]);
      try {
        final c = a * 2.5;
        try {
          expect(c.data[0], closeTo(2.5, 1e-4));
          expect(c.data[1], closeTo(5.0, 1e-4));
          expect(c.data[2], closeTo(7.5, 1e-4));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
      }
    });

    test('div by scalar', () {
      final a = Tensor.fromList([1, 3], [2, 4, 8]);
      try {
        final c = a / 2.0;
        try {
          expect(c.data[0], closeTo(1.0, 1e-4));
          expect(c.data[1], closeTo(2.0, 1e-4));
          expect(c.data[2], closeTo(4.0, 1e-4));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
      }
    });

    test('div by zero throws', () {
      final a = Tensor.fromList([1, 1], [1.0]);
      try {
        expect(() => a / 0, throwsA(isA<UnsupportedError>()));
      } finally {
        a.dispose();
      }
    });
  });

  group('concat & embeddings', () {
    test('concat along last axis', () {
      final a = Tensor.fromList([2, 2], [1, 2, 3, 4]);
      final b = Tensor.fromList([2, 2], [5, 6, 7, 8]);
      try {
        final c = Tensor.concat([a, b]);
        try {
          expect(c.shape, equals([2, 4]));
          expect(c.data, equals([1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('concatAxis0 stacks rows', () {
      final a = Tensor.fromList([1, 3], [1, 2, 3]);
      final b = Tensor.fromList([2, 3], [4, 5, 6, 7, 8, 9]);
      try {
        final c = Tensor.concatAxis0([a, b]);
        try {
          expect(c.shape, equals([3, 3]));
          expect(c.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
        } finally {
          c.dispose();
        }
      } finally {
        a.dispose();
        b.dispose();
      }
    });

    test('embeddings = wte[idx] + wpe[pos]', () {
      final wte = Tensor.fromList([3, 2], [1, 2, 3, 4, 5, 6]);
      final wpe = Tensor.fromList([4, 2], [10, 20, 30, 40, 50, 60, 70, 80]);
      try {
        final out = Tensor.embeddings([0, 2, 1], wte, wpe);
        try {
          expect(out.shape, equals([3, 2]));
          expect(out.data, equals([11.0, 22.0, 35.0, 46.0, 53.0, 64.0]));
        } finally {
          out.dispose();
        }
      } finally {
        wte.dispose();
        wpe.dispose();
      }
    });
  });
}
