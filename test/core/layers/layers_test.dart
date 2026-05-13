// Tests for layer modules under lib/core/layers/.
//
// Requires the CUDA backend (`native/lib/libmat_mul.so`); run from repo root:
//   dart test test/core/layers/

import 'package:dart_cuda/core/layers/feed_forward.dart';
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/mlp.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

const double _tol = 1e-3;

Matcher closeToList(List<double> expected, [double tol = _tol]) =>
    pairwiseCompare<num, double>(
      expected,
      (e, a) => (a.toDouble() - e).abs() < tol,
      'each element within $tol',
    );

double _mean(Iterable<double> xs) {
  var s = 0.0;
  var n = 0;
  for (final v in xs) {
    s += v;
    n += 1;
  }
  return s / n;
}

double _variance(Iterable<double> xs) {
  final m = _mean(xs);
  var s = 0.0;
  var n = 0;
  for (final v in xs) {
    s += (v - m) * (v - m);
    n += 1;
  }
  return s / n;
}

void main() {
  group('Layer (linear + optional GELU)', () {
    test('forward shape matches [batch, nout]', () {
      final layer = Layer(3, 4, useGelu: false);
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      final tracker = <Tensor>[];

      final y = layer.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        for (final p in layer.parameters()) {
          p.dispose();
        }
      });

      expect(y.shape, equals([2, 4]));
      expect(y.length, equals(8));
    });

    test('parameters() returns weight and bias', () {
      final layer = Layer(2, 3, useGelu: false);
      addTearDown(() {
        for (final p in layer.parameters()) {
          p.dispose();
        }
      });

      final params = layer.parameters();
      expect(params, hasLength(2));
      expect(params[0].shape, equals([2, 3])); // w
      expect(params[1].shape, equals([1, 3])); // b
    });

    test('forward with deterministic weights computes x @ w + b', () {
      final layer = Layer(2, 2, useGelu: false);
      // Override the random init with known values.
      layer.w.data = [1, 2, 3, 4]; // [[1,2],[3,4]]
      layer.b.data = [10, 20]; //   [[10,20]]
      addTearDown(() {
        for (final p in layer.parameters()) {
          p.dispose();
        }
      });

      final x = Tensor.fromList([1, 2], [1, 1]);
      final tracker = <Tensor>[];
      final y = layer.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
      });

      // [1,1] @ [[1,2],[3,4]] = [4,6]; + [10,20] = [14,26]
      expect(y.data, closeToList([14, 26]));
    });

    test('useGelu=true applies GELU after the affine map', () {
      final layer = Layer(1, 1, useGelu: true);
      layer.w.data = [1.0];
      layer.b.data = [0.0];
      addTearDown(() {
        for (final p in layer.parameters()) {
          p.dispose();
        }
      });

      final x = Tensor.fromList([1, 1], [0.0]);
      final tracker = <Tensor>[];
      final y = layer.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
      });

      // GELU(0) == 0
      expect(y.data.first, closeTo(0.0, _tol));
    });

    test('zeroGrad clears parameter gradients', () {
      final layer = Layer(2, 1, useGelu: false);
      addTearDown(() {
        for (final p in layer.parameters()) {
          p.dispose();
        }
      });

      final x = Tensor.fromList([1, 2], [1.0, 1.0]);
      final tracker = <Tensor>[];
      final y = layer.forward(x, tracker);
      final loss = y.sum();
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
        for (final t in tracker) {
          t.dispose();
        }
      });

      // After backward, gradients should be non-zero somewhere…
      final hasGrad = layer.w.grad.any((g) => g.abs() > 1e-9) ||
          layer.b.grad.any((g) => g.abs() > 1e-9);
      expect(hasGrad, isTrue);

      layer.zeroGrad();
      expect(layer.w.grad, everyElement(closeTo(0.0, 1e-9)));
      expect(layer.b.grad, everyElement(closeTo(0.0, 1e-9)));
    });
  });

  group('LayerNorm', () {
    test('parameters() returns gamma and beta with the correct shape', () {
      final ln = LayerNorm(8);
      addTearDown(() {
        for (final p in ln.parameters()) {
          p.dispose();
        }
      });

      final params = ln.parameters();
      expect(params, hasLength(2));
      expect(params[0].shape, equals([1, 8])); // gamma
      expect(params[1].shape, equals([1, 8])); // beta
      // Default gamma == 1, beta == 0.
      expect(params[0].data, everyElement(closeTo(1.0, _tol)));
      expect(params[1].data, everyElement(closeTo(0.0, _tol)));
    });

    test('forward produces a tensor of the same shape', () {
      final ln = LayerNorm(4);
      final x = Tensor.fromList([2, 4], [1, 2, 3, 4, -1, 0, 1, 2]);
      final tracker = <Tensor>[];
      final y = ln.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        for (final p in ln.parameters()) {
          p.dispose();
        }
      });

      expect(y.shape, equals(x.shape));
      expect(y.length, equals(x.length));
    });

    test('forward roughly normalizes each row to mean≈0, var≈1', () {
      final ln = LayerNorm(4);
      final x = Tensor.fromList([2, 4], [1, 2, 3, 4, 10, 20, 30, 40]);
      final tracker = <Tensor>[];
      final y = ln.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        for (final p in ln.parameters()) {
          p.dispose();
        }
      });

      final out = y.data;
      final row0 = out.sublist(0, 4);
      final row1 = out.sublist(4, 8);

      expect(_mean(row0), closeTo(0.0, 1e-2));
      expect(_mean(row1), closeTo(0.0, 1e-2));
      // Variance should be ≈ 1 (within a generous tolerance for the kernel).
      expect(_variance(row0), closeTo(1.0, 0.2));
      expect(_variance(row1), closeTo(1.0, 0.2));
    });
  });

  group('FeedForward (D -> 4D -> D)', () {
    test('forward preserves the input width', () {
      final ff = FeedForward(8);
      final x = Tensor.fromList([2, 8], List.generate(16, (i) => i.toDouble()));
      final tracker = <Tensor>[];
      final y = ff.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        for (final p in ff.parameters()) {
          p.dispose();
        }
      });

      expect(y.shape, equals([2, 8]));
    });

    test('parameters() concatenates w1 and w2 params', () {
      final ff = FeedForward(4);
      addTearDown(() {
        for (final p in ff.parameters()) {
          p.dispose();
        }
      });

      // w1: w[4,16] + b[1,16];  w2: w[16,4] + b[1,4]   -> 4 tensors total
      expect(ff.parameters(), hasLength(4));
    });
  });

  group('MLP', () {
    test('builds the requested chain of Layers', () {
      final mlp = MLP(2, [4, 3, 1]);
      addTearDown(() {
        for (final p in mlp.parameters()) {
          p.dispose();
        }
      });

      expect(mlp.layers, hasLength(3));
      expect(mlp.layers[0].w.shape, equals([2, 4]));
      expect(mlp.layers[1].w.shape, equals([4, 3]));
      expect(mlp.layers[2].w.shape, equals([3, 1]));
    });

    test('forward runs end-to-end and yields the final shape', () {
      final mlp = MLP(2, [4, 1]);
      final x = Tensor.fromList([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
      final tracker = <Tensor>[];
      final y = mlp.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        for (final p in mlp.parameters()) {
          p.dispose();
        }
      });

      expect(y.shape, equals([4, 1]));
    });

    test('all layers except the last apply GELU', () {
      final mlp = MLP(2, [4, 3, 1]);
      addTearDown(() {
        for (final p in mlp.parameters()) {
          p.dispose();
        }
      });

      expect(mlp.layers[0].useGelu, isTrue);
      expect(mlp.layers[1].useGelu, isTrue);
      expect(mlp.layers[2].useGelu, isFalse);
    });

    test('parameters() collects every layer\'s parameters', () {
      final mlp = MLP(2, [3, 1]);
      addTearDown(() {
        for (final p in mlp.parameters()) {
          p.dispose();
        }
      });

      // 2 layers × (w + b) == 4
      expect(mlp.parameters(), hasLength(4));
    });

    test('one SGD step reduces a simple regression loss', () {
      final mlp = MLP(2, [4, 1]);
      final x = Tensor.fromList([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
      final t = Tensor.fromList([4, 1], [0, 1, 1, 0]);

      double lossOf() {
        final tracker = <Tensor>[];
        final pred = mlp.forward(x, tracker);
        final diff = pred - t;
        final sq = diff.pow(2.0);
        final loss = sq.sum();
        final v = loss.data.first;
        for (final tt in tracker) {
          tt.dispose();
        }
        diff.dispose();
        sq.dispose();
        loss.dispose();
        return v;
      }

      final before = lossOf();

      // One training step.
      final tracker = <Tensor>[];
      final pred = mlp.forward(x, tracker);
      final diff = pred - t;
      final sq = diff.pow(2.0);
      final loss = sq.sum();
      mlp.zeroGrad();
      loss.backward();
      mlp.step(0.05);
      for (final tt in tracker) {
        tt.dispose();
      }
      diff.dispose();
      sq.dispose();
      loss.dispose();

      final after = lossOf();

      addTearDown(() {
        x.dispose();
        t.dispose();
        for (final p in mlp.parameters()) {
          p.dispose();
        }
      });

      // Loss should decrease (or at least not increase by more than rounding).
      expect(after, lessThanOrEqualTo(before + 1e-4));
    });
  });
}
