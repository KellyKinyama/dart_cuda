// Layer / FeedForward / LayerNorm / MLP tests.

import 'package:dart_cuda/core/layers/feed_forward.dart';
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/mlp.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Layer', () {
    test('forward shape', () {
      final layer = Layer(4, 6, useGelu: false);
      final tracker = <Tensor>[];
      final x = Tensor.fill([3, 4], 1.0);
      tracker.add(x);
      final out = layer.forward(x, tracker);
      expect(out.shape, equals([3, 6]));
      for (final p in layer.parameters()) {
        p.dispose();
      }
      for (final t in tracker) {
        t.dispose();
      }
    });

    test('parameters layout', () {
      final layer = Layer(3, 5, useGelu: true);
      final params = layer.parameters();
      try {
        expect(params.length, equals(2));
        expect(params[0].shape, equals([3, 5]));
        expect(params[1].shape, equals([1, 5]));
      } finally {
        for (final p in params) {
          p.dispose();
        }
      }
    });

    test('backward populates gradients', () {
      final layer = Layer(4, 2, useGelu: false);
      final tracker = <Tensor>[];
      final x = Tensor.fromList([1, 4], [0.1, 0.2, 0.3, 0.4]);
      tracker.add(x);
      final out = layer.forward(x, tracker);
      final loss = out.sum();
      tracker.add(loss);
      layer.zeroGrad();
      loss.backward();
      for (final p in layer.parameters()) {
        expect(p.grad.length, equals(p.length));
      }
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in layer.parameters()) {
        p.dispose();
      }
    });
  });

  group('LayerNorm', () {
    test('forward shape', () {
      final ln = LayerNorm(8);
      final tracker = <Tensor>[];
      final x = Tensor.fromList([
        2,
        8,
      ], List.generate(16, (i) => (i + 1).toDouble() / 16));
      tracker.add(x);
      final out = ln.forward(x, tracker);
      expect(out.shape, equals([2, 8]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in ln.parameters()) {
        p.dispose();
      }
    });
  });

  group('FeedForward', () {
    test('forward preserves embedding dim', () {
      final ff = FeedForward(8);
      final tracker = <Tensor>[];
      final x = Tensor.fromList([
        3,
        8,
      ], List.generate(24, (i) => (i + 1).toDouble() / 24));
      tracker.add(x);
      final out = ff.forward(x, tracker);
      expect(out.shape, equals([3, 8]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in ff.parameters()) {
        p.dispose();
      }
    });
  });

  group('MLP', () {
    test('forward shape', () {
      final mlp = MLP(3, [4, 2]);
      final tracker = <Tensor>[];
      final x = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
      tracker.add(x);
      final out = mlp.forward(x, tracker);
      expect(out.shape, equals([2, 2]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in mlp.parameters()) {
        p.dispose();
      }
    });

    test('overfit tiny regression: loss decreases', () {
      final mlp = MLP(2, [4, 1]);
      final x = Tensor.fromList([2, 2], [0.1, 0.2, 0.3, 0.4]);
      final y = Tensor.fromList([2, 1], [1.0, 0.0]);

      double? initial;
      double? finalLoss;
      for (int step = 0; step < 50; step++) {
        final tracker = <Tensor>[];
        final pred = mlp.forward(x, tracker);
        final diff = pred - y;
        final sq = diff.pow(2.0);
        final loss = sq.sum();
        tracker.addAll([pred, diff, sq, loss]);

        mlp.zeroGrad();
        loss.backward();
        mlp.step(0.1);

        final v = loss.data[0];
        if (step == 0) initial = v;
        finalLoss = v;
        for (final t in tracker) {
          t.dispose();
        }
      }
      expect(initial, isNotNull);
      expect(finalLoss, isNotNull);
      expect(finalLoss!, lessThan(initial!));

      x.dispose();
      y.dispose();
      for (final p in mlp.parameters()) {
        p.dispose();
      }
    });
  });
}
