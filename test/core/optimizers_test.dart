// Optimizer tests: Adam and SDG reduce loss on tiny regression + dispose.

import 'package:dart_cuda/core/layers/mlp.dart';
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/optimizers/stochastic_grad_desc.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

typedef _Loss = ({double initial, double finalLoss});

_Loss _train(dynamic Function(List<Tensor>) optFactory) {
  final mlp = MLP(2, [4, 1]);
  final x = Tensor.fromList([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
  final y = Tensor.fromList([4, 1], [0, 1, 1, 0]);
  final opt = optFactory(mlp.parameters());

  double? initial;
  double? finalLoss;
  for (int step = 0; step < 60; step++) {
    final tracker = <Tensor>[];
    final pred = mlp.forward(x, tracker);
    final diff = pred - y;
    final sq = diff.pow(2.0);
    final loss = sq.sum();
    tracker.addAll([pred, diff, sq, loss]);

    opt.zeroGrad();
    loss.backward();
    opt.step();

    final v = loss.data[0];
    if (step == 0) initial = v;
    finalLoss = v;
    for (final t in tracker) {
      t.dispose();
    }
  }

  x.dispose();
  y.dispose();
  for (final p in mlp.parameters()) {
    p.dispose();
  }
  opt.dispose();
  return (initial: initial!, finalLoss: finalLoss!);
}

void main() {
  group('Adam', () {
    test('decreases loss on XOR-style fit', () {
      final r = _train((ps) => Adam(ps, lr: 0.05));
      expect(r.finalLoss, lessThan(r.initial));
    });
  });

  group('SDG', () {
    test('decreases loss', () {
      final r = _train((ps) => SDG(ps, lr: 0.05));
      expect(r.finalLoss, lessThan(r.initial));
    });
  });

  group('Optimizer dispose', () {
    test('Adam.dispose() empties m/v buffers', () {
      final params = <Tensor>[
        Tensor.zeros([2, 2]),
        Tensor.zeros([1, 4]),
      ];
      final opt = Adam(params, lr: 1e-3);
      expect(opt.m.length, equals(2));
      expect(opt.v.length, equals(2));
      opt.dispose();
      expect(opt.m, isEmpty);
      expect(opt.v, isEmpty);
      for (final p in params) {
        p.dispose();
      }
    });
  });
}
