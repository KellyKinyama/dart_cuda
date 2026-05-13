// Tests for optimizer modules (lib/core/optimizers/).
//
// Requires the CUDA backend; run from repo root:
//   dart test test/core/optimizers/

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/optimizers/stochastic_grad_desc.dart';
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
  group('Adam', () {
    test('allocates m and v buffers matching every parameter', () {
      final p1 = Tensor.fill([2, 3], 0.0);
      final p2 = Tensor.fill([1, 4], 0.0);
      final opt = Adam([p1, p2], lr: 0.01);
      addTearDown(() {
        opt.dispose();
        p1.dispose();
        p2.dispose();
      });

      expect(opt.m, hasLength(2));
      expect(opt.v, hasLength(2));
      expect(opt.m[0].shape, equals([2, 3]));
      expect(opt.m[1].shape, equals([1, 4]));
      expect(opt.v[0].shape, equals([2, 3]));
      expect(opt.v[1].shape, equals([1, 4]));
    });

    test('m and v start at zero', () {
      final p = Tensor.fill([1, 4], 0.0);
      final opt = Adam([p], lr: 0.01);
      addTearDown(() {
        opt.dispose();
        p.dispose();
      });

      expect(opt.m[0].data, everyElement(closeTo(0.0, _tol)));
      expect(opt.v[0].data, everyElement(closeTo(0.0, _tol)));
    });

    test('zeroGrad clears parameter gradients', () {
      final p = Tensor.fromList([1, 3], [1.0, 2.0, 3.0]);
      final loss = p.sum();
      loss.backward();
      addTearDown(() {
        loss.dispose();
        p.dispose();
      });

      // Sanity: grad is non-zero after backward.
      expect(p.grad, closeToList([1.0, 1.0, 1.0]));

      final opt = Adam([p], lr: 0.01);
      addTearDown(opt.dispose);
      opt.zeroGrad();
      expect(p.grad, everyElement(closeTo(0.0, 1e-9)));
    });

    test(
      'step increments t and updates parameters in the gradient direction',
      () {
        final p = Tensor.fromList([1, 2], [10.0, -5.0]);
        final loss = p.sum(); // grad = [1, 1]
        loss.backward();
        addTearDown(() {
          loss.dispose();
          p.dispose();
        });

        final before = List<double>.from(p.data);
        final opt = Adam([p], lr: 0.1);
        addTearDown(opt.dispose);

        expect(opt.t, equals(0));
        opt.step();
        expect(opt.t, equals(1));

        final after = p.data;
        // With grad=+1 and lr=0.1, Adam's first step is approximately
        // p ← p - lr (because m_hat == v_hat-normalized → 1).
        expect(after[0], lessThan(before[0]));
        expect(after[1], lessThan(before[1]));
        // Magnitude of update should be close to lr on the first step.
        expect((before[0] - after[0]).abs(), closeTo(0.1, 0.05));
      },
    );

    test(
      'repeated steps drive a parameter toward zero on a quadratic loss',
      () {
        // Loss = sum(p^2), grad = 2p. Optimum at p = 0.
        final p = Tensor.fromList([1, 3], [1.0, -2.0, 0.5]);
        final opt = Adam([p], lr: 0.1);
        addTearDown(() {
          opt.dispose();
          p.dispose();
        });

        double normSq() => p.data.fold(0.0, (s, v) => s + v * v);
        final initial = normSq();

        for (var i = 0; i < 50; i++) {
          opt.zeroGrad();
          final sq = p.pow(2.0);
          final loss = sq.sum();
          loss.backward();
          opt.step();
          sq.dispose();
          loss.dispose();
        }

        final finalNorm = normSq();
        expect(finalNorm, lessThan(initial));
        expect(finalNorm, lessThan(0.5));
      },
    );

    test('dispose clears m and v buffers', () {
      final p = Tensor.fill([2, 2], 0.0);
      final opt = Adam([p], lr: 0.01);
      addTearDown(p.dispose);

      expect(opt.m, hasLength(1));
      opt.dispose();
      expect(opt.m, isEmpty);
      expect(opt.v, isEmpty);
    });
  });

  group('SDG (gradient-descent variant)', () {
    test('allocates m and v buffers matching every parameter', () {
      final p1 = Tensor.fill([2, 3], 0.0);
      final p2 = Tensor.fill([1, 4], 0.0);
      final opt = SDG([p1, p2], lr: 0.01);
      addTearDown(() {
        opt.dispose();
        p1.dispose();
        p2.dispose();
      });

      expect(opt.m, hasLength(2));
      expect(opt.v, hasLength(2));
      expect(opt.m[0].shape, equals([2, 3]));
      expect(opt.v[1].shape, equals([1, 4]));
    });

    test('zeroGrad clears parameter gradients', () {
      final p = Tensor.fromList([1, 2], [1.0, 1.0]);
      final loss = p.sum();
      loss.backward();
      addTearDown(() {
        loss.dispose();
        p.dispose();
      });

      expect(p.grad, closeToList([1.0, 1.0]));
      final opt = SDG([p], lr: 0.05);
      addTearDown(opt.dispose);

      opt.zeroGrad();
      expect(p.grad, everyElement(closeTo(0.0, 1e-9)));
    });

    test('step changes parameters and increments t', () {
      final p = Tensor.fromList([1, 2], [4.0, -2.0]);
      final loss = p.sum(); // grad = [1, 1]
      loss.backward();
      addTearDown(() {
        loss.dispose();
        p.dispose();
      });

      final before = List<double>.from(p.data);
      final opt = SDG([p], lr: 0.1);
      addTearDown(opt.dispose);

      opt.step();
      expect(opt.t, equals(1));
      // The kernel must move the parameters away from their starting values.
      final after = p.data;
      expect(after[0], isNot(closeTo(before[0], 1e-9)));
      expect(after[1], isNot(closeTo(before[1], 1e-9)));
    });

    test('repeated steps reduce ‖p‖² on a quadratic loss', () {
      // Loss = sum(p^2), grad = 2p. Optimum at p = 0.
      final p = Tensor.fromList([1, 3], [1.0, -2.0, 0.5]);
      final opt = SDG([p], lr: 0.05);
      addTearDown(() {
        opt.dispose();
        p.dispose();
      });

      double normSq() => p.data.fold(0.0, (s, v) => s + v * v);
      final initial = normSq();

      for (var i = 0; i < 50; i++) {
        opt.zeroGrad();
        final sq = p.pow(2.0);
        final loss = sq.sum();
        loss.backward();
        opt.step();
        sq.dispose();
        loss.dispose();
      }

      expect(normSq(), lessThan(initial));
    });

    test('dispose clears m and v buffers', () {
      final p = Tensor.fill([1, 2], 0.0);
      final opt = SDG([p], lr: 0.01);
      addTearDown(p.dispose);

      opt.dispose();
      expect(opt.m, isEmpty);
      expect(opt.v, isEmpty);
    });
  });
}
