// AFT attention smoke tests (forward shape + backward sanity).

import 'package:dart_cuda/core/attention/aft.dart';
import 'package:dart_cuda/core/attention/aft_cross_attention.dart';
import 'package:dart_cuda/core/attention/aft_multi_head_attention.dart';
import 'package:dart_cuda/core/attention/aft_multi_head_cross_attention.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Tensor _randInput(int rows, int cols) {
  final n = rows * cols;
  return Tensor.fromList([
    rows,
    cols,
  ], List.generate(n, (i) => 0.01 * ((i * 17 % 31) - 15)));
}

void main() {
  group('AFTAttention', () {
    test('forward shape (unmasked)', () {
      final att = AFTAttention(8, 4, 4, masked: false);
      final tracker = <Tensor>[];
      final x = _randInput(4, 8);
      tracker.add(x);
      final out = att.forward(x, tracker);
      expect(out.shape, equals([4, 4]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in att.parameters()) {
        p.dispose();
      }
    });

    test('forward shape (masked) + backward runs', () {
      final att = AFTAttention(4, 4, 4, masked: true);
      final tracker = <Tensor>[];
      final x = _randInput(4, 4);
      tracker.add(x);
      final out = att.forward(x, tracker);
      final loss = out.sum();
      tracker.add(loss);
      att.zeroGrad();
      loss.backward();
      for (final p in att.parameters()) {
        expect(p.grad.length, equals(p.length));
      }
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in att.parameters()) {
        p.dispose();
      }
    });
  });

  group('AFTCrossAttention', () {
    test('forward shape', () {
      final att = AFTCrossAttention(8, 6, 4, 3, 5);
      final tracker = <Tensor>[];
      final dec = _randInput(3, 8);
      final enc = _randInput(5, 6);
      tracker.addAll([dec, enc]);
      final out = att.forward(dec, enc, tracker);
      expect(out.shape, equals([3, 4]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in att.parameters()) {
        p.dispose();
      }
    });
  });

  group('MultiHeadAFT', () {
    test('forward shape', () {
      final att = MultiHeadAFT(4, 16, 4);
      final tracker = <Tensor>[];
      final x = _randInput(4, 16);
      tracker.add(x);
      final out = att.forward(x, tracker);
      expect(out.shape, equals([4, 16]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in att.parameters()) {
        p.dispose();
      }
    });
  });

  group('MultiHeadAFTCross', () {
    test('forward shape', () {
      final att = MultiHeadAFTCross(2, 8, 6, 3, 5);
      final tracker = <Tensor>[];
      final dec = _randInput(3, 8);
      final enc = _randInput(5, 6);
      tracker.addAll([dec, enc]);
      final out = att.forward(dec, enc, tracker);
      expect(out.shape, equals([3, 8]));
      for (final t in tracker) {
        t.dispose();
      }
      for (final p in att.parameters()) {
        p.dispose();
      }
    });
  });
}
