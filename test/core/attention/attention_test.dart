// Tests for AFT attention modules (lib/core/attention/).
//
// These tests verify shape correctness, parameter wiring and finite outputs
// for the AFT attention primitives. They require the CUDA backend.
// Run from the repo root: `dart test test/core/attention/`.

import 'package:dart_cuda/core/attention/aft.dart';
import 'package:dart_cuda/core/attention/aft_cross_attention.dart';
import 'package:dart_cuda/core/attention/aft_multi_head_attention.dart';
import 'package:dart_cuda/core/attention/aft_multi_head_cross_attention.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

void disposeAll(Iterable<Tensor> ts) {
  for (final t in ts) {
    t.dispose();
  }
}

bool _allFinite(Iterable<double> xs) =>
    xs.every((v) => !v.isNaN && !v.isInfinite);

void main() {
  group('AFTAttention', () {
    test('forward preserves [T, headSize] shape (unmasked)', () {
      const dim = 8;
      const head = 4;
      const seq = 6;
      final aft = AFTAttention(dim, head, seq, masked: false);
      final x = Tensor.random([seq, dim]);
      final tracker = <Tensor>[];

      final y = aft.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(aft.parameters());
      });

      expect(y.shape, equals([seq, head]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('forward preserves [T, headSize] shape (causal masked)', () {
      const dim = 8;
      const head = 4;
      const seq = 5;
      final aft = AFTAttention(dim, head, seq, masked: true);
      final x = Tensor.random([seq, dim]);
      final tracker = <Tensor>[];

      final y = aft.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(aft.parameters());
      });

      expect(y.shape, equals([seq, head]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() returns Q/K/V (w+b each) plus posBias', () {
      const dim = 8;
      const head = 4;
      const seq = 6;
      final aft = AFTAttention(dim, head, seq);
      addTearDown(() => disposeAll(aft.parameters()));

      final params = aft.parameters();
      // 3 Layer modules × (w + b) = 6, plus posBias = 7 total.
      expect(params, hasLength(7));
      expect(params.last.shape, equals([seq, seq])); // posBias
    });

    test('zeroGrad clears parameter gradients', () {
      const dim = 4;
      const head = 2;
      const seq = 3;
      final aft = AFTAttention(dim, head, seq);
      final x = Tensor.random([seq, dim]);
      final tracker = <Tensor>[];

      final y = aft.forward(x, tracker);
      final loss = y.sum();
      loss.backward();
      addTearDown(() {
        x.dispose();
        loss.dispose();
        disposeAll(tracker);
        disposeAll(aft.parameters());
      });

      aft.zeroGrad();
      for (final p in aft.parameters()) {
        expect(p.grad, everyElement(closeTo(0.0, 1e-9)));
      }
    });
  });

  group('MultiHeadAFT', () {
    test('forward returns [T, embedSize]', () {
      const heads = 4;
      const embed = 16;
      const seq = 5;
      final mha = MultiHeadAFT(heads, embed, seq, masked: false);
      final x = Tensor.random([seq, embed]);
      final tracker = <Tensor>[];

      final y = mha.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(mha.parameters());
      });

      expect(y.shape, equals([seq, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() collects every head plus the output projection', () {
      const heads = 4;
      const embed = 16;
      const seq = 8;
      final mha = MultiHeadAFT(heads, embed, seq);
      addTearDown(() => disposeAll(mha.parameters()));

      // Each head: 7 params (Q,K,V × (w+b) + posBias). Proj: w + b.
      expect(mha.parameters(), hasLength(heads * 7 + 2));
    });

    test('asserts when embedSize is not divisible by numHeads', () {
      expect(
        () => MultiHeadAFT(3, 16, 8),
        throwsA(isA<AssertionError>()),
      );
    });
  });

  group('AFTCrossAttention', () {
    test('forward yields [T_dec, headSize] from decoder + encoder inputs', () {
      const decEmbed = 12;
      const encEmbed = 20;
      const head = 4;
      const tDec = 5;
      const tEnc = 8;
      final cross = AFTCrossAttention(decEmbed, encEmbed, head, tDec, tEnc);

      final xDec = Tensor.random([tDec, decEmbed]);
      final xEnc = Tensor.random([tEnc, encEmbed]);
      final tracker = <Tensor>[];

      final y = cross.forward(xDec, xEnc, tracker);
      addTearDown(() {
        xDec.dispose();
        xEnc.dispose();
        disposeAll(tracker);
        disposeAll(cross.parameters());
      });

      expect(y.shape, equals([tDec, head]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() returns K/Q/V (w+b each) plus posBias', () {
      final cross = AFTCrossAttention(16, 32, 8, 6, 10);
      addTearDown(() => disposeAll(cross.parameters()));

      final params = cross.parameters();
      expect(params, hasLength(7));
      expect(params.last.shape, equals([6, 10])); // posBias [Tdec, Tenc]
    });
  });

  group('MultiHeadAFTCross', () {
    test('forward returns [T_dec, decoderEmbed]', () {
      const heads = 4;
      const decEmbed = 16;
      const encEmbed = 32;
      const tDec = 5;
      const tEnc = 9;
      final cross = MultiHeadAFTCross(
        heads,
        decEmbed,
        encEmbed,
        tDec,
        tEnc,
      );

      final xDec = Tensor.random([tDec, decEmbed]);
      final xEnc = Tensor.random([tEnc, encEmbed]);
      final tracker = <Tensor>[];

      final y = cross.forward(xDec, xEnc, tracker);
      addTearDown(() {
        xDec.dispose();
        xEnc.dispose();
        disposeAll(tracker);
        disposeAll(cross.parameters());
      });

      expect(y.shape, equals([tDec, decEmbed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('asserts when decoderEmbedSize is not divisible by numHeads', () {
      expect(
        () => MultiHeadAFTCross(3, 16, 32, 5, 8),
        throwsA(isA<AssertionError>()),
      );
    });

    test('parameters() collects every head plus the output projection', () {
      const heads = 4;
      final cross = MultiHeadAFTCross(heads, 16, 32, 5, 8);
      addTearDown(() => disposeAll(cross.parameters()));

      // Each head: 7 params, proj: 2.
      expect(cross.parameters(), hasLength(heads * 7 + 2));
    });
  });
}
