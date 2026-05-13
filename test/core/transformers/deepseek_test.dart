// Tests for the DeepSeek MoE AFT decoder (lib/core/transformers/deepseek/).
//
// Requires CUDA backend; run from repo root:
//   dart test --concurrency=1 test/core/transformers/deepseek_test.dart

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';
import 'package:test/test.dart';

void disposeAll(Iterable<Tensor> ts) {
  for (final t in ts) {
    t.dispose();
  }
}

bool _allFinite(Iterable<double> xs) =>
    xs.every((v) => !v.isNaN && !v.isInfinite);

void main() {
  group('Expert', () {
    test('forward preserves [T, dim] shape', () {
      const dim = 8;
      const hidden = 16;
      const seq = 4;
      final expert = Expert(dim, hidden);
      final x = Tensor.random([seq, dim], scale: 0.1);
      final tracker = <Tensor>[];

      final y = expert.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(expert.parameters());
      });

      expect(y.shape, equals([seq, dim]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() returns 4 (w1+b1, w2+b2)', () {
      final expert = Expert(8, 16);
      addTearDown(() => disposeAll(expert.parameters()));
      expect(expert.parameters(), hasLength(4));
    });
  });

  group('MoEFeedForward', () {
    test('forward preserves [T, embedSize] shape', () {
      const embed = 8;
      const seq = 5;
      final moe = MoEFeedForward(
        embedSize: embed,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenSize: 16,
      );
      final x = Tensor.random([seq, embed], scale: 0.1);
      final tracker = <Tensor>[];

      final y = moe.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(moe.parameters());
      });

      expect(y.shape, equals([seq, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() includes gateW + every expert', () {
      const numRouted = 4;
      const numShared = 2;
      final moe = MoEFeedForward(
        embedSize: 8,
        numRoutedExperts: numRouted,
        numSharedExperts: numShared,
        topK: 2,
        expertHiddenSize: 16,
      );
      addTearDown(() => disposeAll(moe.parameters()));

      // gateW + (numRouted + numShared) experts × 4 params each.
      expect(moe.parameters(), hasLength(1 + (numRouted + numShared) * 4));
    });

    test('expertLoad starts at zero for every routed expert', () {
      const numRouted = 4;
      final moe = MoEFeedForward(
        embedSize: 8,
        numRoutedExperts: numRouted,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenSize: 16,
      );
      addTearDown(() => disposeAll(moe.parameters()));

      expect(moe.expertLoad, hasLength(numRouted));
      expect(moe.expertLoad, everyElement(equals(0)));
    });
  });

  group('DeepSeekAFTDecoderBlock', () {
    test('forward returns [T, embedSize] given encoder context', () {
      const embed = 16;
      const heads = 4;
      const encEmbed = 16;
      const seq = 4;
      final block = DeepSeekAFTDecoderBlock(
        embed,
        heads,
        encEmbed,
        seq,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenSize: 16,
      );
      final xDec = Tensor.random([seq, embed], scale: 0.05);
      final xEnc = Tensor.random([seq, encEmbed], scale: 0.05);
      final tracker = <Tensor>[];

      final y = block.forward(xDec, xEnc, tracker);
      addTearDown(() {
        xDec.dispose();
        xEnc.dispose();
        disposeAll(tracker);
        disposeAll(block.parameters());
      });

      expect(y.shape, equals([seq, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });
  });

  group('DeepSeekAFTDecoder', () {
    test('forward returns [T, vocabSize] logits', () {
      const vocab = 16;
      const embed = 16;
      const block = 6;
      final dec = DeepSeekAFTDecoder(
        vocabSize: vocab,
        embedSize: embed,
        blockSize: block,
        numLayers: 1,
        numHeads: 4,
        encoderEmbedSize: embed,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenSize: 16,
      );
      final ids = [0, 1, 2, 3];
      final encOut = Tensor.random([ids.length, embed], scale: 0.05);
      final tracker = <Tensor>[];

      final logits = dec.forward(ids, encOut, tracker);
      addTearDown(() {
        encOut.dispose();
        disposeAll(tracker);
        disposeAll(dec.parameters());
      });

      expect(logits.shape, equals([ids.length, vocab]));
      expect(_allFinite(logits.fetchData()), isTrue);
    });

    test('throws when sequence length exceeds blockSize', () {
      final dec = DeepSeekAFTDecoder(
        vocabSize: 8,
        embedSize: 8,
        blockSize: 3,
        numLayers: 1,
        numHeads: 2,
        encoderEmbedSize: 8,
        numRoutedExperts: 2,
        numSharedExperts: 1,
        topK: 1,
        expertHiddenSize: 8,
      );
      final encOut = Tensor.random([1, 8]);
      addTearDown(() {
        encOut.dispose();
        disposeAll(dec.parameters());
      });

      expect(
        () => dec.forward([0, 1, 2, 3], encOut, <Tensor>[]),
        throwsArgumentError,
      );
    });

    test('updateRoutingBias runs without throwing', () {
      final dec = DeepSeekAFTDecoder(
        vocabSize: 8,
        embedSize: 8,
        blockSize: 4,
        numLayers: 1,
        numHeads: 2,
        encoderEmbedSize: 8,
        numRoutedExperts: 4,
        numSharedExperts: 1,
        topK: 2,
        expertHiddenSize: 8,
      );
      addTearDown(() => disposeAll(dec.parameters()));
      expect(dec.updateRoutingBias, returnsNormally);
    });
  });
}
