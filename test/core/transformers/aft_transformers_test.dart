// Tests for AFT transformer modules (lib/core/transformers/aft/).
//
// Requires the CUDA backend; run from repo root with:
//   dart test --concurrency=1 test/core/transformers/

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/aft/text_decoder_block.dart';
import 'package:dart_cuda/core/transformers/aft/transformer_decoder.dart';
import 'package:dart_cuda/core/transformers/aft/transformer_decoder_block.dart';
import 'package:dart_cuda/core/transformers/aft/transformer_encoder.dart';
import 'package:dart_cuda/core/transformers/aft/transformer_encoder_block.dart';
import 'package:test/test.dart';

void disposeAll(Iterable<Tensor> ts) {
  for (final t in ts) {
    t.dispose();
  }
}

bool _allFinite(Iterable<double> xs) =>
    xs.every((v) => !v.isNaN && !v.isInfinite);

void main() {
  group('TransformerEncoderBlock', () {
    test('forward preserves [T, embedSize] shape', () {
      const embed = 16;
      const heads = 4;
      const seq = 6;
      final block = TransformerEncoderBlock(embed, heads, seq);
      final x = Tensor.random([seq, embed], scale: 0.1);
      final tracker = <Tensor>[];

      final y = block.forward(x, tracker);
      addTearDown(() {
        x.dispose();
        disposeAll(tracker);
        disposeAll(block.parameters());
      });

      expect(y.shape, equals([seq, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('parameters() aggregates attention + ffn + 2× layer-norm', () {
      const embed = 16;
      const heads = 4;
      const seq = 6;
      final block = TransformerEncoderBlock(embed, heads, seq);
      addTearDown(() => disposeAll(block.parameters()));

      // attention: heads*7 + 2 (proj). ffn (Layer): 2. ln1+ln2: 2*2 = 4.
      expect(block.parameters(), hasLength(heads * 7 + 2 + 2 + 4));
    });
  });

  group('TransformerDecoderBlock', () {
    test('forward returns [T_dec, embedSize] given encoder context', () {
      const embed = 16;
      const heads = 4;
      const encEmbed = 24;
      const seq = 5;
      final block = TransformerDecoderBlock(embed, heads, encEmbed, seq);
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

    test('parameters() aggregates self+cross attention, ffn, 3× layer-norm',
        () {
      const embed = 16;
      const heads = 4;
      const encEmbed = 24;
      const seq = 5;
      final block = TransformerDecoderBlock(embed, heads, encEmbed, seq);
      addTearDown(() => disposeAll(block.parameters()));

      // self-attn MultiHeadAFT: heads*7 + 2.
      // cross-attn MultiHeadAFTCross: heads*7 + 2.
      // FeedForward (D→4D→D, 2 Layers): 4.
      // ln1+ln2+ln3: 3*2 = 6.
      expect(
        block.parameters(),
        hasLength((heads * 7 + 2) * 2 + 4 + 6),
      );
    });
  });

  group('TextDecoderBlock', () {
    test('forward returns [T, embedSize] using cross-attention via kv', () {
      const embed = 16;
      const heads = 4;
      const seq = 5;
      final block = TextDecoderBlock(embed, heads, seq);
      final x = Tensor.random([seq, embed], scale: 0.05);
      final enc = Tensor.random([seq, embed], scale: 0.05);
      final tracker = <Tensor>[];

      final y = block.forward(x, enc, tracker);
      addTearDown(() {
        x.dispose();
        enc.dispose();
        disposeAll(tracker);
        disposeAll(block.parameters());
      });

      expect(y.shape, equals([seq, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });
  });

  group('TransformerEncoder', () {
    test('forward over token IDs returns [T, embedSize]', () {
      const vocab = 32;
      const embed = 16;
      const block = 8;
      const layers = 2;
      const heads = 4;
      final enc = TransformerEncoder(
        vocabSize: vocab,
        embedSize: embed,
        blockSize: block,
        numLayers: layers,
        numHeads: heads,
      );
      final ids = [3, 7, 0, 11, 4];
      final tracker = <Tensor>[];

      final y = enc.forward(ids, tracker);
      addTearDown(() {
        disposeAll(tracker);
        disposeAll(enc.parameters());
      });

      expect(y.shape, equals([ids.length, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });

    test('throws when sequence length exceeds blockSize', () {
      final enc = TransformerEncoder(
        vocabSize: 16,
        embedSize: 8,
        blockSize: 4,
        numLayers: 1,
        numHeads: 2,
      );
      addTearDown(() => disposeAll(enc.parameters()));

      expect(
        () => enc.forward([0, 1, 2, 3, 0], <Tensor>[]),
        throwsArgumentError,
      );
    });

    test('asserts when embedSize is not divisible by numHeads', () {
      expect(
        () => TransformerEncoder(
          vocabSize: 16,
          embedSize: 10,
          blockSize: 4,
          numLayers: 1,
          numHeads: 4,
        ),
        throwsA(isA<AssertionError>()),
      );
    });
  });

  group('TransformerDecoder', () {
    test('forward returns logits of shape [T, vocabSize]', () {
      const vocab = 16;
      const embed = 16;
      const block = 6;
      const layers = 1;
      const heads = 4;
      const encEmbed = 16;

      final dec = TransformerDecoder(
        vocabSize: vocab,
        embedSize: embed,
        blockSize: block,
        numLayers: layers,
        numHeads: heads,
        encoderEmbedSize: encEmbed,
      );
      final ids = [0, 1, 2, 3];
      final encOut = Tensor.random([ids.length, encEmbed], scale: 0.05);
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
      final dec = TransformerDecoder(
        vocabSize: 8,
        embedSize: 8,
        blockSize: 3,
        numLayers: 1,
        numHeads: 2,
        encoderEmbedSize: 8,
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
  });
}
