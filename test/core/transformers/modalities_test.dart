// Tests for modality transformer modules (lib/core/transformers/modalities/).
//
// Requires CUDA backend; run from repo root:
//   dart test --concurrency=1 test/core/transformers/modalities_test.dart

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/modalities/audio_transformer.dart';
import 'package:dart_cuda/core/transformers/modalities/multi_modal_transformer.dart';
import 'package:dart_cuda/core/transformers/modalities/text_decoder.dart';
import 'package:dart_cuda/core/transformers/modalities/text_transformer.dart';
import 'package:dart_cuda/core/transformers/modalities/video_transformer.dart';
import 'package:test/test.dart';

void disposeAll(Iterable<Tensor> ts) {
  for (final t in ts) {
    t.dispose();
  }
}

bool _allFinite(Iterable<double> xs) =>
    xs.every((v) => !v.isNaN && !v.isInfinite);

void main() {
  group('TextTransformer', () {
    test('encodes a sequence to [T, embedSize]', () {
      const vocab = 32;
      const embed = 16;
      const maxLen = 8;
      final tt = TextTransformer(
        vocabSize: vocab,
        maxSeqLen: maxLen,
        embedSize: embed,
        numLayers: 1,
        numHeads: 4,
      );
      final ids = [0, 1, 2, 3, 4];
      final tracker = <Tensor>[];

      final y = tt.forward(ids, tracker);
      addTearDown(() {
        disposeAll(tracker);
        disposeAll(tt.parameters());
      });

      expect(y.shape, equals([ids.length, embed]));
      expect(_allFinite(y.fetchData()), isTrue);
    });
  });

  group('AudioTransformer', () {
    test('returns [1, numClasses] logits over a feature sequence', () {
      const featureDim = 8;
      const maxSeq = 16;
      const embed = 16;
      const classes = 4;
      const seqLen = 6;

      final audio = AudioTransformer(
        featureDim: featureDim,
        maxSequenceLength: maxSeq,
        embedSize: embed,
        numClasses: classes,
        numLayers: 1,
        numHeads: 4,
      );
      final feats = Tensor.random([seqLen, featureDim], scale: 0.1);
      final tracker = <Tensor>[];

      final logits = audio.forward(feats, tracker);
      addTearDown(() {
        feats.dispose();
        disposeAll(tracker);
        disposeAll(audio.parameters());
      });

      expect(logits.shape, equals([1, classes]));
      expect(_allFinite(logits.fetchData()), isTrue);
    });
  });

  group('VideoTransformer', () {
    test('returns [1, numClasses] logits when frame embeds need projection',
        () {
      const frameDim = 12;
      const embed = 16;
      const maxLen = 10;
      const classes = 5;
      const numFrames = 4;

      final video = VideoTransformer(
        frameEmbedDim: frameDim,
        embedSize: embed,
        maxVideoSequenceLength: maxLen,
        numClasses: classes,
        numLayers: 1,
        numHeads: 4,
      );
      final frames = Tensor.random([numFrames, frameDim], scale: 0.1);
      final tracker = <Tensor>[];

      final logits = video.forward(frames, tracker);
      addTearDown(() {
        frames.dispose();
        disposeAll(tracker);
        disposeAll(video.parameters());
      });

      expect(logits.shape, equals([1, classes]));
      expect(_allFinite(logits.fetchData()), isTrue);
    });

    test('throws when video exceeds maxVideoSequenceLength', () {
      final video = VideoTransformer(
        frameEmbedDim: 8,
        embedSize: 8,
        maxVideoSequenceLength: 3,
        numClasses: 2,
        numLayers: 1,
        numHeads: 2,
      );
      final frames = Tensor.random([5, 8]);
      addTearDown(() {
        frames.dispose();
        disposeAll(video.parameters());
      });

      expect(
        () => video.forward(frames, <Tensor>[]),
        throwsArgumentError,
      );
    });
  });

  group('MultimodalTransformer', () {
    test('intermediate-fusion forward returns [1, numClasses]', () {
      const audioFeatureDim = 8;
      const audioMaxLen = 10;
      const audioEmbed = 16;
      const videoFrameDim = 12;
      const videoMaxLen = 8;
      const videoEmbed = 16;
      const classes = 3;

      final audio = AudioTransformer(
        featureDim: audioFeatureDim,
        maxSequenceLength: audioMaxLen,
        embedSize: audioEmbed,
        numClasses: classes,
        numLayers: 1,
        numHeads: 4,
      );
      final video = VideoTransformer(
        frameEmbedDim: videoFrameDim,
        embedSize: videoEmbed,
        maxVideoSequenceLength: videoMaxLen,
        numClasses: classes,
        numLayers: 1,
        numHeads: 4,
      );
      final mm = MultimodalTransformer(
        audioModel: audio,
        videoModel: video,
        numClasses: classes,
      );

      final audioIn = Tensor.random([5, audioFeatureDim], scale: 0.1);
      final videoIn = Tensor.random([4, videoFrameDim], scale: 0.1);
      final tracker = <Tensor>[];

      final logits = mm.forward(audioIn, videoIn, tracker);
      addTearDown(() {
        audioIn.dispose();
        videoIn.dispose();
        disposeAll(tracker);
        disposeAll(mm.parameters());
      });

      expect(logits.shape, equals([1, classes]));
      expect(_allFinite(logits.fetchData()), isTrue);
    });
  });

  group('TextDecoder', () {
    test('returns [T, vocabSize] logits given encoder context', () {
      const vocab = 16;
      const embed = 16;
      const maxLen = 6;

      final dec = TextDecoder(
        vocabSize: vocab,
        maxSeqLen: maxLen,
        embedSize: embed,
        numLayers: 1,
        numHeads: 4,
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

    test('throws when text length exceeds maxSeqLen', () {
      final dec = TextDecoder(
        vocabSize: 8,
        maxSeqLen: 3,
        embedSize: 8,
        numLayers: 1,
        numHeads: 2,
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
