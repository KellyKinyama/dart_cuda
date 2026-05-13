// file: lib/mu_zero/example_deepseek_shakespeare.dart
//
// Tiny-Shakespeare char-level training for the DeepSeek-style MoE+AFT
// decoder. Mirrors the toy-corpus example but uses a CharTokenizer over
// `tiny_shakespeare.txt` and random fixed-length batches of `blockSize`.

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

import 'package:dart_cuda/core/models/mu_zero/deepseek_aft_decoder.dart';
import 'package:dart_cuda/mu_zero/example_deepseek_aft_training.dart' show warmupCosineLR, sampleToken;

class CharTokenizer {
  late final List<String> chars;
  late final Map<String, int> stoi;
  late final Map<int, String> itos;

  CharTokenizer(String text) {
    chars = (text.split('').toSet().toList())..sort();
    stoi = {for (var i = 0; i < chars.length; i++) chars[i]: i};
    itos = stoi.map((k, v) => MapEntry(v, k));
  }

  int get vocabSize => chars.length;

  List<int> encode(String s) => s.split('').map((c) => stoi[c] ?? 0).toList();
  String decode(List<int> l) => l.map((i) => itos[i] ?? '').join('');
}

(List<int>, List<int>) _getBatch(List<int> data, int blockSize, math.Random r) {
  final start = r.nextInt(data.length - blockSize - 1);
  final x = data.sublist(start, start + blockSize);
  final y = data.sublist(start + 1, start + blockSize + 1);
  return (x, y);
}

void _safeCleanup(List<Tensor> tracker, List<Tensor> params) {
  final freed = <int>{};
  final paramAddrs = params.map((p) => p.handle.address).toSet();
  for (final t in tracker) {
    final addr = t.handle.address;
    if (addr != 0 &&
        !freed.contains(addr) &&
        !paramAddrs.contains(addr) &&
        t.isView != true) {
      t.dispose();
      freed.add(addr);
    }
  }
}

Future<void> main() async {
  print('--- DeepSeek MoE + AFT — Tiny Shakespeare (char-level) ---');

  final file = File('tiny_shakespeare.txt');
  if (!await file.exists()) {
    print('tiny_shakespeare.txt not found at ${file.absolute.path}');
    return;
  }
  final raw = await file.readAsString();
  final tokenizer = CharTokenizer(raw);
  final data = tokenizer.encode(raw);
  print('Corpus chars: ${data.length}, vocab: ${tokenizer.vocabSize}');

  // Hyperparameters — kept modest for VRAM and CPU-side router cost.
  const int embedSize = 64;
  const int blockSize = 64;
  const int numLayers = 2;
  const int numHeads = 4;

  const int numRoutedExperts = 8;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 64;

  final model = DeepSeekAFTDecoder(
    vocabSize: tokenizer.vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
    numRoutedExperts: numRoutedExperts,
    numSharedExperts: numSharedExperts,
    topK: topK,
    expertHiddenSize: expertHiddenSize,
  );
  print(
    'Model params (tensors): ${model.parameters().length}  '
    'routed=$numRoutedExperts top-$topK shared=$numSharedExperts',
  );

  const double baseLR = 0.001;
  const double minLR = 0.0001;
  const int numSteps = 10000;
  const int balanceEvery = 50;
  const int logEvery = 50;
  final int warmupSteps = (0.05 * numSteps).round();

  final optimizer = Adam(model.parameters(), lr: baseLR);
  final dummyEnc = Tensor.zeros([1, embedSize]);
  final rng = math.Random(1234);

  double runningLoss = 0.0;
  int runningCount = 0;
  for (int step = 0; step < numSteps; step++) {
    optimizer.lr = warmupCosineLR(
      step,
      baseLR: baseLR,
      warmupSteps: warmupSteps,
      totalSteps: numSteps,
      minLR: minLR,
    );

    optimizer.zeroGrad();
    final tracker = <Tensor>[];
    final (x, y) = _getBatch(data, blockSize, rng);

    final logits = model.forward(x, dummyEnc, tracker);
    final loss = logits.crossEntropy(y);
    tracker.add(loss);
    loss.backward();
    runningLoss += loss.fetchData()[0];
    runningCount++;
    optimizer.step();
    _safeCleanup(tracker, model.parameters());

    if ((step + 1) % balanceEvery == 0) {
      // Snapshot before the bias update zeros the load counters.
      final loadSnap = [
        for (final b in model.blocks) List<int>.from(b.moe.expertLoad),
      ];
      model.updateRoutingBias();
      if ((step + 1) % logEvery == 0) {
        print(
          'step ${step + 1}  '
          'lr=${optimizer.lr.toStringAsFixed(5)}  '
          'avg_loss=${(runningLoss / runningCount).toStringAsFixed(4)}',
        );
        for (int li = 0; li < loadSnap.length; li++) {
          print('    layer $li expert load: ${loadSnap[li]}');
        }
        runningLoss = 0.0;
        runningCount = 0;
      }
    }
  }

  // Generation.
  String generate(String prompt, int length, double temperature, int topKGen) {
    final genRng = math.Random(42);
    final tokens = tokenizer.encode(prompt);
    final out = StringBuffer(prompt);
    for (int i = 0; i < length; i++) {
      // Keep the context window <= blockSize.
      final ctx = tokens.length <= blockSize
          ? tokens
          : tokens.sublist(tokens.length - blockSize);
      final tracker = <Tensor>[];
      final logits = model.forward(ctx, dummyEnc, tracker);
      final lastRow = logits.fetchRow(ctx.length - 1);
      final next = sampleToken(
        lastRow,
        temperature: temperature,
        topK: topKGen,
        rng: genRng,
      );
      _safeCleanup(tracker, model.parameters());
      tokens.add(next);
      out.write(tokenizer.decode([next]));
    }
    return out.toString();
  }

  print('\n--- Sample generation ---');
  print(generate('ROMEO: ', 200, 0.8, 10));
}
