// file: lib/mu_zero/example_deepseek_aft_training.dart
//
// Training example for the DeepSeek-style MoE decoder using AFT attention,
// adapted to this codebase's GPU `Tensor` autograd.
//
// Demonstrates:
//   * MoE FFN with routed (top-K) and shared experts.
//   * DeepSeek-V3 aux-loss-free routing-bias balancing.
//   * Warmup + cosine LR schedule.
//   * Temperature / top-K sampling at generation time.

import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';

/// Linear warmup followed by cosine decay to `minLR`.
double warmupCosineLR(
  int step, {
  required double baseLR,
  required int warmupSteps,
  required int totalSteps,
  required double minLR,
}) {
  if (warmupSteps > 0 && step < warmupSteps) {
    return baseLR * (step + 1) / warmupSteps;
  }
  final decaySteps = math.max(1, totalSteps - warmupSteps);
  final progress = ((step - warmupSteps) / decaySteps).clamp(0.0, 1.0);
  final cosine = 0.5 * (1.0 + math.cos(math.pi * progress));
  return minLR + (baseLR - minLR) * cosine;
}

/// Temperature + top-K sampling from a single logits row.
/// `temperature == 0.0` => greedy argmax.
int sampleToken(
  List<double> logits, {
  double temperature = 0.8,
  int topK = 5,
  required math.Random rng,
}) {
  if (temperature <= 0.0) {
    int best = 0;
    double bestV = logits[0];
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > bestV) {
        bestV = logits[i];
        best = i;
      }
    }
    return best;
  }

  final scaled = logits.map((v) => v / temperature).toList();
  // Restrict to top-K.
  final indexed = List<MapEntry<int, double>>.generate(
    scaled.length,
    (i) => MapEntry(i, scaled[i]),
  );
  indexed.sort((a, b) => b.value.compareTo(a.value));
  final k = math.min(topK, indexed.length);
  final candidates = indexed.sublist(0, k);

  // Softmax over candidates.
  final maxV = candidates.first.value;
  final exps = candidates.map((e) => math.exp(e.value - maxV)).toList();
  final sumExp = exps.fold<double>(0.0, (a, b) => a + b);

  final r = rng.nextDouble() * sumExp;
  double cum = 0.0;
  for (int i = 0; i < candidates.length; i++) {
    cum += exps[i];
    if (r <= cum) return candidates[i].key;
  }
  return candidates.last.key;
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

void main() {
  print('--- DeepSeek-style MoE + AFT Decoder Training Example ---');

  // 1. Hyperparameters.
  const int embedSize = 32;
  const int blockSize = 12; // also AFT maxSeqLen
  const int numLayers = 2;
  const int numHeads = 4;

  // MoE configuration (DeepSeekMoE flavour).
  const int numRoutedExperts = 8;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 32;

  // 2. Vocabulary.
  const String startTok = '<start>';
  const String padTok = '<pad>';

  // 3. Toy corpus — varied so routed experts have something to specialise on.
  final List<List<String>> corpus = [
    [startTok, 'hello', 'world', '.'],
    [startTok, 'hello', 'there', '.'],
    [startTok, 'the', 'quick', 'brown', 'fox', 'jumps', '.'],
    [startTok, 'the', 'lazy', 'dog', 'sleeps', '.'],
    [startTok, 'a', 'cat', 'sits', 'on', 'the', 'mat', '.'],
    [startTok, 'this', 'is', 'a', 'test', '.'],
    [startTok, 'i', 'love', 'dart', '.'],
    [startTok, 'we', 'build', 'small', 'models', '.'],
  ];

  final Map<String, int> stoi = {padTok: 0, startTok: 1};
  for (final s in corpus) {
    for (final w in s) {
      stoi.putIfAbsent(w, () => stoi.length);
    }
  }
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));
  final int vocabSize = stoi.length;
  final int startTokenId = stoi[startTok]!;
  print('Vocabulary size: $vocabSize');

  // 4. Build (input, target) pairs WITHOUT padding (variable T <= blockSize).
  final List<List<int>> trainInputs = [];
  final List<List<int>> trainTargets = [];
  for (final sent in corpus) {
    final ids = sent.map((w) => stoi[w]!).toList();
    if (ids.length < 2 || ids.length > blockSize + 1) continue;
    trainInputs.add(ids.sublist(0, ids.length - 1));
    trainTargets.add(ids.sublist(1));
  }

  // 5. Model.
  print('\nInitializing DeepSeekAFTDecoder (MoE + AFT)...');
  final model = DeepSeekAFTDecoder(
    vocabSize: vocabSize,
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
  print('Model initialised. Parameter tensors: ${model.parameters().length}');
  print('  routed experts/layer : $numRoutedExperts (top-$topK active)');
  print('  shared experts/layer : $numSharedExperts');

  // 6. Optimiser + warmup/cosine LR schedule.
  const double baseLR = 0.001;
  const double minLR = 0.0001;
  const int numEpochs = 200;
  final int totalSteps = numEpochs * trainInputs.length;
  final int warmupSteps = (0.05 * totalSteps).round();
  final optimizer = Adam(model.parameters(), lr: baseLR);

  // 7. Dummy encoder output (decoder-only style).
  final dummyEnc = Tensor.zeros([1, embedSize]);

  // 8. Training loop.
  int globalStep = 0;
  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < trainInputs.length; i++) {
      optimizer.lr = warmupCosineLR(
        globalStep,
        baseLR: baseLR,
        warmupSteps: warmupSteps,
        totalSteps: totalSteps,
        minLR: minLR,
      );

      optimizer.zeroGrad();
      final tracker = <Tensor>[];

      final logits = model.forward(trainInputs[i], dummyEnc, tracker);
      final loss = logits.crossEntropy(trainTargets[i]);
      tracker.add(loss);

      loss.backward();
      totalLoss += loss.fetchData()[0];
      optimizer.step();

      _safeCleanup(tracker, model.parameters());
      globalStep++;
    }

    // Snapshot routed-expert load BEFORE the bias update zeros the counters.
    final loadSnapshot = [
      for (final b in model.blocks) List<int>.from(b.moe.expertLoad),
    ];

    // DeepSeek-V3 aux-loss-free routing-bias update once per epoch.
    model.updateRoutingBias();

    if ((epoch + 1) % 10 == 0) {
      print(
        'Epoch ${epoch + 1}  '
        'lr=${optimizer.lr.toStringAsFixed(5)}  '
        'avg_loss=${(totalLoss / trainInputs.length).toStringAsFixed(4)}',
      );
      for (int li = 0; li < loadSnapshot.length; li++) {
        print('    layer $li expert load: ${loadSnapshot[li]}');
      }
    }
  }

  // 9. Generation — temperature + top-K sampling.
  final rng = math.Random(42);

  String generate({
    required int seedTokenId,
    double temperature = 0.8,
    int topK = 5,
  }) {
    final generated = <int>[seedTokenId];
    for (int i = 0; i < blockSize - 1; i++) {
      final tracker = <Tensor>[];
      final logits = model.forward(generated, dummyEnc, tracker);
      final lastRow = logits.fetchRow(generated.length - 1);
      final next = sampleToken(
        lastRow,
        temperature: temperature,
        topK: topK,
        rng: rng,
      );
      _safeCleanup(tracker, model.parameters());
      generated.add(next);
      if (itos[next] == '.') break;
    }
    return generated.map((id) => itos[id]).join(' ');
  }

  print('\n--- Generated samples ---');
  print(
    'greedy           : '
    '${generate(seedTokenId: startTokenId, temperature: 0.0)}',
  );
  print('temp=0.8 top-k=5 : ${generate(seedTokenId: startTokenId)}');
  print(
    'temp=1.2 top-k=5 : '
    '${generate(seedTokenId: startTokenId, temperature: 1.2)}',
  );
  print(
    'temp=0.8 top-k=3 : '
    '${generate(seedTokenId: startTokenId, temperature: 0.8, topK: 3)}',
  );
}
