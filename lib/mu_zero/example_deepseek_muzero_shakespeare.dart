// file: lib/mu_zero/example_deepseek_muzero_shakespeare.dart
//
// MuZero-style training for the DeepSeek MoE+AFT decoder on
// `tiny_shakespeare.txt` (char-level). Adapts the pattern from
// `dart-torch/lib/mu_zero/example_deepseek_muzero.dart` and the local
// `example2.dart` into this codebase's GPU `Tensor` autograd.
//
// MuZero pieces (language-modeling flavour, no RL env):
//   * representation h(x_{<=t}) : token ids -> latent state s_t   [T, D]
//   * dynamics       g(s, a)   : (state, next token) -> next latent state
//   * prediction     f(s)      : latent state -> next-token logits
//
// We alternate two training modes per step:
//   - POLICY mode  : standard teacher-forced LM loss on f(h(ctx)).
//   - DYNAMICS mode: K-step unroll from a single representation, with the
//     dynamics network producing s_{t+1}, ..., s_{t+K}, each scored by f().
//     This is the MuZero "imagined rollout" objective.

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

import 'package:dart_cuda/core/models/mu_zero/deepseek_aft_decoder.dart';
import 'package:dart_cuda/mu_zero/example_deepseek_aft_training.dart' show warmupCosineLR, sampleToken;
import 'package:dart_cuda/mu_zero/example_deepseek_shakespeare.dart' show CharTokenizer;

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

/// MuZero-style agent wrapping a [DeepSeekAFTDecoder].
///
/// The latent state is the decoder's post-LayerNorm activation (i.e. what
/// the LM head normally sees). Dynamics is realized by injecting an action
/// (next-token) embedding into the current state and pushing it through
/// the first decoder block.
class DeepSeekMuZeroAgent {
  final DeepSeekAFTDecoder model;
  final Tensor dummyEnc;
  final double actionScale;

  DeepSeekMuZeroAgent(this.model, {this.actionScale = 5.0})
    : dummyEnc = Tensor.zeros([1, model.encoderEmbedSize]);

  /// h(x_{<=t}) : full decoder forward up to the final LayerNorm.
  /// Returns latent states of shape [T, D].
  Tensor representation(List<int> idx, List<Tensor> tracker) {
    Tensor x = Tensor.embeddings(idx, model.wte, model.wpe);
    tracker.add(x);
    for (final block in model.blocks) {
      x = block.forward(x, dummyEnc, tracker);
    }
    final out = model.finalLayerNorm.forward(x, tracker);
    return out;
  }

  /// g(s_t, a_t) : push (state + action_embed + pos_embed) through one
  /// decoder block, then re-normalize. We use the last row of `state` as
  /// "the" current state for the imagined rollout.
  Tensor dynamics(Tensor stateRow, int action, int step, List<Tensor> tracker) {
    final actionEmb = model.wte.getRow(action);
    final posEmb = model.wpe.getRow(step % model.blockSize);
    final strongAction = actionEmb * actionScale;
    final combined = stateRow + strongAction + posEmb;
    final nextRaw = model.blocks[0].forward(combined, dummyEnc, tracker);
    final next = model.finalLayerNorm.forward(nextRaw, tracker);
    tracker.addAll([actionEmb, strongAction, posEmb, combined]);
    return next;
  }

  /// f(s) : latent state -> next-token logits.
  Tensor predictPolicy(Tensor state, List<Tensor> tracker) {
    return model.lmHead.forward(state, tracker);
  }
}

Future<void> main() async {
  print('--- DeepSeek MoE + AFT MuZero — Tiny Shakespeare (char-level) ---');

  final file = File('tiny_shakespeare.txt');
  if (!await file.exists()) {
    print('tiny_shakespeare.txt not found at ${file.absolute.path}');
    return;
  }
  final raw = await file.readAsString();
  final tokenizer = CharTokenizer(raw);
  final data = tokenizer.encode(raw);
  print('Corpus chars: ${data.length}, vocab: ${tokenizer.vocabSize}');

  // Hyperparameters.
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
  final agent = DeepSeekMuZeroAgent(model);
  print(
    'Model params (tensors): ${model.parameters().length}  '
    'routed=$numRoutedExperts top-$topK shared=$numSharedExperts',
  );

  // Training schedule.
  const double baseLR = 0.001;
  const double minLR = 0.0001;
  const int numSteps = 10000;
  const int balanceEvery = 50;
  const int logEvery = 50;
  const int unrollSteps = 3; // MuZero K-step unroll
  const int dynamicsEvery = 2; // every Nth step is a dynamics-unroll step
  final int warmupSteps = (0.05 * numSteps).round();

  final optimizer = Adam(model.parameters(), lr: baseLR);
  final rng = math.Random(1234);

  double runningPolicyLoss = 0.0;
  double runningDynamicsLoss = 0.0;
  int runningPolicy = 0;
  int runningDynamics = 0;

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

    final bool dynamicsMode = (step % dynamicsEvery == 1);

    if (!dynamicsMode) {
      // ----- POLICY mode: standard teacher-forced LM loss. -----
      final states = agent.representation(x, tracker);
      final logits = agent.predictPolicy(states, tracker);
      final loss = logits.crossEntropy(y);
      tracker.add(loss);
      loss.backward();
      runningPolicyLoss += loss.fetchData()[0];
      runningPolicy++;
    } else {
      // ----- DYNAMICS mode: K-step imagined rollout. -----
      // Pick a random starting position inside the batch with room to unroll.
      final int maxStart = blockSize - unrollSteps - 1;
      final int start = maxStart > 0 ? rng.nextInt(maxStart) : 0;

      // 1. Initial representation on prefix [0..start].
      final prefix = x.sublist(0, start + 1);
      final s0Full = agent.representation(prefix, tracker);
      // Take only the last row (the "current" latent state).
      Tensor stateRow = s0Full.getRow(s0Full.shape[0] - 1);
      tracker.add(stateRow);

      // 2. Predict next token at s0.
      final logits0 = agent.predictPolicy(stateRow, tracker);
      Tensor loss = logits0.crossEntropy([y[start]]);
      tracker.add(loss);

      // 3. Unroll K steps using dynamics(state, action_taken).
      for (int k = 0; k < unrollSteps; k++) {
        final int idx = start + k;
        if (idx + 1 >= y.length) break;
        final int aTaken = x[idx + 1]; // teacher action = next ground-truth tok
        final next = agent.dynamics(stateRow, aTaken, idx + 1, tracker);
        final logitsK = agent.predictPolicy(next, tracker);
        final lossK = logitsK.crossEntropy([y[idx + 1]]);
        tracker.add(lossK);
        loss = loss + lossK;
        tracker.add(loss);
        stateRow = next;
      }

      loss.backward();
      runningDynamicsLoss += loss.fetchData()[0];
      runningDynamics++;
    }

    optimizer.step();
    _safeCleanup(tracker, model.parameters());

    if ((step + 1) % balanceEvery == 0) {
      final loadSnap = [
        for (final b in model.blocks) List<int>.from(b.moe.expertLoad),
      ];
      model.updateRoutingBias();
      if ((step + 1) % logEvery == 0) {
        final pAvg = runningPolicy > 0
            ? (runningPolicyLoss / runningPolicy).toStringAsFixed(4)
            : '   -  ';
        final dAvg = runningDynamics > 0
            ? (runningDynamicsLoss / runningDynamics).toStringAsFixed(4)
            : '   -  ';
        print(
          'step ${step + 1}  '
          'lr=${optimizer.lr.toStringAsFixed(5)}  '
          'policy_loss=$pAvg  dyn_loss=$dAvg',
        );
        for (int li = 0; li < loadSnap.length; li++) {
          print('    layer $li expert load: ${loadSnap[li]}');
        }
        runningPolicyLoss = 0.0;
        runningDynamicsLoss = 0.0;
        runningPolicy = 0;
        runningDynamics = 0;
      }
    }
  }

  // ----- Generation: "pure MuZero" — one representation, then pure dynamics.
  String generatePureMuZero(
    String prompt,
    int length,
    double temperature,
    int topKGen,
  ) {
    final genRng = math.Random(42);
    final ids = tokenizer.encode(prompt);
    final out = StringBuffer(prompt);

    // Initial representation on the prompt.
    final initTracker = <Tensor>[];
    final ctx = ids.length <= blockSize
        ? ids
        : ids.sublist(ids.length - blockSize);
    final s0Full = agent.representation(ctx, initTracker);
    Tensor stateRow = s0Full.getRow(s0Full.shape[0] - 1).detach();
    _safeCleanup(initTracker, model.parameters());

    for (int i = 0; i < length; i++) {
      final tracker = <Tensor>[];
      final logits = agent.predictPolicy(stateRow, tracker);
      final row = logits.fetchData(); // 1 x V
      final next = sampleToken(
        row,
        temperature: temperature,
        topK: topKGen,
        rng: genRng,
      );
      out.write(tokenizer.decode([next]));

      // Imagined transition.
      final nextStateRaw = agent.dynamics(stateRow, next, i + 1, tracker);
      final detached = nextStateRaw.detach();
      _safeCleanup(tracker, model.parameters());
      stateRow.dispose();
      stateRow = detached;
    }
    stateRow.dispose();
    return out.toString();
  }

  // Also provide a standard "teacher-style" sampler that re-encodes context
  // each step (h every step, no pure-dynamics imagination) for comparison.
  String generateWithRepresentation(
    String prompt,
    int length,
    double temperature,
    int topKGen,
  ) {
    final genRng = math.Random(42);
    final ids = tokenizer.encode(prompt);
    final out = StringBuffer(prompt);
    for (int i = 0; i < length; i++) {
      final ctx = ids.length <= blockSize
          ? ids
          : ids.sublist(ids.length - blockSize);
      final tracker = <Tensor>[];
      final states = agent.representation(ctx, tracker);
      final logits = agent.predictPolicy(states, tracker);
      final lastRow = logits.fetchRow(ctx.length - 1);
      final next = sampleToken(
        lastRow,
        temperature: temperature,
        topK: topKGen,
        rng: genRng,
      );
      _safeCleanup(tracker, model.parameters());
      ids.add(next);
      out.write(tokenizer.decode([next]));
    }
    return out.toString();
  }

  print('\n--- Sample generation (representation each step) ---');
  print(generateWithRepresentation('ROMEO: ', 200, 0.8, 10));

  print('\n--- Sample generation (pure MuZero dynamics rollout) ---');
  print(generatePureMuZero('ROMEO: ', 200, 0.8, 10));
}
