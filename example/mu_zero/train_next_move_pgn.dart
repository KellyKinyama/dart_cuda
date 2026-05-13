// Train a tiny `ChessMuZeroAgent`'s policy head to predict the next UCI
// move from PGN-derived games (the bundled `lib/loaders/dataset.dart` set,
// produced by the existing `lib/loaders/chess.dart` parser).
//
// Per game we form input/target pairs by shifting tokens by 1:
//
//   input  = [<start>, m_1, m_2, ..., m_{T-1}]
//   target = [m_1,     m_2, m_3, ..., m_T]
//
// Loss = mean cross-entropy of decoder logits vs `target`.
//
// Usage (from repo root):
//   dart run example/mu_zero/train_next_move_pgn.dart
//
// Optional flags:
//   --games=N        (default 32)   how many PGN games to use
//   --steps=N        (default 600)  total optimizer steps
//   --block=N        (default 32)   max sequence length per game
//   --embed=N        (default 64)   embedding size
//   --layers=N       (default 2)    decoder layers
//   --heads=N        (default 4)    AFT heads
//   --lr=F           (default 1e-3) Adam learning rate
//   --logEvery=N     (default 25)   log loss every N steps
//   --sampleEvery=N  (default 100)  print a greedy continuation every N steps
//   --seed=N         (default 7)    RNG seed for batch sampling

import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';
import 'package:dart_cuda/core/models/mu_zero/muzero_chess_player.dart'
    show ChessMuZeroAgent, MoveTokenizer;
import 'package:dart_cuda/loaders/dataset.dart' show dataset;

int _intFlag(List<String> args, String name, int fallback) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) return int.tryParse(a.substring(p.length)) ?? fallback;
  }
  return fallback;
}

double _doubleFlag(List<String> args, String name, double fallback) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) {
      return double.tryParse(a.substring(p.length)) ?? fallback;
    }
  }
  return fallback;
}

int _argmaxRow(List<double> logits, int rowIdx, int vocabSize) {
  final base = rowIdx * vocabSize;
  var best = 0;
  var bestV = logits[base];
  for (var i = 1; i < vocabSize; i++) {
    final v = logits[base + i];
    if (v > bestV) {
      bestV = v;
      best = i;
    }
  }
  return best;
}

/// Greedy-sample up to `maxNew` next moves from a fresh `<start>` context.
List<String> _sampleGreedy(
  ChessMuZeroAgent agent,
  MoveTokenizer tok,
  int maxNew,
  int blockSize,
) {
  final history = <int>[tok.startId];
  final out = <String>[];
  for (var step = 0; step < maxNew; step++) {
    final ctx = history.length <= blockSize
        ? history
        : history.sublist(history.length - blockSize);
    final tracker = <Tensor>[];
    final state = agent.representation(ctx, tracker);
    final logits = agent.predictPolicy(state, tracker);
    final flat = logits.fetchData();
    final next = _argmaxRow(flat, ctx.length - 1, tok.vocabSize);
    for (final t in tracker) {
      t.dispose();
    }
    if (next == tok.endId) break;
    history.add(next);
    out.add(tok.decode(next));
  }
  return out;
}

Future<void> main(List<String> args) async {
  final numGames = _intFlag(args, 'games', 32).clamp(1, dataset.length);
  final numSteps = _intFlag(args, 'steps', 600);
  final blockSize = _intFlag(args, 'block', 32);
  final embedSize = _intFlag(args, 'embed', 64);
  final numLayers = _intFlag(args, 'layers', 2);
  final numHeads = _intFlag(args, 'heads', 4);
  final lr = _doubleFlag(args, 'lr', 1e-3);
  final logEvery = _intFlag(args, 'logEvery', 25);
  final sampleEvery = _intFlag(args, 'sampleEvery', 100);
  final seed = _intFlag(args, 'seed', 7);

  // ---- 1. Dataset --------------------------------------------------------
  final games = dataset.take(numGames).toList();
  final tok = MoveTokenizer.fromGames(games);
  print(
    'Loaded $numGames games | vocab=${tok.vocabSize} | '
    'avg_len=${(games.map((g) => g.length).reduce((a, b) => a + b) / games.length).toStringAsFixed(1)}',
  );

  // Pre-build per-game token streams: [<start>, m1, ..., mN] truncated to blockSize.
  final streams = <List<int>>[];
  for (final g in games) {
    final ids = <int>[tok.startId];
    for (final mv in g) {
      final id = tok.encode(mv);
      if (id == null) continue;
      ids.add(id);
      if (ids.length >= blockSize + 1) break;
    }
    // Need at least 2 tokens for a (input, target) pair.
    if (ids.length >= 2) streams.add(ids);
  }
  print('Usable streams: ${streams.length}');

  // ---- 2. Model ----------------------------------------------------------
  final model = DeepSeekAFTDecoder(
    vocabSize: tok.vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
    numRoutedExperts: 4,
    numSharedExperts: 1,
    topK: 2,
    expertHiddenSize: embedSize,
  );
  final agent = ChessMuZeroAgent(model);
  final params = agent.parameters();
  final opt = Adam(params, lr: lr);

  print(
    '🚀 Training next-move policy '
    '(steps=$numSteps, lr=$lr, embed=$embedSize, layers=$numLayers, '
    'heads=$numHeads, block=$blockSize)',
  );

  // ---- 3. Training loop --------------------------------------------------
  final rng = math.Random(seed);
  double emaLoss = double.nan;
  for (var step = 1; step <= numSteps; step++) {
    final stream = streams[rng.nextInt(streams.length)];
    // input = stream[0..len-2], target = stream[1..len-1]
    final input = stream.sublist(0, stream.length - 1);
    final target = stream.sublist(1);

    opt.zeroGrad();
    final tracker = <Tensor>[];
    final state = agent.representation(input, tracker);
    final logits = agent.predictPolicy(state, tracker); // [T, V]
    final loss = logits.crossEntropy(target);
    tracker.add(loss);

    loss.backward();
    opt.step();

    final lossVal = loss.fetchData()[0];
    emaLoss = emaLoss.isNaN ? lossVal : 0.9 * emaLoss + 0.1 * lossVal;

    for (final t in tracker) {
      t.dispose();
    }

    if (step == 1 || step % logEvery == 0) {
      // Token-level top-1 accuracy on the same minibatch.
      final tracker2 = <Tensor>[];
      final s2 = agent.representation(input, tracker2);
      final l2 = agent.predictPolicy(s2, tracker2);
      final flat = l2.fetchData();
      var correct = 0;
      for (var i = 0; i < target.length; i++) {
        if (_argmaxRow(flat, i, tok.vocabSize) == target[i]) correct++;
      }
      final acc = correct / target.length;
      for (final t in tracker2) {
        t.dispose();
      }

      print(
        '  step ${step.toString().padLeft(4)} | '
        'loss=${lossVal.toStringAsFixed(4)} '
        '(ema=${emaLoss.toStringAsFixed(4)}) | '
        'acc=${(acc * 100).toStringAsFixed(1)}% | '
        'len=${input.length}',
      );
    }

    if (step % sampleEvery == 0 || step == numSteps) {
      final sampled = _sampleGreedy(agent, tok, 12, blockSize);
      print('    sample(<start> ->): ${sampled.join(" ")}');
    }
  }

  // ---- 4. Cleanup --------------------------------------------------------
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
