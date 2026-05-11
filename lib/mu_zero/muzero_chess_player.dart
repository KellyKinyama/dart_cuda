// file: lib/mu_zero/muzero_chess_player.dart
//
// Standalone MuZero chess player using the DeepSeek MoE + AFT decoder.
//
// What it does:
//   1. Builds a compact move vocabulary from a slice of bundled PGN games.
//   2. Trains a `ChessMuZeroAgent` on those games with alternating
//      POLICY / DYNAMICS objectives (same recipe as
//      `lib/linformer/chess_gpt_example.dart`).
//   3. Plays a self-play game using legal-move-masked argmax on the
//      trained policy head.
//
// This file is intentionally self-contained so it can be invoked directly
// as `dart lib/mu_zero/muzero_chess_player.dart` without depending on
// `lib/linformer/chess_gpt_example.dart`. The two files share the same
// architecture but this one lives next to the other mu_zero examples.

import 'dart:math' as math;

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/dataset/dataset.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/mu_zero/deepseek_aft_decoder.dart';

// ---------------------------------------------------------------------------
// Move tokenizer over moves actually seen in training. Keeping the vocab
// small (vs. a 64*64+2 from-to encoding) is critical for stable training.
// ---------------------------------------------------------------------------

class MoveTokenizer {
  final Map<String, int> stoi;
  final Map<int, String> itos;
  final int startId;
  final int endId;

  MoveTokenizer._(this.stoi, this.itos, this.startId, this.endId);

  factory MoveTokenizer.fromGames(List<List<String>> games) {
    final stoi = <String, int>{'<start>': 0, '<end>': 1};
    int next = 2;
    for (final g in games) {
      for (final mv in g) {
        stoi.putIfAbsent(mv, () => next++);
      }
    }
    final itos = stoi.map((k, v) => MapEntry(v, k));
    return MoveTokenizer._(stoi, itos, stoi['<start>']!, stoi['<end>']!);
  }

  int get vocabSize => stoi.length;
  int? encode(String uci) => stoi[uci];
  String decode(int id) => itos[id] ?? '<unk>';
}

// ---------------------------------------------------------------------------
// MuZero agent. The board encoder is intentionally left out: the AFT
// cross-attention on this engine is unstable with non-trivial encoder
// contexts, so we feed a 1-row zero tensor (matches the working
// `example_deepseek_muzero_shakespeare.dart` pattern).
// ---------------------------------------------------------------------------

class ChessMuZeroAgent {
  final DeepSeekAFTDecoder model;
  final Tensor dummyEnc;
  final double actionScale;

  ChessMuZeroAgent(this.model, {this.actionScale = 5.0})
    : dummyEnc = Tensor.zeros([1, model.encoderEmbedSize]);

  /// h(x) : decoder forward through final LayerNorm. Returns [T, D].
  Tensor representation(List<int> moveIds, List<Tensor> tracker) {
    Tensor x = Tensor.embeddings(moveIds, model.wte, model.wpe);
    tracker.add(x);
    for (final block in model.blocks) {
      x = block.forward(x, dummyEnc, tracker);
    }
    return model.finalLayerNorm.forward(x, tracker);
  }

  /// g(s_t, a_t) : (state-row, action) -> next latent state.
  Tensor dynamics(
    Tensor stateRow,
    int action,
    int step,
    List<Tensor> tracker,
  ) {
    final actionEmb = model.wte.getRow(action);
    final posEmb = model.wpe.getRow(step % model.blockSize);
    final strongAction = actionEmb * actionScale;
    final combined = stateRow + strongAction + posEmb;
    final nextRaw = model.blocks[0].forward(combined, dummyEnc, tracker);
    final next = model.finalLayerNorm.forward(nextRaw, tracker);
    tracker.addAll([actionEmb, strongAction, posEmb, combined]);
    return next;
  }

  /// f(s) : latent state -> move logits.
  Tensor predictPolicy(Tensor state, List<Tensor> tracker) {
    return model.lmHead.forward(state, tracker);
  }

  List<Tensor> parameters() => model.parameters();
}

// ---------------------------------------------------------------------------
// Trajectory & helpers.
// ---------------------------------------------------------------------------

class GameTrajectory {
  final List<int> moveIds; // [<start>, m1, ..., mN, <end>]
  GameTrajectory(this.moveIds);
}

GameTrajectory buildTrajectory(
  List<String> uciMoves,
  int blockSize,
  MoveTokenizer tok,
) {
  final game = Game();
  final moveIds = <int>[tok.startId];
  for (final mv in uciMoves) {
    final id = tok.encode(mv);
    if (id == null) continue;
    moveIds.add(id);
    final m = game.getMove(mv);
    if (m == null) break;
    game.makeMove(m);
    if (moveIds.length >= blockSize) break;
  }
  if (moveIds.length < blockSize) moveIds.add(tok.endId);
  return GameTrajectory(moveIds);
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

double warmupCosineLR(
  int step, {
  required double baseLR,
  required int warmupSteps,
  required int totalSteps,
  required double minLR,
}) {
  if (step < warmupSteps) return baseLR * (step + 1) / warmupSteps;
  final progress = (step - warmupSteps) / math.max(1, totalSteps - warmupSteps);
  final cosine = 0.5 * (1 + math.cos(math.pi * progress.clamp(0.0, 1.0)));
  return minLR + (baseLR - minLR) * cosine;
}

// ---------------------------------------------------------------------------
// Inference: legal-move masked argmax over the current policy head.
// ---------------------------------------------------------------------------

(String, Move)? pickNextMove(
  ChessMuZeroAgent agent,
  MoveTokenizer tok,
  Game game,
  List<int> history,
  int blockSize,
) {
  final tracker = <Tensor>[];
  final ctx = [tok.startId, ...history];
  final trimmed = ctx.length <= blockSize
      ? ctx
      : ctx.sublist(ctx.length - blockSize);

  final states = agent.representation(trimmed, tracker);
  final logits = agent.predictPolicy(states, tracker);
  final lastRow = logits.fetchRow(trimmed.length - 1);

  final legal = game.generateLegalMoves();
  if (legal.isEmpty) {
    _safeCleanup(tracker, agent.parameters());
    return null;
  }

  Move? bestMove;
  String bestUci = '';
  double bestScore = -double.infinity;
  for (final m in legal) {
    final alg = game.toAlgebraic(m);
    final id = tok.encode(alg);
    if (id == null) continue;
    final s = lastRow[id];
    if (s > bestScore) {
      bestScore = s;
      bestMove = m;
      bestUci = alg;
    }
  }

  _safeCleanup(tracker, agent.parameters());
  if (bestMove == null) return null;
  return (bestUci, bestMove);
}

// ---------------------------------------------------------------------------
// Main: train then play.
// ---------------------------------------------------------------------------

Future<void> main() async {
  print('--- MuZero Chess Player (DeepSeek MoE + AFT) ---');

  // Hyperparameters.
  const int embedSize = 64;
  const int blockSize = 32;
  const int numLayers = 2;
  const int numHeads = 4;
  const int numRoutedExperts = 4;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 64;

  final int numGames = math.min(8, dataset.length);
  final games = dataset.take(numGames).toList();
  final tok = MoveTokenizer.fromGames(games);
  print('Move vocab: ${tok.vocabSize}');

  final model = DeepSeekAFTDecoder(
    vocabSize: tok.vocabSize,
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
  final agent = ChessMuZeroAgent(model);
  print(
    'Model params (tensors): ${agent.parameters().length}  '
    'routed=$numRoutedExperts top-$topK shared=$numSharedExperts',
  );

  final trajectories = games
      .map((g) => buildTrajectory(g, blockSize, tok))
      .toList();

  // Training schedule.
  const double baseLR = 1e-3;
  const double minLR = 1e-4;
  const int numSteps = 600;
  const int balanceEvery = 50;
  const int logEvery = 100;
  const int unrollSteps = 3;
  const int dynamicsEvery = 2;
  final int warmupSteps = (0.05 * numSteps).round();

  final optimizer = Adam(agent.parameters(), lr: baseLR);
  final rng = math.Random(7);

  double pLossSum = 0.0;
  double dLossSum = 0.0;
  int pN = 0;
  int dN = 0;

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

    final traj = trajectories[rng.nextInt(trajectories.length)];
    final T = traj.moveIds.length;
    if (T < 2) continue;

    final int seqEnd = math.min(T, blockSize);
    final inputs = traj.moveIds.sublist(0, seqEnd - 1);
    final targets = traj.moveIds.sublist(1, seqEnd);

    final bool dynamicsMode = (step + 1) % dynamicsEvery == 0;

    if (!dynamicsMode) {
      final logits = model.forward(inputs, agent.dummyEnc, tracker);
      final loss = logits.crossEntropy(targets);
      tracker.add(loss);
      final lossVal = loss.fetchData()[0];
      if (lossVal.isNaN || lossVal.isInfinite) {
        _safeCleanup(tracker, agent.parameters());
        continue;
      }
      loss.backward();
      pLossSum += lossVal;
      pN++;
    } else {
      final int maxStart = inputs.length - unrollSteps - 1;
      final int start = maxStart > 0 ? rng.nextInt(maxStart) : 0;

      final prefix = inputs.sublist(0, start + 1);
      final s0Full = agent.representation(prefix, tracker);
      Tensor stateRow = s0Full.getRow(s0Full.shape[0] - 1);
      tracker.add(stateRow);

      final logits0 = agent.predictPolicy(stateRow, tracker);
      Tensor loss = logits0.crossEntropy([targets[start]]);
      tracker.add(loss);

      for (int k = 0; k < unrollSteps; k++) {
        final int idx = start + k;
        if (idx + 1 >= targets.length) break;
        final aTaken = inputs[idx + 1];
        final next = agent.dynamics(stateRow, aTaken, idx + 1, tracker);
        final logitsK = agent.predictPolicy(next, tracker);
        final lossK = logitsK.crossEntropy([targets[idx + 1]]);
        tracker.add(lossK);
        loss = loss + lossK;
        tracker.add(loss);
        stateRow = next;
      }

      final lossVal = loss.fetchData()[0];
      if (lossVal.isNaN || lossVal.isInfinite) {
        _safeCleanup(tracker, agent.parameters());
        continue;
      }
      loss.backward();
      dLossSum += lossVal;
      dN++;
    }

    optimizer.step();
    _safeCleanup(tracker, agent.parameters());

    if ((step + 1) % balanceEvery == 0) {
      final loadSnap = [
        for (final b in model.blocks) List<int>.from(b.moe.expertLoad),
      ];
      model.updateRoutingBias();
      if ((step + 1) % logEvery == 0) {
        final pAvg = pN > 0 ? (pLossSum / pN).toStringAsFixed(4) : '   -  ';
        final dAvg = dN > 0 ? (dLossSum / dN).toStringAsFixed(4) : '   -  ';
        print(
          'step ${step + 1}  '
          'lr=${optimizer.lr.toStringAsFixed(5)}  '
          'policy_loss=$pAvg  dyn_loss=$dAvg',
        );
        for (int li = 0; li < loadSnap.length; li++) {
          print('    layer $li expert load: ${loadSnap[li]}');
        }
        pLossSum = 0.0;
        dLossSum = 0.0;
        pN = 0;
        dN = 0;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Self-play.
  // -----------------------------------------------------------------------
  print('\n--- Self-play (greedy, legal-move masked) ---');
  final game = Game();
  final history = <int>[];
  for (int ply = 0; ply < 20; ply++) {
    final picked = pickNextMove(agent, tok, game, history, blockSize);
    if (picked == null) {
      print('terminal: no in-vocab legal moves');
      break;
    }
    final (mv, move) = picked;
    print('ply ${ply + 1}: $mv');
    game.makeMove(move);
    final id = tok.encode(mv);
    if (id == null) break;
    history.add(id);
    if (game.gameOver) {
      print('Game over: ${game.result?.readable ?? "?"}');
      break;
    }
  }
  print('\nFinal FEN: ${game.fen}');
}
