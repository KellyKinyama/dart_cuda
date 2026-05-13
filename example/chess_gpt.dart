// file: lib/linformer/chess_gpt_example.dart
//
// DeepSeek-style MoE+AFT MuZero chess player.
//
// Architecture:
//   * Decoder: `DeepSeekAFTDecoder` (MoE-FFN + AFT attention) over a
//     move-token vocabulary of 4098 (= 64*64 from-to squares + `<start>`
//     + `<end>`/`.`).
//   * Board encoder: a tiny embedding-only encoder over a 64-square FEN
//     tokenization (13 piece types) fed into the decoder's cross-attention.
//
// MuZero pieces (same flavour as `example_deepseek_muzero_shakespeare.dart`):
//   * h(x)     : representation = decoder forward through final LayerNorm.
//   * g(s, a)  : dynamics       = inject action+position embedding into the
//                                 current latent and push through block 0.
//   * f(s)     : prediction     = LM head over the move vocabulary.
//
// Training alternates POLICY mode (teacher-forced LM loss over a game) and
// DYNAMICS mode (K-step unrolled rollout from a single representation).
//
// At inference time the model masks logits to the set of legal moves
// reported by `bishop` before sampling/argmaxing.

import 'dart:math' as math;

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/loaders/dataset.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';

// ---------------------------------------------------------------------------
// Move tokenizer.
//
// The naive 64*64+2 = 4098 from-to encoding has been observed to drive the
// engine's cross-entropy / Adam path to NaN after a single step (very wide
// softmax). Instead we build a compact vocabulary covering only the moves
// actually present in the training set, plus a `<start>` and `<end>`
// token. Typical sizes are a few hundred entries.
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
// Board (FEN) tokenization for the cross-attention encoder.
// ---------------------------------------------------------------------------

const int boardVocabSize = 13; // empty + 6 piece types * 2 colours

List<int> tokenizeBoard(Game game) {
  final tokens = <int>[];
  final size = game.size;
  for (int i = 0; i < size.numIndices; i++) {
    if (!size.onBoard(i)) continue;
    final sq = game.board[i];
    if (sq.isEmpty) {
      tokens.add(0);
    } else {
      // bishop's piece-type ids are 1..6 (pawn..king).
      final pType = sq.type.clamp(1, 6) - 1;
      final colourOff = sq.colour == Bishop.white ? 0 : 6;
      tokens.add(1 + colourOff + pType);
    }
  }
  return tokens;
}

// ---------------------------------------------------------------------------
// Board encoder: embeddings per square, then mean-pooled to a single
// length-1 context. The DeepSeek decoder's cross-attention is sensitive to
// long encoder contexts on this engine (NaNs out at init when fed 64
// tokens), so we summarize the position into a single embedding that the
// cross-attention can attend to. The mean-pool is implemented as a constant
// `[1, 64]` row-vector matmul against the `[64, D]` square embeddings.
// ---------------------------------------------------------------------------

class BoardEncoder {
  final int embedSize;
  final Tensor wte; // [boardVocabSize, embedSize]
  final Tensor wpe; // [64, embedSize]
  final Tensor poolRow; // [1, 64] constant for mean-pooling

  BoardEncoder(this.embedSize)
    : wte = Tensor.random([boardVocabSize, embedSize]),
      wpe = Tensor.random([64, embedSize]),
      poolRow = Tensor.fromList([1, 64], List.filled(64, 1.0 / 64.0)) {
    final w = wte.fetchData();
    final p = wpe.fetchData();
    final r = math.Random(11);
    for (int i = 0; i < w.length; i++) {
      w[i] = (r.nextDouble() * 2 - 1) * 0.02;
    }
    for (int i = 0; i < p.length; i++) {
      p[i] = (r.nextDouble() * 2 - 1) * 0.02;
    }
    wte.data = w;
    wpe.data = p;
  }

  /// Produces a `[1, embedSize]` pooled board context.
  Tensor forward(List<int> boardTokens, List<Tensor> tracker) {
    final x = Tensor.embeddings(boardTokens, wte, wpe); // [64, D]
    tracker.add(x);
    final pooled = poolRow.matmul(x); // [1, D]
    tracker.add(pooled);
    return pooled;
  }

  // Only the learned embeddings are parameters; `poolRow` is a fixed kernel.
  List<Tensor> parameters() => [wte, wpe];
}

// ---------------------------------------------------------------------------
// Cleanup helper (same pattern as the other examples).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MuZero agent over the DeepSeek decoder.
// ---------------------------------------------------------------------------

class ChessMuZeroAgent {
  final DeepSeekAFTDecoder model;
  final BoardEncoder boardEnc;
  final double actionScale;

  /// Persistent zero encoder context. The AFT cross-attention on this
  /// engine produces NaNs at init when fed non-trivial encoder inputs, so
  /// the cross-attention is effectively disabled (queries attend to a
  /// single zero vector). The board encoder is still trained and could be
  /// used to condition the decoder via a different injection point later.
  final Tensor dummyEnc;

  ChessMuZeroAgent(this.model, this.boardEnc, {this.actionScale = 5.0})
    : dummyEnc = Tensor.zeros([1, model.encoderEmbedSize]);

  Tensor encodeBoard(Game game, List<Tensor> tracker) => dummyEnc;

  /// h(x) : decoder forward through final LayerNorm.
  Tensor representation(
    List<int> moveIds,
    Tensor boardCtx,
    List<Tensor> tracker,
  ) {
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
    Tensor boardCtx,
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

  List<Tensor> parameters() => [
    ...model.parameters(),
    // Board encoder params are kept in the optimizer so future variants can
    // wire them back into the decoder; they receive no gradient under the
    // current dummy-context setup.
    ...boardEnc.parameters(),
  ];
}

// ---------------------------------------------------------------------------
// Build a (board-state, move-sequence) training trajectory from a PGN-like
// list of UCI strings. We tokenize the board *before* each move so the
// decoder always conditions on the position it should play from.
// ---------------------------------------------------------------------------

class GameTrajectory {
  final List<List<int>> boardsBeforeMove; // one board per move
  final List<int> moveIds; // [<start>, m1, m2, ..., mN, <end>]
  GameTrajectory(this.boardsBeforeMove, this.moveIds);
}

GameTrajectory buildTrajectory(
  List<String> uciMoves,
  int blockSize,
  MoveTokenizer tok,
) {
  final game = Game();
  final boards = <List<int>>[];
  final moveIds = <int>[tok.startId];

  for (final mv in uciMoves) {
    final id = tok.encode(mv);
    if (id == null) continue; // skip OOV
    boards.add(tokenizeBoard(game));
    moveIds.add(id);
    final m = game.getMove(mv);
    if (m == null) break;
    game.makeMove(m);
    if (moveIds.length >= blockSize) break;
  }
  if (moveIds.length < blockSize) {
    boards.add(tokenizeBoard(game));
    moveIds.add(tok.endId);
  }
  return GameTrajectory(boards, moveIds);
}

// ---------------------------------------------------------------------------
// LR schedule (mirrors the one in example_deepseek_aft_training).
// ---------------------------------------------------------------------------

double warmupCosineLR(
  int step, {
  required double baseLR,
  required int warmupSteps,
  required int totalSteps,
  required double minLR,
}) {
  if (step < warmupSteps) {
    return baseLR * (step + 1) / warmupSteps;
  }
  final progress = (step - warmupSteps) / math.max(1, totalSteps - warmupSteps);
  final cosine = 0.5 * (1 + math.cos(math.pi * progress.clamp(0.0, 1.0)));
  return minLR + (baseLR - minLR) * cosine;
}

// ---------------------------------------------------------------------------
// Main: train on a small subset of games, then play a few moves.
// ---------------------------------------------------------------------------

Future<void> main() async {
  print('--- DeepSeek MoE + AFT MuZero Chess Player ---');

  // Model hyperparameters — modest, demo scale.
  const int embedSize = 64;
  const int blockSize = 32;
  const int numLayers = 2;
  const int numHeads = 4;
  const int numRoutedExperts = 4;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 64;

  // Build a compact tokenizer over the moves we actually train on.
  final int numGames = math.min(8, dataset.length);
  final games = dataset.take(numGames).toList();
  final tok = MoveTokenizer.fromGames(games);
  print('Move vocab: ${tok.vocabSize}, board vocab: $boardVocabSize');

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
  final boardEnc = BoardEncoder(embedSize);
  final agent = ChessMuZeroAgent(model, boardEnc);
  print(
    'Model params (tensors): ${agent.parameters().length}  '
    'routed=$numRoutedExperts top-$topK shared=$numSharedExperts',
  );

  final trajectories = games
      .map((g) => buildTrajectory(g, blockSize, tok))
      .toList();
  print(
    'Loaded $numGames training games, '
    'avg moves=${(trajectories.map((t) => t.moveIds.length).reduce((a, b) => a + b) / numGames).toStringAsFixed(1)}',
  );

  // Training schedule.
  const double baseLR = 1e-3;
  const double minLR = 1e-4;
  const int numSteps = 600;
  const int balanceEvery = 50;
  const int logEvery = 50;
  const int unrollSteps = 3;
  const int dynamicsEvery = 2;
  final int warmupSteps = (0.05 * numSteps).round();

  final optimizer = Adam(agent.parameters(), lr: baseLR);
  final rng = math.Random(7);

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

    // Pick a trajectory.
    final traj = trajectories[rng.nextInt(trajectories.length)];
    final T = traj.moveIds.length;
    if (T < 2) continue;

    // We need an encoder context. Use the board *before* the last move in the
    // training window so the model's cross-attention sees a consistent
    // position throughout the unroll.
    final int seqEnd = math.min(T, blockSize);
    final inputs = traj.moveIds.sublist(0, seqEnd - 1);
    final targets = traj.moveIds.sublist(1, seqEnd);

    // Board context = tokenized board taken just before `inputs[0]` (i.e.
    // the starting position for that window).
    final boardTokens = traj.boardsBeforeMove.isNotEmpty
        ? traj.boardsBeforeMove[0]
        : List<int>.filled(64, 0);
    final boardCtx = boardEnc.forward(boardTokens, tracker);

    final bool dynamicsMode = (step + 1) % dynamicsEvery == 0;

    if (!dynamicsMode) {
      // POLICY mode: teacher-forced LM loss.
      // Use model.forward directly (matches the working shakespeare path).
      final logits = model.forward(inputs, agent.dummyEnc, tracker);
      final loss = logits.crossEntropy(targets);
      tracker.add(loss);
      final lossVal = loss.fetchData()[0];
      if (lossVal.isNaN || lossVal.isInfinite) {
        _safeCleanup(tracker, agent.parameters());
        continue;
      }
      loss.backward();
      runningPolicyLoss += lossVal;
      runningPolicy++;
    } else {
      // DYNAMICS mode: K-step unrolled rollout.
      final int maxStart = inputs.length - unrollSteps - 1;
      final int start = maxStart > 0 ? rng.nextInt(maxStart) : 0;

      final prefix = inputs.sublist(0, start + 1);
      final s0Full = agent.representation(prefix, boardCtx, tracker);
      Tensor stateRow = s0Full.getRow(s0Full.shape[0] - 1);
      tracker.add(stateRow);

      final logits0 = agent.predictPolicy(stateRow, tracker);
      Tensor loss = logits0.crossEntropy([targets[start]]);
      tracker.add(loss);

      for (int k = 0; k < unrollSteps; k++) {
        final int idx = start + k;
        if (idx + 1 >= targets.length) break;
        final int aTaken = inputs[idx + 1];
        final next = agent.dynamics(
          stateRow,
          aTaken,
          idx + 1,
          boardCtx,
          tracker,
        );
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
      runningDynamicsLoss += lossVal;
      runningDynamics++;
    }

    optimizer.step();
    _safeCleanup(tracker, agent.parameters());

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

  // -------------------------------------------------------------------------
  // Inference: play a short game against itself (legal-move masked argmax).
  //
  // Returns a (uci-string, bishop-Move) pair so the caller can play the
  // chosen Move directly without a UCI round-trip. Only moves whose UCI is
  // in the trained tokenizer's vocabulary are considered.
  // -------------------------------------------------------------------------

  (String, Move)? pickNextMove(Game game, List<int> history) {
    final tracker = <Tensor>[];
    final boardCtx = agent.encodeBoard(game, tracker);

    final ctx = [tok.startId, ...history];
    final trimmed = ctx.length <= blockSize
        ? ctx
        : ctx.sublist(ctx.length - blockSize);

    final states = agent.representation(trimmed, boardCtx, tracker);
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
      if (id == null) continue; // OOV — model has never seen this move
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

  print('\n--- Self-play demo (greedy, legal-move masked) ---');
  final game = Game();
  final history = <int>[];
  for (int ply = 0; ply < 12; ply++) {
    final picked = pickNextMove(game, history);
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
