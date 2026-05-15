// tool/muzero_alphazero_train.dart
//
// AlphaZero-style self-play trainer for the MuZero (DeepSeek-backbone)
// chess model. No external engine: both sides are played by the model
// itself, using PUCT MCTS (`ZobristMcts`) with the model's own policy
// priors and value head. Training targets:
//
//   * Policy: cross-entropy against the action sampled from MCTS visit
//     counts (hard-target proxy for the visit distribution — matches the
//     existing `Tensor.crossEntropy(List<int>)` API).
//   * Value : MSE against the final game outcome z (from the side-to-
//     move's POV at that step). Wins/losses are ±1, draws and maxply
//     cutoffs are 0.
//
// Pipeline per iteration:
//   1. Play [gamesPerIter] self-play games. Each move chosen by MCTS
//      (sampled from visits with temperature for the first --temp-moves
//      plies, then greedy).
//   2. Collect (history-prefix, sampled-move-id, mover-is-white) per step.
//   3. Train MuZero for [trainEpochsPerIter] epochs over the freshly
//      collected pairs (Adam + warmup-cosine LR).
//
// Usage:
//   dart run example/tool/muzero_alphazero_train.dart \
//     [--iters=N] [--games=N] [--epochs=N] [--maxply=N] \
//     [--mcts-sims=N] [--cpuct=F] [--temperature=F] [--temp-moves=N] \
//     [--lr=F] [--value-weight=F] [--seed=N] \
//     [--load=PATH] [--save=PATH] [--save-every=N]

import 'dart:io';
import 'dart:math' as math;

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/loaders/dataset.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';
import 'package:dart_cuda/core/layers/nn.dart' show Module;
import 'package:dart_cuda/core/utils/persistence.dart';
import 'package:dart_cuda/core/models/mu_zero/muzero_chess_player.dart'
    show ChessMuZeroAgent, MoveTokenizer, warmupCosineLR;
import 'package:dart_cuda/core/models/mu_zero/muzero_chess_mcts.dart'
    show ZobristMcts, MctsNode;

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------
class TrainCfg {
  int numIterations = 3;
  int gamesPerIter = 2;
  int trainEpochsPerIter = 2;
  int maxPlies = 40;

  // Self-play search.
  int mctsSims = 32;
  double cPuct = 1.4;
  double temperature = 1.0; // sampling temperature for the early plies
  int tempMoves = 15; // after this many plies, switch to greedy

  // Optimization.
  double baseLR = 1e-3;
  double minLR = 1e-4;
  double valueWeight = 0.5;

  int? seed;

  String? loadPath;
  String? savePath;
  int saveEvery = 1;

  // Live game printing.
  bool showMoves = false;     // print each move as it's played
  bool showBoard = false;     // also print ASCII board after each move
}

// ---------------------------------------------------------------------------
// Trajectory record
// ---------------------------------------------------------------------------
class _Step {
  final List<int> historyIds; // tokens BEFORE this move (incl. <start>)
  final int targetMoveId;
  final bool moverIsWhite;
  _Step(this.historyIds, this.targetMoveId, this.moverIsWhite);
}

class _Trajectory {
  final List<_Step> steps;
  final double finalResultForWhite; // +1, 0, -1
  final bool finished;
  _Trajectory(this.steps, this.finalResultForWhite, this.finished);
}

class _TrainPair {
  final _Step step;
  final _Trajectory trajectory;
  _TrainPair(this.step, this.trajectory);
}

class _AgentModule extends Module {
  final ChessMuZeroAgent agent;
  _AgentModule(this.agent);
  @override
  List<Tensor> parameters() => agent.parameters();
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

// ---------------------------------------------------------------------------
// Pick a move from the MCTS root via visit-count sampling (or greedy).
// Returns null if no legal-and-in-vocab move exists.
// ---------------------------------------------------------------------------
({String uci, int id, Move move})? _sampleFromRoot(
  MctsNode root,
  MoveTokenizer tok,
  math.Random rng, {
  required double temperature,
}) {
  if (root.legalMoves.isEmpty) return null;

  // Restrict to actions whose UCI is in the tokenizer vocab.
  final ids = <int>[];
  final inVocabIdx = <int>[];
  for (int i = 0; i < root.legalMoves.length; i++) {
    final id = tok.encode(root.legalUci[i]);
    if (id == null) continue;
    inVocabIdx.add(i);
    ids.add(id);
  }
  if (inVocabIdx.isEmpty) return null;

  int pickI;
  if (temperature <= 0.0) {
    // Greedy by visit count, prior as tiebreaker.
    pickI = inVocabIdx.first;
    double bestScore = -double.infinity;
    for (final i in inVocabIdx) {
      final s = 10.0 * root.visits[i] + root.priors[i];
      if (s > bestScore) {
        bestScore = s;
        pickI = i;
      }
    }
  } else {
    final invT = 1.0 / temperature;
    final weights = <double>[
      for (final i in inVocabIdx)
        math.pow(root.visits[i] + 1e-9, invT).toDouble(),
    ];
    final total = weights.fold<double>(0.0, (a, b) => a + b);
    if (total <= 0) {
      pickI = inVocabIdx.first;
    } else {
      final pick = rng.nextDouble() * total;
      double acc = 0.0;
      pickI = inVocabIdx.last;
      for (int j = 0; j < weights.length; j++) {
        acc += weights[j];
        if (acc >= pick) {
          pickI = inVocabIdx[j];
          break;
        }
      }
    }
  }

  final uci = root.legalUci[pickI];
  final mv = root.legalMoves[pickI];
  final id = tok.encode(uci)!;
  return (uci: uci, id: id, move: mv);
}

// ---------------------------------------------------------------------------
// Play one self-play game (both sides = MCTS), return the trajectory.
// ---------------------------------------------------------------------------
_Trajectory _playOneGame({
  required ChessMuZeroAgent agent,
  required MoveTokenizer tok,
  required int blockSize,
  required TrainCfg cfg,
  required math.Random rng,
}) {
  final game = Game(variant: Variant.standard());
  final tokHistory = <int>[tok.startId];
  final steps = <_Step>[];

  if (cfg.showMoves) stdout.writeln();

  int ply = 0;
  while (!game.gameOver && ply < cfg.maxPlies) {
    final mover = game.state.turn;

    // Fresh MCTS per move keeps memory bounded; transposition reuse
    // within a single search is what gives us value, anyway.
    final mcts = ZobristMcts(
      agent,
      tok,
      blockSize: blockSize,
      cPuct: cfg.cPuct,
      rng: rng,
    );
    // ZobristMcts expects history WITHOUT the leading <start> token.
    final root = mcts.run(
      rootGame: game,
      history: tokHistory.sublist(1),
      numSimulations: cfg.mctsSims,
    );

    final temperature = ply < cfg.tempMoves ? cfg.temperature : 0.0;
    final pick = _sampleFromRoot(root, tok, rng, temperature: temperature);
    if (pick == null) break;

    steps.add(
      _Step(List<int>.from(tokHistory), pick.id, mover == Bishop.white),
    );

    game.makeMove(pick.move);
    tokHistory.add(pick.id);
    ply++;

    if (cfg.showMoves) {
      // Look up the chosen move's visit count and prior in the root.
      int visits = 0;
      double prior = 0.0;
      for (int i = 0; i < root.legalMoves.length; i++) {
        if (root.legalUci[i] == pick.uci) {
          visits = root.visits[i];
          prior = root.priors[i];
          break;
        }
      }
      final side = mover == Bishop.white ? 'W' : 'B';
      final moveNum = ((ply + 1) / 2).floor();
      stdout.writeln(
        '    ${moveNum.toString().padLeft(3)}.$side ${pick.uci.padRight(6)} '
        '(visits=$visits/${cfg.mctsSims}, '
        'prior=${prior.toStringAsFixed(3)})',
      );
      if (cfg.showBoard) {
        stdout.writeln(game.ascii());
      }
    }

    mcts.clear();
  }

  if (cfg.showMoves) {
    stdout.writeln('    fen: ${game.fen}');
  }

  double resultWhite;
  final finished = game.gameOver;
  if (game.checkmate) {
    // Side to move is mated → opponent (just moved) won.
    resultWhite = game.state.turn == Bishop.white ? -1.0 : 1.0;
  } else {
    // Stalemate, draw rule, or unfinished by maxply → 0.
    resultWhite = 0.0;
  }
  return _Trajectory(steps, resultWhite, finished);
}

// ---------------------------------------------------------------------------
// One training step: policy CE + value MSE on the post-move state.
// ---------------------------------------------------------------------------
({double policy, double value}) _trainStep({
  required ChessMuZeroAgent agent,
  required Adam optimizer,
  required List<int> history,
  required int targetId,
  required double valueTarget,
  required int blockSize,
  required double lr,
  required double valueWeight,
}) {
  optimizer.lr = lr;
  optimizer.zeroGrad();
  final tracker = <Tensor>[];

  final ctx = history.length <= blockSize
      ? history
      : history.sublist(history.length - blockSize);
  if (ctx.isEmpty) {
    _safeCleanup(tracker, agent.parameters());
    return (policy: double.nan, value: double.nan);
  }

  final states = agent.representation(ctx, tracker);
  final logits = agent.predictPolicy(states, tracker);
  final values = agent.predictValue(states, tracker); // [T, 1]
  final targets = <int>[for (int i = 1; i < ctx.length; i++) ctx[i], targetId];

  final pLoss = logits.crossEntropy(targets);
  tracker.add(pLoss);

  final lastValueRow = values.getRow(ctx.length - 1); // [1]
  final target = Tensor.fill(lastValueRow.shape, valueTarget);
  final diff = lastValueRow - target;
  final vLoss = diff.pow(2).mean();
  tracker.addAll([lastValueRow, target, diff, vLoss]);

  final loss = pLoss + (vLoss * valueWeight);
  tracker.add(loss);

  final pVal = pLoss.fetchData()[0];
  final vVal = vLoss.fetchData()[0];
  if (pVal.isNaN || pVal.isInfinite || vVal.isNaN || vVal.isInfinite) {
    _safeCleanup(tracker, agent.parameters());
    return (policy: double.nan, value: double.nan);
  }
  loss.backward();
  optimizer.step();
  _safeCleanup(tracker, agent.parameters());
  return (policy: pVal, value: vVal);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
Future<int> main(List<String> args) async {
  final cfg = TrainCfg();
  for (final a in args) {
    if (a == '--help' || a == '-h') {
      stdout.writeln(
        'Usage: dart run example/tool/muzero_alphazero_train.dart '
        '[--iters=N] [--games=N] [--epochs=N] [--maxply=N] '
        '[--mcts-sims=N] [--cpuct=F] [--temperature=F] [--temp-moves=N] '
        '[--lr=F] [--value-weight=F] [--seed=N] '
        '[--load=PATH] [--save=PATH] [--save-every=N] '
        '[--show-moves] [--show-board]',
      );
      return 0;
    } else if (a.startsWith('--iters=')) {
      cfg.numIterations = int.parse(a.substring('--iters='.length));
    } else if (a.startsWith('--games=')) {
      cfg.gamesPerIter = int.parse(a.substring('--games='.length));
    } else if (a.startsWith('--epochs=')) {
      cfg.trainEpochsPerIter = int.parse(a.substring('--epochs='.length));
    } else if (a.startsWith('--maxply=')) {
      cfg.maxPlies = int.parse(a.substring('--maxply='.length));
    } else if (a.startsWith('--mcts-sims=')) {
      cfg.mctsSims = int.parse(a.substring('--mcts-sims='.length));
    } else if (a.startsWith('--cpuct=')) {
      cfg.cPuct = double.parse(a.substring('--cpuct='.length));
    } else if (a.startsWith('--temperature=')) {
      cfg.temperature = double.parse(a.substring('--temperature='.length));
    } else if (a.startsWith('--temp-moves=')) {
      cfg.tempMoves = int.parse(a.substring('--temp-moves='.length));
    } else if (a.startsWith('--lr=')) {
      cfg.baseLR = double.parse(a.substring('--lr='.length));
    } else if (a.startsWith('--value-weight=')) {
      cfg.valueWeight = double.parse(a.substring('--value-weight='.length));
    } else if (a.startsWith('--seed=')) {
      cfg.seed = int.parse(a.substring('--seed='.length));
    } else if (a.startsWith('--load=')) {
      cfg.loadPath = a.substring('--load='.length);
    } else if (a.startsWith('--save=')) {
      cfg.savePath = a.substring('--save='.length);
    } else if (a.startsWith('--save-every=')) {
      cfg.saveEvery = int.parse(a.substring('--save-every='.length));
    } else if (a == '--show-moves') {
      cfg.showMoves = true;
    } else if (a == '--show-board') {
      cfg.showMoves = true;
      cfg.showBoard = true;
    } else {
      stderr.writeln('Unknown option: $a');
      return 64;
    }
  }

  stdout.writeln('--- MuZero AlphaZero self-play trainer ---');
  stdout.writeln(
    'iters=${cfg.numIterations}  '
    'games/iter=${cfg.gamesPerIter}  '
    'epochs/iter=${cfg.trainEpochsPerIter}  '
    'maxply=${cfg.maxPlies}',
  );
  stdout.writeln(
    'MCTS         : sims=${cfg.mctsSims}  cpuct=${cfg.cPuct}  '
    'temperature=${cfg.temperature} (first ${cfg.tempMoves} plies)',
  );
  if (cfg.loadPath != null) stdout.writeln('Load ckpt    : ${cfg.loadPath}');
  if (cfg.savePath != null) {
    stdout.writeln(
      'Save ckpt    : ${cfg.savePath} (every ${cfg.saveEvery} iter)',
    );
  }

  final rng = math.Random(cfg.seed ?? 42);

  // Hyperparams (kept consistent with `muzero_vs_stockfish_train.dart`).
  const int embedSize = 64;
  const int blockSize = 32;
  const int numLayers = 2;
  const int numHeads = 4;
  const int numRoutedExperts = 4;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 64;

  // Build vocab from bundled PGN dataset so the policy head has a
  // reasonable move table from step 0. (Vocab is not grown after this.)
  final int numGames = math.min(8, dataset.length);
  final games = dataset.take(numGames).toList();
  final tok = MoveTokenizer.fromGames(games);
  stdout.writeln('Move vocab   : ${tok.vocabSize}');

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
  if (cfg.loadPath != null) {
    if (!File(cfg.loadPath!).existsSync()) {
      stderr.writeln('Checkpoint not found: ${cfg.loadPath}');
      return 66;
    }
    await loadModuleBinary(_AgentModule(agent), cfg.loadPath!);
    stdout.writeln('Loaded checkpoint from ${cfg.loadPath}');
  }
  final optimizer = Adam(agent.parameters(), lr: cfg.baseLR);
  stdout.writeln('Model params : ${agent.parameters().length}');

  // Rough total-step counter for warmup-cosine LR.
  int globalStep = 0;
  final approxSteps =
      cfg.numIterations *
      cfg.gamesPerIter *
      cfg.maxPlies *
      cfg.trainEpochsPerIter;
  final warmup = math.max(1, (0.05 * approxSteps).round());

  double currentLR() => warmupCosineLR(
    globalStep,
    baseLR: cfg.baseLR,
    warmupSteps: warmup,
    totalSteps: math.max(approxSteps, warmup + 1),
    minLR: cfg.minLR,
  );

  final sw = Stopwatch()..start();
  int totalWhiteWins = 0;
  int totalBlackWins = 0;
  int totalDraws = 0;
  int totalUnfinished = 0;

  for (int iter = 0; iter < cfg.numIterations; iter++) {
    stdout.writeln('\n=== Iteration ${iter + 1}/${cfg.numIterations} ===');
    final trajectories = <_Trajectory>[];
    int iterW = 0, iterB = 0, iterD = 0, iterU = 0;

    for (int g = 0; g < cfg.gamesPerIter; g++) {
      stdout.write('  game ${g + 1}/${cfg.gamesPerIter} (self-play) ... ');
      final traj = _playOneGame(
        agent: agent,
        tok: tok,
        blockSize: blockSize,
        cfg: cfg,
        rng: rng,
      );
      trajectories.add(traj);
      String outcome;
      if (!traj.finished) {
        outcome = 'unfinished (maxply, z=0)';
        iterU++;
      } else if (traj.finalResultForWhite > 0.5) {
        outcome = 'White wins';
        iterW++;
      } else if (traj.finalResultForWhite < -0.5) {
        outcome = 'Black wins';
        iterB++;
      } else {
        outcome = 'draw';
        iterD++;
      }
      stdout.writeln('${traj.steps.length} plies, $outcome');
    }

    totalWhiteWins += iterW;
    totalBlackWins += iterB;
    totalDraws += iterD;
    totalUnfinished += iterU;
    stdout.writeln(
      '  iter games: W=$iterW  B=$iterB  D=$iterD  unfinished=$iterU',
    );

    model.updateRoutingBias();

    // Build a flat list of training pairs (every step is a training
    // example in AlphaZero — both colors are "the model").
    final pairs = <_TrainPair>[];
    for (final traj in trajectories) {
      for (final s in traj.steps) {
        pairs.add(_TrainPair(s, traj));
      }
    }
    if (pairs.isEmpty) {
      stdout.writeln('  no training pairs collected; skipping training');
      continue;
    }

    for (int epoch = 0; epoch < cfg.trainEpochsPerIter; epoch++) {
      pairs.shuffle(rng);
      double pLossSum = 0.0;
      double vLossSum = 0.0;
      int n = 0;
      for (final p in pairs) {
        final trajResultWhite = p.trajectory.finalResultForWhite;
        final valueTarget = p.step.moverIsWhite
            ? trajResultWhite
            : -trajResultWhite;
        final l = _trainStep(
          agent: agent,
          optimizer: optimizer,
          history: p.step.historyIds,
          targetId: p.step.targetMoveId,
          valueTarget: valueTarget,
          blockSize: blockSize,
          lr: currentLR(),
          valueWeight: cfg.valueWeight,
        );
        globalStep++;
        if (!l.policy.isNaN) {
          pLossSum += l.policy;
          vLossSum += l.value;
          n++;
        }
      }
      if (n > 0) {
        stdout.writeln(
          '  epoch ${epoch + 1}/${cfg.trainEpochsPerIter}  '
          'pairs=$n  '
          'lr=${currentLR().toStringAsFixed(5)}  '
          'policy=${(pLossSum / n).toStringAsFixed(4)}  '
          'value=${(vLossSum / n).toStringAsFixed(4)}',
        );
      }
    }

    if (cfg.savePath != null &&
        ((iter + 1) % cfg.saveEvery == 0 || iter == cfg.numIterations - 1)) {
      await saveModuleBinary(_AgentModule(agent), cfg.savePath!);
      stdout.writeln('  saved checkpoint -> ${cfg.savePath}');
    }
  }

  sw.stop();
  stdout.writeln('\n=== Self-play summary ===');
  stdout.writeln(
    'WhiteWins=$totalWhiteWins  BlackWins=$totalBlackWins  '
    'Draws=$totalDraws  unfinished=$totalUnfinished',
  );
  stdout.writeln(
    '\nTotal wall-clock: '
    '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(2)} s',
  );
  return 0;
}
