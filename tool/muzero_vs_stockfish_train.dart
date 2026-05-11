// tool/muzero_vs_stockfish_train.dart
//
// Trains the MuZero (DeepSeek-backbone) chess model by playing games
// against a real UCI engine (Stockfish, by default) and distilling its
// moves into the policy head.
//
// NOTE — This is the policy-only port of the original `muzero_vs_stockfish_
// train.dart` from the dart_torch codebase. The dart_cuda model used here
// has no value or reward head and no MCTS implementation, so:
//   * MuZero moves are picked by legal-move-masked argmax over the policy
//     head (same as `muzero_chess_player.dart`).
//   * The training signal is cross-entropy on Stockfish's chosen move for
//     SF-to-move states. MuZero's own moves are also distilled against the
//     model's argmax (i.e. self-confirmation) — kept for trajectory shape.
//   * Final-result value targets are recorded into the trajectory but not
//     used yet (placeholder for when a value head is added).
//
// Pipeline per iteration:
//   1. Play [gamesPerIter] full games vs. Stockfish. MuZero alternates
//      White/Black across games.
//   2. Collect (history-prefix, target-move-id) pairs from every step.
//   3. Train MuZero for [trainEpochsPerIter] epochs over the freshly
//      collected pairs using policy cross-entropy with Adam + warmup-cosine
//      LR. Optional periodic dynamics-unroll updates (same recipe as
//      `muzero_chess_player.dart`).
//
// Usage:
//   dart run tool/muzero_vs_stockfish_train.dart tools/stockfish \
//     [--iters=3] [--games=2] [--epochs=2] [--maxply=40] \
//     [--sf-movetime=100] [--sf-skill=0] [--seed=42]

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/dataset/dataset.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/mu_zero/deepseek_aft_decoder.dart';
import 'package:dart_cuda/persistence.dart';
import 'package:dart_cuda/mu_zero/muzero_chess_player.dart'
    show ChessMuZeroAgent, MoveTokenizer, warmupCosineLR;

// ---------------------------------------------------------------------------
// Hyperparameters (overridable via CLI)
// ---------------------------------------------------------------------------
class TrainCfg {
  int numIterations = 3;
  int gamesPerIter = 2;
  int trainEpochsPerIter = 2;
  int maxPlies = 40;

  int sfMovetimeMs = 100;
  int? sfSkill = 0;

  // MuZero training loss / unroll.
  int unrollSteps = 3;
  double baseLR = 1e-3;
  double minLR = 1e-4;

  int? seed;

  // Checkpointing.
  String? loadPath;
  String? savePath;
  int saveEvery = 1; // save every N iterations
}

// ---------------------------------------------------------------------------
// Minimal UCI client.
// ---------------------------------------------------------------------------
class _UciClient {
  final String label;
  final Process process;
  final StreamController<String> _lines = StreamController.broadcast();
  late final StreamSubscription _stdoutSub;
  late final StreamSubscription _stderrSub;
  final bool verbose;

  _UciClient._(this.label, this.process, this.verbose) {
    _stdoutSub = process.stdout
        .transform(const Utf8Decoder())
        .transform(const LineSplitter())
        .listen(_lines.add);
    _stderrSub = process.stderr
        .transform(const Utf8Decoder())
        .transform(const LineSplitter())
        .listen((l) => stderr.writeln('[$label STDERR] $l'));
  }

  static Future<_UciClient> spawn(
    String label,
    String command, {
    bool verbose = false,
  }) async {
    final parts = command.trim().split(RegExp(r'\s+'));
    final exe = parts.first;
    final args = parts.skip(1).toList();
    final proc = await Process.start(
      exe,
      args,
      mode: ProcessStartMode.normal,
      runInShell: true,
    );
    return _UciClient._(label, proc, verbose);
  }

  void _send(String cmd) {
    if (verbose) stdout.writeln('  >> [$label] $cmd');
    process.stdin.writeln(cmd);
  }

  Future<String> _waitFor(
    bool Function(String) match, {
    Duration timeout = const Duration(seconds: 30),
  }) {
    final completer = Completer<String>();
    late StreamSubscription sub;
    sub = _lines.stream.listen((line) {
      if (verbose) stdout.writeln('  << [$label] $line');
      if (match(line) && !completer.isCompleted) {
        completer.complete(line);
        sub.cancel();
      }
    });
    return completer.future.timeout(
      timeout,
      onTimeout: () {
        sub.cancel();
        throw TimeoutException('Timed out waiting for $label');
      },
    );
  }

  Future<void> handshake({int? skillLevel}) async {
    _send('uci');
    await _waitFor((l) => l.trim() == 'uciok');
    if (skillLevel != null) {
      _send('setoption name Skill Level value $skillLevel');
    }
    _send('isready');
    await _waitFor((l) => l.trim() == 'readyok');
  }

  Future<void> newGame() async {
    _send('ucinewgame');
    _send('isready');
    await _waitFor((l) => l.trim() == 'readyok');
  }

  Future<String> bestmoveFromMoves(
    List<String> moves, {
    required int movetimeMs,
  }) async {
    final pos = moves.isEmpty
        ? 'position startpos'
        : 'position startpos moves ${moves.join(' ')}';
    _send(pos);
    _send('isready');
    await _waitFor((l) => l.trim() == 'readyok');
    _send('go movetime $movetimeMs');
    final line = await _waitFor(
      (l) => l.startsWith('bestmove'),
      timeout: Duration(milliseconds: movetimeMs * 5 + 5000),
    );
    return line.split(RegExp(r'\s+'))[1];
  }

  Future<void> quit() async {
    try {
      _send('quit');
    } catch (_) {}
    await process.stdin.close().catchError((_) {});
    final exit = await process.exitCode.timeout(
      const Duration(seconds: 3),
      onTimeout: () {
        process.kill();
        return -1;
      },
    );
    await _stdoutSub.cancel();
    await _stderrSub.cancel();
    await _lines.close();
    if (verbose) stdout.writeln('[$label] exited with code $exit');
  }
}

// ---------------------------------------------------------------------------
// Trajectory record
// ---------------------------------------------------------------------------
class _Step {
  final List<int> historyIds; // tokens BEFORE this move (incl. <start>)
  final int targetMoveId;
  final bool fromStockfish;
  _Step(this.historyIds, this.targetMoveId, this.fromStockfish);
}

class _Trajectory {
  final List<_Step> steps;
  final double finalResultForWhite; // +1, 0, -1
  final bool finished; // true if checkmate/stalemate/draw; false if maxply or abort
  _Trajectory(this.steps, this.finalResultForWhite, this.finished);
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
// MuZero move: legal-move masked argmax over policy head.
// Returns (moveUci, moveId, bishopMove).
// ---------------------------------------------------------------------------
({String uci, int id, Move move})? _muzeroPickMove(
  ChessMuZeroAgent agent,
  MoveTokenizer tok,
  Game game,
  List<int> historyIds,
  int blockSize,
) {
  final legal = game.generateLegalMoves();
  if (legal.isEmpty) return null;

  final tracker = <Tensor>[];
  final ctx = historyIds.length <= blockSize
      ? historyIds
      : historyIds.sublist(historyIds.length - blockSize);

  final states = agent.representation(ctx, tracker);
  final logits = agent.predictPolicy(states, tracker);
  final lastRow = logits.fetchRow(ctx.length - 1);

  Move? bestMove;
  String bestUci = '';
  int bestId = -1;
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
      bestId = id;
    }
  }
  _safeCleanup(tracker, agent.parameters());

  if (bestMove == null) {
    // Fallback: first legal move (may not be in vocab — skip step).
    final m = legal.first;
    final alg = game.toAlgebraic(m);
    final id = tok.encode(alg);
    if (id == null) return null;
    return (uci: alg, id: id, move: m);
  }
  return (uci: bestUci, id: bestId, move: bestMove);
}

// ---------------------------------------------------------------------------
// Play one game MuZero vs Stockfish, return the trajectory.
// ---------------------------------------------------------------------------
Future<_Trajectory> _playOneGame({
  required ChessMuZeroAgent agent,
  required MoveTokenizer tok,
  required int blockSize,
  required _UciClient stockfish,
  required TrainCfg cfg,
  required bool muzeroIsWhite,
}) async {
  final game = Game(variant: Variant.standard());
  final uciHistory = <String>[];
  final tokHistory = <int>[tok.startId];
  final steps = <_Step>[];

  await stockfish.newGame();

  int ply = 0;
  while (!game.gameOver && ply < cfg.maxPlies) {
    final mover = game.state.turn;
    final muzeroToMove = (mover == Bishop.white) == muzeroIsWhite;

    Move chosenMove;
    String chosenUci;
    int targetId;
    bool fromSf;

    if (muzeroToMove) {
      final r = _muzeroPickMove(agent, tok, game, tokHistory, blockSize);
      if (r == null) break;
      chosenMove = r.move;
      chosenUci = r.uci;
      targetId = r.id;
      fromSf = false;
    } else {
      final sfUci = await stockfish.bestmoveFromMoves(
        uciHistory,
        movetimeMs: cfg.sfMovetimeMs,
      );
      if (sfUci == '0000' || sfUci.isEmpty) break;
      final mv = game.getMove(sfUci);
      if (mv == null) break;
      final alg = game.toAlgebraic(mv);
      final id = tok.encode(alg);
      if (id == null) {
        // SF's move isn't in our vocab; play it but skip as a training target.
        chosenMove = mv;
        chosenUci = alg;
        game.makeMove(chosenMove);
        uciHistory.add(alg);
        ply++;
        continue;
      }
      chosenMove = mv;
      chosenUci = alg;
      targetId = id;
      fromSf = true;
    }

    steps.add(_Step(List<int>.from(tokHistory), targetId, fromSf));

    game.makeMove(chosenMove);
    uciHistory.add(chosenUci);
    tokHistory.add(targetId);
    ply++;
  }

  double resultWhite;
  if (game.checkmate) {
    resultWhite = game.state.turn == Bishop.white ? -1.0 : 1.0;
  } else {
    resultWhite = 0.0;
  }
  return _Trajectory(steps, resultWhite, game.gameOver);
}

// ---------------------------------------------------------------------------
// Train one step on a (history, target) pair using policy cross-entropy.
// ---------------------------------------------------------------------------
double _trainStep({
  required ChessMuZeroAgent agent,
  required DeepSeekAFTDecoder model,
  required Adam optimizer,
  required List<int> history,
  required int targetId,
  required int blockSize,
  required double lr,
}) {
  optimizer.lr = lr;
  optimizer.zeroGrad();
  final tracker = <Tensor>[];

  final ctx = history.length <= blockSize
      ? history
      : history.sublist(history.length - blockSize);
  if (ctx.isEmpty) {
    _safeCleanup(tracker, agent.parameters());
    return double.nan;
  }
  // The engine's `crossEntropy` averages over every position in the
  // sequence. For an [<start>, m1, ..., m_{n-1}] prefix we ask the model
  // to predict [m1, ..., m_{n-1}, targetId], i.e. next-token prediction
  // up to and including the Stockfish move we want it to learn.
  final inputs = ctx;
  final targets = <int>[for (int i = 1; i < ctx.length; i++) ctx[i], targetId];
  final logits = model.forward(inputs, agent.dummyEnc, tracker);
  final loss = logits.crossEntropy(targets);
  tracker.add(loss);

  final lossVal = loss.fetchData()[0];
  if (lossVal.isNaN || lossVal.isInfinite) {
    _safeCleanup(tracker, agent.parameters());
    return double.nan;
  }
  loss.backward();
  optimizer.step();
  _safeCleanup(tracker, agent.parameters());
  return lossVal;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
Future<int> main(List<String> args) async {
  if (args.isEmpty || args.first.startsWith('--')) {
    stderr.writeln(
      'Usage: dart run tool/muzero_vs_stockfish_train.dart <stockfish> '
      '[--iters=N] [--games=N] [--epochs=N] [--maxply=N] '
      '[--sf-movetime=MS] [--sf-skill=N] [--seed=N] '
      '[--load=PATH] [--save=PATH] [--save-every=N]',
    );
    return 64;
  }
  final sfPath = args.first;
  final cfg = TrainCfg();
  for (final a in args.skip(1)) {
    if (a.startsWith('--iters=')) {
      cfg.numIterations = int.parse(a.substring('--iters='.length));
    } else if (a.startsWith('--games=')) {
      cfg.gamesPerIter = int.parse(a.substring('--games='.length));
    } else if (a.startsWith('--epochs=')) {
      cfg.trainEpochsPerIter = int.parse(a.substring('--epochs='.length));
    } else if (a.startsWith('--maxply=')) {
      cfg.maxPlies = int.parse(a.substring('--maxply='.length));
    } else if (a.startsWith('--sf-movetime=')) {
      cfg.sfMovetimeMs = int.parse(a.substring('--sf-movetime='.length));
    } else if (a.startsWith('--sf-skill=')) {
      cfg.sfSkill = int.parse(a.substring('--sf-skill='.length));
    } else if (a.startsWith('--seed=')) {
      cfg.seed = int.parse(a.substring('--seed='.length));
    } else if (a.startsWith('--load=')) {
      cfg.loadPath = a.substring('--load='.length);
    } else if (a.startsWith('--save=')) {
      cfg.savePath = a.substring('--save='.length);
    } else if (a.startsWith('--save-every=')) {
      cfg.saveEvery = int.parse(a.substring('--save-every='.length));
    } else {
      stderr.writeln('Unknown option: $a');
      return 64;
    }
  }

  stdout.writeln('--- MuZero vs Stockfish trainer (policy distillation) ---');
  stdout.writeln('Stockfish    : $sfPath');
  stdout.writeln(
    'iters=${cfg.numIterations}  '
    'games/iter=${cfg.gamesPerIter}  '
    'epochs/iter=${cfg.trainEpochsPerIter}  '
    'maxply=${cfg.maxPlies}',
  );
  stdout.writeln('SF movetime  : ${cfg.sfMovetimeMs} ms  skill=${cfg.sfSkill}');
  if (cfg.loadPath != null) stdout.writeln('Load ckpt    : ${cfg.loadPath}');
  if (cfg.savePath != null) {
    stdout.writeln(
      'Save ckpt    : ${cfg.savePath} (every ${cfg.saveEvery} iter)',
    );
  }

  final rng = math.Random(cfg.seed ?? 42);

  // Hyperparams (kept tiny to match `muzero_chess_player.dart`).
  const int embedSize = 64;
  const int blockSize = 32;
  const int numLayers = 2;
  const int numHeads = 4;
  const int numRoutedExperts = 4;
  const int numSharedExperts = 1;
  const int topK = 2;
  const int expertHiddenSize = 64;

  // Build vocab from bundled PGN dataset so MuZero starts with a non-empty
  // move table (vocab grows naturally during training only via the dataset).
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
    await loadModuleBinary(model, cfg.loadPath!);
    stdout.writeln('Loaded checkpoint from ${cfg.loadPath}');
  }
  final optimizer = Adam(agent.parameters(), lr: cfg.baseLR);
  stdout.writeln('Model params : ${agent.parameters().length}');

  final stockfish = await _UciClient.spawn('SF', sfPath);
  await stockfish.handshake(skillLevel: cfg.sfSkill);

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
  int gameIndex = 0;
  int totalWins = 0;
  int totalDraws = 0;
  int totalLosses = 0;
  int totalUnfinished = 0; // games hit maxply with no checkmate
  for (int iter = 0; iter < cfg.numIterations; iter++) {
    stdout.writeln('\n=== Iteration ${iter + 1}/${cfg.numIterations} ===');
    final trajectories = <_Trajectory>[];
    int iterWins = 0;
    int iterDraws = 0;
    int iterLosses = 0;
    int iterUnfinished = 0;

    for (int g = 0; g < cfg.gamesPerIter; g++) {
      final muzeroIsWhite = gameIndex.isEven;
      final sideLabel = muzeroIsWhite ? 'White' : 'Black';
      stdout.write(
        '  game ${g + 1}/${cfg.gamesPerIter} '
        '(MuZero=$sideLabel) ... ',
      );
      final traj = await _playOneGame(
        agent: agent,
        tok: tok,
        blockSize: blockSize,
        stockfish: stockfish,
        cfg: cfg,
        muzeroIsWhite: muzeroIsWhite,
      );
      trajectories.add(traj);
      final muzeroResult = muzeroIsWhite
          ? traj.finalResultForWhite
          : -traj.finalResultForWhite;
      final sfCount = traj.steps.where((s) => s.fromStockfish).length;
      String outcome;
      if (!traj.finished) {
        outcome = 'unfinished';
        iterUnfinished++;
      } else if (muzeroResult > 0.5) {
        outcome = 'WIN';
        iterWins++;
      } else if (muzeroResult < -0.5) {
        outcome = 'loss';
        iterLosses++;
      } else {
        outcome = 'draw';
        iterDraws++;
      }
      stdout.writeln(
        '${traj.steps.length} plies '
        '(${sfCount} SF teacher steps), '
        '$outcome',
      );
      gameIndex++;
    }

    totalWins += iterWins;
    totalDraws += iterDraws;
    totalLosses += iterLosses;
    totalUnfinished += iterUnfinished;
    final iterPlayed = iterWins + iterDraws + iterLosses;
    final iterScore = iterPlayed > 0
        ? (iterWins + 0.5 * iterDraws) / iterPlayed
        : 0.0;
    stdout.writeln(
      '  iter record: ${iterWins}W-${iterDraws}D-${iterLosses}L '
      '(${iterUnfinished} unfinished)  '
      'score=${(iterScore * 100).toStringAsFixed(1)}%',
    );

    model.updateRoutingBias();

    // Build a flat list of (history, target) pairs from SF-teacher steps.
    // (MuZero-own steps would only teach the model to imitate itself.)
    final pairs = <_Step>[];
    for (final traj in trajectories) {
      for (final s in traj.steps) {
        if (s.fromStockfish) pairs.add(s);
      }
    }
    if (pairs.isEmpty) {
      stdout.writeln('  no SF teacher pairs collected; skipping training');
      continue;
    }

    for (int epoch = 0; epoch < cfg.trainEpochsPerIter; epoch++) {
      pairs.shuffle(rng);
      double lossSum = 0.0;
      int n = 0;
      for (final p in pairs) {
        final l = _trainStep(
          agent: agent,
          model: model,
          optimizer: optimizer,
          history: p.historyIds,
          targetId: p.targetMoveId,
          blockSize: blockSize,
          lr: currentLR(),
        );
        globalStep++;
        if (!l.isNaN) {
          lossSum += l;
          n++;
        }
      }
      if (n > 0) {
        stdout.writeln(
          '  epoch ${epoch + 1}/${cfg.trainEpochsPerIter}  '
          'pairs=$n  '
          'lr=${currentLR().toStringAsFixed(5)}  '
          'mean_loss=${(lossSum / n).toStringAsFixed(4)}',
        );
      }
    }

    // Periodic checkpoint save.
    if (cfg.savePath != null &&
        ((iter + 1) % cfg.saveEvery == 0 || iter == cfg.numIterations - 1)) {
      await saveModuleBinary(model, cfg.savePath!);
      stdout.writeln('  saved checkpoint -> ${cfg.savePath}');
    }
  }

  sw.stop();
  final totalPlayed = totalWins + totalDraws + totalLosses;
  final overallScore = totalPlayed > 0
      ? (totalWins + 0.5 * totalDraws) / totalPlayed
      : 0.0;
  stdout.writeln('\n=== Final record vs Stockfish ===');
  stdout.writeln(
    'W=${totalWins}  D=${totalDraws}  L=${totalLosses}  '
    '(unfinished=${totalUnfinished})',
  );
  stdout.writeln(
    'win_rate=${totalPlayed > 0 ? (totalWins / totalPlayed * 100).toStringAsFixed(1) : '0.0'}%  '
    'score=${(overallScore * 100).toStringAsFixed(1)}%',
  );
  stdout.writeln(
    '\nTotal wall-clock: '
    '${(sw.elapsedMilliseconds / 1000).toStringAsFixed(2)} s',
  );

  await stockfish.quit();
  return 0;
}
