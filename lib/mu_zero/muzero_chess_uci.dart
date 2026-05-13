// file: lib/mu_zero/muzero_chess_uci.dart
//
// Universal Chess Interface (UCI) front-end for the MuZero chess player
// defined in `muzero_chess_player.dart`.
//
// Pipeline at startup:
//   1. Build a tokenizer from bundled PGN games.
//   2. Train a `ChessMuZeroAgent` (progress to stderr — UCI requires
//      stdout be reserved for protocol messages).
//   3. Enter the UCI command loop. On `go`, the trained policy head
//      produces a legal-move-masked argmax and we emit it as `bestmove`.
//
// Notes:
//   * The model can only play moves whose UCI strings were seen during
//     training (i.e. exist in the tokenizer). If a position has no
//     in-vocab legal moves, we fall back to the first legal move from
//     `bishop.generateLegalMoves()`.
//   * Search is essentially "depth-1 argmax over the policy net". No
//     tree search yet — `wtime`/`btime`/`movetime` from the GUI are
//     parsed but ignored (a move is produced as soon as the forward
//     pass completes).
//   * Spec: http://wbec-ridderkerk.nl/html/UCIProtocol.html

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/loaders/dataset.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/mu_zero/deepseek_aft_decoder.dart';

import 'package:dart_cuda/mu_zero/muzero_chess_player.dart'
    show
        ChessMuZeroAgent,
        GameTrajectory,
        MoveTokenizer,
        buildTrajectory,
        warmupCosineLR;
import 'package:dart_cuda/mu_zero/muzero_chess_mcts.dart' show pickNextMoveMcts;

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

/// UCI front-end that uses a [ChessMuZeroAgent] to pick moves.
class MuZeroUciEngine {
  final String name;
  final String author;

  final ChessMuZeroAgent agent;
  final MoveTokenizer tok;
  final int blockSize;

  final IOSink _out;
  final Stream<String> _in;

  late Game _game;
  final List<int> _history = []; // tokenized move history (no <start>)

  /// Number of PUCT simulations per `go`. 0 disables MCTS and falls back
  /// to depth-1 policy argmax. Settable via
  /// `setoption name MctsSims value <N>`.
  int mctsSims = 0;

  Completer<void>? _quitCompleter;

  MuZeroUciEngine({
    required this.agent,
    required this.tok,
    required this.blockSize,
    this.name = 'DartCudaMuZero',
    this.author = 'dart_cuda',
    Stream<String>? input,
    IOSink? output,
  }) : _in =
           input ??
           stdin.transform(const Utf8Decoder()).transform(const LineSplitter()),
       _out = output ?? stdout {
    _game = Game(variant: Variant.standard());
  }

  /// Starts the UCI command loop. Completes when `quit` is received.
  Future<void> run() async {
    _quitCompleter = Completer<void>();
    final sub = _in.listen(
      _handleCommand,
      onDone: () {
        if (!(_quitCompleter?.isCompleted ?? true)) _quitCompleter!.complete();
      },
    );
    await _quitCompleter!.future;
    await sub.cancel();
  }

  void _writeln(String line) {
    _out.writeln(line);
  }

  void _handleCommand(String raw) {
    final line = raw.trim();
    if (line.isEmpty) return;
    final parts = line.split(RegExp(r'\s+'));
    final cmd = parts.first;

    switch (cmd) {
      case 'uci':
        _writeln('id name $name');
        _writeln('id author $author');
        _writeln('option name MctsSims type spin default 0 min 0 max 100000');
        _writeln('uciok');
        break;
      case 'isready':
        _writeln('readyok');
        break;
      case 'ucinewgame':
        _game = Game(variant: Variant.standard());
        _history.clear();
        break;
      case 'position':
        _handlePosition(parts);
        break;
      case 'go':
        _handleGo();
        break;
      case 'stop':
        // Search is synchronous; nothing to do.
        break;
      case 'setoption':
        _handleSetOption(parts);
        break;
      case 'ponderhit':
        break;
      case 'quit':
        if (!(_quitCompleter?.isCompleted ?? true)) {
          _quitCompleter!.complete();
        }
        break;
      default:
        break;
    }
  }

  void _handlePosition(List<String> parts) {
    if (parts.length < 2) return;

    final movesIndex = parts.indexOf('moves');
    String fen;

    if (parts[1] == 'startpos') {
      fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    } else if (parts[1] == 'fen') {
      final end = movesIndex != -1 ? movesIndex : parts.length;
      if (end <= 2) return;
      fen = parts.sublist(2, end).join(' ');
    } else {
      return;
    }

    _game = Game(variant: Variant.standard(), fen: fen);
    _history.clear();

    if (movesIndex != -1) {
      for (final moveString in parts.sublist(movesIndex + 1)) {
        final move = _game.getMove(moveString);
        if (move == null) break;
        final alg = _game.toAlgebraic(move);
        _game.makeMove(move, false);
        final id = tok.encode(alg);
        if (id != null) _history.add(id);
      }
    }
  }

  void _handleGo() {
    if (_game.gameOver) {
      _writeln('bestmove 0000');
      return;
    }

    final legal = _game.generateLegalMoves();
    if (legal.isEmpty) {
      _writeln('bestmove 0000');
      return;
    }

    // MCTS branch.
    if (mctsSims > 0) {
      final picked = pickNextMoveMcts(
        agent,
        tok,
        _game,
        _history,
        blockSize,
        numSimulations: mctsSims,
      );
      if (picked != null) {
        _writeln('info depth 1 nodes $mctsSims pv ${picked.uci} string mcts');
        _writeln('bestmove ${picked.uci}');
        return;
      }
      _writeln('info string mcts produced no in-vocab move; falling back');
    }

    // Score legal moves using the trained policy head.
    final tracker = <Tensor>[];
    final ctx = [tok.startId, ..._history];
    final trimmed = ctx.length <= blockSize
        ? ctx
        : ctx.sublist(ctx.length - blockSize);

    final states = agent.representation(trimmed, tracker);
    final logits = agent.predictPolicy(states, tracker);
    final lastRow = logits.fetchRow(trimmed.length - 1);

    Move? bestMove;
    String bestAlg = '';
    double bestScore = -double.infinity;
    for (final m in legal) {
      final alg = _game.toAlgebraic(m);
      final id = tok.encode(alg);
      if (id == null) continue;
      final s = lastRow[id];
      if (s > bestScore) {
        bestScore = s;
        bestMove = m;
        bestAlg = alg;
      }
    }
    _safeCleanup(tracker, agent.parameters());

    // Fallback: no in-vocab legal move — pick the first legal move so we
    // never forfeit on time. Also emit an info string so the GUI can see
    // when the model is guessing.
    if (bestMove == null) {
      final m = legal.first;
      bestAlg = _game.toAlgebraic(m);
      _writeln('info string no in-vocab legal moves; falling back to $bestAlg');
    } else {
      _writeln(
        'info depth 1 score cp ${(bestScore * 100).round()} pv $bestAlg',
      );
    }

    _writeln('bestmove $bestAlg');
  }

  /// Parses `setoption name <Name> value <Value>` lines. Only `MctsSims`
  /// is honored; everything else is silently ignored as the UCI spec
  /// allows.
  void _handleSetOption(List<String> parts) {
    final nameIdx = parts.indexOf('name');
    final valueIdx = parts.indexOf('value');
    if (nameIdx == -1 || valueIdx == -1 || valueIdx <= nameIdx + 1) return;
    final name = parts.sublist(nameIdx + 1, valueIdx).join(' ');
    final value = parts.sublist(valueIdx + 1).join(' ');
    if (name.toLowerCase() == 'mctssims') {
      final n = int.tryParse(value);
      if (n != null && n >= 0) mctsSims = n;
    }
  }

  /// Exposed for tests.
  Game get game => _game;
}

// ---------------------------------------------------------------------------
// Training helper. Mirrors the loop in `muzero_chess_player.dart` but
// directs all human-readable output to stderr so the UCI stdout channel
// stays clean.
// ---------------------------------------------------------------------------

Future<void> trainAgent({
  required ChessMuZeroAgent agent,
  required DeepSeekAFTDecoder model,
  required MoveTokenizer tok,
  required List<GameTrajectory> trajectories,
  required int blockSize,
  int numSteps = 600,
  double baseLR = 1e-3,
  double minLR = 1e-4,
  int dynamicsEvery = 2,
  int unrollSteps = 3,
  int logEvery = 100,
  int balanceEvery = 50,
}) async {
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
      model.updateRoutingBias();
      if ((step + 1) % logEvery == 0) {
        final pAvg = pN > 0 ? (pLossSum / pN).toStringAsFixed(4) : '   -  ';
        final dAvg = dN > 0 ? (dLossSum / dN).toStringAsFixed(4) : '   -  ';
        stderr.writeln(
          '[train] step ${step + 1}/$numSteps  '
          'lr=${optimizer.lr.toStringAsFixed(5)}  '
          'policy=$pAvg  dyn=$dAvg',
        );
        pLossSum = 0.0;
        dLossSum = 0.0;
        pN = 0;
        dN = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Entry point.
// ---------------------------------------------------------------------------

Future<void> main(List<String> args) async {
  // Allow --no-train for quick GUI-handshake debugging.
  final bool train = !args.contains('--no-train');
  final int trainSteps = _readIntFlag(args, '--steps') ?? 600;

  stderr.writeln('[boot] MuZero UCI engine starting (train=$train)');

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
  stderr.writeln('[boot] move vocab: ${tok.vocabSize}');

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
  stderr.writeln(
    '[boot] params=${agent.parameters().length} '
    'routed=$numRoutedExperts top-$topK shared=$numSharedExperts',
  );

  if (train) {
    final trajectories = games
        .map((g) => buildTrajectory(g, blockSize, tok))
        .toList();
    await trainAgent(
      agent: agent,
      model: model,
      tok: tok,
      trajectories: trajectories,
      blockSize: blockSize,
      numSteps: trainSteps,
    );
    stderr.writeln('[boot] training complete; entering UCI loop');
  } else {
    stderr.writeln('[boot] skipping training; entering UCI loop');
  }

  final engine = MuZeroUciEngine(agent: agent, tok: tok, blockSize: blockSize);
  await engine.run();
}

int? _readIntFlag(List<String> args, String flag) {
  final i = args.indexOf(flag);
  if (i < 0 || i + 1 >= args.length) return null;
  return int.tryParse(args[i + 1]);
}
