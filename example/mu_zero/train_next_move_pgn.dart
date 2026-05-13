// Train a tiny `ChessMuZeroAgent`'s policy head to predict the next UCI
// move. Source games come from one of:
//   * the bundled `lib/loaders/dataset.dart` set (default), or
//   * a `.pgn` file on disk via `--pgn=path/to/games.pgn` (standard PGN
//     with SAN moves; comments {...}, variations (...), move numbers,
//     headers [...], and result tokens are stripped).
//
// Per game we form input/target pairs by shifting tokens by 1:
//
//   input  = [<start>, m_1, m_2, ..., m_{T-1}]
//   target = [m_1,     m_2, m_3, ..., m_T]
//
// Loss = mean cross-entropy of decoder logits vs `target`. A fraction of
// games is held out as a validation set; val loss + val top-1 accuracy
// are reported every `--valEvery` steps. Greedy sampling is
// legal-move-masked using a fresh bishop `Game`.
//
// Usage (from repo root):
//   dart run example/mu_zero/train_next_move_pgn.dart
//   dart run example/mu_zero/train_next_move_pgn.dart --pgn=path/to/games.pgn
//
// Optional flags:
//   --pgn=PATH       (none)        load games from this PGN file
//                                  (otherwise uses bundled dataset)
//   --games=N        (default 32)  how many games to use (cap)
//   --steps=N        (default 600) total optimizer steps
//   --block=N        (default 32)  max sequence length per game
//   --embed=N        (default 64)  embedding size
//   --layers=N       (default 2)   decoder layers
//   --heads=N        (default 4)   AFT heads
//   --lr=F           (default 1e-3) Adam learning rate
//   --valSplit=F     (default 0.2) fraction of games held out for val
//   --logEvery=N     (default 25)  log train loss every N steps
//   --valEvery=N     (default 100) eval val loss/acc every N steps
//   --sampleEvery=N  (default 100) print a greedy continuation every N steps
//   --seed=N         (default 7)   RNG seed for sampling

import 'dart:io';
import 'dart:math' as math;

import 'package:bishop/bishop.dart';
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

String? _strFlag(List<String> args, String name) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) return a.substring(p.length);
  }
  return null;
}

// ---------------------------------------------------------------------------
// PGN file loader. Standard PGN: header lines in [], movetext with SAN,
// move numbers (1.  1...), comments {...}, variations (possibly nested),
// NAGs ($n), and a result token at the end. Returns each game as a list
// of UCI strings, replaying SAN through bishop to validate legality.
// ---------------------------------------------------------------------------

String _stripBraces(String s) {
  final sb = StringBuffer();
  var depth = 0;
  for (var i = 0; i < s.length; i++) {
    final c = s[i];
    if (c == '{') {
      depth++;
    } else if (c == '}') {
      if (depth > 0) depth--;
    } else if (depth == 0) {
      sb.write(c);
    }
  }
  return sb.toString();
}

String _stripParens(String s) {
  final sb = StringBuffer();
  var depth = 0;
  for (var i = 0; i < s.length; i++) {
    final c = s[i];
    if (c == '(') {
      depth++;
    } else if (c == ')') {
      if (depth > 0) depth--;
    } else if (depth == 0) {
      sb.write(c);
    }
  }
  return sb.toString();
}

List<List<String>> _parsePgnFile(String path, int maxGames) {
  final raw = File(path).readAsStringSync();
  // Split into per-game blocks. Use the start of a [Event header as the
  // game boundary. Anything before the first [Event is dropped.
  final blocks = <String>[];
  final firstEvent = raw.indexOf('[Event ');
  final body = firstEvent >= 0 ? raw.substring(firstEvent) : raw;
  final pieces = body.split(RegExp(r'(?=^\[Event )', multiLine: true));
  for (final p in pieces) {
    final t = p.trim();
    if (t.isNotEmpty) blocks.add(t);
    if (blocks.length >= maxGames * 4) break; // generous over-fetch
  }

  final results = <List<String>>[];
  final resultTokens = {'1-0', '0-1', '1/2-1/2', '*'};
  for (final block in blocks) {
    // Strip header lines.
    final lines = block
        .split('\n')
        .where((l) => !l.trimLeft().startsWith('['))
        .join(' ');
    var movetext = _stripBraces(lines);
    movetext = _stripParens(movetext);
    // Strip move numbers (`12.`, `12...`) and NAGs (`$3`).
    movetext = movetext.replaceAll(RegExp(r'\d+\.(\.\.)?'), ' ');
    movetext = movetext.replaceAll(RegExp(r'\$\d+'), ' ');

    final tokens = movetext
        .split(RegExp(r'\s+'))
        .where((t) => t.isNotEmpty && !resultTokens.contains(t))
        .toList();
    if (tokens.isEmpty) continue;

    final game = Game();
    final uciMoves = <String>[];
    for (final san in tokens) {
      final m = game.getMoveSan(san, checks: false);
      if (m == null) break; // unparseable -> stop this game
      uciMoves.add(game.toAlgebraic(m));
      game.makeMove(m);
    }
    if (uciMoves.isNotEmpty) results.add(uciMoves);
    if (results.length >= maxGames) break;
  }
  return results;
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

/// Greedy-sample up to `maxNew` next moves from a fresh `<start>` context,
/// masking the policy logits to only the legal moves at each step (in UCI).
/// Stops early on `<end>`, no legal moves, or no legal move present in
/// the tokenizer's vocabulary.
List<String> _sampleGreedy(
  ChessMuZeroAgent agent,
  MoveTokenizer tok,
  int maxNew,
  int blockSize,
) {
  final game = Game();
  final history = <int>[tok.startId];
  final out = <String>[];
  for (var step = 0; step < maxNew; step++) {
    final legal = game.generateLegalMoves();
    if (legal.isEmpty) break;
    // Build {uci -> tokenId} for the legal moves that exist in the vocab.
    final legalIdToUci = <int, String>{};
    for (final m in legal) {
      final uci = game.toAlgebraic(m);
      final id = tok.encode(uci);
      if (id != null) legalIdToUci[id] = uci;
    }
    if (legalIdToUci.isEmpty) break;

    final ctx = history.length <= blockSize
        ? history
        : history.sublist(history.length - blockSize);
    final tracker = <Tensor>[];
    final state = agent.representation(ctx, tracker);
    final logits = agent.predictPolicy(state, tracker);
    final flat = logits.fetchData();
    // Find argmax over the masked legal set on the last row.
    final base = (ctx.length - 1) * tok.vocabSize;
    var bestId = -1;
    var bestV = -double.infinity;
    for (final id in legalIdToUci.keys) {
      final v = flat[base + id];
      if (v > bestV) {
        bestV = v;
        bestId = id;
      }
    }
    for (final t in tracker) {
      t.dispose();
    }
    if (bestId < 0) break;

    final uci = legalIdToUci[bestId]!;
    final move = game.getMove(uci);
    if (move == null) break;
    game.makeMove(move);
    history.add(bestId);
    out.add(uci);
  }
  return out;
}

Future<void> main(List<String> args) async {
  final pgnPath = _strFlag(args, 'pgn');
  final numGamesReq = _intFlag(args, 'games', 32);
  final numSteps = _intFlag(args, 'steps', 600);
  final blockSize = _intFlag(args, 'block', 32);
  final embedSize = _intFlag(args, 'embed', 64);
  final numLayers = _intFlag(args, 'layers', 2);
  final numHeads = _intFlag(args, 'heads', 4);
  final lr = _doubleFlag(args, 'lr', 1e-3);
  final valSplit = _doubleFlag(args, 'valSplit', 0.2).clamp(0.0, 0.9);
  final logEvery = _intFlag(args, 'logEvery', 25);
  final valEvery = _intFlag(args, 'valEvery', 100);
  final sampleEvery = _intFlag(args, 'sampleEvery', 100);
  final seed = _intFlag(args, 'seed', 7);

  // ---- 1. Dataset --------------------------------------------------------
  List<List<String>> games;
  if (pgnPath != null) {
    if (!File(pgnPath).existsSync()) {
      stderr.writeln('error: pgn file not found: $pgnPath');
      exit(1);
    }
    print('📖 Parsing PGN: $pgnPath (cap=$numGamesReq games)');
    games = _parsePgnFile(pgnPath, numGamesReq);
    if (games.isEmpty) {
      stderr.writeln('error: no playable games parsed from $pgnPath');
      exit(2);
    }
  } else {
    final n = numGamesReq.clamp(1, dataset.length);
    games = dataset.take(n).toList();
  }
  final tok = MoveTokenizer.fromGames(games);
  print(
    'Loaded ${games.length} games | vocab=${tok.vocabSize} | '
    'avg_len=${(games.map((g) => g.length).reduce((a, b) => a + b) / games.length).toStringAsFixed(1)}',
  );

  // Pre-build per-game token streams: [<start>, m1, ..., mN] truncated to blockSize.
  final allStreams = <List<int>>[];
  for (final g in games) {
    final ids = <int>[tok.startId];
    for (final mv in g) {
      final id = tok.encode(mv);
      if (id == null) continue;
      ids.add(id);
      if (ids.length >= blockSize + 1) break;
    }
    if (ids.length >= 2) allStreams.add(ids);
  }

  // Shuffle once (seeded) then split into train/val.
  final splitRng = math.Random(seed);
  allStreams.shuffle(splitRng);
  final nVal = (allStreams.length * valSplit).round();
  final valStreams = allStreams.sublist(0, nVal);
  final trainStreams = allStreams.sublist(nVal);
  if (trainStreams.isEmpty) {
    stderr.writeln('error: no training streams after split');
    exit(3);
  }
  print(
    'Streams: train=${trainStreams.length} val=${valStreams.length} '
    '(valSplit=$valSplit)',
  );

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
  final rng = math.Random(seed + 1);
  double emaLoss = double.nan;
  for (var step = 1; step <= numSteps; step++) {
    final stream = trainStreams[rng.nextInt(trainStreams.length)];
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
      // Token-level top-1 accuracy on the same minibatch (cheap).
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
        'train_acc=${(acc * 100).toStringAsFixed(1)}% | '
        'len=${input.length}',
      );
    }

    if (valStreams.isNotEmpty && (step % valEvery == 0 || step == numSteps)) {
      var totalLoss = 0.0;
      var totalTokens = 0;
      var correctTokens = 0;
      for (final s in valStreams) {
        final inp = s.sublist(0, s.length - 1);
        final tgt = s.sublist(1);
        final tk = <Tensor>[];
        final st = agent.representation(inp, tk);
        final lg = agent.predictPolicy(st, tk);
        final lossT = lg.crossEntropy(tgt);
        tk.add(lossT);
        final flat = lg.fetchData();
        for (var i = 0; i < tgt.length; i++) {
          if (_argmaxRow(flat, i, tok.vocabSize) == tgt[i]) correctTokens++;
        }
        // crossEntropy returns mean loss; weight by tokens to get a proper avg.
        totalLoss += lossT.fetchData()[0] * tgt.length;
        totalTokens += tgt.length;
        for (final t in tk) {
          t.dispose();
        }
      }
      final valLoss = totalLoss / totalTokens;
      final valAcc = correctTokens / totalTokens;
      print(
        '    [val] loss=${valLoss.toStringAsFixed(4)} | '
        'acc=${(valAcc * 100).toStringAsFixed(1)}% | '
        'tokens=$totalTokens',
      );
    }

    if (step % sampleEvery == 0 || step == numSteps) {
      final sampled = _sampleGreedy(agent, tok, 12, blockSize);
      print('    sample(<start>, legal-masked ->): ${sampled.join(" ")}');
    }
  }

  // ---- 4. Cleanup --------------------------------------------------------
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
