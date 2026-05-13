// Overfit a `ChessMuZeroAgent` on a single training example — the chess
// start position — using BOTH heads simultaneously:
//   * value head  -> Stockfish's eval (cp -> tanh(cp/600), white POV)
//   * policy head -> Stockfish's best move (cross-entropy)
//
// Pipeline:
//   1. Spawn the bundled Stockfish UCI binary (looks in
//      `example/tools/stockfish` then `tools/stockfish`, override with
//      --stockfish=/path).
//   2. Send `position startpos` + `go movetime`. Parse the final
//      `info ... score cp|mate` line for the value AND the trailing
//      `bestmove <uci>` line for the move.
//   3. Build a tiny 3-token vocab (`<start>`, `<end>`, <bestmove>) and a
//      tiny `ChessMuZeroAgent` (1-layer MoE AFT decoder, 16-d embed).
//   4. Adam loop minimising
//          L = MSE(value_pred, value_target)
//            + CE(policy_logits, bestMoveId)
//      until both heads are within tolerance of their targets.
//
// Usage (from repo root):
//   dart run example/mu_zero/overfit_chess_value_startpos.dart
//
// Optional flags:
//   --epochs=N    (default 300)        max training epochs
//   --lr=F        (default 0.05)       Adam learning rate
//   --movetime=N  (default 500)        Stockfish movetime in ms
//   --tol=F       (default 0.02)       early-stop |val - target| tolerance
//                                      (policy must also be argmax==target)
//   --stockfish=PATH                   override binary location

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/deepseek/deepseek_aft_decoder.dart';
import 'package:dart_cuda/core/models/mu_zero/muzero_chess_player.dart'
    show ChessMuZeroAgent;

const _sfCandidates = <String>['example/tools/stockfish', 'tools/stockfish'];

String? _resolveStockfish(List<String> args) {
  for (final a in args) {
    if (a.startsWith('--stockfish=')) {
      final p = a.substring('--stockfish='.length);
      return File(p).existsSync() ? p : null;
    }
  }
  for (final p in _sfCandidates) {
    if (File(p).existsSync()) return p;
  }
  return null;
}

double _cpToValue(int cp) {
  final x = cp / 600.0;
  final ep = math.exp(x);
  final en = math.exp(-x);
  return (ep - en) / (ep + en);
}

class _SfStartposResult {
  final double value; // [-1, 1] from white POV
  final String bestMoveUci; // e.g. "e2e4"
  _SfStartposResult(this.value, this.bestMoveUci);
}

/// Asks an already-running Stockfish process for its eval AND best move
/// on the start position.
Future<_SfStartposResult> _analyzeStartpos(
  Process sf,
  Stream<String> lines,
  int movetimeMs,
) async {
  void send(String s) => sf.stdin.writeln(s);

  send('position startpos');
  send('isready');
  await lines.firstWhere((l) => l.trim() == 'readyok');

  int? lastCp;
  int? lastMate;
  String? bestMove;
  final done = Completer<void>();
  final sub = lines.listen((line) {
    if (line.startsWith('bestmove')) {
      // "bestmove e2e4 ponder ..."
      final parts = line.split(RegExp(r'\s+'));
      if (parts.length >= 2) bestMove = parts[1];
      if (!done.isCompleted) done.complete();
      return;
    }
    if (!line.startsWith('info ')) return;
    final parts = line.split(RegExp(r'\s+'));
    final i = parts.indexOf('score');
    if (i == -1 || i + 2 >= parts.length) return;
    final kind = parts[i + 1];
    final val = int.tryParse(parts[i + 2]);
    if (val == null) return;
    if (kind == 'cp') {
      lastCp = val;
      lastMate = null;
    } else if (kind == 'mate') {
      lastMate = val;
      lastCp = null;
    }
  });

  send('go movetime $movetimeMs');
  await done.future.timeout(Duration(milliseconds: movetimeMs * 5 + 5000));
  await sub.cancel();

  double value;
  if (lastMate != null) {
    value = lastMate! >= 0 ? 0.99 : -0.99;
  } else if (lastCp != null) {
    value = _cpToValue(lastCp!);
  } else {
    value = 0.0;
  }
  return _SfStartposResult(value, bestMove ?? '');
}

int _argmax(List<double> xs) {
  var best = 0;
  var bestV = xs[0];
  for (var i = 1; i < xs.length; i++) {
    if (xs[i] > bestV) {
      bestV = xs[i];
      best = i;
    }
  }
  return best;
}

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

Future<void> main(List<String> args) async {
  final epochs = _intFlag(args, 'epochs', 300);
  final lr = _doubleFlag(args, 'lr', 0.05);
  final movetimeMs = _intFlag(args, 'movetime', 500);
  final tol = _doubleFlag(args, 'tol', 0.02);

  // ---- 1. Get target from Stockfish ----------------------------------------
  final sfPath = _resolveStockfish(args);
  if (sfPath == null) {
    stderr.writeln(
      'error: stockfish binary not found. Looked in:\n'
      '  ${_sfCandidates.join('\n  ')}\n'
      'Pass --stockfish=/path/to/stockfish or build with:\n'
      '  cd example/tools/sf/stockfish/src && make build',
    );
    exit(1);
  }
  final sf = await Process.start(sfPath, const []);
  final linesCtrl = StreamController<String>.broadcast();
  sf.stdout
      .transform(const Utf8Decoder())
      .transform(const LineSplitter())
      .listen(linesCtrl.add);
  sf.stderr.drain<void>();
  final lines = linesCtrl.stream;

  sf.stdin.writeln('uci');
  await lines.firstWhere((l) => l.trim() == 'uciok');
  sf.stdin.writeln('isready');
  await lines.firstWhere((l) => l.trim() == 'readyok');

  print('🐟 Asking Stockfish about startpos (movetime=${movetimeMs}ms)...');
  final sfResult = await _analyzeStartpos(sf, lines, movetimeMs);
  final targetValue = sfResult.value;
  final bestMoveUci = sfResult.bestMoveUci;
  if (bestMoveUci.isEmpty) {
    stderr.writeln('error: stockfish did not return a bestmove');
    exit(2);
  }
  print('🎯 target value (white POV): ${targetValue.toStringAsFixed(6)}');
  print('🎯 target bestmove        : $bestMoveUci');

  try {
    sf.stdin.writeln('quit');
    await sf.stdin.close();
  } catch (_) {}
  unawaited(sf.exitCode);

  // ---- 2. Build a tiny agent ----------------------------------------------
  // 3-token vocab: <start>=0, <end>=1, <bestmove>=2.
  const startId = 0;
  const bestMoveId = 2;
  const vocabSize = 3;
  const embedSize = 16;
  const blockSize = 4;
  final model = DeepSeekAFTDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 1,
    numHeads: 4,
    encoderEmbedSize: embedSize,
    numRoutedExperts: 2,
    numSharedExperts: 1,
    topK: 1,
    expertHiddenSize: embedSize,
  );
  final agent = ChessMuZeroAgent(model);

  // Constant value target tensor, shape [1, 1] to match value head output.
  final valueTargetTensor = Tensor.fromList([1, 1], [targetValue]);

  final params = agent.parameters();
  final opt = Adam(params, lr: lr);

  print(
    '🚀 Joint overfit (value+policy) on startpos '
    '(epochs=$epochs, lr=$lr, tol=$tol)',
  );

  double finalPredV = double.nan;
  double finalLossV = double.nan;
  double finalLossP = double.nan;
  int finalArgmax = -1;
  for (var epoch = 1; epoch <= epochs; epoch++) {
    opt.zeroGrad();
    final tracker = <Tensor>[];

    // h(start) -> [1, embedSize]
    final state = agent.representation([startId], tracker);

    // f_value(state) -> [1, 1] in (-1, 1)
    final valuePred = agent.predictValue(state, tracker);

    // f_policy(state) -> [1, vocabSize]
    final policyLogits = agent.predictPolicy(state, tracker);

    // Value loss: MSE on the 1x1 scalar tensor.
    final vDiff = valuePred - valueTargetTensor;
    final vLoss = vDiff.pow(2.0);

    // Policy loss: CE against the bestmove token id.
    final pLoss = policyLogits.crossEntropy(const [bestMoveId]);

    // Joint loss.
    final loss = vLoss + pLoss;
    tracker.addAll([vDiff, vLoss, pLoss, loss]);

    loss.backward();
    opt.step();

    finalPredV = valuePred.fetchData()[0];
    finalLossV = vLoss.fetchData()[0];
    finalLossP = pLoss.fetchData()[0];
    finalArgmax = _argmax(policyLogits.fetchData());

    final policyOk = finalArgmax == bestMoveId;
    final valueOk = (finalPredV - targetValue).abs() < tol;

    if (epoch == 1 || epoch % 10 == 0 || (policyOk && valueOk)) {
      print(
        '  epoch ${epoch.toString().padLeft(4)} | '
        'v_pred=${finalPredV.toStringAsFixed(4)} '
        '(tgt=${targetValue.toStringAsFixed(4)}) | '
        'v_loss=${finalLossV.toStringAsExponential(2)} | '
        'p_loss=${finalLossP.toStringAsExponential(2)} | '
        'argmax=$finalArgmax${policyOk ? "\u2713" : ""}',
      );
    }

    for (final t in tracker) {
      t.dispose();
    }

    if (policyOk && valueOk) {
      print(
        '\u2705 both heads converged at epoch $epoch '
        '(policy argmax=bestmove, |v_err|<$tol)',
      );
      break;
    }
  }

  print(
    '\nFinal:\n'
    '  value : pred=${finalPredV.toStringAsFixed(6)}, '
    'target=${targetValue.toStringAsFixed(6)}, '
    '|err|=${(finalPredV - targetValue).abs().toStringAsExponential(3)}\n'
    '  policy: argmax=$finalArgmax, target=$bestMoveId '
    '(${finalArgmax == bestMoveId ? "match" : "mismatch"}), '
    'CE=${finalLossP.toStringAsExponential(3)}',
  );

  // Cleanup
  valueTargetTensor.dispose();
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
