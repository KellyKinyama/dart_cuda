// Overfit a `ChessMuZeroAgent`'s value head on a single training example:
// the chess start position, with the target value coming from Stockfish.
//
// Pipeline:
//   1. Spawn the bundled `tools/stockfish` UCI binary.
//   2. Ask it to evaluate the start position for `--movetime` ms.
//   3. Convert centipawns/mate to a value in [-1, 1] (white POV) using
//      tanh(cp/600) — same conversion the trainer uses.
//   4. Build a tiny `ChessMuZeroAgent` (single-layer MoE decoder) with a
//      one-token vocab (just the `<start>` token, since we never need a
//      real move).
//   5. Train (Adam, MSE) until the value head's prediction matches the
//      Stockfish target — i.e. classical "overfit one example" sanity check
//      that the model + autograd + optimizer all wire correctly.
//
// Usage (from repo root, after `nvcc` build + Stockfish present):
//   dart run example/mu_zero/overfit_chess_value_startpos.dart
//
// Looks for the Stockfish binary in `example/tools/stockfish` first, then
// `tools/stockfish` (legacy). Override with `--stockfish=/path/to/stockfish`.
//
// Optional flags:
//   --epochs=N    (default 200)        max training epochs
//   --lr=F        (default 0.05)       Adam learning rate
//   --movetime=N  (default 500)        Stockfish movetime in ms
//   --tol=F       (default 0.01)       early-stop |pred - target| tolerance

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

/// Asks an already-running Stockfish process for its eval of the start
/// position and returns the value from white's POV in [-1, 1].
Future<double> _evalStartPosForWhite(
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
  final done = Completer<void>();
  final sub = lines.listen((line) {
    if (line.startsWith('bestmove')) {
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

  if (lastMate != null) return lastMate! >= 0 ? 0.99 : -0.99;
  if (lastCp != null) return _cpToValue(lastCp!);
  return 0.0;
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
  final epochs = _intFlag(args, 'epochs', 200);
  final lr = _doubleFlag(args, 'lr', 0.05);
  final movetimeMs = _intFlag(args, 'movetime', 500);
  final tol = _doubleFlag(args, 'tol', 0.01);

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

  print('🐟 Asking Stockfish for startpos eval (movetime=${movetimeMs}ms)...');
  final target = await _evalStartPosForWhite(sf, lines, movetimeMs);
  print('🎯 target value (white POV): ${target.toStringAsFixed(6)}');

  try {
    sf.stdin.writeln('quit');
    await sf.stdin.close();
  } catch (_) {}
  unawaited(sf.exitCode);

  // ---- 2. Build a tiny agent ----------------------------------------------
  // Vocab is just the <start> sentinel; we always feed a 1-token sequence.
  const vocabSize = 2; // <start> + one slot to keep matmul shapes happy
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

  // Build the constant target tensor once. Shape [1, 1] matches the value
  // head's per-row scalar output.
  final targetTensor = Tensor.fromList([1, 1], [target]);

  final params = agent.parameters();
  final opt = Adam(params, lr: lr);

  print(
    '🚀 Overfitting value head on startpos (epochs=$epochs, lr=$lr, tol=$tol)',
  );

  double finalPred = double.nan;
  double finalLoss = double.nan;
  for (var epoch = 1; epoch <= epochs; epoch++) {
    opt.zeroGrad();
    final tracker = <Tensor>[];

    // h(start) -> [1, embedSize]
    final state = agent.representation([0], tracker);

    // f_value(state) -> [1, 1] in (-1, 1)
    final valuePred = agent.predictValue(state, tracker);

    // MSE: (pred - target)^2  (already a 1x1 tensor, no reduce needed)
    final diff = valuePred - targetTensor;
    final loss = diff.pow(2.0);
    tracker.addAll([diff, loss]);

    loss.backward();
    opt.step();

    finalPred = valuePred.fetchData()[0];
    finalLoss = loss.fetchData()[0];

    if (epoch == 1 || epoch % 10 == 0 || finalLoss.abs() < tol * tol) {
      print(
        '  epoch ${epoch.toString().padLeft(4)} | '
        'pred=${finalPred.toStringAsFixed(6)} | '
        'target=${target.toStringAsFixed(6)} | '
        'loss=${finalLoss.toStringAsExponential(3)}',
      );
    }

    for (final t in tracker) {
      t.dispose();
    }

    if ((finalPred - target).abs() < tol) {
      print('✅ converged within tol=$tol at epoch $epoch');
      break;
    }
  }

  print(
    '\nFinal: pred=${finalPred.toStringAsFixed(6)}, '
    'target=${target.toStringAsFixed(6)}, '
    '|err|=${(finalPred - target).abs().toStringAsExponential(3)}',
  );

  // Cleanup
  targetTensor.dispose();
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
