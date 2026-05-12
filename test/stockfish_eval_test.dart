// Integration test for the Stockfish-eval value target used by the
// MuZero trainer on unfinished games. Spawns the bundled `tools/stockfish`
// binary, asks it to evaluate a non-terminal position, parses the same
// `info ... score cp|mate ...` lines the trainer parses, and asserts the
// resulting value-for-white falls inside the [-1, 1] range the value
// head expects.
//
// Skipped if `tools/stockfish` is not present (e.g. on a fresh clone
// where the binary hasn't been built yet).

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:test/test.dart';

const _sfPath = 'tools/stockfish';

/// Runs Stockfish on `position startpos moves <moves>` for [movetimeMs]
/// and returns the value from white's POV in [-1, 1], using the same
/// conversion the trainer applies (cp -> tanh(cp/600), mate -> +/-0.99).
Future<double> _evalForWhite(
  Process sf,
  Stream<String> lines, {
  List<String>? moves,
  String? fen,
  required int movetimeMs,
  required bool whiteToMove,
}) async {
  void send(String s) => sf.stdin.writeln(s);

  final String pos;
  if (fen != null) {
    pos = 'position fen $fen';
  } else {
    final mv = moves ?? const <String>[];
    pos = mv.isEmpty
        ? 'position startpos'
        : 'position startpos moves ${mv.join(' ')}';
  }
  send(pos);
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

  double mover;
  if (lastMate != null) {
    mover = lastMate! >= 0 ? 0.99 : -0.99;
  } else if (lastCp != null) {
    final x = lastCp! / 600.0;
    final ep = math.exp(x);
    final en = math.exp(-x);
    mover = (ep - en) / (ep + en);
  } else {
    mover = 0.0;
  }
  return whiteToMove ? mover : -mover;
}

void main() {
  final sfFile = File(_sfPath);
  final hasStockfish = sfFile.existsSync();

  group(
    'Stockfish eval on unfinished games',
    () {
      late Process sf;
      late StreamController<String> linesCtrl;
      late Stream<String> lines;

      setUp(() async {
        sf = await Process.start(_sfPath, const []);
        linesCtrl = StreamController<String>.broadcast();
        sf.stdout
            .transform(const Utf8Decoder())
            .transform(const LineSplitter())
            .listen(linesCtrl.add);
        sf.stderr.drain<void>();
        lines = linesCtrl.stream;

        sf.stdin.writeln('uci');
        await lines.firstWhere((l) => l.trim() == 'uciok');
        sf.stdin.writeln('isready');
        await lines.firstWhere((l) => l.trim() == 'readyok');
      });

      tearDown(() async {
        try {
          sf.stdin.writeln('quit');
          await sf.stdin.close();
        } catch (_) {}
        await sf.exitCode.timeout(
          const Duration(seconds: 3),
          onTimeout: () {
            sf.kill();
            return -1;
          },
        );
        await linesCtrl.close();
      });

      test('startpos returns a finite value in [-1, 1]', () async {
        final v = await _evalForWhite(
          sf,
          lines,
          moves: const [],
          movetimeMs: 200,
          whiteToMove: true,
        );
        expect(v.isFinite, isTrue);
        expect(v, inInclusiveRange(-1.0, 1.0));
        // Opening should be roughly balanced.
        expect(v.abs(), lessThan(0.5));
      });

      test('after 1.e4 e5 the eval is balanced and in range', () async {
        final v = await _evalForWhite(
          sf,
          lines,
          moves: const ['e2e4', 'e7e5'],
          movetimeMs: 200,
          whiteToMove: true,
        );
        expect(v.isFinite, isTrue);
        expect(v, inInclusiveRange(-1.0, 1.0));
        expect(v.abs(), lessThan(0.5));
      });

      test(
        'lopsided material (white up a queen) yields strongly positive eval',
        () async {
          // White: K on e1 + Q on d1. Black: K on e8 only. White to move.
          // Trivially winning; eval should be clearly positive.
          const fen = '4k3/8/8/8/8/8/8/3QK3 w - - 0 1';
          final v = await _evalForWhite(
            sf,
            lines,
            fen: fen,
            movetimeMs: 300,
            whiteToMove: true,
          );
          expect(v.isFinite, isTrue);
          expect(v, inInclusiveRange(-1.0, 1.0));
          expect(v, greaterThan(0.3));
        },
      );
    },
    skip: hasStockfish
        ? null
        : 'tools/stockfish not present; run `cd tools/sf/stockfish/src && make build` first',
  );
}
