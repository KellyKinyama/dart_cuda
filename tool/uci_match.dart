// UCI match harness: plays a game between two UCI engines via stdio.
//
// Used to verify Bishop's UCI implementation against a real engine
// (e.g. Stockfish). Engine A plays White, Engine B plays Black by default.
//
// Usage:
//   dart run tool/uci_match.dart <engineA> <engineB> [--movetime=300] [--maxply=200]
//
// Example:
//   dart run tool/uci_match.dart "dart run bin/uci.dart" tools/stockfish.exe

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:bishop/bishop.dart';

class UciClient {
  final String label;
  final Process process;
  final StreamController<String> _lines = StreamController.broadcast();
  late final StreamSubscription _stdoutSub;
  late final StreamSubscription _stderrSub;

  UciClient._(this.label, this.process) {
    _stdoutSub = process.stdout
        .transform(const Utf8Decoder())
        .transform(const LineSplitter())
        .listen(_lines.add);
    _stderrSub = process.stderr
        .transform(const Utf8Decoder())
        .transform(const LineSplitter())
        .listen((l) => stderr.writeln('[$label STDERR] $l'));
  }

  static Future<UciClient> spawn(String label, String command) async {
    // Support quoted multi-arg commands (e.g. "dart run bin/uci.dart").
    final parts = _tokenize(command);
    final exe = parts.first;
    final args = parts.skip(1).toList();
    final proc = await Process.start(
      exe,
      args,
      mode: ProcessStartMode.normal,
      runInShell: true,
    );
    return UciClient._(label, proc);
  }

  static List<String> _tokenize(String s) {
    // Minimal: split on whitespace; no shell quoting support needed for our use.
    return s.trim().split(RegExp(r'\s+'));
  }

  void send(String cmd) {
    stdout.writeln('  >> [$label] $cmd');
    process.stdin.writeln(cmd);
  }

  /// Waits for a line that satisfies [match]. Returns the matching line.
  Future<String> waitFor(
    bool Function(String) match, {
    Duration timeout = const Duration(seconds: 30),
  }) {
    final completer = Completer<String>();
    late StreamSubscription sub;
    sub = _lines.stream.listen((line) {
      stdout.writeln('  << [$label] $line');
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

  Future<void> handshake() async {
    send('uci');
    await waitFor((l) => l.trim() == 'uciok');
    send('isready');
    await waitFor((l) => l.trim() == 'readyok');
  }

  Future<String> bestmove({
    required String positionCmd,
    required int movetimeMs,
  }) async {
    send(positionCmd);
    send('isready');
    await waitFor((l) => l.trim() == 'readyok');
    send('go movetime $movetimeMs');
    final line = await waitFor(
      (l) => l.startsWith('bestmove'),
      timeout: Duration(milliseconds: movetimeMs * 5 + 5000),
    );
    return line.split(RegExp(r'\s+'))[1];
  }

  Future<void> quit() async {
    try {
      send('quit');
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
    stdout.writeln('[$label] exited with code $exit');
  }
}

/// Renders a standard-chess FEN as an 8x8 ASCII board with rank/file labels.
/// Uses unicode chess glyphs for clarity in modern terminals.
String _renderBoard(String fen) {
  const glyphs = {
    'P': '\u2659',
    'N': '\u2658',
    'B': '\u2657',
    'R': '\u2656',
    'Q': '\u2655',
    'K': '\u2654',
    'p': '\u265F',
    'n': '\u265E',
    'b': '\u265D',
    'r': '\u265C',
    'q': '\u265B',
    'k': '\u265A',
  };
  final boardField = fen.split(' ').first;
  final ranks = boardField.split('/');
  final buf = StringBuffer();
  buf.writeln('  +------------------------+');
  for (int r = 0; r < ranks.length; r++) {
    final rankNum = ranks.length - r;
    buf.write('$rankNum |');
    for (final ch in ranks[r].split('')) {
      final n = int.tryParse(ch);
      if (n != null) {
        for (int i = 0; i < n; i++) {
          buf.write(' . ');
        }
      } else {
        buf.write(' ${glyphs[ch] ?? ch} ');
      }
    }
    buf.writeln('|');
  }
  buf.writeln('  +------------------------+');
  buf.writeln('    a  b  c  d  e  f  g  h');
  return buf.toString();
}

Future<int> main(List<String> args) async {
  if (args.length < 2) {
    stderr.writeln(
      'Usage: dart run tool/uci_match.dart <engineA> <engineB> [--movetime=MS] [--maxply=N] [--skill=N]',
    );
    return 64;
  }

  final cmdA = args[0];
  final cmdB = args[1];
  int movetime = 300;
  int? maxPly; // null = play to completion
  int? skillLevel; // For Stockfish: 0..20

  for (final a in args.skip(2)) {
    if (a.startsWith('--movetime=')) {
      movetime = int.parse(a.substring('--movetime='.length));
    } else if (a.startsWith('--maxply=')) {
      maxPly = int.parse(a.substring('--maxply='.length));
    } else if (a.startsWith('--skill=')) {
      skillLevel = int.parse(a.substring('--skill='.length));
    }
  }

  stdout.writeln('Engine A (White): $cmdA');
  stdout.writeln('Engine B (Black): $cmdB');
  stdout.writeln(
    'movetime=${movetime}ms, '
    'maxply=${maxPly ?? 'unlimited (play to completion)'}'
    '${skillLevel != null ? ', skill=$skillLevel' : ''}',
  );
  stdout.writeln('');

  final a = await UciClient.spawn('A', cmdA);
  final b = await UciClient.spawn('B', cmdB);

  await a.handshake();
  await b.handshake();

  if (skillLevel != null) {
    // Best-effort: apply to whichever engine accepts it (Stockfish only).
    a.send('setoption name Skill Level value $skillLevel');
    b.send('setoption name Skill Level value $skillLevel');
  }

  a.send('ucinewgame');
  b.send('ucinewgame');

  final game = Game(variant: Variant.standard());
  final moves = <String>[];
  String? resultStr;
  String? terminationReason;

  for (int ply = 0; maxPly == null || ply < maxPly; ply++) {
    if (game.gameOver) {
      final winner = game.winner;
      if (winner == null) {
        resultStr = '1/2-1/2';
        terminationReason = 'draw';
      } else if (winner == Bishop.white) {
        resultStr = '1-0';
        terminationReason = 'white wins';
      } else {
        resultStr = '0-1';
        terminationReason = 'black wins';
      }
      break;
    }

    final mover = (ply % 2 == 0) ? a : b;
    final moverLabel = (ply % 2 == 0) ? 'White (A)' : 'Black (B)';
    final posCmd = moves.isEmpty
        ? 'position startpos'
        : 'position startpos moves ${moves.join(' ')}';

    final String moveUci;
    try {
      moveUci = await mover.bestmove(positionCmd: posCmd, movetimeMs: movetime);
    } catch (e) {
      stderr.writeln('[$moverLabel] failed to produce a move: $e');
      resultStr = (ply % 2 == 0) ? '0-1' : '1-0';
      terminationReason = '$moverLabel crashed/timeout';
      break;
    }

    if (moveUci == '0000' || moveUci.isEmpty) {
      // Engine reports no move — treat as game over / loss for that side.
      resultStr = (ply % 2 == 0) ? '0-1' : '1-0';
      terminationReason = '$moverLabel returned $moveUci';
      break;
    }

    final mv = game.getMove(moveUci);
    if (mv == null) {
      stderr.writeln(
        '[$moverLabel] produced ILLEGAL move "$moveUci" in position $posCmd',
      );
      resultStr = (ply % 2 == 0) ? '0-1' : '1-0';
      terminationReason = '$moverLabel illegal move $moveUci';
      break;
    }
    game.makeMove(mv, false);
    moves.add(moveUci);
    stdout.writeln('');
    stdout.writeln('Ply ${ply + 1}: $moverLabel plays $moveUci');
    stdout.writeln(_renderBoard(game.fen));
  }

  if (resultStr == null) {
    resultStr = '*';
    terminationReason = 'ply limit reached';
  }

  stdout.writeln('');
  stdout.writeln('===== MATCH SUMMARY =====');
  stdout.writeln('Result: $resultStr ($terminationReason)');
  stdout.writeln('Plies played: ${moves.length}');
  stdout.writeln('Moves: ${moves.join(' ')}');
  stdout.writeln('Final FEN: ${game.fen}');
  stdout.writeln('');
  stdout.writeln('Final position:');
  stdout.writeln(_renderBoard(game.fen));

  await a.quit();
  await b.quit();
  return 0;
}
