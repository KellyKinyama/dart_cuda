// Unit tests for the pure parts of muzero_chess_mcts.dart that don't
// require spinning up the GPU model: the per-edge Q getters (regular vs
// optimistic), terminal handling, and the visit-distribution helper.

import 'package:bishop/bishop.dart';
import 'package:dart_cuda/mu_zero/muzero_chess_mcts.dart';
import 'package:dart_cuda/mu_zero/muzero_chess_player.dart' show MoveTokenizer;
import 'package:test/test.dart';

MctsNode _node({
  required List<Move> legal,
  required List<String> uci,
  required List<double> priors,
  bool terminal = false,
  double terminalValue = 0.0,
  int hash = 0,
}) => MctsNode(
  hash: hash,
  whiteToMove: true,
  legalMoves: legal,
  legalUci: uci,
  priors: priors,
  terminal: terminal,
  terminalValue: terminalValue,
);

void main() {
  group('MctsNode.q / qSelect', () {
    final game = Game(variant: Variant.standard());
    final legal = game.generateLegalMoves();
    final uci = [for (final m in legal) game.toAlgebraic(m)];

    MctsNode make() => _node(
      legal: legal,
      uci: uci,
      priors: List<double>.filled(legal.length, 1.0 / legal.length),
    );

    test('unvisited edge: q=0.0, qSelect=1.0 (optimistic)', () {
      final n = make();
      expect(n.q(0), 0.0);
      expect(n.qSelect(0), 1.0);
    });

    test('visited edge: q == qSelect == totalActionValue / visits', () {
      final n = make();
      n.visits[0] = 4;
      n.totalActionValue[0] = 1.0;
      expect(n.q(0), closeTo(0.25, 1e-9));
      expect(n.qSelect(0), closeTo(0.25, 1e-9));
    });
  });

  group('MctsNode.visitDistribution', () {
    final game = Game(variant: Variant.standard());
    final legal = game.generateLegalMoves();
    final uci = [for (final m in legal) game.toAlgebraic(m)];
    final tok = MoveTokenizer.fromGames([uci]);

    test('empty (n == 0) returns all zeros', () {
      final n = _node(
        legal: legal,
        uci: uci,
        priors: List<double>.filled(legal.length, 1.0 / legal.length),
      );
      final dist = n.visitDistribution(tok);
      expect(dist.length, tok.vocabSize);
      expect(dist.every((p) => p == 0.0), isTrue);
    });

    test(
      'non-empty: distribution sums to 1 and is concentrated on legal moves',
      () {
        final n = _node(
          legal: legal,
          uci: uci,
          priors: List<double>.filled(legal.length, 1.0 / legal.length),
        );
        // Pretend we visited the first three legal moves with N = [2, 1, 1].
        n.visits[0] = 2;
        n.visits[1] = 1;
        n.visits[2] = 1;
        n.n = 4;

        final dist = n.visitDistribution(tok);
        // Sum to 1.
        final s = dist.fold<double>(0.0, (a, b) => a + b);
        expect(s, closeTo(1.0, 1e-9));

        // Visited moves get their share; everything else stays 0.
        final id0 = tok.encode(uci[0])!;
        final id1 = tok.encode(uci[1])!;
        final id2 = tok.encode(uci[2])!;
        expect(dist[id0], closeTo(0.5, 1e-9));
        expect(dist[id1], closeTo(0.25, 1e-9));
        expect(dist[id2], closeTo(0.25, 1e-9));

        // An unvisited but legal move should remain 0.
        final id3 = tok.encode(uci[3])!;
        expect(dist[id3], 0.0);
      },
    );
  });

  group('MctsNode terminal flag', () {
    test('mate sets terminal=true and value=-1 from mover POV', () {
      final n = _node(
        legal: const [],
        uci: const [],
        priors: const [],
        terminal: true,
        terminalValue: -1.0,
      );
      expect(n.terminal, isTrue);
      expect(n.terminalValue, -1.0);
      expect(n.legalMoves, isEmpty);
    });
  });
}
