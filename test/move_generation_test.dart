// Move-generation accuracy tests.
//
// Two layers:
//   1. Bishop perft sanity: at known positions, bishop's
//      `generateLegalMoves()` count matches the published perft values.
//      This validates the move generator we delegate to.
//   2. Round-trip: every legal move at startpos can be encoded via
//      `game.toAlgebraic(...)` and re-decoded via `game.getMove(...)`,
//      and the algebraic form is the canonical UCI long-form (from-to[promo]).
//
// We do NOT instantiate the GPU model here; this file exercises only
// bishop + the small adapters our MCTS uses.

import 'package:bishop/bishop.dart';
import 'package:test/test.dart';

/// Recursive perft using bishop. Returns the number of leaf nodes at
/// depth [d] from the position currently held by [game].
int _perft(Game game, int d) {
  if (d == 0) return 1;
  final moves = game.generateLegalMoves();
  if (d == 1) return moves.length;
  int total = 0;
  for (final m in moves) {
    game.makeMove(m, false);
    total += _perft(game, d - 1);
    game.undo();
  }
  return total;
}

void main() {
  group('Bishop perft (move-generation correctness)', () {
    // https://www.chessprogramming.org/Perft_Results
    test('startpos depth 1 = 20', () {
      final g = Game(variant: Variant.standard());
      expect(_perft(g, 1), 20);
    });

    test('startpos depth 2 = 400', () {
      final g = Game(variant: Variant.standard());
      expect(_perft(g, 2), 400);
    });

    test('startpos depth 3 = 8902', () {
      final g = Game(variant: Variant.standard());
      expect(_perft(g, 3), 8902);
    });

    test('Kiwipete depth 1 = 48', () {
      // Famous tactical test position; depth-1 perft is exactly 48.
      const fen =
          'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      expect(_perft(g, 1), 48);
    });

    test('Kiwipete depth 2 = 2039', () {
      const fen =
          'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      expect(_perft(g, 2), 2039);
    });

    test('promotion position has exactly 4 pawn promotions', () {
      // White pawn on a7, otherwise empty (with kings). The only legal
      // pawn move is the promotion a7-a8 — which expands to four
      // promotion choices (Q, R, B, N).
      const fen = '8/P7/8/8/8/8/8/k6K w - - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      final moves = g.generateLegalMoves();
      final pawnMoves = moves.where((m) => g.toAlgebraic(m).startsWith('a7'));
      expect(pawnMoves.length, 4);
    });
  });

  group('Move encode/decode round-trip', () {
    test('every legal startpos move re-decodes to the same move', () {
      final g = Game(variant: Variant.standard());
      final legal = g.generateLegalMoves();
      for (final m in legal) {
        final alg = g.toAlgebraic(m);
        final back = g.getMove(alg);
        expect(back, isNotNull, reason: 'getMove($alg) returned null');
        expect(g.toAlgebraic(back!), alg);
      }
    });

    test('promotion algebraic includes the promotion piece suffix', () {
      const fen = '8/P7/8/8/8/8/8/k6K w - - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      final promoMoves = g
          .generateLegalMoves()
          .map(g.toAlgebraic)
          .where((s) => s.startsWith('a7a8'))
          .toList();
      expect(promoMoves.length, 4);
      // The four promotion choices must produce distinct algebraic
      // strings (e.g. a7a8q, a7a8r, a7a8b, a7a8n).
      expect(promoMoves.toSet().length, 4);
      for (final alg in promoMoves) {
        expect(alg.length, greaterThanOrEqualTo(5));
      }
    });

    test('castling algebraic is the king\'s two-square move', () {
      const fen = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      final algs = g.generateLegalMoves().map(g.toAlgebraic).toSet();
      // King-side: e1g1; queen-side: e1c1. Bishop reports them in UCI
      // long-form (from-to) when no promotion is involved.
      expect(algs.contains('e1g1'), isTrue, reason: 'no O-O move found');
      expect(algs.contains('e1c1'), isTrue, reason: 'no O-O-O move found');
    });
  });

  group('Terminal detection', () {
    test('fool\'s mate position is checkmate, no legal moves', () {
      // 1. f3 e5 2. g4 Qh4# — white is mated.
      final g = Game(variant: Variant.standard());
      for (final uci in const ['f2f3', 'e7e5', 'g2g4', 'd8h4']) {
        final m = g.getMove(uci);
        expect(m, isNotNull, reason: 'could not parse $uci');
        g.makeMove(m!, false);
      }
      expect(g.checkmate, isTrue);
      expect(g.gameOver, isTrue);
      expect(g.generateLegalMoves(), isEmpty);
    });

    test('stalemate has no legal moves but is not checkmate', () {
      // Classic K+Q stalemate: black king on a8, white king on c7, white queen on c8.
      // Wait — that's mate. Use a clean stalemate: black to move, k on h8,
      // white king on f7, white queen on g6 → all black king squares are
      // attacked but the king is not in check.
      const fen = '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1';
      final g = Game(variant: Variant.standard(), fen: fen);
      expect(g.generateLegalMoves(), isEmpty);
      expect(g.checkmate, isFalse);
      expect(g.gameOver, isTrue);
    });
  });
}
