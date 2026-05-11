// file: lib/mu_zero/muzero_chess_mcts.dart
//
// PUCT-style Monte Carlo Tree Search over real bishop positions, with
// node statistics stored in a transposition table keyed by Zobrist hash
// (`game.state.hash`). Priors come from the model's policy head; leaf
// values come from the model's value head. There is no model rollout
// (we re-use bishop for legal-move generation and result detection),
// which keeps the search deterministic and easy to debug.
//
// Goal: produce a stronger move than depth-1 policy argmax by spending
// extra forward passes on a few candidate lines, without changing the
// model. Suitable both for play and (later) for generating MuZero-style
// (state, visit-distribution) training targets.

import 'dart:math' as math;

import 'package:bishop/bishop.dart';

import '../gpu_tensor.dart';
import 'muzero_chess_player.dart';

/// A single node in the search tree. One node per *position* (Zobrist key),
/// not per (parent, move) pair — transpositions share the same node.
class MctsNode {
  final int hash;
  final bool whiteToMove;
  final List<Move> legalMoves;
  final List<String> legalUci; // bishop algebraic for each move
  final List<double> priors; // policy P(move) over legalMoves, sums to 1
  final List<int> visits;
  final List<double> totalActionValue; // sum of values returned by children
  int n; // total visit count for this node
  bool terminal; // true if game is over at this node
  double terminalValue; // value from the side-to-move's POV if terminal

  MctsNode({
    required this.hash,
    required this.whiteToMove,
    required this.legalMoves,
    required this.legalUci,
    required this.priors,
    required this.terminal,
    required this.terminalValue,
  }) : visits = List<int>.filled(legalMoves.length, 0),
       totalActionValue = List<double>.filled(legalMoves.length, 0.0),
       n = 0;

  double q(int i) => visits[i] == 0 ? 0.0 : totalActionValue[i] / visits[i];

  /// Distribution of visit counts over the action space of the tokenizer.
  /// Used as a soft policy target for MuZero-style training.
  List<double> visitDistribution(MoveTokenizer tok) {
    final out = List<double>.filled(tok.vocabSize, 0.0);
    if (n == 0) return out;
    for (int i = 0; i < legalMoves.length; i++) {
      final id = tok.encode(legalUci[i]);
      if (id == null) continue;
      out[id] = visits[i] / n;
    }
    return out;
  }
}

/// PUCT-style MCTS with Zobrist-keyed transposition table.
class ZobristMcts {
  final ChessMuZeroAgent agent;
  final MoveTokenizer tok;
  final int blockSize;
  final double cPuct;
  final math.Random rng;
  final Map<int, MctsNode> _table = {};

  ZobristMcts(
    this.agent,
    this.tok, {
    required this.blockSize,
    this.cPuct = 1.4,
    math.Random? rng,
  }) : rng = rng ?? math.Random();

  /// Drop the transposition table. Call between independent searches if
  /// memory matters.
  void clear() => _table.clear();

  int get nodeCount => _table.length;

  /// Run [numSimulations] PUCT simulations from the position currently
  /// held by [rootGame]. Returns the root node so callers can read its
  /// visit distribution.
  ///
  /// [history] is the tokenized move history (without `<start>`) that
  /// led to the root position; the model's representation needs it.
  MctsNode run({
    required Game rootGame,
    required List<int> history,
    required int numSimulations,
  }) {
    final rootFen = rootGame.fen;
    final root = _ensureNode(rootGame, history);
    for (int s = 0; s < numSimulations; s++) {
      _simulate(
        Game(variant: Variant.standard(), fen: rootFen),
        List<int>.from(history),
      );
    }
    return root;
  }

  /// One simulation: descend with PUCT until we hit a leaf (a node that
  /// hasn't been expanded yet) or a terminal position, then back-propagate
  /// the leaf value along the path. [game] and [history] are mutated.
  void _simulate(Game game, List<int> history) {
    final path = <(MctsNode, int)>[]; // (node, action index in node.legalMoves)
    MctsNode node = _ensureNode(game, history);

    while (!node.terminal) {
      final i = _selectAction(node);
      final mv = node.legalMoves[i];
      final uci = node.legalUci[i];
      path.add((node, i));

      game.makeMove(mv, false);
      final id = tok.encode(uci);
      if (id != null) history.add(id);

      final next = _table[game.state.hash];
      if (next == null) {
        // Leaf: create the node, then back-propagate its value.
        final leaf = _ensureNode(game, history);
        _backprop(path, leaf.terminal ? leaf.terminalValue : _valueOf(history));
        return;
      }
      node = next;
    }
    // Reached an already-expanded terminal via selection.
    _backprop(path, node.terminalValue);
  }

  /// PUCT score: Q(s,a) + cPuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
  int _selectAction(MctsNode node) {
    final sqrtN = math.sqrt(node.n.toDouble() + 1e-8);
    int bestI = 0;
    double bestScore = -double.infinity;
    for (int i = 0; i < node.legalMoves.length; i++) {
      final u = cPuct * node.priors[i] * sqrtN / (1 + node.visits[i]);
      final s = node.q(i) + u;
      if (s > bestScore) {
        bestScore = s;
        bestI = i;
      }
    }
    return bestI;
  }

  /// Back up [leafValue], flipping sign at every ply because the value
  /// is always from the side-to-move's POV.
  void _backprop(List<(MctsNode, int)> path, double leafValue) {
    double v = -leafValue; // last edge's parent moved; flip into parent POV
    for (int p = path.length - 1; p >= 0; p--) {
      final (node, i) = path[p];
      node.visits[i] += 1;
      node.totalActionValue[i] += v;
      node.n += 1;
      v = -v;
    }
  }

  /// Look up or create the node for the current position. Calls the
  /// model once to obtain priors and (for the side-to-move) the leaf
  /// value if it ever needs to be expanded.
  MctsNode _ensureNode(Game game, List<int> history) {
    final h = game.state.hash;
    final cached = _table[h];
    if (cached != null) return cached;

    final legal = game.generateLegalMoves();
    final whiteToMove = game.state.turn == Bishop.white;

    // Terminal detection. We must mark mate/draw before asking the model
    // for a policy because there are no legal moves to put priors on.
    if (game.gameOver) {
      double tv;
      if (game.checkmate) {
        // Side to move is mated → value = -1 for them.
        tv = -1.0;
      } else {
        tv = 0.0; // stalemate / draw rules
      }
      final node = MctsNode(
        hash: h,
        whiteToMove: whiteToMove,
        legalMoves: const [],
        legalUci: const [],
        priors: const [],
        terminal: true,
        terminalValue: tv,
      );
      _table[h] = node;
      return node;
    }

    if (legal.isEmpty) {
      // Defensive: bishop didn't flag gameOver but no moves available.
      final node = MctsNode(
        hash: h,
        whiteToMove: whiteToMove,
        legalMoves: const [],
        legalUci: const [],
        priors: const [],
        terminal: true,
        terminalValue: 0.0,
      );
      _table[h] = node;
      return node;
    }

    final uciList = [for (final m in legal) game.toAlgebraic(m)];
    final priors = _policyPriors(legal, uciList, history);
    final node = MctsNode(
      hash: h,
      whiteToMove: whiteToMove,
      legalMoves: legal,
      legalUci: uciList,
      priors: priors,
      terminal: false,
      terminalValue: 0.0,
    );
    _table[h] = node;
    return node;
  }

  /// Run the policy head once at the current history and return a
  /// softmax restricted to legal moves. Moves not in the tokenizer get
  /// a uniform fallback share.
  List<double> _policyPriors(
    List<Move> legal,
    List<String> uciList,
    List<int> history,
  ) {
    final tracker = <Tensor>[];
    final ctx = _trimCtx(history);
    final states = agent.representation(ctx, tracker);
    final logits = agent.predictPolicy(states, tracker);
    final row = logits.fetchRow(ctx.length - 1);

    final raw = List<double>.filled(legal.length, double.negativeInfinity);
    int knownCount = 0;
    for (int i = 0; i < legal.length; i++) {
      final id = tok.encode(uciList[i]);
      if (id != null) {
        raw[i] = row[id];
        knownCount++;
      }
    }
    _safeCleanupAgent(tracker);

    if (knownCount == 0) {
      // No legal move is in vocab → fall back to uniform priors.
      return List<double>.filled(legal.length, 1.0 / legal.length);
    }

    // Softmax over legal moves; unknown moves get a small floor share.
    double maxL = double.negativeInfinity;
    for (final v in raw) {
      if (v > maxL) maxL = v;
    }
    final exps = <double>[];
    double sum = 0.0;
    for (final v in raw) {
      final e = v.isFinite ? math.exp(v - maxL) : 0.0;
      exps.add(e);
      sum += e;
    }
    if (sum <= 0) {
      return List<double>.filled(legal.length, 1.0 / legal.length);
    }
    final priors = [for (final e in exps) e / sum];
    const floor = 1e-3;
    for (int i = 0; i < priors.length; i++) {
      if (priors[i] < floor) priors[i] = floor;
    }
    final norm = priors.fold<double>(0.0, (a, b) => a + b);
    for (int i = 0; i < priors.length; i++) {
      priors[i] /= norm;
    }
    return priors;
  }

  /// Run the value head once at the current history. Returns the value
  /// from the side-to-move's POV (already tanh-squashed in the agent).
  double _valueOf(List<int> history) {
    final tracker = <Tensor>[];
    final ctx = _trimCtx(history);
    final states = agent.representation(ctx, tracker);
    final values = agent.predictValue(states, tracker);
    final v = values.fetchRow(ctx.length - 1)[0];
    _safeCleanupAgent(tracker);
    return v;
  }

  List<int> _trimCtx(List<int> history) {
    final ctx = [tok.startId, ...history];
    return ctx.length <= blockSize ? ctx : ctx.sublist(ctx.length - blockSize);
  }

  void _safeCleanupAgent(List<Tensor> tracker) {
    final freed = <int>{};
    final paramAddrs = agent.parameters().map((p) => p.handle.address).toSet();
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
}

/// Pick the next move using PUCT MCTS over the current position. Returns
/// `null` if there is no legal move that is also in the tokenizer's
/// vocabulary. When [temperature] is 0 the most-visited action is
/// returned; otherwise a sample is drawn from `visits^(1/temperature)`.
({String uci, int id, Move move})? pickNextMoveMcts(
  ChessMuZeroAgent agent,
  MoveTokenizer tok,
  Game game,
  List<int> history,
  int blockSize, {
  required int numSimulations,
  double cPuct = 1.4,
  double temperature = 0.0,
  math.Random? rng,
}) {
  if (numSimulations <= 0) return null;
  final mcts = ZobristMcts(
    agent,
    tok,
    blockSize: blockSize,
    cPuct: cPuct,
    rng: rng,
  );
  final root = mcts.run(
    rootGame: game,
    history: history,
    numSimulations: numSimulations,
  );
  if (root.legalMoves.isEmpty) return null;

  int pickIdx;
  if (temperature <= 0.0) {
    pickIdx = 0;
    int bestN = -1;
    for (int i = 0; i < root.visits.length; i++) {
      if (root.visits[i] > bestN) {
        bestN = root.visits[i];
        pickIdx = i;
      }
    }
  } else {
    final r = rng ?? math.Random();
    final invT = 1.0 / temperature;
    final weights = [
      for (final v in root.visits) math.pow(v + 1e-9, invT).toDouble(),
    ];
    final total = weights.fold<double>(0.0, (a, b) => a + b);
    if (total <= 0) {
      pickIdx = 0;
    } else {
      final pick = r.nextDouble() * total;
      double acc = 0.0;
      pickIdx = weights.length - 1;
      for (int i = 0; i < weights.length; i++) {
        acc += weights[i];
        if (acc >= pick) {
          pickIdx = i;
          break;
        }
      }
    }
  }

  final uci = root.legalUci[pickIdx];
  final mv = root.legalMoves[pickIdx];
  final id = tok.encode(uci);
  if (id == null) {
    // Fall back to the highest-visit move whose UCI is in vocab.
    for (int i = 0; i < root.visits.length; i++) {
      final mid = tok.encode(root.legalUci[i]);
      if (mid != null) {
        return (uci: root.legalUci[i], id: mid, move: root.legalMoves[i]);
      }
    }
    return null;
  }
  return (uci: uci, id: id, move: mv);
}
