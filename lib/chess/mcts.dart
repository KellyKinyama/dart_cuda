// mcts3.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';

import '../gpu_tensor.dart';

// --- CHESS UTILITIES ---

int encodeMove(String uci) {
  if (uci == "<start>") return 4096;
  if (uci == ".") return 4097;
  int sqToIdx(String sq) {
    int file = sq.codeUnitAt(0) - 'a'.codeUnitAt(0);
    int rank = int.parse(sq[1]) - 1;
    return rank * 8 + file;
  }

  return (sqToIdx(uci.substring(0, 2)) * 64) + sqToIdx(uci.substring(2, 4));
}

String decodeMove(int index) {
  if (index == 4096) return "<start>";
  if (index == 4097) return ".";
  String idxToSq(int idx) {
    return String.fromCharCode('a'.codeUnitAt(0) + (idx % 8)) +
        (idx ~/ 8 + 1).toString();
  }

  return idxToSq(index ~/ 64) + idxToSq(index % 64);
}

class Mcts {
  Node rootNode;
  final int vocabSize = 4098;
  final int bigSize = 16; // SHRUNK: From 128 to 16 (Minimum for 4 heads)
  final int blockSize =
      16; // SHRUNK: From 16 to 8 (Your games are only 6 tokens long)

  final TransformerDecoder gpt;

  final Tensor dummyEnc; // = Tensor.zeros([1, bigSize]);

  // final Map<String, int> stoi;
  // final Map<int, String> itos;
  final Game game;

  Mcts(this.game, this.gpt)
    : rootNode = Node(game.state.hash),
      dummyEnc = Tensor.zeros([
        1, 16, //blockSize
      ]);

  Future<EngineResult> monteCarlo({
    int maxIterations = 5000,
    int timeLimit = 5000,
  }) async {
    final int endTime = DateTime.now().millisecondsSinceEpoch + timeLimit;

    rootNode = Node(game.state.hash);

    for (int i = 0; i < maxIterations; i++) {
      if (DateTime.now().millisecondsSinceEpoch > endTime) {
        print('MCTS search timed out after $i iterations.');
        break;
      }

      // 1. Selection
      List<Edge> path = [];
      Node currentNode = rootNode;
      Game searchGame = Game.fromPgn(game.pgn());

      while (!searchGame.gameOver && currentNode.children.isNotEmpty) {
        Edge bestEdge = _selectBestChild(currentNode, searchGame);
        path.add(bestEdge);
        searchGame.makeMove(bestEdge.move, false);
        currentNode = _getNode(searchGame, currentNode, bestEdge.move);
      }

      // 2. Expansion
      if (!searchGame.gameOver) {
        await _expandNode(currentNode, searchGame);
      }

      // 3. Simulation
      final double simulationResult = _simulate(searchGame);

      // 4. Backpropagation
      _backpropagate(path, simulationResult);
    }

    if (rootNode.children.isEmpty) {
      return const EngineResult();
    }

    Edge bestEdge = rootNode.children.reduce(
      (a, b) => a.visits > b.visits ? a : b,
    );

    return EngineResult(
      move: bestEdge.move,
      eval: bestEdge.actionValue,
      depth: bestEdge.visits,
    );
  }

  Node _getNode(Game searchGame, Node parentNode, Move move) {
    final hash = searchGame.state.hash;
    final existingEdge = parentNode.children.firstWhere(
      (edge) => edge.move == move,
      orElse: () => throw Exception('Node for move not found'),
    );
    if (existingEdge.childNode == null) {
      existingEdge.childNode = Node(hash);
    }
    return existingEdge.childNode!;
  }

  Edge _selectBestChild(Node node, Game game) {
    double bestScore = -double.infinity;
    Edge? bestEdge;

    // The score is based on the current player. If the Q-value is positive,
    // it's good for the player who just moved (the parent node's player).
    // The child node's player wants to minimize this value (maximize their own score).
    for (Edge edge in node.children) {
      final qValue = edge.visits == 0 ? 0.0 : edge.actionValue / edge.visits;
      final score = qValue + _calculatePuctScore(node, edge, 1.0);
      if (score > bestScore) {
        bestScore = score;
        bestEdge = edge;
      }
    }
    return bestEdge!;
  }

  Future<void> _expandNode(Node node, Game game) async {
    final List<Move> legalMoves = game.generateLegalMoves();
    final List<String> history = game.moveHistoryAlgebraic;

    List<Tensor> tracker = [];

    final List<int> inputTokens = [4096];
    if (history.isNotEmpty) {
      inputTokens.addAll(history.map((move) => encodeMove(move)));
    }

    List<int> context = inputTokens.length > blockSize
        ? inputTokens.sublist(inputTokens.length - blockSize)
        : inputTokens;

    final logits = gpt.forward(context, dummyEnc, tracker);
    final lastLogits = logits.softmax();

    List<double> probabilities = lastLogits.fetchRow(context.length - 1);

    // final probabilities = lastLogits.softmax();

    for (Move move in legalMoves) {
      final String moveAlgebraic = game.toAlgebraic(move);
      final int? moveTokenId = encodeMove(moveAlgebraic);
      if (moveTokenId != null) {
        final double priorProb = probabilities[moveTokenId];
        Edge edge = Edge(move, priorProb);
        node.children.add(edge);
      }
    }

    for (var t in tracker) t.dispose();
    logits.dispose();
  }

  double _simulate(Game game) {
    Game rolloutGame = Game.fromPgn(game.pgn());
    while (!rolloutGame.gameOver) {
      final legalMoves = rolloutGame.generateLegalMoves();
      if (legalMoves.isEmpty) break;
      final randomMove = legalMoves[Random().nextInt(legalMoves.length)];
      rolloutGame.makeMove(randomMove);
    }

    final int? winner = rolloutGame.winner;
    if (winner == game.turn) {
      return 1.0;
    } else if (winner == null) {
      return 0.0;
    } else {
      return -1.0;
    }
  }

  void _backpropagate(List<Edge> path, double reward) {
    for (Edge edge in path.reversed) {
      edge.visits++;
      edge.actionValue += reward;
      reward = -reward; // Invert the reward for the previous player
    }
    rootNode.visitCount++;
  }

  double _calculatePuctScore(Node node, Edge edge, double c_puct) {
    final q_value = edge.visits == 0 ? 0.0 : edge.actionValue / edge.visits;
    final explorationTerm =
        c_puct * edge.reward * sqrt(node.visitCount) / (1 + edge.visits);
    return q_value + explorationTerm;
  }
}

class Node {
  final int hash;
  int visitCount;
  List<Edge> children;

  Node(this.hash) : visitCount = 0, children = [];
}

class Edge {
  final Move move;
  double reward;
  double actionValue;
  int visits;
  Node? childNode;

  Edge(this.move, this.reward) : actionValue = 0.0, visits = 0;
}

class EngineResult {
  final Move? move;
  final double? eval;
  final int? depth;
  bool get hasMove => move != null;
  const EngineResult({this.move, this.eval, this.depth});
}
