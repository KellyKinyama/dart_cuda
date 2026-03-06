// uci.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:bishop/bishop.dart';

import 'package:dart_cuda/aft_transformer_decoder.dart';
// import 'gpt_model2.dart';
import '../network_utils.dart';
import 'mcts.dart' as mc;

class UciEngine {
  late Game game;
  mc.Mcts? mcts;
  bool isSearching = false;
  // final GptModel gpt;

  TransformerDecoder gpt;

  UciEngine(this.gpt) {
    game = Game();

    mcts = mc.Mcts(game, gpt);
  }

  void startLoop() {
    stdin.transform(Utf8Decoder()).transform(const LineSplitter()).listen((
      line,
    ) {
      handleCommand(line);
    });
  }

  void handleCommand(String command) {
    final parts = command.split(' ');
    final cmd = parts[0];

    switch (cmd) {
      case 'uci':
        stdout.writeln('id name MyDartMctsEngine');
        stdout.writeln('id author YourName');
        stdout.writeln('uciok');
        break;
      case 'isready':
        stdout.writeln('readyok');
        break;
      case 'ucinewgame':
        game = Game();
        mcts = mc.Mcts(game, gpt);
        break;
      case 'position':
        handlePosition(parts);
        break;
      case 'go':
        handleGo(parts);
        break;
      case 'stop':
        isSearching = false;
        break;
      case 'quit':
        exit(0);
      // New command for training on the current game history
      case 'train':
        // handleTrain();
        break;
      default:
        break;
    }
  }

  void handlePosition(List<String> parts) {
    int movesIndex = parts.indexOf('moves');
    String fen = '';

    if (parts[1] == 'startpos') {
      fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    } else if (parts[1] == 'fen') {
      fen = parts
          .sublist(2, movesIndex != -1 ? movesIndex : parts.length)
          .join(' ');
    }

    game = Game(variant: null);
    game.setup(fen: fen);

    // Re-initialize the MCTS engine with the updated game object
    mcts = mc.Mcts(game, gpt);

    if (movesIndex != -1) {
      List<String> moves = parts.sublist(movesIndex + 1);
      for (String moveString in moves) {
        Move? move = game.getMove(moveString);
        if (move != null) {
          game.makeMove(move);
        }
      }
    }
  }

  void handleGo(List<String> parts) async {
    int moveTime = 5000;
    int movetimeIndex = parts.indexOf('movetime');
    if (movetimeIndex != -1 && movetimeIndex + 1 < parts.length) {
      moveTime = int.tryParse(parts[movetimeIndex + 1]) ?? 5000;
    }
    isSearching = true;

    if (mcts != null) {
      // Pass the time limit to the monteCarlo function
      mc.EngineResult result = await mcts!.monteCarlo(timeLimit: moveTime);
      isSearching = false;

      if (result.hasMove) {
        stdout.writeln('bestmove ${game.toAlgebraic(result.move!)}');
      }
    }
  }

  // Future<void> handleTrain() async {
  //   print('Starting training on current game history...');
  //   await gpt.trainOnSequence(game.moveHistoryAlgebraic);
  //   print('Training complete.');
  // }
}

// --------------------------------------------------------------------------------------------------

Future<void> main() async {
  final int vocabSize = 4098;
  final int bigSize = 16; // SHRUNK: From 128 to 16 (Minimum for 4 heads)
  final int blockSize =
      16; // SHRUNK: From 16 to 8 (Your games are only 6 tokens long)
  TransformerDecoder gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 2, // SHRUNK: From 4 to 1
    numHeads: 4, // SHRUNK: From 4 to 2
    blockSize: blockSize,
  );

  const String weightPath = 'chess_gpt.bin';

  // 1. Try to restore previous state
  bool isLoaded = await loadModuleBinary(gpt, weightPath);
  final uciEngine = UciEngine(gpt);
  uciEngine.startLoop();
}
