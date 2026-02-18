import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';

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

void main() async {
  print("♟️ GPU Chess AFT-GPT: Full Power Mode (Engine-Stabilized)");

  const int vocabSize = 4098;
  const int bigSize = 16; // SHRUNK: From 128 to 16 (Minimum for 4 heads)
  const int blockSize =
      8; // SHRUNK: From 16 to 8 (Your games are only 6 tokens long)

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 1, // SHRUNK: From 4 to 1
    numHeads: 2, // SHRUNK: From 4 to 2
    blockSize: blockSize,
  );

  // 1. STANDARD INITIALIZATION
  // 0.02 is the gold standard for Transformers. Since your C++ engine is now
  // numerically stable, we can use a stronger starting signal.
  final rand = math.Random();
  for (var p in gpt.parameters()) {
    // MUCH smaller scale. We want the model to start almost "blank"
    p.data = List.generate(
      p.length,
      (_) => (rand.nextDouble() * 2 - 1) * 0.0001,
    );
  }

  // 2. OPTIMIZER
  // Using the new Adam kernel with Weight Decay (handled on C++ side)
  final optimizer = Adam(gpt.parameters(), lr: 0.001);
  final dummyEnc = Tensor.zeros([1, bigSize]);

  final dataset = [
    ["<start>", "e2e4", "e7e5", "g1f3", "b8c6", "."],
    ["<start>", "d2d4", "d7d5", "c2c4", "e7e6", "."],
  ].map((seq) => seq.map(encodeMove).toList()).toList();

  print("🚀 Training... (Stable LSE kernels active)");

  for (int epoch = 0; epoch <= 500; epoch++) {
    double epochLoss = 0;

    for (var seq in dataset) {
      List<Tensor> tracker = [];
      optimizer.zeroGrad();

      final x = seq.sublist(0, seq.length - 1);
      final y = seq.sublist(1);

      // FORWARD: Clean and direct. No CPU-intervention "jackets".
      final logits = gpt.forward(x, dummyEnc, tracker);

      // LOSS: The GPU now handles the max-subtraction internally!
      final loss = logits.crossEntropy(y);
      final lossVal = loss.fetchData()[0];

      if (!lossVal.isNaN && !lossVal.isInfinite) {
        loss.backward();
        optimizer.step();
        epochLoss += lossVal;
      }

      for (var t in tracker) t.dispose();
      loss.dispose();
      logits.dispose();
    }

    if (epoch % 50 == 0) {
      print(
        "Epoch $epoch | Loss: ${(epochLoss / dataset.length).toStringAsFixed(6)}",
      );
    }
  }

  print("\n--- Training Complete. Sampling Moves ---");
  generate(gpt, [4096], 4097, vocabSize, blockSize, dummyEnc);
}

// ... [Keep generate and sampleLocal as they were] ...

void generate(
  TransformerDecoder model,
  List<int> gen,
  int endId,
  int vocabSize,
  int blockSize,
  Tensor dummyEnc,
) {
  for (int i = 0; i < 10; i++) {
    List<Tensor> tracker = [];
    List<int> context = gen.length > blockSize
        ? gen.sublist(gen.length - blockSize)
        : gen;

    final logits = model.forward(context, dummyEnc, tracker);
    List<double> lastLogits = logits.fetchRow(context.length - 1);

    int nextId = sampleLocal(lastLogits, 0.1);
    gen.add(nextId);
    print("Move $i -> ${decodeMove(nextId)}");

    for (var t in tracker) t.dispose();
    logits.dispose();
    if (nextId == endId) break;
  }
}

int sampleLocal(List<double> row, double temp) {
  double maxL = row.reduce(math.max);
  List<double> exps = row.map((v) => math.exp((v - maxL) / temp)).toList();
  double sumExp = exps.reduce((a, b) => a + b);
  double r = math.Random().nextDouble() * sumExp;
  double cumulative = 0;
  for (int i = 0; i < exps.length; i++) {
    cumulative += exps[i];
    if (r <= cumulative) return i;
  }
  return 0;
}
