import 'dart:math' as math;

import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';

void main() {
  print("🚀 Starting GPU-Accelerated AFT-GPT Test Drive...");

  const int vocabSize = 25;
  const int bigSize = 128;
  const int blockSize = 16;

  final stoi = {
    "hello": 0,
    "world": 1,
    "the": 2,
    "quick": 3,
    "brown": 4,
    "fox": 5,
    ".": 6,
    "<start>": 7,
    "jumps": 8,
    "over": 9,
    "lazy": 10,
    "dog": 11,
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 4,
    numHeads: 4,
    blockSize: blockSize,
  );

  final optimizer = Adam(gpt.parameters(), lr: 0.001);
  final dummyEnc = Tensor.zeros([1, bigSize]); // Encoder memory on GPU

  final dataset = [
    [7, 0, 1, 6], // <start> hello world .
    [7, 2, 3, 4, 5, 8, 9, 2, 10, 11, 6], // <start> the quick brown fox .
  ];

  for (int epoch = 0; epoch <= 500; epoch++) {
    double epochLoss = 0;

    for (var seq in dataset) {
      // TRACKER: Essential for GPU memory management
      List<Tensor> tracker = [];

      optimizer.zeroGrad();

      final x = seq.sublist(0, seq.length - 1);
      final y = seq.sublist(1);

      // 1. Forward (Entirely on GPU)
      final logits = gpt.forward(x, dummyEnc, tracker);

      // 2. Loss (Using the CUDA CrossEntropy kernel)
      final loss = logits.crossEntropy(y);

      // 3. Backward & Step
      loss.backward();
      optimizer.step();

      // Fetch only the scalar loss value to CPU for logging
      epochLoss += loss.fetchData()[0];

      // 4. DISPOSE: Clear intermediate VRAM
      for (var t in tracker) t.dispose();
      loss.dispose();
    }

    if (epoch % 50 == 0) {
      print(
        "Epoch $epoch | GPU Loss: ${(epochLoss / dataset.length).toStringAsFixed(6)}",
      );
    }
  }

  print("\n--- Model Trained. Sampling from GPU ---");
  generate(
    gpt,
    [stoi["<start>"]!],
    stoi["."]!,
    itos,
    vocabSize,
    blockSize,
    dummyEnc,
  );
}

void generate(
  TransformerDecoder model,
  List<int> gen,
  int endId,
  Map<int, String> itos,
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

    // Fetch only the last row of logits to CPU for sampling
    List<double> lastLogits = logits.fetchRow(context.length - 1);
    int nextId = sampleLocal(lastLogits, 0.1);

    gen.add(nextId);
    print("Next -> ${itos[nextId]}");

    for (var t in tracker) t.dispose();
    logits.dispose();
    if (nextId == endId) break;
  }
  print("Final: ${gen.map((id) => itos[id]).join(' ')}");
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
