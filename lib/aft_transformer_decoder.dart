import 'dart:math' as math;

import 'package:dart_cuda/adam.dart';

import 'aft_transformer_decoder_block.dart';
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart';

class TransformerDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;
  final int encoderEmbedSize;

  final Tensor wte; // [vocabSize, embedSize]
  final Tensor wpe; // [blockSize, embedSize]

  final List<TransformerDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  TransformerDecoder({
    this.vocabSize = 4098,
    this.embedSize = 128,
    this.blockSize = 16,
    this.numLayers = 4,
    this.numHeads = 4,
    this.encoderEmbedSize = 128,
  }) : wte = Tensor.random([vocabSize, embedSize]),
       wpe = Tensor.random([blockSize, embedSize]),
       blocks = List.generate(
         numLayers,
         (i) => TransformerDecoderBlock(
           embedSize,
           numHeads,
           encoderEmbedSize,
           blockSize,
         ),
       ),
       finalLayerNorm = LayerNorm(embedSize),
       lmHead = Layer(embedSize, vocabSize, useGelu: false) {
    // 1. Remove the .step(-0.02) hack.
    // Instead, let's use a proper Xavier/Normal distribution if your Tensor class allows,
    // or manually scale the random data on the CPU once.

    List<double> rawWte = wte.fetchData();
    final rand = math.Random();
    for (int i = 0; i < rawWte.length; i++) {
      rawWte[i] = (rand.nextDouble() * 2 - 1) * 0.02; // Range [-0.02, 0.02]
    }
    wte.data = rawWte;

    // 2. CRITICAL: Zero out the lmHead bias
    // This forces the model to use the embeddings and attention to differentiate moves.
    final params = lmHead.parameters();
    if (params.length > 1) {
      // Assuming params[0] is weights and params[1] is bias
      Tensor bias = params[1];
      bias.data = List.filled(bias.length, 0.0);
      print("🎯 lmHead bias zeroed to prevent index-collapse.");
    }
  }

  Tensor forward(List<int> idx, Tensor encoderOutput, List<Tensor> tracker) {
    final int T = idx.length;

    if (T > blockSize) {
      throw ArgumentError(
        "Sequence length $T exceeds max block size $blockSize",
      );
    }

    // 1. GPU Embedding Lookup
    // We pass T explicitly so the kernel knows how many rows of wpe to use
    Tensor x = Tensor.embeddings(idx, wte, wpe);
    tracker.add(x);

    // 2. Transformer Blocks
    for (final block in blocks) {
      x = block.forward(x, encoderOutput, tracker);
    }

    // 3. Final Norm & Head
    final xNorm = finalLayerNorm.forward(x, tracker);
    return lmHead.forward(xNorm, tracker);
  }

  @override
  List<Tensor> parameters() => [
    wte,
    wpe,
    ...blocks.expand((block) => block.parameters()),
    ...finalLayerNorm.parameters(),
    ...lmHead.parameters(),
  ];
}

// void main() {
//   print("--- Stable Tensor-Engine AFT-GPT Training ---");

//   const int vocabSize = 5;
//   const int embedSize = 32; // Smaller is often more stable for toy examples
//   const int blockSize = 16;
//   const int numLayers = 2; // Start with 1 layer to ensure convergence

//   final Map<String, int> stoi = {
//     "hello": 0,
//     "world": 1,
//     ".": 2,
//     "<start>": 3,
//     "<pad>": 4,
//   };
//   final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));

//   final List<int> inputIds = [3, 0, 1, 4, 4, 4, 4, 4, 4, 4];
//   final List<int> targetIds = [0, 1, 2, 4, 4, 4, 4, 4, 4, 4];

//   final model = TransformerDecoder(
//     vocabSize: vocabSize,
//     embedSize: embedSize,
//     encoderEmbedSize: embedSize,
//     blockSize: blockSize,
//     numLayers: numLayers,
//     numHeads: 4,
//   );

//   List<Tensor> tracker = [];

//   // Use a conservative Learning Rate without momentum first
//   const double learningRate = 0.001;
//   final optimizer = Adam(
//     model.parameters(),
//     lr: learningRate, //momentum: 0.0
//   );
//   final dummyEnc = Tensor.zeros([1, embedSize]);

//   print('Starting training...');
//   for (int epoch = 0; epoch <= 10000; epoch++) {
//     optimizer.zeroGrad();

//     final x = inputIds.sublist(0, inputIds.length - 1);
//     final y = inputIds.sublist(1);
//     final logits = model.forward(x, dummyEnc, tracker);

//     double lossValue = 0;
//     // int activeTokens = 0;
//     final loss = logits.crossEntropy(y);

//     final lossVal = loss.fetchData()[0];

//     // Normalize gradients by the number of samples in the batch
//     // for (int i = 0; i < logits.grad.length; i++) {
//     //   logits.grad[i] /= activeTokens;
//     // }

//     loss.backward();
//     optimizer.step();

//     if (epoch % 50 == 0) {
//       print("Epoch $epoch | Loss: ${lossVal.toStringAsFixed(10)}");
//       if (lossValue.isNaN) break;
//     }

//     for (var t in tracker) {
//       t.dispose();
//     }
//     loss.dispose();
//     logits.dispose();
//   }

//   // 3. Inference
//   // print("\nInference Result:");
//   // List<int> currentSeq = [stoi["<start>"]!];
//   // for (int i = 0; i < 5; i++) {
//   //   final out = model.forward(currentSeq, dummyEnc, tracker);
//   //   int lastOffset = (currentSeq.length - 1) * vocabSize;

//   //   int nextId = 0;
//   //   double best = -double.infinity;
//   //   for (int v = 0; v < vocabSize; v++) {
//   //     if (out.data[lastOffset + v] > best) {
//   //       best = out.data[lastOffset + v];
//   //       nextId = v;
//   //     }
//   //   }
//   //   currentSeq.add(nextId);
//   //   if (nextId == stoi["."]) break;
//   // }
//   // print("Output: ${currentSeq.map((id) => itos[id] ?? "??").join(" ")}");
// }

void main() {
  print("--- Stable Tensor-Engine AFT-GPT Training ---");

  // 1. Hyperparameters
  const int vocabSize = 5;
  const int embedSize = 32;
  const int blockSize = 16;
  const int numLayers = 2;

  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    ".": 2,
    "<start>": 3,
    "<pad>": 4,
  };
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));

  // Data: "hello world ."
  // Input:  <start> hello world
  // Target: hello   world .
  final List<int> inputIds = [3, 0, 1];
  final List<int> targetIds = [0, 1, 2];

  final model = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: 4,
  );

  final List<Tensor> tracker = [];
  const double learningRate = 0.001;
  final optimizer = Adam(model.parameters(), lr: learningRate);

  // Dummy Encoder Context (e.g., could be a single 'thought' vector or a null-state)
  final dummyEnc = Tensor.zeros([1, embedSize]);

  print('Starting training...');
  for (int epoch = 0; epoch <= 1000; epoch++) {
    optimizer.zeroGrad();

    // 2. Forward Pass
    // logits: [T, vocabSize]
    final logits = model.forward(inputIds, dummyEnc, tracker);

    // 3. Loss Calculation
    // Assuming your crossEntropy implementation handles [T, V] logits vs [T] targets
    final loss = logits.crossEntropy(targetIds);
    final double lossVal = loss.fetchData()[0];

    // 4. Backprop
    loss.backward();

    // Optional: Gradient Clipping for safety
    // for (var p in model.parameters()) {
    //   p.grad?.clamp(-1.0, 1.0);
    // }

    optimizer.step();

    if (epoch % 100 == 0) {
      print(
        "Epoch ${epoch.toString().padLeft(4)} | Loss: ${lossVal.toStringAsFixed(10)}",
      );
      if (lossVal.isNaN) break;
    }

    // 5. Explicit GPU Memory Cleanup
    for (var t in tracker) {
      t.dispose();
    }
    tracker.clear();
    loss.dispose();
    logits.dispose();
  }

  // 6. Inference / Generation
  print("\n--- Inference ---");
  List<int> currentSeq = [stoi["<start>"]!];

  for (int i = 0; i < 5; i++) {
    // Generate next token
    final out = model.forward(currentSeq, dummyEnc, tracker);

    // Get the logits for the very last token generated
    final lastTokenLogits = out.fetchData().sublist(
      (currentSeq.length - 1) * vocabSize,
      currentSeq.length * vocabSize,
    );

    int nextId = 0;
    double best = -double.infinity;
    for (int v = 0; v < vocabSize; v++) {
      if (lastTokenLogits[v] > best) {
        best = lastTokenLogits[v];
        nextId = v;
      }
    }

    currentSeq.add(nextId);

    // Cleanup inference tensors
    for (var t in tracker) t.dispose();
    tracker.clear();
    out.dispose();

    if (nextId == stoi["."]) break;
  }

  print("Generated: ${currentSeq.map((id) => itos[id] ?? "??").join(" ")}");
}
