import 'dart:math' as math;

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
