import 'gpu_tensor.dart';
import 'aft_transformer_encoder_block.dart';
import 'layer_norm.dart';
import 'nn.dart';

class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;

  // GPU Tensors for Embeddings
  final Tensor wte; // [vocabSize, embedSize]
  final Tensor wpe; // [blockSize, embedSize]

  final List<TransformerEncoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100,
    this.embedSize = 64,
    this.blockSize = 128,
    this.numLayers = 6,
    this.numHeads = 8,
  }) : assert(
         embedSize % numHeads == 0,
         "embedSize must be divisible by numHeads",
       ),
       // Initialize directly on GPU with scaling
       wte = Tensor.random([vocabSize, embedSize], scale: 0.02),
       wpe = Tensor.random([blockSize, embedSize], scale: 0.02),
       blocks = List.generate(
         numLayers,
         (i) => TransformerEncoderBlock(embedSize, numHeads, blockSize),
       ),
       finalLayerNorm = LayerNorm(embedSize);

  /// idx: List of token IDs.
  /// tracker: List to collect intermediate tensors for disposal.
  Tensor forward(List<int> idx, List<Tensor> tracker) {
    final int T = idx.length;
    if (T > blockSize) {
      throw ArgumentError("Sequence length $T exceeds block size $blockSize");
    }

    // 1. GPU-Accelerated Embedding Lookup
    // This calls your 'embedding_forward' CUDA kernel.
    // No loops, just one kernel launch.
    final x = Tensor.embeddings(idx, wte, wpe);
    tracker.add(x);

    return _processThroughBlocks(x, tracker);
  }

  /// Entry point for vision (image patches) or pre-embedded tokens
  Tensor forwardEmbeddings(Tensor embeddedInputs, List<Tensor> tracker) {
    if (embeddedInputs.shape[0] > blockSize) {
      throw ArgumentError("Input sequence length exceeds block size");
    }
    return _processThroughBlocks(embeddedInputs, tracker);
  }

  /// Internal GPU processing pipeline
  Tensor _processThroughBlocks(Tensor x, List<Tensor> tracker) {
    Tensor current = x;

    // 2. Stack of Encoder Blocks
    for (final block in blocks) {
      // Each block creates multiple intermediate tensors inside.
      // We pass the tracker down to ensure they are all collected.
      current = block.forward(current, tracker);

      // Note: We don't add 'current' to the tracker here because it is
      // passed to the NEXT block. The tracker will dispose of everything
      // collected by the blocks at the end of the training step.
    }

    // 3. Final Layer Norm on GPU
    final out = finalLayerNorm.forward(current, tracker);

    return out;
  }

  @override
  List<Tensor> parameters() => [
    wte,
    wpe,
    ...blocks.expand((block) => block.parameters()),
    ...finalLayerNorm.parameters(),
  ];
}
