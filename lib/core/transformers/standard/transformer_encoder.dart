/// Transformer encoder.
///
/// Takes a token-id sequence and returns contextualised embeddings
/// (no LM head; for that use [Transformer] from `transformer.dart`).
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/standard/transformer_encoder_block.dart';

class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;

  final Tensor wte;
  final Tensor wpe;
  final List<TransformerEncoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100,
    this.embedSize = 64,
    this.blockSize = 128,
    this.numLayers = 6,
    this.numHeads = 8,
  })  : assert(embedSize % numHeads == 0,
            'embedSize must be divisible by numHeads'),
        wte = Tensor.random([vocabSize, embedSize], scale: 0.02),
        wpe = Tensor.random([blockSize, embedSize], scale: 0.02),
        blocks = List.generate(
          numLayers,
          (_) => TransformerEncoderBlock(embedSize, numHeads),
        ),
        finalLayerNorm = LayerNorm(embedSize);

  Tensor forward(List<int> idx, List<Tensor> tracker) {
    final T = idx.length;
    if (T > blockSize) {
      throw ArgumentError(
          'Sequence length $T exceeds block size $blockSize');
    }
    var x = Tensor.embeddings(idx, wte, wpe);
    tracker.add(x);
    return _processThroughBlocks(x, tracker);
  }

  /// Entry point for vision (patch embeddings) or pre-embedded tokens.
  Tensor forwardEmbeddings(Tensor embedded, List<Tensor> tracker) {
    if (embedded.shape[0] > blockSize) {
      throw ArgumentError('Input sequence length exceeds block size');
    }
    return _processThroughBlocks(embedded, tracker);
  }

  Tensor _processThroughBlocks(Tensor x, List<Tensor> tracker) {
    var current = x;
    for (final b in blocks) {
      current = b.forward(current, tracker);
    }
    return finalLayerNorm.forward(current, tracker);
  }

  @override
  List<Tensor> parameters() => [
        wte,
        wpe,
        ...blocks.expand((b) => b.parameters()),
        ...finalLayerNorm.parameters(),
      ];
}
