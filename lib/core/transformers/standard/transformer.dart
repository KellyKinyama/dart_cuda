/// GPT-style decoder-only Transformer.
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/standard/transformer_block.dart';

class Transformer extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;

  final Tensor wte; // [vocabSize, embedSize]
  final Tensor wpe; // [blockSize, embedSize]
  final List<TransformerBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  Transformer({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
  })  : assert(embedSize % numHeads == 0,
            'embedSize must be divisible by numHeads'),
        wte = Tensor.random([vocabSize, embedSize], scale: 0.02),
        wpe = Tensor.random([blockSize, embedSize], scale: 0.02),
        blocks = List.generate(
          numLayers,
          (_) => TransformerBlock(embedSize, numHeads, masked: true),
        ),
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer(embedSize, vocabSize, useGelu: false);

  Tensor forward(List<int> idx, List<Tensor> tracker) {
    final T = idx.length;
    if (T > blockSize) {
      throw ArgumentError(
          'Sequence length $T exceeds block size $blockSize');
    }
    var x = Tensor.embeddings(idx, wte, wpe);
    tracker.add(x);
    for (final b in blocks) {
      x = b.forward(x, tracker);
    }
    x = finalLayerNorm.forward(x, tracker);
    return lmHead.forward(x, tracker);
  }

  @override
  List<Tensor> parameters() => [
        wte,
        wpe,
        ...blocks.expand((b) => b.parameters()),
        ...finalLayerNorm.parameters(),
        ...lmHead.parameters(),
      ];
}
