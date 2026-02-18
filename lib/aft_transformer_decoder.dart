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
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
    this.encoderEmbedSize = 64,
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
    // Initial scaling on GPU (using tensor_step or a custom fill kernel)
    wte.step(-0.02); // Quick hack to scale if you don't have a scale kernel
    wpe.step(-0.02);
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
