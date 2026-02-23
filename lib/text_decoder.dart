// lib/text_decoder.dart (New File)
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart';
import 'aft_text_decoder_block.dart'; // Your new file

class TextDecoder extends Module {
  final int vocabSize;
  final int maxSeqLen;
  final int embedSize;

  final Tensor wte; // Token Embeddings
  final Tensor wpe; // Positional Embeddings
  final List<TextDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead; // Language Model Head (embedSize -> vocabSize)

  TextDecoder({
    required this.vocabSize,
    required this.maxSeqLen,
    required this.embedSize,
    int numLayers = 4,
    int numHeads = 4,
  }) : wte = Tensor.random([vocabSize, embedSize], scale: 0.02),
       wpe = Tensor.random([maxSeqLen, embedSize], scale: 0.02),
       blocks = List.generate(
         numLayers,
         (i) => TextDecoderBlock(embedSize, numHeads, maxSeqLen),
       ),
       finalLayerNorm = LayerNorm(embedSize),
       lmHead = Layer(
         embedSize,
         vocabSize,
         useGelu: false,
       ); // Project to vocabulary

  Tensor forward(
    List<int> textTokens,
    Tensor encoderOutput,
    List<Tensor> tracker,
  ) {
    final int T = textTokens.length;
    if (T > maxSeqLen) {
      throw ArgumentError("Text sequence length exceeds maxSeqLen");
    }

    // 1. Embed text tokens and add positional information
    final tokenEmbeds = Tensor.embeddings(textTokens, wte, wpe);
    tracker.add(tokenEmbeds);

    // 2. Pass through decoder blocks
    Tensor x = tokenEmbeds;
    for (final block in blocks) {
      x = block.forward(x, encoderOutput, tracker);
    }

    // 3. Final LayerNorm and Language Model Head
    final normalized = finalLayerNorm.forward(x, tracker);
    return lmHead.forward(
      normalized,
      tracker,
    ); // Logits for each vocabulary token
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
