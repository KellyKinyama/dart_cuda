import 'aft_multi_head_attention.dart';
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart'; // Contains your 'Layer' class

class TransformerEncoderBlock extends Module {
  final MultiHeadAFT attention;
  final Layer ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  TransformerEncoderBlock(this.embedSize, int numHeads, int maxSeqLen)
    : attention = MultiHeadAFT(
        numHeads,
        embedSize,
        maxSeqLen,
        masked: false, // Encoder is bidirectional, not causal
      ),
      // FFN typically expands and then contracts;
      // here using your Layer class for computation.
      ffn = Layer(embedSize, embedSize, useGelu: true),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize);

  /// x: [T, embedSize]
  Tensor forward(Tensor x, List<Tensor> tracker) {
    // --- 1. Multi-Head AFT Attention Sub-layer ---
    // Pre-Norm logic: ln -> attention -> residual
    final xNorm1 = ln1.forward(x, tracker);
    // Note: ln1.forward likely adds intermediates to tracker internally

    final attnOut = attention.forward(xNorm1, tracker);
    final xRes1 = x + attnOut;

    // We track intermediates created in this scope
    tracker.addAll([attnOut, xRes1]);

    // --- 2. Feed-Forward Network Sub-layer ---
    // Pre-Norm logic: ln -> ffn -> residual
    final xNorm2 = ln2.forward(xRes1, tracker);

    final ffnOut = ffn.forward(xNorm2, tracker);
    final out = xRes1 + ffnOut;

    // Track the MLP output.
    // 'out' is NOT added to tracker because it is the return value
    // and will be used (and tracked/disposed) by the next layer or main loop.
    tracker.add(ffnOut);

    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...attention.parameters(),
    ...ffn.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
  ];
}
