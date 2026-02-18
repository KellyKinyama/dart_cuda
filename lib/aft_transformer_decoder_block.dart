import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross_attention.dart';
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final Layer ffn; // Using your 'Layer' class for the FFN
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final int embedSize;

  TransformerDecoderBlock(
    this.embedSize,
    int numHeads,
    int encoderEmbedSize,
    int maxSeqLen,
  ) : selfAttention = MultiHeadAFT(
        numHeads,
        embedSize,
        maxSeqLen,
        masked: true,
      ),
      crossAttention = MultiHeadAFTCross(
        numHeads,
        embedSize,
        encoderEmbedSize,
        maxSeqLen,
        maxSeqLen,
      ),
      // Typically FFN expands 4x in Transformers
      ffn = Layer(embedSize, embedSize, useGelu: true),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize),
      ln3 = LayerNorm(embedSize);

  /// x_decoder: [T_dec, embedSize]
  /// x_encoder: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // --- 1. Masked AFT Self-Attention + Residual ---
    final xNorm1 = ln1.forward(xDec, tracker);
    final selfAttnOut = selfAttention.forward(xNorm1, tracker);
    final xRes1 = xDec + selfAttnOut;
    tracker.addAll([selfAttnOut, xRes1]);

    // --- 2. AFT Cross-Attention + Residual ---
    final xNorm2 = ln2.forward(xRes1, tracker);
    final crossAttnOut = crossAttention.forward(xNorm2, xEnc, tracker);
    final xRes2 = xRes1 + crossAttnOut;
    tracker.addAll([crossAttnOut, xRes2]);

    // --- 3. Feed-Forward + Residual ---
    final xNorm3 = ln3.forward(xRes2, tracker);
    final ffnOut = ffn.forward(xNorm3, tracker);
    final out = xRes2 + ffnOut;
    tracker.addAll([ffnOut, out]);

    tracker.addAll([crossAttnOut, xRes2]);

    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttention.parameters(),
    ...crossAttention.parameters(),
    ...ffn.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...ln3.parameters(),
  ];
}
