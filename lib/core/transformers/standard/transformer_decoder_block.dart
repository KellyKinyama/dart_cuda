/// Decoder transformer block with three pre-norm sub-layers:
///   1. Masked multi-head self-attention.
///   2. Multi-head cross-attention over encoder output.
///   3. Position-wise feed-forward network.
import 'package:dart_cuda/core/attention/multi_head_attention.dart';
import 'package:dart_cuda/core/attention/multi_head_cross_attention.dart';
import 'package:dart_cuda/core/layers/feed_forward.dart';
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class TransformerDecoderBlock extends Module {
  final int embedSize;
  final int encoderEmbedSize;
  final MultiHeadAttention selfAttention;
  final MultiHeadCrossAttention crossAttention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;

  TransformerDecoderBlock(this.embedSize, int numHeads, this.encoderEmbedSize)
    : selfAttention = MultiHeadAttention(numHeads, embedSize, masked: true),
      crossAttention = MultiHeadCrossAttention(
        numHeads,
        embedSize,
        encoderEmbedSize,
      ),
      ffn = FeedForward(embedSize),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize),
      ln3 = LayerNorm(embedSize);

  Tensor forward(Tensor xDecoder, Tensor xEncoder, List<Tensor> tracker) {
    // 1. Masked self-attention.
    final xNorm1 = ln1.forward(xDecoder, tracker);
    final selfOut = selfAttention.forward(xNorm1, tracker);
    final xRes1 = xDecoder + selfOut;
    tracker.addAll([selfOut, xRes1]);

    // 2. Cross-attention over encoder output.
    final xNorm2 = ln2.forward(xRes1, tracker);
    final crossOut = crossAttention.forward(xNorm2, xEncoder, tracker);
    final xRes2 = xRes1 + crossOut;
    tracker.addAll([crossOut, xRes2]);

    // 3. FFN.
    final xNorm3 = ln3.forward(xRes2, tracker);
    final ffnOut = ffn.forward(xNorm3, tracker);
    final out = xRes2 + ffnOut;
    tracker.add(ffnOut);
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
