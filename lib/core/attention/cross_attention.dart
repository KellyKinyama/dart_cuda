/// Standard scaled-dot-product cross-attention (single head).
///
/// Queries come from the decoder side, keys/values from the encoder side.
/// No causal masking (encoder output is fully visible to the decoder).
import 'package:dart_cuda/core/attention/self_attention.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class CrossAttention extends Module {
  final int decoderEmbedSize;
  final int encoderEmbedSize;
  final int headSize;
  final Layer query;
  final Layer key;
  final Layer value;

  CrossAttention(
    this.decoderEmbedSize,
    this.encoderEmbedSize,
    this.headSize,
  )   : query = Layer(decoderEmbedSize, headSize, useGelu: false),
        key = Layer(encoderEmbedSize, headSize, useGelu: false),
        value = Layer(encoderEmbedSize, headSize, useGelu: false);

  Tensor forward(Tensor xDecoder, Tensor xEncoder, List<Tensor> tracker) {
    final q = query.forward(xDecoder, tracker);
    final k = key.forward(xEncoder, tracker);
    final v = value.forward(xEncoder, tracker);
    return scaledDotProductAttention(q, k, v, tracker, masked: false);
  }

  @override
  List<Tensor> parameters() => [
        ...query.parameters(),
        ...key.parameters(),
        ...value.parameters(),
      ];
}
