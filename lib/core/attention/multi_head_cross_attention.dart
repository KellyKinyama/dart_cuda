/// Multi-head standard cross-attention.
import 'package:dart_cuda/core/attention/cross_attention.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class MultiHeadCrossAttention extends Module {
  final int numHeads;
  final int decoderEmbedSize;
  final int encoderEmbedSize;
  final int headSize;
  final List<CrossAttention> heads;
  final Layer proj;

  MultiHeadCrossAttention(
    this.numHeads,
    this.decoderEmbedSize,
    this.encoderEmbedSize,
  )   : assert(decoderEmbedSize % numHeads == 0,
            'decoderEmbedSize must be divisible by numHeads'),
        headSize = decoderEmbedSize ~/ numHeads,
        heads = List.generate(
          numHeads,
          (_) => CrossAttention(
            decoderEmbedSize,
            encoderEmbedSize,
            decoderEmbedSize ~/ numHeads,
          ),
        ),
        proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu: false);

  Tensor forward(Tensor xDecoder, Tensor xEncoder, List<Tensor> tracker) {
    final outs =
        heads.map((h) => h.forward(xDecoder, xEncoder, tracker)).toList();
    final concat = Tensor.concat(outs);
    tracker.add(concat);
    return proj.forward(concat, tracker);
  }

  @override
  List<Tensor> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}
