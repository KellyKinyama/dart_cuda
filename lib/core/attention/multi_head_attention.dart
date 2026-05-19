/// Multi-head standard self-attention.
import 'package:dart_cuda/core/attention/self_attention.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class MultiHeadAttention extends Module {
  final int numHeads;
  final int embedSize;
  final int headSize;
  final bool masked;
  final List<SelfAttention> heads;
  final Layer proj;

  MultiHeadAttention(this.numHeads, this.embedSize, {this.masked = false})
      : assert(embedSize % numHeads == 0,
            'embedSize must be divisible by numHeads'),
        headSize = embedSize ~/ numHeads,
        heads = List.generate(
          numHeads,
          (_) => SelfAttention(
            embedSize,
            embedSize ~/ numHeads,
            masked: masked,
          ),
        ),
        proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final outs = heads.map((h) => h.forward(x, tracker)).toList();
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
