import 'aft.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class MultiHeadAFT extends Module {
  final List<AFTAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;
  final bool masked;

  MultiHeadAFT(
    this.numHeads,
    int embedSize,
    int maxSeqLen, {
    this.masked = false,
  }) : assert(embedSize % numHeads == 0),
       headSize = embedSize ~/ numHeads,
       heads = List.generate(
         numHeads,
         (i) => AFTAttention(
           embedSize,
           embedSize ~/ numHeads,
           maxSeqLen,
           masked: masked,
         ),
       ),
       proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // 1. Compute each head on GPU
    final List<Tensor> headOutputs = heads
        .map((h) => h.forward(x, tracker))
        .toList();

    // 2. Concatenate results on GPU
    final concatenated = Tensor.concat(headOutputs);
    tracker.add(concatenated);

    // 3. Final Linear Projection
    return proj.forward(concatenated, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...heads.expand((h) => h.parameters()),
    ...proj.parameters(),
  ];
}
