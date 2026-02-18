import 'aft_cross_attention.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class MultiHeadAFTCross extends Module {
  final List<AFTCrossAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFTCross(
    this.numHeads,
    int decoderEmbedSize,
    int encoderEmbedSize,
    int maxTDec,
    int maxTEnc,
  ) : assert(decoderEmbedSize % numHeads == 0),
      headSize = decoderEmbedSize ~/ numHeads,
      heads = List.generate(
        numHeads,
        (i) => AFTCrossAttention(
          decoderEmbedSize,
          encoderEmbedSize,
          decoderEmbedSize ~/ numHeads,
          maxTDec,
          maxTEnc,
        ),
      ),
      proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu: false);

  /// xDec: [T_dec, decoderEmbedSize]
  /// xEnc: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // 1. Run each Cross-Attention head on GPU
    // Each head returns a [TDec, headSize] tensor handle
    final headOutputs = heads
        .map((h) => h.forward(xDec, xEnc, tracker))
        .toList();

    // 2. Concatenate heads along axis 1 (GPU kernel)
    // Uses the concat_tensors_gpu kernel we added to engine.cu
    final concatenated = Tensor.concat(headOutputs);
    tracker.add(concatenated);

    // 3. Final linear projection
    return proj.forward(concatenated, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...heads.expand((h) => h.parameters()),
    ...proj.parameters(),
  ];
}
