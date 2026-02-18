import 'gpu_tensor.dart';
import 'nn.dart';

class AFTCrossAttention extends Module {
  final Layer keyLayer;
  final Layer queryLayer;
  final Layer valueLayer;
  final Tensor posBias;

  AFTCrossAttention(
    int decoderEmbedSize,
    int encoderEmbedSize,
    int headSize,
    int maxTDec,
    int maxTEnc,
  ) : keyLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
      queryLayer = Layer(decoderEmbedSize, headSize, useGelu: false),
      valueLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
      // Xavier-like init for bias
      posBias = Tensor.fromList([
        maxTDec,
        maxTEnc,
      ], List.generate(maxTDec * maxTEnc, (_) => 0.01));

  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // 1. Projections (GPU Matmuls)
    // Q comes from Decoder, K/V come from Encoder
    final q = queryLayer.forward(xDec, tracker);
    final k = keyLayer.forward(xEnc, tracker);
    final v = valueLayer.forward(xEnc, tracker);

    // 2. High-speed CUDA kernel
    final out = Tensor.aftCross(q, k, v, posBias);

    tracker.add(out);
    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...keyLayer.parameters(),
    ...queryLayer.parameters(),
    ...valueLayer.parameters(),
    posBias,
  ];
}
