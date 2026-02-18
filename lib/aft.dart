import 'gpu_tensor.dart';
import 'nn.dart'; // Assuming your Layer/Module base classes are here

class AFTAttention extends Module {
  final Layer queryLayer, keyLayer, valueLayer;
  final Tensor posBias;
  final bool masked;

  AFTAttention(
    int embedSize,
    int headSize,
    int maxSeqLen, {
    this.masked = false,
  }) : queryLayer = Layer(embedSize, headSize, useGelu: false),
       keyLayer = Layer(embedSize, headSize, useGelu: false),
       valueLayer = Layer(embedSize, headSize, useGelu: false),
       // Initialize bias with small random values
       posBias = Tensor.fromList([
         maxSeqLen,
         maxSeqLen,
       ], List.generate(maxSeqLen * maxSeqLen, (_) => 0.01));

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // 1. Projections (On GPU)
    final q = queryLayer.forward(x, tracker);
    final k = keyLayer.forward(x, tracker);
    final v = valueLayer.forward(x, tracker);

    // 2. AFT custom kernel call
    // This replaces the nested Dart loops with one high-speed CUDA call
    final out = Tensor.aft(q, k, v, posBias, masked);

    tracker.add(out);
    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...queryLayer.parameters(),
    ...keyLayer.parameters(),
    ...valueLayer.parameters(),
    posBias,
  ];
}
