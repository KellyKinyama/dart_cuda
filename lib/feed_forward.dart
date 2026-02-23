import 'gpu_tensor.dart';
import 'nn.dart';

class FeedForward extends Module {
  final Layer w1; // Up-projection (embedSize -> embedSize * 4)
  final Layer w2; // Down-projection (embedSize * 4 -> embedSize)

  FeedForward(int dim)
    : w1 = Layer(dim, dim * 4, useGelu: true),
      w2 = Layer(dim * 4, dim, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // 1. Expand and Activate (GELU)
    final hidden = w1.forward(x, tracker);
    // 2. Contract back to original dimension
    final out = w2.forward(hidden, tracker);
    return out;
  }

  @override
  List<Tensor> parameters() => [...w1.parameters(), ...w2.parameters()];
}
