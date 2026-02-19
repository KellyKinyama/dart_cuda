// file: triplet_loss_gpu.dart
import 'gpu_tensor.dart';
import 'nn.dart';

class TripletLossGPU extends Module {
  final double margin;

  TripletLossGPU({this.margin = 0.2});

  Tensor forward(
    Tensor anchor,
    Tensor positive,
    Tensor negative,
    List<Tensor> tracker,
  ) {
    // 1. Calculate squared Euclidean distances
    final diffPos = anchor - positive;
    final diffNeg = anchor - negative;

    final distP = diffPos.pow(2.0).sum();
    final distN = diffNeg.pow(2.0).sum();

    // 2. Triplet Formula: L = distP - distN + margin
    final rawLoss = (distP - distN) + margin;

    // 3. Hinge Loss: L = max(0, L)
    // This prevents the gradient from pulling embeddings into infinity
    // once the margin is already met.
    final loss = rawLoss.relu();

    tracker.addAll([diffPos, diffNeg, distP, distN, rawLoss, loss]);

    // 🔍 Optional Debugging (keep it minimal for performance)
    if (loss.fetchData()[0] > 100) {
      print("⚠️ High Loss detected: ${loss.fetchData()[0]}");
    }

    return loss;
  }

  @override
  List<Tensor> parameters() => [];
}
