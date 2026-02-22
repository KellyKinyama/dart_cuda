import 'gpu_tensor.dart';
import 'mlp.dart';

void main() {
  // 1. Setup Network
  final model = MLP(2, [4, 2]);
  final double learningRate = 0.5; // High LR for XOR

  // 2. Data (XOR)
  final xData = Tensor.fromList([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
  final target = Tensor.fromList([4, 2], [0, 1, 1, 0, 1, 0, 0, 1]);

  print('Starting GPU MLP Training...');

  for (int epoch = 0; epoch <= 2000; epoch++) {
    // This list tracks every tensor created in this iteration
    List<Tensor> tracker = [];

    // Forward
    final pred = model.forward(xData, tracker);

    // Loss = (pred - target)^2
    final diff = pred - target;
    final loss = diff.pow(2.0);
    tracker.addAll([diff, loss]);

    // Backward
    model.zeroGrad();
    loss.backward();

    // Update (On GPU)
    model.step(learningRate);

    if (epoch % 100 == 0) {
      print("Epoch $epoch, Loss: ${loss.data[0].toStringAsFixed(6)}");
    }

    // --- CRITICAL: GPU MEMORY CLEANUP ---
    // Dispose all intermediate tensors created this epoch
    for (var t in tracker) {
      t.dispose();
    }
  }

  // Final Test
  List<Tensor> dummy = [];
  final finalPred = model.forward(xData, dummy);
  print("\nFinal Results:");
  print(finalPred.printMatrix());

  // Final cleanup
  finalPred.dispose();
  xData.dispose();
  target.dispose();
}
