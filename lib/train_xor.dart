import 'gpu_tensor.dart';

void main() {
  final inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
  final targets = [0.0, 1.0, 1.0, 0.0];

  // Weights stay alive for the whole program
  final w1 = Tensor.fromList([2, 3], [0.5, -0.2, 0.1, 0.3, 0.8, -0.5]);
  final w2 = Tensor.fromList([3, 1], [0.7, -0.4, 0.2]);

  const double lr = 0.5;

  for (int epoch = 0; epoch <= 1000; epoch++) {
    double epochLoss = 0;
    for (int i = 0; i < 4; i++) {
      final x = Tensor.fromList([1, 2], inputs[i]);
      final target = Tensor.fromList([1, 1], [targets[i]]);

      // Forward
      final h = x.matmul(w1).sigmoid();
      final pred = h.matmul(w2).sigmoid();
      final diff = pred - target;
      final loss = diff.pow(2.0);

      epochLoss += loss.data[0];

      // Backward
      w1.zeroGrad();
      w2.zeroGrad();
      loss.backward();

      // Step
      w1.step(lr);
      w2.step(lr);

      // --- CRITICAL: Cleanup temporaries ---
      // If you don't dispose these, you'll get a memory leak or a crash
      loss.dispose();
      diff.dispose();
      pred.dispose();
      h.dispose();
      target.dispose();
      x.dispose();
    }
    if (epoch % 100 == 0) print("Epoch $epoch Loss: ${epochLoss / 4}");
  }

  print('\n--- Final Predictions ---');
  for (int i = 0; i < 4; i++) {
    final x = Tensor.fromList([1, 2], inputs[i]);
    final h = x.matmul(w1).sigmoid();
    final pred = h.matmul(w2).sigmoid();
    
    print('Input: ${inputs[i]} -> Prediction: ${pred.data[0].toStringAsFixed(4)} (Target: ${targets[i]})');
    
    // Clean up
    x.dispose(); h.dispose(); pred.dispose();
  }
}