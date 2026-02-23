import 'dart:typed_data';
import 'package:test/test.dart';

import '../gpu_tensor.dart';

void main() {
  test('Cross Entropy Autograd - Numerical Gradient Check', () {
    // 1. Setup Data: 2 time steps, 3 classes
    // Logits (randomly initialized for the test)
    final logits = Tensor.fill([2, 3], 0.0);
    logits.data = [
      1.5, -0.5, 0.2, // Time 0
      -1.0, 2.0, 0.5, // Time 1
    ];

    // Targets: Class indices
    final targets = [
      0,
      1,
    ]; // Correct labels for Time 0 is index 0, Time 1 is index 1

    // 2. Forward Pass
    final loss = logits.crossEntropy(targets);
    print("Initial Loss: ${loss.fetchData()[0]}");

    // 3. Backward Pass
    // Ensure gradients are zeroed before backward
    // (Assuming your engine/tensor class has a zeroGrad method)
    loss.backward();

    // 4. Fetch Analytic Gradients (Calculated by your CUDA kernel)
    final analyticGrads =
        logits.grad; // Assuming fetchGrad exists like fetchData

    // 5. Numerical Gradient Calculation
    final double epsilon = 1e-4;
    final List<double> numericalGrads = [];
    final List<double> originalData = List.from(logits.fetchData());

    for (int i = 0; i < originalData.length; i++) {
      // Nudge UP
      List<double> upData = List.from(originalData);
      upData[i] += epsilon;
      logits.data = upData;
      double lossUp = logits.crossEntropy(targets).fetchData()[0];

      // Nudge DOWN
      List<double> downData = List.from(originalData);
      downData[i] -= epsilon;
      logits.data = downData;
      double lossDown = logits.crossEntropy(targets).fetchData()[0];

      // Formula: (L(x+e) - L(x-e)) / 2e
      numericalGrads.add((lossUp - lossDown) / (2 * epsilon));
    }

    // 6. Compare
    print("\n--- Gradient Comparison ---");
    print("Index | Analytic | Numerical | Difference");
    bool passed = true;
    for (int i = 0; i < analyticGrads.length; i++) {
      double diff = (analyticGrads[i] - numericalGrads[i]).abs();
      print(
        "$i | ${analyticGrads[i].toStringAsFixed(6)} | ${numericalGrads[i].toStringAsFixed(6)} | $diff",
      );

      if (diff > 1e-4) passed = false;
    }

    expect(
      passed,
      isTrue,
      reason: "Analytic and Numerical gradients differ too much!",
    );
    print("\n✅ Autograd check passed!");
  });
}
