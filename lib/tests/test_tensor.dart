import '../gpu_tensor.dart';

void main() {
  print("--- Starting Manual Autograd Verification ---");

  // 0. Bridge Check (Crucial for debugging your current zero-issue)
  verifyBridge();

  // 1. Core Math
  verifySum();
  verifyMean();
  verifySigmoid();
  verifyMatMul();
  verifyPow();
  verifyLog();

  // 2. Transformer Specifics
  verifyCrossEntropySimple();
  verifySoftmax();
  verifyCausalMasking();
  verifyLayerNorm();
  verifyWeightUpdate();

  print("\n--- All tests completed ---");
}

bool closeEnough(double a, double b, [double tol = 1e-3]) {
  return (a - b).abs() < tol;
}

// ---------------------------------------------------------
// 0. BRIDGE Verification (Tests fromList -> fetchData)
// ---------------------------------------------------------
void verifyBridge() {
  final List<double> values = [1.0, 2.0, 3.0, 4.0];
  final x = Tensor.fromList([2, 2], values);

  // This triggers cudaMemcpy from Device to Host
  final fetched = x.fetchData();

  bool ok = true;
  for (int i = 0; i < values.length; i++) {
    if (!closeEnough(fetched[i], values[i])) ok = false;
  }

  print(
    "BRIDGE (FFI): ${ok ? '✅ PASS' : '❌ FAIL (Sent $values, got $fetched)'}",
  );
}

// ---------------------------------------------------------
// 1. SUM Verification
// ---------------------------------------------------------
void verifySum() {
  final x = Tensor.fromList([2, 2], [1.0, 2.0, 3.0, 4.0]);

  final loss = x.sum();
  loss.backward();

  final grads = x.grad; // Assuming .grad also calls fetchData() internally
  bool ok = grads.every((g) => closeEnough(g, 1.0));
  print("SUM: ${ok ? '✅ PASS' : '❌ FAIL (Expected grads of 1.0, got $grads)'}");
}

// ---------------------------------------------------------
// 2. MEAN Verification
// ---------------------------------------------------------
void verifyMean() {
  final x = Tensor.fromList([1, 4], [1.0, 2.0, 3.0, 4.0]);

  final loss = x.mean();
  loss.backward();

  final grads = x.grad;
  bool ok = grads.every((g) => closeEnough(g, 0.25));
  print("MEAN: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.25, got $grads)'}");
}

// ---------------------------------------------------------
// 3. SIGMOID Verification
// ---------------------------------------------------------
void verifySigmoid() {
  final x = Tensor.fromList([1, 1], [0.0]);

  final loss = x.sigmoid().sum();
  loss.backward();

  final grads = x.grad;
  bool ok = closeEnough(grads[0], 0.25);
  print(
    "SIGMOID: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.25, got ${grads[0]})'}",
  );
}

// ---------------------------------------------------------
// 4. MATMUL Verification
// ---------------------------------------------------------
void verifyMatMul() {
  final A = Tensor.fromList([1, 2], [2.0, 3.0]);
  final B = Tensor.fromList([2, 1], [4.0, 5.0]);

  final loss = A.matmul(B).sum();
  loss.backward();

  final gradA = A.grad;
  final gradB = B.grad;

  bool okA = closeEnough(gradA[0], 4.0) && closeEnough(gradA[1], 5.0);
  bool okB = closeEnough(gradB[0], 2.0) && closeEnough(gradB[1], 3.0);

  print("MATMUL: ${okA && okB ? '✅ PASS' : '❌ FAIL'}");
}

// ---------------------------------------------------------
// 5. POW Verification
// ---------------------------------------------------------
void verifyPow() {
  final x = Tensor.fromList([1, 1], [3.0]);

  final loss = x.pow(2.0).sum();
  loss.backward();

  final grads = x.grad;
  bool ok = closeEnough(grads[0], 6.0);
  print("POW: ${ok ? '✅ PASS' : '❌ FAIL (Expected 6.0, got ${grads[0]})'}");
}

// ---------------------------------------------------------
// 6. LOG Verification
// ---------------------------------------------------------
void verifyLog() {
  final x = Tensor.fromList([1, 1], [2.0]);

  final loss = x.log().sum();
  loss.backward();

  final grads = x.grad;
  bool ok = closeEnough(grads[0], 0.5);
  print("LOG: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.5, got ${grads[0]})'}");
}

// ---------------------------------------------------------
// 7. CROSS ENTROPY Verification
// ---------------------------------------------------------
void verifyCrossEntropySimple() {
  final x = Tensor.fromList([1, 2], [0.0, 0.0]);

  final loss = x.crossEntropy([1]);
  loss.backward();

  final grads = x.grad;
  // Expected based on Label Smoothing epsilon 0.1
  bool ok = closeEnough(grads[0], 0.45) && closeEnough(grads[1], -0.45);
  print("CROSS_ENTROPY: ${ok ? '✅ PASS' : '❌ FAIL (Got $grads)'}");
}

// ---------------------------------------------------------
// 8. SOFTMAX Verification
// ---------------------------------------------------------
void verifySoftmax() {
  final x = Tensor.fromList([1, 3], [1.0, 2.0, 3.0]);

  final y = x.softmax();
  final data = y.fetchData(); // Use fetchData instead of .data

  bool forwardOk = closeEnough(data[0], 0.0900) && closeEnough(data[2], 0.6652);

  final loss = y.sum();
  loss.backward();
  final grads = x.grad;

  bool gradOk = grads.every((g) => closeEnough(g, 0.0, 1e-5));
  print("SOFTMAX: ${forwardOk && gradOk ? '✅ PASS' : '❌ FAIL'}");
}

// ---------------------------------------------------------
// 9. CAUSAL MASKING Verification
// ---------------------------------------------------------
void verifyCausalMasking() {
  final attn = Tensor.fromList([3, 3], List.filled(9, 10.0));
  final mask = Tensor.fromList([3, 3], [1, 0, 0, 1, 1, 0, 1, 1, 1]);

  final masked = attn * mask;
  final loss = masked.sum();
  loss.backward();

  final grads = attn.grad;

  bool ok =
      closeEnough(grads[1], 0.0) &&
      closeEnough(grads[2], 0.0) &&
      closeEnough(grads[5], 0.0) &&
      closeEnough(grads[0], 1.0);

  print("CAUSAL MASK: ${ok ? '✅ PASS' : '❌ FAIL'}");
}

// ---------------------------------------------------------
// 10. LAYER NORM Verification
// ---------------------------------------------------------
void verifyLayerNorm() {
  final x = Tensor.fromList([1, 4], [10.0, 2.0, 5.0, 3.0]);

  final y = x.normalize(eps: 1e-6);
  final loss = y.sum();
  loss.backward();

  final grads = x.grad;
  bool ok = grads.every((g) => closeEnough(g, 0.0, 1e-5));
  print("LAYERNORM: ${ok ? '✅ PASS' : '❌ FAIL'}");
}

// ---------------------------------------------------------
// 11. WEIGHT UPDATE Verification
// ---------------------------------------------------------
void verifyWeightUpdate() {
  final X = Tensor.fromList([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  final W = Tensor.fromList([3, 2], List.filled(6, 0.5));

  final out = X.matmul(W);
  final loss = out.sum();
  loss.backward();

  final gradW = W.grad;
  bool ok = closeEnough(gradW[0], 5.0) && closeEnough(gradW[5], 9.0);

  print("WEIGHT UPDATE: ${ok ? '✅ PASS' : '❌ FAIL (Got $gradW)'}");
}
