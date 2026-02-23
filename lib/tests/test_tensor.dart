import '../gpu_tensor.dart';

void main() {
  print("--- Starting Manual Autograd Verification ---");

  verifySum();
  verifyMean();
  verifySigmoid();
  verifyMatMul();
  verifyPow();
  verifyLog();
  verifyCrossEntropySimple();

  print("\n--- All tests completed ---");
}

/// Helper to compare two values with a small tolerance
bool closeEnough(double a, double b, [double tol = 1e-3]) {
  return (a - b).abs() < tol;
}

// ---------------------------------------------------------
// 1. SUM Verification
// ---------------------------------------------------------
void verifySum() {
  final x = Tensor.fill([2, 2], 0.0);
  x.data = [1.0, 2.0, 3.0, 4.0];

  final loss = x.sum();
  loss.backward();

  final grads = x.grad;
  // d(sum(x))/dx should be [1, 1, 1, 1]
  bool ok = grads.every((g) => closeEnough(g, 1.0));
  print("SUM: ${ok ? '✅ PASS' : '❌ FAIL (Expected grads of 1.0, got $grads)'}");
}

// ---------------------------------------------------------
// 2. MEAN Verification
// ---------------------------------------------------------
void verifyMean() {
  final x = Tensor.fill([1, 4], 0.0);
  x.data = [1.0, 2.0, 3.0, 4.0];

  final loss = x.mean();
  loss.backward();

  final grads = x.grad;
  // d(mean(x))/dx should be 1/N. N=4, so [0.25, 0.25, 0.25, 0.25]
  bool ok = grads.every((g) => closeEnough(g, 0.25));
  print("MEAN: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.25, got $grads)'}");
}

// ---------------------------------------------------------
// 3. SIGMOID Verification
// ---------------------------------------------------------
void verifySigmoid() {
  final x = Tensor.fill([1, 1], 0.0);
  x.data = [0.0]; // sigmoid(0) = 0.5

  // We wrap it in sum to get a scalar loss for backward
  final loss = x.sigmoid().sum();
  loss.backward();

  final grads = x.grad;
  // d(sigmoid(x))/dx = sig(x) * (1 - sig(x))
  // At x=0: 0.5 * (1 - 0.5) = 0.25
  bool ok = closeEnough(grads[0], 0.25);
  print(
    "SIGMOID: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.25, got ${grads[0]})'}",
  );
}

// ---------------------------------------------------------
// 4. MATMUL Verification
// ---------------------------------------------------------
void verifyMatMul() {
  // A [1, 2] * B [2, 1] = C [1, 1]
  final A = Tensor.fill([1, 2], 0.0);
  final B = Tensor.fill([2, 1], 0.0);
  A.data = [2.0, 3.0];
  B.data = [4.0, 5.0];

  final loss = A.matmul(B).sum();
  loss.backward();

  final gradA = A.grad;
  final gradB = B.grad;

  // d(AB)/dA = B^T -> [4.0, 5.0]
  // d(AB)/dB = A^T -> [2.0, 3.0]
  bool okA = closeEnough(gradA[0], 4.0) && closeEnough(gradA[1], 5.0);
  bool okB = closeEnough(gradB[0], 2.0) && closeEnough(gradB[1], 3.0);

  print("MATMUL: ${okA && okB ? '✅ PASS' : '❌ FAIL'}");
}

// ---------------------------------------------------------
// 5. POW Verification
// ---------------------------------------------------------
void verifyPow() {
  final x = Tensor.fill([1, 1], 0.0);
  x.data = [3.0];

  final loss = x.pow(2.0).sum(); // x^2
  loss.backward();

  final grads = x.grad;
  // d(x^2)/dx = 2x. For x=3, grad = 6.0
  bool ok = closeEnough(grads[0], 6.0);
  print("POW: ${ok ? '✅ PASS' : '❌ FAIL (Expected 6.0, got ${grads[0]})'}");
}

// ---------------------------------------------------------
// 6. LOG Verification
// ---------------------------------------------------------
void verifyLog() {
  final x = Tensor.fill([1, 1], 0.0);
  x.data = [2.0]; // log(2.0)

  final loss = x.log().sum();
  loss.backward();

  final grads = x.grad;
  // d(log(x))/dx = 1/x. For x=2.0, grad = 0.5
  bool ok = closeEnough(grads[0], 0.5);
  print("LOG: ${ok ? '✅ PASS' : '❌ FAIL (Expected 0.5, got ${grads[0]})'}");
}

// ---------------------------------------------------------
// 7. Cross Entropy (Internal) Verification
// ---------------------------------------------------------
// void verifyCrossEntropySimple() {
//   // 1 row, 2 classes. Target is index 1.
//   final x = Tensor.fill([1, 2], 0.0);
//   x.data = [0.0, 0.0]; // Softmax probabilities will be [0.5, 0.5]

//   final loss = x.crossEntropy([1]);
//   loss.backward();

//   final grads = x.grad;

//   // For Cross Entropy: grad = (Prob - Target) / T
//   // T = 1. Target vector Y = [0, 1]
//   // Grads: [0.5 - 0, 0.5 - 1] = [0.5, -0.5]

//   bool ok = closeEnough(grads[0], 0.5) && closeEnough(grads[1], -0.5);
//   print(
//     "CROSS_ENTROPY: ${ok ? '✅ PASS' : '❌ FAIL (Got $grads, expected [0.5, -0.5])'}",
//   );
// }

void verifyCrossEntropySimple() {
  final x = Tensor.fill([1, 2], 0.0);
  x.data = [0.0, 0.0]; // Softmax probabilities = [0.5, 0.5]

  final loss = x.crossEntropy([1]);
  loss.backward();

  final grads = x.grad;

  // With epsilon 0.1 and V=2:
  // Target Prob = 0.95
  // Grad = 0.5 - 0.95 = -0.45 (for target)
  // Grad = 0.5 - 0.05 =  0.45 (for non-target)

  bool ok = closeEnough(grads[0], 0.45) && closeEnough(grads[1], -0.45);
  print(
    "CROSS_ENTROPY: ${ok ? '✅ PASS' : '❌ FAIL (Got $grads, expected [0.45, -0.45])'}",
  );
}
