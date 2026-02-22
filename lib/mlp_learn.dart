import 'dart:math';

import 'gpu_tensor.dart';

void main() {
  print('\n Matrix multiplcation');
  // Define matrix dimensions: A(M x K) * B(K x N) = C(M x N)
  // Matrix A (4x3)
  final input = Tensor.random([256, 1]);

  // Matrix B (3x4)
  final weight1 = Tensor.random([1, 256]);
  final biases1 = Tensor.random([1, 256]);
  final weight2 = Tensor.random([256, 10]);

  print('\n Matrix multiplcation');
  // final matMuled = matrixA.matmul(matrixB);
  final matMuled1 = input * weight1 + biases1;

  final reLued = matMuled1.relu();

  final matMuled2 = reLued * weight2;
  final reLued2 = matMuled2.relu();

  print("matrix: ${reLued2.printMatrix()}");
}
