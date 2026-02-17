import 'dart:ffi';
import 'package:ffi/ffi.dart';

// 1. Define the FFI function signatures to match the C/C++ wrapper.

// This is the native C function signature.
typedef MatMulFunc = Void Function(
    Pointer<Float> a, Pointer<Float> b, Pointer<Float> c, Int32 M, Int32 N, Int32 K);

// This is the Dart-friendly version of the function.
typedef MatMul = void Function(
    Pointer<Float> a, Pointer<Float> b, Pointer<Float> c, int M, int N, int K);

void main() {
  // Define matrix dimensions: A(M x K) * B(K x N) = C(M x N)
  const M = 4;
  const K = 3;
  const N = 4;

  // 2. Load the compiled .so library from the current directory.
  final dylib = DynamicLibrary.open('./libmatmul.so');

  // 3. Look up the function by its C name.
  final matMulCuda =
      dylib.lookupFunction<MatMulFunc, MatMul>('matrix_multiply_cuda');

  // 4. Allocate memory for the matrices that C can understand.
  // Matrices are stored as flat, 1D arrays in row-major order.
  final Pointer<Float> hostA = calloc<Float>(M * K);
  final Pointer<Float> hostB = calloc<Float>(K * N);
  final Pointer<Float> hostC = calloc<Float>(M * N); // For the result

  // 5. Populate the input matrices A and B with some data.
  final matrixA = hostA.asTypedList(M * K);
  final matrixB = hostB.asTypedList(K * N);

  // Matrix A (4x3)
  matrixA.setAll(0, [
    1, 2, 3,  // row 0
    4, 5, 6,  // row 1
    7, 8, 9,  // row 2
    1, 1, 1,  // row 3
  ]);

  // Matrix B (3x4)
  matrixB.setAll(0, [
    9, 8, 7, 6, // row 0
    5, 4, 3, 2, // row 1
    1, 2, 3, 4, // row 2
  ]);

  print('Matrix A:');
  printMatrix(hostA, M, K);
  print('Matrix B:');
  printMatrix(hostB, K, N);

  // 6. Call the CUDA function!
  print('\nCalling CUDA function...');
  matMulCuda(hostA, hostB, hostC, M, N, K);
  print('CUDA function returned.');

  // 7. Print the result from matrix C.
  print('\nResult Matrix C:');
  printMatrix(hostC, M, N);

  // 8. CRITICAL: Free the allocated memory to avoid leaks.
  calloc.free(hostA);
  calloc.free(hostB);
  calloc.free(hostC);
}

/// Helper function to print a matrix from a flat array.
void printMatrix(Pointer<Float> matrix, int rows, int cols) {
  final list = matrix.asTypedList(rows * cols);
  for (int i = 0; i < rows; i++) {
    final row = list.sublist(i * cols, (i + 1) * cols);
    print(row.map((e) => e.toStringAsFixed(1).padLeft(5)).join(' '));
  }
}
