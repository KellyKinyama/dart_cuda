import 'dart:typed_data';
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import 'engine.dart';

class Tensor {
  // final int M;
  // final int K;
  // final int N;
  // int cols;
  // int rows;
  List<int> shape;

  late Float32List data; // = hostA.asTypedList(M * K);
  late Float32List grad; // = hostA.asTypedList(M * K);

  Tensor(
    this.shape,
    List<double> data,
    //   {
    //   required this.rows,
    //   required this.cols,
    // }
  ) {
    this.data = Float32List.fromList(data);
    grad = Float32List(this.data.length);
  }

  factory Tensor.fromFloatList(Pointer<Float> matrix, int rows, int cols) {
    final list = matrix.asTypedList(rows * cols);
    return Tensor([rows, cols], list);
  }

  Tensor matMul(Tensor other) {
    // Define matrix dimensions: A(M x K) * B(K x N) = C(M x N)
    int M = shape[0];
    int K = shape[1];
    int N = other.shape[1];

    print("M: $M, K: $K, N: $N");

    // 4. Allocate memory for the matrices that C can understand.
    // Matrices are stored as flat, 1D arrays in row-major order.
    final Pointer<Float> hostA = calloc<Float>(M * K);
    final Pointer<Float> hostB = calloc<Float>(K * N);
    final Pointer<Float> hostC = calloc<Float>(M * N); // For the result

    final matrixA = hostA.asTypedList(M * K);
    final matrixB = hostB.asTypedList(K * N);

    // Matrix A (4x3)
    matrixA.setAll(0, data);

    // Matrix B (3x4)
    matrixB.setAll(0, other.data);

    print('Matrix A:');
    printMatrix(hostA, M, K);
    print('Matrix B:');
    printMatrix(hostB, K, N);

    // 6. Call the CUDA function!
    print('\nCalling CUDA function...');
    engine.matMulCuda(hostA, hostB, hostC, M, N, K);
    print('CUDA function returned.');

    // 7. Print the result from matrix C.
    print('\nResult Matrix C:');
    printMatrix(hostC, M, N);

    // 8. CRITICAL: Free the allocated memory to avoid leaks.
    calloc.free(hostA);
    calloc.free(hostB);
    calloc.free(hostC);

    return Tensor([M, N], hostC.asTypedList(M * N));
  }

  /// Helper function to print a matrix from a flat array.
  String printMatrix() {
    // final list = matrix.asTypedList(rows * cols);
    print("Rows: ${shape[0]}, Cols: ${shape[1]}");
    String output = "";
    for (int i = 0; i < shape[0]; i++) {
      final row = data.sublist(i * shape[1], (i + 1) * shape[1]);
      output += '\n ${row.map((e) => e.toString()).join(' ')}';
      // print(row.map((e) => e.toStringAsFixed(1).padLeft(5)).join(' '));
    }
    return output;
  }
}

void main() {
  const M = 4;
  const K = 3;
  const N = 4;

  // Matrix A (4x3)
  final matrixA = Tensor(
    [4, 3],
    [
      1, 2, 3, // row 0
      4, 5, 6, // row 1
      7, 8, 9, // row 2
      1, 1, 1, // row 3
    ],
  );

  // Matrix B (3x4)
  final matrixB = Tensor(
    [3, 4],
    [
      9, 8, 7, 6, // row 0
      5, 4, 3, 2, // row 1
      1, 2, 3, 4, // row 2
    ],
  );

  final out = matrixA.matMul(matrixB);
}
