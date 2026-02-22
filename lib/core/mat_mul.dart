part of 'tensor.dart';

//  import 'package:dart_cuda/gpu_tensor.dart';

extension TensorPart1 on Tensor {
  Tensor matMul(Tensor other) {
    assert(
      shape[1] == other.shape[0],
      "Dimension mismatch: ${shape[1]} != ${other.shape[0]}",
    );

    // CRITICAL: Capture 'this' and 'other' as local final variables.
    // This ensures the backward closure points to the EXACT weight buffers.
    final Tensor input = this;
    final Tensor weights = other;

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
    // printMatrix(hostA, M, K);
    print('Matrix B:');
    // printMatrix(hostB, K, N);

    // 6. Call the CUDA function!
    print('\nCalling CUDA function...');
    engine.matMulCuda(hostA, hostB, hostC, M, N, K);
    print('CUDA function returned.');

    // 7. Print the result from matrix C.
    print('\nResult Matrix C:');
    // printMatrix(hostC, M, N);

    // 8. CRITICAL: Free the allocated memory to avoid leaks.
    calloc.free(hostA);
    calloc.free(hostB);
    calloc.free(hostC);

    final out = Tensor([M, N], hostC.asTypedList(M * N));

    out.onBackward = () {
      for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < N; j++) {
            double gradOut = out.grad[i * N + j];
            // Update the locally captured weight and input tensors
            input.grad[i * K + k] += weights.data[k * N + j] * gradOut;
            weights.grad[k * N + j] += input.data[i * K + k] * gradOut;
          }
        }
      }
    };

    return out;
  }
}
