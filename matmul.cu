#include <iostream>

// This macro automatically chooses the correct syntax.
// In WSL (Linux), it will use the __attribute__... part.
#if defined(_WIN32) || defined(_WIN64)
  #define DLLEXPORT __declspec(dllexport)
#else
  #define DLLEXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief CUDA kernel for matrix multiplication (C = A * B).
 */
__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float C_value = 0.0;
        for (int i = 0; i < K; ++i) {
            C_value += A[row * K + i] * B[i * N + col];
        }
        C[row * M + col] = C_value;
    }
}

__global__ void l2NormalizeKernel(float* data, int rows, int cols, float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum_sq = 0.0f;
        
        // 1. Calculate squared sum for the row
        for (int i = 0; i < cols; ++i) {
            float val = data[row * cols + i];
            sum_sq += val * val;
        }

        // 2. Calculate the norm (with epsilon safety)
        float norm = sqrtf(sum_sq + epsilon);

        // 3. Divide elements by the norm
        for (int i = 0; i < cols; ++i) {
            data[row * cols + i] /= norm;
        }
    }
}

/**
 * @brief C-style wrapper function that Dart FFI will call.
 */
extern "C" DLLEXPORT void matrix_multiply_cuda(
    float* host_A, 
    float* host_B, 
    float* host_C, 
    int M, 
    int N, 
    int K
) {
    float *dev_A, *dev_B, *dev_C;

    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // 1. Allocate GPU memory
    cudaMalloc(&dev_A, size_A);
    cudaMalloc(&dev_B, size_B);
    cudaMalloc(&dev_C, size_C);

    // 2. Copy data from CPU to GPU
    cudaMemcpy(dev_A, host_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, size_B, cudaMemcpyHostToDevice);

    // 3. Configure kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);

    printf("Launching kernel...\n");
    // 4. Launch kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(dev_A, dev_B, dev_C, M, N, K);
    
    cudaDeviceSynchronize();
    printf("Kernel finished.\n");

    // 5. Copy result from GPU to CPU
    cudaMemcpy(host_C, dev_C, size_C, cudaMemcpyDeviceToHost);

    // 6. Free GPU memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
}
