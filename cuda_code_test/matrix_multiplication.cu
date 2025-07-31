// matrix_mul.cu
#include <cuda_runtime.h>
#include <stdio.h>

#define N 16  // matrix size N x N

// CUDA kernel to multiply two matrices A and B, store in C
__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < n && col < n) {
        float val = 0;
        for (int k = 0; k < n; ++k) {
            val += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = val;
    }
}

int main() {
    int size = N * N * sizeof(float);

    // Host matrices
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // or random values
        h_B[i] = 2.0f;
    }

    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel: 16x16 matrix, so 16x16 threads (or blockDim = 16x16, gridDim=1x1)
    dim3 blockDim(16, 16);
    dim3 gridDim(1, 1);
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
