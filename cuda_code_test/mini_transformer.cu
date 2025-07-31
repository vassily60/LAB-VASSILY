#include <iostream>
#include <cstdlib>
#include <ctime>


const int M = 4;  // rows in X
const int D = 4;  // dim

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

float* transpose(float* mat, int rows, int cols) {
    float* trans = new float[rows * cols];
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            trans[c * rows + r] = mat[r * cols + c];
    return trans;
}

void gpu_matmul(float* A, float* B, float* C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void print_matrix(const char* name, float* mat, int rows, int cols) {
    std::cout << name << ":\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            std::cout << mat[r * cols + c] << "\t";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    srand(time(0));

    // Allocate host matrices
    float *X = new float[M * D];
    float *Wq = new float[D * D], *Wk = new float[D * D], *Wv = new float[D * D];
    float *Q = new float[M * D], *K = new float[M * D], *V = new float[M * D];
    float *A = new float[M * M], *O = new float[M * D];

    // Randomly initialize
    for (int i = 0; i < M * D; ++i) {
        X[i] = static_cast<float>(rand() % 10);
    }

    for (int i = 0; i < D * D; ++i) {
        Wq[i] = static_cast<float>(rand() % 10);
        Wk[i] = static_cast<float>(rand() % 10);
        Wv[i] = static_cast<float>(rand() % 10);
    }

    // Multiply: Q = X × Wq, K = X × Wk, V = X × Wv
    gpu_matmul(X, Wq, Q, M, D, D);
    gpu_matmul(X, Wk, K, M, D, D);
    gpu_matmul(X, Wv, V, M, D, D);

    // Transpose K → K_T
    float* K_T = transpose(K, M, D);

    // Multiply: A = Q × Kᵗ (M×D × D×M = M×M)
    gpu_matmul(Q, K_T, A, M, M, D);

    // Multiply: O = A × V (M×M × M×D = M×D)
    gpu_matmul(A, V, O, M, D, M);

    // Display results
    print_matrix("X", X, M, D);
    print_matrix("Wq", Wq, D, D);
    print_matrix("Q = X × Wq", Q, M, D);
    print_matrix("A = Q × Kᵗ", A, M, M);
    print_matrix("Output = A × V", O, M, D);

    // Cleanup
    delete[] X; delete[] Wq; delete[] Wk; delete[] Wv;
    delete[] Q; delete[] K; delete[] V;
    delete[] A; delete[] O; delete[] K_T;

    return 0;
}
