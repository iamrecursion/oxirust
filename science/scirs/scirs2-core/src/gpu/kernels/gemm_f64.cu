// GEMM kernel for f64
// C = alpha * A * B + beta * C

extern "C" __global__ void gemm_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    const double alpha,
    const double beta,
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldb,
    const int ldc
) {
    // Similar to f32 version but with double precision
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        if (row < M && tile * 16 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * lda + tile * 16 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        if (col < N && tile * 16 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}