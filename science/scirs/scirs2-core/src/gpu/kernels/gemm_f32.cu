// GEMM kernel for f32
// C = alpha * A * B + beta * C

extern "C" __global__ void gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const float alpha,
    const float beta,
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldb,
    const int ldc
) {
    // Thread and block indices
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for tile-based computation
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    // Accumulator for this thread
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        // Load tile from A
        if (row < M && tile * 16 + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * lda + tile * 16 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B
        if (col < N && tile * 16 + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}