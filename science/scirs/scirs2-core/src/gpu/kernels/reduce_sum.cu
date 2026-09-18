// Sum reduction kernel

extern "C" __global__ void reduce_sum_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    extern __shared__ float sdata[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void reduce_sum_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    const int n
) {
    extern __shared__ double sdata_d[];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata_d[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_d[tid] += sdata_d[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata_d[0];
    }
}