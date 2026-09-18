// Fill kernel - set all elements to a value

extern "C" __global__ void fill_f32(
    float* __restrict__ dst,
    const float value,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx] = value;
    }
}

extern "C" __global__ void fill_f64(
    double* __restrict__ dst,
    const double value,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx] = value;
    }
}

// Fill with linear sequence (like arange)
extern "C" __global__ void fill_sequence_f32(
    float* __restrict__ dst,
    const float start,
    const float step,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx] = start + idx * step;
    }
}