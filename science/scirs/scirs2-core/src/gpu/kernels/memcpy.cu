// Memory copy kernel

extern "C" __global__ void memcpy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

extern "C" __global__ void memcpy_f64(
    const double* __restrict__ src,
    double* __restrict__ dst,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Strided memory copy
extern "C" __global__ void memcpy_strided_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int n,
    const int src_stride,
    const int dst_stride
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        dst[idx * dst_stride] = src[idx * src_stride];
    }
}