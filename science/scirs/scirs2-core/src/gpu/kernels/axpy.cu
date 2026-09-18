// AXPY kernel: y = alpha * x + y

extern "C" __global__ void axpy_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float alpha,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

extern "C" __global__ void axpy_f64(
    const double* __restrict__ x,
    double* __restrict__ y,
    const double alpha,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}