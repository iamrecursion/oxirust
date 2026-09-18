/**
 * Ultra-optimized CUDA kernel for error estimation in adaptive ODE solvers
 * Computes relative and absolute error estimates for step size control
 * 
 * This kernel implements error estimation with:
 * - Numerically stable error computation
 * - Efficient reduction operations
 * - Support for mixed absolute/relative tolerances
 */

extern "C" __global__ void error_estimate(
    const double* __restrict__ y1,    // Solution from method 1 (higher order)
    const double* __restrict__ y2,    // Solution from method 2 (lower order)
    double* __restrict__ error,       // Output error estimate
    const double rtol,                // Relative tolerance
    const double atol,                // Absolute tolerance
    const int n                       // Problem dimension
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Load values
    const double y1_val = y1[idx];
    const double y2_val = y2[idx];
    
    // Compute absolute error
    const double abs_error = fabs(y1_val - y2_val);
    
    // Compute scaling factor for relative error
    // scale = atol + rtol * max(|y1|, |y2|)
    const double y_scale = fmax(fabs(y1_val), fabs(y2_val));
    const double scale = atol + rtol * y_scale;
    
    // Relative error estimate
    const double rel_error = abs_error / fmax(scale, 1e-16); // Avoid division by zero
    
    // Store the relative error for norm computation
    error[idx] = rel_error;
}

/**
 * Single-precision version
 */
extern "C" __global__ void error_estimate_f32(
    const float* __restrict__ y1,
    const float* __restrict__ y2,
    float* __restrict__ error,
    const float rtol,
    const float atol,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    const float y1_val = y1[idx];
    const float y2_val = y2[idx];
    
    const float abs_error = fabsf(y1_val - y2_val);
    const float y_scale = fmaxf(fabsf(y1_val), fabsf(y2_val));
    const float scale = atol + rtol * y_scale;
    const float rel_error = abs_error / fmaxf(scale, 1e-7f);
    
    error[idx] = rel_error;
}

/**
 * Kernel for computing the norm of error vector (L2 norm)
 * Used for global error estimation in adaptive step control
 */
extern "C" __global__ void error_norm_squared(
    const double* __restrict__ error,    // Input error vector
    double* __restrict__ partial_sums,   // Output partial sums for reduction
    const int n                          // Problem dimension
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for block-level reduction
    extern __shared__ double sdata[];
    
    // Load data and compute squared error
    double sum = 0.0;
    if (idx < n) {
        const double err = error[idx];
        sum = err * err;
    }
    sdata[tid] = sum;
    
    __syncthreads();
    
    // Block-level reduction using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result of this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * Single-precision version of norm computation
 */
extern "C" __global__ void error_norm_squared_f32(
    const float* __restrict__ error,
    float* __restrict__ partial_sums,
    const int n
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata_f32[];
    
    float sum = 0.0f;
    if (idx < n) {
        const float err = error[idx];
        sum = err * err;
    }
    sdata_f32[tid] = sum;
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_f32[tid] += sdata_f32[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata_f32[0];
    }
}

/**
 * Optimized kernel for weighted error norm computation
 * Combines error computation and norm calculation in a single pass
 */
extern "C" __global__ void weighted_error_norm(
    const double* __restrict__ y1,
    const double* __restrict__ y2,
    const double* __restrict__ weights,  // Pre-computed error weights
    double* __restrict__ partial_sums,
    const int n
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ double sdata_weighted[];
    
    double sum = 0.0;
    if (idx < n) {
        const double y1_val = y1[idx];
        const double y2_val = y2[idx];
        const double weight = weights[idx];
        
        const double error = (y1_val - y2_val) * weight;
        sum = error * error;
    }
    sdata_weighted[tid] = sum;
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_weighted[tid] += sdata_weighted[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata_weighted[0];
    }
}