/**
 * Ultra-optimized CUDA kernel for Runge-Kutta 4th order final combination
 * Computes y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6 for RK4 method
 * 
 * This kernel combines all RK4 stages with:
 * - Fused multiply-add (FMA) operations
 * - Optimized memory bandwidth utilization
 * - Reduced arithmetic operations through algebraic optimization
 */

extern "C" __global__ void rk4_combine(
    const double* __restrict__ y,     // Input solution vector
    const double* __restrict__ k1,    // k1 from stage 1
    const double* __restrict__ k2,    // k2 from stage 2
    const double* __restrict__ k3,    // k3 from stage 3
    const double* __restrict__ k4,    // k4 from stage 4
    double* __restrict__ y_new,       // Output solution vector
    const double h,                   // Step size (for verification)
    const int n                       // Problem dimension
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Load all values with coalesced memory access
    const double y_val = y[idx];
    const double k1_val = k1[idx];
    const double k2_val = k2[idx];
    const double k3_val = k3[idx];
    const double k4_val = k4[idx];
    
    // RK4 combination formula: y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
    // Using FMA operations for maximum precision and performance
    
    // Compute 2*k2 + 2*k3 with single multiplication
    const double k23_term = 2.0 * (k2_val + k3_val);
    
    // Compute weighted sum: k1 + 2*k2 + 2*k3 + k4
    const double weighted_sum = k1_val + k23_term + k4_val;
    
    // Final result: y + weighted_sum/6
    const double y_new_val = y_val + weighted_sum * (1.0 / 6.0);
    
    // Store result
    y_new[idx] = y_new_val;
}

/**
 * Single-precision version with optimized arithmetic
 */
extern "C" __global__ void rk4_combine_f32(
    const float* __restrict__ y,
    const float* __restrict__ k1,
    const float* __restrict__ k2,
    const float* __restrict__ k3,
    const float* __restrict__ k4,
    float* __restrict__ y_new,
    const float h,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    const float y_val = y[idx];
    const float k1_val = k1[idx];
    const float k2_val = k2[idx];
    const float k3_val = k3[idx];
    const float k4_val = k4[idx];
    
    // Optimized combination using FMA
    const float k23_term = 2.0f * (k2_val + k3_val);
    const float weighted_sum = k1_val + k23_term + k4_val;
    const float y_new_val = y_val + weighted_sum * (1.0f / 6.0f);
    
    y_new[idx] = y_new_val;
}

/**
 * Vectorized version for larger data types (experimental)
 * Uses float4 for better memory throughput on compatible hardware
 */
extern "C" __global__ void rk4_combine_vectorized(
    const float4* __restrict__ y,
    const float4* __restrict__ k1,
    const float4* __restrict__ k2,
    const float4* __restrict__ k3,
    const float4* __restrict__ k4,
    float4* __restrict__ y_new,
    const float h,
    const int n_vec  // n/4 for float4 vectorization
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_vec) return;
    
    // Load 4 elements at once
    const float4 y_val = y[idx];
    const float4 k1_val = k1[idx];
    const float4 k2_val = k2[idx];
    const float4 k3_val = k3[idx];
    const float4 k4_val = k4[idx];
    
    // Vectorized RK4 combination
    float4 result;
    result.x = y_val.x + (k1_val.x + 2.0f*(k2_val.x + k3_val.x) + k4_val.x) * (1.0f/6.0f);
    result.y = y_val.y + (k1_val.y + 2.0f*(k2_val.y + k3_val.y) + k4_val.y) * (1.0f/6.0f);
    result.z = y_val.z + (k1_val.z + 2.0f*(k2_val.z + k3_val.z) + k4_val.z) * (1.0f/6.0f);
    result.w = y_val.w + (k1_val.w + 2.0f*(k2_val.w + k3_val.w) + k4_val.w) * (1.0f/6.0f);
    
    y_new[idx] = result;
}