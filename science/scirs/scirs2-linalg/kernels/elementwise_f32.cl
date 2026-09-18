// ULTRATHINK MODE: Highly Optimized OpenCL Element-wise Operations Kernel (f32)
// Production-ready GPU kernels for element-wise matrix and vector operations

// Configuration constants
#define BLOCK_SIZE 256
#define VECTOR_WIDTH 4
#define UNROLL_FACTOR 8

// Basic element-wise addition
__kernel void elementwise_add_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = A[i] + B[i];
}

// Vectorized element-wise addition using float4
__kernel void elementwise_add_f32_vectorized(
    __global const float4* restrict A,
    __global const float4* restrict B,
    __global float4* restrict C,
    const int N4 // N divided by 4
) {
    const int i = get_global_id(0);
    
    if (i >= N4) return;
    
    C[i] = A[i] + B[i];
}

// Unrolled element-wise addition for better performance
__kernel void elementwise_add_f32_unrolled(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0) * UNROLL_FACTOR;
    
    if (i >= N) return;
    
    #pragma unroll UNROLL_FACTOR
    for (int k = 0; k < UNROLL_FACTOR && i + k < N; k++) {
        C[i + k] = A[i + k] + B[i + k];
    }
}

// Element-wise multiplication
__kernel void elementwise_mul_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = A[i] * B[i];
}

// Element-wise division with safety check
__kernel void elementwise_div_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N,
    const float epsilon
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    float b_val = B[i];
    C[i] = (fabs(b_val) > epsilon) ? A[i] / b_val : 0.0f;
}

// Element-wise power operation
__kernel void elementwise_pow_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = pow(A[i], B[i]);
}

// Element-wise exponential
__kernel void elementwise_exp_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = exp(A[i]);
}

// Element-wise logarithm with safety check
__kernel void elementwise_log_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N,
    const float epsilon
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    float a_val = A[i];
    B[i] = (a_val > epsilon) ? log(a_val) : -INFINITY;
}

// Element-wise square root
__kernel void elementwise_sqrt_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    float a_val = A[i];
    B[i] = (a_val >= 0.0f) ? sqrt(a_val) : 0.0f;
}

// Element-wise sine
__kernel void elementwise_sin_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = sin(A[i]);
}

// Element-wise cosine
__kernel void elementwise_cos_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = cos(A[i]);
}

// Element-wise tangent
__kernel void elementwise_tan_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = tan(A[i]);
}

// Element-wise absolute value
__kernel void elementwise_abs_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = fabs(A[i]);
}

// Element-wise maximum of two arrays
__kernel void elementwise_max_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = fmax(A[i], B[i]);
}

// Element-wise minimum of two arrays
__kernel void elementwise_min_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = fmin(A[i], B[i]);
}

// Scalar addition (A + scalar)
__kernel void scalar_add_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const float scalar,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = A[i] + scalar;
}

// Scalar multiplication (A * scalar)
__kernel void scalar_mul_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const float scalar,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = A[i] * scalar;
}

// Linear combination (alpha * A + beta * B)
__kernel void linear_combination_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const float alpha,
    const float beta,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = alpha * A[i] + beta * B[i];
}

// Fused multiply-add (A * B + C)
__kernel void fused_multiply_add_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global const float* restrict C,
    __global float* restrict D,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    D[i] = fma(A[i], B[i], C[i]);
}

// Element-wise comparison (A > B ? 1.0 : 0.0)
__kernel void elementwise_greater_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = (A[i] > B[i]) ? 1.0f : 0.0f;
}

// Element-wise comparison (A < B ? 1.0 : 0.0)
__kernel void elementwise_less_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = (A[i] < B[i]) ? 1.0f : 0.0f;
}

// Element-wise equality check with tolerance
__kernel void elementwise_equal_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const float tolerance,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    C[i] = (fabs(A[i] - B[i]) <= tolerance) ? 1.0f : 0.0f;
}

// Element-wise clamp operation
__kernel void elementwise_clamp_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const float min_val,
    const float max_val,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = clamp(A[i], min_val, max_val);
}

// Element-wise sign function
__kernel void elementwise_sign_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    float a_val = A[i];
    B[i] = (a_val > 0.0f) ? 1.0f : ((a_val < 0.0f) ? -1.0f : 0.0f);
}

// Element-wise sigmoid activation
__kernel void elementwise_sigmoid_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = 1.0f / (1.0f + exp(-A[i]));
}

// Element-wise tanh activation
__kernel void elementwise_tanh_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = tanh(A[i]);
}

// Element-wise ReLU activation
__kernel void elementwise_relu_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    B[i] = fmax(0.0f, A[i]);
}

// Element-wise Leaky ReLU activation
__kernel void elementwise_leaky_relu_f32(
    __global const float* restrict A,
    __global float* restrict B,
    const float alpha,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    float a_val = A[i];
    B[i] = (a_val > 0.0f) ? a_val : alpha * a_val;
}

// Vectorized operations with higher throughput
__kernel void vectorized_operations_f32(
    __global const float4* restrict A,
    __global const float4* restrict B,
    __global float4* restrict C,
    const int operation, // 0=add, 1=mul, 2=max, 3=min
    const int N4
) {
    const int i = get_global_id(0);
    
    if (i >= N4) return;
    
    float4 a_vec = A[i];
    float4 b_vec = B[i];
    
    switch (operation) {
        case 0: // Addition
            C[i] = a_vec + b_vec;
            break;
        case 1: // Multiplication
            C[i] = a_vec * b_vec;
            break;
        case 2: // Maximum
            C[i] = fmax(a_vec, b_vec);
            break;
        case 3: // Minimum
            C[i] = fmin(a_vec, b_vec);
            break;
        default:
            C[i] = a_vec + b_vec; // Fallback to addition
            break;
    }
}

// Mixed precision element-wise operations
#ifdef cl_khr_fp16
__kernel void elementwise_add_mixed_precision(
    __global const half* restrict A,
    __global const half* restrict B,
    __global float* restrict C,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    // Convert half to float for computation
    float a_val = vload_half(0, &A[i]);
    float b_val = vload_half(0, &B[i]);
    
    C[i] = a_val + b_val;
}
#endif

// Batch element-wise operations
__kernel void batch_elementwise_add_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N,
    const int batch_size,
    const int A_stride,
    const int B_stride,
    const int C_stride
) {
    const int batch_id = get_global_id(1);
    const int i = get_global_id(0);
    
    if (batch_id >= batch_size || i >= N) return;
    
    // Calculate offsets for this batch
    const int A_offset = batch_id * A_stride;
    const int B_offset = batch_id * B_stride;
    const int C_offset = batch_id * C_stride;
    
    C[C_offset + i] = A[A_offset + i] + B[B_offset + i];
}

// In-place operations for memory efficiency
__kernel void inplace_elementwise_add_f32(
    __global float* restrict A,
    __global const float* restrict B,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    A[i] += B[i];
}

__kernel void inplace_scalar_mul_f32(
    __global float* restrict A,
    const float scalar,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= N) return;
    
    A[i] *= scalar;
}