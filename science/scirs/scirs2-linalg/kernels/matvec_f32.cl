// ULTRATHINK MODE: Highly Optimized OpenCL Matrix-Vector Multiplication Kernel (f32)
// Production-ready GPU kernel with advanced optimizations

// Configuration constants
#define BLOCK_SIZE 256
#define VECTOR_WIDTH 4
#define UNROLL_FACTOR 8

// Basic matrix-vector multiplication
__kernel void matvec_f32_basic(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += A[i * N + j] * x[j];
    }
    
    y[i] = sum;
}

// Optimized matrix-vector multiplication with vectorization
__kernel void matvec_f32_optimized(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    const int row_offset = i * N;
    
    // Vectorized loop with unrolling
    int j = 0;
    for (; j <= N - UNROLL_FACTOR; j += UNROLL_FACTOR) {
        #pragma unroll UNROLL_FACTOR
        for (int k = 0; k < UNROLL_FACTOR; k++) {
            sum += A[row_offset + j + k] * x[j + k];
        }
    }
    
    // Handle remaining elements
    for (; j < N; j++) {
        sum += A[row_offset + j] * x[j];
    }
    
    y[i] = sum;
}

// Matrix-vector multiplication with reduction using local memory
__kernel void matvec_f32_reduction(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N,
    __local float* scratch
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Each work group processes one row
    const int row = group_id;
    
    if (row >= M) return;
    
    // Each thread processes multiple elements
    float sum = 0.0f;
    const int row_offset = row * N;
    
    for (int j = local_id; j < N; j += group_size) {
        sum += A[row_offset + j] * x[j];
    }
    
    // Store partial sum in local memory
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work group
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (local_id == 0) {
        y[row] = scratch[0];
    }
}

// High-precision matrix-vector multiplication with Kahan summation
__kernel void matvec_f32_high_precision(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    float c = 0.0f; // Kahan summation compensation
    const int row_offset = i * N;
    
    for (int j = 0; j < N; j++) {
        float product = A[row_offset + j] * x[j];
        float y_val = product - c;
        float t = sum + y_val;
        c = (t - sum) - y_val;
        sum = t;
    }
    
    y[i] = sum;
}

// Vectorized matrix-vector multiplication using float4
__kernel void matvec_f32_vectorized(
    __global const float4* restrict A,
    __global const float4* restrict x,
    __global float* restrict y,
    const int M,
    const int N4 // N divided by 4
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float4 sum4 = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    const int row_offset = i * N4;
    
    for (int j = 0; j < N4; j++) {
        float4 a_vec = A[row_offset + j];
        float4 x_vec = x[j];
        sum4 += a_vec * x_vec;
    }
    
    // Horizontal sum of the vector
    y[i] = sum4.x + sum4.y + sum4.z + sum4.w;
}

// Blocked matrix-vector multiplication for better cache utilization
__kernel void matvec_f32_blocked(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N,
    const int blockSize
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    const int row_offset = i * N;
    
    // Process in blocks for better cache utilization
    for (int jb = 0; jb < N; jb += blockSize) {
        int jEnd = min(jb + blockSize, N);
        
        float blockSum = 0.0f;
        for (int j = jb; j < jEnd; j++) {
            blockSum += A[row_offset + j] * x[j];
        }
        sum += blockSum;
    }
    
    y[i] = sum;
}

// Transpose matrix-vector multiplication (y = A^T * x)
__kernel void matvec_f32_transpose(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N,
    __local float* scratch
) {
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Each work group processes one column of A (row of A^T)
    const int col = group_id;
    
    if (col >= N) return;
    
    // Each thread processes multiple rows
    float sum = 0.0f;
    
    for (int i = local_id; i < M; i += group_size) {
        sum += A[i * N + col] * x[i];
    }
    
    // Store partial sum in local memory
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work group
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (local_id == 0) {
        y[col] = scratch[0];
    }
}

// Adaptive matrix-vector multiplication
__kernel void matvec_f32_adaptive(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N,
    const int strategy, // 0=basic, 1=optimized, 2=reduction, 3=vectorized
    __local float* scratch
) {
    switch (strategy) {
        case 0: // Basic
            matvec_f32_basic(A, x, y, M, N);
            break;
            
        case 1: // Optimized
            matvec_f32_optimized(A, x, y, M, N);
            break;
            
        case 2: // Reduction
            matvec_f32_reduction(A, x, y, M, N, scratch);
            break;
            
        default: // Fallback to basic
            matvec_f32_basic(A, x, y, M, N);
            break;
    }
}

// Mixed precision matrix-vector multiplication
#ifdef cl_khr_fp16
__kernel void matvec_mixed_precision(
    __global const half* restrict A,
    __global const half* restrict x,
    __global float* restrict y,
    const int M,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    const int row_offset = i * N;
    
    for (int j = 0; j < N; j++) {
        // Convert half to float for computation
        float a_val = vload_half(0, &A[row_offset + j]);
        float x_val = vload_half(0, &x[j]);
        sum += a_val * x_val;
    }
    
    y[i] = sum;
}
#endif

// Sparse matrix-vector multiplication (CSR format)
__kernel void spmv_csr_f32(
    __global const float* restrict values,
    __global const int* restrict col_indices,
    __global const int* restrict row_ptr,
    __global const float* restrict x,
    __global float* restrict y,
    const int M
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    float sum = 0.0f;
    const int start = row_ptr[i];
    const int end = row_ptr[i + 1];
    
    for (int j = start; j < end; j++) {
        sum += values[j] * x[col_indices[j]];
    }
    
    y[i] = sum;
}

// Batch matrix-vector multiplication
__kernel void batch_matvec_f32(
    __global const float* restrict A,
    __global const float* restrict x,
    __global float* restrict y,
    const int M,
    const int N,
    const int batch_size,
    const int A_stride,
    const int x_stride,
    const int y_stride
) {
    const int batch_id = get_global_id(2);
    const int i = get_global_id(0);
    
    if (batch_id >= batch_size || i >= M) return;
    
    // Calculate offsets for this batch
    const int A_offset = batch_id * A_stride;
    const int x_offset = batch_id * x_stride;
    const int y_offset = batch_id * y_stride;
    
    float sum = 0.0f;
    const int row_offset = A_offset + i * N;
    
    for (int j = 0; j < N; j++) {
        sum += A[row_offset + j] * x[x_offset + j];
    }
    
    y[y_offset + i] = sum;
}

// Double precision matrix-vector multiplication
#ifdef cl_khr_fp64
__kernel void matvec_f64_basic(
    __global const double* restrict A,
    __global const double* restrict x,
    __global double* restrict y,
    const int M,
    const int N
) {
    const int i = get_global_id(0);
    
    if (i >= M) return;
    
    double sum = 0.0;
    const int row_offset = i * N;
    
    for (int j = 0; j < N; j++) {
        sum += A[row_offset + j] * x[j];
    }
    
    y[i] = sum;
}
#endif

// Default kernel for backward compatibility
__kernel void matvec_f32(__global const float* A,
                         __global const float* x,
                         __global float* y,
                         const int M,
                         const int N) {
    matvec_f32_optimized(A, x, y, M, N);
}