// ULTRATHINK MODE: Highly Optimized OpenCL Reduction Operations Kernel (f32)
// Production-ready GPU kernels for reduction operations (sum, dot product, norms, etc.)

// Configuration constants
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define UNROLL_FACTOR 8

// Parallel reduction sum using local memory
__kernel void reduce_sum_f32(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Load data into local memory
    float sum = 0.0f;
    int index = global_id;
    
    // Grid-stride loop for large arrays
    while (index < N) {
        sum += input[index];
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction in local memory
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result for this work group
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Optimized sum reduction with warp primitives
__kernel void reduce_sum_f32_optimized(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Load and accumulate multiple elements per thread
    float sum = 0.0f;
    
    #pragma unroll UNROLL_FACTOR
    for (int i = 0; i < UNROLL_FACTOR; i++) {
        int index = global_id * UNROLL_FACTOR + i;
        if (index < N) {
            sum += input[index];
        }
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Optimized reduction with unrolling
    if (group_size >= 512) {
        if (local_id < 256) scratch[local_id] += scratch[local_id + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (group_size >= 256) {
        if (local_id < 128) scratch[local_id] += scratch[local_id + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (group_size >= 128) {
        if (local_id < 64) scratch[local_id] += scratch[local_id + 64];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Final warp reduction (no synchronization needed)
    if (local_id < 32) {
        if (group_size >= 64) scratch[local_id] += scratch[local_id + 32];
        if (group_size >= 32) scratch[local_id] += scratch[local_id + 16];
        if (group_size >= 16) scratch[local_id] += scratch[local_id + 8];
        if (group_size >= 8) scratch[local_id] += scratch[local_id + 4];
        if (group_size >= 4) scratch[local_id] += scratch[local_id + 2];
        if (group_size >= 2) scratch[local_id] += scratch[local_id + 1];
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Dot product of two vectors
__kernel void dot_product_f32(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Compute partial dot product
    float sum = 0.0f;
    int index = global_id;
    
    while (index < N) {
        sum += A[index] * B[index];
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work group
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// High-precision dot product with Kahan summation
__kernel void dot_product_f32_kahan(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Kahan summation for better precision
    float sum = 0.0f;
    float c = 0.0f;
    int index = global_id;
    
    while (index < N) {
        float product = A[index] * B[index];
        float y = product - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Standard reduction for local memory
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Vector L2 norm (Euclidean norm)
__kernel void vector_norm_l2_f32(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    int index = global_id;
    
    while (index < N) {
        float val = input[index];
        sum_sq += val * val;
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce sum of squares
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Take square root of the result
    if (local_id == 0) {
        output[group_id] = sqrt(scratch[0]);
    }
}

// Vector L1 norm (Manhattan norm)
__kernel void vector_norm_l1_f32(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Compute sum of absolute values
    float sum_abs = 0.0f;
    int index = global_id;
    
    while (index < N) {
        sum_abs += fabs(input[index]);
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum_abs;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Vector infinity norm (maximum absolute value)
__kernel void vector_norm_inf_f32(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Find maximum absolute value
    float max_abs = 0.0f;
    int index = global_id;
    
    while (index < N) {
        max_abs = fmax(max_abs, fabs(input[index]));
        index += get_global_size(0);
    }
    
    scratch[local_id] = max_abs;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce using max operation
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Find minimum value and its index
__kernel void find_min_f32(
    __global const float* restrict input,
    __global float* restrict min_val,
    __global int* restrict min_idx,
    __local float* restrict val_scratch,
    __local int* restrict idx_scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Find local minimum
    float local_min = INFINITY;
    int local_min_idx = -1;
    int index = global_id;
    
    while (index < N) {
        if (input[index] < local_min) {
            local_min = input[index];
            local_min_idx = index;
        }
        index += get_global_size(0);
    }
    
    val_scratch[local_id] = local_min;
    idx_scratch[local_id] = local_min_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce to find global minimum
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            if (val_scratch[local_id + stride] < val_scratch[local_id]) {
                val_scratch[local_id] = val_scratch[local_id + stride];
                idx_scratch[local_id] = idx_scratch[local_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        min_val[group_id] = val_scratch[0];
        min_idx[group_id] = idx_scratch[0];
    }
}

// Find maximum value and its index
__kernel void find_max_f32(
    __global const float* restrict input,
    __global float* restrict max_val,
    __global int* restrict max_idx,
    __local float* restrict val_scratch,
    __local int* restrict idx_scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Find local maximum
    float local_max = -INFINITY;
    int local_max_idx = -1;
    int index = global_id;
    
    while (index < N) {
        if (input[index] > local_max) {
            local_max = input[index];
            local_max_idx = index;
        }
        index += get_global_size(0);
    }
    
    val_scratch[local_id] = local_max;
    idx_scratch[local_id] = local_max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce to find global maximum
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            if (val_scratch[local_id + stride] > val_scratch[local_id]) {
                val_scratch[local_id] = val_scratch[local_id + stride];
                idx_scratch[local_id] = idx_scratch[local_id + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        max_val[group_id] = val_scratch[0];
        max_idx[group_id] = idx_scratch[0];
    }
}

// Compute mean and variance in one pass
__kernel void mean_variance_f32(
    __global const float* restrict input,
    __global float* restrict mean_output,
    __global float* restrict var_output,
    __local float* restrict mean_scratch,
    __local float* restrict var_scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Welford's online algorithm for numerical stability
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    int index = global_id;
    
    while (index < N) {
        count++;
        float delta = input[index] - mean;
        mean += delta / count;
        float delta2 = input[index] - mean;
        m2 += delta * delta2;
        index += get_global_size(0);
    }
    
    mean_scratch[local_id] = mean * count; // Store sum for reduction
    var_scratch[local_id] = m2;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce sums
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            mean_scratch[local_id] += mean_scratch[local_id + stride];
            var_scratch[local_id] += var_scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        mean_output[group_id] = mean_scratch[0] / N; // Convert sum back to mean
        var_output[group_id] = var_scratch[0] / (N - 1); // Sample variance
    }
}

// Vectorized dot product using float4
__kernel void dot_product_f32_vectorized(
    __global const float4* restrict A,
    __global const float4* restrict B,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N4
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Compute partial dot product with vectorization
    float sum = 0.0f;
    int index = global_id;
    
    while (index < N4) {
        float4 a_vec = A[index];
        float4 b_vec = B[index];
        float4 prod = a_vec * b_vec;
        sum += prod.x + prod.y + prod.z + prod.w;
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Standard reduction
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Batch reduction operations
__kernel void batch_reduce_sum_f32(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N,
    const int batch_size,
    const int input_stride,
    const int output_stride
) {
    const int batch_id = get_global_id(1);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    if (batch_id >= batch_size) return;
    
    // Calculate offsets for this batch
    const int input_offset = batch_id * input_stride;
    const int output_offset = batch_id * output_stride;
    
    // Compute partial sum for this batch
    float sum = 0.0f;
    int index = get_global_id(0);
    
    while (index < N) {
        sum += input[input_offset + index];
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work group
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[output_offset + group_id] = scratch[0];
    }
}

// Mixed precision reductions
#ifdef cl_khr_fp16
__kernel void reduce_sum_mixed_precision(
    __global const half* restrict input,
    __global float* restrict output,
    __local float* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    // Accumulate in single precision for better accuracy
    float sum = 0.0f;
    int index = global_id;
    
    while (index < N) {
        sum += vload_half(0, &input[index]);
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Standard reduction
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}
#endif

// Double precision operations
#ifdef cl_khr_fp64
__kernel void reduce_sum_f64(
    __global const double* restrict input,
    __global double* restrict output,
    __local double* restrict scratch,
    const int N
) {
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int group_id = get_group_id(0);
    
    double sum = 0.0;
    int index = global_id;
    
    while (index < N) {
        sum += input[index];
        index += get_global_size(0);
    }
    
    scratch[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}
#endif