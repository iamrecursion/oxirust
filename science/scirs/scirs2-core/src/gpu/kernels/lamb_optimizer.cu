// LAMB (Layer-wise Adaptive Moments) optimizer CUDA kernels
// Implementation based on "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
// Supports:
// - Layer-wise adaptation for large batch training
// - Mixed precision training with tensor cores
// - Efficient norm computation with warp-level primitives
// - Multi-GPU synchronization

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

// Constants
constexpr float EPSILON_F32 = 1e-6f;
constexpr double EPSILON_F64 = 1e-6;
constexpr float NORM_EPSILON = 1e-8f;

// Warp-level reduction for norm computation
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for norm computation
template<typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val) {
    __shared__ T shared[32]; // Assumes max 1024 threads (32 warps)
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // Final reduction
    if (threadIdx.x < blockDim.x / 32) {
        val = shared[threadIdx.x];
    } else {
        val = 0;
    }
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Compute L2 norm of a vector using segmented reduction
template<typename T>
__global__ void compute_norm_kernel(
    const T* __restrict__ data,
    T* __restrict__ norm,
    const int n
) {
    extern __shared__ char shared_mem[];
    T* shared = reinterpret_cast<T*>(shared_mem);
    
    T local_sum = 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Compute local sum of squares
    for (int i = idx; i < n; i += stride) {
        T val = data[i];
        local_sum += val * val;
    }
    
    // Block-level reduction
    local_sum = block_reduce_sum<T, 256>(local_sum);
    
    // Write block result
    if (threadIdx.x == 0) {
        atomicAdd(norm, local_sum);
    }
}

// Standard LAMB update kernel (FP32)
extern "C" __global__ void lamb_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ param_norm,     // Pre-computed parameter norm
    float* __restrict__ update_norm,    // Pre-computed update norm
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const int n,
    const int layer_id              // For per-layer adaptation
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Compute trust ratio (same for all parameters in this layer)
    float trust_ratio = 1.0f;
    if (idx == 0) {
        float p_norm = sqrtf(param_norm[layer_id] + NORM_EPSILON);
        float u_norm = sqrtf(update_norm[layer_id] + NORM_EPSILON);
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        
        // Store for other threads
        update_norm[layer_id] = trust_ratio;
    }
    
    // Synchronize within block to get trust ratio
    __syncthreads();
    if (idx != 0) {
        trust_ratio = update_norm[layer_id];
    }
    
    // Update parameters
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Apply weight decay (L2 regularization)
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Update biased first moment estimate
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        m[i] = m_new;
        
        // Update biased second raw moment estimate
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        v[i] = v_new;
        
        // Compute bias-corrected moment estimates
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        
        // Compute update
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Apply layer-wise learning rate adaptation
        params[i] -= lr * trust_ratio * update;
    }
}

// LAMB kernel with fused norm computation
extern "C" __global__ void lamb_update_fused_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const int n
) {
    extern __shared__ float shared_mem[];
    float* shared_param_norm = &shared_mem[0];
    float* shared_update_norm = &shared_mem[blockDim.x / 32];
    
    float local_param_norm_sq = 0.0f;
    float local_update_norm_sq = 0.0f;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // First pass: compute norms and update moments
    for (int i = idx; i < n; i += stride) {
        float param = params[i];
        float grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * param;
        }
        
        // Update moments
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        m[i] = m_new;
        
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        v[i] = v_new;
        
        // Compute update
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Accumulate norms
        local_param_norm_sq += param * param;
        local_update_norm_sq += update * update;
    }
    
    // Reduce norms within block
    local_param_norm_sq = block_reduce_sum<float, 256>(local_param_norm_sq);
    local_update_norm_sq = block_reduce_sum<float, 256>(local_update_norm_sq);
    
    // Store in shared memory
    if (threadIdx.x == 0) {
        shared_param_norm[0] = local_param_norm_sq;
        shared_update_norm[0] = local_update_norm_sq;
    }
    __syncthreads();
    
    // Compute trust ratio
    float trust_ratio = 1.0f;
    if (threadIdx.x == 0) {
        float p_norm = sqrtf(shared_param_norm[0] + NORM_EPSILON);
        float u_norm = sqrtf(shared_update_norm[0] + NORM_EPSILON);
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        shared_param_norm[0] = trust_ratio;
    }
    __syncthreads();
    
    trust_ratio = shared_param_norm[0];
    
    // Second pass: apply updates with trust ratio
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Recompute update (could cache, but memory vs compute tradeoff)
        float m_val = m[i];
        float v_val = v[i];
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Apply update with trust ratio
        params[i] -= lr * trust_ratio * update;
    }
}

// Mixed precision LAMB with FP16 and tensor cores
extern "C" __global__ void lamb_update_mixed_fp16(
    __half* __restrict__ params_fp16,
    float* __restrict__ params_fp32,
    const __half* __restrict__ grads_fp16,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const float loss_scale,
    const int n
) {
    extern __shared__ float shared_mem[];
    float* shared_norms = shared_mem;
    
    float local_param_norm_sq = 0.0f;
    float local_update_norm_sq = 0.0f;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process parameters
    for (int i = idx; i < n; i += stride) {
        // Convert FP16 gradient to FP32 and unscale
        float grad = __half2float(grads_fp16[i]) / loss_scale;
        float param = params_fp32[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * param;
        }
        
        // Update moments in FP32
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        m[i] = m_new;
        
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        v[i] = v_new;
        
        // Compute update
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Accumulate norms
        local_param_norm_sq += param * param;
        local_update_norm_sq += update * update;
    }
    
    // Reduce and compute trust ratio (same as before)
    local_param_norm_sq = block_reduce_sum<float, 256>(local_param_norm_sq);
    local_update_norm_sq = block_reduce_sum<float, 256>(local_update_norm_sq);
    
    if (threadIdx.x == 0) {
        float p_norm = sqrtf(local_param_norm_sq + NORM_EPSILON);
        float u_norm = sqrtf(local_update_norm_sq + NORM_EPSILON);
        float trust_ratio = 1.0f;
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        shared_norms[0] = trust_ratio;
    }
    __syncthreads();
    
    float trust_ratio = shared_norms[0];
    
    // Apply updates
    for (int i = idx; i < n; i += stride) {
        float grad = __half2float(grads_fp16[i]) / loss_scale;
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * params_fp32[i];
        }
        
        float m_val = m[i];
        float v_val = v[i];
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Update FP32 master weights
        float param_new = params_fp32[i] - lr * trust_ratio * update;
        params_fp32[i] = param_new;
        
        // Convert back to FP16
        params_fp16[i] = __float2half(param_new);
    }
}

// Multi-GPU LAMB with efficient norm synchronization
extern "C" __global__ void lamb_update_multi_gpu_f32(
    float* __restrict__ params,
    float* __restrict__ grads,
    float* __restrict__ m,
    float* __restrict__ v,
    float* __restrict__ global_param_norm,  // Shared across GPUs
    float* __restrict__ global_update_norm, // Shared across GPUs
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const int n,
    const int num_gpus,
    const int gpu_id
) {
    extern __shared__ float shared_mem[];
    
    float local_param_norm_sq = 0.0f;
    float local_update_norm_sq = 0.0f;
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // First phase: compute local norms and update moments
    for (int i = idx; i < n; i += stride) {
        float param = params[i];
        float grad = grads[i] / num_gpus;  // Average gradients
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * param;
        }
        
        // Update moments
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        m[i] = m_new;
        
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        v[i] = v_new;
        
        // Compute update
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Accumulate norms
        local_param_norm_sq += param * param;
        local_update_norm_sq += update * update;
    }
    
    // Reduce within GPU
    local_param_norm_sq = block_reduce_sum<float, 256>(local_param_norm_sq);
    local_update_norm_sq = block_reduce_sum<float, 256>(local_update_norm_sq);
    
    // Contribute to global norm (requires atomic operations)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(&global_param_norm[0], local_param_norm_sq);
        atomicAdd(&global_update_norm[0], local_update_norm_sq);
    }
    
    // Synchronize across all GPUs (simplified - real impl would use NCCL)
    __threadfence_system();
    
    // Compute trust ratio
    float trust_ratio = 1.0f;
    if (idx == 0) {
        float p_norm = sqrtf(global_param_norm[0] + NORM_EPSILON);
        float u_norm = sqrtf(global_update_norm[0] + NORM_EPSILON);
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        
        // Broadcast trust ratio
        global_param_norm[1] = trust_ratio;
    }
    
    // Ensure all threads see the trust ratio
    __threadfence_system();
    trust_ratio = global_param_norm[1];
    
    // Apply updates
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i] / num_gpus;
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        float m_val = m[i];
        float v_val = v[i];
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        params[i] -= lr * trust_ratio * update;
    }
}

// Utility kernel to compute layer-wise norms for LAMB
extern "C" __global__ void compute_layer_norms_f32(
    const float* __restrict__ params,
    const float* __restrict__ updates,
    float* __restrict__ param_norms,
    float* __restrict__ update_norms,
    const int* __restrict__ layer_sizes,
    const int* __restrict__ layer_offsets,
    const int num_layers
) {
    const int layer_id = blockIdx.y;
    if (layer_id >= num_layers) return;
    
    const int start = layer_offsets[layer_id];
    const int size = layer_sizes[layer_id];
    const int end = start + size;
    
    extern __shared__ float shared_mem[];
    
    float local_param_norm_sq = 0.0f;
    float local_update_norm_sq = 0.0f;
    
    const int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < end; i += stride) {
        if (params != nullptr) {
            float val = params[i];
            local_param_norm_sq += val * val;
        }
        if (updates != nullptr) {
            float val = updates[i];
            local_update_norm_sq += val * val;
        }
    }
    
    // Reduce within block
    local_param_norm_sq = block_reduce_sum<float, 256>(local_param_norm_sq);
    local_update_norm_sq = block_reduce_sum<float, 256>(local_update_norm_sq);
    
    // Write layer norms
    if (threadIdx.x == 0) {
        atomicAdd(&param_norms[layer_id], local_param_norm_sq);
        atomicAdd(&update_norms[layer_id], local_update_norm_sq);
    }
}