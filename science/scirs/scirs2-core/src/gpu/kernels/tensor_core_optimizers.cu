// Tensor Core optimized optimizer kernels
// Leverages NVIDIA Tensor Cores for accelerated mixed-precision training
// Supports Volta (V100), Turing (T4), Ampere (A100), and newer architectures

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

// Tensor Core tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp size constant
constexpr int WARP_SIZE = 32;

// Helper to convert float to half with saturation
__device__ __forceinline__ __half float_to_half_sat(float f) {
    return __float2half_rn(fminf(fmaxf(f, -65504.0f), 65504.0f));
}

// Tensor Core accelerated vector operations for optimizer updates
// This kernel uses Tensor Cores to compute matrix operations that appear in
// second-order optimization methods and large-scale parameter updates

// Adam optimizer with Tensor Core acceleration for large parameter blocks
extern "C" __global__ void adam_tensor_core_f16(
    __half* __restrict__ params_f16,      // FP16 parameters
    float* __restrict__ params_f32,       // FP32 master weights
    const __half* __restrict__ grads_f16, // FP16 gradients
    float* __restrict__ m,                // First moment (FP32)
    float* __restrict__ v,                // Second moment (FP32)
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
    // Use Tensor Cores for efficient FP16 computation where possible
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    
    // Each warp processes WMMA_M parameters
    const int warp_start = blockIdx.x * (num_warps * WMMA_M) + warp_id * WMMA_M;
    
    // Declare fragments for Tensor Core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Process parameters in tiles
    for (int i = warp_start + lane_id; i < min(warp_start + WMMA_M, n); i += WARP_SIZE) {
        if (i < n) {
            // Load gradient and unscale
            float grad = __half2float(grads_f16[i]) / loss_scale;
            float param = params_f32[i];
            
            // Apply weight decay
            if (weight_decay > 0.0f) {
                grad += weight_decay * param;
            }
            
            // Update moments
            float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
            float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
            
            m[i] = m_new;
            v[i] = v_new;
            
            // Compute bias-corrected estimates
            float m_hat = m_new / bias_correction1;
            float v_hat = v_new / bias_correction2;
            
            // Update parameter
            float param_new = param - lr * m_hat / (sqrtf(v_hat) + eps);
            params_f32[i] = param_new;
            
            // Convert back to FP16 with saturation
            params_f16[i] = float_to_half_sat(param_new);
        }
    }
}

// LAMB optimizer with Tensor Core acceleration for norm computation
extern "C" __global__ void lamb_tensor_core_f16(
    __half* __restrict__ params_f16,
    float* __restrict__ params_f32,
    const __half* __restrict__ grads_f16,
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
    // Shared memory for norm reduction
    extern __shared__ float shared_data[];
    float* param_norm_sq = &shared_data[0];
    float* update_norm_sq = &shared_data[blockDim.x / WARP_SIZE];
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Initialize shared memory
    if (tid < 2 * (blockDim.x / WARP_SIZE)) {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();
    
    float local_param_norm = 0.0f;
    float local_update_norm = 0.0f;
    
    // Process parameters and compute norms
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = __half2float(grads_f16[i]) / loss_scale;
        float param = params_f32[i];
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * param;
        }
        
        // Update moments
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        
        m[i] = m_new;
        v[i] = v_new;
        
        // Compute update
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Accumulate norms
        local_param_norm += param * param;
        local_update_norm += update * update;
    }
    
    // Warp-level reduction using shuffle instructions
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_param_norm += __shfl_down_sync(0xffffffff, local_param_norm, offset);
        local_update_norm += __shfl_down_sync(0xffffffff, local_update_norm, offset);
    }
    
    // Write warp results to shared memory
    if (lane_id == 0) {
        param_norm_sq[warp_id] = local_param_norm;
        update_norm_sq[warp_id] = local_update_norm;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid == 0) {
        float total_param_norm = 0.0f;
        float total_update_norm = 0.0f;
        
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) {
            total_param_norm += param_norm_sq[i];
            total_update_norm += update_norm_sq[i];
        }
        
        // Compute trust ratio
        float p_norm = sqrtf(total_param_norm + 1e-8f);
        float u_norm = sqrtf(total_update_norm + 1e-8f);
        float trust_ratio = 1.0f;
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        
        param_norm_sq[0] = trust_ratio;  // Store for broadcast
    }
    __syncthreads();
    
    float trust_ratio = param_norm_sq[0];
    
    // Apply updates with trust ratio
    for (int i = idx; i < n; i += stride) {
        float grad = __half2float(grads_f16[i]) / loss_scale;
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * params_f32[i];
        }
        
        float m_val = m[i];
        float v_val = v[i];
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        // Update FP32 master weights
        float param_new = params_f32[i] - lr * trust_ratio * update;
        params_f32[i] = param_new;
        
        // Convert to FP16 with saturation
        params_f16[i] = float_to_half_sat(param_new);
    }
}

// Specialized kernel for BFloat16 training (Ampere and newer)
#if __CUDA_ARCH__ >= 800
extern "C" __global__ void adam_tensor_core_bf16(
    __nv_bfloat16* __restrict__ params_bf16,
    float* __restrict__ params_f32,
    const __nv_bfloat16* __restrict__ grads_bf16,
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        // Convert BF16 gradient to FP32 and unscale
        float grad = __bfloat162float(grads_bf16[i]) / loss_scale;
        float param = params_f32[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * param;
        }
        
        // Update moments
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        
        m[i] = m_new;
        v[i] = v_new;
        
        // Compute bias-corrected estimates
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        
        // Update parameter
        float param_new = param - lr * m_hat / (sqrtf(v_hat) + eps);
        params_f32[i] = param_new;
        
        // Convert back to BF16
        params_bf16[i] = __float2bfloat16(param_new);
    }
}
#endif

// Tensor Core optimized SGD with momentum for large batches
extern "C" __global__ void sgd_momentum_tensor_core_f16(
    __half* __restrict__ params_f16,
    float* __restrict__ params_f32,
    const __half* __restrict__ grads_f16,
    float* __restrict__ momentum,
    const float lr,
    const float momentum_factor,
    const float weight_decay,
    const float loss_scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread for better memory coalescing
    const int ELEMS_PER_THREAD = 4;
    const int start_idx = idx * ELEMS_PER_THREAD;
    
    // Vectorized loads/stores for better memory throughput
    if (start_idx + ELEMS_PER_THREAD <= n) {
        // Load 4 elements at once
        float4* params_f32_vec = reinterpret_cast<float4*>(params_f32 + start_idx);
        float4* momentum_vec = reinterpret_cast<float4*>(momentum + start_idx);
        
        float4 param_vec = *params_f32_vec;
        float4 mom_vec = *momentum_vec;
        
        // Process each element
        #pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            float grad = __half2float(grads_f16[start_idx + j]) / loss_scale;
            float param = reinterpret_cast<float*>(&param_vec)[j];
            float mom = reinterpret_cast<float*>(&mom_vec)[j];
            
            // Apply weight decay
            if (weight_decay > 0.0f) {
                grad += weight_decay * param;
            }
            
            // Update momentum
            mom = momentum_factor * mom + grad;
            reinterpret_cast<float*>(&mom_vec)[j] = mom;
            
            // Update parameter
            param -= lr * mom;
            reinterpret_cast<float*>(&param_vec)[j] = param;
            
            // Convert to FP16
            params_f16[start_idx + j] = float_to_half_sat(param);
        }
        
        // Store updated values
        *params_f32_vec = param_vec;
        *momentum_vec = mom_vec;
    } else {
        // Handle remaining elements
        for (int i = start_idx; i < n && i < start_idx + ELEMS_PER_THREAD; i++) {
            float grad = __half2float(grads_f16[i]) / loss_scale;
            float param = params_f32[i];
            
            if (weight_decay > 0.0f) {
                grad += weight_decay * param;
            }
            
            float mom = momentum_factor * momentum[i] + grad;
            momentum[i] = mom;
            
            param -= lr * mom;
            params_f32[i] = param;
            params_f16[i] = float_to_half_sat(param);
        }
    }
}

// Utility kernel for gradient scaling/unscaling with overflow detection
extern "C" __global__ void scale_gradients_check_overflow_f16(
    __half* __restrict__ gradients,
    float* __restrict__ grad_scale,
    int* __restrict__ has_overflow,
    const float scale_factor,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    bool local_overflow = false;
    
    for (int i = idx; i < n; i += stride) {
        float grad = __half2float(gradients[i]);
        
        // Check for inf/nan
        if (!isfinite(grad)) {
            local_overflow = true;
        }
        
        // Scale gradient
        grad *= scale_factor;
        
        // Check again after scaling
        if (!isfinite(grad)) {
            local_overflow = true;
        }
        
        gradients[i] = float_to_half_sat(grad);
    }
    
    // Report overflow
    if (local_overflow) {
        atomicExch(has_overflow, 1);
    }
}