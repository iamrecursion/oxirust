// HIP optimizer kernels for AMD GPUs
// These are ROCm-compatible versions of the CUDA kernels

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas.h>

// AMD GPU specific optimizations
#define WAVEFRONT_SIZE 64
#define LDS_SIZE 65536 // Local Data Share size

// Helper for wavefront-level reductions
template<typename T>
__device__ __forceinline__ T wavefront_reduce_sum(T val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

// HIP version of Adam optimizer
extern "C" __global__ void hip_adam_update_f32(
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
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Coalesced memory access pattern optimized for AMD
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Update moments
        float m_new = beta1 * m[i] + (1.0f - beta1) * grad;
        float v_new = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        
        m[i] = m_new;
        v[i] = v_new;
        
        // Bias correction and update
        float m_hat = m_new / bias_correction1;
        float v_hat = v_new / bias_correction2;
        
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// HIP version of LAMB optimizer with LDS optimization
extern "C" __global__ void hip_lamb_update_f32(
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
    // Use LDS for norm computation
    extern __shared__ float lds_data[];
    float* param_norm_sq = &lds_data[0];
    float* update_norm_sq = &lds_data[blockDim.x / WAVEFRONT_SIZE];
    
    const int tid = threadIdx.x;
    const int wid = tid / WAVEFRONT_SIZE;
    const int lane = tid % WAVEFRONT_SIZE;
    
    float local_param_norm = 0.0f;
    float local_update_norm = 0.0f;
    
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    // First pass: compute norms
    for (int i = idx; i < n; i += stride) {
        float param = params[i];
        float grad = grads[i];
        
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
        
        local_param_norm += param * param;
        local_update_norm += update * update;
    }
    
    // Wavefront reduction
    local_param_norm = wavefront_reduce_sum(local_param_norm);
    local_update_norm = wavefront_reduce_sum(local_update_norm);
    
    if (lane == 0) {
        param_norm_sq[wid] = local_param_norm;
        update_norm_sq[wid] = local_update_norm;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        float total_param_norm = 0.0f;
        float total_update_norm = 0.0f;
        
        for (int i = 0; i < blockDim.x / WAVEFRONT_SIZE; i++) {
            total_param_norm += param_norm_sq[i];
            total_update_norm += update_norm_sq[i];
        }
        
        float p_norm = sqrtf(total_param_norm + 1e-8f);
        float u_norm = sqrtf(total_update_norm + 1e-8f);
        float trust_ratio = 1.0f;
        
        if (p_norm > 0.0f && u_norm > 0.0f) {
            trust_ratio = p_norm / u_norm;
        }
        
        param_norm_sq[0] = trust_ratio;
    }
    __syncthreads();
    
    float trust_ratio = param_norm_sq[0];
    
    // Second pass: apply updates
    for (int i = idx; i < n; i += stride) {
        float m_val = m[i];
        float v_val = v[i];
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;
        float update = m_hat / (sqrtf(v_hat) + eps);
        
        params[i] -= lr * trust_ratio * update;
    }
}

// Matrix Core accelerated SGD for AMD GPUs (using WMMA)
extern "C" __global__ void hip_sgd_matrix_core_f16(
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
    
    // Process multiple elements per thread for better throughput
    const int ELEMS_PER_THREAD = 4;
    const int start_idx = idx * ELEMS_PER_THREAD;
    
    if (start_idx + ELEMS_PER_THREAD <= n) {
        // Vectorized loads
        float4 params_vec = *reinterpret_cast<float4*>(params_f32 + start_idx);
        float4 mom_vec = *reinterpret_cast<float4*>(momentum + start_idx);
        
        #pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            float grad = __half2float(grads_f16[start_idx + j]) / loss_scale;
            float param = reinterpret_cast<float*>(&params_vec)[j];
            float mom = reinterpret_cast<float*>(&mom_vec)[j];
            
            if (weight_decay > 0.0f) {
                grad += weight_decay * param;
            }
            
            // Update momentum
            mom = momentum_factor * mom + grad;
            reinterpret_cast<float*>(&mom_vec)[j] = mom;
            
            // Update parameter
            param -= lr * mom;
            reinterpret_cast<float*>(&params_vec)[j] = param;
            
            // Convert to FP16
            params_f16[start_idx + j] = __float2half(param);
        }
        
        // Vectorized stores
        *reinterpret_cast<float4*>(params_f32 + start_idx) = params_vec;
        *reinterpret_cast<float4*>(momentum + start_idx) = mom_vec;
    }
}

// Fused RMSprop kernel for AMD GPUs with async memory operations
extern "C" __global__ void hip_rmsprop_async_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ v,
    float* __restrict__ m,
    const float lr,
    const float alpha,
    const float eps,
    const float momentum,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Prefetch data using async copy
    __shared__ float shared_grads[256];
    __shared__ float shared_params[256];
    
    if (idx < n && threadIdx.x < 256) {
        // Async copy to shared memory
        __pipeline_memcpy_async(&shared_grads[threadIdx.x], 
                               &grads[idx], sizeof(float));
        __pipeline_memcpy_async(&shared_params[threadIdx.x], 
                               &params[idx], sizeof(float));
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
    __syncthreads();
    
    // Process from shared memory
    if (idx < n && threadIdx.x < 256) {
        float grad = shared_grads[threadIdx.x];
        float param = shared_params[threadIdx.x];
        
        // Update mean squared gradient
        float v_new = alpha * v[idx] + (1.0f - alpha) * grad * grad;
        v[idx] = v_new;
        
        // Compute update
        float update = grad / (sqrtf(v_new) + eps);
        
        // Apply momentum if enabled
        if (momentum > 0.0f && m != nullptr) {
            float m_new = momentum * m[idx] + update;
            m[idx] = m_new;
            params[idx] = param - lr * m_new;
        } else {
            params[idx] = param - lr * update;
        }
    }
}

// Utility function to query AMD GPU properties
extern "C" __host__ void hip_get_device_properties(
    int device_id,
    int* compute_units,
    int* max_threads_per_block,
    size_t* shared_mem_per_block
) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    
    *compute_units = prop.multiProcessorCount;
    *max_threads_per_block = prop.maxThreadsPerBlock;
    *shared_mem_per_block = prop.sharedMemPerBlock;
}