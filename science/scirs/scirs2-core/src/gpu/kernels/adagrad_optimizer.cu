// AdaGrad optimizer CUDA kernels
// Implements Adaptive Gradient Algorithm for adaptive learning rates

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants
constexpr float EPSILON_F32 = 1e-10f;
constexpr double EPSILON_F64 = 1e-10;

// Standard AdaGrad update kernel (FP32)
extern "C" __global__ void adagrad_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ sum_sq_grad,   // Accumulated squared gradients
    const float lr,
    const float eps,
    const float weight_decay,
    const int n,
    const int step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Accumulate squared gradient
        float sum_sq = sum_sq_grad[i] + grad * grad;
        sum_sq_grad[i] = sum_sq;
        
        // Compute adaptive learning rate
        float adaptive_lr = lr / (sqrtf(sum_sq) + eps);
        
        // Update parameters
        params[i] -= adaptive_lr * grad;
    }
}

// Standard AdaGrad update kernel (FP64)
extern "C" __global__ void adagrad_update_f64(
    double* __restrict__ params,
    const double* __restrict__ grads,
    double* __restrict__ sum_sq_grad,
    const double lr,
    const double eps,
    const double weight_decay,
    const int n,
    const int step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        double grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0) {
            grad += weight_decay * params[i];
        }
        
        // Accumulate squared gradient
        double sum_sq = sum_sq_grad[i] + grad * grad;
        sum_sq_grad[i] = sum_sq;
        
        // Compute adaptive learning rate
        double adaptive_lr = lr / (sqrt(sum_sq) + eps);
        
        // Update parameters
        params[i] -= adaptive_lr * grad;
    }
}

// AdaGrad with diagonal preconditioning
extern "C" __global__ void adagrad_diagonal_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ sum_sq_grad,
    float* __restrict__ preconditioner,  // Diagonal preconditioning matrix
    const float lr,
    const float eps,
    const float weight_decay,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Apply preconditioning
        if (preconditioner != nullptr) {
            grad *= preconditioner[i];
        }
        
        // Accumulate squared gradient
        float sum_sq = sum_sq_grad[i] + grad * grad;
        sum_sq_grad[i] = sum_sq;
        
        // Compute adaptive learning rate
        float adaptive_lr = lr / (sqrtf(sum_sq) + eps);
        
        // Update parameters
        params[i] -= adaptive_lr * grad;
    }
}

// AdaDelta variant (extension of AdaGrad)
extern "C" __global__ void adadelta_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ avg_sq_grad,     // Exponential average of squared gradients
    float* __restrict__ avg_sq_update,   // Exponential average of squared updates
    const float rho,                      // Decay rate
    const float eps,
    const float weight_decay,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * params[i];
        }
        
        // Update exponential average of squared gradients
        float avg_sq_g = rho * avg_sq_grad[i] + (1.0f - rho) * grad * grad;
        avg_sq_grad[i] = avg_sq_g;
        
        // Compute update
        float rms_update = sqrtf(avg_sq_update[i] + eps);
        float rms_grad = sqrtf(avg_sq_g + eps);
        float update = (rms_update / rms_grad) * grad;
        
        // Update exponential average of squared updates
        avg_sq_update[i] = rho * avg_sq_update[i] + (1.0f - rho) * update * update;
        
        // Update parameters
        params[i] -= update;
    }
}

// AdaGrad with gradient clipping
extern "C" __global__ void adagrad_clipped_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ sum_sq_grad,
    const float lr,
    const float eps,
    const float max_grad_norm,
    const int n
) {
    extern __shared__ float shmem[];
    
    // First pass: compute gradient norm
    float local_norm_sq = 0.0f;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        local_norm_sq += grad * grad;
    }
    
    // Reduce within block
    shmem[threadIdx.x] = local_norm_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Compute clipping scale
    float clip_scale = 1.0f;
    if (threadIdx.x == 0) {
        float total_norm = sqrtf(shmem[0]);
        if (total_norm > max_grad_norm) {
            clip_scale = max_grad_norm / total_norm;
        }
        shmem[0] = clip_scale;
    }
    __syncthreads();
    clip_scale = shmem[0];
    
    // Second pass: apply AdaGrad with clipped gradients
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i] * clip_scale;
        
        // Accumulate squared gradient
        float sum_sq = sum_sq_grad[i] + grad * grad;
        sum_sq_grad[i] = sum_sq;
        
        // Update parameters
        params[i] -= lr * grad / (sqrtf(sum_sq) + eps);
    }
}

// Window-based AdaGrad (limits memory of past gradients)
extern "C" __global__ void adagrad_window_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ grad_window,      // Circular buffer of past gradients
    int* __restrict__ window_idx,         // Current position in window
    const float lr,
    const float eps,
    const int window_size,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Get current window position
    int win_idx = *window_idx % window_size;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Store gradient in window
        int window_offset = i * window_size + win_idx;
        float old_grad = grad_window[window_offset];
        grad_window[window_offset] = grad;
        
        // Compute sum of squared gradients over window
        float sum_sq = 0.0f;
        for (int w = 0; w < window_size; w++) {
            float g = grad_window[i * window_size + w];
            sum_sq += g * g;
        }
        
        // Average over window
        sum_sq /= float(window_size);
        
        // Update parameters
        params[i] -= lr * grad / (sqrtf(sum_sq) + eps);
    }
}

// Utility kernel to initialize AdaGrad state
extern "C" __global__ void init_adagrad_state_f32(
    float* __restrict__ sum_sq_grad,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        sum_sq_grad[i] = 0.0f;
    }
}