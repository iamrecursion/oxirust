// SGD optimizer CUDA kernels
// Implements stochastic gradient descent with momentum and Nesterov acceleration

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Constants
constexpr float EPSILON = 1e-8f;

// Basic SGD update kernel (FP32)
extern "C" __global__ void sgd_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    const float lr,
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
        
        // Update parameters
        params[i] -= lr * grad;
    }
}

// Basic SGD update kernel (FP64)
extern "C" __global__ void sgd_update_f64(
    double* __restrict__ params,
    const double* __restrict__ grads,
    const double lr,
    const double weight_decay,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        double grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0) {
            grad += weight_decay * params[i];
        }
        
        // Update parameters
        params[i] -= lr * grad;
    }
}

// SGD with momentum update kernel (FP32)
extern "C" __global__ void sgd_momentum_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ momentum,
    const float lr,
    const float momentum_factor,
    const float weight_decay,
    const int nesterov,
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
        
        // Update momentum
        float m = momentum_factor * momentum[i] + grad;
        momentum[i] = m;
        
        // Apply update
        if (nesterov) {
            // Nesterov accelerated gradient
            params[i] -= lr * (momentum_factor * m + grad);
        } else {
            // Standard momentum
            params[i] -= lr * m;
        }
    }
}

// SGD with momentum update kernel (FP64)
extern "C" __global__ void sgd_momentum_update_f64(
    double* __restrict__ params,
    const double* __restrict__ grads,
    double* __restrict__ momentum,
    const double lr,
    const double momentum_factor,
    const double weight_decay,
    const int nesterov,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        double grad = grads[i];
        
        // Apply weight decay
        if (weight_decay > 0.0) {
            grad += weight_decay * params[i];
        }
        
        // Update momentum
        double m = momentum_factor * momentum[i] + grad;
        momentum[i] = m;
        
        // Apply update
        if (nesterov) {
            // Nesterov accelerated gradient
            params[i] -= lr * (momentum_factor * m + grad);
        } else {
            // Standard momentum
            params[i] -= lr * m;
        }
    }
}

// Heavy-ball SGD with separate momentum for parameters
extern "C" __global__ void sgd_heavy_ball_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ velocity,
    float* __restrict__ params_prev,
    const float lr,
    const float momentum,
    const float weight_decay,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        float param_curr = params[i];
        float param_old = params_prev[i];
        
        // Apply weight decay
        if (weight_decay > 0.0f) {
            grad += weight_decay * param_curr;
        }
        
        // Heavy-ball update
        float param_new = param_curr - lr * grad + momentum * (param_curr - param_old);
        
        // Store for next iteration
        params_prev[i] = param_curr;
        params[i] = param_new;
    }
}

// Accelerated SGD kernel with adaptive momentum
extern "C" __global__ void sgd_accelerated_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ velocity,
    float* __restrict__ accumulator,
    const float lr,
    const float beta,
    const float restart_threshold,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        float v = velocity[i];
        float acc = accumulator[i];
        
        // Check for restart condition (gradient and momentum have opposite signs)
        float momentum_grad_product = v * grad;
        if (momentum_grad_product < 0 && fabsf(momentum_grad_product) > restart_threshold) {
            // Restart: reset momentum
            v = 0.0f;
            acc = 0.0f;
        }
        
        // Update velocity with acceleration
        v = beta * v - lr * grad;
        velocity[i] = v;
        
        // Update accumulator for adaptive restart
        acc = beta * acc + (1.0f - beta) * grad * grad;
        accumulator[i] = acc;
        
        // Update parameters
        params[i] += v;
    }
}

// Utility kernel to initialize momentum buffer
extern "C" __global__ void init_sgd_momentum_f32(
    float* __restrict__ momentum,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        momentum[i] = 0.0f;
    }
}