// RMSprop optimizer CUDA kernels
// Implements Root Mean Square Propagation with centered variant

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// Constants
constexpr float EPSILON_F32 = 1e-8f;
constexpr double EPSILON_F64 = 1e-8;

// Standard RMSprop update kernel (FP32)
extern "C" __global__ void rmsprop_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ v,        // Mean squared gradient
    float* __restrict__ m,        // Momentum buffer (optional)
    const float lr,
    const float alpha,            // Decay rate for moving average
    const float eps,
    const float weight_decay,
    const float momentum,
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
        
        // Update mean squared gradient with exponential moving average
        float v_new = alpha * v[i] + (1.0f - alpha) * grad * grad;
        v[i] = v_new;
        
        // Compute update
        float update = grad / (sqrtf(v_new) + eps);
        
        // Apply momentum if enabled
        if (momentum > 0.0f && m != nullptr) {
            float m_new = momentum * m[i] + update;
            m[i] = m_new;
            params[i] -= lr * m_new;
        } else {
            params[i] -= lr * update;
        }
    }
}

// Standard RMSprop update kernel (FP64)
extern "C" __global__ void rmsprop_update_f64(
    double* __restrict__ params,
    const double* __restrict__ grads,
    double* __restrict__ v,
    double* __restrict__ m,
    const double lr,
    const double alpha,
    const double eps,
    const double weight_decay,
    const double momentum,
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
        
        // Update mean squared gradient
        double v_new = alpha * v[i] + (1.0 - alpha) * grad * grad;
        v[i] = v_new;
        
        // Compute update
        double update = grad / (sqrt(v_new) + eps);
        
        // Apply momentum if enabled
        if (momentum > 0.0 && m != nullptr) {
            double m_new = momentum * m[i] + update;
            m[i] = m_new;
            params[i] -= lr * m_new;
        } else {
            params[i] -= lr * update;
        }
    }
}

// Centered RMSprop update kernel (FP32)
extern "C" __global__ void rmsprop_centered_update_f32(
    float* __restrict__ params,
    const float* __restrict__ grads,
    float* __restrict__ v,        // Mean squared gradient
    float* __restrict__ g,        // Mean gradient
    float* __restrict__ m,        // Momentum buffer
    const float lr,
    const float alpha,
    const float eps,
    const float weight_decay,
    const float momentum,
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
        
        // Update mean gradient (for centered variant)
        float g_new = alpha * g[i] + (1.0f - alpha) * grad;
        g[i] = g_new;
        
        // Update mean squared gradient
        float v_new = alpha * v[i] + (1.0f - alpha) * grad * grad;
        v[i] = v_new;
        
        // Compute centered second moment
        float centered_v = v_new - g_new * g_new;
        
        // Compute update with centered variance
        float update = grad / (sqrtf(centered_v + eps));
        
        // Apply momentum if enabled
        if (momentum > 0.0f && m != nullptr) {
            float m_new = momentum * m[i] + update;
            m[i] = m_new;
            params[i] -= lr * m_new;
        } else {
            params[i] -= lr * update;
        }
    }
}

// Centered RMSprop update kernel (FP64)
extern "C" __global__ void rmsprop_centered_update_f64(
    double* __restrict__ params,
    const double* __restrict__ grads,
    double* __restrict__ v,
    double* __restrict__ g,
    double* __restrict__ m,
    const double lr,
    const double alpha,
    const double eps,
    const double weight_decay,
    const double momentum,
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
        
        // Update mean gradient
        double g_new = alpha * g[i] + (1.0 - alpha) * grad;
        g[i] = g_new;
        
        // Update mean squared gradient
        double v_new = alpha * v[i] + (1.0 - alpha) * grad * grad;
        v[i] = v_new;
        
        // Compute centered second moment
        double centered_v = v_new - g_new * g_new;
        
        // Compute update
        double update = grad / (sqrt(centered_v + eps));
        
        // Apply momentum if enabled
        if (momentum > 0.0 && m != nullptr) {
            double m_new = momentum * m[i] + update;
            m[i] = m_new;
            params[i] -= lr * m_new;
        } else {
            params[i] -= lr * update;
        }
    }
}

// RMSprop with Nesterov momentum
extern "C" __global__ void rmsprop_nesterov_f32(
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
    
    for (int i = idx; i < n; i += stride) {
        float grad = grads[i];
        
        // Update mean squared gradient
        float v_new = alpha * v[i] + (1.0f - alpha) * grad * grad;
        v[i] = v_new;
        
        // Compute adaptive learning rate
        float adaptive_lr = lr / (sqrtf(v_new) + eps);
        
        // Update momentum with Nesterov lookahead
        float m_prev = m[i];
        float m_new = momentum * m_prev - adaptive_lr * grad;
        m[i] = m_new;
        
        // Apply Nesterov update
        params[i] += -momentum * m_prev + (1.0f + momentum) * m_new;
    }
}

// Utility kernel to initialize RMSprop state
extern "C" __global__ void init_rmsprop_state_f32(
    float* __restrict__ v,
    float* __restrict__ g,
    float* __restrict__ m,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        v[i] = 0.0f;
        if (g != nullptr) g[i] = 0.0f;
        if (m != nullptr) m[i] = 0.0f;
    }
}