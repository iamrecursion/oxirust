// Multi-GPU synchronization primitives for distributed training
// Implements efficient parameter and gradient synchronization across multiple GPUs
// using NCCL-style collective communication patterns

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// Constants for synchronization
constexpr int MAX_GPUS = 16;
constexpr int SYNC_WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

// Ring all-reduce implementation for gradient averaging
extern "C" __global__ void ring_allreduce_f32(
    float* __restrict__ data,
    float* __restrict__ recv_buffer,
    const int chunk_size,
    const int rank,
    const int world_size,
    const int chunk_id
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate chunk boundaries
    const int start = chunk_id * chunk_size;
    const int end = min((chunk_id + 1) * chunk_size, gridDim.x * blockDim.x);
    
    if (tid + start < end) {
        const int idx = tid + start;
        
        // Ring reduce phase
        for (int step = 0; step < world_size - 1; step++) {
            int send_chunk = (rank - step + world_size) % world_size;
            int recv_chunk = (rank - step - 1 + world_size) % world_size;
            
            // In real implementation, would use peer-to-peer transfers
            // For now, simulate with local operations
            if (chunk_id == send_chunk) {
                // This chunk is being sent
                recv_buffer[idx] = data[idx];
            }
            
            __syncthreads();
            
            if (chunk_id == recv_chunk) {
                // This chunk is being received and accumulated
                data[idx] += recv_buffer[idx];
            }
            
            __syncthreads();
        }
        
        // Ring broadcast phase
        for (int step = 0; step < world_size - 1; step++) {
            int send_chunk = (rank + 1 - step + world_size) % world_size;
            int recv_chunk = (rank - step + world_size) % world_size;
            
            if (chunk_id == send_chunk) {
                recv_buffer[idx] = data[idx];
            }
            
            __syncthreads();
            
            if (chunk_id == recv_chunk) {
                data[idx] = recv_buffer[idx];
            }
            
            __syncthreads();
        }
        
        // Average the gradients
        data[idx] /= float(world_size);
    }
}

// Tree all-reduce for small tensors
extern "C" __global__ void tree_allreduce_f32(
    float* __restrict__ data,
    float* __restrict__ workspace,
    const int size,
    const int rank,
    const int world_size
) {
    extern __shared__ float shared_data[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    
    // Load data to shared memory
    if (idx < size) {
        shared_data[tid] = data[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    __syncthreads();
    
    // Tree reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && idx + stride < size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        workspace[bid] = shared_data[0];
    }
    
    // Global synchronization point (requires cooperative launch)
    cg::grid_group grid = cg::this_grid();
    grid.sync();
    
    // Final reduction across blocks
    if (bid == 0 && tid < gridDim.x) {
        float sum = 0.0f;
        for (int i = tid; i < gridDim.x; i += blockDim.x) {
            sum += workspace[i];
        }
        
        // Warp-level reduction
        for (int offset = SYNC_WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (tid == 0) {
            workspace[0] = sum / float(world_size);
        }
    }
    
    grid.sync();
    
    // Broadcast result
    const float avg = workspace[0];
    if (idx < size) {
        data[idx] = avg;
    }
}

// Asynchronous gradient compression for bandwidth optimization
extern "C" __global__ void compress_gradients_topk_f32(
    const float* __restrict__ gradients,
    float* __restrict__ compressed_values,
    int* __restrict__ compressed_indices,
    float* __restrict__ error_feedback,
    const int size,
    const int k,  // Top-k elements to keep
    const float momentum  // Error feedback momentum
) {
    extern __shared__ char shared_mem[];
    float* values = reinterpret_cast<float*>(shared_mem);
    int* indices = reinterpret_cast<int*>(&values[blockDim.x]);
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    // Load gradient with error feedback
    float grad_val = 0.0f;
    if (idx < size) {
        grad_val = gradients[idx] + momentum * error_feedback[idx];
    }
    
    values[tid] = fabsf(grad_val);
    indices[tid] = idx;
    __syncthreads();
    
    // Bitonic sort to find top-k within block
    for (int k_step = 2; k_step <= blockDim.x; k_step *= 2) {
        for (int j = k_step / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k_step) == 0) {
                    if (values[tid] < values[ixj]) {
                        // Swap
                        float temp_val = values[tid];
                        values[tid] = values[ixj];
                        values[ixj] = temp_val;
                        
                        int temp_idx = indices[tid];
                        indices[tid] = indices[ixj];
                        indices[ixj] = temp_idx;
                    }
                } else {
                    if (values[tid] > values[ixj]) {
                        // Swap
                        float temp_val = values[tid];
                        values[tid] = values[ixj];
                        values[ixj] = temp_val;
                        
                        int temp_idx = indices[tid];
                        indices[tid] = indices[ixj];
                        indices[ixj] = temp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Select top-k per block
    if (tid < k && idx < size) {
        int selected_idx = indices[tid];
        if (selected_idx < size) {
            float selected_grad = gradients[selected_idx] + momentum * error_feedback[selected_idx];
            compressed_values[blockIdx.x * k + tid] = selected_grad;
            compressed_indices[blockIdx.x * k + tid] = selected_idx;
            
            // Update error feedback
            error_feedback[selected_idx] = 0.0f;
        }
    }
    
    // Store error for non-selected gradients
    if (idx < size && tid >= k) {
        error_feedback[idx] = gradients[idx] + momentum * error_feedback[idx];
    }
}

// Decompress sparse gradients
extern "C" __global__ void decompress_gradients_f32(
    float* __restrict__ dense_gradients,
    const float* __restrict__ compressed_values,
    const int* __restrict__ compressed_indices,
    const int num_compressed,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize to zero
    if (idx < size) {
        dense_gradients[idx] = 0.0f;
    }
    
    // Scatter compressed values
    if (idx < num_compressed) {
        int target_idx = compressed_indices[idx];
        if (target_idx >= 0 && target_idx < size) {
            atomicAdd(&dense_gradients[target_idx], compressed_values[idx]);
        }
    }
}

// Hierarchical all-reduce for multi-node setups
extern "C" __global__ void hierarchical_allreduce_f32(
    float* __restrict__ data,
    float* __restrict__ local_workspace,
    float* __restrict__ global_workspace,
    const int size,
    const int local_rank,
    const int local_size,
    const int global_rank,
    const int global_size
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    
    if (idx >= size) return;
    
    // Phase 1: Local all-reduce within node
    // Reduce to local rank 0
    if (local_rank == 0) {
        for (int i = 1; i < local_size; i++) {
            // In real impl, would receive from local rank i
            data[idx] += local_workspace[i * size + idx];
        }
    }
    
    __syncthreads();
    
    // Phase 2: Global all-reduce across nodes (only local rank 0 participates)
    if (local_rank == 0) {
        // Copy to global workspace
        global_workspace[global_rank * size + idx] = data[idx];
        
        __syncthreads();
        
        // All-reduce across nodes
        if (global_rank == 0) {
            for (int i = 1; i < global_size; i++) {
                data[idx] += global_workspace[i * size + idx];
            }
            data[idx] /= float(global_size * local_size);
        }
        
        __syncthreads();
        
        // Broadcast to all nodes
        if (global_rank != 0) {
            data[idx] = global_workspace[idx];
        }
    }
    
    __syncthreads();
    
    // Phase 3: Broadcast within node
    if (local_rank != 0) {
        data[idx] = local_workspace[idx];
    }
}

// Pipeline parallel gradient synchronization
extern "C" __global__ void pipeline_grad_sync_f32(
    float* __restrict__ forward_grads,
    float* __restrict__ backward_grads,
    const float* __restrict__ activations,
    const int* __restrict__ layer_sizes,
    const int* __restrict__ layer_offsets,
    const int pipeline_stage,
    const int num_stages,
    const int micro_batch
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int layer_start = layer_offsets[pipeline_stage];
    const int layer_size = layer_sizes[pipeline_stage];
    
    if (tid < layer_size) {
        const int idx = layer_start + tid;
        
        // Forward gradients from next stage
        if (pipeline_stage < num_stages - 1) {
            float grad = forward_grads[idx];
            backward_grads[idx] += grad;
        }
        
        // Backward gradients to previous stage
        if (pipeline_stage > 0) {
            // Compute gradient w.r.t activations
            float act_grad = backward_grads[idx] * activations[idx];
            atomicAdd(&forward_grads[idx - layer_sizes[pipeline_stage - 1]], act_grad);
        }
    }
}

// Gradient accumulation with overflow detection for mixed precision
extern "C" __global__ void accumulate_gradients_mixed_f16(
    __half* __restrict__ grad_accum,
    const __half* __restrict__ micro_grads,
    float* __restrict__ grad_scale,
    int* __restrict__ has_overflow,
    const int size,
    const int accumulation_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float accum = __half2float(grad_accum[idx]);
        float micro = __half2float(micro_grads[idx]);
        
        // Check for overflow
        if (!isfinite(micro) || !isfinite(accum)) {
            atomicExch(has_overflow, 1);
            return;
        }
        
        // Accumulate
        accum += micro;
        
        // Check result
        if (!isfinite(accum)) {
            atomicExch(has_overflow, 1);
            return;
        }
        
        // Store back
        grad_accum[idx] = __float2half(accum);
        
        // If this is the last accumulation step, scale the gradients
        if (accumulation_steps == 0) {
            float scaled = accum / (*grad_scale);
            grad_accum[idx] = __float2half(scaled);
        }
    }
}

// Zero-redundancy optimizer state partitioning (ZeRO-style)
extern "C" __global__ void partition_optimizer_states_f32(
    const float* __restrict__ full_params,
    float* __restrict__ local_params,
    float* __restrict__ local_moments1,
    float* __restrict__ local_moments2,
    const int total_size,
    const int rank,
    const int world_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate partition boundaries
    const int partition_size = (total_size + world_size - 1) / world_size;
    const int start = rank * partition_size;
    const int end = min((rank + 1) * partition_size, total_size);
    const int local_idx = tid;
    const int global_idx = start + tid;
    
    if (global_idx < end) {
        // Copy assigned parameters
        local_params[local_idx] = full_params[global_idx];
        
        // Initialize optimizer states
        local_moments1[local_idx] = 0.0f;
        local_moments2[local_idx] = 0.0f;
    }
}

// Gather partitioned parameters after optimization
extern "C" __global__ void gather_partitioned_params_f32(
    float* __restrict__ full_params,
    const float* __restrict__ local_params,
    const int total_size,
    const int rank,
    const int world_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int partition_size = (total_size + world_size - 1) / world_size;
    const int start = rank * partition_size;
    const int end = min((rank + 1) * partition_size, total_size);
    const int local_idx = tid;
    const int global_idx = start + tid;
    
    if (global_idx < end) {
        // In real implementation, would use all-gather collective
        full_params[global_idx] = local_params[local_idx];
    }
}