//! General matrix-vector multiplication (GEMV) kernels for GPU
//!
//! Implements y = alpha * A * x + beta * y where:
//! - A is an M x N matrix
//! - x is an N-dimensional vector
//! - y is an M-dimensional vector
//! - alpha and beta are scalar values

use std::collections::HashMap;

use crate::gpu::kernels::{
    BaseKernel, DataType, GpuKernel, KernelMetadata, KernelParams, OperationType,
};
use crate::gpu::{GpuBackend, GpuError};

/// General matrix-vector multiplication kernel
pub struct GemvKernel {
    base: BaseKernel,
}

impl Default for GemvKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl GemvKernel {
    /// Create a new GEMV kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [256, 1, 1],
            local_memory_usage: 1024, // 1 KB local memory for reduction
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void gemv(
    const float* __restrict__ matrix,  // M x N matrix (row-major)
    const float* __restrict__ vector,  // N-dimensional vector
    float* __restrict__ result,        // M-dimensional result vector
    float alpha,
    float beta,
    int M,  // Number of rows
    int N   // Number of columns
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;

        // Compute dot product of matrix row with vector
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col] * vector[col];
        }

        // Apply alpha and beta coefficients
        result[row] = alpha * sum + beta * result[row];
    }
}

// Optimized version using shared memory for larger matrices
extern "C" __global__ void gemv_shared(
    const float* __restrict__ matrix,
    const float* __restrict__ vector,
    float* __restrict__ result,
    float alpha,
    float beta,
    int M,
    int N
) {
    extern __shared__ float shared_vector[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load vector into shared memory in chunks
    for (int i = tid; i < N; i += blockDim.x) {
        if (i < N) {
            shared_vector[i] = vector[i];
        }
    }
    __syncthreads();

    if (row < M) {
        float sum = 0.0f;

        // Compute dot product using shared memory vector
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col] * shared_vector[col];
        }

        result[row] = alpha * sum + beta * result[row];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    alpha: f32,
    beta: f32,
    M: u32,  // Number of rows
    N: u32,  // Number of columns
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;   // M x N matrix
@group(0) @binding(2) var<storage, read> vector: array<f32>;   // N-dimensional vector
@group(0) @binding(3) var<storage, write> result: array<f32>;  // M-dimensional result

@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row < uniforms.M) {
        var sum = 0.0;

        // Compute dot product of matrix row with vector
        for (var col = 0u; col < uniforms.N; col = col + 1u) {
            let matrix_idx = row * uniforms.N + col;
            sum = sum + matrix[matrix_idx] * vector[col];
        }

        // Apply alpha and beta coefficients
        result[row] = uniforms.alpha * sum + uniforms.beta * result[row];
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
    const device float* matrix [[buffer(0)]],    // M x N matrix
    const device float* vector [[buffer(1)]],    // N-dimensional vector
    device float* result [[buffer(2)]],          // M-dimensional result
    constant float& alpha [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant uint& M [[buffer(5)]],              // Number of rows
    constant uint& N [[buffer(6)]],              // Number of columns
    uint gid [[thread_position_in_grid]])
{
    if (gid < M) {
        float sum = 0.0f;

        // Compute dot product of matrix row with vector
        for (uint col = 0; col < N; col++) {
            sum += matrix[gid * N + col] * vector[col];
        }

        // Apply alpha and beta coefficients
        result[gid] = alpha * sum + beta * result[gid];
    }
}

// Optimized version using threadgroup memory
kernel void gemv_tiled(
    const device float* matrix [[buffer(0)]],
    const device float* vector [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint blockSize [[threads_per_threadgroup]])
{
    threadgroup float shared_vector[256];  // Shared vector storage

    // Load vector into threadgroup memory
    for (uint i = lid; i < N; i += blockSize) {
        if (i < N) {
            shared_vector[i] = vector[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < M) {
        float sum = 0.0f;

        // Compute using shared vector
        for (uint col = 0; col < N; col++) {
            sum += matrix[gid * N + col] * shared_vector[col];
        }

        result[gid] = alpha * sum + beta * result[gid];
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void gemv(
    __global const float* matrix,   // M x N matrix
    __global const float* vector,   // N-dimensional vector
    __global float* result,         // M-dimensional result
    const float alpha,
    const float beta,
    const int M,                    // Number of rows
    const int N)                    // Number of columns
{
    int row = get_global_id(0);

    if (row < M) {
        float sum = 0.0f;

        // Compute dot product of matrix row with vector
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col] * vector[col];
        }

        // Apply alpha and beta coefficients
        result[row] = alpha * sum + beta * result[row];
    }
}

// Version with local memory optimization
__kernel void gemv_local(
    __global const float* matrix,
    __global const float* vector,
    __global float* result,
    const float alpha,
    const float beta,
    const int M,
    const int N,
    __local float* local_vector)
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    // Load vector into local memory
    for (int i = lid; i < N; i += local_size) {
        if (i < N) {
            local_vector[i] = vector[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (row < M) {
        float sum = 0.0f;

        // Compute using local vector
        for (int col = 0; col < N; col++) {
            sum += matrix[row * N + col] * local_vector[col];
        }

        result[row] = alpha * sum + beta * result[row];
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "gemv",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for GemvKernel {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        self.base.source_for_backend(backend)
    }

    fn metadata(&self) -> KernelMetadata {
        self.base.metadata()
    }

    fn can_specialize(&self, params: &KernelParams) -> bool {
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        // For now, return the same kernel (no type specialization implemented yet)
        Ok(Box::new(Self::new()))
    }
}

/// Batched GEMV kernel for processing multiple matrix-vector multiplications
pub struct BatchGemvKernel {
    base: BaseKernel,
}

impl Default for BatchGemvKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchGemvKernel {
    /// Create a new batched GEMV kernel
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [16, 16, 1],
            local_memory_usage: 2048,
            supports_tensor_cores: false,
            operationtype: OperationType::ComputeIntensive,
            backend_metadata: HashMap::new(),
        };

        let cuda_source = r#"
extern "C" __global__ void batch_gemv(
    const float* __restrict__ matrices,  // Batch of M x N matrices
    const float* __restrict__ vectors,   // Batch of N-dimensional vectors
    float* __restrict__ results,         // Batch of M-dimensional results
    float alpha,
    float beta,
    int batch_size,
    int M,  // Number of rows per matrix
    int N   // Number of columns per matrix
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && row < M) {
        // Calculate offsets for this batch
        int matrix_offset = batch_idx * M * N;
        int vector_offset = batch_idx * N;
        int result_offset = batch_idx * M;

        float sum = 0.0f;

        // Compute dot product of matrix row with vector
        for (int col = 0; col < N; col++) {
            sum += matrices[matrix_offset + row * N + col] *
                   vectors[vector_offset + col];
        }

        // Apply alpha and beta coefficients
        results[result_offset + row] = alpha * sum + beta * results[result_offset + row];
    }
}
"#
        .to_string();

        let rocm_source = cuda_source.clone();

        let wgpu_source = r#"
struct Uniforms {
    alpha: f32,
    beta: f32,
    batch_size: u32,
    M: u32,  // Number of rows per matrix
    N: u32,  // Number of columns per matrix
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> matrices: array<f32>;  // Batch of matrices
@group(0) @binding(2) var<storage, read> vectors: array<f32>;   // Batch of vectors
@group(0) @binding(3) var<storage, write> results: array<f32>;  // Batch of results

@compute @workgroup_size(16, 16, 1)
fn batch_gemv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.z;
    let row = global_id.x;

    if (batch_idx < uniforms.batch_size && row < uniforms.M) {
        // Calculate offsets for this batch
        let matrix_offset = batch_idx * uniforms.M * uniforms.N;
        let vector_offset = batch_idx * uniforms.N;
        let result_offset = batch_idx * uniforms.M;

        var sum = 0.0;

        // Compute dot product
        for (var col = 0u; col < uniforms.N; col = col + 1u) {
            let matrix_idx = matrix_offset + row * uniforms.N + col;
            let vector_idx = vector_offset + col;
            sum = sum + matrices[matrix_idx] * vectors[vector_idx];
        }

        // Apply coefficients
        let result_idx = result_offset + row;
        results[result_idx] = uniforms.alpha * sum + uniforms.beta * results[result_idx];
    }
}
"#
        .to_string();

        let metal_source = r#"
#include <metal_stdlib>
using namespace metal;

kernel void batch_gemv(
    const device float* matrices [[buffer(0)]],   // Batch of matrices
    const device float* vectors [[buffer(1)]],    // Batch of vectors
    device float* results [[buffer(2)]],          // Batch of results
    constant float& alpha [[buffer(3)]],
    constant float& beta [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& M [[buffer(6)]],               // Rows per matrix
    constant uint& N [[buffer(7)]],               // Columns per matrix
    uint3 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.z;
    uint row = gid.x;

    if (batch_idx < batch_size && row < M) {
        // Calculate offsets
        uint matrix_offset = batch_idx * M * N;
        uint vector_offset = batch_idx * N;
        uint result_offset = batch_idx * M;

        float sum = 0.0f;

        // Compute dot product
        for (uint col = 0; col < N; col++) {
            sum += matrices[matrix_offset + row * N + col] *
                   vectors[vector_offset + col];
        }

        // Apply coefficients
        results[result_offset + row] = alpha * sum + beta * results[result_offset + row];
    }
}
"#
        .to_string();

        let opencl_source = r#"
__kernel void batch_gemv(
    __global const float* matrices,
    __global const float* vectors,
    __global float* results,
    const float alpha,
    const float beta,
    const int batch_size,
    const int M,
    const int N)
{
    int batch_idx = get_global_id(2);
    int row = get_global_id(0);

    if (batch_idx < batch_size && row < M) {
        // Calculate offsets
        int matrix_offset = batch_idx * M * N;
        int vector_offset = batch_idx * N;
        int result_offset = batch_idx * M;

        float sum = 0.0f;

        // Compute dot product
        for (int col = 0; col < N; col++) {
            sum += matrices[matrix_offset + row * N + col] *
                   vectors[vector_offset + col];
        }

        // Apply coefficients
        results[result_offset + row] = alpha * sum + beta * results[result_offset + row];
    }
}
"#
        .to_string();

        Self {
            base: BaseKernel::new(
                "batch_gemv",
                &cuda_source,
                &rocm_source,
                &wgpu_source,
                &metal_source,
                &opencl_source,
                metadata,
            ),
        }
    }
}

impl GpuKernel for BatchGemvKernel {
    fn name(&self) -> &str {
        self.base.name()
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        self.base.source_for_backend(backend)
    }

    fn metadata(&self) -> KernelMetadata {
        self.base.metadata()
    }

    fn can_specialize(&self, params: &KernelParams) -> bool {
        matches!(params.datatype, DataType::Float32 | DataType::Float64)
    }

    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }

        Ok(Box::new(Self::new()))
    }
}
