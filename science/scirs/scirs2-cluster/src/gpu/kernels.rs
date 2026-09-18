//! GPU Kernel Implementations for Clustering Algorithms
//!
//! This module provides GPU kernel source code and execution wrappers for
//! clustering operations including distance computation, K-means, and more.

use crate::error::{ClusteringError, Result};
use serde::{Deserialize, Serialize};

use super::core::GpuBackend;

// ============================================================================
// Distance Kernel Types
// ============================================================================

/// Distance kernel type for GPU acceleration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceKernelType {
    /// Euclidean distance (L2)
    Euclidean,
    /// Squared Euclidean distance (faster, no sqrt)
    SquaredEuclidean,
    /// Manhattan distance (L1)
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Chebyshev distance (L-infinity)
    Chebyshev,
    /// Minkowski distance with parameter p
    Minkowski,
    /// Hamming distance (for binary data)
    Hamming,
}

impl Default for DistanceKernelType {
    fn default() -> Self {
        DistanceKernelType::SquaredEuclidean
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/// Generate CUDA kernel for squared Euclidean distance matrix computation
pub fn generate_cuda_distance_matrix_kernel() -> String {
    r#"
extern "C" __global__ void squared_euclidean_distance_matrix(
    const float* __restrict__ data,
    float* __restrict__ distances,
    const int n_samples,
    const int n_features,
    const int tile_size
) {
    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* tile_a = shared_mem;
    float* tile_b = shared_mem + tile_size * n_features;

    const int row = blockIdx.y * tile_size + threadIdx.y;
    const int col = blockIdx.x * tile_size + threadIdx.x;

    if (row >= n_samples || col >= n_samples) return;

    // Load tiles into shared memory
    for (int k = 0; k < n_features; k += blockDim.x) {
        int feat_idx = k + threadIdx.x;
        if (feat_idx < n_features) {
            if (threadIdx.y < tile_size && row < n_samples) {
                tile_a[threadIdx.y * n_features + feat_idx] = data[row * n_features + feat_idx];
            }
            if (threadIdx.x < tile_size && col < n_samples) {
                tile_b[threadIdx.x * n_features + feat_idx] = data[col * n_features + feat_idx];
            }
        }
    }

    __syncthreads();

    // Compute squared Euclidean distance
    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = tile_a[threadIdx.y * n_features + k] - tile_b[threadIdx.x * n_features + k];
        sum += diff * diff;
    }

    distances[row * n_samples + col] = sum;
}

extern "C" __global__ void euclidean_distance_matrix(
    const float* __restrict__ data,
    float* __restrict__ distances,
    const int n_samples,
    const int n_features
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) {
        // Only compute upper triangle, copy to lower
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    float dist = sqrtf(sum);
    distances[row * n_samples + col] = dist;
    distances[col * n_samples + row] = dist;  // Symmetric
}

extern "C" __global__ void cosine_distance_matrix(
    const float* __restrict__ data,
    float* __restrict__ distances,
    const int n_samples,
    const int n_features
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int k = 0; k < n_features; k++) {
        float a = data[row * n_features + k];
        float b = data[col * n_features + k];
        dot_product += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    float cosine_sim = (norm_a > 0 && norm_b > 0) ? (dot_product / (norm_a * norm_b)) : 0.0f;
    float dist = 1.0f - cosine_sim;

    distances[row * n_samples + col] = dist;
    distances[col * n_samples + row] = dist;
}

extern "C" __global__ void manhattan_distance_matrix(
    const float* __restrict__ data,
    float* __restrict__ distances,
    const int n_samples,
    const int n_features
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += fabsf(diff);
    }

    distances[row * n_samples + col] = sum;
    distances[col * n_samples + row] = sum;
}
"#
    .to_string()
}

/// Generate CUDA kernel for K-means label assignment
pub fn generate_cuda_kmeans_assign_kernel() -> String {
    r#"
extern "C" __global__ void kmeans_assign_labels(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    int* __restrict__ labels,
    float* __restrict__ distances,
    const int n_samples,
    const int n_centroids,
    const int n_features
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= n_samples) return;

    float min_dist = 1e38f;  // Large number
    int min_label = 0;

    for (int c = 0; c < n_centroids; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = data[sample_idx * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_label = c;
        }
    }

    labels[sample_idx] = min_label;
    distances[sample_idx] = min_dist;
}

extern "C" __global__ void kmeans_compute_centroids(
    const float* __restrict__ data,
    const int* __restrict__ labels,
    float* __restrict__ new_centroids,
    int* __restrict__ counts,
    const int n_samples,
    const int n_centroids,
    const int n_features
) {
    const int centroid_idx = blockIdx.x;
    const int feature_idx = threadIdx.x;

    if (centroid_idx >= n_centroids || feature_idx >= n_features) return;

    // Initialize
    if (feature_idx == 0) {
        counts[centroid_idx] = 0;
    }
    new_centroids[centroid_idx * n_features + feature_idx] = 0.0f;

    __syncthreads();

    // Sum points in this cluster
    for (int i = 0; i < n_samples; i++) {
        if (labels[i] == centroid_idx) {
            atomicAdd(&new_centroids[centroid_idx * n_features + feature_idx],
                     data[i * n_features + feature_idx]);
            if (feature_idx == 0) {
                atomicAdd(&counts[centroid_idx], 1);
            }
        }
    }

    __syncthreads();

    // Normalize
    int count = counts[centroid_idx];
    if (count > 0) {
        new_centroids[centroid_idx * n_features + feature_idx] /= (float)count;
    }
}
"#
    .to_string()
}

/// Generate CUDA kernel for batch distance computation (points to centroids)
pub fn generate_cuda_batch_distance_kernel() -> String {
    r#"
extern "C" __global__ void batch_squared_euclidean(
    const float* __restrict__ points,
    const float* __restrict__ centroids,
    float* __restrict__ distances,
    const int n_points,
    const int n_centroids,
    const int n_features
) {
    const int point_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= n_points || centroid_idx >= n_centroids) return;

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = points[point_idx * n_features + k] - centroids[centroid_idx * n_features + k];
        sum += diff * diff;
    }

    distances[point_idx * n_centroids + centroid_idx] = sum;
}

// Tensor core accelerated version (for supported GPUs)
extern "C" __global__ void batch_squared_euclidean_tc(
    const half* __restrict__ points,
    const half* __restrict__ centroids,
    float* __restrict__ distances,
    const int n_points,
    const int n_centroids,
    const int n_features
) {
    // Simplified version - actual implementation would use WMMA intrinsics
    const int point_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= n_points || centroid_idx >= n_centroids) return;

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float p = __half2float(points[point_idx * n_features + k]);
        float c = __half2float(centroids[centroid_idx * n_features + k]);
        float diff = p - c;
        sum += diff * diff;
    }

    distances[point_idx * n_centroids + centroid_idx] = sum;
}
"#
    .to_string()
}

// ============================================================================
// OpenCL Kernels
// ============================================================================

/// Generate OpenCL kernel for squared Euclidean distance matrix
pub fn generate_opencl_distance_matrix_kernel() -> String {
    r#"
__kernel void squared_euclidean_distance_matrix(
    __global const float* data,
    __global float* distances,
    const int n_samples,
    const int n_features
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;  // Compute upper triangle only

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    distances[row * n_samples + col] = sum;
    distances[col * n_samples + row] = sum;  // Symmetric
}

__kernel void euclidean_distance_matrix(
    __global const float* data,
    __global float* distances,
    const int n_samples,
    const int n_features
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    float dist = sqrt(sum);
    distances[row * n_samples + col] = dist;
    distances[col * n_samples + row] = dist;
}

__kernel void kmeans_assign_labels(
    __global const float* data,
    __global const float* centroids,
    __global int* labels,
    __global float* distances,
    const int n_samples,
    const int n_centroids,
    const int n_features
) {
    const int sample_idx = get_global_id(0);

    if (sample_idx >= n_samples) return;

    float min_dist = 1e38f;
    int min_label = 0;

    for (int c = 0; c < n_centroids; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = data[sample_idx * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_label = c;
        }
    }

    labels[sample_idx] = min_label;
    distances[sample_idx] = min_dist;
}
"#
    .to_string()
}

// ============================================================================
// Metal Shaders
// ============================================================================

/// Generate Metal shader for distance computation
pub fn generate_metal_distance_kernel() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void squared_euclidean_distance_matrix(
    device const float* data [[buffer(0)]],
    device float* distances [[buffer(1)]],
    constant uint& n_samples [[buffer(2)]],
    constant uint& n_features [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float sum = 0.0f;
    for (uint k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    distances[row * n_samples + col] = sum;
    distances[col * n_samples + row] = sum;
}

kernel void euclidean_distance_matrix(
    device const float* data [[buffer(0)]],
    device float* distances [[buffer(1)]],
    constant uint& n_samples [[buffer(2)]],
    constant uint& n_features [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float sum = 0.0f;
    for (uint k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    float dist = sqrt(sum);
    distances[row * n_samples + col] = dist;
    distances[col * n_samples + row] = dist;
}

kernel void kmeans_assign_labels(
    device const float* data [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* labels [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& n_samples [[buffer(4)]],
    constant uint& n_centroids [[buffer(5)]],
    constant uint& n_features [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint sample_idx = gid;

    if (sample_idx >= n_samples) return;

    float min_dist = 1e38f;
    int min_label = 0;

    for (uint c = 0; c < n_centroids; c++) {
        float dist = 0.0f;
        for (uint f = 0; f < n_features; f++) {
            float diff = data[sample_idx * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_label = (int)c;
        }
    }

    labels[sample_idx] = min_label;
    distances[sample_idx] = min_dist;
}

kernel void batch_squared_euclidean(
    device const float* points [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& n_points [[buffer(3)]],
    constant uint& n_centroids [[buffer(4)]],
    constant uint& n_features [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint point_idx = gid.y;
    uint centroid_idx = gid.x;

    if (point_idx >= n_points || centroid_idx >= n_centroids) return;

    float sum = 0.0f;
    for (uint k = 0; k < n_features; k++) {
        float diff = points[point_idx * n_features + k] - centroids[centroid_idx * n_features + k];
        sum += diff * diff;
    }

    distances[point_idx * n_centroids + centroid_idx] = sum;
}
"#
    .to_string()
}

// ============================================================================
// ROCm/HIP Kernels
// ============================================================================

/// Generate ROCm/HIP kernel for distance computation
pub fn generate_rocm_distance_kernel() -> String {
    r#"
#include <hip/hip_runtime.h>

extern "C" __global__ void squared_euclidean_distance_matrix(
    const float* __restrict__ data,
    float* __restrict__ distances,
    const int n_samples,
    const int n_features
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n_samples || col >= n_samples) return;
    if (row > col) return;

    float sum = 0.0f;
    for (int k = 0; k < n_features; k++) {
        float diff = data[row * n_features + k] - data[col * n_features + k];
        sum += diff * diff;
    }

    distances[row * n_samples + col] = sum;
    distances[col * n_samples + row] = sum;
}

extern "C" __global__ void kmeans_assign_labels(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    int* __restrict__ labels,
    float* __restrict__ distances,
    const int n_samples,
    const int n_centroids,
    const int n_features
) {
    const int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= n_samples) return;

    float min_dist = 1e38f;
    int min_label = 0;

    for (int c = 0; c < n_centroids; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = data[sample_idx * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }

        if (dist < min_dist) {
            min_dist = dist;
            min_label = c;
        }
    }

    labels[sample_idx] = min_label;
    distances[sample_idx] = min_dist;
}
"#
    .to_string()
}

// ============================================================================
// Kernel Selection and Compilation
// ============================================================================

/// Kernel configuration for execution
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size (threads per block)
    pub block_size: (usize, usize, usize),
    /// Grid size (blocks per grid)
    pub grid_size: (usize, usize, usize),
    /// Shared memory size in bytes
    pub shared_mem_size: usize,
    /// Use tensor cores if available
    pub use_tensor_cores: bool,
    /// Preferred data type
    pub data_type: KernelDataType,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: (16, 16, 1),
            grid_size: (1, 1, 1),
            shared_mem_size: 0,
            use_tensor_cores: false,
            data_type: KernelDataType::Float32,
        }
    }
}

/// Data type for kernel operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelDataType {
    /// 16-bit floating point
    Float16,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 8-bit integer
    Int8,
}

/// Get kernel source for the specified backend and operation
pub fn get_kernel_source(backend: GpuBackend, kernel_type: DistanceKernelType) -> Result<String> {
    match backend {
        GpuBackend::Cuda => Ok(generate_cuda_distance_matrix_kernel()),
        GpuBackend::OpenCl => Ok(generate_opencl_distance_matrix_kernel()),
        GpuBackend::Metal => Ok(generate_metal_distance_kernel()),
        GpuBackend::Rocm => Ok(generate_rocm_distance_kernel()),
        GpuBackend::CpuFallback => Err(ClusteringError::InvalidInput(
            "CPU fallback does not use GPU kernels".to_string(),
        )),
        _ => Err(ClusteringError::InvalidInput(format!(
            "Backend {:?} not supported for kernel generation",
            backend
        ))),
    }
}

/// Get K-means kernel source for the specified backend
pub fn get_kmeans_kernel_source(backend: GpuBackend) -> Result<String> {
    match backend {
        GpuBackend::Cuda => Ok(generate_cuda_kmeans_assign_kernel()),
        GpuBackend::OpenCl => Ok(generate_opencl_distance_matrix_kernel()), // Contains K-means too
        GpuBackend::Metal => Ok(generate_metal_distance_kernel()),
        GpuBackend::Rocm => Ok(generate_rocm_distance_kernel()),
        GpuBackend::CpuFallback => Err(ClusteringError::InvalidInput(
            "CPU fallback does not use GPU kernels".to_string(),
        )),
        _ => Err(ClusteringError::InvalidInput(format!(
            "Backend {:?} not supported for kernel generation",
            backend
        ))),
    }
}

/// Calculate optimal kernel configuration for a given problem size
pub fn calculate_kernel_config(
    n_samples: usize,
    n_features: usize,
    backend: GpuBackend,
) -> KernelConfig {
    let block_size = match backend {
        GpuBackend::Cuda | GpuBackend::Rocm => {
            // NVIDIA/AMD typically use 16x16 or 32x32 blocks
            if n_samples <= 256 {
                (16, 16, 1)
            } else {
                (32, 32, 1)
            }
        }
        GpuBackend::Metal => {
            // Metal typically uses smaller blocks
            (16, 16, 1)
        }
        GpuBackend::OpenCl => {
            // OpenCL varies by device
            (16, 16, 1)
        }
        _ => (16, 16, 1),
    };

    let grid_size = (
        (n_samples + block_size.0 - 1) / block_size.0,
        (n_samples + block_size.1 - 1) / block_size.1,
        1,
    );

    // Calculate shared memory for tiling
    let tile_size = block_size.0;
    let shared_mem_size = 2 * tile_size * n_features * std::mem::size_of::<f32>();

    KernelConfig {
        block_size,
        grid_size,
        shared_mem_size,
        use_tensor_cores: matches!(backend, GpuBackend::Cuda | GpuBackend::Rocm),
        data_type: KernelDataType::Float32,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernel_generation() {
        let kernel = generate_cuda_distance_matrix_kernel();
        assert!(kernel.contains("squared_euclidean_distance_matrix"));
        assert!(kernel.contains("euclidean_distance_matrix"));
        assert!(kernel.contains("cosine_distance_matrix"));
    }

    #[test]
    fn test_cuda_kmeans_kernel_generation() {
        let kernel = generate_cuda_kmeans_assign_kernel();
        assert!(kernel.contains("kmeans_assign_labels"));
        assert!(kernel.contains("kmeans_compute_centroids"));
    }

    #[test]
    fn test_opencl_kernel_generation() {
        let kernel = generate_opencl_distance_matrix_kernel();
        assert!(kernel.contains("__kernel"));
        assert!(kernel.contains("squared_euclidean_distance_matrix"));
    }

    #[test]
    fn test_metal_kernel_generation() {
        let kernel = generate_metal_distance_kernel();
        assert!(kernel.contains("using namespace metal"));
        assert!(kernel.contains("squared_euclidean_distance_matrix"));
    }

    #[test]
    fn test_rocm_kernel_generation() {
        let kernel = generate_rocm_distance_kernel();
        assert!(kernel.contains("hip/hip_runtime.h"));
        assert!(kernel.contains("squared_euclidean_distance_matrix"));
    }

    #[test]
    fn test_get_kernel_source() {
        let cuda_source = get_kernel_source(GpuBackend::Cuda, DistanceKernelType::Euclidean);
        assert!(cuda_source.is_ok());

        let cpu_source = get_kernel_source(GpuBackend::CpuFallback, DistanceKernelType::Euclidean);
        assert!(cpu_source.is_err());
    }

    #[test]
    fn test_kernel_config_calculation() {
        let config = calculate_kernel_config(1000, 50, GpuBackend::Cuda);
        assert!(config.block_size.0 > 0);
        assert!(config.grid_size.0 > 0);
        assert!(config.use_tensor_cores);
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size, (16, 16, 1));
        assert_eq!(config.data_type, KernelDataType::Float32);
    }
}
