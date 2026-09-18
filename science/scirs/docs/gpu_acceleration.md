# GPU Acceleration in SciRS2

This document describes the GPU acceleration framework in SciRS2 and provides guidelines for incorporating GPU-accelerated algorithms across the library's modules.

## Implementation Architecture

The SciRS2 GPU acceleration framework has been implemented with the following key principles:

1. **Backend Abstraction**: Support multiple GPU backends (CUDA, WebGPU, Metal, OpenCL)
2. **Memory Management**: Efficient memory transfers between CPU and GPU
3. **Computation Primitives**: Common operations optimized for GPU execution
4. **Integration**: Seamless integration with other optimization techniques (SIMD, parallel)

## Usage Guide

### Basic Usage

To use GPU acceleration in your module, add the core dependency with the GPU feature:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["gpu"] }
```

Then, use the GPU context and buffers:

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, GpuBuffer};

// Create a GPU context with the default backend
let ctx = GpuContext::new(GpuBackend::default())?;

// Allocate memory on the GPU
let mut buffer = ctx.create_buffer::<f32>(1024);

// Copy data to GPU
let host_data = vec![1.0f32; 1024];
buffer.copy_from_host(&host_data);

// Execute a computation
ctx.execute(|compiler| {
    let kernel = compiler.compile(r#"
        __kernel void vector_add(__global float* a) {
            size_t i = get_global_id(0);
            a[i] = a[i] + 1.0f;
        }
    "#)?;

    kernel.set_buffer(0, &mut buffer);
    kernel.dispatch([1024, 1, 1]);
    Ok(())
})?;

// Copy data back to host
let mut result = vec![0.0f32; 1024];
buffer.copy_to_host(&mut result);
```

### Memory Management

The GPU module provides memory pooling for efficient buffer reuse:

```rust
use std::sync::Arc;
use scirs2_core::gpu::{GpuContext, GpuMemoryPool, GpuBuffer};

// Create a shared context
let context = Arc::new(GpuContext::new(GpuBackend::default())?);

// Create a memory pool
let pool = GpuMemoryPool::new(context.clone());

// Acquire buffers from the pool
let buffer = pool.acquire::<f32>(1024);

// Use the buffer for computations
// ...

// Release the buffer back to the pool
pool.release(buffer);
```

## Implementation Guidelines

When implementing GPU-accelerated algorithms in SciRS2 modules:

1. **Feature Gating**: Always gate GPU implementations behind the `gpu` feature flag
2. **Fallbacks**: Provide CPU fallbacks for all GPU-accelerated functions
3. **Error Handling**: Use the `GpuError` type for error reporting
4. **Memory Efficiency**: Minimize CPU-GPU transfers and reuse GPU memory
5. **Testing**: Add tests that validate GPU results against CPU reference implementations

Example implementation pattern:

```rust
fn my_algorithm<T: GpuDataType>(data: &[T]) -> Result<Vec<T>, Error> {
    #[cfg(feature = "gpu")]
    {
        // Try GPU implementation first
        if let Ok(ctx) = GpuContext::new(GpuBackend::default()) {
            return gpu_implementation(&ctx, data)
                .map_err(|e| Error::GpuError(e.to_string()));
        }
    }
    
    // Fall back to CPU implementation
    cpu_implementation(data)
}

#[cfg(feature = "gpu")]
fn gpu_implementation<T: GpuDataType>(
    ctx: &GpuContext, 
    data: &[T]
) -> Result<Vec<T>, GpuError> {
    // GPU implementation
    // ...
}

fn cpu_implementation<T>(data: &[T]) -> Result<Vec<T>, Error> {
    // CPU implementation
    // ...
}
```

## Original Strategy For GPU Acceleration

## Priority Modules for GPU Acceleration

1. **scirs2-linalg** (Highest Priority)
   - Matrix operations are highly parallelizable
   - Tensor contractions and batch matrix operations
   - GPU acceleration for BLAS/LAPACK operations would provide massive speedups
   - Core module used by many other modules

2. **scirs2-neural** (High Priority)
   - Deep learning operations are inherently GPU-friendly
   - Matrix multiplications in forward/backward passes
   - Batch processing of data
   - Training acceleration

3. **scirs2-ndimage** (High Priority)
   - Image filtering operations (gaussian, median, convolutions)
   - Feature detection algorithms
   - Morphological operations

4. **scirs2-fft** (Medium-High Priority)
   - FFT algorithms are computationally intensive and well-suited for GPU
   - Spectral analysis and signal processing

5. **Additional Priority Modules**:
   - **scirs2-sparse** (Medium): Sparse matrix operations
   - **scirs2-optimize** (Medium): Gradient-based optimization methods
   - **scirs2-signal** (Medium): Signal processing operations
   - **scirs2-stats** (Lower): Distribution sampling and statistical operations
   - **scirs2-interpolate** (Lower): Gridded interpolation

## Proposed Architecture for GPU Implementation

### 1. Core-based Integration with Module Extensions

The optimal approach is to build GPU foundations in `scirs2-core` with module-specific extensions:

```
scirs2-core/src/gpu/
├── mod.rs             # Main module and re-exports
├── traits.rs          # Core abstraction traits
├── memory.rs          # Memory management abstractions
├── error.rs           # GPU-specific error types
├── backend/           # Backend implementations
└── ops/               # Basic operations
```

### 2. Trait-based Abstraction

```rust
pub trait ComputeBackend {
    type Scalar: ScalarOps;
    type Vector: VectorOps<Scalar=Self::Scalar>;
    type Matrix: MatrixOps<Scalar=Self::Scalar>;
    
    fn name() -> &'static str;
    fn capabilities() -> BackendCapabilities;
    fn is_available() -> bool;
}

// Backend implementations
pub struct CpuBackend;
pub struct CudaBackend;
pub struct WgpuBackend;
```

### 3. Feature Flag Structure

```toml
# In scirs2-core/Cargo.toml
[features]
default = []
simd = ["wide"]
parallel = ["rayon"]
gpu = ["gpu-abstractions"]
cuda = ["gpu", "rust-cuda"]
wgpu = ["gpu", "wgpu"]
```

### 4. Compatible Rust GPU Libraries

1. **CUDA via rust-cuda**: For NVIDIA GPUs
   - Best performance for numerical computations
   - Most mature ecosystem for scientific computing

2. **WGPU via wgpu**: For cross-platform support
   - Works on all major GPU vendors (NVIDIA, AMD, Intel)
   - Web compatibility and modern GPU API

### 5. User-Facing API

```rust
// User code
let array = Array2::random((1000, 1000));

// CPU computation
let result = some_algorithm(&array);

// GPU-accelerated computation
let gpu_context = GpuContext::new();
let result = gpu_context.compute(|ctx| {
    // Transfer data to GPU
    let gpu_array = ctx.upload(&array)?;
    
    // Compute on GPU
    let gpu_result = some_algorithm_gpu(ctx, &gpu_array)?;
    
    // Transfer result back to CPU
    ctx.download(&gpu_result)
});
```

### 6. Implementation Strategy

1. **Phase 1**: Create core GPU abstraction layer in `scirs2-core`
2. **Phase 2**: Implement BLAS operations accelerated with GPU (prioritize scirs2-linalg)
3. **Phase 3**: Extend to other computational modules (FFT, image processing, statistics)
4. **Phase 4**: Optimize advanced algorithms (neural networks, optimization)

## Key Architecture Features

1. **Layered Architecture**:
   - Core GPU abstractions in scirs2-core
   - Module-specific extensions in domain crates
   - Generic algorithms working across backends

2. **Flexible Backend Selection**:
   - Runtime selection of optimal backend
   - Compile-time feature flags for supported backends
   - CPU fallback for all operations

3. **Memory Management**:
   - Transparent data transfer between CPU and GPU
   - Memory pool and buffer caching
   - Lazy evaluation where beneficial

4. **Maintainability**:
   - Follows existing project structure
   - Consistent with SIMD and parallel modules
   - Clear separation of concerns
   - Well-defined trait boundaries

This architecture provides a comprehensive approach to integrating GPU acceleration while maintaining the project's modular design and ensuring flexibility for users to choose the appropriate backend for their needs.# Test addition
