# GPU Kernel Library Implementation

This document provides a detailed implementation plan and code examples for creating a comprehensive GPU kernel library as part of the `scirs2-core` GPU acceleration module.

## Overview

The GPU kernel library will provide pre-optimized kernels for common mathematical operations across different GPU backends (CUDA, WebGPU, Metal, OpenCL). These kernels will serve as building blocks for more complex algorithms implemented in various SciRS2 modules.

## Directory Structure

```
scirs2-core/src/gpu/
├── kernels/
│   ├── mod.rs                  # Main module exports
│   ├── common.rs               # Common kernel utilities
│   ├── blas/                   # BLAS-like operations
│   │   ├── mod.rs
│   │   ├── gemm.rs             # Matrix multiplication
│   │   ├── axpy.rs             # Vector operations
│   │   └── ...
│   ├── transform/              # Data transformations
│   │   ├── mod.rs
│   │   ├── fft.rs              # FFT operations
│   │   └── ...
│   ├── reduction/              # Reduction operations
│   │   ├── mod.rs
│   │   ├── sum.rs              # Summation
│   │   ├── norm.rs             # Vector/matrix norms
│   │   └── ...
│   └── ml/                     # Machine learning primitives
│       ├── mod.rs
│       ├── activation.rs       # Activation functions
│       ├── pooling.rs          # Pooling operations
│       └── ...
└── backend/
    ├── mod.rs
    ├── cuda.rs                 # CUDA-specific implementations
    ├── wgpu.rs                 # WebGPU-specific implementations
    ├── metal.rs                # Metal-specific implementations
    └── opencl.rs               # OpenCL-specific implementations
```

## Implementation Plan

### 1. Kernel Interface Design

```rust
/// GPU Kernel interface
pub trait GpuKernel {
    /// The name of the kernel
    fn name(&self) -> &str;
    
    /// Get kernel source for the specified backend
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError>;
    
    /// Get kernel metadata (workgroup size, memory requirements, etc.)
    fn metadata(&self) -> KernelMetadata;
    
    /// Can this kernel be specialized for the given parameters?
    fn can_specialize(&self, params: &KernelParams) -> bool;
    
    /// Create a specialized version of this kernel for the given parameters
    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError>;
}

/// Kernel metadata for optimal execution
pub struct KernelMetadata {
    /// Recommended workgroup size
    pub workgroup_size: [u32; 3],
    /// Local memory usage in bytes
    pub local_memory_usage: usize,
    /// Whether the kernel supports tensor cores (NVIDIA) or similar
    pub supports_tensor_cores: bool,
    /// Operations type (compute intensive, memory intensive, balanced)
    pub operation_type: OperationType,
}

/// Parameters for kernel specialization
pub struct KernelParams {
    /// Numeric type (f32, f64, etc.)
    pub data_type: DataType,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Additional numeric parameters
    pub numeric_params: HashMap<String, f64>,
    /// Additional string parameters
    pub string_params: HashMap<String, String>,
}
```

### 2. Base Kernel Implementation

```rust
pub struct BaseKernel {
    name: String,
    cuda_source: String,
    wgpu_source: String,
    metal_source: String,
    opencl_source: String,
    metadata: KernelMetadata,
}

impl BaseKernel {
    pub fn new(
        name: &str,
        cuda_source: &str,
        wgpu_source: &str,
        metal_source: &str,
        opencl_source: &str,
        metadata: KernelMetadata,
    ) -> Self {
        Self {
            name: name.to_string(),
            cuda_source: cuda_source.to_string(),
            wgpu_source: wgpu_source.to_string(),
            metal_source: metal_source.to_string(),
            opencl_source: opencl_source.to_string(),
            metadata,
        }
    }
}

impl GpuKernel for BaseKernel {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Cuda => Ok(self.cuda_source.clone()),
            GpuBackend::Wgpu => Ok(self.wgpu_source.clone()),
            GpuBackend::Metal => Ok(self.metal_source.clone()),
            GpuBackend::OpenCL => Ok(self.opencl_source.clone()),
            _ => Err(GpuError::UnsupportedBackend(backend)),
        }
    }
    
    fn metadata(&self) -> KernelMetadata {
        self.metadata.clone()
    }
    
    fn can_specialize(&self, _params: &KernelParams) -> bool {
        false // Base implementation doesn't support specialization
    }
    
    fn specialize(&self, _params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        Err(GpuError::SpecializationNotSupported)
    }
}
```

### 3. GEMM (Matrix Multiplication) Kernel Example

```rust
pub struct GemmKernel {
    base: BaseKernel,
}

impl GemmKernel {
    pub fn new() -> Self {
        let metadata = KernelMetadata {
            workgroup_size: [16, 16, 1],
            local_memory_usage: 8192, // 8 KB
            supports_tensor_cores: true,
            operation_type: OperationType::ComputeIntensive,
        };
        
        Self {
            base: BaseKernel::new(
                "gemm",
                include_str!("../../../kernels/cuda/gemm.cu"),
                include_str!("../../../kernels/wgpu/gemm.wgsl"),
                include_str!("../../../kernels/metal/gemm.metal"),
                include_str!("../../../kernels/opencl/gemm.cl"),
                metadata,
            )
        }
    }
    
    pub fn with_alpha_beta(alpha: f32, beta: f32) -> Box<dyn GpuKernel> {
        let mut kernel = Self::new();
        // Customize kernel for alpha/beta values
        // ...
        Box::new(kernel)
    }
}

impl GpuKernel for GemmKernel {
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
        // Check if we can specialize for these parameters
        match params.data_type {
            DataType::Float32 | DataType::Float64 => true,
            _ => false,
        }
    }
    
    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        if !self.can_specialize(params) {
            return Err(GpuError::SpecializationNotSupported);
        }
        
        // Implement specialization logic here
        // For example, we could generate different kernels based on matrix dimensions
        // to optimize for different hardware or matrix sizes
        
        let m = params.input_dims.get(0).copied().unwrap_or(0);
        let k = params.input_dims.get(1).copied().unwrap_or(0);
        let n = params.output_dims.get(1).copied().unwrap_or(0);
        
        let specialized = match params.data_type {
            DataType::Float32 => {
                if m % 64 == 0 && n % 64 == 0 && k % 64 == 0 {
                    // Use a specialized kernel for dimensions divisible by 64
                    self.generate_specialized_kernel_64(m, n, k)?
                } else {
                    // Use a more general kernel
                    self.generate_specialized_kernel_general(m, n, k)?
                }
            },
            DataType::Float64 => {
                // Different optimization for double precision
                self.generate_specialized_kernel_double(m, n, k)?
            },
            _ => return Err(GpuError::UnsupportedDataType(params.data_type)),
        };
        
        Ok(Box::new(specialized))
    }
}

impl GemmKernel {
    fn generate_specialized_kernel_64(&self, m: usize, n: usize, k: usize) -> Result<GemmKernel, GpuError> {
        // Generate a specialized kernel for dimensions divisible by 64
        // This would typically involve code generation or template instantiation
        Ok(GemmKernel::new()) // Simplified for this example
    }
    
    fn generate_specialized_kernel_general(&self, m: usize, n: usize, k: usize) -> Result<GemmKernel, GpuError> {
        // Generate a general kernel
        Ok(GemmKernel::new()) // Simplified for this example
    }
    
    fn generate_specialized_kernel_double(&self, m: usize, n: usize, k: usize) -> Result<GemmKernel, GpuError> {
        // Generate a kernel optimized for double precision
        Ok(GemmKernel::new()) // Simplified for this example
    }
}
```

### 4. Kernel Registry

```rust
pub struct KernelRegistry {
    kernels: HashMap<String, Box<dyn GpuKernel>>,
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }
    
    pub fn with_default_kernels() -> Self {
        let mut registry = Self::new();
        
        // Register BLAS kernels
        registry.register(Box::new(GemmKernel::new()));
        registry.register(Box::new(AxpyKernel::new()));
        
        // Register transform kernels
        registry.register(Box::new(FftKernel::new()));
        
        // Register reduction kernels
        registry.register(Box::new(SumKernel::new()));
        registry.register(Box::new(NormKernel::new()));
        
        // Register ML kernels
        registry.register(Box::new(ReluKernel::new()));
        registry.register(Box::new(SigmoidKernel::new()));
        registry.register(Box::new(MaxPoolKernel::new()));
        
        registry
    }
    
    pub fn register(&mut self, kernel: Box<dyn GpuKernel>) {
        self.kernels.insert(kernel.name().to_string(), kernel);
    }
    
    pub fn get(&self, name: &str) -> Option<&Box<dyn GpuKernel>> {
        self.kernels.get(name)
    }
    
    pub fn get_specialized(
        &self, 
        name: &str, 
        params: &KernelParams
    ) -> Result<Box<dyn GpuKernel>, GpuError> {
        let kernel = self.get(name).ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;
        
        if kernel.can_specialize(params) {
            kernel.specialize(params)
        } else {
            Err(GpuError::SpecializationNotSupported)
        }
    }
}
```

### 5. Integration with GpuContext

```rust
impl GpuContext {
    // Existing implementation...
    
    /// Get a kernel from the registry
    pub fn get_kernel(&self, name: &str) -> Result<GpuKernelHandle, GpuError> {
        let kernel = self.kernel_registry.get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;
            
        let kernel_source = kernel.source_for_backend(self.backend)?;
        let metadata = kernel.metadata();
        
        let handle = self.compile_kernel_with_metadata(&kernel_source, &metadata)?;
        Ok(handle)
    }
    
    /// Get a specialized kernel from the registry
    pub fn get_specialized_kernel(
        &self, 
        name: &str, 
        params: &KernelParams
    ) -> Result<GpuKernelHandle, GpuError> {
        let specialized = self.kernel_registry.get_specialized(name, params)?;
        let kernel_source = specialized.source_for_backend(self.backend)?;
        let metadata = specialized.metadata();
        
        let handle = self.compile_kernel_with_metadata(&kernel_source, &metadata)?;
        Ok(handle)
    }
}
```

### 6. User API Examples

```rust
// Example 1: Basic matrix multiplication
fn matrix_multiply(ctx: &GpuContext, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>, GpuError> {
    let a_buffer = ctx.create_buffer_from_slice(a.as_slice().unwrap());
    let b_buffer = ctx.create_buffer_from_slice(b.as_slice().unwrap());
    let c_buffer = ctx.create_buffer::<f32>(a.nrows() * b.ncols());
    
    // Get pre-optimized GEMM kernel
    let kernel = ctx.get_kernel("gemm")?;
    
    // Set kernel parameters
    kernel.set_buffer("a", &a_buffer);
    kernel.set_buffer("b", &b_buffer);
    kernel.set_buffer("c", &c_buffer);
    kernel.set_u32("m", a.nrows() as u32);
    kernel.set_u32("n", b.ncols() as u32);
    kernel.set_u32("k", a.ncols() as u32);
    
    // Execute kernel
    kernel.dispatch([
        (b.ncols() as u32 + 15) / 16,
        (a.nrows() as u32 + 15) / 16,
        1
    ]);
    
    // Read result
    let result_vec = c_buffer.to_vec();
    
    // Reshape into Array2
    let result = Array2::from_shape_vec((a.nrows(), b.ncols()), result_vec)?;
    
    Ok(result)
}

// Example 2: Specialized matrix multiplication for specific dimensions
fn matrix_multiply_specialized(
    ctx: &GpuContext, 
    a: &Array2<f32>, 
    b: &Array2<f32>
) -> Result<Array2<f32>, GpuError> {
    let params = KernelParams {
        data_type: DataType::Float32,
        input_dims: vec![a.nrows(), a.ncols()],
        output_dims: vec![a.nrows(), b.ncols()],
        numeric_params: HashMap::new(),
        string_params: HashMap::new(),
    };
    
    // Get specialized kernel
    let kernel = ctx.get_specialized_kernel("gemm", &params)?;
    
    // Use kernel...
    // (similar to previous example)
    
    Ok(result)
}
```

## CUDA GEMM Kernel Example

```cuda
extern "C" __global__ void gemm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m, int n, int k
) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    int aBegin = k * BLOCK_SIZE * by;
    
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + k - 1;
    
    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    
    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * n;
    
    // The element of the block sub-matrix that is computed
    // by the thread
    float Csub = 0;
    
    // Loop over all the sub-matrices of A and B required to
    // compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {
        
        // Shared memory for the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        
        // Shared memory for the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // Load the matrices from global memory to shared memory
        As[ty][tx] = a[a + k * ty + tx];
        Bs[ty][tx] = b[b + n * ty + tx];
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the block sub-matrix to global memory
    int c_idx = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[c_idx + n * ty + tx] = Csub;
}
```

## WGSL GEMM Kernel Example

```wgsl
struct Uniforms {
    m: u32,
    n: u32,
    k: u32,
    block_size: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, write> c: array<f32>;

var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let bx = workgroup_id.x;
    let by = workgroup_id.y;
    
    let tx = local_id.x;
    let ty = local_id.y;
    
    let block_size = 16u;
    
    // Index of c
    let row = by * block_size + ty;
    let col = bx * block_size + tx;
    
    var sum = 0.0;
    
    // Loop over A and B tiles
    for (var t = 0u; t < (uniforms.k + block_size - 1u) / block_size; t = t + 1u) {
        // Load A tile
        if (row < uniforms.m && t * block_size + tx < uniforms.k) {
            As[ty][tx] = a[row * uniforms.k + t * block_size + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        // Load B tile
        if (t * block_size + ty < uniforms.k && col < uniforms.n) {
            Bs[ty][tx] = b[(t * block_size + ty) * uniforms.n + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute
        for (var k = 0u; k < block_size; k = k + 1u) {
            sum = sum + As[ty][k] * Bs[k][tx];
        }
        
        workgroupBarrier();
    }
    
    // Write result
    if (row < uniforms.m && col < uniforms.n) {
        c[row * uniforms.n + col] = sum;
    }
}
```

## Benefits of the Kernel Library

1. **Performance Optimization**: Kernels can be highly optimized for each backend
2. **Code Reuse**: Core algorithms only need to be implemented once
3. **Specialization**: Kernels can be specialized for specific data types and dimensions
4. **Maintainability**: Separates algorithm implementations from the framework code
5. **Extensibility**: New kernels can be added without modifying the core framework

## Integration with Other Modules

The kernel library will be particularly useful for:

1. **scirs2-linalg**: Matrix operations, decompositions, and solvers
2. **scirs2-fft**: Fast Fourier Transform implementations
3. **scirs2-neural**: Neural network layer implementations
4. **scirs2-ndimage**: Image filtering and processing operations
5. **scirs2-optimize**: Gradient-based optimization algorithms

## Next Steps

1. Implement the base kernel infrastructure
2. Create optimized versions of core BLAS operations (GEMM, AXPY, etc.)
3. Add transform operations like FFT
4. Add reduction operations (sum, norm, etc.)
5. Implement machine learning primitives
6. Create specialized versions for different hardware targets
7. Add benchmarking and auto-tuning capabilities