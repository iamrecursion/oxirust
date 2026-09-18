//! GPU Kernel Registry System
//!
//! This module provides a centralized registry for all GPU kernels used across
//! the SciRS2 ecosystem. All modules must register their GPU kernels here
//! instead of implementing them directly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "gpu")]
use crate::gpu::{GpuBackend, GpuDevice, GpuError, GpuKernel};

/// GPU kernel identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct KernelId {
    /// Module that owns this kernel (e.g., "linalg", "fft", "neural")
    pub module: String,
    /// Operation name (e.g., "gemm", "fft2d", "conv2d")
    pub operation: String,
    /// Data type (e.g., "f32", "f64", "c32", "c64")
    pub dtype: String,
    /// Optional variant (e.g., "transposed", "batched", "strided")
    pub variant: Option<String>,
}

impl KernelId {
    /// Create a new kernel identifier
    pub fn new(module: &str, operation: &str, dtype: &str) -> Self {
        Self {
            module: module.to_string(),
            operation: operation.to_string(),
            dtype: dtype.to_string(),
            variant: None,
        }
    }

    /// Create a new kernel identifier with a variant
    pub fn with_variant(module: &str, operation: &str, dtype: &str, variant: &str) -> Self {
        Self {
            module: module.to_string(),
            operation: operation.to_string(),
            dtype: dtype.to_string(),
            variant: Some(variant.to_string()),
        }
    }

    /// Get a string representation suitable for kernel naming
    pub fn as_kernel_name(&self) -> String {
        match &self.variant {
            Some(variant) => format!(
                "{}_{}_{}__{}",
                self.module, self.operation, self.dtype, variant
            ),
            None => format!(
                "{module}_{operation}__{dtype}",
                module = self.module,
                operation = self.operation,
                dtype = self.dtype
            ),
        }
    }
}

/// GPU kernel source code and metadata
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// The actual kernel source code
    pub source: String,
    /// The backend this kernel is written for
    pub backend: GpuBackend,
    /// Entry point function name
    pub entry_point: String,
    /// Required workgroup/block size (x, y, z)
    pub workgroup_size: (u32, u32, u32),
    /// Shared memory requirements in bytes
    pub shared_memory: usize,
    /// Whether this kernel uses tensor cores or similar
    pub uses_tensor_cores: bool,
}

/// Compiled kernel cache entry
#[cfg(feature = "gpu")]
struct CompiledKernel {
    kernel: Arc<GpuKernel>,
    device_id: usize,
}

/// Global GPU kernel registry
static KERNEL_REGISTRY: OnceLock<Mutex<KernelRegistry>> = OnceLock::new();

/// GPU kernel registry
pub struct KernelRegistry {
    /// Registered kernel sources
    sources: HashMap<KernelId, Vec<KernelSource>>,
    /// Compiled kernel cache
    #[cfg(feature = "gpu")]
    compiled_cache: HashMap<(KernelId, usize), CompiledKernel>,
}

impl KernelRegistry {
    /// Create a new kernel registry
    fn new() -> Self {
        Self {
            sources: HashMap::new(),
            #[cfg(feature = "gpu")]
            compiled_cache: HashMap::new(),
        }
    }

    /// Get the global kernel registry
    pub fn global() -> &'static Mutex<KernelRegistry> {
        KERNEL_REGISTRY.get_or_init(|| {
            let mut registry = KernelRegistry::new();
            // Register built-in kernels
            registry.register_builtin_kernels();
            Mutex::new(registry)
        })
    }

    /// Register built-in kernels from scirs2-core
    fn register_builtin_kernels(&mut self) {
        // Register BLAS kernels
        self.register_blas_kernels();

        // Register reduction kernels
        self.register_reduction_kernels();

        // Register utility kernels
        self.register_utility_kernels();
    }

    /// Register a kernel source
    pub fn register_kernel(&mut self, id: KernelId, source: KernelSource) {
        self.sources.entry(id).or_default().push(source);
    }

    /// Get kernel sources for a given ID
    pub fn get_sources(&self, id: &KernelId) -> Option<&Vec<KernelSource>> {
        self.sources.get(id)
    }

    /// Get a compiled kernel for the current device
    #[cfg(feature = "gpu")]
    pub fn get_kernel(
        &mut self,
        id: &KernelId,
        device: &GpuDevice,
    ) -> Result<Arc<GpuKernel>, GpuError> {
        let device_id = device.device_id();
        let cache_key = (id.clone(), device_id);

        // Check cache first
        if let Some(cached) = self.compiled_cache.get(&cache_key) {
            if cached.device_id == device_id {
                return Ok(cached.kernel.clone());
            }
        }

        // Find appropriate source for the device's backend
        let sources = self
            .sources
            .get(id)
            .ok_or_else(|| GpuError::KernelNotFound(id.as_kernel_name()))?;

        let source = sources
            .iter()
            .find(|s| s.backend == device.backend())
            .ok_or_else(|| GpuError::BackendNotSupported(device.backend()))?;

        // Compile the kernel
        let kernel = device.compile_kernel(&source.source, &source.entry_point)?;
        let kernel = Arc::new(kernel);

        // Cache the compiled kernel
        self.compiled_cache.insert(
            cache_key,
            CompiledKernel {
                kernel: kernel.clone(),
                device_id,
            },
        );

        Ok(kernel)
    }

    /// Clear the compiled kernel cache
    #[cfg(feature = "gpu")]
    pub fn clear_cache(&mut self) {
        self.compiled_cache.clear();
    }

    /// List all registered kernels
    pub fn list_kernels(&self) -> Vec<KernelId> {
        self.sources.keys().cloned().collect()
    }

    /// Check if a kernel is registered
    pub fn has_kernel(&self, id: &KernelId) -> bool {
        self.sources.contains_key(id)
    }
}

// Built-in kernel registration implementations
impl KernelRegistry {
    fn register_blas_kernels(&mut self) {
        // GEMM kernels for f32
        self.register_kernel(
            KernelId::new("core", "gemm", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/gemm_f32.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "gemm_f32".to_string(),
                workgroup_size: (16, 16, 1),
                shared_memory: 4096,
                uses_tensor_cores: false,
            },
        );

        // GEMM kernels for f64
        self.register_kernel(
            KernelId::new("core", "gemm", "f64"),
            KernelSource {
                source: include_str!("gpu/kernels/gemm_f64.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "gemm_f64".to_string(),
                workgroup_size: (16, 16, 1),
                shared_memory: 8192,
                uses_tensor_cores: false,
            },
        );

        // AXPY kernels
        self.register_kernel(
            KernelId::new("core", "axpy", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/axpy.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "axpy_f32".to_string(),
                workgroup_size: (256, 1, 1),
                shared_memory: 0,
                uses_tensor_cores: false,
            },
        );
    }

    fn register_reduction_kernels(&mut self) {
        // Sum reduction kernels
        self.register_kernel(
            KernelId::new("core", "reduce_sum", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/reduce_sum.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "reduce_sum_f32".to_string(),
                workgroup_size: (256, 1, 1),
                shared_memory: 1024,
                uses_tensor_cores: false,
            },
        );

        // Max reduction kernels
        self.register_kernel(
            KernelId::new("core", "reduce_max", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/reduce_max.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "reduce_max_f32".to_string(),
                workgroup_size: (256, 1, 1),
                shared_memory: 1024,
                uses_tensor_cores: false,
            },
        );
    }

    fn register_utility_kernels(&mut self) {
        // Memory copy kernels
        self.register_kernel(
            KernelId::new("core", "memcpy", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/memcpy.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "memcpy_f32".to_string(),
                workgroup_size: (256, 1, 1),
                shared_memory: 0,
                uses_tensor_cores: false,
            },
        );

        // Fill kernels
        self.register_kernel(
            KernelId::new("core", "fill", "f32"),
            KernelSource {
                source: include_str!("gpu/kernels/fill.cu").to_string(),
                backend: GpuBackend::Cuda,
                entry_point: "fill_f32".to_string(),
                workgroup_size: (256, 1, 1),
                shared_memory: 0,
                uses_tensor_cores: false,
            },
        );
    }
}

/// Register a kernel from a module
///
/// This should be called during module initialization to register
/// module-specific kernels with the global registry.
///
/// # Example
///
/// ```rust
/// use scirs2_core::gpu_registry::{register_module_kernel, KernelId, KernelSource};
/// use scirs2_core::gpu::GpuBackend;
///
/// fn register_fft_kernels() {
///     let fft_kernel_source = r#"
///         extern "C" __global__ void fft2d_c32(float2* data, int n) {
///             // FFT kernel implementation
///         }
///     "#;
///
///     register_module_kernel(
///         KernelId::new("fft", "fft2d", "c32"),
///         KernelSource {
///             source: fft_kernel_source.to_string(),
///             backend: GpuBackend::Cuda,
///             entry_point: "fft2d_c32".to_string(),
///             workgroup_size: (32, 8, 1),
///             shared_memory: 16384,
///             uses_tensor_cores: false,
///         },
///     );
/// }
/// ```
#[allow(dead_code)]
pub fn register_module_kernel(id: KernelId, source: KernelSource) {
    let registry = KernelRegistry::global();
    let mut registry = registry.lock().expect("Operation failed");
    registry.register_kernel(id, source);
}

/// Get a compiled kernel for the current device
#[cfg(feature = "gpu")]
#[allow(dead_code)]
pub fn get_kernel(id: &KernelId, device: &GpuDevice) -> Result<Arc<GpuKernel>, GpuError> {
    let registry = KernelRegistry::global();
    let mut registry = registry.lock().expect("Operation failed");
    registry.get_kernel(id, device)
}

/// Check if a kernel is registered
#[allow(dead_code)]
pub fn has_kernel(id: &KernelId) -> bool {
    let registry = KernelRegistry::global();
    let registry = registry.lock().expect("Operation failed");
    registry.has_kernel(id)
}

/// List all registered kernels
#[allow(dead_code)]
pub fn list_kernels() -> Vec<KernelId> {
    let registry = KernelRegistry::global();
    let registry = registry.lock().expect("Operation failed");
    registry.list_kernels()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_id() {
        let id = KernelId::new("linalg", "gemm", "f32");
        assert_eq!(id.as_kernel_name(), "linalg_gemm__f32");

        let id_with_variant = KernelId::with_variant("fft", "fft2d", "c64", "batched");
        assert_eq!(id_with_variant.as_kernel_name(), "fft_fft2d_c64__batched");
    }

    #[test]
    fn test_kernel_registration() {
        let id = KernelId::new("test", "dummy", "f32");
        let source = KernelSource {
            source: "dummy kernel".to_string(),
            backend: GpuBackend::Cuda,
            entry_point: "dummy".to_string(),
            workgroup_size: (1, 1, 1),
            shared_memory: 0,
            uses_tensor_cores: false,
        };

        register_module_kernel(id.clone(), source);
        assert!(has_kernel(&id));
    }

    #[test]
    fn test_builtin_kernels() {
        // Check that some built-in kernels are registered
        assert!(has_kernel(&KernelId::new("core", "gemm", "f32")));
        assert!(has_kernel(&KernelId::new("core", "reduce_sum", "f32")));
        assert!(has_kernel(&KernelId::new("core", "fill", "f32")));
    }
}
