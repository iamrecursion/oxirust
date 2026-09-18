//! Backend implementations for GPU and CPU compute

pub mod cpu;
pub mod vulkan;

pub use cpu::CpuBackend;
pub use vulkan::VulkanBackend;

use crate::Result;

/// Backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Vulkan backend
    Vulkan,
    /// Metal backend (macOS/iOS)
    Metal,
    /// DirectX 12 backend (Windows)
    DX12,
    /// CPU SIMD fallback
    CPU,
}

impl BackendType {
    /// Get the backend name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::DX12 => "DirectX 12",
            Self::CPU => "CPU SIMD",
        }
    }

    /// Check if this is a GPU backend
    #[must_use]
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::CPU)
    }
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend type
    pub backend_type: BackendType,
    /// Maximum workgroup size (x, y, z)
    pub max_workgroup_size: (u32, u32, u32),
    /// Maximum workgroup invocations
    pub max_workgroup_invocations: u32,
    /// Maximum buffer size in bytes
    pub max_buffer_size: u64,
    /// Supports compute shaders
    pub compute_shaders: bool,
    /// Supports subgroups
    pub subgroups: bool,
    /// Supports push constants
    pub push_constants: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Vulkan,
            max_workgroup_size: (256, 256, 64),
            max_workgroup_invocations: 256,
            max_buffer_size: 1024 * 1024 * 1024, // 1 GB
            compute_shaders: true,
            subgroups: false,
            push_constants: false,
        }
    }
}

/// Backend trait for different compute backends
pub trait Backend {
    /// Get backend capabilities
    fn capabilities(&self) -> &BackendCapabilities;

    /// Get backend type
    fn backend_type(&self) -> BackendType {
        self.capabilities().backend_type
    }

    /// Check if this backend is available on the current system
    fn is_available() -> bool
    where
        Self: Sized;

    /// Initialize the backend
    fn initialize() -> Result<Self>
    where
        Self: Sized;
}
