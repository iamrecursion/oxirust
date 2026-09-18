//! Vulkan compute backend implementation
//!
//! This module provides Vulkan-based GPU compute acceleration using wgpu.
//! wgpu abstracts the low-level Vulkan API and provides a safe Rust interface.

use super::{Backend, BackendCapabilities, BackendType};
use crate::{GpuDevice, Result};

/// Vulkan backend for GPU compute
pub struct VulkanBackend {
    capabilities: BackendCapabilities,
    device: Option<GpuDevice>,
}

impl VulkanBackend {
    /// Create a new Vulkan backend
    pub fn new() -> Result<Self> {
        let device = GpuDevice::new(None)?;
        let capabilities = Self::query_capabilities(&device);

        Ok(Self {
            capabilities,
            device: Some(device),
        })
    }

    /// Query backend capabilities from the device
    fn query_capabilities(device: &GpuDevice) -> BackendCapabilities {
        let info = device.info();

        // Determine backend type from device info
        let backend_type = match info.backend.as_str() {
            "Vulkan" => BackendType::Vulkan,
            "Metal" => BackendType::Metal,
            "DirectX 12" => BackendType::DX12,
            _ => BackendType::Vulkan, // Default
        };

        // wgpu provides reasonable defaults
        BackendCapabilities {
            backend_type,
            max_workgroup_size: (256, 256, 64),
            max_workgroup_invocations: 256,
            max_buffer_size: 1024 * 1024 * 1024, // 1 GB default
            compute_shaders: true,
            subgroups: false,
            push_constants: false,
        }
    }

    /// Get the device
    #[must_use]
    pub fn device(&self) -> Option<&GpuDevice> {
        self.device.as_ref()
    }

    /// Check if Vulkan is the active backend
    #[must_use]
    pub fn is_vulkan(&self) -> bool {
        matches!(self.capabilities.backend_type, BackendType::Vulkan)
    }

    /// Get device information
    #[must_use]
    pub fn device_info(&self) -> Option<String> {
        self.device.as_ref().map(|d| {
            let info = d.info();
            format!("{} ({}) - {}", info.name, info.device_type, info.backend)
        })
    }
}

impl Backend for VulkanBackend {
    fn capabilities(&self) -> &BackendCapabilities {
        &self.capabilities
    }

    fn is_available() -> bool {
        // Try to create a device to check if Vulkan is available
        GpuDevice::new(None).is_ok()
    }

    fn initialize() -> Result<Self> {
        Self::new()
    }
}

impl Default for VulkanBackend {
    /// Creates a Vulkan backend with default settings.
    ///
    /// # Panics
    ///
    /// Panics if no Vulkan-capable GPU device is available. Prefer
    /// [`VulkanBackend::new()`] for fallible construction.
    fn default() -> Self {
        match Self::new() {
            Ok(backend) => backend,
            Err(e) => panic!("Failed to initialize Vulkan backend: {e}"),
        }
    }
}

/// Vulkan-specific features and extensions
pub struct VulkanFeatures {
    /// Descriptor indexing support
    pub descriptor_indexing: bool,
    /// Buffer device address support
    pub buffer_device_address: bool,
    /// Shader subgroup support
    pub subgroup_operations: bool,
    /// Ray tracing support
    pub ray_tracing: bool,
}

impl VulkanFeatures {
    /// Query available Vulkan features
    #[must_use]
    pub fn query() -> Self {
        // In wgpu, feature detection is abstracted
        // We would need to query wgpu::Features for available features
        Self {
            descriptor_indexing: false,
            buffer_device_address: false,
            subgroup_operations: false,
            ray_tracing: false,
        }
    }

    /// Check if any advanced features are available
    #[must_use]
    pub fn has_advanced_features(&self) -> bool {
        self.descriptor_indexing
            || self.buffer_device_address
            || self.subgroup_operations
            || self.ray_tracing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU hardware probe; run with --ignored
    fn test_vulkan_backend_available() {
        // This test will pass if Vulkan is available on the system
        let available = VulkanBackend::is_available();
        println!("Vulkan available: {available}");
    }

    #[test]
    fn test_vulkan_features() {
        let features = VulkanFeatures::query();
        println!("Vulkan features: {:?}", features.has_advanced_features());
    }
}
