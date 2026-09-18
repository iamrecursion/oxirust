//! GPU device abstraction and enumeration.

use crate::device::DeviceType;

/// Represents a GPU compute device with its capabilities and memory information.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuDevice {
    /// Device index in the system.
    pub index: usize,
    /// Device type (CUDA, Metal, Vulkan, ROCm).
    pub device_type: DeviceType,
    /// Human-readable device name.
    pub name: String,
    /// Total device memory in bytes.
    pub total_memory_bytes: u64,
    /// Available (free) device memory in bytes.
    pub free_memory_bytes: u64,
    /// CUDA compute capability as (major, minor), if applicable.
    pub compute_capability: Option<(u32, u32)>,
    /// Whether the device supports FP16 (half-precision) operations.
    pub supports_fp16: bool,
    /// Whether the device supports BF16 (bfloat16) operations.
    pub supports_bf16: bool,
}

/// Requirements that can be checked against a GPU device.
#[derive(Debug, Clone)]
pub enum GpuRequirement {
    /// Device must have at least this many bytes of free memory.
    MinMemoryBytes(u64),
    /// Device must have at least this compute capability (major, minor).
    MinComputeCapability(u32, u32),
    /// Device must support FP16 operations.
    Fp16Support,
    /// Device must support BF16 operations.
    Bf16Support,
    /// Device must have tensor cores (CUDA compute capability >= 7.0).
    TensorCores,
}

impl GpuDevice {
    /// Create a new GPU device with the given index, type, and name.
    /// Memory fields default to 0; other fields default to disabled.
    pub fn new(index: usize, device_type: DeviceType, name: String) -> Self {
        Self {
            index,
            device_type,
            name,
            total_memory_bytes: 0,
            free_memory_bytes: 0,
            compute_capability: None,
            supports_fp16: false,
            supports_bf16: false,
        }
    }

    /// Returns a numeric score for the compute capability (major * 100 + minor).
    /// Returns 0 if no compute capability is reported.
    pub fn compute_capability_score(&self) -> u32 {
        match self.compute_capability {
            Some((major, minor)) => major * 100 + minor,
            None => 0,
        }
    }

    /// Returns true if the device satisfies the given requirement.
    pub fn is_capable_of(&self, req: &GpuRequirement) -> bool {
        match req {
            GpuRequirement::MinMemoryBytes(min_bytes) => self.free_memory_bytes >= *min_bytes,
            GpuRequirement::MinComputeCapability(req_major, req_minor) => {
                match self.compute_capability {
                    Some((major, minor)) => {
                        major > *req_major || (major == *req_major && minor >= *req_minor)
                    }
                    None => false,
                }
            }
            GpuRequirement::Fp16Support => self.supports_fp16,
            GpuRequirement::Bf16Support => self.supports_bf16,
            GpuRequirement::TensorCores => {
                // Tensor cores require CUDA compute capability >= 7.0
                match self.compute_capability {
                    Some((major, _)) => major >= 7,
                    None => false,
                }
            }
        }
    }

    /// Enumerate available GPU devices.
    /// Without CUDA bindings, this always returns an empty vector.
    pub fn enumerate() -> Vec<GpuDevice> {
        // Pure Rust stub — no CUDA bindings, no C/Fortran dependencies.
        // Real enumeration would require CUDA runtime bindings.
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        let device = GpuDevice::new(0, DeviceType::Cuda, "Tesla V100".to_string());
        assert_eq!(device.index, 0);
        assert_eq!(device.device_type, DeviceType::Cuda);
        assert_eq!(device.name, "Tesla V100");
        assert_eq!(device.total_memory_bytes, 0);
        assert_eq!(device.free_memory_bytes, 0);
        assert!(device.compute_capability.is_none());
        assert!(!device.supports_fp16);
        assert!(!device.supports_bf16);
    }

    #[test]
    fn test_gpu_device_compute_score() {
        let mut device = GpuDevice::new(0, DeviceType::Cuda, "A100".to_string());
        assert_eq!(device.compute_capability_score(), 0);

        device.compute_capability = Some((8, 0));
        assert_eq!(device.compute_capability_score(), 800);

        device.compute_capability = Some((7, 5));
        assert_eq!(device.compute_capability_score(), 705);
    }

    #[test]
    fn test_is_capable_min_memory() {
        let mut device = GpuDevice::new(0, DeviceType::Cuda, "RTX 3090".to_string());
        device.free_memory_bytes = 16 * 1024 * 1024 * 1024; // 16 GB

        assert!(device.is_capable_of(&GpuRequirement::MinMemoryBytes(8 * 1024 * 1024 * 1024)));
        assert!(device.is_capable_of(&GpuRequirement::MinMemoryBytes(16 * 1024 * 1024 * 1024)));
        assert!(!device.is_capable_of(&GpuRequirement::MinMemoryBytes(32 * 1024 * 1024 * 1024)));
    }

    #[test]
    fn test_is_capable_compute_capability() {
        let mut device = GpuDevice::new(0, DeviceType::Cuda, "RTX 3080".to_string());

        // No capability — should fail
        assert!(!device.is_capable_of(&GpuRequirement::MinComputeCapability(7, 0)));

        device.compute_capability = Some((8, 6));
        assert!(device.is_capable_of(&GpuRequirement::MinComputeCapability(7, 0)));
        assert!(device.is_capable_of(&GpuRequirement::MinComputeCapability(8, 6)));
        assert!(!device.is_capable_of(&GpuRequirement::MinComputeCapability(9, 0)));
        assert!(!device.is_capable_of(&GpuRequirement::MinComputeCapability(8, 7)));
    }

    #[test]
    fn test_is_capable_fp16() {
        let mut device = GpuDevice::new(0, DeviceType::Cuda, "V100".to_string());
        assert!(!device.is_capable_of(&GpuRequirement::Fp16Support));

        device.supports_fp16 = true;
        assert!(device.is_capable_of(&GpuRequirement::Fp16Support));
        assert!(!device.is_capable_of(&GpuRequirement::Bf16Support));
    }

    #[test]
    fn test_enumerate_returns_empty_without_cuda() {
        let devices = GpuDevice::enumerate();
        // Without real CUDA bindings, enumeration must return empty.
        assert!(devices.is_empty());
    }
}
