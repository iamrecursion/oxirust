#![allow(dead_code)]
//! GPU device capability detection and feature reporting.
//!
//! Queries and stores hardware capabilities such as maximum texture sizes,
//! compute shader limits, supported features, and memory constraints so that
//! the acceleration layer can make informed scheduling decisions.

use std::collections::HashSet;
use std::fmt;

/// Supported GPU compute features that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFeature {
    /// 16-bit (half-precision) floating-point compute.
    Float16Compute,
    /// 64-bit (double-precision) floating-point compute.
    Float64Compute,
    /// 16-bit integer atomics in shaders.
    Int16Atomics,
    /// Shared memory atomics.
    SharedMemoryAtomics,
    /// Subgroup (warp/wavefront) operations.
    SubgroupOps,
    /// Indirect compute dispatch.
    IndirectDispatch,
    /// Descriptor indexing (bindless resources).
    DescriptorIndexing,
    /// Push constants support.
    PushConstants,
    /// Timeline semaphores for synchronization.
    TimelineSemaphores,
    /// Cooperative matrix (tensor core) operations.
    CooperativeMatrix,
}

impl fmt::Display for GpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float16Compute => write!(f, "fp16_compute"),
            Self::Float64Compute => write!(f, "fp64_compute"),
            Self::Int16Atomics => write!(f, "int16_atomics"),
            Self::SharedMemoryAtomics => write!(f, "shared_mem_atomics"),
            Self::SubgroupOps => write!(f, "subgroup_ops"),
            Self::IndirectDispatch => write!(f, "indirect_dispatch"),
            Self::DescriptorIndexing => write!(f, "descriptor_indexing"),
            Self::PushConstants => write!(f, "push_constants"),
            Self::TimelineSemaphores => write!(f, "timeline_semaphores"),
            Self::CooperativeMatrix => write!(f, "cooperative_matrix"),
        }
    }
}

/// Memory information for a GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemoryInfo {
    /// Total device-local (VRAM) memory in bytes.
    pub device_local_bytes: u64,
    /// Total host-visible memory in bytes.
    pub host_visible_bytes: u64,
    /// Maximum single allocation size in bytes.
    pub max_allocation_bytes: u64,
    /// Number of memory heaps.
    pub heap_count: u32,
}

impl DeviceMemoryInfo {
    /// Returns the total VRAM in megabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn vram_mb(&self) -> f64 {
        self.device_local_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns the total VRAM in gigabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn vram_gb(&self) -> f64 {
        self.device_local_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Returns true if the device has at least the specified amount of VRAM (in bytes).
    #[must_use]
    pub fn has_minimum_vram(&self, min_bytes: u64) -> bool {
        self.device_local_bytes >= min_bytes
    }
}

/// Compute shader dimension limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeLimits {
    /// Maximum workgroup size in the X dimension.
    pub max_workgroup_size_x: u32,
    /// Maximum workgroup size in the Y dimension.
    pub max_workgroup_size_y: u32,
    /// Maximum workgroup size in the Z dimension.
    pub max_workgroup_size_z: u32,
    /// Maximum total invocations per workgroup.
    pub max_workgroup_invocations: u32,
    /// Maximum number of workgroups in X dimension per dispatch.
    pub max_dispatch_x: u32,
    /// Maximum number of workgroups in Y dimension per dispatch.
    pub max_dispatch_y: u32,
    /// Maximum number of workgroups in Z dimension per dispatch.
    pub max_dispatch_z: u32,
    /// Size of shared memory per workgroup in bytes.
    pub max_shared_memory_bytes: u32,
}

impl ComputeLimits {
    /// Returns the maximum total workgroup size (product of x * y * z capped by invocations).
    #[must_use]
    pub fn max_total_workgroup_size(&self) -> u32 {
        let product = self
            .max_workgroup_size_x
            .saturating_mul(self.max_workgroup_size_y)
            .saturating_mul(self.max_workgroup_size_z);
        product.min(self.max_workgroup_invocations)
    }

    /// Suggests an optimal 1D workgroup size for a given total element count.
    #[must_use]
    pub fn suggest_1d_workgroup_size(&self, _element_count: u32) -> u32 {
        // Use the smaller of max_x and max_invocations, rounded down to power-of-2-like values
        let cap = self
            .max_workgroup_size_x
            .min(self.max_workgroup_invocations);
        cap.min(256) // 256 is a common sweet spot
    }

    /// Returns the maximum dispatch size as a tuple (x, y, z).
    #[must_use]
    pub fn max_dispatch(&self) -> (u32, u32, u32) {
        (
            self.max_dispatch_x,
            self.max_dispatch_y,
            self.max_dispatch_z,
        )
    }
}

impl Default for ComputeLimits {
    fn default() -> Self {
        // Vulkan minimum guaranteed limits
        Self {
            max_workgroup_size_x: 128,
            max_workgroup_size_y: 128,
            max_workgroup_size_z: 64,
            max_workgroup_invocations: 128,
            max_dispatch_x: 65535,
            max_dispatch_y: 65535,
            max_dispatch_z: 65535,
            max_shared_memory_bytes: 16384,
        }
    }
}

/// GPU vendor identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA Corporation.
    Nvidia,
    /// AMD / ATI Technologies.
    Amd,
    /// Intel Corporation.
    Intel,
    /// Apple (`MoltenVK` layer).
    Apple,
    /// Qualcomm (Adreno).
    Qualcomm,
    /// ARM (Mali).
    Arm,
    /// Unknown vendor.
    Unknown(u32),
}

impl GpuVendor {
    /// Creates a vendor from a PCI vendor ID.
    #[must_use]
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            0x5143 => Self::Qualcomm,
            0x13B5 => Self::Arm,
            other => Self::Unknown(other),
        }
    }

    /// Returns the vendor name as a string.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Qualcomm => "Qualcomm",
            Self::Arm => "ARM",
            Self::Unknown(_) => "Unknown",
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Complete capability report for a GPU device.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name string.
    pub device_name: String,
    /// GPU vendor.
    pub vendor: GpuVendor,
    /// Driver version (vendor-specific encoding).
    pub driver_version: u32,
    /// Vulkan API version supported (major, minor, patch).
    pub api_version: (u32, u32, u32),
    /// Memory information.
    pub memory: DeviceMemoryInfo,
    /// Compute shader limits.
    pub compute_limits: ComputeLimits,
    /// Set of supported features.
    pub features: HashSet<GpuFeature>,
    /// Maximum 2D image/texture dimension.
    pub max_image_2d: u32,
    /// Maximum storage buffer size in bytes.
    pub max_storage_buffer_bytes: u64,
    /// Maximum push constant size in bytes.
    pub max_push_constant_bytes: u32,
}

impl DeviceCapabilities {
    /// Creates a new capabilities report with minimal defaults (useful for CPU fallback).
    #[must_use]
    pub fn cpu_fallback() -> Self {
        Self {
            device_name: "CPU Fallback".to_string(),
            vendor: GpuVendor::Unknown(0),
            driver_version: 0,
            api_version: (0, 0, 0),
            memory: DeviceMemoryInfo {
                device_local_bytes: 0,
                host_visible_bytes: 0,
                max_allocation_bytes: 0,
                heap_count: 0,
            },
            compute_limits: ComputeLimits::default(),
            features: HashSet::new(),
            max_image_2d: 16384,
            max_storage_buffer_bytes: u64::MAX,
            max_push_constant_bytes: 0,
        }
    }

    /// Returns true if the device supports a specific feature.
    #[must_use]
    pub fn has_feature(&self, feature: GpuFeature) -> bool {
        self.features.contains(&feature)
    }

    /// Returns the Vulkan API version as a formatted string.
    #[must_use]
    pub fn api_version_string(&self) -> String {
        format!(
            "{}.{}.{}",
            self.api_version.0, self.api_version.1, self.api_version.2
        )
    }

    /// Returns a capability score (higher is better) for scheduling decisions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn capability_score(&self) -> f64 {
        let mem_score = (self.memory.device_local_bytes as f64).log2().max(0.0);
        let feature_score = self.features.len() as f64 * 10.0;
        let compute_score = f64::from(self.compute_limits.max_workgroup_invocations).log2();
        mem_score + feature_score + compute_score
    }

    /// Returns true if the device meets minimum requirements for a given workload.
    #[must_use]
    pub fn meets_requirements(
        &self,
        min_vram_bytes: u64,
        required_features: &[GpuFeature],
    ) -> bool {
        self.memory.has_minimum_vram(min_vram_bytes)
            && required_features.iter().all(|f| self.has_feature(*f))
    }

    /// Returns a human-readable summary of the device.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} ({}) - VRAM: {:.1} GB, API: {}, Features: {}",
            self.device_name,
            self.vendor,
            self.memory.vram_gb(),
            self.api_version_string(),
            self.features.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_feature_display() {
        assert_eq!(format!("{}", GpuFeature::Float16Compute), "fp16_compute");
        assert_eq!(format!("{}", GpuFeature::SubgroupOps), "subgroup_ops");
    }

    #[test]
    fn test_device_memory_vram_mb() {
        let mem = DeviceMemoryInfo {
            device_local_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            host_visible_bytes: 16 * 1024 * 1024 * 1024,
            max_allocation_bytes: 4 * 1024 * 1024 * 1024,
            heap_count: 2,
        };
        assert!((mem.vram_mb() - 8192.0).abs() < 1.0);
        assert!((mem.vram_gb() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_device_memory_minimum_vram() {
        let mem = DeviceMemoryInfo {
            device_local_bytes: 4 * 1024 * 1024 * 1024,
            host_visible_bytes: 0,
            max_allocation_bytes: 0,
            heap_count: 1,
        };
        assert!(mem.has_minimum_vram(2 * 1024 * 1024 * 1024));
        assert!(!mem.has_minimum_vram(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_compute_limits_default() {
        let limits = ComputeLimits::default();
        assert_eq!(limits.max_workgroup_invocations, 128);
        assert_eq!(limits.max_shared_memory_bytes, 16384);
    }

    #[test]
    fn test_compute_limits_total_workgroup_size() {
        let limits = ComputeLimits {
            max_workgroup_size_x: 1024,
            max_workgroup_size_y: 1024,
            max_workgroup_size_z: 64,
            max_workgroup_invocations: 1024,
            ..ComputeLimits::default()
        };
        // Product would be huge, but capped by max_workgroup_invocations
        assert_eq!(limits.max_total_workgroup_size(), 1024);
    }

    #[test]
    fn test_compute_limits_suggest_1d() {
        let limits = ComputeLimits {
            max_workgroup_size_x: 1024,
            max_workgroup_invocations: 1024,
            ..ComputeLimits::default()
        };
        assert_eq!(limits.suggest_1d_workgroup_size(10000), 256);
    }

    #[test]
    fn test_compute_limits_max_dispatch() {
        let limits = ComputeLimits::default();
        assert_eq!(limits.max_dispatch(), (65535, 65535, 65535));
    }

    #[test]
    fn test_gpu_vendor_from_id() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_vendor_id(0x106B), GpuVendor::Apple);
        assert_eq!(
            GpuVendor::from_vendor_id(0xFFFF),
            GpuVendor::Unknown(0xFFFF)
        );
    }

    #[test]
    fn test_gpu_vendor_name() {
        assert_eq!(GpuVendor::Nvidia.name(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.name(), "AMD");
        assert_eq!(GpuVendor::Unknown(0).name(), "Unknown");
    }

    #[test]
    fn test_device_capabilities_cpu_fallback() {
        let caps = DeviceCapabilities::cpu_fallback();
        assert_eq!(caps.device_name, "CPU Fallback");
        assert!(caps.features.is_empty());
        assert!(!caps.has_feature(GpuFeature::Float16Compute));
    }

    #[test]
    fn test_device_capabilities_has_feature() {
        let mut caps = DeviceCapabilities::cpu_fallback();
        caps.features.insert(GpuFeature::SubgroupOps);
        assert!(caps.has_feature(GpuFeature::SubgroupOps));
        assert!(!caps.has_feature(GpuFeature::Float64Compute));
    }

    #[test]
    fn test_device_capabilities_api_version_string() {
        let mut caps = DeviceCapabilities::cpu_fallback();
        caps.api_version = (1, 3, 250);
        assert_eq!(caps.api_version_string(), "1.3.250");
    }

    #[test]
    fn test_device_capabilities_meets_requirements() {
        let mut caps = DeviceCapabilities::cpu_fallback();
        caps.memory.device_local_bytes = 4 * 1024 * 1024 * 1024;
        caps.features.insert(GpuFeature::Float16Compute);
        caps.features.insert(GpuFeature::SubgroupOps);

        assert!(caps.meets_requirements(2 * 1024 * 1024 * 1024, &[GpuFeature::Float16Compute]));
        assert!(!caps.meets_requirements(8 * 1024 * 1024 * 1024, &[GpuFeature::Float16Compute]));
        assert!(!caps.meets_requirements(0, &[GpuFeature::CooperativeMatrix]));
    }

    #[test]
    fn test_device_capabilities_summary() {
        let mut caps = DeviceCapabilities::cpu_fallback();
        caps.device_name = "Test GPU".to_string();
        caps.vendor = GpuVendor::Nvidia;
        caps.api_version = (1, 3, 0);
        caps.memory.device_local_bytes = 8 * 1024 * 1024 * 1024;
        let summary = caps.summary();
        assert!(summary.contains("Test GPU"));
        assert!(summary.contains("NVIDIA"));
    }

    #[test]
    fn test_device_capabilities_capability_score() {
        let mut caps = DeviceCapabilities::cpu_fallback();
        caps.memory.device_local_bytes = 8 * 1024 * 1024 * 1024;
        caps.features.insert(GpuFeature::Float16Compute);
        let score = caps.capability_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(format!("{}", GpuVendor::Intel), "Intel");
        assert_eq!(format!("{}", GpuVendor::Unknown(42)), "Unknown");
    }
}
