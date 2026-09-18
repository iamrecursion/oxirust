#![allow(dead_code)]
//! GPU resource tracking for render farm nodes.
//!
//! Provides GPU discovery, VRAM allocation tracking, and thermal monitoring
//! across render workers. Detection is best-effort: on platforms without
//! accessible GPU drivers the tracker still works with manually-registered
//! [`GpuInfo`] entries.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// ── GpuVendor ───────────────────────────────────────────────────────────────

/// GPU hardware vendor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuVendor {
    /// NVIDIA Corporation
    Nvidia,
    /// Advanced Micro Devices
    Amd,
    /// Intel Corporation
    Intel,
    /// Apple Silicon
    Apple,
    /// Unknown or unrecognised vendor
    Unknown,
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "NVIDIA"),
            Self::Amd => write!(f, "AMD"),
            Self::Intel => write!(f, "Intel"),
            Self::Apple => write!(f, "Apple"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── GpuInfo ─────────────────────────────────────────────────────────────────

/// Information about a single GPU device on a render node.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Zero-based device index.
    pub index: u32,
    /// Hardware vendor.
    pub vendor: GpuVendor,
    /// Marketing model name (e.g. "RTX 4090").
    pub model: String,
    /// Total video RAM in mebibytes.
    pub vram_mb: u64,
    /// Current die temperature in °C, if readable.
    pub temperature_celsius: Option<f32>,
    /// GPU utilisation percentage (0–100), if readable.
    pub utilization_percent: Option<f32>,
}

impl GpuInfo {
    /// Create a new [`GpuInfo`] with the minimum required fields.
    #[must_use]
    pub fn new(index: u32, vendor: GpuVendor, model: impl Into<String>, vram_mb: u64) -> Self {
        Self {
            index,
            vendor,
            model: model.into(),
            vram_mb,
            temperature_celsius: None,
            utilization_percent: None,
        }
    }

    /// Attach a temperature reading and return `self` (builder style).
    #[must_use]
    pub fn with_temperature(mut self, temp_celsius: f32) -> Self {
        self.temperature_celsius = Some(temp_celsius);
        self
    }

    /// Attach a utilisation reading and return `self` (builder style).
    #[must_use]
    pub fn with_utilization(mut self, utilization_percent: f32) -> Self {
        self.utilization_percent = Some(utilization_percent.clamp(0.0, 100.0));
        self
    }

    /// Returns `true` when the reported temperature exceeds 85 °C.
    ///
    /// Returns `false` if no temperature data is available.
    #[must_use]
    pub fn is_overheating(&self) -> bool {
        self.temperature_celsius.map(|t| t > 85.0).unwrap_or(false)
    }

    /// VRAM expressed as gibibytes (convenient for display).
    #[must_use]
    pub fn vram_gb(&self) -> f64 {
        self.vram_mb as f64 / 1024.0
    }

    /// Returns `true` if this GPU is from NVIDIA (useful for CUDA checks).
    #[must_use]
    pub fn is_nvidia(&self) -> bool {
        self.vendor == GpuVendor::Nvidia
    }
}

// ── GpuResourceTracker ──────────────────────────────────────────────────────

/// Tracks VRAM allocation across render jobs on this node.
///
/// The tracker maintains a registry of [`GpuInfo`] entries and maps
/// job identifiers to the number of VRAM mebibytes they have reserved.
/// Allocation is checked against total available VRAM to prevent
/// over-subscription.
#[derive(Debug, Default)]
pub struct GpuResourceTracker {
    gpus: Arc<RwLock<Vec<GpuInfo>>>,
    /// Maps `job_id` → allocated VRAM in MiB.
    job_vram: Arc<RwLock<HashMap<String, u64>>>,
}

impl GpuResourceTracker {
    /// Create an empty tracker with no GPUs registered.
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpus: Arc::new(RwLock::new(Vec::new())),
            job_vram: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a GPU device with this tracker.
    pub fn register_gpu(&self, gpu: GpuInfo) {
        self.gpus.write().push(gpu);
    }

    /// Number of registered GPU devices.
    #[must_use]
    pub fn gpu_count(&self) -> usize {
        self.gpus.read().len()
    }

    /// Sum of VRAM across all registered GPUs (MiB).
    #[must_use]
    pub fn total_vram_mb(&self) -> u64 {
        self.gpus.read().iter().map(|g| g.vram_mb).sum()
    }

    /// Attempt to reserve `vram_mb` MiB of VRAM for `job_id`.
    ///
    /// Returns `true` if the reservation succeeded (sufficient free VRAM was
    /// available) or `false` if the request would exceed available capacity.
    /// Repeated calls for the same `job_id` replace the previous allocation.
    pub fn allocate_vram(&self, job_id: &str, vram_mb: u64) -> bool {
        let total = self.total_vram_mb();
        let current_for_job = self.job_vram.read().get(job_id).copied().unwrap_or(0);
        let other_allocated: u64 = self
            .job_vram
            .read()
            .values()
            .sum::<u64>()
            .saturating_sub(current_for_job);

        if other_allocated + vram_mb > total {
            return false;
        }

        self.job_vram.write().insert(job_id.to_string(), vram_mb);
        true
    }

    /// Release VRAM allocation for a finished or cancelled job.
    pub fn release_vram(&self, job_id: &str) {
        self.job_vram.write().remove(job_id);
    }

    /// Total VRAM currently allocated across all tracked jobs (MiB).
    #[must_use]
    pub fn allocated_vram_mb(&self) -> u64 {
        self.job_vram.read().values().sum()
    }

    /// VRAM not yet allocated (MiB). Saturates at zero.
    #[must_use]
    pub fn available_vram_mb(&self) -> u64 {
        self.total_vram_mb()
            .saturating_sub(self.allocated_vram_mb())
    }

    /// Returns a snapshot of all registered [`GpuInfo`] entries.
    #[must_use]
    pub fn get_gpus(&self) -> Vec<GpuInfo> {
        self.gpus.read().clone()
    }

    /// Returns GPUs whose temperature exceeds 85 °C.
    #[must_use]
    pub fn overheating_gpus(&self) -> Vec<GpuInfo> {
        self.gpus
            .read()
            .iter()
            .filter(|g| g.is_overheating())
            .cloned()
            .collect()
    }

    /// Update temperature and utilisation for the GPU at `index`.
    ///
    /// No-op if no GPU with that index is registered.
    pub fn update_thermal(&self, index: u32, temp_celsius: f32, util_percent: f32) {
        for gpu in self.gpus.write().iter_mut() {
            if gpu.index == index {
                gpu.temperature_celsius = Some(temp_celsius);
                gpu.utilization_percent = Some(util_percent.clamp(0.0, 100.0));
                break;
            }
        }
    }

    /// Attempt to detect GPUs via the `sysinfo` crate (best-effort).
    ///
    /// `sysinfo` 0.36 does not expose a GPU enumeration API, so this method
    /// currently returns an empty [`Vec`]. Real detection would parse
    /// `/proc/driver/nvidia/gpus/` on Linux, query `IOKit` on macOS, or call
    /// the NVML / ROCm SMI libraries. This stub exists so callers can call
    /// `detect_gpus_sysinfo()` and register the results without code changes
    /// once the underlying library gains GPU support.
    #[must_use]
    pub fn detect_gpus_sysinfo() -> Vec<GpuInfo> {
        // sysinfo 0.36 does not enumerate GPUs; return empty list.
        // Future: use sysinfo::System::components() for thermal sensors and
        // extend once the crate exposes a GPU list.
        vec![]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── GpuInfo tests ────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_info_construction() {
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "RTX 4090", 24576);
        assert_eq!(gpu.index, 0);
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert_eq!(gpu.model, "RTX 4090");
        assert_eq!(gpu.vram_mb, 24576);
        assert!(gpu.temperature_celsius.is_none());
        assert!(gpu.utilization_percent.is_none());
    }

    #[test]
    fn test_gpu_info_vram_gb() {
        let gpu = GpuInfo::new(0, GpuVendor::Amd, "RX 7900 XTX", 24576);
        let gb = gpu.vram_gb();
        assert!((gb - 24.0).abs() < 0.1, "expected ~24 GiB, got {gb}");
    }

    #[test]
    fn test_gpu_info_vram_gb_small() {
        let gpu = GpuInfo::new(1, GpuVendor::Intel, "Arc A770", 16 * 1024);
        assert!((gpu.vram_gb() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_is_overheating_below_threshold() {
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "GPU", 8192).with_temperature(80.0);
        assert!(!gpu.is_overheating());
    }

    #[test]
    fn test_is_overheating_at_threshold() {
        // Exactly 85 °C is NOT overheating (threshold is > 85)
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "GPU", 8192).with_temperature(85.0);
        assert!(!gpu.is_overheating());
    }

    #[test]
    fn test_is_overheating_above_threshold() {
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "GPU", 8192).with_temperature(90.0);
        assert!(gpu.is_overheating());
    }

    #[test]
    fn test_is_overheating_no_temp() {
        let gpu = GpuInfo::new(0, GpuVendor::Unknown, "GPU", 4096);
        // No temperature data → not overheating
        assert!(!gpu.is_overheating());
    }

    #[test]
    fn test_with_utilization_clamps() {
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "GPU", 8192).with_utilization(150.0);
        assert_eq!(gpu.utilization_percent, Some(100.0));
    }

    #[test]
    fn test_is_nvidia() {
        let gpu = GpuInfo::new(0, GpuVendor::Nvidia, "A100", 40960);
        assert!(gpu.is_nvidia());
        let amd = GpuInfo::new(1, GpuVendor::Amd, "MI250", 65536);
        assert!(!amd.is_nvidia());
    }

    // ── GpuResourceTracker tests ─────────────────────────────────────────────

    #[test]
    fn test_tracker_empty_on_new() {
        let tracker = GpuResourceTracker::new();
        assert_eq!(tracker.gpu_count(), 0);
        assert_eq!(tracker.total_vram_mb(), 0);
        assert_eq!(tracker.available_vram_mb(), 0);
    }

    #[test]
    fn test_tracker_register_gpu() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "A100", 40960));
        assert_eq!(tracker.gpu_count(), 1);
        assert_eq!(tracker.total_vram_mb(), 40960);
    }

    #[test]
    fn test_tracker_total_vram_multiple_gpus() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "A100", 40960));
        tracker.register_gpu(GpuInfo::new(1, GpuVendor::Nvidia, "A100", 40960));
        assert_eq!(tracker.total_vram_mb(), 81920);
    }

    #[test]
    fn test_tracker_allocate_vram_success() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "RTX 4090", 24576));
        assert!(tracker.allocate_vram("job-1", 8192));
        assert_eq!(tracker.allocated_vram_mb(), 8192);
        assert_eq!(tracker.available_vram_mb(), 24576 - 8192);
    }

    #[test]
    fn test_tracker_allocate_vram_failure_over_capacity() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "RTX 3090", 24576));
        assert!(!tracker.allocate_vram("job-big", 30000));
        assert_eq!(tracker.allocated_vram_mb(), 0);
    }

    #[test]
    fn test_tracker_release_vram_restores_available() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "RTX 4090", 24576));
        tracker.allocate_vram("job-1", 12288);
        assert_eq!(tracker.available_vram_mb(), 24576 - 12288);
        tracker.release_vram("job-1");
        assert_eq!(tracker.available_vram_mb(), 24576);
    }

    #[test]
    fn test_tracker_overheating_gpus_filter() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(
            GpuInfo::new(0, GpuVendor::Nvidia, "Hot GPU", 8192).with_temperature(92.0),
        );
        tracker.register_gpu(
            GpuInfo::new(1, GpuVendor::Nvidia, "Cool GPU", 8192).with_temperature(65.0),
        );
        let hot = tracker.overheating_gpus();
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].model, "Hot GPU");
    }

    #[test]
    fn test_tracker_detect_gpus_sysinfo_returns_vec() {
        // May be empty on CI/test machines — just confirm it doesn't panic
        let gpus = GpuResourceTracker::detect_gpus_sysinfo();
        // The stub always returns empty; confirm the return type is correct
        let _ = gpus.len();
    }

    #[test]
    fn test_tracker_update_thermal() {
        let tracker = GpuResourceTracker::new();
        tracker.register_gpu(GpuInfo::new(0, GpuVendor::Nvidia, "GPU", 8192));
        tracker.update_thermal(0, 75.0, 80.0);
        let gpus = tracker.get_gpus();
        assert_eq!(gpus[0].temperature_celsius, Some(75.0));
        assert_eq!(gpus[0].utilization_percent, Some(80.0));
    }

    #[test]
    fn test_vendor_display() {
        assert_eq!(GpuVendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.to_string(), "AMD");
        assert_eq!(GpuVendor::Unknown.to_string(), "Unknown");
    }
}
