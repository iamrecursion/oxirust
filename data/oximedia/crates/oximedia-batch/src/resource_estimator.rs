#![allow(dead_code)]
//! Resource estimation for batch processing jobs.
//!
//! Before scheduling a job the engine needs to know approximate CPU, memory
//! and disk requirements.  This module provides estimators that inspect job
//! parameters and produce a [`ResourceEstimate`] used by the scheduler.

use std::collections::HashMap;

/// Units for memory sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUnit {
    /// Bytes.
    Bytes,
    /// Kibibytes (1024 bytes).
    KiB,
    /// Mebibytes (1024 KiB).
    MiB,
    /// Gibibytes (1024 MiB).
    GiB,
}

impl MemoryUnit {
    /// Convert a value in this unit to bytes.
    #[must_use]
    pub fn to_bytes(self, value: u64) -> u64 {
        match self {
            Self::Bytes => value,
            Self::KiB => value * 1024,
            Self::MiB => value * 1024 * 1024,
            Self::GiB => value * 1024 * 1024 * 1024,
        }
    }

    /// Convert bytes to this unit (integer division).
    #[must_use]
    pub fn from_bytes(self, bytes: u64) -> u64 {
        match self {
            Self::Bytes => bytes,
            Self::KiB => bytes / 1024,
            Self::MiB => bytes / (1024 * 1024),
            Self::GiB => bytes / (1024 * 1024 * 1024),
        }
    }
}

/// Estimated resources a single job will need.
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Estimated CPU cores (can be fractional, e.g. 0.5 means half a core).
    pub cpu_cores: f64,
    /// Estimated peak memory in bytes.
    pub memory_bytes: u64,
    /// Estimated temporary disk space in bytes.
    pub disk_bytes: u64,
    /// Estimated GPU memory in bytes (0 if no GPU needed).
    pub gpu_memory_bytes: u64,
    /// Estimated wall-clock duration in seconds.
    pub estimated_duration_secs: f64,
    /// Confidence level 0.0..=1.0 (how reliable this estimate is).
    pub confidence: f64,
}

impl Default for ResourceEstimate {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_bytes: 256 * 1024 * 1024, // 256 MiB
            disk_bytes: 0,
            gpu_memory_bytes: 0,
            estimated_duration_secs: 60.0,
            confidence: 0.5,
        }
    }
}

impl ResourceEstimate {
    /// Create a new estimate with all fields specified.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        cpu_cores: f64,
        memory_bytes: u64,
        disk_bytes: u64,
        gpu_memory_bytes: u64,
        estimated_duration_secs: f64,
        confidence: f64,
    ) -> Self {
        Self {
            cpu_cores,
            memory_bytes,
            disk_bytes,
            gpu_memory_bytes,
            estimated_duration_secs,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return memory estimate in the given unit.
    #[must_use]
    pub fn memory_in(&self, unit: MemoryUnit) -> u64 {
        unit.from_bytes(self.memory_bytes)
    }

    /// Return disk estimate in the given unit.
    #[must_use]
    pub fn disk_in(&self, unit: MemoryUnit) -> u64 {
        unit.from_bytes(self.disk_bytes)
    }

    /// Whether this job needs a GPU.
    #[must_use]
    pub fn needs_gpu(&self) -> bool {
        self.gpu_memory_bytes > 0
    }

    /// Merge with another estimate by taking the maximum of each field.
    #[must_use]
    pub fn merge_max(&self, other: &Self) -> Self {
        Self {
            cpu_cores: self.cpu_cores.max(other.cpu_cores),
            memory_bytes: self.memory_bytes.max(other.memory_bytes),
            disk_bytes: self.disk_bytes.max(other.disk_bytes),
            gpu_memory_bytes: self.gpu_memory_bytes.max(other.gpu_memory_bytes),
            estimated_duration_secs: self
                .estimated_duration_secs
                .max(other.estimated_duration_secs),
            confidence: self.confidence.min(other.confidence),
        }
    }

    /// Sum two estimates (for parallel job scheduling headroom).
    #[must_use]
    pub fn sum(&self, other: &Self) -> Self {
        Self {
            cpu_cores: self.cpu_cores + other.cpu_cores,
            memory_bytes: self.memory_bytes + other.memory_bytes,
            disk_bytes: self.disk_bytes + other.disk_bytes,
            gpu_memory_bytes: self.gpu_memory_bytes + other.gpu_memory_bytes,
            estimated_duration_secs: self
                .estimated_duration_secs
                .max(other.estimated_duration_secs),
            confidence: self.confidence.min(other.confidence),
        }
    }

    /// Apply a safety margin multiplier to memory and disk estimates.
    #[must_use]
    pub fn with_safety_margin(mut self, factor: f64) -> Self {
        let factor = factor.max(1.0);
        #[allow(clippy::cast_precision_loss)]
        {
            self.memory_bytes = (self.memory_bytes as f64 * factor) as u64;
            self.disk_bytes = (self.disk_bytes as f64 * factor) as u64;
            self.gpu_memory_bytes = (self.gpu_memory_bytes as f64 * factor) as u64;
        }
        self
    }
}

/// Codec identifier used for estimation lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CodecProfile {
    /// H.264 / AVC encode.
    H264,
    /// H.265 / HEVC encode.
    H265,
    /// AV1 encode.
    Av1,
    /// `ProRes` encode.
    ProRes,
    /// `DNxHD` / `DNxHR` encode.
    DnxHd,
    /// Copy (no transcode).
    Copy,
    /// Custom codec identifier.
    Custom(String),
}

/// Resolution tier for estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolutionTier {
    /// SD (up to 720x576).
    Sd,
    /// HD (up to 1920x1080).
    Hd,
    /// UHD / 4K (up to 3840x2160).
    Uhd4K,
    /// 8K (up to 7680x4320).
    Uhd8K,
}

/// The resource estimator applies heuristic rules to predict job costs.
#[derive(Debug, Clone)]
pub struct ResourceEstimator {
    /// Per-codec base memory cost in bytes.
    codec_memory: HashMap<CodecProfile, u64>,
    /// Per-resolution multiplier applied to base memory.
    resolution_multiplier: HashMap<ResolutionTier, f64>,
    /// Global safety margin factor (>= 1.0).
    safety_margin: f64,
}

impl Default for ResourceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceEstimator {
    /// Create an estimator with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        let mut codec_memory = HashMap::new();
        codec_memory.insert(CodecProfile::H264, 512 * 1024 * 1024); // 512 MiB
        codec_memory.insert(CodecProfile::H265, 768 * 1024 * 1024); // 768 MiB
        codec_memory.insert(CodecProfile::Av1, 1024 * 1024 * 1024); // 1 GiB
        codec_memory.insert(CodecProfile::ProRes, 256 * 1024 * 1024); // 256 MiB
        codec_memory.insert(CodecProfile::DnxHd, 256 * 1024 * 1024); // 256 MiB
        codec_memory.insert(CodecProfile::Copy, 64 * 1024 * 1024); // 64 MiB

        let mut resolution_multiplier = HashMap::new();
        resolution_multiplier.insert(ResolutionTier::Sd, 1.0);
        resolution_multiplier.insert(ResolutionTier::Hd, 2.0);
        resolution_multiplier.insert(ResolutionTier::Uhd4K, 6.0);
        resolution_multiplier.insert(ResolutionTier::Uhd8K, 16.0);

        Self {
            codec_memory,
            resolution_multiplier,
            safety_margin: 1.2,
        }
    }

    /// Set the global safety margin (clamped to >= 1.0).
    pub fn set_safety_margin(&mut self, margin: f64) {
        self.safety_margin = margin.max(1.0);
    }

    /// Return the current safety margin.
    #[must_use]
    pub fn safety_margin(&self) -> f64 {
        self.safety_margin
    }

    /// Override the base memory for a codec profile.
    pub fn set_codec_memory(&mut self, codec: CodecProfile, bytes: u64) {
        self.codec_memory.insert(codec, bytes);
    }

    /// Estimate resources for a transcode job.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_transcode(
        &self,
        codec: &CodecProfile,
        resolution: ResolutionTier,
        duration_secs: f64,
        thread_count: u32,
    ) -> ResourceEstimate {
        let base_mem = self
            .codec_memory
            .get(codec)
            .copied()
            .unwrap_or(512 * 1024 * 1024);
        let res_mult = self
            .resolution_multiplier
            .get(&resolution)
            .copied()
            .unwrap_or(1.0);

        let memory = (base_mem as f64 * res_mult) as u64;
        let disk = (duration_secs * 5.0 * 1024.0 * 1024.0 * res_mult) as u64; // ~5 MB/s * res

        let speed_factor = match codec {
            CodecProfile::Copy => 100.0,
            CodecProfile::ProRes | CodecProfile::DnxHd => 2.0,
            CodecProfile::H264 => 1.0,
            CodecProfile::H265 => 0.5,
            CodecProfile::Av1 => 0.25,
            CodecProfile::Custom(_) => 1.0,
        };
        let est_duration = if speed_factor > 0.0 {
            duration_secs / speed_factor
        } else {
            duration_secs
        };

        ResourceEstimate::new(f64::from(thread_count), memory, disk, 0, est_duration, 0.7)
            .with_safety_margin(self.safety_margin)
    }

    /// Estimate resources for a simple file copy/move.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_file_op(&self, file_size_bytes: u64) -> ResourceEstimate {
        let duration = file_size_bytes as f64 / (200.0 * 1024.0 * 1024.0); // ~200 MB/s
        ResourceEstimate::new(0.1, 64 * 1024 * 1024, file_size_bytes, 0, duration, 0.9)
    }

    /// Aggregate estimates for a batch of jobs.
    #[must_use]
    pub fn aggregate(estimates: &[ResourceEstimate]) -> ResourceEstimate {
        if estimates.is_empty() {
            return ResourceEstimate::default();
        }
        let mut total = estimates[0].clone();
        for est in &estimates[1..] {
            total = total.sum(est);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_unit_to_bytes() {
        assert_eq!(MemoryUnit::Bytes.to_bytes(100), 100);
        assert_eq!(MemoryUnit::KiB.to_bytes(1), 1024);
        assert_eq!(MemoryUnit::MiB.to_bytes(1), 1_048_576);
        assert_eq!(MemoryUnit::GiB.to_bytes(1), 1_073_741_824);
    }

    #[test]
    fn test_memory_unit_from_bytes() {
        assert_eq!(MemoryUnit::KiB.from_bytes(2048), 2);
        assert_eq!(MemoryUnit::MiB.from_bytes(1_048_576), 1);
        assert_eq!(MemoryUnit::GiB.from_bytes(2_147_483_648), 2);
    }

    #[test]
    fn test_resource_estimate_default() {
        let est = ResourceEstimate::default();
        assert!((est.cpu_cores - 1.0).abs() < f64::EPSILON);
        assert_eq!(est.memory_bytes, 256 * 1024 * 1024);
        assert!(!est.needs_gpu());
    }

    #[test]
    fn test_resource_estimate_needs_gpu() {
        let est = ResourceEstimate::new(4.0, 1024, 0, 512 * 1024 * 1024, 30.0, 0.8);
        assert!(est.needs_gpu());
    }

    #[test]
    fn test_resource_estimate_memory_in() {
        let est = ResourceEstimate::new(1.0, 2 * 1024 * 1024 * 1024, 0, 0, 10.0, 0.9);
        assert_eq!(est.memory_in(MemoryUnit::GiB), 2);
    }

    #[test]
    fn test_resource_estimate_merge_max() {
        let a = ResourceEstimate::new(2.0, 1000, 500, 0, 10.0, 0.8);
        let b = ResourceEstimate::new(4.0, 800, 1000, 256, 20.0, 0.6);
        let merged = a.merge_max(&b);
        assert!((merged.cpu_cores - 4.0).abs() < f64::EPSILON);
        assert_eq!(merged.memory_bytes, 1000);
        assert_eq!(merged.disk_bytes, 1000);
        assert_eq!(merged.gpu_memory_bytes, 256);
        assert!((merged.confidence - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_estimate_sum() {
        let a = ResourceEstimate::new(2.0, 1000, 500, 0, 10.0, 0.8);
        let b = ResourceEstimate::new(2.0, 1000, 500, 0, 5.0, 0.9);
        let total = a.sum(&b);
        assert!((total.cpu_cores - 4.0).abs() < f64::EPSILON);
        assert_eq!(total.memory_bytes, 2000);
        assert_eq!(total.disk_bytes, 1000);
        // Duration takes max
        assert!((total.estimated_duration_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_estimate_safety_margin() {
        let est = ResourceEstimate::new(1.0, 1000, 2000, 500, 10.0, 0.5);
        let safe = est.with_safety_margin(2.0);
        assert_eq!(safe.memory_bytes, 2000);
        assert_eq!(safe.disk_bytes, 4000);
        assert_eq!(safe.gpu_memory_bytes, 1000);
    }

    #[test]
    fn test_estimator_transcode_h264_hd() {
        let estimator = ResourceEstimator::new();
        let est = estimator.estimate_transcode(&CodecProfile::H264, ResolutionTier::Hd, 60.0, 4);
        assert!(est.cpu_cores >= 4.0);
        assert!(est.memory_bytes > 0);
        assert!(est.disk_bytes > 0);
    }

    #[test]
    fn test_estimator_transcode_av1_4k() {
        let estimator = ResourceEstimator::new();
        let est = estimator.estimate_transcode(&CodecProfile::Av1, ResolutionTier::Uhd4K, 120.0, 8);
        // AV1 should take longer than H264
        let h264_est =
            estimator.estimate_transcode(&CodecProfile::H264, ResolutionTier::Uhd4K, 120.0, 8);
        assert!(est.estimated_duration_secs > h264_est.estimated_duration_secs);
    }

    #[test]
    fn test_estimator_file_op() {
        let estimator = ResourceEstimator::new();
        let est = estimator.estimate_file_op(1_073_741_824); // 1 GiB
        assert!(est.estimated_duration_secs > 0.0);
        assert_eq!(est.disk_bytes, 1_073_741_824);
    }

    #[test]
    fn test_estimator_aggregate() {
        let a = ResourceEstimate::new(2.0, 1000, 500, 0, 10.0, 0.8);
        let b = ResourceEstimate::new(2.0, 2000, 1000, 0, 20.0, 0.7);
        let agg = ResourceEstimator::aggregate(&[a, b]);
        assert!((agg.cpu_cores - 4.0).abs() < f64::EPSILON);
        assert_eq!(agg.memory_bytes, 3000);
    }

    #[test]
    fn test_estimator_aggregate_empty() {
        let agg = ResourceEstimator::aggregate(&[]);
        assert!((agg.cpu_cores - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimator_set_safety_margin() {
        let mut estimator = ResourceEstimator::new();
        estimator.set_safety_margin(1.5);
        assert!((estimator.safety_margin() - 1.5).abs() < f64::EPSILON);
        estimator.set_safety_margin(0.5); // should clamp to 1.0
        assert!((estimator.safety_margin() - 1.0).abs() < f64::EPSILON);
    }
}
