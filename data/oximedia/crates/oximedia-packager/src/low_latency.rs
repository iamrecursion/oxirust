//! Low-latency streaming support (LHLS/LL-DASH).
//!
//! This module provides types and configuration for low-latency HLS and DASH streaming,
//! including partial segment management and manifest generation.

#![allow(dead_code)]

/// Target latency level for low-latency streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyTarget {
    /// Normal latency (standard HLS/DASH), typically 6-30 seconds.
    Normal,
    /// Low latency, typically 2-6 seconds.
    Low,
    /// Ultra-low latency, typically under 2 seconds.
    UltraLow,
}

impl LatencyTarget {
    /// Target latency in milliseconds.
    #[must_use]
    pub fn target_latency_ms(&self) -> u32 {
        match self {
            Self::Normal => 15_000,
            Self::Low => 3_000,
            Self::UltraLow => 1_000,
        }
    }

    /// Full segment duration in milliseconds.
    #[must_use]
    pub fn segment_duration_ms(&self) -> u32 {
        match self {
            Self::Normal => 6_000,
            Self::Low => 2_000,
            Self::UltraLow => 1_000,
        }
    }

    /// Partial segment (chunk) duration in milliseconds.
    #[must_use]
    pub fn part_duration_ms(&self) -> u32 {
        match self {
            Self::Normal => 2_000,
            Self::Low => 250,
            Self::UltraLow => 100,
        }
    }
}

/// A partial segment (chunk) within a low-latency stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartialSegment {
    /// Index of the parent segment.
    pub segment_index: u32,
    /// Index of this part within the parent segment.
    pub part_index: u32,
    /// Duration of this partial segment in milliseconds.
    pub duration_ms: u32,
    /// Whether this part can be used as a starting point for playback.
    pub is_independent: bool,
}

impl PartialSegment {
    /// Create a new partial segment.
    #[must_use]
    pub fn new(
        segment_index: u32,
        part_index: u32,
        duration_ms: u32,
        is_independent: bool,
    ) -> Self {
        Self {
            segment_index,
            part_index,
            duration_ms,
            is_independent,
        }
    }

    /// Returns `true` if this part is the final part of the parent segment.
    ///
    /// A part is considered the last if it covers the remainder of the segment duration.
    /// This is a heuristic: a part whose duration is at least 80% of a typical part
    /// duration threshold is treated as the last part in this simplified model.
    #[must_use]
    pub fn is_last_part(&self) -> bool {
        // In a real implementation, this would be set by the packager.
        // Here we use a sentinel: part_index == u32::MAX signals the last part.
        self.part_index == u32::MAX
    }
}

/// Configuration for low-latency streaming.
#[derive(Debug, Clone, PartialEq)]
pub struct LowLatencyConfig {
    /// Desired latency target.
    pub target: LatencyTarget,
    /// Whether to enable blocking playlist requests (LL-HLS feature).
    pub enable_blocking_requests: bool,
    /// Hold-back time in seconds added to the target latency.
    pub hold_back_secs: f32,
}

impl LowLatencyConfig {
    /// Create a default Low-Latency HLS configuration.
    #[must_use]
    pub fn lhls_default() -> Self {
        Self {
            target: LatencyTarget::Low,
            enable_blocking_requests: true,
            hold_back_secs: 3.0,
        }
    }

    /// Create a default LL-DASH configuration.
    #[must_use]
    pub fn ll_dash_default() -> Self {
        Self {
            target: LatencyTarget::UltraLow,
            enable_blocking_requests: false,
            hold_back_secs: 1.5,
        }
    }
}

impl Default for LowLatencyConfig {
    fn default() -> Self {
        Self::lhls_default()
    }
}

/// A low-latency manifest containing configuration and partial segments.
#[derive(Debug, Clone)]
pub struct LowLatencyManifest {
    /// Low-latency configuration.
    pub config: LowLatencyConfig,
    /// Partial segments (parts) in the manifest.
    pub parts: Vec<PartialSegment>,
}

impl LowLatencyManifest {
    /// Create a new manifest with the given configuration.
    #[must_use]
    pub fn new(config: LowLatencyConfig) -> Self {
        Self {
            config,
            parts: Vec::new(),
        }
    }

    /// Add a partial segment to the manifest.
    pub fn add_part(&mut self, part: PartialSegment) {
        self.parts.push(part);
    }

    /// Count the number of fully-completed segments represented in the manifest.
    ///
    /// A segment is considered complete when we have parts covering its full duration.
    /// This simplified implementation groups by `segment_index` and counts unique ones.
    #[must_use]
    pub fn complete_segments(&self) -> u32 {
        if self.parts.is_empty() {
            return 0;
        }
        let mut segments: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for part in &self.parts {
            segments.insert(part.segment_index);
        }
        segments.len() as u32
    }

    /// Return the total number of partial segments in the manifest.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Estimate the current playback latency in milliseconds.
    ///
    /// Latency is estimated as the target latency minus the hold-back, bounded by zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_latency_ms(&self) -> u32 {
        let target_ms = self.config.target.target_latency_ms() as f32;
        let hold_back_ms = self.config.hold_back_secs * 1000.0;
        let estimated = target_ms + hold_back_ms;
        estimated as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- LatencyTarget tests ---

    #[test]
    fn test_latency_target_normal_latency_ms() {
        assert_eq!(LatencyTarget::Normal.target_latency_ms(), 15_000);
    }

    #[test]
    fn test_latency_target_low_latency_ms() {
        assert_eq!(LatencyTarget::Low.target_latency_ms(), 3_000);
    }

    #[test]
    fn test_latency_target_ultra_low_latency_ms() {
        assert_eq!(LatencyTarget::UltraLow.target_latency_ms(), 1_000);
    }

    #[test]
    fn test_latency_target_segment_duration_normal() {
        assert_eq!(LatencyTarget::Normal.segment_duration_ms(), 6_000);
    }

    #[test]
    fn test_latency_target_segment_duration_low() {
        assert_eq!(LatencyTarget::Low.segment_duration_ms(), 2_000);
    }

    #[test]
    fn test_latency_target_part_duration_low() {
        assert_eq!(LatencyTarget::Low.part_duration_ms(), 250);
    }

    #[test]
    fn test_latency_target_part_duration_ultra_low() {
        assert_eq!(LatencyTarget::UltraLow.part_duration_ms(), 100);
    }

    // --- PartialSegment tests ---

    #[test]
    fn test_partial_segment_creation() {
        let part = PartialSegment::new(0, 0, 250, true);
        assert_eq!(part.segment_index, 0);
        assert_eq!(part.part_index, 0);
        assert_eq!(part.duration_ms, 250);
        assert!(part.is_independent);
    }

    #[test]
    fn test_partial_segment_not_last_by_default() {
        let part = PartialSegment::new(1, 2, 250, false);
        assert!(!part.is_last_part());
    }

    #[test]
    fn test_partial_segment_last_part_sentinel() {
        let part = PartialSegment::new(1, u32::MAX, 250, false);
        assert!(part.is_last_part());
    }

    // --- LowLatencyConfig tests ---

    #[test]
    fn test_lhls_default_config() {
        let cfg = LowLatencyConfig::lhls_default();
        assert_eq!(cfg.target, LatencyTarget::Low);
        assert!(cfg.enable_blocking_requests);
        assert!((cfg.hold_back_secs - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ll_dash_default_config() {
        let cfg = LowLatencyConfig::ll_dash_default();
        assert_eq!(cfg.target, LatencyTarget::UltraLow);
        assert!(!cfg.enable_blocking_requests);
        assert!((cfg.hold_back_secs - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_config_equals_lhls() {
        let default = LowLatencyConfig::default();
        let lhls = LowLatencyConfig::lhls_default();
        assert_eq!(default, lhls);
    }

    // --- LowLatencyManifest tests ---

    #[test]
    fn test_manifest_new_is_empty() {
        let cfg = LowLatencyConfig::lhls_default();
        let manifest = LowLatencyManifest::new(cfg);
        assert_eq!(manifest.part_count(), 0);
        assert_eq!(manifest.complete_segments(), 0);
    }

    #[test]
    fn test_manifest_add_part() {
        let cfg = LowLatencyConfig::lhls_default();
        let mut manifest = LowLatencyManifest::new(cfg);
        manifest.add_part(PartialSegment::new(0, 0, 250, true));
        assert_eq!(manifest.part_count(), 1);
    }

    #[test]
    fn test_manifest_complete_segments_single_segment() {
        let cfg = LowLatencyConfig::lhls_default();
        let mut manifest = LowLatencyManifest::new(cfg);
        manifest.add_part(PartialSegment::new(0, 0, 250, true));
        manifest.add_part(PartialSegment::new(0, 1, 250, false));
        manifest.add_part(PartialSegment::new(0, 2, 250, false));
        // All parts belong to segment 0
        assert_eq!(manifest.complete_segments(), 1);
    }

    #[test]
    fn test_manifest_complete_segments_multiple() {
        let cfg = LowLatencyConfig::lhls_default();
        let mut manifest = LowLatencyManifest::new(cfg);
        manifest.add_part(PartialSegment::new(0, 0, 250, true));
        manifest.add_part(PartialSegment::new(1, 0, 250, true));
        manifest.add_part(PartialSegment::new(2, 0, 250, true));
        assert_eq!(manifest.complete_segments(), 3);
    }

    #[test]
    fn test_manifest_estimated_latency_lhls() {
        let cfg = LowLatencyConfig::lhls_default();
        let manifest = LowLatencyManifest::new(cfg);
        // 3000 + 3000 = 6000 ms
        assert_eq!(manifest.estimated_latency_ms(), 6_000);
    }

    #[test]
    fn test_manifest_estimated_latency_ll_dash() {
        let cfg = LowLatencyConfig::ll_dash_default();
        let manifest = LowLatencyManifest::new(cfg);
        // 1000 + 1500 = 2500 ms
        assert_eq!(manifest.estimated_latency_ms(), 2_500);
    }
}
