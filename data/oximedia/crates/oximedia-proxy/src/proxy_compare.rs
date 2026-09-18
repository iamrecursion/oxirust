#![allow(dead_code)]

//! Proxy vs original quality comparison and validation metrics.
//!
//! This module provides tools for comparing proxy files against their
//! original source media. It computes resolution ratios, bitrate ratios,
//! frame rate matches, and generates comparison reports used during
//! quality-control workflows.

use std::collections::HashMap;

/// Resolution of a media file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Total number of pixels.
    pub const fn pixel_count(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Compute the ratio of this resolution's pixel count to another's.
    #[allow(clippy::cast_precision_loss)]
    pub fn ratio_to(&self, other: &Resolution) -> f64 {
        if other.pixel_count() == 0 {
            return 0.0;
        }
        self.pixel_count() as f64 / other.pixel_count() as f64
    }
}

impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Metadata for a media file used in comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaInfo {
    /// File path.
    pub path: String,
    /// Resolution.
    pub resolution: Resolution,
    /// Bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Codec name.
    pub codec: String,
    /// File size in bytes.
    pub file_size_bytes: u64,
}

impl MediaInfo {
    /// Create new media info.
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            resolution: Resolution::new(0, 0),
            bitrate_bps: 0,
            frame_rate: 0.0,
            duration_ms: 0,
            codec: String::new(),
            file_size_bytes: 0,
        }
    }

    /// Set the resolution.
    pub fn with_resolution(mut self, w: u32, h: u32) -> Self {
        self.resolution = Resolution::new(w, h);
        self
    }

    /// Set the bitrate.
    pub fn with_bitrate(mut self, bps: u64) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set the frame rate.
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.frame_rate = fps;
        self
    }

    /// Set the duration.
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Set the codec name.
    pub fn with_codec(mut self, codec: &str) -> Self {
        self.codec = codec.to_string();
        self
    }

    /// Set the file size.
    pub fn with_file_size(mut self, bytes: u64) -> Self {
        self.file_size_bytes = bytes;
        self
    }
}

/// Result of comparing a proxy against its original.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Proxy media info.
    pub proxy: MediaInfo,
    /// Original media info.
    pub original: MediaInfo,
    /// Resolution ratio (proxy pixels / original pixels).
    pub resolution_ratio: f64,
    /// Bitrate ratio (proxy / original).
    pub bitrate_ratio: f64,
    /// Whether frame rates match.
    pub frame_rate_match: bool,
    /// Whether durations are within tolerance.
    pub duration_match: bool,
    /// File size reduction ratio (proxy size / original size).
    pub size_ratio: f64,
    /// Duration difference in milliseconds.
    pub duration_diff_ms: i64,
}

/// Tolerance settings for proxy comparison.
#[derive(Debug, Clone)]
pub struct ComparisonTolerance {
    /// Maximum allowed frame rate difference (fps).
    pub frame_rate_tolerance: f64,
    /// Maximum allowed duration difference (ms).
    pub duration_tolerance_ms: u64,
    /// Maximum allowed resolution ratio (e.g., 0.5 for half-res).
    pub max_resolution_ratio: f64,
    /// Minimum acceptable resolution ratio.
    pub min_resolution_ratio: f64,
}

impl Default for ComparisonTolerance {
    fn default() -> Self {
        Self {
            frame_rate_tolerance: 0.01,
            duration_tolerance_ms: 100,
            max_resolution_ratio: 1.0,
            min_resolution_ratio: 0.01,
        }
    }
}

/// Engine that compares proxy and original media.
#[derive(Debug)]
pub struct ProxyCompareEngine {
    /// Tolerance settings.
    tolerance: ComparisonTolerance,
}

impl ProxyCompareEngine {
    /// Create a new comparison engine with default tolerances.
    pub fn new() -> Self {
        Self {
            tolerance: ComparisonTolerance::default(),
        }
    }

    /// Create a comparison engine with custom tolerances.
    pub fn with_tolerance(tolerance: ComparisonTolerance) -> Self {
        Self { tolerance }
    }

    /// Compare a proxy to its original source.
    #[allow(clippy::cast_precision_loss)]
    pub fn compare(&self, proxy: &MediaInfo, original: &MediaInfo) -> ComparisonResult {
        let resolution_ratio = proxy.resolution.ratio_to(&original.resolution);

        let bitrate_ratio = if original.bitrate_bps > 0 {
            proxy.bitrate_bps as f64 / original.bitrate_bps as f64
        } else {
            0.0
        };

        let frame_rate_match =
            (proxy.frame_rate - original.frame_rate).abs() <= self.tolerance.frame_rate_tolerance;

        let duration_diff_ms = proxy.duration_ms as i64 - original.duration_ms as i64;
        let duration_match =
            (duration_diff_ms.unsigned_abs()) <= self.tolerance.duration_tolerance_ms;

        let size_ratio = if original.file_size_bytes > 0 {
            proxy.file_size_bytes as f64 / original.file_size_bytes as f64
        } else {
            0.0
        };

        ComparisonResult {
            proxy: proxy.clone(),
            original: original.clone(),
            resolution_ratio,
            bitrate_ratio,
            frame_rate_match,
            duration_match,
            size_ratio,
            duration_diff_ms,
        }
    }

    /// Check whether a comparison result passes all quality gates.
    pub fn passes_qc(&self, result: &ComparisonResult) -> bool {
        result.frame_rate_match
            && result.duration_match
            && result.resolution_ratio >= self.tolerance.min_resolution_ratio
            && result.resolution_ratio <= self.tolerance.max_resolution_ratio
    }

    /// Compare a batch of proxy-original pairs and return results.
    pub fn compare_batch(&self, pairs: &[(MediaInfo, MediaInfo)]) -> Vec<ComparisonResult> {
        pairs
            .iter()
            .map(|(proxy, original)| self.compare(proxy, original))
            .collect()
    }

    /// Compute aggregate statistics from a batch of comparison results.
    #[allow(clippy::cast_precision_loss)]
    pub fn aggregate_stats(results: &[ComparisonResult]) -> ComparisonStats {
        if results.is_empty() {
            return ComparisonStats::default();
        }
        let total = results.len();
        let frame_rate_matches = results.iter().filter(|r| r.frame_rate_match).count();
        let duration_matches = results.iter().filter(|r| r.duration_match).count();
        let avg_resolution_ratio: f64 =
            results.iter().map(|r| r.resolution_ratio).sum::<f64>() / total as f64;
        let avg_bitrate_ratio: f64 =
            results.iter().map(|r| r.bitrate_ratio).sum::<f64>() / total as f64;
        let avg_size_ratio: f64 = results.iter().map(|r| r.size_ratio).sum::<f64>() / total as f64;

        ComparisonStats {
            total,
            frame_rate_matches,
            duration_matches,
            avg_resolution_ratio,
            avg_bitrate_ratio,
            avg_size_ratio,
        }
    }

    /// Group comparison results by codec.
    pub fn group_by_codec(results: &[ComparisonResult]) -> HashMap<String, Vec<usize>> {
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in results.iter().enumerate() {
            groups.entry(r.proxy.codec.clone()).or_default().push(i);
        }
        groups
    }
}

impl Default for ProxyCompareEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate statistics from a batch comparison.
#[derive(Debug, Clone, Default)]
pub struct ComparisonStats {
    /// Total comparisons.
    pub total: usize,
    /// How many had matching frame rates.
    pub frame_rate_matches: usize,
    /// How many had matching durations.
    pub duration_matches: usize,
    /// Average resolution ratio across all comparisons.
    pub avg_resolution_ratio: f64,
    /// Average bitrate ratio across all comparisons.
    pub avg_bitrate_ratio: f64,
    /// Average file size ratio across all comparisons.
    pub avg_size_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_original() -> MediaInfo {
        MediaInfo::new("/src/clip.mxf")
            .with_resolution(3840, 2160)
            .with_bitrate(100_000_000)
            .with_frame_rate(23.976)
            .with_duration_ms(60_000)
            .with_codec("ProRes")
            .with_file_size(750_000_000)
    }

    fn make_proxy() -> MediaInfo {
        MediaInfo::new("/proxy/clip.mp4")
            .with_resolution(1920, 1080)
            .with_bitrate(5_000_000)
            .with_frame_rate(23.976)
            .with_duration_ms(60_000)
            .with_codec("H264")
            .with_file_size(37_500_000)
    }

    #[test]
    fn test_resolution_pixel_count() {
        let r = Resolution::new(1920, 1080);
        assert_eq!(r.pixel_count(), 2_073_600);
    }

    #[test]
    fn test_resolution_ratio() {
        let proxy = Resolution::new(1920, 1080);
        let original = Resolution::new(3840, 2160);
        let ratio = proxy.ratio_to(&original);
        assert!((ratio - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_resolution_ratio_zero() {
        let a = Resolution::new(1920, 1080);
        let zero = Resolution::new(0, 0);
        assert!((a.ratio_to(&zero) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolution_display() {
        let r = Resolution::new(1920, 1080);
        assert_eq!(format!("{r}"), "1920x1080");
    }

    #[test]
    fn test_compare_frame_rate_match() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!(result.frame_rate_match);
    }

    #[test]
    fn test_compare_duration_match() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!(result.duration_match);
        assert_eq!(result.duration_diff_ms, 0);
    }

    #[test]
    fn test_compare_resolution_ratio() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!((result.resolution_ratio - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_compare_bitrate_ratio() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!((result.bitrate_ratio - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_compare_size_ratio() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!((result.size_ratio - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_passes_qc_default() {
        let engine = ProxyCompareEngine::new();
        let result = engine.compare(&make_proxy(), &make_original());
        assert!(engine.passes_qc(&result));
    }

    #[test]
    fn test_fails_qc_frame_rate_mismatch() {
        let engine = ProxyCompareEngine::new();
        let proxy = make_proxy().with_frame_rate(30.0);
        let result = engine.compare(&proxy, &make_original());
        assert!(!result.frame_rate_match);
        assert!(!engine.passes_qc(&result));
    }

    #[test]
    fn test_fails_qc_duration_mismatch() {
        let engine = ProxyCompareEngine::new();
        let proxy = make_proxy().with_duration_ms(65_000);
        let result = engine.compare(&proxy, &make_original());
        assert!(!result.duration_match);
    }

    #[test]
    fn test_compare_batch() {
        let engine = ProxyCompareEngine::new();
        let pairs = vec![
            (make_proxy(), make_original()),
            (make_proxy(), make_original()),
        ];
        let results = engine.compare_batch(&pairs);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_aggregate_stats() {
        let engine = ProxyCompareEngine::new();
        let results = vec![
            engine.compare(&make_proxy(), &make_original()),
            engine.compare(&make_proxy(), &make_original()),
        ];
        let stats = ProxyCompareEngine::aggregate_stats(&results);
        assert_eq!(stats.total, 2);
        assert_eq!(stats.frame_rate_matches, 2);
        assert_eq!(stats.duration_matches, 2);
    }

    #[test]
    fn test_aggregate_stats_empty() {
        let stats = ProxyCompareEngine::aggregate_stats(&[]);
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_group_by_codec() {
        let engine = ProxyCompareEngine::new();
        let results = vec![
            engine.compare(&make_proxy(), &make_original()),
            engine.compare(&make_proxy().with_codec("VP9"), &make_original()),
        ];
        let groups = ProxyCompareEngine::group_by_codec(&results);
        assert!(groups.contains_key("H264"));
        assert!(groups.contains_key("VP9"));
    }

    #[test]
    fn test_custom_tolerance() {
        let tolerance = ComparisonTolerance {
            frame_rate_tolerance: 1.0,
            duration_tolerance_ms: 5000,
            max_resolution_ratio: 1.0,
            min_resolution_ratio: 0.001,
        };
        let engine = ProxyCompareEngine::with_tolerance(tolerance);
        let proxy = make_proxy().with_frame_rate(24.0);
        let result = engine.compare(&proxy, &make_original());
        assert!(result.frame_rate_match);
    }

    #[test]
    fn test_default_engine() {
        let engine = ProxyCompareEngine::default();
        assert!((engine.tolerance.frame_rate_tolerance - 0.01).abs() < f64::EPSILON);
    }
}
