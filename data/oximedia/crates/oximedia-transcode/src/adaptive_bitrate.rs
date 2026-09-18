//! Adaptive bitrate ladder generation for HLS/DASH streaming.
//!
//! This module provides tools for generating Netflix-style ABR ladders,
//! per-title encoding optimization, and bandwidth estimation.

use std::collections::VecDeque;

/// A single rendition in an ABR ladder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionBitrate {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Codec name (e.g., "h264", "vp9", "av1").
    pub codec: String,
}

impl ResolutionBitrate {
    /// Creates a new resolution/bitrate rendition.
    #[must_use]
    pub fn new(width: u32, height: u32, bitrate_kbps: u32, codec: impl Into<String>) -> Self {
        Self {
            width,
            height,
            bitrate_kbps,
            codec: codec.into(),
        }
    }

    /// Returns the pixel count for this rendition.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the aspect ratio as a float.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// An adaptive bitrate ladder containing multiple renditions.
#[derive(Debug, Clone)]
pub struct AbrLadder {
    /// Renditions sorted by bitrate descending.
    pub renditions: Vec<ResolutionBitrate>,
}

impl AbrLadder {
    /// Creates a new ABR ladder.
    #[must_use]
    pub fn new(mut renditions: Vec<ResolutionBitrate>) -> Self {
        // Sort descending by bitrate
        renditions.sort_by(|a, b| b.bitrate_kbps.cmp(&a.bitrate_kbps));
        Self { renditions }
    }

    /// Selects the optimal rendition for a given available bandwidth.
    ///
    /// Returns the highest-bitrate rendition whose bitrate is ≤ 80% of available bandwidth.
    #[must_use]
    pub fn optimal_bitrate_kbps(&self, bandwidth_kbps: u32) -> Option<&ResolutionBitrate> {
        let threshold = (f64::from(bandwidth_kbps) * 0.8) as u32;
        self.renditions.iter().find(|r| r.bitrate_kbps <= threshold)
    }

    /// Returns the number of renditions in the ladder.
    #[must_use]
    pub fn len(&self) -> usize {
        self.renditions.len()
    }

    /// Returns true if the ladder has no renditions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.renditions.is_empty()
    }

    /// Returns the highest-quality rendition.
    #[must_use]
    pub fn highest_quality(&self) -> Option<&ResolutionBitrate> {
        self.renditions.first()
    }

    /// Returns the lowest-quality rendition.
    #[must_use]
    pub fn lowest_quality(&self) -> Option<&ResolutionBitrate> {
        self.renditions.last()
    }
}

/// Generator for ABR ladders targeting specific resolutions and codecs.
#[derive(Debug, Clone, Default)]
pub struct AbrLadderGenerator;

impl AbrLadderGenerator {
    /// Creates a new `AbrLadderGenerator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Generates a Netflix-style ABR ladder for the given target resolution and codec.
    ///
    /// Standard ladder: 1080p/8000, 720p/4500, 540p/2000, 360p/800, 240p/300 kbps.
    /// Only renditions at or below the target resolution are included.
    #[must_use]
    pub fn generate(&self, target_resolution: (u32, u32), codec: &str) -> AbrLadder {
        let (_target_w, target_h) = target_resolution;

        // Netflix-style standard ladder
        let standard_rungs: &[(u32, u32, u32)] = &[
            (1920, 1080, 8000),
            (1280, 720, 4500),
            (960, 540, 2000),
            (640, 360, 800),
            (426, 240, 300),
        ];

        let renditions = standard_rungs
            .iter()
            .filter(|(_, h, _)| *h <= target_h)
            .map(|(w, h, kbps)| ResolutionBitrate::new(*w, *h, *kbps, codec))
            .collect();

        AbrLadder::new(renditions)
    }
}

/// Per-title encoding optimization for content-aware ABR ladders.
#[derive(Debug, Clone, Default)]
pub struct PerTitleEncoding;

impl PerTitleEncoding {
    /// Creates a new `PerTitleEncoding` optimizer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyzes content complexity from frame variance values.
    ///
    /// Returns the mean variance as a complexity score.
    #[must_use]
    pub fn analyze_complexity(frame_variance: &[f32]) -> f32 {
        if frame_variance.is_empty() {
            return 0.0;
        }
        let sum: f32 = frame_variance.iter().sum();
        sum / frame_variance.len() as f32
    }

    /// Optimizes a base ABR ladder based on content complexity.
    ///
    /// Complexity factor is clamped to 0.5–2.0. Bitrates are scaled accordingly.
    #[must_use]
    pub fn optimize_ladder(base_ladder: &AbrLadder, complexity: f32) -> AbrLadder {
        // Clamp complexity factor to [0.5, 2.0]
        let factor = complexity.clamp(0.5, 2.0);

        let adjusted = base_ladder
            .renditions
            .iter()
            .map(|r| {
                let new_bitrate = ((r.bitrate_kbps as f32) * factor).round() as u32;
                ResolutionBitrate::new(r.width, r.height, new_bitrate, r.codec.clone())
            })
            .collect();

        AbrLadder::new(adjusted)
    }
}

/// A bandwidth measurement at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandwidthPoint {
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Measured bandwidth in kilobits per second.
    pub kbps: u32,
}

/// Ring buffer of recent bandwidth samples (last 30 samples).
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    samples: VecDeque<BandwidthPoint>,
    capacity: usize,
}

impl BandwidthEstimator {
    /// Creates a new bandwidth estimator with a 30-sample buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(30),
            capacity: 30,
        }
    }

    /// Adds a bandwidth sample, evicting oldest if at capacity.
    pub fn add_sample(&mut self, point: BandwidthPoint) {
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(point);
    }

    /// Returns the number of samples in the buffer.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the estimated bandwidth as the average of recent samples.
    #[must_use]
    pub fn estimated_kbps(&self) -> Option<u32> {
        if self.samples.is_empty() {
            return None;
        }
        let sum: u64 = self.samples.iter().map(|s| u64::from(s.kbps)).sum();
        Some((sum / self.samples.len() as u64) as u32)
    }

    /// Returns all samples in the buffer.
    #[must_use]
    pub fn samples(&self) -> &VecDeque<BandwidthPoint> {
        &self.samples
    }
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_bitrate_new() {
        let r = ResolutionBitrate::new(1920, 1080, 8000, "h264");
        assert_eq!(r.width, 1920);
        assert_eq!(r.height, 1080);
        assert_eq!(r.bitrate_kbps, 8000);
        assert_eq!(r.codec, "h264");
    }

    #[test]
    fn test_resolution_bitrate_pixel_count() {
        let r = ResolutionBitrate::new(1920, 1080, 8000, "h264");
        assert_eq!(r.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_bitrate_aspect_ratio() {
        let r = ResolutionBitrate::new(1920, 1080, 8000, "h264");
        let ar = r.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_abr_ladder_sorted_descending() {
        let renditions = vec![
            ResolutionBitrate::new(640, 360, 800, "h264"),
            ResolutionBitrate::new(1920, 1080, 8000, "h264"),
            ResolutionBitrate::new(1280, 720, 4500, "h264"),
        ];
        let ladder = AbrLadder::new(renditions);
        assert_eq!(ladder.renditions[0].bitrate_kbps, 8000);
        assert_eq!(ladder.renditions[1].bitrate_kbps, 4500);
        assert_eq!(ladder.renditions[2].bitrate_kbps, 800);
    }

    #[test]
    fn test_abr_ladder_optimal_bitrate() {
        let gen = AbrLadderGenerator::new();
        let ladder = gen.generate((1920, 1080), "h264");

        // 80% of 10000 = 8000 → exactly 8000 qualifies
        let opt = ladder.optimal_bitrate_kbps(10000);
        assert!(opt.is_some());
        assert_eq!(opt.expect("should succeed in test").bitrate_kbps, 8000);

        // 80% of 5000 = 4000 → best is 2000
        let opt2 = ladder.optimal_bitrate_kbps(5000);
        assert!(opt2.is_some());
        assert_eq!(opt2.expect("should succeed in test").bitrate_kbps, 2000);
    }

    #[test]
    fn test_abr_ladder_optimal_bitrate_too_low() {
        let gen = AbrLadderGenerator::new();
        let ladder = gen.generate((1920, 1080), "h264");

        // 80% of 100 = 80 → none qualify
        let opt = ladder.optimal_bitrate_kbps(100);
        assert!(opt.is_none());
    }

    #[test]
    fn test_abr_ladder_empty_check() {
        let ladder = AbrLadder::new(vec![]);
        assert!(ladder.is_empty());
        assert_eq!(ladder.len(), 0);
        assert!(ladder.highest_quality().is_none());
        assert!(ladder.lowest_quality().is_none());
    }

    #[test]
    fn test_abr_ladder_generator_full_1080p() {
        let gen = AbrLadderGenerator::new();
        let ladder = gen.generate((1920, 1080), "h264");
        assert_eq!(ladder.len(), 5);
        assert_eq!(
            ladder
                .highest_quality()
                .expect("should succeed in test")
                .bitrate_kbps,
            8000
        );
        assert_eq!(
            ladder
                .lowest_quality()
                .expect("should succeed in test")
                .bitrate_kbps,
            300
        );
    }

    #[test]
    fn test_abr_ladder_generator_720p_limit() {
        let gen = AbrLadderGenerator::new();
        let ladder = gen.generate((1280, 720), "vp9");
        // Should only include 720p, 540p, 360p, 240p (not 1080p)
        for r in &ladder.renditions {
            assert!(r.height <= 720);
        }
    }

    #[test]
    fn test_per_title_analyze_complexity_empty() {
        let c = PerTitleEncoding::analyze_complexity(&[]);
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_per_title_analyze_complexity() {
        let variances = vec![1.0, 2.0, 3.0, 4.0];
        let c = PerTitleEncoding::analyze_complexity(&variances);
        assert!((c - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_per_title_optimize_ladder_clamping() {
        let gen = AbrLadderGenerator::new();
        let base = gen.generate((1920, 1080), "h264");

        // Complexity 3.0 should be clamped to 2.0
        let optimized = PerTitleEncoding::optimize_ladder(&base, 3.0);
        let base_highest = base
            .highest_quality()
            .expect("should succeed in test")
            .bitrate_kbps;
        let opt_highest = optimized
            .highest_quality()
            .expect("should succeed in test")
            .bitrate_kbps;
        assert_eq!(opt_highest, (base_highest as f32 * 2.0).round() as u32);
    }

    #[test]
    fn test_per_title_optimize_ladder_lower() {
        let gen = AbrLadderGenerator::new();
        let base = gen.generate((1920, 1080), "h264");

        // Complexity 0.5 halves bitrates
        let optimized = PerTitleEncoding::optimize_ladder(&base, 0.5);
        let base_highest = base
            .highest_quality()
            .expect("should succeed in test")
            .bitrate_kbps;
        let opt_highest = optimized
            .highest_quality()
            .expect("should succeed in test")
            .bitrate_kbps;
        assert_eq!(opt_highest, (base_highest as f32 * 0.5).round() as u32);
    }

    #[test]
    fn test_bandwidth_estimator_add_sample() {
        let mut est = BandwidthEstimator::new();
        assert_eq!(est.sample_count(), 0);
        assert!(est.estimated_kbps().is_none());

        est.add_sample(BandwidthPoint {
            timestamp_ms: 0,
            kbps: 5000,
        });
        assert_eq!(est.sample_count(), 1);
        assert_eq!(est.estimated_kbps(), Some(5000));
    }

    #[test]
    fn test_bandwidth_estimator_ring_buffer() {
        let mut est = BandwidthEstimator::new();
        for i in 0..35u64 {
            est.add_sample(BandwidthPoint {
                timestamp_ms: i * 1000,
                kbps: 1000,
            });
        }
        // Should only keep last 30
        assert_eq!(est.sample_count(), 30);
    }

    #[test]
    fn test_bandwidth_estimator_average() {
        let mut est = BandwidthEstimator::new();
        est.add_sample(BandwidthPoint {
            timestamp_ms: 0,
            kbps: 4000,
        });
        est.add_sample(BandwidthPoint {
            timestamp_ms: 1000,
            kbps: 6000,
        });
        assert_eq!(est.estimated_kbps(), Some(5000));
    }
}
