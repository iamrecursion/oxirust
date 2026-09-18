//! Dynamic range processing primitives.
//!
//! Provides loudness mapping, compression, brick-wall limiting, and true-peak
//! estimation — all operating in the dB/dBFS domain.

#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// DynamicRangeSpec
// ──────────────────────────────────────────────────────────────────────────────

/// A specification describing the minimum and maximum loudness of a signal
/// in dB (or LUFS, depending on context).
#[derive(Debug, Clone, Copy)]
pub struct DynamicRangeSpec {
    /// Minimum (quietest) level in dB.
    pub min_db: f32,
    /// Maximum (loudest) level in dB.
    pub max_db: f32,
}

impl DynamicRangeSpec {
    /// Create a new specification.
    ///
    /// `min_db` must be ≤ `max_db`.
    pub fn new(min_db: f32, max_db: f32) -> Self {
        debug_assert!(min_db <= max_db, "min_db must be <= max_db");
        Self { min_db, max_db }
    }

    /// The total range in dB (`max_db − min_db`).
    pub fn range_db(&self) -> f32 {
        self.max_db - self.min_db
    }

    /// Returns `true` if `db` lies within `[min_db, max_db]` (inclusive).
    pub fn contains(&self, db: f32) -> bool {
        db >= self.min_db && db <= self.max_db
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// LoudnessMap
// ──────────────────────────────────────────────────────────────────────────────

/// Time-ordered loudness map: a sequence of `(frame, loudness_db)` measurements.
#[derive(Debug, Default)]
pub struct LoudnessMap {
    /// Ordered list of `(frame_index, loudness_db)` pairs.
    pub frames: Vec<(u64, f32)>,
}

impl LoudnessMap {
    /// Create an empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a loudness measurement.
    pub fn add(&mut self, frame: u64, loudness_db: f32) {
        self.frames.push((frame, loudness_db));
    }

    /// Mean loudness across all measurements.
    ///
    /// Returns `f32::NEG_INFINITY` when empty.
    pub fn average_loudness(&self) -> f32 {
        if self.frames.is_empty() {
            return f32::NEG_INFINITY;
        }
        let sum: f32 = self.frames.iter().map(|&(_, l)| l).sum();
        sum / self.frames.len() as f32
    }

    /// Loudest (maximum) measurement.
    ///
    /// Returns `f32::NEG_INFINITY` when empty.
    pub fn peak_loudness(&self) -> f32 {
        self.frames
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Return the dynamic range as a [`DynamicRangeSpec`].
    ///
    /// Returns a zero-range spec at 0 dB when empty.
    pub fn range(&self) -> DynamicRangeSpec {
        if self.frames.is_empty() {
            return DynamicRangeSpec::new(0.0, 0.0);
        }
        let min = self
            .frames
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::INFINITY, f32::min);
        let max = self
            .frames
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        DynamicRangeSpec::new(min, max)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DynamicRangeCompressor
// ──────────────────────────────────────────────────────────────────────────────

/// Simple feed-forward dynamic range compressor.
///
/// Applies gain reduction to levels above `threshold_db` according to `ratio`.
#[derive(Debug, Clone, Copy)]
pub struct DynamicRangeCompressor {
    /// Level above which compression begins, in dB.
    pub threshold_db: f32,
    /// Compression ratio (e.g. 4.0 = 4:1).  Must be ≥ 1.0.
    pub ratio: f32,
}

impl DynamicRangeCompressor {
    /// Create a new compressor.
    pub fn new(threshold_db: f32, ratio: f32) -> Self {
        debug_assert!(ratio >= 1.0, "ratio must be >= 1.0");
        Self {
            threshold_db,
            ratio,
        }
    }

    /// Compute the gain reduction (in dB, always ≤ 0) for an input level.
    ///
    /// No reduction is applied below the threshold.
    pub fn compress(&self, level_db: f32) -> f32 {
        if level_db <= self.threshold_db {
            return 0.0;
        }
        let overshoot = level_db - self.threshold_db;
        let output_overshoot = overshoot / self.ratio;
        output_overshoot - overshoot // always ≤ 0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BrickwallLimiter
// ──────────────────────────────────────────────────────────────────────────────

/// Instantaneous brick-wall sample limiter.
#[derive(Debug, Clone, Copy)]
pub struct BrickwallLimiter {
    /// Maximum absolute sample value (derived from `ceiling_db`).
    ceiling_db: f32,
    ceiling_linear: f32,
}

impl BrickwallLimiter {
    /// Create a limiter with the given ceiling in dBFS.
    pub fn new(ceiling_db: f32) -> Self {
        let ceiling_linear = 10.0_f32.powf(ceiling_db / 20.0);
        Self {
            ceiling_db,
            ceiling_linear,
        }
    }

    /// Return the ceiling in dBFS.
    pub fn ceiling_db(&self) -> f32 {
        self.ceiling_db
    }

    /// Clip `sample` so that its magnitude does not exceed the ceiling.
    pub fn limit(&self, sample: f32) -> f32 {
        sample.clamp(-self.ceiling_linear, self.ceiling_linear)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TruePeakEstimator
// ──────────────────────────────────────────────────────────────────────────────

/// Estimates true peak level via 4x linear interpolation oversampling.
pub struct TruePeakEstimator;

impl TruePeakEstimator {
    /// Estimate the true peak of `samples` by upsampling 4x with linear
    /// interpolation and finding the maximum absolute value.
    ///
    /// Returns the peak as a linear (non-dB) value.
    pub fn estimate(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let factor = 4usize;
        let mut peak = 0.0_f32;

        for i in 0..samples.len() {
            let s0 = samples[i];
            let s1 = if i + 1 < samples.len() {
                samples[i + 1]
            } else {
                0.0
            };

            for sub in 0..factor {
                let t = sub as f32 / factor as f32;
                let interp = s0 + t * (s1 - s0);
                peak = peak.max(interp.abs());
            }
        }

        peak
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // DynamicRangeSpec ────────────────────────────────────────────────────────

    #[test]
    fn test_range_db() {
        let s = DynamicRangeSpec::new(-30.0, -10.0);
        assert!((s.range_db() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_contains_inside() {
        let s = DynamicRangeSpec::new(-30.0, -10.0);
        assert!(s.contains(-20.0));
    }

    #[test]
    fn test_contains_boundary() {
        let s = DynamicRangeSpec::new(-30.0, -10.0);
        assert!(s.contains(-30.0));
        assert!(s.contains(-10.0));
    }

    #[test]
    fn test_contains_outside() {
        let s = DynamicRangeSpec::new(-30.0, -10.0);
        assert!(!s.contains(-31.0));
        assert!(!s.contains(-9.0));
    }

    // LoudnessMap ─────────────────────────────────────────────────────────────

    #[test]
    fn test_loudness_map_empty_average() {
        let m = LoudnessMap::new();
        assert_eq!(m.average_loudness(), f32::NEG_INFINITY);
    }

    #[test]
    fn test_loudness_map_average() {
        let mut m = LoudnessMap::new();
        m.add(0, -20.0);
        m.add(1, -10.0);
        assert!((m.average_loudness() - (-15.0)).abs() < 1e-5);
    }

    #[test]
    fn test_loudness_map_peak() {
        let mut m = LoudnessMap::new();
        m.add(0, -30.0);
        m.add(1, -10.0);
        m.add(2, -20.0);
        assert!((m.peak_loudness() - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_loudness_map_range() {
        let mut m = LoudnessMap::new();
        m.add(0, -30.0);
        m.add(1, -10.0);
        let r = m.range();
        assert!((r.min_db - (-30.0)).abs() < 1e-5);
        assert!((r.max_db - (-10.0)).abs() < 1e-5);
    }

    // DynamicRangeCompressor ──────────────────────────────────────────────────

    #[test]
    fn test_compressor_below_threshold_no_gain_reduction() {
        let c = DynamicRangeCompressor::new(-20.0, 4.0);
        assert_eq!(c.compress(-30.0), 0.0);
    }

    #[test]
    fn test_compressor_at_threshold_no_gain_reduction() {
        let c = DynamicRangeCompressor::new(-20.0, 4.0);
        assert_eq!(c.compress(-20.0), 0.0);
    }

    #[test]
    fn test_compressor_above_threshold_gain_reduction() {
        let c = DynamicRangeCompressor::new(-20.0, 4.0);
        // 4 dB above threshold → output overshoot = 1 dB → reduction = -3 dB.
        let gr = c.compress(-16.0);
        assert!((gr - (-3.0)).abs() < 1e-5, "Expected -3 dB, got {}", gr);
    }

    // BrickwallLimiter ────────────────────────────────────────────────────────

    #[test]
    fn test_limiter_below_ceiling_unchanged() {
        let lim = BrickwallLimiter::new(-1.0); // ≈ 0.891 linear
        let sample = 0.5_f32;
        assert!((lim.limit(sample) - sample).abs() < 1e-6);
    }

    #[test]
    fn test_limiter_clips_positive() {
        let lim = BrickwallLimiter::new(0.0); // 1.0 linear
        assert!((lim.limit(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_limiter_clips_negative() {
        let lim = BrickwallLimiter::new(0.0);
        assert!((lim.limit(-2.0) - (-1.0)).abs() < 1e-6);
    }

    // TruePeakEstimator ───────────────────────────────────────────────────────

    #[test]
    fn test_true_peak_empty() {
        assert_eq!(TruePeakEstimator::estimate(&[]), 0.0);
    }

    #[test]
    fn test_true_peak_constant() {
        let samples = vec![0.5f32; 100];
        let peak = TruePeakEstimator::estimate(&samples);
        assert!(peak >= 0.5, "peak should be >= 0.5, got {}", peak);
    }

    #[test]
    fn test_true_peak_single_sample() {
        let peak = TruePeakEstimator::estimate(&[0.8]);
        assert!((peak - 0.8).abs() < 1e-5);
    }
}
