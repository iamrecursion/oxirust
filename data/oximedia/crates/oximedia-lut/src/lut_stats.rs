#![allow(dead_code)]
//! Statistical analysis of LUT output distributions.
//!
//! Provides:
//! * [`LutChannel`]      – identifies a colour channel.
//! * [`LutChannelStats`] – min/max/midpoint statistics for one channel.
//! * [`Lut3dStats`]      – full per-channel statistics for a 3-D LUT with clipping detection.

use crate::Rgb;

// ---------------------------------------------------------------------------
// LutChannel
// ---------------------------------------------------------------------------

/// Identifies a colour channel in a 3-component RGB LUT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LutChannel {
    /// The red channel (index 0).
    Red,
    /// The green channel (index 1).
    Green,
    /// The blue channel (index 2).
    Blue,
}

impl LutChannel {
    /// Returns the slice index for this channel (`0`, `1`, or `2`).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Green => 1,
            Self::Blue => 2,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Red => "Red",
            Self::Green => "Green",
            Self::Blue => "Blue",
        }
    }
}

// ---------------------------------------------------------------------------
// LutChannelStats
// ---------------------------------------------------------------------------

/// Statistics for a single channel across all 3-D LUT lattice points.
#[derive(Debug, Clone)]
pub struct LutChannelStats {
    /// Minimum output value observed across all lattice points.
    min: f64,
    /// Maximum output value observed across all lattice points.
    max: f64,
    /// Arithmetic mean of output values.
    mean: f64,
    /// Population standard deviation.
    std_dev: f64,
    /// Total number of lattice points sampled.
    count: usize,
}

impl LutChannelStats {
    /// Compute channel statistics from a flat 3-D LUT for `channel`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(lut: &[Rgb], channel: LutChannel) -> Self {
        let ch = channel.index();
        let count = lut.len();
        if count == 0 {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                count: 0,
            };
        }

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0_f64;

        for entry in lut {
            let v = entry[ch];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }

        let mean = sum / count as f64;
        let variance = lut.iter().map(|e| (e[ch] - mean).powi(2)).sum::<f64>() / count as f64;

        Self {
            min,
            max,
            mean,
            std_dev: variance.sqrt(),
            count,
        }
    }

    /// The midpoint between min and max.
    #[must_use]
    pub fn midpoint(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    /// Minimum output value.
    #[must_use]
    pub fn min_output(&self) -> f64 {
        self.min
    }

    /// Maximum output value.
    #[must_use]
    pub fn max_output(&self) -> f64 {
        self.max
    }

    /// Arithmetic mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Total lattice points sampled.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Dynamic range (max − min).
    #[must_use]
    pub fn dynamic_range(&self) -> f64 {
        self.max - self.min
    }
}

// ---------------------------------------------------------------------------
// Lut3dStats
// ---------------------------------------------------------------------------

/// Full statistical profile for all three channels of a 3-D LUT.
#[derive(Debug, Clone)]
pub struct Lut3dStats {
    red: LutChannelStats,
    green: LutChannelStats,
    blue: LutChannelStats,
}

impl Lut3dStats {
    /// Compute statistics for all channels.
    #[must_use]
    pub fn compute(lut: &[Rgb]) -> Self {
        Self {
            red: LutChannelStats::compute(lut, LutChannel::Red),
            green: LutChannelStats::compute(lut, LutChannel::Green),
            blue: LutChannelStats::compute(lut, LutChannel::Blue),
        }
    }

    /// Statistics for the requested channel.
    #[must_use]
    pub fn channel_stats(&self, channel: LutChannel) -> &LutChannelStats {
        match channel {
            LutChannel::Red => &self.red,
            LutChannel::Green => &self.green,
            LutChannel::Blue => &self.blue,
        }
    }

    /// Returns `true` when any channel clips to exactly `0.0` or `1.0`.
    ///
    /// Clipping is detected when the channel minimum is `≤ 0.0` (with a small
    /// tolerance) or the maximum is `≥ 1.0`.
    #[must_use]
    pub fn is_clipping(&self) -> bool {
        const EPS: f64 = 1e-6;
        for ch in [LutChannel::Red, LutChannel::Green, LutChannel::Blue] {
            let s = self.channel_stats(ch);
            if s.min_output() <= EPS || s.max_output() >= 1.0 - EPS {
                return true;
            }
        }
        false
    }

    /// Overall maximum output across all three channels.
    #[must_use]
    pub fn overall_max(&self) -> f64 {
        self.red
            .max_output()
            .max(self.green.max_output())
            .max(self.blue.max_output())
    }

    /// Overall minimum output across all three channels.
    #[must_use]
    pub fn overall_min(&self) -> f64 {
        self.red
            .min_output()
            .min(self.green.min_output())
            .min(self.blue.min_output())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an identity 3-D LUT of the given size.
    #[allow(clippy::cast_precision_loss)]
    fn identity_3d(size: usize) -> Vec<Rgb> {
        let scale = (size - 1) as f64;
        let mut data = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    data.push([r as f64 / scale, g as f64 / scale, b as f64 / scale]);
                }
            }
        }
        data
    }

    #[test]
    fn test_channel_index() {
        assert_eq!(LutChannel::Red.index(), 0);
        assert_eq!(LutChannel::Green.index(), 1);
        assert_eq!(LutChannel::Blue.index(), 2);
    }

    #[test]
    fn test_channel_name() {
        assert_eq!(LutChannel::Red.name(), "Red");
        assert_eq!(LutChannel::Green.name(), "Green");
        assert_eq!(LutChannel::Blue.name(), "Blue");
    }

    #[test]
    fn test_channel_stats_min_max_identity() {
        let lut = identity_3d(5);
        let stats = LutChannelStats::compute(&lut, LutChannel::Red);
        assert!((stats.min_output() - 0.0).abs() < 1e-9);
        assert!((stats.max_output() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_stats_midpoint() {
        let lut = identity_3d(5);
        let stats = LutChannelStats::compute(&lut, LutChannel::Green);
        assert!((stats.midpoint() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_channel_stats_count() {
        let lut = identity_3d(4);
        let stats = LutChannelStats::compute(&lut, LutChannel::Blue);
        assert_eq!(stats.count(), 64);
    }

    #[test]
    fn test_channel_stats_dynamic_range_identity() {
        let lut = identity_3d(5);
        let stats = LutChannelStats::compute(&lut, LutChannel::Red);
        assert!((stats.dynamic_range() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_channel_stats_mean_identity() {
        let lut = identity_3d(5);
        let stats = LutChannelStats::compute(&lut, LutChannel::Red);
        // Mean should be 0.5 for a symmetric identity distribution
        assert!((stats.mean() - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_channel_stats_empty_lut() {
        let stats = LutChannelStats::compute(&[], LutChannel::Red);
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.min_output(), 0.0);
        assert_eq!(stats.max_output(), 0.0);
    }

    #[test]
    fn test_lut3d_stats_channel_stats_red() {
        let lut = identity_3d(5);
        let stats = Lut3dStats::compute(&lut);
        let r = stats.channel_stats(LutChannel::Red);
        assert!((r.min_output() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lut3d_stats_overall_max() {
        let lut = identity_3d(5);
        let stats = Lut3dStats::compute(&lut);
        assert!((stats.overall_max() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lut3d_stats_overall_min() {
        let lut = identity_3d(5);
        let stats = Lut3dStats::compute(&lut);
        assert!((stats.overall_min() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_lut3d_stats_not_clipping_for_interior() {
        // A LUT that only outputs interior values should not clip.
        let lut: Vec<Rgb> = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let stats = Lut3dStats::compute(&lut);
        assert!(!stats.is_clipping());
    }

    #[test]
    fn test_lut3d_stats_clipping_detected_at_one() {
        let lut: Vec<Rgb> = vec![[0.1, 0.2, 1.0]];
        let stats = Lut3dStats::compute(&lut);
        assert!(stats.is_clipping());
    }

    #[test]
    fn test_lut3d_stats_clipping_detected_at_zero() {
        let lut: Vec<Rgb> = vec![[0.0, 0.5, 0.5]];
        let stats = Lut3dStats::compute(&lut);
        assert!(stats.is_clipping());
    }
}
