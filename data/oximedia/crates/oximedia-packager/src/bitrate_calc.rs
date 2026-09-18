#![allow(dead_code)]
//! Bitrate calculations and quality metrics for adaptive streaming.
//!
//! This module provides utilities for computing target bitrates,
//! bits-per-pixel ratios, bandwidth budgets, and quality estimations
//! used when building adaptive bitrate ladders.

use crate::error::{PackagerError, PackagerResult};

/// Bits-per-pixel calculator for a given resolution and frame rate.
#[derive(Debug, Clone, Copy)]
pub struct BppCalculator {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate.
    pub fps: f64,
}

impl BppCalculator {
    /// Create a new BPP calculator.
    ///
    /// # Errors
    ///
    /// Returns an error if width, height, or fps is zero.
    pub fn new(width: u32, height: u32, fps: f64) -> PackagerResult<Self> {
        if width == 0 || height == 0 {
            return Err(PackagerError::InvalidConfig(
                "Width and height must be > 0".into(),
            ));
        }
        if fps <= 0.0 {
            return Err(PackagerError::InvalidConfig("FPS must be > 0".into()));
        }
        Ok(Self { width, height, fps })
    }

    /// Total pixels per frame.
    #[must_use]
    pub fn pixels_per_frame(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Total pixels per second.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pixels_per_second(&self) -> f64 {
        self.pixels_per_frame() as f64 * self.fps
    }

    /// Compute bits per pixel for a given bitrate (bits/s).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bpp(&self, bitrate_bps: u64) -> f64 {
        bitrate_bps as f64 / self.pixels_per_second()
    }

    /// Compute the required bitrate (bits/s) for a target BPP.
    #[must_use]
    pub fn bitrate_for_bpp(&self, target_bpp: f64) -> u64 {
        (target_bpp * self.pixels_per_second()) as u64
    }
}

/// Quality tier labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTier {
    /// Low quality (mobile, bandwidth-constrained).
    Low,
    /// Medium quality (standard streaming).
    Medium,
    /// High quality (HD streaming).
    High,
    /// Premium quality (4K / high fidelity).
    Premium,
}

impl QualityTier {
    /// Suggested BPP range for AV1 codec.
    #[must_use]
    pub const fn av1_bpp_range(&self) -> (f64, f64) {
        match self {
            Self::Low => (0.02, 0.05),
            Self::Medium => (0.05, 0.10),
            Self::High => (0.10, 0.20),
            Self::Premium => (0.20, 0.40),
        }
    }

    /// Determine quality tier from a BPP value (AV1 assumptions).
    #[must_use]
    pub fn from_bpp(bpp: f64) -> Self {
        if bpp < 0.05 {
            Self::Low
        } else if bpp < 0.10 {
            Self::Medium
        } else if bpp < 0.20 {
            Self::High
        } else {
            Self::Premium
        }
    }

    /// Display name.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Premium => "Premium",
        }
    }
}

/// A bandwidth budget entry for one variant in the ladder.
#[derive(Debug, Clone)]
pub struct BandwidthBudget {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Video bitrate (bits/s).
    pub video_bitrate: u64,
    /// Audio bitrate (bits/s).
    pub audio_bitrate: u64,
    /// Quality tier.
    pub tier: QualityTier,
}

impl BandwidthBudget {
    /// Total bandwidth (video + audio).
    #[must_use]
    pub fn total_bandwidth(&self) -> u64 {
        self.video_bitrate + self.audio_bitrate
    }

    /// Compute the HLS BANDWIDTH tag value (rounded up to nearest 1000).
    #[must_use]
    pub fn hls_bandwidth(&self) -> u64 {
        let total = self.total_bandwidth();
        ((total + 999) / 1000) * 1000
    }
}

/// Generate a simple adaptive bitrate ladder from source parameters.
///
/// # Errors
///
/// Returns an error if the source dimensions or frame rate are invalid.
#[allow(clippy::cast_precision_loss)]
pub fn generate_ladder(
    source_width: u32,
    source_height: u32,
    source_fps: f64,
    audio_bitrate: u64,
) -> PackagerResult<Vec<BandwidthBudget>> {
    if source_width == 0 || source_height == 0 {
        return Err(PackagerError::InvalidConfig(
            "Source dimensions must be > 0".into(),
        ));
    }

    // Standard resolution steps (height-based)
    let steps: &[(u32, QualityTier)] = &[
        (2160, QualityTier::Premium),
        (1080, QualityTier::High),
        (720, QualityTier::Medium),
        (480, QualityTier::Low),
        (360, QualityTier::Low),
    ];

    let aspect = source_width as f64 / source_height as f64;
    let mut ladder = Vec::new();

    for &(h, tier) in steps {
        if h > source_height {
            continue;
        }
        let w = ((h as f64 * aspect) as u32) & !1; // even width
        let calc = BppCalculator::new(w, h, source_fps)?;
        let (_, bpp_high) = tier.av1_bpp_range();
        let mid_bpp = bpp_high * 0.7; // target ~70% of tier ceiling
        let video_bitrate = calc.bitrate_for_bpp(mid_bpp);

        ladder.push(BandwidthBudget {
            width: w,
            height: h,
            video_bitrate,
            audio_bitrate,
            tier,
        });
    }

    Ok(ladder)
}

/// Estimate file size in bytes for a given bitrate and duration.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_file_size(bitrate_bps: u64, duration_seconds: f64) -> u64 {
    (bitrate_bps as f64 * duration_seconds / 8.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpp_calculator_creation() {
        let calc = BppCalculator::new(1920, 1080, 30.0);
        assert!(calc.is_ok());
    }

    #[test]
    fn test_bpp_calculator_zero_width() {
        let calc = BppCalculator::new(0, 1080, 30.0);
        assert!(calc.is_err());
    }

    #[test]
    fn test_bpp_calculator_zero_fps() {
        let calc = BppCalculator::new(1920, 1080, 0.0);
        assert!(calc.is_err());
    }

    #[test]
    fn test_pixels_per_frame() {
        let calc = BppCalculator::new(1920, 1080, 30.0).expect("should succeed in test");
        assert_eq!(calc.pixels_per_frame(), 1920 * 1080);
    }

    #[test]
    fn test_bpp_calculation() {
        let calc = BppCalculator::new(1920, 1080, 30.0).expect("should succeed in test");
        let bpp = calc.bpp(5_000_000);
        assert!(bpp > 0.0 && bpp < 1.0);
    }

    #[test]
    fn test_bpp_roundtrip() {
        let calc = BppCalculator::new(1280, 720, 24.0).expect("should succeed in test");
        let target_bpp = 0.1;
        let bitrate = calc.bitrate_for_bpp(target_bpp);
        let actual_bpp = calc.bpp(bitrate);
        assert!((actual_bpp - target_bpp).abs() < 0.001);
    }

    #[test]
    fn test_quality_tier_from_bpp() {
        assert_eq!(QualityTier::from_bpp(0.01), QualityTier::Low);
        assert_eq!(QualityTier::from_bpp(0.07), QualityTier::Medium);
        assert_eq!(QualityTier::from_bpp(0.15), QualityTier::High);
        assert_eq!(QualityTier::from_bpp(0.30), QualityTier::Premium);
    }

    #[test]
    fn test_quality_tier_label() {
        assert_eq!(QualityTier::Low.label(), "Low");
        assert_eq!(QualityTier::Premium.label(), "Premium");
    }

    #[test]
    fn test_bandwidth_budget_total() {
        let b = BandwidthBudget {
            width: 1920,
            height: 1080,
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            tier: QualityTier::High,
        };
        assert_eq!(b.total_bandwidth(), 5_128_000);
    }

    #[test]
    fn test_hls_bandwidth_rounding() {
        let b = BandwidthBudget {
            width: 1280,
            height: 720,
            video_bitrate: 3_000_500,
            audio_bitrate: 128_000,
            tier: QualityTier::Medium,
        };
        let hls = b.hls_bandwidth();
        assert_eq!(hls % 1000, 0);
        assert!(hls >= b.total_bandwidth());
    }

    #[test]
    fn test_generate_ladder_1080p() {
        let ladder = generate_ladder(1920, 1080, 30.0, 128_000).expect("should succeed in test");
        assert!(!ladder.is_empty());
        // Should include 1080, 720, 480, 360
        let heights: Vec<u32> = ladder.iter().map(|b| b.height).collect();
        assert!(heights.contains(&1080));
        assert!(heights.contains(&720));
    }

    #[test]
    fn test_generate_ladder_invalid() {
        let result = generate_ladder(0, 1080, 30.0, 128_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_file_size() {
        // 5 Mbps * 60 seconds = 37.5 MB
        let size = estimate_file_size(5_000_000, 60.0);
        assert_eq!(size, 37_500_000);
    }

    #[test]
    fn test_av1_bpp_range() {
        let (lo, hi) = QualityTier::Medium.av1_bpp_range();
        assert!(lo < hi);
        assert!(lo > 0.0);
    }
}
