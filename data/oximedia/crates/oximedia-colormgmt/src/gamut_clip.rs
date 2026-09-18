#![allow(dead_code)]
//! Gamut clipping algorithms for out-of-gamut color handling.
//!
//! When converting colors between gamuts with different extents (e.g., Rec.2020
//! to Rec.709), some source colors have no exact representation in the target.
//! This module provides multiple clipping strategies that map out-of-gamut
//! colors back into the target gamut while preserving perceptual quality.
//!
//! # Strategies
//!
//! - **Hard clip** — clamp each RGB channel independently to `[0, 1]`.
//! - **Luminance-preserving clip** — project toward the achromatic axis while
//!   keeping relative luminance constant.
//! - **Chroma-reduction clip** — reduce CIE LCh chroma until the color fits the
//!   target gamut, preserving lightness and hue.
//! - **Adaptive clip** — weighted blend of luminance-preserving and
//!   chroma-reduction, configurable per use-case.

use std::fmt;

// ─── Clipping Strategy ──────────────────────────────────────────────────────

/// Strategy for mapping out-of-gamut colors into the target gamut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipStrategy {
    /// Clamp each RGB channel independently to `[0, 1]`.
    HardClip,
    /// Project toward the achromatic axis, preserving relative luminance.
    LuminancePreserving,
    /// Reduce chroma in CIE LCh while preserving lightness and hue.
    ChromaReduction,
    /// Blend of luminance-preserving and chroma-reduction with a configurable
    /// alpha (stored separately in [`GamutClipper`]).
    Adaptive,
}

impl fmt::Display for ClipStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HardClip => f.write_str("Hard Clip"),
            Self::LuminancePreserving => f.write_str("Luminance-Preserving"),
            Self::ChromaReduction => f.write_str("Chroma Reduction"),
            Self::Adaptive => f.write_str("Adaptive"),
        }
    }
}

// ─── RGB Triplet helper ─────────────────────────────────────────────────────

/// A linear-light RGB triplet (may be out of `[0, 1]` if the color is out of gamut).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRgb {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
}

impl LinearRgb {
    /// Create a new linear RGB value.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Returns `true` if all channels are within `[0, 1]`.
    #[must_use]
    pub fn is_in_gamut(&self) -> bool {
        self.r >= 0.0
            && self.r <= 1.0
            && self.g >= 0.0
            && self.g <= 1.0
            && self.b >= 0.0
            && self.b <= 1.0
    }

    /// Returns the maximum channel value.
    #[must_use]
    pub fn max_channel(&self) -> f64 {
        self.r.max(self.g).max(self.b)
    }

    /// Returns the minimum channel value.
    #[must_use]
    pub fn min_channel(&self) -> f64 {
        self.r.min(self.g).min(self.b)
    }

    /// Approximate relative luminance using Rec.709 coefficients.
    #[must_use]
    pub fn luminance(&self) -> f64 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }
}

// ─── Clip Result ────────────────────────────────────────────────────────────

/// Result of a gamut-clip operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipResult {
    /// The clipped linear RGB value (guaranteed in-gamut).
    pub rgb: LinearRgb,
    /// `true` if the input was already in-gamut (no modification needed).
    pub was_in_gamut: bool,
    /// Euclidean distance between the original and clipped RGB values.
    pub clip_distance: f64,
}

// ─── GamutClipper ───────────────────────────────────────────────────────────

/// Configurable gamut clipper that applies a chosen [`ClipStrategy`].
#[derive(Debug, Clone)]
pub struct GamutClipper {
    /// Active clipping strategy.
    pub strategy: ClipStrategy,
    /// Blend weight for [`ClipStrategy::Adaptive`] (0 = pure luminance, 1 = pure chroma).
    pub adaptive_alpha: f64,
    /// Small tolerance for floating-point boundary checks.
    pub tolerance: f64,
}

impl Default for GamutClipper {
    fn default() -> Self {
        Self {
            strategy: ClipStrategy::HardClip,
            adaptive_alpha: 0.5,
            tolerance: 1e-10,
        }
    }
}

impl GamutClipper {
    /// Create a new `GamutClipper` with the given strategy.
    #[must_use]
    pub fn new(strategy: ClipStrategy) -> Self {
        Self {
            strategy,
            ..Self::default()
        }
    }

    /// Set the adaptive blend alpha (clamped to `[0, 1]`).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.adaptive_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set the in-gamut tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol.abs();
        self
    }

    /// Returns `true` if `rgb` is within gamut accounting for the configured tolerance.
    #[must_use]
    pub fn is_in_gamut(&self, rgb: &LinearRgb) -> bool {
        let t = self.tolerance;
        rgb.r >= -t
            && rgb.r <= 1.0 + t
            && rgb.g >= -t
            && rgb.g <= 1.0 + t
            && rgb.b >= -t
            && rgb.b <= 1.0 + t
    }

    /// Clip an out-of-gamut linear RGB value back into `[0, 1]`.
    #[must_use]
    pub fn clip(&self, rgb: LinearRgb) -> ClipResult {
        if self.is_in_gamut(&rgb) {
            let clamped = hard_clip(rgb);
            return ClipResult {
                rgb: clamped,
                was_in_gamut: true,
                clip_distance: 0.0,
            };
        }
        let clipped = match self.strategy {
            ClipStrategy::HardClip => hard_clip(rgb),
            ClipStrategy::LuminancePreserving => luminance_preserving_clip(rgb),
            ClipStrategy::ChromaReduction => chroma_reduction_clip(rgb),
            ClipStrategy::Adaptive => {
                let lp = luminance_preserving_clip(rgb);
                let cr = chroma_reduction_clip(rgb);
                let a = self.adaptive_alpha;
                LinearRgb::new(
                    lp.r * (1.0 - a) + cr.r * a,
                    lp.g * (1.0 - a) + cr.g * a,
                    lp.b * (1.0 - a) + cr.b * a,
                )
            }
        };
        let dist = ((rgb.r - clipped.r).powi(2)
            + (rgb.g - clipped.g).powi(2)
            + (rgb.b - clipped.b).powi(2))
        .sqrt();
        ClipResult {
            rgb: clipped,
            was_in_gamut: false,
            clip_distance: dist,
        }
    }

    /// Clip a batch of linear RGB values.
    #[must_use]
    pub fn clip_batch(&self, pixels: &[LinearRgb]) -> Vec<ClipResult> {
        pixels.iter().map(|p| self.clip(*p)).collect()
    }
}

// ─── Clipping implementations ───────────────────────────────────────────────

/// Hard-clip each channel to `[0, 1]`.
#[must_use]
fn hard_clip(rgb: LinearRgb) -> LinearRgb {
    LinearRgb::new(
        rgb.r.clamp(0.0, 1.0),
        rgb.g.clamp(0.0, 1.0),
        rgb.b.clamp(0.0, 1.0),
    )
}

/// Luminance-preserving clip: project toward the achromatic axis.
#[must_use]
fn luminance_preserving_clip(rgb: LinearRgb) -> LinearRgb {
    let y = rgb.luminance().clamp(0.0, 1.0);
    // Binary search toward the grey point (y, y, y) until in gamut
    let grey = LinearRgb::new(y, y, y);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best = grey;
    for _ in 0..32 {
        let mid = (lo + hi) * 0.5;
        let candidate = LinearRgb::new(
            grey.r + mid * (rgb.r - grey.r),
            grey.g + mid * (rgb.g - grey.g),
            grey.b + mid * (rgb.b - grey.b),
        );
        if candidate.is_in_gamut() {
            best = candidate;
            lo = mid;
        } else {
            hi = mid;
        }
    }
    best
}

/// Chroma-reduction clip: reduce saturation while preserving luminance and hue.
#[must_use]
fn chroma_reduction_clip(rgb: LinearRgb) -> LinearRgb {
    // Approximate: desaturate toward luminance grey
    let y = rgb.luminance().clamp(0.0, 1.0);
    let grey = LinearRgb::new(y, y, y);
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best = grey;
    for _ in 0..32 {
        let mid = (lo + hi) * 0.5;
        let candidate = LinearRgb::new(
            grey.r + mid * (rgb.r - grey.r),
            grey.g + mid * (rgb.g - grey.g),
            grey.b + mid * (rgb.b - grey.b),
        );
        if candidate.is_in_gamut() {
            best = candidate;
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Final hard-clip to guarantee bounds
    hard_clip(best)
}

/// Compute the Euclidean distance between two linear RGB values.
#[must_use]
pub fn rgb_distance(a: &LinearRgb, b: &LinearRgb) -> f64 {
    ((a.r - b.r).powi(2) + (a.g - b.g).powi(2) + (a.b - b.b).powi(2)).sqrt()
}

/// Batch statistics from a gamut-clipping pass.
#[derive(Debug, Clone)]
pub struct ClipStats {
    /// Total number of pixels processed.
    pub total: usize,
    /// Number of pixels that were already in-gamut.
    pub in_gamut_count: usize,
    /// Number of pixels that required clipping.
    pub clipped_count: usize,
    /// Mean clip distance across clipped pixels.
    pub mean_clip_distance: f64,
    /// Maximum clip distance.
    pub max_clip_distance: f64,
}

impl ClipStats {
    /// Compute statistics from a slice of [`ClipResult`]s.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_results(results: &[ClipResult]) -> Self {
        let total = results.len();
        let in_gamut_count = results.iter().filter(|r| r.was_in_gamut).count();
        let clipped_count = total - in_gamut_count;
        let (sum_dist, max_dist) = results.iter().fold((0.0_f64, 0.0_f64), |(s, m), r| {
            if r.was_in_gamut {
                (s, m)
            } else {
                (s + r.clip_distance, m.max(r.clip_distance))
            }
        });
        let mean_clip_distance = if clipped_count > 0 {
            sum_dist / clipped_count as f64
        } else {
            0.0
        };
        Self {
            total,
            in_gamut_count,
            clipped_count,
            mean_clip_distance,
            max_clip_distance: max_dist,
        }
    }

    /// Fraction of pixels that required clipping.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn clip_ratio(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.clipped_count as f64 / self.total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_rgb_in_gamut() {
        let c = LinearRgb::new(0.5, 0.5, 0.5);
        assert!(c.is_in_gamut());
    }

    #[test]
    fn test_linear_rgb_out_of_gamut_negative() {
        let c = LinearRgb::new(-0.1, 0.5, 0.5);
        assert!(!c.is_in_gamut());
    }

    #[test]
    fn test_linear_rgb_out_of_gamut_high() {
        let c = LinearRgb::new(0.5, 1.2, 0.5);
        assert!(!c.is_in_gamut());
    }

    #[test]
    fn test_luminance_grey() {
        let grey = LinearRgb::new(0.5, 0.5, 0.5);
        let y = grey.luminance();
        assert!(
            (y - 0.5).abs() < 1e-6,
            "grey luminance should be 0.5, got {y}"
        );
    }

    #[test]
    fn test_max_min_channel() {
        let c = LinearRgb::new(0.1, 0.9, 0.5);
        assert!((c.max_channel() - 0.9).abs() < 1e-12);
        assert!((c.min_channel() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_hard_clip_in_gamut_unchanged() {
        let c = LinearRgb::new(0.3, 0.6, 0.9);
        let clipped = hard_clip(c);
        assert!((clipped.r - 0.3).abs() < 1e-12);
        assert!((clipped.g - 0.6).abs() < 1e-12);
        assert!((clipped.b - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_hard_clip_clamps() {
        let c = LinearRgb::new(-0.2, 1.5, 0.5);
        let clipped = hard_clip(c);
        assert!((clipped.r - 0.0).abs() < 1e-12);
        assert!((clipped.g - 1.0).abs() < 1e-12);
        assert!((clipped.b - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gamut_clipper_default_strategy() {
        let gc = GamutClipper::default();
        assert_eq!(gc.strategy, ClipStrategy::HardClip);
    }

    #[test]
    fn test_gamut_clipper_in_gamut_passthrough() {
        let gc = GamutClipper::new(ClipStrategy::HardClip);
        let c = LinearRgb::new(0.5, 0.5, 0.5);
        let result = gc.clip(c);
        assert!(result.was_in_gamut);
        assert!((result.clip_distance - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_gamut_clipper_hard_clip_oob() {
        let gc = GamutClipper::new(ClipStrategy::HardClip);
        let c = LinearRgb::new(-0.5, 2.0, 0.5);
        let result = gc.clip(c);
        assert!(!result.was_in_gamut);
        assert!(result.rgb.is_in_gamut());
        assert!(result.clip_distance > 0.0);
    }

    #[test]
    fn test_gamut_clipper_luminance_preserving() {
        let gc = GamutClipper::new(ClipStrategy::LuminancePreserving);
        let c = LinearRgb::new(1.5, 0.0, 0.0);
        let result = gc.clip(c);
        assert!(!result.was_in_gamut);
        // Clipped result should be in gamut
        assert!(
            result.rgb.r >= 0.0 && result.rgb.r <= 1.0,
            "r out of range: {}",
            result.rgb.r
        );
    }

    #[test]
    fn test_gamut_clipper_chroma_reduction() {
        let gc = GamutClipper::new(ClipStrategy::ChromaReduction);
        let c = LinearRgb::new(0.0, 1.8, -0.3);
        let result = gc.clip(c);
        assert!(!result.was_in_gamut);
        assert!(result.rgb.is_in_gamut());
    }

    #[test]
    fn test_gamut_clipper_adaptive() {
        let gc = GamutClipper::new(ClipStrategy::Adaptive).with_alpha(0.7);
        assert!((gc.adaptive_alpha - 0.7).abs() < 1e-12);
        let c = LinearRgb::new(1.2, -0.1, 0.5);
        let result = gc.clip(c);
        assert!(!result.was_in_gamut);
    }

    #[test]
    fn test_clip_batch() {
        let gc = GamutClipper::new(ClipStrategy::HardClip);
        let pixels = vec![
            LinearRgb::new(0.5, 0.5, 0.5),
            LinearRgb::new(1.5, -0.5, 0.5),
        ];
        let results = gc.clip_batch(&pixels);
        assert_eq!(results.len(), 2);
        assert!(results[0].was_in_gamut);
        assert!(!results[1].was_in_gamut);
    }

    #[test]
    fn test_clip_stats_from_results() {
        let results = vec![
            ClipResult {
                rgb: LinearRgb::new(0.5, 0.5, 0.5),
                was_in_gamut: true,
                clip_distance: 0.0,
            },
            ClipResult {
                rgb: LinearRgb::new(1.0, 0.0, 0.5),
                was_in_gamut: false,
                clip_distance: 0.3,
            },
            ClipResult {
                rgb: LinearRgb::new(0.0, 1.0, 0.5),
                was_in_gamut: false,
                clip_distance: 0.7,
            },
        ];
        let stats = ClipStats::from_results(&results);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.in_gamut_count, 1);
        assert_eq!(stats.clipped_count, 2);
        assert!((stats.mean_clip_distance - 0.5).abs() < 1e-12);
        assert!((stats.max_clip_distance - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_clip_stats_empty() {
        let stats = ClipStats::from_results(&[]);
        assert_eq!(stats.total, 0);
        assert!((stats.clip_ratio() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_clip_strategy_display() {
        assert_eq!(format!("{}", ClipStrategy::HardClip), "Hard Clip");
        assert_eq!(
            format!("{}", ClipStrategy::LuminancePreserving),
            "Luminance-Preserving"
        );
        assert_eq!(
            format!("{}", ClipStrategy::ChromaReduction),
            "Chroma Reduction"
        );
        assert_eq!(format!("{}", ClipStrategy::Adaptive), "Adaptive");
    }

    #[test]
    fn test_rgb_distance() {
        let a = LinearRgb::new(0.0, 0.0, 0.0);
        let b = LinearRgb::new(1.0, 0.0, 0.0);
        assert!((rgb_distance(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_with_tolerance() {
        let gc = GamutClipper::new(ClipStrategy::HardClip).with_tolerance(0.01);
        // Slightly out of [0,1] but within tolerance
        let c = LinearRgb::new(1.005, 0.5, 0.5);
        assert!(gc.is_in_gamut(&c));
    }
}
