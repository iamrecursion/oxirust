#![allow(dead_code)]
//! Film grain restoration and matching for degraded media.
//!
//! This module provides tools for analyzing, extracting, and restoring
//! film grain patterns from degraded video and photographic sources.
//! Grain characteristics (size, intensity, color bias) are estimated
//! from reference regions and applied to degrained footage for a
//! natural, film-like appearance.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Statistical profile of film grain extracted from a reference region.
#[derive(Debug, Clone)]
pub struct GrainProfile {
    /// Mean grain intensity (0.0 = no grain, 1.0 = maximum).
    pub mean_intensity: f64,
    /// Standard deviation of grain intensity.
    pub std_dev: f64,
    /// Average grain particle size in pixels.
    pub avg_grain_size: f64,
    /// Colour bias of the grain (R, G, B offsets).
    pub color_bias: [f64; 3],
    /// Number of analysis frames that contributed.
    pub frame_count: usize,
}

/// Configuration for grain analysis.
#[derive(Debug, Clone)]
pub struct GrainAnalysisConfig {
    /// Block size (pixels) for local analysis.
    pub block_size: usize,
    /// Overlap between adjacent blocks (pixels).
    pub overlap: usize,
    /// Minimum variance to consider a block "grainy".
    pub min_variance: f64,
    /// Maximum variance — above this we likely have edges, not grain.
    pub max_variance: f64,
}

/// A single grain particle descriptor.
#[derive(Debug, Clone, Copy)]
pub struct GrainParticle {
    /// X position (normalised 0..1).
    pub x: f64,
    /// Y position (normalised 0..1).
    pub y: f64,
    /// Radius in pixels.
    pub radius: f64,
    /// Intensity (-1..1, where negative = darker grain, positive = lighter).
    pub intensity: f64,
}

/// Strategy for restoring grain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainRestoreStrategy {
    /// Re-synthesise grain from a statistical profile.
    Synthesize,
    /// Transfer grain from a reference frame.
    Transfer,
    /// Blend synthesised and transferred grain.
    Hybrid,
}

/// Result of a grain restoration pass.
#[derive(Debug, Clone)]
pub struct GrainRestoreResult {
    /// Restored pixel values.
    pub pixels: Vec<f64>,
    /// Width of the restored image.
    pub width: usize,
    /// Height of the restored image.
    pub height: usize,
    /// Profile used for restoration.
    pub profile: GrainProfile,
    /// Mean-squared error between original and restored grain region.
    pub mse: f64,
}

/// Running estimator that accumulates grain statistics over multiple frames.
#[derive(Debug, Clone)]
pub struct GrainEstimator {
    /// Window of recent intensity values.
    window: VecDeque<f64>,
    /// Maximum window length.
    max_window: usize,
    /// Running sum.
    sum: f64,
    /// Running sum of squares.
    sum_sq: f64,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for GrainAnalysisConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            overlap: 4,
            min_variance: 0.001,
            max_variance: 0.15,
        }
    }
}

impl GrainProfile {
    /// Create a flat (no-grain) profile.
    pub fn flat() -> Self {
        Self {
            mean_intensity: 0.0,
            std_dev: 0.0,
            avg_grain_size: 0.0,
            color_bias: [0.0; 3],
            frame_count: 0,
        }
    }

    /// Compute a blended profile from two profiles with a given mix factor (0..1).
    #[allow(clippy::cast_precision_loss)]
    pub fn blend(a: &Self, b: &Self, mix: f64) -> Self {
        let inv = 1.0 - mix;
        Self {
            mean_intensity: a.mean_intensity * inv + b.mean_intensity * mix,
            std_dev: a.std_dev * inv + b.std_dev * mix,
            avg_grain_size: a.avg_grain_size * inv + b.avg_grain_size * mix,
            color_bias: [
                a.color_bias[0] * inv + b.color_bias[0] * mix,
                a.color_bias[1] * inv + b.color_bias[1] * mix,
                a.color_bias[2] * inv + b.color_bias[2] * mix,
            ],
            frame_count: a.frame_count + b.frame_count,
        }
    }

    /// Return the dominant colour channel of the grain bias.
    pub fn dominant_channel(&self) -> usize {
        let mut idx = 0usize;
        let mut best = self.color_bias[0].abs();
        for (i, &v) in self.color_bias.iter().enumerate().skip(1) {
            if v.abs() > best {
                best = v.abs();
                idx = i;
            }
        }
        idx
    }
}

impl GrainEstimator {
    /// Create a new estimator with a given window size.
    pub fn new(max_window: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_window),
            max_window,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a new intensity sample.
    pub fn push(&mut self, value: f64) {
        if self.window.len() == self.max_window {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Current mean intensity.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.sum / self.window.len() as f64
    }

    /// Current variance.
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> f64 {
        let n = self.window.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        self.sum_sq / n as f64 - mean * mean
    }

    /// Current standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Number of samples in the window.
    pub fn count(&self) -> usize {
        self.window.len()
    }
}

/// Analyse a flat pixel buffer and produce a `GrainProfile`.
#[allow(clippy::cast_precision_loss)]
pub fn analyse_grain(
    pixels: &[f64],
    width: usize,
    height: usize,
    config: &GrainAnalysisConfig,
) -> GrainProfile {
    if pixels.is_empty() || width == 0 || height == 0 {
        return GrainProfile::flat();
    }

    let step = config.block_size.saturating_sub(config.overlap).max(1);
    let mut intensities: Vec<f64> = Vec::new();

    let mut by = 0;
    while by + config.block_size <= height {
        let mut bx = 0;
        while bx + config.block_size <= width {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0usize;
            for row in by..by + config.block_size {
                for col in bx..bx + config.block_size {
                    let idx = row * width + col;
                    if idx < pixels.len() {
                        let v = pixels[idx];
                        sum += v;
                        sum_sq += v * v;
                        count += 1;
                    }
                }
            }
            if count > 1 {
                let mean = sum / count as f64;
                let var = (sum_sq / count as f64) - mean * mean;
                if var >= config.min_variance && var <= config.max_variance {
                    intensities.push(var.sqrt());
                }
            }
            bx += step;
        }
        by += step;
    }

    if intensities.is_empty() {
        return GrainProfile::flat();
    }

    let n = intensities.len() as f64;
    let mean_i: f64 = intensities.iter().sum::<f64>() / n;
    let var_i: f64 = intensities
        .iter()
        .map(|v| (v - mean_i).powi(2))
        .sum::<f64>()
        / n;

    GrainProfile {
        mean_intensity: mean_i,
        std_dev: var_i.sqrt(),
        avg_grain_size: 1.5, // heuristic default
        color_bias: [0.0; 3],
        frame_count: 1,
    }
}

/// Synthesise grain noise and add to a pixel buffer.
#[allow(clippy::cast_precision_loss)]
pub fn apply_grain(pixels: &[f64], profile: &GrainProfile, seed: u64) -> Vec<f64> {
    // Simple LCG-based deterministic noise
    let mut state = seed;
    pixels
        .iter()
        .map(|&p| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = ((state >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            (p + noise * profile.mean_intensity).clamp(0.0, 1.0)
        })
        .collect()
}

/// Compute mean-squared error between two buffers.
pub fn mse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::NAN;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = a.len() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_profile() {
        let p = GrainProfile::flat();
        assert_eq!(p.mean_intensity, 0.0);
        assert_eq!(p.std_dev, 0.0);
        assert_eq!(p.frame_count, 0);
    }

    #[test]
    fn test_blend_profiles() {
        let a = GrainProfile {
            mean_intensity: 0.2,
            std_dev: 0.1,
            avg_grain_size: 1.0,
            color_bias: [0.1, 0.0, 0.0],
            frame_count: 1,
        };
        let b = GrainProfile {
            mean_intensity: 0.8,
            std_dev: 0.3,
            avg_grain_size: 3.0,
            color_bias: [0.0, 0.0, 0.2],
            frame_count: 1,
        };
        let c = GrainProfile::blend(&a, &b, 0.5);
        assert!((c.mean_intensity - 0.5).abs() < 1e-9);
        assert!((c.avg_grain_size - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_dominant_channel_red() {
        let p = GrainProfile {
            mean_intensity: 0.5,
            std_dev: 0.1,
            avg_grain_size: 1.0,
            color_bias: [0.5, 0.1, -0.2],
            frame_count: 1,
        };
        assert_eq!(p.dominant_channel(), 0);
    }

    #[test]
    fn test_dominant_channel_blue() {
        let p = GrainProfile {
            mean_intensity: 0.5,
            std_dev: 0.1,
            avg_grain_size: 1.0,
            color_bias: [0.1, 0.1, -0.9],
            frame_count: 1,
        };
        assert_eq!(p.dominant_channel(), 2);
    }

    #[test]
    fn test_estimator_empty() {
        let e = GrainEstimator::new(10);
        assert_eq!(e.mean(), 0.0);
        assert_eq!(e.variance(), 0.0);
        assert_eq!(e.count(), 0);
    }

    #[test]
    fn test_estimator_push_mean() {
        let mut e = GrainEstimator::new(10);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            e.push(v);
        }
        assert!((e.mean() - 3.0).abs() < 1e-9);
        assert_eq!(e.count(), 5);
    }

    #[test]
    fn test_estimator_window_eviction() {
        let mut e = GrainEstimator::new(3);
        for v in [10.0, 20.0, 30.0, 40.0] {
            e.push(v);
        }
        // Window should be [20, 30, 40]
        assert_eq!(e.count(), 3);
        assert!((e.mean() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimator_std_dev() {
        let mut e = GrainEstimator::new(100);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            e.push(v);
        }
        assert!(e.std_dev() > 0.0);
    }

    #[test]
    fn test_analyse_grain_empty() {
        let p = analyse_grain(&[], 0, 0, &GrainAnalysisConfig::default());
        assert_eq!(p.mean_intensity, 0.0);
    }

    #[test]
    fn test_analyse_grain_uniform() {
        // Uniform block has zero variance -> flat profile
        let pixels = vec![0.5; 256];
        let p = analyse_grain(&pixels, 16, 16, &GrainAnalysisConfig::default());
        assert_eq!(p.mean_intensity, 0.0);
    }

    #[test]
    fn test_apply_grain_length() {
        let pixels = vec![0.5; 100];
        let profile = GrainProfile {
            mean_intensity: 0.05,
            std_dev: 0.01,
            avg_grain_size: 1.0,
            color_bias: [0.0; 3],
            frame_count: 1,
        };
        let out = apply_grain(&pixels, &profile, 42);
        assert_eq!(out.len(), pixels.len());
    }

    #[test]
    fn test_apply_grain_clamped() {
        let pixels = vec![0.99; 50];
        let profile = GrainProfile {
            mean_intensity: 0.5,
            std_dev: 0.1,
            avg_grain_size: 1.0,
            color_bias: [0.0; 3],
            frame_count: 1,
        };
        let out = apply_grain(&pixels, &profile, 7);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_mse_identical() {
        let a = vec![0.1, 0.2, 0.3];
        assert!((mse(&a, &a)).abs() < 1e-12);
    }

    #[test]
    fn test_mse_different() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        assert!((mse(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mse_mismatched_len() {
        let a = vec![0.0];
        let b = vec![0.0, 1.0];
        assert!(mse(&a, &b).is_nan());
    }

    #[test]
    fn test_default_config() {
        let c = GrainAnalysisConfig::default();
        assert_eq!(c.block_size, 16);
        assert_eq!(c.overlap, 4);
    }
}
