//! Complexity histogram-based scene detection.
//!
//! This module provides two primary abstractions:
//!
//! * [`ComplexityHistogramDetector`] — computes a *frame complexity score* (0–1)
//!   from the standard deviation of the pixel-intensity histogram.  High standard
//!   deviation indicates a broadly distributed (rich/complex) image; low standard
//!   deviation indicates a uniform (simple/dark/bright) image.
//!
//! * [`AdaptiveThreshold`] — maintains a rolling average of recent complexity
//!   scores and uses it to dynamically choose a scene-cut detection threshold,
//!   lowering the threshold for simple content (where even small changes matter)
//!   and raising it for complex content (where large changes are normal).
//!
//! # Algorithm
//!
//! ## Complexity score
//!
//! For an 8-bit grayscale image (or the luma channel of an RGB frame) with pixel
//! values `p₀…pₙ₋₁`:
//!
//! 1. Build a 256-bin histogram `H[0..256]`.
//! 2. Normalise to a probability mass function: `h[i] = H[i] / N`.
//! 3. Compute the mean bin index:   `μ = Σ i·h[i]`
//! 4. Compute the variance:         `σ² = Σ (i−μ)²·h[i]`
//! 5. Compute the standard deviation: `σ = √σ²`
//! 6. Normalise to \[0,1\]: `score = σ / 127.5`  (127.5 is the max possible σ for
//!    a two-spike distribution at 0 and 255 with equal weight).
//!
//! ## Adaptive threshold
//!
//! Given a rolling window of `N` complexity scores:
//!
//! ```text
//! mean_c = mean(window)
//! threshold = min_threshold + mean_c × (max_threshold − min_threshold)
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_scene::complexity_detector::{ComplexityHistogramDetector, AdaptiveThreshold};
//!
//! let detector = ComplexityHistogramDetector::new();
//! let width = 4;
//! let height = 4;
//!
//! // All-black frame — zero complexity
//! let black: Vec<u8> = vec![0u8; (width * height) as usize * 3];
//! let score_black = detector.detect(&black, width, height);
//! assert!(score_black < 0.05);
//!
//! // Checker-board (alternating 0 / 255) — maximum complexity
//! let checker: Vec<u8> = (0..(width * height) as usize)
//!     .flat_map(|i| {
//!         let v: u8 = if i % 2 == 0 { 0 } else { 255 };
//!         [v, v, v]
//!     })
//!     .collect();
//! let score_checker = detector.detect(&checker, width, height);
//! assert!(score_checker > 0.5);
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// ComplexityHistogramDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Computes a normalised frame complexity score in \[0, 1\] from the standard
/// deviation of the pixel-intensity histogram.
///
/// **Input format**: `frame` must be an interleaved RGB byte slice of length
/// `width × height × 3`.  The luma (Y) value is derived via the BT.601
/// coefficients:
///
/// ```text
/// Y = 0.299·R + 0.587·G + 0.114·B
/// ```
///
/// For single-channel (grayscale) frames pass `width × height` bytes; the
/// detector auto-detects grayscale input when `frame.len() == width * height`.
#[derive(Debug, Clone, Default)]
pub struct ComplexityHistogramDetector {
    /// Number of histogram bins (fixed at 256 for 8-bit intensity).
    bins: usize,
}

impl ComplexityHistogramDetector {
    /// Create a new detector with 256-bin intensity histogram.
    #[must_use]
    pub fn new() -> Self {
        Self { bins: 256 }
    }

    /// Compute the frame complexity score.
    ///
    /// Returns a value in \[0, 1\] where 0 = perfectly uniform frame (single
    /// colour) and 1 = maximum possible spread (equal weight at intensity 0
    /// and 255).
    ///
    /// Returns `0.0` for empty frames.
    #[must_use]
    pub fn detect(&self, frame: &[u8], width: u32, height: u32) -> f32 {
        let n_pixels = (width as usize) * (height as usize);
        if n_pixels == 0 || frame.is_empty() {
            return 0.0;
        }

        // Build luma histogram
        let histogram = self.build_histogram(frame, n_pixels);

        // Compute complexity as normalised standard deviation of the histogram
        self.complexity_from_histogram(&histogram, n_pixels)
    }

    /// Build a 256-bin intensity histogram from a frame.
    ///
    /// Supports both grayscale (1 channel) and RGB (3 channels) input.
    fn build_histogram(&self, frame: &[u8], n_pixels: usize) -> [u64; 256] {
        let mut hist = [0u64; 256];

        if frame.len() == n_pixels {
            // Grayscale: one byte per pixel
            for &byte in frame.iter().take(n_pixels) {
                hist[byte as usize] += 1;
            }
        } else {
            // RGB (or larger): compute BT.601 luma
            let samples = frame.len() / 3;
            let pixels = samples.min(n_pixels);
            for i in 0..pixels {
                let r = frame[i * 3] as f32;
                let g = frame[i * 3 + 1] as f32;
                let b = frame[i * 3 + 2] as f32;
                let y = (0.299 * r + 0.587 * g + 0.114 * b).round() as usize;
                let y = y.min(255);
                hist[y] += 1;
            }
        }

        hist
    }

    /// Compute normalised complexity from a 256-bin histogram.
    ///
    /// Uses the standard deviation of the luma distribution, normalised by
    /// the theoretical maximum standard deviation (127.5 for a bimodal
    /// distribution equally split between intensity 0 and 255).
    fn complexity_from_histogram(&self, histogram: &[u64; 256], n_pixels: usize) -> f32 {
        if n_pixels == 0 {
            return 0.0;
        }

        let n = n_pixels as f64;

        // Weighted mean intensity
        let mean: f64 = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| i as f64 * count as f64)
            .sum::<f64>()
            / n;

        // Variance of intensity
        let variance: f64 = histogram
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let diff = i as f64 - mean;
                diff * diff * count as f64
            })
            .sum::<f64>()
            / n;

        let std_dev = variance.sqrt();

        // Maximum possible standard deviation for 8-bit intensity:
        // equal mass at 0 and 255 → mean = 127.5, σ = 127.5
        let max_std_dev = 127.5_f64;

        ((std_dev / max_std_dev) as f32).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveThreshold
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamically adjusts a scene-cut threshold based on a rolling average of
/// recent frame complexity scores.
///
/// High complexity → higher threshold (the scene is already "busy", so small
/// inter-frame differences are expected and should not trigger a cut).
///
/// Low complexity → lower threshold (uniform/dark content, so even subtle
/// changes deserve detection).
///
/// # Formula
///
/// ```text
/// threshold(t) = min_threshold + mean_complexity(window) × (max_threshold − min_threshold)
/// ```
///
/// When fewer than `window_size` frames have been pushed the available scores
/// are used; the threshold falls back to `base_threshold` for an empty window.
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Lower bound for the detection threshold (used for simple/dark content).
    pub min_threshold: f32,
    /// Upper bound for the detection threshold (used for complex/rich content).
    pub max_threshold: f32,
    /// Threshold returned when the window is empty.
    pub base_threshold: f32,
    /// Maximum number of complexity scores retained in the rolling window.
    pub window_size: usize,
    /// Rolling window of recent complexity scores.
    window: VecDeque<f32>,
}

impl AdaptiveThreshold {
    /// Create a new `AdaptiveThreshold` with the given bounds and window size.
    ///
    /// # Panics
    ///
    /// Panics if `min_threshold > max_threshold` or `window_size == 0`.
    #[must_use]
    pub fn new(min_threshold: f32, max_threshold: f32, window_size: usize) -> Self {
        assert!(
            min_threshold <= max_threshold,
            "min_threshold must be ≤ max_threshold"
        );
        assert!(window_size > 0, "window_size must be > 0");
        let base = (min_threshold + max_threshold) / 2.0;
        Self {
            min_threshold,
            max_threshold,
            base_threshold: base,
            window_size,
            window: VecDeque::with_capacity(window_size),
        }
    }

    /// Create with all defaults: min=0.2, max=0.5, window=30.
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(0.2, 0.5, 30)
    }

    /// Push a new complexity score into the rolling window.
    ///
    /// If the window is full the oldest score is evicted.
    pub fn push_complexity(&mut self, score: f32) {
        if self.window.len() == self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(score.clamp(0.0, 1.0));
    }

    /// Compute the current adaptive threshold from the rolling window mean.
    ///
    /// Returns `base_threshold` when the window is empty.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        if self.window.is_empty() {
            return self.base_threshold;
        }
        let mean: f32 = self.window.iter().sum::<f32>() / self.window.len() as f32;
        let t = self.min_threshold + mean * (self.max_threshold - self.min_threshold);
        t.clamp(self.min_threshold, self.max_threshold)
    }

    /// Determine whether a given inter-frame difference constitutes a scene cut.
    ///
    /// Returns `true` when `diff >= threshold()`.
    #[must_use]
    pub fn is_scene_cut(&self, diff: f32) -> bool {
        diff >= self.threshold()
    }

    /// Number of scores currently in the window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Clear the rolling window.
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::default_params()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Generate an RGB frame filled with a constant grey value.
    fn constant_rgb(value: u8, width: u32, height: u32) -> Vec<u8> {
        vec![value; (width * height * 3) as usize]
    }

    /// Generate an RGB frame with alternating black and white pixels.
    fn checkerboard_rgb(width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        (0..n)
            .flat_map(|i| {
                let v: u8 = if i % 2 == 0 { 0 } else { 255 };
                [v, v, v]
            })
            .collect()
    }

    /// Generate a gradient grayscale frame (0..255, repeating).
    fn gradient_gray(n_pixels: usize) -> Vec<u8> {
        (0..n_pixels).map(|i| (i % 256) as u8).collect()
    }

    // ── ComplexityHistogramDetector ───────────────────────────────────────────

    #[test]
    fn test_black_frame_is_zero_complexity() {
        let det = ComplexityHistogramDetector::new();
        let frame = constant_rgb(0, 16, 16);
        let score = det.detect(&frame, 16, 16);
        assert!(
            score < 1e-6,
            "black frame complexity should be 0, got {score}"
        );
    }

    #[test]
    fn test_white_frame_is_zero_complexity() {
        let det = ComplexityHistogramDetector::new();
        let frame = constant_rgb(255, 16, 16);
        let score = det.detect(&frame, 16, 16);
        assert!(
            score < 1e-6,
            "white frame complexity should be 0, got {score}"
        );
    }

    #[test]
    fn test_checkerboard_has_high_complexity() {
        let det = ComplexityHistogramDetector::new();
        let frame = checkerboard_rgb(8, 8);
        let score = det.detect(&frame, 8, 8);
        assert!(
            score > 0.9,
            "checkerboard complexity should be near 1, got {score}"
        );
    }

    #[test]
    fn test_complex_frame_more_complex_than_constant() {
        let det = ComplexityHistogramDetector::new();
        let complex = gradient_gray(1024);
        let constant = vec![128u8; 1024];

        let complex_score = det.detect(&complex, 32, 32);
        let constant_score = det.detect(&constant, 32, 32);

        assert!(
            complex_score > constant_score,
            "gradient should be more complex ({}) than constant ({})",
            complex_score,
            constant_score
        );
    }

    #[test]
    fn test_empty_frame_returns_zero() {
        let det = ComplexityHistogramDetector::new();
        let score = det.detect(&[], 0, 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_grayscale_single_channel() {
        let det = ComplexityHistogramDetector::new();
        // One-channel: len == width*height
        let gray = gradient_gray(256);
        let score = det.detect(&gray, 16, 16);
        assert!(score > 0.0, "gradient gray should have non-zero complexity");
    }

    #[test]
    fn test_score_in_unit_range() {
        let det = ComplexityHistogramDetector::new();
        let checker = checkerboard_rgb(32, 32);
        let score = det.detect(&checker, 32, 32);
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1], got {score}"
        );
    }

    // ── Scene transition detection ────────────────────────────────────────────

    #[test]
    fn test_black_to_white_transition_is_detectable() {
        // Simulate a black→white scene transition:
        // - black frames have low complexity and low luma mean
        // - white frames have low complexity but completely different luma
        // The *inter-frame difference* (in luma mean) is large; verify that
        // the adaptive threshold correctly allows detection.

        let det = ComplexityHistogramDetector::new();
        let mut adaptive = AdaptiveThreshold::new(0.1, 0.5, 5);

        let black = constant_rgb(0, 16, 16);
        let _white = constant_rgb(255, 16, 16);

        // Push several black frames into the complexity window
        for _ in 0..5 {
            let s = det.detect(&black, 16, 16);
            adaptive.push_complexity(s);
        }

        // The threshold should be low because complexity is near zero
        let threshold = adaptive.threshold();
        assert!(
            threshold < 0.3,
            "threshold for black frames should be low, got {threshold}"
        );

        // Compute a normalised luma-difference between black and white
        // Δ = |mean_white - mean_black| / 255 = |255 - 0| / 255 = 1.0
        let luma_diff = 1.0_f32;
        assert!(
            adaptive.is_scene_cut(luma_diff),
            "black→white transition must be detected as a scene cut (threshold={threshold})"
        );
    }

    // ── AdaptiveThreshold ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_window_returns_base_threshold() {
        let at = AdaptiveThreshold::new(0.2, 0.6, 10);
        assert!(
            (at.threshold() - at.base_threshold).abs() < 1e-6,
            "empty window must return base_threshold"
        );
    }

    #[test]
    fn test_high_complexity_raises_threshold() {
        let mut at = AdaptiveThreshold::new(0.1, 0.6, 5);
        for _ in 0..5 {
            at.push_complexity(1.0); // maximum complexity
        }
        assert!(
            at.threshold() > 0.5,
            "max complexity should push threshold to max ({:.3})",
            at.threshold()
        );
    }

    #[test]
    fn test_low_complexity_lowers_threshold() {
        let mut at = AdaptiveThreshold::new(0.1, 0.6, 5);
        for _ in 0..5 {
            at.push_complexity(0.0); // minimum complexity
        }
        assert!(
            (at.threshold() - 0.1).abs() < 1e-6,
            "zero complexity should keep threshold at min, got {:.4}",
            at.threshold()
        );
    }

    #[test]
    fn test_window_eviction_respects_window_size() {
        let mut at = AdaptiveThreshold::new(0.0, 1.0, 3);
        for _ in 0..10 {
            at.push_complexity(0.5);
        }
        assert_eq!(at.window_len(), 3, "window must be capped at window_size");
    }

    #[test]
    fn test_is_scene_cut_above_threshold() {
        let at = AdaptiveThreshold::new(0.3, 0.7, 5); // base = 0.5
        assert!(at.is_scene_cut(0.6));
    }

    #[test]
    fn test_is_not_scene_cut_below_threshold() {
        let at = AdaptiveThreshold::new(0.3, 0.7, 5); // base = 0.5
        assert!(!at.is_scene_cut(0.4));
    }

    #[test]
    fn test_reset_clears_window() {
        let mut at = AdaptiveThreshold::new(0.2, 0.8, 10);
        for i in 0..5 {
            at.push_complexity(i as f32 * 0.1);
        }
        at.reset();
        assert_eq!(at.window_len(), 0);
        assert!(
            (at.threshold() - at.base_threshold).abs() < 1e-6,
            "after reset threshold must revert to base"
        );
    }

    #[test]
    fn test_threshold_interpolation() {
        // With window of complexity 0.5 exactly, threshold should be midpoint.
        let mut at = AdaptiveThreshold::new(0.0, 1.0, 5);
        for _ in 0..5 {
            at.push_complexity(0.5);
        }
        let t = at.threshold();
        assert!(
            (t - 0.5).abs() < 1e-5,
            "complexity=0.5 with range [0,1] should give threshold=0.5, got {t}"
        );
    }
}
