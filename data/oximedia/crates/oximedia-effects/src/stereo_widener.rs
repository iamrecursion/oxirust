//! Stereo field widening and narrowing effect.
//!
//! This module manipulates the perceived stereo width of an audio signal
//! using mid/side processing. The [`StereoWidener`] converts a stereo pair
//! to M/S representation, adjusts the side level, and converts back.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::stereo_widener::{StereoWidener, WidenerMode};
//!
//! let mut w = StereoWidener::new(WidenerMode::MidSide, 1.5);
//! let (l, r) = w.process_sample(0.5, -0.3);
//! assert!(l.is_finite());
//! assert!(r.is_finite());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Simple stereo widener configuration with a single `width` parameter.
///
/// This is a lightweight alternative to [`WidenerConfig`] for callers that
/// only need to set the width factor.
///
/// * `0.0` = mono (full mid, no side)
/// * `1.0` = original stereo width (identity)
/// * `2.0` = exaggerated wide stereo
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoWidenerConfig {
    /// Width factor: 0.0 = mono, 1.0 = normal, 2.0 = extra wide.
    pub width: f32,
}

impl Default for StereoWidenerConfig {
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

impl StereoWidenerConfig {
    /// Create a new config with the given width (clamped to ≥ 0.0).
    #[must_use]
    pub fn new(width: f32) -> Self {
        Self {
            width: width.max(0.0),
        }
    }

    /// Mono preset (width = 0.0).
    #[must_use]
    pub fn mono() -> Self {
        Self { width: 0.0 }
    }

    /// Normal stereo preset (width = 1.0, identity).
    #[must_use]
    pub fn normal() -> Self {
        Self { width: 1.0 }
    }

    /// Extra-wide stereo preset (width = 2.0).
    #[must_use]
    pub fn wide() -> Self {
        Self { width: 2.0 }
    }
}

/// Algorithm used for stereo width manipulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WidenerMode {
    /// Classic mid/side balance. Width 1.0 = original, >1 = wider, <1 = narrower.
    MidSide,
    /// Haas-effect delay (adds a short delay to one channel).
    HaasDelay,
    /// Phase-based widening using a small pitch-shift on one channel.
    PhaseSpread,
}

/// Configuration for the stereo widener.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WidenerConfig {
    /// Processing mode.
    pub mode: WidenerMode,
    /// Width factor. 1.0 = unchanged, 0.0 = mono, 2.0 = exaggerated stereo.
    pub width: f32,
    /// Compensation gain to keep loudness consistent, 0.0 -- 2.0.
    pub compensation: f32,
}

impl Default for WidenerConfig {
    fn default() -> Self {
        Self {
            mode: WidenerMode::MidSide,
            width: 1.0,
            compensation: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Stereo widener
// ---------------------------------------------------------------------------

/// Processes a stereo signal to adjust perceived width.
#[derive(Debug, Clone)]
pub struct StereoWidener {
    mode: WidenerMode,
    width: f32,
    compensation: f32,
    // Haas delay buffer (simple ring buffer for one channel).
    haas_buffer: Vec<f32>,
    haas_write: usize,
    haas_delay: usize,
}

impl StereoWidener {
    /// Create a new widener with the given mode and width factor.
    #[must_use]
    pub fn new(mode: WidenerMode, width: f32) -> Self {
        let haas_delay = 20; // ~0.4 ms at 48 kHz, a subtle Haas effect
        Self {
            mode,
            width: width.max(0.0),
            compensation: 1.0,
            haas_buffer: vec![0.0; 512],
            haas_write: 0,
            haas_delay,
        }
    }

    /// Create from a config struct.
    #[must_use]
    pub fn from_config(config: WidenerConfig) -> Self {
        let mut w = Self::new(config.mode, config.width);
        w.compensation = config.compensation.clamp(0.0, 2.0);
        w
    }

    /// Set the width factor.
    pub fn set_width(&mut self, width: f32) {
        self.width = width.max(0.0);
    }

    /// Return the current width.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Set the mode.
    pub fn set_mode(&mut self, mode: WidenerMode) {
        self.mode = mode;
    }

    /// Return the current mode.
    #[must_use]
    pub fn mode(&self) -> WidenerMode {
        self.mode
    }

    /// Process one stereo sample pair, returning `(left, right)`.
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        match self.mode {
            WidenerMode::MidSide => self.process_mid_side(left, right),
            WidenerMode::HaasDelay => self.process_haas(left, right),
            WidenerMode::PhaseSpread => self.process_phase_spread(left, right),
        }
    }

    /// Process buffers in-place.
    pub fn process_buffers(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let (l, r) = self.process_sample(left[i], right[i]);
            left[i] = l;
            right[i] = r;
        }
    }

    /// Process owned slices, returning new `(left_out, right_out)` vectors.
    ///
    /// Only up to `min(left.len(), right.len())` samples are processed.
    pub fn process(&mut self, left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = left.len().min(right.len());
        let mut l_out = vec![0.0f32; n];
        let mut r_out = vec![0.0f32; n];
        for i in 0..n {
            let (l, r) = self.process_sample(left[i], right[i]);
            l_out[i] = l;
            r_out[i] = r;
        }
        (l_out, r_out)
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.haas_buffer.iter_mut().for_each(|s| *s = 0.0);
        self.haas_write = 0;
    }

    // -- internal algorithms ------------------------------------------------

    fn process_mid_side(&self, left: f32, right: f32) -> (f32, f32) {
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        let new_side = side * self.width;
        let l = (mid + new_side) * self.compensation;
        let r = (mid - new_side) * self.compensation;
        (l, r)
    }

    fn process_haas(&mut self, left: f32, right: f32) -> (f32, f32) {
        // Write current right channel into delay buffer
        let _mask = self.haas_buffer.len() - 1; // assume power of two is close enough
        let write_pos = self.haas_write % self.haas_buffer.len();
        self.haas_buffer[write_pos] = right;
        self.haas_write += 1;

        // Read delayed sample
        let read_pos =
            (self.haas_write + self.haas_buffer.len() - self.haas_delay) % self.haas_buffer.len();
        let delayed = self.haas_buffer[read_pos];

        let blend = (self.width - 1.0).clamp(0.0, 1.0);
        let r_out = right * (1.0 - blend) + delayed * blend;
        (left * self.compensation, r_out * self.compensation)
    }

    fn process_phase_spread(&self, left: f32, right: f32) -> (f32, f32) {
        // Simple approximation: scale the difference channel.
        let mid = (left + right) * 0.5;
        let side = (left - right) * 0.5;
        let spread = side * self.width;
        let l = (mid + spread) * self.compensation;
        let r = (mid - spread) * self.compensation;
        (l, r)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mid_side_unity_width() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        let (l, r) = w.process_sample(0.8, -0.4);
        // Width 1.0 should reproduce the input
        assert!((l - 0.8).abs() < 1e-5);
        assert!((r - (-0.4)).abs() < 1e-5);
    }

    #[test]
    fn test_mid_side_mono_collapse() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 0.0);
        let (l, r) = w.process_sample(1.0, -1.0);
        // Width 0 -> mono: L=R=mid
        assert!((l - r).abs() < 1e-6);
    }

    #[test]
    fn test_mid_side_wider_than_original() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 2.0);
        let (l, r) = w.process_sample(0.5, -0.5);
        // Side component should be doubled
        let orig_side = (0.5 - (-0.5)) * 0.5; // = 0.5
        let new_side = (l - r) * 0.5;
        assert!((new_side - orig_side * 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_haas_mode_runs() {
        let mut w = StereoWidener::new(WidenerMode::HaasDelay, 1.5);
        for _ in 0..100 {
            let (l, r) = w.process_sample(0.3, 0.3);
            assert!(l.is_finite());
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_phase_spread_runs() {
        let mut w = StereoWidener::new(WidenerMode::PhaseSpread, 1.2);
        let (l, r) = w.process_sample(0.6, -0.2);
        assert!(l.is_finite());
        assert!(r.is_finite());
    }

    #[test]
    fn test_set_width() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        w.set_width(2.5);
        assert!((w.width() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_width_non_negative() {
        let w = StereoWidener::new(WidenerMode::MidSide, -1.0);
        assert!(w.width() >= 0.0);
    }

    #[test]
    fn test_set_mode() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        w.set_mode(WidenerMode::HaasDelay);
        assert_eq!(w.mode(), WidenerMode::HaasDelay);
    }

    #[test]
    fn test_process_buffers() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        let mut left = vec![0.5; 16];
        let mut right = vec![-0.5; 16];
        w.process_buffers(&mut left, &mut right);
        for (l, r) in left.iter().zip(right.iter()) {
            assert!(l.is_finite());
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut w = StereoWidener::new(WidenerMode::HaasDelay, 1.5);
        for _ in 0..50 {
            w.process_sample(0.4, 0.4);
        }
        w.reset();
        assert_eq!(w.haas_write, 0);
    }

    #[test]
    fn test_from_config() {
        let config = WidenerConfig {
            mode: WidenerMode::PhaseSpread,
            width: 1.8,
            compensation: 0.9,
        };
        let w = StereoWidener::from_config(config);
        assert_eq!(w.mode(), WidenerMode::PhaseSpread);
        assert!((w.width() - 1.8).abs() < 1e-6);
    }

    #[test]
    fn test_default_config() {
        let config = WidenerConfig::default();
        assert_eq!(config.mode, WidenerMode::MidSide);
        assert!((config.width - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compensation_clamp() {
        let config = WidenerConfig {
            mode: WidenerMode::MidSide,
            width: 1.0,
            compensation: 5.0,
        };
        let w = StereoWidener::from_config(config);
        assert!(w.compensation <= 2.0);
    }

    #[test]
    fn test_identical_channels_stay_identical_at_unity() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        let (l, r) = w.process_sample(0.7, 0.7);
        assert!((l - r).abs() < 1e-6);
    }

    // ---- StereoWidenerConfig tests -----------------------------------------

    #[test]
    fn test_stereo_widener_config_default() {
        let c = StereoWidenerConfig::default();
        assert!((c.width - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_widener_config_mono() {
        let c = StereoWidenerConfig::mono();
        assert!((c.width - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_widener_config_wide() {
        let c = StereoWidenerConfig::wide();
        assert!((c.width - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_widener_config_new_clamps_negative() {
        let c = StereoWidenerConfig::new(-0.5);
        assert!(c.width >= 0.0);
    }

    #[test]
    fn test_stereo_widener_config_normal() {
        let c = StereoWidenerConfig::normal();
        assert!((c.width - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_widener_process_slices() {
        let mut w = StereoWidener::new(WidenerMode::MidSide, 1.0);
        let left = vec![0.5f32; 16];
        let right = vec![-0.3f32; 16];
        let (l_out, r_out) = w.process(&left, &right);
        assert_eq!(l_out.len(), 16);
        assert_eq!(r_out.len(), 16);
        for (&l, &r) in l_out.iter().zip(r_out.iter()) {
            assert!(l.is_finite());
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_stereo_widener_process_preserves_mono_at_zero_width() {
        let cfg = StereoWidenerConfig::mono();
        let mut w = StereoWidener::new(WidenerMode::MidSide, cfg.width);
        let left = vec![0.8f32; 8];
        let right = vec![-0.4f32; 8];
        let (l_out, r_out) = w.process(&left, &right);
        // At width=0 L should equal R (collapsed to mid)
        for (&l, &r) in l_out.iter().zip(r_out.iter()) {
            assert!((l - r).abs() < 1e-5);
        }
    }
}
