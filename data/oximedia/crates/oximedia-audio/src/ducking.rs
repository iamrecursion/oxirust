//! Audio ducking — automatically lower music/background audio when voiceover is present.
//!
//! Audio ducking (also known as auto-ducking or side-chain ducking) reduces the level
//! of a "background" signal (e.g., music) whenever a "foreground" signal (e.g., voice)
//! exceeds a configurable threshold.  This module implements a full-featured ducking
//! processor with:
//!
//! - Side-chain level detection (RMS or peak)
//! - Configurable threshold, depth, attack, hold, and release parameters
//! - Smoothed gain reduction via an RC envelope follower
//! - Optional stereo (linked) and multi-channel support
//!
//! # Example
//!
//! ```
//! use oximedia_audio::ducking::{Ducker, DuckerConfig};
//!
//! let config = DuckerConfig {
//!     sample_rate: 48_000.0,
//!     threshold_db: -20.0,
//!     depth_db: 10.0,
//!     attack_ms: 10.0,
//!     hold_ms: 200.0,
//!     release_ms: 500.0,
//!     ..DuckerConfig::default()
//! };
//! let mut ducker = Ducker::new(config);
//!
//! // Process background samples driven by a voiceover sidechain.
//! let background = vec![0.8_f32; 1024];
//! let voiceover   = vec![0.5_f32; 1024];
//! let output = ducker.process_stereo(&background, &voiceover);
//! assert_eq!(output.len(), background.len());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

// ─────────────────────────────────────────────────────────────────────────────
// Detection mode
// ─────────────────────────────────────────────────────────────────────────────

/// Side-chain level detection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectionMode {
    /// RMS (root-mean-square) level over a short window.  Smoother and more
    /// representative of perceived loudness.
    #[default]
    Rms,
    /// Instantaneous peak level.  Faster but can react to transients.
    Peak,
}

// ─────────────────────────────────────────────────────────────────────────────
// DuckerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration parameters for [`Ducker`].
#[derive(Debug, Clone)]
pub struct DuckerConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Level (dBFS) above which ducking activates.  Must be ≤ 0.
    pub threshold_db: f32,
    /// Maximum gain reduction applied when fully ducked (positive dB, e.g., 10 = –10 dB).
    pub depth_db: f32,
    /// Attack time in milliseconds (time to reach full duck from silence).
    pub attack_ms: f32,
    /// Hold time in milliseconds (keep ducking this long after level drops below threshold).
    pub hold_ms: f32,
    /// Release time in milliseconds (time to restore full level after hold expires).
    pub release_ms: f32,
    /// Side-chain window length in milliseconds for RMS integration.
    pub rms_window_ms: f32,
    /// Side-chain level detection mode.
    pub detection: DetectionMode,
}

impl Default for DuckerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000.0,
            threshold_db: -20.0,
            depth_db: 10.0,
            attack_ms: 10.0,
            hold_ms: 200.0,
            release_ms: 500.0,
            rms_window_ms: 30.0,
            detection: DetectionMode::Rms,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: dB conversions
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[inline]
fn linear_to_db(lin: f32) -> f32 {
    if lin <= 1e-12 {
        return -240.0;
    }
    20.0 * lin.log10()
}

/// Compute one-pole RC coefficient for the given time constant and sample rate.
///
/// `tau_ms` is the time constant in milliseconds.  Returns the coefficient α
/// such that `y[n] = α * y[n-1] + (1-α) * x[n]`.
#[inline]
fn rc_coeff(tau_ms: f32, sample_rate: f32) -> f32 {
    if tau_ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let tau_samples = tau_ms * 0.001 * sample_rate;
    (-1.0_f32 / tau_samples).exp()
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS tracker (circular buffer)
// ─────────────────────────────────────────────────────────────────────────────

struct RmsTracker {
    buf: Vec<f32>, // squared samples
    pos: usize,
    sum_sq: f64,
}

impl RmsTracker {
    fn new(window_samples: usize) -> Self {
        let len = window_samples.max(1);
        Self {
            buf: vec![0.0; len],
            pos: 0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, sample: f32) -> f32 {
        let sq = (sample as f64) * (sample as f64);
        self.sum_sq -= self.buf[self.pos] as f64;
        self.buf[self.pos] = sq as f32;
        self.sum_sq += sq;
        self.pos = (self.pos + 1) % self.buf.len();
        let mean = (self.sum_sq / self.buf.len() as f64).max(0.0);
        mean.sqrt() as f32
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
        self.sum_sq = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ducker state machine
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DuckState {
    /// Side-chain below threshold; no ducking.
    Idle,
    /// Side-chain above threshold; gain is being reduced.
    Attacking,
    /// Fully ducked; maintaining maximum gain reduction.
    Hold,
    /// Side-chain dropped; gain is recovering.
    Releasing,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ducker
// ─────────────────────────────────────────────────────────────────────────────

/// Audio ducking processor.
///
/// `Ducker` monitors a side-chain (voiceover) signal and reduces the gain
/// of a background (music) signal whenever the side-chain exceeds the
/// configured threshold.  The gain reduction envelope uses separate attack,
/// hold, and release time constants for natural-sounding results.
pub struct Ducker {
    config: DuckerConfig,
    rms: RmsTracker,
    /// Current gain reduction level (linear, 1.0 = no reduction).
    gain: f32,
    /// Target gain when fully ducked (linear).
    duck_gain: f32,
    /// Current state.
    state: DuckState,
    /// Hold counter (samples remaining in hold phase).
    hold_counter: u64,
    /// Pre-computed attack RC coefficient.
    attack_coeff: f32,
    /// Pre-computed release RC coefficient.
    release_coeff: f32,
    /// Hold duration in samples.
    hold_samples: u64,
    /// Threshold (linear).
    threshold_lin: f32,
}

impl Ducker {
    /// Create a new `Ducker` with the given configuration.
    #[must_use]
    pub fn new(config: DuckerConfig) -> Self {
        let sr = config.sample_rate;
        let window_samples = ((config.rms_window_ms * 0.001 * sr).round() as usize).max(1);
        let attack_coeff = rc_coeff(config.attack_ms, sr);
        let release_coeff = rc_coeff(config.release_ms, sr);
        let hold_samples = (config.hold_ms * 0.001 * sr).round() as u64;
        let threshold_lin = db_to_linear(config.threshold_db);
        let duck_gain = db_to_linear(-config.depth_db.abs()); // e.g. -10 dB

        Self {
            config,
            rms: RmsTracker::new(window_samples),
            gain: 1.0,
            duck_gain,
            state: DuckState::Idle,
            hold_counter: 0,
            attack_coeff,
            release_coeff,
            hold_samples,
            threshold_lin,
        }
    }

    /// Update internal parameters from a new configuration without resetting state.
    pub fn update_config(&mut self, config: DuckerConfig) {
        let sr = config.sample_rate;
        let window_samples = ((config.rms_window_ms * 0.001 * sr).round() as usize).max(1);
        self.attack_coeff = rc_coeff(config.attack_ms, sr);
        self.release_coeff = rc_coeff(config.release_ms, sr);
        self.hold_samples = (config.hold_ms * 0.001 * sr).round() as u64;
        self.threshold_lin = db_to_linear(config.threshold_db);
        self.duck_gain = db_to_linear(-config.depth_db.abs());
        self.rms = RmsTracker::new(window_samples);
        self.config = config;
    }

    /// Reset all state to initial conditions (no ducking).
    pub fn reset(&mut self) {
        self.rms.reset();
        self.gain = 1.0;
        self.state = DuckState::Idle;
        self.hold_counter = 0;
    }

    /// Current gain reduction as a linear multiplier (1.0 = no reduction).
    #[must_use]
    pub fn current_gain(&self) -> f32 {
        self.gain
    }

    /// Current gain reduction in dB (0.0 = no reduction, negative = ducked).
    #[must_use]
    pub fn current_gain_db(&self) -> f32 {
        linear_to_db(self.gain)
    }

    /// Whether the ducker is currently applying gain reduction.
    #[must_use]
    pub fn is_ducking(&self) -> bool {
        self.state != DuckState::Idle
    }

    /// Process a single pair of (background, sidechain) samples.
    ///
    /// Returns the gain-adjusted background sample.
    pub fn process_sample(&mut self, background: f32, sidechain: f32) -> f32 {
        // 1. Measure sidechain level
        let level = match self.config.detection {
            DetectionMode::Rms => self.rms.push(sidechain),
            DetectionMode::Peak => {
                // Still push to keep rms buf consistent
                let _ = self.rms.push(sidechain);
                sidechain.abs()
            }
        };

        // 2. Advance state machine
        match self.state {
            DuckState::Idle => {
                if level >= self.threshold_lin {
                    self.state = DuckState::Attacking;
                }
            }
            DuckState::Attacking => {
                if level < self.threshold_lin {
                    // Started to fall — go to hold
                    self.state = DuckState::Hold;
                    self.hold_counter = self.hold_samples;
                }
            }
            DuckState::Hold => {
                if level >= self.threshold_lin {
                    // Level rose again — back to attacking
                    self.state = DuckState::Attacking;
                    self.hold_counter = 0;
                } else if self.hold_counter == 0 {
                    self.state = DuckState::Releasing;
                } else {
                    self.hold_counter -= 1;
                }
            }
            DuckState::Releasing => {
                if level >= self.threshold_lin {
                    self.state = DuckState::Attacking;
                } else if (self.gain - 1.0).abs() < 1e-5 {
                    self.state = DuckState::Idle;
                }
            }
        }

        // 3. Move gain toward target
        let target = match self.state {
            DuckState::Idle => 1.0,
            DuckState::Attacking | DuckState::Hold => self.duck_gain,
            DuckState::Releasing => 1.0,
        };

        let coeff = match self.state {
            DuckState::Attacking | DuckState::Hold => self.attack_coeff,
            DuckState::Releasing | DuckState::Idle => self.release_coeff,
        };

        self.gain = coeff * self.gain + (1.0 - coeff) * target;
        // Clamp to valid range
        self.gain = self.gain.clamp(self.duck_gain, 1.0);

        background * self.gain
    }

    /// Process a block of background samples with a corresponding sidechain block.
    ///
    /// Both slices must have the same length; if they differ, the shorter length
    /// is used.  Returns a `Vec<f32>` of gain-adjusted background samples.
    #[must_use]
    pub fn process_stereo(&mut self, background: &[f32], sidechain: &[f32]) -> Vec<f32> {
        let n = background.len().min(sidechain.len());
        (0..n)
            .map(|i| self.process_sample(background[i], sidechain[i]))
            .collect()
    }

    /// Process background and sidechain blocks in-place.
    ///
    /// `background` is modified to contain the ducked output.  The shorter of
    /// the two slice lengths is processed.
    pub fn process_inplace(&mut self, background: &mut [f32], sidechain: &[f32]) {
        let n = background.len().min(sidechain.len());
        for i in 0..n {
            background[i] = self.process_sample(background[i], sidechain[i]);
        }
    }

    /// Process stereo-interleaved background and sidechain buffers.
    ///
    /// Both buffers must be interleaved stereo (L, R, L, R, …).  The sidechain
    /// level is taken as the average of both channels.  Returns a
    /// `Vec<f32>` of gain-reduced interleaved stereo background samples.
    #[must_use]
    pub fn process_stereo_interleaved(
        &mut self,
        background: &[f32],
        sidechain: &[f32],
    ) -> Vec<f32> {
        let n_frames = background.len().min(sidechain.len()) / 2;
        let mut out = Vec::with_capacity(n_frames * 2);
        for i in 0..n_frames {
            let sc = (sidechain[i * 2].abs() + sidechain[i * 2 + 1].abs()) * 0.5;
            let _gain = {
                // Advance gain for this frame (use average SC level)
                let dummy_bg = background[i * 2];
                let ducked = self.process_sample(dummy_bg, sc);
                ducked / dummy_bg.abs().max(1e-10) * dummy_bg.signum()
            };
            // Apply same gain to both channels
            let g = self.gain; // after process_sample updated self.gain
            out.push(background[i * 2] * g);
            out.push(background[i * 2 + 1] * g);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ducker() -> Ducker {
        Ducker::new(DuckerConfig::default())
    }

    #[test]
    fn test_no_ducking_when_sidechain_silent() {
        let mut d = default_ducker();
        // With silent sidechain the background should pass through at gain ~ 1
        let bg: Vec<f32> = vec![0.5; 2048];
        let sc: Vec<f32> = vec![0.0; 2048];
        let out = d.process_stereo(&bg, &sc);
        let final_out = *out.last().expect("non-empty");
        assert!(
            (final_out - 0.5).abs() < 0.01,
            "Silent SC should not duck; got {final_out}"
        );
    }

    #[test]
    fn test_ducking_reduces_level() {
        let mut d = default_ducker();
        let bg: Vec<f32> = vec![1.0; 4096];
        let sc: Vec<f32> = vec![1.0; 4096]; // loud sidechain
        let out = d.process_stereo(&bg, &sc);
        // After enough samples, gain should have dropped significantly
        let final_out = *out.last().expect("non-empty");
        assert!(
            final_out < 0.5,
            "Loud SC should duck background; got {final_out}"
        );
    }

    #[test]
    fn test_is_ducking_flag() {
        let mut d = default_ducker();
        // Initially not ducking
        assert!(!d.is_ducking());
        // Feed loud sidechain
        for _ in 0..100 {
            d.process_sample(1.0, 1.0);
        }
        assert!(d.is_ducking(), "Should be ducking after loud SC");
    }

    #[test]
    fn test_gain_in_range() {
        let mut d = default_ducker();
        let bg = vec![0.8_f32; 1024];
        let sc = vec![0.9_f32; 1024];
        let out = d.process_stereo(&bg, &sc);
        for &s in &out {
            assert!(s.is_finite(), "Output must be finite");
            assert!(s >= -2.0 && s <= 2.0, "Output out of range: {s}");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let mut d = default_ducker();
        for _ in 0..500 {
            d.process_sample(1.0, 1.0);
        }
        d.reset();
        assert!(!d.is_ducking());
        assert!((d.current_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_current_gain_db_no_ducking() {
        let d = Ducker::new(DuckerConfig::default());
        // Initially no ducking => gain = 1.0 => 0 dB
        assert!(d.current_gain_db().abs() < 0.1);
    }

    #[test]
    fn test_process_inplace_matches_process_stereo() {
        let mut d1 = default_ducker();
        let mut d2 = default_ducker();
        let bg = vec![0.6_f32; 256];
        let sc = vec![0.4_f32; 256];
        let out1 = d1.process_stereo(&bg, &sc);
        let mut bg2 = bg.clone();
        d2.process_inplace(&mut bg2, &sc);
        for (a, b) in out1.iter().zip(bg2.iter()) {
            assert!((a - b).abs() < 1e-6, "inplace vs stereo mismatch");
        }
    }

    #[test]
    fn test_process_stereo_shorter_length() {
        let mut d = default_ducker();
        let bg = vec![0.5_f32; 100];
        let sc = vec![0.3_f32; 50];
        let out = d.process_stereo(&bg, &sc);
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn test_update_config() {
        let mut d = default_ducker();
        let new_cfg = DuckerConfig {
            depth_db: 20.0,
            ..DuckerConfig::default()
        };
        d.update_config(new_cfg);
        // After deep ducking with loud SC, level should drop very low
        for _ in 0..2000 {
            d.process_sample(1.0, 1.0);
        }
        assert!(d.current_gain() < 0.2, "20 dB duck should be deep");
    }

    #[test]
    fn test_peak_detection_mode() {
        let mut d = Ducker::new(DuckerConfig {
            detection: DetectionMode::Peak,
            ..DuckerConfig::default()
        });
        // A single loud peak should trigger ducking
        for _ in 0..1000 {
            d.process_sample(1.0, 1.0);
        }
        assert!(d.is_ducking());
    }

    #[test]
    fn test_rms_tracker_push_zero() {
        let mut t = RmsTracker::new(100);
        let rms = t.push(0.0);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_rms_tracker_constant_signal() {
        let mut t = RmsTracker::new(100);
        let mut last = 0.0_f32;
        for _ in 0..200 {
            last = t.push(0.5);
        }
        // RMS of constant 0.5 is 0.5
        assert!(
            (last - 0.5).abs() < 0.01,
            "RMS of 0.5 const = 0.5; got {last}"
        );
    }

    #[test]
    fn test_db_to_linear_zero_db() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_db_to_linear_minus_20() {
        let lin = db_to_linear(-20.0);
        assert!((lin - 0.1).abs() < 1e-5);
    }
}
