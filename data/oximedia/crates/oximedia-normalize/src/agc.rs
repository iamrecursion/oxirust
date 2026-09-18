//! Automatic Gain Control (AGC) for audio normalization.
//!
//! Provides configurable attack/release envelope following, per-sample gain
//! application, and a gain scheduler for keyframe-based automation.

// AGC operates entirely in the f32 domain; the casts are intentional.
#![allow(clippy::cast_precision_loss)]
#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// AgcConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Automatic Gain Control processor.
#[derive(Debug, Clone)]
pub struct AgcConfig {
    /// Target output level in dB (typically negative, e.g. -18.0 dBFS).
    pub target_level_db: f32,
    /// Maximum gain the AGC is allowed to apply, in dB.
    pub max_gain_db: f32,
    /// Minimum gain (or maximum attenuation) the AGC may apply, in dB.
    pub min_gain_db: f32,
    /// Attack time in milliseconds (how quickly gain rises when level drops).
    pub attack_ms: f32,
    /// Release time in milliseconds (how quickly gain falls when level rises).
    pub release_ms: f32,
}

impl AgcConfig {
    /// Preset suited to broadcast dialogue/voice-over content.
    pub fn broadcast() -> Self {
        Self {
            target_level_db: -18.0,
            max_gain_db: 20.0,
            min_gain_db: -20.0,
            attack_ms: 10.0,
            release_ms: 200.0,
        }
    }

    /// Preset suited to speech (podcast / call-centre / conference).
    pub fn speech() -> Self {
        Self {
            target_level_db: -20.0,
            max_gain_db: 30.0,
            min_gain_db: -10.0,
            attack_ms: 5.0,
            release_ms: 150.0,
        }
    }

    /// Preset suited to music mastering.
    pub fn music() -> Self {
        Self {
            target_level_db: -14.0,
            max_gain_db: 12.0,
            min_gain_db: -12.0,
            attack_ms: 20.0,
            release_ms: 500.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AgcState
// ──────────────────────────────────────────────────────────────────────────────

/// Run-time state carried between blocks for the AGC processor.
#[derive(Debug, Clone)]
pub struct AgcState {
    /// Current gain applied by the AGC, in dB.
    pub current_gain_db: f32,
    /// Smoothed signal envelope, in dB.
    pub envelope_db: f32,
}

impl AgcState {
    /// Initialise a new state from the given configuration.
    ///
    /// The initial gain is set to 0 dB (unity); the envelope is initialised
    /// to the configured target level.
    pub fn new(config: &AgcConfig) -> Self {
        Self {
            current_gain_db: 0.0,
            envelope_db: config.target_level_db,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the RMS level of `samples` in dB (reference 1.0 full-scale).
///
/// Returns a large negative value (~-120 dB) when the input is silent.
pub fn compute_envelope(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -120.0_f32;
    }
    let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq <= 0.0 {
        return -120.0_f32;
    }
    10.0 * mean_sq.log10()
}

/// Apply AGC to `samples`, updating `state` in-place.
///
/// Returns a `Vec<f32>` containing the gain-adjusted samples.
pub fn apply_agc(
    samples: &[f32],
    config: &AgcConfig,
    state: &mut AgcState,
    sample_rate: u32,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Compute per-sample attack / release coefficients.
    let sr = sample_rate as f32;
    let attack_coef = (-1.0_f32 / (config.attack_ms * 0.001 * sr)).exp();
    let release_coef = (-1.0_f32 / (config.release_ms * 0.001 * sr)).exp();

    let mut out = Vec::with_capacity(samples.len());

    for &sample in samples {
        // Instantaneous level in dB.
        let inst_db = if sample.abs() > 1e-10 {
            20.0 * sample.abs().log10()
        } else {
            -100.0_f32
        };

        // Envelope follower: use attack when rising, release when falling.
        if inst_db > state.envelope_db {
            state.envelope_db = attack_coef * state.envelope_db + (1.0 - attack_coef) * inst_db;
        } else {
            state.envelope_db = release_coef * state.envelope_db + (1.0 - release_coef) * inst_db;
        }

        // Desired gain to reach target.
        let desired_gain_db = config.target_level_db - state.envelope_db;
        let clamped = desired_gain_db.clamp(config.min_gain_db, config.max_gain_db);
        state.current_gain_db = clamped;

        let gain_linear = db_to_linear(clamped);
        out.push(sample * gain_linear);
    }

    out
}

/// Convert a gain in dB to a linear multiplier.
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// ──────────────────────────────────────────────────────────────────────────────
// GainScheduler
// ──────────────────────────────────────────────────────────────────────────────

/// Keyframe-based gain automation curve.
///
/// Stores `(frame_index, gain_db)` pairs and interpolates linearly between them.
#[derive(Debug, Default)]
pub struct GainScheduler {
    /// Sorted keyframe list of `(frame, gain_db)`.
    pub segments: Vec<(u64, f32)>,
}

impl GainScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a keyframe.  Keyframes are maintained in ascending frame order.
    pub fn add_keyframe(&mut self, frame: u64, gain_db: f32) {
        // Remove any existing keyframe at the same position.
        self.segments.retain(|&(f, _)| f != frame);
        self.segments.push((frame, gain_db));
        self.segments.sort_by_key(|&(f, _)| f);
    }

    /// Return the interpolated gain (dB) at `frame`.
    ///
    /// - Before the first keyframe: returns the first keyframe's gain.
    /// - After the last keyframe: returns the last keyframe's gain.
    /// - Between keyframes: linearly interpolates.
    pub fn gain_at(&self, frame: u64) -> f32 {
        if self.segments.is_empty() {
            return 0.0;
        }
        if frame <= self.segments[0].0 {
            return self.segments[0].1;
        }
        if frame >= self.segments[self.segments.len() - 1].0 {
            return self.segments[self.segments.len() - 1].1;
        }
        // Find surrounding keyframes.
        let idx = self.segments.partition_point(|&(f, _)| f <= frame) - 1;
        let (f0, g0) = self.segments[idx];
        let (f1, g1) = self.segments[idx + 1];
        let t = (frame - f0) as f32 / (f1 - f0) as f32;
        g0 + t * (g1 - g0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // AgcConfig presets ───────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_preset() {
        let cfg = AgcConfig::broadcast();
        assert_eq!(cfg.target_level_db, -18.0);
        assert!(cfg.max_gain_db > 0.0);
        assert!(cfg.attack_ms > 0.0 && cfg.release_ms > 0.0);
    }

    #[test]
    fn test_speech_preset() {
        let cfg = AgcConfig::speech();
        assert!(cfg.attack_ms < cfg.release_ms);
    }

    #[test]
    fn test_music_preset() {
        let cfg = AgcConfig::music();
        assert_eq!(cfg.target_level_db, -14.0);
    }

    // AgcState ────────────────────────────────────────────────────────────────

    #[test]
    fn test_agc_state_new_initial_gain_zero() {
        let cfg = AgcConfig::broadcast();
        let state = AgcState::new(&cfg);
        assert_eq!(state.current_gain_db, 0.0);
    }

    #[test]
    fn test_agc_state_envelope_equals_target() {
        let cfg = AgcConfig::broadcast();
        let state = AgcState::new(&cfg);
        assert_eq!(state.envelope_db, cfg.target_level_db);
    }

    // compute_envelope ────────────────────────────────────────────────────────

    #[test]
    fn test_compute_envelope_empty() {
        assert_eq!(compute_envelope(&[]), -120.0);
    }

    #[test]
    fn test_compute_envelope_silence() {
        assert_eq!(compute_envelope(&[0.0, 0.0, 0.0]), -120.0);
    }

    #[test]
    fn test_compute_envelope_full_scale() {
        // RMS of +1.0 constant = 0 dBFS.
        let samples = vec![1.0f32; 1000];
        let db = compute_envelope(&samples);
        assert!((db - 0.0).abs() < 0.1, "Expected ~0 dBFS, got {}", db);
    }

    #[test]
    fn test_compute_envelope_half() {
        // RMS of 0.5 constant ≈ -6.02 dBFS.
        let samples = vec![0.5f32; 1000];
        let db = compute_envelope(&samples);
        assert!(
            (db - (-6.02)).abs() < 0.1,
            "Expected ~-6.02 dBFS, got {}",
            db
        );
    }

    // apply_agc ───────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_agc_output_length() {
        let cfg = AgcConfig::broadcast();
        let mut state = AgcState::new(&cfg);
        let input = vec![0.1f32; 480];
        let out = apply_agc(&input, &cfg, &mut state, 48000);
        assert_eq!(out.len(), input.len());
    }

    #[test]
    fn test_apply_agc_empty_input() {
        let cfg = AgcConfig::broadcast();
        let mut state = AgcState::new(&cfg);
        assert!(apply_agc(&[], &cfg, &mut state, 48000).is_empty());
    }

    // GainScheduler ───────────────────────────────────────────────────────────

    #[test]
    fn test_gain_scheduler_empty_returns_zero() {
        let gs = GainScheduler::new();
        assert_eq!(gs.gain_at(0), 0.0);
    }

    #[test]
    fn test_gain_scheduler_single_keyframe() {
        let mut gs = GainScheduler::new();
        gs.add_keyframe(100, 6.0);
        assert_eq!(gs.gain_at(0), 6.0);
        assert_eq!(gs.gain_at(100), 6.0);
        assert_eq!(gs.gain_at(200), 6.0);
    }

    #[test]
    fn test_gain_scheduler_interpolation() {
        let mut gs = GainScheduler::new();
        gs.add_keyframe(0, 0.0);
        gs.add_keyframe(100, 10.0);
        let mid = gs.gain_at(50);
        assert!((mid - 5.0).abs() < 0.01, "Expected 5.0, got {}", mid);
    }

    #[test]
    fn test_gain_scheduler_sorted_insertion() {
        let mut gs = GainScheduler::new();
        gs.add_keyframe(200, 4.0);
        gs.add_keyframe(0, 0.0);
        assert_eq!(gs.segments[0].0, 0);
        assert_eq!(gs.segments[1].0, 200);
    }
}
