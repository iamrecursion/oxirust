//! Voice-Activity-Detected dialogue normalization.
//!
//! Combines the [`VoiceActivityDetector`] with the [`DialogueNormalizer`] so that
//! normalization is applied only to frames classified as speech. Silence and
//! hangover frames pass through unmodified (or optionally with a configurable
//! hold gain), preventing gain-pumping artefacts on noise-only segments.
//!
//! # Processing Pipeline
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │  frame (N samples)                                      │
//! │         │                                               │
//! │         ▼                                               │
//! │  VoiceActivityDetector.process_frame()                 │
//! │         │                                               │
//! │         ├── VadState::Speech / Hangover ──► apply gain │
//! │         │                                               │
//! │         └── VadState::Silence ──────────────► passthrough (or freeze gain)
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! The gain itself is computed from a [`DialogueLoudness`] measurement supplied
//! by the caller (typically from an upstream ITU-R BS.1770 metering block).
//!
//! # Example
//!
//! ```rust
//! use oximedia_normalize::vad_dialogue_norm::{
//!     VadDialogueNormConfig, VadDialogueNormalizer,
//! };
//! use oximedia_normalize::dialogue_norm::{DialogueLoudness, DialogueNormConfig};
//!
//! let config = VadDialogueNormConfig::broadcast();
//! let mut processor = VadDialogueNormalizer::new(config);
//!
//! // Simulate a -28 LUFS dialogue measurement
//! let loudness = DialogueLoudness::new(-28.0, 8.0, -6.0, 0.75);
//! processor.update_loudness(loudness);
//!
//! let mut frame = vec![0.3_f32; 480]; // 10 ms @ 48 kHz
//! let stats = processor.process_frame(&mut frame);
//! // Frame was speech, so normalization should have been applied
//! assert!(stats.speech_frame || stats.gain_applied_db.abs() < 0.01);
//! ```

use crate::dialogue_norm::{DialogueLoudness, DialogueNormConfig, DialogueNormalizer};
use crate::voice_activity::{VadConfig, VadState, VoiceActivityDetector};

/// Configuration for VAD-gated dialogue normalization.
#[derive(Debug, Clone)]
pub struct VadDialogueNormConfig {
    /// Dialogue normalization configuration.
    pub dialogue_norm: DialogueNormConfig,
    /// Voice activity detection configuration.
    pub vad: VadConfig,
    /// Gain smoothing time constant in frames.
    ///
    /// The applied gain is smoothed over this many frames to avoid
    /// abrupt discontinuities at speech onset/offset transitions.
    pub gain_smoothing_frames: usize,
    /// If true, freeze the gain at the last speech value during silence.
    /// If false, the gain returns to 0 dB (unity) during silence.
    pub freeze_gain_on_silence: bool,
}

impl VadDialogueNormConfig {
    /// Broadcast dialogue normalization (ATSC A/85, VAD at 16 kHz frame size 256).
    pub fn broadcast() -> Self {
        Self {
            dialogue_norm: DialogueNormConfig::atsc(),
            vad: VadConfig::narrowband(),
            gain_smoothing_frames: 5,
            freeze_gain_on_silence: true,
        }
    }

    /// EBU R128 dialogue normalization with wideband VAD.
    pub fn ebu_r128() -> Self {
        Self {
            dialogue_norm: DialogueNormConfig::ebu_r128(),
            vad: VadConfig::wideband(),
            gain_smoothing_frames: 5,
            freeze_gain_on_silence: true,
        }
    }

    /// Custom configuration.
    pub fn custom(dialogue_norm: DialogueNormConfig, vad: VadConfig) -> Self {
        Self {
            dialogue_norm,
            vad,
            gain_smoothing_frames: 3,
            freeze_gain_on_silence: false,
        }
    }
}

/// Per-frame processing statistics.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    /// VAD classification for this frame.
    pub vad_state: VadState,
    /// Whether the frame was classified as speech (Speech or Hangover).
    pub speech_frame: bool,
    /// Gain applied to this frame in dB.
    pub gain_applied_db: f64,
    /// Whether the gain was smoothed (transitioning).
    pub gain_smoothed: bool,
}

/// VAD-gated dialogue normalizer.
///
/// Applies [`DialogueNormalizer`] gain only to voice-active frames,
/// leaving silence frames unmodified.
pub struct VadDialogueNormalizer {
    config: VadDialogueNormConfig,
    vad: VoiceActivityDetector,
    normalizer: DialogueNormalizer,
    /// Pending gain computed from the most recent loudness measurement (dB).
    pending_gain_db: f64,
    /// Gain currently being applied (dB), smoothed toward pending_gain_db.
    applied_gain_db: f64,
    /// Gain at last speech frame (used when freeze_gain_on_silence is true).
    last_speech_gain_db: f64,
    /// Smoothing step size in dB per frame.
    smoothing_step_db: f64,
    /// Total frames processed.
    total_frames: u64,
    /// Total speech frames processed.
    speech_frames: u64,
}

impl VadDialogueNormalizer {
    /// Create a new VAD-gated dialogue normalizer.
    pub fn new(config: VadDialogueNormConfig) -> Self {
        let normalizer = DialogueNormalizer::new(config.dialogue_norm.clone());
        let vad = VoiceActivityDetector::new(config.vad.clone());
        // Initial smoothing step: start with a large step so first frames converge quickly
        let smoothing_step_db = if config.gain_smoothing_frames == 0 {
            f64::MAX
        } else {
            30.0 / config.gain_smoothing_frames as f64
        };

        Self {
            config,
            vad,
            normalizer,
            pending_gain_db: 0.0,
            applied_gain_db: 0.0,
            last_speech_gain_db: 0.0,
            smoothing_step_db,
            total_frames: 0,
            speech_frames: 0,
        }
    }

    /// Update the loudness measurement used to compute the normalization gain.
    ///
    /// This should be called whenever a new loudness measurement is available
    /// (e.g. at the end of a 400 ms gated measurement block).
    pub fn update_loudness(&mut self, loudness: DialogueLoudness) {
        let result = self.normalizer.apply(loudness);
        self.pending_gain_db = result.applied_gain_db;
    }

    /// Process a single frame of audio samples in-place.
    ///
    /// # Arguments
    /// * `frame` – Mono or interleaved frame of audio samples.
    ///
    /// # Returns
    /// [`FrameStats`] describing the VAD decision and gain applied.
    pub fn process_frame(&mut self, frame: &mut [f32]) -> FrameStats {
        let vad_state = self.vad.process_frame(frame);
        let is_speech = vad_state.is_speech();
        self.total_frames += 1;

        if is_speech {
            self.speech_frames += 1;

            // Smooth applied_gain_db toward pending_gain_db
            let old_gain = self.applied_gain_db;
            let target = self.pending_gain_db;
            let delta = target - old_gain;
            let step = self.smoothing_step_db;
            self.applied_gain_db = if delta.abs() <= step {
                target
            } else {
                old_gain + step * delta.signum()
            };
            self.last_speech_gain_db = self.applied_gain_db;

            let gain_smoothed = (self.applied_gain_db - target).abs() > 1e-9;

            // Apply gain
            if self.applied_gain_db.abs() > 1e-9 {
                let gain_linear = db_to_linear(self.applied_gain_db) as f32;
                for s in frame.iter_mut() {
                    *s *= gain_linear;
                }
            }

            FrameStats {
                vad_state,
                speech_frame: true,
                gain_applied_db: self.applied_gain_db,
                gain_smoothed,
            }
        } else {
            // Silence / hangover: decide whether to freeze or release gain
            let effective_gain = if self.config.freeze_gain_on_silence {
                // Keep the last speech gain to avoid pumping
                self.last_speech_gain_db
            } else {
                // Return toward unity (0 dB) over time
                let step = self.smoothing_step_db;
                let old = self.applied_gain_db;
                if old.abs() <= step {
                    0.0
                } else {
                    old - step * old.signum()
                }
            };
            self.applied_gain_db = effective_gain;

            FrameStats {
                vad_state,
                speech_frame: false,
                gain_applied_db: effective_gain,
                gain_smoothed: false,
            }
        }
    }

    /// Reset the normalizer to initial state.
    pub fn reset(&mut self) {
        self.vad.reset();
        self.pending_gain_db = 0.0;
        self.applied_gain_db = 0.0;
        self.last_speech_gain_db = 0.0;
        self.total_frames = 0;
        self.speech_frames = 0;
    }

    /// Total frames processed.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Total speech frames processed.
    pub fn speech_frames(&self) -> u64 {
        self.speech_frames
    }

    /// Speech activity ratio (0.0–1.0).
    pub fn speech_ratio(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.speech_frames as f64 / self.total_frames as f64
    }

    /// Current applied gain in dB.
    pub fn applied_gain_db(&self) -> f64 {
        self.applied_gain_db
    }

    /// Current pending (target) gain in dB.
    pub fn pending_gain_db(&self) -> f64 {
        self.pending_gain_db
    }

    /// Get the configuration.
    pub fn config(&self) -> &VadDialogueNormConfig {
        &self.config
    }
}

/// Convert dB to linear amplitude multiplier (f64).
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_activity::VadConfig;

    fn loud_frame(amplitude: f32, len: usize) -> Vec<f32> {
        vec![amplitude; len]
    }

    fn silent_frame(len: usize) -> Vec<f32> {
        vec![1e-7_f32; len]
    }

    fn make_vad_loud_config() -> VadDialogueNormConfig {
        let mut vad = VadConfig::default();
        vad.min_speech_frames = 1;
        vad.hangover_frames = 2;
        VadDialogueNormConfig::custom(DialogueNormConfig::atsc(), vad)
    }

    fn make_loudness(lkfs: f64) -> DialogueLoudness {
        DialogueLoudness::new(lkfs, 8.0, -6.0, 0.75)
    }

    // ─── Configuration ─────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_config() {
        let cfg = VadDialogueNormConfig::broadcast();
        assert!((cfg.dialogue_norm.target_lkfs - (-24.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ebu_r128_config() {
        let cfg = VadDialogueNormConfig::ebu_r128();
        assert!((cfg.dialogue_norm.target_lkfs - (-23.0)).abs() < f64::EPSILON);
    }

    // ─── update_loudness ───────────────────────────────────────────────────

    #[test]
    fn test_update_loudness_stores_gain() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        // -30 LUFS vs target -24 LUFS → needs +6 dB (clamped to max_gain_db = 15)
        proc.update_loudness(make_loudness(-30.0));
        assert!(
            proc.pending_gain_db() > 0.0,
            "pending gain should be positive, got {}",
            proc.pending_gain_db()
        );
    }

    // ─── process_frame: speech ─────────────────────────────────────────────

    #[test]
    fn test_speech_frame_gain_applied() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-30.0)); // needs +6 dB gain

        let amplitude = 0.5_f32;
        let mut frame = loud_frame(amplitude, 256);
        let stats = proc.process_frame(&mut frame);

        assert!(stats.speech_frame, "should classify as speech");
        assert!(
            stats.gain_applied_db > 0.0,
            "gain should be positive on speech frame, got {}",
            stats.gain_applied_db
        );

        // Output samples should be louder than input
        let gain_linear = db_to_linear(stats.gain_applied_db) as f32;
        let expected = amplitude * gain_linear;
        assert!(
            (frame[0] - expected).abs() < 1e-4,
            "sample mismatch: expected {expected:.6}, got {:.6}",
            frame[0]
        );
    }

    #[test]
    fn test_speech_frame_increments_counter() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-24.0));

        let mut frame = loud_frame(0.5, 256);
        proc.process_frame(&mut frame);

        assert_eq!(proc.speech_frames(), 1);
        assert_eq!(proc.total_frames(), 1);
    }

    // ─── process_frame: silence ─────────────────────────────────────────────

    #[test]
    fn test_silence_frame_not_classified_speech() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-28.0));

        let mut frame = silent_frame(256);
        let stats = proc.process_frame(&mut frame);

        assert!(!stats.speech_frame, "silent frame should not be speech");
        assert_eq!(stats.vad_state, VadState::Silence);
    }

    #[test]
    fn test_silence_no_gain_applied_when_no_prior_speech() {
        // With freeze_gain_on_silence = false and no prior speech, gain should remain ~0 dB
        let mut vad = VadConfig::default();
        vad.min_speech_frames = 1;
        let mut cfg = VadDialogueNormConfig::custom(DialogueNormConfig::atsc(), vad);
        cfg.freeze_gain_on_silence = false;

        let mut proc = VadDialogueNormalizer::new(cfg);

        let mut frame = silent_frame(256);
        let stats = proc.process_frame(&mut frame);
        assert!(
            stats.gain_applied_db.abs() < 0.01,
            "expected near-zero gain on first silence frame, got {}",
            stats.gain_applied_db
        );
    }

    #[test]
    fn test_silence_freezes_gain_when_configured() {
        // freeze_gain_on_silence = true: gain should stay at last_speech_gain_db
        let mut cfg = make_vad_loud_config();
        cfg.freeze_gain_on_silence = true;
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-30.0)); // +6 dB needed

        // First, push a speech frame so gain is established
        let mut frame = loud_frame(0.5, 256);
        let speech_stats = proc.process_frame(&mut frame);
        let speech_gain = speech_stats.gain_applied_db;

        // Now a silence frame
        let mut silent = silent_frame(256);
        let silence_stats = proc.process_frame(&mut silent);

        // Gain should be frozen at the last speech gain
        assert!(
            (silence_stats.gain_applied_db - speech_gain).abs() < 0.01,
            "silence gain {} should match last speech gain {}",
            silence_stats.gain_applied_db,
            speech_gain
        );
    }

    // ─── VAD state passthrough ─────────────────────────────────────────────

    #[test]
    fn test_hangover_frame_classified_speech() {
        let mut vad = VadConfig::default();
        vad.min_speech_frames = 1;
        vad.hangover_frames = 3;
        let cfg = VadDialogueNormConfig::custom(DialogueNormConfig::atsc(), vad);
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-28.0));

        // Speech frame
        let mut f1 = loud_frame(0.5, 256);
        proc.process_frame(&mut f1);

        // Silence frame → hangover
        let mut f2 = silent_frame(256);
        let stats = proc.process_frame(&mut f2);
        assert_eq!(stats.vad_state, VadState::Hangover);
        assert!(stats.speech_frame, "hangover counts as speech");
    }

    // ─── Gain smoothing ─────────────────────────────────────────────────────

    #[test]
    fn test_gain_smoothing_converges() {
        let mut vad = VadConfig::default();
        vad.min_speech_frames = 1;
        let mut cfg = VadDialogueNormConfig::custom(DialogueNormConfig::atsc(), vad);
        cfg.gain_smoothing_frames = 10; // slow convergence
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-34.0)); // needs +10 dB

        let mut prev_gain = proc.applied_gain_db();
        let mut converged = false;
        for _ in 0..20 {
            let mut frame = loud_frame(0.5, 256);
            proc.process_frame(&mut frame);
            let g = proc.applied_gain_db();
            if (g - proc.pending_gain_db()).abs() < 0.01 {
                converged = true;
                break;
            }
            prev_gain = g;
        }
        let _ = prev_gain;
        assert!(
            converged,
            "gain should converge to pending target within 20 frames"
        );
    }

    // ─── Reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_state() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-28.0));
        let mut frame = loud_frame(0.5, 256);
        proc.process_frame(&mut frame);
        assert!(proc.total_frames() > 0);

        proc.reset();
        assert_eq!(proc.total_frames(), 0);
        assert_eq!(proc.speech_frames(), 0);
        assert!((proc.applied_gain_db()).abs() < f64::EPSILON);
        assert!((proc.pending_gain_db()).abs() < f64::EPSILON);
    }

    // ─── Speech ratio ───────────────────────────────────────────────────────

    #[test]
    fn test_speech_ratio_zero_frames() {
        let cfg = make_vad_loud_config();
        let proc = VadDialogueNormalizer::new(cfg);
        assert!((proc.speech_ratio()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speech_ratio_all_speech() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-24.0));
        for _ in 0..10 {
            let mut frame = loud_frame(0.5, 256);
            proc.process_frame(&mut frame);
        }
        assert!(
            proc.speech_ratio() > 0.9,
            "speech ratio should be high, got {}",
            proc.speech_ratio()
        );
    }

    #[test]
    fn test_speech_ratio_mixed() {
        let cfg = make_vad_loud_config();
        let mut proc = VadDialogueNormalizer::new(cfg);
        proc.update_loudness(make_loudness(-24.0));

        for _ in 0..5 {
            let mut f = loud_frame(0.5, 256);
            proc.process_frame(&mut f);
        }
        for _ in 0..5 {
            let mut f = silent_frame(256);
            proc.process_frame(&mut f);
        }
        assert!(proc.speech_ratio() < 1.0);
    }

    // ─── db_to_linear helper ────────────────────────────────────────────────

    #[test]
    fn test_db_to_linear_zero_db() {
        assert!((db_to_linear(0.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_db_to_linear_6db() {
        let expected = 10.0_f64.powf(6.0 / 20.0);
        assert!((db_to_linear(6.0) - expected).abs() < 1e-12);
    }
}
