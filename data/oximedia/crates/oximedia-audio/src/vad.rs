//! Voice Activity Detection (VAD).
//!
//! Provides an energy-based VAD with spectral flatness measure,
//! hangover logic, and minimum speech duration filtering.

/// Decision returned by the VAD for each processed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    /// Active speech detected.
    Speech,
    /// Silence (no speech activity).
    Silence,
    /// Transitioning between silence and speech (ramp-up or hangover).
    Transition,
}

/// Configuration for the voice activity detector.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Energy threshold in dBFS (e.g. -40.0).
    ///
    /// Frames with energy above this threshold are candidates for speech.
    pub threshold_db: f32,
    /// Minimum consecutive speech duration in milliseconds.
    ///
    /// Short bursts shorter than this remain `Transition` rather than `Speech`.
    pub min_speech_ms: u32,
    /// Hangover duration in milliseconds.
    ///
    /// After energy drops below threshold the VAD keeps reporting `Speech`
    /// (or `Transition`) for this many milliseconds.
    pub hangover_ms: u32,
}

impl VadConfig {
    /// Creates a default EBU / telephony-suitable VAD configuration.
    #[must_use]
    pub fn default_telephony() -> Self {
        Self {
            threshold_db: -40.0,
            min_speech_ms: 30,
            hangover_ms: 100,
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self::default_telephony()
    }
}

// ---------------------------------------------------------------------------
// Internal state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    /// Currently in silence.
    Silence,
    /// Energy above threshold but not long enough for confirmed speech.
    RampUp,
    /// Confirmed speech.
    Speech,
    /// Energy dropped; holding speech decision for hangover period.
    Hangover,
}

/// Voice activity detector.
pub struct VadDetector {
    config: VadConfig,
    sample_rate: u32,
    /// Current internal FSM state.
    state: VadState,
    /// Energy of the most recently processed frame in dBFS.
    current_energy_db: f32,
    /// Spectral flatness of the most recently processed frame.
    current_sfm: f32,
    /// Number of consecutive samples with energy above threshold.
    above_threshold_samples: u64,
    /// Number of consecutive samples spent in hangover.
    hangover_samples_remaining: u64,
    /// Minimum samples required for confirmed speech.
    min_speech_samples: u64,
    /// Total hangover samples per configuration.
    hangover_total_samples: u64,
}

impl VadDetector {
    /// Create a new `VadDetector`.
    ///
    /// # Panics
    ///
    /// Never panics; if `sample_rate` is 0 the detector uses 0-sample
    /// thresholds (effectively always transitions immediately).
    #[must_use]
    pub fn new(config: VadConfig, sample_rate: u32) -> Self {
        let sr = sample_rate as u64;
        let min_speech_samples = if sr > 0 {
            sr * u64::from(config.min_speech_ms) / 1_000
        } else {
            0
        };
        let hangover_total_samples = if sr > 0 {
            sr * u64::from(config.hangover_ms) / 1_000
        } else {
            0
        };

        Self {
            config,
            sample_rate,
            state: VadState::Silence,
            current_energy_db: f32::NEG_INFINITY,
            current_sfm: 1.0,
            above_threshold_samples: 0,
            hangover_samples_remaining: 0,
            min_speech_samples,
            hangover_total_samples,
        }
    }

    /// Process a single frame of audio samples and return the VAD decision.
    ///
    /// The frame length should match the expected block size (e.g. 10–30 ms
    /// at the configured sample rate). Shorter frames are accepted but
    /// internal timing counters may be less accurate.
    ///
    /// Returns `VadDecision::Silence` on an empty frame.
    pub fn process_frame(&mut self, samples: &[f32]) -> VadDecision {
        if samples.is_empty() {
            return VadDecision::Silence;
        }

        self.current_energy_db = compute_energy_db(samples);
        self.current_sfm = self.spectral_flatness_measure(samples);

        let n = samples.len() as u64;
        let is_active = self.current_energy_db >= self.config.threshold_db;

        self.state = match self.state {
            VadState::Silence => {
                if is_active {
                    self.above_threshold_samples = n;
                    VadState::RampUp
                } else {
                    self.above_threshold_samples = 0;
                    VadState::Silence
                }
            }
            VadState::RampUp => {
                if is_active {
                    self.above_threshold_samples += n;
                    if self.above_threshold_samples >= self.min_speech_samples {
                        VadState::Speech
                    } else {
                        VadState::RampUp
                    }
                } else {
                    // Too short – drop back to silence
                    self.above_threshold_samples = 0;
                    VadState::Silence
                }
            }
            VadState::Speech => {
                if is_active {
                    self.above_threshold_samples += n;
                    VadState::Speech
                } else {
                    // Begin hangover
                    self.hangover_samples_remaining = self.hangover_total_samples;
                    VadState::Hangover
                }
            }
            VadState::Hangover => {
                if is_active {
                    // Speech resumed
                    self.hangover_samples_remaining = 0;
                    VadState::Speech
                } else if self.hangover_samples_remaining > n {
                    self.hangover_samples_remaining -= n;
                    VadState::Hangover
                } else {
                    self.hangover_samples_remaining = 0;
                    self.above_threshold_samples = 0;
                    VadState::Silence
                }
            }
        };

        self.decision_for_state()
    }

    /// Returns `true` if the current state is speech or hangover.
    #[must_use]
    pub fn is_speech(&self) -> bool {
        matches!(self.state, VadState::Speech | VadState::Hangover)
    }

    /// Returns the energy of the last processed frame in dBFS.
    #[must_use]
    pub fn frame_energy_db(&self) -> f32 {
        self.current_energy_db
    }

    /// Returns the spectral flatness measure of the last processed frame.
    ///
    /// Values close to 1.0 indicate noise-like signal; close to 0.0 indicate
    /// tonal / speech-like signal.
    #[must_use]
    pub fn last_sfm(&self) -> f32 {
        self.current_sfm
    }

    /// Compute a spectral flatness proxy for the given samples.
    ///
    /// Uses the variance of local short-term energy (computed in 8 non-overlapping
    /// sub-frames) relative to the mean energy. Tonal signals have highly periodic
    /// energy envelopes resulting in larger variance; noise-like signals have lower
    /// variance.  The result is inverted so that:
    ///
    /// - Values close to 0.0 indicate tonal (periodic, speech-like) content.
    /// - Values close to 1.0 indicate noise-like (flat-spectrum) content.
    ///
    /// Returns a value in [0.0, 1.0].
    pub fn spectral_flatness_measure(&self, samples: &[f32]) -> f32 {
        const NUM_BINS: usize = 8;
        if samples.len() < NUM_BINS * 2 {
            return 1.0; // not enough data
        }

        let bin_len = samples.len() / NUM_BINS;
        let mut energies = [0.0f64; NUM_BINS];
        for (b, chunk) in samples.chunks(bin_len).take(NUM_BINS).enumerate() {
            energies[b] =
                chunk.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / chunk.len() as f64;
        }

        let mean = energies.iter().sum::<f64>() / NUM_BINS as f64;
        if mean < 1e-30 {
            return 1.0; // silence
        }

        let variance = energies
            .iter()
            .map(|&e| {
                let d = e - mean;
                d * d
            })
            .sum::<f64>()
            / NUM_BINS as f64;

        // Coefficient of variation (std/mean) normalised to [0,1]
        let cv = (variance.sqrt() / mean) as f32;

        // Tonal → high CV (large swings) → low flatness
        // Noise  → low CV (even energy) → high flatness
        // We cap CV at 2.0 and invert.
        let flatness = 1.0 - (cv / 2.0).clamp(0.0, 1.0);
        flatness.clamp(0.0, 1.0)
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.current_energy_db = f32::NEG_INFINITY;
        self.current_sfm = 1.0;
        self.above_threshold_samples = 0;
        self.hangover_samples_remaining = 0;
    }

    /// Return the current VAD decision without processing a new frame.
    #[must_use]
    pub fn current_decision(&self) -> VadDecision {
        self.decision_for_state()
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Return the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn decision_for_state(&self) -> VadDecision {
        match self.state {
            VadState::Silence => VadDecision::Silence,
            VadState::RampUp | VadState::Hangover => VadDecision::Transition,
            VadState::Speech => VadDecision::Speech,
        }
    }
}

// ---------------------------------------------------------------------------
// Free helper: compute RMS energy in dBFS
// ---------------------------------------------------------------------------

/// Compute the RMS energy of `samples` in dBFS.
///
/// Returns `f32::NEG_INFINITY` for silence (all-zero input).
fn compute_energy_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq: f32 = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq < f32::EPSILON {
        f32::NEG_INFINITY
    } else {
        10.0 * mean_sq.log10()
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SAMPLE_RATE: u32 = 48_000;

    /// Generate a 1 kHz sine wave frame of `num_samples` samples at `amplitude`.
    fn sine_frame(amplitude: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect()
    }

    /// Generate silence.
    fn silence_frame(num_samples: usize) -> Vec<f32> {
        vec![0.0f32; num_samples]
    }

    // ------------------------------------------------------------------
    // VadConfig
    // ------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.threshold_db, -40.0);
        assert_eq!(cfg.min_speech_ms, 30);
        assert_eq!(cfg.hangover_ms, 100);
    }

    // ------------------------------------------------------------------
    // Silence detection
    // ------------------------------------------------------------------

    #[test]
    fn test_silence_frame_returns_silence() {
        let mut vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let frame = silence_frame(480); // 10 ms at 48 kHz
        let decision = vad.process_frame(&frame);
        assert_eq!(decision, VadDecision::Silence);
    }

    #[test]
    fn test_empty_frame_returns_silence() {
        let mut vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let decision = vad.process_frame(&[]);
        assert_eq!(decision, VadDecision::Silence);
    }

    // ------------------------------------------------------------------
    // Speech detection
    // ------------------------------------------------------------------

    #[test]
    fn test_loud_tone_eventually_becomes_speech() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 10, // 480 samples at 48 kHz
            hangover_ms: 50,
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let frame = sine_frame(0.5, 480); // -6 dBFS → well above -40 dB

        // Feed 5 frames (50 ms) to satisfy min_speech_ms=10 ms
        let mut decisions = Vec::new();
        for _ in 0..5 {
            decisions.push(vad.process_frame(&frame));
        }

        assert!(
            decisions.contains(&VadDecision::Speech),
            "Should reach Speech state: {decisions:?}"
        );
    }

    // ------------------------------------------------------------------
    // Short burst remains Transition
    // ------------------------------------------------------------------

    #[test]
    fn test_short_burst_stays_transition() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 100, // require 100 ms before confirming speech
            hangover_ms: 50,
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let frame = sine_frame(0.5, 480); // 10 ms frame

        // Only feed ONE frame (10 ms < 100 ms min_speech)
        let d = vad.process_frame(&frame);
        assert_eq!(
            d,
            VadDecision::Transition,
            "One 10ms frame should be Transition, not Speech"
        );
    }

    // ------------------------------------------------------------------
    // Hangover behaviour
    // ------------------------------------------------------------------

    #[test]
    fn test_hangover_maintains_non_silence_after_speech_ends() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 10,
            hangover_ms: 100, // 4800 samples at 48 kHz
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let speech_frame = sine_frame(0.5, 480);
        let silent_frame = silence_frame(480);

        // Establish speech (3 frames = 30 ms ≥ min_speech 10 ms)
        for _ in 0..3 {
            vad.process_frame(&speech_frame);
        }
        assert_eq!(vad.process_frame(&speech_frame), VadDecision::Speech);

        // Now feed silence; hangover should keep non-Silence for ~100 ms
        let mut after_speech = Vec::new();
        for _ in 0..5 {
            // 5 frames = 50 ms, still within hangover
            after_speech.push(vad.process_frame(&silent_frame));
        }

        assert!(
            after_speech.iter().any(|d| *d != VadDecision::Silence),
            "Hangover should prevent immediate Silence: {after_speech:?}"
        );
    }

    #[test]
    fn test_hangover_eventually_returns_to_silence() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 10,
            hangover_ms: 50, // 2400 samples = 5 × 480-sample frames
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let speech_frame = sine_frame(0.5, 480);
        let silent_frame = silence_frame(480);

        // Establish speech
        for _ in 0..3 {
            vad.process_frame(&speech_frame);
        }

        // Feed 20 silent frames (400 ms >> 50 ms hangover)
        let mut last_decision = VadDecision::Speech;
        for _ in 0..20 {
            last_decision = vad.process_frame(&silent_frame);
        }

        assert_eq!(
            last_decision,
            VadDecision::Silence,
            "After hangover expires, VAD must return Silence"
        );
    }

    // ------------------------------------------------------------------
    // Reset behaviour
    // ------------------------------------------------------------------

    #[test]
    fn test_reset_returns_to_silence() {
        let mut vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let speech_frame = sine_frame(0.5, 1440); // 30 ms
        for _ in 0..5 {
            vad.process_frame(&speech_frame);
        }

        vad.reset();
        assert_eq!(vad.current_decision(), VadDecision::Silence);
        assert!(!vad.is_speech());
    }

    // ------------------------------------------------------------------
    // Energy measurement accuracy
    // ------------------------------------------------------------------

    #[test]
    fn test_energy_db_of_sine_is_reasonable() {
        let mut vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let frame = sine_frame(1.0, 4800); // 0 dBFS peak sine → RMS = -3 dBFS
        vad.process_frame(&frame);
        let e = vad.frame_energy_db();
        // 0 dBFS sine: RMS ≈ -3.01 dBFS → in power terms ≈ -3 dB
        assert!(
            e > -10.0,
            "Energy should be > -10 dB for full-scale sine: {e}"
        );
        assert!(e < 1.0, "Energy should be < 1 dB for full-scale sine: {e}");
    }

    #[test]
    fn test_silence_energy_is_neg_infinity() {
        let mut vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let frame = silence_frame(4800);
        vad.process_frame(&frame);
        assert!(
            vad.frame_energy_db().is_infinite() && vad.frame_energy_db() < 0.0,
            "Silence energy must be -∞"
        );
    }

    // ------------------------------------------------------------------
    // Spectral flatness
    // ------------------------------------------------------------------

    #[test]
    fn test_sfm_sine_returns_valid_range() {
        let vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        let sine = sine_frame(0.5, 4800);
        let sfm = vad.spectral_flatness_measure(&sine);
        // SFM must always be in [0.0, 1.0]
        assert!(
            (0.0..=1.0).contains(&sfm),
            "SFM must be in [0,1], got {sfm}"
        );
    }

    #[test]
    fn test_sfm_noise_returns_valid_range() {
        let vad = VadDetector::new(VadConfig::default(), SAMPLE_RATE);
        // Pseudo-random noise via deterministic multi-frequency sequence
        let noise: Vec<f32> = (0..4800)
            .map(|i| {
                let x = (i as f32 * 1234.567_f32).sin() * 0.5
                    + (i as f32 * 7654.321_f32).cos() * 0.3
                    + (i as f32 * 3141.592_f32).sin() * 0.2;
                x.clamp(-1.0, 1.0)
            })
            .collect();

        let sfm = vad.spectral_flatness_measure(&noise);
        assert!(
            (0.0..=1.0).contains(&sfm),
            "Noise SFM must be in [0,1], got {sfm}"
        );
    }

    // ------------------------------------------------------------------
    // Mixed silence / speech sequence
    // ------------------------------------------------------------------

    #[test]
    fn test_mixed_sequence_silence_speech_silence() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 10,
            hangover_ms: 20,
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let speech = sine_frame(0.5, 480);
        let silence = silence_frame(480);

        // Silence
        for _ in 0..5 {
            let d = vad.process_frame(&silence);
            assert_eq!(d, VadDecision::Silence);
        }
        // Speech onset
        let mut saw_speech = false;
        for _ in 0..10 {
            let d = vad.process_frame(&speech);
            if d == VadDecision::Speech {
                saw_speech = true;
            }
        }
        assert!(saw_speech, "Should detect speech in speech segment");

        // Hangover + silence
        let mut returned_to_silence = false;
        for _ in 0..30 {
            if vad.process_frame(&silence) == VadDecision::Silence {
                returned_to_silence = true;
                break;
            }
        }
        assert!(returned_to_silence, "Should eventually return to Silence");
    }

    // ------------------------------------------------------------------
    // Various threshold values
    // ------------------------------------------------------------------

    #[test]
    fn test_high_threshold_treats_quiet_as_silence() {
        let cfg = VadConfig {
            threshold_db: -10.0, // very high threshold
            min_speech_ms: 10,
            hangover_ms: 20,
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        // Quiet sine at -20 dBFS (amplitude ≈ 0.1)
        let frame = sine_frame(0.1, 4800);
        let d = vad.process_frame(&frame);
        // -20 dBFS < -10 dBFS threshold → silence
        assert!(
            d != VadDecision::Speech,
            "Quiet signal below high threshold must not be Speech: {d:?}"
        );
    }

    #[test]
    fn test_is_speech_after_confirmed_speech() {
        let cfg = VadConfig {
            threshold_db: -40.0,
            min_speech_ms: 10,
            hangover_ms: 100,
        };
        let mut vad = VadDetector::new(cfg, SAMPLE_RATE);
        let speech = sine_frame(0.5, 1440); // 30 ms
        for _ in 0..5 {
            vad.process_frame(&speech);
        }
        assert!(
            vad.is_speech(),
            "is_speech() should be true after confirmed speech"
        );
    }
}
