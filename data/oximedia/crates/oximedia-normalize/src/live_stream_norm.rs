//! Live streaming normalization with adaptive loudness target.
//!
//! This module provides real-time loudness normalization for live audio streams,
//! where the content type may change (speech, music, mixed, silence) and the
//! normalization target should adapt accordingly.
//!
//! # Architecture
//!
//! ```text
//! Audio input ──► ContentClassifier ──► target selector ──► AdaptiveNorm ──► output
//! ```
//!
//! 1. **Content classification**: A short-time classifier detects whether the
//!    current segment is speech-dominant, music-dominant, or mixed.
//! 2. **Adaptive target selection**: Each content class has its own loudness target
//!    (e.g. speech at −23 LUFS, music at −14 LUFS).  The active target is
//!    transitioned smoothly when the class changes.
//! 3. **Gain computation**: A first-order gain follower tracks the running loudness
//!    and steers toward the active target with configurable attack/release.
//! 4. **Limiting**: An optional brick-wall peak limiter protects against transient
//!    clipping after the gain stage.
//!
//! # Example
//!
//! ```rust
//! use oximedia_normalize::live_stream_norm::{LiveStreamNormConfig, LiveStreamNormalizer};
//!
//! let cfg = LiveStreamNormConfig::default_broadcast();
//! let mut norm = LiveStreamNormalizer::new(cfg).expect("valid config");
//!
//! // Feed 20 ms chunks at 48 kHz / stereo
//! let chunk = vec![0.1_f32; 48000 / 50 * 2];
//! let mut output = vec![0.0_f32; chunk.len()];
//! norm.process_chunk(&chunk, &mut output).expect("process OK");
//! ```

/// Content type classification for adaptive target selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentClass {
    /// Predominantly speech / dialogue.
    Speech,
    /// Predominantly music.
    Music,
    /// Mixed speech and music.
    Mixed,
    /// Silence or near-silence.
    Silence,
}

impl ContentClass {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Speech => "speech",
            Self::Music => "music",
            Self::Mixed => "mixed",
            Self::Silence => "silence",
        }
    }
}

/// Per-content-class loudness target.
#[derive(Debug, Clone)]
pub struct ContentTargets {
    /// Target integrated loudness for speech in LUFS.
    pub speech_lufs: f64,
    /// Target integrated loudness for music in LUFS.
    pub music_lufs: f64,
    /// Target integrated loudness for mixed content in LUFS.
    pub mixed_lufs: f64,
    /// Silence threshold in LUFS (below this the gain follower is frozen).
    pub silence_threshold_lufs: f64,
}

impl ContentTargets {
    /// Broadcast targets: speech −23 LUFS, music −16 LUFS, mixed −20 LUFS.
    pub fn broadcast() -> Self {
        Self {
            speech_lufs: -23.0,
            music_lufs: -16.0,
            mixed_lufs: -20.0,
            silence_threshold_lufs: -60.0,
        }
    }

    /// Streaming targets: speech −16 LUFS, music −14 LUFS, mixed −14 LUFS.
    pub fn streaming() -> Self {
        Self {
            speech_lufs: -16.0,
            music_lufs: -14.0,
            mixed_lufs: -14.0,
            silence_threshold_lufs: -60.0,
        }
    }

    /// Return the loudness target for a given content class.
    pub fn target_for(&self, class: ContentClass) -> f64 {
        match class {
            ContentClass::Speech => self.speech_lufs,
            ContentClass::Music => self.music_lufs,
            ContentClass::Mixed => self.mixed_lufs,
            ContentClass::Silence => self.silence_threshold_lufs,
        }
    }
}

/// Configuration for the live stream normalizer.
#[derive(Debug, Clone)]
pub struct LiveStreamNormConfig {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of audio channels.
    pub channels: usize,
    /// Per-class loudness targets.
    pub targets: ContentTargets,
    /// Gain follower attack time in milliseconds.
    pub attack_ms: f64,
    /// Gain follower release time in milliseconds.
    pub release_ms: f64,
    /// Target transition smoothing time in milliseconds (when class changes).
    pub transition_ms: f64,
    /// Maximum gain the normalizer may apply in dB.
    pub max_gain_db: f64,
    /// Maximum attenuation the normalizer may apply in dB (positive value).
    pub max_attenuation_db: f64,
    /// Enable brick-wall peak limiter after gain stage.
    pub enable_limiter: bool,
    /// True peak ceiling when limiter is enabled (dBTP).
    pub true_peak_ceiling_dbtp: f64,
}

impl LiveStreamNormConfig {
    /// Create a broadcast-oriented configuration.
    pub fn default_broadcast() -> Self {
        Self {
            sample_rate: 48000.0,
            channels: 2,
            targets: ContentTargets::broadcast(),
            attack_ms: 100.0,
            release_ms: 2000.0,
            transition_ms: 500.0,
            max_gain_db: 15.0,
            max_attenuation_db: 30.0,
            enable_limiter: true,
            true_peak_ceiling_dbtp: -1.0,
        }
    }

    /// Create a streaming-oriented configuration.
    pub fn default_streaming() -> Self {
        Self {
            sample_rate: 48000.0,
            channels: 2,
            targets: ContentTargets::streaming(),
            attack_ms: 200.0,
            release_ms: 3000.0,
            transition_ms: 1000.0,
            max_gain_db: 12.0,
            max_attenuation_db: 20.0,
            enable_limiter: true,
            true_peak_ceiling_dbtp: -1.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate < 8000.0 || self.sample_rate > 192_000.0 {
            return Err(format!("Invalid sample rate: {}", self.sample_rate));
        }
        if self.channels == 0 || self.channels > 16 {
            return Err(format!("Invalid channel count: {}", self.channels));
        }
        if self.attack_ms <= 0.0 {
            return Err("attack_ms must be > 0".to_string());
        }
        if self.release_ms <= 0.0 {
            return Err("release_ms must be > 0".to_string());
        }
        if self.max_gain_db <= 0.0 || self.max_gain_db > 60.0 {
            return Err(format!(
                "max_gain_db must be in (0, 60]: {}",
                self.max_gain_db
            ));
        }
        if self.max_attenuation_db <= 0.0 || self.max_attenuation_db > 60.0 {
            return Err(format!(
                "max_attenuation_db must be in (0, 60]: {}",
                self.max_attenuation_db
            ));
        }
        Ok(())
    }
}

/// Simple content classifier based on spectral centroid heuristics.
///
/// Uses the ratio of high-frequency to low-frequency energy as a proxy:
/// speech tends to concentrate energy in the 1–4 kHz range, music is
/// more broadband, and silence has very low overall energy.
struct ContentClassifier {
    /// Sample rate.
    sample_rate: f64,
    /// Silence threshold (linear RMS).
    silence_rms_threshold: f32,
    /// Minimum fraction of energy above 1 kHz to classify as speech.
    speech_hf_min: f32,
    /// Maximum fraction of energy above 1 kHz before switching to music.
    music_hf_max: f32,
}

impl ContentClassifier {
    fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            silence_rms_threshold: 1e-5,
            speech_hf_min: 0.25,
            music_hf_max: 0.75,
        }
    }

    /// Classify a block of interleaved audio samples.
    fn classify(&self, samples: &[f32], channels: usize) -> ContentClass {
        if samples.is_empty() {
            return ContentClass::Silence;
        }

        // Compute overall RMS
        let rms = compute_rms(samples);
        if rms < self.silence_rms_threshold {
            return ContentClass::Silence;
        }

        // Compute high-frequency energy ratio via a first-order high-pass
        // filter proxy (difference filter: y[n] = x[n] - x[n-1])
        let hop = channels.max(1);
        let mut prev = 0.0_f32;
        let mut sum_hp_sq = 0.0_f64;
        let mut sum_total_sq = 0.0_f64;

        for chunk in samples.chunks_exact(hop) {
            let mono = chunk.iter().map(|&s| f64::from(s)).sum::<f64>() / hop as f64;
            let mono_f = mono as f32;
            let hp = mono_f - prev;
            prev = mono_f;
            sum_hp_sq += f64::from(hp * hp);
            sum_total_sq += mono * mono;
        }

        if sum_total_sq < 1e-12 {
            return ContentClass::Silence;
        }

        let hf_ratio = (sum_hp_sq / sum_total_sq) as f32;

        // Heuristic thresholds (empirically tuned, not based on ITU)
        if hf_ratio >= self.speech_hf_min && hf_ratio < self.music_hf_max {
            ContentClass::Speech
        } else if hf_ratio >= self.music_hf_max {
            ContentClass::Music
        } else {
            ContentClass::Mixed
        }
    }
}

/// Real-time RMS computation (mono mix).
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Convert dB to a linear amplitude multiplier.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
#[inline]
fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        -120.0
    } else {
        20.0 * linear.log10()
    }
}

/// Per-chunk statistics produced by the normalizer.
#[derive(Debug, Clone)]
pub struct ChunkStats {
    /// Detected content class.
    pub content_class: ContentClass,
    /// Active loudness target (LUFS) at the time of processing.
    pub active_target_lufs: f64,
    /// Current gain applied (dB).
    pub gain_db: f64,
    /// Whether the peak limiter was engaged.
    pub limiter_engaged: bool,
    /// RMS level of the input chunk (dBFS).
    pub input_rms_db: f64,
}

/// Live streaming normalizer with adaptive loudness target.
pub struct LiveStreamNormalizer {
    config: LiveStreamNormConfig,
    classifier: ContentClassifier,
    /// Current gain envelope in dB.
    current_gain_db: f64,
    /// Current active loudness target in LUFS.
    active_target_lufs: f64,
    /// Current content class.
    current_class: ContentClass,
    /// Running short-term loudness estimate in dBFS (exponential moving average).
    loudness_ema_db: f64,
    /// Attack coefficient (per-sample).
    attack_coeff: f64,
    /// Release coefficient (per-sample).
    release_coeff: f64,
    /// Target transition coefficient (per-sample), used when class changes.
    transition_coeff: f64,
    /// Total chunks processed.
    chunks_processed: u64,
}

impl LiveStreamNormalizer {
    /// Create a new live stream normalizer.
    ///
    /// Returns `Err` if the configuration is invalid.
    pub fn new(config: LiveStreamNormConfig) -> Result<Self, String> {
        config.validate()?;

        let sr = config.sample_rate;
        let attack_coeff = compute_envelope_coeff(config.attack_ms, sr);
        let release_coeff = compute_envelope_coeff(config.release_ms, sr);
        let transition_coeff = compute_envelope_coeff(config.transition_ms, sr);

        let initial_target = config.targets.speech_lufs;

        let classifier = ContentClassifier::new(sr);

        Ok(Self {
            config,
            classifier,
            current_gain_db: 0.0,
            active_target_lufs: initial_target,
            current_class: ContentClass::Speech,
            loudness_ema_db: initial_target,
            attack_coeff,
            release_coeff,
            transition_coeff,
            chunks_processed: 0,
        })
    }

    /// Process a chunk of audio samples, applying adaptive normalization.
    ///
    /// # Arguments
    /// * `input` – Interleaved input samples (f32).
    /// * `output` – Interleaved output buffer; must be the same length as `input`.
    ///
    /// # Returns
    /// [`ChunkStats`] describing the decisions made for this chunk.
    pub fn process_chunk(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<ChunkStats, String> {
        if input.len() != output.len() {
            return Err(format!(
                "Input length {} != output length {}",
                input.len(),
                output.len()
            ));
        }
        if input.is_empty() {
            return Ok(ChunkStats {
                content_class: ContentClass::Silence,
                active_target_lufs: self.active_target_lufs,
                gain_db: self.current_gain_db,
                limiter_engaged: false,
                input_rms_db: -120.0,
            });
        }

        // 1. Classify content
        let class = self.classifier.classify(input, self.config.channels);

        // 2. Smoothly transition target when class changes
        let new_target = self.config.targets.target_for(class);
        self.active_target_lufs = self.active_target_lufs
            + self.transition_coeff * input.len() as f64 * (new_target - self.active_target_lufs);
        self.active_target_lufs = self.active_target_lufs.clamp(-96.0, 0.0);
        self.current_class = class;

        // 3. Measure short-term RMS of input
        let rms = compute_rms(input);
        let input_rms_db = if rms > 1e-10 {
            linear_to_db(f64::from(rms))
        } else {
            -120.0
        };

        // 4. Update loudness EMA
        if class != ContentClass::Silence {
            let coeff = if input_rms_db > self.loudness_ema_db {
                // rising signal: use attack coefficient scaled for chunk size
                1.0 - (1.0 - self.attack_coeff).powi(input.len() as i32)
            } else {
                1.0 - (1.0 - self.release_coeff).powi(input.len() as i32)
            };
            self.loudness_ema_db += coeff * (input_rms_db - self.loudness_ema_db);
        }

        // 5. Compute desired gain
        let desired_gain_db = if class == ContentClass::Silence {
            // Freeze gain during silence: do not boost noise floor
            self.current_gain_db
        } else {
            let raw = self.active_target_lufs - self.loudness_ema_db;
            raw.clamp(-self.config.max_attenuation_db, self.config.max_gain_db)
        };

        // 6. Smooth gain transitions
        let coeff_gain = if desired_gain_db > self.current_gain_db {
            1.0 - (1.0 - self.attack_coeff).powi(input.len() as i32)
        } else {
            1.0 - (1.0 - self.release_coeff).powi(input.len() as i32)
        };
        self.current_gain_db += coeff_gain * (desired_gain_db - self.current_gain_db);
        self.current_gain_db = self
            .current_gain_db
            .clamp(-self.config.max_attenuation_db, self.config.max_gain_db);

        // 7. Apply gain
        let gain_linear = db_to_linear(self.current_gain_db) as f32;
        for (in_s, out_s) in input.iter().zip(output.iter_mut()) {
            *out_s = *in_s * gain_linear;
        }

        // 8. Brick-wall limiter
        let mut limiter_engaged = false;
        if self.config.enable_limiter {
            let ceiling = db_to_linear(self.config.true_peak_ceiling_dbtp) as f32;
            for s in output.iter_mut() {
                if s.abs() > ceiling {
                    *s = s.signum() * ceiling;
                    limiter_engaged = true;
                }
            }
        }

        self.chunks_processed += 1;

        Ok(ChunkStats {
            content_class: class,
            active_target_lufs: self.active_target_lufs,
            gain_db: self.current_gain_db,
            limiter_engaged,
            input_rms_db,
        })
    }

    /// Reset the normalizer to initial state.
    pub fn reset(&mut self) {
        self.current_gain_db = 0.0;
        self.active_target_lufs = self.config.targets.speech_lufs;
        self.current_class = ContentClass::Speech;
        self.loudness_ema_db = self.config.targets.speech_lufs;
        self.chunks_processed = 0;
    }

    /// Current gain being applied in dB.
    pub fn current_gain_db(&self) -> f64 {
        self.current_gain_db
    }

    /// Current content class.
    pub fn current_class(&self) -> ContentClass {
        self.current_class
    }

    /// Active loudness target in LUFS.
    pub fn active_target_lufs(&self) -> f64 {
        self.active_target_lufs
    }

    /// Total chunks processed.
    pub fn chunks_processed(&self) -> u64 {
        self.chunks_processed
    }

    /// Get the configuration.
    pub fn config(&self) -> &LiveStreamNormConfig {
        &self.config
    }
}

/// Compute a first-order smoothing coefficient from a time constant in ms.
///
/// `coeff = exp(-1 / (time_ms * sr / 1000))`
fn compute_envelope_coeff(time_ms: f64, sample_rate: f64) -> f64 {
    if time_ms <= 0.0 || sample_rate <= 0.0 {
        return 0.0;
    }
    let tau_samples = time_ms * sample_rate / 1000.0;
    (-1.0 / tau_samples).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ContentClass ───────────────────────────────────────────────────────

    #[test]
    fn test_content_class_labels() {
        assert_eq!(ContentClass::Speech.label(), "speech");
        assert_eq!(ContentClass::Music.label(), "music");
        assert_eq!(ContentClass::Mixed.label(), "mixed");
        assert_eq!(ContentClass::Silence.label(), "silence");
    }

    // ─── ContentTargets ─────────────────────────────────────────────────────

    #[test]
    fn test_content_targets_broadcast() {
        let t = ContentTargets::broadcast();
        assert!((t.speech_lufs - (-23.0)).abs() < f64::EPSILON);
        assert!((t.music_lufs - (-16.0)).abs() < f64::EPSILON);
        assert!((t.mixed_lufs - (-20.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_targets_streaming() {
        let t = ContentTargets::streaming();
        assert!((t.speech_lufs - (-16.0)).abs() < f64::EPSILON);
        assert!((t.music_lufs - (-14.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_content_targets_target_for() {
        let t = ContentTargets::broadcast();
        assert!((t.target_for(ContentClass::Speech) - (-23.0)).abs() < f64::EPSILON);
        assert!((t.target_for(ContentClass::Music) - (-16.0)).abs() < f64::EPSILON);
        assert!((t.target_for(ContentClass::Mixed) - (-20.0)).abs() < f64::EPSILON);
    }

    // ─── LiveStreamNormConfig ───────────────────────────────────────────────

    #[test]
    fn test_config_broadcast_valid() {
        assert!(LiveStreamNormConfig::default_broadcast().validate().is_ok());
    }

    #[test]
    fn test_config_streaming_valid() {
        assert!(LiveStreamNormConfig::default_streaming().validate().is_ok());
    }

    #[test]
    fn test_config_invalid_sample_rate() {
        let mut cfg = LiveStreamNormConfig::default_broadcast();
        cfg.sample_rate = 100.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_channels() {
        let mut cfg = LiveStreamNormConfig::default_broadcast();
        cfg.channels = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_invalid_attack() {
        let mut cfg = LiveStreamNormConfig::default_broadcast();
        cfg.attack_ms = 0.0;
        assert!(cfg.validate().is_err());
    }

    // ─── LiveStreamNormalizer ───────────────────────────────────────────────

    #[test]
    fn test_normalizer_creation() {
        let norm = LiveStreamNormalizer::new(LiveStreamNormConfig::default_broadcast());
        assert!(norm.is_ok());
    }

    #[test]
    fn test_normalizer_initial_state() {
        let norm = LiveStreamNormalizer::new(LiveStreamNormConfig::default_broadcast())
            .expect("valid config");
        assert!((norm.current_gain_db()).abs() < f64::EPSILON);
        assert_eq!(norm.chunks_processed(), 0);
    }

    #[test]
    fn test_process_chunk_silence() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let input = vec![0.0_f32; 960]; // 20 ms at 48 kHz stereo
        let mut output = vec![0.0_f32; 960];
        let stats = norm.process_chunk(&input, &mut output).expect("ok");
        assert_eq!(stats.content_class, ContentClass::Silence);
        assert_eq!(norm.chunks_processed(), 1);
    }

    #[test]
    fn test_process_chunk_output_length() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let n = 960;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.1)
            .collect();
        let mut output = vec![0.0_f32; n];
        norm.process_chunk(&input, &mut output).expect("ok");
        assert_eq!(output.len(), n);
    }

    #[test]
    fn test_process_chunk_mismatched_lengths() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let input = vec![0.0_f32; 960];
        let mut output = vec![0.0_f32; 480];
        assert!(norm.process_chunk(&input, &mut output).is_err());
    }

    #[test]
    fn test_process_chunk_output_is_finite() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let n = 960;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.3)
            .collect();
        let mut output = vec![0.0_f32; n];
        norm.process_chunk(&input, &mut output).expect("ok");
        assert!(
            output.iter().all(|s| s.is_finite()),
            "output must be finite"
        );
    }

    #[test]
    fn test_process_multiple_chunks_incrementing_counter() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let n = 480;
        for _ in 0..5 {
            let input = vec![0.1_f32; n];
            let mut output = vec![0.0_f32; n];
            norm.process_chunk(&input, &mut output).expect("ok");
        }
        assert_eq!(norm.chunks_processed(), 5);
    }

    #[test]
    fn test_limiter_ceiling_respected() {
        let mut cfg = LiveStreamNormConfig::default_broadcast();
        cfg.enable_limiter = true;
        cfg.true_peak_ceiling_dbtp = -1.0;
        cfg.max_gain_db = 20.0;
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");

        // Loud signal that would trigger gain boost then clip
        let mut samples = vec![0.9_f32; 960];
        let mut output = vec![0.0_f32; 960];
        norm.process_chunk(&samples, &mut output).expect("ok");

        let ceiling = 10.0_f32.powf(-1.0_f32 / 20.0);
        for &s in &output {
            assert!(
                s.abs() <= ceiling + 1e-5,
                "sample {} exceeds ceiling {}",
                s,
                ceiling
            );
        }
        drop(samples.iter_mut()); // suppress unused mut warning
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let mut norm = LiveStreamNormalizer::new(cfg).expect("valid");
        let n = 480;
        let input = vec![0.5_f32; n];
        let mut output = vec![0.0_f32; n];
        norm.process_chunk(&input, &mut output).expect("ok");
        assert!(norm.chunks_processed() > 0);
        norm.reset();
        assert_eq!(norm.chunks_processed(), 0);
    }

    #[test]
    fn test_active_target_initialized_to_speech() {
        let cfg = LiveStreamNormConfig::default_broadcast();
        let norm = LiveStreamNormalizer::new(cfg.clone()).expect("valid");
        assert!((norm.active_target_lufs() - cfg.targets.speech_lufs).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_envelope_coeff_range() {
        // Coefficient must be in (0, 1) for positive time constants
        let coeff = compute_envelope_coeff(100.0, 48000.0);
        assert!(
            coeff > 0.0 && coeff < 1.0,
            "coefficient must be in (0,1), got {}",
            coeff
        );
    }

    #[test]
    fn test_compute_envelope_coeff_longer_slower() {
        // Longer time constant → coefficient closer to 1 → slower response
        let c100ms = compute_envelope_coeff(100.0, 48000.0);
        let c1s = compute_envelope_coeff(1000.0, 48000.0);
        assert!(
            c1s > c100ms,
            "longer time constant should give higher (slower) coefficient: {} vs {}",
            c1s,
            c100ms
        );
    }

    // ─── ContentClassifier ─────────────────────────────────────────────────

    #[test]
    fn test_classifier_silence() {
        let classifier = ContentClassifier::new(48000.0);
        let samples = vec![0.0_f32; 480];
        let class = classifier.classify(&samples, 2);
        assert_eq!(class, ContentClass::Silence);
    }

    #[test]
    fn test_classifier_non_silence_on_signal() {
        let classifier = ContentClassifier::new(48000.0);
        let n = 480;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin() * 0.5)
            .collect();
        let class = classifier.classify(&samples, 1);
        assert_ne!(
            class,
            ContentClass::Silence,
            "expected non-silence classification for a sine wave"
        );
    }
}
