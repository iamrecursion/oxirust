//! Audio scene classification using spectral features.
//!
//! Classifies audio into seven semantic categories:
//! `Indoor`, `Outdoor`, `Quiet`, `Noisy`, `Speech`, `Music`, `Mixed`.
//!
//! The classifier extracts three lightweight time-domain and spectral
//! descriptors — spectral centroid, zero-crossing rate, and short-term energy —
//! and maps them to scene labels via hand-tuned decision thresholds. No
//! machine-learning training data or weights are required; the thresholds are
//! derived from established acoustic literature on indoor/outdoor and
//! speech/music discrimination.

use crate::{compute_rms, zero_crossing_rate};

/// Seven-way audio scene label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioScene {
    /// Indoor room environment (low reverberation, relatively quiet background).
    Indoor,
    /// Outdoor environment (wind, traffic, nature sounds, etc.).
    Outdoor,
    /// Very quiet environment — near-silence or minimal ambient sound.
    Quiet,
    /// Highly noisy environment with broadband masking noise.
    Noisy,
    /// Predominantly speech content.
    Speech,
    /// Predominantly music content (harmonic, rhythmic energy).
    Music,
    /// Mixed content — no single category dominates.
    Mixed,
}

impl AudioScene {
    /// Human-readable label for the scene.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Indoor => "Indoor",
            Self::Outdoor => "Outdoor",
            Self::Quiet => "Quiet",
            Self::Noisy => "Noisy",
            Self::Speech => "Speech",
            Self::Music => "Music",
            Self::Mixed => "Mixed",
        }
    }
}

/// Per-frame acoustic features used by the classifier.
#[derive(Debug, Clone)]
pub struct SceneFeatures {
    /// Spectral centroid in normalised units (0..1 relative to Nyquist).
    pub spectral_centroid_norm: f32,
    /// Zero-crossing rate (0..1, fraction of sign changes per sample).
    pub zcr: f32,
    /// Short-term RMS energy (linear).
    pub energy: f32,
    /// Spectral flatness (0 = tonal, 1 = white noise).
    pub flatness: f32,
    /// Spectral rolloff at 85th-percentile energy (normalised 0..1).
    pub rolloff_norm: f32,
}

/// Configuration for [`AudioSceneClassifier`].
#[derive(Debug, Clone)]
pub struct SceneClassifierConfig {
    /// FFT size used for spectral analysis.
    pub fft_size: usize,
    /// Hop length between analysis frames.
    pub hop_size: usize,
    /// Energy threshold below which a frame is considered silent.
    pub silence_threshold: f32,
    /// Fraction of silent frames above which the scene is `Quiet`.
    pub quiet_frame_fraction: f32,
}

impl Default for SceneClassifierConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            silence_threshold: 0.005,
            quiet_frame_fraction: 0.85,
        }
    }
}

/// Classification result including the predicted label and per-class scores.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted scene label.
    pub scene: AudioScene,
    /// Confidence score for the predicted label (0..1).
    pub confidence: f32,
    /// Aggregate acoustic features averaged over the input.
    pub features: SceneFeatures,
}

/// Threshold-based audio scene classifier.
///
/// Extracts spectral centroid, ZCR, RMS energy, spectral flatness, and
/// spectral rolloff from overlapping frames and classifies the aggregate
/// feature vector using a decision-tree of acoustic thresholds.
pub struct AudioSceneClassifier {
    config: SceneClassifierConfig,
}

impl AudioSceneClassifier {
    /// Create a new classifier with the given configuration.
    #[must_use]
    pub fn new(config: SceneClassifierConfig) -> Self {
        Self { config }
    }

    /// Classify the audio scene of the provided mono samples.
    ///
    /// # Arguments
    /// * `samples` - Mono audio samples (f32).
    /// * `sample_rate` - Audio sample rate in Hz.
    ///
    /// # Returns
    /// [`ClassificationResult`] with scene label, confidence, and features.
    ///
    /// # Errors
    /// Returns [`crate::AnalysisError`] if samples are too short.
    pub fn classify(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> crate::Result<ClassificationResult> {
        if samples.len() < self.config.fft_size {
            return Err(crate::AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        let features = self.extract_features(samples, sample_rate);
        let (scene, confidence) = classify_features(&features, &self.config, samples);

        Ok(ClassificationResult {
            scene,
            confidence,
            features,
        })
    }

    /// Extract aggregate acoustic features from the full signal.
    fn extract_features(&self, samples: &[f32], sample_rate: u32) -> SceneFeatures {
        let n_fft = self.config.fft_size;
        let hop = self.config.hop_size;
        let n_bins = n_fft / 2 + 1;
        let nyquist = sample_rate as f32 / 2.0;

        // Pre-compute Hann window
        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let x = std::f64::consts::PI * i as f64 / (n_fft - 1) as f64;
                (0.5 * (1.0 - x.cos())) as f32
            })
            .collect();

        let num_frames = (samples.len().saturating_sub(n_fft)) / hop + 1;
        let mut sum_centroid = 0.0_f64;
        let mut sum_zcr = 0.0_f64;
        let mut sum_energy = 0.0_f64;
        let mut sum_flatness = 0.0_f64;
        let mut sum_rolloff = 0.0_f64;
        let mut counted = 0_usize;

        for fi in 0..num_frames {
            let start = fi * hop;
            let end = start + n_fft;
            if end > samples.len() {
                break;
            }

            let frame = &samples[start..end];

            // RMS energy of raw frame
            let energy = compute_rms(frame);
            sum_energy += f64::from(energy);

            // ZCR of raw frame
            let zcr = zero_crossing_rate(frame);
            sum_zcr += f64::from(zcr);

            // Windowed magnitude spectrum via OxiFFT
            let windowed: Vec<oxifft::Complex<f64>> = frame
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| oxifft::Complex::new(f64::from(s * w), 0.0))
                .collect();

            let spectrum = oxifft::fft(&windowed);
            let magnitude: Vec<f32> = spectrum[..n_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
                .collect();

            sum_centroid += f64::from(compute_spectral_centroid_norm(
                &magnitude,
                n_fft,
                sample_rate,
            ));
            sum_flatness += f64::from(compute_spectral_flatness(&magnitude));
            sum_rolloff += f64::from(compute_spectral_rolloff_norm(
                &magnitude,
                n_fft,
                sample_rate,
            ));

            counted += 1;
        }

        let n = counted.max(1) as f64;
        SceneFeatures {
            spectral_centroid_norm: (sum_centroid / n / f64::from(nyquist)) as f32,
            zcr: (sum_zcr / n) as f32,
            energy: (sum_energy / n) as f32,
            flatness: (sum_flatness / n) as f32,
            rolloff_norm: (sum_rolloff / n) as f32,
        }
    }
}

// ── Spectral descriptors ──────────────────────────────────────────────────────

/// Spectral centroid in Hz.
fn compute_spectral_centroid_norm(magnitude: &[f32], n_fft: usize, sample_rate: u32) -> f32 {
    let mut weighted = 0.0_f32;
    let mut total = 0.0_f32;
    for (k, &m) in magnitude.iter().enumerate() {
        let freq = k as f32 * sample_rate as f32 / n_fft as f32;
        weighted += freq * m;
        total += m;
    }
    if total > 0.0 {
        weighted / total
    } else {
        0.0
    }
}

/// Wiener spectral flatness in [0, 1].
fn compute_spectral_flatness(magnitude: &[f32]) -> f32 {
    let n = magnitude.len();
    if n == 0 {
        return 0.0;
    }
    let eps = 1e-10_f64;
    let log_sum: f64 = magnitude.iter().map(|&m| (f64::from(m) + eps).ln()).sum();
    let geo_mean = (log_sum / n as f64).exp();
    let arith_mean: f64 = magnitude.iter().map(|&m| f64::from(m) + eps).sum::<f64>() / n as f64;
    if arith_mean > 0.0 {
        (geo_mean / arith_mean).clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// 85th-percentile spectral rolloff as a fraction of Nyquist.
fn compute_spectral_rolloff_norm(magnitude: &[f32], n_fft: usize, sample_rate: u32) -> f32 {
    let total_energy: f32 = magnitude.iter().map(|&m| m * m).sum();
    if total_energy <= 0.0 {
        return 0.0;
    }
    let threshold = 0.85 * total_energy;
    let mut cum = 0.0_f32;
    let nyquist = sample_rate as f32 / 2.0;
    for (k, &m) in magnitude.iter().enumerate() {
        cum += m * m;
        if cum >= threshold {
            let freq = k as f32 * sample_rate as f32 / n_fft as f32;
            return (freq / nyquist).clamp(0.0, 1.0);
        }
    }
    1.0
}

// ── Decision logic ────────────────────────────────────────────────────────────

/// Map the aggregate feature vector to a scene label and confidence.
///
/// Decision thresholds are derived from acoustic analysis literature:
///
/// - **Quiet**: overall RMS below silence threshold on most frames.
/// - **Speech**: centroid in speech formant range (500–3500 Hz), ZCR moderate,
///   flatness low (tonal/harmonic content).
/// - **Music**: centroid spans wider range, rolloff higher, flatness moderate,
///   energy sustained.
/// - **Noisy**: high flatness (broadband), high energy.
/// - **Indoor/Outdoor**: residual classification based on centroid and rolloff.
/// - **Mixed**: none of the above clearly holds.
fn classify_features(
    f: &SceneFeatures,
    cfg: &SceneClassifierConfig,
    samples: &[f32],
) -> (AudioScene, f32) {
    // Silence / quiet check
    let silent_fraction = {
        let hop = cfg.hop_size;
        let n_fft = cfg.fft_size;
        let num_frames = (samples.len().saturating_sub(n_fft)) / hop + 1;
        let silent: usize = (0..num_frames)
            .filter(|&fi| {
                let start = fi * hop;
                let end = (start + n_fft).min(samples.len());
                compute_rms(&samples[start..end]) < cfg.silence_threshold
            })
            .count();
        if num_frames > 0 {
            silent as f32 / num_frames as f32
        } else {
            1.0
        }
    };

    if silent_fraction >= cfg.quiet_frame_fraction {
        return (AudioScene::Quiet, 0.85 + 0.15 * silent_fraction);
    }

    // Noisy: high flatness + high energy
    if f.flatness > 0.55 && f.energy > 0.06 {
        let conf = (f.flatness * 0.7 + (f.energy / 0.5).min(1.0) * 0.3).min(1.0);
        return (AudioScene::Noisy, conf);
    }

    // Speech: centroid in 500–3500 Hz, moderate ZCR, low flatness
    // spectral_centroid_norm is centroid_hz / nyquist
    // For 22050 Hz Nyquist: 500/22050 ≈ 0.023, 3500/22050 ≈ 0.159
    // For 8000 Hz Nyquist:  500/8000  ≈ 0.063, 3500/8000  ≈ 0.437
    // We use relative thresholds flexible across sample rates:
    let is_speech_centroid = f.spectral_centroid_norm > 0.02 && f.spectral_centroid_norm < 0.30;
    let is_speech_zcr = f.zcr > 0.02 && f.zcr < 0.30;
    let is_speech_flatness = f.flatness < 0.40;
    if is_speech_centroid && is_speech_zcr && is_speech_flatness && f.energy > cfg.silence_threshold
    {
        let centroid_score = 1.0 - (f.spectral_centroid_norm - 0.10).abs() / 0.20;
        let zcr_score = 1.0 - (f.zcr - 0.10).abs() / 0.20;
        let conf = ((centroid_score + zcr_score) / 2.0).clamp(0.5, 0.95);
        return (AudioScene::Speech, conf);
    }

    // Music: tonal (low flatness), wide spectral rolloff, sustained energy
    let is_music_flatness = f.flatness < 0.35;
    let is_music_rolloff = f.rolloff_norm > 0.20;
    let is_music_energy = f.energy > cfg.silence_threshold * 2.0;
    if is_music_flatness && is_music_rolloff && is_music_energy {
        let roll_score = (f.rolloff_norm * 2.0).min(1.0);
        let conf = (0.5 + roll_score * 0.4).min(0.95);
        return (AudioScene::Music, conf);
    }

    // Outdoor: high centroid (wind) + moderate flatness
    if f.spectral_centroid_norm > 0.25 && f.flatness > 0.30 {
        return (AudioScene::Outdoor, 0.65);
    }

    // Indoor: lower centroid, low flatness (reverberant room tone)
    if f.spectral_centroid_norm < 0.20 && f.flatness < 0.40 {
        return (AudioScene::Indoor, 0.60);
    }

    // Default: mixed
    (AudioScene::Mixed, 0.50)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sample_rate: u32, duration_secs: f64) -> Vec<f32> {
        let num = (f64::from(sample_rate) * duration_secs) as usize;
        (0..num)
            .map(|i| {
                (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(sample_rate)).sin() as f32
            })
            .collect()
    }

    fn white_noise(n: usize, amplitude: f32) -> Vec<f32> {
        // Simple PRNG-based white noise (no rand dependency)
        let mut state: u64 = 0xDEAD_BEEF_1234_5678;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let v = (state as i64 as f64) / (i64::MAX as f64);
                v as f32 * amplitude
            })
            .collect()
    }

    #[test]
    fn test_classify_quiet_silence() {
        let silence = vec![0.0_f32; 44100];
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let result = classifier
            .classify(&silence, 44100)
            .expect("should succeed");
        assert_eq!(result.scene, AudioScene::Quiet, "silence should be Quiet");
    }

    #[test]
    fn test_classify_noisy_white_noise() {
        let noise = white_noise(44100, 0.5);
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let result = classifier.classify(&noise, 44100).expect("should succeed");
        // High amplitude white noise should be classified as Noisy or Mixed
        assert!(
            matches!(result.scene, AudioScene::Noisy | AudioScene::Mixed),
            "white noise should be Noisy/Mixed, got {:?}",
            result.scene
        );
    }

    #[test]
    fn test_classify_music_tonal_sine() {
        // A single pure sine should be very tonal → Music
        let samples = sine_wave(440.0, 44100, 1.0);
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let result = classifier
            .classify(&samples, 44100)
            .expect("should succeed");
        assert!(
            matches!(
                result.scene,
                AudioScene::Music | AudioScene::Speech | AudioScene::Indoor
            ),
            "pure tone should be Music/Speech/Indoor, got {:?}",
            result.scene
        );
    }

    #[test]
    fn test_classify_insufficient_samples() {
        let short = vec![0.0_f32; 100];
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        assert!(classifier.classify(&short, 44100).is_err());
    }

    #[test]
    fn test_features_energy_for_sine() {
        let samples = sine_wave(440.0, 44100, 0.5);
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let features = classifier.extract_features(&samples, 44100);
        // RMS of a unit sine is 1/sqrt(2) ≈ 0.707
        assert!(
            features.energy > 0.5 && features.energy < 0.8,
            "RMS energy of unit sine should be ~0.707, got {}",
            features.energy
        );
    }

    #[test]
    fn test_features_flatness_range() {
        let noise = white_noise(22050, 0.3);
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let f = classifier.extract_features(&noise, 22050);
        assert!(
            f.flatness >= 0.0 && f.flatness <= 1.0,
            "flatness out of range: {}",
            f.flatness
        );
    }

    #[test]
    fn test_scene_label_strings() {
        assert_eq!(AudioScene::Indoor.label(), "Indoor");
        assert_eq!(AudioScene::Outdoor.label(), "Outdoor");
        assert_eq!(AudioScene::Quiet.label(), "Quiet");
        assert_eq!(AudioScene::Noisy.label(), "Noisy");
        assert_eq!(AudioScene::Speech.label(), "Speech");
        assert_eq!(AudioScene::Music.label(), "Music");
        assert_eq!(AudioScene::Mixed.label(), "Mixed");
    }

    #[test]
    fn test_confidence_in_range() {
        let samples = sine_wave(1000.0, 22050, 0.5);
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let result = classifier.classify(&samples, 22050).expect("ok");
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "confidence {} out of [0,1]",
            result.confidence
        );
    }

    #[test]
    fn test_classify_low_energy_is_quiet() {
        // Very low amplitude signal should be Quiet
        let samples: Vec<f32> = sine_wave(440.0, 44100, 0.5)
            .into_iter()
            .map(|s| s * 0.001) // multiply down to near-silence
            .collect();
        let classifier = AudioSceneClassifier::new(SceneClassifierConfig::default());
        let result = classifier.classify(&samples, 44100).expect("ok");
        assert_eq!(
            result.scene,
            AudioScene::Quiet,
            "very low amplitude should be Quiet, got {:?}",
            result.scene
        );
    }

    #[test]
    fn test_spectral_flatness_pure_tone() {
        // A single-bin spike → flatness near 0
        let mut mag = vec![0.0_f32; 513];
        mag[100] = 1.0;
        let flatness = compute_spectral_flatness(&mag);
        assert!(
            flatness < 0.05,
            "pure tone flatness should be near 0, got {}",
            flatness
        );
    }

    #[test]
    fn test_spectral_flatness_uniform() {
        // Uniform spectrum → flatness = 1
        let mag = vec![1.0_f32; 512];
        let flatness = compute_spectral_flatness(&mag);
        assert!(
            (flatness - 1.0).abs() < 0.01,
            "uniform flatness should be ~1, got {}",
            flatness
        );
    }
}
