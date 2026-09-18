//! Music mood and emotion detection using the Russell circumplex model.
//!
//! Implements a rule-based valence/arousal classifier that maps low-level
//! audio features to eight discrete emotional categories.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ── Mood enum ─────────────────────────────────────────────────────────────────

/// Discrete emotional categories derived from the Russell circumplex model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    /// High valence, high arousal – fast tempo, bright timbre, major mode.
    Happy,
    /// Low valence, low arousal – slow tempo, dark timbre, minor mode.
    Sad,
    /// High arousal, moderate-to-high valence – loud, energetic, percussive.
    Energetic,
    /// Low arousal, high valence – slow, soft, warm timbre.
    Calm,
    /// Low valence, very high arousal – loud, harsh, percussive, minor mode.
    Aggressive,
    /// Moderate-to-high valence, low arousal – warm timbre, moderate tempo.
    Romantic,
    /// Low valence, low-to-moderate arousal – slow, minor mode, introspective.
    Melancholic,
    /// Ambiguous or contradictory features.
    Neutral,
}

impl Mood {
    /// Human-readable label for this mood.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Energetic => "energetic",
            Self::Calm => "calm",
            Self::Aggressive => "aggressive",
            Self::Romantic => "romantic",
            Self::Melancholic => "melancholic",
            Self::Neutral => "neutral",
        }
    }
}

// ── MoodFeatures ──────────────────────────────────────────────────────────────

/// Low-level audio features used as inputs to the mood classifier.
#[derive(Debug, Clone)]
pub struct MoodFeatures {
    /// Affective valence in [−1.0, 1.0].  −1 = strongly negative, +1 = strongly positive.
    pub valence: f32,
    /// Affective arousal in [−1.0, 1.0].  −1 = very calm, +1 = very energetic.
    pub arousal: f32,
    /// Estimated tempo in beats per minute.
    pub tempo_bpm: f32,
    /// Tonality mode: 0.0 = pure minor, 1.0 = pure major.
    pub mode: f32,
    /// Spectral brightness: energy fraction above 1 kHz.
    pub spectral_brightness: f32,
    /// RMS energy of the audio signal (0.0–1.0).
    pub energy: f32,
    /// Rhythmic regularity proxy (auto-correlation at 4 beat lag, normalised 0..1).
    pub danceability: f32,
}

// ── MoodResult ────────────────────────────────────────────────────────────────

/// Full output of the mood-detection pipeline.
#[derive(Debug, Clone)]
pub struct MoodResult {
    /// The dominant mood category.
    pub primary_mood: Mood,
    /// Classifier confidence (0.0–1.0).
    pub confidence: f32,
    /// Continuous valence score (−1.0 to 1.0).
    pub valence: f32,
    /// Continuous arousal score (−1.0 to 1.0).
    pub arousal: f32,
    /// All extracted features that drove the classification.
    pub features: MoodFeatures,
}

// ── MoodDetector ──────────────────────────────────────────────────────────────

/// Stateless mood detector; all methods take their inputs explicitly.
#[derive(Debug, Clone, Default)]
pub struct MoodDetector;

impl MoodDetector {
    /// Create a new `MoodDetector`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    // ── Feature extraction ─────────────────────────────────────────────────

    /// Extract [`MoodFeatures`] from raw audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples`     – mono audio samples (f32)
    /// * `sample_rate` – sample rate in Hz
    /// * `tempo_bpm`   – pre-computed tempo estimate
    /// * `key_mode`    – tonality mode (0.0 = minor, 1.0 = major)
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn extract_features(
        &self,
        samples: &[f32],
        sample_rate: u32,
        tempo_bpm: f32,
        key_mode: f32,
    ) -> MoodFeatures {
        let sr = sample_rate as f32;

        // --- RMS energy ---
        let energy = if samples.is_empty() {
            0.0
        } else {
            let sum_sq: f32 = samples.iter().map(|x| x * x).sum();
            (sum_sq / samples.len() as f32).sqrt().clamp(0.0, 1.0)
        };

        // --- Spectral brightness (high-frequency energy ratio) ---
        // Approximate using DFT magnitudes via Goertzel / naive DFT on downsampled windows.
        let spectral_brightness = compute_spectral_brightness(samples, sr);

        // --- Danceability: normalised auto-correlation at 4-beat lag ---
        let danceability = if tempo_bpm > 0.0 {
            compute_danceability(samples, sr, tempo_bpm)
        } else {
            0.0
        };

        // --- Valence heuristic: mode + spectral brightness + tempo ---
        // Weighted combination; major mode and bright timbre → positive valence.
        let valence = compute_valence(key_mode, spectral_brightness, tempo_bpm, energy);

        // --- Arousal heuristic: energy + tempo + brightness ---
        let arousal = compute_arousal(energy, tempo_bpm, spectral_brightness);

        MoodFeatures {
            valence,
            arousal,
            tempo_bpm,
            mode: key_mode.clamp(0.0, 1.0),
            spectral_brightness,
            energy,
            danceability,
        }
    }

    // ── Classification ─────────────────────────────────────────────────────

    /// Classify a set of [`MoodFeatures`] into a [`MoodResult`] using the
    /// Russell circumplex rule set.
    #[must_use]
    pub fn classify(&self, features: &MoodFeatures) -> MoodResult {
        let v = features.valence; // −1..1
        let a = features.arousal; // −1..1

        // Quadrant thresholds (non-symmetric to match common musical intuition)
        const POS_V: f32 = 0.1;
        const NEG_V: f32 = -0.1;
        const HIGH_A: f32 = 0.2;
        const LOW_A: f32 = -0.1;

        let (primary_mood, confidence) = if v > POS_V && a > HIGH_A {
            // Upper-right: Happy (energetic + positive)
            let conf = ((v + 1.0) * 0.5 * 0.6 + (a + 1.0) * 0.5 * 0.4).clamp(0.0, 1.0);
            (Mood::Happy, conf)
        } else if v < NEG_V && a > HIGH_A {
            // Upper-left: Aggressive (energetic + negative)
            // Distinguish from Energetic by mode (minor → Aggressive)
            if features.mode < 0.4 {
                let conf = ((1.0 - v) * 0.5 * 0.5 + (a + 1.0) * 0.5 * 0.5).clamp(0.0, 1.0);
                (Mood::Aggressive, conf)
            } else {
                let conf = ((a + 1.0) * 0.5).clamp(0.0, 1.0);
                (Mood::Energetic, conf)
            }
        } else if v > POS_V && a < LOW_A {
            // Lower-right: Calm or Romantic
            if features.mode > 0.5 && features.tempo_bpm > 60.0 && features.tempo_bpm < 120.0 {
                let conf = ((v + 1.0) * 0.5 * 0.5 + (1.0 - a) * 0.5 * 0.5).clamp(0.0, 1.0);
                (Mood::Romantic, conf)
            } else {
                let conf = ((v + 1.0) * 0.5 * 0.4 + (1.0 - a) * 0.5 * 0.6).clamp(0.0, 1.0);
                (Mood::Calm, conf)
            }
        } else if v < NEG_V && a < LOW_A {
            // Lower-left: Sad or Melancholic
            if a < -0.4 {
                let conf = ((1.0 - v) * 0.5 * 0.5 + (1.0 - a) * 0.5 * 0.5).clamp(0.0, 1.0);
                (Mood::Sad, conf)
            } else {
                let conf = ((1.0 - v) * 0.5 * 0.6 + (1.0 - a) * 0.5 * 0.4).clamp(0.0, 1.0);
                (Mood::Melancholic, conf)
            }
        } else if a > HIGH_A {
            // High arousal without strong valence polarity → Energetic
            let conf = ((a + 1.0) * 0.5).clamp(0.0, 1.0);
            (Mood::Energetic, conf)
        } else {
            // Ambiguous
            (Mood::Neutral, 0.5_f32)
        };

        MoodResult {
            primary_mood,
            confidence,
            valence: v,
            arousal: a,
            features: features.clone(),
        }
    }

    // ── Convenience pipeline ───────────────────────────────────────────────

    /// Convenience method: extract features and classify in one call.
    #[must_use]
    pub fn detect(
        &self,
        samples: &[f32],
        sample_rate: u32,
        tempo_bpm: f32,
        key_mode: f32,
    ) -> MoodResult {
        let features = self.extract_features(samples, sample_rate, tempo_bpm, key_mode);
        self.classify(&features)
    }
}

// ── Internal helper functions ─────────────────────────────────────────────────

/// Compute spectral brightness: fraction of signal energy above 1 kHz.
///
/// Uses a simple windowed DFT on the first 2048 samples to approximate
/// the high-frequency energy ratio.
fn compute_spectral_brightness(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() || sample_rate <= 0.0 {
        return 0.0;
    }

    let window_size = 2048_usize.min(samples.len());
    let half = window_size / 2;
    if half == 0 {
        return 0.0;
    }

    // Hann window + DFT via naive O(N²) computation over a small window
    let bin_hz = sample_rate / window_size as f32;
    let cutoff_bin = (1000.0 / bin_hz).round() as usize;
    let cutoff_bin = cutoff_bin.min(half);

    let mut high_energy = 0.0_f32;
    let mut total_energy = 0.0_f32;

    for k in 0..half {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (n, &s) in samples[..window_size].iter().enumerate() {
            let phase = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / window_size as f32;
            re += s * phase.cos();
            im += s * phase.sin();
        }
        let mag_sq = re * re + im * im;
        total_energy += mag_sq;
        if k >= cutoff_bin {
            high_energy += mag_sq;
        }
    }

    if total_energy > 0.0 {
        (high_energy / total_energy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute danceability via auto-correlation at the 4-beat lag.
fn compute_danceability(samples: &[f32], sample_rate: f32, tempo_bpm: f32) -> f32 {
    if samples.is_empty() || tempo_bpm <= 0.0 {
        return 0.0;
    }

    // Lag = 4 beats in samples
    let beat_samples = (60.0 * sample_rate / tempo_bpm) as usize;
    let lag = 4 * beat_samples;

    if lag >= samples.len() {
        return 0.0;
    }

    let n = samples.len() - lag;
    let numerator: f32 = (0..n).map(|i| samples[i] * samples[i + lag]).sum();
    let denom_a: f32 = (0..n).map(|i| samples[i] * samples[i]).sum();
    let denom_b: f32 = (lag..lag + n).map(|i| samples[i] * samples[i]).sum();

    let denom = (denom_a * denom_b).sqrt();
    if denom > 0.0 {
        (numerator / denom).abs().clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Heuristic valence from mode, spectral brightness, tempo and energy.
fn compute_valence(mode: f32, brightness: f32, tempo_bpm: f32, energy: f32) -> f32 {
    // Major mode → positive valence contribution
    let mode_contribution = (mode - 0.5) * 0.8;

    // Bright timbre → positive valence
    let brightness_contribution = (brightness - 0.5) * 0.6;

    // Mid-range tempo (90–130 BPM) is slightly positive; extremes are neutral
    let tempo_contribution = {
        let t = ((tempo_bpm - 100.0) / 80.0).clamp(-1.0, 1.0);
        t * 0.2
    };

    // Energy has a small positive association with valence (active = alert)
    let energy_contribution = (energy - 0.5) * 0.1;

    (mode_contribution + brightness_contribution + tempo_contribution + energy_contribution)
        .clamp(-1.0, 1.0)
}

/// Heuristic arousal from RMS energy, tempo, and spectral brightness.
fn compute_arousal(energy: f32, tempo_bpm: f32, brightness: f32) -> f32 {
    // Energy is the primary driver (0.4 weight)
    let energy_contribution = (energy - 0.3) * 1.4;

    // Fast tempo → high arousal
    let tempo_contribution = (tempo_bpm - 110.0) / 100.0;

    // Bright timbre → slightly higher arousal
    let brightness_contribution = (brightness - 0.5) * 0.4;

    (energy_contribution * 0.5 + tempo_contribution * 0.35 + brightness_contribution * 0.15)
        .clamp(-1.0, 1.0)
}

// ── Backward-compatible types from the original module ───────────────────────
// (kept so any existing code that imports from this module continues to compile)

/// Emotional valence level (negative ↔ positive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValenceLevel {
    /// Strongly negative affect.
    VeryNegative,
    /// Mildly negative affect.
    Negative,
    /// Neutral / ambiguous valence.
    Neutral,
    /// Mildly positive affect.
    Positive,
    /// Strongly positive affect.
    VeryPositive,
}

impl ValenceLevel {
    /// Numeric score in [−1.0, 1.0].
    #[must_use]
    pub fn score(&self) -> f32 {
        match self {
            Self::VeryNegative => -1.0,
            Self::Negative => -0.5,
            Self::Neutral => 0.0,
            Self::Positive => 0.5,
            Self::VeryPositive => 1.0,
        }
    }
}

/// Emotional arousal level (calm ↔ energetic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArousalLevel {
    /// Extremely calm.
    VeryCalm,
    /// Calm.
    Calm,
    /// Moderate.
    Moderate,
    /// High energy.
    Energetic,
    /// Very high energy.
    VeryEnergetic,
}

impl ArousalLevel {
    /// BPM range `(min, max)` typically associated with this arousal level.
    #[must_use]
    pub fn bpm_range(&self) -> (f32, f32) {
        match self {
            Self::VeryCalm => (40.0, 70.0),
            Self::Calm => (70.0, 95.0),
            Self::Moderate => (95.0, 120.0),
            Self::Energetic => (120.0, 160.0),
            Self::VeryEnergetic => (160.0, 240.0),
        }
    }
}

/// A 2-D mood vector in the valence/arousal space.
#[derive(Debug, Clone)]
pub struct MoodVector {
    /// Valence level.
    pub valence: ValenceLevel,
    /// Arousal level.
    pub arousal: ArousalLevel,
    /// Confidence in [0.0, 1.0].
    pub confidence: f32,
}

impl MoodVector {
    /// Returns `true` if confidence exceeds `t`.
    #[must_use]
    pub fn is_confident(&self, t: f32) -> bool {
        self.confidence > t
    }

    /// Emotional quadrant label.
    #[must_use]
    pub fn quadrant(&self) -> &str {
        let pos_v = matches!(
            self.valence,
            ValenceLevel::Positive | ValenceLevel::VeryPositive
        );
        let high_a = matches!(
            self.arousal,
            ArousalLevel::Energetic | ArousalLevel::VeryEnergetic
        );
        match (pos_v, high_a) {
            (true, true) => "happy",
            (true, false) => "calm",
            (false, true) => "angry",
            (false, false) => "sad",
        }
    }
}

/// Heuristic mood classifier based on tempo, spectral centroid, and energy.
#[derive(Debug, Clone)]
pub struct MoodClassifier {
    /// Estimated tempo in BPM.
    pub tempo_bpm: f32,
    /// Spectral centroid in Hz.
    pub spectral_centroid: f32,
    /// RMS energy.
    pub energy: f32,
}

impl MoodClassifier {
    /// Classify mood.
    #[must_use]
    pub fn classify(&self) -> MoodVector {
        let arousal = if self.tempo_bpm < 70.0 {
            ArousalLevel::VeryCalm
        } else if self.tempo_bpm < 95.0 {
            ArousalLevel::Calm
        } else if self.tempo_bpm < 120.0 {
            ArousalLevel::Moderate
        } else if self.tempo_bpm < 160.0 {
            ArousalLevel::Energetic
        } else {
            ArousalLevel::VeryEnergetic
        };

        let valence = if self.spectral_centroid < 1000.0 {
            ValenceLevel::VeryNegative
        } else if self.spectral_centroid < 2000.0 {
            ValenceLevel::Negative
        } else if self.spectral_centroid < 3500.0 {
            ValenceLevel::Neutral
        } else if self.spectral_centroid < 5000.0 {
            ValenceLevel::Positive
        } else {
            ValenceLevel::VeryPositive
        };

        let confidence = (self.energy.abs().min(1.0) * 0.8 + 0.2).clamp(0.0, 1.0);
        MoodVector {
            valence,
            arousal,
            confidence,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MoodDetector::new ─────────────────────────────────────────────────────

    #[test]
    fn test_mood_detector_new() {
        let _det = MoodDetector::new();
    }

    // ── extract_features ──────────────────────────────────────────────────────

    #[test]
    fn test_extract_features_silence() {
        let det = MoodDetector::new();
        let features = det.extract_features(&[], 44100, 120.0, 1.0);
        assert!((features.energy - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_features_energy_range() {
        let det = MoodDetector::new();
        let samples = vec![0.5_f32; 44100];
        let features = det.extract_features(&samples, 44100, 120.0, 1.0);
        assert!(features.energy >= 0.0 && features.energy <= 1.0);
    }

    #[test]
    fn test_extract_features_mode_clamped() {
        let det = MoodDetector::new();
        let samples = vec![0.3_f32; 8192];
        // Pass mode outside [0,1] – should be clamped
        let features = det.extract_features(&samples, 44100, 120.0, 1.5);
        assert!(features.mode >= 0.0 && features.mode <= 1.0);
    }

    #[test]
    fn test_extract_features_valence_range() {
        let det = MoodDetector::new();
        let samples = vec![0.3_f32; 8192];
        let features = det.extract_features(&samples, 44100, 120.0, 0.8);
        assert!(
            features.valence >= -1.0 && features.valence <= 1.0,
            "valence={}",
            features.valence
        );
    }

    #[test]
    fn test_extract_features_arousal_range() {
        let det = MoodDetector::new();
        let samples = vec![0.3_f32; 8192];
        let features = det.extract_features(&samples, 44100, 120.0, 0.8);
        assert!(
            features.arousal >= -1.0 && features.arousal <= 1.0,
            "arousal={}",
            features.arousal
        );
    }

    // ── classify (circumplex rules) ───────────────────────────────────────────

    #[test]
    fn test_classify_high_valence_high_arousal_is_happy() {
        let det = MoodDetector::new();
        let features = MoodFeatures {
            valence: 0.7,
            arousal: 0.8,
            tempo_bpm: 140.0,
            mode: 0.9,
            spectral_brightness: 0.7,
            energy: 0.8,
            danceability: 0.7,
        };
        let result = det.classify(&features);
        assert_eq!(
            result.primary_mood,
            Mood::Happy,
            "got {:?}",
            result.primary_mood
        );
    }

    #[test]
    fn test_classify_low_valence_high_arousal_minor_is_aggressive() {
        let det = MoodDetector::new();
        let features = MoodFeatures {
            valence: -0.7,
            arousal: 0.8,
            tempo_bpm: 160.0,
            mode: 0.1,
            spectral_brightness: 0.6,
            energy: 0.9,
            danceability: 0.3,
        };
        let result = det.classify(&features);
        assert_eq!(
            result.primary_mood,
            Mood::Aggressive,
            "got {:?}",
            result.primary_mood
        );
    }

    #[test]
    fn test_classify_high_valence_low_arousal_is_calm_or_romantic() {
        let det = MoodDetector::new();
        let features = MoodFeatures {
            valence: 0.5,
            arousal: -0.5,
            tempo_bpm: 80.0,
            mode: 0.8,
            spectral_brightness: 0.4,
            energy: 0.2,
            danceability: 0.4,
        };
        let result = det.classify(&features);
        assert!(
            result.primary_mood == Mood::Calm || result.primary_mood == Mood::Romantic,
            "got {:?}",
            result.primary_mood
        );
    }

    #[test]
    fn test_classify_low_valence_very_low_arousal_is_sad() {
        let det = MoodDetector::new();
        let features = MoodFeatures {
            valence: -0.6,
            arousal: -0.7,
            tempo_bpm: 50.0,
            mode: 0.1,
            spectral_brightness: 0.2,
            energy: 0.1,
            danceability: 0.1,
        };
        let result = det.classify(&features);
        assert_eq!(
            result.primary_mood,
            Mood::Sad,
            "got {:?}",
            result.primary_mood
        );
    }

    #[test]
    fn test_classify_confidence_range() {
        let det = MoodDetector::new();
        let features = MoodFeatures {
            valence: 0.3,
            arousal: 0.6,
            tempo_bpm: 130.0,
            mode: 0.7,
            spectral_brightness: 0.5,
            energy: 0.6,
            danceability: 0.5,
        };
        let result = det.classify(&features);
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "confidence={}",
            result.confidence
        );
    }

    // ── convenience detect ─────────────────────────────────────────────────

    #[test]
    fn test_detect_returns_valid_mood() {
        let det = MoodDetector::new();
        let samples = vec![0.4_f32; 8192];
        let result = det.detect(&samples, 44100, 120.0, 0.8);
        // Just ensure it doesn't panic and returns sane values
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    // ── backward-compat ValenceLevel ──────────────────────────────────────────

    #[test]
    fn test_valence_very_negative_score() {
        assert!((ValenceLevel::VeryNegative.score() - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_valence_very_positive_score() {
        assert!((ValenceLevel::VeryPositive.score() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_valence_neutral_score() {
        assert!((ValenceLevel::Neutral.score() - 0.0).abs() < 1e-5);
    }

    // ── backward-compat ArousalLevel ──────────────────────────────────────────

    #[test]
    fn test_arousal_very_calm_bpm_range() {
        let (lo, hi) = ArousalLevel::VeryCalm.bpm_range();
        assert!(lo < hi);
        assert!((lo - 40.0).abs() < 1e-5);
    }

    #[test]
    fn test_arousal_very_energetic_bpm_range() {
        let (lo, hi) = ArousalLevel::VeryEnergetic.bpm_range();
        assert!(lo >= 160.0 && hi > lo);
    }

    // ── backward-compat MoodClassifier ────────────────────────────────────────

    #[test]
    fn test_legacy_classify_fast_bright_is_happy() {
        let clf = MoodClassifier {
            tempo_bpm: 140.0,
            spectral_centroid: 4500.0,
            energy: 0.8,
        };
        assert_eq!(clf.classify().quadrant(), "happy");
    }

    #[test]
    fn test_legacy_classify_slow_dark_is_sad() {
        let clf = MoodClassifier {
            tempo_bpm: 55.0,
            spectral_centroid: 800.0,
            energy: 0.3,
        };
        assert_eq!(clf.classify().quadrant(), "sad");
    }
}
