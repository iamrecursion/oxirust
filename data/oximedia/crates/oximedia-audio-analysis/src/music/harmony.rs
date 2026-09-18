//! Harmonic analysis for music.
//!
//! Provides automatic key detection, chord recognition, and harmonic complexity
//! analysis using chroma features and Krumhansl-Schmuckler key profiles.

use crate::{generate_window, AnalysisConfig, AnalysisError, Result, WindowType};
use oxifft::Complex;
use rayon::prelude::*;

/// Names of the 12 pitch classes, starting from C.
const PITCH_CLASS_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Krumhansl-Schmuckler major key profile (1982).
const MAJOR_PROFILE: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Krumhansl-Schmuckler natural minor key profile.
const MINOR_PROFILE: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Chord template: pitch class indices relative to root for common chord types.
struct ChordTemplate {
    name: &'static str,
    intervals: &'static [usize],
}

const CHORD_TEMPLATES: &[ChordTemplate] = &[
    ChordTemplate {
        name: "maj",
        intervals: &[0, 4, 7],
    },
    ChordTemplate {
        name: "min",
        intervals: &[0, 3, 7],
    },
    ChordTemplate {
        name: "dim",
        intervals: &[0, 3, 6],
    },
    ChordTemplate {
        name: "aug",
        intervals: &[0, 4, 8],
    },
    ChordTemplate {
        name: "7",
        intervals: &[0, 4, 7, 10],
    },
    ChordTemplate {
        name: "maj7",
        intervals: &[0, 4, 7, 11],
    },
    ChordTemplate {
        name: "min7",
        intervals: &[0, 3, 7, 10],
    },
    ChordTemplate {
        name: "sus4",
        intervals: &[0, 5, 7],
    },
    ChordTemplate {
        name: "sus2",
        intervals: &[0, 2, 7],
    },
];

/// Harmony analyzer for detecting key, chords, and harmonic progressions.
pub struct HarmonyAnalyzer {
    config: AnalysisConfig,
}

impl HarmonyAnalyzer {
    /// Create a new harmony analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze harmonic content of audio samples.
    ///
    /// Performs:
    /// 1. Chroma feature extraction via FFT and pitch-class folding
    /// 2. Key detection using Krumhansl-Schmuckler correlation
    /// 3. Frame-by-frame chord detection using template matching
    /// 4. Harmonic complexity estimation
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<HarmonyResult> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Extract chroma features from spectrogram
        let chroma_frames = self.extract_chroma_frames(samples, sample_rate)?;

        if chroma_frames.is_empty() {
            return Ok(HarmonyResult {
                key: "Unknown".to_string(),
                key_confidence: 0.0,
                key_pitch_class: 0,
                key_is_major: true,
                chords: vec![],
                chord_confidences: vec![],
                harmonic_complexity: 0.0,
                mean_chroma: [0.0; 12],
            });
        }

        // Compute mean chroma for key detection
        let mean_chroma = compute_mean_chroma(&chroma_frames);

        // Detect key using Krumhansl-Schmuckler algorithm
        let (key_pc, is_major, key_confidence) = detect_key(&mean_chroma);

        let key_name = format!(
            "{} {}",
            PITCH_CLASS_NAMES[key_pc],
            if is_major { "major" } else { "minor" }
        );

        // Detect chords per frame
        let (chords, chord_confidences) = self.detect_chords(&chroma_frames);

        // Compute harmonic complexity
        let harmonic_complexity = compute_harmonic_complexity(&chroma_frames, &chords);

        Ok(HarmonyResult {
            key: key_name,
            key_confidence,
            key_pitch_class: key_pc,
            key_is_major: is_major,
            chords,
            chord_confidences,
            harmonic_complexity,
            mean_chroma,
        })
    }

    /// Extract per-frame 12-bin chroma vectors from audio.
    ///
    /// Frame computation is parallelized with rayon — each frame's FFT and
    /// pitch-class folding are fully independent.
    fn extract_chroma_frames(&self, samples: &[f32], sample_rate: f32) -> Result<Vec<[f32; 12]>> {
        let fft_size = self.config.fft_size;
        let hop_size = self.config.hop_size;
        let window = generate_window(WindowType::Hann, fft_size);
        let num_bins = fft_size / 2 + 1;

        let num_frames = if samples.len() >= fft_size {
            (samples.len() - fft_size) / hop_size + 1
        } else {
            0
        };

        // Collect valid frame start positions.
        let frame_starts: Vec<usize> = (0..num_frames)
            .map(|fi| fi * hop_size)
            .take_while(|&start| start + fft_size <= samples.len())
            .collect();

        // Parallel: each frame's windowed FFT + chroma folding is independent.
        let chroma_frames: Vec<[f32; 12]> = frame_starts
            .par_iter()
            .map(|&start| {
                let end = start + fft_size;
                let complex_input: Vec<Complex<f64>> = samples[start..end]
                    .iter()
                    .zip(&window)
                    .map(|(&s, &w)| Complex::new(f64::from(s * w), 0.0))
                    .collect();

                let fft_output = oxifft::fft(&complex_input);

                let magnitude: Vec<f32> = fft_output[..num_bins]
                    .iter()
                    .map(|c| c.norm() as f32)
                    .collect();

                fold_to_chroma(&magnitude, sample_rate, fft_size)
            })
            .collect();

        Ok(chroma_frames)
    }

    /// Detect chords in each chroma frame using template matching.
    ///
    /// Template matching is embarrassingly parallel across frames.
    fn detect_chords(&self, chroma_frames: &[[f32; 12]]) -> (Vec<String>, Vec<f32>) {
        let pairs: Vec<(String, f32)> = chroma_frames
            .par_iter()
            .map(|chroma| match_chord(chroma))
            .collect();

        pairs.into_iter().unzip()
    }
}

/// Fold magnitude spectrum into 12 pitch-class bins.
fn fold_to_chroma(magnitude: &[f32], sample_rate: f32, fft_size: usize) -> [f32; 12] {
    let mut chroma = [0.0_f32; 12];
    let min_freq = 27.5_f32; // A0
    let max_freq = 4186.0_f32; // C8
    let a4_hz = 440.0_f32;

    for (bin, &mag) in magnitude.iter().enumerate() {
        if mag <= 0.0 {
            continue;
        }
        let freq = bin as f32 * sample_rate / fft_size as f32;
        if freq < min_freq || freq > max_freq {
            continue;
        }
        let semitones = 12.0 * (freq / a4_hz).log2();
        let rounded = semitones.round() as i32;
        let pc = ((rounded + 9).rem_euclid(12)) as usize;
        chroma[pc] += mag;
    }

    // L1 normalize
    let sum: f32 = chroma.iter().sum();
    if sum > f32::EPSILON {
        for v in &mut chroma {
            *v /= sum;
        }
    }

    chroma
}

/// Compute mean chroma over all frames.
fn compute_mean_chroma(frames: &[[f32; 12]]) -> [f32; 12] {
    let mut mean = [0.0_f32; 12];
    if frames.is_empty() {
        return mean;
    }

    for frame in frames {
        for (m, &v) in mean.iter_mut().zip(frame.iter()) {
            *m += v;
        }
    }

    let n = frames.len() as f32;
    for m in &mut mean {
        *m /= n;
    }

    mean
}

/// Detect the most likely key using Krumhansl-Schmuckler key-profile correlation.
///
/// Returns (pitch_class, is_major, confidence).
/// `confidence` is the Pearson correlation coefficient of the best match (0..1 range clipped).
pub fn detect_key(chroma: &[f32; 12]) -> (usize, bool, f32) {
    let chroma_mean: f32 = chroma.iter().sum::<f32>() / 12.0;
    let chroma_var: f32 = chroma
        .iter()
        .map(|&v| (v - chroma_mean).powi(2))
        .sum::<f32>()
        / 12.0;
    let chroma_std = chroma_var.sqrt();

    if chroma_std < f32::EPSILON {
        return (0, true, 0.0);
    }

    let profile_stats = |p: &[f32; 12]| -> (f32, f32) {
        let mean = p.iter().sum::<f32>() / 12.0;
        let var: f32 = p.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / 12.0;
        (mean, var.sqrt())
    };

    let (major_mean, major_std) = profile_stats(&MAJOR_PROFILE);
    let (minor_mean, minor_std) = profile_stats(&MINOR_PROFILE);

    let mut best_key = 0usize;
    let mut best_is_major = true;
    let mut best_corr = f32::NEG_INFINITY;

    for root in 0..12 {
        // Try major
        if major_std > f32::EPSILON {
            let rotated = rotate_profile(&MAJOR_PROFILE, root);
            let corr = pearson_correlation(
                chroma,
                &rotated,
                chroma_mean,
                chroma_std,
                major_mean,
                major_std,
            );
            if corr > best_corr {
                best_corr = corr;
                best_key = root;
                best_is_major = true;
            }
        }

        // Try minor
        if minor_std > f32::EPSILON {
            let rotated = rotate_profile(&MINOR_PROFILE, root);
            let corr = pearson_correlation(
                chroma,
                &rotated,
                chroma_mean,
                chroma_std,
                minor_mean,
                minor_std,
            );
            if corr > best_corr {
                best_corr = corr;
                best_key = root;
                best_is_major = false;
            }
        }
    }

    // Confidence: normalized from correlation range [-1, 1] to [0, 1]
    let confidence = ((best_corr + 1.0) / 2.0).clamp(0.0, 1.0);

    (best_key, best_is_major, confidence)
}

/// Rotate a 12-element profile by `shift` semitones.
fn rotate_profile(profile: &[f32; 12], shift: usize) -> [f32; 12] {
    let mut out = [0.0_f32; 12];
    for i in 0..12 {
        out[i] = profile[(i + 12 - shift) % 12];
    }
    out
}

/// Pearson correlation between chroma and a key profile.
fn pearson_correlation(
    chroma: &[f32; 12],
    profile: &[f32; 12],
    chroma_mean: f32,
    chroma_std: f32,
    profile_mean: f32,
    profile_std: f32,
) -> f32 {
    if chroma_std < f32::EPSILON || profile_std < f32::EPSILON {
        return 0.0;
    }
    let cov: f32 = chroma
        .iter()
        .zip(profile.iter())
        .map(|(&c, &p)| (c - chroma_mean) * (p - profile_mean))
        .sum::<f32>()
        / 12.0;
    (cov / (chroma_std * profile_std)).clamp(-1.0, 1.0)
}

/// Match a chroma vector against chord templates, returning the best chord name
/// and confidence.
fn match_chord(chroma: &[f32; 12]) -> (String, f32) {
    let mut best_name = String::from("N"); // "No chord"
    let mut best_score = 0.0_f32;

    for root in 0..12 {
        for template in CHORD_TEMPLATES {
            // Build a binary template vector for this root+chord type
            let mut tmpl = [0.0_f32; 12];
            for &interval in template.intervals {
                tmpl[(root + interval) % 12] = 1.0;
            }

            // Compute cosine similarity
            let dot: f32 = chroma.iter().zip(tmpl.iter()).map(|(&a, &b)| a * b).sum();
            let norm_a: f32 = chroma.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = tmpl.iter().map(|&v| v * v).sum::<f32>().sqrt();

            let score = if norm_a > f32::EPSILON && norm_b > f32::EPSILON {
                dot / (norm_a * norm_b)
            } else {
                0.0
            };

            if score > best_score {
                best_score = score;
                best_name = format!("{}{}", PITCH_CLASS_NAMES[root], template.name);
            }
        }
    }

    (best_name, best_score.clamp(0.0, 1.0))
}

/// Compute harmonic complexity from chord diversity and chroma distribution.
///
/// Returns a value in [0, 1] where 0 = very simple (one chord), 1 = very complex.
fn compute_harmonic_complexity(chroma_frames: &[[f32; 12]], chords: &[String]) -> f32 {
    if chords.is_empty() || chroma_frames.is_empty() {
        return 0.0;
    }

    // Factor 1: Number of unique chords / total chords (diversity)
    let mut unique_chords: Vec<&String> = chords.iter().collect();
    unique_chords.sort();
    unique_chords.dedup();
    let chord_diversity =
        (unique_chords.len() as f32 - 1.0).max(0.0) / (chords.len() as f32).max(1.0);

    // Factor 2: Chroma entropy (how evenly distributed the energy is)
    let mean_chroma = compute_mean_chroma(chroma_frames);
    let sum: f32 = mean_chroma.iter().sum();
    let entropy = if sum > f32::EPSILON {
        let mut h = 0.0_f32;
        for &v in &mean_chroma {
            let p = v / sum;
            if p > f32::EPSILON {
                h -= p * p.log2();
            }
        }
        // Normalize: max entropy = log2(12) ≈ 3.585
        h / 12.0_f32.log2()
    } else {
        0.0
    };

    // Factor 3: Chord change rate
    let mut changes = 0;
    for i in 1..chords.len() {
        if chords[i] != chords[i - 1] {
            changes += 1;
        }
    }
    let change_rate = changes as f32 / (chords.len().saturating_sub(1).max(1)) as f32;

    // Weighted combination
    let complexity = 0.3 * chord_diversity + 0.4 * entropy + 0.3 * change_rate;
    complexity.clamp(0.0, 1.0)
}

/// Detect the key of audio from raw samples (convenience function).
///
/// # Arguments
/// * `samples` - Mono audio samples
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Key name (e.g. "C major", "A minor") and confidence (0..1).
pub fn detect_key_from_audio(samples: &[f32], sample_rate: f32) -> Result<(String, f32)> {
    let config = AnalysisConfig::default();
    let analyzer = HarmonyAnalyzer::new(config);
    let result = analyzer.analyze(samples, sample_rate)?;
    Ok((result.key, result.key_confidence))
}

/// Harmony analysis result.
#[derive(Debug, Clone)]
pub struct HarmonyResult {
    /// Detected musical key (e.g. "C major", "A minor")
    pub key: String,
    /// Confidence in key detection (0.0 - 1.0)
    pub key_confidence: f32,
    /// Key pitch class index (0 = C, 1 = C#, ..., 11 = B)
    pub key_pitch_class: usize,
    /// Whether the detected key is major (true) or minor (false)
    pub key_is_major: bool,
    /// Detected chord sequence (one per analysis frame)
    pub chords: Vec<String>,
    /// Confidence for each detected chord (0.0 - 1.0)
    pub chord_confidences: Vec<f32>,
    /// Harmonic complexity measure (0.0 - 1.0)
    pub harmonic_complexity: f32,
    /// Mean 12-bin chroma vector for the entire signal
    pub mean_chroma: [f32; 12],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a sine wave at the given frequency.
    fn sine_wave(freq: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
        let n = (sample_rate * duration) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    /// Generate a chord (sum of sine waves at given frequencies).
    fn chord_signal(freqs: &[f32], sample_rate: f32, duration: f32) -> Vec<f32> {
        let n = (sample_rate * duration) as usize;
        let amplitude = 1.0 / freqs.len() as f32;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                freqs
                    .iter()
                    .map(|&f| amplitude * (2.0 * PI * f * t).sin())
                    .sum::<f32>()
            })
            .collect()
    }

    #[test]
    fn test_harmony_analyzer_basic() {
        let config = AnalysisConfig::default();
        let analyzer = HarmonyAnalyzer::new(config);
        let samples = sine_wave(440.0, 44100.0, 0.5);
        let result = analyzer.analyze(&samples, 44100.0);
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(!result.key.is_empty());
        assert!(result.key_confidence >= 0.0 && result.key_confidence <= 1.0);
    }

    #[test]
    fn test_harmony_analyzer_insufficient_samples() {
        let config = AnalysisConfig::default();
        let analyzer = HarmonyAnalyzer::new(config);
        let samples = vec![0.1; 100]; // too short
        let result = analyzer.analyze(&samples, 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_key_c_major_chord() {
        // C major chord: C4 (261.63), E4 (329.63), G4 (392.00)
        let samples = chord_signal(&[261.63, 329.63, 392.00], 44100.0, 1.0);
        let config = AnalysisConfig::default();
        let analyzer = HarmonyAnalyzer::new(config);
        let result = analyzer.analyze(&samples, 44100.0).expect("should succeed");
        // The key should be C major (or closely related)
        assert!(
            result.key_confidence > 0.3,
            "Key confidence should be reasonable: {}",
            result.key_confidence
        );
        // The key pitch class should be C (0) for a C major chord
        assert_eq!(
            result.key_pitch_class, 0,
            "Expected C major key from C major chord, got {}",
            result.key
        );
    }

    #[test]
    fn test_detect_key_a_minor_chord() {
        // A minor chord: A3 (220.0), C4 (261.63), E4 (329.63)
        let samples = chord_signal(&[220.0, 261.63, 329.63], 44100.0, 1.0);
        let config = AnalysisConfig::default();
        let analyzer = HarmonyAnalyzer::new(config);
        let result = analyzer.analyze(&samples, 44100.0).expect("should succeed");
        assert!(
            result.key_confidence > 0.3,
            "Key detection confidence: {}",
            result.key_confidence
        );
    }

    #[test]
    fn test_detect_key_function() {
        let mut chroma = [0.0_f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 0.7; // E
        chroma[7] = 0.8; // G
        let (pc, is_major, conf) = detect_key(&chroma);
        assert_eq!(pc, 0, "Expected C, got pitch class {pc}");
        assert!(is_major, "Expected major key");
        assert!(conf > 0.5, "Confidence should be high: {conf}");
    }

    #[test]
    fn test_detect_key_g_major() {
        // G major: G (7), B (11), D (2) prominent
        let mut chroma = [0.0_f32; 12];
        chroma[7] = 1.0; // G
        chroma[11] = 0.7; // B
        chroma[2] = 0.8; // D
        let (pc, is_major, _) = detect_key(&chroma);
        assert_eq!(pc, 7, "Expected G (7), got {pc}");
        assert!(is_major, "Expected major key");
    }

    #[test]
    fn test_detect_key_flat_chroma() {
        let chroma = [1.0_f32 / 12.0; 12];
        let (_, _, conf) = detect_key(&chroma);
        // Flat chroma should yield low confidence (near 0.5 after normalization)
        assert!(
            conf < 0.7,
            "Flat chroma should have low confidence, got {conf}"
        );
    }

    #[test]
    fn test_match_chord_c_major() {
        let mut chroma = [0.0_f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 0.8; // E
        chroma[7] = 0.9; // G
        let (name, conf) = match_chord(&chroma);
        assert!(
            name.starts_with('C'),
            "Expected chord starting with C, got {name}"
        );
        assert!(conf > 0.5, "Chord confidence should be high: {conf}");
    }

    #[test]
    fn test_match_chord_a_minor() {
        let mut chroma = [0.0_f32; 12];
        chroma[9] = 1.0; // A
        chroma[0] = 0.8; // C
        chroma[4] = 0.9; // E
        let (name, conf) = match_chord(&chroma);
        assert!(
            name.contains('A') || name.contains('C'),
            "Expected A minor or related, got {name}"
        );
        assert!(conf > 0.3, "Chord confidence: {conf}");
    }

    #[test]
    fn test_harmonic_complexity_single_chord() {
        // Same chord repeated = low complexity
        let chroma = [[0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]; 10];
        let chords: Vec<String> = vec!["Cmaj".to_string(); 10];
        let complexity = compute_harmonic_complexity(&chroma, &chords);
        assert!(
            complexity < 0.5,
            "Single repeated chord should have low complexity: {complexity}"
        );
    }

    #[test]
    fn test_harmonic_complexity_many_chords() {
        let chroma_frames: Vec<[f32; 12]> = (0..12)
            .map(|i| {
                let mut c = [0.0_f32; 12];
                c[i] = 1.0;
                c[(i + 4) % 12] = 0.7;
                c[(i + 7) % 12] = 0.8;
                c
            })
            .collect();
        let chords: Vec<String> = (0..12)
            .map(|i| format!("{}maj", PITCH_CLASS_NAMES[i]))
            .collect();
        let complexity = compute_harmonic_complexity(&chroma_frames, &chords);
        assert!(
            complexity > 0.3,
            "Many different chords should have higher complexity: {complexity}"
        );
    }

    #[test]
    fn test_detect_key_from_audio() {
        let samples = chord_signal(&[261.63, 329.63, 392.00], 44100.0, 0.5);
        let result = detect_key_from_audio(&samples, 44100.0);
        assert!(result.is_ok());
        let (key, conf) = result.expect("should succeed");
        assert!(!key.is_empty());
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_chord_detection_has_results() {
        let samples = chord_signal(&[261.63, 329.63, 392.00], 44100.0, 1.0);
        let config = AnalysisConfig::default();
        let analyzer = HarmonyAnalyzer::new(config);
        let result = analyzer.analyze(&samples, 44100.0).expect("should succeed");
        assert!(
            !result.chords.is_empty(),
            "Should detect at least one chord"
        );
        assert_eq!(result.chords.len(), result.chord_confidences.len());
    }

    #[test]
    fn test_rotate_profile_identity() {
        let rotated = rotate_profile(&MAJOR_PROFILE, 0);
        for i in 0..12 {
            assert!((rotated[i] - MAJOR_PROFILE[i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_rotate_profile_by_one() {
        let rotated = rotate_profile(&MAJOR_PROFILE, 1);
        // rotated[0] should be MAJOR_PROFILE[11]
        assert!((rotated[0] - MAJOR_PROFILE[11]).abs() < f32::EPSILON);
        assert!((rotated[1] - MAJOR_PROFILE[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mean_chroma_computation() {
        let frames = vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mean = compute_mean_chroma(&frames);
        assert!((mean[0] - 0.5).abs() < 1e-5);
        assert!((mean[4] - 0.5).abs() < 1e-5);
    }
}
