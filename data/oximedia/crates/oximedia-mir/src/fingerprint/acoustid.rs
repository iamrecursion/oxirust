//! Acoustic fingerprinting inspired by Chromaprint/AcoustID.
//!
//! Provides a simplified Chromaprint-inspired fingerprint pipeline:
//! chroma feature extraction via short-time DFT, followed by FNV-based
//! hash compression into a compact fingerprint vector.

#![allow(dead_code)]

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// ChromaFeature
// ---------------------------------------------------------------------------

/// A 12-bin chroma vector representing the relative energy of each pitch
/// class (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
#[derive(Debug, Clone, Copy)]
pub struct ChromaFeature(pub [f32; 12]);

impl ChromaFeature {
    /// Create a zero-initialised chroma feature.
    #[must_use]
    pub fn zero() -> Self {
        Self([0.0; 12])
    }

    /// Normalise the chroma vector to unit L2 norm (in-place).
    pub fn normalise(&mut self) {
        let norm: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut self.0 {
                *v /= norm;
            }
        }
    }

    /// Cosine similarity with another `ChromaFeature`.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let na: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-9 || nb < 1e-9 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

// ---------------------------------------------------------------------------
// ChromaExtractor
// ---------------------------------------------------------------------------

/// Extracts per-frame chroma features from audio samples.
pub struct ChromaExtractor;

impl ChromaExtractor {
    /// Extract chroma features using a simplified short-time DFT.
    ///
    /// # Arguments
    ///
    /// * `samples` — Mono audio samples (f32, normalised).
    /// * `sample_rate` — Sample rate in Hz.
    /// * `frame_size` — DFT window length in samples.
    #[must_use]
    pub fn extract(samples: &[f32], sample_rate: u32, frame_size: usize) -> Vec<ChromaFeature> {
        if samples.is_empty() || frame_size == 0 {
            return Vec::new();
        }

        let hop = frame_size / 2;
        let sr = sample_rate as f32;
        let mut features = Vec::new();

        let mut start = 0usize;
        while start + frame_size <= samples.len() {
            let frame = &samples[start..start + frame_size];
            let mut chroma = ChromaFeature::zero();

            // Apply a simple Hann window and compute power per frequency bin
            // We map each frequency bin to a pitch class using the chroma mapping
            let n = frame_size as f32;
            for k in 1..frame_size / 2 {
                // DFT magnitude via Goertzel-like direct formula (simplified)
                let freq_hz = k as f32 * sr / n;
                if !(27.5..=4186.0).contains(&freq_hz) {
                    // Outside piano range — skip
                    continue;
                }

                // Power approximation: sum windowed cosine correlation
                let mut re = 0.0_f32;
                let mut im = 0.0_f32;
                for (i, &sample) in frame.iter().enumerate() {
                    let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / (frame_size as f32 - 1.0)).cos();
                    let angle = 2.0 * PI * k as f32 * i as f32 / n;
                    re += sample * w * angle.cos();
                    im -= sample * w * angle.sin();
                }
                let power = re * re + im * im;

                // Map frequency to chroma bin
                // MIDI note for a frequency: 69 + 12 * log2(freq / 440)
                let midi = 69.0 + 12.0 * (freq_hz / 440.0).log2();
                let pitch_class = (midi.round() as i32).rem_euclid(12) as usize;
                chroma.0[pitch_class] += power;
            }

            chroma.normalise();
            features.push(chroma);
            start += hop;
        }

        features
    }
}

// ---------------------------------------------------------------------------
// AcoustidFingerprint
// ---------------------------------------------------------------------------

/// A compact acoustic fingerprint.
#[derive(Debug, Clone)]
pub struct AcoustidFingerprint {
    /// Sequence of 32-bit hash values derived from chroma features.
    pub fingerprint: Vec<u32>,
    /// Duration of the audio used to generate the fingerprint.
    pub duration_secs: f32,
}

impl AcoustidFingerprint {
    /// Number of hash values in the fingerprint.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprint.len()
    }

    /// Whether the fingerprint is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprint.is_empty()
    }
}

// ---------------------------------------------------------------------------
// AcoustidEncoder
// ---------------------------------------------------------------------------

/// Encodes audio samples into an [`AcoustidFingerprint`].
pub struct AcoustidEncoder;

impl AcoustidEncoder {
    /// Compute a fingerprint from mono audio samples.
    ///
    /// Internally extracts chroma features per frame and hashes them using
    /// a variant of FNV-1a over quantised chroma values.
    ///
    /// # Arguments
    ///
    /// * `samples` — Mono audio samples.
    /// * `sample_rate` — Sample rate in Hz.
    #[must_use]
    pub fn compute(samples: &[f32], sample_rate: u32) -> AcoustidFingerprint {
        Self::compute_with_frame_size(samples, sample_rate, 4096)
    }

    /// Compute a fingerprint with a custom frame size (useful for testing).
    ///
    /// # Arguments
    ///
    /// * `samples` — Mono audio samples.
    /// * `sample_rate` — Sample rate in Hz.
    /// * `frame_size` — DFT frame size in samples.
    #[must_use]
    pub fn compute_with_frame_size(
        samples: &[f32],
        sample_rate: u32,
        frame_size: usize,
    ) -> AcoustidFingerprint {
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        let chroma_frames = ChromaExtractor::extract(samples, sample_rate, frame_size);

        let fingerprint: Vec<u32> = chroma_frames.iter().map(Self::hash_chroma).collect();

        AcoustidFingerprint {
            fingerprint,
            duration_secs,
        }
    }

    /// Hash a single chroma feature using FNV-1a over quantised values.
    fn hash_chroma(cf: &ChromaFeature) -> u32 {
        const FNV_OFFSET: u32 = 2_166_136_261;
        const FNV_PRIME: u32 = 16_777_619;

        let mut hash = FNV_OFFSET;
        for &v in &cf.0 {
            // Quantise to 4 bits (0-15)
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let quantised = (v * 16.0).clamp(0.0, 15.0) as u8;
            hash ^= u32::from(quantised);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// FingerprintMatcher
// ---------------------------------------------------------------------------

/// Compares two fingerprints for similarity.
pub struct FingerprintMatcher;

impl FingerprintMatcher {
    /// Compute the similarity between two fingerprints as the fraction of
    /// matching 32-bit hash values at aligned positions.
    ///
    /// Returns a value in [0, 1] where 1.0 means all hashes agree.
    #[must_use]
    pub fn similarity(a: &AcoustidFingerprint, b: &AcoustidFingerprint) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let len = a.fingerprint.len().min(b.fingerprint.len());
        let matching = a.fingerprint[..len]
            .iter()
            .zip(b.fingerprint[..len].iter())
            .filter(|(x, y)| x == y)
            .count();

        matching as f32 / len as f32
    }

    /// Similarity based on bit-agreement ratio (fraction of identical bits).
    #[must_use]
    pub fn bit_similarity(a: &AcoustidFingerprint, b: &AcoustidFingerprint) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let len = a.fingerprint.len().min(b.fingerprint.len());
        let total_bits = len * 32;
        let differing_bits: u32 = a.fingerprint[..len]
            .iter()
            .zip(b.fingerprint[..len].iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();

        1.0 - differing_bits as f32 / total_bits as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_chroma_feature_zero() {
        let cf = ChromaFeature::zero();
        assert!(cf.0.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_chroma_feature_normalise() {
        let mut cf = ChromaFeature([1.0; 12]);
        cf.normalise();
        let norm: f32 = cf.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_chroma_cosine_similarity_identical() {
        let cf = ChromaFeature([0.5; 12]);
        assert!((cf.cosine_similarity(&cf) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_chroma_extractor_empty() {
        let result = ChromaExtractor::extract(&[], 44100, 2048);
        assert!(result.is_empty());
    }

    #[test]
    fn test_chroma_extractor_produces_frames() {
        let samples = sine_wave(440.0, 8000, 1.0);
        let frames = ChromaExtractor::extract(&samples, 8000, 512);
        assert!(!frames.is_empty());
    }

    /// Helper: compute a fingerprint quickly using a small frame_size.
    fn fast_fingerprint(freq: f32, duration_secs: f32) -> AcoustidFingerprint {
        // frame_size=256 keeps the O(N²) DFT per-frame at 128*256 = 32k ops,
        // orders of magnitude cheaper than the default 4096.
        let sr = 8000_u32;
        let samples = sine_wave(freq, sr, duration_secs);
        AcoustidEncoder::compute_with_frame_size(&samples, sr, 256)
    }

    #[test]
    fn test_acoustid_encoder_basic() {
        let fp = fast_fingerprint(440.0, 0.25);
        assert!(!fp.is_empty());
        assert!((fp.duration_secs - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_acoustid_encoder_empty() {
        let fp = AcoustidEncoder::compute(&[], 44100);
        assert!(fp.is_empty());
        assert_eq!(fp.duration_secs, 0.0);
    }

    #[test]
    fn test_fingerprint_self_similarity() {
        let fp = fast_fingerprint(440.0, 0.25);
        let sim = FingerprintMatcher::similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fingerprint_different_audio() {
        let fp_a = fast_fingerprint(440.0, 0.25);
        let fp_b = fast_fingerprint(523.25, 0.25); // C5 — different chroma class
        let sim = FingerprintMatcher::similarity(&fp_a, &fp_b);
        // Different frequencies should produce different fingerprints
        assert!(sim < 1.0);
    }

    #[test]
    fn test_bit_similarity_self() {
        let fp = fast_fingerprint(440.0, 0.25);
        let sim = FingerprintMatcher::bit_similarity(&fp, &fp);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fingerprint_empty_similarity() {
        let empty = AcoustidFingerprint {
            fingerprint: vec![],
            duration_secs: 0.0,
        };
        let fp = fast_fingerprint(440.0, 0.25);
        assert_eq!(FingerprintMatcher::similarity(&empty, &fp), 0.0);
    }
}
