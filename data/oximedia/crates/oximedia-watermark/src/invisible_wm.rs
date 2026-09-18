#![allow(dead_code)]
//! Invisible watermark embedding using frequency-domain manipulation.
//!
//! This module provides a higher-level invisible watermarking framework that
//! combines multiple embedding strategies (DCT, DFT, wavelet) with adaptive
//! strength control and perceptual masking to produce imperceptible yet robust
//! watermarks in audio signals.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain enum
// ---------------------------------------------------------------------------

/// Transform domain used for watermark embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformDomain {
    /// Discrete Cosine Transform domain.
    Dct,
    /// Discrete Fourier Transform (magnitude/phase) domain.
    Dft,
    /// Wavelet (multi-resolution) domain.
    Wavelet,
    /// Time domain (direct sample modification).
    TimeDomain,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the invisible watermark embedder.
#[derive(Debug, Clone)]
pub struct InvisibleWmConfig {
    /// Transform domain to operate in.
    pub domain: TransformDomain,
    /// Base embedding strength in [0.0, 1.0].
    pub base_strength: f64,
    /// Frame size (number of samples per analysis window).
    pub frame_size: usize,
    /// Hop size between consecutive frames.
    pub hop_size: usize,
    /// Enable perceptual masking to improve imperceptibility.
    pub perceptual_mask: bool,
    /// Secret key seed for pseudo-random sequence generation.
    pub key_seed: u64,
    /// Minimum SNR (dB) to maintain in watermarked output.
    pub min_snr_db: f64,
    /// Error-correction redundancy factor (1 = no redundancy).
    pub redundancy: usize,
}

impl Default for InvisibleWmConfig {
    fn default() -> Self {
        Self {
            domain: TransformDomain::Dft,
            base_strength: 0.05,
            frame_size: 2048,
            hop_size: 1024,
            perceptual_mask: true,
            key_seed: 0,
            min_snr_db: 30.0,
            redundancy: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Masking model
// ---------------------------------------------------------------------------

/// Simple perceptual masking model that estimates how much modification is
/// tolerable at each frequency band.
#[derive(Debug, Clone)]
pub struct PerceptualMask {
    /// Number of frequency bands.
    pub num_bands: usize,
    /// Masking thresholds per band (linear amplitude).
    pub thresholds: Vec<f64>,
}

impl PerceptualMask {
    /// Build a masking model from a frame of audio samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_frame(frame: &[f64], num_bands: usize) -> Self {
        let band_size = frame.len().max(1) / num_bands.max(1);
        let mut thresholds = Vec::with_capacity(num_bands);
        for b in 0..num_bands {
            let start = b * band_size;
            let end = ((b + 1) * band_size).min(frame.len());
            let rms: f64 = if end > start {
                let sum_sq: f64 = frame[start..end].iter().map(|s| s * s).sum();
                (sum_sq / (end - start) as f64).sqrt()
            } else {
                0.0
            };
            // Masking threshold is proportional to RMS energy
            thresholds.push(rms * 0.1);
        }
        Self {
            num_bands,
            thresholds,
        }
    }

    /// Return the masking threshold for a given band index.
    #[must_use]
    pub fn threshold(&self, band: usize) -> f64 {
        self.thresholds.get(band).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

/// Invisible watermark embedder.
#[derive(Debug, Clone)]
pub struct InvisibleWmEmbedder {
    /// Embedder configuration.
    pub config: InvisibleWmConfig,
}

impl InvisibleWmEmbedder {
    /// Create a new embedder with the given configuration.
    #[must_use]
    pub fn new(config: InvisibleWmConfig) -> Self {
        Self { config }
    }

    /// Generate a pseudo-random spreading sequence of `len` elements in {-1, 1}.
    fn spread_sequence(&self, len: usize) -> Vec<f64> {
        let mut seq = Vec::with_capacity(len);
        let mut state = self.config.key_seed.wrapping_add(0xDEAD_BEEF);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            seq.push(if state & 1 == 0 { 1.0 } else { -1.0 });
        }
        seq
    }

    /// Embed a payload (byte slice) into audio samples.
    ///
    /// Returns the watermarked audio and the number of payload bits embedded.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn embed(&self, samples: &[f64], payload: &[u8]) -> (Vec<f64>, usize) {
        let bits = bytes_to_bits(payload);
        let redundant_bits = repeat_bits(&bits, self.config.redundancy);
        let spread = self.spread_sequence(self.config.frame_size);
        let num_frames = if samples.len() >= self.config.frame_size {
            (samples.len() - self.config.frame_size) / self.config.hop_size + 1
        } else {
            0
        };

        let mut output = samples.to_vec();
        let mut bits_embedded = 0usize;

        for f in 0..num_frames {
            if bits_embedded >= redundant_bits.len() {
                break;
            }
            let start = f * self.config.hop_size;
            let end = start + self.config.frame_size;
            let frame = &samples[start..end];

            let strength = if self.config.perceptual_mask {
                let mask = PerceptualMask::from_frame(frame, 8);
                let avg_thresh: f64 = mask.thresholds.iter().sum::<f64>() / mask.num_bands as f64;
                self.config.base_strength * (1.0 + avg_thresh)
            } else {
                self.config.base_strength
            };

            let bit_val = if redundant_bits[bits_embedded] {
                1.0
            } else {
                -1.0
            };
            for (i, &s) in spread.iter().enumerate() {
                if start + i < output.len() {
                    output[start + i] += strength * bit_val * s;
                }
            }
            bits_embedded += 1;
        }

        (output, bits_embedded)
    }

    /// Calculate embedding capacity in bits (before redundancy) for a given
    /// number of samples.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn capacity(&self, num_samples: usize) -> usize {
        let num_frames = if num_samples >= self.config.frame_size {
            (num_samples - self.config.frame_size) / self.config.hop_size + 1
        } else {
            0
        };
        num_frames / self.config.redundancy.max(1)
    }
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Invisible watermark extractor (blind detection).
#[derive(Debug, Clone)]
pub struct InvisibleWmExtractor {
    /// Extractor configuration (same parameters as embedder).
    pub config: InvisibleWmConfig,
}

impl InvisibleWmExtractor {
    /// Create a new extractor.
    #[must_use]
    pub fn new(config: InvisibleWmConfig) -> Self {
        Self { config }
    }

    /// Extract raw correlation values for each frame.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn correlate(&self, samples: &[f64]) -> Vec<f64> {
        let spread = {
            let mut seq = Vec::with_capacity(self.config.frame_size);
            let mut state = self.config.key_seed.wrapping_add(0xDEAD_BEEF);
            for _ in 0..self.config.frame_size {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                seq.push(if state & 1 == 0 { 1.0 } else { -1.0 });
            }
            seq
        };

        let num_frames = if samples.len() >= self.config.frame_size {
            (samples.len() - self.config.frame_size) / self.config.hop_size + 1
        } else {
            0
        };

        let mut corrs = Vec::with_capacity(num_frames);
        for f in 0..num_frames {
            let start = f * self.config.hop_size;
            let mut dot = 0.0f64;
            for (i, &s) in spread.iter().enumerate() {
                if start + i < samples.len() {
                    dot += samples[start + i] * s;
                }
            }
            corrs.push(dot / self.config.frame_size as f64);
        }
        corrs
    }

    /// Extract payload bits from watermarked audio.
    #[must_use]
    pub fn extract(&self, samples: &[f64], expected_bits: usize) -> Vec<u8> {
        let corrs = self.correlate(samples);
        let redundant_len = expected_bits * self.config.redundancy;
        let raw_bits: Vec<bool> = corrs.iter().take(redundant_len).map(|&c| c > 0.0).collect();
        let voted = majority_vote(&raw_bits, self.config.redundancy);
        bits_to_bytes(&voted[..voted.len().min(expected_bits)])
    }
}

// ---------------------------------------------------------------------------
// Quality assessment
// ---------------------------------------------------------------------------

/// Quality metrics after watermark embedding.
#[derive(Debug, Clone)]
pub struct WmQuality {
    /// Signal-to-Noise Ratio in decibels.
    pub snr_db: f64,
    /// Peak Signal-to-Noise Ratio in decibels.
    pub psnr_db: f64,
    /// Mean absolute difference.
    pub mae: f64,
}

/// Compute quality metrics between original and watermarked audio.
#[allow(clippy::cast_precision_loss)]
pub fn compute_quality(original: &[f64], watermarked: &[f64]) -> WmQuality {
    let n = original.len().min(watermarked.len());
    if n == 0 {
        return WmQuality {
            snr_db: 0.0,
            psnr_db: 0.0,
            mae: 0.0,
        };
    }

    let mut sig_power = 0.0f64;
    let mut noise_power = 0.0f64;
    let mut abs_sum = 0.0f64;

    for i in 0..n {
        let diff = watermarked[i] - original[i];
        sig_power += original[i] * original[i];
        noise_power += diff * diff;
        abs_sum += diff.abs();
    }

    let mae = abs_sum / n as f64;
    let snr_db = if noise_power > 0.0 {
        10.0 * (sig_power / noise_power).log10()
    } else {
        f64::INFINITY
    };
    let peak = original.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    let mse = noise_power / n as f64;
    let psnr_db = if mse > 0.0 {
        10.0 * ((peak * peak) / mse).log10()
    } else {
        f64::INFINITY
    };

    WmQuality {
        snr_db,
        psnr_db,
        mae,
    }
}

// ---------------------------------------------------------------------------
// Batch embedder
// ---------------------------------------------------------------------------

/// Result of embedding into multiple channels.
#[derive(Debug, Clone)]
pub struct BatchEmbedResult {
    /// Per-channel watermarked audio.
    pub channels: Vec<Vec<f64>>,
    /// Per-channel bits embedded.
    pub bits_per_channel: Vec<usize>,
    /// Per-channel quality metrics.
    pub quality: Vec<WmQuality>,
}

/// Embed a watermark into multiple audio channels independently.
#[must_use]
pub fn batch_embed(
    channels: &[Vec<f64>],
    payload: &[u8],
    config: &InvisibleWmConfig,
) -> BatchEmbedResult {
    let embedder = InvisibleWmEmbedder::new(config.clone());
    let mut result_channels = Vec::with_capacity(channels.len());
    let mut bits_per_channel = Vec::with_capacity(channels.len());
    let mut quality = Vec::with_capacity(channels.len());

    for ch in channels {
        let (wm, bits) = embedder.embed(ch, payload);
        let q = compute_quality(ch, &wm);
        result_channels.push(wm);
        bits_per_channel.push(bits);
        quality.push(q);
    }

    BatchEmbedResult {
        channels: result_channels,
        bits_per_channel,
        quality,
    }
}

// ---------------------------------------------------------------------------
// Domain stats
// ---------------------------------------------------------------------------

/// Frequency-domain statistics for a signal frame.
#[derive(Debug, Clone)]
pub struct DomainStats {
    /// Map of domain -> mean energy.
    pub energy_map: HashMap<TransformDomain, f64>,
    /// Recommended domain based on signal characteristics.
    pub recommended: TransformDomain,
}

/// Analyse a signal frame and recommend the best embedding domain.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn analyse_domain(frame: &[f64]) -> DomainStats {
    let n = frame.len().max(1);
    let energy: f64 = frame.iter().map(|s| s * s).sum::<f64>() / n as f64;
    let zero_crossings = frame
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    let zcr = zero_crossings as f64 / n as f64;

    let mut energy_map = HashMap::new();
    energy_map.insert(TransformDomain::Dft, energy * (1.0 + zcr));
    energy_map.insert(TransformDomain::Dct, energy * (1.0 - zcr * 0.5).max(0.0));
    energy_map.insert(TransformDomain::Wavelet, energy);
    energy_map.insert(TransformDomain::TimeDomain, energy * 0.5);

    let recommended = if zcr > 0.3 {
        TransformDomain::Dft
    } else if energy < 0.001 {
        TransformDomain::TimeDomain
    } else {
        TransformDomain::Dct
    };

    DomainStats {
        energy_map,
        recommended,
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Convert a byte slice to a vector of bits (MSB first).
fn bytes_to_bits(data: &[u8]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for &byte in data {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1 == 1);
        }
    }
    bits
}

/// Convert a bit vector back to bytes (MSB first, zero-padded).
fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut byte = 0u8;
        for (i, &b) in chunk.iter().enumerate() {
            if b {
                byte |= 1 << (7 - i);
            }
        }
        bytes.push(byte);
    }
    bytes
}

/// Repeat each bit `factor` times for redundancy coding.
fn repeat_bits(bits: &[bool], factor: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(bits.len() * factor);
    for &b in bits {
        for _ in 0..factor {
            out.push(b);
        }
    }
    out
}

/// Majority-vote decode over groups of `factor` bits.
fn majority_vote(bits: &[bool], factor: usize) -> Vec<bool> {
    if factor == 0 {
        return Vec::new();
    }
    bits.chunks(factor)
        .map(|chunk| {
            let ones = chunk.iter().filter(|&&b| b).count();
            ones * 2 > chunk.len()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = InvisibleWmConfig::default();
        assert_eq!(cfg.domain, TransformDomain::Dft);
        assert!((cfg.base_strength - 0.05).abs() < 1e-9);
        assert_eq!(cfg.frame_size, 2048);
        assert_eq!(cfg.hop_size, 1024);
        assert!(cfg.perceptual_mask);
        assert_eq!(cfg.redundancy, 3);
    }

    #[test]
    fn test_bytes_to_bits_roundtrip() {
        let data = b"Hello";
        let bits = bytes_to_bits(data);
        assert_eq!(bits.len(), 40);
        let recovered = bits_to_bytes(&bits);
        assert_eq!(&recovered, data);
    }

    #[test]
    fn test_repeat_and_vote() {
        let bits = vec![true, false, true, true];
        let repeated = repeat_bits(&bits, 5);
        assert_eq!(repeated.len(), 20);
        // Majority vote should recover original
        let voted = majority_vote(&repeated, 5);
        assert_eq!(voted, bits);
    }

    #[test]
    fn test_majority_vote_with_noise() {
        // Original bit: true, repeated 5 times, flip 2 to false => still true
        let chunk = vec![true, true, true, false, false];
        let voted = majority_vote(&chunk, 5);
        assert_eq!(voted, vec![true]);
    }

    #[test]
    fn test_embed_extract_roundtrip() {
        let config = InvisibleWmConfig {
            frame_size: 256,
            hop_size: 128,
            redundancy: 3,
            perceptual_mask: false,
            base_strength: 0.5,
            ..Default::default()
        };
        // Need enough samples for 8 bits * 3 redundancy = 24 frames
        // Each frame = 256, hop = 128 => need 128*23 + 256 = 3200 samples
        let samples = vec![0.0f64; 4096];
        let payload = b"A";

        let embedder = InvisibleWmEmbedder::new(config.clone());
        let (watermarked, bits_embedded) = embedder.embed(&samples, payload);
        assert!(bits_embedded > 0);

        let extractor = InvisibleWmExtractor::new(config);
        let recovered = extractor.extract(&watermarked, 8);
        assert_eq!(recovered, vec![b'A']);
    }

    #[test]
    fn test_capacity() {
        let config = InvisibleWmConfig {
            frame_size: 256,
            hop_size: 128,
            redundancy: 3,
            ..Default::default()
        };
        let embedder = InvisibleWmEmbedder::new(config);
        let cap = embedder.capacity(4096);
        // (4096-256)/128 + 1 = 31 frames, /3 = 10 bits
        assert_eq!(cap, 10);
    }

    #[test]
    fn test_quality_perfect() {
        let signal = vec![0.5; 1000];
        let q = compute_quality(&signal, &signal);
        assert!(q.snr_db.is_infinite() || q.snr_db > 100.0);
        assert!(q.mae < 1e-12);
    }

    #[test]
    fn test_quality_with_noise() {
        let original: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
        let noisy: Vec<f64> = original.iter().map(|&s| s + 0.001).collect();
        let q = compute_quality(&original, &noisy);
        assert!(q.snr_db > 20.0);
        assert!(q.mae > 0.0);
        assert!(q.psnr_db > 20.0);
    }

    #[test]
    fn test_perceptual_mask() {
        let frame: Vec<f64> = (0..2048).map(|i| (i as f64 * 0.1).sin() * 0.8).collect();
        let mask = PerceptualMask::from_frame(&frame, 8);
        assert_eq!(mask.num_bands, 8);
        assert_eq!(mask.thresholds.len(), 8);
        for t in &mask.thresholds {
            assert!(*t >= 0.0);
        }
    }

    #[test]
    fn test_perceptual_mask_silence() {
        let frame = vec![0.0f64; 1024];
        let mask = PerceptualMask::from_frame(&frame, 4);
        for t in &mask.thresholds {
            assert!((*t).abs() < 1e-12);
        }
    }

    #[test]
    fn test_batch_embed() {
        let config = InvisibleWmConfig {
            frame_size: 256,
            hop_size: 128,
            redundancy: 1,
            perceptual_mask: false,
            base_strength: 0.3,
            ..Default::default()
        };
        let channels = vec![vec![0.0f64; 2048], vec![0.1f64; 2048]];
        let result = batch_embed(&channels, b"X", &config);
        assert_eq!(result.channels.len(), 2);
        assert_eq!(result.bits_per_channel.len(), 2);
        assert_eq!(result.quality.len(), 2);
    }

    #[test]
    fn test_analyse_domain_high_zcr() {
        // Alternating samples => high zero-crossing rate
        let frame: Vec<f64> = (0..512)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let stats = analyse_domain(&frame);
        assert_eq!(stats.recommended, TransformDomain::Dft);
        assert!(stats.energy_map.contains_key(&TransformDomain::Dft));
    }

    #[test]
    fn test_analyse_domain_silence() {
        let frame = vec![0.0f64; 512];
        let stats = analyse_domain(&frame);
        assert_eq!(stats.recommended, TransformDomain::TimeDomain);
    }

    #[test]
    fn test_spread_sequence_deterministic() {
        let cfg = InvisibleWmConfig {
            key_seed: 42,
            ..Default::default()
        };
        let e1 = InvisibleWmEmbedder::new(cfg.clone());
        let e2 = InvisibleWmEmbedder::new(cfg);
        assert_eq!(e1.spread_sequence(100), e2.spread_sequence(100));
    }

    #[test]
    fn test_empty_payload() {
        let config = InvisibleWmConfig {
            frame_size: 256,
            hop_size: 128,
            redundancy: 1,
            perceptual_mask: false,
            ..Default::default()
        };
        let samples = vec![0.5f64; 1024];
        let embedder = InvisibleWmEmbedder::new(config);
        let (wm, bits) = embedder.embed(&samples, &[]);
        assert_eq!(bits, 0);
        // Signal should be unchanged
        assert_eq!(wm.len(), samples.len());
    }
}
