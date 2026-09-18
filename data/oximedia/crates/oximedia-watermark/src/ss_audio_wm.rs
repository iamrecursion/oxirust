//! Spread-spectrum audio watermarking with PN-sequence correlation detection.
//!
//! Embeds a binary message into audio samples by adding a scaled
//! pseudo-noise (PN) sequence whose polarity encodes each bit.  Detection
//! is performed by correlating the received signal against the known PN
//! sequence.

/// A 63-chip maximal-length sequence (m-sequence, polynomial x^6+x+1).
///
/// Values are ±1.0.  This length (2^6 - 1 = 63) gives good auto-correlation
/// properties suitable for spread-spectrum embedding.
pub const PN_SEQUENCE: [f32; 63] = [
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
];

/// Configuration for the spread-spectrum audio watermarker.
#[derive(Debug, Clone)]
pub struct AudioWatermarkConfig {
    /// Sample rate of the audio signal.
    pub target_sample_rate: u32,
    /// Watermark amplitude as a fraction of full scale (e.g. 0.01 = 1 %).
    pub amplitude_fraction: f32,
    /// Centre frequency used for sinusoidal spreading (Hz).
    ///
    /// Not used in the current implementation (reserved for future
    /// band-limited embedding); kept for API compatibility.
    pub frequency_hz: f32,
}

impl Default for AudioWatermarkConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 44100,
            amplitude_fraction: 0.05,
            frequency_hz: 1000.0,
        }
    }
}

impl AudioWatermarkConfig {
    /// Create a configuration designed to be inaudible.
    ///
    /// Uses a very low amplitude fraction (0.003) appropriate for content
    /// that will be distributed at high quality.
    #[must_use]
    pub fn inaudible() -> Self {
        Self {
            target_sample_rate: 44100,
            amplitude_fraction: 0.003,
            frequency_hz: 18_000.0,
        }
    }
}

/// Embed a binary message into audio samples using spread-spectrum.
///
/// Each message bit is spread over one repetition of `PN_SEQUENCE`:
/// - bit `true`  → add  `amplitude_fraction * PN_SEQUENCE[n]` to each chip sample.
/// - bit `false` → add `-amplitude_fraction * PN_SEQUENCE[n]`.
///
/// If the samples slice is shorter than required for all bits, embedding
/// stops silently at the end of the buffer.
///
/// # Arguments
/// * `samples`      – mutable audio sample buffer (f32, normalised ±1).
/// * `message_bits` – slice of booleans to embed.
/// * `config`       – watermark configuration.
pub fn embed_spread_spectrum(
    samples: &mut [f32],
    message_bits: &[bool],
    config: &AudioWatermarkConfig,
) {
    let chip_len = PN_SEQUENCE.len();
    let amp = config.amplitude_fraction;
    for (bit_idx, &bit) in message_bits.iter().enumerate() {
        let polarity = if bit { 1.0_f32 } else { -1.0_f32 };
        let start = bit_idx * chip_len;
        if start >= samples.len() {
            break;
        }
        let end = (start + chip_len).min(samples.len());
        for (s, &pn) in samples[start..end].iter_mut().zip(PN_SEQUENCE.iter()) {
            *s += polarity * amp * pn;
        }
    }
}

/// Detect a spread-spectrum watermark by correlating against `PN_SEQUENCE`.
///
/// For each expected bit position the dot product of the received samples
/// with `PN_SEQUENCE` is computed.  A positive correlation → `true`;
/// negative → `false`.  Confidence is the magnitude of the average
/// per-bit correlation normalised to `[0, 1]` (capped at 1).
///
/// # Arguments
/// * `samples`      – received audio sample buffer.
/// * `config`       – watermark configuration (`amplitude_fraction` used for
///                    normalisation).
///
/// # Returns
/// A tuple `(bits, confidence)`:
/// - `bits`       – detected boolean message bits (length = number of complete
///                  PN periods that fit in `samples`).
/// - `confidence` – average correlation strength in `[0, 1]`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_spread_spectrum(samples: &[f32], config: &AudioWatermarkConfig) -> (Vec<bool>, f32) {
    let chip_len = PN_SEQUENCE.len();
    if samples.is_empty() || chip_len == 0 {
        return (Vec::new(), 0.0);
    }

    let n_bits = samples.len() / chip_len;
    if n_bits == 0 {
        return (Vec::new(), 0.0);
    }

    let mut bits = Vec::with_capacity(n_bits);
    let mut corr_sum = 0.0_f32;

    for bit_idx in 0..n_bits {
        let start = bit_idx * chip_len;
        let chunk = &samples[start..start + chip_len];
        let corr: f32 = chunk
            .iter()
            .zip(PN_SEQUENCE.iter())
            .map(|(s, p)| s * p)
            .sum::<f32>()
            / chip_len as f32;
        bits.push(corr >= 0.0);
        corr_sum += corr.abs();
    }

    let avg_corr = corr_sum / n_bits as f32;
    // Normalise: expected correlation = amplitude_fraction
    let amp = config.amplitude_fraction.max(1e-9);
    let confidence = (avg_corr / amp).clamp(0.0, 1.0);

    (bits, confidence)
}

/// Signal-to-watermark noise ratio utilities.
pub struct AudioWatermarkStrength;

impl AudioWatermarkStrength {
    /// Compute the signal-to-watermark-noise ratio in dB.
    ///
    /// `original`   – the un-watermarked reference signal.
    /// `watermarked` – the watermarked signal.
    ///
    /// Returns `f32::INFINITY` if the watermark is zero (no noise added).
    /// Returns `-f32::INFINITY` if the signal is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn snr_db(original: &[f32], watermarked: &[f32]) -> f32 {
        let len = original.len().min(watermarked.len());
        if len == 0 {
            return 0.0;
        }
        let signal_power: f32 = original[..len].iter().map(|&s| s * s).sum::<f32>() / len as f32;
        let noise_power: f32 = original[..len]
            .iter()
            .zip(watermarked[..len].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / len as f32;

        if noise_power < 1e-20 {
            return f32::INFINITY;
        }
        if signal_power < 1e-20 {
            return f32::NEG_INFINITY;
        }
        10.0 * (signal_power / noise_power).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PN_SEQUENCE ───────────────────────────────────────────────────────────

    #[test]
    fn test_pn_sequence_length() {
        assert_eq!(PN_SEQUENCE.len(), 63);
    }

    #[test]
    fn test_pn_sequence_values_pm1() {
        for &v in PN_SEQUENCE.iter() {
            assert!(
                (v - 1.0).abs() < 1e-5 || (v + 1.0).abs() < 1e-5,
                "PN value not ±1: {v}"
            );
        }
    }

    #[test]
    fn test_pn_sequence_not_all_same() {
        let all_one = PN_SEQUENCE.iter().all(|&v| v > 0.0);
        let all_neg = PN_SEQUENCE.iter().all(|&v| v < 0.0);
        assert!(!all_one && !all_neg);
    }

    // ── AudioWatermarkConfig ──────────────────────────────────────────────────

    #[test]
    fn test_default_config_sample_rate() {
        let cfg = AudioWatermarkConfig::default();
        assert_eq!(cfg.target_sample_rate, 44100);
    }

    #[test]
    fn test_default_config_amplitude() {
        let cfg = AudioWatermarkConfig::default();
        assert!(cfg.amplitude_fraction > 0.0);
        assert!(cfg.amplitude_fraction < 1.0);
    }

    #[test]
    fn test_inaudible_config_low_amplitude() {
        let cfg = AudioWatermarkConfig::inaudible();
        assert!(cfg.amplitude_fraction < 0.01);
    }

    // ── embed_spread_spectrum ─────────────────────────────────────────────────

    #[test]
    fn test_embed_changes_samples() {
        let config = AudioWatermarkConfig::default();
        let original = vec![0.5_f32; PN_SEQUENCE.len() * 4];
        let mut watermarked = original.clone();
        let bits = vec![true, false, true, false];
        embed_spread_spectrum(&mut watermarked, &bits, &config);
        assert_ne!(watermarked, original);
    }

    #[test]
    fn test_embed_does_not_panic_short_buffer() {
        let config = AudioWatermarkConfig::default();
        let mut samples = vec![0.0_f32; 10]; // shorter than one PN period
        embed_spread_spectrum(&mut samples, &[true, false], &config);
        // No panic — just partial embedding
    }

    #[test]
    fn test_embed_empty_bits_no_change() {
        let config = AudioWatermarkConfig::default();
        let original = vec![0.3_f32; 200];
        let mut samples = original.clone();
        embed_spread_spectrum(&mut samples, &[], &config);
        assert_eq!(samples, original);
    }

    // ── detect_spread_spectrum ────────────────────────────────────────────────

    #[test]
    fn test_detect_roundtrip_true_bits() {
        let config = AudioWatermarkConfig {
            amplitude_fraction: 0.5,
            ..Default::default()
        };
        let bits_in = vec![true; 4];
        let mut samples = vec![0.0_f32; PN_SEQUENCE.len() * bits_in.len()];
        embed_spread_spectrum(&mut samples, &bits_in, &config);
        let (bits_out, _) = detect_spread_spectrum(&samples, &config);
        assert_eq!(bits_out, bits_in);
    }

    #[test]
    fn test_detect_roundtrip_false_bits() {
        let config = AudioWatermarkConfig {
            amplitude_fraction: 0.5,
            ..Default::default()
        };
        let bits_in = vec![false; 3];
        let mut samples = vec![0.0_f32; PN_SEQUENCE.len() * bits_in.len()];
        embed_spread_spectrum(&mut samples, &bits_in, &config);
        let (bits_out, _) = detect_spread_spectrum(&samples, &config);
        assert_eq!(bits_out, bits_in);
    }

    #[test]
    fn test_detect_empty_samples_returns_empty() {
        let config = AudioWatermarkConfig::default();
        let (bits, confidence) = detect_spread_spectrum(&[], &config);
        assert!(bits.is_empty());
        assert!((confidence - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_detect_confidence_in_range() {
        let config = AudioWatermarkConfig {
            amplitude_fraction: 0.5,
            ..Default::default()
        };
        let mut samples = vec![0.0_f32; PN_SEQUENCE.len() * 2];
        embed_spread_spectrum(&mut samples, &[true, false], &config);
        let (_, confidence) = detect_spread_spectrum(&samples, &config);
        assert!((0.0..=1.0).contains(&confidence));
    }

    // ── AudioWatermarkStrength::snr_db ────────────────────────────────────────

    #[test]
    fn test_snr_identical_signals_infinite() {
        let sig = vec![0.5_f32; 100];
        let snr = AudioWatermarkStrength::snr_db(&sig, &sig);
        assert!(snr.is_infinite() && snr > 0.0);
    }

    #[test]
    fn test_snr_known_value() {
        // signal: constant 1.0; watermark adds 0.1 → noise = 0.1
        // SNR = 10 * log10(1 / 0.01) = 20 dB
        let original = vec![1.0_f32; 100];
        let watermarked = vec![1.1_f32; 100];
        let snr = AudioWatermarkStrength::snr_db(&original, &watermarked);
        assert!((snr - 20.0).abs() < 1.0, "expected ~20 dB, got {snr}");
    }

    #[test]
    fn test_snr_empty_returns_zero() {
        let snr = AudioWatermarkStrength::snr_db(&[], &[]);
        assert!((snr - 0.0).abs() < 1e-5);
    }
}
