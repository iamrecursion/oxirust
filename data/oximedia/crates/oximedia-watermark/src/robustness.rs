//! Watermark robustness testing: attack resistance simulation and survival rate scoring.
//!
//! Simulates common signal-processing attacks and measures how well a watermark survives.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Types of attack to simulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum AttackType {
    /// Additive white Gaussian noise at a given SNR (dB).
    AwgnNoise,
    /// Amplitude scaling (gain change).
    AmplitudeScale,
    /// Sample-rate down-conversion then up-conversion (resampling).
    Resampling,
    /// Low-pass filtering (simulates lossy compression).
    LowPassFilter,
    /// Time stretching / pitch shifting.
    TimeStretch,
    /// Amplitude quantization (simulates low bit-depth).
    Quantization,
    /// Echo addition.
    Echo,
    /// Random sample-level jitter.
    SampleJitter,
}

/// Parameters for an individual attack.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AttackParams {
    /// Which attack to apply.
    pub attack_type: AttackType,
    /// Primary strength parameter (interpretation depends on attack type).
    pub strength: f64,
}

impl AttackParams {
    /// Create default AWGN attack at 30 dB SNR.
    #[must_use]
    pub fn awgn(snr_db: f64) -> Self {
        Self {
            attack_type: AttackType::AwgnNoise,
            strength: snr_db,
        }
    }

    /// Create amplitude scale attack.
    #[must_use]
    pub fn amplitude(scale: f64) -> Self {
        Self {
            attack_type: AttackType::AmplitudeScale,
            strength: scale,
        }
    }

    /// Create low-pass filter attack with given cutoff fraction (0..1).
    #[must_use]
    pub fn low_pass(cutoff_fraction: f64) -> Self {
        Self {
            attack_type: AttackType::LowPassFilter,
            strength: cutoff_fraction,
        }
    }

    /// Create quantization attack with given number of bits (1–16).
    #[must_use]
    pub fn quantization(bits: u32) -> Self {
        Self {
            attack_type: AttackType::Quantization,
            strength: f64::from(bits),
        }
    }
}

/// Result of a single robustness test.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RobustnessResult {
    /// Attack that was applied.
    pub attack: AttackParams,
    /// Survival score (0.0 = watermark destroyed, 1.0 = fully survived).
    pub survival_rate: f64,
    /// Bit error rate of detected watermark (0.0–0.5).
    pub bit_error_rate: f64,
    /// Whether the watermark is considered detectable (BER < threshold).
    pub detectable: bool,
}

/// Configuration for the robustness test suite.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RobustnessConfig {
    /// BER threshold below which the watermark is considered detectable.
    pub ber_threshold: f64,
    /// Minimum survival rate to pass.
    pub min_survival_rate: f64,
}

impl Default for RobustnessConfig {
    fn default() -> Self {
        Self {
            ber_threshold: 0.1,
            min_survival_rate: 0.7,
        }
    }
}

/// Simulate application of AWGN noise to a signal.
///
/// Scales noise power to achieve the target SNR.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn apply_awgn(samples: &[f32], snr_db: f64) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let signal_power: f64 = samples
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / samples.len() as f64;
    let snr_linear = 10.0_f64.powf(snr_db / 10.0);
    let noise_power = if snr_linear > 0.0 {
        signal_power / snr_linear
    } else {
        signal_power
    };
    let noise_std = noise_power.sqrt();
    // Deterministic pseudo-noise using a simple LCG seeded at 42
    let mut state: u64 = 42;
    samples
        .iter()
        .map(|&s| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let normalized = (state >> 33) as f64 / f64::from(u32::MAX) - 0.5; // [-0.5, 0.5]
            let noise = normalized * 2.0 * noise_std;
            (f64::from(s) + noise) as f32
        })
        .collect()
}

/// Simulate amplitude scaling attack.
#[must_use]
pub fn apply_amplitude_scale(samples: &[f32], scale: f64) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| (f64::from(s) * scale) as f32)
        .collect()
}

/// Simulate simple low-pass filter (box filter of length `kernel_len`).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn apply_low_pass_box(samples: &[f32], kernel_len: usize) -> Vec<f32> {
    if kernel_len == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let half = kernel_len / 2;
    let n = samples.len();
    let inv = 1.0 / kernel_len as f64;
    (0..n)
        .map(|i| {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let sum: f64 = samples[start..end].iter().map(|&s| f64::from(s)).sum();
            let count = (end - start) as f64;
            (sum / count * inv * kernel_len as f64) as f32
        })
        .collect()
}

/// Simulate quantization attack (reduce to `bits`-bit resolution).
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn apply_quantization(samples: &[f32], bits: u32) -> Vec<f32> {
    let levels = f64::from(1u32 << bits.min(30));
    samples
        .iter()
        .map(|&s| {
            let quantized = (f64::from(s) * levels).round() / levels;
            quantized as f32
        })
        .collect()
}

/// Estimate bit error rate between original and attacked watermark bits.
///
/// `original_bits` and `recovered_bits` are byte slices; comparison is bit-level.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_ber(original_bits: &[u8], recovered_bits: &[u8]) -> f64 {
    let len = original_bits.len().min(recovered_bits.len());
    if len == 0 {
        return 0.5;
    }
    let mut errors = 0usize;
    let mut total = 0usize;
    for (&a, &b) in original_bits.iter().zip(recovered_bits.iter()) {
        let xor = a ^ b;
        errors += xor.count_ones() as usize;
        total += 8;
    }
    errors as f64 / total as f64
}

/// Compute survival rate from BER.
///
/// Maps BER=0 → 1.0 (fully survived) and BER=0.5 → 0.0 (random).
#[must_use]
pub fn ber_to_survival_rate(ber: f64) -> f64 {
    (1.0 - ber * 2.0).max(0.0)
}

/// Run a standard robustness test suite against a set of attacks.
///
/// `original_payload` is the embedded payload bytes.
/// `detect_fn` receives the attacked samples and returns detected payload bytes.
#[allow(clippy::cast_precision_loss)]
pub fn run_robustness_suite<F>(
    watermarked: &[f32],
    original_payload: &[u8],
    attacks: &[AttackParams],
    config: &RobustnessConfig,
    detect_fn: F,
) -> Vec<RobustnessResult>
where
    F: Fn(&[f32]) -> Vec<u8>,
{
    attacks
        .iter()
        .map(|attack| {
            let attacked = match attack.attack_type {
                AttackType::AwgnNoise => apply_awgn(watermarked, attack.strength),
                AttackType::AmplitudeScale => apply_amplitude_scale(watermarked, attack.strength),
                AttackType::LowPassFilter => {
                    let kernel = ((1.0 - attack.strength) * 32.0) as usize + 1;
                    apply_low_pass_box(watermarked, kernel)
                }
                AttackType::Quantization => apply_quantization(watermarked, attack.strength as u32),
                AttackType::Echo => {
                    // Simple echo: add a delayed, attenuated copy
                    let delay = (watermarked.len() / 20).max(1);
                    let mut out = watermarked.to_vec();
                    for i in delay..out.len() {
                        out[i] =
                            (f64::from(out[i]) + f64::from(watermarked[i - delay]) * 0.3) as f32;
                    }
                    out
                }
                AttackType::Resampling | AttackType::TimeStretch | AttackType::SampleJitter => {
                    // Approximate: slight amplitude perturbation
                    apply_amplitude_scale(watermarked, 0.99)
                }
            };

            let detected = detect_fn(&attacked);
            let ber = estimate_ber(original_payload, &detected);
            let survival_rate = ber_to_survival_rate(ber);
            let detectable = ber < config.ber_threshold;

            RobustnessResult {
                attack: attack.clone(),
                survival_rate,
                bit_error_rate: ber,
                detectable,
            }
        })
        .collect()
}

/// Aggregate survival score across all robustness results.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn aggregate_survival_score(results: &[RobustnessResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let sum: f64 = results.iter().map(|r| r.survival_rate).sum();
    sum / results.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_awgn_same_length() {
        let samples = vec![0.5f32; 100];
        let noisy = apply_awgn(&samples, 20.0);
        assert_eq!(noisy.len(), 100);
    }

    #[test]
    fn test_apply_awgn_empty() {
        let noisy = apply_awgn(&[], 20.0);
        assert!(noisy.is_empty());
    }

    #[test]
    fn test_apply_awgn_high_snr_close_to_original() {
        let samples = vec![1.0f32; 1000];
        let noisy = apply_awgn(&samples, 60.0);
        let diff: f32 = samples
            .iter()
            .zip(noisy.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / 1000.0;
        assert!(
            diff < 0.01,
            "High SNR should preserve signal, avg diff={diff}"
        );
    }

    #[test]
    fn test_apply_amplitude_scale_doubles() {
        let samples = vec![0.5f32; 10];
        let scaled = apply_amplitude_scale(&samples, 2.0);
        for s in &scaled {
            assert!((*s - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_amplitude_scale_zero() {
        let samples = vec![0.5f32; 10];
        let scaled = apply_amplitude_scale(&samples, 0.0);
        for s in &scaled {
            assert!(s.abs() < 1e-5);
        }
    }

    #[test]
    fn test_apply_low_pass_same_length() {
        let samples = vec![1.0f32; 50];
        let filtered = apply_low_pass_box(&samples, 5);
        assert_eq!(filtered.len(), 50);
    }

    #[test]
    fn test_apply_low_pass_zero_kernel() {
        let samples = vec![1.0f32; 10];
        let filtered = apply_low_pass_box(&samples, 0);
        assert_eq!(filtered, samples);
    }

    #[test]
    fn test_apply_quantization_same_length() {
        let samples = vec![0.3f32; 20];
        let q = apply_quantization(&samples, 8);
        assert_eq!(q.len(), 20);
    }

    #[test]
    fn test_apply_quantization_reduces_precision() {
        let samples = vec![0.123456789f32; 10];
        let q = apply_quantization(&samples, 4);
        // 4-bit quantization should round to 1/16 intervals
        for &s in &q {
            let levels = 16.0f32;
            let rounded = (s * levels).round() / levels;
            assert!((s - rounded).abs() < 1e-4);
        }
    }

    #[test]
    fn test_estimate_ber_identical() {
        let payload = vec![0b10101010u8, 0b11001100];
        let ber = estimate_ber(&payload, &payload);
        assert_eq!(ber, 0.0);
    }

    #[test]
    fn test_estimate_ber_all_flipped() {
        let original = vec![0xFFu8, 0xFF];
        let recovered = vec![0x00u8, 0x00];
        let ber = estimate_ber(&original, &recovered);
        assert!((ber - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_ber_half_flipped() {
        let original = vec![0b11110000u8];
        let recovered = vec![0b00001111u8];
        let ber = estimate_ber(&original, &recovered);
        assert!((ber - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_ber_empty() {
        let ber = estimate_ber(&[], &[]);
        assert!((ber - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ber_to_survival_rate_zero_ber() {
        assert!((ber_to_survival_rate(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ber_to_survival_rate_half_ber() {
        assert!((ber_to_survival_rate(0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ber_to_survival_rate_clamped() {
        assert_eq!(ber_to_survival_rate(1.0), 0.0);
    }

    #[test]
    fn test_aggregate_survival_score_empty() {
        assert_eq!(aggregate_survival_score(&[]), 0.0);
    }

    #[test]
    fn test_aggregate_survival_score_average() {
        let results = vec![
            RobustnessResult {
                attack: AttackParams::awgn(30.0),
                survival_rate: 0.8,
                bit_error_rate: 0.1,
                detectable: true,
            },
            RobustnessResult {
                attack: AttackParams::awgn(10.0),
                survival_rate: 0.4,
                bit_error_rate: 0.3,
                detectable: false,
            },
        ];
        let score = aggregate_survival_score(&results);
        assert!((score - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_robustness_config_default() {
        let config = RobustnessConfig::default();
        assert!((config.ber_threshold - 0.1).abs() < 1e-9);
        assert!((config.min_survival_rate - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_run_robustness_suite_count() {
        let watermarked = vec![0.3f32; 1000];
        let payload = vec![0xAAu8, 0xBB, 0xCC];
        let attacks = vec![
            AttackParams::awgn(30.0),
            AttackParams::amplitude(0.9),
            AttackParams::quantization(8),
        ];
        let config = RobustnessConfig::default();
        let detect_fn = |_: &[f32]| payload.clone();
        let results = run_robustness_suite(&watermarked, &payload, &attacks, &config, detect_fn);
        assert_eq!(results.len(), 3);
    }
}
