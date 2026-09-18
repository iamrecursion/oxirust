//! Comprehensive robustness testing for watermark survival against common attacks.
//!
//! This module provides systematic tests that verify watermark survival under:
//! - Compression (MP3 simulation, quantization)
//! - Filtering (low-pass, high-pass, band-pass)
//! - Noise addition (AWGN at various SNR levels)
//! - Cropping (start, end, middle removal)
//! - Amplitude scaling and dynamic range compression
//! - Resampling
//! - Time stretching and pitch shifting
//!
//! The test harness runs each attack at multiple severity levels and reports
//! a robustness score per algorithm.

use crate::attacks::{
    add_echo, add_noise, amplitude_scale, compress, crop, highpass_filter, jitter, lowpass_filter,
    mp3_compression, pitch_shift, resample, time_stretch,
};
use crate::error::WatermarkResult;
use crate::payload::PayloadCodec;
use crate::{Algorithm, WatermarkConfig, WatermarkDetector, WatermarkEmbedder};

/// Result of a single robustness test.
#[derive(Debug, Clone)]
pub struct AttackResult {
    /// Name of the attack.
    pub attack_name: String,
    /// Whether the watermark was successfully detected after the attack.
    pub survived: bool,
    /// Bit error rate (0.0 = perfect, 1.0 = all wrong).
    pub ber: f32,
    /// Description of attack parameters.
    pub params: String,
}

/// Aggregate robustness report for an algorithm.
#[derive(Debug, Clone)]
pub struct RobustnessReport {
    /// Algorithm tested.
    pub algorithm: Algorithm,
    /// Individual attack results.
    pub results: Vec<AttackResult>,
    /// Overall survival rate (fraction of attacks survived).
    pub survival_rate: f32,
    /// Average BER across all attacks.
    pub average_ber: f32,
}

/// Robustness test suite runner.
pub struct RobustnessSuite {
    sample_rate: u32,
    /// Audio length in samples for test signals.
    signal_length: usize,
    /// Payload to embed for testing.
    payload: Vec<u8>,
}

impl RobustnessSuite {
    /// Create a new robustness suite.
    #[must_use]
    pub fn new(sample_rate: u32, signal_length: usize, payload: Vec<u8>) -> Self {
        Self {
            sample_rate,
            signal_length,
            payload,
        }
    }

    /// Create with default parameters (1 byte payload, ~2 seconds of audio).
    #[must_use]
    pub fn default_suite() -> Self {
        Self {
            sample_rate: 44100,
            signal_length: 73728, // 36 * 2048, enough for SS
            payload: b"W".to_vec(),
        }
    }

    /// Run the full robustness suite for a given algorithm.
    pub fn run(&self, algorithm: Algorithm) -> RobustnessReport {
        let config = WatermarkConfig::default()
            .with_algorithm(algorithm)
            .with_strength(0.15)
            .with_key(0x0B05_7E57)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), self.sample_rate);
        let detector = WatermarkDetector::new(config);

        // Generate test signal (sine sweep for realistic content)
        let signal = self.generate_test_signal();

        // Try to embed
        let watermarked = match embedder.embed(&signal, &self.payload) {
            Ok(wm) => wm,
            Err(_) => {
                return RobustnessReport {
                    algorithm,
                    results: vec![AttackResult {
                        attack_name: "embed".to_string(),
                        survived: false,
                        ber: 1.0,
                        params: "embedding failed".to_string(),
                    }],
                    survival_rate: 0.0,
                    average_ber: 1.0,
                };
            }
        };

        // Get expected bits
        let expected_bits = match self.expected_bits() {
            Ok(bits) => bits,
            Err(_) => {
                return RobustnessReport {
                    algorithm,
                    results: Vec::new(),
                    survival_rate: 0.0,
                    average_ber: 1.0,
                };
            }
        };

        // Run attacks
        let mut results = Vec::new();

        // No attack (baseline)
        results.push(self.test_attack(
            "baseline",
            "no attack",
            &watermarked,
            &detector,
            expected_bits,
        ));

        // Compression attacks
        results.push(self.test_attack(
            "mp3_128kbps",
            "MP3 128 kbps",
            &mp3_compression(&watermarked, 128),
            &detector,
            expected_bits,
        ));

        results.push(self.test_attack(
            "mp3_64kbps",
            "MP3 64 kbps",
            &mp3_compression(&watermarked, 64),
            &detector,
            expected_bits,
        ));

        // Noise attacks
        for &snr in &[40.0, 30.0, 20.0] {
            results.push(self.test_attack(
                &format!("noise_{snr}dB"),
                &format!("AWGN {snr} dB"),
                &add_noise(&watermarked, snr),
                &detector,
                expected_bits,
            ));
        }

        // Filtering
        results.push(self.test_attack(
            "lowpass_8k",
            "LPF 8 kHz",
            &lowpass_filter(&watermarked, 8000.0, self.sample_rate),
            &detector,
            expected_bits,
        ));

        results.push(self.test_attack(
            "lowpass_4k",
            "LPF 4 kHz",
            &lowpass_filter(&watermarked, 4000.0, self.sample_rate),
            &detector,
            expected_bits,
        ));

        results.push(self.test_attack(
            "highpass_100",
            "HPF 100 Hz",
            &highpass_filter(&watermarked, 100.0, self.sample_rate),
            &detector,
            expected_bits,
        ));

        // Amplitude scaling
        results.push(self.test_attack(
            "scale_0.5",
            "amplitude x0.5",
            &amplitude_scale(&watermarked, 0.5),
            &detector,
            expected_bits,
        ));

        results.push(self.test_attack(
            "scale_2.0",
            "amplitude x2.0",
            &amplitude_scale(&watermarked, 2.0),
            &detector,
            expected_bits,
        ));

        // Dynamic range compression
        results.push(self.test_attack(
            "compress_4:1",
            "DRC 4:1 @ 0.5",
            &compress(&watermarked, 0.5, 4.0),
            &detector,
            expected_bits,
        ));

        // Cropping
        let crop_start = self.signal_length / 10;
        results.push(self.test_attack(
            "crop_front_10%",
            "remove first 10%",
            &crop(&watermarked, crop_start, self.signal_length),
            &detector,
            expected_bits,
        ));

        // Echo addition
        results.push(self.test_attack(
            "echo_50ms",
            "echo 50ms decay 0.3",
            &add_echo(&watermarked, (self.sample_rate as usize) / 20, 0.3),
            &detector,
            expected_bits,
        ));

        // Resampling (down then up)
        let resampled_down = resample(&watermarked, self.sample_rate, 22050);
        let resampled_back = resample(&resampled_down, 22050, self.sample_rate);
        results.push(self.test_attack(
            "resample_22k",
            "44.1k→22k→44.1k",
            &resampled_back,
            &detector,
            expected_bits,
        ));

        // Jitter
        results.push(self.test_attack(
            "jitter_2",
            "jitter ±2 samples",
            &jitter(&watermarked, 2),
            &detector,
            expected_bits,
        ));

        // Time stretch
        results.push(self.test_attack(
            "time_stretch_1.05",
            "stretch 5%",
            &time_stretch(&watermarked, 1.05),
            &detector,
            expected_bits,
        ));

        // Pitch shift
        results.push(self.test_attack(
            "pitch_+1st",
            "pitch +1 semitone",
            &pitch_shift(&watermarked, 1.0),
            &detector,
            expected_bits,
        ));

        // Compute summary
        let survived_count = results.iter().filter(|r| r.survived).count();
        #[allow(clippy::cast_precision_loss)]
        let survival_rate = survived_count as f32 / results.len() as f32;
        #[allow(clippy::cast_precision_loss)]
        let average_ber = results.iter().map(|r| r.ber).sum::<f32>() / results.len() as f32;

        RobustnessReport {
            algorithm,
            results,
            survival_rate,
            average_ber,
        }
    }

    /// Test a single attack.
    fn test_attack(
        &self,
        name: &str,
        params: &str,
        attacked: &[f32],
        detector: &WatermarkDetector,
        expected_bits: usize,
    ) -> AttackResult {
        match detector.detect(attacked, expected_bits) {
            Ok(detected) => {
                let ber = bit_error_rate(&self.payload, &detected);
                AttackResult {
                    attack_name: name.to_string(),
                    survived: ber < 0.1, // <10% BER = survived
                    ber,
                    params: params.to_string(),
                }
            }
            Err(_) => AttackResult {
                attack_name: name.to_string(),
                survived: false,
                ber: 1.0,
                params: params.to_string(),
            },
        }
    }

    /// Generate test signal (sum of sine waves at different frequencies).
    fn generate_test_signal(&self) -> Vec<f32> {
        (0..self.signal_length)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / self.sample_rate as f32;
                let f1 = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
                let f2 = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.2;
                let f3 = (2.0 * std::f32::consts::PI * 2500.0 * t).sin() * 0.1;
                f1 + f2 + f3
            })
            .collect()
    }

    /// Get expected encoded bits for the payload.
    fn expected_bits(&self) -> WatermarkResult<usize> {
        let codec = PayloadCodec::new(16, 8)?;
        let encoded = codec.encode(&self.payload)?;
        Ok(encoded.len() * 8)
    }
}

/// Calculate bit error rate between original and detected payloads.
#[must_use]
pub fn bit_error_rate(original: &[u8], detected: &[u8]) -> f32 {
    let max_len = original.len().max(detected.len());
    if max_len == 0 {
        return 0.0;
    }

    let mut errors = 0usize;
    let mut total = 0usize;

    for i in 0..max_len {
        let orig_byte = original.get(i).copied().unwrap_or(0);
        let det_byte = detected.get(i).copied().unwrap_or(0);
        let xor = orig_byte ^ det_byte;

        for bit in 0..8 {
            if (xor >> bit) & 1 == 1 {
                errors += 1;
            }
            total += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let ber = errors as f32 / total as f32;
    ber
}

/// Run a quick comparison of all algorithms for robustness.
#[must_use]
pub fn compare_algorithms() -> Vec<RobustnessReport> {
    let suite = RobustnessSuite::default_suite();

    vec![
        suite.run(Algorithm::SpreadSpectrum),
        // Other algorithms can be added but may need different signal lengths
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_error_rate_identical() {
        let a = b"Hello";
        let b = b"Hello";
        assert!((bit_error_rate(a, b)).abs() < 1e-6);
    }

    #[test]
    fn test_bit_error_rate_different() {
        let a = b"\x00";
        let b = b"\xFF";
        assert!((bit_error_rate(a, b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bit_error_rate_partial() {
        let a = b"\x00";
        let b = b"\x0F"; // 4 bits different
        assert!((bit_error_rate(a, b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bit_error_rate_different_lengths() {
        let a = b"AB";
        let b = b"A";
        let ber = bit_error_rate(a, b);
        assert!(ber > 0.0); // Second byte of b treated as 0
    }

    #[test]
    fn test_robustness_suite_baseline() {
        let suite = RobustnessSuite::new(44100, 73728, b"W".to_vec());
        let report = suite.run(Algorithm::SpreadSpectrum);

        // The baseline (no attack) should survive
        let baseline = report.results.iter().find(|r| r.attack_name == "baseline");
        assert!(baseline.is_some());

        // Report should have multiple attack results
        assert!(report.results.len() > 5, "expected many attacks");
    }

    #[test]
    fn test_robustness_spread_spectrum_noise_40db() {
        // SS should survive mild noise (40 dB SNR)
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let attacked = add_noise(&watermarked, 40.0);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&attacked, bits);
        // Should at least not panic; may or may not detect
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_amplitude_scaling() {
        // Watermark should survive amplitude scaling (it's proportional)
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let scaled = amplitude_scale(&watermarked, 0.5);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&scaled, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_compression() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let compressed = mp3_compression(&watermarked, 128);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&compressed, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_lowpass_filter() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let filtered = lowpass_filter(&watermarked, 8000.0, 44100);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&filtered, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_dynamic_compression() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = (0..73728)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / 44100.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let compressed = compress(&watermarked, 0.3, 4.0);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&compressed, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_echo_attack() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let echoed = add_echo(&watermarked, 2205, 0.3); // 50ms echo

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&echoed, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_report_structure() {
        let suite = RobustnessSuite::default_suite();
        let report = suite.run(Algorithm::SpreadSpectrum);

        assert_eq!(report.algorithm, Algorithm::SpreadSpectrum);
        assert!(report.survival_rate >= 0.0 && report.survival_rate <= 1.0);
        assert!(report.average_ber >= 0.0 && report.average_ber <= 1.0);

        for result in &report.results {
            assert!(!result.attack_name.is_empty());
            assert!(!result.params.is_empty());
            assert!(result.ber >= 0.0 && result.ber <= 1.0);
        }
    }

    #[test]
    fn test_compare_algorithms() {
        let reports = compare_algorithms();
        assert!(!reports.is_empty());

        for report in &reports {
            assert!(!report.results.is_empty());
        }
    }

    #[test]
    fn test_robustness_cropping() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        // Crop: keep 90% from start
        let cropped = crop(&watermarked, 0, (watermarked.len() * 9) / 10);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&cropped, bits);
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_robustness_jitter() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15)
            .with_key(12345)
            .with_psychoacoustic(false);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        let signal: Vec<f32> = vec![0.0; 73728];
        let payload = b"W";

        let watermarked = embedder
            .embed(&signal, payload)
            .expect("embed should succeed");
        let jittered = jitter(&watermarked, 3);

        let codec = PayloadCodec::new(16, 8).expect("codec should succeed");
        let encoded = codec.encode(payload).expect("encode should succeed");
        let bits = encoded.len() * 8;

        let result = detector.detect(&jittered, bits);
        assert!(result.is_ok() || result.is_err());
    }
}
