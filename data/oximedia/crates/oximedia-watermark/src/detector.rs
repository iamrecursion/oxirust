//! Unified watermark detection and extraction.
//!
//! This module provides a unified interface for detecting and extracting
//! watermarks regardless of the embedding algorithm used.

use crate::payload::SyncDetector;
use oxifft::Complex;

/// Watermark detection result.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether watermark was detected.
    pub detected: bool,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f32,
    /// Extracted payload (if detected).
    pub payload: Option<Vec<u8>>,
    /// Signal-to-Watermark Ratio in dB.
    pub swr_db: Option<f32>,
    /// Bit error rate (if applicable).
    pub ber: Option<f32>,
}

/// Blind watermark detector (no original required).
pub struct BlindDetector {
    #[allow(dead_code)]
    sample_rate: u32,
    sync_detector: Option<SyncDetector>,
}

impl BlindDetector {
    /// Create a new blind detector.
    #[must_use]
    pub fn new(sample_rate: u32, sync_key: Option<u64>) -> Self {
        let sync_detector = sync_key.map(|key| SyncDetector::new(key, 256));

        Self {
            sample_rate,
            sync_detector,
        }
    }

    /// Detect watermark presence.
    #[must_use]
    pub fn detect(&self, samples: &[f32]) -> DetectionResult {
        // Try synchronization detection first
        if let Some(ref sync) = self.sync_detector {
            if let Some(_offset) = sync.detect(samples, 0.7) {
                return DetectionResult {
                    detected: true,
                    confidence: 0.9,
                    payload: None,
                    swr_db: None,
                    ber: None,
                };
            }
        }

        // Analyze spectral characteristics
        let spectral_score = self.analyze_spectral_characteristics(samples);

        DetectionResult {
            detected: spectral_score > 0.5,
            confidence: spectral_score,
            payload: None,
            swr_db: None,
            ber: None,
        }
    }

    /// Analyze spectral characteristics for watermark detection.
    fn analyze_spectral_characteristics(&self, samples: &[f32]) -> f32 {
        let frame_size = 2048;
        if samples.len() < frame_size {
            return 0.0;
        }

        let freq_input: Vec<Complex<f32>> = samples[..frame_size]
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();

        let freq_data = oxifft::fft(&freq_input);

        // Calculate spectral flatness
        let flatness = self.calculate_spectral_flatness(&freq_data);

        // Watermarks typically reduce spectral flatness
        (1.0 - flatness).clamp(0.0, 1.0)
    }

    /// Calculate spectral flatness measure.
    fn calculate_spectral_flatness(&self, freq_data: &[Complex<f32>]) -> f32 {
        let n = freq_data.len() / 2;
        let mut geometric_mean = 1.0f32;
        let mut arithmetic_mean = 0.0f32;

        #[allow(clippy::cast_precision_loss)]
        let divisor = 1.0 / n as f32;
        for i in 1..n {
            let mag = freq_data[i].norm().max(1e-10);
            geometric_mean *= mag.powf(divisor);
            arithmetic_mean += mag * divisor;
        }

        if arithmetic_mean > 1e-10 {
            geometric_mean / arithmetic_mean
        } else {
            0.0
        }
    }

    /// Estimate watermark strength.
    #[must_use]
    pub fn estimate_strength(&self, original: &[f32], watermarked: &[f32]) -> f32 {
        let n = original.len().min(watermarked.len());
        let mut watermark_energy = 0.0f32;
        let mut signal_energy = 0.0f32;

        for i in 0..n {
            let wm = watermarked[i] - original[i];
            watermark_energy += wm * wm;
            signal_energy += original[i] * original[i];
        }

        if signal_energy > 1e-10 {
            10.0 * (watermark_energy / signal_energy).log10()
        } else {
            -100.0
        }
    }
}

/// Non-blind detector (requires original).
pub struct NonBlindDetector {
    threshold: f32,
}

impl NonBlindDetector {
    /// Create a new non-blind detector.
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Detect watermark by comparing with original.
    #[must_use]
    pub fn detect(&self, original: &[f32], watermarked: &[f32]) -> DetectionResult {
        let correlation = self.calculate_correlation(original, watermarked);
        let swr_db = self.calculate_swr(original, watermarked);

        DetectionResult {
            detected: correlation > self.threshold,
            confidence: correlation,
            payload: None,
            swr_db: Some(swr_db),
            ber: None,
        }
    }

    /// Calculate correlation between original and watermarked.
    fn calculate_correlation(&self, original: &[f32], watermarked: &[f32]) -> f32 {
        let n = original.len().min(watermarked.len());
        let mut sum = 0.0f32;
        let mut orig_energy = 0.0f32;
        let mut wm_energy = 0.0f32;

        for i in 0..n {
            sum += original[i] * watermarked[i];
            orig_energy += original[i] * original[i];
            wm_energy += watermarked[i] * watermarked[i];
        }

        if orig_energy > 1e-10 && wm_energy > 1e-10 {
            sum / (orig_energy.sqrt() * wm_energy.sqrt())
        } else {
            0.0
        }
    }

    /// Calculate Signal-to-Watermark Ratio.
    fn calculate_swr(&self, original: &[f32], watermarked: &[f32]) -> f32 {
        let n = original.len().min(watermarked.len());
        let mut signal_energy = 0.0f32;
        let mut noise_energy = 0.0f32;

        for i in 0..n {
            signal_energy += original[i] * original[i];
            let diff = watermarked[i] - original[i];
            noise_energy += diff * diff;
        }

        if noise_energy > 1e-10 {
            10.0 * (signal_energy / noise_energy).log10()
        } else {
            100.0
        }
    }
}

/// Multi-algorithm detector that tries different detection methods.
pub struct MultiDetector {
    detectors: Vec<DetectorType>,
}

#[derive(Debug, Clone)]
enum DetectorType {
    Sync { key: u64, threshold: f32 },
    Spectral { threshold: f32 },
    Correlation { threshold: f32 },
}

impl MultiDetector {
    /// Create a new multi-detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
        }
    }

    /// Add synchronization detector.
    pub fn add_sync_detector(&mut self, key: u64, threshold: f32) {
        self.detectors.push(DetectorType::Sync { key, threshold });
    }

    /// Add spectral detector.
    pub fn add_spectral_detector(&mut self, threshold: f32) {
        self.detectors.push(DetectorType::Spectral { threshold });
    }

    /// Add correlation detector.
    pub fn add_correlation_detector(&mut self, threshold: f32) {
        self.detectors.push(DetectorType::Correlation { threshold });
    }

    /// Detect using all configured detectors.
    #[must_use]
    pub fn detect(&self, samples: &[f32], original: Option<&[f32]>) -> Vec<DetectionResult> {
        let mut results = Vec::new();

        for detector in &self.detectors {
            match detector {
                DetectorType::Sync { key, threshold } => {
                    let sync = SyncDetector::new(*key, 256);
                    let detected = sync.detect(samples, *threshold).is_some();
                    results.push(DetectionResult {
                        detected,
                        confidence: if detected { 0.9 } else { 0.1 },
                        payload: None,
                        swr_db: None,
                        ber: None,
                    });
                }
                DetectorType::Spectral { threshold } => {
                    let blind = BlindDetector::new(44100, None);
                    let mut result = blind.detect(samples);
                    result.detected = result.confidence > *threshold;
                    results.push(result);
                }
                DetectorType::Correlation { threshold } => {
                    if let Some(orig) = original {
                        let non_blind = NonBlindDetector::new(*threshold);
                        results.push(non_blind.detect(orig, samples));
                    }
                }
            }
        }

        results
    }
}

impl Default for MultiDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Bit Error Rate between two bit sequences.
#[must_use]
pub fn calculate_ber(original_bits: &[bool], extracted_bits: &[bool]) -> f32 {
    let n = original_bits.len().min(extracted_bits.len());
    if n == 0 {
        return 1.0;
    }

    let mut errors = 0;
    for i in 0..n {
        if original_bits[i] != extracted_bits[i] {
            errors += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let result = errors as f32 / n as f32;
    result
}

/// Calculate detection probability using Neyman-Pearson criterion.
#[must_use]
pub fn calculate_detection_probability(correlation: f32, noise_power: f32) -> f32 {
    // Simplified detection probability model
    let snr = if noise_power > 1e-10 {
        correlation / noise_power.sqrt()
    } else {
        correlation
    };

    // Approximate probability using error function
    0.5 * (1.0 + (snr / std::f32::consts::SQRT_2).tanh())
}

/// Calculate false positive rate.
#[must_use]
pub fn calculate_false_positive_rate(threshold: f32, noise_power: f32) -> f32 {
    // Simplified false positive model
    let normalized_threshold = threshold / noise_power.sqrt().max(1e-10);
    0.5 * (1.0 - (normalized_threshold / std::f32::consts::SQRT_2).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blind_detection() {
        let detector = BlindDetector::new(44100, Some(12345));
        let samples: Vec<f32> = vec![0.1; 10000];

        let result = detector.detect(&samples);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_non_blind_detection() {
        let detector = NonBlindDetector::new(0.8);

        let original: Vec<f32> = vec![0.1; 1000];
        let watermarked: Vec<f32> = original.iter().map(|&s| s * 1.01).collect();

        let result = detector.detect(&original, &watermarked);
        assert!(result.swr_db.is_some());
    }

    #[test]
    fn test_multi_detector() {
        let mut detector = MultiDetector::new();
        detector.add_sync_detector(12345, 0.7);
        detector.add_spectral_detector(0.5);

        let samples: Vec<f32> = vec![0.1; 10000];
        let results = detector.detect(&samples, None);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ber_calculation() {
        let orig = vec![true, false, true, true, false];
        let extr = vec![true, true, true, false, false];

        let ber = calculate_ber(&orig, &extr);
        assert!((ber - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_detection_probability() {
        let prob = calculate_detection_probability(0.8, 0.1);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_spectral_flatness() {
        let detector = BlindDetector::new(44100, None);

        let frame_size = 2048;

        let freq_input: Vec<Complex<f32>> =
            (0..frame_size).map(|_| Complex::new(1.0, 0.0)).collect();

        let freq_data = oxifft::fft(&freq_input);

        let flatness = detector.calculate_spectral_flatness(&freq_data);
        assert!(flatness >= 0.0 && flatness <= 1.0);
    }
}
