#![allow(dead_code)]
//! Pitch detection via autocorrelation and the YIN algorithm.

/// A single pitch candidate with frequency and confidence.
#[derive(Debug, Clone)]
pub struct PitchCandidate {
    /// Estimated fundamental frequency in Hz.
    pub frequency: f32,
    /// Confidence score in the range 0.0–1.0.
    pub confidence: f32,
}

impl PitchCandidate {
    /// Create a new `PitchCandidate`.
    #[must_use]
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns `true` if the confidence meets `threshold`.
    #[must_use]
    pub fn confidence_ok(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

// ---------------------------------------------------------------------------

/// Autocorrelation-based pitch estimator.
pub struct AutoCorrelation {
    /// Minimum detectable frequency (Hz).
    pub min_freq: f32,
    /// Maximum detectable frequency (Hz).
    pub max_freq: f32,
}

impl AutoCorrelation {
    /// Create a new `AutoCorrelation` estimator.
    #[must_use]
    pub fn new(min_freq: f32, max_freq: f32) -> Self {
        Self { min_freq, max_freq }
    }

    /// Compute normalised autocorrelation for lags corresponding to
    /// [min\_freq, max\_freq] and return a `PitchCandidate`.
    ///
    /// Returns `None` if the signal is too short or silent.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(&self, samples: &[f32], sample_rate: f32) -> Option<PitchCandidate> {
        if sample_rate <= 0.0 || samples.is_empty() {
            return None;
        }

        let lag_min = (sample_rate / self.max_freq).round() as usize;
        let lag_max = (sample_rate / self.min_freq).round() as usize;

        if lag_min >= samples.len() || lag_max >= samples.len() || lag_min >= lag_max {
            return None;
        }

        // r0 — autocorrelation at lag 0 (energy)
        let r0: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if r0 == 0.0 {
            return None;
        }

        let mut best_corr = -1.0_f64;
        let mut best_lag = lag_min;

        for lag in lag_min..=lag_max.min(samples.len() - 1) {
            let mut r = 0.0_f64;
            for i in 0..(samples.len() - lag) {
                r += f64::from(samples[i]) * f64::from(samples[i + lag]);
            }
            let norm = r / r0;
            if norm > best_corr {
                best_corr = norm;
                best_lag = lag;
            }
        }

        let freq = sample_rate / best_lag as f32;
        let confidence = ((best_corr + 1.0) / 2.0).clamp(0.0, 1.0) as f32;
        Some(PitchCandidate::new(freq, confidence))
    }
}

// ---------------------------------------------------------------------------

/// Convenience function: detect pitch using autocorrelation.
///
/// Returns `None` if pitch cannot be reliably detected.
#[must_use]
pub fn detect_pitch(samples: &[f32], sample_rate: f32) -> Option<PitchCandidate> {
    AutoCorrelation::new(60.0, 1600.0).compute(samples, sample_rate)
}

// ---------------------------------------------------------------------------

/// YIN pitch detector — uses the difference function to avoid octave errors.
pub struct YinPitchDetector {
    /// Threshold for the normalised difference function (typically 0.1–0.15).
    pub threshold: f32,
    /// Minimum frequency to consider (Hz).
    pub min_freq: f32,
    /// Maximum frequency to consider (Hz).
    pub max_freq: f32,
}

impl YinPitchDetector {
    /// Create a `YinPitchDetector` with the given threshold.
    #[must_use]
    pub fn new(threshold: f32, min_freq: f32, max_freq: f32) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            min_freq,
            max_freq,
        }
    }

    /// Detect the fundamental frequency using the YIN algorithm.
    ///
    /// Returns `None` if no pitch is found below `threshold`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect(&self, samples: &[f32], sample_rate: f32) -> Option<PitchCandidate> {
        if samples.is_empty() || sample_rate <= 0.0 {
            return None;
        }

        let lag_min = (sample_rate / self.max_freq).ceil() as usize;
        let lag_max = (sample_rate / self.min_freq).floor() as usize;

        if lag_max >= samples.len() || lag_min >= lag_max {
            return None;
        }

        let n = samples.len();

        // Step 1: difference function d(tau)
        let mut d: Vec<f64> = vec![0.0; lag_max + 1];
        for tau in 1..=lag_max {
            for j in 0..(n - tau) {
                let diff = f64::from(samples[j]) - f64::from(samples[j + tau]);
                d[tau] += diff * diff;
            }
        }

        // Step 2: cumulative mean normalised difference
        let mut d_prime: Vec<f64> = vec![0.0; lag_max + 1];
        d_prime[0] = 1.0;
        let mut running_sum = 0.0_f64;
        for tau in 1..=lag_max {
            running_sum += d[tau];
            if running_sum == 0.0 {
                d_prime[tau] = 1.0;
            } else {
                d_prime[tau] = d[tau] * tau as f64 / running_sum;
            }
        }

        // Step 3: absolute threshold — find first lag where d' < threshold
        let mut best_lag = None;
        for tau in lag_min..=lag_max {
            if d_prime[tau] < f64::from(self.threshold) {
                // Find local minimum in neighbourhood
                let mut min_tau = tau;
                while min_tau < lag_max && d_prime[min_tau + 1] < d_prime[min_tau] {
                    min_tau += 1;
                }
                best_lag = Some(min_tau);
                break;
            }
        }

        let lag = best_lag?;
        let freq = sample_rate / lag as f32;
        let aperiodicity = (d_prime[lag] as f32).clamp(0.0, 1.0);
        let confidence = 1.0 - aperiodicity;

        Some(PitchCandidate::new(freq, confidence))
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a pure sine wave at `freq` Hz.
    fn sine_wave(freq: f32, sample_rate: f32, n_samples: usize) -> Vec<f32> {
        (0..n_samples)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    // --- PitchCandidate ---

    #[test]
    fn test_confidence_ok_above_threshold() {
        let c = PitchCandidate::new(440.0, 0.9);
        assert!(c.confidence_ok(0.8));
    }

    #[test]
    fn test_confidence_ok_below_threshold() {
        let c = PitchCandidate::new(440.0, 0.5);
        assert!(!c.confidence_ok(0.8));
    }

    #[test]
    fn test_confidence_clamped_high() {
        let c = PitchCandidate::new(440.0, 1.5);
        assert!((c.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_clamped_low() {
        let c = PitchCandidate::new(440.0, -0.5);
        assert!((c.confidence - 0.0).abs() < f32::EPSILON);
    }

    // --- AutoCorrelation ---

    #[test]
    fn test_autocorr_sine_440hz() {
        let samples = sine_wave(440.0, 44100.0, 4096);
        let ac = AutoCorrelation::new(60.0, 1600.0);
        let candidate = ac.compute(&samples, 44100.0).expect("should find pitch");
        let err = (candidate.frequency - 440.0).abs() / 440.0;
        assert!(err < 0.05, "freq={}", candidate.frequency);
    }

    #[test]
    fn test_autocorr_silent_signal() {
        let samples = vec![0.0_f32; 2048];
        let ac = AutoCorrelation::new(60.0, 1600.0);
        assert!(ac.compute(&samples, 44100.0).is_none());
    }

    #[test]
    fn test_autocorr_empty_signal() {
        let ac = AutoCorrelation::new(60.0, 1600.0);
        assert!(ac.compute(&[], 44100.0).is_none());
    }

    #[test]
    fn test_autocorr_zero_sample_rate() {
        let samples = vec![1.0_f32; 512];
        let ac = AutoCorrelation::new(60.0, 1600.0);
        assert!(ac.compute(&samples, 0.0).is_none());
    }

    // --- detect_pitch ---

    #[test]
    fn test_detect_pitch_convenience() {
        let samples = sine_wave(220.0, 44100.0, 4096);
        let c = detect_pitch(&samples, 44100.0).expect("should detect pitch");
        let err = (c.frequency - 220.0).abs() / 220.0;
        assert!(err < 0.05, "freq={}", c.frequency);
    }

    // --- YinPitchDetector ---

    #[test]
    fn test_yin_sine_440hz() {
        let samples = sine_wave(440.0, 44100.0, 4096);
        let yin = YinPitchDetector::new(0.1, 60.0, 1600.0);
        let c = yin
            .detect(&samples, 44100.0)
            .expect("YIN should find pitch");
        let err = (c.frequency - 440.0).abs() / 440.0;
        assert!(err < 0.05, "yin freq={}", c.frequency);
    }

    #[test]
    fn test_yin_empty_signal() {
        let yin = YinPitchDetector::new(0.1, 60.0, 1600.0);
        assert!(yin.detect(&[], 44100.0).is_none());
    }

    #[test]
    fn test_yin_threshold_clamped() {
        let yin = YinPitchDetector::new(2.0, 60.0, 1600.0);
        assert!((yin.threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yin_zero_sample_rate() {
        let samples = vec![0.5_f32; 1024];
        let yin = YinPitchDetector::new(0.1, 60.0, 1600.0);
        assert!(yin.detect(&samples, 0.0).is_none());
    }

    #[test]
    fn test_yin_confidence_range() {
        let samples = sine_wave(330.0, 44100.0, 4096);
        let yin = YinPitchDetector::new(0.1, 60.0, 1600.0);
        if let Some(c) = yin.detect(&samples, 44100.0) {
            assert!((0.0..=1.0).contains(&c.confidence));
        }
    }
}
