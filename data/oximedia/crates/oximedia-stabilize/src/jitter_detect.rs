#![allow(dead_code)]
//! Jitter detection and classification for video stabilization.
//!
//! This module detects and classifies different types of camera jitter including
//! high-frequency micro-vibrations, periodic oscillations, and impulsive jolts.
//! The classification helps the stabilizer choose optimal filtering parameters.

/// Type of jitter detected in the motion signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterType {
    /// No significant jitter detected.
    None,
    /// High-frequency micro-vibration (e.g., engine rumble).
    MicroVibration,
    /// Periodic oscillation (e.g., walking, vehicle bounce).
    PeriodicOscillation,
    /// Impulsive jolt (e.g., bump or impact).
    ImpulsiveJolt,
    /// Random hand shake.
    HandShake,
    /// Mixed jitter from multiple sources.
    Mixed,
}

/// Severity level of detected jitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JitterSeverity {
    /// Negligible jitter, no correction needed.
    Negligible,
    /// Mild jitter, light smoothing recommended.
    Mild,
    /// Moderate jitter, standard correction recommended.
    Moderate,
    /// Severe jitter, aggressive correction recommended.
    Severe,
    /// Extreme jitter, content may be partially unrecoverable.
    Extreme,
}

/// Analysis result for a jitter detection pass.
#[derive(Debug, Clone)]
pub struct JitterAnalysis {
    /// Detected jitter type.
    pub jitter_type: JitterType,
    /// Overall severity.
    pub severity: JitterSeverity,
    /// Root mean square of displacement in pixels.
    pub rms_displacement: f64,
    /// Peak displacement in pixels.
    pub peak_displacement: f64,
    /// Dominant frequency of oscillation (Hz), if periodic.
    pub dominant_frequency: Option<f64>,
    /// Recommended smoothing window size.
    pub recommended_window: usize,
    /// Number of impulsive events detected.
    pub impulse_count: usize,
}

/// A single motion sample for jitter analysis.
#[derive(Debug, Clone, Copy)]
pub struct MotionSample {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
    /// Timestamp in seconds.
    pub timestamp: f64,
}

impl MotionSample {
    /// Create a new motion sample.
    #[must_use]
    pub fn new(dx: f64, dy: f64, timestamp: f64) -> Self {
        Self { dx, dy, timestamp }
    }

    /// Compute the displacement magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// Configuration for the jitter detector.
#[derive(Debug, Clone)]
pub struct JitterDetectorConfig {
    /// Frame rate of the source video (fps).
    pub frame_rate: f64,
    /// Threshold for micro-vibration RMS (pixels).
    pub micro_vibration_threshold: f64,
    /// Threshold for impulsive jolt detection (pixels).
    pub impulse_threshold: f64,
    /// Minimum number of samples for reliable analysis.
    pub min_samples: usize,
    /// Periodicity detection tolerance (Hz).
    pub frequency_tolerance: f64,
}

impl Default for JitterDetectorConfig {
    fn default() -> Self {
        Self {
            frame_rate: 30.0,
            micro_vibration_threshold: 0.5,
            impulse_threshold: 15.0,
            min_samples: 10,
            frequency_tolerance: 0.5,
        }
    }
}

/// Jitter detector that analyzes motion signals.
#[derive(Debug)]
pub struct JitterDetector {
    /// Detector configuration.
    config: JitterDetectorConfig,
}

impl JitterDetector {
    /// Create a new jitter detector with the given configuration.
    #[must_use]
    pub fn new(config: JitterDetectorConfig) -> Self {
        Self { config }
    }

    /// Create a jitter detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(JitterDetectorConfig::default())
    }

    /// Analyze a sequence of motion samples for jitter.
    #[must_use]
    pub fn analyze(&self, samples: &[MotionSample]) -> JitterAnalysis {
        if samples.len() < self.config.min_samples {
            return JitterAnalysis {
                jitter_type: JitterType::None,
                severity: JitterSeverity::Negligible,
                rms_displacement: 0.0,
                peak_displacement: 0.0,
                dominant_frequency: None,
                recommended_window: 15,
                impulse_count: 0,
            };
        }

        let magnitudes: Vec<f64> = samples.iter().map(|s| s.magnitude()).collect();
        let rms = compute_rms(&magnitudes);
        let peak = magnitudes.iter().copied().fold(0.0_f64, f64::max);
        let impulse_count = self.count_impulses(&magnitudes);
        let dominant_freq = self.detect_dominant_frequency(samples);

        let jitter_type = self.classify_jitter(rms, peak, impulse_count, dominant_freq);
        let severity = self.classify_severity(rms, peak);
        let recommended_window = self.recommend_window(&jitter_type, &severity);

        JitterAnalysis {
            jitter_type,
            severity,
            rms_displacement: rms,
            peak_displacement: peak,
            dominant_frequency: dominant_freq,
            recommended_window,
            impulse_count,
        }
    }

    /// Classify the type of jitter from statistics.
    fn classify_jitter(
        &self,
        rms: f64,
        peak: f64,
        impulse_count: usize,
        dominant_freq: Option<f64>,
    ) -> JitterType {
        if rms < self.config.micro_vibration_threshold * 0.1 {
            return JitterType::None;
        }

        let has_impulses = impulse_count > 0;
        let has_periodicity = dominant_freq.is_some();
        let is_micro = rms < self.config.micro_vibration_threshold;

        match (has_impulses, has_periodicity, is_micro) {
            (true, true, _) => JitterType::Mixed,
            (true, false, _) => JitterType::ImpulsiveJolt,
            (false, true, true) => JitterType::MicroVibration,
            (false, true, false) => JitterType::PeriodicOscillation,
            (false, false, true) => JitterType::MicroVibration,
            (false, false, false) => {
                if peak / rms > 3.0 {
                    JitterType::HandShake
                } else {
                    JitterType::HandShake
                }
            }
        }
    }

    /// Classify jitter severity.
    fn classify_severity(&self, rms: f64, peak: f64) -> JitterSeverity {
        let score = rms * 0.6 + peak * 0.4;
        if score < 0.5 {
            JitterSeverity::Negligible
        } else if score < 2.0 {
            JitterSeverity::Mild
        } else if score < 8.0 {
            JitterSeverity::Moderate
        } else if score < 25.0 {
            JitterSeverity::Severe
        } else {
            JitterSeverity::Extreme
        }
    }

    /// Count impulsive events (sudden large displacements).
    fn count_impulses(&self, magnitudes: &[f64]) -> usize {
        magnitudes
            .iter()
            .filter(|&&m| m > self.config.impulse_threshold)
            .count()
    }

    /// Detect dominant frequency using zero-crossing analysis.
    fn detect_dominant_frequency(&self, samples: &[MotionSample]) -> Option<f64> {
        if samples.len() < 4 {
            return None;
        }

        // Compute mean to center the signal
        let mean_dx: f64 = samples.iter().map(|s| s.dx).sum::<f64>() / samples.len() as f64;
        let centered: Vec<f64> = samples.iter().map(|s| s.dx - mean_dx).collect();

        // Count zero crossings
        let mut crossings = 0usize;
        for i in 1..centered.len() {
            if centered[i - 1] * centered[i] < 0.0 {
                crossings += 1;
            }
        }

        if crossings < 2 {
            return None;
        }

        // Estimate frequency from zero crossings
        let duration = samples.last().map_or(0.0, |s| s.timestamp)
            - samples.first().map_or(0.0, |s| s.timestamp);
        if duration <= 0.0 {
            return None;
        }

        let freq = crossings as f64 / (2.0 * duration);
        if freq > self.config.frequency_tolerance {
            Some(freq)
        } else {
            None
        }
    }

    /// Recommend smoothing window based on jitter classification.
    fn recommend_window(&self, jitter_type: &JitterType, severity: &JitterSeverity) -> usize {
        let base = match jitter_type {
            JitterType::None => 10,
            JitterType::MicroVibration => 5,
            JitterType::PeriodicOscillation => 30,
            JitterType::ImpulsiveJolt => 15,
            JitterType::HandShake => 20,
            JitterType::Mixed => 25,
        };
        let multiplier = match severity {
            JitterSeverity::Negligible => 1.0,
            JitterSeverity::Mild => 1.0,
            JitterSeverity::Moderate => 1.5,
            JitterSeverity::Severe => 2.0,
            JitterSeverity::Extreme => 3.0,
        };
        #[allow(clippy::cast_possible_truncation)]
        let result = (base as f64 * multiplier) as usize;
        result.max(3)
    }
}

/// Compute root mean square of a slice.
fn compute_rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = values.iter().map(|v| v * v).sum();
    (sum_sq / values.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_sample_magnitude() {
        let s = MotionSample::new(3.0, 4.0, 0.0);
        assert!((s.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_sample_zero() {
        let s = MotionSample::new(0.0, 0.0, 1.0);
        assert!((s.magnitude() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rms_empty() {
        assert!((compute_rms(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rms_known() {
        let values = [3.0, 4.0];
        let expected = (25.0_f64 / 2.0).sqrt();
        assert!((compute_rms(&values) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_insufficient_samples() {
        let detector = JitterDetector::with_defaults();
        let samples = vec![MotionSample::new(1.0, 0.0, 0.0)];
        let result = detector.analyze(&samples);
        assert_eq!(result.jitter_type, JitterType::None);
        assert_eq!(result.severity, JitterSeverity::Negligible);
    }

    #[test]
    fn test_analyze_no_jitter() {
        let detector = JitterDetector::with_defaults();
        let samples: Vec<MotionSample> = (0..20)
            .map(|i| MotionSample::new(0.001, 0.001, i as f64 / 30.0))
            .collect();
        let result = detector.analyze(&samples);
        assert_eq!(result.jitter_type, JitterType::None);
    }

    #[test]
    fn test_analyze_hand_shake() {
        let detector = JitterDetector::with_defaults();
        let samples: Vec<MotionSample> = (0..30)
            .map(|i| {
                let t = i as f64 / 30.0;
                MotionSample::new((t * 2.3).sin() * 3.0, (t * 3.7).cos() * 2.5, t)
            })
            .collect();
        let result = detector.analyze(&samples);
        assert!(result.rms_displacement > 0.0);
        assert!(result.severity >= JitterSeverity::Mild);
    }

    #[test]
    fn test_analyze_impulse() {
        let detector = JitterDetector::new(JitterDetectorConfig {
            impulse_threshold: 10.0,
            ..JitterDetectorConfig::default()
        });
        let mut samples: Vec<MotionSample> = (0..20)
            .map(|i| MotionSample::new(0.5, 0.5, i as f64 / 30.0))
            .collect();
        // Insert an impulse
        samples[10] = MotionSample::new(20.0, 15.0, 10.0 / 30.0);
        let result = detector.analyze(&samples);
        assert!(result.impulse_count >= 1);
    }

    #[test]
    fn test_analyze_periodic() {
        let detector = JitterDetector::with_defaults();
        let samples: Vec<MotionSample> = (0..60)
            .map(|i| {
                let t = i as f64 / 30.0;
                MotionSample::new((t * 6.0 * std::f64::consts::PI).sin() * 2.0, 0.0, t)
            })
            .collect();
        let result = detector.analyze(&samples);
        assert!(result.dominant_frequency.is_some());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(JitterSeverity::Negligible < JitterSeverity::Mild);
        assert!(JitterSeverity::Mild < JitterSeverity::Moderate);
        assert!(JitterSeverity::Moderate < JitterSeverity::Severe);
        assert!(JitterSeverity::Severe < JitterSeverity::Extreme);
    }

    #[test]
    fn test_recommend_window_scales_with_severity() {
        let detector = JitterDetector::with_defaults();
        let w_mild = detector.recommend_window(&JitterType::HandShake, &JitterSeverity::Mild);
        let w_severe = detector.recommend_window(&JitterType::HandShake, &JitterSeverity::Severe);
        assert!(w_severe > w_mild);
    }

    #[test]
    fn test_default_config() {
        let config = JitterDetectorConfig::default();
        assert!((config.frame_rate - 30.0).abs() < 1e-10);
        assert_eq!(config.min_samples, 10);
    }

    #[test]
    fn test_classify_severity_levels() {
        let detector = JitterDetector::with_defaults();
        assert_eq!(
            detector.classify_severity(0.1, 0.2),
            JitterSeverity::Negligible
        );
        assert_eq!(detector.classify_severity(1.0, 1.5), JitterSeverity::Mild);
        assert_eq!(
            detector.classify_severity(5.0, 6.0),
            JitterSeverity::Moderate
        );
        assert_eq!(
            detector.classify_severity(15.0, 20.0),
            JitterSeverity::Severe
        );
        assert_eq!(
            detector.classify_severity(50.0, 60.0),
            JitterSeverity::Extreme
        );
    }
}
