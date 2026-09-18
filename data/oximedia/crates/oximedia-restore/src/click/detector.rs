//! Click and pop detection.

use crate::error::RestoreResult;

/// Click detection configuration.
#[derive(Debug, Clone)]
pub struct ClickDetectorConfig {
    /// Detection sensitivity (0.0 = low, 1.0 = high).
    pub sensitivity: f32,
    /// Minimum click duration in samples.
    pub min_duration: usize,
    /// Maximum click duration in samples.
    pub max_duration: usize,
    /// Threshold multiplier for detection.
    pub threshold_multiplier: f32,
}

impl Default for ClickDetectorConfig {
    fn default() -> Self {
        Self {
            sensitivity: 0.5,
            min_duration: 1,
            max_duration: 100,
            threshold_multiplier: 3.0,
        }
    }
}

/// Severity classification of a detected click or pop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClickSeverity {
    /// Low-level transient, barely above threshold.
    Low,
    /// Moderate click, clearly audible.
    Medium,
    /// Heavy pop or loud impulse artifact.
    High,
}

impl ClickSeverity {
    /// Derive severity from normalised magnitude relative to the detection threshold.
    ///
    /// * `< 2×` threshold ratio → [`Low`](ClickSeverity::Low)
    /// * `2× – 5×` → [`Medium`](ClickSeverity::Medium)
    /// * `> 5×` → [`High`](ClickSeverity::High)
    #[must_use]
    pub fn from_ratio(magnitude_to_threshold_ratio: f32) -> Self {
        if magnitude_to_threshold_ratio >= 5.0 {
            Self::High
        } else if magnitude_to_threshold_ratio >= 2.0 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// Detected click or pop.
#[derive(Debug, Clone)]
pub struct Click {
    /// Start sample index.
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
    /// Peak magnitude of the difference signal at detection.
    pub magnitude: f32,
    /// Severity classification derived from magnitude vs threshold ratio.
    pub severity: ClickSeverity,
    /// Confidence score in [0.0, 1.0]: how certain the detector is this is a real click.
    ///
    /// Computed from the ratio of the peak magnitude to the adaptive threshold and
    /// normalised by the configured `threshold_multiplier`.
    pub confidence: f32,
}

/// Click detector.
#[derive(Debug, Clone)]
pub struct ClickDetector {
    config: ClickDetectorConfig,
}

impl ClickDetector {
    /// Create a new click detector.
    #[must_use]
    pub fn new(config: ClickDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect clicks in samples.
    ///
    /// Each returned [`Click`] is annotated with [`ClickSeverity`] and a `confidence`
    /// score in \[0, 1\] so callers can triage and prioritise restoration effort.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    ///
    /// # Returns
    ///
    /// List of detected clicks with severity and confidence.
    pub fn detect(&self, samples: &[f32]) -> RestoreResult<Vec<Click>> {
        if samples.len() < 3 {
            return Ok(Vec::new());
        }

        // Compute first-order difference to highlight transients
        let mut diff: Vec<f32> = Vec::with_capacity(samples.len() - 1);
        for i in 1..samples.len() {
            diff.push((samples[i] - samples[i - 1]).abs());
        }

        // Compute adaptive threshold based on median + MAD
        let threshold = self.compute_threshold(&diff);

        // Find regions above threshold
        let mut clicks = Vec::new();
        let mut in_click = false;
        let mut click_start = 0;
        let mut peak_magnitude = 0.0f32;

        for (i, &d) in diff.iter().enumerate() {
            if !in_click && d > threshold {
                // Start of click
                in_click = true;
                click_start = i;
                peak_magnitude = d;
            } else if in_click {
                if d > threshold {
                    // Continue click
                    peak_magnitude = peak_magnitude.max(d);
                } else {
                    // End of click
                    let duration = i - click_start;
                    if duration >= self.config.min_duration && duration <= self.config.max_duration
                    {
                        let (severity, confidence) =
                            self.compute_severity_confidence(peak_magnitude, threshold);
                        clicks.push(Click {
                            start: click_start,
                            end: i,
                            magnitude: peak_magnitude,
                            severity,
                            confidence,
                        });
                    }
                    in_click = false;
                    peak_magnitude = 0.0;
                }
            }
        }

        // Handle click at end of buffer
        if in_click {
            let duration = diff.len() - click_start;
            if duration >= self.config.min_duration && duration <= self.config.max_duration {
                let (severity, confidence) =
                    self.compute_severity_confidence(peak_magnitude, threshold);
                clicks.push(Click {
                    start: click_start,
                    end: diff.len(),
                    magnitude: peak_magnitude,
                    severity,
                    confidence,
                });
            }
        }

        Ok(clicks)
    }

    /// Compute severity and confidence for a detected click.
    ///
    /// `confidence` is the clamped ratio of `peak_magnitude / threshold`, normalised
    /// to the range \[0, 1\] by considering the `threshold_multiplier` as an upper
    /// reference (ratio ≥ `threshold_multiplier × 3` → confidence 1.0).
    fn compute_severity_confidence(
        &self,
        peak_magnitude: f32,
        threshold: f32,
    ) -> (ClickSeverity, f32) {
        let ratio = if threshold > f32::EPSILON {
            peak_magnitude / threshold
        } else {
            1.0
        };

        let severity = ClickSeverity::from_ratio(ratio);

        // Confidence: ratio of 1.0 (just at threshold) → 0.0 confidence;
        // ratio of `threshold_multiplier * 3` or above → 1.0 confidence.
        let upper_ref = self.config.threshold_multiplier * 3.0;
        let confidence = ((ratio - 1.0) / (upper_ref - 1.0).max(f32::EPSILON)).clamp(0.0, 1.0);

        (severity, confidence)
    }

    /// Compute adaptive threshold using median-based method.
    fn compute_threshold(&self, diff: &[f32]) -> f32 {
        if diff.is_empty() {
            return 0.0;
        }

        // Compute median
        let mut sorted = diff.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Compute MAD (Median Absolute Deviation)
        let mut deviations: Vec<f32> = diff.iter().map(|&d| (d - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if deviations.len() % 2 == 0 {
            (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
        } else {
            deviations[deviations.len() / 2]
        };

        // Threshold = median + multiplier * MAD
        let base_threshold = median + self.config.threshold_multiplier * mad;

        // Adjust for sensitivity
        base_threshold * (1.0 - self.config.sensitivity * 0.5)
    }
}

/// Detect clicks using a simple energy-based method.
///
/// Severity and confidence are set to [`ClickSeverity::Medium`] / `0.5` by default
/// since this simpler method does not have access to an adaptive threshold for
/// precise scoring.  Callers that need accurate severity should use [`ClickDetector`].
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `threshold` - Energy threshold
/// * `window_size` - Window size for energy computation
///
/// # Returns
///
/// List of detected clicks.
#[must_use]
pub fn detect_clicks_simple(samples: &[f32], threshold: f32, window_size: usize) -> Vec<Click> {
    if samples.len() < window_size {
        return Vec::new();
    }

    let mut clicks: Vec<Click> = Vec::new();
    let half_window = window_size / 2;

    for i in half_window..samples.len() - half_window {
        // Compute local energy
        let start = i.saturating_sub(half_window);
        let end = (i + half_window).min(samples.len());

        let energy: f32 = samples[start..end].iter().map(|&s| s * s).sum();
        let avg_energy = energy / (end - start) as f32;

        if avg_energy > threshold {
            // Check if this is a new click or part of existing one
            if let Some(last_click) = clicks.last_mut() {
                if i <= last_click.end + window_size {
                    // Extend existing click
                    last_click.end = i + 1;
                    last_click.magnitude = last_click.magnitude.max(avg_energy);
                    continue;
                }
            }

            // Simple confidence: clamped ratio of energy to threshold
            let ratio = avg_energy / threshold.max(f32::EPSILON);
            let confidence = ((ratio - 1.0) / 9.0).clamp(0.0, 1.0);
            let severity = ClickSeverity::from_ratio(ratio);

            // New click
            clicks.push(Click {
                start: i.saturating_sub(window_size),
                end: i + window_size,
                magnitude: avg_energy,
                severity,
                confidence,
            });
        }
    }

    clicks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_click_detector() {
        let mut samples = vec![0.0; 1000];

        // Add some clicks
        samples[100] = 1.0;
        samples[101] = -0.8;
        samples[500] = 0.9;
        samples[501] = -0.7;

        let detector = ClickDetector::new(ClickDetectorConfig::default());
        let clicks = detector.detect(&samples).expect("should succeed in test");

        assert!(!clicks.is_empty());
    }

    #[test]
    fn test_click_severity_and_confidence_populated() {
        let mut samples = vec![0.0f32; 1000];
        // Large amplitude impulse — should produce high severity / confidence
        samples[300] = 5.0;
        samples[301] = -5.0;

        let detector = ClickDetector::new(ClickDetectorConfig::default());
        let clicks = detector.detect(&samples).expect("should succeed in test");

        assert!(!clicks.is_empty(), "impulse should be detected");
        for click in &clicks {
            assert!(
                click.confidence >= 0.0 && click.confidence <= 1.0,
                "confidence out of range: {}",
                click.confidence
            );
        }
    }

    #[test]
    fn test_click_severity_levels() {
        assert_eq!(ClickSeverity::from_ratio(1.5), ClickSeverity::Low);
        assert_eq!(ClickSeverity::from_ratio(3.0), ClickSeverity::Medium);
        assert_eq!(ClickSeverity::from_ratio(6.0), ClickSeverity::High);
    }

    #[test]
    fn test_detect_clicks_simple() {
        let mut samples = vec![0.0; 1000];

        // Add impulse
        samples[500] = 1.0;

        let clicks = detect_clicks_simple(&samples, 0.1, 5);
        assert!(!clicks.is_empty());
        // Severity and confidence must be valid
        for c in &clicks {
            assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
        }
    }

    #[test]
    fn test_config_default() {
        let config = ClickDetectorConfig::default();
        assert_eq!(config.sensitivity, 0.5);
        assert!(config.min_duration > 0);
        assert!(config.max_duration > config.min_duration);
    }
}
