//! Clipping detection.

use crate::error::RestoreResult;

/// Clipping detection configuration.
#[derive(Debug, Clone)]
pub struct ClipDetectorConfig {
    /// Threshold for clipping detection (0.0 to 1.0).
    pub threshold: f32,
    /// Minimum consecutive clipped samples.
    pub min_consecutive: usize,
}

impl Default for ClipDetectorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.99,
            min_consecutive: 2,
        }
    }
}

/// Detected clipping region.
#[derive(Debug, Clone)]
pub struct ClippingRegion {
    /// Start sample index.
    pub start: usize,
    /// End sample index (exclusive).
    pub end: usize,
    /// Peak value (positive or negative).
    pub peak: f32,
    /// Direction of clipping (true = positive, false = negative).
    pub positive: bool,
}

/// Clipping detector.
#[derive(Debug, Clone)]
pub struct ClipDetector {
    config: ClipDetectorConfig,
}

impl ClipDetector {
    /// Create a new clipping detector.
    #[must_use]
    pub fn new(config: ClipDetectorConfig) -> Self {
        Self { config }
    }

    /// Detect clipping in samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    ///
    /// # Returns
    ///
    /// List of detected clipping regions.
    pub fn detect(&self, samples: &[f32]) -> RestoreResult<Vec<ClippingRegion>> {
        let mut regions = Vec::new();

        if samples.is_empty() {
            return Ok(regions);
        }

        let mut in_clip = false;
        let mut clip_start = 0;
        let mut clip_positive = true;
        let mut clip_peak = 0.0;
        let mut consecutive_count = 0;

        for (i, &sample) in samples.iter().enumerate() {
            let is_clipped = sample.abs() >= self.config.threshold;
            let is_positive = sample > 0.0;

            if is_clipped {
                if !in_clip {
                    // Potential start of clipping
                    clip_start = i;
                    clip_positive = is_positive;
                    clip_peak = sample;
                    consecutive_count = 1;
                    in_clip = true;
                } else if is_positive == clip_positive {
                    // Continue clipping in same direction
                    consecutive_count += 1;
                    clip_peak = if sample.abs() > clip_peak.abs() {
                        sample
                    } else {
                        clip_peak
                    };
                } else {
                    // Direction changed - end previous clip if valid
                    if consecutive_count >= self.config.min_consecutive {
                        regions.push(ClippingRegion {
                            start: clip_start,
                            end: i,
                            peak: clip_peak,
                            positive: clip_positive,
                        });
                    }

                    // Start new clip
                    clip_start = i;
                    clip_positive = is_positive;
                    clip_peak = sample;
                    consecutive_count = 1;
                }
            } else if in_clip {
                // End of clipping
                if consecutive_count >= self.config.min_consecutive {
                    regions.push(ClippingRegion {
                        start: clip_start,
                        end: i,
                        peak: clip_peak,
                        positive: clip_positive,
                    });
                }
                in_clip = false;
                consecutive_count = 0;
            }
        }

        // Handle clipping at end
        if in_clip && consecutive_count >= self.config.min_consecutive {
            regions.push(ClippingRegion {
                start: clip_start,
                end: samples.len(),
                peak: clip_peak,
                positive: clip_positive,
            });
        }

        Ok(regions)
    }

    /// Estimate clipping severity.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    ///
    /// # Returns
    ///
    /// Clipping severity ratio (0.0 = no clipping, 1.0 = fully clipped).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_severity(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let clipped_count = samples
            .iter()
            .filter(|&&s| s.abs() >= self.config.threshold)
            .count();

        clipped_count as f32 / samples.len() as f32
    }
}

/// Detect clipping using derivative analysis.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `threshold` - Clipping threshold
///
/// # Returns
///
/// List of detected clipping regions.
#[must_use]
pub fn detect_clipping_derivative(samples: &[f32], threshold: f32) -> Vec<ClippingRegion> {
    let mut regions = Vec::new();

    if samples.len() < 3 {
        return regions;
    }

    let mut in_clip = false;
    let mut clip_start = 0;
    let mut clip_positive = true;
    let mut clip_peak = 0.0;

    for i in 1..samples.len() - 1 {
        let is_clipped = samples[i].abs() >= threshold;
        let derivative = (samples[i + 1] - samples[i - 1]) / 2.0;
        let is_flat = derivative.abs() < 0.001; // Very small derivative indicates flat clip

        if is_clipped && is_flat {
            if !in_clip {
                clip_start = i;
                clip_positive = samples[i] > 0.0;
                clip_peak = samples[i];
                in_clip = true;
            } else {
                clip_peak = if samples[i].abs() > clip_peak.abs() {
                    samples[i]
                } else {
                    clip_peak
                };
            }
        } else if in_clip {
            regions.push(ClippingRegion {
                start: clip_start,
                end: i,
                peak: clip_peak,
                positive: clip_positive,
            });
            in_clip = false;
        }
    }

    if in_clip {
        regions.push(ClippingRegion {
            start: clip_start,
            end: samples.len(),
            peak: clip_peak,
            positive: clip_positive,
        });
    }

    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_detector() {
        let mut samples = vec![0.0; 100];

        // Add clipping at peak
        for i in 40..50 {
            samples[i] = 1.0;
        }

        let detector = ClipDetector::new(ClipDetectorConfig::default());
        let regions = detector.detect(&samples).expect("should succeed in test");

        assert!(!regions.is_empty());
        assert!(regions[0].positive);
        assert!((regions[0].peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_clipping() {
        let mut samples = vec![0.0; 100];

        for i in 40..50 {
            samples[i] = -0.995;
        }

        let detector = ClipDetector::new(ClipDetectorConfig::default());
        let regions = detector.detect(&samples).expect("should succeed in test");

        assert!(!regions.is_empty());
        assert!(!regions[0].positive);
    }

    #[test]
    fn test_estimate_severity() {
        let mut samples = vec![0.0; 100];
        for i in 0..50 {
            samples[i] = 1.0;
        }

        let detector = ClipDetector::new(ClipDetectorConfig::default());
        let severity = detector.estimate_severity(&samples);

        assert!((severity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_detect_clipping_derivative() {
        let mut samples = vec![0.0; 100];

        // Create flat clipping region
        for i in 40..50 {
            samples[i] = 1.0;
        }

        let regions = detect_clipping_derivative(&samples, 0.99);
        assert!(!regions.is_empty());
    }

    #[test]
    fn test_config_default() {
        let config = ClipDetectorConfig::default();
        assert!(config.threshold > 0.9);
        assert!(config.min_consecutive > 0);
    }
}
