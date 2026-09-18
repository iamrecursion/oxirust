//! Clipping detection.

/// Clipping detection result.
#[derive(Debug, Clone)]
pub struct ClippingResult {
    /// Whether clipping is detected
    pub has_clipping: bool,
    /// Number of clipped samples
    pub clipped_samples: usize,
    /// Ratio of clipped samples to total samples
    pub clipping_ratio: f32,
    /// Maximum consecutive clipped samples
    pub max_consecutive_clipped: usize,
}

/// Detect clipping in audio samples.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `threshold` - Clipping threshold (typically 0.99 for digital audio)
///
/// # Returns
/// Clipping detection result
#[must_use]
pub fn detect_clipping(samples: &[f32], threshold: f32) -> ClippingResult {
    if samples.is_empty() {
        return ClippingResult {
            has_clipping: false,
            clipped_samples: 0,
            clipping_ratio: 0.0,
            max_consecutive_clipped: 0,
        };
    }

    let mut clipped_samples = 0;
    let mut consecutive = 0;
    let mut max_consecutive = 0;

    for &sample in samples {
        if sample.abs() >= threshold {
            clipped_samples += 1;
            consecutive += 1;
            max_consecutive = max_consecutive.max(consecutive);
        } else {
            consecutive = 0;
        }
    }

    let clipping_ratio = clipped_samples as f32 / samples.len() as f32;

    ClippingResult {
        has_clipping: clipped_samples > 0,
        clipped_samples,
        clipping_ratio,
        max_consecutive_clipped: max_consecutive,
    }
}

/// Detect clipping with hysteresis (more robust to noise).
#[must_use]
pub fn detect_clipping_with_hysteresis(
    samples: &[f32],
    threshold_high: f32,
    threshold_low: f32,
) -> ClippingResult {
    if samples.is_empty() {
        return ClippingResult {
            has_clipping: false,
            clipped_samples: 0,
            clipping_ratio: 0.0,
            max_consecutive_clipped: 0,
        };
    }

    let mut clipped_samples = 0;
    let mut consecutive = 0;
    let mut max_consecutive = 0;
    let mut in_clip = false;

    for &sample in samples {
        let abs_sample = sample.abs();

        if !in_clip && abs_sample >= threshold_high {
            in_clip = true;
            clipped_samples += 1;
            consecutive += 1;
        } else if in_clip {
            clipped_samples += 1;
            consecutive += 1;

            if abs_sample < threshold_low {
                in_clip = false;
                max_consecutive = max_consecutive.max(consecutive);
                consecutive = 0;
            }
        }
    }

    if in_clip {
        max_consecutive = max_consecutive.max(consecutive);
    }

    let clipping_ratio = clipped_samples as f32 / samples.len() as f32;

    ClippingResult {
        has_clipping: clipped_samples > 0,
        clipped_samples,
        clipping_ratio,
        max_consecutive_clipped: max_consecutive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipping_detection() {
        // No clipping
        let samples = vec![0.0, 0.5, -0.5, 0.8, -0.8];
        let result = detect_clipping(&samples, 0.99);
        assert!(!result.has_clipping);

        // With clipping
        let clipped = vec![0.0, 1.0, 1.0, -1.0, 0.5];
        let result = detect_clipping(&clipped, 0.99);
        assert!(result.has_clipping);
        assert!(result.clipped_samples >= 2); // At least the 1.0 and -1.0 values
    }

    #[test]
    fn test_clipping_with_hysteresis() {
        let samples = vec![0.0, 0.95, 1.0, 0.98, 0.9, 0.5];
        let result = detect_clipping_with_hysteresis(&samples, 0.99, 0.9);

        assert!(result.has_clipping);
    }
}
