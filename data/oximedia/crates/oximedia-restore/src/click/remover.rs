//! Click and pop removal using interpolation.

use crate::click::detector::Click;
use crate::error::RestoreResult;
use crate::utils::interpolation::{interpolate, InterpolationMethod};

/// Click remover.
#[derive(Debug, Clone)]
pub struct ClickRemover {
    method: InterpolationMethod,
    padding: usize,
}

impl ClickRemover {
    /// Create a new click remover.
    ///
    /// # Arguments
    ///
    /// * `method` - Interpolation method to use
    /// * `padding` - Extra samples to include on each side
    #[must_use]
    pub fn new(method: InterpolationMethod, padding: usize) -> Self {
        Self { method, padding }
    }

    /// Remove clicks from samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    /// * `clicks` - Detected clicks to remove
    ///
    /// # Returns
    ///
    /// Samples with clicks removed.
    pub fn remove(&self, samples: &[f32], clicks: &[Click]) -> RestoreResult<Vec<f32>> {
        let mut output = samples.to_vec();

        // Process clicks in reverse order to maintain indices
        for click in clicks.iter().rev() {
            let start = click.start.saturating_sub(self.padding);
            let end = (click.end + self.padding).min(samples.len());

            if start >= end || end > samples.len() {
                continue;
            }

            // Interpolate the click region
            let interpolated = interpolate(samples, start, end, self.method)?;

            // Replace samples
            for (i, &value) in interpolated.iter().enumerate() {
                if start + i < output.len() {
                    output[start + i] = value;
                }
            }
        }

        Ok(output)
    }

    /// Remove single click.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    /// * `click` - Click to remove
    ///
    /// # Returns
    ///
    /// Samples with click removed.
    pub fn remove_single(&self, samples: &[f32], click: &Click) -> RestoreResult<Vec<f32>> {
        self.remove(samples, std::slice::from_ref(click))
    }

    /// Remove clicks from samples **in-place**.
    ///
    /// Unlike [`remove`][Self::remove] which allocates a full output buffer,
    /// this method operates directly on `&mut [f32]` using a single small
    /// allocation for the sorted click index.  The interpolated values replace
    /// the corrupted samples without copying the entire buffer.
    ///
    /// Click regions are processed in ascending order; each region is
    /// interpolated with cubic Hermite blending between the two boundary
    /// samples immediately outside the region.
    ///
    /// # Arguments
    ///
    /// * `samples` - Mutable sample buffer; modified in place.
    /// * `clicks` - Detected clicks to remove.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::RestoreError`] on internal failures (e.g. from the underlying
    /// interpolation call).
    pub fn remove_in_place(&self, samples: &mut [f32], clicks: &[Click]) -> RestoreResult<()> {
        if clicks.is_empty() || samples.is_empty() {
            return Ok(());
        }

        // One allocation: a sorted copy of the click list (indices only).
        let mut sorted: Vec<&Click> = clicks.iter().collect();
        sorted.sort_by_key(|c| c.start);

        for click in sorted {
            let start = click.start.saturating_sub(self.padding);
            let end = (click.end + self.padding).min(samples.len());

            if start >= end || end > samples.len() {
                continue;
            }

            // We need the boundary values *before* we mutate the region.
            // Read from the surrounding samples (not inside the click region).
            let left = if start > 0 { samples[start - 1] } else { 0.0 };
            let right = if end < samples.len() {
                samples[end]
            } else {
                0.0
            };

            let len = end - start;
            if len == 0 {
                continue;
            }

            // Cubic Hermite blend between left and right boundary values.
            // t runs from 1/(len+1) to len/(len+1) so the boundaries are not
            // overwritten (they were already good samples outside the click).
            #[allow(clippy::cast_precision_loss)]
            for i in 0..len {
                let t = (i + 1) as f32 / (len + 1) as f32;
                // Smoothstep: 3t²-2t³
                let smooth_t = t * t * (3.0 - 2.0 * t);
                samples[start + i] = left + smooth_t * (right - left);
            }
        }

        Ok(())
    }
}

impl Default for ClickRemover {
    fn default() -> Self {
        Self::new(InterpolationMethod::Cubic, 2)
    }
}

/// Remove click using autoregressive (AR) modeling.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `click` - Click to remove
/// * `order` - AR model order
///
/// # Returns
///
/// Samples with click removed using AR prediction.
pub fn remove_click_ar(samples: &[f32], click: &Click, order: usize) -> RestoreResult<Vec<f32>> {
    if click.start >= click.end || click.end > samples.len() {
        return Ok(samples.to_vec());
    }

    let mut output = samples.to_vec();

    // Get samples before click for AR modeling
    if click.start < order {
        // Not enough history, fall back to cubic interpolation
        return interpolate(samples, click.start, click.end, InterpolationMethod::Cubic);
    }

    // Compute AR coefficients using Burg's method
    let history = &samples[click.start - order..click.start];
    let coeffs = burg_ar(history, order);

    // Predict samples in click region
    for i in click.start..click.end {
        let mut prediction = 0.0;

        for (j, &coeff) in coeffs.iter().enumerate() {
            if i > j {
                prediction += coeff * output[i - j - 1];
            }
        }

        output[i] = prediction;
    }

    Ok(output)
}

/// Burg's algorithm for AR coefficient estimation.
fn burg_ar(samples: &[f32], order: usize) -> Vec<f32> {
    let n = samples.len();
    if n <= order || order == 0 {
        return vec![0.0; order];
    }

    let mut coeffs = vec![0.0; order];
    let mut forward = samples.to_vec();
    let mut backward = samples.to_vec();

    for m in 0..order {
        // Compute reflection coefficient
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in m + 1..n {
            numerator += forward[i] * backward[i - 1];
            denominator += forward[i] * forward[i] + backward[i - 1] * backward[i - 1];
        }

        let k = if denominator > f32::EPSILON {
            -2.0 * numerator / denominator
        } else {
            0.0
        };

        coeffs[m] = k;

        // Update AR coefficients
        for i in 0..m {
            let temp = coeffs[i];
            coeffs[i] += k * coeffs[m - i - 1];
            coeffs[m - i - 1] += k * temp;
        }

        // Update forward and backward predictions
        for i in m + 1..n {
            let temp_forward = forward[i];
            forward[i] += k * backward[i - 1];
            backward[i - 1] += k * temp_forward;
        }
    }

    coeffs
}

/// Remove click using median filtering.
///
/// # Arguments
///
/// * `samples` - Input samples
/// * `click` - Click to remove
/// * `window_size` - Median filter window size (should be odd)
///
/// # Returns
///
/// Samples with click removed.
pub fn remove_click_median(
    samples: &[f32],
    click: &Click,
    window_size: usize,
) -> RestoreResult<Vec<f32>> {
    if click.start >= click.end || click.end > samples.len() {
        return Ok(samples.to_vec());
    }

    let mut output = samples.to_vec();
    let half_window = window_size / 2;

    for i in click.start..click.end {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(samples.len());

        // Collect samples in window, excluding the click region
        let mut window_samples: Vec<f32> = samples[start..end]
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let sample_idx = start + idx;
                sample_idx < click.start || sample_idx >= click.end
            })
            .map(|(_, &s)| s)
            .collect();

        if window_samples.is_empty() {
            continue;
        }

        // Compute median
        window_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if window_samples.len() % 2 == 0 {
            (window_samples[window_samples.len() / 2 - 1]
                + window_samples[window_samples.len() / 2])
                / 2.0
        } else {
            window_samples[window_samples.len() / 2]
        };

        output[i] = median;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_click_remover() {
        let mut samples = vec![0.0; 100];
        samples[50] = 1.0; // Click

        let click = Click {
            start: 50,
            end: 51,
            magnitude: 1.0,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };

        let remover = ClickRemover::default();
        let restored = remover
            .remove(&samples, &[click])
            .expect("should succeed in test");

        assert_eq!(restored.len(), samples.len());
        assert!(restored[50].abs() < 0.5); // Click should be reduced
    }

    #[test]
    fn test_remove_click_ar() {
        let mut samples = vec![0.0; 100];
        for i in 0..100 {
            use std::f32::consts::PI;
            samples[i] = (2.0 * PI * i as f32 / 10.0).sin();
        }
        samples[50] = 2.0; // Click

        let click = Click {
            start: 50,
            end: 51,
            magnitude: 2.0,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };

        let restored = remove_click_ar(&samples, &click, 10).expect("should succeed in test");
        assert_eq!(restored.len(), samples.len());
        assert!(restored[50].abs() < 1.5);
    }

    #[test]
    fn test_remove_click_median() {
        let mut samples = vec![0.0; 100];
        samples[50] = 1.0; // Click

        let click = Click {
            start: 50,
            end: 51,
            magnitude: 1.0,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };

        let restored = remove_click_median(&samples, &click, 5).expect("should succeed in test");
        assert_eq!(restored.len(), samples.len());
        assert!(restored[50].abs() < 0.5);
    }

    #[test]
    fn test_burg_ar() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let coeffs = burg_ar(&samples, 2);
        assert_eq!(coeffs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // In-place click removal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_click_remover_in_place_basic() {
        use std::f32::consts::PI;

        // 1000-sample sine wave at a normalised frequency.
        let n = 1000usize;
        let mut samples: Vec<f32> = (0..n).map(|i| (2.0 * PI * i as f32 / 50.0).sin()).collect();

        // Snapshot of samples outside the corrupted region for later comparison.
        let before_prefix: Vec<f32> = samples[..95].to_vec();
        let before_suffix: Vec<f32> = samples[115..].to_vec();

        // Corrupt samples 100..110 with large values.
        let corruption = 100.0_f32;
        for s in &mut samples[100..110] {
            *s = corruption;
        }

        // Verify the corruption is visible before removal.
        let max_before: f32 = samples[100..110]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_before > 10.0,
            "corruption should be present before removal (max={max_before})"
        );

        let click = Click {
            start: 100,
            end: 110,
            magnitude: corruption,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };

        let remover = ClickRemover::new(InterpolationMethod::Cubic, 0);
        remover
            .remove_in_place(&mut samples, &[click])
            .expect("in-place removal should succeed");

        // After removal the region should be much smaller than the corruption.
        let max_after: f32 = samples[100..110]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_after < corruption / 2.0,
            "click region should be interpolated (max after = {max_after}, corruption = {corruption})"
        );

        // Samples *before* the click region must be unchanged.
        assert_eq!(
            &samples[..95],
            before_prefix.as_slice(),
            "samples before click region must be unchanged"
        );

        // Samples *after* the click region must be unchanged.
        assert_eq!(
            &samples[115..],
            before_suffix.as_slice(),
            "samples after click region must be unchanged"
        );
    }

    #[test]
    fn test_click_remover_in_place_empty_clicks() {
        // No clicks → buffer unchanged.
        let mut samples = vec![0.5f32; 100];
        let original = samples.clone();
        let remover = ClickRemover::default();
        remover
            .remove_in_place(&mut samples, &[])
            .expect("empty clicks should succeed");
        assert_eq!(samples, original, "no-op when clicks list is empty");
    }

    #[test]
    fn test_click_remover_in_place_at_boundaries() {
        // Click at the very start and very end should not panic.
        let mut samples = vec![1.0f32; 50];
        let click_start = Click {
            start: 0,
            end: 3,
            magnitude: 5.0,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };
        let click_end = Click {
            start: 47,
            end: 50,
            magnitude: 5.0,
            confidence: 1.0,
            severity: crate::click::detector::ClickSeverity::High,
        };
        let remover = ClickRemover::new(InterpolationMethod::Cubic, 0);
        remover
            .remove_in_place(&mut samples, &[click_start, click_end])
            .expect("boundary clicks should not panic");
        for (i, &s) in samples.iter().enumerate() {
            assert!(
                s.is_finite(),
                "sample {i} is not finite after in-place removal"
            );
        }
    }
}
