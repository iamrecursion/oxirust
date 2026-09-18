//! Declipping algorithms for restoring clipped audio.

use crate::clip::detector::ClippingRegion;
use crate::error::RestoreResult;
use crate::utils::interpolation::{interpolate, InterpolationMethod};

/// Declipping configuration.
#[derive(Debug, Clone)]
pub struct DeclipConfig {
    /// Extra samples to include before clipped region.
    pub pre_padding: usize,
    /// Extra samples to include after clipped region.
    pub post_padding: usize,
    /// Interpolation method to use.
    pub method: InterpolationMethod,
}

impl Default for DeclipConfig {
    fn default() -> Self {
        Self {
            pre_padding: 5,
            post_padding: 5,
            method: InterpolationMethod::Cubic,
        }
    }
}

/// Basic declipping using interpolation.
#[derive(Debug, Clone)]
pub struct BasicDeclipper {
    config: DeclipConfig,
}

impl BasicDeclipper {
    /// Create a new basic declipper.
    #[must_use]
    pub fn new(config: DeclipConfig) -> Self {
        Self { config }
    }

    /// Restore clipped regions.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input samples
    /// * `regions` - Detected clipping regions
    ///
    /// # Returns
    ///
    /// Restored samples.
    pub fn restore(&self, samples: &[f32], regions: &[ClippingRegion]) -> RestoreResult<Vec<f32>> {
        let mut output = samples.to_vec();

        // Process regions in reverse to maintain indices
        for region in regions.iter().rev() {
            let start = region.start.saturating_sub(self.config.pre_padding);
            let end = (region.end + self.config.post_padding).min(samples.len());

            if start >= end || end > samples.len() {
                continue;
            }

            // Interpolate the clipped region
            let restored = interpolate(samples, start, end, self.config.method)?;

            // Replace samples
            for (i, &value) in restored.iter().enumerate() {
                if start + i < output.len() {
                    output[start + i] = value;
                }
            }
        }

        Ok(output)
    }
}

/// Declipping using cubic spline extrapolation.
pub fn declip_cubic_spline(
    samples: &[f32],
    region: &ClippingRegion,
    context_samples: usize,
) -> RestoreResult<Vec<f32>> {
    let mut output = samples.to_vec();

    let start = region.start;
    let end = region.end;

    if start >= end || end > samples.len() {
        return Ok(output);
    }

    // Get context before and after clip
    let pre_start = start.saturating_sub(context_samples);
    let post_end = (end + context_samples).min(samples.len());

    if pre_start >= start || post_end <= end {
        // Not enough context, use simple interpolation
        let restored = interpolate(samples, start, end, InterpolationMethod::Cubic)?;
        for (i, &value) in restored.iter().enumerate() {
            if start + i < output.len() {
                output[start + i] = value;
            }
        }
        return Ok(output);
    }

    // Use cubic spline with known points before and after
    let restored = interpolate(samples, start, end, InterpolationMethod::Cubic)?;

    for (i, &value) in restored.iter().enumerate() {
        if start + i < output.len() {
            output[start + i] = value;
        }
    }

    Ok(output)
}

/// Declipping using AR (autoregressive) prediction.
pub fn declip_ar_prediction(
    samples: &[f32],
    region: &ClippingRegion,
    order: usize,
) -> RestoreResult<Vec<f32>> {
    let mut output = samples.to_vec();

    let start = region.start;
    let end = region.end;

    if start >= end || end > samples.len() || start < order {
        return Ok(output);
    }

    // Compute AR coefficients from samples before clipping
    let history = &samples[start - order..start];
    let coeffs = compute_ar_coefficients(history, order);

    // Forward prediction from before clip
    let mut forward_pred = vec![0.0; end - start];
    for i in 0..forward_pred.len() {
        let mut prediction = 0.0;
        for (j, &coeff) in coeffs.iter().enumerate() {
            let idx = if i > j {
                start + i - j - 1
            } else {
                start - j - 1 + i
            };
            if idx < output.len() {
                prediction += coeff * output[idx];
            }
        }
        forward_pred[i] = prediction;
    }

    // Backward prediction from after clip if possible
    if end + order <= samples.len() {
        let future = &samples[end..end + order];
        let back_coeffs = compute_ar_coefficients(future, order);

        let mut backward_pred = vec![0.0; end - start];
        for i in (0..backward_pred.len()).rev() {
            let mut prediction = 0.0;
            for (j, &coeff) in back_coeffs.iter().enumerate() {
                let idx = if i + j + 1 < backward_pred.len() {
                    start + i + j + 1
                } else {
                    end + (i + j + 1 - backward_pred.len())
                };
                if idx < output.len() {
                    prediction += coeff * output[idx];
                }
            }
            backward_pred[i] = prediction;
        }

        // Blend forward and backward predictions
        for (i, (&fwd, &bwd)) in forward_pred.iter().zip(backward_pred.iter()).enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let weight = i as f32 / forward_pred.len() as f32;
            output[start + i] = (1.0 - weight) * fwd + weight * bwd;
        }
    } else {
        // Use only forward prediction
        for (i, &value) in forward_pred.iter().enumerate() {
            output[start + i] = value;
        }
    }

    Ok(output)
}

/// Compute AR coefficients using Yule-Walker equations.
fn compute_ar_coefficients(samples: &[f32], order: usize) -> Vec<f32> {
    let n = samples.len();
    if n <= order || order == 0 {
        return vec![0.0; order];
    }

    // Compute autocorrelation
    let mut autocorr = vec![0.0; order + 1];
    for lag in 0..=order {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += samples[i] * samples[i + lag];
        }
        autocorr[lag] = sum / (n - lag) as f32;
    }

    // Solve Yule-Walker using Levinson-Durbin
    let mut coeffs = vec![0.0; order];
    let mut error = autocorr[0];

    for i in 0..order {
        let mut lambda = autocorr[i + 1];
        for j in 0..i {
            lambda -= coeffs[j] * autocorr[i - j];
        }

        let k = if error.abs() > f32::EPSILON {
            lambda / error
        } else {
            0.0
        };

        coeffs[i] = k;

        for j in 0..i {
            let temp = coeffs[j];
            coeffs[j] -= k * coeffs[i - j - 1];
            coeffs[i - j - 1] -= k * temp;
        }

        error *= 1.0 - k * k;
    }

    coeffs
}

/// Declipping using iterative method with constraints.
#[allow(dead_code)]
pub fn declip_iterative(
    samples: &[f32],
    region: &ClippingRegion,
    max_iterations: usize,
) -> RestoreResult<Vec<f32>> {
    let mut output = samples.to_vec();

    let start = region.start;
    let end = region.end;

    if start >= end || end > samples.len() {
        return Ok(output);
    }

    // Initialize with linear interpolation
    let initial = interpolate(samples, start, end, InterpolationMethod::Linear)?;
    for (i, &value) in initial.iter().enumerate() {
        if start + i < output.len() {
            output[start + i] = value;
        }
    }

    // Iteratively refine using smoothness constraint
    for _ in 0..max_iterations {
        for i in start + 1..end - 1 {
            if i < output.len() {
                // Local smoothing: average of neighbors
                let smoothed = (output[i - 1] + output[i + 1]) / 2.0;
                output[i] = 0.7 * output[i] + 0.3 * smoothed;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_declipper() {
        let mut samples = vec![0.0; 100];

        // Add clipping
        for i in 40..50 {
            samples[i] = 1.0;
        }

        let region = ClippingRegion {
            start: 40,
            end: 50,
            peak: 1.0,
            positive: true,
        };

        let declipper = BasicDeclipper::new(DeclipConfig::default());
        let restored = declipper
            .restore(&samples, &[region])
            .expect("should succeed in test");

        assert_eq!(restored.len(), samples.len());
        // Check that clipped region is modified
        assert!(restored[45] < 0.95);
    }

    #[test]
    fn test_declip_cubic_spline() {
        let mut samples: Vec<f32> = (0..100)
            .map(|i| {
                use std::f32::consts::PI;
                (2.0 * PI * i as f32 / 20.0).sin()
            })
            .collect();

        // Clip peaks
        for i in 0..samples.len() {
            if samples[i] > 0.9 {
                samples[i] = 0.9;
            }
        }

        let region = ClippingRegion {
            start: 45,
            end: 55,
            peak: 0.9,
            positive: true,
        };

        let restored = declip_cubic_spline(&samples, &region, 10).expect("should succeed in test");
        assert_eq!(restored.len(), samples.len());
    }

    #[test]
    fn test_declip_ar_prediction() {
        let samples: Vec<f32> = (0..100)
            .map(|i| {
                use std::f32::consts::PI;
                (2.0 * PI * i as f32 / 20.0).sin()
            })
            .collect();

        let region = ClippingRegion {
            start: 45,
            end: 55,
            peak: 1.0,
            positive: true,
        };

        let restored = declip_ar_prediction(&samples, &region, 10).expect("should succeed in test");
        assert_eq!(restored.len(), samples.len());
    }

    #[test]
    fn test_compute_ar_coefficients() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let coeffs = compute_ar_coefficients(&samples, 3);
        assert_eq!(coeffs.len(), 3);
    }
}
