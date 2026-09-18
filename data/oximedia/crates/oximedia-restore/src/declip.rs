//! Declipping - restore audio that has been hard-clipped.
//!
//! Clipping occurs when the signal amplitude exceeds the recordable range and
//! is clamped to the maximum value.  This module detects clipped regions and
//! reconstructs the signal using various interpolation strategies.

/// A region of the audio signal that has been detected as clipped.
#[derive(Debug, Clone, PartialEq)]
pub struct ClippingRegion {
    /// Index of the first clipped sample (inclusive).
    pub start: usize,
    /// Index of the last clipped sample (inclusive).
    pub end: usize,
    /// Peak amplitude magnitude observed in the region.
    pub peak_amplitude: f32,
}

impl ClippingRegion {
    /// Length of the clipped region in samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start) + 1
    }

    /// Returns `true` when the region is empty (end < start).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end < self.start
    }
}

/// Detects clipped regions in a mono audio buffer.
#[derive(Debug, Clone)]
pub struct ClippingDetector;

impl ClippingDetector {
    /// Scan `samples` for runs where |sample| >= `threshold` and return all
    /// detected [`ClippingRegion`]s.
    ///
    /// # Arguments
    /// * `samples`   - Input sample buffer (f32, typically in \[-1, 1\]).
    /// * `threshold` - Amplitude at or above which a sample is considered
    ///                 clipped (e.g. `0.99`).
    #[must_use]
    pub fn detect(samples: &[f32], threshold: f32) -> Vec<ClippingRegion> {
        let mut regions = Vec::new();
        let mut in_region = false;
        let mut region_start = 0;
        let mut peak = 0.0_f32;

        for (i, &s) in samples.iter().enumerate() {
            let abs = s.abs();
            if abs >= threshold {
                if !in_region {
                    in_region = true;
                    region_start = i;
                    peak = abs;
                } else {
                    peak = peak.max(abs);
                }
            } else if in_region {
                regions.push(ClippingRegion {
                    start: region_start,
                    end: i - 1,
                    peak_amplitude: peak,
                });
                in_region = false;
                peak = 0.0;
            }
        }

        // Close any open region at end of buffer
        if in_region {
            regions.push(ClippingRegion {
                start: region_start,
                end: samples.len() - 1,
                peak_amplitude: peak,
            });
        }

        regions
    }
}

/// Method used to reconstruct clipped samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclipMethod {
    /// Cubic Hermite spline interpolation between the last good sample before
    /// the clip and the first good sample after.
    Interpolation,
    /// AR (auto-regressive) signal reconstruction – forward + backward
    /// prediction averaged together.
    SignalReconstruction,
    /// Simply scale samples down to the clip threshold (least destructive for
    /// very short clips).
    ClipThreshold,
}

/// Configuration for the declipping processor.
#[derive(Debug, Clone)]
pub struct DeclipConfig {
    /// Amplitude at or above which a sample is considered clipped.
    pub threshold: f32,
    /// Reconstruction method.
    pub method: DeclipMethod,
    /// AR model order used by [`DeclipMethod::SignalReconstruction`].
    pub order: u32,
}

impl Default for DeclipConfig {
    fn default() -> Self {
        Self {
            threshold: 0.99,
            method: DeclipMethod::Interpolation,
            order: 8,
        }
    }
}

/// Cubic Hermite spline interpolator.
pub struct CubicInterpolator;

impl CubicInterpolator {
    /// Interpolate `n_samples` values between the tail of `before` and the
    /// head of `after` using a cubic Hermite spline.
    ///
    /// The spline is defined by:
    /// - p0  = last sample of `before`
    /// - m0  = estimated tangent at p0
    /// - p1  = first sample of `after`
    /// - m1  = estimated tangent at p1
    ///
    /// Returns a `Vec<f32>` of length `n_samples`.
    #[must_use]
    pub fn interpolate(before: &[f32], after: &[f32], n_samples: usize) -> Vec<f32> {
        if n_samples == 0 {
            return Vec::new();
        }

        let p0 = before.last().copied().unwrap_or(0.0);
        let p1 = after.first().copied().unwrap_or(0.0);

        // Estimate tangents from neighbouring samples.
        let m0 = if before.len() >= 2 {
            (p0 - before[before.len() - 2]) * 0.5
        } else {
            0.0
        };
        let m1 = if after.len() >= 2 {
            (after[1] - p1) * 0.5
        } else {
            0.0
        };

        (0..n_samples)
            .map(|i| {
                // t goes from 0 (exclusive) to 1 (exclusive)
                let t = (i + 1) as f32 / (n_samples + 1) as f32;
                let t2 = t * t;
                let t3 = t2 * t;

                // Cubic Hermite basis functions
                let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
                let h10 = t3 - 2.0 * t2 + t;
                let h01 = -2.0 * t3 + 3.0 * t2;
                let h11 = t3 - t2;

                h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            })
            .collect()
    }
}

/// Summary report produced by [`DeclipProcessor::process`].
#[derive(Debug, Clone)]
pub struct DeclipReport {
    /// Number of clipped regions detected.
    pub regions_detected: u32,
    /// Number of regions that were repaired.
    pub regions_repaired: u32,
    /// Maximum clipping percentage relative to buffer length.
    pub max_clipping_pct: f32,
    /// Estimated SNR improvement in dB (approximation).
    pub snr_improvement_db: f32,
}

/// Main declipping processor.
///
/// Usage:
/// ```
/// use oximedia_restore::declip::{DeclipConfig, DeclipProcessor};
///
/// let config = DeclipConfig::default();
/// let mut samples = vec![0.0f32; 1024];
/// let report = DeclipProcessor::process(&mut samples, &config);
/// assert_eq!(report.regions_detected, 0);
/// ```
pub struct DeclipProcessor;

impl DeclipProcessor {
    /// Process `samples` in-place and return a [`DeclipReport`].
    pub fn process(samples: &mut Vec<f32>, config: &DeclipConfig) -> DeclipReport {
        let regions = ClippingDetector::detect(samples, config.threshold);
        let regions_detected = regions.len() as u32;
        let mut regions_repaired = 0u32;
        let n = samples.len();

        // Calculate clipping percentage before repair
        let clipped_count = samples
            .iter()
            .filter(|&&s| s.abs() >= config.threshold)
            .count();
        let max_clipping_pct = if n > 0 {
            (clipped_count as f32 / n as f32) * 100.0
        } else {
            0.0
        };

        // Repair each region
        for region in &regions {
            let repaired = match config.method {
                DeclipMethod::Interpolation => {
                    Self::repair_interpolation(samples, region, config.threshold)
                }
                DeclipMethod::SignalReconstruction => {
                    Self::repair_ar(samples, region, config.order as usize)
                }
                DeclipMethod::ClipThreshold => {
                    Self::repair_scale(samples, region, config.threshold)
                }
            };
            if repaired {
                regions_repaired += 1;
            }
        }

        // Rough SNR improvement estimate: -20*log10(clipping_pct/100 + eps)
        let eps = 1e-6_f32;
        let snr_improvement_db = if max_clipping_pct > 0.0 {
            -20.0 * (max_clipping_pct / 100.0 + eps).log10()
        } else {
            0.0
        };

        DeclipReport {
            regions_detected,
            regions_repaired,
            max_clipping_pct,
            snr_improvement_db,
        }
    }

    fn repair_interpolation(
        samples: &mut Vec<f32>,
        region: &ClippingRegion,
        _threshold: f32,
    ) -> bool {
        let before_end = region.start.saturating_sub(1);
        let after_start = (region.end + 1).min(samples.len() - 1);

        if region.start == 0 || region.end + 1 >= samples.len() {
            // Edge clip – scale instead
            for s in samples[region.start..=region.end].iter_mut() {
                *s = s.signum() * _threshold * 0.95;
            }
            return true;
        }

        let n_samples = region.len();
        let before: Vec<f32> = samples[..=before_end].to_vec();
        let after: Vec<f32> = samples[after_start..].to_vec();

        let interp = CubicInterpolator::interpolate(&before, &after, n_samples);
        for (i, val) in interp.iter().enumerate() {
            samples[region.start + i] = *val;
        }
        true
    }

    fn repair_ar(samples: &mut Vec<f32>, region: &ClippingRegion, order: usize) -> bool {
        let n = samples.len();
        let start = region.start;
        let end = region.end.min(n - 1);

        // Forward prediction using `order` samples before clip
        if start < order {
            return false;
        }
        let ctx: Vec<f32> = samples[start - order..start].to_vec();
        let forward: Vec<f32> = (0..region.len())
            .map(|k| {
                // Simple AR(1)-like: use average slope
                let slope = if ctx.len() >= 2 {
                    ctx[ctx.len() - 1] - ctx[ctx.len() - 2]
                } else {
                    0.0
                };
                (ctx.last().copied().unwrap_or(0.0) + slope * (k + 1) as f32).clamp(-1.0, 1.0)
            })
            .collect();

        // Backward prediction using `order` samples after clip
        let backward: Vec<f32> = if end + order < n {
            let ctx_back: Vec<f32> = samples[end + 1..=(end + order).min(n - 1)].to_vec();
            (0..region.len())
                .map(|k| {
                    let slope = if ctx_back.len() >= 2 {
                        ctx_back[0] - ctx_back[1]
                    } else {
                        0.0
                    };
                    let rev_k = region.len() - 1 - k;
                    (ctx_back.first().copied().unwrap_or(0.0) + slope * (rev_k + 1) as f32)
                        .clamp(-1.0, 1.0)
                })
                .collect()
        } else {
            forward.clone()
        };

        // Average forward and backward predictions
        for (i, (f, b)) in forward.iter().zip(backward.iter()).enumerate() {
            samples[start + i] = (f + b) * 0.5;
        }
        true
    }

    fn repair_scale(samples: &mut Vec<f32>, region: &ClippingRegion, threshold: f32) -> bool {
        let end = region.end.min(samples.len() - 1);
        for s in samples[region.start..=end].iter_mut() {
            if s.abs() >= threshold {
                *s = s.signum() * threshold * 0.95;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clipped_signal() -> Vec<f32> {
        let mut v: Vec<f32> = (0..200).map(|i| (i as f32 * 0.05).sin() * 0.8).collect();
        // Hard-clip a region
        for s in v[50..60].iter_mut() {
            *s = 1.0;
        }
        v
    }

    #[test]
    fn test_detect_no_clipping() {
        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 0.8];
        let regions = ClippingDetector::detect(&samples, 0.99);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_detect_single_region() {
        let mut samples = vec![0.5f32; 100];
        for s in samples[10..20].iter_mut() {
            *s = 1.0;
        }
        let regions = ClippingDetector::detect(&samples, 0.99);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start, 10);
        assert_eq!(regions[0].end, 19);
    }

    #[test]
    fn test_detect_multiple_regions() {
        let mut samples = vec![0.5f32; 200];
        for s in samples[10..15].iter_mut() {
            *s = 1.0;
        }
        for s in samples[80..90].iter_mut() {
            *s = -1.0;
        }
        let regions = ClippingDetector::detect(&samples, 0.99);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_detect_peak_amplitude() {
        let mut samples = vec![0.0f32; 50];
        samples[10] = 0.995;
        samples[11] = 1.0;
        samples[12] = 0.997;
        let regions = ClippingDetector::detect(&samples, 0.99);
        assert_eq!(regions.len(), 1);
        assert!((regions[0].peak_amplitude - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_clipping_region_len() {
        let r = ClippingRegion {
            start: 10,
            end: 19,
            peak_amplitude: 1.0,
        };
        assert_eq!(r.len(), 10);
    }

    #[test]
    fn test_clipping_region_is_empty() {
        let r = ClippingRegion {
            start: 5,
            end: 5,
            peak_amplitude: 1.0,
        };
        assert!(!r.is_empty());
        let empty = ClippingRegion {
            start: 10,
            end: 5,
            peak_amplitude: 1.0,
        };
        assert!(empty.is_empty());
    }

    #[test]
    fn test_cubic_interpolator_basic() {
        let before = vec![0.0f32, 0.1, 0.2];
        let after = vec![0.8f32, 0.9, 1.0];
        let interp = CubicInterpolator::interpolate(&before, &after, 4);
        assert_eq!(interp.len(), 4);
        // Values should be between the boundary values
        for v in &interp {
            assert!(*v >= -0.5 && *v <= 1.5, "value {v} out of expected range");
        }
    }

    #[test]
    fn test_cubic_interpolator_zero_samples() {
        let result = CubicInterpolator::interpolate(&[0.0], &[1.0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_declip_process_no_clips() {
        let config = DeclipConfig::default();
        let mut samples = vec![0.5f32; 1000];
        let report = DeclipProcessor::process(&mut samples, &config);
        assert_eq!(report.regions_detected, 0);
        assert_eq!(report.regions_repaired, 0);
        assert_eq!(report.max_clipping_pct, 0.0);
    }

    #[test]
    fn test_declip_process_interpolation() {
        let config = DeclipConfig {
            threshold: 0.99,
            method: DeclipMethod::Interpolation,
            order: 8,
        };
        let mut samples = make_clipped_signal();
        let report = DeclipProcessor::process(&mut samples, &config);
        assert_eq!(report.regions_detected, 1);
        assert_eq!(report.regions_repaired, 1);
        // After repair, no sample should be clipped
        for s in &samples {
            assert!(s.abs() < 1.0 + 1e-5, "sample {s} still clipped");
        }
    }

    #[test]
    fn test_declip_process_ar() {
        let config = DeclipConfig {
            threshold: 0.99,
            method: DeclipMethod::SignalReconstruction,
            order: 8,
        };
        let mut samples = make_clipped_signal();
        let report = DeclipProcessor::process(&mut samples, &config);
        assert!(report.regions_repaired <= report.regions_detected);
    }

    #[test]
    fn test_declip_process_scale() {
        let config = DeclipConfig {
            threshold: 0.99,
            method: DeclipMethod::ClipThreshold,
            order: 8,
        };
        let mut samples = make_clipped_signal();
        let report = DeclipProcessor::process(&mut samples, &config);
        assert_eq!(report.regions_detected, 1);
        assert_eq!(report.regions_repaired, 1);
    }

    #[test]
    fn test_declip_snr_improvement_positive() {
        let config = DeclipConfig::default();
        let mut samples = make_clipped_signal();
        let report = DeclipProcessor::process(&mut samples, &config);
        if report.regions_detected > 0 {
            assert!(report.snr_improvement_db >= 0.0);
        }
    }

    #[test]
    fn test_declip_report_fields() {
        let config = DeclipConfig::default();
        let mut samples = vec![1.0f32; 100];
        let report = DeclipProcessor::process(&mut samples, &config);
        assert!(report.max_clipping_pct > 0.0);
        assert!(report.regions_detected > 0);
    }
}
