//! Click and pop removal for vinyl restoration workflows.
//!
//! Clicks and pops are impulsive noise artefacts characteristic of vinyl records
//! and old tape recordings. They appear as short-duration, high-amplitude spikes
//! that are easily distinguished from the underlying audio.
//!
//! ## Algorithm
//!
//! 1. **Detection** — A median-absolute-deviation (MAD) detector identifies
//!    suspected click positions. The local median of a short window is used as
//!    the "expected" signal; any sample that deviates by more than `threshold`
//!    times the MAD is flagged.
//!
//! 2. **Interpolation** — Flagged regions are replaced using linear interpolation
//!    between the last clean sample before the click and the first clean sample
//!    after it.  For wider pops a cubic spline interpolator preserves continuity
//!    of both value and slope.
//!
//! 3. **Suppression** — An optional short auto-regressive (AR) model fitted to
//!    the surrounding context is used for regions wider than `max_linear_span`
//!    samples. The AR coefficients are solved via a small Levinson–Durbin step.

#![allow(dead_code)]

/// Severity of a detected impulsive noise event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpulseKind {
    /// A short click (≤ `max_click_span` samples).
    Click,
    /// A wider pop (> `max_click_span` up to `max_pop_span` samples).
    Pop,
}

/// A detected click or pop region.
#[derive(Debug, Clone, Copy)]
pub struct ImpulseRegion {
    /// Sample index of the first affected sample (inclusive).
    pub start: usize,
    /// Sample index past the last affected sample (exclusive).
    pub end: usize,
    /// Classification.
    pub kind: ImpulseKind,
    /// Impulse magnitude: the region's peak absolute amplitude divided by the
    /// RMS of the adjacent clean context (how far above the local noise floor
    /// the click/pop rises). Falls back to the raw peak when no clean context
    /// is available.
    pub magnitude: f32,
}

/// Configuration for click/pop removal.
#[derive(Debug, Clone)]
pub struct ClickRemoverConfig {
    /// Detection threshold in multiples of the local MAD.
    /// Typical range 4.0–12.0; lower = more sensitive.
    pub threshold: f32,
    /// Window length (samples) used to estimate local statistics.
    /// Should be long enough to capture at least a few cycles at the
    /// lowest audio frequency of interest (e.g. 64 at 44.1 kHz).
    pub window: usize,
    /// Maximum span of a single click (samples). Wider artefacts are
    /// classified as pops.
    pub max_click_span: usize,
    /// Maximum span of a pop to attempt repair (samples). Wider regions
    /// are flagged but not repaired (returned in the report unchanged).
    pub max_pop_span: usize,
    /// Number of AR model coefficients for wide-pop interpolation.
    pub ar_order: usize,
}

impl Default for ClickRemoverConfig {
    fn default() -> Self {
        Self {
            threshold: 6.0,
            window: 64,
            max_click_span: 6,
            max_pop_span: 64,
            ar_order: 12,
        }
    }
}

/// Vinyl click and pop remover.
///
/// Apply with [`ClickRemover::process`] on mono f32 samples.
pub struct ClickRemover {
    config: ClickRemoverConfig,
}

impl ClickRemover {
    /// Create a new click remover with the given configuration.
    #[must_use]
    pub fn new(config: ClickRemoverConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(ClickRemoverConfig::default())
    }

    /// Detect and repair clicks/pops in `samples` in-place.
    ///
    /// Returns a list of detected and (attempted) repaired regions.
    pub fn process(&self, samples: &mut [f32]) -> Vec<ImpulseRegion> {
        let n = samples.len();
        if n < self.config.window * 2 {
            return Vec::new();
        }

        // Step 1: Detect candidate positions
        let flags = self.detect(samples);

        // Step 2: Group consecutive flagged samples into regions
        let regions = self.group_regions(samples, &flags);

        // Step 3: Repair each region
        let mut repaired = Vec::with_capacity(regions.len());
        for region in regions {
            if region.end - region.start <= self.config.max_pop_span {
                self.repair(samples, &region);
            }
            repaired.push(region);
        }

        repaired
    }

    /// Detect impulsive samples; returns a boolean mask.
    fn detect(&self, samples: &[f32]) -> Vec<bool> {
        let n = samples.len();
        let half = self.config.window / 2;
        let mut flags = vec![false; n];

        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let window = &samples[lo..hi];

            let med = median(window);
            let mad_val = mad(window, med);

            if mad_val > 1e-10 {
                let deviation = (samples[i] - med).abs();
                if deviation > self.config.threshold * mad_val {
                    flags[i] = true;
                }
            }
        }

        flags
    }

    /// Group consecutive flagged samples into `ImpulseRegion`s.
    ///
    /// Each region's `magnitude` is computed by [`Self::region_magnitude`] as the
    /// ratio of the region's peak absolute amplitude to the RMS of the clean
    /// signal immediately surrounding it — i.e. how far above the local noise
    /// floor the impulse rises.
    fn group_regions(&self, samples: &[f32], flags: &[bool]) -> Vec<ImpulseRegion> {
        let mut regions = Vec::new();
        let mut i = 0;
        while i < flags.len() {
            if flags[i] {
                let start = i;
                while i < flags.len() && flags[i] {
                    i += 1;
                }
                let end = i;
                let span = end - start;
                let kind = if span <= self.config.max_click_span {
                    ImpulseKind::Click
                } else {
                    ImpulseKind::Pop
                };
                let magnitude = self.region_magnitude(samples, start, end);
                regions.push(ImpulseRegion {
                    start,
                    end,
                    kind,
                    magnitude,
                });
            } else {
                i += 1;
            }
        }
        regions
    }

    /// Compute the magnitude of an impulsive region `[start, end)` as the ratio
    /// of its peak absolute amplitude to the RMS of the adjacent clean context.
    ///
    /// The context is drawn from up to `self.config.window` samples on each side
    /// of the region (clamped to the signal bounds and excluding the region
    /// itself). A magnitude near `1.0` means the impulse barely exceeds the
    /// surrounding signal energy; larger values indicate sharper clicks/pops.
    /// When no clean context exists, the raw peak amplitude is returned so the
    /// value still reflects the impulse strength rather than a fabricated
    /// constant.
    fn region_magnitude(&self, samples: &[f32], start: usize, end: usize) -> f32 {
        let n = samples.len();
        let end = end.min(n);
        if start >= end {
            return 0.0;
        }

        // Peak absolute amplitude within the flagged region.
        let peak = samples[start..end]
            .iter()
            .fold(0.0f32, |m, &s| m.max(s.abs()));

        // RMS of the clean context immediately surrounding the region.
        let ctx = self.config.window.max(1);
        let left_lo = start.saturating_sub(ctx);
        let right_hi = (end + ctx).min(n);

        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for &s in &samples[left_lo..start] {
            sum_sq += f64::from(s) * f64::from(s);
            count += 1;
        }
        for &s in &samples[end..right_hi] {
            sum_sq += f64::from(s) * f64::from(s);
            count += 1;
        }

        if count == 0 {
            return peak;
        }

        let rms = (sum_sq / count as f64).sqrt() as f32;
        if rms > 1e-9 {
            peak / rms
        } else {
            // Silent surrounding context: the impulse stands alone.
            peak
        }
    }

    /// Repair a single impulsive region using interpolation or AR prediction.
    fn repair(&self, samples: &mut [f32], region: &ImpulseRegion) {
        let n = samples.len();
        let start = region.start;
        let end = region.end.min(n);
        let span = end - start;

        if span == 0 {
            return;
        }

        // Gather clean context around the region
        let left = if start > 0 { samples[start - 1] } else { 0.0 };
        let right = if end < n { samples[end] } else { left };

        if span <= self.config.max_click_span {
            // Linear interpolation for short clicks
            for (j, idx) in (start..end).enumerate() {
                let t = (j + 1) as f32 / (span + 1) as f32;
                samples[idx] = left * (1.0 - t) + right * t;
            }
        } else if span <= 3 * self.config.max_click_span {
            // Cubic Hermite spline for medium spans
            cubic_hermite_fill(samples, start, end, left, right);
        } else {
            // AR prediction for wider pops
            let order = self.config.ar_order.min(start).min(n - end);
            if order >= 2 {
                let coeffs =
                    ar_coefficients(&samples[start.saturating_sub(order * 4)..start], order);
                for idx in start..end {
                    let predicted = ar_predict(&samples[..idx], &coeffs);
                    samples[idx] = predicted;
                }
            } else {
                // Fall back to linear
                for (j, idx) in (start..end).enumerate() {
                    let t = (j + 1) as f32 / (span + 1) as f32;
                    samples[idx] = left * (1.0 - t) + right * t;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cubic Hermite fill
// ---------------------------------------------------------------------------

fn cubic_hermite_fill(samples: &mut [f32], start: usize, end: usize, left: f32, right: f32) {
    let n = samples.len();
    // Estimate tangents from context
    let m0 = if start >= 2 {
        (left - samples[start - 2]) * 0.5
    } else {
        0.0
    };
    let m1 = if end + 1 < n {
        (samples[end + 1] - right) * 0.5
    } else {
        0.0
    };

    let span = (end - start + 1) as f32;
    for (j, idx) in (start..end).enumerate() {
        let t = (j + 1) as f32 / span;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        samples[idx] = h00 * left + h10 * m0 + h01 * right + h11 * m1;
    }
}

// ---------------------------------------------------------------------------
// AR (auto-regressive) prediction via Levinson–Durbin
// ---------------------------------------------------------------------------

/// Estimate AR coefficients of order `p` from `context` using Levinson–Durbin.
fn ar_coefficients(context: &[f32], p: usize) -> Vec<f32> {
    let n = context.len();
    if n < p + 1 || p == 0 {
        return vec![0.0; p];
    }

    // Compute biased autocorrelation r[0..=p]
    let mut r = vec![0.0_f64; p + 1];
    for lag in 0..=p {
        let mut acc = 0.0_f64;
        for i in lag..n {
            acc += f64::from(context[i]) * f64::from(context[i - lag]);
        }
        r[lag] = acc / n as f64;
    }

    if r[0].abs() < 1e-12 {
        return vec![0.0; p];
    }

    // Levinson–Durbin recursion
    let mut a = vec![0.0_f64; p]; // AR coefficients
    let mut a_prev = vec![0.0_f64; p];
    let mut error = r[0];

    for order in 1..=p {
        // Reflection coefficient
        let mut lambda = 0.0_f64;
        for j in 0..order - 1 {
            lambda += a_prev[j] * r[order - 1 - j];
        }
        lambda = (r[order] - lambda) / error;

        a[order - 1] = lambda;
        for j in 0..order - 1 {
            a[j] = a_prev[j] - lambda * a_prev[order - 2 - j];
        }

        error *= 1.0 - lambda * lambda;
        if error.abs() < 1e-12 {
            break;
        }
        a_prev = a.clone();
    }

    a.iter().map(|&c| c as f32).collect()
}

/// Predict one sample ahead using AR coefficients applied to recent history.
fn ar_predict(history: &[f32], coeffs: &[f32]) -> f32 {
    let _p = coeffs.len();
    let n = history.len();
    let mut sum = 0.0_f32;
    for (j, &c) in coeffs.iter().enumerate() {
        if j < n {
            sum += c * history[n - 1 - j];
        }
    }
    // Clamp to [-1, 1] to prevent runaway predictions
    sum.clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Compute the median of a slice (returns 0 for empty slices).
fn median(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[m - 1] + sorted[m]) * 0.5
    } else {
        sorted[m]
    }
}

/// Compute the median absolute deviation of `data` around `center`.
fn mad(data: &[f32], center: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let deviations: Vec<f32> = data.iter().map(|&x| (x - center).abs()).collect();
    median(&deviations)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (TAU * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect()
    }

    fn inject_click(signal: &mut Vec<f32>, pos: usize) {
        signal[pos] = 5.0; // strong impulse well outside [-1, 1]
    }

    #[test]
    fn test_click_remover_creation() {
        let _ = ClickRemover::default_config();
    }

    #[test]
    fn test_click_removal_detects_click() {
        let mut signal = sine(4096);
        inject_click(&mut signal, 512);
        let remover = ClickRemover::default_config();
        let mut proc = signal.clone();
        let regions = remover.process(&mut proc);
        let found = regions.iter().any(|r| r.start <= 512 && r.end > 512);
        assert!(found, "Should detect the injected click");
    }

    #[test]
    fn test_click_removal_repaired_sample_finite() {
        let mut signal = sine(4096);
        inject_click(&mut signal, 256);
        let remover = ClickRemover::default_config();
        remover.process(&mut signal);
        for (i, &s) in signal.iter().enumerate() {
            assert!(s.is_finite(), "Sample {i} is not finite after repair");
        }
    }

    #[test]
    fn test_click_removal_amplitude_reduced() {
        let mut signal = sine(4096);
        signal[1024] = 10.0; // large click
        let remover = ClickRemover::default_config();
        remover.process(&mut signal);
        // The repaired sample should be within reasonable range
        assert!(
            signal[1024].abs() < 2.0,
            "Click amplitude should be reduced, got {}",
            signal[1024]
        );
    }

    #[test]
    fn test_clean_signal_unchanged() {
        let mut signal = sine(4096);
        let original = signal.clone();
        let remover = ClickRemover::default_config();
        let regions = remover.process(&mut signal);
        assert!(
            regions.is_empty() || regions.len() < 5,
            "Should not find clicks in a clean signal"
        );
        // Most samples should be unchanged (allow for tiny numerical differences)
        let changed: usize = signal
            .iter()
            .zip(original.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-5)
            .count();
        assert!(
            changed < signal.len() / 100,
            "More than 1% of clean signal was modified: {changed} samples"
        );
    }

    #[test]
    fn test_multiple_clicks() {
        let mut signal = sine(8192);
        inject_click(&mut signal, 200);
        inject_click(&mut signal, 1000);
        inject_click(&mut signal, 4000);
        let remover = ClickRemover::default_config();
        let regions = remover.process(&mut signal);
        assert!(
            regions.len() >= 3,
            "Should detect at least 3 clicks, found {}",
            regions.len()
        );
        for &s in &signal {
            assert!(
                s.is_finite(),
                "All samples must be finite after multi-click repair"
            );
        }
    }

    #[test]
    fn test_short_signal_no_panic() {
        let mut signal = vec![0.5_f32; 10];
        let remover = ClickRemover::default_config();
        let _ = remover.process(&mut signal);
    }

    #[test]
    fn test_median_even() {
        let data = vec![1.0_f32, 3.0, 2.0, 4.0];
        let m = median(&data);
        assert!((m - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_median_odd() {
        let data = vec![5.0_f32, 1.0, 3.0];
        let m = median(&data);
        assert!((m - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mad_zero_for_constant() {
        let data = vec![2.0_f32; 10];
        assert_eq!(mad(&data, 2.0), 0.0);
    }

    #[test]
    fn test_ar_coefficients_returns_correct_length() {
        let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let coeffs = ar_coefficients(&signal, 8);
        assert_eq!(coeffs.len(), 8);
        for &c in &coeffs {
            assert!(c.is_finite(), "AR coefficient must be finite");
        }
    }

    #[test]
    fn test_region_kind_click_vs_pop() {
        let config = ClickRemoverConfig {
            max_click_span: 4,
            ..Default::default()
        };
        let remover = ClickRemover::new(config);
        let short_flags = [false, true, true, true, false];
        let long_flags = [false, true, true, true, true, true, true, false];
        let short_samples = vec![0.1f32; short_flags.len()];
        let long_samples = vec![0.1f32; long_flags.len()];
        let short_regions = remover.group_regions(&short_samples, &short_flags);
        let long_regions = remover.group_regions(&long_samples, &long_flags);
        assert_eq!(short_regions[0].kind, ImpulseKind::Click);
        assert_eq!(long_regions[0].kind, ImpulseKind::Pop);
    }
}
