//! Phase correction utilities for `OxiMedia` normalize crate.
//!
//! Detects and corrects phase relationships between audio channels.

#![allow(dead_code)]

use crate::{NormalizeError, NormalizeResult};
use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

/// A phase shift value, stored in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseShift {
    degrees: f32,
}

impl PhaseShift {
    /// Create a `PhaseShift` from a degree value.
    pub fn from_degrees(degrees: f32) -> Self {
        // Normalise to (-180, 180]
        let mut d = degrees % 360.0;
        if d > 180.0 {
            d -= 360.0;
        } else if d <= -180.0 {
            d += 360.0;
        }
        Self { degrees: d }
    }

    /// Create a `PhaseShift` from a radian value.
    pub fn from_radians(radians: f32) -> Self {
        Self::from_degrees(radians.to_degrees())
    }

    /// Phase shift in degrees.
    pub fn degrees(self) -> f32 {
        self.degrees
    }

    /// Phase shift in radians.
    pub fn radians(self) -> f32 {
        self.degrees.to_radians()
    }

    /// True if the shift represents an approximate polarity inversion (~180°).
    pub fn is_inverted(self, tolerance_deg: f32) -> bool {
        (self.degrees.abs() - 180.0).abs() <= tolerance_deg
    }

    /// True if the shift is approximately zero.
    pub fn is_in_phase(self, tolerance_deg: f32) -> bool {
        self.degrees.abs() <= tolerance_deg
    }
}

/// Configuration for the phase corrector.
#[derive(Clone, Debug)]
pub struct PhaseCorrectorConfig {
    /// Maximum shift to apply in degrees.
    pub max_shift_deg: f32,
    /// If the detected shift is within this tolerance (°), skip correction.
    pub tolerance_deg: f32,
    /// Number of channels in the stream.
    pub channels: usize,
}

impl Default for PhaseCorrectorConfig {
    fn default() -> Self {
        Self {
            max_shift_deg: 180.0,
            tolerance_deg: 5.0,
            channels: 2,
        }
    }
}

impl PhaseCorrectorConfig {
    /// Construct with given channel count.
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            ..Default::default()
        }
    }
}

/// A cached forward/inverse FFT plan pair for one buffer length.
struct FftPlanPair {
    n: usize,
    forward: Plan<f64>,
    inverse: Plan<f64>,
}

/// Applies a constant (frequency-independent) phase rotation to an audio channel.
///
/// # How it works
///
/// A true broadband phase shift cannot be done in the sample domain with a plain
/// per-sample scale (that only changes amplitude — it does not shift the waveform
/// in time/phase at all). Instead this corrector implements a real FFT-based
/// all-pass filter:
///
/// 1. Forward FFT the whole input buffer to the frequency domain.
/// 2. Rotate every positive-frequency bin by `e^{-jθ}` and its conjugate-mirror
///    negative-frequency bin by `e^{+jθ}` (leaving the DC bin, and the Nyquist bin
///    for even-length buffers, untouched — a real-valued signal carries no phase
///    information in those two bins, and rotating them would inject a spurious
///    imaginary component into the reconstructed signal).
/// 3. Inverse FFT back to the time domain and take the real part.
///
/// Because every bin is only *rotated* (multiplied by a unit-magnitude complex
/// number), `|X'[k]| == |X[k]|` for every bin: the magnitude spectrum — and hence
/// the signal's energy/RMS — is preserved exactly (up to floating-point and FFT
/// round-trip error), while the phase of every frequency component shifts by
/// `θ`. This is the textbook definition of an all-pass filter, and is what lets a
/// sinusoid `sin(ωt)` become `sin(ωt - θ)` rather than merely a scaled-down copy
/// of itself.
pub struct PhaseCorrector {
    config: PhaseCorrectorConfig,
    /// Per-channel applied shift.
    shifts: Vec<PhaseShift>,
    /// Cached FFT/IFFT plan pair for the most recently used buffer length, so
    /// repeated calls at a stable block size do not re-plan every time.
    fft_cache: Option<FftPlanPair>,
}

impl PhaseCorrector {
    /// Create a new corrector; all channel shifts initialised to zero.
    pub fn new(config: PhaseCorrectorConfig) -> Self {
        let n = config.channels;
        Self {
            config,
            shifts: vec![PhaseShift::from_degrees(0.0); n],
            fft_cache: None,
        }
    }

    /// Shift one channel's samples by the specified `PhaseShift`.
    ///
    /// This is a real broadband (all-pass) phase shift computed via FFT — see the
    /// [`PhaseCorrector`] docs for the algorithm. It exactly preserves the
    /// magnitude spectrum of `samples` while rotating the phase of every
    /// frequency component by `shift`.
    ///
    /// A buffer of length 0 or 1 carries no phase information (there is no
    /// frequency component to rotate) and is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`NormalizeError::ProcessingError`] if an FFT plan cannot be
    /// constructed for `samples.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `channel >= self.config.channels` (an out-of-range channel index
    /// is a programmer error, consistent with this module's other bounds checks).
    pub fn shift_channel(
        &mut self,
        channel: usize,
        shift: PhaseShift,
        samples: &mut [f32],
    ) -> NormalizeResult<()> {
        assert!(channel < self.config.channels, "channel index out of range");
        self.shifts[channel] = shift;

        let n = samples.len();
        if n < 2 {
            // No frequency bin carries phase information for a 0- or 1-sample buffer.
            return Ok(());
        }

        self.ensure_plans(n)?;
        let cache = self.fft_cache.as_ref().ok_or_else(|| {
            NormalizeError::ProcessingError("FFT plan cache is empty".to_string())
        })?;

        let mut spectrum: Vec<Complex<f64>> = samples
            .iter()
            .map(|&s| Complex::new(f64::from(s), 0.0))
            .collect();
        cache.forward.execute_inplace(&mut spectrum);

        let theta = f64::from(shift.radians());
        let (sin_t, cos_t) = theta.sin_cos();
        let rotate_positive = Complex::new(cos_t, -sin_t);
        let rotate_negative = rotate_positive.conj();
        let half = n / 2;

        for (k, bin) in spectrum.iter_mut().enumerate().skip(1) {
            if n % 2 == 0 && k == half {
                continue; // Nyquist bin: must stay real for a real-valued IFFT result.
            }
            if k <= half {
                *bin *= rotate_positive;
            } else {
                *bin *= rotate_negative;
            }
        }

        cache.inverse.execute_inplace(&mut spectrum);

        // oxifft's backward transform is unnormalized (FFTW convention, matching the
        // `Istft::inverse` convention used elsewhere in the workspace) — divide by n.
        let scale = 1.0 / n as f64;
        for (dst, c) in samples.iter_mut().zip(spectrum.iter()) {
            *dst = (c.re * scale) as f32;
        }

        Ok(())
    }

    /// Ensure `fft_cache` holds a forward/inverse plan pair sized for `n`.
    fn ensure_plans(&mut self, n: usize) -> NormalizeResult<()> {
        if let Some(cache) = &self.fft_cache {
            if cache.n == n {
                return Ok(());
            }
        }
        let forward =
            Plan::<f64>::dft_1d(n, Direction::Forward, Flags::ESTIMATE).ok_or_else(|| {
                NormalizeError::ProcessingError(format!(
                    "failed to construct forward FFT plan for phase shift (n={n})"
                ))
            })?;
        let inverse =
            Plan::<f64>::dft_1d(n, Direction::Backward, Flags::ESTIMATE).ok_or_else(|| {
                NormalizeError::ProcessingError(format!(
                    "failed to construct inverse FFT plan for phase shift (n={n})"
                ))
            })?;
        self.fft_cache = Some(FftPlanPair {
            n,
            forward,
            inverse,
        });
        Ok(())
    }

    /// Return the recorded shift for a channel.
    pub fn channel_shift(&self, channel: usize) -> PhaseShift {
        self.shifts[channel]
    }

    /// Reset all shifts to zero.
    pub fn reset(&mut self) {
        for s in &mut self.shifts {
            *s = PhaseShift::from_degrees(0.0);
        }
    }
}

/// Computes cross-correlation-based phase relationships between channels.
pub struct PhaseInspector {
    sample_rate: f32,
}

impl PhaseInspector {
    /// Create with the given sample rate.
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Compute normalised cross-correlation coefficient between two equal-length buffers.
    ///
    /// Returns a value in `[-1.0, 1.0]`.
    pub fn correlation(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "buffers must be same length");
        let n = a.len();
        if n == 0 {
            return 0.0;
        }
        let mean_a: f32 = a.iter().sum::<f32>() / n as f32;
        let mean_b: f32 = b.iter().sum::<f32>() / n as f32;
        let mut num = 0.0_f32;
        let mut denom_a = 0.0_f32;
        let mut denom_b = 0.0_f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let xa = x - mean_a;
            let yb = y - mean_b;
            num += xa * yb;
            denom_a += xa * xa;
            denom_b += yb * yb;
        }
        let denom = (denom_a * denom_b).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            num / denom
        }
    }

    /// Sample rate this inspector was built for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

/// A report summarising phase inspection results.
#[derive(Clone, Debug)]
pub struct PhaseReport {
    /// Measured cross-correlation coefficient.
    pub correlation: f32,
    /// Estimated phase shift.
    pub estimated_shift: PhaseShift,
    /// Tolerance used when judging in-phase.
    pub tolerance_deg: f32,
}

impl PhaseReport {
    /// Construct a report.
    pub fn new(correlation: f32, estimated_shift: PhaseShift, tolerance_deg: f32) -> Self {
        Self {
            correlation,
            estimated_shift,
            tolerance_deg,
        }
    }

    /// True if the channels are considered in-phase (shift within tolerance).
    pub fn is_in_phase(&self) -> bool {
        self.estimated_shift.is_in_phase(self.tolerance_deg)
    }

    /// True if the channels appear to be polarity-inverted.
    pub fn is_inverted(&self) -> bool {
        // High negative correlation → inverted
        self.correlation < -0.9
    }
}

/// Estimate phase shift from correlation coefficient (rough heuristic).
///
/// Maps correlation [-1,1] to a phase angle in degrees via arccos.
#[allow(clippy::cast_precision_loss)]
pub fn estimate_shift_from_correlation(corr: f32) -> PhaseShift {
    // arccos maps +1 → 0°, 0 → 90°, -1 → 180°
    let clamped = corr.clamp(-1.0, 1.0);
    let rad = clamped.acos(); // [0, π]
    PhaseShift::from_radians(rad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_phase_shift_degrees_normalised() {
        let s = PhaseShift::from_degrees(370.0);
        assert!((s.degrees() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_phase_shift_negative_normalised() {
        let s = PhaseShift::from_degrees(-190.0);
        assert!((s.degrees() - 170.0).abs() < 1e-4);
    }

    #[test]
    fn test_phase_shift_from_radians() {
        let s = PhaseShift::from_radians(PI);
        assert!((s.degrees().abs() - 180.0).abs() < 1e-3);
    }

    #[test]
    fn test_phase_shift_is_inverted() {
        let s = PhaseShift::from_degrees(178.0);
        assert!(s.is_inverted(5.0));
    }

    #[test]
    fn test_phase_shift_not_inverted() {
        let s = PhaseShift::from_degrees(90.0);
        assert!(!s.is_inverted(5.0));
    }

    #[test]
    fn test_phase_shift_is_in_phase() {
        let s = PhaseShift::from_degrees(3.0);
        assert!(s.is_in_phase(5.0));
    }

    #[test]
    fn test_phase_shift_not_in_phase() {
        let s = PhaseShift::from_degrees(30.0);
        assert!(!s.is_in_phase(5.0));
    }

    #[test]
    fn test_corrector_shift_channel_zero_shift_is_identity() {
        let cfg = PhaseCorrectorConfig::new(2);
        let mut corrector = PhaseCorrector::new(cfg);
        // A constant (DC-only) buffer: forward FFT has all energy in bin 0, which is
        // never rotated, so a 0° shift must round-trip to (numerically) the original.
        let original = vec![1.0_f32; 16];
        let mut samples = original.clone();
        let shift = PhaseShift::from_degrees(0.0);
        corrector
            .shift_channel(0, shift, &mut samples)
            .expect("shift_channel should succeed");
        assert!(samples.iter().all(|s| s.is_finite()));
        for (&a, &b) in original.iter().zip(samples.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "0 deg shift should be identity: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_corrector_reset() {
        let cfg = PhaseCorrectorConfig::new(2);
        let mut corrector = PhaseCorrector::new(cfg);
        let mut samples = vec![0.5_f32; 8];
        let shift = PhaseShift::from_degrees(45.0);
        corrector
            .shift_channel(0, shift, &mut samples)
            .expect("shift_channel should succeed");
        corrector.reset();
        assert!((corrector.channel_shift(0).degrees()).abs() < 1e-5);
    }

    /// This is the central regression test for the fabricated-success bug: the old
    /// implementation computed `sample *= shift.radians().cos()`, i.e. a pure
    /// amplitude scale with the *same* phase, not a phase shift at all. A real
    /// all-pass phase shift must (a) preserve the magnitude/RMS of a sinusoid and
    /// (b) actually move its phase — proven here against the closed-form identity
    /// that shifting every frequency component of `sin(ωt)` by `-θ` yields exactly
    /// `sin(ωt - θ)` when the tone sits on an exact FFT bin (no spectral leakage).
    #[test]
    fn test_shift_channel_matches_analytic_phase_shifted_sinusoid() {
        let cfg = PhaseCorrectorConfig::new(1);
        let mut corrector = PhaseCorrector::new(cfg);

        let n = 256_usize;
        let bin = 5.0_f32; // exact integer bin -> no spectral leakage
        let original: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * bin * i as f32 / n as f32).sin())
            .collect();

        for shift_deg in [30.0_f32, 90.0, 135.0, -60.0] {
            let mut samples = original.clone();
            let shift = PhaseShift::from_degrees(shift_deg);
            corrector
                .shift_channel(0, shift, &mut samples)
                .expect("shift_channel should succeed");

            let theta = shift.radians();
            let expected: Vec<f32> = (0..n)
                .map(|i| (std::f32::consts::TAU * bin * i as f32 / n as f32 - theta).sin())
                .collect();

            let max_err = samples
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err < 1e-3,
                "shift {shift_deg} deg: max error {max_err} vs analytic sin(wt - theta)"
            );
        }
    }

    #[test]
    fn test_shift_channel_preserves_rms_but_changes_waveform() {
        let cfg = PhaseCorrectorConfig::new(1);
        let mut corrector = PhaseCorrector::new(cfg);

        let n = 512_usize;
        let bin = 11.0_f32;
        let original: Vec<f32> = (0..n)
            .map(|i| 0.8 * (std::f32::consts::TAU * bin * i as f32 / n as f32).sin())
            .collect();
        let mut samples = original.clone();

        let shift = PhaseShift::from_degrees(90.0);
        corrector
            .shift_channel(0, shift, &mut samples)
            .expect("shift_channel should succeed");

        let rms =
            |s: &[f32]| -> f32 { (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt() };
        let rms_before = rms(&original);
        let rms_after = rms(&samples);
        assert!(
            (rms_before - rms_after).abs() < 0.01,
            "an all-pass phase shift must preserve RMS: before {rms_before}, after {rms_after}"
        );

        // Prove it is *not* a no-op / pure amplitude scale: the waveform must have
        // measurably changed shape. A pure amplitude scale by cos(90 deg) == 0 would
        // (per the old bug) collapse the signal to ~silence; a real phase shift keeps
        // the same RMS while moving energy in time.
        let max_abs_diff = original
            .iter()
            .zip(samples.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff > 0.5,
            "waveform should differ substantially after a 90 deg shift, got max diff {max_abs_diff}"
        );

        // Cross-correlation against the original must drop well below 1.0 — a pure
        // amplitude scale (the old bug) would leave normalized correlation at
        // *exactly* 1.0 (or -1.0), since it does not change the signal's shape.
        let inspector = PhaseInspector::new(48_000.0);
        let corr = inspector.correlation(&original, &samples);
        assert!(
            corr.abs() < 0.9,
            "a genuine phase shift must reduce self-correlation; amplitude-only scaling \
             would leave it at +/-1.0, got {corr}"
        );
    }

    #[test]
    fn test_shift_channel_empty_and_single_sample_are_noop() {
        let cfg = PhaseCorrectorConfig::new(1);
        let mut corrector = PhaseCorrector::new(cfg);

        let mut empty: Vec<f32> = Vec::new();
        corrector
            .shift_channel(0, PhaseShift::from_degrees(45.0), &mut empty)
            .expect("empty buffer should succeed as a no-op");
        assert!(empty.is_empty());

        let mut single = vec![0.42_f32];
        corrector
            .shift_channel(0, PhaseShift::from_degrees(45.0), &mut single)
            .expect("single-sample buffer should succeed as a no-op");
        assert!((single[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_inspector_correlation_identical() {
        let inspector = PhaseInspector::new(48_000.0);
        let a: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let corr = inspector.correlation(&a, &a);
        assert!((corr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_inspector_correlation_inverted() {
        let inspector = PhaseInspector::new(48_000.0);
        let a: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = a.iter().map(|s| -s).collect();
        let corr = inspector.correlation(&a, &b);
        assert!((corr + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_inspector_correlation_empty() {
        let inspector = PhaseInspector::new(48_000.0);
        let corr = inspector.correlation(&[], &[]);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_phase_report_is_in_phase() {
        let shift = PhaseShift::from_degrees(2.0);
        let report = PhaseReport::new(0.98, shift, 5.0);
        assert!(report.is_in_phase());
    }

    #[test]
    fn test_phase_report_is_inverted() {
        let shift = PhaseShift::from_degrees(180.0);
        let report = PhaseReport::new(-0.99, shift, 5.0);
        assert!(report.is_inverted());
    }

    #[test]
    fn test_estimate_shift_from_correlation_unity() {
        let s = estimate_shift_from_correlation(1.0);
        assert!(s.degrees().abs() < 1e-3);
    }

    #[test]
    fn test_estimate_shift_from_correlation_inverted() {
        let s = estimate_shift_from_correlation(-1.0);
        assert!((s.degrees().abs() - 180.0).abs() < 1e-3);
    }
}
