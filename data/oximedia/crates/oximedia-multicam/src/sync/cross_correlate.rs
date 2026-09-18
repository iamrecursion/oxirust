//! FFT-based audio cross-correlation for multi-camera synchronization.
//!
//! Computes the cross-correlation of two audio buffers via the FFT/IFFT
//! convolution theorem, then finds the lag at the peak correlation value to
//! determine the time offset between two camera audio tracks.
//!
//! # Algorithm
//!
//! Given signals `a` and `b`:
//! 1. Downsample both signals so that the cross-correlation FFT size stays
//!    within the limit supported by the underlying FFT engine (2^16 = 65536).
//! 2. Zero-pad both to the next power-of-two size ≥ `2N - 1` for linear
//!    (non-circular) correlation.
//! 3. Compute FFTs: `A = FFT(a)`, `B = FFT(b)`.
//! 4. Multiply element-wise: `R[k] = A[k] * conj(B[k])`.
//! 5. Compute IFFT: `r = IFFT(R)`.
//! 6. Interpret the result as the cross-correlation sequence and locate its
//!    peak to obtain the integer-sample lag.
//! 7. Apply parabolic interpolation around the peak for sub-sample accuracy.
//! 8. Scale the lag back to the original sample rate.
//! 9. Convert sample lag to [`std::time::Duration`].

use std::time::Duration;

use oxifft::api::{fft, ifft};
use oxifft::Complex;

use crate::{MultiCamError, Result};

// ── constants ─────────────────────────────────────────────────────────────────

/// Maximum FFT size supported by the underlying engine without panicking.
/// 2^16 = 65536 complex samples.
const MAX_FFT_SIZE: usize = 1 << 16;

/// Maximum half-length (i.e. max number of samples per signal after
/// truncation) to keep the linear-correlation FFT size ≤ MAX_FFT_SIZE.
/// For linear correlation: fft_size ≥ 2N−1 → N ≤ (MAX_FFT_SIZE + 1) / 2.
const MAX_SIGNAL_LEN: usize = (MAX_FFT_SIZE + 1) / 2; // 32768

// ── helpers ───────────────────────────────────────────────────────────────────

/// Round up to the next power of two (returns `n` itself if it already is one).
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

/// Compute the RMS energy of a signal slice.
fn rms_energy(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / signal.len() as f64).sqrt()
}

/// Downsample `signal` by `factor` using simple stride decimation.
fn downsample(signal: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return signal.to_vec();
    }
    signal.iter().step_by(factor).copied().collect()
}

/// Compute the minimum downsample factor needed so that `len` samples fit
/// within [`MAX_SIGNAL_LEN`].
fn required_downsample_factor(len: usize) -> usize {
    if len <= MAX_SIGNAL_LEN {
        return 1;
    }
    // We need ceil(len / MAX_SIGNAL_LEN).
    (len + MAX_SIGNAL_LEN - 1) / MAX_SIGNAL_LEN
}

/// Parabolically interpolate around the peak index to obtain a sub-sample lag.
fn parabolic_interpolate(xcorr: &[f64], peak_idx: usize) -> f64 {
    let n = xcorr.len();
    if peak_idx == 0 || peak_idx + 1 >= n {
        return peak_idx as f64;
    }
    let y_m = xcorr[peak_idx - 1];
    let y_0 = xcorr[peak_idx];
    let y_p = xcorr[peak_idx + 1];
    let denom = 2.0 * y_0 - y_m - y_p;
    if denom.abs() < 1e-12 {
        return peak_idx as f64;
    }
    peak_idx as f64 + (y_m - y_p) / (2.0 * denom)
}

/// Core FFT cross-correlation returning the signed lag in downsampled samples.
///
/// Both `a` and `b` must have length ≤ `MAX_SIGNAL_LEN`.
///
/// Returns `(lag_ds, peak_xcorr_value)` where `lag_ds` is the signed lag
/// in downsampled samples (positive = b is delayed relative to a).
fn fft_xcorr(a: &[f32], b: &[f32]) -> (f64, f64) {
    debug_assert!(a.len() <= MAX_SIGNAL_LEN);
    debug_assert!(b.len() <= MAX_SIGNAL_LEN);

    let linear_len = a.len() + b.len() - 1;
    let fft_size = next_power_of_two(linear_len).min(MAX_FFT_SIZE);

    // Build zero-padded complex input buffers.
    let zero = Complex::new(0.0_f64, 0.0_f64);

    let mut fa: Vec<Complex<f64>> = a.iter().map(|&x| Complex::new(f64::from(x), 0.0)).collect();
    fa.resize(fft_size, zero);

    let mut fb: Vec<Complex<f64>> = b.iter().map(|&x| Complex::new(f64::from(x), 0.0)).collect();
    fb.resize(fft_size, zero);

    // Forward FFTs.
    let fa_spec = fft(&fa);
    let fb_spec = fft(&fb);

    // Cross-power spectrum: A * conj(B).
    let cross: Vec<Complex<f64>> = fa_spec
        .iter()
        .zip(fb_spec.iter())
        .map(|(a_val, b_val)| {
            Complex::new(
                a_val.re * b_val.re + a_val.im * b_val.im,
                a_val.im * b_val.re - a_val.re * b_val.im,
            )
        })
        .collect();

    // Inverse FFT to obtain correlation sequence.
    let xcorr_complex = ifft(&cross);

    // Normalisation: divide by signal energies × fft_size to get NCC in [-1,1].
    let energy_a = rms_energy(a);
    let energy_b = rms_energy(b);
    let denom = energy_a * energy_b * fft_size as f64;
    let norm = if denom > 0.0 { denom } else { fft_size as f64 };

    let xcorr: Vec<f64> = xcorr_complex.iter().map(|c| c.re / norm).collect();

    // Search for the peak in the positive-lag half [0, len_a) and the
    // negative-lag wrap-around [fft_size - len_b + 1, fft_size).
    let max_pos = a.len();
    let neg_start = fft_size.saturating_sub(b.len().saturating_sub(1));

    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;

    for i in 0..max_pos.min(xcorr.len()) {
        if xcorr[i] > best_val {
            best_val = xcorr[i];
            best_idx = i;
        }
    }
    for i in neg_start..xcorr.len() {
        if xcorr[i] > best_val {
            best_val = xcorr[i];
            best_idx = i;
        }
    }

    // Sub-sample interpolation.
    let fractional_idx = parabolic_interpolate(&xcorr, best_idx);

    // Convert to signed lag.
    let lag_ds = if fractional_idx < (fft_size as f64) / 2.0 {
        fractional_idx
    } else {
        fractional_idx - fft_size as f64
    };

    (lag_ds, best_val)
}

// ── Public FFT cross-correlation API ─────────────────────────────────────────

/// Compute the FFT-based cross-correlation of two signals.
///
/// Both signals are zero-padded to the next power of two ≥ `len(a) + len(b) - 1`
/// for linear (non-circular) correlation.  The result vector has length equal to
/// that power-of-two FFT size.
///
/// Positive lags in the returned vector represent how many samples `b` is
/// delayed relative to `a`.  Use [`find_sync_offset`] to obtain a single
/// signed integer offset.
///
/// # Panics
///
/// Does not panic; returns an empty `Vec` when either input is empty.
#[must_use]
pub fn fft_cross_correlate(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    // Clamp to MAX_SIGNAL_LEN so the FFT size stays within the engine limit.
    let a_clamped = if a.len() > MAX_SIGNAL_LEN {
        &a[..MAX_SIGNAL_LEN]
    } else {
        a
    };
    let b_clamped = if b.len() > MAX_SIGNAL_LEN {
        &b[..MAX_SIGNAL_LEN]
    } else {
        b
    };

    let linear_len = a_clamped.len() + b_clamped.len() - 1;
    let fft_size = next_power_of_two(linear_len).min(MAX_FFT_SIZE);

    let zero = oxifft::Complex::new(0.0_f64, 0.0_f64);

    let mut fa: Vec<oxifft::Complex<f64>> = a_clamped
        .iter()
        .map(|&x| oxifft::Complex::new(f64::from(x), 0.0))
        .collect();
    fa.resize(fft_size, zero);

    let mut fb: Vec<oxifft::Complex<f64>> = b_clamped
        .iter()
        .map(|&x| oxifft::Complex::new(f64::from(x), 0.0))
        .collect();
    fb.resize(fft_size, zero);

    let fa_spec = oxifft::api::fft(&fa);
    let fb_spec = oxifft::api::fft(&fb);

    // Cross-power spectrum: A * conj(B)
    let cross: Vec<oxifft::Complex<f64>> = fa_spec
        .iter()
        .zip(fb_spec.iter())
        .map(|(a_val, b_val)| {
            oxifft::Complex::new(
                a_val.re * b_val.re + a_val.im * b_val.im,
                a_val.im * b_val.re - a_val.re * b_val.im,
            )
        })
        .collect();

    let xcorr_complex = oxifft::api::ifft(&cross);

    // Normalise and convert to f32.
    let norm = fft_size as f64;
    xcorr_complex.iter().map(|c| (c.re / norm) as f32).collect()
}

/// Find the time offset (in samples) between two audio signals using FFT
/// cross-correlation.
///
/// A positive return value means `signal_b` starts later than `signal_a`
/// (i.e. `b` is delayed by the returned number of samples).  A negative
/// value means `b` leads `a`.
///
/// The `sample_rate` parameter is accepted for future normalisation but the
/// returned value is always in *samples* (not seconds).
///
/// # Returns
///
/// The signed sample lag.  Returns `0` when either input is empty.
#[must_use]
pub fn find_sync_offset(signal_a: &[f32], signal_b: &[f32], _sample_rate: u32) -> i64 {
    if signal_a.is_empty() || signal_b.is_empty() {
        return 0;
    }

    // Use the CrossCorrelator which handles downsampling for long buffers.
    let sr = _sample_rate;
    let correlator = CrossCorrelator::new(sr);
    // find_offset_samples returns Ok only when both are non-empty (guaranteed above).
    correlator
        .find_offset_samples(signal_a, signal_b)
        .unwrap_or(0)
}

// ── CrossCorrelator ───────────────────────────────────────────────────────────

/// FFT-based cross-correlator for two mono audio buffers.
///
/// Automatically downsamples long buffers to keep the FFT size within the
/// engine limit, then scales the result back to original sample-count space.
///
/// # Example
///
/// ```
/// use oximedia_multicam::sync::cross_correlate::CrossCorrelator;
///
/// let sample_rate = 48_000_u32;
/// let correlator = CrossCorrelator::new(sample_rate);
///
/// // Build a simple 1 kHz tone and a version delayed by 100 ms.
/// let freq = 1_000.0_f32;
/// let offset_samples = (0.1 * sample_rate as f32) as usize; // 4 800
/// let n = 96_000_usize; // 2 seconds
/// let a: Vec<f32> = (0..n)
///     .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
///     .collect();
/// let mut b = vec![0.0_f32; n];
/// b[offset_samples..].copy_from_slice(&a[..n - offset_samples]);
///
/// let duration = correlator.find_offset(&a, &b).expect("should succeed");
/// let offset_ms = duration.as_millis();
/// assert!((offset_ms as i64 - 100).abs() <= 5, "offset was {offset_ms} ms");
/// ```
#[derive(Debug, Clone)]
pub struct CrossCorrelator {
    /// Sample rate of the audio buffers (Hz).
    pub sample_rate: u32,
}

impl CrossCorrelator {
    /// Create a new `CrossCorrelator` for the given `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Find the time offset between two audio buffers by locating the peak of
    /// their FFT cross-correlation.
    ///
    /// Returns a [`Duration`] representing the absolute magnitude of the lag.
    /// To obtain the signed lag (b leads vs b lags), use `find_offset_samples`.
    ///
    /// # Errors
    ///
    /// Returns [`MultiCamError::InsufficientData`] if either slice is empty.
    pub fn find_offset(&self, a: &[f32], b: &[f32]) -> Result<Duration> {
        let lag_samples = self.find_offset_samples(a, b)?;
        let abs_samples = lag_samples.unsigned_abs() as u64;
        let nanos = abs_samples
            .saturating_mul(1_000_000_000)
            .checked_div(u64::from(self.sample_rate))
            .unwrap_or(0);
        Ok(Duration::from_nanos(nanos))
    }

    /// Like `find_offset` but returns the signed sample lag.
    ///
    /// A positive value means `b` starts later than `a` by that many samples
    /// (b is delayed).  A negative value means `b` leads `a`.
    ///
    /// # Errors
    ///
    /// Returns [`MultiCamError::InsufficientData`] when either buffer is empty.
    pub fn find_offset_samples(&self, a: &[f32], b: &[f32]) -> Result<i64> {
        if a.is_empty() || b.is_empty() {
            return Err(MultiCamError::InsufficientData(
                "audio buffers must not be empty".into(),
            ));
        }

        // Compute the downsample factor to keep each signal ≤ MAX_SIGNAL_LEN.
        let max_len = a.len().max(b.len());
        let ds_factor = required_downsample_factor(max_len);

        let a_ds = downsample(a, ds_factor);
        let b_ds = downsample(b, ds_factor);

        // FFT cross-correlation in downsampled domain.
        // The FFT convention here produces: r[k] = sum_n a[n]*b[n-k].
        // Peak at k = -offset when b is delayed by `offset` relative to a.
        // We negate so that positive lag = b is delayed.
        let (lag_ds, _peak) = fft_xcorr(&a_ds, &b_ds);

        // Scale lag from downsampled domain back to original sample rate.
        // Negate to match the convention: positive = b is delayed.
        let lag_orig = -lag_ds * ds_factor as f64;
        Ok(lag_orig.round() as i64)
    }

    /// Return the peak normalised cross-correlation value in [-1.0, 1.0].
    ///
    /// Computes the Pearson NCC at the lag found by `find_offset_samples`.
    ///
    /// # Errors
    ///
    /// Returns [`MultiCamError::InsufficientData`] when either buffer is empty.
    pub fn peak_correlation(&self, a: &[f32], b: &[f32]) -> Result<f64> {
        if a.is_empty() || b.is_empty() {
            return Err(MultiCamError::InsufficientData(
                "audio buffers must not be empty".into(),
            ));
        }

        let lag = self.find_offset_samples(a, b)?;

        // NCC at the found lag.
        let (start_a, start_b) = if lag >= 0 {
            (0usize, lag as usize)
        } else {
            ((-lag) as usize, 0usize)
        };

        let count = a
            .len()
            .saturating_sub(start_a)
            .min(b.len().saturating_sub(start_b));

        if count == 0 {
            return Ok(0.0);
        }

        let mut sum_ab = 0.0f64;
        let mut sum_aa = 0.0f64;
        let mut sum_bb = 0.0f64;

        for i in 0..count {
            let av = f64::from(a[start_a + i]);
            let bv = f64::from(b[start_b + i]);
            sum_ab += av * bv;
            sum_aa += av * av;
            sum_bb += bv * bv;
        }

        let denom = (sum_aa * sum_bb).sqrt();
        if denom > 0.0 {
            Ok((sum_ab / denom).clamp(-1.0, 1.0))
        } else {
            Ok(0.0)
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const SAMPLE_RATE: u32 = 48_000;

    /// Generate a sine tone of `n` samples at `sample_rate`.
    fn sine_tone(freq_hz: f32, n: usize, sample_rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    /// Delay `signal` by `offset` samples (zero-pad start, truncate end).
    fn delay_signal(signal: &[f32], offset: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; signal.len()];
        if offset < signal.len() {
            out[offset..].copy_from_slice(&signal[..signal.len() - offset]);
        }
        out
    }

    // ── basic API tests ───────────────────────────────────────────────────────

    #[test]
    fn test_new_stores_sample_rate() {
        let c = CrossCorrelator::new(44_100);
        assert_eq!(c.sample_rate, 44_100);
    }

    #[test]
    fn test_empty_a_returns_error() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let b = vec![1.0_f32; 100];
        assert!(c.find_offset(&[], &b).is_err());
    }

    #[test]
    fn test_empty_b_returns_error() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let a = vec![1.0_f32; 100];
        assert!(c.find_offset(&a, &[]).is_err());
    }

    // ── known-offset synthetic signal tests ───────────────────────────────────
    //
    // IMPORTANT: Pure sine tones create periodic cross-correlation functions,
    // which means any lag that is a multiple of the sine's period gives the
    // same maximum value.  To break this ambiguity we use broadband mixed-tone
    // or AM-modulated signals so that the cross-correlation has a unique global
    // peak.

    /// Generate a non-periodic test signal: 1 kHz AM-modulated by a 50 Hz
    /// envelope, giving a unique cross-correlation peak.
    fn broadband_signal(n: usize, sample_rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Carrier 1 kHz with 50 Hz envelope modulation (AM).
                let env = 0.5 + 0.5 * (2.0 * PI * 50.0 * t).sin();
                env * (2.0 * PI * 1_000.0 * t).sin()
                    + 0.3 * (2.0 * PI * 1_337.0 * t).sin()
                    + 0.2 * (2.0 * PI * 283.0 * t).sin()
            })
            .collect()
    }

    /// Broadband signal, b delayed by exactly 100 ms → 4 800 samples at 48 kHz.
    #[test]
    fn test_100ms_offset_1khz_tone() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let n = 96_000; // 2 seconds
        let a = broadband_signal(n, SAMPLE_RATE);
        let offset_samples = (0.1 * SAMPLE_RATE as f32) as usize; // 4 800
        let b = delay_signal(&a, offset_samples);
        let lag = c.find_offset_samples(&a, &b).expect("should succeed");
        assert!(
            (lag - offset_samples as i64).abs() <= 200,
            "Expected lag ≈ {offset_samples}, got {lag}"
        );
    }

    /// Broadband signal, b delayed by exactly 50 ms.
    #[test]
    fn test_50ms_offset_1khz_tone() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let n = 96_000;
        let a = broadband_signal(n, SAMPLE_RATE);
        let offset_samples = (0.05 * SAMPLE_RATE as f32) as usize; // 2 400
        let b = delay_signal(&a, offset_samples);
        let lag = c.find_offset_samples(&a, &b).expect("should succeed");
        assert!(
            (lag - offset_samples as i64).abs() <= 200,
            "Expected lag ≈ {offset_samples}, got {lag}"
        );
    }

    /// Broadband signal, b delayed by 200 ms.
    #[test]
    fn test_200ms_offset_500hz_tone() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let n = 144_000; // 3 seconds
        let a = broadband_signal(n, SAMPLE_RATE);
        let offset_samples = (0.2 * SAMPLE_RATE as f32) as usize; // 9 600
        let b = delay_signal(&a, offset_samples);
        let lag = c.find_offset_samples(&a, &b).expect("should succeed");
        assert!(
            (lag - offset_samples as i64).abs() <= 500,
            "Expected lag ≈ {offset_samples}, got {lag}"
        );
    }

    /// Zero offset: identical signals should yield lag ≈ 0.
    #[test]
    fn test_zero_offset_identical_signals() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let a = broadband_signal(48_000, SAMPLE_RATE);
        let lag = c.find_offset_samples(&a, &a).expect("should succeed");
        // With downsampling the tolerance is one downsample period.
        let ds = required_downsample_factor(a.len()) as i64;
        assert!(
            lag.abs() <= ds * 2,
            "Expected lag ≈ 0, got {lag} (ds_factor={ds})"
        );
    }

    /// Duration for 100 ms offset should be within ~100 ms.
    #[test]
    fn test_find_offset_returns_correct_duration() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let n = 96_000;
        let a = broadband_signal(n, SAMPLE_RATE);
        let offset_samples = (0.1 * SAMPLE_RATE as f32) as usize;
        let b = delay_signal(&a, offset_samples);
        let dur = c.find_offset(&a, &b).expect("should succeed");
        let ms = dur.as_millis() as i64;
        assert!((ms - 100).abs() <= 15, "Expected ~100 ms, got {ms} ms");
    }

    /// Peak correlation of identical signals should be near 1.0.
    #[test]
    fn test_peak_correlation_identical() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let a = broadband_signal(48_000, SAMPLE_RATE);
        let corr = c.peak_correlation(&a, &a).expect("should succeed");
        assert!(corr > 0.8, "Expected near 1.0 correlation, got {corr}");
    }

    /// Peak correlation of uncorrelated (different) signals should be lower
    /// than self-correlation.
    #[test]
    fn test_peak_correlation_uncorrelated() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let a = broadband_signal(48_000, SAMPLE_RATE);
        // A completely different signal: 200 Hz pure tone.
        let b = sine_tone(200.0, 48_000, SAMPLE_RATE);
        let corr_self = c.peak_correlation(&a, &a).expect("should succeed");
        let corr_diff = c.peak_correlation(&a, &b).expect("should succeed");
        // Self-correlation should beat cross-correlation.
        assert!(
            corr_self > corr_diff,
            "Expected self-corr ({corr_self:.3}) > cross-corr ({corr_diff:.3})"
        );
    }

    /// Broadband signal, b delayed by 10 ms.
    #[test]
    fn test_10ms_offset_2khz_tone() {
        let c = CrossCorrelator::new(SAMPLE_RATE);
        let n = 96_000;
        let a = broadband_signal(n, SAMPLE_RATE);
        let offset_samples = (0.01 * SAMPLE_RATE as f32) as usize; // 480
        let b = delay_signal(&a, offset_samples);
        let lag = c.find_offset_samples(&a, &b).expect("should succeed");
        assert!(
            (lag - offset_samples as i64).abs() <= 100,
            "Expected lag ≈ {offset_samples}, got {lag}"
        );
    }
}
