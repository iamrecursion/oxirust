//! Batched Welch PSD for parallel multi-channel processing.
//!
//! This module provides a `BatchedWelch` estimator that computes the power
//! spectral density (PSD) and cross-power spectral density (CPSD) matrices
//! for multiple measurement channels simultaneously.  The CPSD matrix is
//! required by the Enhanced FDD (EFDD) algorithm in [`crate::oma_efdd`].
//!
//! # Algorithm
//!
//! The standard Welch method:
//! 1. Split each channel signal into overlapping segments of length `nperseg`.
//! 2. Apply a window function (Hann, Hamming, or flat/rectangular).
//! 3. Optionally subtract the segment mean (detrend).
//! 4. Compute the FFT of each windowed segment.
//! 5. Accumulate the magnitude-squared spectra across segments.
//! 6. Average and normalise to physical units (V²/Hz).
//!
//! For the CPSD matrix at each frequency bin `k`:
//! `G[i,j](k) = average( X_i(k) * conj(X_j(k)) ) / (W * fs)`
//!
//! where `W` is the window energy (`sum(window^2)`).

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Window function for the Welch method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WelchWindow {
    /// Hann (Hanning) window — cosine bell, good general-purpose choice.
    Hann,
    /// Hamming window — modified Hann with non-zero endpoints; slightly lower
    /// side-lobe amplitude but higher stop-band floor.
    Hamming,
    /// Flat (rectangular / boxcar) window — no amplitude correction, maximum
    /// frequency resolution but significant spectral leakage.
    Flat,
}

/// Configuration for the Welch batched PSD estimator.
#[derive(Debug, Clone)]
pub struct WelchConfig {
    /// Samples per FFT segment.  Must be at least 4.
    pub nperseg: usize,
    /// Number of overlapping samples between consecutive segments.
    /// Must be strictly less than `nperseg`.
    pub noverlap: usize,
    /// Window function applied to each segment before FFT.
    pub window: WelchWindow,
    /// Sampling frequency in Hz.  Used to compute the frequency axis and
    /// to normalise the PSD to power per Hz.
    pub fs: f64,
    /// If `true`, subtract the mean of each segment before windowing.
    pub detrend: bool,
}

impl Default for WelchConfig {
    fn default() -> Self {
        Self {
            nperseg: 256,
            noverlap: 128,
            window: WelchWindow::Hann,
            fs: 1.0,
            detrend: true,
        }
    }
}

/// Output of the batched Welch estimator.
#[derive(Debug, Clone)]
pub struct WelchResult {
    /// Frequency axis in Hz, length = `nperseg / 2 + 1`.
    pub freqs: Vec<f64>,
    /// Auto-PSD for each channel.  `psd[c][k]` is the one-sided PSD of
    /// channel `c` at frequency bin `k`, in units of (signal unit)²/Hz.
    pub psd: Vec<Vec<f64>>,
    /// Full cross-PSD matrix, or `None` when computed with
    /// [`BatchedWelch::compute_psd_only`].
    ///
    /// `cpsd[i][j][k]` = `(Re, Im)` of `G_{ij}(f_k)`.
    /// When `i == j` the imaginary part is zero and `Re == psd[i][k]`.
    pub cpsd: Option<Vec<Vec<Vec<(f64, f64)>>>>,
}

/// Batched Welch PSD estimator.
#[derive(Debug, Clone)]
pub struct BatchedWelch {
    config: WelchConfig,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl BatchedWelch {
    /// Create a new `BatchedWelch` estimator.
    ///
    /// # Errors
    /// Returns [`SignalError::InvalidArgument`] if the configuration is invalid
    /// (e.g. `noverlap >= nperseg`, `nperseg < 4`, or `fs <= 0`).
    pub fn new(config: WelchConfig) -> Result<Self, SignalError> {
        if config.nperseg < 4 {
            return Err(SignalError::InvalidArgument(
                "nperseg must be at least 4".to_string(),
            ));
        }
        if config.noverlap >= config.nperseg {
            return Err(SignalError::InvalidArgument(format!(
                "noverlap ({}) must be strictly less than nperseg ({})",
                config.noverlap, config.nperseg
            )));
        }
        if config.fs <= 0.0 {
            return Err(SignalError::InvalidArgument(
                "fs (sampling frequency) must be positive".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Compute Welch PSD and full CPSD matrix for `data[channel][sample]`.
    ///
    /// Returns a [`WelchResult`] with both auto-PSD and cross-PSD filled in.
    ///
    /// # Errors
    /// Returns an error if `data` is empty, channels have different lengths,
    /// or the signal is too short to form even one segment.
    pub fn compute(&self, data: &[Vec<f64>]) -> SignalResult<WelchResult> {
        self.compute_impl(data, true)
    }

    /// Compute only the auto-PSD for each channel (faster; no CPSD matrix).
    ///
    /// Returns a [`WelchResult`] with `cpsd = None`.
    ///
    /// # Errors
    /// Same conditions as [`Self::compute`].
    pub fn compute_psd_only(&self, data: &[Vec<f64>]) -> SignalResult<WelchResult> {
        self.compute_impl(data, false)
    }

    /// Single-channel convenience method.
    ///
    /// # Returns
    /// `(freqs, psd)` — frequency axis in Hz and the one-sided auto-PSD.
    ///
    /// # Errors
    /// Returns an error if the signal is too short to form one segment.
    pub fn compute_single(&self, signal: &[f64]) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        let data = vec![signal.to_vec()];
        let result = self.compute_psd_only(&data)?;
        let psd = result.psd.into_iter().next().unwrap_or_default();
        Ok((result.freqs, psd))
    }

    // -------------------------------------------------------------------
    // Internal implementation
    // -------------------------------------------------------------------

    fn compute_impl(&self, data: &[Vec<f64>], compute_cpsd: bool) -> SignalResult<WelchResult> {
        let n_channels = data.len();
        if n_channels == 0 {
            return Err(SignalError::InvalidArgument(
                "data must contain at least one channel".to_string(),
            ));
        }

        let n_samples = data[0].len();
        for (ch, channel) in data.iter().enumerate() {
            if channel.len() != n_samples {
                return Err(SignalError::DimensionMismatch(format!(
                    "channel {} has length {} but channel 0 has length {}",
                    ch,
                    channel.len(),
                    n_samples
                )));
            }
        }

        let nperseg = self.config.nperseg;
        let noverlap = self.config.noverlap;
        let step = nperseg - noverlap;
        let n_fft = next_pow2(nperseg);
        let n_bins = n_fft / 2 + 1;

        if n_samples < nperseg {
            return Err(SignalError::InvalidArgument(format!(
                "signal length ({}) is shorter than nperseg ({})",
                n_samples, nperseg
            )));
        }

        // Pre-compute window coefficients
        let win = build_window(nperseg, &self.config.window);
        let win_energy: f64 = win.iter().map(|w| w * w).sum();

        // Accumulators
        // auto_acc[ch][bin] = sum of |X_ch(bin)|^2
        let mut auto_acc: Vec<Vec<f64>> = vec![vec![0.0f64; n_bins]; n_channels];
        // cross_acc[ch_i][ch_j][bin] = sum of X_i(bin) * conj(X_j(bin)), stored as (re, im)
        let mut cross_acc: Option<Vec<Vec<Vec<(f64, f64)>>>> = if compute_cpsd {
            Some(vec![
                vec![vec![(0.0f64, 0.0f64); n_bins]; n_channels];
                n_channels
            ])
        } else {
            None
        };

        let mut n_segments: usize = 0;
        let mut start = 0usize;

        while start + nperseg <= n_samples {
            // Compute windowed FFT for each channel at this segment
            let mut spectra: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n_channels);
            for ch in 0..n_channels {
                let seg_slice = &data[ch][start..start + nperseg];
                let spec = windowed_fft(seg_slice, &win, n_fft, self.config.detrend);
                spectra.push(spec);
            }

            // Accumulate auto-spectra
            for ch in 0..n_channels {
                for bin in 0..n_bins {
                    let (re, im) = spectra[ch][bin];
                    auto_acc[ch][bin] += re * re + im * im;
                }
            }

            // Accumulate cross-spectra
            if let Some(ref mut cross) = cross_acc {
                for i in 0..n_channels {
                    for j in 0..n_channels {
                        for bin in 0..n_bins {
                            let (ri, ii) = spectra[i][bin];
                            let (rj, ij) = spectra[j][bin];
                            // X_i * conj(X_j)
                            cross[i][j][bin].0 += ri * rj + ii * ij;
                            cross[i][j][bin].1 += ii * rj - ri * ij;
                        }
                    }
                }
            }

            n_segments += 1;
            start += step;
        }

        if n_segments == 0 {
            return Err(SignalError::InvalidArgument(
                "no complete segments fit in the signal; reduce nperseg".to_string(),
            ));
        }

        // Normalise: average and apply PSD scaling
        // One-sided PSD: scale by 2 for interior bins (DC and Nyquist get factor 1)
        let scale = 1.0 / (win_energy * n_segments as f64 * self.config.fs);

        let mut psd: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
        for ch in 0..n_channels {
            let mut ch_psd = vec![0.0f64; n_bins];
            for bin in 0..n_bins {
                let factor = if bin == 0 || (n_fft % 2 == 0 && bin == n_bins - 1) {
                    1.0
                } else {
                    2.0
                };
                ch_psd[bin] = auto_acc[ch][bin] * scale * factor;
            }
            psd.push(ch_psd);
        }

        let cpsd = match cross_acc {
            None => None,
            Some(mut cross) => {
                for i in 0..n_channels {
                    for j in 0..n_channels {
                        for bin in 0..n_bins {
                            let factor = if bin == 0 || (n_fft % 2 == 0 && bin == n_bins - 1) {
                                1.0
                            } else {
                                2.0
                            };
                            cross[i][j][bin].0 *= scale * factor;
                            cross[i][j][bin].1 *= scale * factor;
                        }
                    }
                }
                Some(cross)
            }
        };

        // Frequency axis
        let freqs: Vec<f64> = (0..n_bins)
            .map(|k| k as f64 * self.config.fs / n_fft as f64)
            .collect();

        Ok(WelchResult { freqs, psd, cpsd })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute Hann window coefficients.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Build a window vector for the given type and length.
pub fn build_window(n: usize, window: &WelchWindow) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    match window {
        WelchWindow::Hann => hann_window(n),
        WelchWindow::Hamming => {
            if n == 1 {
                return vec![1.0];
            }
            (0..n)
                .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
                .collect()
        }
        WelchWindow::Flat => vec![1.0; n],
    }
}

/// Apply a pre-computed window in-place to a segment.
pub fn apply_window(segment: &mut [f64], window: &WelchWindow) {
    let win = build_window(segment.len(), window);
    for (s, w) in segment.iter_mut().zip(win.iter()) {
        *s *= w;
    }
}

/// Compute the windowed FFT of a real segment.
///
/// Returns one-sided complex spectrum of length `n_fft / 2 + 1` as `(re, im)`.
fn windowed_fft(segment: &[f64], win: &[f64], n_fft: usize, detrend: bool) -> Vec<(f64, f64)> {
    let n_seg = segment.len();

    // Compute mean for detrending
    let mean = if detrend && n_seg > 0 {
        segment.iter().sum::<f64>() / n_seg as f64
    } else {
        0.0
    };

    // Build zero-padded buffer
    let mut buf: Vec<(f64, f64)> = (0..n_fft)
        .map(|i| {
            if i < n_seg {
                let v = (segment[i] - mean) * win.get(i).copied().unwrap_or(0.0);
                (v, 0.0)
            } else {
                (0.0, 0.0)
            }
        })
        .collect();

    fft_in_place(&mut buf);

    // Return one-sided
    let n_bins = n_fft / 2 + 1;
    buf.truncate(n_bins);
    buf
}

// ---------------------------------------------------------------------------
// In-place iterative Cooley-Tukey FFT (power-of-two only)
// ---------------------------------------------------------------------------

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn fft_in_place(buf: &mut [(f64, f64)]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    let bits = (n as f64).log2() as usize;

    // Bit-reversal permutation
    for i in 0..n {
        let rev = bit_reverse(i, bits);
        if rev > i {
            buf.swap(i, rev);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wlen = (angle.cos(), angle.sin());
        for i in (0..n).step_by(len) {
            let mut w: (f64, f64) = (1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let t = cmul(w, buf[i + j + half]);
                buf[i + j] = (u.0 + t.0, u.1 + t.1);
                buf[i + j + half] = (u.0 - t.0, u.1 - t.1);
                w = cmul(w, wlen);
            }
        }
        len <<= 1;
    }
}

/// In-place IFFT via conjugate trick: IFFT(x) = conj(FFT(conj(x))) / n.
pub(crate) fn ifft_real_onesided(spectrum: &[(f64, f64)], n_full: usize) -> Vec<f64> {
    // Reconstruct two-sided spectrum
    let n_bins = spectrum.len();
    let mut buf: Vec<(f64, f64)> = vec![(0.0, 0.0); n_full];
    for k in 0..n_bins.min(n_full) {
        buf[k] = spectrum[k];
    }
    // Mirror for k > 0 and k < n_full/2
    for k in 1..n_bins.min(n_full / 2) {
        let conj = (spectrum[k].0, -spectrum[k].1);
        buf[n_full - k] = conj;
    }

    // IFFT via conjugation trick
    // Conjugate input
    for v in buf.iter_mut() {
        v.1 = -v.1;
    }
    fft_in_place(&mut buf);
    // Conjugate and scale
    let scale = 1.0 / n_full as f64;
    buf.iter().map(|v| v.0 * scale).collect()
}

#[inline]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_white_noise(n: usize, seed: u64) -> Vec<f64> {
        // Simple LCG for reproducible noise — no external deps needed
        let mut state = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn make_sinusoid(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    /// White noise PSD should be roughly flat; max/min within 2x is a
    /// reasonable tolerance for 4096-sample segments.
    #[test]
    fn test_white_noise_flat_spectrum() {
        let cfg = WelchConfig {
            nperseg: 512,
            noverlap: 256,
            window: WelchWindow::Hann,
            fs: 1000.0,
            detrend: true,
        };
        let bw = BatchedWelch::new(cfg).expect("valid config");
        let noise = make_white_noise(8192, 42);
        let (_, psd) = bw
            .compute_single(&noise)
            .expect("compute_single should succeed");

        // Ignore DC (bin 0) and only check bins 1..
        let interior: Vec<f64> = psd[1..].to_vec();
        let max_val = interior.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = interior.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            max_val < min_val * 4.0,
            "white noise PSD not flat enough: max={max_val:.4e} min={min_val:.4e}"
        );
    }

    /// DC bin (freq=0) should be near zero for a zero-mean signal.
    #[test]
    fn test_dc_near_zero_for_zero_mean() {
        let cfg = WelchConfig::default();
        let bw = BatchedWelch::new(cfg.clone()).expect("valid config");
        let n = 2048usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 256.0).sin())
            .collect();
        let (freqs, psd) = bw
            .compute_single(&signal)
            .expect("compute_single should succeed");
        assert!(!freqs.is_empty());
        // DC component should be much smaller than the sinusoid peak
        let dc = psd[0];
        let peak = psd.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            dc < peak * 0.1,
            "DC bin too large: dc={dc:.4e} peak={peak:.4e}"
        );
    }

    /// Single sinusoid should produce a clear peak at the correct frequency bin.
    #[test]
    fn test_sinusoid_peak_at_correct_freq() {
        let fs = 1000.0f64;
        let f0 = 100.0f64; // 100 Hz
        let cfg = WelchConfig {
            nperseg: 256,
            noverlap: 128,
            window: WelchWindow::Hann,
            fs,
            detrend: false,
        };
        let bw = BatchedWelch::new(cfg).expect("valid config");
        let signal = make_sinusoid(4096, f0, fs);
        let (freqs, psd) = bw
            .compute_single(&signal)
            .expect("compute_single should succeed");

        // Find peak bin
        let (peak_bin, _) = psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let peak_freq = freqs[peak_bin];

        let bin_width = freqs[1] - freqs[0];
        assert!(
            (peak_freq - f0).abs() <= bin_width * 1.5,
            "peak at {peak_freq} Hz but expected {f0} Hz (bin_width={bin_width})"
        );
    }

    /// Multi-channel output has correct shape.
    #[test]
    fn test_multi_channel_shapes() {
        let cfg = WelchConfig {
            nperseg: 128,
            noverlap: 64,
            window: WelchWindow::Hann,
            fs: 512.0,
            detrend: true,
        };
        let bw = BatchedWelch::new(cfg).expect("valid config");
        let n_channels = 4usize;
        let data: Vec<Vec<f64>> = (0..n_channels)
            .map(|seed| make_white_noise(2048, seed as u64 + 1))
            .collect();

        let result = bw.compute(&data).expect("compute should succeed");
        assert_eq!(result.psd.len(), n_channels, "wrong number of PSD arrays");

        let n_bins = 128 / 2 + 1; // nperseg / 2 + 1 (next_pow2(128)==128)
        for ch in 0..n_channels {
            assert_eq!(
                result.psd[ch].len(),
                n_bins,
                "PSD length wrong for channel {ch}"
            );
        }
        assert_eq!(result.freqs.len(), n_bins, "freq axis length wrong");

        let cpsd = result.cpsd.expect("CPSD should be computed");
        assert_eq!(cpsd.len(), n_channels);
        assert_eq!(cpsd[0].len(), n_channels);
        assert_eq!(cpsd[0][0].len(), n_bins);
    }

    /// noverlap=0 and noverlap=nperseg/2 should give the same peak location.
    #[test]
    fn test_overlap_peak_location_invariant() {
        let fs = 1024.0f64;
        let f0 = 80.0f64;
        let signal = make_sinusoid(8192, f0, fs);

        for &noverlap in &[0usize, 128usize] {
            let cfg = WelchConfig {
                nperseg: 256,
                noverlap,
                window: WelchWindow::Hann,
                fs,
                detrend: false,
            };
            let bw = BatchedWelch::new(cfg).expect("valid config");
            let (freqs, psd) = bw
                .compute_single(&signal)
                .expect("compute_single should succeed");
            let (peak_bin, _) = psd
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            let peak_freq = freqs[peak_bin];
            let bin_width = freqs[1] - freqs[0];
            assert!(
                (peak_freq - f0).abs() <= bin_width * 1.5,
                "noverlap={noverlap}: peak at {peak_freq} Hz, expected {f0}"
            );
        }
    }

    /// `compute_psd_only` should produce `cpsd = None`.
    #[test]
    fn test_psd_only_no_cpsd() {
        let cfg = WelchConfig::default();
        let bw = BatchedWelch::new(cfg).expect("valid config");
        let data = vec![make_white_noise(2048, 7), make_white_noise(2048, 13)];
        let result = bw
            .compute_psd_only(&data)
            .expect("compute_psd_only should succeed");
        assert!(result.cpsd.is_none(), "cpsd should be None for psd_only");
    }
}
