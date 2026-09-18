// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! FFT based building blocks on top of the pure-Rust `oxifft` crate.
//!
//! The algorithms of this crate whose direct formulation is quadratic in the
//! length of the data (correlation functions of long molecular-dynamics
//! series, for instance) call into this module instead of looping over all
//! pairs of samples.

use crate::compensated::neumaier_sum;
use oxifft::Complex;

/// Autocovariance sums of `x` about its mean, computed with the Wiener-Khinchin
/// theorem.
///
/// Returns `r[0..=last]` with `last = min(max_lag, x.len() - 1)` and
///
/// `r[k] = Σ_{i=0}^{n-1-k} (x_i - x̄)(x_{i+k} - x̄)`,
///
/// the raw sums over the `n - k` available pairs (no `1/n` or `1/(n - k)`
/// factor), where `x̄` is the mean of `x`; `r[0]` is the sum of squared
/// deviations.  The centred series is zero-padded to a power of two
/// `m >= 2n - 1`, so that the circular correlation of the padded sequence is
/// the linear one, transformed with a real FFT, and the squared magnitudes of
/// the spectrum are transformed back.  The cost is `O(n log n)`, against the
/// `O(n * (max_lag + 1))` of the direct sums, and the result agrees with them to
/// rounding error relative to `r[0]`.
///
/// Returns an empty vector for an empty series.
pub(crate) fn autocorrelation_fft(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        // A single sample is its own mean.
        return vec![0.0];
    }
    let last_lag = max_lag.min(n - 1);
    let mean = neumaier_sum(x) / n as f64;
    let padded_len = (2 * n - 1).next_power_of_two();
    let mut padded = vec![0.0_f64; padded_len];
    for (slot, &value) in padded.iter_mut().zip(x) {
        *slot = value - mean;
    }
    let spectrum = oxifft::rfft(&padded);
    let power: Vec<Complex<f64>> = spectrum
        .iter()
        .map(|bin| Complex::new(bin.re * bin.re + bin.im * bin.im, 0.0))
        .collect();
    // `irfft` carries the 1/m of the inverse transform.
    let mut correlation = oxifft::irfft(&power, padded_len);
    correlation.truncate(last_lag + 1);
    correlation
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random series in `[-1, 1)` (SplitMix64).
    fn pseudo_random_series(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    /// The direct `O(n^2)` autocovariance sums about the mean.
    fn direct_sums(x: &[f64], max_lag: usize) -> Vec<f64> {
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
        (0..=max_lag.min(n - 1))
            .map(|lag| (0..n - lag).map(|i| centred[i] * centred[i + lag]).sum())
            .collect()
    }

    #[test]
    fn matches_the_direct_sums_on_a_random_series() {
        let x = pseudo_random_series(1000, 7);
        let fast = autocorrelation_fft(&x, 999);
        let direct = direct_sums(&x, 999);
        assert_eq!(fast.len(), 1000);
        let scale = direct[0];
        for (lag, (a, b)) in fast.iter().zip(&direct).enumerate() {
            assert!(
                (a - b).abs() <= 1e-10 * scale,
                "lag {lag}: fft {a}, direct {b}"
            );
        }
    }

    #[test]
    fn matches_the_direct_sums_for_awkward_lengths() {
        // Lengths below, at and just above powers of two, odd and prime ones.
        for n in [
            2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 64, 97, 100, 127, 128, 129, 997,
        ] {
            let x = pseudo_random_series(n, n as u64);
            let fast = autocorrelation_fft(&x, n);
            let direct = direct_sums(&x, n);
            assert_eq!(fast.len(), n, "n = {n}");
            let scale = direct[0].max(f64::MIN_POSITIVE);
            for (lag, (a, b)) in fast.iter().zip(&direct).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12 * scale,
                    "n = {n}, lag {lag}: fft {a}, direct {b}"
                );
            }
        }
    }

    #[test]
    fn max_lag_truncates_and_is_clamped_to_the_series_length() {
        let x = pseudo_random_series(50, 3);
        let full = autocorrelation_fft(&x, 49);
        assert_eq!(autocorrelation_fft(&x, 0).len(), 1);
        assert_eq!(autocorrelation_fft(&x, 10).len(), 11);
        assert_eq!(autocorrelation_fft(&x, 49).len(), 50);
        assert_eq!(autocorrelation_fft(&x, 10_000).len(), 50);
        assert_eq!(autocorrelation_fft(&x, 10), full[..11].to_vec());
    }

    #[test]
    fn removes_the_mean_before_correlating() {
        let x = pseudo_random_series(256, 11);
        let shifted: Vec<f64> = x.iter().map(|v| v + 1.0e6).collect();
        let a = autocorrelation_fft(&x, 255);
        let b = autocorrelation_fft(&shifted, 255);
        let scale = a[0];
        for (lag, (u, v)) in a.iter().zip(&b).enumerate() {
            // The shift costs about 1e-10 of absolute precision per sample.
            assert!((u - v).abs() <= 1e-8 * scale, "lag {lag}: {u} vs {v}");
        }
    }

    #[test]
    fn zero_lag_is_the_sum_of_squared_deviations() {
        let x = [1.0, 2.0, 4.0, 8.0];
        let mean = 3.75;
        let expected: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
        let r = autocorrelation_fft(&x, 3);
        assert!((r[0] - expected).abs() < 1e-12, "{} vs {expected}", r[0]);
    }

    #[test]
    fn trivial_series_have_trivial_correlations() {
        assert!(autocorrelation_fft(&[], 5).is_empty());
        assert_eq!(autocorrelation_fft(&[3.5], 5), vec![0.0]);
        let constant = autocorrelation_fft(&[2.0; 17], 16);
        assert_eq!(constant.len(), 17);
        assert!(constant.iter().all(|r| r.abs() < 1e-28), "{constant:?}");
    }

    #[test]
    fn a_long_series_is_handled_in_log_linear_time() {
        // 2^16 samples: the direct sums would need 2e9 products for the full
        // lag range.  Compare a few lags with their O(n) definition.
        let n = 1 << 16;
        let x = pseudo_random_series(n, 99);
        let r = autocorrelation_fft(&x, n - 1);
        assert_eq!(r.len(), n);
        let mean = x.iter().sum::<f64>() / n as f64;
        for lag in [0, 1, 2, 17, 1024, n - 2, n - 1] {
            let direct: f64 = (0..n - lag)
                .map(|i| (x[i] - mean) * (x[i + lag] - mean))
                .sum();
            assert!(
                (r[lag] - direct).abs() <= 1e-10 * r[0],
                "lag {lag}: fft {}, direct {direct}",
                r[lag]
            );
        }
    }
}
