// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Sperling ride comfort index (Wz) of a measured or simulated acceleration
//! record.
//!
//! The index of Sperling and Helberg is defined on the *discrete spectrum* of
//! the vibration: every spectral line of amplitude `a_i` \[cm/s²\] at the
//! frequency `f_i` \[Hz\] contributes
//!
//! ```text
//! Wz_i = (a_i³ · B(f_i)³)^0.1 = (a_i · B(f_i))^0.3
//! ```
//!
//! and the lines are accumulated through their tenth powers,
//!
//! ```text
//! Wz = (Σ Wz_i¹⁰)^0.1 = (Σ a_i³ · B(f_i)³)^0.1 ,
//! ```
//!
//! where `B(f)` is the frequency weighting of human sensitivity to vertical or
//! lateral vibration (see [`sperling_frequency_weight`]). Only the band
//! 0.5–30 Hz contributes.
//!
//! [`sperling_wz_axis`] evaluates the index from time-domain samples: the record
//! is Hann-windowed and transformed with a real FFT (`oxifft`), the spectrum is
//! split into spectral lines, and each line is assigned the amplitude of the
//! sinusoid that carries its power.

use oxifft::rfft;
use std::f64::consts::PI;

/// Lower edge of the frequency range that contributes to the Sperling index (Hz).
pub const SPERLING_MIN_FREQUENCY_HZ: f64 = 0.5;

/// Upper edge of the frequency range that contributes to the Sperling index (Hz).
pub const SPERLING_MAX_FREQUENCY_HZ: f64 = 30.0;

/// Direction of the vibration assessed by the Sperling index.
///
/// Passengers are less sensitive to lateral than to vertical vibration, which
/// shows in the scale factor of the frequency weighting (0.737 vs 0.588).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SperlingAxis {
    /// Vertical vibration of the car body.
    Vertical,
    /// Lateral vibration of the car body.
    Lateral,
}

/// Frequency weighting `B(f)` of the Sperling index (dimensionless).
///
/// ```text
///               ⎛ 1.911 f² + (0.25 f²)²                       ⎞ ½
/// B(f) = k_axis ⎜ ───────────────────────────────────────────── ⎟
///               ⎝ (1 − 0.277 f²)² + (1.563 f − 0.0368 f³)²     ⎠
/// ```
///
/// with `f` in Hz and `k_axis = 0.588` for vertical, `0.737` for lateral
/// vibration. The curve vanishes at `f = 0`, peaks at about 5 Hz and decays like
/// `1/f` above 10 Hz. For vertical vibration it reproduces, between 1 and 10 Hz
/// to within about 4 %, the weighting implied by the classical piecewise
/// definition (`F(f) = 0.325 f²` up to 5.9 Hz, `400/f²` up to 20 Hz) of the
/// single-frequency formula `Wz = 0.896 (a³ F(f) / f)^0.1`.
pub fn sperling_frequency_weight(frequency_hz: f64, axis: SperlingAxis) -> f64 {
    let f = frequency_hz;
    let f2 = f * f;
    let numerator = 1.911 * f2 + (0.25 * f2).powi(2);
    let denominator = (1.0 - 0.277 * f2).powi(2) + (1.563 * f - 0.0368 * f2 * f).powi(2);
    let axis_scale = match axis {
        SperlingAxis::Vertical => 0.588,
        SperlingAxis::Lateral => 0.737,
    };
    axis_scale * (numerator / denominator).sqrt()
}

/// Sperling ride comfort index (Wz) for vertical vibration.
///
/// Shorthand for [`sperling_wz_axis`] with [`SperlingAxis::Vertical`].
///
/// * `accelerations` — acceleration samples (m/s²).
/// * `sample_rate` — sampling frequency (Hz).
pub fn sperling_wz(accelerations: &[f64], sample_rate: f64) -> f64 {
    sperling_wz_axis(accelerations, sample_rate, SperlingAxis::Vertical)
}

/// Sperling ride comfort index (Wz) of an acceleration record.
///
/// * `accelerations` — equally spaced acceleration samples (m/s²) of the car
///   body along the chosen axis.
/// * `sample_rate` — sampling frequency (Hz).
/// * `axis` — vertical or lateral weighting.
///
/// The mean is removed, the record is multiplied by a (periodic) Hann window
/// and transformed with a real FFT. The one-sided power spectrum is then
/// divided into spectral lines: every run of bins between two successive local
/// minima belongs to one line. A line carries the power
///
/// ```text
/// P = Σ_bins 2|X_k|² / (Σw)²  /  ENBW        (m²/s⁴, ENBW = N Σw² / (Σw)² = 1.5)
/// ```
///
/// at its power-weighted centroid frequency `f`, and enters the index as the
/// sinusoid of the same power, `a = √(2P)` (converted to cm/s²), with
/// `a³ B(f)³`. Dividing by the equivalent noise bandwidth makes the amplitude of
/// an isolated tone exact for *any* position of its frequency relative to the
/// FFT bins: the Hann window spreads a tone over three to five bins, and
/// summing `a³` per bin instead would overestimate Wz by about 2 %. Only lines
/// with a centroid between [`SPERLING_MIN_FREQUENCY_HZ`] and
/// [`SPERLING_MAX_FREQUENCY_HZ`] contribute.
///
/// The index is defined on discrete lines, so tones closer than about two FFT
/// bins merge into one line (their powers add), and for broadband records the
/// value depends on the frequency resolution `sample_rate / N`; compare records
/// of equal length. Values below 2.5 are considered comfortable, above 3.25
/// very uncomfortable.
///
/// Returns `0.0` for fewer than two samples, for a sample rate below `1e-6` Hz
/// and for a record without vibration in the band, and `NaN` if the input
/// contains non-finite values.
pub fn sperling_wz_axis(accelerations: &[f64], sample_rate: f64, axis: SperlingAxis) -> f64 {
    let n = accelerations.len();
    if n < 2 || sample_rate < 1e-6 {
        return 0.0;
    }
    if !sample_rate.is_finite() || accelerations.iter().any(|a| !a.is_finite()) {
        return f64::NAN;
    }

    // Remove the mean and apply the periodic Hann window.
    let count = n as f64;
    let mean = accelerations.iter().sum::<f64>() / count;
    let window: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / count).cos())
        .collect();
    let windowed: Vec<f64> = accelerations
        .iter()
        .zip(&window)
        .map(|(a, w)| (a - mean) * w)
        .collect();
    let sum_w: f64 = window.iter().sum();
    let sum_w2: f64 = window.iter().map(|w| w * w).sum();
    let enbw = count * sum_w2 / (sum_w * sum_w);

    // One-sided spectrum; power[k] is the mean square (m²/s⁴) of the sinusoid
    // that would produce |X_k| at a bin centre.
    let spectrum = rfft(&windowed);
    let power: Vec<f64> = spectrum
        .iter()
        .enumerate()
        .map(|(k, bin)| {
            let self_conjugate = k == 0 || (n.is_multiple_of(2) && k == n / 2);
            let fold = if self_conjugate { 1.0 } else { 2.0 };
            fold * (bin.re * bin.re + bin.im * bin.im) / (sum_w * sum_w)
        })
        .collect();

    // Split the spectrum into lines (peak-to-valley segments) and accumulate
    // Σ a³ B³ over the lines inside the evaluation band.
    let bin_width = sample_rate / count;
    let mut accumulated = 0.0;
    let mut line_start = 0;
    let mut falling = false;
    for k in 0..power.len() {
        let last = k + 1 == power.len();
        let rises_next = !last && power[k + 1] > power[k];
        let falls_next = !last && power[k + 1] < power[k];
        if last || (falling && rises_next) {
            accumulated +=
                line_contribution(&power[line_start..=k], line_start, bin_width, enbw, axis);
            line_start = k + 1;
            falling = false;
        } else if falls_next {
            falling = true;
        }
    }
    accumulated.powf(0.1)
}

/// `a³ B(f)³` of the spectral line formed by `power` (the bins starting at bin
/// `first_bin`); zero if the line has no power or lies outside the band.
fn line_contribution(
    power: &[f64],
    first_bin: usize,
    bin_width: f64,
    enbw: f64,
    axis: SperlingAxis,
) -> f64 {
    let total: f64 = power.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return 0.0;
    }
    let centroid_bin = power
        .iter()
        .enumerate()
        .map(|(i, p)| p * (first_bin + i) as f64)
        .sum::<f64>()
        / total;
    let frequency = centroid_bin * bin_width;
    if !(SPERLING_MIN_FREQUENCY_HZ..=SPERLING_MAX_FREQUENCY_HZ).contains(&frequency) {
        return 0.0;
    }
    // Amplitude of the sinusoid carrying the line power, in cm/s².
    let amplitude_cm = 100.0 * (2.0 * total / enbw).sqrt();
    (amplitude_cm * sperling_frequency_weight(frequency, axis)).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `amplitude · sin(2π f t)` sampled `n` times at `sample_rate`.
    fn tone(amplitude: f64, frequency: f64, sample_rate: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / sample_rate).sin())
            .collect()
    }

    /// Closed-form index of a single sinusoid: `(a³ B³)^0.1`, `a` in cm/s².
    fn single_tone_wz(amplitude_ms2: f64, frequency: f64, axis: SperlingAxis) -> f64 {
        let weighted = 100.0 * amplitude_ms2 * sperling_frequency_weight(frequency, axis);
        weighted.powi(3).powf(0.1)
    }

    fn assert_close(actual: f64, expected: f64, relative: f64) {
        assert!(
            (actual - expected).abs() <= relative * expected.abs(),
            "actual {actual}, expected {expected} (relative tolerance {relative})"
        );
    }

    #[test]
    fn weight_matches_hand_evaluated_values() {
        // f = 5 Hz: numerator 86.8375, denominator 45.44185.
        let vertical = sperling_frequency_weight(5.0, SperlingAxis::Vertical);
        assert_close(vertical, 0.812836, 1e-5);
        let lateral = sperling_frequency_weight(5.0, SperlingAxis::Lateral);
        assert_close(lateral, 1.018810, 1e-5);
        assert_eq!(sperling_frequency_weight(0.0, SperlingAxis::Vertical), 0.0);
    }

    #[test]
    fn weight_has_a_band_pass_shape_and_lateral_scale() {
        let b = |f: f64| sperling_frequency_weight(f, SperlingAxis::Vertical);
        assert!(b(0.5) < b(1.0) && b(1.0) < b(3.0) && b(3.0) < b(5.0));
        assert!(b(5.0) > b(10.0) && b(10.0) > b(20.0) && b(20.0) > b(30.0));
        for f in [0.7, 2.0, 5.0, 13.0, 29.0] {
            let ratio = sperling_frequency_weight(f, SperlingAxis::Lateral) / b(f);
            assert_close(ratio, 0.737 / 0.588, 1e-12);
        }
    }

    #[test]
    fn weight_agrees_with_the_piecewise_single_frequency_formula() {
        // Wz = 0.896 (a³ F(f)/f)^0.1 = (a·B)^0.3 requires
        // B = (0.896¹⁰ F(f)/f)^(1/3) with F = 0.325 f² (≤ 5.9 Hz), 400/f² (≤ 20 Hz).
        for f in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0] {
            let piecewise_f = if f <= 5.9 {
                0.325 * f * f
            } else {
                400.0 / (f * f)
            };
            let reference = (0.896_f64.powi(10) * piecewise_f / f).cbrt();
            let weight = sperling_frequency_weight(f, SperlingAxis::Vertical);
            assert!(
                (weight / reference - 1.0).abs() < 0.04,
                "f = {f} Hz: B = {weight}, piecewise {reference}"
            );
        }
    }

    #[test]
    fn bin_centred_tone_matches_the_closed_form() {
        // 10 s at 200 Hz: 5 Hz is exactly bin 50. Summing a³ over the Hann main
        // lobe bin by bin would give 1.25 A³ and overshoot Wz by 2.3 %.
        let samples = tone(0.5, 5.0, 200.0, 2000);
        let wz = sperling_wz(&samples, 200.0);
        assert_close(wz, single_tone_wz(0.5, 5.0, SperlingAxis::Vertical), 1e-6);
        assert_close(wz, 3.038728, 1e-6);
    }

    #[test]
    fn off_bin_and_odd_length_tones_match_the_closed_form() {
        // 5.37 Hz sits between bins (5 s record, 0.2 Hz resolution).
        let samples = tone(0.4, 5.37, 200.0, 1000);
        let wz = sperling_wz(&samples, 200.0);
        assert_close(wz, single_tone_wz(0.4, 5.37, SperlingAxis::Vertical), 1e-6);

        // Half-bin offset (worst case for scalloping): 5.1 Hz at 0.2 Hz resolution.
        let samples = tone(0.4, 5.1, 200.0, 1000);
        let wz = sperling_wz(&samples, 200.0);
        assert_close(wz, single_tone_wz(0.4, 5.1, SperlingAxis::Vertical), 1e-6);

        // Odd record length and a frequency that is not a multiple of the resolution.
        let samples = tone(0.25, 3.3, 100.0, 777);
        let wz = sperling_wz(&samples, 100.0);
        assert_close(wz, single_tone_wz(0.25, 3.3, SperlingAxis::Vertical), 1e-6);
    }

    #[test]
    fn lateral_axis_uses_the_lateral_weighting() {
        let samples = tone(0.3, 4.0, 100.0, 1000);
        let vertical = sperling_wz_axis(&samples, 100.0, SperlingAxis::Vertical);
        let lateral = sperling_wz_axis(&samples, 100.0, SperlingAxis::Lateral);
        assert_close(lateral / vertical, (0.737_f64 / 0.588).powf(0.3), 1e-6);
        assert_close(
            lateral,
            single_tone_wz(0.3, 4.0, SperlingAxis::Lateral),
            1e-6,
        );
        assert_eq!(sperling_wz(&samples, 100.0), vertical);
    }

    #[test]
    fn index_grows_with_the_third_tenth_power_of_the_amplitude() {
        let mild = tone(0.2, 6.0, 200.0, 1000);
        let rough = tone(0.4, 6.0, 200.0, 1000);
        let ratio = sperling_wz(&rough, 200.0) / sperling_wz(&mild, 200.0);
        assert_close(ratio, 2.0_f64.powf(0.3), 1e-6);
    }

    #[test]
    fn separated_tones_combine_through_the_tenth_power_sum() {
        let low = tone(0.3, 2.0, 200.0, 2000);
        let high = tone(0.2, 8.0, 200.0, 2000);
        let both: Vec<f64> = low.iter().zip(&high).map(|(a, b)| a + b).collect();
        let w1 = single_tone_wz(0.3, 2.0, SperlingAxis::Vertical);
        let w2 = single_tone_wz(0.2, 8.0, SperlingAxis::Vertical);
        let expected = (w1.powi(10) + w2.powi(10)).powf(0.1);
        let wz = sperling_wz(&both, 200.0);
        assert_close(wz, expected, 1e-6);
        assert!(wz > w1.max(w2), "combination must exceed the larger line");
    }

    #[test]
    fn components_outside_the_evaluation_band_are_ignored() {
        // 0.2 Hz (below the band) and 40 Hz (above it) alone give no index: only
        // the rounding noise of the FFT is left, which the 0.1 power lifts to
        // about 1e-4 — several orders of magnitude below the same tone in the band.
        let in_band = sperling_wz(&tone(1.0, 5.0, 100.0, 4000), 100.0);
        assert!(in_band > 3.0, "reference index {in_band}");
        assert!(sperling_wz(&tone(1.0, 0.2, 100.0, 4000), 100.0) < 1e-3);
        assert!(sperling_wz(&tone(1.0, 40.0, 200.0, 2000), 200.0) < 1e-3);

        // Added to a 5 Hz tone they change nothing.
        let base = tone(0.5, 5.0, 200.0, 2000);
        let spurious = tone(2.0, 40.0, 200.0, 2000);
        let mixed: Vec<f64> = base.iter().zip(&spurious).map(|(a, b)| a + b).collect();
        assert_close(sperling_wz(&mixed, 200.0), sperling_wz(&base, 200.0), 1e-6);
    }

    #[test]
    fn constant_offset_is_ignored() {
        let base = tone(0.5, 5.0, 200.0, 1000);
        let offset: Vec<f64> = base.iter().map(|a| a + 9.81).collect();
        assert_close(sperling_wz(&offset, 200.0), sperling_wz(&base, 200.0), 1e-9);
    }

    #[test]
    fn degenerate_inputs() {
        assert_eq!(sperling_wz(&[], 100.0), 0.0);
        assert_eq!(sperling_wz(&[1.0], 100.0), 0.0);
        assert_eq!(sperling_wz(&[0.0; 64], 100.0), 0.0);
        assert_eq!(sperling_wz(&tone(1.0, 5.0, 100.0, 100), 0.0), 0.0);
        assert!(sperling_wz(&[0.0, f64::NAN, 1.0, 0.5], 100.0).is_nan());
        assert!(sperling_wz(&[0.0, 1.0, 0.0, -1.0], f64::INFINITY).is_nan());
    }

    #[test]
    fn broadband_record_gives_a_finite_positive_index() {
        // Deterministic pseudo-noise: sum of many incommensurate tones.
        let samples: Vec<f64> = (0..2048)
            .map(|i| {
                let t = i as f64 / 128.0;
                (1..=40)
                    .map(|j| {
                        0.01 * (2.0 * PI * (0.37 * j as f64 + 0.011 * (j * j) as f64) * t).sin()
                    })
                    .sum()
            })
            .collect();
        let wz = sperling_wz(&samples, 128.0);
        assert!(wz.is_finite() && wz > 0.0, "wz = {wz}");
        let louder: Vec<f64> = samples.iter().map(|a| 3.0 * a).collect();
        assert!(sperling_wz(&louder, 128.0) > wz);
    }
}
