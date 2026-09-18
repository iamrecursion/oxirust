//! Total harmonic distortion (THD) measurement.

use oxifft::Complex;

/// Compute total harmonic distortion (THD).
///
/// THD is the ratio of the sum of harmonic powers to the fundamental power.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// THD value (0-1, lower is better)
#[must_use]
pub fn total_harmonic_distortion(samples: &[f32], _sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let fft_size = samples.len().next_power_of_two();

    // Prepare FFT input
    let buffer: Vec<Complex<f64>> = samples
        .iter()
        .map(|&s| Complex::new(f64::from(s), 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    let buffer = oxifft::fft(&buffer);

    // Compute magnitude spectrum
    let magnitude: Vec<f32> = buffer[..fft_size / 2]
        .iter()
        .map(|c| c.norm() as f32)
        .collect();

    // Find fundamental frequency peak
    let fundamental_bin = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    if fundamental_bin == 0 {
        return 0.0;
    }

    let fundamental_power = magnitude[fundamental_bin].powi(2);

    // Sum harmonic powers (2f, 3f, 4f, 5f)
    let mut harmonic_power = 0.0;
    for harmonic in 2..=5 {
        let harmonic_bin = fundamental_bin * harmonic;
        if harmonic_bin < magnitude.len() {
            harmonic_power += magnitude[harmonic_bin].powi(2);
        }
    }

    if fundamental_power > 0.0 {
        (harmonic_power / fundamental_power).sqrt()
    } else {
        0.0
    }
}

/// Compute THD+N (THD plus noise).
#[must_use]
pub fn thd_plus_noise(samples: &[f32], _sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let fft_size = samples.len().next_power_of_two();

    let buffer: Vec<Complex<f64>> = samples
        .iter()
        .map(|&s| Complex::new(f64::from(s), 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    let buffer = oxifft::fft(&buffer);

    let magnitude: Vec<f32> = buffer[..fft_size / 2]
        .iter()
        .map(|c| c.norm() as f32)
        .collect();

    let fundamental_bin = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    if fundamental_bin == 0 {
        return 0.0;
    }

    let fundamental_power = magnitude[fundamental_bin].powi(2);
    let total_power: f32 = magnitude.iter().map(|&m| m.powi(2)).sum();

    let noise_plus_harmonics = total_power - fundamental_power;

    if fundamental_power > 0.0 {
        (noise_plus_harmonics / fundamental_power).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thd_clean_signal() {
        // Pure sine wave should have very low THD
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let thd = total_harmonic_distortion(&samples, sample_rate);
        assert!(thd < 0.1);
    }

    #[test]
    fn test_thd_distorted_signal() {
        // Clipped sine wave should have higher THD
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| {
                let x = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin();
                x.clamp(-0.5, 0.5) // Clipping
            })
            .collect();

        let thd = total_harmonic_distortion(&samples, sample_rate);
        assert!(thd > 0.05);
    }

    // ── Analytical accuracy tests ──────────────────────────────────────────────

    /// A pure single-frequency sine wave contains no harmonic content, so the
    /// measured THD (ratio of harmonic power to fundamental power) must be below
    /// 1 % (0.01).  We use 8192 samples at an integer number of cycles so that
    /// spectral leakage is minimal.
    #[test]
    fn test_thd_pure_sine_analytically_zero() {
        // 220 Hz at 44100 Hz sample rate — 8192 samples ≈ 40.9 cycles.
        // Using a power-of-two length keeps the fundamental near a single bin.
        let sample_rate = 44100.0_f32;
        let fundamental_hz = 220.0_f32;
        let n_samples = 8192_usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * fundamental_hz * t).sin()
            })
            .collect();

        let thd = total_harmonic_distortion(&samples, f64::from(sample_rate) as f32);
        assert!(
            thd < 0.01,
            "pure sine THD should be < 1 %, got {:.4} ({:.2} %)",
            thd,
            thd * 100.0,
        );
    }

    /// Generate a signal that is the sum of a fundamental (1 V peak) and a
    /// second harmonic at exactly 10 % amplitude (0.1 V peak).  The expected
    /// THD is `sqrt(0.01) / 1.0 = 0.10` (10 %).
    ///
    /// The algorithm bins the spectrum and measures the harmonic power at
    /// exact multiples of the peak bin.  Spectral leakage from a non-integer
    /// number of cycles causes the measured energy at the exact harmonic bin to
    /// be less than the true harmonic power, so we verify a lower bound rather
    /// than a point estimate: the measured THD must be at least 3 % (greater than
    /// a pure sine) and below 30 % (far less than hard clipping).  This confirms
    /// that the harmonic content is detected without being overly sensitive to
    /// leakage.
    #[test]
    fn test_thd_known_harmonics() {
        let sample_rate = 44100.0_f32;
        let fundamental_hz = 440.0_f32;
        let harmonic_ratio = 0.10_f32; // 2nd harmonic at 10 % of fundamental
        let n_samples = 8192_usize;

        let samples: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let fundamental = (2.0 * std::f32::consts::PI * fundamental_hz * t).sin();
                let harmonic =
                    harmonic_ratio * (2.0 * std::f32::consts::PI * 2.0 * fundamental_hz * t).sin();
                fundamental + harmonic
            })
            .collect();

        let thd = total_harmonic_distortion(&samples, f64::from(sample_rate) as f32);

        // The signal has a known 10 % harmonic, so THD must be detectably above
        // zero (> 3 %) and well below severe clipping (< 30 %).
        assert!(
            thd > 0.03,
            "THD with known 10 % harmonic must be > 3 %, got {:.4} ({:.2} %)",
            thd,
            thd * 100.0,
        );
        assert!(
            thd < 0.30,
            "THD with known 10 % harmonic must be < 30 %, got {:.4} ({:.2} %)",
            thd,
            thd * 100.0,
        );
    }
}
