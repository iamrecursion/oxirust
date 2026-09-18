//! Utility functions for MIR analysis.

use crate::{MirError, MirResult};
use oxifft::Complex;
use std::f32::consts::PI;

/// Apply Hann window to signal.
#[must_use]
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let factor = 2.0 * PI * i as f32 / (size - 1) as f32;
            0.5 * (1.0 - factor.cos())
        })
        .collect()
}

/// Apply Hamming window to signal.
#[must_use]
#[allow(dead_code)]
pub fn hamming_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let factor = 2.0 * PI * i as f32 / (size - 1) as f32;
            0.54 - 0.46 * factor.cos()
        })
        .collect()
}

/// Apply Blackman window to signal.
#[must_use]
#[allow(dead_code)]
pub fn blackman_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let factor = 2.0 * PI * i as f32 / (size - 1) as f32;
            0.42 - 0.5 * factor.cos() + 0.08 * (2.0 * factor).cos()
        })
        .collect()
}

/// Compute Short-Time Fourier Transform (STFT).
///
/// # Errors
///
/// Returns error if FFT computation fails.
pub fn stft(
    signal: &[f32],
    window_size: usize,
    hop_size: usize,
) -> MirResult<Vec<Vec<Complex<f32>>>> {
    if signal.is_empty() {
        return Err(MirError::InvalidInput("Empty signal for STFT".to_string()));
    }

    let window = hann_window(window_size);

    let num_frames = (signal.len().saturating_sub(window_size)) / hop_size + 1;
    let mut result = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = (start + window_size).min(signal.len());

        if end - start < window_size {
            break;
        }

        let buffer: Vec<Complex<f32>> = signal[start..end]
            .iter()
            .zip(&window)
            .map(|(s, w)| Complex::new(s * w, 0.0))
            .collect();

        let fft_result = oxifft::fft(&buffer);
        result.push(fft_result);
    }

    Ok(result)
}

/// Compute magnitude spectrum from complex FFT output.
#[must_use]
pub fn magnitude_spectrum(fft_output: &[Complex<f32>]) -> Vec<f32> {
    fft_output.iter().map(|c| c.norm()).collect()
}

/// Compute power spectrum from complex FFT output.
#[must_use]
#[allow(dead_code)]
pub fn power_spectrum(fft_output: &[Complex<f32>]) -> Vec<f32> {
    fft_output.iter().map(|c| c.norm_sqr()).collect()
}

/// Compute mel filterbank.
#[must_use]
#[allow(dead_code)]
#[allow(clippy::cast_precision_loss)]
pub fn mel_filterbank(
    num_filters: usize,
    fft_size: usize,
    sample_rate: f32,
    min_freq: f32,
    max_freq: f32,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel(min_freq);
    let mel_max = hz_to_mel(max_freq);

    let mel_points: Vec<f32> = (0..=num_filters + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_filters + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((fft_size + 1) as f32 * f / sample_rate).floor() as usize)
        .collect();

    let mut filterbank = vec![vec![0.0; fft_size / 2 + 1]; num_filters];

    for i in 0..num_filters {
        let start = bin_points[i];
        let center = bin_points[i + 1];
        let end = bin_points[i + 2];

        for (k, fb) in filterbank[i]
            .iter_mut()
            .enumerate()
            .take(center)
            .skip(start)
        {
            if center > start {
                *fb = (k - start) as f32 / (center - start) as f32;
            }
        }

        for (k, fb) in filterbank[i].iter_mut().enumerate().take(end).skip(center) {
            if end > center {
                *fb = (end - k) as f32 / (end - center) as f32;
            }
        }
    }

    filterbank
}

/// Convert Hz to Mel scale.
#[must_use]
#[allow(dead_code)]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale to Hz.
#[must_use]
#[allow(dead_code)]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Compute autocorrelation using FFT.
///
/// # Errors
///
/// Returns error if FFT computation fails.
pub fn autocorrelation(signal: &[f32]) -> MirResult<Vec<f32>> {
    if signal.is_empty() {
        return Err(MirError::InvalidInput(
            "Empty signal for autocorrelation".to_string(),
        ));
    }

    let n = signal.len();
    let fft_size = n.next_power_of_two() * 2;

    // Zero-padded signal
    let buffer: Vec<Complex<f32>> = signal
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    // Forward FFT
    let mut fft_result = oxifft::fft(&buffer);

    // Compute power spectrum
    for x in &mut fft_result {
        let mag_sq = x.norm_sqr();
        *x = Complex::new(mag_sq, 0.0);
    }

    // Inverse FFT
    let ifft_result = oxifft::ifft(&fft_result);

    // Normalize and extract real part
    #[allow(clippy::cast_precision_loss)]
    let scale = fft_size as f32;
    let result: Vec<f32> = ifft_result[..n].iter().map(|c| c.re / scale).collect();

    Ok(result)
}

/// Compute mean of a vector.
#[must_use]
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute standard deviation of a vector.
#[must_use]
pub fn std_dev(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Normalize vector to [0, 1] range.
#[must_use]
pub fn normalize(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < f32::EPSILON {
        return vec![0.5; values.len()];
    }

    values
        .iter()
        .map(|&v| (v - min_val) / (max_val - min_val))
        .collect()
}

/// Find peaks in a signal.
#[must_use]
pub fn find_peaks(signal: &[f32], min_distance: usize) -> Vec<usize> {
    if signal.len() < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            // Check if this peak is far enough from the last one
            if peaks.is_empty() || i - peaks[peaks.len() - 1] >= min_distance {
                peaks.push(i);
            } else if signal[i] > signal[peaks[peaks.len() - 1]] {
                // Replace last peak if this one is higher
                peaks.pop();
                peaks.push(i);
            }
        }
    }

    peaks
}

/// Compute median of a vector.
#[must_use]
pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Downsample signal by integer factor.
#[must_use]
#[allow(dead_code)]
pub fn downsample(signal: &[f32], factor: usize) -> Vec<f32> {
    if factor == 0 || signal.is_empty() {
        return Vec::new();
    }

    signal.iter().step_by(factor).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_median() {
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_normalize() {
        let normalized = normalize(&[0.0, 5.0, 10.0]);
        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_peaks() {
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let peaks = find_peaks(&signal, 1);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
    }

    #[test]
    fn test_autocorrelation() {
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = autocorrelation(&signal);
        assert!(result.is_ok());
        let acf = result.expect("should succeed in test");
        assert_eq!(acf.len(), signal.len());
    }
}
