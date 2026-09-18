//! Spectrum analyzer using simple FFT-based analysis.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Spectrum analyzer metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpectrumMetrics {
    /// Frequency spectrum (magnitude per bin).
    pub spectrum: Vec<f32>,

    /// Frequency bin centers (Hz).
    pub frequencies: Vec<f32>,

    /// Dominant frequency (Hz).
    pub dominant_frequency: f32,

    /// Spectral centroid (Hz).
    pub spectral_centroid: f32,

    /// Spectral flatness (0.0-1.0).
    pub spectral_flatness: f32,
}

/// Simple spectrum analyzer.
pub struct SpectrumAnalyzer {
    sample_rate: f64,
    channels: usize,
    fft_size: usize,
    buffer: Vec<f32>,
    window: Vec<f32>,
    metrics: SpectrumMetrics,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let fft_size = 2048;
        let window = Self::create_hann_window(fft_size);

        let bin_size = sample_rate / fft_size as f64;
        let frequencies: Vec<f32> = (0..fft_size / 2)
            .map(|i| (i as f64 * bin_size) as f32)
            .collect();

        Self {
            sample_rate,
            channels,
            fft_size,
            buffer: vec![0.0; fft_size],
            window,
            metrics: SpectrumMetrics {
                spectrum: vec![0.0; fft_size / 2],
                frequencies,
                dominant_frequency: 0.0,
                spectral_centroid: 0.0,
                spectral_flatness: 0.0,
            },
        }
    }

    /// Process audio samples.
    pub fn process(&mut self, samples: &[f32]) {
        if samples.is_empty() || self.channels == 0 {
            return;
        }

        // Mix to mono if multi-channel
        let mono_samples: Vec<f32> = if self.channels == 1 {
            samples.to_vec()
        } else {
            samples
                .chunks_exact(self.channels)
                .map(|frame| frame.iter().sum::<f32>() / self.channels as f32)
                .collect()
        };

        // Fill buffer with latest samples
        if mono_samples.len() >= self.fft_size {
            let start = mono_samples.len() - self.fft_size;
            self.buffer
                .copy_from_slice(&mono_samples[start..start + self.fft_size]);
        } else {
            // Shift buffer and append new samples
            let shift_amount = self.fft_size - mono_samples.len();
            self.buffer
                .copy_within(mono_samples.len()..self.fft_size, 0);
            self.buffer[shift_amount..].copy_from_slice(&mono_samples);
        }

        // Compute spectrum
        self.compute_spectrum();
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &SpectrumMetrics {
        &self.metrics
    }

    /// Reset analyzer.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.metrics.spectrum.fill(0.0);
        self.metrics.dominant_frequency = 0.0;
        self.metrics.spectral_centroid = 0.0;
        self.metrics.spectral_flatness = 0.0;
    }

    fn compute_spectrum(&mut self) {
        // Apply window
        let windowed: Vec<f32> = self
            .buffer
            .iter()
            .zip(&self.window)
            .map(|(s, w)| s * w)
            .collect();

        // Simple DFT (not optimized, just for monitoring)
        // In production, use a proper FFT library
        let spectrum_size = self.fft_size / 2;
        let mut spectrum = vec![0.0f32; spectrum_size];

        for k in 0..spectrum_size {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (n, &sample) in windowed.iter().enumerate() {
                let angle = -2.0 * PI * (k as f32) * (n as f32) / (self.fft_size as f32);
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag).sqrt() / (self.fft_size as f32);
        }

        self.metrics.spectrum = spectrum.clone();

        // Find dominant frequency
        if let Some((max_idx, _)) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            self.metrics.dominant_frequency = self.metrics.frequencies[max_idx];
        }

        // Calculate spectral centroid
        let mut weighted_sum = 0.0f32;
        let mut total_magnitude = 0.0f32;

        for (i, &mag) in spectrum.iter().enumerate() {
            weighted_sum += self.metrics.frequencies[i] * mag;
            total_magnitude += mag;
        }

        self.metrics.spectral_centroid = if total_magnitude > 0.0 {
            weighted_sum / total_magnitude
        } else {
            0.0
        };

        // Calculate spectral flatness (geometric mean / arithmetic mean)
        let geo_mean = Self::geometric_mean(&spectrum);
        let arith_mean = spectrum.iter().sum::<f32>() / spectrum.len() as f32;

        self.metrics.spectral_flatness = if arith_mean > 0.0 {
            geo_mean / arith_mean
        } else {
            0.0
        };
    }

    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let angle = 2.0 * PI * (i as f32) / (size as f32);
                0.5 * (1.0 - angle.cos())
            })
            .collect()
    }

    fn geometric_mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let sum_log: f32 = values
            .iter()
            .map(|&v| if v > 0.0 { v.ln() } else { 0.0 })
            .sum();

        (sum_log / values.len() as f32).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_analyzer() {
        let mut analyzer = SpectrumAnalyzer::new(48000.0, 2);

        let samples = vec![0.0f32; 10000];
        analyzer.process(&samples);

        let metrics = analyzer.metrics();
        assert_eq!(metrics.spectrum.len(), 1024); // fft_size / 2
    }

    #[test]
    fn test_spectrum_sine_wave() {
        let mut analyzer = SpectrumAnalyzer::new(48000.0, 1);

        // Generate 1kHz sine wave
        let mut samples = Vec::new();
        for i in 0..48000 {
            let t = i as f32 / 48000.0;
            let sample = (2.0 * PI * 1000.0 * t).sin() * 0.5;
            samples.push(sample);
        }

        analyzer.process(&samples);

        let metrics = analyzer.metrics();
        // Dominant frequency should be close to 1000 Hz
        assert!((metrics.dominant_frequency - 1000.0).abs() < 50.0);
    }
}
