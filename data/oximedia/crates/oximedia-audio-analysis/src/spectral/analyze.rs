//! Main spectral analysis implementation.

use crate::{get_or_compute_window, AnalysisConfig, AnalysisError, Result};
use oxifft::{Complex, Direction, Flags, Plan};
use std::sync::{Arc, Mutex};

/// Spectral analyzer for frequency-domain analysis.
///
/// # Scratch-buffer reuse
///
/// Two `Mutex`-protected scratch buffers (`fft_in` and `fft_out`) are reused
/// across `compute_spectrum` calls to avoid per-frame heap allocation.
/// `Mutex` (rather than `RefCell`) is used so that `SpectralAnalyzer` is
/// `Sync`, enabling it to be shared across rayon threads inside
/// `AudioAnalyzer::analyze`.  The lock is held only for the duration of the
/// FFT computation and released before returning.
///
/// # Window-coefficient sharing
///
/// The window coefficients are stored as an `Arc<Vec<f32>>` and come from
/// the global cache in `crate::get_or_compute_window`.  Multiple
/// `SpectralAnalyzer` instances with the same `(window_type, fft_size)` share
/// the same backing `Vec` allocation.
pub struct SpectralAnalyzer {
    config: AnalysisConfig,
    /// Shared, immutable window coefficients (Arc avoids per-instance copy).
    window: Arc<Vec<f32>>,
    /// Reusable FFT input buffer (windowed complex samples).
    fft_in: Mutex<Vec<Complex<f64>>>,
    /// Reusable FFT output buffer.
    fft_out: Mutex<Vec<Complex<f64>>>,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer with the given configuration.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        // Fetch (or compute once) the window coefficients from the global cache.
        let window = get_or_compute_window(config.window_type, config.fft_size);
        let fft_size = config.fft_size;

        Self {
            config,
            window,
            fft_in: Mutex::new(vec![Complex::new(0.0, 0.0); fft_size]),
            fft_out: Mutex::new(vec![Complex::new(0.0, 0.0); fft_size]),
        }
    }

    /// Perform spectral analysis on audio samples.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<SpectralFeatures> {
        if samples.len() < self.config.fft_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.fft_size,
                got: samples.len(),
            });
        }

        // Compute average spectrum over all frames
        let num_frames = (samples.len() - self.config.fft_size) / self.config.hop_size + 1;
        let mut avg_magnitude = vec![0.0_f32; self.config.fft_size / 2 + 1];
        let mut prev_magnitude: Option<Vec<f32>> = None;
        let mut total_flux = 0.0_f32;
        let mut flux_count = 0usize;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.config.hop_size;
            let end = start + self.config.fft_size;
            if end > samples.len() {
                break;
            }

            let frame_spectrum = self.compute_spectrum(&samples[start..end])?;

            // Accumulate flux
            if let Some(ref prev) = prev_magnitude {
                total_flux += super::flux::spectral_flux(&frame_spectrum, prev);
                flux_count += 1;
            }

            for (i, &mag) in frame_spectrum.iter().enumerate() {
                avg_magnitude[i] += mag;
            }

            prev_magnitude = Some(frame_spectrum);
        }

        // Average the magnitude spectrum
        let actual_frames = num_frames.max(1) as f32;
        for mag in &mut avg_magnitude {
            *mag /= actual_frames;
        }

        // Compute spectral features
        let centroid = super::centroid::spectral_centroid(&avg_magnitude, sample_rate);
        let flatness = super::flatness::spectral_flatness(&avg_magnitude);
        let crest = super::crest::spectral_crest(&avg_magnitude);
        let bandwidth = super::bandwidth::spectral_bandwidth(&avg_magnitude, sample_rate, centroid);
        let rolloff = super::rolloff::spectral_rolloff_85(&avg_magnitude, sample_rate);
        let flux = if flux_count > 0 {
            total_flux / flux_count as f32
        } else {
            0.0
        };

        Ok(SpectralFeatures {
            centroid,
            flatness,
            crest,
            bandwidth,
            rolloff,
            flux,
            magnitude_spectrum: avg_magnitude,
        })
    }

    /// Analyze a single frame for real-time processing.
    pub fn analyze_frame(&self, samples: &[f32], sample_rate: f32) -> Result<SpectralFeatures> {
        let magnitude = self.compute_spectrum(samples)?;

        let centroid = super::centroid::spectral_centroid(&magnitude, sample_rate);
        let flatness = super::flatness::spectral_flatness(&magnitude);
        let crest = super::crest::spectral_crest(&magnitude);
        let bandwidth = super::bandwidth::spectral_bandwidth(&magnitude, sample_rate, centroid);
        let rolloff = super::rolloff::spectral_rolloff_85(&magnitude, sample_rate);

        Ok(SpectralFeatures {
            centroid,
            flatness,
            crest,
            bandwidth,
            rolloff,
            flux: 0.0,
            magnitude_spectrum: magnitude,
        })
    }

    /// Compute magnitude spectrum for a frame.
    ///
    /// Reuses the internal `fft_in` and `fft_out` scratch buffers to avoid
    /// allocating a new `Vec<Complex<f64>>` on every call.  The buffers are
    /// cleared (overwritten) at the start of each call so there is no
    /// risk of stale data from a previous frame.
    fn compute_spectrum(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let fft_size = self.config.fft_size;
        if samples.len() != fft_size {
            return Err(AnalysisError::InvalidInput(format!(
                "Expected {} samples, got {}",
                fft_size,
                samples.len()
            )));
        }

        // Acquire the scratch buffers.  Mutex is used (rather than RefCell) so
        // that SpectralAnalyzer remains Sync and can be shared across rayon threads.
        let mut fft_in = self.fft_in.lock().unwrap_or_else(|e| e.into_inner());
        let mut fft_out = self.fft_out.lock().unwrap_or_else(|e| e.into_inner());

        // Resize if the FFT size changed (e.g. after config reload).
        if fft_in.len() != fft_size {
            fft_in.resize(fft_size, Complex::new(0.0, 0.0));
        }
        if fft_out.len() != fft_size {
            fft_out.resize(fft_size, Complex::new(0.0, 0.0));
        }

        // Fill the input buffer with windowed samples (overwrites any stale data).
        for ((dst, &s), &w) in fft_in
            .iter_mut()
            .zip(samples.iter())
            .zip(self.window.iter())
        {
            *dst = Complex::new(f64::from(s * w), 0.0);
        }

        // Execute the FFT in-place using the persistent Plan API.
        let plan = Plan::<f64>::dft_1d(fft_size, Direction::Forward, Flags::ESTIMATE).ok_or_else(
            || AnalysisError::FftError(format!("Failed to create FFT plan for size {fft_size}")),
        )?;
        plan.execute(&*fft_in, &mut *fft_out);

        // Compute magnitude spectrum (only positive frequencies).
        let magnitude: Vec<f32> = fft_out[..=fft_size / 2]
            .iter()
            .map(|c| c.norm() as f32)
            .collect();

        Ok(magnitude)
    }

    /// Test-visible wrapper around the private `compute_spectrum`.
    ///
    /// Exposes the scratch-buffer reuse path to unit tests without making
    /// `compute_spectrum` part of the public API.
    #[cfg(test)]
    pub fn compute_spectrum_pub(&self, samples: &[f32]) -> Result<Vec<f32>> {
        self.compute_spectrum(samples)
    }
}

/// Spectral features extracted from audio.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid in Hz (center of mass of spectrum)
    pub centroid: f32,
    /// Spectral flatness (0-1, higher = more noise-like)
    pub flatness: f32,
    /// Spectral crest factor (peak-to-average ratio)
    pub crest: f32,
    /// Spectral bandwidth in Hz
    pub bandwidth: f32,
    /// Spectral rolloff frequency in Hz (85% energy threshold)
    pub rolloff: f32,
    /// Spectral flux (change from previous frame)
    pub flux: f32,
    /// Full magnitude spectrum
    pub magnitude_spectrum: Vec<f32>,
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            centroid: 0.0,
            flatness: 0.0,
            crest: 0.0,
            bandwidth: 0.0,
            rolloff: 0.0,
            flux: 0.0,
            magnitude_spectrum: Vec::new(),
        }
    }
}

/// Convert FFT bin index to frequency in Hz.
#[must_use]
pub fn bin_to_frequency(bin: usize, sample_rate: f32, fft_size: usize) -> f32 {
    bin as f32 * sample_rate / fft_size as f32
}

/// Convert frequency to FFT bin index.
#[must_use]
pub fn frequency_to_bin(frequency: f32, sample_rate: f32, fft_size: usize) -> usize {
    ((frequency * fft_size as f32 / sample_rate).round() as usize).min(fft_size / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Item 3: FFT scratch-buffer reuse tests ────────────────────────────────

    /// Calling `compute_spectrum` twice on the same input must return identical
    /// results, confirming that the scratch buffers are properly reset between
    /// calls and that no stale data from one call bleeds into the next.
    #[test]
    fn test_fft_scratch_reuse_identical_to_allocating() {
        let config = AnalysisConfig::default();
        let analyzer = SpectralAnalyzer::new(config.clone());

        // Build a deterministic non-trivial input: 440 Hz sine with full FFT window.
        let sample_rate = 44100.0_f32;
        let samples: Vec<f32> = (0..config.fft_size)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let mag_first = analyzer
            .compute_spectrum_pub(&samples)
            .expect("first compute_spectrum call failed");
        let mag_second = analyzer
            .compute_spectrum_pub(&samples)
            .expect("second compute_spectrum call failed");

        assert_eq!(
            mag_first.len(),
            mag_second.len(),
            "magnitude spectrum lengths differ"
        );
        for (i, (a, b)) in mag_first.iter().zip(mag_second.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "magnitude at bin {i} differs between calls: {a} vs {b}"
            );
        }
    }

    /// Calling `compute_spectrum` on a different input after a previous call
    /// must not contaminate the result with data from the previous frame.
    #[test]
    fn test_fft_scratch_no_cross_contamination() {
        let config = AnalysisConfig::default();
        let analyzer = SpectralAnalyzer::new(config.clone());

        let sample_rate = 44100.0_f32;
        // Frame A: 440 Hz sine.
        let samples_a: Vec<f32> = (0..config.fft_size)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();
        // Frame B: 880 Hz sine (clearly different spectrum).
        let samples_b: Vec<f32> = (0..config.fft_size)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate).sin())
            .collect();

        let mag_a1 = analyzer
            .compute_spectrum_pub(&samples_a)
            .expect("frame A first call failed");
        // Process B to "dirty" the scratch buffers.
        let _ = analyzer
            .compute_spectrum_pub(&samples_b)
            .expect("frame B call failed");
        // Process A again — must match the first A result exactly.
        let mag_a2 = analyzer
            .compute_spectrum_pub(&samples_a)
            .expect("frame A second call failed");

        for (i, (x, y)) in mag_a1.iter().zip(mag_a2.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "bin {i}: A result corrupted by intermediate B frame"
            );
        }
    }

    #[test]
    fn test_spectral_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = SpectralAnalyzer::new(config);

        // Generate 1 second of 440 Hz sine wave
        let sample_rate = 44100.0;
        let duration = 1.0;
        let frequency = 440.0;
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect();

        let result = analyzer
            .analyze(&samples, sample_rate)
            .expect("analysis should succeed");

        // Just verify we get reasonable values
        assert!(result.centroid > 0.0 && result.centroid < sample_rate / 2.0);
        // Flatness should be between 0 and 1
        assert!(result.flatness >= 0.0 && result.flatness <= 1.0);
    }

    #[test]
    fn test_bin_frequency_conversion() {
        let sample_rate = 44100.0;
        let fft_size = 2048;

        let freq = 440.0;
        let bin = frequency_to_bin(freq, sample_rate, fft_size);
        let back = bin_to_frequency(bin, sample_rate, fft_size);

        assert!((freq - back).abs() < sample_rate / fft_size as f32);
    }
}
