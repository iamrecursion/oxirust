#![allow(dead_code)]
//! Spectral contrast feature extraction for music analysis.
//!
//! Spectral contrast measures the difference between spectral peaks and
//! valleys in each sub-band, providing a compact descriptor that is useful
//! for genre classification, instrument recognition, and timbre analysis.

use std::f32::consts::PI;

/// Configuration for spectral contrast extraction.
#[derive(Debug, Clone)]
pub struct SpectralContrastConfig {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples.
    pub hop_size: usize,
    /// Number of sub-bands to split the spectrum into.
    pub n_bands: usize,
    /// Fraction of bins used for peak / valley estimation (0.0 .. 0.5).
    pub alpha: f32,
}

impl Default for SpectralContrastConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            window_size: 2048,
            hop_size: 512,
            n_bands: 6,
            alpha: 0.2,
        }
    }
}

/// Spectral contrast for a single frame.
#[derive(Debug, Clone)]
pub struct FrameContrast {
    /// Peak values for each sub-band (dB-like scale).
    pub peaks: Vec<f32>,
    /// Valley values for each sub-band.
    pub valleys: Vec<f32>,
    /// Contrast = peak - valley for each sub-band.
    pub contrast: Vec<f32>,
}

/// Result of spectral contrast analysis over an entire audio buffer.
#[derive(Debug, Clone)]
pub struct SpectralContrastResult {
    /// Per-frame contrast data.
    pub frames: Vec<FrameContrast>,
    /// Mean contrast across all frames for each sub-band.
    pub mean_contrast: Vec<f32>,
    /// Standard deviation of contrast for each sub-band.
    pub std_contrast: Vec<f32>,
    /// Number of sub-bands used.
    pub n_bands: usize,
}

/// Spectral contrast extractor.
pub struct SpectralContrastExtractor {
    config: SpectralContrastConfig,
}

impl SpectralContrastExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: SpectralContrastConfig) -> Self {
        Self { config }
    }

    /// Extract spectral contrast features from a mono audio buffer.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn extract(&self, samples: &[f32]) -> SpectralContrastResult {
        let n = self.config.window_size;
        let hop = self.config.hop_size;
        let n_bands = self.config.n_bands;

        let mut frames = Vec::new();
        let mut pos = 0;
        while pos + n <= samples.len() {
            let frame = &samples[pos..pos + n];
            let mag = simple_magnitude_spectrum(frame, n);
            let fc = self.compute_frame_contrast(&mag, n_bands);
            frames.push(fc);
            pos += hop;
        }

        let (mean_contrast, std_contrast) = aggregate_contrast(&frames, n_bands);

        SpectralContrastResult {
            frames,
            mean_contrast,
            std_contrast,
            n_bands,
        }
    }

    /// Compute contrast for a single magnitude spectrum.
    #[allow(clippy::cast_precision_loss)]
    fn compute_frame_contrast(&self, magnitudes: &[f32], n_bands: usize) -> FrameContrast {
        let half = magnitudes.len();
        let band_size = half / n_bands;
        let alpha_count = ((band_size as f32 * self.config.alpha) as usize).max(1);

        let mut peaks = Vec::with_capacity(n_bands);
        let mut valleys = Vec::with_capacity(n_bands);
        let mut contrast = Vec::with_capacity(n_bands);

        for b in 0..n_bands {
            let start = b * band_size;
            let end = if b == n_bands - 1 {
                half
            } else {
                start + band_size
            };
            let mut band: Vec<f32> = magnitudes[start..end].to_vec();
            band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let valley = mean_of_slice(&band[..alpha_count]);
            let peak = mean_of_slice(&band[band.len().saturating_sub(alpha_count)..]);

            let valley_db = to_db(valley);
            let peak_db = to_db(peak);

            peaks.push(peak_db);
            valleys.push(valley_db);
            contrast.push(peak_db - valley_db);
        }

        FrameContrast {
            peaks,
            valleys,
            contrast,
        }
    }
}

/// Aggregate mean and std of contrast across frames.
#[allow(clippy::cast_precision_loss)]
fn aggregate_contrast(frames: &[FrameContrast], n_bands: usize) -> (Vec<f32>, Vec<f32>) {
    if frames.is_empty() {
        return (vec![0.0; n_bands], vec![0.0; n_bands]);
    }
    let n = frames.len() as f32;
    let mut mean = vec![0.0_f32; n_bands];
    for f in frames {
        for (i, &c) in f.contrast.iter().enumerate() {
            if i < n_bands {
                mean[i] += c;
            }
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    let mut std_dev = vec![0.0_f32; n_bands];
    for f in frames {
        for (i, &c) in f.contrast.iter().enumerate() {
            if i < n_bands {
                let d = c - mean[i];
                std_dev[i] += d * d;
            }
        }
    }
    for s in &mut std_dev {
        *s = (*s / n).sqrt();
    }

    (mean, std_dev)
}

/// Simplified magnitude spectrum (DFT of first N/2+1 bins).
#[allow(clippy::cast_precision_loss)]
fn simple_magnitude_spectrum(frame: &[f32], n: usize) -> Vec<f32> {
    let half = n / 2 + 1;
    let mut mags = vec![0.0_f32; half];
    let n_f = n as f32;
    for (k, mag) in mags.iter_mut().enumerate() {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for (i, &sample) in frame.iter().enumerate().take(n) {
            let angle = 2.0 * PI * k as f32 * i as f32 / n_f;
            re += sample * angle.cos();
            im -= sample * angle.sin();
        }
        *mag = (re * re + im * im).sqrt();
    }
    mags
}

/// Convert a linear magnitude to a dB-like scale.
fn to_db(val: f32) -> f32 {
    20.0 * (val.max(1e-10)).log10()
}

/// Compute the arithmetic mean of a slice of `f32`.
#[allow(clippy::cast_precision_loss)]
fn mean_of_slice(s: &[f32]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    s.iter().sum::<f32>() / s.len() as f32
}

/// Compute the spectral flatness of a magnitude spectrum.
///
/// Flatness = geometric mean / arithmetic mean. Values near 1.0 indicate
/// noise-like spectra; values near 0.0 indicate tonal spectra.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn spectral_flatness(magnitudes: &[f32]) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }
    let n = magnitudes.len() as f32;
    let log_sum: f32 = magnitudes.iter().map(|&m| (m.max(1e-10)).ln()).sum();
    let geo_mean = (log_sum / n).exp();
    let arith_mean = magnitudes.iter().sum::<f32>() / n;
    if arith_mean < 1e-12 {
        return 0.0;
    }
    (geo_mean / arith_mean).min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sr: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_extract_silence() {
        let ext = SpectralContrastExtractor::new(SpectralContrastConfig::default());
        let silence = vec![0.0_f32; 4096];
        let result = ext.extract(&silence);
        assert_eq!(result.n_bands, 6);
        assert!(!result.frames.is_empty());
    }

    #[test]
    fn test_extract_empty() {
        let ext = SpectralContrastExtractor::new(SpectralContrastConfig::default());
        let result = ext.extract(&[]);
        assert!(result.frames.is_empty());
        assert_eq!(result.mean_contrast.len(), 6);
    }

    #[test]
    fn test_extract_sine_tone() {
        // Use a small window and low sample rate to keep the O(N²) DFT fast.
        let ext = SpectralContrastExtractor::new(SpectralContrastConfig {
            sample_rate: 8000.0,
            window_size: 256,
            hop_size: 128,
            ..SpectralContrastConfig::default()
        });
        // 0.5 s at 8 kHz = 4000 samples; only a few frames processed.
        let tone = sine_wave(1000.0, 8000.0, 4000);
        let result = ext.extract(&tone);
        assert!(!result.frames.is_empty());
        // A pure tone should produce high contrast in at least one band.
        let max_contrast = result.mean_contrast.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            max_contrast > 0.0,
            "Expected some contrast, got {max_contrast}"
        );
    }

    #[test]
    fn test_frame_contrast_band_count() {
        let ext = SpectralContrastExtractor::new(SpectralContrastConfig {
            n_bands: 4,
            ..SpectralContrastConfig::default()
        });
        let tone = sine_wave(440.0, 44100.0, 4096);
        let result = ext.extract(&tone);
        for f in &result.frames {
            assert_eq!(f.peaks.len(), 4);
            assert_eq!(f.valleys.len(), 4);
            assert_eq!(f.contrast.len(), 4);
        }
    }

    #[test]
    fn test_contrast_equals_peak_minus_valley() {
        let ext = SpectralContrastExtractor::new(SpectralContrastConfig::default());
        let samples = sine_wave(880.0, 44100.0, 4096);
        let result = ext.extract(&samples);
        for frame in &result.frames {
            for i in 0..frame.contrast.len() {
                let expected = frame.peaks[i] - frame.valleys[i];
                assert!(
                    (frame.contrast[i] - expected).abs() < 1e-4,
                    "contrast mismatch at band {i}"
                );
            }
        }
    }

    #[test]
    fn test_to_db_unity() {
        let db = to_db(1.0);
        assert!((db - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_to_db_ten() {
        let db = to_db(10.0);
        assert!((db - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_mean_of_slice_basic() {
        let vals = [1.0, 2.0, 3.0, 4.0];
        assert!((mean_of_slice(&vals) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_of_slice_empty() {
        let vals: [f32; 0] = [];
        assert!((mean_of_slice(&vals) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectral_flatness_pure_tone() {
        // Pure tone should have low flatness.
        let mags = sine_wave(440.0, 44100.0, 64)
            .iter()
            .map(|s| s.abs())
            .collect::<Vec<_>>();
        let flat = spectral_flatness(&mags);
        assert!(flat < 1.0, "pure-tone flatness should be < 1, got {flat}");
    }

    #[test]
    fn test_spectral_flatness_empty() {
        assert!((spectral_flatness(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_default() {
        let cfg = SpectralContrastConfig::default();
        assert_eq!(cfg.window_size, 2048);
        assert_eq!(cfg.n_bands, 6);
        assert!((cfg.alpha - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aggregate_contrast_empty_frames() {
        let (mean, std) = aggregate_contrast(&[], 4);
        assert_eq!(mean.len(), 4);
        assert_eq!(std.len(), 4);
        assert!(mean.iter().all(|&v| v == 0.0));
    }
}
