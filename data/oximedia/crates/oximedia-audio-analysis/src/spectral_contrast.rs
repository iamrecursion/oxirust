#![allow(dead_code)]
//! Spectral contrast analysis for audio characterisation.
//!
//! Spectral contrast measures the difference between spectral peaks and
//! valleys in sub-bands, providing a robust descriptor for music genre
//! classification, timbre analysis, and content-based audio retrieval.
//! The implementation follows the octave-band approach described in the
//! literature, with configurable band count and neighbourhood size.

use std::f64::consts::PI;

/// Number of default octave sub-bands (covers roughly 100 Hz to 11 kHz at 44.1 kHz).
const DEFAULT_BAND_COUNT: usize = 6;

/// Configuration for spectral contrast analysis.
#[derive(Debug, Clone)]
pub struct SpectralContrastConfig {
    /// FFT size for spectral analysis.
    pub fft_size: usize,
    /// Number of octave sub-bands.
    pub band_count: usize,
    /// Fraction of bins considered as peaks (top alpha fraction).
    pub alpha: f64,
    /// Minimum frequency of the lowest band (Hz).
    pub min_freq_hz: f64,
    /// Sample rate (Hz).
    pub sample_rate: f64,
    /// Floor value to avoid log-of-zero.
    pub floor: f64,
}

impl Default for SpectralContrastConfig {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            band_count: DEFAULT_BAND_COUNT,
            alpha: 0.2,
            min_freq_hz: 100.0,
            sample_rate: 44100.0,
            floor: 1e-10,
        }
    }
}

/// Spectral contrast result for a single frame.
#[derive(Debug, Clone)]
pub struct SpectralContrastFrame {
    /// Peak magnitudes per sub-band (linear scale).
    pub peaks: Vec<f64>,
    /// Valley magnitudes per sub-band (linear scale).
    pub valleys: Vec<f64>,
    /// Contrast values per sub-band (peak - valley in dB).
    pub contrast_db: Vec<f64>,
    /// Mean contrast across all bands (dB).
    pub mean_contrast_db: f64,
}

/// Spectral contrast result over multiple frames.
#[derive(Debug, Clone)]
pub struct SpectralContrastResult {
    /// Per-frame results.
    pub frames: Vec<SpectralContrastFrame>,
    /// Mean contrast per band averaged over all frames (dB).
    pub mean_contrast_per_band: Vec<f64>,
    /// Standard deviation of contrast per band over all frames (dB).
    pub std_contrast_per_band: Vec<f64>,
    /// Overall mean contrast (dB).
    pub overall_mean_contrast_db: f64,
}

/// Analyser for spectral contrast computation.
#[derive(Debug, Clone)]
pub struct SpectralContrastAnalyzer {
    /// Analysis configuration.
    config: SpectralContrastConfig,
    /// Precomputed band edge indices into the FFT magnitude spectrum.
    band_edges: Vec<usize>,
}

impl SpectralContrastAnalyzer {
    /// Create a new analyser with the given configuration.
    #[must_use]
    pub fn new(config: SpectralContrastConfig) -> Self {
        let band_edges = compute_band_edges(&config);
        Self { config, band_edges }
    }

    /// Create an analyser with default settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SpectralContrastConfig::default())
    }

    /// Analyse spectral contrast for a single magnitude spectrum.
    ///
    /// `magnitudes` is a slice of FFT magnitude bins (length = `fft_size / 2 + 1`).
    #[must_use]
    pub fn analyze_frame(&self, magnitudes: &[f64]) -> SpectralContrastFrame {
        let mut peaks = Vec::with_capacity(self.config.band_count);
        let mut valleys = Vec::with_capacity(self.config.band_count);
        let mut contrast_db = Vec::with_capacity(self.config.band_count);

        for band in 0..self.config.band_count {
            let lo = self.band_edges[band];
            let hi = self.band_edges[band + 1];
            if lo >= hi || lo >= magnitudes.len() {
                peaks.push(self.config.floor);
                valleys.push(self.config.floor);
                contrast_db.push(0.0);
                continue;
            }
            let hi = hi.min(magnitudes.len());
            let band_mags = &magnitudes[lo..hi];

            let (peak_val, valley_val) =
                peak_valley(band_mags, self.config.alpha, self.config.floor);
            let contrast = 20.0 * (peak_val / valley_val).log10();

            peaks.push(peak_val);
            valleys.push(valley_val);
            contrast_db.push(contrast);
        }

        let mean_contrast_db = if contrast_db.is_empty() {
            0.0
        } else {
            contrast_db.iter().sum::<f64>() / contrast_db.len() as f64
        };

        SpectralContrastFrame {
            peaks,
            valleys,
            contrast_db,
            mean_contrast_db,
        }
    }

    /// Analyse spectral contrast over a series of magnitude frames.
    #[must_use]
    pub fn analyze(&self, magnitude_frames: &[Vec<f64>]) -> SpectralContrastResult {
        let frames: Vec<SpectralContrastFrame> = magnitude_frames
            .iter()
            .map(|m| self.analyze_frame(m))
            .collect();

        let n_bands = self.config.band_count;
        let n_frames = frames.len();

        let mut mean_per_band = vec![0.0; n_bands];
        let mut std_per_band = vec![0.0; n_bands];

        if n_frames > 0 {
            for frame in &frames {
                for (b, &c) in frame.contrast_db.iter().enumerate() {
                    mean_per_band[b] += c;
                }
            }
            for v in &mut mean_per_band {
                *v /= n_frames as f64;
            }

            if n_frames > 1 {
                for frame in &frames {
                    for (b, &c) in frame.contrast_db.iter().enumerate() {
                        let diff = c - mean_per_band[b];
                        std_per_band[b] += diff * diff;
                    }
                }
                for v in &mut std_per_band {
                    *v = (*v / (n_frames - 1) as f64).sqrt();
                }
            }
        }

        let overall_mean = if mean_per_band.is_empty() {
            0.0
        } else {
            mean_per_band.iter().sum::<f64>() / mean_per_band.len() as f64
        };

        SpectralContrastResult {
            frames,
            mean_contrast_per_band: mean_per_band,
            std_contrast_per_band: std_per_band,
            overall_mean_contrast_db: overall_mean,
        }
    }
}

/// Compute octave-spaced band edge indices for the FFT magnitude spectrum.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
fn compute_band_edges(config: &SpectralContrastConfig) -> Vec<usize> {
    let nyquist = config.sample_rate / 2.0;
    let bin_hz = nyquist / (config.fft_size as f64 / 2.0);

    let mut edges = Vec::with_capacity(config.band_count + 1);
    for i in 0..=config.band_count {
        let freq = config.min_freq_hz * 2.0_f64.powi(i as i32);
        let bin = (freq / bin_hz).round() as usize;
        edges.push(bin);
    }
    edges
}

/// Compute peak and valley values for a sub-band using the alpha fraction.
fn peak_valley(band: &[f64], alpha: f64, floor: f64) -> (f64, f64) {
    if band.is_empty() {
        return (floor, floor);
    }

    let mut sorted: Vec<f64> = band.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let k = ((alpha * n as f64).ceil() as usize).max(1).min(n);

    // Valley: mean of bottom-k bins
    let valley = sorted[..k].iter().sum::<f64>() / k as f64;
    // Peak: mean of top-k bins
    let peak = sorted[n - k..].iter().sum::<f64>() / k as f64;

    (peak.max(floor), valley.max(floor))
}

/// Compute a simple Hann-windowed magnitude spectrum from time-domain samples.
///
/// Returns a magnitude vector of length `fft_size / 2 + 1`.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn magnitude_spectrum(samples: &[f32], fft_size: usize) -> Vec<f64> {
    let n = fft_size.min(samples.len());
    let mut windowed = vec![0.0_f64; fft_size];
    for (i, &s) in samples.iter().take(n).enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos());
        windowed[i] = f64::from(s) * w;
    }

    // Compute magnitude via DFT (slow but dependency-free for small sizes).
    let half = fft_size / 2 + 1;
    let mut mag = vec![0.0_f64; half];
    #[allow(clippy::needless_range_loop)]
    for k in 0..half {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (n_idx, &x) in windowed.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * n_idx as f64 / fft_size as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        mag[k] = (re * re + im * im).sqrt();
    }
    mag
}

/// Spectral flatness of a sub-band (geometric mean / arithmetic mean).
///
/// Returns a value between 0 (tonal) and 1 (noise-like).
#[must_use]
pub fn spectral_flatness(band: &[f64]) -> f64 {
    if band.is_empty() {
        return 0.0;
    }
    let n = band.len() as f64;
    let log_sum: f64 = band.iter().map(|&x| (x.max(1e-30)).ln()).sum();
    let geo_mean = (log_sum / n).exp();
    let arith_mean = band.iter().sum::<f64>() / n;
    if arith_mean <= 0.0 {
        0.0
    } else {
        geo_mean / arith_mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SpectralContrastConfig::default();
        assert_eq!(cfg.fft_size, 2048);
        assert_eq!(cfg.band_count, DEFAULT_BAND_COUNT);
        assert!((cfg.alpha - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_band_edges_count() {
        let cfg = SpectralContrastConfig::default();
        let edges = compute_band_edges(&cfg);
        assert_eq!(edges.len(), cfg.band_count + 1);
    }

    #[test]
    fn test_band_edges_increasing() {
        let cfg = SpectralContrastConfig::default();
        let edges = compute_band_edges(&cfg);
        for w in edges.windows(2) {
            assert!(w[1] >= w[0], "Band edges should be non-decreasing");
        }
    }

    #[test]
    fn test_peak_valley_uniform() {
        let band = vec![1.0; 100];
        let (peak, valley) = peak_valley(&band, 0.2, 1e-10);
        assert!((peak - 1.0).abs() < 1e-6);
        assert!((valley - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_peak_valley_varying() {
        let band: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (peak, valley) = peak_valley(&band, 0.2, 1e-10);
        assert!(peak > valley, "Peak should exceed valley");
        // Top 20 bins: 81..100, mean = 90.5
        assert!((peak - 90.5).abs() < 0.01);
        // Bottom 20 bins: 1..20, mean = 10.5
        assert!((valley - 10.5).abs() < 0.01);
    }

    #[test]
    fn test_peak_valley_empty() {
        let (p, v) = peak_valley(&[], 0.2, 1e-10);
        assert!((p - 1e-10).abs() < 1e-20);
        assert!((v - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_analyze_frame_constant_spectrum() {
        let analyzer = SpectralContrastAnalyzer::with_defaults();
        let mag = vec![1.0; 1025]; // fft_size/2 + 1
        let frame = analyzer.analyze_frame(&mag);
        assert_eq!(frame.contrast_db.len(), DEFAULT_BAND_COUNT);
        for &c in &frame.contrast_db {
            assert!(
                c.abs() < 1e-3,
                "Constant spectrum should have near-zero contrast"
            );
        }
    }

    #[test]
    fn test_analyze_frame_band_count() {
        let cfg = SpectralContrastConfig {
            band_count: 4,
            ..Default::default()
        };
        let analyzer = SpectralContrastAnalyzer::new(cfg);
        let mag = vec![1.0; 1025];
        let frame = analyzer.analyze_frame(&mag);
        assert_eq!(frame.peaks.len(), 4);
        assert_eq!(frame.valleys.len(), 4);
        assert_eq!(frame.contrast_db.len(), 4);
    }

    #[test]
    fn test_analyze_multi_frame() {
        let analyzer = SpectralContrastAnalyzer::with_defaults();
        let frames_data: Vec<Vec<f64>> = (0..5).map(|_| vec![1.0; 1025]).collect();
        let result = analyzer.analyze(&frames_data);
        assert_eq!(result.frames.len(), 5);
        assert_eq!(result.mean_contrast_per_band.len(), DEFAULT_BAND_COUNT);
    }

    #[test]
    fn test_analyze_empty_frames() {
        let analyzer = SpectralContrastAnalyzer::with_defaults();
        let result = analyzer.analyze(&[]);
        assert!(result.frames.is_empty());
        assert!((result.overall_mean_contrast_db).abs() < f64::EPSILON);
    }

    #[test]
    fn test_magnitude_spectrum_length() {
        let samples: Vec<f32> = vec![0.0; 2048];
        let mag = magnitude_spectrum(&samples, 2048);
        assert_eq!(mag.len(), 1025);
    }

    #[test]
    fn test_magnitude_spectrum_silence() {
        let samples: Vec<f32> = vec![0.0; 512];
        let mag = magnitude_spectrum(&samples, 512);
        for &m in &mag {
            assert!(m.abs() < 1e-10, "Silence should have near-zero magnitude");
        }
    }

    #[test]
    fn test_spectral_flatness_noise() {
        // Uniform spectrum => flatness near 1.0
        let band = vec![1.0; 100];
        let flat = spectral_flatness(&band);
        assert!(
            (flat - 1.0).abs() < 1e-6,
            "Uniform spectrum flatness should be ~1.0"
        );
    }

    #[test]
    fn test_spectral_flatness_tonal() {
        // Single peak with zeros => flatness near 0
        let mut band = vec![0.001; 100];
        band[50] = 100.0;
        let flat = spectral_flatness(&band);
        assert!(
            flat < 0.1,
            "Tonal signal flatness should be near 0, got {flat}"
        );
    }

    #[test]
    fn test_spectral_flatness_empty() {
        assert!((spectral_flatness(&[]) - 0.0).abs() < f64::EPSILON);
    }
}
