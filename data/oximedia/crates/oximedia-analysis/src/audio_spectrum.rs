#![allow(dead_code)]
//! Audio spectrum analysis utilities.
//!
//! Provides tools for computing magnitude spectra, spectral centroid,
//! spectral rolloff, spectral flux, band energy ratios, and other
//! frequency-domain features commonly used in audio content analysis
//! and music information retrieval.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for spectrum analysis.
#[derive(Debug, Clone)]
pub struct SpectrumConfig {
    /// FFT size (must be a power of two).
    pub fft_size: usize,
    /// Hop size between consecutive analysis windows.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Apply Hann window before FFT.
    pub use_hann_window: bool,
    /// Number of frequency bands for band-energy analysis.
    pub num_bands: usize,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 512,
            sample_rate: 48000,
            use_hann_window: true,
            num_bands: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Spectral features for a single frame
// ---------------------------------------------------------------------------

/// Spectral features computed from a single analysis window.
#[derive(Debug, Clone)]
pub struct SpectralFrame {
    /// Frame index (0-based).
    pub index: usize,
    /// Magnitude spectrum (linear, non-negative). Length = `fft_size / 2 + 1`.
    pub magnitudes: Vec<f64>,
    /// Spectral centroid in Hz.
    pub centroid_hz: f64,
    /// Spectral rolloff frequency in Hz (85% energy).
    pub rolloff_hz: f64,
    /// Spectral flux (L2 norm of magnitude change from previous frame).
    pub flux: f64,
    /// Spectral flatness (geometric mean / arithmetic mean of magnitudes).
    pub flatness: f64,
    /// Band energies (sum of squared magnitudes per band).
    pub band_energies: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Aggregate result
// ---------------------------------------------------------------------------

/// Aggregated spectrum analysis result over an entire audio clip.
#[derive(Debug, Clone)]
pub struct SpectrumAnalysisResult {
    /// Number of frames analysed.
    pub num_frames: usize,
    /// Per-frame spectral features.
    pub frames: Vec<SpectralFrame>,
    /// Average spectral centroid across all frames (Hz).
    pub avg_centroid_hz: f64,
    /// Average spectral rolloff across all frames (Hz).
    pub avg_rolloff_hz: f64,
    /// Average spectral flux.
    pub avg_flux: f64,
    /// Average spectral flatness.
    pub avg_flatness: f64,
    /// Mean band energies across all frames.
    pub mean_band_energies: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful spectrum analyzer that processes audio in chunks.
#[derive(Debug)]
pub struct SpectrumAnalyzer {
    /// Configuration.
    config: SpectrumConfig,
    /// Accumulated frames.
    frames: Vec<SpectralFrame>,
    /// Previous magnitude spectrum for flux calculation.
    prev_mags: Option<Vec<f64>>,
    /// Pre-computed Hann window coefficients.
    hann_window: Vec<f64>,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    pub fn new(config: SpectrumConfig) -> Self {
        let hann_window = compute_hann_window(config.fft_size);
        Self {
            config,
            frames: Vec::new(),
            prev_mags: None,
            hann_window,
        }
    }

    /// Analyze all windows in the given sample buffer.
    ///
    /// Samples should be mono f32 PCM in the range \[-1.0, 1.0\].
    pub fn analyze_buffer(&mut self, samples: &[f32]) {
        let fft_size = self.config.fft_size;
        let hop = self.config.hop_size;
        if samples.len() < fft_size {
            return;
        }
        let mut offset = 0;
        while offset + fft_size <= samples.len() {
            let window = &samples[offset..offset + fft_size];
            self.analyze_window(window);
            offset += hop;
        }
    }

    /// Analyze a single window of samples.
    fn analyze_window(&mut self, window: &[f32]) {
        let _n = self.config.fft_size;
        // Apply window function
        let windowed: Vec<f64> = if self.config.use_hann_window {
            window
                .iter()
                .enumerate()
                .map(|(i, &s)| f64::from(s) * self.hann_window[i])
                .collect()
        } else {
            window.iter().map(|&s| f64::from(s)).collect()
        };

        // Compute magnitude spectrum using naive DFT (sufficient for analysis sizes)
        let mags = compute_magnitude_spectrum(&windowed);

        let centroid = compute_centroid(&mags, self.config.sample_rate);
        let rolloff = compute_rolloff(&mags, self.config.sample_rate, 0.85);
        let flatness = compute_flatness(&mags);
        let flux = self
            .prev_mags
            .as_ref()
            .map_or(0.0, |prev| compute_flux(&mags, prev));
        let band_energies = compute_band_energies(&mags, self.config.num_bands);

        let index = self.frames.len();
        self.frames.push(SpectralFrame {
            index,
            magnitudes: mags.clone(),
            centroid_hz: centroid,
            rolloff_hz: rolloff,
            flux,
            flatness,
            band_energies,
        });
        self.prev_mags = Some(mags);
    }

    /// Finalize and return aggregated results.
    pub fn finalize(self) -> SpectrumAnalysisResult {
        let n = self.frames.len();
        if n == 0 {
            return SpectrumAnalysisResult {
                num_frames: 0,
                frames: Vec::new(),
                avg_centroid_hz: 0.0,
                avg_rolloff_hz: 0.0,
                avg_flux: 0.0,
                avg_flatness: 0.0,
                mean_band_energies: Vec::new(),
            };
        }
        #[allow(clippy::cast_precision_loss)]
        let nf = n as f64;
        let avg_centroid = self.frames.iter().map(|f| f.centroid_hz).sum::<f64>() / nf;
        let avg_rolloff = self.frames.iter().map(|f| f.rolloff_hz).sum::<f64>() / nf;
        let avg_flux = self.frames.iter().map(|f| f.flux).sum::<f64>() / nf;
        let avg_flatness = self.frames.iter().map(|f| f.flatness).sum::<f64>() / nf;

        let num_bands = self.config.num_bands;
        let mut mean_bands = vec![0.0f64; num_bands];
        for frame in &self.frames {
            for (j, &e) in frame.band_energies.iter().enumerate() {
                if j < num_bands {
                    mean_bands[j] += e;
                }
            }
        }
        for v in &mut mean_bands {
            *v /= nf;
        }

        SpectrumAnalysisResult {
            num_frames: n,
            frames: self.frames,
            avg_centroid_hz: avg_centroid,
            avg_rolloff_hz: avg_rolloff,
            avg_flux,
            avg_flatness,
            mean_band_energies: mean_bands,
        }
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------

/// Compute a Hann window of given size.
fn compute_hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / size as f64).cos());
            w
        })
        .collect()
}

/// Compute magnitude spectrum via naive DFT (first N/2+1 bins).
fn compute_magnitude_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let half = n / 2 + 1;
    let mut mags = Vec::with_capacity(half);
    for k in 0..half {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (i, &x) in signal.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let angle = 2.0 * PI * k as f64 * i as f64 / n as f64;
            re += x * angle.cos();
            im -= x * angle.sin();
        }
        #[allow(clippy::cast_precision_loss)]
        let mag = (re * re + im * im).sqrt() / n as f64;
        mags.push(mag);
    }
    mags
}

/// Spectral centroid: weighted average of frequencies by magnitude.
fn compute_centroid(mags: &[f64], sample_rate: u32) -> f64 {
    let total: f64 = mags.iter().sum();
    if total < 1e-12 {
        return 0.0;
    }
    let n = (mags.len() - 1) * 2; // original FFT size
    #[allow(clippy::cast_precision_loss)]
    let bin_hz = f64::from(sample_rate) / n as f64;
    let weighted: f64 = mags
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            #[allow(clippy::cast_precision_loss)]
            let freq = i as f64 * bin_hz;
            freq * m
        })
        .sum();
    weighted / total
}

/// Spectral rolloff: frequency below which `ratio` of total energy resides.
fn compute_rolloff(mags: &[f64], sample_rate: u32, ratio: f64) -> f64 {
    let total_energy: f64 = mags.iter().map(|m| m * m).sum();
    let threshold = total_energy * ratio;
    let n = (mags.len() - 1) * 2;
    #[allow(clippy::cast_precision_loss)]
    let bin_hz = f64::from(sample_rate) / n as f64;
    let mut cumulative = 0.0;
    for (i, &m) in mags.iter().enumerate() {
        cumulative += m * m;
        if cumulative >= threshold {
            #[allow(clippy::cast_precision_loss)]
            return i as f64 * bin_hz;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let nyquist = f64::from(sample_rate) / 2.0;
    nyquist
}

/// Spectral flatness: geometric mean / arithmetic mean.
fn compute_flatness(mags: &[f64]) -> f64 {
    let n = mags.len();
    if n == 0 {
        return 0.0;
    }
    // Use log-domain for geometric mean to avoid overflow/underflow
    let positive: Vec<f64> = mags.iter().copied().filter(|&m| m > 1e-20).collect();
    if positive.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let log_geo = positive.iter().map(|m| m.ln()).sum::<f64>() / positive.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let arith = positive.iter().sum::<f64>() / positive.len() as f64;
    if arith < 1e-20 {
        return 0.0;
    }
    let geo = log_geo.exp();
    geo / arith
}

/// Spectral flux: L2 distance between consecutive magnitude spectra.
fn compute_flux(current: &[f64], previous: &[f64]) -> f64 {
    current
        .iter()
        .zip(previous.iter())
        .map(|(c, p)| {
            let d = c - p;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Split magnitude spectrum into `num_bands` equal-width bands and sum energy.
fn compute_band_energies(mags: &[f64], num_bands: usize) -> Vec<f64> {
    if num_bands == 0 || mags.is_empty() {
        return vec![0.0; num_bands];
    }
    let bins_per_band = mags.len() / num_bands;
    let mut energies = Vec::with_capacity(num_bands);
    for b in 0..num_bands {
        let start = b * bins_per_band;
        let end = if b == num_bands - 1 {
            mags.len()
        } else {
            start + bins_per_band
        };
        let energy: f64 = mags[start..end].iter().map(|m| m * m).sum();
        energies.push(energy);
    }
    energies
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f64 / sample_rate as f64;
                (2.0 * PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_config_defaults() {
        let cfg = SpectrumConfig::default();
        assert_eq!(cfg.fft_size, 1024);
        assert_eq!(cfg.hop_size, 512);
        assert_eq!(cfg.sample_rate, 48000);
        assert!(cfg.use_hann_window);
    }

    #[test]
    fn test_empty_analyzer() {
        let a = SpectrumAnalyzer::new(SpectrumConfig::default());
        let result = a.finalize();
        assert_eq!(result.num_frames, 0);
    }

    #[test]
    fn test_silence_spectrum() {
        let cfg = SpectrumConfig {
            fft_size: 64,
            hop_size: 32,
            sample_rate: 8000,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        let silence = vec![0.0f32; 128];
        a.analyze_buffer(&silence);
        let result = a.finalize();
        assert!(result.num_frames > 0);
        assert!(result.avg_centroid_hz.abs() < 0.01);
    }

    #[test]
    fn test_sine_centroid() {
        let cfg = SpectrumConfig {
            fft_size: 256,
            hop_size: 256,
            sample_rate: 8000,
            use_hann_window: false,
            num_bands: 4,
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        // 1000 Hz sine — centroid should be near 1000 Hz
        let samples = sine_wave(1000.0, 8000, 256);
        a.analyze_buffer(&samples);
        let result = a.finalize();
        assert_eq!(result.num_frames, 1);
        assert!((result.avg_centroid_hz - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_spectral_flux_zero_for_constant() {
        let cfg = SpectrumConfig {
            fft_size: 64,
            hop_size: 64,
            sample_rate: 8000,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        // Repeating identical windows => flux should be zero after first frame
        let samples = sine_wave(500.0, 8000, 256);
        a.analyze_buffer(&samples);
        let result = a.finalize();
        // Second frame onwards should have near-zero flux (identical signal)
        if result.num_frames >= 2 {
            assert!(result.frames[1].flux < 0.01);
        }
    }

    #[test]
    fn test_flatness_pure_tone_vs_noise() {
        // Pure tone: flatness should be low
        let cfg = SpectrumConfig {
            fft_size: 128,
            hop_size: 128,
            sample_rate: 8000,
            use_hann_window: false,
            num_bands: 4,
        };
        let mut a = SpectrumAnalyzer::new(cfg.clone());
        a.analyze_buffer(&sine_wave(1000.0, 8000, 128));
        let tone_result = a.finalize();

        // "Noise-like" signal: multi-frequency sum for broadband energy
        let mut a2 = SpectrumAnalyzer::new(cfg);
        let noise: Vec<f32> = (0..128)
            .map(|i| {
                let t = i as f32 / 8000.0;
                0.2 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 2500.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 3500.0 * t).sin()
            })
            .collect();
        a2.analyze_buffer(&noise);
        let noise_result = a2.finalize();

        // Noise should have higher flatness than pure tone
        assert!(noise_result.avg_flatness > tone_result.avg_flatness);
    }

    #[test]
    fn test_band_energies_count() {
        let cfg = SpectrumConfig {
            fft_size: 64,
            hop_size: 64,
            sample_rate: 8000,
            num_bands: 8,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        a.analyze_buffer(&sine_wave(1000.0, 8000, 64));
        let result = a.finalize();
        assert_eq!(result.mean_band_energies.len(), 8);
    }

    #[test]
    fn test_rolloff_below_nyquist() {
        let sr = 8000u32;
        let cfg = SpectrumConfig {
            fft_size: 128,
            hop_size: 128,
            sample_rate: sr,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        a.analyze_buffer(&sine_wave(1000.0, sr, 128));
        let result = a.finalize();
        #[allow(clippy::cast_precision_loss)]
        let nyquist = sr as f64 / 2.0;
        assert!(result.avg_rolloff_hz <= nyquist);
    }

    #[test]
    fn test_hann_window_length() {
        let w = compute_hann_window(512);
        assert_eq!(w.len(), 512);
        // Endpoints should be near zero
        assert!(w[0].abs() < 0.001);
    }

    #[test]
    fn test_magnitude_spectrum_length() {
        let signal = vec![0.0f64; 16];
        let mags = compute_magnitude_spectrum(&signal);
        assert_eq!(mags.len(), 9); // 16/2 + 1
    }

    #[test]
    fn test_short_buffer_no_crash() {
        let cfg = SpectrumConfig {
            fft_size: 256,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        // Buffer shorter than FFT size — should produce zero frames
        a.analyze_buffer(&[0.1f32; 100]);
        let result = a.finalize();
        assert_eq!(result.num_frames, 0);
    }

    #[test]
    fn test_multiple_windows() {
        let cfg = SpectrumConfig {
            fft_size: 64,
            hop_size: 32,
            sample_rate: 8000,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        a.analyze_buffer(&sine_wave(500.0, 8000, 256));
        let result = a.finalize();
        // (256 - 64) / 32 + 1 = 7 frames
        assert!(result.num_frames >= 5);
    }

    #[test]
    fn test_flux_increases_on_change() {
        let cfg = SpectrumConfig {
            fft_size: 64,
            hop_size: 64,
            sample_rate: 8000,
            use_hann_window: false,
            ..Default::default()
        };
        let mut a = SpectrumAnalyzer::new(cfg);
        let mut buf = sine_wave(500.0, 8000, 64);
        buf.extend(sine_wave(3000.0, 8000, 64));
        a.analyze_buffer(&buf);
        let result = a.finalize();
        assert_eq!(result.num_frames, 2);
        // Flux on second frame should be non-trivial
        assert!(result.frames[1].flux > 0.001);
    }
}
