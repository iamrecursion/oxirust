//! Constant-Q Transform (CQT) for music-oriented frequency analysis.
//!
//! The CQT provides logarithmically-spaced frequency bins, making it ideal
//! for music analysis where octave relationships are important.
//!
//! Reference: J. C. Brown (1991), "Calculation of a constant Q spectral transform",
//! JASA 89(1), 425–434.

use crate::{AnalysisError, Result};
use oxifft::Complex;
use std::f64::consts::PI;

/// Configuration for CQT computation.
#[derive(Debug, Clone)]
pub struct CqtConfig {
    /// Minimum frequency in Hz (e.g., C1 = 32.7 Hz)
    pub f_min: f64,
    /// Number of frequency bins per octave (e.g., 12 for semitones)
    pub bins_per_octave: usize,
    /// Total number of frequency bins
    pub n_bins: usize,
    /// Quality factor Q (controls frequency resolution vs time resolution)
    pub quality_factor: f64,
    /// Hop size in samples
    pub hop_size: usize,
}

impl Default for CqtConfig {
    fn default() -> Self {
        Self {
            f_min: 32.703,       // C1
            bins_per_octave: 36, // 3 bins per semitone
            n_bins: 252,         // 7 octaves * 36 bins
            quality_factor: 1.0,
            hop_size: 512,
        }
    }
}

/// Constant-Q Transform analyzer.
pub struct CqtAnalyzer {
    config: CqtConfig,
    /// Precomputed CQT kernels (one per frequency bin)
    kernels: Vec<Vec<Complex<f64>>>,
    /// Center frequencies for each bin
    frequencies: Vec<f64>,
}

impl CqtAnalyzer {
    /// Create a new CQT analyzer.
    ///
    /// Precomputes the CQT kernels for efficient repeated analysis.
    pub fn new(config: CqtConfig, sample_rate: f64) -> Result<Self> {
        if sample_rate <= 0.0 {
            return Err(AnalysisError::InvalidConfig(
                "Sample rate must be positive".to_string(),
            ));
        }
        if config.bins_per_octave == 0 {
            return Err(AnalysisError::InvalidConfig(
                "bins_per_octave must be >= 1".to_string(),
            ));
        }

        // Q factor: Q = 1 / (2^(1/B) - 1) * quality_factor
        let q = config.quality_factor / (2.0_f64.powf(1.0 / config.bins_per_octave as f64) - 1.0);

        let mut kernels = Vec::with_capacity(config.n_bins);
        let mut frequencies = Vec::with_capacity(config.n_bins);

        for k in 0..config.n_bins {
            let f_k = config.f_min * 2.0_f64.powf(k as f64 / config.bins_per_octave as f64);
            frequencies.push(f_k);

            // Window length for this bin: N_k = Q * fs / f_k
            let n_k = (q * sample_rate / f_k).round() as usize;
            if n_k == 0 {
                return Err(AnalysisError::InvalidConfig(format!(
                    "Kernel length is zero for bin {k}, frequency {f_k:.1} Hz"
                )));
            }

            // Temporal kernel: Hann-windowed complex sinusoid
            let mut kernel = Vec::with_capacity(n_k);
            for n in 0..n_k {
                let window = 0.5 * (1.0 - (2.0 * PI * n as f64 / (n_k - 1) as f64).cos());
                let angle = 2.0 * PI * q * n as f64 / n_k as f64;
                let real = window * angle.cos() / n_k as f64;
                let imag = -window * angle.sin() / n_k as f64;
                kernel.push(Complex::new(real, imag));
            }

            kernels.push(kernel);
        }

        Ok(Self {
            config,
            kernels,
            frequencies,
        })
    }

    /// Compute CQT magnitude spectrum for a block of audio samples.
    ///
    /// # Arguments
    /// * `samples` - Audio samples (should be at least as long as the longest kernel)
    ///
    /// # Returns
    /// Complex CQT coefficients for each frequency bin.
    pub fn compute(&self, samples: &[f32]) -> Result<Vec<Complex<f64>>> {
        if samples.is_empty() {
            return Err(AnalysisError::InsufficientSamples { needed: 1, got: 0 });
        }

        let mut cqt = Vec::with_capacity(self.config.n_bins);

        for kernel in &self.kernels {
            let n_k = kernel.len();
            if n_k > samples.len() {
                // Insufficient samples: return zero for this bin
                cqt.push(Complex::new(0.0, 0.0));
                continue;
            }

            // Compute inner product: cqt[k] = sum_n samples[n] * conj(kernel[n])
            // Since kernel already has conj applied (negative imag), just multiply
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let offset = samples.len().saturating_sub(n_k) / 2;

            for (i, k) in kernel.iter().enumerate() {
                let s = f64::from(samples[offset + i]);
                re += s * k.re;
                im += s * k.im;
            }

            cqt.push(Complex::new(re, im));
        }

        Ok(cqt)
    }

    /// Compute CQT magnitude (absolute value of each coefficient).
    pub fn magnitude(&self, samples: &[f32]) -> Result<Vec<f64>> {
        let cqt = self.compute(samples)?;
        Ok(cqt.iter().map(|c| c.norm()).collect())
    }

    /// Compute CQT over time using hop-based framing.
    ///
    /// Returns a 2D matrix (frames × bins) of CQT magnitudes.
    pub fn spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f64>>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let hop = self.config.hop_size;
        let n_frames = samples.len() / hop;
        let mut result = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let frame = &samples[start..samples.len().min(start + samples.len())];
            let mag = self.magnitude(frame)?;
            result.push(mag);
        }

        Ok(result)
    }

    /// Return the center frequency of each CQT bin in Hz.
    #[must_use]
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Convert bin index to musical note name (C, C#, D, …, B).
    #[must_use]
    pub fn bin_to_note_name(bin: usize, bins_per_octave: usize) -> &'static str {
        const NOTE_NAMES: &[&str] = &[
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let semitone_idx = (bin * 12 / bins_per_octave) % 12;
        NOTE_NAMES[semitone_idx]
    }
}

/// CQT-based chromagram: sum CQT magnitudes across octaves into 12 pitch classes.
///
/// # Arguments
/// * `cqt_magnitude` - Magnitude spectrum from `CqtAnalyzer::magnitude()`
/// * `bins_per_octave` - Must match the `CqtConfig` used
///
/// # Returns
/// 12-element chroma vector (one value per pitch class, C through B)
#[must_use]
pub fn cqt_chroma(cqt_magnitude: &[f64], bins_per_octave: usize) -> Vec<f64> {
    let mut chroma = vec![0.0_f64; 12];
    let bins_per_semitone = bins_per_octave / 12;
    let bins_per_semitone = bins_per_semitone.max(1);

    for (k, &mag) in cqt_magnitude.iter().enumerate() {
        let semitone = (k / bins_per_semitone) % 12;
        chroma[semitone] += mag * mag; // accumulate energy
    }

    // Normalize by L2 norm
    let norm: f64 = chroma.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for c in &mut chroma {
            *c /= norm;
        }
    }

    chroma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cqt_analyzer_construction() {
        let config = CqtConfig::default();
        let analyzer = CqtAnalyzer::new(config, 44100.0);
        assert!(analyzer.is_ok());
        let a = analyzer.expect("should construct");
        assert_eq!(a.frequencies().len(), 252);
        assert!(a.frequencies()[0] > 30.0 && a.frequencies()[0] < 40.0);
    }

    #[test]
    fn test_cqt_frequencies_logarithmic() {
        let config = CqtConfig {
            bins_per_octave: 12,
            n_bins: 24,
            ..Default::default()
        };
        let analyzer = CqtAnalyzer::new(config, 44100.0).expect("should construct");
        let freqs = analyzer.frequencies();
        // Every 12 bins should double the frequency (one octave)
        let ratio = freqs[12] / freqs[0];
        assert!((ratio - 2.0).abs() < 0.01, "One octave ratio: {ratio}");
    }

    #[test]
    fn test_cqt_magnitude_sine_wave() {
        let config = CqtConfig {
            bins_per_octave: 12,
            n_bins: 48,     // 4 octaves
            f_min: 130.813, // C3
            ..Default::default()
        };
        let sample_rate = 44100.0_f64;
        let analyzer = CqtAnalyzer::new(config.clone(), sample_rate).expect("should construct");

        // Generate A4 = 440 Hz sine wave
        let samples: Vec<f32> = (0..8192)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * 440.0 * t).sin() as f32
            })
            .collect();

        let mag = analyzer.magnitude(&samples).expect("should compute");
        assert_eq!(mag.len(), config.n_bins);
        // All magnitudes should be non-negative
        for &m in &mag {
            assert!(m >= 0.0, "Magnitude should be non-negative: {m}");
        }
    }

    #[test]
    fn test_cqt_chroma_from_magnitude() {
        let mag = vec![1.0; 36]; // 36 bins = 1 octave at bins_per_octave=36
        let chroma = cqt_chroma(&mag, 36);
        assert_eq!(chroma.len(), 12);
        // Should be roughly uniform since input is uniform
        let max = chroma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = chroma.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (max - min).abs() < 0.5,
            "Uniform input should produce near-uniform chroma"
        );
    }

    #[test]
    fn test_cqt_spectrogram() {
        let config = CqtConfig {
            bins_per_octave: 12,
            n_bins: 12,
            hop_size: 256,
            ..Default::default()
        };
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI as f32 * 440.0 * i as f32 / 44100.0).sin())
            .collect();
        let analyzer = CqtAnalyzer::new(config, 44100.0).expect("should construct");
        let spec = analyzer.spectrogram(&samples).expect("should compute");
        assert!(!spec.is_empty());
        for frame in &spec {
            assert_eq!(frame.len(), 12);
        }
    }

    #[test]
    fn test_note_name() {
        let name = CqtAnalyzer::bin_to_note_name(0, 12);
        assert_eq!(name, "C");
        let name9 = CqtAnalyzer::bin_to_note_name(9, 12);
        assert_eq!(name9, "A");
    }
}
