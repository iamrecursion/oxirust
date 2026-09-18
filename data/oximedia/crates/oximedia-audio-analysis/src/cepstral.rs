#![allow(dead_code)]
//! Cepstral analysis for audio signals.
//!
//! Cepstral analysis operates in the "quefrency" domain (the cepstral domain)
//! to separate the excitation source from the vocal tract response. This module
//! provides MFCC extraction, cepstral peak prominence, and liftering utilities.

use std::f64::consts::PI;

/// Configuration for cepstral analysis.
#[derive(Debug, Clone)]
pub struct CepstralConfig {
    /// Number of cepstral coefficients to extract (default 13).
    pub num_coefficients: usize,
    /// Number of Mel filter banks (default 26).
    pub num_filters: usize,
    /// FFT size in samples (default 1024).
    pub fft_size: usize,
    /// Low frequency edge for Mel filter bank in Hz.
    pub low_freq: f64,
    /// High frequency edge for Mel filter bank in Hz.
    pub high_freq: f64,
    /// Whether to apply liftering (default true).
    pub apply_lifter: bool,
    /// Liftering coefficient L (default 22).
    pub lifter_coeff: usize,
}

impl Default for CepstralConfig {
    fn default() -> Self {
        Self {
            num_coefficients: 13,
            num_filters: 26,
            fft_size: 1024,
            low_freq: 20.0,
            high_freq: 8000.0,
            apply_lifter: true,
            lifter_coeff: 22,
        }
    }
}

/// Result of cepstral analysis on a single frame.
#[derive(Debug, Clone)]
pub struct CepstralFrame {
    /// MFCC coefficients for this frame.
    pub mfcc: Vec<f64>,
    /// Cepstral peak prominence (CPP) in dB.
    pub cpp_db: f64,
    /// Index of the dominant cepstral peak (quefrency bin).
    pub peak_quefrency_bin: usize,
    /// Energy of the frame in dB.
    pub energy_db: f64,
}

/// Cepstral analyzer for extracting MFCCs and cepstral features.
#[derive(Debug, Clone)]
pub struct CepstralAnalyzer {
    config: CepstralConfig,
    mel_filters: Vec<Vec<f64>>,
}

/// Convert frequency in Hz to Mel scale.
#[allow(clippy::cast_precision_loss)]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale value back to Hz.
#[allow(clippy::cast_precision_loss)]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build triangular Mel filter bank.
#[allow(clippy::cast_precision_loss, clippy::needless_range_loop)]
fn build_mel_filters(
    num_filters: usize,
    fft_size: usize,
    sample_rate: f64,
    low_freq: f64,
    high_freq: f64,
) -> Vec<Vec<f64>> {
    let num_bins = fft_size / 2 + 1;
    let mel_low = hz_to_mel(low_freq);
    let mel_high = hz_to_mel(high_freq);

    // Create equally spaced Mel points
    let mel_points: Vec<f64> = (0..=(num_filters + 1))
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (num_filters + 1) as f64)
        .collect();

    // Convert back to Hz and then to FFT bin indices
    let bin_indices: Vec<usize> = mel_points
        .iter()
        .map(|&m| {
            let hz = mel_to_hz(m);
            let bin = (hz * fft_size as f64 / sample_rate).floor() as usize;
            bin.min(num_bins.saturating_sub(1))
        })
        .collect();

    let mut filters = Vec::with_capacity(num_filters);
    for f in 0..num_filters {
        let mut filter = vec![0.0_f64; num_bins];
        let left = bin_indices[f];
        let center = bin_indices[f + 1];
        let right = bin_indices[f + 2];

        // Rising slope
        if center > left {
            for k in left..=center {
                filter[k] = (k - left) as f64 / (center - left) as f64;
            }
        }
        // Falling slope
        if right > center {
            for k in center..=right.min(num_bins - 1) {
                filter[k] = (right - k) as f64 / (right - center) as f64;
            }
        }
        filters.push(filter);
    }
    filters
}

/// Apply Discrete Cosine Transform (type II) to log Mel energies.
#[allow(clippy::cast_precision_loss)]
fn dct_ii(input: &[f64], num_output: usize) -> Vec<f64> {
    let n = input.len();
    (0..num_output)
        .map(|k| {
            let sum: f64 = input
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    val * (PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos()
                })
                .sum();
            sum
        })
        .collect()
}

/// Apply sinusoidal liftering to cepstral coefficients.
#[allow(clippy::cast_precision_loss)]
fn lifter(coeffs: &mut [f64], lift_coeff: usize) {
    let l = lift_coeff as f64;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c *= 1.0 + (l / 2.0) * (PI * i as f64 / l).sin();
    }
}

impl CepstralAnalyzer {
    /// Create a new cepstral analyzer for the given sample rate.
    #[must_use]
    pub fn new(config: CepstralConfig, sample_rate: f64) -> Self {
        let mel_filters = build_mel_filters(
            config.num_filters,
            config.fft_size,
            sample_rate,
            config.low_freq,
            config.high_freq.min(sample_rate / 2.0),
        );
        Self {
            config,
            mel_filters,
        }
    }

    /// Analyze a single frame of power spectrum data.
    ///
    /// `power_spectrum` should contain `fft_size / 2 + 1` bins.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn analyze_frame(&self, power_spectrum: &[f64]) -> CepstralFrame {
        // Apply Mel filter bank
        let mel_energies: Vec<f64> = self
            .mel_filters
            .iter()
            .map(|filter| {
                let energy: f64 = filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                if energy > 1e-30 {
                    energy.ln()
                } else {
                    -69.0
                } // floor
            })
            .collect();

        // DCT to get MFCCs
        let mut mfcc = dct_ii(&mel_energies, self.config.num_coefficients);

        // Optional liftering
        if self.config.apply_lifter {
            lifter(&mut mfcc, self.config.lifter_coeff);
        }

        // Frame energy
        let total_energy: f64 = power_spectrum.iter().sum();
        let energy_db = if total_energy > 1e-30 {
            10.0 * total_energy.log10()
        } else {
            -100.0
        };

        // Cepstral peak prominence: compute real cepstrum, find peak
        let cepstrum: Vec<f64> = power_spectrum
            .iter()
            .map(|&p| if p > 1e-30 { p.ln() } else { -69.0 })
            .collect();

        let (peak_bin, peak_val) = cepstrum
            .iter()
            .enumerate()
            .skip(2) // skip DC and first bin
            .fold((0_usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });

        // CPP = peak value minus average
        let avg = cepstrum.iter().sum::<f64>() / cepstrum.len().max(1) as f64;
        let cpp_db = peak_val - avg;

        CepstralFrame {
            mfcc,
            cpp_db,
            peak_quefrency_bin: peak_bin,
            energy_db,
        }
    }

    /// Analyze multiple frames and return all cepstral frames.
    #[must_use]
    pub fn analyze_frames(&self, power_spectra: &[Vec<f64>]) -> Vec<CepstralFrame> {
        power_spectra
            .iter()
            .map(|spectrum| self.analyze_frame(spectrum))
            .collect()
    }

    /// Compute delta (first derivative) features from a sequence of MFCC vectors.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute_deltas(frames: &[CepstralFrame], context: usize) -> Vec<Vec<f64>> {
        let n = frames.len();
        if n == 0 {
            return Vec::new();
        }
        let num_coeffs = frames[0].mfcc.len();
        let mut deltas = Vec::with_capacity(n);
        let denom: f64 = (1..=context).map(|k| 2.0 * (k as f64) * (k as f64)).sum();
        let denom = if denom > 0.0 { denom } else { 1.0 };

        for t in 0..n {
            let mut delta = vec![0.0_f64; num_coeffs];
            for k in 1..=context {
                let prev = t.saturating_sub(k);
                let next = if t + k < n { t + k } else { n - 1 };
                for (c, d) in delta.iter_mut().enumerate() {
                    *d += k as f64 * (frames[next].mfcc[c] - frames[prev].mfcc[c]);
                }
            }
            for d in &mut delta {
                *d /= denom;
            }
            deltas.push(delta);
        }
        deltas
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &CepstralConfig {
        &self.config
    }
}

/// Compute cepstral distance between two MFCC vectors (Euclidean).
#[must_use]
pub fn cepstral_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let sum: f64 = (0..n).map(|i| (a[i] - b[i]).powi(2)).sum();
    sum.sqrt()
}

/// Compute mean MFCC vector over a set of frames.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn mean_mfcc(frames: &[CepstralFrame]) -> Vec<f64> {
    if frames.is_empty() {
        return Vec::new();
    }
    let n = frames[0].mfcc.len();
    let mut mean = vec![0.0_f64; n];
    for frame in frames {
        for (i, &v) in frame.mfcc.iter().enumerate() {
            if i < n {
                mean[i] += v;
            }
        }
    }
    let count = frames.len() as f64;
    for m in &mut mean {
        *m /= count;
    }
    mean
}

// ── Full MFCC extraction from raw audio samples ─────────────────────────────

/// Configuration for full MFCC extraction from raw audio.
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Number of MFCC coefficients to extract (default 13).
    pub num_coefficients: usize,
    /// Number of Mel filter banks (default 26).
    pub num_filters: usize,
    /// FFT size (default 1024).
    pub fft_size: usize,
    /// Hop size between frames (default 512).
    pub hop_size: usize,
    /// Low frequency cutoff in Hz (default 20).
    pub low_freq: f64,
    /// High frequency cutoff in Hz (default 8000).
    pub high_freq: f64,
    /// Pre-emphasis coefficient (0.0 = none, 0.97 = typical, default 0.97).
    pub pre_emphasis: f64,
    /// Whether to include delta (first-derivative) features.
    pub include_deltas: bool,
    /// Whether to include delta-delta (second-derivative) features.
    pub include_delta_deltas: bool,
    /// Delta context window size (default 2).
    pub delta_context: usize,
    /// Whether to apply sinusoidal liftering (default true).
    pub apply_lifter: bool,
    /// Liftering coefficient (default 22).
    pub lifter_coeff: usize,
    /// Whether to replace C0 with log energy (default true).
    pub use_energy: bool,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            num_coefficients: 13,
            num_filters: 26,
            fft_size: 1024,
            hop_size: 512,
            low_freq: 20.0,
            high_freq: 8000.0,
            pre_emphasis: 0.97,
            include_deltas: true,
            include_delta_deltas: false,
            delta_context: 2,
            apply_lifter: true,
            lifter_coeff: 22,
            use_energy: true,
        }
    }
}

/// Complete MFCC extraction result from raw audio.
#[derive(Debug, Clone)]
pub struct MfccResult {
    /// MFCC coefficients per frame (num_frames x num_coefficients).
    pub mfcc: Vec<Vec<f64>>,
    /// Delta (first-derivative) MFCC coefficients per frame, if computed.
    pub deltas: Option<Vec<Vec<f64>>>,
    /// Delta-delta (second-derivative) coefficients per frame, if computed.
    pub delta_deltas: Option<Vec<Vec<f64>>>,
    /// Frame-level log energies.
    pub frame_energies: Vec<f64>,
    /// Number of frames extracted.
    pub num_frames: usize,
    /// Number of MFCC coefficients per frame.
    pub num_coefficients: usize,
    /// Sample rate of the analyzed audio.
    pub sample_rate: f64,
}

/// Extract MFCCs from raw audio samples.
///
/// This is a complete pipeline: pre-emphasis -> framing -> windowing -> FFT ->
/// power spectrum -> mel filter bank -> log compression -> DCT -> liftering ->
/// optional deltas. Suitable for speech recognition and audio ML features.
///
/// # Arguments
/// * `samples` - Mono audio samples (f32)
/// * `sample_rate` - Sample rate in Hz
/// * `config` - MFCC extraction configuration
///
/// # Returns
/// `MfccResult` containing per-frame MFCC vectors and optional delta features.
///
/// # Errors
/// Returns error if samples are too short or sample rate is invalid.
pub fn extract_mfcc(
    samples: &[f32],
    sample_rate: f64,
    config: &MfccConfig,
) -> std::result::Result<MfccResult, crate::AnalysisError> {
    if sample_rate <= 0.0 || sample_rate > 192_000.0 {
        return Err(crate::AnalysisError::InvalidSampleRate(sample_rate as f32));
    }
    if samples.len() < config.fft_size {
        return Err(crate::AnalysisError::InsufficientSamples {
            needed: config.fft_size,
            got: samples.len(),
        });
    }

    let effective_high = config.high_freq.min(sample_rate / 2.0);

    // Build mel filter bank
    let mel_filters = build_mel_filters(
        config.num_filters,
        config.fft_size,
        sample_rate,
        config.low_freq,
        effective_high,
    );

    // Pre-emphasis
    let emphasized = if config.pre_emphasis > 0.0 {
        let mut out = Vec::with_capacity(samples.len());
        out.push(f64::from(samples[0]));
        for i in 1..samples.len() {
            out.push(f64::from(samples[i]) - config.pre_emphasis * f64::from(samples[i - 1]));
        }
        out
    } else {
        samples.iter().map(|&s| f64::from(s)).collect()
    };

    // Hann window
    let window: Vec<f64> = (0..config.fft_size)
        .map(|i| {
            let x = PI * i as f64 / (config.fft_size.saturating_sub(1)) as f64;
            0.5 * (1.0 - x.cos())
        })
        .collect();

    let num_bins = config.fft_size / 2 + 1;

    // Process frames
    let num_frames = if emphasized.len() >= config.fft_size {
        (emphasized.len() - config.fft_size) / config.hop_size + 1
    } else {
        0
    };

    let mut all_mfcc = Vec::with_capacity(num_frames);
    let mut frame_energies = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * config.hop_size;
        let end = start + config.fft_size;
        if end > emphasized.len() {
            break;
        }

        let frame = &emphasized[start..end];

        // Apply window
        let windowed: Vec<f64> = frame.iter().zip(&window).map(|(&s, &w)| s * w).collect();

        // Compute frame energy (before FFT for log energy feature)
        let frame_energy: f64 = windowed.iter().map(|&x| x * x).sum();
        let log_energy = if frame_energy > 1e-30 {
            frame_energy.ln()
        } else {
            -69.0
        };
        frame_energies.push(log_energy);

        // FFT using oxifft
        let complex_input: Vec<oxifft::Complex<f64>> = windowed
            .iter()
            .map(|&s| oxifft::Complex::new(s, 0.0))
            .collect();

        let fft_output = oxifft::fft(&complex_input);

        // Power spectrum
        let power_spectrum: Vec<f64> = fft_output[..num_bins]
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        // Apply mel filter bank
        let mel_energies: Vec<f64> = mel_filters
            .iter()
            .map(|filter| {
                let energy: f64 = filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                if energy > 1e-30 {
                    energy.ln()
                } else {
                    -69.0
                }
            })
            .collect();

        // DCT-II to get MFCCs
        let mut mfcc = dct_ii(&mel_energies, config.num_coefficients);

        // Replace C0 with log energy if requested
        if config.use_energy && !mfcc.is_empty() {
            mfcc[0] = log_energy;
        }

        // Liftering
        if config.apply_lifter {
            lifter(&mut mfcc, config.lifter_coeff);
        }

        all_mfcc.push(mfcc);
    }

    // Compute deltas
    let deltas = if config.include_deltas && !all_mfcc.is_empty() {
        Some(compute_deltas_from_vecs(&all_mfcc, config.delta_context))
    } else {
        None
    };

    // Compute delta-deltas
    let delta_deltas = if config.include_delta_deltas {
        deltas
            .as_ref()
            .map(|d| compute_deltas_from_vecs(d, config.delta_context))
    } else {
        None
    };

    let actual_frames = all_mfcc.len();

    Ok(MfccResult {
        mfcc: all_mfcc,
        deltas,
        delta_deltas,
        frame_energies,
        num_frames: actual_frames,
        num_coefficients: config.num_coefficients,
        sample_rate,
    })
}

/// Compute delta features from a sequence of coefficient vectors.
#[allow(clippy::cast_precision_loss)]
fn compute_deltas_from_vecs(frames: &[Vec<f64>], context: usize) -> Vec<Vec<f64>> {
    let n = frames.len();
    if n == 0 {
        return Vec::new();
    }
    let num_coeffs = frames[0].len();
    let denom: f64 = (1..=context).map(|k| 2.0 * (k as f64) * (k as f64)).sum();
    let denom = if denom > 0.0 { denom } else { 1.0 };

    let mut deltas = Vec::with_capacity(n);
    for t in 0..n {
        let mut delta = vec![0.0_f64; num_coeffs];
        for k in 1..=context {
            let prev = t.saturating_sub(k);
            let next = if t + k < n { t + k } else { n - 1 };
            for c in 0..num_coeffs {
                delta[c] += k as f64 * (frames[next][c] - frames[prev][c]);
            }
        }
        for d in &mut delta {
            *d /= denom;
        }
        deltas.push(delta);
    }
    deltas
}

/// Compute MFCC variance across frames (useful for voice activity detection).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mfcc_variance(mfcc_result: &MfccResult) -> Vec<f64> {
    if mfcc_result.mfcc.is_empty() {
        return Vec::new();
    }
    let n = mfcc_result.num_coefficients;
    let mean = mean_mfcc_from_vecs(&mfcc_result.mfcc);
    let count = mfcc_result.mfcc.len() as f64;

    let mut variance = vec![0.0_f64; n];
    for frame in &mfcc_result.mfcc {
        for (i, &v) in frame.iter().enumerate() {
            if i < n {
                variance[i] += (v - mean[i]) * (v - mean[i]);
            }
        }
    }
    for v in &mut variance {
        *v /= count.max(1.0);
    }
    variance
}

/// Compute mean MFCC from raw coefficient vectors.
#[allow(clippy::cast_precision_loss)]
fn mean_mfcc_from_vecs(frames: &[Vec<f64>]) -> Vec<f64> {
    if frames.is_empty() {
        return Vec::new();
    }
    let n = frames[0].len();
    let mut mean = vec![0.0_f64; n];
    for frame in frames {
        for (i, &v) in frame.iter().enumerate() {
            if i < n {
                mean[i] += v;
            }
        }
    }
    let count = frames.len() as f64;
    for m in &mut mean {
        *m /= count;
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_flat_spectrum(num_bins: usize, value: f64) -> Vec<f64> {
        vec![value; num_bins]
    }

    #[test]
    fn test_hz_to_mel_and_back() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 1e-6);
    }

    #[test]
    fn test_hz_to_mel_zero() {
        assert!((hz_to_mel(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mel_filters_shape() {
        let filters = build_mel_filters(26, 1024, 16000.0, 20.0, 8000.0);
        assert_eq!(filters.len(), 26);
        for f in &filters {
            assert_eq!(f.len(), 513); // 1024/2+1
        }
    }

    #[test]
    fn test_mel_filters_non_negative() {
        let filters = build_mel_filters(10, 512, 16000.0, 0.0, 8000.0);
        for f in &filters {
            for &v in f {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_dct_ii_length() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = dct_ii(&input, 3);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_lifter_modifies_coefficients() {
        let mut coeffs = vec![1.0; 13];
        let original = coeffs.clone();
        lifter(&mut coeffs, 22);
        // First coefficient (i=0) stays the same (sin(0)=0, so multiplier=1.0)
        assert!((coeffs[0] - original[0]).abs() < 1e-10);
        // Others should be different
        assert!((coeffs[6] - original[6]).abs() > 0.01);
    }

    #[test]
    fn test_analyzer_creation() {
        let config = CepstralConfig::default();
        let analyzer = CepstralAnalyzer::new(config, 16000.0);
        assert_eq!(analyzer.config().num_coefficients, 13);
        assert_eq!(analyzer.config().num_filters, 26);
    }

    #[test]
    fn test_analyze_frame_mfcc_count() {
        let config = CepstralConfig {
            num_coefficients: 13,
            num_filters: 26,
            fft_size: 512,
            ..CepstralConfig::default()
        };
        let analyzer = CepstralAnalyzer::new(config, 16000.0);
        let spectrum = make_flat_spectrum(257, 0.01);
        let frame = analyzer.analyze_frame(&spectrum);
        assert_eq!(frame.mfcc.len(), 13);
    }

    #[test]
    fn test_analyze_frame_energy() {
        let config = CepstralConfig {
            fft_size: 512,
            ..CepstralConfig::default()
        };
        let analyzer = CepstralAnalyzer::new(config, 16000.0);
        let loud = make_flat_spectrum(257, 1.0);
        let quiet = make_flat_spectrum(257, 0.001);
        let frame_loud = analyzer.analyze_frame(&loud);
        let frame_quiet = analyzer.analyze_frame(&quiet);
        assert!(frame_loud.energy_db > frame_quiet.energy_db);
    }

    #[test]
    fn test_analyze_frames_multiple() {
        let config = CepstralConfig {
            fft_size: 512,
            ..CepstralConfig::default()
        };
        let analyzer = CepstralAnalyzer::new(config, 16000.0);
        let spectra = vec![
            make_flat_spectrum(257, 0.01),
            make_flat_spectrum(257, 0.02),
            make_flat_spectrum(257, 0.03),
        ];
        let results = analyzer.analyze_frames(&spectra);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compute_deltas() {
        let config = CepstralConfig {
            fft_size: 512,
            num_coefficients: 5,
            num_filters: 10,
            ..CepstralConfig::default()
        };
        let analyzer = CepstralAnalyzer::new(config, 16000.0);
        let spectra: Vec<Vec<f64>> = (1..=5)
            .map(|i| make_flat_spectrum(257, 0.01 * i as f64))
            .collect();
        let frames = analyzer.analyze_frames(&spectra);
        let deltas = CepstralAnalyzer::compute_deltas(&frames, 2);
        assert_eq!(deltas.len(), 5);
        assert_eq!(deltas[0].len(), 5);
    }

    #[test]
    fn test_cepstral_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cepstral_distance(&a, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_cepstral_distance_nonzero() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        assert!((cepstral_distance(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_mfcc_empty() {
        let result = mean_mfcc(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mean_mfcc_single() {
        let frame = CepstralFrame {
            mfcc: vec![1.0, 2.0, 3.0],
            cpp_db: 0.0,
            peak_quefrency_bin: 0,
            energy_db: 0.0,
        };
        let mean = mean_mfcc(&[frame]);
        assert!((mean[0] - 1.0).abs() < 1e-10);
        assert!((mean[1] - 2.0).abs() < 1e-10);
        assert!((mean[2] - 3.0).abs() < 1e-10);
    }

    // ── extract_mfcc tests ──────────────────────────────────────────────────

    fn generate_sine(freq: f64, sample_rate: f64, duration: f64) -> Vec<f32> {
        let num_samples = (sample_rate * duration) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_extract_mfcc_basic() {
        let samples = generate_sine(440.0, 16000.0, 0.5);
        let config = MfccConfig::default();
        let result = extract_mfcc(&samples, 16000.0, &config);
        assert!(result.is_ok());
        let result = result.expect("mfcc extraction should succeed");
        assert!(result.num_frames > 0);
        assert_eq!(result.num_coefficients, 13);
        for frame in &result.mfcc {
            assert_eq!(frame.len(), 13);
        }
    }

    #[test]
    fn test_extract_mfcc_with_deltas() {
        let samples = generate_sine(300.0, 16000.0, 0.5);
        let config = MfccConfig {
            include_deltas: true,
            include_delta_deltas: true,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, 16000.0, &config).expect("should succeed");
        assert!(result.deltas.is_some());
        assert!(result.delta_deltas.is_some());
        let deltas = result.deltas.as_ref().expect("deltas should exist");
        assert_eq!(deltas.len(), result.num_frames);
        let dd = result
            .delta_deltas
            .as_ref()
            .expect("delta-deltas should exist");
        assert_eq!(dd.len(), result.num_frames);
    }

    #[test]
    fn test_extract_mfcc_no_deltas() {
        let samples = generate_sine(440.0, 16000.0, 0.3);
        let config = MfccConfig {
            include_deltas: false,
            include_delta_deltas: false,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, 16000.0, &config).expect("should succeed");
        assert!(result.deltas.is_none());
        assert!(result.delta_deltas.is_none());
    }

    #[test]
    fn test_extract_mfcc_invalid_sample_rate() {
        let samples = vec![0.0_f32; 2048];
        let config = MfccConfig::default();
        let result = extract_mfcc(&samples, -1.0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_mfcc_too_short() {
        let samples = vec![0.0_f32; 100];
        let config = MfccConfig {
            fft_size: 1024,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, 16000.0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_mfcc_energy_feature() {
        let loud = generate_sine(440.0, 16000.0, 0.3);
        let quiet: Vec<f32> = loud.iter().map(|&s| s * 0.01).collect();
        let config = MfccConfig {
            use_energy: true,
            ..MfccConfig::default()
        };
        let r_loud = extract_mfcc(&loud, 16000.0, &config).expect("should succeed");
        let r_quiet = extract_mfcc(&quiet, 16000.0, &config).expect("should succeed");
        // Loud signal should have higher energy in C0
        let loud_c0_mean: f64 =
            r_loud.mfcc.iter().map(|f| f[0]).sum::<f64>() / r_loud.num_frames as f64;
        let quiet_c0_mean: f64 =
            r_quiet.mfcc.iter().map(|f| f[0]).sum::<f64>() / r_quiet.num_frames as f64;
        assert!(
            loud_c0_mean > quiet_c0_mean,
            "Loud C0 ({loud_c0_mean}) should exceed quiet C0 ({quiet_c0_mean})"
        );
    }

    #[test]
    fn test_extract_mfcc_different_frequencies() {
        let low = generate_sine(200.0, 16000.0, 0.5);
        let high = generate_sine(2000.0, 16000.0, 0.5);
        let config = MfccConfig::default();
        let r_low = extract_mfcc(&low, 16000.0, &config).expect("should succeed");
        let r_high = extract_mfcc(&high, 16000.0, &config).expect("should succeed");
        // MFCCs should differ between low and high frequency signals
        let mean_low = mean_mfcc_from_vecs(&r_low.mfcc);
        let mean_high = mean_mfcc_from_vecs(&r_high.mfcc);
        let dist: f64 = mean_low
            .iter()
            .zip(&mean_high)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            dist > 0.1,
            "MFCCs for 200Hz and 2000Hz should differ, distance={dist}"
        );
    }

    #[test]
    fn test_extract_mfcc_no_pre_emphasis() {
        let samples = generate_sine(440.0, 16000.0, 0.3);
        let config = MfccConfig {
            pre_emphasis: 0.0,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, 16000.0, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mfcc_variance_nonzero_for_varying_signal() {
        // Create a signal that changes over time (frequency sweep)
        let sample_rate = 16000.0;
        let num_samples = 8000;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let freq = 200.0 + 2000.0 * t; // sweep from 200 to ~3200 Hz
                (2.0 * PI * freq * t).sin() as f32
            })
            .collect();
        let config = MfccConfig {
            include_deltas: false,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, sample_rate, &config).expect("should succeed");
        let var = mfcc_variance(&result);
        assert!(!var.is_empty());
        // At least some coefficients should have non-trivial variance
        let total_var: f64 = var.iter().sum();
        assert!(
            total_var > 0.0,
            "Varying signal should have nonzero MFCC variance"
        );
    }

    #[test]
    fn test_mfcc_variance_empty() {
        let result = MfccResult {
            mfcc: vec![],
            deltas: None,
            delta_deltas: None,
            frame_energies: vec![],
            num_frames: 0,
            num_coefficients: 13,
            sample_rate: 16000.0,
        };
        let var = mfcc_variance(&result);
        assert!(var.is_empty());
    }

    #[test]
    fn test_extract_mfcc_frame_count() {
        let sample_rate = 16000.0;
        let duration = 1.0;
        let samples = generate_sine(440.0, sample_rate, duration);
        let config = MfccConfig {
            fft_size: 512,
            hop_size: 256,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, sample_rate, &config).expect("should succeed");
        let expected_frames = (samples.len() - config.fft_size) / config.hop_size + 1;
        assert_eq!(result.num_frames, expected_frames);
    }

    #[test]
    fn test_extract_mfcc_custom_coefficients() {
        let samples = generate_sine(440.0, 16000.0, 0.3);
        let config = MfccConfig {
            num_coefficients: 20,
            num_filters: 40,
            ..MfccConfig::default()
        };
        let result = extract_mfcc(&samples, 16000.0, &config).expect("should succeed");
        assert_eq!(result.num_coefficients, 20);
        for frame in &result.mfcc {
            assert_eq!(frame.len(), 20);
        }
    }
}
