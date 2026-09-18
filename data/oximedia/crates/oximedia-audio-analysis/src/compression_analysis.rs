//! Audio compression analysis: bitrate estimation, codec artefact detection,
//! spectral hole detection, and compression history scoring.
//!
//! These tools are useful for audio forensics, quality control, and codec
//! research.  They operate entirely in the time-domain or on magnitude spectra
//! and require no external dependencies beyond the standard library.

#![allow(dead_code)]

/// Result of a bitrate estimation heuristic.
#[derive(Debug, Clone, PartialEq)]
pub struct BitrateEstimate {
    /// Estimated nominal bitrate class in kbps (e.g. 64, 128, 192, 320).
    pub kbps_class: u32,
    /// Confidence score in [0, 1].
    pub confidence: f32,
}

impl BitrateEstimate {
    /// Creates a new [`BitrateEstimate`].
    #[must_use]
    pub fn new(kbps_class: u32, confidence: f32) -> Self {
        Self {
            kbps_class,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Represents a spectral hole – a narrow frequency band with anomalously low
/// energy relative to neighbouring bands.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralHole {
    /// Centre frequency of the hole in Hz.
    pub center_hz: f32,
    /// Width of the hole in Hz.
    pub width_hz: f32,
    /// Relative depth (how much lower than neighbours, in dB).
    pub depth_db: f32,
}

/// A scored summary of the compression history analysis.
#[derive(Debug, Clone)]
pub struct CompressionScore {
    /// Number of spectral holes detected (proxy for codec artefacts).
    pub hole_count: usize,
    /// Mean depth of detected holes in dB.
    pub mean_hole_depth_db: f32,
    /// Estimated maximum frequency cutoff (often lowered by lossy codecs) in Hz.
    pub estimated_cutoff_hz: f32,
    /// Normalised compression score: 0 = no compression detected, 1 = heavily compressed.
    pub score: f32,
}

/// Detects spectral holes in a magnitude spectrum.
///
/// A "hole" is a bin (or small range of bins) whose magnitude is significantly
/// lower than the average of its immediate neighbours.
///
/// # Arguments
/// * `magnitudes` – Magnitude spectrum (linear scale).
/// * `sample_rate` – Audio sample rate in Hz.
/// * `threshold_db` – Depth threshold: holes must be at least this many dB below neighbours.
/// * `window_bins` – Number of bins on each side used for the local average.
#[must_use]
pub fn detect_spectral_holes(
    magnitudes: &[f32],
    sample_rate: f32,
    threshold_db: f32,
    window_bins: usize,
) -> Vec<SpectralHole> {
    let n = magnitudes.len();
    if n < 2 * window_bins + 1 || sample_rate <= 0.0 {
        return Vec::new();
    }

    let n_fft = (n - 1) * 2;
    let hz_per_bin = sample_rate / n_fft as f32;

    let mut holes = Vec::new();
    let mut skip_until = 0usize;

    for i in window_bins..n - window_bins {
        if i < skip_until {
            continue;
        }

        let left_avg: f32 = magnitudes[i - window_bins..i].iter().sum::<f32>() / window_bins as f32;
        let right_avg: f32 =
            magnitudes[i + 1..i + 1 + window_bins].iter().sum::<f32>() / window_bins as f32;
        let neighbour_avg = (left_avg + right_avg) * 0.5;

        let bin_val = magnitudes[i].max(1e-10);
        let ratio = neighbour_avg / bin_val;
        let depth_db = 20.0 * ratio.max(1e-10).log10();

        if depth_db >= threshold_db {
            let center_hz = i as f32 * hz_per_bin;
            holes.push(SpectralHole {
                center_hz,
                width_hz: hz_per_bin,
                depth_db,
            });
            skip_until = i + window_bins; // suppress adjacent detections
        }
    }

    holes
}

/// Estimates the effective high-frequency cutoff of the spectrum.
///
/// Lossy codecs often attenuate or zero out frequencies above a cutoff (e.g.
/// 16 kHz for 128 kbps MP3).  This function searches from the top of the
/// spectrum downward for the last bin above a noise floor.
///
/// # Arguments
/// * `magnitudes` – Magnitude spectrum.
/// * `sample_rate` – Audio sample rate in Hz.
/// * `noise_floor_db` – Bins below this level relative to the peak are ignored.
#[must_use]
pub fn estimate_cutoff_frequency(magnitudes: &[f32], sample_rate: f32, noise_floor_db: f32) -> f32 {
    let n = magnitudes.len();
    if n == 0 || sample_rate <= 0.0 {
        return 0.0;
    }

    let peak = magnitudes.iter().copied().fold(0.0_f32, f32::max);
    if peak < 1e-10 {
        return 0.0;
    }

    let min_linear = peak * 10.0_f32.powf(noise_floor_db / 20.0);
    let n_fft = (n - 1) * 2;
    let hz_per_bin = sample_rate / n_fft as f32;

    for (i, &m) in magnitudes.iter().enumerate().rev() {
        if m >= min_linear {
            return i as f32 * hz_per_bin;
        }
    }
    0.0
}

/// Estimates the bitrate class of a potentially compressed audio signal based
/// on the effective frequency cutoff.
///
/// This heuristic uses typical MP3/AAC cutoff frequencies.
#[must_use]
pub fn estimate_bitrate_class(cutoff_hz: f32) -> BitrateEstimate {
    // Typical cutoffs (approximate):
    //   64 kbps:  ~11 kHz
    //   96 kbps:  ~14 kHz
    //  128 kbps:  ~16 kHz
    //  192 kbps:  ~19 kHz
    //  320 kbps:  ~20 kHz (near Nyquist at 44.1 kHz)
    let (kbps, confidence): (u32, f32) = if cutoff_hz < 12_000.0 {
        (64, 0.8)
    } else if cutoff_hz < 15_000.0 {
        (96, 0.7)
    } else if cutoff_hz < 17_500.0 {
        (128, 0.75)
    } else if cutoff_hz < 19_500.0 {
        (192, 0.7)
    } else {
        (320, 0.6)
    };

    BitrateEstimate::new(kbps, confidence)
}

/// Computes a composite compression score for the audio.
#[must_use]
pub fn analyse_compression(
    magnitudes: &[f32],
    sample_rate: f32,
    noise_floor_db: f32,
) -> CompressionScore {
    let holes = detect_spectral_holes(magnitudes, sample_rate, 15.0, 4);
    let hole_count = holes.len();

    let mean_hole_depth_db = if hole_count == 0 {
        0.0
    } else {
        holes.iter().map(|h| h.depth_db).sum::<f32>() / hole_count as f32
    };

    let estimated_cutoff_hz = estimate_cutoff_frequency(magnitudes, sample_rate, noise_floor_db);
    let nyquist = sample_rate / 2.0;

    // Score increases when cutoff is below Nyquist and holes are present
    let cutoff_score = if nyquist > 0.0 {
        1.0 - (estimated_cutoff_hz / nyquist).min(1.0)
    } else {
        0.0
    };
    let hole_score = (hole_count as f32 / 10.0).min(1.0);
    let score = (cutoff_score * 0.6 + hole_score * 0.4).clamp(0.0, 1.0);

    CompressionScore {
        hole_count,
        mean_hole_depth_db,
        estimated_cutoff_hz,
        score,
    }
}

/// Computes the entropy of a magnitude spectrum as a measure of complexity.
///
/// Higher entropy indicates a more spectrally complex (less compressed) signal.
#[must_use]
pub fn spectral_entropy(magnitudes: &[f32]) -> f32 {
    let total: f32 = magnitudes.iter().sum();
    if total < 1e-10 {
        return 0.0;
    }

    magnitudes
        .iter()
        .map(|&m| {
            let p = m / total;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Measures the spectral continuity index: average absolute difference between
/// adjacent bins normalised by the mean magnitude.
///
/// High continuity (low index) suggests smooth spectra typical of heavily coded
/// audio, while complex spectra have higher index values.
#[must_use]
pub fn spectral_continuity_index(magnitudes: &[f32]) -> f32 {
    let n = magnitudes.len();
    if n < 2 {
        return 0.0;
    }

    let mean: f32 = magnitudes.iter().sum::<f32>() / n as f32;
    if mean < 1e-10 {
        return 0.0;
    }

    let diff_sum: f32 = magnitudes.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

    diff_sum / ((n - 1) as f32 * mean)
}

/// Generates a test magnitude spectrum with a simulated high-frequency cutoff.
#[must_use]
pub fn synthetic_compressed_spectrum(n_bins: usize, cutoff_bin: usize) -> Vec<f32> {
    (0..n_bins)
        .map(|i| if i <= cutoff_bin { 1.0f32 } else { 1e-6 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_spectrum(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    #[test]
    fn test_bitrate_estimate_clamp() {
        let b = BitrateEstimate::new(128, 1.5);
        assert!((b.confidence - 1.0).abs() < 1e-6);
        let b2 = BitrateEstimate::new(128, -0.5);
        assert!((b2.confidence).abs() < 1e-6);
    }

    #[test]
    fn test_detect_spectral_holes_empty_on_flat() {
        // A perfectly flat spectrum should have no holes
        let spec = flat_spectrum(513);
        let holes = detect_spectral_holes(&spec, 44100.0, 15.0, 4);
        assert!(
            holes.is_empty(),
            "expected no holes in flat spectrum, got {}",
            holes.len()
        );
    }

    #[test]
    fn test_detect_spectral_holes_finds_notch() {
        let mut spec = flat_spectrum(513);
        // Introduce a notch at bin 100
        spec[100] = 1e-6;
        let holes = detect_spectral_holes(&spec, 44100.0, 10.0, 3);
        assert!(!holes.is_empty(), "should detect notch");
    }

    #[test]
    fn test_detect_spectral_holes_positive_depth() {
        let mut spec = flat_spectrum(513);
        spec[200] = 1e-6;
        let holes = detect_spectral_holes(&spec, 44100.0, 10.0, 3);
        for h in &holes {
            assert!(h.depth_db > 0.0, "hole depth should be positive");
        }
    }

    #[test]
    fn test_estimate_cutoff_full_spectrum() {
        let spec = flat_spectrum(513);
        let cutoff = estimate_cutoff_frequency(&spec, 44100.0, -60.0);
        // Full flat spectrum: cutoff should be near Nyquist
        assert!(cutoff > 20_000.0, "cutoff = {cutoff}");
    }

    #[test]
    fn test_estimate_cutoff_low_cutoff() {
        let spec = synthetic_compressed_spectrum(513, 200);
        let cutoff = estimate_cutoff_frequency(&spec, 44100.0, -60.0);
        // cutoff bin 200 out of 512 → freq ≈ 200 * 44100 / 1024 ≈ 8613 Hz
        assert!(cutoff < 12_000.0, "cutoff = {cutoff}");
    }

    #[test]
    fn test_estimate_bitrate_class_low_cutoff() {
        let est = estimate_bitrate_class(10_000.0);
        assert_eq!(est.kbps_class, 64);
    }

    #[test]
    fn test_estimate_bitrate_class_high_cutoff() {
        let est = estimate_bitrate_class(20_000.0);
        assert_eq!(est.kbps_class, 320);
    }

    #[test]
    fn test_estimate_bitrate_class_mid_cutoff() {
        let est = estimate_bitrate_class(16_000.0);
        assert_eq!(est.kbps_class, 128);
    }

    #[test]
    fn test_analyse_compression_score_range() {
        let spec = flat_spectrum(513);
        let result = analyse_compression(&spec, 44100.0, -60.0);
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_analyse_compression_compressed_higher_score() {
        let compressed = synthetic_compressed_spectrum(513, 180);
        let full = flat_spectrum(513);
        let cs = analyse_compression(&compressed, 44100.0, -60.0);
        let fs = analyse_compression(&full, 44100.0, -60.0);
        assert!(
            cs.score >= fs.score,
            "compressed score {} should be >= full score {}",
            cs.score,
            fs.score
        );
    }

    #[test]
    fn test_spectral_entropy_flat() {
        let spec = flat_spectrum(512);
        let e = spectral_entropy(&spec);
        // All equal probabilities → maximum entropy
        let expected = (512.0_f32).ln();
        assert!(
            (e - expected).abs() < 0.01,
            "entropy = {e}, expected ~{expected}"
        );
    }

    #[test]
    fn test_spectral_entropy_impulse() {
        let mut spec = vec![0.0f32; 512];
        spec[0] = 1.0;
        let e = spectral_entropy(&spec);
        // Single non-zero bin → entropy = 0
        assert!(e.abs() < 1e-5, "impulse entropy should be 0, got {e}");
    }

    #[test]
    fn test_spectral_continuity_index_flat_is_zero() {
        let spec = flat_spectrum(128);
        let ci = spectral_continuity_index(&spec);
        assert!(
            ci < 1e-5,
            "flat spectrum continuity index should be ~0, got {ci}"
        );
    }

    #[test]
    fn test_spectral_continuity_index_noisy_positive() {
        // Alternating high/low → large differences
        let spec: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let ci = spectral_continuity_index(&spec);
        assert!(
            ci > 0.0,
            "noisy spectrum should have positive continuity index"
        );
    }

    #[test]
    fn test_synthetic_compressed_spectrum_length() {
        let spec = synthetic_compressed_spectrum(513, 300);
        assert_eq!(spec.len(), 513);
    }

    #[test]
    fn test_synthetic_compressed_spectrum_cutoff() {
        let spec = synthetic_compressed_spectrum(100, 50);
        assert!((spec[50] - 1.0).abs() < 1e-6);
        assert!(spec[51] < 1e-4);
    }
}
