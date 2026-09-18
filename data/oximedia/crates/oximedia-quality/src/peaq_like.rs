//! PEAQ-inspired perceptual audio quality metric.
//!
//! PEAQ (Perceptual Evaluation of Audio Quality, ITU-R BS.1387) maps three
//! Model Output Variables (MOVs) onto an Objective Difference Grade (ODG):
//!
//! | ODG  | Impairment |
//! |------|-----------|
//! |  0   | Imperceptible |
//! | -1   | Perceptible but not annoying |
//! | -2   | Slightly annoying |
//! | -3   | Annoying |
//! | -4   | Very annoying |
//!
//! This module provides a lightweight, patent-free approximation using three
//! MOVs computed directly from signal statistics:
//!
//! - **NoiseLoudness** — loudness of the error signal (ref − test)
//! - **BandwidthRef** — effective bandwidth of the reference (as fraction of Nyquist)
//! - **BandwidthTest** — effective bandwidth of the test signal
//!
//! ## Extended metrics
//!
//! - **MOV prediction** — convert ODG to a 1-5 Mean Opinion Value
//! - **Frequency masking** — bark-scale spreading function for critical bands
//! - **Loudness-weighted disturbance** — weight error by perceptual loudness model
//! - **Bandwidth limitation** — detect low-pass or band-limited codecs
//! - **Noise-to-mask ratio** — ratio of distortion energy to masking threshold
//!
//! # Reference
//!
//! ITU-R BS.1387-1 "Method for objective measurements of perceived audio quality."

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::f32::consts::PI;

// ── Weights (calibrated to approximate PEAQ Basic) ───────────────────────────
const W_NOISE: f64 = -3.2; // noise loudness → drives ODG down
const W_BW_DIFF: f64 = -1.0; // bandwidth mismatch penalty
const ODG_BIAS: f64 = 0.0; // perfect match → 0

/// Number of Bark critical bands (up to ~15.5 kHz at 44.1 kHz).
const NUM_BARK_BANDS: usize = 24;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the PEAQ-like quality estimator.
#[derive(Debug, Clone)]
pub struct PeaqLikeConfig {
    /// Audio sample rate in Hz (e.g. 44100 or 48000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u8,
}

impl PeaqLikeConfig {
    /// Construct a new configuration.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
        }
    }
}

impl Default for PeaqLikeConfig {
    fn default() -> Self {
        Self::new(44100, 1)
    }
}

// ── Model Output Variables ─────────────────────────────────────────────────

/// The three Model Output Variables (MOVs) computed from the signal pair.
#[derive(Debug, Clone)]
pub struct MovValues {
    /// Noise loudness: RMS of (ref − test) relative to RMS of ref. In \[0, ∞).
    pub noise_loudness: f64,
    /// Bandwidth of the reference signal as a fraction of Nyquist. In \[0, 1\].
    pub bandwidth_ref: f64,
    /// Bandwidth of the test signal as a fraction of Nyquist. In \[0, 1\].
    pub bandwidth_test: f64,
}

/// Extended analysis results including masking and NMR.
#[derive(Debug, Clone)]
pub struct ExtendedPeaqResult {
    /// Standard ODG in \[-4, 0\].
    pub odg: f64,
    /// Predicted Mean Opinion Value in \[1, 5\].
    pub mov: f64,
    /// Per-Bark-band noise-to-mask ratios (dB). Negative = masked, positive = audible.
    pub nmr_per_band: Vec<f64>,
    /// Total noise-to-mask ratio (dB).
    pub nmr_total: f64,
    /// Loudness-weighted disturbance in \[0, ∞). Lower is better.
    pub loudness_disturbance: f64,
    /// Whether bandwidth limitation was detected.
    pub bandwidth_limited: bool,
    /// Detected bandwidth cutoff as fraction of Nyquist (0.0 if not limited).
    pub bandwidth_cutoff: f64,
}

// ── Main API ──────────────────────────────────────────────────────────────────

/// Entry point for PEAQ-like score computation.
pub struct PeaqScore;

impl PeaqScore {
    /// Compute an ODG-like score from a reference/test audio pair.
    ///
    /// Both slices must contain interleaved samples (for stereo, L₀R₀L₁R₁…).
    /// The lengths do not need to match exactly; the shorter one determines the
    /// comparison window.
    ///
    /// Returns a value in \[-4.0, 0.0\] where 0.0 is imperceptible.
    #[must_use]
    pub fn compute(ref_audio: &[f32], test_audio: &[f32], config: &PeaqLikeConfig) -> f64 {
        if ref_audio.is_empty() || test_audio.is_empty() {
            return 0.0; // nothing to compare
        }

        let movs = Self::compute_movs(ref_audio, test_audio, config);
        Self::movs_to_odg(&movs)
    }

    /// Compute the three MOVs without combining them into a final ODG.
    #[must_use]
    pub fn compute_movs(
        ref_audio: &[f32],
        test_audio: &[f32],
        config: &PeaqLikeConfig,
    ) -> MovValues {
        let n = ref_audio.len().min(test_audio.len());

        let ref_s = &ref_audio[..n];
        let test_s = &test_audio[..n];

        let noise_loudness = compute_noise_loudness(ref_s, test_s);
        let bandwidth_ref = compute_bandwidth(ref_s, config.sample_rate);
        let bandwidth_test = compute_bandwidth(test_s, config.sample_rate);

        MovValues {
            noise_loudness,
            bandwidth_ref,
            bandwidth_test,
        }
    }

    /// Map MOVs → ODG in \[-4, 0\].
    #[must_use]
    pub fn movs_to_odg(movs: &MovValues) -> f64 {
        // Bandwidth difference penalty (0 when equal, up to 1 when very different)
        let bw_diff = (movs.bandwidth_ref - movs.bandwidth_test).abs().min(1.0);

        let raw = ODG_BIAS + W_NOISE * movs.noise_loudness + W_BW_DIFF * bw_diff;
        raw.clamp(-4.0, 0.0)
    }

    /// Convert ODG to predicted Mean Opinion Value (MOV) on a 1-5 scale.
    ///
    /// Maps linearly: ODG 0 → MOV 5, ODG -4 → MOV 1.
    #[must_use]
    pub fn odg_to_mov(odg: f64) -> f64 {
        // MOV = 5 + ODG (since ODG ∈ [-4, 0])
        (5.0 + odg.clamp(-4.0, 0.0)).clamp(1.0, 5.0)
    }

    /// Full extended analysis: ODG, MOV, NMR, masking, bandwidth detection.
    #[must_use]
    pub fn compute_extended(
        ref_audio: &[f32],
        test_audio: &[f32],
        config: &PeaqLikeConfig,
    ) -> ExtendedPeaqResult {
        let n = ref_audio.len().min(test_audio.len());
        if n == 0 {
            return ExtendedPeaqResult {
                odg: 0.0,
                mov: 5.0,
                nmr_per_band: vec![0.0; NUM_BARK_BANDS],
                nmr_total: f64::NEG_INFINITY,
                loudness_disturbance: 0.0,
                bandwidth_limited: false,
                bandwidth_cutoff: 0.0,
            };
        }

        let ref_s = &ref_audio[..n];
        let test_s = &test_audio[..n];

        let movs = Self::compute_movs(ref_audio, test_audio, config);
        let odg = Self::movs_to_odg(&movs);
        let mov = Self::odg_to_mov(odg);

        // Compute spectral representations
        let fft_n = fft_size(n.min(4096));
        let ref_mags = naive_rdft_magnitudes(ref_s, fft_n);
        let test_mags = naive_rdft_magnitudes(test_s, fft_n);

        // Bark-band energies
        let nyquist = config.sample_rate as f64 / 2.0;
        let ref_bark = bark_band_energies(&ref_mags, fft_n, nyquist);
        let test_bark = bark_band_energies(&test_mags, fft_n, nyquist);

        // Masking thresholds via spreading function
        let mask_thresh = compute_masking_thresholds(&ref_bark);

        // Noise-to-mask ratio per band
        let nmr_per_band = compute_nmr_per_band(&ref_bark, &test_bark, &mask_thresh);
        let nmr_total = compute_total_nmr(&nmr_per_band);

        // Loudness-weighted disturbance
        let loudness_disturbance = compute_loudness_disturbance(ref_s, test_s, &ref_bark);

        // Bandwidth limitation detection
        let (bandwidth_limited, bandwidth_cutoff) =
            detect_bandwidth_limitation(&test_mags, fft_n, nyquist);

        ExtendedPeaqResult {
            odg,
            mov,
            nmr_per_band,
            nmr_total,
            loudness_disturbance,
            bandwidth_limited,
            bandwidth_cutoff,
        }
    }
}

// ── NoiseLoudness ─────────────────────────────────────────────────────────────
//
// Defined as RMS(error) / (RMS(ref) + ε) so that:
//  • identical signals → 0.0
//  • silence ref vs loud noise → large value

/// Compute NoiseLoudness: RMS of (ref − test) / (RMS(ref) + ε).
fn compute_noise_loudness(ref_s: &[f32], test_s: &[f32]) -> f64 {
    let n = ref_s.len().min(test_s.len());
    if n == 0 {
        return 0.0;
    }

    let rms_err: f64 = ref_s
        .iter()
        .zip(test_s.iter())
        .map(|(&r, &t)| {
            let e = f64::from(r) - f64::from(t);
            e * e
        })
        .sum::<f64>()
        / n as f64;
    let rms_err = rms_err.sqrt();

    let rms_ref: f64 = ref_s
        .iter()
        .map(|&r| {
            let v = f64::from(r);
            v * v
        })
        .sum::<f64>()
        / n as f64;
    let rms_ref = rms_ref.sqrt();

    rms_err / (rms_ref + 1e-10)
}

// ── Bandwidth ────────────────────────────────────────────────────────────────
//
// Estimates the effective bandwidth from the DFT magnitude spectrum.
// Returns the fraction of the Nyquist frequency below which 99 % of the
// spectral energy resides.

/// Compute effective bandwidth as a fraction of Nyquist.
fn compute_bandwidth(signal: &[f32], sample_rate: u32) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }

    // Use first power-of-2 ≤ n, max 4096, for the DFT.
    let fft_n = fft_size(n.min(4096));

    // Compute real DFT magnitude via explicit trig (no external FFT crate needed).
    let magnitudes = naive_rdft_magnitudes(signal, fft_n);
    let n_bins = magnitudes.len(); // = fft_n/2 + 1

    let total_energy: f64 = magnitudes.iter().map(|m| m * m).sum();
    if total_energy < 1e-12 {
        return 0.0;
    }

    let target = 0.99 * total_energy;
    let mut cumulative = 0.0_f64;
    let mut rolloff_bin = n_bins - 1;
    for (i, &m) in magnitudes.iter().enumerate() {
        cumulative += m * m;
        if cumulative >= target {
            rolloff_bin = i;
            break;
        }
    }

    let nyquist = sample_rate as f64 / 2.0;
    let hz_per_bin = nyquist / (n_bins as f64 - 1.0).max(1.0);
    let rolloff_hz = rolloff_bin as f64 * hz_per_bin;

    (rolloff_hz / nyquist).clamp(0.0, 1.0)
}

/// Largest power of 2 ≤ n (minimum 1).
fn fft_size(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1_usize;
    while p * 2 <= n {
        p *= 2;
    }
    p
}

/// Compute DFT magnitude spectrum using O(N²) DFT for patent-free simplicity.
///
/// Returns `fft_n/2 + 1` magnitude values (one-sided real spectrum).
#[allow(clippy::cast_precision_loss)]
fn naive_rdft_magnitudes(signal: &[f32], fft_n: usize) -> Vec<f64> {
    let n_out = fft_n / 2 + 1;
    let n = signal.len().min(fft_n);
    let mut mags = vec![0.0_f64; n_out];

    for k in 0..n_out {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (t, &s) in signal.iter().enumerate().take(n) {
            let angle = 2.0 * PI as f64 * k as f64 * t as f64 / fft_n as f64;
            re += f64::from(s) * angle.cos();
            im -= f64::from(s) * angle.sin();
        }
        mags[k] = (re * re + im * im).sqrt();
    }
    mags
}

// ── Bark-scale critical band analysis ────────────────────────────────────────

/// Convert frequency (Hz) to Bark scale.
fn hz_to_bark(hz: f64) -> f64 {
    // Traunmüller (1990) formula
    let bark = (26.81 * hz / (1960.0 + hz)) - 0.53;
    bark.max(0.0)
}

/// Bark band edge frequencies (Hz) for bands 0..NUM_BARK_BANDS.
fn bark_band_edges() -> Vec<f64> {
    // Standard Bark band centre frequencies (Zwicker)
    let centres: [f64; NUM_BARK_BANDS] = [
        50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0, 1600.0,
        1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0, 5800.0, 7000.0, 8500.0, 10500.0,
        13500.0,
    ];
    let mut edges = Vec::with_capacity(NUM_BARK_BANDS + 1);
    edges.push(0.0);
    for i in 0..NUM_BARK_BANDS.saturating_sub(1) {
        edges.push((centres[i] + centres[i + 1]) / 2.0);
    }
    edges.push(22050.0); // cap at typical Nyquist
    edges
}

/// Accumulate DFT magnitude² into Bark critical bands.
fn bark_band_energies(magnitudes: &[f64], fft_n: usize, nyquist: f64) -> Vec<f64> {
    let n_bins = magnitudes.len();
    let edges = bark_band_edges();
    let hz_per_bin = if n_bins > 1 {
        nyquist / (n_bins as f64 - 1.0)
    } else {
        1.0
    };

    let mut energies = vec![0.0_f64; NUM_BARK_BANDS];
    let _ = fft_n; // used implicitly through magnitudes length

    for (i, &m) in magnitudes.iter().enumerate() {
        let freq = i as f64 * hz_per_bin;
        // Find which Bark band this bin belongs to
        let band = edges
            .windows(2)
            .position(|w| freq >= w[0] && freq < w[1])
            .unwrap_or(NUM_BARK_BANDS.saturating_sub(1));
        if band < NUM_BARK_BANDS {
            energies[band] += m * m;
        }
    }
    energies
}

// ── Frequency masking (spreading function) ───────────────────────────────────

/// Compute masking thresholds from Bark-band energies using a simplified
/// spreading function model.
///
/// The spreading function attenuates by ~27 dB/Bark upward and ~25 dB/Bark
/// downward, approximating the cochlear excitation pattern.
fn compute_masking_thresholds(ref_bark_energies: &[f64]) -> Vec<f64> {
    let n = ref_bark_energies.len().min(NUM_BARK_BANDS);
    let mut thresholds = vec![0.0_f64; n];

    for (j, thresh) in thresholds.iter_mut().enumerate() {
        let mut excitation = 0.0_f64;
        for (i, &e) in ref_bark_energies.iter().enumerate().take(n) {
            let dist = j as f64 - i as f64; // distance in Bark bands
                                            // Simplified spreading function
            let spread_db = if dist >= 0.0 {
                // Upward spread (high-freq masking by low-freq)
                -27.0 * dist
            } else {
                // Downward spread
                -25.0 * dist.abs()
            };
            let spread_linear = 10.0_f64.powf(spread_db / 10.0);
            excitation += e * spread_linear;
        }
        // Masking threshold = excitation * tone-to-noise offset (~-6 dB for noise masking)
        *thresh = excitation * 0.25; // -6 dB offset
    }
    thresholds
}

// ── Noise-to-mask ratio ──────────────────────────────────────────────────────

/// Per-band NMR in dB: positive means distortion exceeds mask (audible).
fn compute_nmr_per_band(ref_bark: &[f64], test_bark: &[f64], mask_thresholds: &[f64]) -> Vec<f64> {
    let n = ref_bark
        .len()
        .min(test_bark.len())
        .min(mask_thresholds.len());
    let mut nmr = Vec::with_capacity(n);

    for i in 0..n {
        let noise_energy = (ref_bark[i] - test_bark[i]).abs().max(1e-30);
        let mask = mask_thresholds[i].max(1e-30);
        let ratio_db = 10.0 * (noise_energy / mask).log10();
        nmr.push(ratio_db);
    }
    nmr
}

/// Total NMR: energy-weighted average across all bands.
fn compute_total_nmr(nmr_per_band: &[f64]) -> f64 {
    if nmr_per_band.is_empty() {
        return f64::NEG_INFINITY;
    }
    // Convert back to linear, average, convert to dB
    let sum_linear: f64 = nmr_per_band
        .iter()
        .map(|&db| 10.0_f64.powf(db / 10.0))
        .sum();
    let avg_linear = sum_linear / nmr_per_band.len() as f64;
    10.0 * avg_linear.max(1e-30).log10()
}

// ── Loudness-weighted disturbance ────────────────────────────────────────────

/// Compute loudness-weighted disturbance metric.
///
/// Weight the sample-level error by a simple loudness model based on
/// the reference signal's spectral energy (higher-energy bands contribute
/// more to perceived disturbance).
fn compute_loudness_disturbance(ref_s: &[f32], test_s: &[f32], ref_bark_energies: &[f64]) -> f64 {
    let n = ref_s.len().min(test_s.len());
    if n == 0 {
        return 0.0;
    }

    // Global loudness weight: sum of specific loudness (Bark energy ^ 0.23 Zwicker model)
    let total_loudness: f64 = ref_bark_energies
        .iter()
        .map(|&e| e.max(1e-20).powf(0.23))
        .sum();

    // Weighted RMS error
    let err_energy: f64 = ref_s
        .iter()
        .zip(test_s.iter())
        .map(|(&r, &t)| {
            let e = f64::from(r) - f64::from(t);
            e * e
        })
        .sum::<f64>()
        / n as f64;
    let rms_err = err_energy.sqrt();

    // Scale by loudness (louder signals → more noticeable distortion)
    let loudness_scale = (total_loudness / NUM_BARK_BANDS as f64).max(1e-10);
    rms_err * loudness_scale
}

// ── Bandwidth limitation detection ───────────────────────────────────────────

/// Detect bandwidth limitation by finding the 99% energy cutoff point.
///
/// Returns (is_limited, cutoff_fraction_of_nyquist).
/// A signal is considered bandwidth-limited if 99% of its energy
/// is concentrated below 80% of Nyquist.
fn detect_bandwidth_limitation(test_mags: &[f64], _fft_n: usize, nyquist: f64) -> (bool, f64) {
    let n_bins = test_mags.len();
    if n_bins < 4 {
        return (false, 0.0);
    }

    let hz_per_bin = nyquist / (n_bins as f64 - 1.0).max(1.0);

    // Compute total energy (skip DC)
    let total_energy: f64 = test_mags.iter().skip(1).map(|&m| m * m).sum();
    if total_energy < 1e-20 {
        return (false, 0.0);
    }

    // Find the bin where cumulative energy reaches 99%
    let target = 0.99 * total_energy;
    let mut cumulative = 0.0_f64;
    let mut cutoff_bin = n_bins - 1;

    for (i, &m) in test_mags.iter().enumerate().skip(1) {
        cumulative += m * m;
        if cumulative >= target {
            cutoff_bin = i;
            break;
        }
    }

    let cutoff_hz = cutoff_bin as f64 * hz_per_bin;
    let cutoff_frac = (cutoff_hz / nyquist).clamp(0.0, 1.0);

    // Consider "limited" if 99% energy cutoff is below 80% of Nyquist
    let is_limited = cutoff_frac < 0.80;

    (is_limited, cutoff_frac)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    // ── PeaqLikeConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = PeaqLikeConfig::default();
        assert_eq!(cfg.sample_rate, 44100);
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn test_config_new() {
        let cfg = PeaqLikeConfig::new(48000, 2);
        assert_eq!(cfg.sample_rate, 48000);
        assert_eq!(cfg.channels, 2);
    }

    // ── Identical signals ──────────────────────────────────────────────────────

    #[test]
    fn test_identical_signals_odg_zero() {
        let cfg = PeaqLikeConfig::default();
        let sig = sine_wave(1000.0, 44100, 0.1);
        let odg = PeaqScore::compute(&sig, &sig, &cfg);
        assert!(
            (odg - 0.0).abs() < 1e-9,
            "identical → ODG = 0, got {odg:.4}"
        );
    }

    #[test]
    fn test_identical_silence_odg_zero() {
        let cfg = PeaqLikeConfig::default();
        let silence = vec![0.0_f32; 4410];
        let odg = PeaqScore::compute(&silence, &silence, &cfg);
        assert!((odg - 0.0).abs() < 1e-9);
    }

    // ── Distorted signals ─────────────────────────────────────────────────────

    #[test]
    fn test_inverted_signal_very_annoying() {
        let cfg = PeaqLikeConfig::default();
        let ref_sig = sine_wave(440.0, 44100, 0.1);
        let test_sig: Vec<f32> = ref_sig.iter().map(|&v| -v).collect();
        let odg = PeaqScore::compute(&ref_sig, &test_sig, &cfg);
        assert!(
            odg < -2.0,
            "inverted signal should give ODG < -2, got {odg:.4}"
        );
    }

    #[test]
    fn test_noise_vs_sine_low_odg() {
        let cfg = PeaqLikeConfig::default();
        let ref_sig = sine_wave(1000.0, 44100, 0.1);
        // White noise approximation using sawtooth
        let test_sig: Vec<f32> = (0..ref_sig.len())
            .map(|i| ((i % 256) as f32 / 128.0) - 1.0)
            .collect();
        let odg = PeaqScore::compute(&ref_sig, &test_sig, &cfg);
        assert!(odg <= 0.0);
        assert!(odg >= -4.0);
    }

    // ── ODG range ─────────────────────────────────────────────────────────────

    #[test]
    fn test_odg_always_in_range() {
        let cfg = PeaqLikeConfig::default();
        let ref_sig = sine_wave(440.0, 44100, 0.05);
        let test_sig: Vec<f32> = (0..ref_sig.len()).map(|i| (i % 3) as f32).collect();
        let odg = PeaqScore::compute(&ref_sig, &test_sig, &cfg);
        assert!(odg >= -4.0 && odg <= 0.0, "ODG out of range: {odg:.4}");
    }

    // ── Empty input ───────────────────────────────────────────────────────────

    #[test]
    fn test_empty_ref_returns_zero() {
        let cfg = PeaqLikeConfig::default();
        let odg = PeaqScore::compute(&[], &[0.5], &cfg);
        assert_eq!(odg, 0.0);
    }

    #[test]
    fn test_empty_test_returns_zero() {
        let cfg = PeaqLikeConfig::default();
        let odg = PeaqScore::compute(&[0.5], &[], &cfg);
        assert_eq!(odg, 0.0);
    }

    // ── Bandwidth computation ─────────────────────────────────────────────────

    #[test]
    fn test_bandwidth_sine_reasonable() {
        let sr = 44100_u32;
        let sig = sine_wave(4000.0, sr, 0.05);
        let bw = compute_bandwidth(&sig, sr);
        // 4 kHz sine should have bandwidth well below Nyquist (22.05 kHz)
        assert!(
            bw > 0.0 && bw < 1.0,
            "bandwidth should be in (0,1), got {bw:.4}"
        );
    }

    #[test]
    fn test_bandwidth_silence_is_zero() {
        let silence = vec![0.0_f32; 1024];
        let bw = compute_bandwidth(&silence, 44100);
        assert_eq!(bw, 0.0);
    }

    #[test]
    fn test_bandwidth_in_range() {
        let sr = 48000_u32;
        let sig: Vec<f32> = (0..2048).map(|i| ((i % 64) as f32 / 32.0) - 1.0).collect();
        let bw = compute_bandwidth(&sig, sr);
        assert!((0.0..=1.0).contains(&bw));
    }

    // ── Noise loudness ────────────────────────────────────────────────────────

    #[test]
    fn test_noise_loudness_identical_is_zero() {
        let sig: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        let nl = compute_noise_loudness(&sig, &sig);
        assert!(nl.abs() < 1e-9);
    }

    #[test]
    fn test_noise_loudness_inverted_large() {
        let sig: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        let inv: Vec<f32> = sig.iter().map(|&v| -v).collect();
        let nl = compute_noise_loudness(&sig, &inv);
        assert!(
            nl > 0.5,
            "inverted signal → high noise loudness, got {nl:.4}"
        );
    }

    // ── Stereo config ─────────────────────────────────────────────────────────

    #[test]
    fn test_stereo_config_identical() {
        let cfg = PeaqLikeConfig::new(48000, 2);
        let sig = sine_wave(1000.0, 48000, 0.1);
        let odg = PeaqScore::compute(&sig, &sig, &cfg);
        assert!((odg - 0.0).abs() < 1e-9);
    }

    // ── compute_movs ─────────────────────────────────────────────────────────

    #[test]
    fn test_compute_movs_identical() {
        let cfg = PeaqLikeConfig::default();
        let sig = sine_wave(440.0, 44100, 0.05);
        let movs = PeaqScore::compute_movs(&sig, &sig, &cfg);
        assert!(movs.noise_loudness.abs() < 1e-9);
        assert!((movs.bandwidth_ref - movs.bandwidth_test).abs() < 1e-9);
    }

    #[test]
    fn test_movs_to_odg_range() {
        let movs = MovValues {
            noise_loudness: 2.0,
            bandwidth_ref: 0.8,
            bandwidth_test: 0.1,
        };
        let odg = PeaqScore::movs_to_odg(&movs);
        assert!(odg >= -4.0 && odg <= 0.0);
    }

    // ── ODG-to-MOV conversion ────────────────────────────────────────────────

    #[test]
    fn test_odg_to_mov_imperceptible() {
        let mov = PeaqScore::odg_to_mov(0.0);
        assert!((mov - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_odg_to_mov_very_annoying() {
        let mov = PeaqScore::odg_to_mov(-4.0);
        assert!((mov - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_odg_to_mov_mid() {
        let mov = PeaqScore::odg_to_mov(-2.0);
        assert!((mov - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_odg_to_mov_clamps() {
        assert!((PeaqScore::odg_to_mov(-10.0) - 1.0).abs() < 1e-9);
        assert!((PeaqScore::odg_to_mov(5.0) - 5.0).abs() < 1e-9);
    }

    // ── Extended analysis ────────────────────────────────────────────────────

    #[test]
    fn test_extended_identical_signals() {
        let cfg = PeaqLikeConfig::default();
        let sig = sine_wave(1000.0, 44100, 0.1);
        let result = PeaqScore::compute_extended(&sig, &sig, &cfg);
        assert!((result.odg - 0.0).abs() < 1e-9);
        assert!((result.mov - 5.0).abs() < 1e-9);
        assert!((result.loudness_disturbance - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_extended_empty_signals() {
        let cfg = PeaqLikeConfig::default();
        let result = PeaqScore::compute_extended(&[], &[], &cfg);
        assert!((result.odg - 0.0).abs() < 1e-9);
        assert!((result.mov - 5.0).abs() < 1e-9);
        assert_eq!(result.nmr_per_band.len(), NUM_BARK_BANDS);
    }

    #[test]
    fn test_extended_distorted_signal() {
        let cfg = PeaqLikeConfig::default();
        let ref_sig = sine_wave(440.0, 44100, 0.1);
        let test_sig: Vec<f32> = ref_sig.iter().map(|&v| -v).collect();
        let result = PeaqScore::compute_extended(&ref_sig, &test_sig, &cfg);
        assert!(result.odg < -1.0);
        assert!(result.mov < 4.0);
        assert!(result.loudness_disturbance > 0.0);
    }

    // ── Bark-scale helpers ───────────────────────────────────────────────────

    #[test]
    fn test_hz_to_bark_zero() {
        let bark = hz_to_bark(0.0);
        assert!(bark >= 0.0, "Bark should be non-negative");
    }

    #[test]
    fn test_hz_to_bark_1000hz() {
        let bark = hz_to_bark(1000.0);
        // ~8.5 Bark at 1000 Hz
        assert!(
            bark > 7.0 && bark < 10.0,
            "1000 Hz ≈ 8.5 Bark, got {bark:.2}"
        );
    }

    #[test]
    fn test_hz_to_bark_monotonic() {
        let b1 = hz_to_bark(500.0);
        let b2 = hz_to_bark(2000.0);
        let b3 = hz_to_bark(8000.0);
        assert!(b1 < b2 && b2 < b3, "Bark should increase with Hz");
    }

    // ── Masking thresholds ───────────────────────────────────────────────────

    #[test]
    fn test_masking_thresholds_non_negative() {
        let energies = vec![1.0_f64; NUM_BARK_BANDS];
        let thresholds = compute_masking_thresholds(&energies);
        assert_eq!(thresholds.len(), NUM_BARK_BANDS);
        for (i, &t) in thresholds.iter().enumerate() {
            assert!(t >= 0.0, "threshold at band {i} should be ≥ 0, got {t}");
        }
    }

    #[test]
    fn test_masking_thresholds_silence() {
        let energies = vec![0.0_f64; NUM_BARK_BANDS];
        let thresholds = compute_masking_thresholds(&energies);
        for &t in &thresholds {
            assert!((t - 0.0).abs() < 1e-20);
        }
    }

    // ── Noise-to-mask ratio ──────────────────────────────────────────────────

    #[test]
    fn test_nmr_identical_spectra() {
        let energies = vec![10.0_f64; NUM_BARK_BANDS];
        let thresholds = compute_masking_thresholds(&energies);
        let nmr = compute_nmr_per_band(&energies, &energies, &thresholds);
        assert_eq!(nmr.len(), NUM_BARK_BANDS);
        // Identical → noise energy ≈ 0 → very negative NMR
        for (i, &n) in nmr.iter().enumerate() {
            assert!(
                n < 0.0,
                "band {i} NMR should be very negative for identical, got {n:.2}"
            );
        }
    }

    #[test]
    fn test_nmr_total_range() {
        let ref_e = vec![100.0_f64; NUM_BARK_BANDS];
        let test_e = vec![50.0_f64; NUM_BARK_BANDS];
        let thresholds = compute_masking_thresholds(&ref_e);
        let nmr = compute_nmr_per_band(&ref_e, &test_e, &thresholds);
        let total = compute_total_nmr(&nmr);
        // total should be a finite number
        assert!(total.is_finite(), "total NMR should be finite, got {total}");
    }

    #[test]
    fn test_nmr_empty() {
        let total = compute_total_nmr(&[]);
        assert!(total == f64::NEG_INFINITY);
    }

    // ── Loudness-weighted disturbance ────────────────────────────────────────

    #[test]
    fn test_loudness_disturbance_identical_zero() {
        let sig = sine_wave(440.0, 44100, 0.05);
        let bark = vec![1.0; NUM_BARK_BANDS];
        let ld = compute_loudness_disturbance(&sig, &sig, &bark);
        assert!(
            ld.abs() < 1e-9,
            "identical should have 0 disturbance, got {ld}"
        );
    }

    #[test]
    fn test_loudness_disturbance_increases_with_error() {
        let ref_sig = sine_wave(440.0, 44100, 0.05);
        let bark = vec![1.0; NUM_BARK_BANDS];

        let small_err: Vec<f32> = ref_sig.iter().map(|&v| v + 0.01).collect();
        let large_err: Vec<f32> = ref_sig.iter().map(|&v| v + 0.5).collect();

        let ld_small = compute_loudness_disturbance(&ref_sig, &small_err, &bark);
        let ld_large = compute_loudness_disturbance(&ref_sig, &large_err, &bark);

        assert!(
            ld_large > ld_small,
            "larger error should give larger disturbance: {ld_small:.6} vs {ld_large:.6}"
        );
    }

    // ── Bandwidth limitation detection ───────────────────────────────────────

    #[test]
    fn test_bandwidth_not_limited_full_spectrum() {
        // Wideband signal: energy spread across high freqs via white-noise-like pattern
        let sr = 44100_u32;
        let sig: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / sr as f32;
                // Many harmonics spread energy to high frequencies
                (2.0 * PI * 440.0 * t).sin()
                    + (2.0 * PI * 5000.0 * t).sin()
                    + (2.0 * PI * 10000.0 * t).sin()
                    + (2.0 * PI * 18000.0 * t).sin()
                    + (2.0 * PI * 20000.0 * t).sin()
            })
            .collect();
        let fft_n = fft_size(sig.len());
        let mags = naive_rdft_magnitudes(&sig, fft_n);
        let nyquist = sr as f64 / 2.0;
        let (limited, cutoff) = detect_bandwidth_limitation(&mags, fft_n, nyquist);
        assert!(
            !limited,
            "full-spectrum signal should not be limited, cutoff={cutoff:.3}"
        );
    }

    #[test]
    fn test_bandwidth_limited_lowpass() {
        // Very low frequency signal only → all energy below Nyquist * 0.8
        let sr = 44100_u32;
        let sig = sine_wave(200.0, sr, 0.1);
        let fft_n = fft_size(sig.len().min(4096));
        let mags = naive_rdft_magnitudes(&sig, fft_n);
        let nyquist = sr as f64 / 2.0;
        let (limited, cutoff) = detect_bandwidth_limitation(&mags, fft_n, nyquist);
        assert!(
            limited,
            "low-freq-only signal should be bandwidth limited, cutoff={cutoff:.3}"
        );
        assert!(cutoff < 0.5, "cutoff should be low, got {cutoff:.3}");
    }

    #[test]
    fn test_bandwidth_detection_silence() {
        let mags = vec![0.0_f64; 100];
        let (limited, _) = detect_bandwidth_limitation(&mags, 198, 22050.0);
        assert!(!limited);
    }

    #[test]
    fn test_bark_band_energies_count() {
        let mags = vec![1.0_f64; 2049]; // typical for 4096-pt FFT
        let energies = bark_band_energies(&mags, 4096, 22050.0);
        assert_eq!(energies.len(), NUM_BARK_BANDS);
    }
}
