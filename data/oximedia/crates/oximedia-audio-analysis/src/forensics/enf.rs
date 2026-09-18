//! Electrical Network Frequency (ENF) analysis for audio forensics.
//!
//! ENF analysis is used in audio forensics to authenticate recordings by
//! extracting the instantaneous frequency of the mains hum (50 Hz in Europe /
//! Asia, 60 Hz in North America) and comparing it against a reference ENF
//! database. Even without a reference, the extracted ENF track reveals
//! discontinuities that indicate splicing or post-production edits.
//!
//! # Algorithm
//! 1. Divide the signal into overlapping short-time frames.
//! 2. Apply a Hann window and compute the FFT of each frame.
//! 3. Identify the fundamental (50 or 60 Hz) and its 2nd harmonic to improve
//!    estimation robustness via parabolic interpolation around the peak bin.
//! 4. Collect per-frame instantaneous frequency estimates into an ENF track.
//! 5. Compute statistics: mean, standard deviation, drift, and anomaly flags.

use crate::AnalysisError;

/// Nominal mains frequency selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainsFrequency {
    /// 50 Hz mains (Europe, Asia, Africa, most of the world).
    Hz50,
    /// 60 Hz mains (North America, parts of South America, Japan).
    Hz60,
}

impl MainsFrequency {
    /// Nominal fundamental frequency in Hz.
    #[must_use]
    pub fn nominal(self) -> f64 {
        match self {
            Self::Hz50 => 50.0,
            Self::Hz60 => 60.0,
        }
    }

    /// Maximum plausible deviation from nominal (Hz).
    ///
    /// Grids typically regulate to ±0.2 Hz in normal operation; we allow a
    /// wider ±1 Hz window to handle unusual conditions.
    #[must_use]
    pub fn search_radius(self) -> f64 {
        1.0
    }
}

/// Per-frame ENF estimate.
#[derive(Debug, Clone)]
pub struct EnfFrameEstimate {
    /// Frame index (0-based).
    pub frame_index: usize,
    /// Time offset of the frame centre in seconds.
    pub time_seconds: f64,
    /// Estimated instantaneous ENF in Hz (fundamental), or `None` if the
    /// frame had insufficient energy in the ENF band.
    pub frequency_hz: Option<f64>,
    /// Magnitude of the ENF peak (linear, not dB).
    pub peak_magnitude: f32,
    /// Signal-to-noise ratio of the ENF peak relative to neighbouring bins.
    pub snr_db: f32,
}

/// Aggregate result of ENF analysis.
#[derive(Debug, Clone)]
pub struct EnfResult {
    /// Per-frame ENF estimates.
    pub frames: Vec<EnfFrameEstimate>,
    /// Nominal mains frequency used for analysis.
    pub mains_frequency: MainsFrequency,
    /// Mean ENF frequency across valid frames (Hz).
    pub mean_hz: f64,
    /// Standard deviation of ENF across valid frames (Hz).
    pub std_hz: f64,
    /// Maximum frequency deviation from mean (Hz).
    pub max_deviation_hz: f64,
    /// Linear frequency drift estimate (Hz/minute).
    pub drift_hz_per_minute: f64,
    /// Fraction of frames where ENF was detectable.
    pub coverage: f32,
    /// Indices of frames whose ENF deviated by more than 3σ from the mean
    /// — potential authenticity anomalies.
    pub anomaly_frame_indices: Vec<usize>,
}

/// Configuration for [`EnfAnalyzer`].
#[derive(Debug, Clone)]
pub struct EnfConfig {
    /// Which mains standard to target.
    pub mains: MainsFrequency,
    /// FFT window length in samples.
    pub frame_size: usize,
    /// Hop between frames in samples.
    pub hop_size: usize,
    /// Minimum SNR (dB) for a frame to be considered valid.
    pub min_snr_db: f32,
    /// Whether to use the 2nd harmonic to improve estimation accuracy.
    pub use_harmonic: bool,
    /// Whether to perform weighted interpolation (parabolic peak fitting).
    pub parabolic_interp: bool,
}

impl Default for EnfConfig {
    fn default() -> Self {
        Self {
            mains: MainsFrequency::Hz50,
            frame_size: 8192,
            hop_size: 4096,
            min_snr_db: 6.0,
            use_harmonic: true,
            parabolic_interp: true,
        }
    }
}

/// ENF analyzer for forensic recording authentication.
pub struct EnfAnalyzer {
    config: EnfConfig,
}

impl EnfAnalyzer {
    /// Create a new ENF analyzer with the given configuration.
    #[must_use]
    pub fn new(config: EnfConfig) -> Self {
        Self { config }
    }

    /// Analyse the ENF content of the provided mono audio samples.
    ///
    /// # Arguments
    /// * `samples` - Mono audio samples (f32).
    /// * `sample_rate` - Sample rate in Hz.
    ///
    /// # Returns
    /// [`EnfResult`] containing per-frame estimates and aggregate statistics.
    ///
    /// # Errors
    /// Returns [`AnalysisError`] if the sample rate is too low for ENF
    /// analysis or the samples are insufficient.
    pub fn analyze(&self, samples: &[f32], sample_rate: u32) -> crate::Result<EnfResult> {
        let nominal = self.config.mains.nominal();

        // Sample rate must be at least 4× the 2nd harmonic of the mains freq
        // to avoid aliasing.  In practice any rate ≥ 400 Hz is fine for 50/60
        // Hz work; real recordings are typically 44.1 kHz or 48 kHz.
        if f64::from(sample_rate) < nominal * 4.0 {
            return Err(AnalysisError::InvalidSampleRate(sample_rate as f32));
        }

        if samples.len() < self.config.frame_size {
            return Err(AnalysisError::InsufficientSamples {
                needed: self.config.frame_size,
                got: samples.len(),
            });
        }

        let sr = f64::from(sample_rate);
        let frame_size = self.config.frame_size;
        let hop_size = self.config.hop_size;

        // Pre-compute Hann window
        let window: Vec<f64> = (0..frame_size)
            .map(|i| {
                let x = std::f64::consts::PI * i as f64 / (frame_size - 1) as f64;
                0.5 * (1.0 - x.cos())
            })
            .collect();

        let num_frames = (samples.len() - frame_size) / hop_size + 1;
        let mut frame_estimates: Vec<EnfFrameEstimate> = Vec::with_capacity(num_frames);

        for fi in 0..num_frames {
            let start = fi * hop_size;
            let end = start + frame_size;
            if end > samples.len() {
                break;
            }

            let time_seconds = (start + frame_size / 2) as f64 / sr;
            let frame = &samples[start..end];

            // Apply Hann window
            let windowed: Vec<oxifft::Complex<f64>> = frame
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| oxifft::Complex::new(f64::from(s) * w, 0.0))
                .collect();

            // FFT
            let spectrum = oxifft::fft(&windowed);
            let n_bins = frame_size / 2 + 1;
            let magnitude: Vec<f32> = spectrum[..n_bins]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im).sqrt() as f32)
                .collect();

            // Estimate ENF frequency from fundamental (and optionally harmonic)
            let estimate = estimate_enf_from_spectrum(
                &magnitude,
                sr,
                frame_size,
                self.config.mains,
                self.config.parabolic_interp,
                self.config.use_harmonic,
                self.config.min_snr_db,
                fi,
                time_seconds,
            );

            frame_estimates.push(estimate);
        }

        // Aggregate statistics
        let result = compute_aggregate(&frame_estimates, self.config.mains);
        Ok(result)
    }
}

/// Convenience function matching the task specification signature.
///
/// Uses default configuration (50 Hz mains, 8192-sample window).
///
/// # Errors
/// See [`EnfAnalyzer::analyze`].
pub fn analyze_enf(samples: &[f32], sample_rate: u32) -> crate::Result<EnfResult> {
    EnfAnalyzer::new(EnfConfig::default()).analyze(samples, sample_rate)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Estimate ENF for a single frame given its magnitude spectrum.
#[allow(clippy::too_many_arguments)]
fn estimate_enf_from_spectrum(
    magnitude: &[f32],
    sample_rate: f64,
    frame_size: usize,
    mains: MainsFrequency,
    parabolic: bool,
    use_harmonic: bool,
    min_snr_db: f32,
    frame_index: usize,
    time_seconds: f64,
) -> EnfFrameEstimate {
    let nominal = mains.nominal();
    let radius = mains.search_radius();
    let bin_hz = sample_rate / frame_size as f64;

    // Search window around fundamental
    let lo_bin = ((nominal - radius) / bin_hz).floor() as usize;
    let hi_bin = ((nominal + radius) / bin_hz).ceil() as usize;
    let hi_bin = hi_bin.min(magnitude.len().saturating_sub(1));

    let (fund_bin, fund_mag) = peak_bin(magnitude, lo_bin, hi_bin);

    // Reject frames with negligible energy in the ENF band
    let minimum_magnitude: f32 = 1e-6;
    if fund_mag < minimum_magnitude {
        return EnfFrameEstimate {
            frame_index,
            time_seconds,
            frequency_hz: None,
            peak_magnitude: fund_mag,
            snr_db: 0.0,
        };
    }

    // Compute SNR: peak vs. average of neighbouring bins (excluding peak ±2)
    let snr_db = compute_snr_db(magnitude, fund_bin, lo_bin, hi_bin);

    if snr_db < min_snr_db {
        return EnfFrameEstimate {
            frame_index,
            time_seconds,
            frequency_hz: None,
            peak_magnitude: fund_mag,
            snr_db,
        };
    }

    // Parabolic interpolation to refine sub-bin frequency estimate
    let mut freq_est = if parabolic {
        parabolic_interpolate(magnitude, fund_bin, bin_hz)
    } else {
        fund_bin as f64 * bin_hz
    };

    // Optionally blend with 2nd-harmonic estimate (reduces variance ~√2)
    if use_harmonic {
        let nominal_h2 = nominal * 2.0;
        let lo_h2 = ((nominal_h2 - radius * 2.0) / bin_hz).floor() as usize;
        let hi_h2 = ((nominal_h2 + radius * 2.0) / bin_hz).ceil() as usize;
        let hi_h2 = hi_h2.min(magnitude.len().saturating_sub(1));

        if hi_h2 > lo_h2 {
            let (h2_bin, h2_mag) = peak_bin(magnitude, lo_h2, hi_h2);
            if h2_bin > 0 && h2_mag > fund_mag * 0.1 {
                let h2_est = if parabolic {
                    parabolic_interpolate(magnitude, h2_bin, bin_hz) / 2.0
                } else {
                    h2_bin as f64 * bin_hz / 2.0
                };
                // Weighted average: fundamental weight=1, harmonic weight=0.5
                freq_est = (freq_est + 0.5 * h2_est) / 1.5;
            }
        }
    }

    EnfFrameEstimate {
        frame_index,
        time_seconds,
        frequency_hz: Some(freq_est),
        peak_magnitude: fund_mag,
        snr_db,
    }
}

/// Find the bin with the highest magnitude within `[lo, hi]`.
///
/// Returns `(bin_index, magnitude)`.  Returns `(0, 0.0)` if the range is empty.
fn peak_bin(magnitude: &[f32], lo: usize, hi: usize) -> (usize, f32) {
    if lo >= magnitude.len() || hi < lo {
        return (0, 0.0);
    }
    let hi_clamped = hi.min(magnitude.len() - 1);
    let (idx, &mag) = magnitude[lo..=hi_clamped]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    (lo + idx, mag)
}

/// Parabolic interpolation around a spectral peak for sub-bin accuracy.
///
/// Given the peak at `bin`, uses bins `bin-1` and `bin+1` to fit a parabola
/// and return the refined frequency estimate in Hz.
fn parabolic_interpolate(magnitude: &[f32], bin: usize, bin_hz: f64) -> f64 {
    if bin == 0 || bin + 1 >= magnitude.len() {
        return bin as f64 * bin_hz;
    }
    let alpha = f64::from(magnitude[bin - 1]);
    let beta = f64::from(magnitude[bin]);
    let gamma = f64::from(magnitude[bin + 1]);
    let denom = alpha - 2.0 * beta + gamma;
    if denom.abs() < 1e-30 {
        return bin as f64 * bin_hz;
    }
    let p = 0.5 * (alpha - gamma) / denom;
    (bin as f64 + p) * bin_hz
}

/// Compute SNR (dB) of a spectral peak relative to the noise floor.
///
/// Noise is estimated from bins in a wider neighbourhood outside `[lo, hi]`
/// (avoiding the ENF search band to prevent contaminating the noise estimate).
/// If the neighbourhood is empty, a local guard of ±1 bin is tried instead.
fn compute_snr_db(magnitude: &[f32], peak_bin: usize, lo: usize, hi: usize) -> f32 {
    let peak_mag = magnitude[peak_bin.min(magnitude.len() - 1)];
    if peak_mag < 1e-30 {
        return 0.0;
    }

    // Primary noise estimate: bins adjacent to the search band, width = band width
    let band_width = hi.saturating_sub(lo) + 1;
    let noise_lo_outer = lo.saturating_sub(band_width);
    let noise_hi_outer = (hi + band_width).min(magnitude.len().saturating_sub(1));

    // Collect bins outside [lo, hi]
    let noise_bins_outer: Vec<f32> = (noise_lo_outer..=noise_hi_outer)
        .filter(|&k| k < lo || k > hi)
        .map(|k| magnitude[k])
        .collect();

    let noise_mean = if !noise_bins_outer.is_empty() {
        noise_bins_outer.iter().sum::<f32>() / noise_bins_outer.len() as f32
    } else {
        // Fallback: immediate neighbours within search band (guard = 1 bin)
        let guard = 1_usize;
        let local: Vec<f32> = (lo..=hi.min(magnitude.len() - 1))
            .filter(|&k| {
                (k + guard < peak_bin) || (peak_bin <= k && k > peak_bin + guard) || (k != peak_bin)
            })
            .filter(|&k| k != peak_bin)
            .map(|k| magnitude[k])
            .collect();
        if local.is_empty() {
            // Only one bin in search range — no noise estimate possible.
            // Treat high SNR so the frame is kept (magnitude was already checked above).
            return 40.0;
        }
        local.iter().sum::<f32>() / local.len() as f32
    };

    if noise_mean <= 1e-30 {
        // No noise floor → very high SNR
        return 60.0;
    }

    20.0 * (f64::from(peak_mag) / f64::from(noise_mean)).log10() as f32
}

/// Compute aggregate statistics from per-frame estimates.
fn compute_aggregate(frames: &[EnfFrameEstimate], mains: MainsFrequency) -> EnfResult {
    let valid: Vec<(usize, f64)> = frames
        .iter()
        .filter_map(|f| f.frequency_hz.map(|hz| (f.frame_index, hz)))
        .collect();

    if valid.is_empty() {
        return EnfResult {
            frames: frames.to_vec(),
            mains_frequency: mains,
            mean_hz: mains.nominal(),
            std_hz: 0.0,
            max_deviation_hz: 0.0,
            drift_hz_per_minute: 0.0,
            coverage: 0.0,
            anomaly_frame_indices: Vec::new(),
        };
    }

    let n = valid.len() as f64;
    let mean_hz: f64 = valid.iter().map(|(_, hz)| hz).sum::<f64>() / n;
    let variance: f64 = valid
        .iter()
        .map(|(_, hz)| (hz - mean_hz).powi(2))
        .sum::<f64>()
        / n;
    let std_hz = variance.sqrt();

    let max_deviation_hz = valid
        .iter()
        .map(|(_, hz)| (hz - mean_hz).abs())
        .fold(0.0_f64, f64::max);

    // Anomalies: frames with |f - mean| > 3σ
    let threshold = 3.0 * std_hz;
    let anomaly_frame_indices: Vec<usize> = valid
        .iter()
        .filter(|(_, hz)| (hz - mean_hz).abs() > threshold && std_hz > 0.0)
        .map(|(idx, _)| *idx)
        .collect();

    // Linear drift: least-squares slope over frame index (as proxy for time)
    let drift_hz_per_minute = compute_drift(&valid, frames);

    let coverage = valid.len() as f32 / frames.len().max(1) as f32;

    EnfResult {
        frames: frames.to_vec(),
        mains_frequency: mains,
        mean_hz,
        std_hz,
        max_deviation_hz,
        drift_hz_per_minute,
        coverage,
        anomaly_frame_indices,
    }
}

/// Estimate linear drift in Hz/minute via simple linear regression.
fn compute_drift(valid: &[(usize, f64)], all_frames: &[EnfFrameEstimate]) -> f64 {
    if valid.len() < 2 {
        return 0.0;
    }

    // Convert frame indices to time in minutes
    let time_per_frame_minutes = all_frames
        .get(1)
        .and_then(|f| {
            all_frames.first().map(|f0| {
                let dt = f.time_seconds - f0.time_seconds;
                if dt > 0.0 {
                    dt / 60.0
                } else {
                    1.0 / 60.0
                }
            })
        })
        .unwrap_or(1.0 / 60.0);

    let n = valid.len() as f64;
    let times: Vec<f64> = valid
        .iter()
        .map(|(idx, _)| *idx as f64 * time_per_frame_minutes)
        .collect();
    let freqs: Vec<f64> = valid.iter().map(|(_, hz)| *hz).collect();

    let mean_t = times.iter().sum::<f64>() / n;
    let mean_f = freqs.iter().sum::<f64>() / n;

    let num: f64 = times
        .iter()
        .zip(freqs.iter())
        .map(|(t, f)| (t - mean_t) * (f - mean_f))
        .sum();
    let den: f64 = times.iter().map(|t| (t - mean_t).powi(2)).sum();

    if den.abs() < 1e-30 {
        0.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_enf_signal(
        nominal_hz: f64,
        deviation: f64,
        sample_rate: u32,
        duration_secs: f64,
        amplitude: f32,
    ) -> Vec<f32> {
        let n = (f64::from(sample_rate) * duration_secs) as usize;
        let sr = f64::from(sample_rate);
        let mut phase = 0.0_f64;
        (0..n)
            .map(|i| {
                // Slowly drift the frequency to simulate real grid behaviour
                let t = i as f64 / sr;
                let freq = nominal_hz + deviation * (2.0 * std::f64::consts::PI * 0.01 * t).sin();
                phase += 2.0 * std::f64::consts::PI * freq / sr;
                (phase.sin() as f32) * amplitude
            })
            .collect()
    }

    fn make_analyzer(mains: MainsFrequency) -> EnfAnalyzer {
        EnfAnalyzer::new(EnfConfig {
            mains,
            frame_size: 4096,
            hop_size: 2048,
            min_snr_db: 3.0,
            use_harmonic: false,
            parabolic_interp: true,
        })
    }

    #[test]
    fn test_analyze_50hz_signal() {
        let samples = generate_enf_signal(50.0, 0.05, 8000, 5.0, 0.1);
        let analyzer = make_analyzer(MainsFrequency::Hz50);
        let result = analyzer.analyze(&samples, 8000).expect("should succeed");

        assert!(
            result.coverage > 0.5,
            "coverage too low: {}",
            result.coverage
        );
        assert!(
            (result.mean_hz - 50.0).abs() < 0.5,
            "mean ENF {:.4} Hz not near 50 Hz",
            result.mean_hz
        );
    }

    #[test]
    fn test_analyze_60hz_signal() {
        let samples = generate_enf_signal(60.0, 0.05, 8000, 5.0, 0.1);
        let analyzer = make_analyzer(MainsFrequency::Hz60);
        let result = analyzer.analyze(&samples, 8000).expect("should succeed");

        assert!(
            (result.mean_hz - 60.0).abs() < 0.5,
            "mean ENF {:.4} Hz not near 60 Hz",
            result.mean_hz
        );
    }

    #[test]
    fn test_insufficient_samples_error() {
        let short = vec![0.0_f32; 100];
        let analyzer = make_analyzer(MainsFrequency::Hz50);
        assert!(analyzer.analyze(&short, 8000).is_err());
    }

    #[test]
    fn test_sample_rate_too_low_error() {
        let samples = vec![0.0_f32; 8192];
        let analyzer = make_analyzer(MainsFrequency::Hz60);
        // 100 Hz sample rate is way too low for 60 Hz analysis (< 4× Nyquist)
        assert!(analyzer.analyze(&samples, 100).is_err());
    }

    #[test]
    fn test_coverage_pure_signal() {
        let samples = generate_enf_signal(50.0, 0.0, 8000, 10.0, 0.2);
        let analyzer = make_analyzer(MainsFrequency::Hz50);
        let result = analyzer.analyze(&samples, 8000).expect("ok");
        assert!(
            result.coverage > 0.5,
            "pure ENF signal should have high coverage: {}",
            result.coverage
        );
    }

    #[test]
    fn test_silence_low_coverage() {
        // Silence → no detectable ENF → low coverage
        let silence = vec![0.0_f32; 8192 * 4];
        let analyzer = EnfAnalyzer::new(EnfConfig {
            min_snr_db: 3.0,
            frame_size: 8192,
            hop_size: 4096,
            use_harmonic: false,
            parabolic_interp: false,
            mains: MainsFrequency::Hz50,
        });
        let result = analyzer.analyze(&silence, 8000).expect("ok");
        assert_eq!(
            result.coverage, 0.0,
            "silence should yield zero coverage, got {}",
            result.coverage
        );
    }

    #[test]
    fn test_enf_result_fields_valid() {
        let samples = generate_enf_signal(50.0, 0.02, 8000, 3.0, 0.1);
        let analyzer = make_analyzer(MainsFrequency::Hz50);
        let result = analyzer.analyze(&samples, 8000).expect("ok");

        assert!(result.std_hz >= 0.0, "std_hz should be non-negative");
        assert!(
            result.max_deviation_hz >= 0.0,
            "max_deviation_hz should be non-negative"
        );
        assert!(
            result.coverage >= 0.0 && result.coverage <= 1.0,
            "coverage out of [0,1]: {}",
            result.coverage
        );
    }

    #[test]
    fn test_parabolic_interpolate_on_peak() {
        // Construct a spectrum with a clean quadratic peak at bin 10
        let mut mag = vec![0.0_f32; 32];
        mag[9] = 0.5;
        mag[10] = 1.0;
        mag[11] = 0.5;
        let bin_hz = 10.0; // 10 Hz per bin (arbitrary)
        let freq = parabolic_interpolate(&mag, 10, bin_hz);
        // Symmetric peak → exactly at bin 10
        assert!(
            (freq - 100.0).abs() < 1e-6,
            "symmetric peak should give bin×hz=100, got {freq}"
        );
    }

    #[test]
    fn test_parabolic_interpolate_asymmetric() {
        let mut mag = vec![0.0_f32; 32];
        mag[9] = 0.3;
        mag[10] = 1.0;
        mag[11] = 0.6;
        let bin_hz = 1.0;
        let freq = parabolic_interpolate(&mag, 10, bin_hz);
        // Peak is shifted slightly toward bin 11
        assert!(
            freq > 10.0,
            "asymmetric peak should shift toward higher bin, got {freq}"
        );
    }

    #[test]
    fn test_mains_frequency_nominal() {
        assert_eq!(MainsFrequency::Hz50.nominal(), 50.0);
        assert_eq!(MainsFrequency::Hz60.nominal(), 60.0);
    }

    #[test]
    fn test_analyze_enf_convenience_fn() {
        let samples = generate_enf_signal(50.0, 0.05, 8000, 5.0, 0.1);
        let result = analyze_enf(&samples, 8000).expect("convenience fn should work");
        assert_eq!(result.mains_frequency, MainsFrequency::Hz50);
    }

    #[test]
    fn test_drift_near_zero_for_stable_signal() {
        // A perfectly stable 50 Hz signal should have near-zero drift
        let samples = generate_enf_signal(50.0, 0.0, 8000, 10.0, 0.15);
        let analyzer = make_analyzer(MainsFrequency::Hz50);
        let result = analyzer.analyze(&samples, 8000).expect("ok");
        // Drift should be tiny (±0.01 Hz/min) for a stable synthetic signal
        assert!(
            result.drift_hz_per_minute.abs() < 0.5,
            "drift should be near 0 for stable signal: {}",
            result.drift_hz_per_minute
        );
    }
}
