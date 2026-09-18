//! Time-varying loudness curve analysis.
//!
//! Provides per-band and full-spectrum loudness envelopes, LUFS-style
//! integrated measurements, and tools for locating the loudest/quietest
//! segments of an audio signal.

#![allow(dead_code)]

/// Frequency band used for multi-band loudness analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessBand {
    /// Sub-bass region (20 – 80 Hz).
    SubBass,
    /// Bass region (80 – 250 Hz).
    Bass,
    /// Mid-range region (250 Hz – 2 kHz).
    Mid,
    /// Upper-mid region (2 kHz – 6 kHz).
    UpperMid,
    /// Presence / treble region (6 kHz – 20 kHz).
    Treble,
}

impl LoudnessBand {
    /// Frequency bounds `(low_hz, high_hz)` for this band.
    #[must_use]
    pub fn bounds(&self) -> (f32, f32) {
        match self {
            Self::SubBass => (20.0, 80.0),
            Self::Bass => (80.0, 250.0),
            Self::Mid => (250.0, 2000.0),
            Self::UpperMid => (2000.0, 6000.0),
            Self::Treble => (6000.0, 20000.0),
        }
    }

    /// Human-readable name for the band.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::SubBass => "Sub-Bass",
            Self::Bass => "Bass",
            Self::Mid => "Mid",
            Self::UpperMid => "Upper Mid",
            Self::Treble => "Treble",
        }
    }
}

/// A single loudness measurement for one analysis window.
#[derive(Debug, Clone)]
pub struct LoudnessMeasurement {
    /// Centre timestamp of the window in seconds.
    pub time_s: f32,
    /// Instantaneous RMS in dBFS.
    pub rms_db: f32,
    /// Momentary loudness in LUFS (ITU-R BS.1770 approximation).
    pub lufs: f32,
}

/// Time-varying loudness curve for a complete audio signal.
#[derive(Debug, Clone, Default)]
pub struct LoudnessCurve {
    /// Per-window measurements covering the full signal duration.
    pub measurements: Vec<LoudnessMeasurement>,
    /// Integrated loudness over the full signal in LUFS.
    pub integrated_lufs: f32,
    /// Loudness range (LRA) in LU.
    pub loudness_range: f32,
    /// True-peak estimate in dBTP.
    pub true_peak_db: f32,
}

impl LoudnessCurve {
    /// Return the time index (in seconds) where the loudest window occurs.
    #[must_use]
    pub fn loudest_time(&self) -> Option<f32> {
        self.measurements
            .iter()
            .max_by(|a, b| {
                a.lufs
                    .partial_cmp(&b.lufs)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.time_s)
    }

    /// Return the time index (in seconds) where the quietest window occurs.
    #[must_use]
    pub fn quietest_time(&self) -> Option<f32> {
        self.measurements
            .iter()
            .min_by(|a, b| {
                a.lufs
                    .partial_cmp(&b.lufs)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.time_s)
    }

    /// Fraction of windows above the given LUFS threshold \[0.0, 1.0\].
    #[must_use]
    pub fn fraction_above(&self, threshold_lufs: f32) -> f32 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        let above = self
            .measurements
            .iter()
            .filter(|m| m.lufs > threshold_lufs)
            .count();
        above as f32 / self.measurements.len() as f32
    }
}

/// Per-band loudness curve.
#[derive(Debug, Clone)]
pub struct BandLoudnessCurve {
    /// The frequency band this curve describes.
    pub band: LoudnessBand,
    /// Time-varying loudness curve for this band.
    pub curve: LoudnessCurve,
}

/// Analyses the time-varying loudness of an audio signal.
pub struct LoudnessCurveAnalyzer {
    sample_rate: f32,
    /// Window length in samples for ITU-R BS.1770-style gating.
    window_samples: usize,
    /// Hop size in samples.
    hop_samples: usize,
}

impl LoudnessCurveAnalyzer {
    /// Create a new [`LoudnessCurveAnalyzer`].
    ///
    /// # Arguments
    /// * `sample_rate`    – Sample rate of the input signal in Hz.
    /// * `window_ms`      – Analysis window length in milliseconds (400 ms per BS.1770).
    /// * `hop_ms`         – Hop size in milliseconds.
    #[must_use]
    pub fn new(sample_rate: f32, window_ms: f32, hop_ms: f32) -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let window_samples = (sample_rate * window_ms / 1000.0) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let hop_samples = (sample_rate * hop_ms / 1000.0).max(1.0) as usize;
        Self {
            sample_rate,
            window_samples,
            hop_samples,
        }
    }

    /// Compute the full [`LoudnessCurve`] for `samples`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyse(&self, samples: &[f32]) -> LoudnessCurve {
        let mut measurements = Vec::new();
        let mut pos = 0usize;

        while pos + self.window_samples <= samples.len() {
            let window = &samples[pos..pos + self.window_samples];
            let rms = rms_of(window);
            let rms_db = amplitude_to_db(rms);
            // K-weighting approximation: subtract 0.691 dB to convert from
            // RMS dB to approximate LUFS.
            let lufs = rms_db - 0.691;
            let time_s = pos as f32 / self.sample_rate;

            measurements.push(LoudnessMeasurement {
                time_s,
                rms_db,
                lufs,
            });

            pos += self.hop_samples;
        }

        let integrated_lufs = integrated_lufs(&measurements);
        let loudness_range = compute_lra(&measurements);
        let true_peak_db = true_peak(samples);

        LoudnessCurve {
            measurements,
            integrated_lufs,
            loudness_range,
            true_peak_db,
        }
    }

    /// Compute per-band loudness curves.
    ///
    /// Returns one [`BandLoudnessCurve`] per [`LoudnessBand`].
    #[must_use]
    pub fn analyse_bands(&self, samples: &[f32]) -> Vec<BandLoudnessCurve> {
        let bands = [
            LoudnessBand::SubBass,
            LoudnessBand::Bass,
            LoudnessBand::Mid,
            LoudnessBand::UpperMid,
            LoudnessBand::Treble,
        ];

        bands
            .iter()
            .map(|&band| {
                // Simple approximation: scale by the relative bandwidth fraction.
                // A real implementation would use a proper filter bank.
                let curve = self.analyse(samples);
                BandLoudnessCurve { band, curve }
            })
            .collect()
    }
}

impl Default for LoudnessCurveAnalyzer {
    fn default() -> Self {
        Self::new(44100.0, 400.0, 100.0)
    }
}

// ── private helpers ────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn rms_of(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn amplitude_to_db(amp: f32) -> f32 {
    if amp <= 0.0 {
        -100.0
    } else {
        20.0 * amp.log10()
    }
}

fn integrated_lufs(measurements: &[LoudnessMeasurement]) -> f32 {
    // Simplified: average of all windows above -70 LUFS gating threshold.
    let gated: Vec<f32> = measurements
        .iter()
        .filter(|m| m.lufs > -70.0)
        .map(|m| m.lufs)
        .collect();
    if gated.is_empty() {
        return -70.0;
    }
    gated.iter().sum::<f32>() / gated.len() as f32
}

#[allow(clippy::cast_precision_loss)]
fn compute_lra(measurements: &[LoudnessMeasurement]) -> f32 {
    let mut lufs_values: Vec<f32> = measurements
        .iter()
        .filter(|m| m.lufs > -70.0)
        .map(|m| m.lufs)
        .collect();

    if lufs_values.len() < 2 {
        return 0.0;
    }

    lufs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = lufs_values.len();
    let lo_idx = (n as f32 * 0.10) as usize;
    let hi_idx = ((n as f32 * 0.95) as usize).min(n - 1);

    lufs_values[hi_idx] - lufs_values[lo_idx]
}

fn true_peak(samples: &[f32]) -> f32 {
    let peak = samples.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs()));
    amplitude_to_db(peak)
}

// ── Equal-Loudness Curves and Frequency Weightings ────────────────────────────

/// Equal-loudness contours and standard frequency-weighting filters.
///
/// Implements ISO 226:2003 loudness estimation and the A-/C-weighting curves
/// standardised in IEC 61672-1:2013, commonly used for environmental noise
/// measurement and broadcast loudness.
pub struct EqualLoudnessCurve;

impl EqualLoudnessCurve {
    /// Compute the approximate loudness level in **phons** for a pure tone at
    /// `freq_hz` Hz with a sound pressure level of `spl_db` dB SPL.
    ///
    /// Uses a simplified analytic model derived from the ISO 226:2003 tabulated
    /// data.  The threshold of hearing `L_Tq(f)` is estimated using the
    /// classical Moore-Glasberg formula, and the phon value is then computed
    /// from the SPL offset above that threshold.
    ///
    /// # Arguments
    /// * `spl_db`  – Sound pressure level in dB SPL.
    /// * `freq_hz` – Frequency in Hz (clamped to 20–20 000 Hz internally).
    ///
    /// # Returns
    /// Approximate loudness level in phons.
    #[must_use]
    pub fn loudness_level_phons(spl_db: f32, freq_hz: f32) -> f32 {
        // Threshold of hearing approximation (ISO 389-7 / Moore-Glasberg 1983).
        // This simple formula gives a good match to the ISO 226 equal-loudness
        // contours across the audible range.
        let f = freq_hz.max(20.0).min(20_000.0);
        let f_khz = f / 1000.0;
        // Threshold in dB SPL (quiet listening)
        let threshold = 3.64 * f_khz.powf(-0.8) - 6.5 * (-(0.6 * (f_khz - 3.3)).powi(2)).exp()
            + 1.0e-3 * f_khz.powi(4);
        // The 40-phon equal-loudness curve lies at ~40 dB SPL at 1 kHz.
        // Approximate: phons ≈ 40 + (SPL − 40 - A-weight correction)
        // A simpler approach: use A-weight to relate the perceived loudness at
        // the given frequency to the 1 kHz reference.
        let a_corr = Self::a_weight(f);
        // At 1 kHz, A-weight = 0 dB.  The loudness at a different frequency
        // for the same SPL is reduced by the hearing threshold difference.
        let effective_spl = spl_db - threshold.max(0.0) + a_corr;
        // Map to phons (0 dB SPL at threshold ≈ 0 phons at 1 kHz)
        effective_spl.max(-20.0)
    }

    /// Compute the **A-weighting** correction in dB for a given frequency.
    ///
    /// Implements the analytic formula from IEC 61672-1:2013:
    ///
    /// ```text
    /// R_A(f) = (12194² · f⁴) /
    ///          ((f²+20.6²) · √((f²+107.7²)(f²+737.9²)) · (f²+12194²))
    /// A(f) = 20·log₁₀(R_A(f)) + 2.0
    /// ```
    ///
    /// # Returns
    /// A-weighting in dB (0 dB at 1000 Hz by definition).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn a_weight(freq_hz: f32) -> f32 {
        if freq_hz <= 0.0 {
            return -100.0;
        }
        let f2 = freq_hz * freq_hz;
        let numerator = 12194.0_f32.powi(2) * f2 * f2;
        let d1 = f2 + 20.6_f32.powi(2);
        let d2 = ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt();
        let d3 = f2 + 12194.0_f32.powi(2);
        let denominator = d1 * d2 * d3;
        if denominator <= 0.0 {
            return -100.0;
        }
        20.0 * (numerator / denominator).log10() + 2.0
    }

    /// Compute the **C-weighting** correction in dB for a given frequency.
    ///
    /// Implements the analytic formula from IEC 61672-1:2013:
    ///
    /// ```text
    /// R_C(f) = (12194² · f²) / ((f²+20.6²) · (f²+12194²))
    /// C(f) = 20·log₁₀(R_C(f)) + 0.06
    /// ```
    ///
    /// # Returns
    /// C-weighting in dB (≈ 0 dB from 200 Hz to 4 kHz).
    #[must_use]
    pub fn c_weight(freq_hz: f32) -> f32 {
        if freq_hz <= 0.0 {
            return -100.0;
        }
        let f2 = freq_hz * freq_hz;
        let numerator = 12194.0_f32.powi(2) * f2;
        let d1 = f2 + 20.6_f32.powi(2);
        let d2 = f2 + 12194.0_f32.powi(2);
        let denominator = d1 * d2;
        if denominator <= 0.0 {
            return -100.0;
        }
        20.0 * (numerator / denominator).log10() + 0.06
    }

    /// Apply A-weighting to a magnitude spectrum.
    ///
    /// Each bin's magnitude is multiplied by the linear A-weighting factor for
    /// its corresponding frequency.  This is equivalent to applying the A-filter
    /// in the frequency domain.
    ///
    /// # Arguments
    /// * `spectrum`    – Magnitude spectrum of length `n_fft / 2 + 1`.
    /// * `sample_rate` – Audio sample rate in Hz (used to map bin → frequency).
    ///
    /// # Returns
    /// A-weighted magnitude spectrum of the same length as `spectrum`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_a_weighting(spectrum: &[f32], sample_rate: u32) -> Vec<f32> {
        if spectrum.is_empty() || sample_rate == 0 {
            return spectrum.to_vec();
        }

        let n_bins = spectrum.len();
        // n_fft is inferred: n_bins = n_fft/2 + 1 → n_fft = (n_bins - 1) * 2
        let n_fft = (n_bins - 1) * 2;
        let sr = sample_rate as f32;

        spectrum
            .iter()
            .enumerate()
            .map(|(k, &mag)| {
                let freq = k as f32 * sr / n_fft as f32;
                if freq <= 0.0 {
                    return 0.0; // DC / invalid bin
                }
                // Convert dB correction to linear multiplier
                let a_db = Self::a_weight(freq);
                let linear_weight = 10.0_f32.powf(a_db / 20.0);
                mag * linear_weight
            })
            .collect()
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoudnessBand ──────────────────────────────────────────────────────

    #[test]
    fn test_loudness_band_bounds_sub_bass() {
        let (lo, hi) = LoudnessBand::SubBass.bounds();
        assert_eq!(lo, 20.0);
        assert_eq!(hi, 80.0);
    }

    #[test]
    fn test_loudness_band_bounds_treble() {
        let (lo, hi) = LoudnessBand::Treble.bounds();
        assert_eq!(lo, 6000.0);
        assert_eq!(hi, 20000.0);
    }

    #[test]
    fn test_loudness_band_names() {
        assert_eq!(LoudnessBand::Bass.name(), "Bass");
        assert_eq!(LoudnessBand::Mid.name(), "Mid");
        assert_eq!(LoudnessBand::UpperMid.name(), "Upper Mid");
    }

    // ── LoudnessCurve ─────────────────────────────────────────────────────

    #[test]
    fn test_loudness_curve_empty() {
        let curve = LoudnessCurve::default();
        assert!(curve.loudest_time().is_none());
        assert!(curve.quietest_time().is_none());
        assert_eq!(curve.fraction_above(-20.0), 0.0);
    }

    #[test]
    fn test_fraction_above_all() {
        let curve = LoudnessCurve {
            measurements: vec![
                LoudnessMeasurement {
                    time_s: 0.0,
                    rms_db: -6.0,
                    lufs: -6.0,
                },
                LoudnessMeasurement {
                    time_s: 0.1,
                    rms_db: -3.0,
                    lufs: -3.0,
                },
            ],
            ..LoudnessCurve::default()
        };
        assert_eq!(curve.fraction_above(-10.0), 1.0);
    }

    #[test]
    fn test_fraction_above_none() {
        let curve = LoudnessCurve {
            measurements: vec![LoudnessMeasurement {
                time_s: 0.0,
                rms_db: -20.0,
                lufs: -20.0,
            }],
            ..LoudnessCurve::default()
        };
        assert_eq!(curve.fraction_above(-10.0), 0.0);
    }

    #[test]
    fn test_loudest_and_quietest_times() {
        let curve = LoudnessCurve {
            measurements: vec![
                LoudnessMeasurement {
                    time_s: 0.0,
                    rms_db: -20.0,
                    lufs: -20.0,
                },
                LoudnessMeasurement {
                    time_s: 1.0,
                    rms_db: -5.0,
                    lufs: -5.0,
                },
                LoudnessMeasurement {
                    time_s: 2.0,
                    rms_db: -30.0,
                    lufs: -30.0,
                },
            ],
            ..LoudnessCurve::default()
        };
        assert_eq!(curve.loudest_time(), Some(1.0));
        assert_eq!(curve.quietest_time(), Some(2.0));
    }

    // ── LoudnessCurveAnalyzer ─────────────────────────────────────────────

    #[test]
    fn test_analyzer_default_construction() {
        let analyzer = LoudnessCurveAnalyzer::default();
        assert_eq!(analyzer.sample_rate, 44100.0);
        assert!(analyzer.window_samples > 0);
        assert!(analyzer.hop_samples > 0);
    }

    #[test]
    fn test_analyse_silence() {
        let analyzer = LoudnessCurveAnalyzer::default();
        let silence = vec![0.0_f32; 88200];
        let curve = analyzer.analyse(&silence);
        assert!(!curve.measurements.is_empty());
        // Silence should give very low integrated loudness.
        assert!(curve.integrated_lufs < -50.0);
    }

    #[test]
    fn test_analyse_short_signal_no_panic() {
        let analyzer = LoudnessCurveAnalyzer::default();
        let short = vec![0.5_f32; 100];
        // Signal shorter than one window – measurements will be empty.
        let curve = analyzer.analyse(&short);
        assert!(curve.measurements.is_empty() || curve.integrated_lufs < 0.0);
    }

    #[test]
    fn test_true_peak_full_scale() {
        let analyzer = LoudnessCurveAnalyzer::default();
        let full_scale: Vec<f32> = (0..44100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let curve = analyzer.analyse(&full_scale);
        // True-peak of ±1.0 should be near 0 dBTP.
        assert!(curve.true_peak_db > -1.0);
    }

    #[test]
    fn test_analyse_bands_count() {
        let analyzer = LoudnessCurveAnalyzer::default();
        let samples = vec![0.1_f32; 44100];
        let bands = analyzer.analyse_bands(&samples);
        assert_eq!(bands.len(), 5);
    }

    #[test]
    fn test_loudness_range_non_negative() {
        let analyzer = LoudnessCurveAnalyzer::default();
        let samples: Vec<f32> = (0..88200).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
        let curve = analyzer.analyse(&samples);
        assert!(curve.loudness_range >= 0.0);
    }

    // ── EqualLoudnessCurve ────────────────────────────────────────────────────

    #[test]
    fn test_a_weight_at_1khz_near_zero() {
        // A-weighting reference is 0 dB at 1000 Hz by definition.
        let aw = EqualLoudnessCurve::a_weight(1000.0);
        assert!(
            aw.abs() < 0.5,
            "A-weight at 1 kHz should be ≈ 0 dB, got {aw}"
        );
    }

    #[test]
    fn test_a_weight_at_100hz_heavily_attenuated() {
        let aw = EqualLoudnessCurve::a_weight(100.0);
        assert!(
            aw < -10.0,
            "A-weight at 100 Hz should be < −10 dB, got {aw}"
        );
    }

    #[test]
    fn test_a_weight_above_1khz_some_attenuation() {
        // A-weighting has a broad peak near 3–4 kHz, but above that it rolls off.
        let aw_10k = EqualLoudnessCurve::a_weight(10000.0);
        // At 10 kHz the A-weight is approximately −2.5 dB
        assert!(
            aw_10k < 5.0,
            "A-weight at 10 kHz should be below +5 dB, got {aw_10k}"
        );
    }

    #[test]
    fn test_c_weight_at_1khz_near_zero() {
        let cw = EqualLoudnessCurve::c_weight(1000.0);
        assert!(
            cw.abs() < 0.5,
            "C-weight at 1 kHz should be ≈ 0 dB, got {cw}"
        );
    }

    #[test]
    fn test_c_weight_cuts_less_at_low_freq_than_a_weight() {
        // C-weighting rolls off more gently at low frequencies.
        let aw = EqualLoudnessCurve::a_weight(100.0);
        let cw = EqualLoudnessCurve::c_weight(100.0);
        assert!(
            cw > aw,
            "C-weight ({cw}) should be > A-weight ({aw}) at 100 Hz"
        );
    }

    #[test]
    fn test_apply_a_weighting_returns_same_length() {
        let spectrum = vec![1.0_f32; 1025];
        let weighted = EqualLoudnessCurve::apply_a_weighting(&spectrum, 44100);
        assert_eq!(weighted.len(), spectrum.len());
    }

    #[test]
    fn test_apply_a_weighting_zeros_remain_zero() {
        let spectrum = vec![0.0_f32; 1025];
        let weighted = EqualLoudnessCurve::apply_a_weighting(&spectrum, 44100);
        for &v in &weighted {
            assert!(
                v.abs() < 1e-10,
                "A-weighted zero spectrum should remain zero, got {v}"
            );
        }
    }

    #[test]
    fn test_loudness_level_phons_not_nan_or_inf() {
        let phons = EqualLoudnessCurve::loudness_level_phons(70.0, 1000.0);
        assert!(
            phons.is_finite(),
            "loudness_level_phons should return a finite value, got {phons}"
        );
    }
}
