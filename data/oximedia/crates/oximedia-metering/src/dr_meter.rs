//! Dynamic range metering: peak, RMS, crest factor, and DR scoring.
//!
//! Implements a subset of the DR (Dynamic Range) measurement methodology used
//! in tools such as the TT Dynamic Range Meter.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Small constant added before log to avoid −∞ when the signal is silent.
const FLOOR: f32 = 1e-10;

/// Compute the peak level in dBFS from a slice of samples.
///
/// Returns `−∞` (as a large negative number) when `samples` is empty.
#[must_use]
pub fn compute_peak_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    20.0 * (peak + FLOOR).log10()
}

/// Compute the RMS level in dBFS from a slice of samples.
///
/// Returns `−∞` when `samples` is empty.
#[must_use]
pub fn compute_rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    20.0 * (mean_sq.sqrt() + FLOOR).log10()
}

/// Compute the dynamic range (peak − RMS) in dB.
///
/// Returns `0.0` when `samples` is empty.
#[must_use]
pub fn compute_dr(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    compute_peak_db(samples) - compute_rms_db(samples)
}

/// Aggregated dynamic range metrics for a signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicRangeMeter {
    /// Peak level in dBFS.
    pub peak_db: f32,
    /// RMS level in dBFS.
    pub rms_db: f32,
    /// Crest factor (peak − RMS) in dB.
    pub crest_factor_db: f32,
    /// Estimated dynamic range in dB.
    pub dynamic_range_db: f32,
}

impl DynamicRangeMeter {
    /// Compute metrics from a slice of audio samples.
    #[must_use]
    pub fn from_samples(samples: &[f32]) -> Self {
        let peak_db = compute_peak_db(samples);
        let rms_db = compute_rms_db(samples);
        let crest_factor_db = if peak_db.is_finite() && rms_db.is_finite() {
            peak_db - rms_db
        } else {
            0.0
        };
        let dynamic_range_db = crest_factor_db;
        Self {
            peak_db,
            rms_db,
            crest_factor_db,
            dynamic_range_db,
        }
    }

    /// Return `true` when the dynamic range is below 8 dB (heavily compressed).
    #[must_use]
    pub fn is_highly_compressed(&self) -> bool {
        self.dynamic_range_db < 8.0
    }
}

/// A labelled DR measurement suitable for scoring and reporting.
#[derive(Debug, Clone)]
pub struct DrMeasurement {
    /// Identifier for the track or segment.
    pub track_id: String,
    /// Dynamic range in dB.
    pub dr_db: f32,
    /// Peak level in dBFS.
    pub peak_db: f32,
    /// RMS level in dBFS.
    pub rms_db: f32,
}

impl DrMeasurement {
    /// Create a measurement from samples.
    #[must_use]
    pub fn from_samples(track_id: impl Into<String>, samples: &[f32]) -> Self {
        let meter = DynamicRangeMeter::from_samples(samples);
        Self {
            track_id: track_id.into(),
            dr_db: meter.dynamic_range_db,
            peak_db: meter.peak_db,
            rms_db: meter.rms_db,
        }
    }

    /// Qualitative DR score.
    ///
    /// * `"Excellent"` — DR ≥ 20 dB
    /// * `"Good"`      — DR ≥ 13 dB
    /// * `"Fair"`      — DR ≥ 8 dB
    /// * `"Poor"`      — DR < 8 dB
    #[must_use]
    pub fn dr_score(&self) -> &str {
        if self.dr_db >= 20.0 {
            "Excellent"
        } else if self.dr_db >= 13.0 {
            "Good"
        } else if self.dr_db >= 8.0 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a constant-amplitude signal.
    fn constant_signal(amp: f32, n: usize) -> Vec<f32> {
        vec![amp; n]
    }

    /// Generate a sine wave at full scale.
    fn sine_wave(freq_hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    // ---------- compute_peak_db ----------

    #[test]
    fn test_peak_db_full_scale() {
        let s = constant_signal(1.0, 100);
        let peak = compute_peak_db(&s);
        // 20*log10(1.0) ≈ 0 dBFS (ignoring floor)
        assert!(peak > -0.01 && peak < 0.01, "peak={peak}");
    }

    #[test]
    fn test_peak_db_half_scale() {
        let s = constant_signal(0.5, 100);
        let peak = compute_peak_db(&s);
        // 20*log10(0.5) ≈ −6 dBFS
        assert!(peak < -5.0 && peak > -7.0, "peak={peak}");
    }

    #[test]
    fn test_peak_db_empty_is_neg_inf() {
        assert_eq!(compute_peak_db(&[]), f32::NEG_INFINITY);
    }

    // ---------- compute_rms_db ----------

    #[test]
    fn test_rms_db_constant_signal() {
        let s = constant_signal(1.0, 1000);
        let rms = compute_rms_db(&s);
        assert!(rms > -0.01 && rms < 0.01, "rms={rms}");
    }

    #[test]
    fn test_rms_db_empty_is_neg_inf() {
        assert_eq!(compute_rms_db(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn test_rms_db_sine_approx_minus_3db() {
        // RMS of a sine wave ≈ A/√2 → −3 dBFS for A=1
        let s = sine_wave(1_000.0, 44_100.0, 44_100);
        let rms = compute_rms_db(&s);
        assert!(rms > -4.0 && rms < -2.0, "rms={rms}");
    }

    // ---------- compute_dr ----------

    #[test]
    fn test_dr_empty_is_zero() {
        assert!((compute_dr(&[]) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dr_constant_signal_near_zero() {
        // peak ≈ RMS for constant signal
        let s = constant_signal(0.8, 1000);
        let dr = compute_dr(&s);
        assert!(dr.abs() < 0.5, "dr={dr}");
    }

    // ---------- DynamicRangeMeter ----------

    #[test]
    fn test_meter_from_samples_fields() {
        let s = sine_wave(440.0, 44_100.0, 44_100);
        let m = DynamicRangeMeter::from_samples(&s);
        assert!(m.peak_db.is_finite());
        assert!(m.rms_db.is_finite());
        assert!(m.crest_factor_db >= 0.0, "crest={}", m.crest_factor_db);
    }

    #[test]
    fn test_is_highly_compressed_true() {
        let m = DynamicRangeMeter {
            peak_db: -0.1,
            rms_db: -3.0,
            crest_factor_db: 5.0,
            dynamic_range_db: 5.0,
        };
        assert!(m.is_highly_compressed());
    }

    #[test]
    fn test_is_highly_compressed_false() {
        let m = DynamicRangeMeter {
            peak_db: -0.1,
            rms_db: -20.0,
            crest_factor_db: 19.9,
            dynamic_range_db: 19.9,
        };
        assert!(!m.is_highly_compressed());
    }

    // ---------- DrMeasurement ----------

    #[test]
    fn test_dr_score_excellent() {
        let m = DrMeasurement {
            track_id: "t".into(),
            dr_db: 22.0,
            peak_db: -0.1,
            rms_db: -22.1,
        };
        assert_eq!(m.dr_score(), "Excellent");
    }

    #[test]
    fn test_dr_score_good() {
        let m = DrMeasurement {
            track_id: "t".into(),
            dr_db: 15.0,
            peak_db: -0.1,
            rms_db: -15.1,
        };
        assert_eq!(m.dr_score(), "Good");
    }

    #[test]
    fn test_dr_score_fair() {
        let m = DrMeasurement {
            track_id: "t".into(),
            dr_db: 9.0,
            peak_db: -0.1,
            rms_db: -9.1,
        };
        assert_eq!(m.dr_score(), "Fair");
    }

    #[test]
    fn test_dr_score_poor() {
        let m = DrMeasurement {
            track_id: "t".into(),
            dr_db: 5.0,
            peak_db: -0.1,
            rms_db: -5.1,
        };
        assert_eq!(m.dr_score(), "Poor");
    }

    #[test]
    fn test_dr_measurement_from_samples() {
        let s = sine_wave(440.0, 44_100.0, 44_100);
        let m = DrMeasurement::from_samples("track-1", &s);
        assert_eq!(m.track_id, "track-1");
        assert!(m.dr_db >= 0.0);
    }
}
