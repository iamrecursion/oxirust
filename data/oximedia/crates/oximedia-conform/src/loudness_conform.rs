//! Loudness conformance for broadcast and streaming delivery.
//!
//! Implements EBU R128, ATSC A/85, and custom loudness correction,
//! including integrated loudness measurement gating, correction gain
//! calculation, and per-segment loudness normalisation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// Loudness standard to conform to.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoudnessStandard {
    /// EBU R128 — -23 LUFS, -1 dBTP max true peak
    EbuR128,
    /// ATSC A/85 — -24 LKFS, -2 dBTP max true peak
    AtscA85,
    /// Netflix OC — -27 LUFS, -2 dBTP max true peak
    NetflixOC,
    /// Apple/iTunes — -16 LUFS, -1 dBTP max true peak
    AppleItunes,
    /// Spotify — -14 LUFS, -1 dBTP max true peak
    Spotify,
    /// Custom user-defined target
    Custom {
        /// Target integrated loudness (LUFS/LKFS)
        target_lufs: f32,
        /// Max true peak (dBTP)
        max_true_peak_dbtp: f32,
    },
}

impl LoudnessStandard {
    /// Returns the target integrated loudness in LUFS/LKFS.
    #[must_use]
    pub fn target_lufs(self) -> f32 {
        match self {
            Self::EbuR128 => -23.0,
            Self::AtscA85 => -24.0,
            Self::NetflixOC => -27.0,
            Self::AppleItunes => -16.0,
            Self::Spotify => -14.0,
            Self::Custom { target_lufs, .. } => target_lufs,
        }
    }

    /// Returns the maximum true peak in dBTP.
    #[must_use]
    pub fn max_true_peak_dbtp(self) -> f32 {
        match self {
            Self::EbuR128 => -1.0,
            Self::AtscA85 => -2.0,
            Self::NetflixOC => -2.0,
            Self::AppleItunes => -1.0,
            Self::Spotify => -1.0,
            Self::Custom {
                max_true_peak_dbtp, ..
            } => max_true_peak_dbtp,
        }
    }
}

/// Result of a loudness measurement pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessMeasurement {
    /// Integrated (program) loudness in LUFS
    pub integrated_lufs: f32,
    /// True peak level in dBTP
    pub true_peak_dbtp: f32,
    /// Loudness range (LRA) in LU
    pub loudness_range_lu: f32,
    /// Short-term loudness (maximum over all 3-second windows) in LUFS
    pub max_short_term_lufs: f32,
    /// Momentary loudness (maximum over all 400ms windows) in LUFS
    pub max_momentary_lufs: f32,
    /// Duration of the measured audio in seconds
    pub duration_seconds: f64,
    /// Number of gated blocks used for integrated loudness
    pub gated_block_count: u64,
}

impl LoudnessMeasurement {
    /// Creates a mock measurement for testing.
    #[must_use]
    pub fn mock(integrated: f32, true_peak: f32, lra: f32, duration: f64) -> Self {
        Self {
            integrated_lufs: integrated,
            true_peak_dbtp: true_peak,
            loudness_range_lu: lra,
            max_short_term_lufs: integrated + 6.0,
            max_momentary_lufs: integrated + 9.0,
            duration_seconds: duration,
            gated_block_count: (duration * 10.0) as u64,
        }
    }
}

/// Gating configuration for loudness measurement (ITU-R BS.1770-4).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Absolute gate threshold in LKFS (default: -70 LKFS)
    pub absolute_gate_lufs: f32,
    /// Relative gate offset below ungated mean (default: -10 LU)
    pub relative_gate_offset_lu: f32,
    /// Block length in milliseconds (default: 400 ms)
    pub block_length_ms: u32,
    /// Block overlap fraction (default: 0.75 = 75%)
    pub overlap_fraction: f32,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            absolute_gate_lufs: -70.0,
            relative_gate_offset_lu: -10.0,
            block_length_ms: 400,
            overlap_fraction: 0.75,
        }
    }
}

impl GatingConfig {
    /// Returns the hop size between blocks in milliseconds.
    #[must_use]
    pub fn hop_ms(&self) -> f32 {
        self.block_length_ms as f32 * (1.0 - self.overlap_fraction)
    }

    /// Returns the absolute gate threshold as a linear power value.
    #[must_use]
    pub fn absolute_gate_linear(&self) -> f32 {
        10.0_f32.powf(self.absolute_gate_lufs / 10.0)
    }
}

/// Loudness correction action to apply to a clip or program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoudnessCorrection {
    /// Apply a linear gain (in dB)
    GainDb(f32),
    /// Dynamic range compression with knee and ratio
    DynamicCompression {
        /// Threshold in dBFS
        threshold_dbfs: f32,
        /// Compression ratio (e.g. 2.0 = 2:1)
        ratio: f32,
        /// Knee width in dB
        knee_db: f32,
        /// Makeup gain in dB applied after compression
        makeup_gain_db: f32,
    },
    /// True peak limiting to a maximum dBTP level
    TruePeakLimit {
        /// Maximum true peak in dBTP
        max_dbtp: f32,
    },
    /// No correction needed
    NoCorrection,
}

/// Calculates the required gain correction (in dB) to reach a target loudness.
///
/// # Arguments
///
/// * `measured_lufs` – measured integrated loudness of the content
/// * `target_lufs`   – desired integrated loudness
///
/// Returns the gain adjustment in dB (positive = louder, negative = quieter).
#[must_use]
pub fn calculate_gain_correction(measured_lufs: f32, target_lufs: f32) -> f32 {
    target_lufs - measured_lufs
}

/// Checks whether a measurement complies with a given loudness standard.
///
/// Returns a list of compliance issues. An empty list means full compliance.
#[must_use]
pub fn check_compliance(
    measurement: &LoudnessMeasurement,
    standard: LoudnessStandard,
    tolerance_lu: f32,
) -> Vec<LoudnessComplianceIssue> {
    let mut issues = Vec::new();

    let target = standard.target_lufs();
    let diff = (measurement.integrated_lufs - target).abs();
    if diff > tolerance_lu {
        issues.push(LoudnessComplianceIssue::IntegratedOutOfRange {
            target,
            tolerance: tolerance_lu,
            actual: measurement.integrated_lufs,
        });
    }

    if measurement.true_peak_dbtp > standard.max_true_peak_dbtp() {
        issues.push(LoudnessComplianceIssue::TruePeakExceeded {
            limit: standard.max_true_peak_dbtp(),
            actual: measurement.true_peak_dbtp,
        });
    }

    issues
}

/// A loudness compliance issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoudnessComplianceIssue {
    /// Integrated loudness is outside the tolerance window.
    IntegratedOutOfRange {
        /// Target LUFS
        target: f32,
        /// Tolerance in LU
        tolerance: f32,
        /// Measured value
        actual: f32,
    },
    /// True peak exceeds the allowed maximum.
    TruePeakExceeded {
        /// Limit in dBTP
        limit: f32,
        /// Measured value
        actual: f32,
    },
}

/// A single loudness-gated block used in the BS.1770 algorithm.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GatedBlock {
    /// Mean square value (linear) for this block
    pub mean_square: f32,
    /// Start offset within the stream in milliseconds
    pub start_ms: u64,
}

impl GatedBlock {
    /// Converts mean square to LKFS.
    #[must_use]
    pub fn to_lkfs(self) -> f32 {
        if self.mean_square <= 0.0 {
            return f32::NEG_INFINITY;
        }
        -0.691 + 10.0 * self.mean_square.log10()
    }
}

/// Applies absolute and relative gates to a sequence of gated blocks,
/// returning the mean-square values that pass both gates.
#[must_use]
pub fn apply_gating(blocks: &[GatedBlock], config: &GatingConfig) -> Vec<f32> {
    // Step 1: absolute gate
    let abs_gate_lin = config.absolute_gate_linear();
    let abs_gated: Vec<f32> = blocks
        .iter()
        .filter(|b| b.mean_square >= abs_gate_lin)
        .map(|b| b.mean_square)
        .collect();

    if abs_gated.is_empty() {
        return abs_gated;
    }

    // Step 2: compute ungated mean for relative gate
    let ungated_mean: f32 = abs_gated.iter().sum::<f32>() / abs_gated.len() as f32;
    let rel_gate_lin = ungated_mean * 10.0_f32.powf(config.relative_gate_offset_lu / 10.0);

    abs_gated
        .into_iter()
        .filter(|&v| v >= rel_gate_lin)
        .collect()
}

/// Computes integrated loudness (LUFS) from a set of gated mean-square values.
#[must_use]
pub fn integrated_loudness_from_gated(gated_values: &[f32]) -> f32 {
    if gated_values.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean: f32 = gated_values.iter().sum::<f32>() / gated_values.len() as f32;
    if mean <= 0.0 {
        return f32::NEG_INFINITY;
    }
    -0.691 + 10.0 * mean.log10()
}

/// Per-segment loudness data (e.g. for chapter-based normalisation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentLoudness {
    /// Segment identifier (e.g. chapter index or clip name)
    pub id: String,
    /// Start time in seconds
    pub start_seconds: f64,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Integrated loudness measurement
    pub measurement: LoudnessMeasurement,
    /// Gain correction to apply (dB), calculated after measurement
    pub gain_correction_db: f32,
}

impl SegmentLoudness {
    /// Creates a new segment loudness entry and computes the gain correction.
    #[must_use]
    pub fn new(
        id: String,
        start_seconds: f64,
        measurement: LoudnessMeasurement,
        standard: LoudnessStandard,
    ) -> Self {
        let gain_correction_db =
            calculate_gain_correction(measurement.integrated_lufs, standard.target_lufs());
        let duration_seconds = measurement.duration_seconds;
        Self {
            id,
            start_seconds,
            duration_seconds,
            measurement,
            gain_correction_db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebu_r128_target() {
        assert!((LoudnessStandard::EbuR128.target_lufs() - (-23.0)).abs() < 1e-5);
    }

    #[test]
    fn test_atsc_a85_true_peak() {
        assert!((LoudnessStandard::AtscA85.max_true_peak_dbtp() - (-2.0)).abs() < 1e-5);
    }

    #[test]
    fn test_custom_loudness_standard() {
        let std = LoudnessStandard::Custom {
            target_lufs: -20.0,
            max_true_peak_dbtp: -0.5,
        };
        assert!((std.target_lufs() - (-20.0)).abs() < 1e-5);
        assert!((std.max_true_peak_dbtp() - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_gain_correction_positive() {
        // Content is -28 LUFS, target is -23 LUFS → need +5 dB
        let gain = calculate_gain_correction(-28.0, -23.0);
        assert!((gain - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_gain_correction_negative() {
        // Content is -18 LUFS, target is -23 LUFS → need -5 dB
        let gain = calculate_gain_correction(-18.0, -23.0);
        assert!((gain - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn test_calculate_gain_correction_zero() {
        let gain = calculate_gain_correction(-23.0, -23.0);
        assert!(gain.abs() < 1e-5);
    }

    #[test]
    fn test_check_compliance_pass() {
        let m = LoudnessMeasurement::mock(-23.0, -2.0, 8.0, 60.0);
        let issues = check_compliance(&m, LoudnessStandard::EbuR128, 1.0);
        assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
    }

    #[test]
    fn test_check_compliance_loudness_fail() {
        let m = LoudnessMeasurement::mock(-18.0, -2.0, 8.0, 60.0);
        let issues = check_compliance(&m, LoudnessStandard::EbuR128, 1.0);
        assert!(!issues.is_empty());
        let has_integrated = issues
            .iter()
            .any(|i| matches!(i, LoudnessComplianceIssue::IntegratedOutOfRange { .. }));
        assert!(has_integrated);
    }

    #[test]
    fn test_check_compliance_true_peak_fail() {
        let m = LoudnessMeasurement::mock(-23.0, 0.0, 8.0, 60.0); // true peak 0 dBTP
        let issues = check_compliance(&m, LoudnessStandard::EbuR128, 1.0);
        let has_peak = issues
            .iter()
            .any(|i| matches!(i, LoudnessComplianceIssue::TruePeakExceeded { .. }));
        assert!(has_peak);
    }

    #[test]
    fn test_gating_config_hop_ms() {
        let cfg = GatingConfig::default();
        assert!((cfg.hop_ms() - 100.0).abs() < 1e-3); // 400ms * 0.25 = 100ms
    }

    #[test]
    fn test_gating_config_absolute_gate_linear() {
        let cfg = GatingConfig::default();
        let lin = cfg.absolute_gate_linear();
        assert!(lin > 0.0 && lin < 1e-6); // -70 LKFS is a very small power
    }

    #[test]
    fn test_apply_gating_removes_silent_blocks() {
        let cfg = GatingConfig::default();
        let blocks = vec![
            GatedBlock {
                mean_square: 1e-10,
                start_ms: 0,
            }, // silent
            GatedBlock {
                mean_square: 0.01,
                start_ms: 100,
            }, // audible
            GatedBlock {
                mean_square: 0.02,
                start_ms: 200,
            }, // audible
        ];
        let gated = apply_gating(&blocks, &cfg);
        assert!(!gated.is_empty());
        // The silent block should be filtered out
        for v in &gated {
            assert!(*v > 1e-8);
        }
    }

    #[test]
    fn test_integrated_loudness_from_gated_empty() {
        let result = integrated_loudness_from_gated(&[]);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_integrated_loudness_from_gated_nonzero() {
        // mean_square of 0.01 → LUFS = -0.691 + 10 * log10(0.01) = -0.691 - 20 = -20.691
        let vals = vec![0.01f32; 100];
        let lufs = integrated_loudness_from_gated(&vals);
        assert!((lufs - (-20.691)).abs() < 0.01);
    }

    #[test]
    fn test_segment_loudness_gain_correction() {
        let m = LoudnessMeasurement::mock(-27.0, -3.0, 6.0, 45.0);
        let seg = SegmentLoudness::new("ch01".to_string(), 0.0, m, LoudnessStandard::EbuR128);
        // -27 → -23 = +4 dB
        assert!((seg.gain_correction_db - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_gated_block_to_lkfs() {
        let block = GatedBlock {
            mean_square: 0.01,
            start_ms: 0,
        };
        let lkfs = block.to_lkfs();
        assert!((lkfs - (-20.691)).abs() < 0.01);
    }
}
