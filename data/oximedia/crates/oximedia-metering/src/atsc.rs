//! ATSC A/85 loudness standard for digital television.
//!
//! Implements the Advanced Television Systems Committee A/85:2013
//! "Techniques for Establishing and Maintaining Audio Loudness for Digital Television".
//!
//! # Standards
//!
//! - ATSC A/85:2013: Recommended Practice for digital television audio
//! - Uses ITU-R BS.1770-4 LKFS measurement (identical to LUFS)
//! - Target: -24 LKFS ±2 dB
//! - Maximum True Peak: typically -2.0 dBTP (to prevent overshoots)
//!
//! # Note on Terminology
//!
//! LKFS (Loudness, K-weighted, relative to Full Scale) is the term used by ATSC.
//! It is numerically identical to LUFS (Loudness Units relative to Full Scale)
//! used by EBU R128. Both use the same ITU-R BS.1770-4 measurement algorithm.

use crate::{LoudnessMeter, LoudnessMetrics, MeterConfig, Standard};

/// ATSC A/85 target loudness in LKFS.
pub const ATSC_TARGET_LKFS: f64 = -24.0;

/// ATSC A/85 tolerance in dB.
pub const ATSC_TOLERANCE_DB: f64 = 2.0;

/// ATSC A/85 recommended maximum true peak in dBTP.
pub const ATSC_MAX_TRUEPEAK_DBTP: f64 = -2.0;

/// ATSC program type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtscProgramType {
    /// General broadcast programs.
    General,
    /// Commercials and advertisements.
    Commercial,
    /// News programs.
    News,
    /// Sports programming.
    Sports,
    /// Long-form programming (movies, dramas).
    LongForm,
}

impl AtscProgramType {
    /// Get program type name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Commercial => "Commercial",
            Self::News => "News",
            Self::Sports => "Sports",
            Self::LongForm => "Long-form",
        }
    }

    /// Get recommended target loudness for this program type.
    ///
    /// ATSC A/85 allows some flexibility based on program type.
    pub fn target_lkfs(&self) -> f64 {
        match self {
            Self::General | Self::Commercial | Self::News | Self::Sports => ATSC_TARGET_LKFS,
            Self::LongForm => -24.0, // Same as general, but may have wider dynamic range
        }
    }

    /// Get tolerance for this program type.
    pub fn tolerance(&self) -> f64 {
        match self {
            Self::Commercial => 2.0, // Commercials must be within ±2 dB
            _ => 2.0,
        }
    }
}

/// ATSC A/85 compliance status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtscComplianceStatus {
    /// Fully compliant with ATSC A/85.
    Compliant,
    /// Loudness exceeds +2 dB tolerance.
    TooLoud,
    /// Loudness exceeds -2 dB tolerance.
    TooQuiet,
    /// True peak exceeds recommended limit.
    PeakExceeded,
    /// Multiple compliance issues.
    Multiple,
    /// Insufficient data.
    Unknown,
}

impl AtscComplianceStatus {
    /// Check if compliant.
    pub fn is_compliant(&self) -> bool {
        matches!(self, Self::Compliant)
    }

    /// Get status description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Compliant => "Compliant with ATSC A/85",
            Self::TooLoud => "Programme loudness exceeds +2 dB tolerance",
            Self::TooQuiet => "Programme loudness exceeds -2 dB tolerance",
            Self::PeakExceeded => "True peak exceeds -2.0 dBTP",
            Self::Multiple => "Multiple compliance issues",
            Self::Unknown => "Insufficient data for compliance check",
        }
    }
}

/// ATSC A/85 compliance result.
#[derive(Clone, Debug)]
pub struct AtscA85Compliance {
    /// Compliance status.
    pub status: AtscComplianceStatus,
    /// Measured integrated loudness in LKFS.
    pub integrated_lkfs: f64,
    /// Measured true peak in dBTP.
    pub true_peak_dbtp: f64,
    /// Measured loudness range in LU.
    pub loudness_range: f64,
    /// Deviation from target in dB.
    pub deviation_db: f64,
    /// Is loudness within tolerance?
    pub loudness_ok: bool,
    /// Is true peak within limit?
    pub peak_ok: bool,
    /// Program type being checked.
    pub program_type: AtscProgramType,
}

impl AtscA85Compliance {
    /// Create compliance result from metrics.
    ///
    /// # Arguments
    ///
    /// * `metrics` - Loudness metrics
    /// * `program_type` - ATSC program type
    pub fn from_metrics(metrics: &LoudnessMetrics, program_type: AtscProgramType) -> Self {
        let target = program_type.target_lkfs();
        let tolerance = program_type.tolerance();

        let integrated = metrics.integrated_lufs; // LKFS = LUFS
        let peak = metrics.true_peak_dbtp;
        let lra = metrics.loudness_range;

        let loudness_ok = if integrated.is_finite() {
            integrated >= target - tolerance && integrated <= target + tolerance
        } else {
            false
        };

        let peak_ok = peak <= ATSC_MAX_TRUEPEAK_DBTP;

        let deviation = if integrated.is_finite() {
            integrated - target
        } else {
            0.0
        };

        let status = if !integrated.is_finite() {
            AtscComplianceStatus::Unknown
        } else if loudness_ok && peak_ok {
            AtscComplianceStatus::Compliant
        } else if !loudness_ok && !peak_ok {
            AtscComplianceStatus::Multiple
        } else if !peak_ok {
            AtscComplianceStatus::PeakExceeded
        } else if deviation > tolerance {
            AtscComplianceStatus::TooLoud
        } else {
            AtscComplianceStatus::TooQuiet
        };

        Self {
            status,
            integrated_lkfs: integrated,
            true_peak_dbtp: peak,
            loudness_range: lra,
            deviation_db: deviation,
            loudness_ok,
            peak_ok,
            program_type,
        }
    }

    /// Get recommended gain adjustment.
    pub fn recommended_gain(&self) -> f64 {
        if self.integrated_lkfs.is_finite() {
            self.program_type.target_lkfs() - self.integrated_lkfs
        } else {
            0.0
        }
    }

    /// Check if gain adjustment would cause clipping.
    pub fn would_clip(&self, gain_db: f64) -> bool {
        let adjusted_peak = self.true_peak_dbtp + gain_db;
        adjusted_peak > ATSC_MAX_TRUEPEAK_DBTP
    }

    /// Get safe gain adjustment that won't cause clipping.
    pub fn safe_gain(&self) -> f64 {
        let desired_gain = self.recommended_gain();
        let max_safe_gain = ATSC_MAX_TRUEPEAK_DBTP - self.true_peak_dbtp;
        desired_gain.min(max_safe_gain)
    }
}

/// ATSC A/85 meter.
pub struct AtscA85Meter {
    meter: LoudnessMeter,
    program_type: AtscProgramType,
}

impl AtscA85Meter {
    /// Create a new ATSC A/85 meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `program_type` - Type of program being measured
    pub fn new(
        sample_rate: f64,
        channels: usize,
        program_type: AtscProgramType,
    ) -> crate::MeteringResult<Self> {
        let config = MeterConfig::new(Standard::AtscA85, sample_rate, channels);
        let meter = LoudnessMeter::new(config)?;

        Ok(Self {
            meter,
            program_type,
        })
    }

    /// Process f32 audio samples.
    pub fn process_f32(&mut self, samples: &[f32]) {
        self.meter.process_f32(samples);
    }

    /// Process f64 audio samples.
    pub fn process_f64(&mut self, samples: &[f64]) {
        self.meter.process_f64(samples);
    }

    /// Get current loudness metrics.
    pub fn metrics(&mut self) -> LoudnessMetrics {
        self.meter.metrics()
    }

    /// Check ATSC A/85 compliance.
    pub fn check_compliance(&mut self) -> AtscA85Compliance {
        let metrics = self.metrics();
        AtscA85Compliance::from_metrics(&metrics, self.program_type)
    }

    /// Get program type.
    pub fn program_type(&self) -> AtscProgramType {
        self.program_type
    }

    /// Set program type.
    pub fn set_program_type(&mut self, program_type: AtscProgramType) {
        self.program_type = program_type;
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.meter.reset();
    }

    /// Get the underlying meter.
    pub fn meter(&self) -> &LoudnessMeter {
        &self.meter
    }

    /// Get mutable reference to underlying meter.
    pub fn meter_mut(&mut self) -> &mut LoudnessMeter {
        &mut self.meter
    }
}

/// Check if loudness value is compliant with ATSC A/85.
///
/// # Arguments
///
/// * `lkfs` - Measured loudness in LKFS
/// * `program_type` - Program type
///
/// # Returns
///
/// `true` if within tolerance
pub fn is_lkfs_compliant(lkfs: f64, program_type: AtscProgramType) -> bool {
    if !lkfs.is_finite() {
        return false;
    }

    let target = program_type.target_lkfs();
    let tolerance = program_type.tolerance();

    lkfs >= target - tolerance && lkfs <= target + tolerance
}

/// Check if true peak is compliant with ATSC A/85.
///
/// # Arguments
///
/// * `true_peak_dbtp` - True peak in dBTP
///
/// # Returns
///
/// `true` if within limit
pub fn is_peak_compliant(true_peak_dbtp: f64) -> bool {
    true_peak_dbtp <= ATSC_MAX_TRUEPEAK_DBTP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atsc_constants() {
        assert_eq!(ATSC_TARGET_LKFS, -24.0);
        assert_eq!(ATSC_TOLERANCE_DB, 2.0);
        assert_eq!(ATSC_MAX_TRUEPEAK_DBTP, -2.0);
    }

    #[test]
    fn test_program_type_names() {
        assert_eq!(AtscProgramType::General.name(), "General");
        assert_eq!(AtscProgramType::Commercial.name(), "Commercial");
    }

    #[test]
    fn test_program_type_targets() {
        assert_eq!(AtscProgramType::General.target_lkfs(), -24.0);
        assert_eq!(AtscProgramType::Commercial.target_lkfs(), -24.0);
    }

    #[test]
    fn test_compliance_status() {
        assert!(AtscComplianceStatus::Compliant.is_compliant());
        assert!(!AtscComplianceStatus::TooLoud.is_compliant());
    }

    #[test]
    fn test_compliance_from_metrics() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -24.0,
            true_peak_dbtp: -3.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = AtscA85Compliance::from_metrics(&metrics, AtscProgramType::General);
        assert!(compliance.status.is_compliant());
        assert!(compliance.loudness_ok);
        assert!(compliance.peak_ok);
    }

    #[test]
    fn test_compliance_too_loud() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -20.0,
            true_peak_dbtp: -3.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = AtscA85Compliance::from_metrics(&metrics, AtscProgramType::General);
        assert_eq!(compliance.status, AtscComplianceStatus::TooLoud);
    }

    #[test]
    fn test_compliance_peak_exceeded() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -24.0,
            true_peak_dbtp: -1.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = AtscA85Compliance::from_metrics(&metrics, AtscProgramType::General);
        assert_eq!(compliance.status, AtscComplianceStatus::PeakExceeded);
    }

    #[test]
    fn test_is_lkfs_compliant() {
        assert!(is_lkfs_compliant(-24.0, AtscProgramType::General));
        assert!(is_lkfs_compliant(-22.0, AtscProgramType::General));
        assert!(is_lkfs_compliant(-26.0, AtscProgramType::General));
        assert!(!is_lkfs_compliant(-20.0, AtscProgramType::General));
        assert!(!is_lkfs_compliant(-28.0, AtscProgramType::General));
    }

    #[test]
    fn test_is_peak_compliant() {
        assert!(is_peak_compliant(-3.0));
        assert!(is_peak_compliant(-2.0));
        assert!(!is_peak_compliant(-1.0));
        assert!(!is_peak_compliant(0.0));
    }

    #[test]
    fn test_recommended_gain() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -20.0,
            true_peak_dbtp: -3.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = AtscA85Compliance::from_metrics(&metrics, AtscProgramType::General);
        assert_eq!(compliance.recommended_gain(), -4.0);
    }

    #[test]
    fn test_safe_gain() {
        let metrics = LoudnessMetrics {
            integrated_lufs: -30.0,
            true_peak_dbtp: -3.0,
            loudness_range: 10.0,
            ..Default::default()
        };

        let compliance = AtscA85Compliance::from_metrics(&metrics, AtscProgramType::General);
        let safe = compliance.safe_gain();

        // Safe gain should not cause peak to exceed -2 dBTP
        assert!(compliance.true_peak_dbtp + safe <= ATSC_MAX_TRUEPEAK_DBTP);
    }
}
