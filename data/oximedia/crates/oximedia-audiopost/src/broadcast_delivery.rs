//! Broadcast delivery specifications and validation.
//!
//! Provides standard broadcast loudness specifications (ATSC A/85, EBU R128,
//! AES streaming, Netflix, YouTube) and validation utilities for audio delivery.

#![allow(dead_code)]

/// Broadcast delivery specification.
#[derive(Debug, Clone)]
pub struct BroadcastSpec {
    /// Specification name (e.g. "EBU R128").
    pub name: String,
    /// Target integrated loudness in LUFS.
    pub loudness_lufs: f64,
    /// Maximum true peak in dBFS.
    pub true_peak_dbfs: f64,
    /// Required sample rate in Hz.
    pub sample_rate: u32,
    /// Required bit depth.
    pub bit_depth: u16,
    /// Required number of audio channels.
    pub channels: u8,
    /// Required codec (e.g. "PCM", "AAC", "AC-3").
    pub codec: String,
}

impl BroadcastSpec {
    /// Create a new broadcast spec.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        loudness_lufs: f64,
        true_peak_dbfs: f64,
        sample_rate: u32,
        bit_depth: u16,
        channels: u8,
        codec: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            loudness_lufs,
            true_peak_dbfs,
            sample_rate,
            bit_depth,
            channels,
            codec: codec.into(),
        }
    }

    /// ATSC A/85 specification (US broadcast standard, -24 LUFS).
    #[must_use]
    pub fn atsc_a85() -> Self {
        Self::new("ATSC A/85", -24.0, -2.0, 48000, 24, 2, "AC-3")
    }

    /// EBU R128 specification (European broadcast standard, -23 LUFS).
    #[must_use]
    pub fn ebu_r128() -> Self {
        Self::new("EBU R128", -23.0, -1.0, 48000, 24, 2, "PCM")
    }

    /// AES streaming specification (-18 LUFS).
    #[must_use]
    pub fn aes_streaming() -> Self {
        Self::new("AES Streaming", -18.0, -1.0, 48000, 16, 2, "AAC")
    }

    /// Netflix specification (-27 LUFS).
    #[must_use]
    pub fn netflix() -> Self {
        Self::new("Netflix", -27.0, -2.0, 48000, 24, 2, "AAC")
    }

    /// YouTube specification (-14 LUFS).
    #[must_use]
    pub fn youtube() -> Self {
        Self::new("YouTube", -14.0, -1.0, 44100, 16, 2, "AAC")
    }

    /// Loudness tolerance in LUFS (±1 LUFS for most specs).
    #[must_use]
    pub fn loudness_tolerance(&self) -> f64 {
        1.0
    }
}

/// Type of delivery issue found during validation.
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    /// Integrated loudness is below the target.
    LoudnessTooLow,
    /// Integrated loudness is above the target.
    LoudnessTooHigh,
    /// True peak exceeds the maximum allowed level.
    TruePeakExceeded,
    /// Sample rate does not match specification.
    SampleRateMismatch,
}

/// A delivery validation issue.
#[derive(Debug, Clone)]
pub struct DeliveryIssue {
    /// Type of issue.
    pub issue_type: IssueType,
    /// The measured value.
    pub measured: f64,
    /// The allowed/target value.
    pub allowed: f64,
    /// Difference between measured and allowed.
    pub difference: f64,
}

impl DeliveryIssue {
    /// Create a new delivery issue.
    #[must_use]
    pub fn new(issue_type: IssueType, measured: f64, allowed: f64) -> Self {
        Self {
            issue_type,
            measured,
            allowed,
            difference: measured - allowed,
        }
    }
}

/// Validates audio delivery against broadcast specifications.
pub struct DeliveryValidator;

impl DeliveryValidator {
    /// Validate measured audio levels against a broadcast specification.
    ///
    /// Returns a list of issues found. Empty list means the audio passes the spec.
    #[must_use]
    pub fn validate(
        measured_lufs: f64,
        measured_peak: f64,
        spec: &BroadcastSpec,
    ) -> Vec<DeliveryIssue> {
        let mut issues = Vec::new();

        let tolerance = spec.loudness_tolerance();

        // Check loudness (with tolerance window)
        if measured_lufs < spec.loudness_lufs - tolerance {
            issues.push(DeliveryIssue::new(
                IssueType::LoudnessTooLow,
                measured_lufs,
                spec.loudness_lufs,
            ));
        } else if measured_lufs > spec.loudness_lufs + tolerance {
            issues.push(DeliveryIssue::new(
                IssueType::LoudnessTooHigh,
                measured_lufs,
                spec.loudness_lufs,
            ));
        }

        // Check true peak
        if measured_peak > spec.true_peak_dbfs {
            issues.push(DeliveryIssue::new(
                IssueType::TruePeakExceeded,
                measured_peak,
                spec.true_peak_dbfs,
            ));
        }

        issues
    }
}

/// Complete delivery validation report.
#[derive(Debug, Clone)]
pub struct DeliveryReport {
    /// Specification name validated against.
    pub spec_name: String,
    /// Whether the audio passed all checks.
    pub passed: bool,
    /// Issues found during validation.
    pub issues: Vec<DeliveryIssue>,
    /// Corrective gain in dB to reach the target loudness.
    pub corrective_gain_db: f64,
}

impl DeliveryReport {
    /// Generate a delivery report for the given measurements against a spec.
    #[must_use]
    pub fn generate(measured_lufs: f64, measured_peak: f64, spec: &BroadcastSpec) -> Self {
        let issues = DeliveryValidator::validate(measured_lufs, measured_peak, spec);
        let passed = issues.is_empty();

        // Corrective gain needed to hit target loudness
        let corrective_gain_db = spec.loudness_lufs - measured_lufs;

        DeliveryReport {
            spec_name: spec.name.clone(),
            passed,
            issues,
            corrective_gain_db,
        }
    }

    /// Get all issues of a specific type.
    #[must_use]
    pub fn issues_of_type(&self, issue_type: &IssueType) -> Vec<&DeliveryIssue> {
        self.issues
            .iter()
            .filter(|i| &i.issue_type == issue_type)
            .collect()
    }

    /// Whether the audio needs to be louder.
    #[must_use]
    pub fn needs_gain_increase(&self) -> bool {
        self.corrective_gain_db > 0.0
    }

    /// Whether the audio needs to be quieter.
    #[must_use]
    pub fn needs_gain_reduction(&self) -> bool {
        self.corrective_gain_db < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atsc_a85_spec() {
        let spec = BroadcastSpec::atsc_a85();
        assert_eq!(spec.name, "ATSC A/85");
        assert_eq!(spec.loudness_lufs, -24.0);
        assert_eq!(spec.true_peak_dbfs, -2.0);
        assert_eq!(spec.sample_rate, 48000);
    }

    #[test]
    fn test_ebu_r128_spec() {
        let spec = BroadcastSpec::ebu_r128();
        assert_eq!(spec.loudness_lufs, -23.0);
        assert_eq!(spec.true_peak_dbfs, -1.0);
    }

    #[test]
    fn test_aes_streaming_spec() {
        let spec = BroadcastSpec::aes_streaming();
        assert_eq!(spec.loudness_lufs, -18.0);
        assert_eq!(spec.codec, "AAC");
    }

    #[test]
    fn test_netflix_spec() {
        let spec = BroadcastSpec::netflix();
        assert_eq!(spec.loudness_lufs, -27.0);
    }

    #[test]
    fn test_youtube_spec() {
        let spec = BroadcastSpec::youtube();
        assert_eq!(spec.loudness_lufs, -14.0);
        assert_eq!(spec.sample_rate, 44100);
    }

    #[test]
    fn test_validate_passing() {
        let spec = BroadcastSpec::ebu_r128();
        let issues = DeliveryValidator::validate(-23.0, -2.0, &spec);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_loudness_too_low() {
        let spec = BroadcastSpec::ebu_r128(); // -23 LUFS target
        let issues = DeliveryValidator::validate(-26.0, -3.0, &spec);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::LoudnessTooLow);
        assert_eq!(issues[0].measured, -26.0);
        assert_eq!(issues[0].allowed, -23.0);
    }

    #[test]
    fn test_validate_loudness_too_high() {
        let spec = BroadcastSpec::ebu_r128(); // -23 LUFS target
        let issues = DeliveryValidator::validate(-20.0, -3.0, &spec);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::LoudnessTooHigh);
    }

    #[test]
    fn test_validate_true_peak_exceeded() {
        let spec = BroadcastSpec::ebu_r128(); // -1 dBFS true peak limit
        let issues = DeliveryValidator::validate(-23.0, 0.5, &spec);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::TruePeakExceeded);
        assert_eq!(issues[0].measured, 0.5);
        assert_eq!(issues[0].allowed, -1.0);
    }

    #[test]
    fn test_validate_multiple_issues() {
        let spec = BroadcastSpec::ebu_r128();
        let issues = DeliveryValidator::validate(-28.0, 0.0, &spec);
        assert_eq!(issues.len(), 2); // too quiet AND true peak exceeded
    }

    #[test]
    fn test_delivery_report_passed() {
        let spec = BroadcastSpec::atsc_a85();
        let report = DeliveryReport::generate(-24.0, -3.0, &spec);
        assert!(report.passed);
        assert!(report.issues.is_empty());
        assert_eq!(report.spec_name, "ATSC A/85");
    }

    #[test]
    fn test_delivery_report_corrective_gain_increase() {
        let spec = BroadcastSpec::ebu_r128(); // -23 LUFS
        let report = DeliveryReport::generate(-26.0, -5.0, &spec);
        assert!(report.needs_gain_increase());
        assert!((report.corrective_gain_db - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_delivery_report_corrective_gain_reduction() {
        let spec = BroadcastSpec::ebu_r128(); // -23 LUFS
        let report = DeliveryReport::generate(-20.0, -5.0, &spec);
        assert!(report.needs_gain_reduction());
        assert!((report.corrective_gain_db - (-3.0)).abs() < 0.001);
    }

    #[test]
    fn test_delivery_report_within_tolerance() {
        // EBU R128 allows ±1 LUFS tolerance
        let spec = BroadcastSpec::ebu_r128();
        let report = DeliveryReport::generate(-23.5, -2.0, &spec); // within ±1 LUFS
        assert!(report.passed);
    }
}
