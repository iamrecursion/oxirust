#![allow(dead_code)]
//! Source media verification and integrity checks for conform workflows.
//!
//! Validates that source media files are intact, meet expected specifications,
//! and are suitable for the target conform before processing begins.

use std::collections::HashMap;

/// The kind of verification check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckKind {
    /// File exists and is readable.
    FileExists,
    /// File size matches expectation.
    FileSize,
    /// Checksum (hash) verification.
    Checksum,
    /// Codec matches expected codec.
    CodecMatch,
    /// Frame rate matches expected rate.
    FrameRateMatch,
    /// Resolution matches expected resolution.
    ResolutionMatch,
    /// Duration is within expected range.
    DurationRange,
    /// Audio channel count matches.
    AudioChannels,
    /// Sample rate matches expected rate.
    SampleRate,
    /// Color space matches expected space.
    ColorSpace,
}

impl std::fmt::Display for CheckKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileExists => write!(f, "file_exists"),
            Self::FileSize => write!(f, "file_size"),
            Self::Checksum => write!(f, "checksum"),
            Self::CodecMatch => write!(f, "codec_match"),
            Self::FrameRateMatch => write!(f, "frame_rate_match"),
            Self::ResolutionMatch => write!(f, "resolution_match"),
            Self::DurationRange => write!(f, "duration_range"),
            Self::AudioChannels => write!(f, "audio_channels"),
            Self::SampleRate => write!(f, "sample_rate"),
            Self::ColorSpace => write!(f, "color_space"),
        }
    }
}

/// Result of a single verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckResult {
    /// Check passed.
    Pass,
    /// Check failed with a reason.
    Fail(String),
    /// Check was skipped (no reference value available).
    Skipped,
    /// Check produced a warning (non-critical deviation).
    Warning(String),
}

impl CheckResult {
    /// Whether the check passed or was skipped (not a failure).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Pass | Self::Skipped | Self::Warning(_))
    }

    /// Whether the check is a hard failure.
    #[must_use]
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }
}

/// Expected properties of a source media file.
#[derive(Debug, Clone, Default)]
pub struct SourceExpectation {
    /// Expected file size in bytes (if known).
    pub expected_size: Option<u64>,
    /// Expected checksum (hex string, if known).
    pub expected_checksum: Option<String>,
    /// Expected video codec name (e.g. "h264", "prores").
    pub expected_codec: Option<String>,
    /// Expected frame rate as a float.
    pub expected_fps: Option<f64>,
    /// Expected width in pixels.
    pub expected_width: Option<u32>,
    /// Expected height in pixels.
    pub expected_height: Option<u32>,
    /// Expected duration in seconds (with tolerance).
    pub expected_duration_secs: Option<f64>,
    /// Tolerance for duration in seconds.
    pub duration_tolerance_secs: f64,
    /// Expected audio channel count.
    pub expected_audio_channels: Option<u32>,
    /// Expected audio sample rate.
    pub expected_sample_rate: Option<u32>,
    /// Expected color space name.
    pub expected_color_space: Option<String>,
}

impl SourceExpectation {
    /// Create a new expectation with default tolerance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            duration_tolerance_secs: 0.5,
            ..Default::default()
        }
    }
}

/// Actual measured properties of a source media file.
#[derive(Debug, Clone, Default)]
pub struct SourceProperties {
    /// File path.
    pub path: String,
    /// Whether the file exists.
    pub exists: bool,
    /// File size in bytes.
    pub size: u64,
    /// Computed checksum (hex string).
    pub checksum: Option<String>,
    /// Video codec name.
    pub codec: Option<String>,
    /// Frame rate.
    pub fps: Option<f64>,
    /// Width in pixels.
    pub width: Option<u32>,
    /// Height in pixels.
    pub height: Option<u32>,
    /// Duration in seconds.
    pub duration_secs: Option<f64>,
    /// Audio channel count.
    pub audio_channels: Option<u32>,
    /// Audio sample rate.
    pub sample_rate: Option<u32>,
    /// Color space name.
    pub color_space: Option<String>,
}

/// Report from verifying a single source file.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Path of the verified file.
    pub path: String,
    /// Individual check results.
    pub checks: HashMap<CheckKind, CheckResult>,
}

impl VerificationReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new(path: String) -> Self {
        Self {
            path,
            checks: HashMap::new(),
        }
    }

    /// Whether all checks passed (no failures).
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.checks.values().all(CheckResult::is_ok)
    }

    /// Count of failed checks.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.checks.values().filter(|r| r.is_fail()).count()
    }

    /// Count of warning checks.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.checks
            .values()
            .filter(|r| matches!(r, CheckResult::Warning(_)))
            .count()
    }

    /// Count of total checks performed (excluding skipped).
    #[must_use]
    pub fn performed_count(&self) -> usize {
        self.checks
            .values()
            .filter(|r| !matches!(r, CheckResult::Skipped))
            .count()
    }
}

/// Verify source properties against expectations.
#[must_use]
pub fn verify_source(props: &SourceProperties, expect: &SourceExpectation) -> VerificationReport {
    let mut report = VerificationReport::new(props.path.clone());

    // File existence
    report.checks.insert(
        CheckKind::FileExists,
        if props.exists {
            CheckResult::Pass
        } else {
            CheckResult::Fail("File does not exist".to_string())
        },
    );

    // File size
    if let Some(expected) = expect.expected_size {
        report.checks.insert(
            CheckKind::FileSize,
            if props.size == expected {
                CheckResult::Pass
            } else {
                CheckResult::Fail(format!(
                    "Size mismatch: expected {expected}, got {}",
                    props.size
                ))
            },
        );
    } else {
        report
            .checks
            .insert(CheckKind::FileSize, CheckResult::Skipped);
    }

    // Checksum
    match (&props.checksum, &expect.expected_checksum) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::Checksum,
                if actual == expected {
                    CheckResult::Pass
                } else {
                    CheckResult::Fail(format!(
                        "Checksum mismatch: expected {expected}, got {actual}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::Checksum, CheckResult::Skipped);
        }
    }

    // Codec
    match (&props.codec, &expect.expected_codec) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::CodecMatch,
                if actual.to_lowercase() == expected.to_lowercase() {
                    CheckResult::Pass
                } else {
                    CheckResult::Fail(format!("Codec mismatch: expected {expected}, got {actual}"))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::CodecMatch, CheckResult::Skipped);
        }
    }

    // Frame rate
    match (props.fps, expect.expected_fps) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::FrameRateMatch,
                if (actual - expected).abs() < 0.01 {
                    CheckResult::Pass
                } else {
                    CheckResult::Fail(format!(
                        "Frame rate mismatch: expected {expected:.3}, got {actual:.3}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::FrameRateMatch, CheckResult::Skipped);
        }
    }

    // Resolution
    match (
        props.width,
        props.height,
        expect.expected_width,
        expect.expected_height,
    ) {
        (Some(w), Some(h), Some(ew), Some(eh)) => {
            report.checks.insert(
                CheckKind::ResolutionMatch,
                if w == ew && h == eh {
                    CheckResult::Pass
                } else {
                    CheckResult::Fail(format!(
                        "Resolution mismatch: expected {ew}x{eh}, got {w}x{h}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::ResolutionMatch, CheckResult::Skipped);
        }
    }

    // Duration
    match (props.duration_secs, expect.expected_duration_secs) {
        (Some(actual), Some(expected)) => {
            let diff = (actual - expected).abs();
            report.checks.insert(
                CheckKind::DurationRange,
                if diff <= expect.duration_tolerance_secs {
                    CheckResult::Pass
                } else {
                    CheckResult::Fail(format!(
                        "Duration mismatch: expected {expected:.2}s +/-{:.2}s, got {actual:.2}s",
                        expect.duration_tolerance_secs
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::DurationRange, CheckResult::Skipped);
        }
    }

    // Audio channels
    match (props.audio_channels, expect.expected_audio_channels) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::AudioChannels,
                if actual == expected {
                    CheckResult::Pass
                } else {
                    CheckResult::Warning(format!(
                        "Audio channels differ: expected {expected}, got {actual}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::AudioChannels, CheckResult::Skipped);
        }
    }

    // Sample rate
    match (props.sample_rate, expect.expected_sample_rate) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::SampleRate,
                if actual == expected {
                    CheckResult::Pass
                } else {
                    CheckResult::Warning(format!(
                        "Sample rate differs: expected {expected}, got {actual}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::SampleRate, CheckResult::Skipped);
        }
    }

    // Color space
    match (&props.color_space, &expect.expected_color_space) {
        (Some(actual), Some(expected)) => {
            report.checks.insert(
                CheckKind::ColorSpace,
                if actual.to_lowercase() == expected.to_lowercase() {
                    CheckResult::Pass
                } else {
                    CheckResult::Warning(format!(
                        "Color space differs: expected {expected}, got {actual}"
                    ))
                },
            );
        }
        _ => {
            report
                .checks
                .insert(CheckKind::ColorSpace, CheckResult::Skipped);
        }
    }

    report
}

/// Verify multiple sources at once, returning a report for each.
#[must_use]
pub fn verify_batch(items: &[(SourceProperties, SourceExpectation)]) -> Vec<VerificationReport> {
    items
        .iter()
        .map(|(props, expect)| verify_source(props, expect))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_props() -> SourceProperties {
        SourceProperties {
            path: "/media/source.mxf".to_string(),
            exists: true,
            size: 1_000_000,
            checksum: Some("abc123".to_string()),
            codec: Some("prores".to_string()),
            fps: Some(24.0),
            width: Some(1920),
            height: Some(1080),
            duration_secs: Some(60.0),
            audio_channels: Some(2),
            sample_rate: Some(48000),
            color_space: Some("bt709".to_string()),
        }
    }

    fn good_expect() -> SourceExpectation {
        SourceExpectation {
            expected_size: Some(1_000_000),
            expected_checksum: Some("abc123".to_string()),
            expected_codec: Some("prores".to_string()),
            expected_fps: Some(24.0),
            expected_width: Some(1920),
            expected_height: Some(1080),
            expected_duration_secs: Some(60.0),
            duration_tolerance_secs: 0.5,
            expected_audio_channels: Some(2),
            expected_sample_rate: Some(48000),
            expected_color_space: Some("bt709".to_string()),
        }
    }

    #[test]
    fn test_all_pass() {
        let report = verify_source(&good_props(), &good_expect());
        assert!(report.all_passed());
        assert_eq!(report.failure_count(), 0);
    }

    #[test]
    fn test_file_not_exist() {
        let mut props = good_props();
        props.exists = false;
        let report = verify_source(&props, &good_expect());
        assert!(!report.all_passed());
        assert!(report.checks[&CheckKind::FileExists].is_fail());
    }

    #[test]
    fn test_size_mismatch() {
        let mut props = good_props();
        props.size = 999;
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::FileSize].is_fail());
    }

    #[test]
    fn test_checksum_mismatch() {
        let mut props = good_props();
        props.checksum = Some("wrong".to_string());
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::Checksum].is_fail());
    }

    #[test]
    fn test_codec_mismatch() {
        let mut props = good_props();
        props.codec = Some("h264".to_string());
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::CodecMatch].is_fail());
    }

    #[test]
    fn test_fps_mismatch() {
        let mut props = good_props();
        props.fps = Some(25.0);
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::FrameRateMatch].is_fail());
    }

    #[test]
    fn test_resolution_mismatch() {
        let mut props = good_props();
        props.width = Some(3840);
        props.height = Some(2160);
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::ResolutionMatch].is_fail());
    }

    #[test]
    fn test_duration_out_of_range() {
        let mut props = good_props();
        props.duration_secs = Some(65.0);
        let report = verify_source(&props, &good_expect());
        assert!(report.checks[&CheckKind::DurationRange].is_fail());
    }

    #[test]
    fn test_audio_channels_warning() {
        let mut props = good_props();
        props.audio_channels = Some(6);
        let report = verify_source(&props, &good_expect());
        let result = &report.checks[&CheckKind::AudioChannels];
        assert!(matches!(result, CheckResult::Warning(_)));
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn test_skipped_when_no_expectation() {
        let props = good_props();
        let expect = SourceExpectation::new();
        let report = verify_source(&props, &expect);
        assert!(report.all_passed());
        assert!(matches!(
            report.checks[&CheckKind::FileSize],
            CheckResult::Skipped
        ));
    }

    #[test]
    fn test_verify_batch() {
        let items = vec![(good_props(), good_expect()), (good_props(), good_expect())];
        let reports = verify_batch(&items);
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(super::VerificationReport::all_passed));
    }

    #[test]
    fn test_check_kind_display() {
        assert_eq!(format!("{}", CheckKind::FileExists), "file_exists");
        assert_eq!(format!("{}", CheckKind::Checksum), "checksum");
    }

    #[test]
    fn test_performed_count() {
        let props = good_props();
        let expect = good_expect();
        let report = verify_source(&props, &expect);
        // All 10 checks should be performed (none skipped)
        assert_eq!(report.performed_count(), 10);
    }
}
