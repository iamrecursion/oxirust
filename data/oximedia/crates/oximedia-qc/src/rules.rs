//! Quality control rule definitions and trait system.
//!
//! This module provides the foundational trait [`QcRule`] and related types
//! for implementing quality control checks. Rules can validate video quality,
//! audio quality, container integrity, and compliance with delivery specifications.

use oximedia_core::OxiResult;
use std::fmt;

/// Severity level for QC findings.
///
/// Indicates the importance and impact of a quality control issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub enum Severity {
    /// Informational message (no action required).
    Info,
    /// Warning - issue should be reviewed but may be acceptable.
    Warning,
    /// Error - issue that prevents meeting quality standards.
    Error,
    /// Critical - serious issue that makes content unusable.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Quality control check result.
///
/// Represents the outcome of a single QC rule check, including
/// pass/fail status, severity, and detailed messages.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct CheckResult {
    /// Name of the rule that was checked.
    pub rule_name: String,

    /// Whether the check passed.
    pub passed: bool,

    /// Severity level if the check failed.
    pub severity: Severity,

    /// Detailed message about the check result.
    pub message: String,

    /// Optional recommendation for fixing the issue.
    pub recommendation: Option<String>,

    /// Stream index this check applies to (if applicable).
    pub stream_index: Option<usize>,

    /// Timestamp where the issue was detected (if applicable, in seconds).
    pub timestamp: Option<f64>,

    /// Additional metadata about the check.
    #[cfg(feature = "json")]
    #[cfg_attr(feature = "json", serde(skip_serializing_if = "Option::is_none"))]
    pub metadata: Option<serde_json::Value>,

    /// Additional metadata about the check (when JSON feature is disabled).
    #[cfg(not(feature = "json"))]
    #[cfg_attr(feature = "json", serde(skip))]
    pub metadata: Option<String>,
}

impl CheckResult {
    /// Creates a passing check result.
    #[must_use]
    pub fn pass(rule_name: impl Into<String>) -> Self {
        Self {
            rule_name: rule_name.into(),
            passed: true,
            severity: Severity::Info,
            message: "Check passed".to_string(),
            recommendation: None,
            stream_index: None,
            timestamp: None,
            metadata: None,
        }
    }

    /// Creates a failing check result.
    #[must_use]
    pub fn fail(
        rule_name: impl Into<String>,
        severity: Severity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            rule_name: rule_name.into(),
            passed: false,
            severity,
            message: message.into(),
            recommendation: None,
            stream_index: None,
            timestamp: None,
            metadata: None,
        }
    }

    /// Sets the recommendation for this check result.
    #[must_use]
    pub fn with_recommendation(mut self, recommendation: impl Into<String>) -> Self {
        self.recommendation = Some(recommendation.into());
        self
    }

    /// Sets the stream index for this check result.
    #[must_use]
    pub const fn with_stream(mut self, stream_index: usize) -> Self {
        self.stream_index = Some(stream_index);
        self
    }

    /// Sets the timestamp for this check result.
    #[must_use]
    pub const fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Sets metadata for this check result (JSON value).
    #[cfg(feature = "json")]
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Sets metadata for this check result (string).
    #[cfg(not(feature = "json"))]
    #[must_use]
    pub fn with_metadata(mut self, metadata: impl Into<String>) -> Self {
        self.metadata = Some(metadata.into());
        self
    }
}

/// Category of quality control rule.
///
/// Used to classify rules by the aspect of media they validate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub enum RuleCategory {
    /// Video quality checks.
    Video,
    /// Audio quality checks.
    Audio,
    /// Container/format checks.
    Container,
    /// Compliance with delivery specifications.
    Compliance,
}

impl fmt::Display for RuleCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::Container => write!(f, "container"),
            Self::Compliance => write!(f, "compliance"),
        }
    }
}

/// Context passed to QC rules during validation.
///
/// Contains file path and stream information needed for validation.
#[derive(Clone, Debug)]
pub struct QcContext {
    /// Path to the file being validated.
    pub file_path: String,

    /// Stream information from the container.
    pub streams: Vec<oximedia_container::StreamInfo>,

    /// Total duration of the file in seconds (if known).
    pub duration: Option<f64>,

    /// Overall file bitrate in bits per second, derived from file size / duration.
    pub file_bitrate: Option<u64>,
}

impl QcContext {
    /// Creates a new QC context.
    #[must_use]
    pub fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            streams: Vec::new(),
            duration: None,
            file_bitrate: None,
        }
    }

    /// Adds stream information to the context.
    pub fn add_stream(&mut self, stream: oximedia_container::StreamInfo) {
        self.streams.push(stream);
    }

    /// Sets the total duration.
    pub fn set_duration(&mut self, duration: f64) {
        self.duration = Some(duration);
    }

    /// Returns video streams.
    #[must_use]
    pub fn video_streams(&self) -> Vec<&oximedia_container::StreamInfo> {
        self.streams.iter().filter(|s| s.is_video()).collect()
    }

    /// Returns audio streams.
    #[must_use]
    pub fn audio_streams(&self) -> Vec<&oximedia_container::StreamInfo> {
        self.streams.iter().filter(|s| s.is_audio()).collect()
    }
}

/// Trait for quality control rules.
///
/// Implement this trait to create custom QC rules that can be used
/// with the [`crate::QualityControl`] system.
///
/// # Examples
///
/// ```ignore
/// use oximedia_qc::rules::{QcRule, CheckResult, QcContext, RuleCategory};
/// use oximedia_core::OxiResult;
///
/// struct MyCustomRule;
///
/// impl QcRule for MyCustomRule {
///     fn name(&self) -> &str {
///         "custom_check"
///     }
///
///     fn category(&self) -> RuleCategory {
///         RuleCategory::Video
///     }
///
///     fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
///         let mut results = Vec::new();
///
///         for stream in context.video_streams() {
///             let result = CheckResult::pass(self.name());
///             results.push(result);
///         }
///
///         Ok(results)
///     }
/// }
/// ```
pub trait QcRule: Send + Sync {
    /// Returns the name of this rule.
    fn name(&self) -> &str;

    /// Returns the category of this rule.
    fn category(&self) -> RuleCategory;

    /// Returns a description of what this rule checks.
    fn description(&self) -> &str {
        ""
    }

    /// Performs the quality control check.
    ///
    /// # Errors
    ///
    /// Returns an error if the check cannot be performed (e.g., file I/O error).
    /// Check failures should be returned as `CheckResult` with `passed = false`.
    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>>;

    /// Returns whether this rule is applicable to the given context.
    ///
    /// This can be used to skip rules that don't apply to certain files.
    fn is_applicable(&self, _context: &QcContext) -> bool {
        true
    }
}

/// Threshold configuration for quality checks.
///
/// Many QC rules require thresholds (e.g., minimum bitrate, maximum silence duration).
/// This struct provides a way to configure these thresholds.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct Thresholds {
    /// Minimum video bitrate in bits per second.
    pub min_video_bitrate: Option<u64>,

    /// Maximum video bitrate in bits per second.
    pub max_video_bitrate: Option<u64>,

    /// Minimum audio bitrate in bits per second.
    pub min_audio_bitrate: Option<u64>,

    /// Maximum audio bitrate in bits per second.
    pub max_audio_bitrate: Option<u64>,

    /// Maximum silence duration in seconds.
    pub max_silence_duration: Option<f64>,

    /// Maximum black frame duration in seconds.
    pub max_black_duration: Option<f64>,

    /// Loudness target in LUFS (default: -23.0 for broadcast).
    pub loudness_target: Option<f64>,

    /// Loudness tolerance in LU (default: ±1.0).
    pub loudness_tolerance: Option<f64>,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_video_bitrate: None,
            max_video_bitrate: None,
            min_audio_bitrate: None,
            max_audio_bitrate: None,
            max_silence_duration: Some(2.0),
            max_black_duration: Some(2.0),
            loudness_target: Some(-23.0),
            loudness_tolerance: Some(1.0),
        }
    }
}

impl Thresholds {
    /// Creates a new threshold configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the minimum video bitrate.
    #[must_use]
    pub const fn with_min_video_bitrate(mut self, bitrate: u64) -> Self {
        self.min_video_bitrate = Some(bitrate);
        self
    }

    /// Sets the maximum video bitrate.
    #[must_use]
    pub const fn with_max_video_bitrate(mut self, bitrate: u64) -> Self {
        self.max_video_bitrate = Some(bitrate);
        self
    }

    /// Sets the loudness target in LUFS.
    #[must_use]
    pub const fn with_loudness_target(mut self, target: f64) -> Self {
        self.loudness_target = Some(target);
        self
    }
}

/// Threshold-based severity classification for numeric QC measurements.
///
/// Allows mapping measured values to [`Severity`] levels based on configurable
/// boundary ranges. Values within the acceptable range map to [`Severity::Info`];
/// successive boundaries escalate through Warning, Error, and Critical.
///
/// # Example
///
/// ```
/// use oximedia_qc::rules::{SeverityClassifier, Severity};
///
/// // Loudness check: target -23 LUFS ± 1 LU (warning), ± 3 LU (error), ± 6 LU (critical)
/// let classifier = SeverityClassifier::new(-24.0, -22.0)
///     .with_warning_range(-26.0, -20.0)
///     .with_error_range(-29.0, -17.0)
///     .with_critical_range(-35.0, -10.0);
///
/// assert_eq!(classifier.classify(-23.0), Severity::Info);
/// assert_eq!(classifier.classify(-25.0), Severity::Warning);
/// assert_eq!(classifier.classify(-28.0), Severity::Error);
/// assert_eq!(classifier.classify(-34.0), Severity::Critical);
/// // Values outside even the critical range map to Critical
/// assert_eq!(classifier.classify(-40.0), Severity::Critical);
/// ```
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
pub struct SeverityClassifier {
    /// Inclusive lower bound of the Info (acceptable) range.
    pub info_min: f64,
    /// Inclusive upper bound of the Info (acceptable) range.
    pub info_max: f64,
    /// Inclusive lower bound of the Warning range (wider than Info).
    pub warning_min: Option<f64>,
    /// Inclusive upper bound of the Warning range (wider than Info).
    pub warning_max: Option<f64>,
    /// Inclusive lower bound of the Error range (wider than Warning).
    pub error_min: Option<f64>,
    /// Inclusive upper bound of the Error range (wider than Warning).
    pub error_max: Option<f64>,
    /// Inclusive lower bound of the Critical range (wider than Error).
    pub critical_min: Option<f64>,
    /// Inclusive upper bound of the Critical range (wider than Error).
    pub critical_max: Option<f64>,
}

impl SeverityClassifier {
    /// Creates a classifier with only an Info (acceptable) band.
    ///
    /// Values outside `[info_min, info_max]` immediately escalate to Critical
    /// unless additional bands are configured via the builder methods.
    #[must_use]
    pub const fn new(info_min: f64, info_max: f64) -> Self {
        Self {
            info_min,
            info_max,
            warning_min: None,
            warning_max: None,
            error_min: None,
            error_max: None,
            critical_min: None,
            critical_max: None,
        }
    }

    /// Sets the Warning band boundaries.
    #[must_use]
    pub const fn with_warning_range(mut self, min: f64, max: f64) -> Self {
        self.warning_min = Some(min);
        self.warning_max = Some(max);
        self
    }

    /// Sets the Error band boundaries.
    #[must_use]
    pub const fn with_error_range(mut self, min: f64, max: f64) -> Self {
        self.error_min = Some(min);
        self.error_max = Some(max);
        self
    }

    /// Sets the Critical band boundaries.
    #[must_use]
    pub const fn with_critical_range(mut self, min: f64, max: f64) -> Self {
        self.critical_min = Some(min);
        self.critical_max = Some(max);
        self
    }

    /// Classifies a measured value into a [`Severity`] level.
    ///
    /// Evaluation is from tightest (Info) to widest (Critical) band. Values
    /// that fall outside all configured bands return [`Severity::Critical`].
    #[must_use]
    pub fn classify(&self, value: f64) -> Severity {
        // Inside the acceptable (Info) band?
        if value >= self.info_min && value <= self.info_max {
            return Severity::Info;
        }

        // Inside Warning band (if configured)?
        if let (Some(wmin), Some(wmax)) = (self.warning_min, self.warning_max) {
            if value >= wmin && value <= wmax {
                return Severity::Warning;
            }
        }

        // Inside Error band (if configured)?
        if let (Some(emin), Some(emax)) = (self.error_min, self.error_max) {
            if value >= emin && value <= emax {
                return Severity::Error;
            }
        }

        // Inside Critical band (if configured)?
        if let (Some(cmin), Some(cmax)) = (self.critical_min, self.critical_max) {
            if value >= cmin && value <= cmax {
                return Severity::Critical;
            }
        }

        // Outside all configured bands — always Critical.
        Severity::Critical
    }

    /// Convenience: creates a standard loudness classifier for EBU R128 (-23 LUFS).
    ///
    /// - Info:     −24.0 … −22.0 LUFS  (±1 LU)
    /// - Warning:  −26.0 … −20.0 LUFS  (±3 LU)
    /// - Error:    −29.0 … −17.0 LUFS  (±6 LU)
    /// - Critical: −35.0 … −10.0 LUFS  (±12 LU)
    #[must_use]
    pub const fn ebu_r128_loudness() -> Self {
        Self::new(-24.0, -22.0)
            .with_warning_range(-26.0, -20.0)
            .with_error_range(-29.0, -17.0)
            .with_critical_range(-35.0, -10.0)
    }

    /// Convenience: creates a luma-range classifier for broadcast-legal 8-bit video.
    ///
    /// Luma must stay within 16–235. Ranges:
    /// - Info:     16 … 235  (legal)
    /// - Warning:   8 … 247  (near-legal)
    /// - Error:     4 … 251
    /// - Critical:  0 … 255
    #[must_use]
    pub const fn broadcast_luma_8bit() -> Self {
        Self::new(16.0, 235.0)
            .with_warning_range(8.0, 247.0)
            .with_error_range(4.0, 251.0)
            .with_critical_range(0.0, 255.0)
    }

    /// Convenience: creates a chroma-range classifier for broadcast-legal 8-bit video.
    ///
    /// Chroma (Cb/Cr) must stay within 16–240. Ranges:
    /// - Info:     16 … 240  (legal)
    /// - Warning:   8 … 248  (near-legal)
    /// - Error:     4 … 252
    /// - Critical:  0 … 255
    #[must_use]
    pub const fn broadcast_chroma_8bit() -> Self {
        Self::new(16.0, 240.0)
            .with_warning_range(8.0, 248.0)
            .with_error_range(4.0, 252.0)
            .with_critical_range(0.0, 255.0)
    }
}

#[cfg(test)]
mod severity_classifier_tests {
    use super::*;

    #[test]
    fn test_info_range_included() {
        let c = SeverityClassifier::new(16.0, 235.0);
        assert_eq!(c.classify(16.0), Severity::Info);
        assert_eq!(c.classify(235.0), Severity::Info);
        assert_eq!(c.classify(128.0), Severity::Info);
    }

    #[test]
    fn test_warning_range() {
        let c = SeverityClassifier::new(16.0, 235.0).with_warning_range(8.0, 247.0);
        assert_eq!(c.classify(10.0), Severity::Warning);
        assert_eq!(c.classify(240.0), Severity::Warning);
    }

    #[test]
    fn test_error_range() {
        let c = SeverityClassifier::new(16.0, 235.0)
            .with_warning_range(8.0, 247.0)
            .with_error_range(4.0, 251.0);
        assert_eq!(c.classify(5.0), Severity::Error);
        assert_eq!(c.classify(250.0), Severity::Error);
    }

    #[test]
    fn test_critical_range() {
        let c = SeverityClassifier::ebu_r128_loudness();
        assert_eq!(c.classify(-34.0), Severity::Critical);
        assert_eq!(c.classify(-40.0), Severity::Critical); // outside all bands
    }

    #[test]
    fn test_outside_all_bands_is_critical() {
        let c = SeverityClassifier::new(16.0, 235.0).with_warning_range(8.0, 247.0);
        assert_eq!(c.classify(0.0), Severity::Critical);
        assert_eq!(c.classify(255.0), Severity::Critical);
    }

    #[test]
    fn test_ebu_r128_target() {
        let c = SeverityClassifier::ebu_r128_loudness();
        assert_eq!(c.classify(-23.0), Severity::Info);
        assert_eq!(c.classify(-25.0), Severity::Warning);
        assert_eq!(c.classify(-28.0), Severity::Error);
    }

    #[test]
    fn test_broadcast_luma_preset() {
        let c = SeverityClassifier::broadcast_luma_8bit();
        assert_eq!(c.classify(128.0), Severity::Info);
        assert_eq!(c.classify(10.0), Severity::Warning);
        assert_eq!(c.classify(253.0), Severity::Critical);
    }
}
