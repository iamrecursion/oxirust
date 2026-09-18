//! Audio issue reporting for post-production quality control.
//!
//! Provides structured reporting of audio quality issues found during
//! analysis, including clipping, level, phase, and DC offset problems.

#![allow(dead_code)]

/// Severity level for an audio issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational — does not require action.
    Info,
    /// Warning — should be reviewed but may be acceptable.
    Warning,
    /// Error — must be fixed before delivery.
    Error,
    /// Critical — render or broadcast will fail.
    Critical,
}

impl IssueSeverity {
    /// Returns `true` if the severity is `Error` or `Critical`.
    #[must_use]
    pub fn is_blocking(self) -> bool {
        matches!(self, IssueSeverity::Error | IssueSeverity::Critical)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            IssueSeverity::Info => "INFO",
            IssueSeverity::Warning => "WARNING",
            IssueSeverity::Error => "ERROR",
            IssueSeverity::Critical => "CRITICAL",
        }
    }
}

/// Category of an audio quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioIssueType {
    /// Sample values exceeded ±1.0 (digital clipping).
    Clipping,
    /// Programme loudness is below required target.
    LowLevel,
    /// Significant DC offset detected in the signal.
    DcOffset,
    /// Out-of-phase or mono-compatibility problem.
    PhaseIssue,
    /// Unexpected silence (drop-out or gap).
    Silence,
    /// Intermittent noise burst or click detected.
    NoiseBurst,
}

impl AudioIssueType {
    /// Returns the default severity for this issue type.
    #[must_use]
    pub fn severity(self) -> IssueSeverity {
        match self {
            AudioIssueType::Clipping => IssueSeverity::Error,
            AudioIssueType::LowLevel => IssueSeverity::Warning,
            AudioIssueType::DcOffset => IssueSeverity::Warning,
            AudioIssueType::PhaseIssue => IssueSeverity::Warning,
            AudioIssueType::Silence => IssueSeverity::Error,
            AudioIssueType::NoiseBurst => IssueSeverity::Info,
        }
    }

    /// Short description of the issue type.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            AudioIssueType::Clipping => "Clipping",
            AudioIssueType::LowLevel => "Low Level",
            AudioIssueType::DcOffset => "DC Offset",
            AudioIssueType::PhaseIssue => "Phase Issue",
            AudioIssueType::Silence => "Silence",
            AudioIssueType::NoiseBurst => "Noise Burst",
        }
    }
}

/// A single detected audio issue with timing and detail information.
#[derive(Debug, Clone)]
pub struct AudioIssue {
    /// Category of the issue.
    pub issue_type: AudioIssueType,
    /// Override severity (falls back to type default if `None`).
    pub severity_override: Option<IssueSeverity>,
    /// Start position in samples.
    pub start_sample: u64,
    /// Duration in samples (`0` = instantaneous).
    pub duration_samples: u64,
    /// Additional context string.
    pub detail: String,
    /// Channel index (0-based); `None` means all channels.
    pub channel: Option<usize>,
}

impl AudioIssue {
    /// Create a new audio issue.
    #[must_use]
    pub fn new(
        issue_type: AudioIssueType,
        start_sample: u64,
        duration_samples: u64,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            issue_type,
            severity_override: None,
            start_sample,
            duration_samples,
            detail: detail.into(),
            channel: None,
        }
    }

    /// Attach a specific channel index.
    #[must_use]
    pub fn with_channel(mut self, ch: usize) -> Self {
        self.channel = Some(ch);
        self
    }

    /// Override the default severity.
    #[must_use]
    pub fn with_severity(mut self, sev: IssueSeverity) -> Self {
        self.severity_override = Some(sev);
        self
    }

    /// Effective severity (override wins over type default).
    #[must_use]
    pub fn severity(&self) -> IssueSeverity {
        self.severity_override
            .unwrap_or_else(|| self.issue_type.severity())
    }

    /// Human-readable description combining type name, severity, and detail.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "[{}] {} at sample {}: {}",
            self.severity().label(),
            self.issue_type.name(),
            self.start_sample,
            self.detail
        )
    }

    /// Returns `true` if this issue is blocking delivery.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.severity().is_blocking()
    }
}

/// Aggregate report of all audio issues found during analysis.
#[derive(Debug, Default, Clone)]
pub struct AudioReport {
    issues: Vec<AudioIssue>,
    /// Label identifying the source (file name, clip ID, etc.).
    pub source_label: String,
    /// Sample rate used during analysis.
    pub sample_rate: u32,
}

impl AudioReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new(source_label: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            issues: Vec::new(),
            source_label: source_label.into(),
            sample_rate,
        }
    }

    /// Append an issue to the report.
    pub fn add_issue(&mut self, issue: AudioIssue) {
        self.issues.push(issue);
    }

    /// Total number of recorded issues.
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    /// Returns `true` if any issue has blocking severity.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.issues.iter().any(|i| i.is_critical())
    }

    /// Iterate over all issues.
    pub fn issues(&self) -> impl Iterator<Item = &AudioIssue> {
        self.issues.iter()
    }

    /// Filter issues by type.
    pub fn issues_of_type(&self, t: AudioIssueType) -> impl Iterator<Item = &AudioIssue> {
        self.issues.iter().filter(move |i| i.issue_type == t)
    }

    /// Count issues at or above a given severity.
    #[must_use]
    pub fn count_at_severity(&self, min: IssueSeverity) -> usize {
        self.issues.iter().filter(|i| i.severity() >= min).count()
    }

    /// Convert time in seconds to sample position.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn seconds_to_sample(&self, secs: f64) -> u64 {
        (secs * self.sample_rate as f64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::Error);
        assert!(IssueSeverity::Error > IssueSeverity::Warning);
        assert!(IssueSeverity::Warning > IssueSeverity::Info);
    }

    #[test]
    fn test_severity_is_blocking() {
        assert!(IssueSeverity::Error.is_blocking());
        assert!(IssueSeverity::Critical.is_blocking());
        assert!(!IssueSeverity::Warning.is_blocking());
        assert!(!IssueSeverity::Info.is_blocking());
    }

    #[test]
    fn test_issue_type_severity_defaults() {
        assert_eq!(AudioIssueType::Clipping.severity(), IssueSeverity::Error);
        assert_eq!(AudioIssueType::LowLevel.severity(), IssueSeverity::Warning);
        assert_eq!(AudioIssueType::DcOffset.severity(), IssueSeverity::Warning);
        assert_eq!(
            AudioIssueType::PhaseIssue.severity(),
            IssueSeverity::Warning
        );
        assert_eq!(AudioIssueType::Silence.severity(), IssueSeverity::Error);
        assert_eq!(AudioIssueType::NoiseBurst.severity(), IssueSeverity::Info);
    }

    #[test]
    fn test_issue_type_names() {
        assert_eq!(AudioIssueType::Clipping.name(), "Clipping");
        assert_eq!(AudioIssueType::DcOffset.name(), "DC Offset");
    }

    #[test]
    fn test_audio_issue_description() {
        let issue = AudioIssue::new(AudioIssueType::Clipping, 1000, 5, "peak = 1.2");
        let desc = issue.description();
        assert!(desc.contains("ERROR"));
        assert!(desc.contains("Clipping"));
        assert!(desc.contains("1000"));
        assert!(desc.contains("peak = 1.2"));
    }

    #[test]
    fn test_audio_issue_severity_override() {
        let issue = AudioIssue::new(AudioIssueType::Clipping, 0, 0, "test")
            .with_severity(IssueSeverity::Info);
        assert_eq!(issue.severity(), IssueSeverity::Info);
        assert!(!issue.is_critical());
    }

    #[test]
    fn test_audio_issue_with_channel() {
        let issue =
            AudioIssue::new(AudioIssueType::PhaseIssue, 0, 100, "phase flip").with_channel(1);
        assert_eq!(issue.channel, Some(1));
    }

    #[test]
    fn test_report_add_and_count() {
        let mut report = AudioReport::new("test.wav", 48000);
        assert_eq!(report.issue_count(), 0);
        report.add_issue(AudioIssue::new(AudioIssueType::Clipping, 0, 1, "clip"));
        report.add_issue(AudioIssue::new(AudioIssueType::LowLevel, 500, 0, "low"));
        assert_eq!(report.issue_count(), 2);
    }

    #[test]
    fn test_report_has_critical() {
        let mut report = AudioReport::new("test.wav", 48000);
        report.add_issue(AudioIssue::new(AudioIssueType::LowLevel, 0, 0, "low"));
        assert!(!report.has_critical());
        report.add_issue(AudioIssue::new(AudioIssueType::Clipping, 100, 1, "clip"));
        assert!(report.has_critical());
    }

    #[test]
    fn test_report_issues_of_type() {
        let mut report = AudioReport::new("clip.wav", 48000);
        report.add_issue(AudioIssue::new(AudioIssueType::Clipping, 0, 1, "a"));
        report.add_issue(AudioIssue::new(AudioIssueType::LowLevel, 10, 0, "b"));
        report.add_issue(AudioIssue::new(AudioIssueType::Clipping, 20, 1, "c"));
        let clips: Vec<_> = report.issues_of_type(AudioIssueType::Clipping).collect();
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_report_count_at_severity() {
        let mut report = AudioReport::new("clip.wav", 48000);
        report.add_issue(AudioIssue::new(AudioIssueType::NoiseBurst, 0, 0, "info"));
        report.add_issue(AudioIssue::new(AudioIssueType::LowLevel, 0, 0, "warn"));
        report.add_issue(AudioIssue::new(AudioIssueType::Clipping, 0, 0, "err"));
        assert_eq!(report.count_at_severity(IssueSeverity::Warning), 2);
        assert_eq!(report.count_at_severity(IssueSeverity::Error), 1);
        assert_eq!(report.count_at_severity(IssueSeverity::Critical), 0);
    }

    #[test]
    fn test_seconds_to_sample() {
        let report = AudioReport::new("x.wav", 48000);
        assert_eq!(report.seconds_to_sample(1.0), 48000);
        assert_eq!(report.seconds_to_sample(0.5), 24000);
    }

    #[test]
    fn test_report_default() {
        let report = AudioReport::default();
        assert_eq!(report.issue_count(), 0);
        assert!(!report.has_critical());
    }

    #[test]
    fn test_report_source_label() {
        let report = AudioReport::new("my_mix.wav", 44100);
        assert_eq!(report.source_label, "my_mix.wav");
        assert_eq!(report.sample_rate, 44100);
    }
}
