//! Media authenticity verification — detect metadata inconsistencies, timestamp
//! anomalies, and suspicious compression patterns.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single issue found during an authenticity check.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthenticityIssue {
    /// A field in the file's metadata contradicts another field.
    MetadataInconsistency(String),
    /// A timestamp value is anomalous or out of expected order.
    TimestampAnomaly(String),
    /// Unexpected or suspicious compression artefact detected.
    CompressionArtifact(String),
    /// Bitrate is inconsistent with declared encoding parameters.
    BitrateInconsistency(String),
}

impl AuthenticityIssue {
    /// Return a human-readable description of the issue.
    pub fn description(&self) -> &str {
        match self {
            Self::MetadataInconsistency(s)
            | Self::TimestampAnomaly(s)
            | Self::CompressionArtifact(s)
            | Self::BitrateInconsistency(s) => s.as_str(),
        }
    }
}

/// A specific check that the `AuthenticityChecker` can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthenticityCheck {
    /// Verify that metadata fields are mutually consistent.
    MetadataConsistency,
    /// Verify that timestamps are monotonically increasing and reasonable.
    TimestampAnalysis,
    /// Detect re-encoding or unexpected compression history in the stream.
    CompressionHistory,
    /// Detect resolution changes that indicate covert re-encoding.
    ResolutionHistory,
}

/// Summary of an authenticity analysis run.
#[derive(Debug, Clone)]
pub struct AuthenticityReport {
    /// Name or path of the file that was analysed.
    pub filename: String,
    /// `true` if the file passes all enabled checks.
    pub is_authentic: bool,
    /// Overall confidence that the file is authentic (0.0–1.0).
    pub confidence: f64,
    /// Ordered list of issues found during analysis.
    pub issues: Vec<AuthenticityIssue>,
}

impl AuthenticityReport {
    /// Create a new report that initially considers the file authentic.
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            is_authentic: true,
            confidence: 1.0,
            issues: Vec::new(),
        }
    }

    /// Record an issue and reduce the confidence score accordingly.
    pub fn add_issue(&mut self, issue: AuthenticityIssue) {
        self.issues.push(issue);
        // Each issue reduces confidence; cap at 0.
        self.confidence = (self.confidence - 0.15).max(0.0);
        self.is_authentic = self.confidence > 0.5;
    }
}

/// Configurable checker that runs a selected subset of authenticity tests.
#[derive(Debug, Clone)]
pub struct AuthenticityChecker {
    /// Checks that are enabled for this instance.
    pub checks: Vec<AuthenticityCheck>,
}

impl AuthenticityChecker {
    /// Create a checker with all available checks enabled.
    pub fn all_checks() -> Self {
        Self {
            checks: vec![
                AuthenticityCheck::MetadataConsistency,
                AuthenticityCheck::TimestampAnalysis,
                AuthenticityCheck::CompressionHistory,
                AuthenticityCheck::ResolutionHistory,
            ],
        }
    }

    /// Create an empty checker (no checks enabled).
    pub fn none() -> Self {
        Self { checks: Vec::new() }
    }
}

impl Default for AuthenticityChecker {
    fn default() -> Self {
        Self::all_checks()
    }
}

/// Verify that creation, modification, and optional encoding dates are consistent.
///
/// - `modification_date` must be >= `creation_date`.
/// - `encoding_date`, if provided, must be >= `creation_date`.
pub fn check_metadata_consistency(
    creation_date: u64,
    modification_date: u64,
    encoding_date: Option<u64>,
) -> Vec<AuthenticityIssue> {
    let mut issues = Vec::new();
    if modification_date < creation_date {
        issues.push(AuthenticityIssue::MetadataInconsistency(format!(
            "Modification date ({modification_date}) is earlier than creation date ({creation_date})"
        )));
    }
    if let Some(enc) = encoding_date {
        if enc < creation_date {
            issues.push(AuthenticityIssue::MetadataInconsistency(format!(
                "Encoding date ({enc}) precedes creation date ({creation_date})"
            )));
        }
    }
    issues
}

/// Detect anomalies in a sequence of timestamps.
///
/// Issues reported:
/// - Non-monotonic timestamps (out-of-order segments).
/// - Gaps > 10× the median inter-frame interval.
pub fn check_timestamp_anomalies(timestamps: &[u64]) -> Vec<AuthenticityIssue> {
    let mut issues = Vec::new();
    if timestamps.len() < 2 {
        return issues;
    }

    let mut deltas: Vec<u64> = timestamps
        .windows(2)
        .map(|w| {
            if w[1] >= w[0] {
                w[1] - w[0]
            } else {
                // Non-monotonic — flag and treat delta as 0.
                0
            }
        })
        .collect();

    // Check for non-monotonic entries.
    for (i, w) in timestamps.windows(2).enumerate() {
        if w[1] < w[0] {
            issues.push(AuthenticityIssue::TimestampAnomaly(format!(
                "Timestamp at index {} ({}) is less than previous ({})",
                i + 1,
                w[1],
                w[0]
            )));
        }
    }

    // Compute median delta.
    deltas.sort_unstable();
    let median_delta = if deltas.is_empty() {
        return issues;
    } else {
        deltas[deltas.len() / 2]
    };

    if median_delta == 0 {
        return issues;
    }

    // Flag gaps more than 10× the median.
    for (i, w) in timestamps.windows(2).enumerate() {
        if w[1] >= w[0] {
            let delta = w[1] - w[0];
            if delta > 10 * median_delta {
                issues.push(AuthenticityIssue::TimestampAnomaly(format!(
                    "Large timestamp gap at index {i}: delta={delta}, median={median_delta}"
                )));
            }
        }
    }
    issues
}

/// Analyse the I-frame pattern of a compressed video stream for signs of re-encoding.
///
/// `frame_types` is a sequence of `'I'`, `'P'`, or `'B'` characters.
///
/// Flags raised:
/// - Suspiciously high I-frame ratio (>50 % indicates re-encoding or cuts).
/// - Consecutive I-frames (two or more in a row) in a stream that also contains P/B frames.
pub fn analyze_compression_history(frame_types: &[char]) -> Vec<AuthenticityIssue> {
    let mut issues = Vec::new();
    if frame_types.is_empty() {
        return issues;
    }

    let total = frame_types.len();
    let i_count = frame_types.iter().filter(|&&c| c == 'I').count();
    let i_ratio = i_count as f64 / total as f64;

    if i_ratio > 0.5 && total > 4 {
        issues.push(AuthenticityIssue::CompressionArtifact(format!(
            "Abnormally high I-frame ratio: {:.1}% (expected <25%)",
            i_ratio * 100.0
        )));
    }

    // Detect consecutive I-frames in a mixed stream.
    let has_non_i = frame_types.iter().any(|&c| c == 'P' || c == 'B');
    if has_non_i {
        for w in frame_types.windows(2) {
            if w[0] == 'I' && w[1] == 'I' {
                issues.push(AuthenticityIssue::CompressionArtifact(
                    "Consecutive I-frames detected in mixed stream — possible re-encode or splice"
                        .to_string(),
                ));
                break; // Report once.
            }
        }
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_new_is_authentic() {
        let r = AuthenticityReport::new("test.mp4");
        assert!(r.is_authentic);
        assert!((r.confidence - 1.0).abs() < 1e-10);
        assert!(r.issues.is_empty());
    }

    #[test]
    fn test_report_add_issue_reduces_confidence() {
        let mut r = AuthenticityReport::new("test.mp4");
        r.add_issue(AuthenticityIssue::MetadataInconsistency("bad".to_string()));
        assert!(r.confidence < 1.0);
        assert_eq!(r.issues.len(), 1);
    }

    #[test]
    fn test_report_multiple_issues_can_flag_inauthentic() {
        let mut r = AuthenticityReport::new("test.mp4");
        for _ in 0..4 {
            r.add_issue(AuthenticityIssue::TimestampAnomaly("x".to_string()));
        }
        assert!(!r.is_authentic);
    }

    #[test]
    fn test_metadata_consistency_ok() {
        let issues = check_metadata_consistency(1000, 2000, Some(1500));
        assert!(issues.is_empty());
    }

    #[test]
    fn test_metadata_consistency_modification_before_creation() {
        let issues = check_metadata_consistency(2000, 1000, None);
        assert!(!issues.is_empty());
        assert!(matches!(
            issues[0],
            AuthenticityIssue::MetadataInconsistency(_)
        ));
    }

    #[test]
    fn test_metadata_consistency_encoding_before_creation() {
        let issues = check_metadata_consistency(1000, 2000, Some(500));
        assert!(!issues.is_empty());
        assert!(matches!(
            issues[0],
            AuthenticityIssue::MetadataInconsistency(_)
        ));
    }

    #[test]
    fn test_timestamp_anomalies_empty() {
        assert!(check_timestamp_anomalies(&[]).is_empty());
    }

    #[test]
    fn test_timestamp_anomalies_monotonic() {
        let ts: Vec<u64> = (0..10).map(|i| i * 33).collect();
        assert!(check_timestamp_anomalies(&ts).is_empty());
    }

    #[test]
    fn test_timestamp_anomalies_non_monotonic() {
        let ts = vec![0u64, 33, 66, 50, 99];
        let issues = check_timestamp_anomalies(&ts);
        assert!(!issues.is_empty());
        assert!(matches!(issues[0], AuthenticityIssue::TimestampAnomaly(_)));
    }

    #[test]
    fn test_compression_history_normal_gop() {
        // Typical I P P P B B I P P P pattern.
        let types: Vec<char> = "IPPPBBIPPP".chars().collect();
        let issues = analyze_compression_history(&types);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_compression_history_high_i_ratio() {
        let types: Vec<char> = "IIIIIPIPPI".chars().collect();
        let issues = analyze_compression_history(&types);
        assert!(!issues.is_empty());
        assert!(matches!(
            issues[0],
            AuthenticityIssue::CompressionArtifact(_)
        ));
    }

    #[test]
    fn test_compression_history_consecutive_i_in_mixed_stream() {
        let types: Vec<char> = "IPPIIBBP".chars().collect();
        let issues = analyze_compression_history(&types);
        assert!(issues
            .iter()
            .any(|i| matches!(i, AuthenticityIssue::CompressionArtifact(_))));
    }

    #[test]
    fn test_authenticity_issue_description() {
        let issue = AuthenticityIssue::MetadataInconsistency("test".to_string());
        assert_eq!(issue.description(), "test");
    }

    #[test]
    fn test_authenticity_checker_default_has_checks() {
        let checker = AuthenticityChecker::default();
        assert!(!checker.checks.is_empty());
    }

    #[test]
    fn test_authenticity_checker_none_is_empty() {
        let checker = AuthenticityChecker::none();
        assert!(checker.checks.is_empty());
    }
}
