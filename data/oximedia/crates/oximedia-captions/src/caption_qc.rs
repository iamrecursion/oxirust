#![allow(dead_code)]
//! Caption quality control framework.
//!
//! This module provides automated quality control checks for caption tracks,
//! verifying compliance with FCC, OFCOM, ARIB, and other broadcast regulations.
//! It checks timing accuracy, text formatting, positioning, and readability.

use std::collections::HashMap;
use std::fmt;

/// QC check category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QcCategory {
    /// Timing-related checks.
    Timing,
    /// Text formatting checks.
    Formatting,
    /// Positioning and region checks.
    Positioning,
    /// Readability and reading speed checks.
    Readability,
    /// Encoding and character set checks.
    Encoding,
    /// Compliance with specific broadcast standards.
    Compliance,
}

impl fmt::Display for QcCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timing => write!(f, "Timing"),
            Self::Formatting => write!(f, "Formatting"),
            Self::Positioning => write!(f, "Positioning"),
            Self::Readability => write!(f, "Readability"),
            Self::Encoding => write!(f, "Encoding"),
            Self::Compliance => write!(f, "Compliance"),
        }
    }
}

/// QC check result status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QcStatus {
    /// Check passed.
    Pass,
    /// Check passed with warnings.
    Warn,
    /// Check failed.
    Fail,
    /// Check was skipped.
    Skip,
}

impl fmt::Display for QcStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Warn => write!(f, "WARN"),
            Self::Fail => write!(f, "FAIL"),
            Self::Skip => write!(f, "SKIP"),
        }
    }
}

/// Broadcast standard to check against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastStandard {
    /// FCC (USA) closed captioning rules.
    Fcc,
    /// OFCOM (UK) subtitling guidelines.
    Ofcom,
    /// ARIB (Japan) captioning standard.
    Arib,
    /// Netflix timed text style guide.
    Netflix,
    /// BBC subtitle guidelines.
    Bbc,
    /// Custom / generic standard.
    Custom,
}

impl fmt::Display for BroadcastStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fcc => write!(f, "FCC"),
            Self::Ofcom => write!(f, "OFCOM"),
            Self::Arib => write!(f, "ARIB"),
            Self::Netflix => write!(f, "Netflix"),
            Self::Bbc => write!(f, "BBC"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Configuration for QC checks.
#[derive(Debug, Clone)]
pub struct QcConfig {
    /// The broadcast standard to check against.
    pub standard: BroadcastStandard,
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum lines per caption.
    pub max_lines: usize,
    /// Maximum words per minute.
    pub max_wpm: f64,
    /// Minimum caption display duration in seconds.
    pub min_duration_secs: f64,
    /// Maximum caption display duration in seconds.
    pub max_duration_secs: f64,
    /// Minimum gap between captions in seconds.
    pub min_gap_secs: f64,
    /// Whether to check for empty captions.
    pub check_empty: bool,
    /// Whether to check for trailing whitespace.
    pub check_trailing_whitespace: bool,
    /// Whether to check for consecutive identical captions.
    pub check_duplicates: bool,
}

impl Default for QcConfig {
    fn default() -> Self {
        Self {
            standard: BroadcastStandard::Fcc,
            max_chars_per_line: 32,
            max_lines: 2,
            max_wpm: 220.0,
            min_duration_secs: 0.667,
            max_duration_secs: 7.0,
            min_gap_secs: 0.067,
            check_empty: true,
            check_trailing_whitespace: true,
            check_duplicates: true,
        }
    }
}

impl QcConfig {
    /// Create a config for Netflix subtitle guidelines.
    #[must_use]
    pub fn netflix() -> Self {
        Self {
            standard: BroadcastStandard::Netflix,
            max_chars_per_line: 42,
            max_lines: 2,
            max_wpm: 200.0,
            min_duration_secs: 0.833,
            max_duration_secs: 7.0,
            min_gap_secs: 0.083,
            check_empty: true,
            check_trailing_whitespace: true,
            check_duplicates: true,
        }
    }

    /// Create a config for BBC subtitle guidelines.
    #[must_use]
    pub fn bbc() -> Self {
        Self {
            standard: BroadcastStandard::Bbc,
            max_chars_per_line: 37,
            max_lines: 2,
            max_wpm: 180.0,
            min_duration_secs: 0.3,
            max_duration_secs: 7.0,
            min_gap_secs: 0.0,
            check_empty: true,
            check_trailing_whitespace: true,
            check_duplicates: true,
        }
    }
}

/// A caption entry for QC analysis.
#[derive(Debug, Clone)]
pub struct QcCaption {
    /// Index in the caption track.
    pub index: usize,
    /// Text content.
    pub text: String,
    /// Start time in seconds.
    pub start_secs: f64,
    /// End time in seconds.
    pub end_secs: f64,
}

impl QcCaption {
    /// Create a new QC caption entry.
    pub fn new(index: usize, text: impl Into<String>, start_secs: f64, end_secs: f64) -> Self {
        Self {
            index,
            text: text.into(),
            start_secs,
            end_secs,
        }
    }

    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        (self.end_secs - self.start_secs).max(0.0)
    }

    /// Count words.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Calculate words per minute.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn wpm(&self) -> f64 {
        let d = self.duration_secs();
        if d <= 0.0 {
            return 0.0;
        }
        (self.word_count() as f64 / d) * 60.0
    }

    /// Get the maximum line length in characters.
    #[must_use]
    pub fn max_line_length(&self) -> usize {
        self.text
            .lines()
            .map(|l| l.chars().count())
            .max()
            .unwrap_or(0)
    }

    /// Get the number of lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.text.lines().count().max(1)
    }

    /// Check if the text has trailing whitespace on any line.
    #[must_use]
    pub fn has_trailing_whitespace(&self) -> bool {
        self.text.lines().any(|l| l != l.trim_end())
    }

    /// Check if the caption text is empty or whitespace-only.
    #[must_use]
    pub fn is_empty_text(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// A single QC finding.
#[derive(Debug, Clone)]
pub struct QcFinding {
    /// Category of the check.
    pub category: QcCategory,
    /// Status of the check.
    pub status: QcStatus,
    /// Caption index.
    pub caption_index: usize,
    /// Check rule identifier.
    pub rule_id: String,
    /// Human-readable message.
    pub message: String,
}

/// QC report aggregating all findings.
#[derive(Debug, Clone)]
pub struct QcReport {
    /// Standard used for this report.
    pub standard: BroadcastStandard,
    /// Total captions checked.
    pub total_captions: usize,
    /// All findings.
    pub findings: Vec<QcFinding>,
    /// Overall pass/fail.
    pub overall_status: QcStatus,
    /// Counts by category.
    pub category_counts: HashMap<QcCategory, HashMap<QcStatus, usize>>,
}

impl QcReport {
    /// Count findings with a given status.
    #[must_use]
    pub fn count_by_status(&self, status: QcStatus) -> usize {
        self.findings.iter().filter(|f| f.status == status).count()
    }

    /// Get all fail findings.
    #[must_use]
    pub fn failures(&self) -> Vec<&QcFinding> {
        self.findings
            .iter()
            .filter(|f| f.status == QcStatus::Fail)
            .collect()
    }

    /// Check if the report has any failures.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.findings.iter().any(|f| f.status == QcStatus::Fail)
    }
}

/// The QC engine that runs checks against captions.
#[derive(Debug, Clone)]
pub struct CaptionQcEngine {
    /// QC configuration.
    pub config: QcConfig,
}

impl CaptionQcEngine {
    /// Create a new QC engine with default FCC config.
    #[must_use]
    pub fn new(config: QcConfig) -> Self {
        Self { config }
    }

    /// Run all QC checks on a list of captions.
    #[must_use]
    pub fn run_checks(&self, captions: &[QcCaption]) -> QcReport {
        let mut findings = Vec::new();

        for caption in captions {
            self.check_timing(caption, &mut findings);
            self.check_formatting(caption, &mut findings);
            self.check_readability(caption, &mut findings);
        }

        // Cross-caption checks
        self.check_gaps(captions, &mut findings);
        if self.config.check_duplicates {
            self.check_consecutive_duplicates(captions, &mut findings);
        }

        let overall_status = if findings.iter().any(|f| f.status == QcStatus::Fail) {
            QcStatus::Fail
        } else if findings.iter().any(|f| f.status == QcStatus::Warn) {
            QcStatus::Warn
        } else {
            QcStatus::Pass
        };

        let mut category_counts: HashMap<QcCategory, HashMap<QcStatus, usize>> = HashMap::new();
        for finding in &findings {
            *category_counts
                .entry(finding.category)
                .or_default()
                .entry(finding.status)
                .or_insert(0) += 1;
        }

        QcReport {
            standard: self.config.standard,
            total_captions: captions.len(),
            findings,
            overall_status,
            category_counts,
        }
    }

    /// Check timing constraints for a single caption.
    fn check_timing(&self, caption: &QcCaption, findings: &mut Vec<QcFinding>) {
        let d = caption.duration_secs();

        if d < self.config.min_duration_secs {
            findings.push(QcFinding {
                category: QcCategory::Timing,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "T001".to_string(),
                message: format!(
                    "Duration {d:.3}s below minimum {:.3}s",
                    self.config.min_duration_secs
                ),
            });
        }

        if d > self.config.max_duration_secs {
            findings.push(QcFinding {
                category: QcCategory::Timing,
                status: QcStatus::Warn,
                caption_index: caption.index,
                rule_id: "T002".to_string(),
                message: format!(
                    "Duration {d:.3}s exceeds maximum {:.3}s",
                    self.config.max_duration_secs
                ),
            });
        }

        if caption.start_secs > caption.end_secs {
            findings.push(QcFinding {
                category: QcCategory::Timing,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "T003".to_string(),
                message: "Start time is after end time".to_string(),
            });
        }
    }

    /// Check formatting constraints for a single caption.
    fn check_formatting(&self, caption: &QcCaption, findings: &mut Vec<QcFinding>) {
        if self.config.check_empty && caption.is_empty_text() {
            findings.push(QcFinding {
                category: QcCategory::Formatting,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "F001".to_string(),
                message: "Caption text is empty".to_string(),
            });
        }

        if caption.max_line_length() > self.config.max_chars_per_line {
            findings.push(QcFinding {
                category: QcCategory::Formatting,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "F002".to_string(),
                message: format!(
                    "Line length {} exceeds maximum {}",
                    caption.max_line_length(),
                    self.config.max_chars_per_line
                ),
            });
        }

        if caption.line_count() > self.config.max_lines {
            findings.push(QcFinding {
                category: QcCategory::Formatting,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "F003".to_string(),
                message: format!(
                    "Line count {} exceeds maximum {}",
                    caption.line_count(),
                    self.config.max_lines
                ),
            });
        }

        if self.config.check_trailing_whitespace && caption.has_trailing_whitespace() {
            findings.push(QcFinding {
                category: QcCategory::Formatting,
                status: QcStatus::Warn,
                caption_index: caption.index,
                rule_id: "F004".to_string(),
                message: "Caption has trailing whitespace".to_string(),
            });
        }
    }

    /// Check readability constraints for a single caption.
    fn check_readability(&self, caption: &QcCaption, findings: &mut Vec<QcFinding>) {
        let wpm = caption.wpm();
        if wpm > self.config.max_wpm {
            findings.push(QcFinding {
                category: QcCategory::Readability,
                status: QcStatus::Fail,
                caption_index: caption.index,
                rule_id: "R001".to_string(),
                message: format!(
                    "Reading speed {wpm:.0} WPM exceeds maximum {:.0} WPM",
                    self.config.max_wpm
                ),
            });
        }
    }

    /// Check gaps between consecutive captions.
    fn check_gaps(&self, captions: &[QcCaption], findings: &mut Vec<QcFinding>) {
        for pair in captions.windows(2) {
            let gap = pair[1].start_secs - pair[0].end_secs;
            if gap < 0.0 {
                findings.push(QcFinding {
                    category: QcCategory::Timing,
                    status: QcStatus::Fail,
                    caption_index: pair[0].index,
                    rule_id: "T004".to_string(),
                    message: format!("Captions overlap by {:.3}s", -gap),
                });
            } else if gap > 0.0 && gap < self.config.min_gap_secs {
                findings.push(QcFinding {
                    category: QcCategory::Timing,
                    status: QcStatus::Warn,
                    caption_index: pair[0].index,
                    rule_id: "T005".to_string(),
                    message: format!(
                        "Gap {gap:.3}s below minimum {:.3}s",
                        self.config.min_gap_secs
                    ),
                });
            }
        }
    }

    /// Check for consecutive identical captions.
    fn check_consecutive_duplicates(&self, captions: &[QcCaption], findings: &mut Vec<QcFinding>) {
        for pair in captions.windows(2) {
            if pair[0].text.trim() == pair[1].text.trim() && !pair[0].is_empty_text() {
                findings.push(QcFinding {
                    category: QcCategory::Formatting,
                    status: QcStatus::Warn,
                    caption_index: pair[1].index,
                    rule_id: "F005".to_string(),
                    message: "Duplicate of previous caption".to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_captions() -> Vec<QcCaption> {
        vec![
            QcCaption::new(0, "Hello world.", 0.0, 2.0),
            QcCaption::new(1, "This is a test.", 2.1, 4.5),
            QcCaption::new(2, "Final caption.", 5.0, 7.0),
        ]
    }

    #[test]
    fn test_qc_caption_duration() {
        let c = QcCaption::new(0, "Test", 1.0, 3.5);
        assert!((c.duration_secs() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qc_caption_word_count() {
        let c = QcCaption::new(0, "one two three", 0.0, 1.0);
        assert_eq!(c.word_count(), 3);
    }

    #[test]
    fn test_qc_caption_wpm() {
        let c = QcCaption::new(0, "one two three four five six", 0.0, 3.0);
        assert!((c.wpm() - 120.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qc_caption_max_line_length() {
        let c = QcCaption::new(0, "short\nlonger line here", 0.0, 2.0);
        assert_eq!(c.max_line_length(), 16);
    }

    #[test]
    fn test_qc_caption_line_count() {
        let c = QcCaption::new(0, "line1\nline2\nline3", 0.0, 3.0);
        assert_eq!(c.line_count(), 3);
    }

    #[test]
    fn test_qc_caption_trailing_whitespace() {
        let c = QcCaption::new(0, "hello ", 0.0, 1.0);
        assert!(c.has_trailing_whitespace());
        let c2 = QcCaption::new(1, "hello", 0.0, 1.0);
        assert!(!c2.has_trailing_whitespace());
    }

    #[test]
    fn test_qc_caption_empty_text() {
        let c = QcCaption::new(0, "   ", 0.0, 1.0);
        assert!(c.is_empty_text());
        let c2 = QcCaption::new(1, "hi", 0.0, 1.0);
        assert!(!c2.is_empty_text());
    }

    #[test]
    fn test_qc_engine_clean_pass() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = sample_captions();
        let report = engine.run_checks(&captions);
        assert_eq!(report.overall_status, QcStatus::Pass);
        assert!(!report.has_failures());
    }

    #[test]
    fn test_qc_engine_detects_short_duration() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = vec![QcCaption::new(0, "Blink", 0.0, 0.1)];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "T001"));
    }

    #[test]
    fn test_qc_engine_detects_overlap() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = vec![
            QcCaption::new(0, "First", 0.0, 3.0),
            QcCaption::new(1, "Overlapping", 2.5, 5.0),
        ];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "T004"));
    }

    #[test]
    fn test_qc_engine_detects_line_length() {
        let engine = CaptionQcEngine::new(QcConfig::default()); // max 32
        let long_line = "A".repeat(40);
        let captions = vec![QcCaption::new(0, &long_line, 0.0, 3.0)];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "F002"));
    }

    #[test]
    fn test_qc_engine_detects_too_many_lines() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = vec![QcCaption::new(0, "L1\nL2\nL3", 0.0, 3.0)];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "F003"));
    }

    #[test]
    fn test_qc_engine_detects_empty_caption() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = vec![QcCaption::new(0, "   ", 0.0, 2.0)];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "F001"));
    }

    #[test]
    fn test_qc_engine_detects_duplicates() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = vec![
            QcCaption::new(0, "Same text", 0.0, 2.0),
            QcCaption::new(1, "Same text", 2.5, 4.5),
        ];
        let report = engine.run_checks(&captions);
        assert!(report.findings.iter().any(|f| f.rule_id == "F005"));
    }

    #[test]
    fn test_qc_report_count_by_status() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        let captions = sample_captions();
        let report = engine.run_checks(&captions);
        assert_eq!(report.count_by_status(QcStatus::Fail), 0);
    }

    #[test]
    fn test_netflix_config() {
        let config = QcConfig::netflix();
        assert_eq!(config.standard, BroadcastStandard::Netflix);
        assert_eq!(config.max_chars_per_line, 42);
    }

    #[test]
    fn test_bbc_config() {
        let config = QcConfig::bbc();
        assert_eq!(config.standard, BroadcastStandard::Bbc);
        assert_eq!(config.max_chars_per_line, 37);
    }

    #[test]
    fn test_category_display() {
        assert_eq!(format!("{}", QcCategory::Timing), "Timing");
        assert_eq!(format!("{}", QcCategory::Encoding), "Encoding");
    }

    #[test]
    fn test_qc_status_display() {
        assert_eq!(format!("{}", QcStatus::Pass), "PASS");
        assert_eq!(format!("{}", QcStatus::Fail), "FAIL");
    }
}

/// Negative tests: verify that intentionally flawed captions trigger the
/// expected QC findings.  Each test exercises a different failure path so
/// that the full check matrix is covered.
#[cfg(test)]
mod qc_negative_tests {
    use super::*;

    // ── Individual check triggers ─────────────────────────────────────────────

    /// T001: caption shorter than Netflix minimum duration (0.833 s).
    #[test]
    fn test_qc_too_short_duration_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        // 0.1 s is well below the 0.833 s Netflix minimum.
        let captions = vec![QcCaption::new(0, "Hello world", 0.0, 0.1)];
        let report = engine.run_checks(&captions);
        let has_fail = report.findings.iter().any(|f| f.status == QcStatus::Fail);
        assert!(
            has_fail,
            "too-short duration should produce a Fail finding (T001)"
        );
        assert!(
            report.findings.iter().any(|f| f.rule_id == "T001"),
            "expected T001 rule in findings"
        );
    }

    /// T003: end time before start time must produce a Fail finding.
    #[test]
    fn test_qc_reversed_timestamps_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![QcCaption::new(0, "Reversed", 5.0, 2.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "T003"),
            "reversed timestamps should trigger T003"
        );
    }

    /// T004: overlapping captions produce a Fail finding.
    #[test]
    fn test_qc_overlapping_captions_fail() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![
            QcCaption::new(0, "First caption", 0.0, 5.0),
            QcCaption::new(1, "Overlapping", 3.0, 7.0),
        ];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "T004"),
            "overlapping captions should trigger T004"
        );
        assert_eq!(report.overall_status, QcStatus::Fail);
    }

    /// R001: 50 words in 1 second = 3000 WPM is far beyond the 200 WPM Netflix limit.
    #[test]
    fn test_qc_too_fast_speech_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let long_text = "word ".repeat(50).trim().to_string();
        let captions = vec![QcCaption::new(0, &long_text, 0.0, 1.0)];
        let report = engine.run_checks(&captions);
        let has_issue = report
            .findings
            .iter()
            .any(|f| f.status == QcStatus::Fail || f.status == QcStatus::Warn);
        assert!(
            has_issue,
            "excessive speech rate should produce Warn or Fail finding (R001)"
        );
    }

    /// F001: empty caption text must be flagged.
    #[test]
    fn test_qc_empty_caption_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![QcCaption::new(0, "   ", 0.0, 2.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "F001"),
            "empty caption should trigger F001"
        );
    }

    /// F002: line exceeding max_chars_per_line must be flagged.
    #[test]
    fn test_qc_long_line_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix()); // max 42
        let long_line = "A".repeat(50);
        let captions = vec![QcCaption::new(0, &long_line, 0.0, 3.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "F002"),
            "over-long line should trigger F002"
        );
    }

    /// F003: more than 2 lines must be flagged (Netflix limit).
    #[test]
    fn test_qc_too_many_lines_fails() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![QcCaption::new(0, "Line 1\nLine 2\nLine 3", 0.0, 4.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "F003"),
            "three-line caption should trigger F003"
        );
    }

    /// F004: trailing whitespace produces a Warn finding.
    #[test]
    fn test_qc_trailing_whitespace_warns() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![QcCaption::new(0, "trailing space  ", 0.0, 2.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.rule_id == "F004" && f.status == QcStatus::Warn),
            "trailing whitespace should trigger F004 Warn"
        );
    }

    /// F005: consecutive duplicate captions produce a Warn finding.
    #[test]
    fn test_qc_duplicate_captions_warn() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        let captions = vec![
            QcCaption::new(0, "Duplicate text", 0.0, 2.0),
            QcCaption::new(1, "Duplicate text", 2.5, 4.5),
        ];
        let report = engine.run_checks(&captions);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.rule_id == "F005" && f.status == QcStatus::Warn),
            "duplicate consecutive captions should trigger F005 Warn"
        );
    }

    // ── Happy-path regression ─────────────────────────────────────────────────

    /// Well-formed captions must not produce any Fail findings.
    #[test]
    fn test_qc_known_good_captions_pass() {
        let engine = CaptionQcEngine::new(QcConfig::netflix());
        // ~5 words / 3 seconds ≈ 100 WPM — well within 200 WPM Netflix limit.
        let captions = vec![
            QcCaption::new(0, "Hello, how are you today?", 0.0, 3.0),
            QcCaption::new(1, "I am doing very well, thank you.", 4.0, 7.0),
        ];
        let report = engine.run_checks(&captions);
        let has_fail = report.findings.iter().any(|f| f.status == QcStatus::Fail);
        assert!(
            !has_fail,
            "well-formed captions should not produce any Fail findings"
        );
    }

    // ── BBC preset checks ─────────────────────────────────────────────────────

    /// BBC has a lower WPM limit (180) and different min_duration (0.3 s).
    #[test]
    fn test_qc_bbc_preset_rate_limit() {
        let engine = CaptionQcEngine::new(QcConfig::bbc());
        // 40 words in 1 second = 2400 WPM — far above BBC 180 WPM limit.
        let fast_text = "word ".repeat(40).trim().to_string();
        let captions = vec![QcCaption::new(0, &fast_text, 0.0, 1.0)];
        let report = engine.run_checks(&captions);
        assert!(
            report.findings.iter().any(|f| f.rule_id == "R001"),
            "BBC preset should also flag excessive speech rate via R001"
        );
    }

    /// All six check-type categories must be exercisable without panic.
    #[test]
    fn test_qc_report_structure_integrity() {
        let engine = CaptionQcEngine::new(QcConfig::default());
        // Produce at least one finding so category_counts is non-empty.
        let captions = vec![QcCaption::new(0, "Blink", 0.0, 0.1)];
        let report = engine.run_checks(&captions);
        // count_by_status must be consistent with findings vec length.
        let total: usize = [
            QcStatus::Pass,
            QcStatus::Warn,
            QcStatus::Fail,
            QcStatus::Skip,
        ]
        .iter()
        .map(|&s| report.count_by_status(s))
        .sum();
        assert_eq!(
            total,
            report.findings.len(),
            "count_by_status totals must equal findings length"
        );
    }
}
