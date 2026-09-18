//! IMF package validation report generation.
//!
//! After validating an IMF package, the results are collected into a
//! structured report comprising sections, individual issues, and an overall
//! pass/fail status. This module provides the types used by the validator
//! to present its findings.

#![allow(dead_code)]

use std::fmt;

// ---------------------------------------------------------------------------
// ImfSeverity
// ---------------------------------------------------------------------------

/// Severity level of an issue found during validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ImfSeverity {
    /// Informational note, not a problem.
    Info,
    /// Potential issue that may cause problems in some workflows.
    Warning,
    /// Definite problem that violates a specification requirement.
    Error,
    /// Critical failure that prevents further processing.
    Fatal,
}

impl ImfSeverity {
    /// Short label for display.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
        }
    }

    /// Returns `true` if this severity represents a failure (error or fatal).
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Error | Self::Fatal)
    }
}

impl fmt::Display for ImfSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// ImfIssue
// ---------------------------------------------------------------------------

/// Identifies the category / domain of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImfIssue {
    /// Hash mismatch between PKL declaration and actual file.
    HashMismatch,
    /// Missing asset referenced in CPL or PKL.
    MissingAsset,
    /// Edit rate inconsistency between CPL and essence.
    EditRateMismatch,
    /// Duration declared in CPL does not match essence.
    DurationMismatch,
    /// Invalid or malformed UUID.
    InvalidUuid,
    /// SMPTE conformance rule violation.
    ConformanceViolation,
    /// XML structure error (missing required element, etc.).
    XmlStructure,
    /// Essence descriptor constraint violated.
    EssenceConstraint,
    /// Timeline gap or overlap detected.
    TimelineGap,
    /// Unsupported feature or profile.
    UnsupportedFeature,
    /// Generic issue not covered by the above.
    Other,
}

impl ImfIssue {
    /// Short label for the issue category.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HashMismatch => "hash-mismatch",
            Self::MissingAsset => "missing-asset",
            Self::EditRateMismatch => "edit-rate-mismatch",
            Self::DurationMismatch => "duration-mismatch",
            Self::InvalidUuid => "invalid-uuid",
            Self::ConformanceViolation => "conformance",
            Self::XmlStructure => "xml-structure",
            Self::EssenceConstraint => "essence-constraint",
            Self::TimelineGap => "timeline-gap",
            Self::UnsupportedFeature => "unsupported",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ImfIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// ImfReportEntry
// ---------------------------------------------------------------------------

/// A single finding in an IMF validation report.
#[derive(Debug, Clone)]
pub struct ImfReportEntry {
    /// Issue category.
    pub issue: ImfIssue,
    /// Severity level.
    pub severity: ImfSeverity,
    /// Human-readable message describing the finding.
    pub message: String,
    /// Optional location reference (e.g., file path, element XPath).
    pub location: Option<String>,
}

impl ImfReportEntry {
    /// Create a new report entry.
    #[must_use]
    pub fn new(issue: ImfIssue, severity: ImfSeverity, message: impl Into<String>) -> Self {
        Self {
            issue,
            severity,
            message: message.into(),
            location: None,
        }
    }

    /// Attach a location to the entry.
    #[must_use]
    pub fn with_location(mut self, loc: impl Into<String>) -> Self {
        self.location = Some(loc.into());
        self
    }

    /// Returns `true` if this entry represents a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.severity.is_failure()
    }
}

impl fmt::Display for ImfReportEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.issue, self.message)?;
        if let Some(ref loc) = self.location {
            write!(f, " (at {loc})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImfReportSection
// ---------------------------------------------------------------------------

/// A named section within a validation report (e.g., "CPL", "PKL", "Essence").
#[derive(Debug, Clone)]
pub struct ImfReportSection {
    /// Section name.
    pub name: String,
    /// Entries in this section.
    pub entries: Vec<ImfReportEntry>,
}

impl ImfReportSection {
    /// Create a new section.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: Vec::new(),
        }
    }

    /// Add an entry to this section.
    pub fn add(&mut self, entry: ImfReportEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of failure-level entries.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_failure()).count()
    }

    /// Returns `true` if the section has no failures.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failure_count() == 0
    }

    /// Iterate entries.
    pub fn iter(&self) -> impl Iterator<Item = &ImfReportEntry> {
        self.entries.iter()
    }
}

impl fmt::Display for ImfReportSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "--- {} ({} entries) ---", self.name, self.entries.len())?;
        for entry in &self.entries {
            writeln!(f, "  {entry}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImfReport
// ---------------------------------------------------------------------------

/// Top-level IMF validation report aggregating all sections.
#[derive(Debug, Clone)]
pub struct ImfReport {
    /// Report title / description.
    pub title: String,
    /// Sections of the report.
    pub sections: Vec<ImfReportSection>,
}

impl ImfReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            sections: Vec::new(),
        }
    }

    /// Add a section to the report.
    pub fn add_section(&mut self, section: ImfReportSection) {
        self.sections.push(section);
    }

    /// Total number of entries across all sections.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.sections.iter().map(|s| s.entry_count()).sum()
    }

    /// Total number of failure entries across all sections.
    #[must_use]
    pub fn total_failures(&self) -> usize {
        self.sections.iter().map(|s| s.failure_count()).sum()
    }

    /// Returns `true` if the entire report has no failures.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.total_failures() == 0
    }

    /// Number of sections.
    #[must_use]
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Find a section by name.
    #[must_use]
    pub fn find_section(&self, name: &str) -> Option<&ImfReportSection> {
        self.sections.iter().find(|s| s.name == name)
    }

    /// All entries of a given issue type across all sections.
    #[must_use]
    pub fn entries_by_issue(&self, issue: ImfIssue) -> Vec<&ImfReportEntry> {
        self.sections
            .iter()
            .flat_map(|s| s.entries.iter())
            .filter(|e| e.issue == issue)
            .collect()
    }

    /// Summary line.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{}: {} sections, {} entries, {} failures => {}",
            self.title,
            self.sections.len(),
            self.total_entries(),
            self.total_failures(),
            if self.passed() { "PASS" } else { "FAIL" },
        )
    }
}

impl fmt::Display for ImfReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} ===", self.title)?;
        for section in &self.sections {
            write!(f, "{section}")?;
        }
        writeln!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_label() {
        assert_eq!(ImfSeverity::Info.label(), "INFO");
        assert_eq!(ImfSeverity::Fatal.label(), "FATAL");
    }

    #[test]
    fn test_severity_is_failure() {
        assert!(!ImfSeverity::Info.is_failure());
        assert!(!ImfSeverity::Warning.is_failure());
        assert!(ImfSeverity::Error.is_failure());
        assert!(ImfSeverity::Fatal.is_failure());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ImfSeverity::Info < ImfSeverity::Warning);
        assert!(ImfSeverity::Warning < ImfSeverity::Error);
        assert!(ImfSeverity::Error < ImfSeverity::Fatal);
    }

    #[test]
    fn test_issue_label() {
        assert_eq!(ImfIssue::HashMismatch.label(), "hash-mismatch");
        assert_eq!(ImfIssue::TimelineGap.label(), "timeline-gap");
    }

    #[test]
    fn test_issue_display() {
        let s = format!("{}", ImfIssue::MissingAsset);
        assert_eq!(s, "missing-asset");
    }

    #[test]
    fn test_entry_new() {
        let e = ImfReportEntry::new(ImfIssue::HashMismatch, ImfSeverity::Error, "SHA mismatch");
        assert!(e.is_failure());
        assert_eq!(e.issue, ImfIssue::HashMismatch);
    }

    #[test]
    fn test_entry_with_location() {
        let e = ImfReportEntry::new(ImfIssue::XmlStructure, ImfSeverity::Warning, "missing Id")
            .with_location("/CPL/SegmentList");
        assert_eq!(e.location.as_deref(), Some("/CPL/SegmentList"));
    }

    #[test]
    fn test_entry_display() {
        let e = ImfReportEntry::new(
            ImfIssue::MissingAsset,
            ImfSeverity::Error,
            "track.mxf not found",
        )
        .with_location("PKL");
        let s = format!("{e}");
        assert!(s.contains("ERROR"));
        assert!(s.contains("missing-asset"));
        assert!(s.contains("PKL"));
    }

    #[test]
    fn test_section_add_and_counts() {
        let mut sec = ImfReportSection::new("CPL");
        sec.add(ImfReportEntry::new(
            ImfIssue::InvalidUuid,
            ImfSeverity::Error,
            "bad uuid",
        ));
        sec.add(ImfReportEntry::new(
            ImfIssue::DurationMismatch,
            ImfSeverity::Warning,
            "mismatch",
        ));
        assert_eq!(sec.entry_count(), 2);
        assert_eq!(sec.failure_count(), 1);
        assert!(!sec.is_clean());
    }

    #[test]
    fn test_section_clean() {
        let mut sec = ImfReportSection::new("PKL");
        sec.add(ImfReportEntry::new(
            ImfIssue::Other,
            ImfSeverity::Info,
            "all good",
        ));
        assert!(sec.is_clean());
    }

    #[test]
    fn test_section_display() {
        let mut sec = ImfReportSection::new("Essence");
        sec.add(ImfReportEntry::new(
            ImfIssue::EssenceConstraint,
            ImfSeverity::Error,
            "bad",
        ));
        let s = format!("{sec}");
        assert!(s.contains("Essence"));
        assert!(s.contains("1 entries"));
    }

    #[test]
    fn test_report_passed() {
        let mut report = ImfReport::new("Test Report");
        let mut sec = ImfReportSection::new("CPL");
        sec.add(ImfReportEntry::new(
            ImfIssue::Other,
            ImfSeverity::Info,
            "ok",
        ));
        report.add_section(sec);
        assert!(report.passed());
        assert_eq!(report.total_entries(), 1);
        assert_eq!(report.total_failures(), 0);
    }

    #[test]
    fn test_report_failed() {
        let mut report = ImfReport::new("Test Report");
        let mut sec = ImfReportSection::new("PKL");
        sec.add(ImfReportEntry::new(
            ImfIssue::HashMismatch,
            ImfSeverity::Fatal,
            "corrupt",
        ));
        report.add_section(sec);
        assert!(!report.passed());
        assert_eq!(report.total_failures(), 1);
    }

    #[test]
    fn test_report_find_section() {
        let mut report = ImfReport::new("R");
        report.add_section(ImfReportSection::new("CPL"));
        report.add_section(ImfReportSection::new("PKL"));
        assert!(report.find_section("CPL").is_some());
        assert!(report.find_section("NONE").is_none());
    }

    #[test]
    fn test_report_entries_by_issue() {
        let mut report = ImfReport::new("R");
        let mut s1 = ImfReportSection::new("A");
        s1.add(ImfReportEntry::new(
            ImfIssue::HashMismatch,
            ImfSeverity::Error,
            "x",
        ));
        let mut s2 = ImfReportSection::new("B");
        s2.add(ImfReportEntry::new(
            ImfIssue::HashMismatch,
            ImfSeverity::Warning,
            "y",
        ));
        s2.add(ImfReportEntry::new(
            ImfIssue::MissingAsset,
            ImfSeverity::Error,
            "z",
        ));
        report.add_section(s1);
        report.add_section(s2);
        let hashes = report.entries_by_issue(ImfIssue::HashMismatch);
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_report_summary() {
        let report = ImfReport::new("My Package");
        let summary = report.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("My Package"));
    }

    #[test]
    fn test_report_display() {
        let mut report = ImfReport::new("Full Report");
        report.add_section(ImfReportSection::new("Section1"));
        let s = format!("{report}");
        assert!(s.contains("Full Report"));
        assert!(s.contains("Section1"));
    }
}
