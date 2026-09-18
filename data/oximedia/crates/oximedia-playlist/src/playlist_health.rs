#![allow(dead_code)]
//! Playlist health monitoring and diagnostics.
//!
//! Provides analysis of playlist integrity including overlap detection,
//! duration anomalies, missing media references, and overall schedule
//! health scoring.

use std::collections::HashMap;

/// Severity level for health issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational only.
    Info,
    /// Warning that may need attention.
    Warning,
    /// Error that requires correction.
    Error,
    /// Critical problem that will cause playout failure.
    Critical,
}

impl Severity {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

/// Category of health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueCategory {
    /// Items overlap in time.
    Overlap,
    /// Gap detected between items.
    Gap,
    /// Item duration is abnormal.
    DurationAnomaly,
    /// Referenced media file is missing.
    MissingMedia,
    /// Duplicate item in the playlist.
    Duplicate,
    /// Scheduling conflict.
    ScheduleConflict,
    /// Metadata is incomplete.
    IncompleteMetadata,
}

impl IssueCategory {
    /// Short description of the category.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Overlap => "Time overlap between items",
            Self::Gap => "Gap between scheduled items",
            Self::DurationAnomaly => "Abnormal item duration",
            Self::MissingMedia => "Missing media reference",
            Self::Duplicate => "Duplicate playlist entry",
            Self::ScheduleConflict => "Scheduling conflict",
            Self::IncompleteMetadata => "Incomplete metadata",
        }
    }
}

/// A single health issue detected in the playlist.
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// Category of the issue.
    pub category: IssueCategory,
    /// Descriptive message.
    pub message: String,
    /// Index of the affected item (if applicable).
    pub item_index: Option<usize>,
}

impl HealthIssue {
    /// Create a new health issue.
    pub fn new(severity: Severity, category: IssueCategory, message: &str) -> Self {
        Self {
            severity,
            category,
            message: message.to_string(),
            item_index: None,
        }
    }

    /// Attach an item index.
    pub fn with_index(mut self, index: usize) -> Self {
        self.item_index = Some(index);
        self
    }
}

/// Represents a simple playlist entry for health analysis.
#[derive(Debug, Clone)]
pub struct HealthEntry {
    /// Entry identifier.
    pub id: String,
    /// Start offset in milliseconds.
    pub start_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Media file path.
    pub media_path: String,
    /// Whether metadata is complete.
    pub metadata_complete: bool,
}

impl HealthEntry {
    /// Create a new health entry.
    pub fn new(id: &str, start_ms: u64, duration_ms: u64, media_path: &str) -> Self {
        Self {
            id: id.to_string(),
            start_ms,
            duration_ms,
            media_path: media_path.to_string(),
            metadata_complete: true,
        }
    }

    /// End time in milliseconds.
    pub fn end_ms(&self) -> u64 {
        self.start_ms + self.duration_ms
    }
}

/// Overall health score for a playlist.
#[derive(Debug, Clone)]
pub struct HealthScore {
    /// Score from 0 to 100.
    pub score: u32,
    /// Total number of issues.
    pub total_issues: usize,
    /// Issues by severity.
    pub by_severity: HashMap<Severity, usize>,
    /// Issues by category.
    pub by_category: HashMap<IssueCategory, usize>,
}

impl HealthScore {
    /// Whether the playlist is considered healthy (score >= 80).
    pub fn is_healthy(&self) -> bool {
        self.score >= 80
    }

    /// Whether there are any critical issues.
    pub fn has_critical(&self) -> bool {
        self.by_severity
            .get(&Severity::Critical)
            .copied()
            .unwrap_or(0)
            > 0
    }
}

/// Configuration for health checks.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Maximum allowed gap in milliseconds.
    pub max_gap_ms: u64,
    /// Minimum expected item duration in milliseconds.
    pub min_duration_ms: u64,
    /// Maximum expected item duration in milliseconds.
    pub max_duration_ms: u64,
    /// Whether to check for missing media.
    pub check_media_exists: bool,
    /// Whether to check for duplicate entries.
    pub check_duplicates: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            max_gap_ms: 5_000,
            min_duration_ms: 1_000,
            max_duration_ms: 14_400_000, // 4 hours
            check_media_exists: false,
            check_duplicates: true,
        }
    }
}

/// Health checker that runs diagnostics on playlist entries.
pub struct PlaylistHealthChecker {
    /// Configuration.
    config: HealthCheckConfig,
}

impl PlaylistHealthChecker {
    /// Create a new health checker with default configuration.
    pub fn new() -> Self {
        Self {
            config: HealthCheckConfig::default(),
        }
    }

    /// Create a new health checker with custom configuration.
    pub fn with_config(config: HealthCheckConfig) -> Self {
        Self { config }
    }

    /// Run all health checks and return discovered issues.
    pub fn check(&self, entries: &[HealthEntry]) -> Vec<HealthIssue> {
        let mut issues = Vec::new();
        self.check_overlaps(entries, &mut issues);
        self.check_gaps(entries, &mut issues);
        self.check_durations(entries, &mut issues);
        self.check_metadata(entries, &mut issues);
        if self.config.check_duplicates {
            self.check_duplicates(entries, &mut issues);
        }
        issues
    }

    /// Compute a health score from the issues found.
    #[allow(clippy::cast_precision_loss)]
    pub fn score(&self, entries: &[HealthEntry]) -> HealthScore {
        let issues = self.check(entries);
        let mut by_severity: HashMap<Severity, usize> = HashMap::new();
        let mut by_category: HashMap<IssueCategory, usize> = HashMap::new();

        for issue in &issues {
            *by_severity.entry(issue.severity).or_insert(0) += 1;
            *by_category.entry(issue.category).or_insert(0) += 1;
        }

        let penalty: f64 = issues
            .iter()
            .map(|i| match i.severity {
                Severity::Info => 0.5,
                Severity::Warning => 2.0,
                Severity::Error => 5.0,
                Severity::Critical => 15.0,
            })
            .sum();

        let raw = (100.0 - penalty).max(0.0);
        let score = raw.min(100.0) as u32;

        HealthScore {
            score,
            total_issues: issues.len(),
            by_severity,
            by_category,
        }
    }

    /// Check for overlapping entries.
    fn check_overlaps(&self, entries: &[HealthEntry], issues: &mut Vec<HealthIssue>) {
        for i in 1..entries.len() {
            let prev_end = entries[i - 1].end_ms();
            let cur_start = entries[i].start_ms;
            if cur_start < prev_end {
                issues.push(
                    HealthIssue::new(
                        Severity::Error,
                        IssueCategory::Overlap,
                        &format!(
                            "Overlap of {}ms between '{}' and '{}'",
                            prev_end - cur_start,
                            entries[i - 1].id,
                            entries[i].id,
                        ),
                    )
                    .with_index(i),
                );
            }
        }
    }

    /// Check for gaps between entries.
    fn check_gaps(&self, entries: &[HealthEntry], issues: &mut Vec<HealthIssue>) {
        for i in 1..entries.len() {
            let prev_end = entries[i - 1].end_ms();
            let cur_start = entries[i].start_ms;
            if cur_start > prev_end {
                let gap = cur_start - prev_end;
                if gap > self.config.max_gap_ms {
                    issues.push(
                        HealthIssue::new(
                            Severity::Warning,
                            IssueCategory::Gap,
                            &format!("Gap of {}ms before '{}'", gap, entries[i].id),
                        )
                        .with_index(i),
                    );
                }
            }
        }
    }

    /// Check for abnormal durations.
    fn check_durations(&self, entries: &[HealthEntry], issues: &mut Vec<HealthIssue>) {
        for (i, entry) in entries.iter().enumerate() {
            if entry.duration_ms < self.config.min_duration_ms {
                issues.push(
                    HealthIssue::new(
                        Severity::Warning,
                        IssueCategory::DurationAnomaly,
                        &format!(
                            "Duration {}ms below minimum {}ms for '{}'",
                            entry.duration_ms, self.config.min_duration_ms, entry.id,
                        ),
                    )
                    .with_index(i),
                );
            }
            if entry.duration_ms > self.config.max_duration_ms {
                issues.push(
                    HealthIssue::new(
                        Severity::Warning,
                        IssueCategory::DurationAnomaly,
                        &format!(
                            "Duration {}ms above maximum {}ms for '{}'",
                            entry.duration_ms, self.config.max_duration_ms, entry.id,
                        ),
                    )
                    .with_index(i),
                );
            }
        }
    }

    /// Check for incomplete metadata.
    fn check_metadata(&self, entries: &[HealthEntry], issues: &mut Vec<HealthIssue>) {
        for (i, entry) in entries.iter().enumerate() {
            if !entry.metadata_complete {
                issues.push(
                    HealthIssue::new(
                        Severity::Info,
                        IssueCategory::IncompleteMetadata,
                        &format!("Incomplete metadata for '{}'", entry.id),
                    )
                    .with_index(i),
                );
            }
        }
    }

    /// Check for duplicate entries.
    fn check_duplicates(&self, entries: &[HealthEntry], issues: &mut Vec<HealthIssue>) {
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for (i, entry) in entries.iter().enumerate() {
            if let Some(&first_idx) = seen.get(entry.media_path.as_str()) {
                issues.push(
                    HealthIssue::new(
                        Severity::Info,
                        IssueCategory::Duplicate,
                        &format!(
                            "Duplicate media '{}' (first at index {})",
                            entry.media_path, first_idx,
                        ),
                    )
                    .with_index(i),
                );
            } else {
                seen.insert(&entry.media_path, i);
            }
        }
    }
}

impl Default for PlaylistHealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<HealthEntry> {
        vec![
            HealthEntry::new("item_1", 0, 10_000, "/media/clip1.mxf"),
            HealthEntry::new("item_2", 10_000, 15_000, "/media/clip2.mxf"),
            HealthEntry::new("item_3", 25_000, 20_000, "/media/clip3.mxf"),
        ]
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_severity_label() {
        assert_eq!(Severity::Info.label(), "INFO");
        assert_eq!(Severity::Critical.label(), "CRITICAL");
    }

    #[test]
    fn test_issue_category_description() {
        assert_eq!(
            IssueCategory::Overlap.description(),
            "Time overlap between items"
        );
        assert_eq!(
            IssueCategory::MissingMedia.description(),
            "Missing media reference"
        );
    }

    #[test]
    fn test_health_entry_end_ms() {
        let entry = HealthEntry::new("a", 1000, 5000, "/clip.mxf");
        assert_eq!(entry.end_ms(), 6000);
    }

    #[test]
    fn test_no_issues_for_clean_playlist() {
        let checker = PlaylistHealthChecker::new();
        let entries = sample_entries();
        let issues = checker.check(&entries);
        assert!(issues.is_empty(), "Expected no issues, got {:?}", issues);
    }

    #[test]
    fn test_overlap_detection() {
        let checker = PlaylistHealthChecker::new();
        let entries = vec![
            HealthEntry::new("a", 0, 10_000, "/clip1.mxf"),
            HealthEntry::new("b", 8_000, 10_000, "/clip2.mxf"),
        ];
        let issues = checker.check(&entries);
        assert!(issues.iter().any(|i| i.category == IssueCategory::Overlap));
    }

    #[test]
    fn test_gap_detection() {
        let mut config = HealthCheckConfig::default();
        config.max_gap_ms = 1_000;
        let checker = PlaylistHealthChecker::with_config(config);
        let entries = vec![
            HealthEntry::new("a", 0, 10_000, "/clip1.mxf"),
            HealthEntry::new("b", 15_000, 10_000, "/clip2.mxf"),
        ];
        let issues = checker.check(&entries);
        assert!(issues.iter().any(|i| i.category == IssueCategory::Gap));
    }

    #[test]
    fn test_duration_anomaly_too_short() {
        let mut config = HealthCheckConfig::default();
        config.min_duration_ms = 5_000;
        let checker = PlaylistHealthChecker::with_config(config);
        let entries = vec![HealthEntry::new("a", 0, 500, "/clip.mxf")];
        let issues = checker.check(&entries);
        assert!(issues
            .iter()
            .any(|i| i.category == IssueCategory::DurationAnomaly));
    }

    #[test]
    fn test_duration_anomaly_too_long() {
        let mut config = HealthCheckConfig::default();
        config.max_duration_ms = 10_000;
        let checker = PlaylistHealthChecker::with_config(config);
        let entries = vec![HealthEntry::new("a", 0, 50_000, "/clip.mxf")];
        let issues = checker.check(&entries);
        assert!(issues
            .iter()
            .any(|i| i.category == IssueCategory::DurationAnomaly));
    }

    #[test]
    fn test_duplicate_detection() {
        let checker = PlaylistHealthChecker::new();
        let entries = vec![
            HealthEntry::new("a", 0, 10_000, "/clip.mxf"),
            HealthEntry::new("b", 10_000, 10_000, "/clip.mxf"),
        ];
        let issues = checker.check(&entries);
        assert!(issues
            .iter()
            .any(|i| i.category == IssueCategory::Duplicate));
    }

    #[test]
    fn test_incomplete_metadata() {
        let checker = PlaylistHealthChecker::new();
        let mut entry = HealthEntry::new("a", 0, 10_000, "/clip.mxf");
        entry.metadata_complete = false;
        let issues = checker.check(&[entry]);
        assert!(issues
            .iter()
            .any(|i| i.category == IssueCategory::IncompleteMetadata));
    }

    #[test]
    fn test_health_score_perfect() {
        let checker = PlaylistHealthChecker::new();
        let entries = sample_entries();
        let score = checker.score(&entries);
        assert_eq!(score.score, 100);
        assert!(score.is_healthy());
        assert!(!score.has_critical());
    }

    #[test]
    fn test_health_score_with_issues() {
        let checker = PlaylistHealthChecker::new();
        let entries = vec![
            HealthEntry::new("a", 0, 10_000, "/clip1.mxf"),
            HealthEntry::new("b", 5_000, 10_000, "/clip2.mxf"),
        ];
        let score = checker.score(&entries);
        assert!(score.score < 100);
        assert!(score.total_issues > 0);
    }

    #[test]
    fn test_health_issue_with_index() {
        let issue =
            HealthIssue::new(Severity::Error, IssueCategory::Overlap, "test").with_index(42);
        assert_eq!(issue.item_index, Some(42));
    }
}
