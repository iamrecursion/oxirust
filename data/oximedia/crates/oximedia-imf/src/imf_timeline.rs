//! IMF timeline construction and validation helpers.
//!
//! Provides a `Timeline` type that assembles ordered `TimelineEntry` records
//! from CPL segment/sequence data, checks for gaps and overlaps, and computes
//! aggregate metrics such as total duration and edit-unit statistics.

#![allow(dead_code)]

/// A single entry on an IMF timeline, corresponding to one resolved resource
/// within a CPL segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEntry {
    /// Segment UUID this entry belongs to.
    pub segment_id: String,
    /// Sequence UUID this entry belongs to.
    pub sequence_id: String,
    /// Track-file asset UUID.
    pub track_file_id: String,
    /// Absolute edit-unit offset within the timeline (timeline position).
    pub timeline_offset: u64,
    /// Number of edit units contributed by this entry.
    pub duration: u64,
}

impl TimelineEntry {
    /// Create a new [`TimelineEntry`].
    #[must_use]
    pub fn new(
        segment_id: impl Into<String>,
        sequence_id: impl Into<String>,
        track_file_id: impl Into<String>,
        timeline_offset: u64,
        duration: u64,
    ) -> Self {
        Self {
            segment_id: segment_id.into(),
            sequence_id: sequence_id.into(),
            track_file_id: track_file_id.into(),
            timeline_offset,
            duration,
        }
    }

    /// Exclusive end position of this entry (offset + duration).
    #[must_use]
    pub fn end_offset(&self) -> u64 {
        self.timeline_offset + self.duration
    }

    /// Returns `true` if this entry overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.timeline_offset < other.end_offset() && other.timeline_offset < self.end_offset()
    }
}

/// Diagnostic severity level for timeline issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Informational notice; does not affect playback.
    Info,
    /// Non-fatal warning that may affect quality or interoperability.
    Warning,
    /// Fatal error that prevents playback or packaging.
    Error,
}

/// A timeline issue found during validation.
#[derive(Debug, Clone)]
pub struct TimelineIssue {
    /// Severity of this issue.
    pub severity: IssueSeverity,
    /// Machine-readable issue code (e.g. `"GAP_DETECTED"`).
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Edit-unit position at which the issue was detected.
    pub position: u64,
}

impl TimelineIssue {
    /// Create a new [`TimelineIssue`].
    #[must_use]
    pub fn new(
        severity: IssueSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
        position: u64,
    ) -> Self {
        Self {
            severity: severity,
            code: code.into(),
            message: message.into(),
            position,
        }
    }
}

/// Result of a timeline validation pass.
#[derive(Debug, Clone, Default)]
pub struct TimelineValidationResult {
    /// All issues found during validation (may be empty).
    pub issues: Vec<TimelineIssue>,
}

impl TimelineValidationResult {
    /// Returns `true` when there are no error-severity issues.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }

    /// Number of error-severity issues.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count()
    }

    /// Number of warning-severity issues.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count()
    }
}

/// An assembled IMF timeline ready for validation or playback scheduling.
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    /// Ordered list of timeline entries.
    entries: Vec<TimelineEntry>,
    /// Edit rate as `(numerator, denominator)`.
    edit_rate: (u32, u32),
}

impl Timeline {
    /// Create an empty [`Timeline`] with the given edit rate.
    #[must_use]
    pub fn new(edit_rate: (u32, u32)) -> Self {
        Self {
            entries: Vec::new(),
            edit_rate,
        }
    }

    /// Append an entry to the timeline.
    pub fn add_entry(&mut self, entry: TimelineEntry) {
        self.entries.push(entry);
    }

    /// All entries in the order they were added.
    #[must_use]
    pub fn entries(&self) -> &[TimelineEntry] {
        &self.entries
    }

    /// Total duration (edit units from offset 0 to the last entry's end).
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.entries
            .iter()
            .map(TimelineEntry::end_offset)
            .max()
            .unwrap_or(0)
    }

    /// Total duration in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        let (num, den) = self.edit_rate;
        if num == 0 {
            return 0.0;
        }
        self.total_duration() as f64 * den as f64 / num as f64
    }

    /// Number of distinct track-file UUIDs referenced.
    #[must_use]
    pub fn unique_track_file_count(&self) -> usize {
        let mut ids: Vec<&str> = self
            .entries
            .iter()
            .map(|e| e.track_file_id.as_str())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// Validate the timeline for gaps and overlaps between consecutive entries
    /// that share the same `sequence_id`.
    #[must_use]
    pub fn validate(&self) -> TimelineValidationResult {
        let mut result = TimelineValidationResult::default();

        // Gather all unique sequence IDs.
        let mut seq_ids: Vec<&str> = self
            .entries
            .iter()
            .map(|e| e.sequence_id.as_str())
            .collect();
        seq_ids.sort_unstable();
        seq_ids.dedup();

        for seq_id in seq_ids {
            let mut seq_entries: Vec<&TimelineEntry> = self
                .entries
                .iter()
                .filter(|e| e.sequence_id == seq_id)
                .collect();
            seq_entries.sort_by_key(|e| e.timeline_offset);

            let mut expected = seq_entries.first().map_or(0, |e| e.timeline_offset);

            for entry in &seq_entries {
                if entry.timeline_offset > expected {
                    result.issues.push(TimelineIssue::new(
                        IssueSeverity::Error,
                        "GAP_DETECTED",
                        format!(
                            "Gap in sequence {seq_id} at position {expected} \
                             (expected {expected}, got {})",
                            entry.timeline_offset
                        ),
                        expected,
                    ));
                } else if entry.timeline_offset < expected {
                    result.issues.push(TimelineIssue::new(
                        IssueSeverity::Error,
                        "OVERLAP_DETECTED",
                        format!(
                            "Overlap in sequence {seq_id} at position {}",
                            entry.timeline_offset
                        ),
                        entry.timeline_offset,
                    ));
                }
                expected = entry.end_offset();
            }
        }

        result
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seq: &str, offset: u64, dur: u64) -> TimelineEntry {
        TimelineEntry::new("seg-1", seq, "tf-001", offset, dur)
    }

    // ── TimelineEntry ─────────────────────────────────────────────────────

    #[test]
    fn test_entry_end_offset() {
        let e = entry("seq-1", 100, 50);
        assert_eq!(e.end_offset(), 150);
    }

    #[test]
    fn test_entry_overlaps_true() {
        let a = entry("seq-1", 0, 100);
        let b = entry("seq-1", 50, 100);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_entry_overlaps_false_adjacent() {
        let a = entry("seq-1", 0, 100);
        let b = entry("seq-1", 100, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_entry_overlaps_false_gap() {
        let a = entry("seq-1", 0, 50);
        let b = entry("seq-1", 100, 50);
        assert!(!a.overlaps(&b));
    }

    // ── TimelineIssue ─────────────────────────────────────────────────────

    #[test]
    fn test_issue_construction() {
        let issue = TimelineIssue::new(IssueSeverity::Error, "GAP", "gap at 100", 100);
        assert_eq!(issue.severity, IssueSeverity::Error);
        assert_eq!(issue.code, "GAP");
        assert_eq!(issue.position, 100);
    }

    // ── TimelineValidationResult ──────────────────────────────────────────

    #[test]
    fn test_validation_result_empty_is_valid() {
        let result = TimelineValidationResult::default();
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
        assert_eq!(result.warning_count(), 0);
    }

    #[test]
    fn test_validation_result_error_invalidates() {
        let mut result = TimelineValidationResult::default();
        result
            .issues
            .push(TimelineIssue::new(IssueSeverity::Error, "E", "err", 0));
        assert!(!result.is_valid());
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_validation_result_warning_still_valid() {
        let mut result = TimelineValidationResult::default();
        result
            .issues
            .push(TimelineIssue::new(IssueSeverity::Warning, "W", "warn", 0));
        assert!(result.is_valid());
        assert_eq!(result.warning_count(), 1);
    }

    // ── Timeline ──────────────────────────────────────────────────────────

    #[test]
    fn test_timeline_empty() {
        let tl = Timeline::new((24, 1));
        assert_eq!(tl.total_duration(), 0);
        assert_eq!(tl.unique_track_file_count(), 0);
    }

    #[test]
    fn test_timeline_total_duration() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(entry("seq-1", 0, 2400));
        assert_eq!(tl.total_duration(), 2400);
    }

    #[test]
    fn test_timeline_total_duration_secs() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(entry("seq-1", 0, 2400));
        assert!((tl.total_duration_secs() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_timeline_unique_track_files() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(TimelineEntry::new("seg", "seq", "tf-A", 0, 100));
        tl.add_entry(TimelineEntry::new("seg", "seq", "tf-A", 100, 100));
        tl.add_entry(TimelineEntry::new("seg", "seq", "tf-B", 200, 100));
        assert_eq!(tl.unique_track_file_count(), 2);
    }

    #[test]
    fn test_timeline_validate_clean() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(entry("seq-1", 0, 100));
        tl.add_entry(entry("seq-1", 100, 100));
        let result = tl.validate();
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
    }

    #[test]
    fn test_timeline_validate_gap_detected() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(entry("seq-1", 0, 100));
        tl.add_entry(entry("seq-1", 200, 100)); // gap at 100
        let result = tl.validate();
        assert!(!result.is_valid());
        assert!(result.issues.iter().any(|i| i.code == "GAP_DETECTED"));
    }

    #[test]
    fn test_timeline_validate_overlap_detected() {
        let mut tl = Timeline::new((24, 1));
        tl.add_entry(entry("seq-1", 0, 150));
        tl.add_entry(entry("seq-1", 100, 100)); // overlap at 100
        let result = tl.validate();
        assert!(!result.is_valid());
        assert!(result.issues.iter().any(|i| i.code == "OVERLAP_DETECTED"));
    }
}
