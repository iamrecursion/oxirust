#![allow(dead_code)]
//! Ingest logging: action classification, log entries, and queryable log.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Classifies an ingest-pipeline action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestAction {
    /// File was successfully copied to the archive.
    FileCopied,
    /// Checksum was verified successfully.
    ChecksumVerified,
    /// Metadata was extracted and stored.
    MetadataExtracted,
    /// Checksum mismatch or data corruption detected.
    ChecksumError(String),
    /// An I/O error occurred during ingest.
    IoError(String),
    /// A format validation error was reported.
    ValidationError(String),
    /// Ingest was explicitly skipped (e.g. duplicate detected).
    Skipped(String),
}

impl IngestAction {
    /// Returns `true` if this action represents an error condition.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            Self::ChecksumError(_) | Self::IoError(_) | Self::ValidationError(_)
        )
    }

    /// Returns `true` if this is a successful/non-error action.
    #[must_use]
    pub fn is_success(&self) -> bool {
        !self.is_error()
    }

    /// Returns a short descriptive label for logging.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::FileCopied => "FILE_COPIED",
            Self::ChecksumVerified => "CHECKSUM_OK",
            Self::MetadataExtracted => "METADATA_OK",
            Self::ChecksumError(_) => "CHECKSUM_ERR",
            Self::IoError(_) => "IO_ERR",
            Self::ValidationError(_) => "VALIDATE_ERR",
            Self::Skipped(_) => "SKIPPED",
        }
    }
}

/// A single log entry recording one ingest action.
#[derive(Debug, Clone)]
pub struct IngestLogEntry {
    /// Asset path or identifier this entry refers to.
    pub asset: String,
    /// The action that was taken.
    pub action: IngestAction,
    /// Unix timestamp (seconds since epoch) when the action occurred.
    pub timestamp_secs: u64,
    /// Optional free-form note.
    pub note: Option<String>,
}

impl IngestLogEntry {
    /// Creates a new `IngestLogEntry` at the given Unix timestamp.
    #[must_use]
    pub fn new(asset: impl Into<String>, action: IngestAction, timestamp_secs: u64) -> Self {
        Self {
            asset: asset.into(),
            action,
            timestamp_secs,
            note: None,
        }
    }

    /// Creates an entry stamped with the current system time.
    ///
    /// Falls back to timestamp 0 if the system clock is unavailable.
    #[must_use]
    pub fn now(asset: impl Into<String>, action: IngestAction) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self::new(asset, action, ts)
    }

    /// Attaches a free-form note and returns `self` for method chaining.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Returns `true` if the entry is within the given number of seconds of the reference time.
    #[must_use]
    pub fn is_recent(&self, reference_secs: u64, window_secs: u64) -> bool {
        let age = reference_secs.saturating_sub(self.timestamp_secs);
        age <= window_secs
    }
}

/// Append-only log of `IngestLogEntry` records.
#[derive(Debug, Default, Clone)]
pub struct IngestLog {
    entries: Vec<IngestLogEntry>,
}

impl IngestLog {
    /// Creates an empty `IngestLog`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Appends an `IngestLogEntry` to the log.
    pub fn record(&mut self, entry: IngestLogEntry) {
        self.entries.push(entry);
    }

    /// Returns a slice of all log entries.
    #[must_use]
    pub fn all(&self) -> &[IngestLogEntry] {
        &self.entries
    }

    /// Returns a vec of references to entries whose action is an error.
    #[must_use]
    pub fn errors(&self) -> Vec<&IngestLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.action.is_error())
            .collect()
    }

    /// Returns entries recorded within `window_secs` of `reference_secs`.
    #[must_use]
    pub fn recent_entries(&self, reference_secs: u64, window_secs: u64) -> Vec<&IngestLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_recent(reference_secs, window_secs))
            .collect()
    }

    /// Returns the total number of log entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the count of error entries.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors().len()
    }

    /// Returns the count of successful (non-error) entries.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.action.is_success())
            .collect::<Vec<_>>()
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- IngestAction ---

    #[test]
    fn file_copied_is_not_error() {
        assert!(!IngestAction::FileCopied.is_error());
    }

    #[test]
    fn checksum_error_is_error() {
        assert!(IngestAction::ChecksumError("mismatch".into()).is_error());
    }

    #[test]
    fn io_error_is_error() {
        assert!(IngestAction::IoError("disk full".into()).is_error());
    }

    #[test]
    fn validation_error_is_error() {
        assert!(IngestAction::ValidationError("bad header".into()).is_error());
    }

    #[test]
    fn skipped_is_not_error() {
        assert!(!IngestAction::Skipped("duplicate".into()).is_error());
    }

    #[test]
    fn label_nonempty_for_all_variants() {
        let variants = [
            IngestAction::FileCopied,
            IngestAction::ChecksumVerified,
            IngestAction::MetadataExtracted,
            IngestAction::ChecksumError("e".into()),
            IngestAction::IoError("e".into()),
            IngestAction::ValidationError("e".into()),
            IngestAction::Skipped("s".into()),
        ];
        for v in &variants {
            assert!(!v.label().is_empty());
        }
    }

    // --- IngestLogEntry ---

    #[test]
    fn is_recent_within_window() {
        let entry = IngestLogEntry::new("asset.mxf", IngestAction::FileCopied, 1000);
        assert!(entry.is_recent(1050, 60));
    }

    #[test]
    fn is_recent_outside_window() {
        let entry = IngestLogEntry::new("asset.mxf", IngestAction::FileCopied, 900);
        assert!(!entry.is_recent(1000, 60));
    }

    #[test]
    fn entry_with_note_stores_note() {
        let entry =
            IngestLogEntry::new("a.mxf", IngestAction::FileCopied, 0).with_note("test note");
        assert_eq!(entry.note.as_deref(), Some("test note"));
    }

    // --- IngestLog ---

    #[test]
    fn empty_log() {
        let log = IngestLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn record_increments_len() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new("a.mxf", IngestAction::FileCopied, 0));
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn errors_returns_only_error_entries() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new("a.mxf", IngestAction::FileCopied, 0));
        log.record(IngestLogEntry::new(
            "b.mxf",
            IngestAction::IoError("fail".into()),
            0,
        ));
        assert_eq!(log.errors().len(), 1);
    }

    #[test]
    fn error_count_correct() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new(
            "a.mxf",
            IngestAction::ChecksumError("x".into()),
            0,
        ));
        log.record(IngestLogEntry::new("b.mxf", IngestAction::FileCopied, 0));
        assert_eq!(log.error_count(), 1);
    }

    #[test]
    fn success_count_correct() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new("a.mxf", IngestAction::FileCopied, 0));
        log.record(IngestLogEntry::new(
            "b.mxf",
            IngestAction::ChecksumVerified,
            0,
        ));
        log.record(IngestLogEntry::new(
            "c.mxf",
            IngestAction::IoError("e".into()),
            0,
        ));
        assert_eq!(log.success_count(), 2);
    }

    #[test]
    fn recent_entries_filters_by_window() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new(
            "old.mxf",
            IngestAction::FileCopied,
            100,
        ));
        log.record(IngestLogEntry::new(
            "new.mxf",
            IngestAction::FileCopied,
            950,
        ));
        let recent = log.recent_entries(1000, 60);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].asset, "new.mxf");
    }

    #[test]
    fn all_returns_all_entries() {
        let mut log = IngestLog::new();
        log.record(IngestLogEntry::new("a.mxf", IngestAction::FileCopied, 0));
        log.record(IngestLogEntry::new("b.mxf", IngestAction::FileCopied, 0));
        assert_eq!(log.all().len(), 2);
    }
}
