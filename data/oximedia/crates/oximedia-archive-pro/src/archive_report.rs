//! Archive reporting: structured event logs and summary reports.
//!
//! Every significant preservation action — ingest, migration, fixity check,
//! restore — is recorded as an [`ArchiveEvent`].  An [`ArchiveReport`]
//! aggregates a collection of events and exposes summary statistics.

#![allow(dead_code)]

use std::path::PathBuf;

/// The type of archival event that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveEventType {
    /// A new asset was ingested into the archive.
    Ingest,
    /// An existing asset was migrated to a new format.
    Migration,
    /// A fixity (checksum) check was performed.
    FixityCheck,
    /// An asset was restored from cold or deep storage.
    Restore,
    /// An asset was deleted according to a retention policy.
    Deletion,
    /// A policy was applied (tiering, retention rule, etc.).
    PolicyApplied,
    /// An integrity error was detected.
    IntegrityError,
    /// Metadata for an asset was updated.
    MetadataUpdate,
}

impl std::fmt::Display for ArchiveEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Ingest => "Ingest",
            Self::Migration => "Migration",
            Self::FixityCheck => "FixityCheck",
            Self::Restore => "Restore",
            Self::Deletion => "Deletion",
            Self::PolicyApplied => "PolicyApplied",
            Self::IntegrityError => "IntegrityError",
            Self::MetadataUpdate => "MetadataUpdate",
        };
        write!(f, "{s}")
    }
}

impl ArchiveEventType {
    /// Returns `true` for error-class events that need immediate attention.
    #[must_use]
    pub fn is_error(self) -> bool {
        matches!(self, Self::IntegrityError)
    }

    /// Returns `true` for events that modify the stored asset.
    #[must_use]
    pub fn is_mutating(self) -> bool {
        matches!(
            self,
            Self::Ingest | Self::Migration | Self::Deletion | Self::MetadataUpdate
        )
    }
}

// ---------------------------------------------------------------------------
// ArchiveEvent
// ---------------------------------------------------------------------------

/// A single recorded preservation action.
#[derive(Debug, Clone)]
pub struct ArchiveEvent {
    /// Monotonically increasing event sequence number.
    pub seq: u64,
    /// Type of event.
    pub event_type: ArchiveEventType,
    /// Asset path the event relates to.
    pub asset: PathBuf,
    /// Wall-clock timestamp as Unix epoch seconds.
    pub timestamp_secs: u64,
    /// Human-readable description.
    pub description: String,
    /// Whether the event represents a success (`true`) or failure (`false`).
    pub success: bool,
    /// Optional agent or system that generated the event.
    pub actor: Option<String>,
}

impl ArchiveEvent {
    /// Create a successful event.
    #[must_use]
    pub fn success(
        seq: u64,
        event_type: ArchiveEventType,
        asset: PathBuf,
        timestamp_secs: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            event_type,
            asset,
            timestamp_secs,
            description: description.into(),
            success: true,
            actor: None,
        }
    }

    /// Create a failure event.
    #[must_use]
    pub fn failure(
        seq: u64,
        event_type: ArchiveEventType,
        asset: PathBuf,
        timestamp_secs: u64,
        description: impl Into<String>,
    ) -> Self {
        Self {
            seq,
            event_type,
            asset,
            timestamp_secs,
            description: description.into(),
            success: false,
            actor: None,
        }
    }

    /// Attach an actor to this event.
    #[must_use]
    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }
}

impl std::fmt::Display for ArchiveEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.success { "OK" } else { "FAIL" };
        write!(
            f,
            "[{}] seq={} {} {:?} — {}",
            status, self.seq, self.event_type, self.asset, self.description
        )
    }
}

// ---------------------------------------------------------------------------
// ArchiveReport
// ---------------------------------------------------------------------------

/// Summary statistics for a collection of archive events.
#[derive(Debug, Clone)]
pub struct ReportSummary {
    /// Total number of events.
    pub total: usize,
    /// Number of successful events.
    pub successful: usize,
    /// Number of failed events.
    pub failed: usize,
    /// Number of integrity-error events.
    pub integrity_errors: usize,
    /// Number of distinct assets referenced.
    pub distinct_assets: usize,
    /// Earliest event timestamp (Unix secs), or `None` if empty.
    pub earliest_ts: Option<u64>,
    /// Latest event timestamp (Unix secs), or `None` if empty.
    pub latest_ts: Option<u64>,
}

/// Aggregates archive events and generates reports.
pub struct ArchiveReport {
    events: Vec<ArchiveEvent>,
    next_seq: u64,
}

impl ArchiveReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_seq: 1,
        }
    }

    /// Record an event, assigning the next sequence number automatically.
    pub fn record(
        &mut self,
        event_type: ArchiveEventType,
        asset: PathBuf,
        timestamp_secs: u64,
        description: impl Into<String>,
        success: bool,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let evt = if success {
            ArchiveEvent::success(seq, event_type, asset, timestamp_secs, description)
        } else {
            ArchiveEvent::failure(seq, event_type, asset, timestamp_secs, description)
        };
        self.events.push(evt);
        seq
    }

    /// Add a pre-built event, re-assigning its sequence number.
    pub fn push(&mut self, mut event: ArchiveEvent) {
        event.seq = self.next_seq;
        self.next_seq += 1;
        self.events.push(event);
    }

    /// Compute a summary over all recorded events.
    #[must_use]
    pub fn summary(&self) -> ReportSummary {
        let total = self.events.len();
        let successful = self.events.iter().filter(|e| e.success).count();
        let failed = total - successful;
        let integrity_errors = self
            .events
            .iter()
            .filter(|e| e.event_type == ArchiveEventType::IntegrityError)
            .count();

        let mut paths: Vec<&PathBuf> = self.events.iter().map(|e| &e.asset).collect();
        paths.sort();
        paths.dedup();
        let distinct_assets = paths.len();

        let earliest_ts = self.events.iter().map(|e| e.timestamp_secs).min();
        let latest_ts = self.events.iter().map(|e| e.timestamp_secs).max();

        ReportSummary {
            total,
            successful,
            failed,
            integrity_errors,
            distinct_assets,
            earliest_ts,
            latest_ts,
        }
    }

    /// Return events filtered by type.
    #[must_use]
    pub fn events_of_type(&self, t: ArchiveEventType) -> Vec<&ArchiveEvent> {
        self.events.iter().filter(|e| e.event_type == t).collect()
    }

    /// Return all failure events.
    #[must_use]
    pub fn failures(&self) -> Vec<&ArchiveEvent> {
        self.events.iter().filter(|e| !e.success).collect()
    }

    /// Return all events in chronological order (by timestamp, then seq).
    #[must_use]
    pub fn chronological(&self) -> Vec<&ArchiveEvent> {
        let mut sorted: Vec<&ArchiveEvent> = self.events.iter().collect();
        sorted.sort_by_key(|e| (e.timestamp_secs, e.seq));
        sorted
    }

    /// Total number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

impl Default for ArchiveReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str) -> PathBuf {
        PathBuf::from(format!("/archive/{name}"))
    }

    fn populate_report() -> ArchiveReport {
        let mut r = ArchiveReport::new();
        r.record(
            ArchiveEventType::Ingest,
            asset("a.mkv"),
            1000,
            "Ingested",
            true,
        );
        r.record(
            ArchiveEventType::FixityCheck,
            asset("a.mkv"),
            2000,
            "Checksum OK",
            true,
        );
        r.record(
            ArchiveEventType::IntegrityError,
            asset("b.avi"),
            3000,
            "Bit rot",
            false,
        );
        r.record(
            ArchiveEventType::Migration,
            asset("b.avi"),
            4000,
            "Migrated to MKV",
            true,
        );
        r
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(ArchiveEventType::Ingest.to_string(), "Ingest");
        assert_eq!(
            ArchiveEventType::IntegrityError.to_string(),
            "IntegrityError"
        );
    }

    #[test]
    fn test_event_type_is_error() {
        assert!(ArchiveEventType::IntegrityError.is_error());
        assert!(!ArchiveEventType::Ingest.is_error());
    }

    #[test]
    fn test_event_type_is_mutating() {
        assert!(ArchiveEventType::Ingest.is_mutating());
        assert!(ArchiveEventType::Deletion.is_mutating());
        assert!(!ArchiveEventType::FixityCheck.is_mutating());
    }

    #[test]
    fn test_event_success_display() {
        let e = ArchiveEvent::success(1, ArchiveEventType::Ingest, asset("x.mkv"), 0, "ok");
        let s = e.to_string();
        assert!(s.contains("OK"));
        assert!(s.contains("Ingest"));
    }

    #[test]
    fn test_event_failure_display() {
        let e = ArchiveEvent::failure(
            2,
            ArchiveEventType::IntegrityError,
            asset("y.avi"),
            0,
            "bad",
        );
        let s = e.to_string();
        assert!(s.contains("FAIL"));
    }

    #[test]
    fn test_event_with_actor() {
        let e = ArchiveEvent::success(1, ArchiveEventType::Ingest, asset("z.mkv"), 0, "ok")
            .with_actor("archivist-bot");
        assert_eq!(e.actor.as_deref(), Some("archivist-bot"));
    }

    #[test]
    fn test_report_record_assigns_seq() {
        let mut r = ArchiveReport::new();
        let s1 = r.record(ArchiveEventType::Ingest, asset("a.mkv"), 0, "ok", true);
        let s2 = r.record(ArchiveEventType::Ingest, asset("b.mkv"), 0, "ok", true);
        assert_eq!(s2, s1 + 1);
    }

    #[test]
    fn test_report_event_count() {
        let r = populate_report();
        assert_eq!(r.event_count(), 4);
    }

    #[test]
    fn test_summary_totals() {
        let r = populate_report();
        let s = r.summary();
        assert_eq!(s.total, 4);
        assert_eq!(s.successful, 3);
        assert_eq!(s.failed, 1);
    }

    #[test]
    fn test_summary_integrity_errors() {
        let r = populate_report();
        assert_eq!(r.summary().integrity_errors, 1);
    }

    #[test]
    fn test_summary_distinct_assets() {
        let r = populate_report();
        assert_eq!(r.summary().distinct_assets, 2);
    }

    #[test]
    fn test_summary_timestamps() {
        let r = populate_report();
        let s = r.summary();
        assert_eq!(s.earliest_ts, Some(1000));
        assert_eq!(s.latest_ts, Some(4000));
    }

    #[test]
    fn test_summary_empty_report() {
        let r = ArchiveReport::new();
        let s = r.summary();
        assert_eq!(s.total, 0);
        assert!(s.earliest_ts.is_none());
        assert!(s.latest_ts.is_none());
    }

    #[test]
    fn test_events_of_type() {
        let r = populate_report();
        let checks = r.events_of_type(ArchiveEventType::FixityCheck);
        assert_eq!(checks.len(), 1);
    }

    #[test]
    fn test_failures() {
        let r = populate_report();
        let fails = r.failures();
        assert_eq!(fails.len(), 1);
        assert!(!fails[0].success);
    }

    #[test]
    fn test_chronological_order() {
        let mut r = ArchiveReport::new();
        r.record(ArchiveEventType::Ingest, asset("a.mkv"), 5000, "late", true);
        r.record(
            ArchiveEventType::Ingest,
            asset("b.mkv"),
            1000,
            "early",
            true,
        );
        let sorted = r.chronological();
        assert_eq!(sorted[0].timestamp_secs, 1000);
        assert_eq!(sorted[1].timestamp_secs, 5000);
    }

    #[test]
    fn test_push_pre_built_event() {
        let mut r = ArchiveReport::new();
        let e = ArchiveEvent::success(999, ArchiveEventType::Restore, asset("c.mkv"), 100, "done");
        r.push(e);
        assert_eq!(r.event_count(), 1);
        // seq should be re-assigned to 1
        assert_eq!(r.events_of_type(ArchiveEventType::Restore)[0].seq, 1);
    }
}
