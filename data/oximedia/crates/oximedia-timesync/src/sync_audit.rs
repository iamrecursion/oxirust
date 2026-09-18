//! Synchronization audit logging and quality metrics.
//!
//! This module records synchronization events, quality snapshots, and alarm
//! conditions for compliance and diagnostics. It maintains a rolling audit log
//! with configurable retention.
//!
//! File persistence is provided by [`FileAuditLogger`], which writes
//! JSON-lines format entries and supports size-triggered rotation.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// Category of a synchronization audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditEventKind {
    /// Clock synchronized to reference.
    LockAcquired,
    /// Clock lost synchronization.
    LockLost,
    /// Phase step was applied.
    PhaseStep,
    /// Frequency adjustment was applied.
    FrequencyAdjust,
    /// Holdover mode entered.
    HoldoverEnter,
    /// Holdover mode exited.
    HoldoverExit,
    /// Reference source changed.
    SourceChange,
    /// Quality threshold alarm triggered.
    QualityAlarm,
    /// Manual time set was performed.
    ManualSet,
    /// Configuration was changed.
    ConfigChange,
}

impl AuditEventKind {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::LockAcquired => "Lock Acquired",
            Self::LockLost => "Lock Lost",
            Self::PhaseStep => "Phase Step",
            Self::FrequencyAdjust => "Frequency Adjust",
            Self::HoldoverEnter => "Holdover Enter",
            Self::HoldoverExit => "Holdover Exit",
            Self::SourceChange => "Source Change",
            Self::QualityAlarm => "Quality Alarm",
            Self::ManualSet => "Manual Set",
            Self::ConfigChange => "Config Change",
        }
    }

    /// Returns `true` if this event indicates a problem.
    #[must_use]
    pub const fn is_alarm(&self) -> bool {
        matches!(
            self,
            Self::LockLost | Self::QualityAlarm | Self::HoldoverEnter
        )
    }
}

/// A single audit event record.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Elapsed seconds since the audit log started (monotonic).
    pub elapsed_secs: f64,
    /// The kind of event.
    pub kind: AuditEventKind,
    /// Measured offset in nanoseconds at the time of the event.
    pub offset_ns: f64,
    /// Optional detail message.
    pub detail: String,
}

impl AuditEvent {
    /// Creates a new audit event.
    #[must_use]
    pub fn new(seq: u64, elapsed_secs: f64, kind: AuditEventKind, offset_ns: f64) -> Self {
        Self {
            seq,
            elapsed_secs,
            kind,
            offset_ns,
            detail: String::new(),
        }
    }

    /// Adds a detail message.
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = detail.to_string();
        self
    }
}

/// Quality snapshot recorded periodically.
#[derive(Debug, Clone, Copy)]
pub struct QualitySnapshot {
    /// Elapsed seconds since log start.
    pub elapsed_secs: f64,
    /// Mean offset in nanoseconds.
    pub mean_offset_ns: f64,
    /// Maximum offset in nanoseconds.
    pub max_offset_ns: f64,
    /// Drift rate in nanoseconds per second.
    pub drift_rate_ns_per_sec: f64,
    /// Whether the clock is currently locked.
    pub locked: bool,
}

/// Alarm threshold configuration.
#[derive(Debug, Clone, Copy)]
pub struct AlarmThresholds {
    /// Maximum acceptable offset in nanoseconds.
    pub max_offset_ns: f64,
    /// Maximum acceptable drift rate in ns/s.
    pub max_drift_ns_per_sec: f64,
    /// Maximum holdover duration in seconds before alarm.
    pub max_holdover_secs: f64,
}

impl Default for AlarmThresholds {
    fn default() -> Self {
        Self {
            max_offset_ns: 100_000.0,      // 100 us
            max_drift_ns_per_sec: 1_000.0, // 1 us/s
            max_holdover_secs: 60.0,       // 1 minute
        }
    }
}

/// Rolling audit log for synchronization events.
#[derive(Debug)]
pub struct SyncAuditLog {
    /// Maximum number of events to retain.
    max_events: usize,
    /// Event log.
    events: VecDeque<AuditEvent>,
    /// Quality snapshots.
    snapshots: VecDeque<QualitySnapshot>,
    /// Maximum number of snapshots to retain.
    max_snapshots: usize,
    /// Next sequence number.
    next_seq: u64,
    /// Alarm thresholds.
    thresholds: AlarmThresholds,
    /// Number of alarm events triggered.
    alarm_count: u64,
}

impl SyncAuditLog {
    /// Creates a new audit log with the given capacity.
    #[must_use]
    pub fn new(max_events: usize, max_snapshots: usize) -> Self {
        Self {
            max_events,
            events: VecDeque::with_capacity(max_events),
            snapshots: VecDeque::with_capacity(max_snapshots),
            max_snapshots,
            next_seq: 1,
            thresholds: AlarmThresholds::default(),
            alarm_count: 0,
        }
    }

    /// Sets alarm thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: AlarmThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Records an audit event.
    pub fn record_event(
        &mut self,
        elapsed_secs: f64,
        kind: AuditEventKind,
        offset_ns: f64,
        detail: &str,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        if kind.is_alarm() {
            self.alarm_count += 1;
        }

        let event = AuditEvent::new(seq, elapsed_secs, kind, offset_ns).with_detail(detail);

        if self.events.len() >= self.max_events {
            self.events.pop_front();
        }
        self.events.push_back(event);
        seq
    }

    /// Records a quality snapshot.
    pub fn record_snapshot(&mut self, snapshot: QualitySnapshot) {
        if self.snapshots.len() >= self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Checks the given metrics against alarm thresholds and records alarms.
    pub fn check_alarms(&mut self, elapsed_secs: f64, offset_ns: f64, drift_ns_per_sec: f64) {
        if offset_ns.abs() > self.thresholds.max_offset_ns {
            self.record_event(
                elapsed_secs,
                AuditEventKind::QualityAlarm,
                offset_ns,
                &format!(
                    "Offset {offset_ns:.0} ns exceeds threshold {:.0} ns",
                    self.thresholds.max_offset_ns
                ),
            );
        }
        if drift_ns_per_sec.abs() > self.thresholds.max_drift_ns_per_sec {
            self.record_event(
                elapsed_secs,
                AuditEventKind::QualityAlarm,
                offset_ns,
                &format!(
                    "Drift {drift_ns_per_sec:.1} ns/s exceeds threshold {:.1} ns/s",
                    self.thresholds.max_drift_ns_per_sec
                ),
            );
        }
    }

    /// Returns the total number of events recorded (including evicted ones).
    #[must_use]
    pub fn total_events_recorded(&self) -> u64 {
        self.next_seq - 1
    }

    /// Returns the number of events currently in the log.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns the number of snapshots currently in the log.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns the number of alarm events.
    #[must_use]
    pub fn alarm_count(&self) -> u64 {
        self.alarm_count
    }

    /// Returns a slice of all current events.
    #[must_use]
    pub fn events(&self) -> &VecDeque<AuditEvent> {
        &self.events
    }

    /// Returns events of a given kind.
    #[must_use]
    pub fn events_of_kind(&self, kind: AuditEventKind) -> Vec<&AuditEvent> {
        self.events.iter().filter(|e| e.kind == kind).collect()
    }

    /// Returns the most recent event, if any.
    #[must_use]
    pub fn latest_event(&self) -> Option<&AuditEvent> {
        self.events.back()
    }

    /// Returns the most recent quality snapshot, if any.
    #[must_use]
    pub fn latest_snapshot(&self) -> Option<&QualitySnapshot> {
        self.snapshots.back()
    }

    /// Clears all events and snapshots.
    pub fn clear(&mut self) {
        self.events.clear();
        self.snapshots.clear();
        self.alarm_count = 0;
        self.next_seq = 1;
    }

    /// Returns `true` if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.snapshots.is_empty()
    }
}

impl Default for SyncAuditLog {
    fn default() -> Self {
        Self::new(1000, 100)
    }
}

// ---------------------------------------------------------------------------
// Persistent file-based audit logger
// ---------------------------------------------------------------------------

/// A serialisable audit entry suitable for JSON-lines persistence.
///
/// This mirrors [`AuditEvent`] but uses only `serde`-friendly primitive types
/// so it can be written to a file without pulling in `Instant` serialisation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditEntry {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Elapsed seconds since the audit log started.
    pub elapsed_secs: f64,
    /// Event kind as a string label.
    pub kind: String,
    /// Measured offset in nanoseconds.
    pub offset_ns: f64,
    /// Optional detail message.
    pub detail: String,
}

impl From<&AuditEvent> for AuditEntry {
    fn from(ev: &AuditEvent) -> Self {
        Self {
            seq: ev.seq,
            elapsed_secs: ev.elapsed_secs,
            kind: ev.kind.label().to_string(),
            offset_ns: ev.offset_ns,
            detail: ev.detail.clone(),
        }
    }
}

/// File-based audit logger that writes one JSON object per line
/// (JSON-lines / NDJSON format) and rotates when the file exceeds
/// `max_size_bytes`.
///
/// # Rotation
/// When [`rotate`](FileAuditLogger::rotate) is called (or triggered
/// automatically when `max_size_bytes` is exceeded after an append), the
/// current log file is renamed to `<path>.bak`, replacing any previous backup,
/// and a fresh log file is opened at `<path>`.
pub struct FileAuditLogger {
    /// Destination path for the active log.
    path: PathBuf,
    /// Size threshold (in bytes) that triggers automatic rotation.
    max_size_bytes: u64,
    /// Buffered writer to the active log file.
    writer: BufWriter<File>,
    /// Approximate number of bytes written to the current file.
    bytes_written: u64,
}

impl FileAuditLogger {
    /// Opens (or creates) the log file at `path` and prepares for appending.
    ///
    /// Returns `Err` if the file cannot be opened or if metadata cannot be
    /// read to determine the current file size.
    pub fn new(path: impl Into<PathBuf>, max_size_bytes: u64) -> Result<Self, std::io::Error> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let bytes_written = file.metadata()?.len();
        let writer = BufWriter::new(file);
        Ok(Self {
            path,
            max_size_bytes,
            writer,
            bytes_written,
        })
    }

    /// Serialises `entry` as a single JSON line and appends it to the log.
    ///
    /// Flushes the internal buffer after every write so that entries are
    /// durable.  If the file size would exceed `max_size_bytes` after the
    /// write, [`rotate`](FileAuditLogger::rotate) is called automatically.
    pub fn append(&mut self, entry: &AuditEntry) -> Result<(), std::io::Error> {
        let line = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let bytes = line.as_bytes();
        self.writer.write_all(bytes)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.bytes_written += bytes.len() as u64 + 1;

        // Auto-rotate when size threshold is exceeded.
        if self.bytes_written > self.max_size_bytes {
            self.rotate()?;
        }

        Ok(())
    }

    /// Renames the current log file to `<path>.bak` and opens a fresh log at
    /// `<path>`.
    ///
    /// Any existing `.bak` file is silently overwritten.  Returns `Err` if
    /// the rename or re-open fails.
    pub fn rotate(&mut self) -> Result<(), std::io::Error> {
        // Flush before renaming.
        self.writer.flush()?;

        let mut bak = self.path.clone();
        let ext = bak
            .extension()
            .map(|e| {
                let mut s = e.to_os_string();
                s.push(".bak");
                s
            })
            .unwrap_or_else(|| std::ffi::OsString::from("bak"));
        bak.set_extension(ext);

        std::fs::rename(&self.path, &bak)?;

        // Open a fresh file.
        let new_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.bytes_written = 0;
        self.writer = BufWriter::new(new_file);

        Ok(())
    }

    /// Returns the approximate number of bytes written to the current log
    /// file since it was opened or last rotated.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Returns a reference to the active log file path.
    #[must_use]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_kind_labels() {
        assert_eq!(AuditEventKind::LockAcquired.label(), "Lock Acquired");
        assert_eq!(AuditEventKind::PhaseStep.label(), "Phase Step");
        assert_eq!(AuditEventKind::HoldoverEnter.label(), "Holdover Enter");
    }

    #[test]
    fn test_event_kind_is_alarm() {
        assert!(AuditEventKind::LockLost.is_alarm());
        assert!(AuditEventKind::QualityAlarm.is_alarm());
        assert!(AuditEventKind::HoldoverEnter.is_alarm());
        assert!(!AuditEventKind::LockAcquired.is_alarm());
        assert!(!AuditEventKind::FrequencyAdjust.is_alarm());
    }

    #[test]
    fn test_new_audit_log() {
        let log = SyncAuditLog::new(100, 50);
        assert_eq!(log.event_count(), 0);
        assert_eq!(log.snapshot_count(), 0);
        assert_eq!(log.alarm_count(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_record_event() {
        let mut log = SyncAuditLog::new(100, 50);
        let seq = log.record_event(1.0, AuditEventKind::LockAcquired, 50.0, "PTP lock");
        assert_eq!(seq, 1);
        assert_eq!(log.event_count(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_event_eviction() {
        let mut log = SyncAuditLog::new(3, 10);
        log.record_event(1.0, AuditEventKind::LockAcquired, 0.0, "");
        log.record_event(2.0, AuditEventKind::FrequencyAdjust, 10.0, "");
        log.record_event(3.0, AuditEventKind::PhaseStep, 20.0, "");
        log.record_event(4.0, AuditEventKind::LockLost, 500.0, "");
        assert_eq!(log.event_count(), 3);
        assert_eq!(log.total_events_recorded(), 4);
    }

    #[test]
    fn test_alarm_count() {
        let mut log = SyncAuditLog::new(100, 50);
        log.record_event(1.0, AuditEventKind::LockAcquired, 0.0, "");
        log.record_event(2.0, AuditEventKind::LockLost, 500.0, "");
        log.record_event(3.0, AuditEventKind::QualityAlarm, 200_000.0, "");
        assert_eq!(log.alarm_count(), 2);
    }

    #[test]
    fn test_events_of_kind() {
        let mut log = SyncAuditLog::new(100, 50);
        log.record_event(1.0, AuditEventKind::PhaseStep, 100.0, "step1");
        log.record_event(2.0, AuditEventKind::LockAcquired, 0.0, "lock");
        log.record_event(3.0, AuditEventKind::PhaseStep, 200.0, "step2");
        let steps = log.events_of_kind(AuditEventKind::PhaseStep);
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_latest_event() {
        let mut log = SyncAuditLog::new(100, 50);
        assert!(log.latest_event().is_none());
        log.record_event(1.0, AuditEventKind::LockAcquired, 0.0, "first");
        log.record_event(2.0, AuditEventKind::LockLost, 100.0, "second");
        let latest = log.latest_event().expect("should succeed in test");
        assert_eq!(latest.kind, AuditEventKind::LockLost);
        assert_eq!(latest.detail, "second");
    }

    #[test]
    fn test_record_snapshot() {
        let mut log = SyncAuditLog::new(100, 3);
        for i in 0..5 {
            #[allow(clippy::cast_precision_loss)]
            let secs = i as f64;
            log.record_snapshot(QualitySnapshot {
                elapsed_secs: secs,
                mean_offset_ns: 50.0,
                max_offset_ns: 100.0,
                drift_rate_ns_per_sec: 5.0,
                locked: true,
            });
        }
        assert_eq!(log.snapshot_count(), 3); // capped at max
    }

    #[test]
    fn test_check_alarms_offset() {
        let mut log = SyncAuditLog::new(100, 50);
        log.check_alarms(1.0, 200_000.0, 0.0); // exceeds 100us threshold
        assert_eq!(log.alarm_count(), 1);
    }

    #[test]
    fn test_check_alarms_drift() {
        let mut log = SyncAuditLog::new(100, 50);
        log.check_alarms(1.0, 50.0, 5_000.0); // exceeds 1us/s threshold
        assert_eq!(log.alarm_count(), 1);
    }

    #[test]
    fn test_check_alarms_no_alarm() {
        let mut log = SyncAuditLog::new(100, 50);
        log.check_alarms(1.0, 50.0, 100.0); // within thresholds
        assert_eq!(log.alarm_count(), 0);
    }

    #[test]
    fn test_clear() {
        let mut log = SyncAuditLog::new(100, 50);
        log.record_event(1.0, AuditEventKind::LockLost, 100.0, "");
        log.record_snapshot(QualitySnapshot {
            elapsed_secs: 1.0,
            mean_offset_ns: 50.0,
            max_offset_ns: 100.0,
            drift_rate_ns_per_sec: 5.0,
            locked: false,
        });
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.alarm_count(), 0);
    }

    // -----------------------------------------------------------------------
    // FileAuditLogger tests
    // -----------------------------------------------------------------------

    fn make_entry(seq: u64) -> AuditEntry {
        AuditEntry {
            seq,
            elapsed_secs: seq as f64 * 0.1,
            kind: "Lock Acquired".to_string(),
            offset_ns: seq as f64 * 10.0,
            detail: format!("entry {seq}"),
        }
    }

    #[test]
    fn test_file_logger_write_100_entries() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_sync_audit_100.jsonl");
        // Remove any leftover from a previous run.
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = FileAuditLogger::new(&path, 1_000_000).expect("should create logger");
            for i in 0u64..100 {
                logger
                    .append(&make_entry(i))
                    .expect("append should succeed");
            }
        }

        // Verify the file contains 100 lines.
        let content = std::fs::read_to_string(&path).expect("should read log file");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 100, "should have written 100 lines");

        // Verify each line is valid JSON.
        for line in &lines {
            let parsed: serde_json::Value =
                serde_json::from_str(line).expect("each line should be valid JSON");
            assert!(parsed.get("seq").is_some(), "entry should have 'seq' field");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_file_logger_from_audit_event() {
        // Verify conversion from AuditEvent to AuditEntry.
        let ev =
            AuditEvent::new(42, 3.14, AuditEventKind::PhaseStep, 12345.0).with_detail("test step");
        let entry = AuditEntry::from(&ev);
        assert_eq!(entry.seq, 42);
        assert_eq!(entry.kind, "Phase Step");
        assert_eq!(entry.detail, "test step");
    }

    #[test]
    fn test_file_logger_rotation_trigger() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_sync_audit_rotate.jsonl");
        let bak_path = dir.join("test_sync_audit_rotate.jsonl.bak");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak_path);

        // Set a very small threshold so rotation is triggered quickly.
        let max_bytes: u64 = 200;
        {
            let mut logger = FileAuditLogger::new(&path, max_bytes).expect("should create logger");
            // Write enough entries to exceed the threshold.
            for i in 0u64..20 {
                logger
                    .append(&make_entry(i))
                    .expect("append should succeed");
            }
        }

        // After rotation the .bak file should exist.
        assert!(bak_path.exists(), "backup file should exist after rotation");
        // The primary log should also still exist (re-opened after rotate).
        assert!(path.exists(), "primary log should exist after rotation");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak_path);
    }

    #[test]
    fn test_file_logger_manual_rotate() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_sync_audit_manual_rotate.jsonl");
        let bak_path = dir.join("test_sync_audit_manual_rotate.jsonl.bak");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak_path);

        {
            let mut logger = FileAuditLogger::new(&path, 1_000_000).expect("should create logger");
            for i in 0u64..5 {
                logger.append(&make_entry(i)).expect("append");
            }
            // Manual rotate.
            logger.rotate().expect("rotate should succeed");
            // Write more entries after rotation.
            for i in 5u64..8 {
                logger.append(&make_entry(i)).expect("append after rotate");
            }
        }

        // Backup should contain the first 5 entries.
        let bak_content = std::fs::read_to_string(&bak_path).expect("bak should be readable");
        assert_eq!(
            bak_content.lines().count(),
            5,
            "backup should contain 5 pre-rotation entries"
        );

        // Primary should contain the 3 post-rotation entries.
        let primary_content = std::fs::read_to_string(&path).expect("primary should be readable");
        assert_eq!(
            primary_content.lines().count(),
            3,
            "primary should contain 3 post-rotation entries"
        );

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bak_path);
    }
}
