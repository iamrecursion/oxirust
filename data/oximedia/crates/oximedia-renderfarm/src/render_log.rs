//! Render farm logging — log levels, individual entries, a bounded in-memory
//! log store, and a batching, file-backed persistence writer.
//!
//! ## Batched persistence
//!
//! A render job can emit tens of thousands of frame-level log entries. Writing
//! each entry to durable storage individually (one syscall / one DB round-trip
//! per frame) turns logging into a bottleneck at scale. [`BatchedRenderLogWriter`]
//! solves this by accumulating entries in an in-memory buffer and emitting them
//! in batches — a single append write per flush instead of one write per frame.
//! See the type-level documentation for the durability trade-offs.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Severity level of a render log entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RenderLogLevel {
    /// Verbose diagnostic information.
    Debug,
    /// Normal operational information.
    Info,
    /// Potential issue that does not stop rendering.
    Warning,
    /// Recoverable error condition.
    Error,
    /// Unrecoverable failure requiring immediate attention.
    Critical,
}

impl RenderLogLevel {
    /// Returns `true` for levels that indicate a problem (`Warning` and above).
    #[must_use]
    pub fn is_problem(&self) -> bool {
        matches!(self, Self::Warning | Self::Error | Self::Critical)
    }

    /// Returns a numeric code for the level (Debug=0 … Critical=4).
    #[must_use]
    pub fn code(&self) -> u8 {
        match self {
            Self::Debug => 0,
            Self::Info => 1,
            Self::Warning => 2,
            Self::Error => 3,
            Self::Critical => 4,
        }
    }
}

/// A single entry in the render log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderLogEntry {
    /// ID of the render job this entry belongs to.
    pub job_id: u64,
    /// Optional frame number (absent for job-level messages).
    pub frame: Option<u32>,
    /// Severity level.
    pub level: RenderLogLevel,
    /// Human-readable message.
    pub message: String,
    /// Unix epoch timestamp (seconds).
    pub timestamp_epoch: u64,
}

impl RenderLogEntry {
    /// Creates a new job-level log entry (no frame number).
    #[must_use]
    pub fn new(job_id: u64, level: RenderLogLevel, msg: impl Into<String>, epoch: u64) -> Self {
        Self {
            job_id,
            frame: None,
            level,
            message: msg.into(),
            timestamp_epoch: epoch,
        }
    }

    /// Creates a new frame-level log entry.
    #[must_use]
    pub fn with_frame(
        job_id: u64,
        frame: u32,
        level: RenderLogLevel,
        msg: impl Into<String>,
        epoch: u64,
    ) -> Self {
        Self {
            job_id,
            frame: Some(frame),
            level,
            message: msg.into(),
            timestamp_epoch: epoch,
        }
    }
}

/// A bounded, append-only collection of [`RenderLogEntry`] entries.
///
/// When the capacity is reached, the oldest entry is dropped to make room.
#[derive(Debug)]
pub struct RenderLog {
    entries: Vec<RenderLogEntry>,
    max_entries: usize,
}

impl RenderLog {
    /// Creates a new `RenderLog` with the given maximum capacity.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Appends an entry.  If at capacity, the oldest entry is discarded.
    pub fn add(&mut self, entry: RenderLogEntry) {
        if self.max_entries == 0 {
            return;
        }
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Returns all `Error` and `Critical` entries belonging to `id`.
    #[must_use]
    pub fn errors_for_job(&self, id: u64) -> Vec<&RenderLogEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.job_id == id
                    && matches!(e.level, RenderLogLevel::Error | RenderLogLevel::Critical)
            })
            .collect()
    }

    /// Returns all entries at the `Warning` level (across all jobs).
    #[must_use]
    pub fn warnings(&self) -> Vec<&RenderLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.level == RenderLogLevel::Warning)
            .collect()
    }

    /// Returns the current number of entries in the log.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Batched, file-backed persistence
// ---------------------------------------------------------------------------

/// Default number of buffered entries that triggers an automatic flush.
pub const DEFAULT_LOG_BATCH_SIZE: usize = 128;

/// Configuration for a [`BatchedRenderLogWriter`].
#[derive(Debug, Clone)]
pub struct RenderLogWriterConfig {
    /// Destination file. Entries are appended as newline-delimited JSON (JSONL),
    /// one JSON object per line, so the file is both append-friendly and
    /// streamable back one entry at a time.
    pub path: PathBuf,
    /// Number of buffered entries that triggers an automatic flush.
    ///
    /// Effectively clamped to at least `1`. A larger value amortizes more
    /// entries per append write, but also widens the window of entries that are
    /// lost if the process crashes before the next flush.
    pub batch_size: usize,
    /// When `true`, every flush additionally calls [`File::sync_all`] (fsync) so
    /// a completed batch survives an OS-level crash or power loss, not just a
    /// process crash. This is markedly slower; leave it `false` (the default)
    /// unless per-batch power-loss durability is required.
    pub sync_on_flush: bool,
}

impl RenderLogWriterConfig {
    /// Creates a configuration for `path` using [`DEFAULT_LOG_BATCH_SIZE`] and
    /// no fsync-on-flush.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            batch_size: DEFAULT_LOG_BATCH_SIZE,
            sync_on_flush: false,
        }
    }

    /// Overrides the batch size (applied with a floor of `1`).
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enables or disables fsync after every flush.
    #[must_use]
    pub fn with_sync_on_flush(mut self, sync_on_flush: bool) -> Self {
        self.sync_on_flush = sync_on_flush;
        self
    }
}

/// A batching, file-backed writer for [`RenderLogEntry`] records.
///
/// # Why batch?
///
/// A naive render log performs one durable write — one syscall, or one database
/// round-trip — per frame. A single job can emit tens of thousands of
/// frame-level entries, turning logging into a syscall storm. This writer
/// accumulates entries in an in-memory buffer and emits them in batches: each
/// flush serializes the whole buffer and performs a **single** `write_all`
/// append, turning `N` per-frame [`log_frame`](Self::log_frame) calls into
/// `ceil(N / batch_size)` append writes.
///
/// # Durability (read this)
///
/// Buffered entries live only in memory until a flush. They are written to the
/// file when:
///
/// * the buffer reaches `batch_size` (automatic threshold flush),
/// * [`flush`](Self::flush) or [`flush_and_sync`](Self::flush_and_sync) is
///   called explicitly, or
/// * the writer is dropped (best-effort flush; any error is swallowed because
///   [`Drop`] cannot return one).
///
/// **A crash, panic, or power loss before a flush loses every still-buffered
/// entry.** This is the fundamental trade-off batching makes in exchange for
/// throughput, and it is not hidden: callers that need a specific entry
/// persisted immediately (for example a `Critical` failure record) must call
/// [`flush`](Self::flush) right after logging it.
///
/// A plain [`flush`](Self::flush) guarantees the bytes have left this process
/// for the kernel, so the batch survives a process crash. It does **not** force
/// the kernel to write through to the physical device, so an OS crash or power
/// loss immediately afterward may still lose it. For that stronger guarantee,
/// enable `sync_on_flush` or call [`flush_and_sync`](Self::flush_and_sync),
/// which additionally fsyncs.
///
/// The writer is single-owner (`&mut self` for logging); share it across
/// threads by wrapping it in a `Mutex`.
#[derive(Debug)]
pub struct BatchedRenderLogWriter {
    file: File,
    config: RenderLogWriterConfig,
    buffer: Vec<RenderLogEntry>,
    physical_writes: u64,
    entries_written: u64,
}

impl BatchedRenderLogWriter {
    /// Opens (creating if necessary) the log file at `path` in append mode,
    /// using [`DEFAULT_LOG_BATCH_SIZE`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Io`] if the file cannot be opened or
    /// created.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        Self::with_config(RenderLogWriterConfig::new(path))
    }

    /// Opens (creating if necessary) the log file described by `config`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Io`] if the file cannot be opened or
    /// created.
    pub fn with_config(config: RenderLogWriterConfig) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.path)?;
        let batch_size = config.batch_size.max(1);
        Ok(Self {
            file,
            config: RenderLogWriterConfig {
                batch_size,
                ..config
            },
            buffer: Vec::new(),
            physical_writes: 0,
            entries_written: 0,
        })
    }

    /// The effective batch size (always at least `1`).
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    /// Number of entries currently buffered in memory and not yet written.
    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` when no entries are currently buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Number of batch append operations performed so far (one per non-empty
    /// flush).
    ///
    /// This is the cost metric batching is designed to minimize: it is always
    /// `<=` the number of logged entries, and strictly less whenever any batch
    /// carried more than one entry.
    #[must_use]
    pub fn physical_writes(&self) -> u64 {
        self.physical_writes
    }

    /// Total number of entries persisted to the file so far (excludes entries
    /// still sitting in the buffer).
    #[must_use]
    pub fn entries_written(&self) -> u64 {
        self.entries_written
    }

    /// Buffers `entry`, flushing automatically once the buffer reaches
    /// `batch_size`.
    ///
    /// # Errors
    ///
    /// Propagates any I/O or serialization error from an automatic flush.
    pub fn log_entry(&mut self, entry: RenderLogEntry) -> Result<()> {
        self.buffer.push(entry);
        if self.buffer.len() >= self.config.batch_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Buffers a frame-level entry. Convenience wrapper over
    /// [`log_entry`](Self::log_entry); callable once per frame.
    ///
    /// # Errors
    ///
    /// Propagates any I/O or serialization error from an automatic flush.
    pub fn log_frame(
        &mut self,
        job_id: u64,
        frame: u32,
        level: RenderLogLevel,
        message: impl Into<String>,
        timestamp_epoch: u64,
    ) -> Result<()> {
        self.log_entry(RenderLogEntry::with_frame(
            job_id,
            frame,
            level,
            message,
            timestamp_epoch,
        ))
    }

    /// Buffers a job-level entry (no frame number).
    ///
    /// # Errors
    ///
    /// Propagates any I/O or serialization error from an automatic flush.
    pub fn log_job(
        &mut self,
        job_id: u64,
        level: RenderLogLevel,
        message: impl Into<String>,
        timestamp_epoch: u64,
    ) -> Result<()> {
        self.log_entry(RenderLogEntry::new(job_id, level, message, timestamp_epoch))
    }

    /// Writes every buffered entry to the file in a single append, then clears
    /// the buffer. A no-op (no append, no counter change) when the buffer is
    /// empty.
    ///
    /// On `Ok`, the entries have been handed to the kernel and survive a process
    /// crash. Power-loss durability additionally requires an fsync — see
    /// [`flush_and_sync`](Self::flush_and_sync) or the `sync_on_flush` option.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Serialization`] if an entry cannot be
    /// encoded, or [`crate::error::Error::Io`] if the append (or fsync) fails.
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        // Serialize the whole batch into one payload so the batch becomes a
        // single `write_all` append rather than one write per entry.
        let mut payload = String::new();
        for entry in &self.buffer {
            let line = serde_json::to_string(entry)?;
            payload.push_str(&line);
            payload.push('\n');
        }
        self.file.write_all(payload.as_bytes())?;
        if self.config.sync_on_flush {
            self.file.sync_all()?;
        }
        self.physical_writes += 1;
        self.entries_written += self.buffer.len() as u64;
        self.buffer.clear();
        Ok(())
    }

    /// Like [`flush`](Self::flush) but always fsyncs the file afterward
    /// (regardless of the `sync_on_flush` setting), for power-loss durability.
    ///
    /// # Errors
    ///
    /// Propagates any serialization, append, or fsync error.
    pub fn flush_and_sync(&mut self) -> Result<()> {
        let had_entries = !self.buffer.is_empty();
        self.flush()?;
        // `flush` already fsynced when `sync_on_flush` is set; otherwise fsync
        // here so the just-written batch is durable against power loss.
        if had_entries && !self.config.sync_on_flush {
            self.file.sync_all()?;
        }
        Ok(())
    }

    /// Reads every entry back from a JSONL log file produced by this writer.
    ///
    /// Intended for replay, inspection, and round-trip tests. Blank lines are
    /// skipped.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::Error::Io`] if the file cannot be read, or
    /// [`crate::error::Error::Serialization`] if a line is not a valid encoded
    /// entry.
    pub fn read_entries(path: impl AsRef<Path>) -> Result<Vec<RenderLogEntry>> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: RenderLogEntry = serde_json::from_str(&line)?;
            entries.push(entry);
        }
        Ok(entries)
    }
}

impl Drop for BatchedRenderLogWriter {
    fn drop(&mut self) {
        // Best-effort: persist whatever is still buffered. `Drop` cannot surface
        // an error, so a failure here (e.g. a full disk) silently drops the
        // tail — callers that require durability must flush explicitly first.
        let _ = self.flush();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_not_problem() {
        assert!(!RenderLogLevel::Debug.is_problem());
    }

    #[test]
    fn test_info_not_problem() {
        assert!(!RenderLogLevel::Info.is_problem());
    }

    #[test]
    fn test_warning_is_problem() {
        assert!(RenderLogLevel::Warning.is_problem());
    }

    #[test]
    fn test_error_is_problem() {
        assert!(RenderLogLevel::Error.is_problem());
    }

    #[test]
    fn test_critical_is_problem() {
        assert!(RenderLogLevel::Critical.is_problem());
    }

    #[test]
    fn test_level_codes() {
        assert_eq!(RenderLogLevel::Debug.code(), 0);
        assert_eq!(RenderLogLevel::Info.code(), 1);
        assert_eq!(RenderLogLevel::Warning.code(), 2);
        assert_eq!(RenderLogLevel::Error.code(), 3);
        assert_eq!(RenderLogLevel::Critical.code(), 4);
    }

    #[test]
    fn test_entry_new_no_frame() {
        let e = RenderLogEntry::new(42, RenderLogLevel::Info, "started", 1_000_000);
        assert_eq!(e.job_id, 42);
        assert!(e.frame.is_none());
        assert_eq!(e.message, "started");
    }

    #[test]
    fn test_entry_with_frame() {
        let e = RenderLogEntry::with_frame(1, 99, RenderLogLevel::Warning, "slow", 5000);
        assert_eq!(e.frame, Some(99));
    }

    #[test]
    fn test_log_add_and_count() {
        let mut log = RenderLog::new(100);
        log.add(RenderLogEntry::new(1, RenderLogLevel::Info, "ok", 0));
        log.add(RenderLogEntry::new(1, RenderLogLevel::Error, "fail", 1));
        assert_eq!(log.entry_count(), 2);
    }

    #[test]
    fn test_log_capacity_enforced() {
        let mut log = RenderLog::new(3);
        for i in 0..5u64 {
            log.add(RenderLogEntry::new(i, RenderLogLevel::Debug, "msg", i));
        }
        assert_eq!(log.entry_count(), 3);
    }

    #[test]
    fn test_log_errors_for_job() {
        let mut log = RenderLog::new(50);
        log.add(RenderLogEntry::new(7, RenderLogLevel::Error, "e1", 0));
        log.add(RenderLogEntry::new(7, RenderLogLevel::Info, "i1", 1));
        log.add(RenderLogEntry::new(8, RenderLogLevel::Error, "other", 2));
        log.add(RenderLogEntry::new(7, RenderLogLevel::Critical, "c1", 3));
        let errs = log.errors_for_job(7);
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn test_log_errors_for_job_none() {
        let mut log = RenderLog::new(10);
        log.add(RenderLogEntry::new(1, RenderLogLevel::Info, "ok", 0));
        assert!(log.errors_for_job(1).is_empty());
    }

    #[test]
    fn test_log_warnings() {
        let mut log = RenderLog::new(20);
        log.add(RenderLogEntry::new(1, RenderLogLevel::Warning, "w1", 0));
        log.add(RenderLogEntry::new(2, RenderLogLevel::Error, "e1", 1));
        log.add(RenderLogEntry::new(3, RenderLogLevel::Warning, "w2", 2));
        let warns = log.warnings();
        assert_eq!(warns.len(), 2);
    }

    #[test]
    fn test_log_zero_capacity_ignores_entries() {
        let mut log = RenderLog::new(0);
        log.add(RenderLogEntry::new(1, RenderLogLevel::Info, "msg", 0));
        assert_eq!(log.entry_count(), 0);
    }

    #[test]
    fn test_log_oldest_dropped_on_overflow() {
        let mut log = RenderLog::new(2);
        log.add(RenderLogEntry::new(1, RenderLogLevel::Debug, "first", 0));
        log.add(RenderLogEntry::new(2, RenderLogLevel::Debug, "second", 1));
        log.add(RenderLogEntry::new(3, RenderLogLevel::Debug, "third", 2));
        // Only "second" and "third" should remain.
        let ids: Vec<u64> = log.entries.iter().map(|e| e.job_id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    // -----------------------------------------------------------------------
    // BatchedRenderLogWriter tests
    // -----------------------------------------------------------------------

    /// Creates a throwaway temp directory (auto-removed on drop) plus a log
    /// path inside it. The returned `TempDir` must stay bound for the lifetime
    /// of the test so the directory is not deleted early.
    fn temp_log() -> Result<(tempfile::TempDir, PathBuf)> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("render.log");
        Ok((dir, path))
    }

    #[test]
    fn test_writer_round_trip_preserves_entries() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let originals = vec![
            RenderLogEntry::new(1, RenderLogLevel::Info, "job started", 1_000),
            RenderLogEntry::with_frame(1, 0, RenderLogLevel::Debug, "frame 0", 1_001),
            RenderLogEntry::with_frame(1, 1, RenderLogLevel::Warning, "slow frame", 1_002),
            RenderLogEntry::new(1, RenderLogLevel::Critical, "gpu fault", 1_003),
        ];

        let mut writer = BatchedRenderLogWriter::new(path.clone())?;
        for entry in &originals {
            writer.log_entry(entry.clone())?;
        }
        writer.flush()?;

        let restored = BatchedRenderLogWriter::read_entries(&path)?;
        assert_eq!(restored, originals);
        Ok(())
    }

    #[test]
    fn test_writer_n_log_frame_calls_produce_n_entries() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let n: u32 = 37;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(8);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        for frame in 0..n {
            writer.log_frame(7, frame, RenderLogLevel::Info, format!("frame {frame}"), 1)?;
        }
        writer.flush()?;

        assert_eq!(writer.entries_written(), u64::from(n));
        let restored = BatchedRenderLogWriter::read_entries(&path)?;
        assert_eq!(restored.len(), n as usize);
        for (frame, entry) in restored.iter().enumerate() {
            assert_eq!(entry.frame, Some(frame as u32));
            assert_eq!(entry.job_id, 7);
        }
        Ok(())
    }

    #[test]
    fn test_writer_threshold_auto_flush_boundary() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(4);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        assert_eq!(writer.batch_size(), 4);

        // One below the threshold: nothing written yet.
        for frame in 0..3 {
            writer.log_frame(1, frame, RenderLogLevel::Debug, "buffered", 0)?;
        }
        assert_eq!(writer.buffered_len(), 3);
        assert_eq!(writer.physical_writes(), 0);

        // The fourth entry hits the threshold and triggers exactly one flush.
        writer.log_frame(1, 3, RenderLogLevel::Debug, "boundary", 0)?;
        assert_eq!(writer.buffered_len(), 0);
        assert_eq!(writer.physical_writes(), 1);
        assert_eq!(BatchedRenderLogWriter::read_entries(&path)?.len(), 4);
        Ok(())
    }

    #[test]
    fn test_writer_explicit_flush() -> Result<()> {
        let (_dir, path) = temp_log()?;
        // Batch larger than the workload so only an explicit flush persists.
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(100);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        for frame in 0..5 {
            writer.log_frame(2, frame, RenderLogLevel::Info, "pending", 0)?;
        }
        assert_eq!(writer.buffered_len(), 5);
        assert_eq!(writer.physical_writes(), 0);
        // Nothing is on disk before the explicit flush.
        assert!(BatchedRenderLogWriter::read_entries(&path)?.is_empty());

        writer.flush()?;
        assert_eq!(writer.buffered_len(), 0);
        assert_eq!(writer.physical_writes(), 1);
        assert_eq!(BatchedRenderLogWriter::read_entries(&path)?.len(), 5);
        Ok(())
    }

    #[test]
    fn test_writer_fewer_writes_than_entries() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let entries: u64 = 50;
        let batch: usize = 10;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(batch);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        for frame in 0..entries {
            writer.log_frame(3, frame as u32, RenderLogLevel::Info, "f", frame)?;
        }
        writer.flush()?; // no-op: 50 is an exact multiple of the batch size

        // 50 entries, batch size 10 -> 5 append writes, far fewer than 50.
        assert_eq!(writer.physical_writes(), 5);
        assert_eq!(writer.entries_written(), entries);
        assert!(writer.physical_writes() < entries);
        assert_eq!(
            BatchedRenderLogWriter::read_entries(&path)?.len(),
            entries as usize
        );
        Ok(())
    }

    #[test]
    fn test_writer_flush_on_drop() -> Result<()> {
        let (_dir, path) = temp_log()?;
        {
            // Default batch size (128) far exceeds the 6 entries, so they stay
            // buffered until the writer is dropped at the end of this block.
            let mut writer = BatchedRenderLogWriter::new(path.clone())?;
            for frame in 0..6 {
                writer.log_frame(4, frame, RenderLogLevel::Info, "drop me", 0)?;
            }
            assert_eq!(writer.buffered_len(), 6);
            assert_eq!(writer.physical_writes(), 0);
        } // drop -> best-effort flush

        let restored = BatchedRenderLogWriter::read_entries(&path)?;
        assert_eq!(restored.len(), 6);
        Ok(())
    }

    #[test]
    fn test_writer_empty_flush_is_noop() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let mut writer = BatchedRenderLogWriter::new(path.clone())?;
        writer.flush()?;
        assert_eq!(writer.physical_writes(), 0);
        assert_eq!(writer.entries_written(), 0);
        assert!(writer.is_empty());
        assert!(BatchedRenderLogWriter::read_entries(&path)?.is_empty());
        Ok(())
    }

    #[test]
    fn test_writer_batch_size_zero_clamped_to_one() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(0);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        // A batch size of 0 is clamped to 1: every entry flushes immediately
        // (the degenerate per-entry case).
        assert_eq!(writer.batch_size(), 1);
        for frame in 0..3 {
            writer.log_frame(5, frame, RenderLogLevel::Info, "immediate", 0)?;
        }
        assert_eq!(writer.physical_writes(), 3);
        assert_eq!(writer.buffered_len(), 0);
        assert_eq!(BatchedRenderLogWriter::read_entries(&path)?.len(), 3);
        Ok(())
    }

    #[test]
    fn test_writer_flush_and_sync_round_trip() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(64);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        for frame in 0..4 {
            writer.log_frame(6, frame, RenderLogLevel::Error, "durable", frame.into())?;
        }
        writer.flush_and_sync()?;
        assert_eq!(writer.physical_writes(), 1);
        assert_eq!(writer.buffered_len(), 0);
        assert_eq!(BatchedRenderLogWriter::read_entries(&path)?.len(), 4);
        Ok(())
    }

    #[test]
    fn test_writer_sync_on_flush_config_auto_flush() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let config = RenderLogWriterConfig::new(path.clone())
            .with_batch_size(3)
            .with_sync_on_flush(true);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;
        for frame in 0..3 {
            writer.log_frame(8, frame, RenderLogLevel::Info, "synced", 0)?;
        }
        // Threshold reached -> auto flush that also fsyncs.
        assert_eq!(writer.physical_writes(), 1);
        assert_eq!(BatchedRenderLogWriter::read_entries(&path)?.len(), 3);
        Ok(())
    }

    #[test]
    fn test_writer_appends_across_multiple_flushes_in_order() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let config = RenderLogWriterConfig::new(path.clone()).with_batch_size(100);
        let mut writer = BatchedRenderLogWriter::with_config(config)?;

        writer.log_frame(9, 0, RenderLogLevel::Info, "a", 0)?;
        writer.log_frame(9, 1, RenderLogLevel::Info, "b", 1)?;
        writer.flush()?;
        writer.log_frame(9, 2, RenderLogLevel::Info, "c", 2)?;
        writer.log_frame(9, 3, RenderLogLevel::Info, "d", 3)?;
        writer.flush()?;

        assert_eq!(writer.physical_writes(), 2);
        let restored = BatchedRenderLogWriter::read_entries(&path)?;
        let frames: Vec<Option<u32>> = restored.iter().map(|e| e.frame).collect();
        assert_eq!(frames, vec![Some(0), Some(1), Some(2), Some(3)]);
        Ok(())
    }

    #[test]
    fn test_writer_default_batch_size() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let writer = BatchedRenderLogWriter::new(path)?;
        assert_eq!(writer.batch_size(), DEFAULT_LOG_BATCH_SIZE);
        assert!(DEFAULT_LOG_BATCH_SIZE >= 1);
        Ok(())
    }

    #[test]
    fn test_writer_log_job_level_entry_has_no_frame() -> Result<()> {
        let (_dir, path) = temp_log()?;
        let mut writer = BatchedRenderLogWriter::new(path.clone())?;
        writer.log_job(10, RenderLogLevel::Info, "job-level", 42)?;
        writer.flush()?;
        let restored = BatchedRenderLogWriter::read_entries(&path)?;
        assert_eq!(restored.len(), 1);
        assert!(restored[0].frame.is_none());
        assert_eq!(restored[0].job_id, 10);
        assert_eq!(restored[0].message, "job-level");
        Ok(())
    }
}
