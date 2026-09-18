// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! WAL-based persistence — file-backed, append-only write-ahead log.
//!
//! Each entry is a single JSON line terminated by `\n`.  Entries are never
//! mutated; new state is expressed by appending a later entry for the same
//! job.  When the WAL is replayed, the *last* entry for each job ID wins,
//! giving a simple last-write-wins semantics.
//!
//! # Format
//!
//! Every line is a JSON object with the following mandatory top-level keys:
//!
//! ```json
//! {
//!   "seq":       <u64>,       // monotonically increasing sequence number
//!   "ts":        "<rfc3339>", // wall-clock timestamp
//!   "op":        "<op>",      // "upsert" | "delete" | "checkpoint"
//!   "job_id":    "<uuid>",    // stable job identity
//!   "payload":   { ... }      // op-specific data (absent for "delete")
//! }
//! ```
//!
//! # Compaction
//!
//! Call `Wal::compact` to rewrite the WAL file, keeping only the last live
//! entry for each job (deletes are dropped).  This reduces replay time after
//! long-running processes.
//!
//! # Crash safety
//!
//! The WAL uses `O_APPEND` semantics.  Each `write_entry` call flushes and
//! syncs (`sync_all`) before returning so that a power-loss after a successful
//! return leaves the entry durable.  Partially-written trailing lines (caused
//! by a crash mid-write) are silently skipped during replay.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

use crate::job::Job;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by WAL operations.
#[derive(Debug, Error)]
pub enum WalError {
    /// An I/O error occurred while reading or writing the WAL file.
    #[error("WAL I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A line could not be serialised to JSON.
    #[error("WAL serialisation error: {0}")]
    Serialise(#[from] serde_json::Error),

    /// The WAL file contains a corrupt entry that cannot be recovered.
    #[error("WAL corrupt entry at sequence {0}: {1}")]
    CorruptEntry(u64, String),
}

// ─────────────────────────────────────────────────────────────────────────────
// WAL operation
// ─────────────────────────────────────────────────────────────────────────────

/// The type of operation recorded in a WAL entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WalOp {
    /// Insert or update a job record.
    Upsert,
    /// Mark a job as permanently deleted (tombstone).
    Delete,
    /// A compaction checkpoint; carries no live job data.
    Checkpoint,
}

// ─────────────────────────────────────────────────────────────────────────────
// WalEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry serialised to one JSON line in the WAL file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Monotonically increasing sequence number (1-based).
    pub seq: u64,
    /// Wall-clock timestamp when the entry was written.
    pub ts: DateTime<Utc>,
    /// The operation type.
    pub op: WalOp,
    /// The job this entry refers to.
    pub job_id: Uuid,
    /// Serialised `Job` for `Upsert` entries; `null` for `Delete`/`Checkpoint`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl WalEntry {
    /// Construct an `Upsert` entry.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Serialise`] if `job` cannot be serialised.
    pub fn upsert(seq: u64, job: &Job) -> Result<Self, WalError> {
        let payload = serde_json::to_value(job)?;
        Ok(Self {
            seq,
            ts: Utc::now(),
            op: WalOp::Upsert,
            job_id: job.id,
            payload: Some(payload),
        })
    }

    /// Construct a `Delete` (tombstone) entry.
    pub fn delete(seq: u64, job_id: Uuid) -> Self {
        Self {
            seq,
            ts: Utc::now(),
            op: WalOp::Delete,
            job_id,
            payload: None,
        }
    }

    /// Construct a `Checkpoint` marker entry.
    pub fn checkpoint(seq: u64) -> Self {
        Self {
            seq,
            ts: Utc::now(),
            op: WalOp::Checkpoint,
            job_id: Uuid::nil(),
            payload: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WalStats
// ─────────────────────────────────────────────────────────────────────────────

/// Replay and compaction statistics.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WalStats {
    /// Total lines read during replay.
    pub lines_read: u64,
    /// Lines that were skipped (corrupt / empty).
    pub lines_skipped: u64,
    /// Number of live job records recovered.
    pub live_jobs: usize,
    /// Number of entries dropped during compaction (tombstones + superseded).
    pub entries_dropped: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Wal
// ─────────────────────────────────────────────────────────────────────────────

/// File-backed, append-only write-ahead log for job state.
pub struct Wal {
    /// Path to the primary WAL file.
    path: PathBuf,
    /// Buffered append writer (always open).
    writer: BufWriter<File>,
    /// Monotonically increasing sequence counter.
    next_seq: u64,
    /// How many entries may be appended before an automatic compaction is triggered.
    /// `0` means never compact automatically.
    auto_compact_threshold: u64,
    /// Entries appended since the last compaction.
    entries_since_compact: u64,
}

impl Wal {
    /// Open (or create) a WAL at `path`.
    ///
    /// If the file already exists it is replayed first to restore
    /// `next_seq`.  Call [`Wal::replay`] separately to recover `Job` state.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Io`] if the file cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WalError> {
        Self::open_with_threshold(path, 0)
    }

    /// Open with an automatic compaction threshold.
    ///
    /// When `auto_compact_threshold > 0`, `Wal::write_entry` will
    /// automatically call `Wal::compact` every `auto_compact_threshold`
    /// appended entries.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Io`] if the file cannot be opened.
    pub fn open_with_threshold<P: AsRef<Path>>(
        path: P,
        auto_compact_threshold: u64,
    ) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();

        // Determine next_seq by doing a lightweight replay of sequence numbers.
        let next_seq = if path.exists() {
            Self::scan_max_seq(&path)? + 1
        } else {
            1
        };

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
            next_seq,
            auto_compact_threshold,
            entries_since_compact: 0,
        })
    }

    /// Replay the WAL file and reconstruct the latest state for every job.
    ///
    /// Returns a map from job ID → `Job` (tombstoned jobs are excluded) and
    /// replay statistics.
    ///
    /// # Errors
    ///
    /// Returns [`WalError::Io`] if the file cannot be read.
    pub fn replay(&self) -> Result<(HashMap<Uuid, Job>, WalStats), WalError> {
        if !self.path.exists() {
            return Ok((HashMap::new(), WalStats::default()));
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        // last-write-wins: later entries overwrite earlier ones.
        let mut state: HashMap<Uuid, Option<Job>> = HashMap::new();
        let mut stats = WalStats::default();

        for line_result in reader.lines() {
            let line = line_result?;
            stats.lines_read += 1;

            let trimmed = line.trim();
            if trimmed.is_empty() {
                stats.lines_skipped += 1;
                continue;
            }

            let entry: WalEntry = match serde_json::from_str(trimmed) {
                Ok(e) => e,
                Err(_) => {
                    stats.lines_skipped += 1;
                    continue;
                }
            };

            match entry.op {
                WalOp::Checkpoint => {
                    // Marker only; no state change.
                }
                WalOp::Delete => {
                    state.insert(entry.job_id, None);
                }
                WalOp::Upsert => {
                    if let Some(payload) = entry.payload {
                        match serde_json::from_value::<Job>(payload) {
                            Ok(job) => {
                                state.insert(entry.job_id, Some(job));
                            }
                            Err(_) => {
                                stats.lines_skipped += 1;
                            }
                        }
                    } else {
                        stats.lines_skipped += 1;
                    }
                }
            }
        }

        let live: HashMap<Uuid, Job> = state
            .into_iter()
            .filter_map(|(id, opt)| opt.map(|j| (id, j)))
            .collect();

        stats.live_jobs = live.len();
        Ok((live, stats))
    }

    /// Append an `Upsert` entry for `job` to the WAL.
    ///
    /// # Errors
    ///
    /// Returns [`WalError`] if serialisation or I/O fails.
    pub fn append_upsert(&mut self, job: &Job) -> Result<u64, WalError> {
        let seq = self.next_seq;
        let entry = WalEntry::upsert(seq, job)?;
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Append a `Delete` tombstone for `job_id`.
    ///
    /// # Errors
    ///
    /// Returns [`WalError`] if I/O fails.
    pub fn append_delete(&mut self, job_id: Uuid) -> Result<u64, WalError> {
        let seq = self.next_seq;
        let entry = WalEntry::delete(seq, job_id);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Append a `Checkpoint` marker.
    ///
    /// # Errors
    ///
    /// Returns [`WalError`] if I/O fails.
    pub fn append_checkpoint(&mut self) -> Result<u64, WalError> {
        let seq = self.next_seq;
        let entry = WalEntry::checkpoint(seq);
        self.write_entry(&entry)?;
        Ok(seq)
    }

    /// Compact the WAL in-place: rewrite keeping only the last live `Upsert`
    /// for each job, discarding tombstones and superseded entries.
    ///
    /// A `Checkpoint` entry is appended at the end of the compacted file so
    /// readers know compaction completed cleanly.
    ///
    /// # Errors
    ///
    /// Returns [`WalError`] if I/O or serialisation fails.
    pub fn compact(&mut self) -> Result<WalStats, WalError> {
        let (live, mut stats) = self.replay()?;

        // Write the compacted content to a temporary sibling file.
        let tmp_path = self.path.with_extension("wal.tmp");
        {
            let tmp_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            let mut tmp_writer = BufWriter::new(tmp_file);

            // Reset sequence for compacted file; keep existing next_seq so new
            // entries remain ordered after compaction.
            let mut seq = 1u64;
            for job in live.values() {
                let entry = WalEntry::upsert(seq, job)?;
                let line = serde_json::to_string(&entry)?;
                tmp_writer.write_all(line.as_bytes())?;
                tmp_writer.write_all(b"\n")?;
                seq += 1;
            }

            // Terminal checkpoint marker.
            let chk = WalEntry::checkpoint(seq);
            let line = serde_json::to_string(&chk)?;
            tmp_writer.write_all(line.as_bytes())?;
            tmp_writer.write_all(b"\n")?;

            tmp_writer.flush()?;
            tmp_writer
                .into_inner()
                .map_err(|e| e.into_error())?
                .sync_all()?;
        }

        // Atomically replace the WAL with the compacted version.
        std::fs::rename(&tmp_path, &self.path)?;

        // Re-open the writer pointing at the (now smaller) file.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        self.entries_since_compact = 0;

        stats.entries_dropped = stats.lines_read.saturating_sub(stats.live_jobs as u64);
        Ok(stats)
    }

    /// The current WAL file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The sequence number that will be assigned to the *next* entry.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Number of entries appended since the last compaction (or since open).
    pub fn entries_since_compact(&self) -> u64 {
        self.entries_since_compact
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Serialise `entry` to a JSON line, write it, flush, and sync.
    fn write_entry(&mut self, entry: &WalEntry) -> Result<(), WalError> {
        let line = serde_json::to_string(entry)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        // Sync to durable storage.
        self.writer.get_ref().sync_all()?;

        self.next_seq += 1;
        self.entries_since_compact += 1;

        // Auto-compact if threshold reached.
        if self.auto_compact_threshold > 0
            && self.entries_since_compact >= self.auto_compact_threshold
        {
            self.compact()?;
        }

        Ok(())
    }

    /// Scan the WAL file extracting only sequence numbers to find the maximum.
    /// Lines that cannot be parsed are silently ignored.
    fn scan_max_seq(path: &Path) -> Result<u64, WalError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut max_seq = 0u64;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Parse only the `seq` field using a minimal struct to avoid full
            // deserialisation overhead.
            #[derive(Deserialize)]
            struct SeqOnly {
                seq: u64,
            }
            if let Ok(s) = serde_json::from_str::<SeqOnly>(trimmed) {
                if s.seq > max_seq {
                    max_seq = s.seq;
                }
            }
        }

        Ok(max_seq)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{Job, JobPayload, Priority, TranscodeParams};
    use std::env;

    fn temp_wal_path(name: &str) -> PathBuf {
        let mut p = env::temp_dir();
        p.push(format!(
            "oximedia_jobs_wal_test_{name}_{}.wal",
            Uuid::new_v4()
        ));
        p
    }

    fn make_job(name: &str) -> Job {
        let params = TranscodeParams {
            input: "in.mp4".to_string(),
            output: "out.mp4".to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".to_string(),
            hw_accel: None,
        };
        Job::new(
            name.to_string(),
            Priority::Normal,
            JobPayload::Transcode(params),
        )
    }

    // ── open / create ─────────────────────────────────────────────────────────

    #[test]
    fn test_open_creates_file() {
        let path = temp_wal_path("create");
        let wal = Wal::open(&path).expect("open should succeed");
        assert_eq!(wal.next_seq(), 1);
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_existing_restores_seq() {
        let path = temp_wal_path("seq_restore");
        {
            let mut wal = Wal::open(&path).expect("open should succeed");
            let job = make_job("j1");
            wal.append_upsert(&job).expect("append should succeed");
            wal.append_upsert(&job).expect("append should succeed");
            // After 2 appends, next_seq should be 3.
            assert_eq!(wal.next_seq(), 3);
        }
        // Re-open — should restore seq.
        let wal2 = Wal::open(&path).expect("re-open should succeed");
        assert_eq!(wal2.next_seq(), 3);
        let _ = std::fs::remove_file(&path);
    }

    // ── append_upsert ─────────────────────────────────────────────────────────

    #[test]
    fn test_append_upsert_returns_seq() {
        let path = temp_wal_path("upsert_seq");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let job = make_job("j1");
        let seq = wal.append_upsert(&job).expect("append should succeed");
        assert_eq!(seq, 1);
        let seq2 = wal.append_upsert(&job).expect("append should succeed");
        assert_eq!(seq2, 2);
        let _ = std::fs::remove_file(&path);
    }

    // ── append_delete ─────────────────────────────────────────────────────────

    #[test]
    fn test_append_delete_creates_tombstone() {
        let path = temp_wal_path("delete");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let job = make_job("j1");
        let id = job.id;
        wal.append_upsert(&job).expect("append should succeed");
        wal.append_delete(id).expect("delete should succeed");

        let (jobs, _stats) = wal.replay().expect("replay should succeed");
        assert!(!jobs.contains_key(&id), "tombstoned job should be absent");
        let _ = std::fs::remove_file(&path);
    }

    // ── replay ────────────────────────────────────────────────────────────────

    #[test]
    fn test_replay_recovers_single_job() {
        let path = temp_wal_path("replay_single");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let job = make_job("j1");
        let id = job.id;
        wal.append_upsert(&job).expect("append should succeed");

        let (jobs, stats) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 1);
        assert!(jobs.contains_key(&id));
        assert_eq!(stats.live_jobs, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_replay_last_write_wins() {
        let path = temp_wal_path("replay_lww");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let mut job = make_job("j1");
        let id = job.id;
        wal.append_upsert(&job).expect("append should succeed");

        // Mutate the job and upsert again.
        job.name = "j1-updated".to_string();
        wal.append_upsert(&job).expect("append should succeed");

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs[&id].name, "j1-updated");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_replay_multiple_jobs() {
        let path = temp_wal_path("replay_multi");
        let mut wal = Wal::open(&path).expect("open should succeed");
        for i in 0..5_usize {
            let job = make_job(&format!("j{i}"));
            wal.append_upsert(&job).expect("append should succeed");
        }
        let (jobs, stats) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 5);
        assert_eq!(stats.live_jobs, 5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_replay_empty_file_returns_empty_map() {
        let path = temp_wal_path("replay_empty");
        let wal = Wal::open(&path).expect("open should succeed");
        let (jobs, stats) = wal.replay().expect("replay should succeed");
        assert!(jobs.is_empty());
        assert_eq!(stats.live_jobs, 0);
        let _ = std::fs::remove_file(&path);
    }

    // ── compact ───────────────────────────────────────────────────────────────

    #[test]
    fn test_compact_removes_tombstones() {
        let path = temp_wal_path("compact_tombstone");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let j1 = make_job("live");
        let j2 = make_job("dead");
        let dead_id = j2.id;
        wal.append_upsert(&j1).expect("append should succeed");
        wal.append_upsert(&j2).expect("append should succeed");
        wal.append_delete(dead_id).expect("delete should succeed");

        wal.compact().expect("compact should succeed");

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 1);
        assert!(!jobs.contains_key(&dead_id));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compact_keeps_live_jobs() {
        let path = temp_wal_path("compact_live");
        let mut wal = Wal::open(&path).expect("open should succeed");
        for i in 0..10_usize {
            let job = make_job(&format!("j{i}"));
            wal.append_upsert(&job).expect("append should succeed");
        }
        wal.compact().expect("compact should succeed");

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 10);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_compact_allows_further_writes() {
        let path = temp_wal_path("compact_write_after");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let j1 = make_job("j1");
        wal.append_upsert(&j1).expect("append should succeed");
        wal.compact().expect("compact should succeed");

        // After compaction, we should still be able to append new entries.
        let j2 = make_job("j2");
        wal.append_upsert(&j2)
            .expect("append after compact should succeed");

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    // ── auto-compact ──────────────────────────────────────────────────────────

    #[test]
    fn test_auto_compact_triggers_on_threshold() {
        let path = temp_wal_path("auto_compact");
        let mut wal = Wal::open_with_threshold(&path, 3).expect("open should succeed");

        // Write 3 upserts; the 3rd should trigger auto-compact.
        for i in 0..3_usize {
            let job = make_job(&format!("j{i}"));
            wal.append_upsert(&job).expect("append should succeed");
        }
        // After auto-compact, entries_since_compact should be reset.
        assert_eq!(wal.entries_since_compact(), 0);

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 3);
        let _ = std::fs::remove_file(&path);
    }

    // ── checkpoint ────────────────────────────────────────────────────────────

    #[test]
    fn test_append_checkpoint_does_not_affect_replay() {
        let path = temp_wal_path("checkpoint");
        let mut wal = Wal::open(&path).expect("open should succeed");
        let job = make_job("j1");
        let id = job.id;
        wal.append_upsert(&job).expect("append should succeed");
        wal.append_checkpoint().expect("checkpoint should succeed");

        let (jobs, _) = wal.replay().expect("replay should succeed");
        assert_eq!(jobs.len(), 1);
        assert!(jobs.contains_key(&id));
        let _ = std::fs::remove_file(&path);
    }

    // ── WalEntry helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_wal_entry_upsert_roundtrip() {
        let job = make_job("roundtrip");
        let entry = WalEntry::upsert(1, &job).expect("upsert entry should be created");
        assert_eq!(entry.op, WalOp::Upsert);
        assert_eq!(entry.job_id, job.id);
        assert!(entry.payload.is_some());
    }

    #[test]
    fn test_wal_entry_delete_has_no_payload() {
        let id = Uuid::new_v4();
        let entry = WalEntry::delete(2, id);
        assert_eq!(entry.op, WalOp::Delete);
        assert!(entry.payload.is_none());
    }

    #[test]
    fn test_wal_entry_checkpoint_has_nil_job_id() {
        let entry = WalEntry::checkpoint(3);
        assert_eq!(entry.op, WalOp::Checkpoint);
        assert_eq!(entry.job_id, Uuid::nil());
    }

    // ── path / seq accessors ──────────────────────────────────────────────────

    #[test]
    fn test_wal_path_accessor() {
        let path = temp_wal_path("path_accessor");
        let wal = Wal::open(&path).expect("open should succeed");
        assert_eq!(wal.path(), path.as_path());
        let _ = std::fs::remove_file(&path);
    }
}
