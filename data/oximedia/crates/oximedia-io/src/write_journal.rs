#![allow(dead_code)]

//! Write-ahead journaling for crash-safe media I/O.
//!
//! This module provides a simple write-ahead journal (WAL) that records
//! pending write operations before they are applied, enabling recovery
//! after unexpected crashes or power failures during media file writes.
//!
//! # Features
//!
//! - [`JournalEntry`] - A single journaled write operation
//! - [`WriteJournal`] - In-memory journal that tracks pending writes
//! - [`JournalConfig`] - Configuration for journal behaviour
//! - [`JournalCheckpoint`] - Snapshot of journal state for recovery

use std::collections::VecDeque;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Configuration for the write journal.
#[derive(Debug, Clone)]
pub struct JournalConfig {
    /// Maximum number of entries to keep before forcing a checkpoint.
    pub max_entries: usize,
    /// Maximum total data size (bytes) before forcing a checkpoint.
    pub max_data_bytes: u64,
    /// Whether to compute checksums for journal entries.
    pub checksums_enabled: bool,
    /// Flush interval hint (how often the journal should be synced).
    pub flush_interval: Duration,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            max_data_bytes: 64 * 1024 * 1024,
            checksums_enabled: true,
            flush_interval: Duration::from_millis(100),
        }
    }
}

impl JournalConfig {
    /// Create a config suitable for low-latency streaming.
    #[must_use]
    pub fn for_streaming() -> Self {
        Self {
            max_entries: 1_000,
            max_data_bytes: 8 * 1024 * 1024,
            checksums_enabled: false,
            flush_interval: Duration::from_millis(10),
        }
    }

    /// Create a config suitable for archival writes.
    #[must_use]
    pub fn for_archival() -> Self {
        Self {
            max_entries: 100_000,
            max_data_bytes: 512 * 1024 * 1024,
            checksums_enabled: true,
            flush_interval: Duration::from_secs(1),
        }
    }
}

/// The type of journal operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalOp {
    /// Write data at a given offset.
    Write,
    /// Truncate the file to a given length.
    Truncate,
    /// Append data to the end of the file.
    Append,
    /// Sync / flush to stable storage.
    Sync,
}

impl fmt::Display for JournalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Write => write!(f, "WRITE"),
            Self::Truncate => write!(f, "TRUNCATE"),
            Self::Append => write!(f, "APPEND"),
            Self::Sync => write!(f, "SYNC"),
        }
    }
}

/// A single journal entry representing one I/O operation.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    /// Monotonic sequence number within the journal.
    pub seq: u64,
    /// The operation type.
    pub op: JournalOp,
    /// Target file offset (meaningful for Write; 0 for others).
    pub offset: u64,
    /// Length of the data (0 for Sync).
    pub data_len: u32,
    /// CRC-32 checksum of the data (0 if checksums disabled or no data).
    pub checksum: u32,
    /// Timestamp (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
}

impl JournalEntry {
    /// Create a new journal entry.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(seq: u64, op: JournalOp, offset: u64, data_len: u32, checksum: u32) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            seq,
            op,
            offset,
            data_len,
            checksum,
            timestamp_ms,
        }
    }

    /// Serialise the entry to a fixed-size byte representation (40 bytes).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut buf = [0u8; 40];
        buf[0..8].copy_from_slice(&self.seq.to_le_bytes());
        buf[8] = match self.op {
            JournalOp::Write => 1,
            JournalOp::Truncate => 2,
            JournalOp::Append => 3,
            JournalOp::Sync => 4,
        };
        buf[16..24].copy_from_slice(&self.offset.to_le_bytes());
        buf[24..28].copy_from_slice(&self.data_len.to_le_bytes());
        buf[28..32].copy_from_slice(&self.checksum.to_le_bytes());
        buf[32..40].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf
    }

    /// Deserialise an entry from a 40-byte buffer.
    #[must_use]
    pub fn from_bytes(buf: &[u8; 40]) -> Self {
        let seq = u64::from_le_bytes(buf[0..8].try_into().unwrap_or_default());
        let op = match buf[8] {
            1 => JournalOp::Write,
            2 => JournalOp::Truncate,
            3 => JournalOp::Append,
            _ => JournalOp::Sync,
        };
        let offset = u64::from_le_bytes(buf[16..24].try_into().unwrap_or_default());
        let data_len = u32::from_le_bytes(buf[24..28].try_into().unwrap_or_default());
        let checksum = u32::from_le_bytes(buf[28..32].try_into().unwrap_or_default());
        let timestamp_ms = u64::from_le_bytes(buf[32..40].try_into().unwrap_or_default());

        Self {
            seq,
            op,
            offset,
            data_len,
            checksum,
            timestamp_ms,
        }
    }
}

/// A snapshot of journal state that can be used for recovery.
#[derive(Debug, Clone)]
pub struct JournalCheckpoint {
    /// Sequence number of the checkpoint.
    pub checkpoint_seq: u64,
    /// Number of entries at checkpoint time.
    pub entry_count: usize,
    /// Total data bytes at checkpoint time.
    pub total_data_bytes: u64,
    /// Timestamp.
    pub timestamp_ms: u64,
}

impl JournalCheckpoint {
    /// Create a new checkpoint from current journal state.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(checkpoint_seq: u64, entry_count: usize, total_data_bytes: u64) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            checkpoint_seq,
            entry_count,
            total_data_bytes,
            timestamp_ms,
        }
    }
}

/// Simple CRC-32 implementation for journal checksums.
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

/// An in-memory write-ahead journal.
///
/// Records write operations before they are committed, allowing replay
/// or rollback after a crash.
pub struct WriteJournal {
    /// Journal configuration.
    config: JournalConfig,
    /// The journal entries.
    entries: VecDeque<JournalEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Total data bytes tracked.
    total_data_bytes: u64,
    /// List of checkpoints.
    checkpoints: Vec<JournalCheckpoint>,
}

impl WriteJournal {
    /// Create a new, empty journal.
    #[must_use]
    pub fn new(config: JournalConfig) -> Self {
        Self {
            config,
            entries: VecDeque::new(),
            next_seq: 1,
            total_data_bytes: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Create a journal with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(JournalConfig::default())
    }

    /// Record a write operation.
    #[allow(clippy::cast_possible_truncation)]
    pub fn record_write(&mut self, offset: u64, data: &[u8]) -> u64 {
        let checksum = if self.config.checksums_enabled {
            crc32_simple(data)
        } else {
            0
        };
        let seq = self.next_seq;
        let entry = JournalEntry::new(seq, JournalOp::Write, offset, data.len() as u32, checksum);
        self.entries.push_back(entry);
        self.next_seq += 1;
        self.total_data_bytes += data.len() as u64;
        seq
    }

    /// Record a truncate operation.
    pub fn record_truncate(&mut self, new_length: u64) -> u64 {
        let seq = self.next_seq;
        let entry = JournalEntry::new(seq, JournalOp::Truncate, new_length, 0, 0);
        self.entries.push_back(entry);
        self.next_seq += 1;
        seq
    }

    /// Record an append operation.
    #[allow(clippy::cast_possible_truncation)]
    pub fn record_append(&mut self, data: &[u8]) -> u64 {
        let checksum = if self.config.checksums_enabled {
            crc32_simple(data)
        } else {
            0
        };
        let seq = self.next_seq;
        let entry = JournalEntry::new(seq, JournalOp::Append, 0, data.len() as u32, checksum);
        self.entries.push_back(entry);
        self.next_seq += 1;
        self.total_data_bytes += data.len() as u64;
        seq
    }

    /// Record a sync operation.
    pub fn record_sync(&mut self) -> u64 {
        let seq = self.next_seq;
        let entry = JournalEntry::new(seq, JournalOp::Sync, 0, 0, 0);
        self.entries.push_back(entry);
        self.next_seq += 1;
        seq
    }

    /// Return the number of pending entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Return the total tracked data bytes.
    #[must_use]
    pub fn total_data_bytes(&self) -> u64 {
        self.total_data_bytes
    }

    /// Check whether the journal needs a checkpoint (based on config limits).
    #[must_use]
    pub fn needs_checkpoint(&self) -> bool {
        self.entries.len() >= self.config.max_entries
            || self.total_data_bytes >= self.config.max_data_bytes
    }

    /// Create a checkpoint, clearing committed entries up to the current point.
    pub fn checkpoint(&mut self) -> JournalCheckpoint {
        let cp =
            JournalCheckpoint::new(self.next_seq - 1, self.entries.len(), self.total_data_bytes);
        self.entries.clear();
        self.total_data_bytes = 0;
        self.checkpoints.push(cp.clone());
        cp
    }

    /// Return entries since a given sequence number (inclusive).
    #[must_use]
    pub fn entries_since(&self, since_seq: u64) -> Vec<&JournalEntry> {
        self.entries.iter().filter(|e| e.seq >= since_seq).collect()
    }

    /// Return all entries.
    #[must_use]
    pub fn all_entries(&self) -> Vec<&JournalEntry> {
        self.entries.iter().collect()
    }

    /// Return the latest checkpoint, if any.
    #[must_use]
    pub fn latest_checkpoint(&self) -> Option<&JournalCheckpoint> {
        self.checkpoints.last()
    }

    /// Return the next sequence number that will be assigned.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &JournalConfig {
        &self.config
    }
}

impl fmt::Debug for WriteJournal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteJournal")
            .field("config", &self.config)
            .field("entries_count", &self.entries.len())
            .field("next_seq", &self.next_seq)
            .field("total_data_bytes", &self.total_data_bytes)
            .field("checkpoints_count", &self.checkpoints.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_journal_config_default() {
        let cfg = JournalConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert!(cfg.checksums_enabled);
        assert_eq!(cfg.flush_interval, Duration::from_millis(100));
    }

    #[test]
    fn test_journal_config_streaming() {
        let cfg = JournalConfig::for_streaming();
        assert_eq!(cfg.max_entries, 1_000);
        assert!(!cfg.checksums_enabled);
    }

    #[test]
    fn test_journal_config_archival() {
        let cfg = JournalConfig::for_archival();
        assert_eq!(cfg.max_entries, 100_000);
        assert!(cfg.checksums_enabled);
    }

    #[test]
    fn test_journal_op_display() {
        assert_eq!(JournalOp::Write.to_string(), "WRITE");
        assert_eq!(JournalOp::Truncate.to_string(), "TRUNCATE");
        assert_eq!(JournalOp::Append.to_string(), "APPEND");
        assert_eq!(JournalOp::Sync.to_string(), "SYNC");
    }

    #[test]
    fn test_journal_entry_roundtrip() {
        let entry = JournalEntry::new(42, JournalOp::Write, 1024, 512, 0xDEAD_BEEF);
        let bytes = entry.to_bytes();
        let decoded = JournalEntry::from_bytes(&bytes);
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.op, JournalOp::Write);
        assert_eq!(decoded.offset, 1024);
        assert_eq!(decoded.data_len, 512);
        assert_eq!(decoded.checksum, 0xDEAD_BEEF);
    }

    #[test]
    fn test_journal_entry_all_ops_roundtrip() {
        for op in [
            JournalOp::Write,
            JournalOp::Truncate,
            JournalOp::Append,
            JournalOp::Sync,
        ] {
            let entry = JournalEntry::new(1, op, 0, 0, 0);
            let bytes = entry.to_bytes();
            let decoded = JournalEntry::from_bytes(&bytes);
            assert_eq!(decoded.op, op);
        }
    }

    #[test]
    fn test_write_journal_record_write() {
        let mut journal = WriteJournal::with_defaults();
        let data = b"hello world";
        let seq = journal.record_write(100, data);
        assert_eq!(seq, 1);
        assert_eq!(journal.entry_count(), 1);
        assert_eq!(journal.total_data_bytes(), data.len() as u64);
    }

    #[test]
    fn test_write_journal_record_truncate() {
        let mut journal = WriteJournal::with_defaults();
        let seq = journal.record_truncate(4096);
        assert_eq!(seq, 1);
        assert_eq!(journal.entry_count(), 1);
        let entries = journal.all_entries();
        assert_eq!(entries[0].op, JournalOp::Truncate);
        assert_eq!(entries[0].offset, 4096);
    }

    #[test]
    fn test_write_journal_record_append() {
        let mut journal = WriteJournal::with_defaults();
        let data = b"appended data";
        let seq = journal.record_append(data);
        assert_eq!(seq, 1);
        assert_eq!(journal.total_data_bytes(), data.len() as u64);
    }

    #[test]
    fn test_write_journal_record_sync() {
        let mut journal = WriteJournal::with_defaults();
        let seq = journal.record_sync();
        assert_eq!(seq, 1);
        let entries = journal.all_entries();
        assert_eq!(entries[0].op, JournalOp::Sync);
    }

    #[test]
    fn test_write_journal_sequence_numbers() {
        let mut journal = WriteJournal::with_defaults();
        let s1 = journal.record_write(0, b"a");
        let s2 = journal.record_write(1, b"b");
        let s3 = journal.record_sync();
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(journal.next_seq(), 4);
    }

    #[test]
    fn test_write_journal_checkpoint() {
        let mut journal = WriteJournal::with_defaults();
        journal.record_write(0, b"data1");
        journal.record_write(100, b"data2");
        assert_eq!(journal.entry_count(), 2);

        let cp = journal.checkpoint();
        assert_eq!(cp.entry_count, 2);
        assert_eq!(journal.entry_count(), 0);
        assert_eq!(journal.total_data_bytes(), 0);

        assert!(journal.latest_checkpoint().is_some());
    }

    #[test]
    fn test_write_journal_entries_since() {
        let mut journal = WriteJournal::with_defaults();
        journal.record_write(0, b"first");
        journal.record_write(10, b"second");
        journal.record_write(20, b"third");

        let since_2 = journal.entries_since(2);
        assert_eq!(since_2.len(), 2);
        assert_eq!(since_2[0].seq, 2);
        assert_eq!(since_2[1].seq, 3);
    }

    #[test]
    fn test_write_journal_needs_checkpoint() {
        let config = JournalConfig {
            max_entries: 3,
            ..JournalConfig::default()
        };
        let mut journal = WriteJournal::new(config);
        journal.record_write(0, b"a");
        journal.record_write(1, b"b");
        assert!(!journal.needs_checkpoint());
        journal.record_write(2, b"c");
        assert!(journal.needs_checkpoint());
    }

    #[test]
    fn test_crc32_simple() {
        let c1 = crc32_simple(b"hello");
        let c2 = crc32_simple(b"hello");
        assert_eq!(c1, c2);

        let c3 = crc32_simple(b"world");
        assert_ne!(c1, c3);

        let c4 = crc32_simple(b"");
        assert_ne!(c4, 0); // CRC of empty is not 0 with this polynomial
    }
}
