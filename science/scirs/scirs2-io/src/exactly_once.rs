//! Exactly-once delivery semantics for streaming pipeline sinks.
//!
//! Exactly-once guarantees are provided by combining two mechanisms:
//!
//! 1. **Idempotency keys** — every message carries a caller-supplied string key
//!    that uniquely identifies the logical event.  Delivering the same key
//!    twice is a no-op.
//!
//! 2. **Write-ahead log (WAL)** — before processing a message the key and its
//!    monotonic sequence number are durably recorded.  On restart the WAL is
//!    replayed to rebuild the set of already-committed keys, so late arrivals
//!    or retries after a crash are still deduplicated correctly.
//!
//! # Design overview
//!
//! ```text
//!   producer ──► ExactlyOnceSink ──► committed output
//!                      │
//!                      ▼
//!                 WriteAheadLog
//!               (in-memory or disk)
//! ```
//!
//! The [`WriteAheadLog`] persists committed idempotency keys as `key\tseq\n`
//! lines.  [`ExactlyOnceSink`] wraps the WAL and exposes a simple
//! `submit` / `retry` API.
//!
//! # Example
//!
//! ```rust
//! use scirs2_io::exactly_once::{ExactlyOnceSink, WriteAheadLog};
//!
//! let wal = WriteAheadLog::in_memory();
//! let mut sink: ExactlyOnceSink<String> = ExactlyOnceSink::with_wal(wal);
//!
//! // First delivery — processed.
//! assert_eq!(sink.submit("msg-1", "hello".to_string()).unwrap(), true);
//!
//! // Retry / duplicate — ignored.
//! assert_eq!(sink.retry("msg-1", "hello".to_string()).unwrap(), false);
//!
//! assert_eq!(sink.processed_count(), 1);
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by exactly-once delivery operations.
#[derive(Debug, thiserror::Error)]
pub enum ExactlyOnceError {
    /// An I/O error occurred while reading or writing the WAL file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The WAL file contains an unrecognisable or corrupted entry.
    #[error("WAL corruption: {0}")]
    Corruption(String),
}

// ---------------------------------------------------------------------------
// WriteAheadLog
// ---------------------------------------------------------------------------

/// The storage medium used by the WAL.
enum WalStorage {
    /// Backed by a file on disk.
    Disk { path: PathBuf, file: std::fs::File },
    /// Backed purely by an in-memory map (no disk I/O; for testing).
    Memory,
}

/// Write-ahead log for exactly-once sink.
///
/// Records committed idempotency keys and their sequence numbers.  On
/// construction from a file path the existing entries are loaded into memory so
/// that keys committed before a restart are still recognised as duplicates.
pub struct WriteAheadLog {
    storage: WalStorage,
    /// Map from idempotency key to sequence number.
    committed_ids: HashMap<String, u64>,
}

impl WriteAheadLog {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Open (or create) a WAL backed by a file at `log_path`.
    ///
    /// Existing entries in the file are replayed to initialise the in-memory
    /// committed-key map.
    ///
    /// # Errors
    ///
    /// Returns [`ExactlyOnceError::Io`] if the file cannot be opened or read,
    /// or [`ExactlyOnceError::Corruption`] if a line cannot be parsed.
    pub fn new(log_path: &Path) -> Result<Self, ExactlyOnceError> {
        let committed_ids = Self::load_from_path(log_path)?;
        let file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(log_path)?;
        Ok(Self {
            storage: WalStorage::Disk {
                path: log_path.to_owned(),
                file,
            },
            committed_ids,
        })
    }

    /// Create a purely in-memory WAL (no disk I/O).
    ///
    /// Suitable for unit tests and scenarios where durability is not required.
    pub fn in_memory() -> Self {
        Self {
            storage: WalStorage::Memory,
            committed_ids: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    /// Return `true` if `key` has already been committed.
    pub fn is_committed(&self, key: &str) -> bool {
        self.committed_ids.contains_key(key)
    }

    /// Return the sequence number associated with a committed key, or `None`
    /// if the key has not been committed.
    pub fn get_sequence(&self, key: &str) -> Option<u64> {
        self.committed_ids.get(key).copied()
    }

    /// Return the total number of committed keys.
    pub fn committed_count(&self) -> usize {
        self.committed_ids.len()
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Record that `key` was processed with the given sequence number `seq`.
    ///
    /// If `key` is already committed this method is a **no-op** (idempotent).
    ///
    /// For disk-backed WALs the entry is appended to the file immediately;
    /// call [`flush`](Self::flush) afterwards to guarantee it is written to the
    /// OS page cache.
    ///
    /// # Errors
    ///
    /// Returns [`ExactlyOnceError::Io`] on write failure.
    pub fn commit(&mut self, key: &str, seq: u64) -> Result<(), ExactlyOnceError> {
        if self.committed_ids.contains_key(key) {
            return Ok(()); // already committed — idempotent
        }
        // Write to disk before updating the in-memory map (WAL protocol).
        if let WalStorage::Disk { ref mut file, .. } = self.storage {
            writeln!(file, "{}\t{}", key, seq)?;
        }
        self.committed_ids.insert(key.to_owned(), seq);
        Ok(())
    }

    /// Flush any pending writes to the OS kernel buffer.
    ///
    /// This is a no-op for in-memory WALs.
    ///
    /// # Errors
    ///
    /// Returns [`ExactlyOnceError::Io`] if the flush fails.
    pub fn flush(&mut self) -> Result<(), ExactlyOnceError> {
        if let WalStorage::Disk { ref mut file, .. } = self.storage {
            file.flush()?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Read and parse the existing WAL file at `path`.
    fn load_from_path(path: &Path) -> Result<HashMap<String, u64>, ExactlyOnceError> {
        let mut map = HashMap::new();
        if !path.exists() {
            return Ok(map);
        }
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        for (line_no, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if line.is_empty() {
                continue;
            }
            let tab_pos = line.find('\t').ok_or_else(|| {
                ExactlyOnceError::Corruption(format!(
                    "line {}: missing tab separator: {line:?}",
                    line_no + 1
                ))
            })?;
            let key = &line[..tab_pos];
            let seq_str = &line[tab_pos + 1..];
            let seq: u64 = seq_str.parse().map_err(|_| {
                ExactlyOnceError::Corruption(format!(
                    "line {}: cannot parse sequence number: {seq_str:?}",
                    line_no + 1
                ))
            })?;
            map.insert(key.to_owned(), seq);
        }
        Ok(map)
    }
}

// ---------------------------------------------------------------------------
// ExactlyOnceSink
// ---------------------------------------------------------------------------

/// Exactly-once sink that deduplicates messages using idempotency keys.
///
/// Internally wraps a [`WriteAheadLog`] to persist committed keys.
///
/// # Type parameter
///
/// `T` is the payload type.  It must implement [`Clone`] so that the sink can
/// retain a copy of every successfully processed item.
pub struct ExactlyOnceSink<T: Clone> {
    wal: WriteAheadLog,
    /// Items that were actually processed (de-duplicated list).
    processed: Vec<(String, T)>,
    /// Monotonically increasing sequence counter.
    next_seq: u64,
}

impl<T: Clone> ExactlyOnceSink<T> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a sink backed by an in-memory WAL.
    pub fn new() -> Self {
        Self {
            wal: WriteAheadLog::in_memory(),
            processed: Vec::new(),
            next_seq: 0,
        }
    }

    /// Create a sink using the supplied `wal`.
    ///
    /// Keys already present in `wal` (e.g. loaded from a previous run) will be
    /// treated as already-processed and will not appear in
    /// [`processed_items`](Self::processed_items).
    pub fn with_wal(wal: WriteAheadLog) -> Self {
        let next_seq = wal.committed_count() as u64;
        Self {
            wal,
            processed: Vec::new(),
            next_seq,
        }
    }

    // -----------------------------------------------------------------------
    // Delivery
    // -----------------------------------------------------------------------

    /// Submit a message with an idempotency key.
    ///
    /// - If `key` has **not** been seen before the data is recorded as
    ///   processed and `Ok(true)` is returned.
    /// - If `key` has already been committed (duplicate / retry) the message
    ///   is silently discarded and `Ok(false)` is returned.
    ///
    /// # Errors
    ///
    /// Propagates any [`ExactlyOnceError`] from the underlying WAL.
    pub fn submit(&mut self, key: &str, data: T) -> Result<bool, ExactlyOnceError> {
        if self.wal.is_committed(key) {
            return Ok(false); // duplicate — skip
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.wal.commit(key, seq)?;
        self.processed.push((key.to_owned(), data));
        Ok(true)
    }

    /// Convenience wrapper for retrying a message.
    ///
    /// Identical to [`submit`](Self::submit) — returns `false` if the key was
    /// already processed.
    pub fn retry(&mut self, key: &str, data: T) -> Result<bool, ExactlyOnceError> {
        self.submit(key, data)
    }

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /// Return the list of items that were actually processed (no duplicates).
    ///
    /// Each entry is `(idempotency_key, payload)`.  The order matches the
    /// order in which items were first submitted.
    pub fn processed_items(&self) -> &[(String, T)] {
        &self.processed
    }

    /// Return the number of items that were actually processed.
    pub fn processed_count(&self) -> usize {
        self.processed.len()
    }

    /// Return a shared reference to the underlying WAL.
    pub fn wal(&self) -> &WriteAheadLog {
        &self.wal
    }

    /// Return a mutable reference to the underlying WAL (e.g. to call `flush`).
    pub fn wal_mut(&mut self) -> &mut WriteAheadLog {
        &mut self.wal
    }
}

impl<T: Clone> Default for ExactlyOnceSink<T> {
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

    // -----------------------------------------------------------------------
    // WriteAheadLog tests
    // -----------------------------------------------------------------------

    /// in_memory WAL starts empty.
    #[test]
    fn test_wal_in_memory_starts_empty() {
        let wal = WriteAheadLog::in_memory();
        assert!(!wal.is_committed("x"));
        assert_eq!(wal.committed_count(), 0);
    }

    /// commit inserts a key; is_committed returns true.
    #[test]
    fn test_wal_commit_and_query() {
        let mut wal = WriteAheadLog::in_memory();
        wal.commit("event-42", 42).expect("commit");
        assert!(wal.is_committed("event-42"));
        assert_eq!(wal.get_sequence("event-42"), Some(42));
    }

    /// commit is idempotent: committing the same key twice does not panic.
    #[test]
    fn test_wal_commit_idempotent() {
        let mut wal = WriteAheadLog::in_memory();
        wal.commit("k", 1).expect("first commit");
        wal.commit("k", 99).expect("second commit (no-op)");
        // Sequence number from the first commit is preserved.
        assert_eq!(wal.get_sequence("k"), Some(1));
        assert_eq!(wal.committed_count(), 1);
    }

    /// flush is a no-op for in-memory WAL.
    #[test]
    fn test_wal_flush_noop_for_memory() {
        let mut wal = WriteAheadLog::in_memory();
        wal.commit("a", 0).expect("commit");
        wal.flush().expect("flush should not error");
    }

    /// Disk-backed WAL persists entries and reloads them correctly.
    #[test]
    fn test_wal_disk_persist_and_reload() {
        let dir = std::env::temp_dir();
        let log_path = dir.join("scirs2_io_test_wal_disk.log");
        // Remove any leftover from a previous run.
        let _ = std::fs::remove_file(&log_path);

        {
            let mut wal = WriteAheadLog::new(&log_path).expect("create wal");
            wal.commit("evt-1", 10).expect("commit 1");
            wal.commit("evt-2", 20).expect("commit 2");
            wal.flush().expect("flush");
        }

        // Reload from disk.
        let wal2 = WriteAheadLog::new(&log_path).expect("reload wal");
        assert!(wal2.is_committed("evt-1"));
        assert!(wal2.is_committed("evt-2"));
        assert_eq!(wal2.get_sequence("evt-1"), Some(10));
        assert_eq!(wal2.get_sequence("evt-2"), Some(20));
        assert_eq!(wal2.committed_count(), 2);

        let _ = std::fs::remove_file(&log_path);
    }

    // -----------------------------------------------------------------------
    // ExactlyOnceSink tests
    // -----------------------------------------------------------------------

    /// Submitting the same key twice: second call returns false.
    #[test]
    fn test_sink_duplicate_key_returns_false() {
        let mut sink: ExactlyOnceSink<u32> = ExactlyOnceSink::new();
        assert!(sink.submit("msg-1", 100).expect("first"));
        assert!(!sink.submit("msg-1", 100).expect("second"));
    }

    /// Submitting different keys: both return true.
    #[test]
    fn test_sink_different_keys_both_processed() {
        let mut sink: ExactlyOnceSink<&str> = ExactlyOnceSink::new();
        assert!(sink.submit("a", "alpha").expect("a"));
        assert!(sink.submit("b", "beta").expect("b"));
        assert_eq!(sink.processed_count(), 2);
    }

    /// processed_count matches non-duplicate submissions.
    #[test]
    fn test_sink_processed_count() {
        let mut sink: ExactlyOnceSink<i64> = ExactlyOnceSink::new();
        for i in 0..5 {
            sink.submit(&format!("key-{i}"), i as i64).expect("submit");
        }
        // Re-submit keys 0 and 1 — should be no-ops.
        sink.submit("key-0", 999).expect("dup 0");
        sink.submit("key-1", 999).expect("dup 1");

        assert_eq!(sink.processed_count(), 5);
        assert_eq!(sink.processed_items().len(), 5);
    }

    /// WAL in_memory().is_committed("x") starts false.
    #[test]
    fn test_wal_in_memory_is_committed_starts_false() {
        let wal = WriteAheadLog::in_memory();
        assert!(!wal.is_committed("any-key"));
    }

    /// retry with the same key returns false (no reprocessing).
    #[test]
    fn test_sink_retry_returns_false() {
        let mut sink: ExactlyOnceSink<String> = ExactlyOnceSink::new();
        sink.submit("event-x", "first delivery".into())
            .expect("first");
        let result = sink
            .retry("event-x", "retry delivery".into())
            .expect("retry");
        assert!(!result, "retry must return false");
        assert_eq!(sink.processed_count(), 1);
    }

    /// processed_items preserves insertion order and contains correct data.
    #[test]
    fn test_sink_processed_items_order() {
        let mut sink: ExactlyOnceSink<u32> = ExactlyOnceSink::new();
        sink.submit("first", 1).expect("1");
        sink.submit("second", 2).expect("2");
        sink.submit("third", 3).expect("3");
        sink.submit("second", 99).expect("dup"); // duplicate, ignored

        let items = sink.processed_items();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, "first");
        assert_eq!(items[0].1, 1);
        assert_eq!(items[1].0, "second");
        assert_eq!(items[1].1, 2);
        assert_eq!(items[2].0, "third");
        assert_eq!(items[2].1, 3);
    }

    /// Sink backed by an existing WAL reuses already-committed keys.
    #[test]
    fn test_sink_with_prepopulated_wal() {
        let dir = std::env::temp_dir();
        let log_path = dir.join("scirs2_io_test_sink_wal.log");
        let _ = std::fs::remove_file(&log_path);

        // Pre-populate the WAL.
        {
            let mut wal = WriteAheadLog::new(&log_path).expect("create wal");
            wal.commit("old-key", 0).expect("commit old");
            wal.flush().expect("flush");
        }

        // Build a sink from the pre-populated WAL.
        let wal = WriteAheadLog::new(&log_path).expect("reload wal");
        let mut sink: ExactlyOnceSink<&str> = ExactlyOnceSink::with_wal(wal);

        // "old-key" was committed before — submit should return false.
        assert!(!sink.submit("old-key", "payload").expect("old"));
        // "new-key" is fresh.
        assert!(sink.submit("new-key", "new payload").expect("new"));
        assert_eq!(sink.processed_count(), 1);

        let _ = std::fs::remove_file(&log_path);
    }

    /// Large number of unique keys: all processed, none duplicated.
    #[test]
    fn test_sink_large_unique_key_set() {
        let mut sink: ExactlyOnceSink<usize> = ExactlyOnceSink::new();
        let n = 1000usize;
        for i in 0..n {
            let processed = sink.submit(&format!("k-{i}"), i).expect("submit");
            assert!(processed, "key {i} should be processed");
        }
        assert_eq!(sink.processed_count(), n);
    }
}
