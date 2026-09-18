//! Persistent-style deduplication index.
//!
//! Tracks content hashes together with occurrence counts and space-saving
//! metrics.  The index operates entirely in memory; persistence to disk can
//! be added by serialising `DedupIndex` with serde.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::path::{Path, PathBuf};

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// DedupEntry
// ---------------------------------------------------------------------------

/// A single entry in the deduplication index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupEntry {
    /// Unique identifier assigned by the index.
    pub id: u64,
    /// Content hash (arbitrary bytes, e.g. BLAKE3 digest).
    pub content_hash: Vec<u8>,
    /// Size of the deduplicated content in bytes.
    pub size_bytes: u64,
    /// Unix epoch timestamp when this hash was first seen.
    pub first_seen_epoch: u64,
    /// Number of times this hash has been seen (≥ 1 after creation).
    pub occurrence_count: u32,
}

impl DedupEntry {
    /// Returns `true` if this hash has been seen more than once.
    #[must_use]
    pub fn is_duplicate(&self) -> bool {
        self.occurrence_count > 1
    }

    /// Space saved by deduplication.
    ///
    /// If the content appeared N times, only one copy is stored.
    /// Savings = `(N - 1) * size_bytes`.
    #[must_use]
    pub fn space_savings(&self) -> u64 {
        if self.occurrence_count <= 1 {
            return 0;
        }
        (self.occurrence_count as u64 - 1).saturating_mul(self.size_bytes)
    }

    /// Return the last-seen epoch, which equals `first_seen_epoch` until we
    /// track updates (here we simply alias for interface completeness).
    #[must_use]
    pub fn first_seen(&self) -> u64 {
        self.first_seen_epoch
    }

    /// Hex representation of the content hash.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        self.content_hash
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DedupIndex
// ---------------------------------------------------------------------------

/// In-memory deduplication index.
///
/// Each unique content hash is stored once; subsequent insertions increment
/// the occurrence counter.
pub struct DedupIndex {
    /// All known entries, ordered by insertion.
    pub entries: Vec<DedupEntry>,
    /// Next ID to assign.
    pub next_id: u64,
}

impl DedupIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a content hash to the index, or increment its counter if already present.
    ///
    /// Returns the `id` of the (existing or newly created) entry.
    pub fn add_or_increment(&mut self, hash: Vec<u8>, size_bytes: u64, epoch: u64) -> u64 {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.content_hash == hash) {
            entry.occurrence_count += 1;
            return entry.id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(DedupEntry {
            id,
            content_hash: hash,
            size_bytes,
            first_seen_epoch: epoch,
            occurrence_count: 1,
        });
        id
    }

    /// Find an entry by its content hash.
    #[must_use]
    pub fn find_by_hash(&self, hash: &[u8]) -> Option<&DedupEntry> {
        self.entries.iter().find(|e| e.content_hash == hash)
    }

    /// Find an entry by its assigned ID.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&DedupEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Return all entries that are duplicates (occurrence_count > 1).
    #[must_use]
    pub fn find_duplicates(&self) -> Vec<&DedupEntry> {
        self.entries.iter().filter(|e| e.is_duplicate()).collect()
    }

    /// Total space saved across all duplicate entries.
    #[must_use]
    pub fn total_space_savings(&self) -> u64 {
        self.entries.iter().map(|e| e.space_savings()).sum()
    }

    /// Number of unique content hashes in the index.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.entries.len()
    }

    /// Total number of content insertions (sum of all occurrence counts).
    #[must_use]
    pub fn total_insertions(&self) -> u64 {
        self.entries.iter().map(|e| e.occurrence_count as u64).sum()
    }

    /// Remove an entry by hash.  Returns `true` if an entry was removed.
    pub fn remove_by_hash(&mut self, hash: &[u8]) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.content_hash == hash) {
            self.entries.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Clear the entire index.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 1;
    }

    /// Return entries sorted by occurrence count (descending).
    #[must_use]
    pub fn most_common(&self) -> Vec<&DedupEntry> {
        let mut sorted: Vec<&DedupEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));
        sorted
    }

    /// Report: percentage of insertions that were duplicates.
    ///
    /// Returns `0.0` if no insertions have been made.
    #[must_use]
    pub fn duplicate_rate(&self) -> f64 {
        let total = self.total_insertions();
        if total == 0 {
            return 0.0;
        }
        let unique = self.unique_count() as u64;
        let dupes = total.saturating_sub(unique);
        dupes as f64 / total as f64
    }
}

impl Default for DedupIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Parallel add_files support
// ---------------------------------------------------------------------------

/// Features extracted from a file for deduplication purposes.
#[derive(Debug, Clone)]
pub struct FileFeatures {
    /// Path of the source file.
    pub path: PathBuf,
    /// BLAKE3-style content hash of the file (or a synthetic hash for tests).
    pub content_hash: Vec<u8>,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Modification time as seconds since UNIX epoch.
    pub mtime_epoch: u64,
}

/// Compute features for a single file.
///
/// Uses a streaming FNV-1a hash over the file bytes as a lightweight
/// approximation of a cryptographic digest.  For production use, callers
/// should substitute a BLAKE3 hasher here.
pub fn compute_file_features(path: &Path) -> std::io::Result<FileFeatures> {
    use std::io::Read;
    use std::time::UNIX_EPOCH;

    let meta = std::fs::metadata(path)?;
    let size_bytes = meta.len();
    let mtime_epoch = meta
        .modified()
        .unwrap_or(UNIX_EPOCH)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // FNV-1a streaming hash over file contents.
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash: u64 = FNV_OFFSET;

    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(65_536, file);
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            hash = (hash ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    }

    Ok(FileFeatures {
        path: path.to_path_buf(),
        content_hash: hash.to_le_bytes().to_vec(),
        size_bytes,
        mtime_epoch,
    })
}

impl DedupIndex {
    /// Add multiple files to the index in parallel (feature extraction) and
    /// insert results sequentially.
    ///
    /// Files that cannot be read are silently skipped; their paths are returned
    /// in the error list.
    ///
    /// # Returns
    ///
    /// A tuple `(added_ids, skipped_paths)`.
    pub fn add_files(&mut self, paths: &[impl AsRef<Path> + Sync]) -> (Vec<u64>, Vec<PathBuf>) {
        // --- parallel feature extraction ---
        let results: Vec<(PathBuf, std::io::Result<FileFeatures>)> = paths
            .par_iter()
            .map(|p| {
                let path = p.as_ref().to_path_buf();
                let feat = compute_file_features(&path);
                (path, feat)
            })
            .collect();

        // --- sequential DB insertion ---
        let mut added_ids = Vec::with_capacity(results.len());
        let mut skipped = Vec::new();

        for (path, result) in results {
            match result {
                Ok(feat) => {
                    let id =
                        self.add_or_increment(feat.content_hash, feat.size_bytes, feat.mtime_epoch);
                    added_ids.push(id);
                }
                Err(_) => {
                    skipped.push(path);
                }
            }
        }

        (added_ids, skipped)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn hash(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    /// Write `n` temp files with distinct content and return their paths.
    fn make_temp_files(n: usize) -> Vec<std::path::PathBuf> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let base = temp_dir();
        let pid = std::process::id();
        (0..n)
            .map(|i| {
                let uid = COUNTER.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!("dedup_idx_{pid}_{uid}_{i}.bin"));
                std::fs::write(&path, format!("synthetic content for file {pid}_{uid}_{i}"))
                    .expect("write temp file");
                path
            })
            .collect()
    }

    // ---- DedupEntry tests ----

    #[test]
    fn test_entry_is_duplicate_false_when_once() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1024,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert!(!e.is_duplicate());
    }

    #[test]
    fn test_entry_is_duplicate_true_when_multiple() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1024,
            first_seen_epoch: 0,
            occurrence_count: 3,
        };
        assert!(e.is_duplicate());
    }

    #[test]
    fn test_entry_space_savings_zero_when_once() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 500,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert_eq!(e.space_savings(), 0);
    }

    #[test]
    fn test_entry_space_savings_correct() {
        let e = DedupEntry {
            id: 1,
            content_hash: hash("abc"),
            size_bytes: 1000,
            first_seen_epoch: 0,
            occurrence_count: 4,
        };
        // (4-1) * 1000 = 3000
        assert_eq!(e.space_savings(), 3000);
    }

    #[test]
    fn test_entry_hash_hex() {
        let e = DedupEntry {
            id: 1,
            content_hash: vec![0xDE, 0xAD, 0xBE, 0xEF],
            size_bytes: 0,
            first_seen_epoch: 0,
            occurrence_count: 1,
        };
        assert_eq!(e.hash_hex(), "deadbeef");
    }

    // ---- DedupIndex tests ----

    #[test]
    fn test_index_new_empty() {
        let idx = DedupIndex::new();
        assert_eq!(idx.unique_count(), 0);
        assert_eq!(idx.total_insertions(), 0);
        assert_eq!(idx.total_space_savings(), 0);
    }

    #[test]
    fn test_add_or_increment_new_entry() {
        let mut idx = DedupIndex::new();
        let id = idx.add_or_increment(hash("file_a"), 100, 1000);
        assert_eq!(idx.unique_count(), 1);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_add_or_increment_existing_entry() {
        let mut idx = DedupIndex::new();
        let id1 = idx.add_or_increment(hash("file_a"), 100, 1000);
        let id2 = idx.add_or_increment(hash("file_a"), 100, 2000);
        assert_eq!(id1, id2, "Same hash should return same ID");
        assert_eq!(idx.unique_count(), 1, "Should still be one unique entry");
        let entry = idx
            .find_by_hash(&hash("file_a"))
            .expect("operation should succeed");
        assert_eq!(entry.occurrence_count, 2);
    }

    #[test]
    fn test_find_by_hash_found() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("alpha"), 200, 0);
        let entry = idx.find_by_hash(&hash("alpha"));
        assert!(entry.is_some());
        assert_eq!(entry.expect("operation should succeed").size_bytes, 200);
    }

    #[test]
    fn test_find_by_hash_not_found() {
        let idx = DedupIndex::new();
        assert!(idx.find_by_hash(&hash("missing")).is_none());
    }

    #[test]
    fn test_find_by_id() {
        let mut idx = DedupIndex::new();
        let id = idx.add_or_increment(hash("entry_1"), 512, 100);
        let entry = idx.find_by_id(id);
        assert!(entry.is_some());
        assert_eq!(
            entry.expect("operation should succeed").content_hash,
            hash("entry_1")
        );
    }

    #[test]
    fn test_find_duplicates() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("unique"), 10, 0);
        idx.add_or_increment(hash("dup"), 20, 0);
        idx.add_or_increment(hash("dup"), 20, 1);
        idx.add_or_increment(hash("dup"), 20, 2);

        let dups = idx.find_duplicates();
        assert_eq!(dups.len(), 1);
        assert_eq!(dups[0].content_hash, hash("dup"));
        assert_eq!(dups[0].occurrence_count, 3);
    }

    #[test]
    fn test_total_space_savings() {
        let mut idx = DedupIndex::new();
        // 1 occurrence → 0 savings
        idx.add_or_increment(hash("a"), 100, 0);
        // 3 occurrences → (3-1)*200 = 400 savings
        idx.add_or_increment(hash("b"), 200, 0);
        idx.add_or_increment(hash("b"), 200, 1);
        idx.add_or_increment(hash("b"), 200, 2);

        assert_eq!(idx.total_space_savings(), 400);
    }

    #[test]
    fn test_remove_by_hash() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("to_remove"), 50, 0);
        assert_eq!(idx.unique_count(), 1);
        let removed = idx.remove_by_hash(&hash("to_remove"));
        assert!(removed);
        assert_eq!(idx.unique_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut idx = DedupIndex::new();
        assert!(!idx.remove_by_hash(&hash("ghost")));
    }

    #[test]
    fn test_clear() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("x"), 10, 0);
        idx.add_or_increment(hash("y"), 20, 0);
        idx.clear();
        assert_eq!(idx.unique_count(), 0);
        assert_eq!(idx.next_id, 1);
    }

    #[test]
    fn test_duplicate_rate() {
        let mut idx = DedupIndex::new();
        // 1 unique, seen 3 times → 2 duplicate insertions out of 3 total ≈ 0.667
        idx.add_or_increment(hash("h"), 100, 0);
        idx.add_or_increment(hash("h"), 100, 1);
        idx.add_or_increment(hash("h"), 100, 2);

        let rate = idx.duplicate_rate();
        let expected = 2.0 / 3.0;
        assert!((rate - expected).abs() < 1e-9);
    }

    #[test]
    fn test_most_common_ordering() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("rare"), 10, 0);
        idx.add_or_increment(hash("common"), 100, 0);
        idx.add_or_increment(hash("common"), 100, 1);
        idx.add_or_increment(hash("common"), 100, 2);
        idx.add_or_increment(hash("mid"), 50, 0);
        idx.add_or_increment(hash("mid"), 50, 1);

        let mc = idx.most_common();
        assert_eq!(mc[0].content_hash, hash("common"));
        assert_eq!(mc[1].content_hash, hash("mid"));
        assert_eq!(mc[2].content_hash, hash("rare"));
    }

    #[test]
    fn test_total_insertions() {
        let mut idx = DedupIndex::new();
        idx.add_or_increment(hash("a"), 1, 0);
        idx.add_or_increment(hash("a"), 1, 1);
        idx.add_or_increment(hash("b"), 1, 0);
        // a: 2, b: 1 → total = 3
        assert_eq!(idx.total_insertions(), 3);
    }

    // ── add_files parallel tests ──────────────────────────────────────────────

    #[test]
    fn test_add_files_parallel_same_result_as_sequential() {
        let paths = make_temp_files(10);

        // Parallel path.
        let mut par_idx = DedupIndex::new();
        let (par_ids, par_skipped) = par_idx.add_files(&paths);
        assert!(par_skipped.is_empty(), "no files should be skipped");
        assert_eq!(par_ids.len(), 10, "all 10 files should produce an ID");

        // Sequential reference path: compute features one by one.
        let mut seq_idx = DedupIndex::new();
        let mut seq_ids = Vec::with_capacity(paths.len());
        for p in &paths {
            let feat = compute_file_features(p.as_ref()).expect("compute features");
            let id = seq_idx.add_or_increment(feat.content_hash, feat.size_bytes, feat.mtime_epoch);
            seq_ids.push(id);
        }

        // Both indexes should contain the same number of unique hashes.
        assert_eq!(
            par_idx.unique_count(),
            seq_idx.unique_count(),
            "unique count must match between parallel and sequential"
        );

        // Each file had distinct content, so all IDs should be unique (no dups).
        assert_eq!(par_idx.find_duplicates().len(), 0);
        assert_eq!(seq_idx.find_duplicates().len(), 0);

        // Clean up.
        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn test_add_files_skips_missing_files() {
        let real = make_temp_files(3);
        let mut all_paths = real.clone();
        all_paths.push(temp_dir().join("nonexistent_dedup_test_file.bin"));

        let mut idx = DedupIndex::new();
        let (_ids, skipped) = idx.add_files(&all_paths);
        assert_eq!(
            skipped.len(),
            1,
            "exactly one missing file should be skipped"
        );
        assert_eq!(idx.unique_count(), 3, "three real files should be indexed");

        for p in &real {
            let _ = std::fs::remove_file(p);
        }
    }
}
