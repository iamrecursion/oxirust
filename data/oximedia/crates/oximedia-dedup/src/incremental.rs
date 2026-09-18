//! Incremental deduplication: only scan new or modified files.
//!
//! Tracks file state (path, size, modification timestamp) across sessions so
//! that subsequent dedup passes only process files that have been added or
//! changed since the last scan.  This dramatically reduces work for large,
//! slowly-evolving media libraries.
//!
//! # Design
//!
//! [`IncrementalIndex`] maintains an in-memory map from file path to
//! [`FileState`] (size + mtime).  On each scan cycle:
//!
//! 1. Walk the candidate file list.
//! 2. Compare each file's current state against the stored state.
//! 3. Classify as **New**, **Modified**, or **Unchanged**.
//! 4. Return only New/Modified files for processing by the dedup pipeline.
//! 5. After processing, update the index with the new state.
//!
//! The index can be serialised to / deserialised from JSON for persistence.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{DedupError, DedupResult};

// ---------------------------------------------------------------------------
// FileState
// ---------------------------------------------------------------------------

/// Snapshot of a file's identity-relevant metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileState {
    /// File size in bytes.
    pub size: u64,
    /// Modification time as seconds since UNIX epoch.
    pub mtime_secs: u64,
    /// BLAKE3 content hash hex (computed on first scan; re-verified on change).
    pub content_hash: Option<String>,
}

impl FileState {
    /// Read the current state of `path` from the filesystem.
    ///
    /// # Errors
    ///
    /// Returns `DedupError::Io` if metadata cannot be read.
    pub fn from_path(path: &Path) -> DedupResult<Self> {
        let meta = std::fs::metadata(path)?;
        let size = meta.len();
        let mtime_secs = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(Self {
            size,
            mtime_secs,
            content_hash: None,
        })
    }

    /// Returns `true` if the file appears unchanged compared to `other`.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.size == other.size && self.mtime_secs == other.mtime_secs
    }
}

// ---------------------------------------------------------------------------
// FileChange
// ---------------------------------------------------------------------------

/// Classification of a file's change status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChange {
    /// File is new (not previously tracked).
    New,
    /// File has been modified (size or mtime changed).
    Modified,
    /// File is unchanged since the last scan.
    Unchanged,
    /// File was previously tracked but no longer exists.
    Deleted,
}

impl FileChange {
    /// Returns `true` if this change requires re-processing.
    #[must_use]
    pub fn needs_processing(self) -> bool {
        matches!(self, Self::New | Self::Modified)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Modified => "modified",
            Self::Unchanged => "unchanged",
            Self::Deleted => "deleted",
        }
    }
}

// ---------------------------------------------------------------------------
// ScanResult
// ---------------------------------------------------------------------------

/// Result of an incremental scan.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Files that need processing (new or modified).
    pub to_process: Vec<PathBuf>,
    /// Files that are unchanged.
    pub unchanged: Vec<PathBuf>,
    /// Files that were deleted since the last scan.
    pub deleted: Vec<PathBuf>,
    /// Per-file change classification.
    pub changes: Vec<(PathBuf, FileChange)>,
}

impl ScanResult {
    /// Total files examined.
    #[must_use]
    pub fn total_examined(&self) -> usize {
        self.to_process.len() + self.unchanged.len()
    }

    /// Fraction of files that need processing (0.0 - 1.0).
    #[must_use]
    pub fn processing_ratio(&self) -> f64 {
        let total = self.total_examined();
        if total == 0 {
            return 0.0;
        }
        self.to_process.len() as f64 / total as f64
    }

    /// Number of new files.
    #[must_use]
    pub fn new_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|(_, c)| *c == FileChange::New)
            .count()
    }

    /// Number of modified files.
    #[must_use]
    pub fn modified_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|(_, c)| *c == FileChange::Modified)
            .count()
    }

    /// Human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} to process ({} new, {} modified), {} unchanged, {} deleted",
            self.to_process.len(),
            self.new_count(),
            self.modified_count(),
            self.unchanged.len(),
            self.deleted.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// IncrementalIndex
// ---------------------------------------------------------------------------

/// Persistent index for incremental deduplication.
///
/// Tracks which files have been seen and their state at the time of the last
/// scan, enabling subsequent scans to skip unchanged files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalIndex {
    /// Map from canonical file path string to its last-known state.
    files: HashMap<String, FileState>,
    /// Epoch timestamp of the last completed scan.
    last_scan_epoch: u64,
    /// Number of scans performed.
    scan_count: u64,
}

impl IncrementalIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            last_scan_epoch: 0,
            scan_count: 0,
        }
    }

    /// Number of tracked files.
    #[must_use]
    pub fn tracked_count(&self) -> usize {
        self.files.len()
    }

    /// Number of scans completed.
    #[must_use]
    pub fn scan_count(&self) -> u64 {
        self.scan_count
    }

    /// Epoch of the last scan.
    #[must_use]
    pub fn last_scan_epoch(&self) -> u64 {
        self.last_scan_epoch
    }

    /// Classify a single file against the stored state.
    ///
    /// # Errors
    ///
    /// Returns an error if the file's metadata cannot be read.
    pub fn classify(&self, path: &Path) -> DedupResult<(FileChange, FileState)> {
        let current = FileState::from_path(path)?;
        let key = path.to_string_lossy().to_string();

        let change = match self.files.get(&key) {
            Some(stored) if stored.matches(&current) => FileChange::Unchanged,
            Some(_) => FileChange::Modified,
            None => FileChange::New,
        };

        Ok((change, current))
    }

    /// Perform an incremental scan over a list of candidate paths.
    ///
    /// Classifies each file, identifies deleted files (tracked but not in
    /// the candidate list), and returns a [`ScanResult`].
    ///
    /// **Does not** update the index -- call `commit` after processing.
    pub fn scan(&self, candidates: &[PathBuf]) -> ScanResult {
        let mut to_process = Vec::new();
        let mut unchanged = Vec::new();
        let mut changes = Vec::new();

        let candidate_set: std::collections::HashSet<String> = candidates
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        for path in candidates {
            match self.classify(path) {
                Ok((change, _state)) => {
                    if change.needs_processing() {
                        to_process.push(path.clone());
                    } else {
                        unchanged.push(path.clone());
                    }
                    changes.push((path.clone(), change));
                }
                Err(_) => {
                    // File cannot be read; treat as deleted/inaccessible
                    changes.push((path.clone(), FileChange::Deleted));
                }
            }
        }

        // Find files that were tracked but are no longer in the candidate list.
        let mut deleted = Vec::new();
        for key in self.files.keys() {
            if !candidate_set.contains(key) {
                deleted.push(PathBuf::from(key));
                changes.push((PathBuf::from(key), FileChange::Deleted));
            }
        }

        ScanResult {
            to_process,
            unchanged,
            deleted,
            changes,
        }
    }

    /// Commit processed files to the index, updating their state.
    ///
    /// Call this after successfully processing the files from a scan.
    pub fn commit(&mut self, paths: &[PathBuf]) {
        for path in paths {
            let key = path.to_string_lossy().to_string();
            if let Ok(state) = FileState::from_path(path) {
                self.files.insert(key, state);
            }
        }
        self.last_scan_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.scan_count += 1;
    }

    /// Commit a single file with an explicit state (e.g. with content hash).
    pub fn commit_file(&mut self, path: &Path, state: FileState) {
        let key = path.to_string_lossy().to_string();
        self.files.insert(key, state);
    }

    /// Remove deleted files from the index.
    pub fn prune_deleted(&mut self, deleted: &[PathBuf]) {
        for path in deleted {
            let key = path.to_string_lossy().to_string();
            self.files.remove(&key);
        }
    }

    /// Get the stored state for a file.
    #[must_use]
    pub fn get_state(&self, path: &Path) -> Option<&FileState> {
        let key = path.to_string_lossy().to_string();
        self.files.get(&key)
    }

    /// Check if a file is tracked.
    #[must_use]
    pub fn is_tracked(&self, path: &Path) -> bool {
        let key = path.to_string_lossy().to_string();
        self.files.contains_key(&key)
    }

    /// Serialise the index to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn to_json(&self) -> DedupResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| DedupError::Hash(format!("JSON serialise: {e}")))
    }

    /// Deserialise an index from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid.
    pub fn from_json(json: &str) -> DedupResult<Self> {
        serde_json::from_str(json).map_err(|e| DedupError::Hash(format!("JSON deserialise: {e}")))
    }

    /// Save the index to a file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be written.
    pub fn save_to_file(&self, path: &Path) -> DedupResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load the index from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load_from_file(path: &Path) -> DedupResult<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Clear the entire index.
    pub fn clear(&mut self) {
        self.files.clear();
        self.last_scan_epoch = 0;
        self.scan_count = 0;
    }

    /// Return all tracked file paths.
    #[must_use]
    pub fn tracked_paths(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }

    /// Merge another index into this one. Files in `other` override this index.
    pub fn merge(&mut self, other: &IncrementalIndex) {
        for (key, state) in &other.files {
            self.files.insert(key.clone(), state.clone());
        }
        self.last_scan_epoch = self.last_scan_epoch.max(other.last_scan_epoch);
    }
}

impl Default for IncrementalIndex {
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

    fn make_temp_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).expect("write temp file");
        path
    }

    #[test]
    fn test_file_state_from_path() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_state");
        let _ = std::fs::create_dir_all(&dir);
        let path = make_temp_file(&dir, "test_state.bin", &[0u8; 100]);

        let state = FileState::from_path(&path).expect("should read state");
        assert_eq!(state.size, 100);
        assert!(state.mtime_secs > 0);
        assert!(state.content_hash.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_file_state_matches() {
        let a = FileState {
            size: 1000,
            mtime_secs: 12345,
            content_hash: None,
        };
        let b = FileState {
            size: 1000,
            mtime_secs: 12345,
            content_hash: Some("abc".to_string()),
        };
        assert!(a.matches(&b)); // content_hash not compared

        let c = FileState {
            size: 2000,
            mtime_secs: 12345,
            content_hash: None,
        };
        assert!(!a.matches(&c));
    }

    #[test]
    fn test_file_change_needs_processing() {
        assert!(FileChange::New.needs_processing());
        assert!(FileChange::Modified.needs_processing());
        assert!(!FileChange::Unchanged.needs_processing());
        assert!(!FileChange::Deleted.needs_processing());
    }

    #[test]
    fn test_file_change_labels() {
        assert_eq!(FileChange::New.label(), "new");
        assert_eq!(FileChange::Modified.label(), "modified");
        assert_eq!(FileChange::Unchanged.label(), "unchanged");
        assert_eq!(FileChange::Deleted.label(), "deleted");
    }

    #[test]
    fn test_incremental_index_new_empty() {
        let idx = IncrementalIndex::new();
        assert_eq!(idx.tracked_count(), 0);
        assert_eq!(idx.scan_count(), 0);
        assert_eq!(idx.last_scan_epoch(), 0);
    }

    #[test]
    fn test_classify_new_file() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_new");
        let _ = std::fs::create_dir_all(&dir);
        let path = make_temp_file(&dir, "new_file.bin", &[1u8; 50]);

        let idx = IncrementalIndex::new();
        let (change, state) = idx.classify(&path).expect("classify");
        assert_eq!(change, FileChange::New);
        assert_eq!(state.size, 50);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_classify_unchanged_file() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_unchanged");
        let _ = std::fs::create_dir_all(&dir);
        let path = make_temp_file(&dir, "unchanged.bin", &[2u8; 75]);

        let mut idx = IncrementalIndex::new();
        idx.commit(std::slice::from_ref(&path));

        let (change, _) = idx.classify(&path).expect("classify");
        assert_eq!(change, FileChange::Unchanged);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_classify_modified_file() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_modified");
        let _ = std::fs::create_dir_all(&dir);
        let path = make_temp_file(&dir, "modifiable.bin", &[3u8; 100]);

        let mut idx = IncrementalIndex::new();
        idx.commit(std::slice::from_ref(&path));

        // Modify the file (change size)
        std::fs::write(&path, &[4u8; 200]).expect("rewrite");

        let (change, _) = idx.classify(&path).expect("classify");
        assert_eq!(change, FileChange::Modified);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scan_mixed_files() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_scan");
        let _ = std::fs::create_dir_all(&dir);

        let f1 = make_temp_file(&dir, "existing.bin", &[5u8; 60]);
        let f2 = make_temp_file(&dir, "new_one.bin", &[6u8; 80]);

        let mut idx = IncrementalIndex::new();
        idx.commit(std::slice::from_ref(&f1));

        let result = idx.scan(&[f1.clone(), f2.clone()]);
        assert_eq!(result.unchanged.len(), 1);
        assert_eq!(result.to_process.len(), 1);
        assert_eq!(result.to_process[0], f2);
        assert_eq!(result.new_count(), 1);
        assert_eq!(result.modified_count(), 0);
        assert!(result.summary().contains("1 to process"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scan_detects_deleted_files() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_deleted");
        let _ = std::fs::create_dir_all(&dir);

        let f1 = make_temp_file(&dir, "will_delete.bin", &[7u8; 40]);
        let f2 = make_temp_file(&dir, "stays.bin", &[8u8; 40]);

        let mut idx = IncrementalIndex::new();
        idx.commit(&[f1.clone(), f2.clone()]);

        // Scan with only f2 in candidates (f1 is "deleted")
        let result = idx.scan(std::slice::from_ref(&f2));
        assert_eq!(result.deleted.len(), 1);
        assert_eq!(result.deleted[0], f1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prune_deleted() {
        let mut idx = IncrementalIndex::new();
        idx.files.insert(
            "/old/file.bin".to_string(),
            FileState {
                size: 100,
                mtime_secs: 0,
                content_hash: None,
            },
        );
        assert_eq!(idx.tracked_count(), 1);

        idx.prune_deleted(&[PathBuf::from("/old/file.bin")]);
        assert_eq!(idx.tracked_count(), 0);
    }

    #[test]
    fn test_commit_updates_scan_count() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_commit");
        let _ = std::fs::create_dir_all(&dir);
        let f = make_temp_file(&dir, "commit_test.bin", &[9u8; 30]);

        let mut idx = IncrementalIndex::new();
        assert_eq!(idx.scan_count(), 0);

        idx.commit(&[f]);
        assert_eq!(idx.scan_count(), 1);
        assert!(idx.last_scan_epoch() > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut idx = IncrementalIndex::new();
        idx.files.insert(
            "/some/file.mp4".to_string(),
            FileState {
                size: 999,
                mtime_secs: 1700000000,
                content_hash: Some("abcd1234".to_string()),
            },
        );
        idx.scan_count = 5;
        idx.last_scan_epoch = 1700000100;

        let json = idx.to_json().expect("serialise");
        let restored = IncrementalIndex::from_json(&json).expect("deserialise");

        assert_eq!(restored.tracked_count(), 1);
        assert_eq!(restored.scan_count(), 5);
        assert_eq!(restored.last_scan_epoch(), 1700000100);

        let state = restored
            .get_state(Path::new("/some/file.mp4"))
            .expect("state should exist");
        assert_eq!(state.size, 999);
        assert_eq!(state.content_hash.as_deref(), Some("abcd1234"));
    }

    #[test]
    fn test_save_and_load_file() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_persist");
        let _ = std::fs::create_dir_all(&dir);
        let index_path = dir.join("dedup_index.json");

        let mut idx = IncrementalIndex::new();
        idx.files.insert(
            "video.mp4".to_string(),
            FileState {
                size: 500,
                mtime_secs: 12345,
                content_hash: None,
            },
        );

        idx.save_to_file(&index_path).expect("save");
        let loaded = IncrementalIndex::load_from_file(&index_path).expect("load");
        assert_eq!(loaded.tracked_count(), 1);
        assert!(loaded.is_tracked(Path::new("video.mp4")));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merge_indices() {
        let mut idx1 = IncrementalIndex::new();
        idx1.files.insert(
            "a.mp4".to_string(),
            FileState {
                size: 100,
                mtime_secs: 1,
                content_hash: None,
            },
        );
        idx1.last_scan_epoch = 100;

        let mut idx2 = IncrementalIndex::new();
        idx2.files.insert(
            "b.mp4".to_string(),
            FileState {
                size: 200,
                mtime_secs: 2,
                content_hash: None,
            },
        );
        idx2.last_scan_epoch = 200;

        idx1.merge(&idx2);
        assert_eq!(idx1.tracked_count(), 2);
        assert!(idx1.is_tracked(Path::new("a.mp4")));
        assert!(idx1.is_tracked(Path::new("b.mp4")));
        assert_eq!(idx1.last_scan_epoch(), 200);
    }

    #[test]
    fn test_commit_file_with_hash() {
        let mut idx = IncrementalIndex::new();
        let state = FileState {
            size: 1024,
            mtime_secs: 1700000000,
            content_hash: Some("deadbeef".to_string()),
        };
        idx.commit_file(Path::new("/media/video.mp4"), state);

        let stored = idx
            .get_state(Path::new("/media/video.mp4"))
            .expect("should exist");
        assert_eq!(stored.content_hash.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn test_clear_index() {
        let mut idx = IncrementalIndex::new();
        idx.files.insert(
            "x.mp4".to_string(),
            FileState {
                size: 1,
                mtime_secs: 1,
                content_hash: None,
            },
        );
        idx.scan_count = 10;
        idx.clear();
        assert_eq!(idx.tracked_count(), 0);
        assert_eq!(idx.scan_count(), 0);
    }

    #[test]
    fn test_tracked_paths() {
        let mut idx = IncrementalIndex::new();
        idx.files.insert(
            "a.mp4".to_string(),
            FileState {
                size: 1,
                mtime_secs: 1,
                content_hash: None,
            },
        );
        idx.files.insert(
            "b.mp4".to_string(),
            FileState {
                size: 2,
                mtime_secs: 2,
                content_hash: None,
            },
        );
        let mut paths = idx.tracked_paths();
        paths.sort();
        assert_eq!(paths, vec!["a.mp4", "b.mp4"]);
    }

    #[test]
    fn test_processing_ratio() {
        let result = ScanResult {
            to_process: vec![PathBuf::from("a"), PathBuf::from("b")],
            unchanged: vec![PathBuf::from("c"), PathBuf::from("d"), PathBuf::from("e")],
            deleted: Vec::new(),
            changes: Vec::new(),
        };
        assert!((result.processing_ratio() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_processing_ratio_empty() {
        let result = ScanResult {
            to_process: Vec::new(),
            unchanged: Vec::new(),
            deleted: Vec::new(),
            changes: Vec::new(),
        };
        assert_eq!(result.processing_ratio(), 0.0);
    }

    #[test]
    fn test_full_incremental_workflow() {
        let dir = std::env::temp_dir().join("oximedia_dedup_incr_workflow");
        let _ = std::fs::create_dir_all(&dir);

        // Session 1: All files are new
        let f1 = make_temp_file(&dir, "video1.bin", &[10u8; 100]);
        let f2 = make_temp_file(&dir, "video2.bin", &[20u8; 200]);

        let mut idx = IncrementalIndex::new();
        let scan1 = idx.scan(&[f1.clone(), f2.clone()]);
        assert_eq!(scan1.to_process.len(), 2);
        assert_eq!(scan1.new_count(), 2);

        idx.commit(&scan1.to_process);

        // Session 2: No changes -> nothing to process
        let scan2 = idx.scan(&[f1.clone(), f2.clone()]);
        assert_eq!(scan2.to_process.len(), 0);
        assert_eq!(scan2.unchanged.len(), 2);

        // Session 3: Modify one file, add a new one
        std::fs::write(&f1, &[11u8; 150]).expect("modify f1");
        let f3 = make_temp_file(&dir, "video3.bin", &[30u8; 300]);

        let scan3 = idx.scan(&[f1.clone(), f2.clone(), f3.clone()]);
        assert_eq!(scan3.to_process.len(), 2); // f1 (modified) + f3 (new)
        assert_eq!(scan3.unchanged.len(), 1); // f2
        assert_eq!(scan3.modified_count(), 1);
        assert_eq!(scan3.new_count(), 1);

        idx.commit(&scan3.to_process);
        assert_eq!(idx.scan_count(), 2);
        assert_eq!(idx.tracked_count(), 3);

        // Session 4: Delete f2
        let scan4 = idx.scan(&[f1.clone(), f3.clone()]);
        assert_eq!(scan4.deleted.len(), 1);
        idx.prune_deleted(&scan4.deleted);
        assert_eq!(idx.tracked_count(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
