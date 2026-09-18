//! File handle tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// File handle information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHandle {
    /// File path.
    pub path: PathBuf,

    /// When opened.
    #[serde(skip, default = "Instant::now")]
    pub opened_at: Instant,

    /// Access mode.
    pub mode: FileMode,

    /// Number of reads.
    pub read_count: u64,

    /// Number of writes.
    pub write_count: u64,
}

/// File access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileMode {
    /// Read-only.
    Read,

    /// Write-only.
    Write,

    /// Read-write.
    ReadWrite,
}

/// File handle tracker.
#[derive(Debug)]
pub struct FileTracker {
    handles: HashMap<u64, FileHandle>,
    next_id: u64,
}

impl FileTracker {
    /// Create a new file tracker.
    pub fn new() -> Self {
        Self {
            handles: HashMap::new(),
            next_id: 0,
        }
    }

    /// Track a file open.
    pub fn track_open(&mut self, path: PathBuf, mode: FileMode) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let handle = FileHandle {
            path,
            opened_at: Instant::now(),
            mode,
            read_count: 0,
            write_count: 0,
        };

        self.handles.insert(id, handle);
        id
    }

    /// Track a file close.
    pub fn track_close(&mut self, id: u64) -> bool {
        self.handles.remove(&id).is_some()
    }

    /// Track a read operation.
    pub fn track_read(&mut self, id: u64) {
        if let Some(handle) = self.handles.get_mut(&id) {
            handle.read_count += 1;
        }
    }

    /// Track a write operation.
    pub fn track_write(&mut self, id: u64) {
        if let Some(handle) = self.handles.get_mut(&id) {
            handle.write_count += 1;
        }
    }

    /// Get open file count.
    pub fn open_count(&self) -> usize {
        self.handles.len()
    }

    /// Get all open files.
    pub fn open_files(&self) -> Vec<&FileHandle> {
        self.handles.values().collect()
    }
}

impl Default for FileTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_tracker() {
        let mut tracker = FileTracker::new();
        let id = tracker.track_open(PathBuf::from("test.txt"), FileMode::Read);
        assert_eq!(tracker.open_count(), 1);

        tracker.track_close(id);
        assert_eq!(tracker.open_count(), 0);
    }

    #[test]
    fn test_file_operations() {
        let mut tracker = FileTracker::new();
        let id = tracker.track_open(PathBuf::from("test.txt"), FileMode::ReadWrite);

        tracker.track_read(id);
        tracker.track_write(id);

        let handle = tracker.handles.get(&id).expect("should succeed in test");
        assert_eq!(handle.read_count, 1);
        assert_eq!(handle.write_count, 1);
    }
}
