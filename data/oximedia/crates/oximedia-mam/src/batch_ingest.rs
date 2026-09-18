//! Batch ingest pipeline for media assets
//!
//! Provides utilities for planning, validating and executing batch ingest
//! operations across large sets of media files, including watch-folder
//! support for automatic ingest of newly appearing files.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// A single item to be ingested in a batch operation
#[derive(Debug, Clone)]
pub struct IngestItem {
    /// Source file path
    pub path: String,
    /// Destination path or folder within the MAM
    pub destination: String,
    /// Additional metadata to attach on ingest
    pub metadata: HashMap<String, String>,
    /// Processing priority (0 = lowest, 255 = highest)
    pub priority: u8,
}

impl IngestItem {
    /// Create a new ingest item
    #[must_use]
    pub fn new(path: impl Into<String>, destination: impl Into<String>, priority: u8) -> Self {
        Self {
            path: path.into(),
            destination: destination.into(),
            metadata: HashMap::new(),
            priority,
        }
    }

    /// Builder-style method to add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Outcome of a single ingest attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestStatus {
    /// File was ingested successfully
    Success,
    /// Ingest encountered a fatal error
    Failed,
    /// File was intentionally skipped (e.g. unsupported format)
    Skipped,
    /// File was already present in the MAM (duplicate detected)
    Duplicate,
}

impl IngestStatus {
    /// Returns `true` if the status indicates a successful ingest
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Human-readable label
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Duplicate => "duplicate",
        }
    }
}

/// Result of a single ingest item being processed
#[derive(Debug, Clone)]
pub struct IngestResult {
    /// The item that was processed
    pub item: IngestItem,
    /// Final status of the ingest attempt
    pub status: IngestStatus,
    /// Error message if the ingest failed
    pub error: Option<String>,
    /// Time taken to process the item in milliseconds
    pub duration_ms: u64,
    /// Asset ID assigned by the MAM on success
    pub asset_id: Option<String>,
}

impl IngestResult {
    /// Create a successful result
    #[must_use]
    pub fn success(item: IngestItem, duration_ms: u64, asset_id: String) -> Self {
        Self {
            item,
            status: IngestStatus::Success,
            error: None,
            duration_ms,
            asset_id: Some(asset_id),
        }
    }

    /// Create a failed result
    #[must_use]
    pub fn failed(item: IngestItem, duration_ms: u64, error: impl Into<String>) -> Self {
        Self {
            item,
            status: IngestStatus::Failed,
            error: Some(error.into()),
            duration_ms,
            asset_id: None,
        }
    }

    /// Create a skipped result
    #[must_use]
    pub fn skipped(item: IngestItem, reason: impl Into<String>) -> Self {
        Self {
            item,
            status: IngestStatus::Skipped,
            error: Some(reason.into()),
            duration_ms: 0,
            asset_id: None,
        }
    }

    /// Create a duplicate result
    #[must_use]
    pub fn duplicate(item: IngestItem, existing_asset_id: impl Into<String>) -> Self {
        Self {
            item,
            status: IngestStatus::Duplicate,
            error: None,
            duration_ms: 0,
            asset_id: Some(existing_asset_id.into()),
        }
    }
}

/// Validates individual ingest items before they are processed
pub struct IngestValidator;

impl IngestValidator {
    /// Validate that a path string is syntactically acceptable.
    ///
    /// Returns `Ok(())` if valid, or `Err(reason)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error string if the path is empty or contains
    /// characters that are not allowed in a file path.
    pub fn validate_path(path: &str) -> Result<(), String> {
        if path.is_empty() {
            return Err("Path must not be empty".to_string());
        }

        // Reject null bytes
        if path.contains('\0') {
            return Err("Path contains null bytes".to_string());
        }

        // Reject path traversal attempts
        if path.contains("..") {
            return Err("Path contains directory traversal sequence (..)".to_string());
        }

        Ok(())
    }

    /// Check whether a checksum is already present in the list of existing checksums.
    ///
    /// Returns `true` if this is a duplicate.
    #[must_use]
    pub fn check_duplicate(checksum: &str, existing: &[String]) -> bool {
        existing.iter().any(|e| e == checksum)
    }
}

/// A plan for a batch ingest operation
#[derive(Debug, Clone)]
pub struct BatchIngestPlan {
    /// Items to be ingested
    pub items: Vec<IngestItem>,
    /// Total estimated size of all source files in bytes
    pub total_size_bytes: u64,
    /// Estimated total duration in seconds
    pub estimated_duration_secs: f64,
}

impl BatchIngestPlan {
    /// Create a batch ingest plan
    #[must_use]
    pub fn new(
        items: Vec<IngestItem>,
        total_size_bytes: u64,
        estimated_duration_secs: f64,
    ) -> Self {
        Self {
            items,
            total_size_bytes,
            estimated_duration_secs,
        }
    }

    /// Return the number of items in the plan
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Estimate throughput in MB/s
    #[must_use]
    pub fn estimated_throughput_mbps(&self) -> f64 {
        if self.estimated_duration_secs <= 0.0 {
            return 0.0;
        }
        (self.total_size_bytes as f64 / 1_048_576.0) / self.estimated_duration_secs
    }
}

/// Summary report for a completed batch ingest
#[derive(Debug, Clone)]
pub struct BatchIngestReport {
    /// Individual results for every item in the batch
    pub results: Vec<IngestResult>,
    /// Number of successfully ingested items
    pub success_count: u32,
    /// Number of items that failed
    pub failed_count: u32,
    /// Total number of new assets created in the MAM
    pub total_assets_created: u32,
}

impl BatchIngestReport {
    /// Build a report from a list of results
    #[must_use]
    pub fn from_results(results: Vec<IngestResult>) -> Self {
        let success_count = results.iter().filter(|r| r.status.is_success()).count() as u32;
        let failed_count = results
            .iter()
            .filter(|r| r.status == IngestStatus::Failed)
            .count() as u32;
        let total_assets_created = results
            .iter()
            .filter(|r| r.status.is_success() && r.asset_id.is_some())
            .count() as u32;

        Self {
            results,
            success_count,
            failed_count,
            total_assets_created,
        }
    }

    /// Returns the fraction of items that succeeded (0.0 – 1.0)
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let total = self.results.len();
        if total == 0 {
            return 1.0;
        }
        self.success_count as f32 / total as f32
    }

    /// Returns the number of skipped items
    #[must_use]
    pub fn skipped_count(&self) -> u32 {
        self.results
            .iter()
            .filter(|r| r.status == IngestStatus::Skipped)
            .count() as u32
    }

    /// Returns the number of duplicate items
    #[must_use]
    pub fn duplicate_count(&self) -> u32 {
        self.results
            .iter()
            .filter(|r| r.status == IngestStatus::Duplicate)
            .count() as u32
    }
}

/// Priority queue for ingest items (higher priority value = processed sooner)
#[derive(Debug, Default)]
pub struct IngestQueue {
    /// Items stored in the queue (not sorted; popping performs a linear scan)
    items: Vec<IngestItem>,
}

impl IngestQueue {
    /// Create a new empty queue
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an item onto the queue
    pub fn push(&mut self, item: IngestItem) {
        self.items.push(item);
    }

    /// Pop the highest-priority item from the queue.
    ///
    /// When multiple items share the same priority, the one inserted
    /// earliest is returned first (stable FIFO within a priority level).
    #[must_use]
    pub fn pop(&mut self) -> Option<IngestItem> {
        if self.items.is_empty() {
            return None;
        }

        // Find the index of the highest-priority item (stable: first occurrence wins on tie)
        let idx = self
            .items
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                a.priority.cmp(&b.priority).then_with(|| ib.cmp(ia)) // earlier index wins on tie
            })
            .map(|(i, _)| i)?;

        Some(self.items.remove(idx))
    }

    /// Return the number of items currently in the queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Return `true` if the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Peek at the highest-priority item without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&IngestItem> {
        self.items.iter().max_by(|a, b| a.priority.cmp(&b.priority))
    }
}

// ---------------------------------------------------------------------------
// Watch Folder Support
// ---------------------------------------------------------------------------

/// File extensions that are eligible for automatic watch-folder ingest.
pub const DEFAULT_MEDIA_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "mkv", "avi", "mxf", "ts", "m2ts", "webm", "flv", "mp3", "wav", "flac", "aiff",
    "ogg", "opus", "m4a", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "gif", "webp", "dpx", "exr",
    "pdf",
];

/// Minimum interval between consecutive poll cycles to avoid tight-looping.
const MIN_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Strategy used to decide whether a file is fully written before ingesting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityStrategy {
    /// Ingest once the file's mtime has not changed for `stable_secs` seconds.
    MtimeQuiesce,
    /// Ingest immediately when the file appears (suitable for atomic moves).
    Immediate,
}

/// Configuration for a watch folder monitor.
#[derive(Debug, Clone)]
pub struct WatchFolderConfig {
    /// Root directory to watch (not recursive by default).
    pub folder: PathBuf,
    /// Destination path/collection inside the MAM.
    pub destination: String,
    /// How often to scan the directory.
    pub poll_interval: Duration,
    /// How long a file must be stable (mtime not changing) before ingesting.
    pub stable_secs: u64,
    /// File stability strategy.
    pub stability_strategy: StabilityStrategy,
    /// File extensions to accept (case-insensitive). Empty = accept all.
    pub allowed_extensions: Vec<String>,
    /// Priority assigned to watch-folder ingest items.
    pub priority: u8,
    /// Recurse into sub-directories.
    pub recursive: bool,
    /// Metadata to attach to every automatically ingested item.
    pub default_metadata: HashMap<String, String>,
}

impl WatchFolderConfig {
    /// Create a new config with sensible defaults.
    #[must_use]
    pub fn new(folder: impl Into<PathBuf>, destination: impl Into<String>) -> Self {
        Self {
            folder: folder.into(),
            destination: destination.into(),
            poll_interval: Duration::from_secs(10),
            stable_secs: 5,
            stability_strategy: StabilityStrategy::MtimeQuiesce,
            allowed_extensions: DEFAULT_MEDIA_EXTENSIONS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            priority: 128,
            recursive: false,
            default_metadata: HashMap::new(),
        }
    }

    /// Set the poll interval.
    #[must_use]
    pub fn with_poll_interval(mut self, d: Duration) -> Self {
        self.poll_interval = d.max(MIN_POLL_INTERVAL);
        self
    }

    /// Set the file-stability quiesce time.
    #[must_use]
    pub fn with_stable_secs(mut self, secs: u64) -> Self {
        self.stable_secs = secs;
        self
    }

    /// Switch to the `Immediate` stability strategy.
    #[must_use]
    pub fn immediate(mut self) -> Self {
        self.stability_strategy = StabilityStrategy::Immediate;
        self
    }

    /// Enable recursive directory watching.
    #[must_use]
    pub fn recursive(mut self) -> Self {
        self.recursive = true;
        self
    }

    /// Override the allowed-extension list.
    #[must_use]
    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.allowed_extensions = exts;
        self
    }

    /// Check whether a file extension is allowed.
    #[must_use]
    pub fn is_extension_allowed(&self, path: &Path) -> bool {
        if self.allowed_extensions.is_empty() {
            return true;
        }
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| {
                let lower = e.to_lowercase();
                self.allowed_extensions
                    .iter()
                    .any(|allowed| allowed.to_lowercase() == lower)
            })
            .unwrap_or(false)
    }
}

/// Internal tracking record for a file seen in the watch folder.
#[derive(Debug, Clone)]
struct WatchedFile {
    /// Absolute path to the file.
    path: PathBuf,
    /// Last observed modification time.
    last_mtime: SystemTime,
    /// When this file was first noticed by the watcher.
    first_seen: Instant,
}

/// Result of a single watch-folder scan cycle.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Newly discovered candidate files.
    pub new_candidates: usize,
    /// Files promoted to the ingest queue this cycle.
    pub queued: usize,
    /// Files still waiting for stability.
    pub pending_stability: usize,
    /// Files skipped due to extension filtering.
    pub skipped_extension: usize,
}

/// State shared between the watch-folder scanner and external callers.
pub struct WatchFolderState {
    config: WatchFolderConfig,
    /// Files currently being tracked (path → record).
    pending: HashMap<PathBuf, WatchedFile>,
    /// Paths that have already been queued (to avoid re-queuing on restart).
    ingested: HashSet<PathBuf>,
    /// Ingest items ready to be processed.
    ready_queue: IngestQueue,
    /// Total files queued since the watcher was created.
    total_queued: u64,
}

impl WatchFolderState {
    /// Create a new watch-folder state from the provided config.
    #[must_use]
    pub fn new(config: WatchFolderConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            ingested: HashSet::new(),
            ready_queue: IngestQueue::new(),
            total_queued: 0,
        }
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &WatchFolderConfig {
        &self.config
    }

    /// Return total files queued since creation.
    #[must_use]
    pub fn total_queued(&self) -> u64 {
        self.total_queued
    }

    /// Number of items currently in the ready queue.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Pop the next ready item from the ingest queue.
    pub fn pop_ready(&mut self) -> Option<IngestItem> {
        self.ready_queue.pop()
    }

    /// Mark a path as successfully ingested so it is not re-queued.
    pub fn mark_ingested(&mut self, path: PathBuf) {
        self.pending.remove(&path);
        self.ingested.insert(path);
    }

    /// Scan the watch folder once, updating internal state.
    ///
    /// This performs a **synchronous** directory read and is intended to be
    /// called from within a dedicated polling loop (e.g. a background thread
    /// or an async task that offloads I/O).  It does **not** block on I/O
    /// beyond a single directory listing and per-file metadata reads.
    ///
    /// # Errors
    ///
    /// Returns an `std::io::Error` if the watch folder cannot be read.
    pub fn scan(&mut self) -> std::io::Result<ScanResult> {
        let now = Instant::now();
        let stable_duration = Duration::from_secs(self.config.stable_secs);

        let mut result = ScanResult {
            new_candidates: 0,
            queued: 0,
            pending_stability: 0,
            skipped_extension: 0,
        };

        // Collect files from the watch folder (optionally recursive).
        let discovered = collect_files(&self.config.folder, self.config.recursive)?;

        for path in discovered {
            // Skip already-ingested files.
            if self.ingested.contains(&path) {
                continue;
            }
            // Extension filter.
            if !self.config.is_extension_allowed(&path) {
                result.skipped_extension += 1;
                continue;
            }

            // Fetch file metadata.
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue, // file disappeared between listing and stat
            };

            if !meta.is_file() {
                continue;
            }

            let current_mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);

            match self.pending.get_mut(&path) {
                None => {
                    // First time we see this file.
                    result.new_candidates += 1;
                    self.pending.insert(
                        path.clone(),
                        WatchedFile {
                            path,
                            last_mtime: current_mtime,
                            first_seen: now,
                        },
                    );
                }
                Some(record) => {
                    let ready = match self.config.stability_strategy {
                        StabilityStrategy::Immediate => true,
                        StabilityStrategy::MtimeQuiesce => {
                            if current_mtime != record.last_mtime {
                                // File is still being written; reset the clock.
                                record.last_mtime = current_mtime;
                                false
                            } else {
                                // mtime stable — check elapsed time
                                now.duration_since(record.first_seen) >= stable_duration
                            }
                        }
                    };

                    if ready {
                        // Promote to ready queue.
                        let path_str = path.to_string_lossy().into_owned();
                        let mut item = IngestItem::new(
                            path_str,
                            self.config.destination.clone(),
                            self.config.priority,
                        );
                        for (k, v) in &self.config.default_metadata {
                            item = item.with_metadata(k.clone(), v.clone());
                        }
                        self.ready_queue.push(item);
                        self.total_queued += 1;
                        result.queued += 1;
                        // Move from pending to a "queued-but-not-yet-confirmed" state.
                        // The caller is responsible for calling `mark_ingested` on success.
                        self.pending.remove(&path);
                        self.ingested.insert(path);
                    } else {
                        record.last_mtime = current_mtime;
                        result.pending_stability += 1;
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Collect regular files under `root`.  When `recursive` is true, descends
/// into sub-directories; otherwise only the top-level entries are examined.
///
/// Hidden files (names starting with `.`) are always skipped.
fn collect_files(root: &Path, recursive: bool) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    let entries = std::fs::read_dir(root)?;
    for entry in entries.flatten() {
        let path = entry.path();

        // Skip hidden files/dirs
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with('.'))
            .unwrap_or(false)
        {
            continue;
        }

        if path.is_dir() && recursive {
            files.extend(collect_files(&path, true)?);
        } else if path.is_file() {
            files.push(path);
        }
    }

    Ok(files)
}

/// A thread-safe wrapper around [`WatchFolderState`] for use in concurrent
/// environments.
pub struct WatchFolder {
    state: Arc<Mutex<WatchFolderState>>,
}

impl WatchFolder {
    /// Create a new watch folder from configuration.
    #[must_use]
    pub fn new(config: WatchFolderConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(WatchFolderState::new(config))),
        }
    }

    /// Clone the inner state handle (cheap — only clones the `Arc`).
    #[must_use]
    pub fn handle(&self) -> Arc<Mutex<WatchFolderState>> {
        Arc::clone(&self.state)
    }

    /// Perform one synchronous scan cycle, returning a summary.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from the directory scan.
    pub fn scan(&self) -> std::io::Result<ScanResult> {
        self.state
            .lock()
            .map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::Other, "watch folder mutex poisoned")
            })?
            .scan()
    }

    /// Pop the next ingest-ready item, if any.
    pub fn pop_ready(&self) -> Option<IngestItem> {
        self.state.lock().ok().and_then(|mut g| g.pop_ready())
    }

    /// Mark a path as ingested.
    pub fn mark_ingested(&self, path: PathBuf) {
        if let Ok(mut g) = self.state.lock() {
            g.mark_ingested(path);
        }
    }

    /// Total files queued since the watcher was created.
    #[must_use]
    pub fn total_queued(&self) -> u64 {
        self.state.lock().map(|g| g.total_queued()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(path: &str, priority: u8) -> IngestItem {
        IngestItem::new(path, "/dest", priority)
    }

    #[test]
    fn test_ingest_status_is_success() {
        assert!(IngestStatus::Success.is_success());
        assert!(!IngestStatus::Failed.is_success());
        assert!(!IngestStatus::Skipped.is_success());
        assert!(!IngestStatus::Duplicate.is_success());
    }

    #[test]
    fn test_ingest_status_labels() {
        assert_eq!(IngestStatus::Success.label(), "success");
        assert_eq!(IngestStatus::Failed.label(), "failed");
        assert_eq!(IngestStatus::Skipped.label(), "skipped");
        assert_eq!(IngestStatus::Duplicate.label(), "duplicate");
    }

    #[test]
    fn test_ingest_result_success() {
        let r = IngestResult::success(item("a.mp4", 5), 200, "asset-001".to_string());
        assert!(r.status.is_success());
        assert_eq!(r.asset_id.as_deref(), Some("asset-001"));
        assert!(r.error.is_none());
    }

    #[test]
    fn test_ingest_result_failed() {
        let r = IngestResult::failed(item("b.mp4", 5), 100, "disk full");
        assert_eq!(r.status, IngestStatus::Failed);
        assert!(r.error.is_some());
    }

    #[test]
    fn test_ingest_result_skipped() {
        let r = IngestResult::skipped(item("c.xyz", 5), "unsupported format");
        assert_eq!(r.status, IngestStatus::Skipped);
    }

    #[test]
    fn test_ingest_result_duplicate() {
        let r = IngestResult::duplicate(item("d.mp4", 5), "existing-123");
        assert_eq!(r.status, IngestStatus::Duplicate);
        assert_eq!(r.asset_id.as_deref(), Some("existing-123"));
    }

    #[test]
    fn test_validator_validate_path_ok() {
        assert!(IngestValidator::validate_path("/media/foo/bar.mp4").is_ok());
    }

    #[test]
    fn test_validator_validate_path_empty() {
        assert!(IngestValidator::validate_path("").is_err());
    }

    #[test]
    fn test_validator_validate_path_traversal() {
        assert!(IngestValidator::validate_path("/media/../secret").is_err());
    }

    #[test]
    fn test_validator_check_duplicate_found() {
        let existing = vec!["abc123".to_string(), "def456".to_string()];
        assert!(IngestValidator::check_duplicate("abc123", &existing));
    }

    #[test]
    fn test_validator_check_duplicate_not_found() {
        let existing = vec!["abc123".to_string()];
        assert!(!IngestValidator::check_duplicate("xyz999", &existing));
    }

    #[test]
    fn test_batch_ingest_report_success_rate() {
        let results = vec![
            IngestResult::success(item("a.mp4", 5), 100, "a1".to_string()),
            IngestResult::success(item("b.mp4", 5), 100, "a2".to_string()),
            IngestResult::failed(item("c.mp4", 5), 50, "error"),
        ];
        let report = BatchIngestReport::from_results(results);
        assert_eq!(report.success_count, 2);
        assert_eq!(report.failed_count, 1);
        assert!((report.success_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_ingest_report_empty() {
        let report = BatchIngestReport::from_results(vec![]);
        assert_eq!(report.success_rate(), 1.0);
    }

    #[test]
    fn test_ingest_queue_priority_ordering() {
        let mut q = IngestQueue::new();
        q.push(item("low.mp4", 1));
        q.push(item("high.mp4", 10));
        q.push(item("mid.mp4", 5));

        let first = q.pop().expect("should succeed in test");
        assert_eq!(first.path, "high.mp4");
        let second = q.pop().expect("should succeed in test");
        assert_eq!(second.path, "mid.mp4");
        let third = q.pop().expect("should succeed in test");
        assert_eq!(third.path, "low.mp4");
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_ingest_queue_len_and_empty() {
        let mut q = IngestQueue::new();
        assert!(q.is_empty());
        q.push(item("a.mp4", 5));
        assert_eq!(q.len(), 1);
        let _ = q.pop();
        assert!(q.is_empty());
    }

    // -------------------------------------------------------------------------
    // Watch folder tests
    // -------------------------------------------------------------------------

    fn temp_watch_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("should create temp dir")
    }

    #[test]
    fn test_watch_folder_config_defaults() {
        let cfg = WatchFolderConfig::new("/watch", "/dest");
        assert_eq!(cfg.folder, PathBuf::from("/watch"));
        assert_eq!(cfg.destination, "/dest");
        assert!(!cfg.recursive);
        assert_eq!(cfg.stability_strategy, StabilityStrategy::MtimeQuiesce);
        assert!(!cfg.allowed_extensions.is_empty());
    }

    #[test]
    fn test_watch_folder_config_builder() {
        let cfg = WatchFolderConfig::new("/watch", "/dest")
            .with_stable_secs(30)
            .immediate()
            .recursive()
            .with_extensions(vec!["mp4".to_string()]);
        assert_eq!(cfg.stable_secs, 30);
        assert_eq!(cfg.stability_strategy, StabilityStrategy::Immediate);
        assert!(cfg.recursive);
        assert_eq!(cfg.allowed_extensions, vec!["mp4".to_string()]);
    }

    #[test]
    fn test_watch_folder_extension_filter_allowed() {
        let cfg = WatchFolderConfig::new("/w", "/d")
            .with_extensions(vec!["mp4".to_string(), "mov".to_string()]);
        assert!(cfg.is_extension_allowed(Path::new("video.mp4")));
        assert!(cfg.is_extension_allowed(Path::new("clip.MOV"))); // case-insensitive
        assert!(!cfg.is_extension_allowed(Path::new("doc.pdf")));
    }

    #[test]
    fn test_watch_folder_extension_filter_empty_allows_all() {
        let cfg = WatchFolderConfig::new("/w", "/d").with_extensions(vec![]);
        assert!(cfg.is_extension_allowed(Path::new("any_file.xyz")));
    }

    #[test]
    fn test_watch_folder_extension_no_extension() {
        let cfg = WatchFolderConfig::new("/w", "/d").with_extensions(vec!["mp4".to_string()]);
        assert!(!cfg.is_extension_allowed(Path::new("Makefile")));
    }

    #[test]
    fn test_watch_folder_scan_empty_dir() {
        let dir = temp_watch_dir();
        let cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        let wf = WatchFolder::new(cfg);
        let result = wf.scan().expect("scan should succeed");
        assert_eq!(result.new_candidates, 0);
        assert_eq!(result.queued, 0);
    }

    #[test]
    fn test_watch_folder_scan_new_file_immediate() {
        let dir = temp_watch_dir();
        std::fs::write(dir.path().join("test.mp4"), b"fakevideo").expect("write test file");

        let cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        let wf = WatchFolder::new(cfg);

        // First scan: file is new candidate
        let r1 = wf.scan().expect("scan 1");
        assert_eq!(r1.new_candidates, 1);
        // Immediate strategy: promoted right away in first scan
        // (first scan adds to pending; second scan with Immediate promotes it)
        let r2 = wf.scan().expect("scan 2");
        assert_eq!(r2.queued, 1);
        assert_eq!(wf.total_queued(), 1);

        let popped = wf.pop_ready();
        assert!(popped.is_some());
        let itm = popped.expect("item should be present");
        assert!(itm.path.ends_with("test.mp4"));
    }

    #[test]
    fn test_watch_folder_scan_dedup_after_mark_ingested() {
        let dir = temp_watch_dir();
        let file = dir.path().join("clip.mp4");
        std::fs::write(&file, b"data").expect("write test file");

        let cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        let wf = WatchFolder::new(cfg);

        // Scan twice to promote
        wf.scan().expect("scan 1");
        wf.scan().expect("scan 2");
        wf.mark_ingested(file.clone());

        // A third scan must not re-queue the already-ingested file
        let r3 = wf.scan().expect("scan 3");
        assert_eq!(r3.queued, 0);
        assert_eq!(r3.new_candidates, 0);
    }

    #[test]
    fn test_watch_folder_skips_hidden_files() {
        let dir = temp_watch_dir();
        std::fs::write(dir.path().join(".hidden.mp4"), b"data").expect("write hidden file");
        std::fs::write(dir.path().join("visible.mp4"), b"data").expect("write visible file");

        let cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        let wf = WatchFolder::new(cfg);
        wf.scan().expect("scan 1");
        let r2 = wf.scan().expect("scan 2");

        // Only the visible file should be queued
        assert_eq!(r2.queued, 1);
    }

    #[test]
    fn test_watch_folder_skips_extension_filtered_files() {
        let dir = temp_watch_dir();
        std::fs::write(dir.path().join("document.docx"), b"data").expect("write docx");
        std::fs::write(dir.path().join("video.mp4"), b"data").expect("write mp4");

        let cfg = WatchFolderConfig::new(dir.path(), "/dest")
            .with_extensions(vec!["mp4".to_string()])
            .immediate();
        let wf = WatchFolder::new(cfg);
        wf.scan().expect("scan 1");
        let r2 = wf.scan().expect("scan 2");

        assert_eq!(r2.queued, 1);
    }

    #[test]
    fn test_watch_folder_default_metadata_attached() {
        let dir = temp_watch_dir();
        std::fs::write(dir.path().join("asset.mp4"), b"data").expect("write file");

        let mut cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        cfg.default_metadata
            .insert("project".to_string(), "test-project".to_string());

        let wf = WatchFolder::new(cfg);
        wf.scan().expect("scan 1");
        wf.scan().expect("scan 2");

        let it = wf.pop_ready().expect("should have item");
        assert_eq!(
            it.metadata.get("project").map(String::as_str),
            Some("test-project")
        );
    }

    #[test]
    fn test_watch_folder_state_total_queued() {
        let dir = temp_watch_dir();
        for i in 0..3u8 {
            std::fs::write(dir.path().join(format!("f{i}.mp4")), b"data").expect("write test file");
        }

        let cfg = WatchFolderConfig::new(dir.path(), "/dest").immediate();
        let wf = WatchFolder::new(cfg);
        wf.scan().expect("scan 1");
        wf.scan().expect("scan 2");

        assert_eq!(wf.total_queued(), 3);
    }

    #[test]
    fn test_watch_folder_mtime_quiesce_not_ready_first_scan() {
        let dir = temp_watch_dir();
        std::fs::write(dir.path().join("slow.mp4"), b"data").expect("write file");

        // MtimeQuiesce with 60-second stability (will never be reached in test)
        let cfg = WatchFolderConfig::new(dir.path(), "/dest").with_stable_secs(60);
        let wf = WatchFolder::new(cfg);
        wf.scan().expect("scan 1");
        let r2 = wf.scan().expect("scan 2");

        // File is stable in mtime but 60s haven't passed
        assert_eq!(r2.queued, 0);
        assert_eq!(r2.pending_stability, 1);
    }

    #[test]
    fn test_collect_files_non_recursive() {
        let dir = temp_watch_dir();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).expect("create sub");
        std::fs::write(dir.path().join("top.mp4"), b"d").expect("write top");
        std::fs::write(sub.join("nested.mp4"), b"d").expect("write nested");

        let files = collect_files(dir.path(), false).expect("collect");
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("top.mp4"));
    }

    #[test]
    fn test_collect_files_recursive() {
        let dir = temp_watch_dir();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).expect("create sub");
        std::fs::write(dir.path().join("top.mp4"), b"d").expect("write top");
        std::fs::write(sub.join("nested.mp4"), b"d").expect("write nested");

        let files = collect_files(dir.path(), true).expect("collect");
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_watch_folder_poll_interval_min() {
        let cfg = WatchFolderConfig::new("/w", "/d").with_poll_interval(Duration::from_millis(1));
        // Minimum is 1 second
        assert!(cfg.poll_interval >= MIN_POLL_INTERVAL);
    }
}
