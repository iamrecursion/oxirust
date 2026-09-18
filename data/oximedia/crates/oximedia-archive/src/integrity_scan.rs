//! Scheduled integrity scanning for archived media
//!
//! Provides periodic full-archive and incremental integrity scans. Detects
//! bit-rot, silent data corruption, and missing files. Reports per-file
//! status and aggregate health metrics.
//!
//! [`ScanCache`] implements a real incremental integrity scan: files whose
//! `(size, mtime)` pair is unchanged since the previous scan are skipped
//! (their cached checksum is retained), while new or modified files are
//! re-hashed via [`crate::mmap_checksum::compute_checksums_mmap`].

use crate::mmap_checksum::{compute_checksums_mmap, MmapChecksumConfig};
use crate::{ArchiveError, ArchiveResult};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

/// Status of a single file after an integrity scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileIntegrity {
    /// File is intact.
    Ok,
    /// File has been modified since last scan.
    Modified,
    /// File is missing from storage.
    Missing,
    /// File is corrupted (checksum mismatch).
    Corrupted,
    /// File has not yet been scanned.
    Unknown,
}

impl fmt::Display for FileIntegrity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Ok => "OK",
            Self::Modified => "MODIFIED",
            Self::Missing => "MISSING",
            Self::Corrupted => "CORRUPTED",
            Self::Unknown => "UNKNOWN",
        };
        write!(f, "{s}")
    }
}

/// Policy for how often and what to scan.
#[derive(Debug, Clone)]
pub struct ScanPolicy {
    /// Interval between full scans in hours.
    pub full_scan_interval_hours: u32,
    /// Interval between incremental scans in hours.
    pub incremental_interval_hours: u32,
    /// Maximum number of files per incremental scan batch.
    pub incremental_batch_size: usize,
    /// Whether to halt on first corruption found.
    pub stop_on_first_error: bool,
    /// Number of parallel scan threads.
    pub parallelism: usize,
}

impl Default for ScanPolicy {
    fn default() -> Self {
        Self {
            full_scan_interval_hours: 168, // weekly
            incremental_interval_hours: 24,
            incremental_batch_size: 1000,
            stop_on_first_error: false,
            parallelism: 4,
        }
    }
}

/// Record of a single file's scan result.
#[derive(Debug, Clone)]
pub struct FileScanRecord {
    /// File path.
    pub path: String,
    /// Expected checksum.
    pub expected_checksum: String,
    /// Actual checksum computed during scan.
    pub actual_checksum: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Integrity status.
    pub status: FileIntegrity,
    /// Scan timestamp in epoch milliseconds.
    pub scanned_at_ms: u64,
    /// File last-modification time in epoch milliseconds, if known.
    ///
    /// Populated by incremental scanning (see [`ScanCache`]) and used together
    /// with [`Self::size_bytes`] as a cheap change-detection key: a file whose
    /// `(size_bytes, mtime_ms)` pair is unchanged since the previous scan can
    /// skip a full re-hash.
    pub mtime_ms: Option<u64>,
}

impl FileScanRecord {
    /// Create a new scan record.
    pub fn new(
        path: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        size_bytes: u64,
        scanned_at_ms: u64,
    ) -> Self {
        let expected_checksum = expected.into();
        let actual_checksum = actual.into();
        let status = if expected_checksum == actual_checksum {
            FileIntegrity::Ok
        } else {
            FileIntegrity::Corrupted
        };
        Self {
            path: path.into(),
            expected_checksum,
            actual_checksum,
            size_bytes,
            status,
            scanned_at_ms,
            mtime_ms: None,
        }
    }

    /// Create a record for a missing file.
    pub fn missing(path: impl Into<String>, scanned_at_ms: u64) -> Self {
        Self {
            path: path.into(),
            expected_checksum: String::new(),
            actual_checksum: String::new(),
            size_bytes: 0,
            status: FileIntegrity::Missing,
            scanned_at_ms,
            mtime_ms: None,
        }
    }

    /// Attach a last-modification time (epoch milliseconds) to this record.
    ///
    /// Builder-style; leaves [`Self::new`] / [`Self::missing`] source-compatible
    /// for callers that do not track mtime.
    #[must_use]
    pub fn with_mtime(mut self, mtime_ms: u64) -> Self {
        self.mtime_ms = Some(mtime_ms);
        self
    }

    /// Whether this record's `(size_bytes, mtime_ms)` change-key matches the
    /// given file metadata. Returns `false` if this record has no recorded
    /// mtime (cannot prove the file is unchanged).
    #[must_use]
    pub fn matches_metadata(&self, size_bytes: u64, mtime_ms: u64) -> bool {
        self.mtime_ms == Some(mtime_ms) && self.size_bytes == size_bytes
    }
}

/// Aggregate health metrics from a scan.
#[derive(Debug, Clone)]
pub struct ScanHealthMetrics {
    /// Total files scanned.
    pub total_scanned: usize,
    /// Number of OK files.
    pub ok_count: usize,
    /// Number of corrupted files.
    pub corrupted_count: usize,
    /// Number of missing files.
    pub missing_count: usize,
    /// Number of modified files.
    pub modified_count: usize,
    /// Total bytes scanned.
    pub total_bytes_scanned: u64,
    /// Duration of the scan in milliseconds.
    pub duration_ms: u64,
}

impl ScanHealthMetrics {
    /// Compute the health score as a fraction from 0.0 to 1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn health_score(&self) -> f64 {
        if self.total_scanned == 0 {
            return 1.0;
        }
        self.ok_count as f64 / self.total_scanned as f64
    }

    /// Whether the archive is considered healthy (score >= threshold).
    #[allow(clippy::cast_precision_loss)]
    pub fn is_healthy(&self, threshold: f64) -> bool {
        self.health_score() >= threshold
    }
}

/// An integrity scan session that accumulates file scan records.
#[derive(Debug)]
pub struct IntegrityScan {
    /// Scan policy.
    policy: ScanPolicy,
    /// All file scan records.
    records: Vec<FileScanRecord>,
    /// Start time in epoch ms.
    start_ms: u64,
    /// End time in epoch ms (0 if not finished).
    end_ms: u64,
}

impl IntegrityScan {
    /// Create a new integrity scan session.
    pub fn new(policy: ScanPolicy, start_ms: u64) -> Self {
        Self {
            policy,
            records: Vec::new(),
            start_ms,
            end_ms: 0,
        }
    }

    /// Create with default policy.
    pub fn with_defaults(start_ms: u64) -> Self {
        Self::new(ScanPolicy::default(), start_ms)
    }

    /// Get the policy.
    pub fn policy(&self) -> &ScanPolicy {
        &self.policy
    }

    /// Add a file scan record.
    pub fn add_record(&mut self, record: FileScanRecord) {
        self.records.push(record);
    }

    /// Number of records collected so far.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Mark the scan as finished.
    pub fn finish(&mut self, end_ms: u64) {
        self.end_ms = end_ms;
    }

    /// Whether the scan has finished.
    pub fn is_finished(&self) -> bool {
        self.end_ms > 0
    }

    /// Get all records.
    pub fn records(&self) -> &[FileScanRecord] {
        &self.records
    }

    /// Get only corrupted records.
    pub fn corrupted(&self) -> Vec<&FileScanRecord> {
        self.records
            .iter()
            .filter(|r| r.status == FileIntegrity::Corrupted)
            .collect()
    }

    /// Get only missing records.
    pub fn missing(&self) -> Vec<&FileScanRecord> {
        self.records
            .iter()
            .filter(|r| r.status == FileIntegrity::Missing)
            .collect()
    }

    /// Compute aggregate health metrics.
    pub fn metrics(&self) -> ScanHealthMetrics {
        let mut ok = 0usize;
        let mut corrupted = 0usize;
        let mut missing = 0usize;
        let mut modified = 0usize;
        let mut total_bytes = 0u64;
        for r in &self.records {
            match r.status {
                FileIntegrity::Ok => ok += 1,
                FileIntegrity::Corrupted => corrupted += 1,
                FileIntegrity::Missing => missing += 1,
                FileIntegrity::Modified => modified += 1,
                FileIntegrity::Unknown => {}
            }
            total_bytes += r.size_bytes;
        }
        ScanHealthMetrics {
            total_scanned: self.records.len(),
            ok_count: ok,
            corrupted_count: corrupted,
            missing_count: missing,
            modified_count: modified,
            total_bytes_scanned: total_bytes,
            duration_ms: self.end_ms.saturating_sub(self.start_ms),
        }
    }

    /// Group records by integrity status.
    pub fn group_by_status(&self) -> HashMap<String, Vec<&FileScanRecord>> {
        let mut map: HashMap<String, Vec<&FileScanRecord>> = HashMap::new();
        for r in &self.records {
            map.entry(r.status.to_string()).or_default().push(r);
        }
        map
    }
}

/// Convert a [`std::time::SystemTime`] to epoch milliseconds without panicking.
///
/// Returns an [`ArchiveError::Validation`] if the time is before the Unix epoch
/// (which would otherwise make `duration_since` fail).
fn system_time_to_epoch_ms(t: std::time::SystemTime) -> ArchiveResult<u64> {
    t.duration_since(UNIX_EPOCH)
        .map(|d| {
            let ms = d.as_millis();
            // Clamp rather than panic: u128 → u64 cannot overflow for any
            // realistic mtime, but stay total just in case.
            u64::try_from(ms).unwrap_or(u64::MAX)
        })
        .map_err(|e| ArchiveError::Validation(format!("file mtime is before Unix epoch: {e}")))
}

/// Incremental integrity-scan cache.
///
/// Holds the most recent [`FileScanRecord`] per path. On each [`Self::scan`]
/// call, a file whose `(size, mtime)` pair matches its cached record is
/// **skipped** (its previously computed checksum is retained); otherwise the
/// file is fully re-hashed and the cache entry is replaced.
///
/// The `skip_count` / `rehash_count` counters are reset at the start of every
/// [`Self::scan`] call so that each call's skip-vs-rehash decision is directly
/// observable without any timing or sleeping — they report the deltas for that
/// call only.
///
/// # Example
///
/// ```no_run
/// use std::path::PathBuf;
/// use oximedia_archive::integrity_scan::ScanCache;
///
/// let mut cache = ScanCache::new();
/// let paths = vec![PathBuf::from("/srv/archive/a.mxf")];
///
/// cache.scan(&paths)?;            // first pass: hashes everything
/// assert_eq!(cache.rehash_count(), 1);
///
/// cache.scan(&paths)?;            // second pass, file unchanged: skips it
/// assert_eq!(cache.skip_count(), 1);
/// assert_eq!(cache.rehash_count(), 0);
/// # Ok::<(), oximedia_archive::ArchiveError>(())
/// ```
#[derive(Debug, Default)]
pub struct ScanCache {
    /// Cached scan records, keyed by file path.
    records: HashMap<PathBuf, FileScanRecord>,
    /// Number of files re-hashed during the most recent `scan` call.
    rehash_count: usize,
    /// Number of files skipped (unchanged) during the most recent `scan` call.
    skip_count: usize,
    /// Checksum configuration used when (re-)hashing files.
    config: MmapChecksumConfig,
}

impl ScanCache {
    /// Create an empty scan cache using the default checksum configuration.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            rehash_count: 0,
            skip_count: 0,
            config: MmapChecksumConfig::default(),
        }
    }

    /// Create an empty scan cache with an explicit checksum configuration.
    pub fn with_config(config: MmapChecksumConfig) -> Self {
        Self {
            records: HashMap::new(),
            rehash_count: 0,
            skip_count: 0,
            config,
        }
    }

    /// Perform an incremental integrity scan over `paths`.
    ///
    /// For each path:
    /// * reads `std::fs::metadata` to obtain the current size and mtime
    ///   (mtime is converted to epoch milliseconds without `unwrap`);
    /// * if a cached record exists **and** its `(size, mtime)` both match the
    ///   current metadata, the file is **skipped** and its cached checksum is
    ///   retained (`skip_count` incremented);
    /// * otherwise the file is re-hashed via
    ///   [`compute_checksums_mmap`], and the cache entry is replaced with a
    ///   fresh record carrying the new size, mtime and checksum
    ///   (`rehash_count` incremented).
    ///
    /// The `skip_count` and `rehash_count` counters are **reset at the start**
    /// of this call, so on return they describe only this scan.
    ///
    /// # Errors
    ///
    /// Returns [`ArchiveError::Io`] if a file's metadata cannot be read or the
    /// file cannot be hashed, and [`ArchiveError::Validation`] if a file's
    /// mtime predates the Unix epoch.
    pub fn scan(&mut self, paths: &[PathBuf]) -> ArchiveResult<()> {
        self.rehash_count = 0;
        self.skip_count = 0;

        // Wall-clock scan timestamp (epoch ms); best-effort, never panics.
        let scanned_at_ms = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);

        for path in paths {
            let metadata = std::fs::metadata(path)?;
            let size_bytes = metadata.len();
            let mtime_ms = system_time_to_epoch_ms(metadata.modified()?)?;

            if let Some(existing) = self.records.get(path) {
                if existing.matches_metadata(size_bytes, mtime_ms) {
                    // Unchanged: keep the cached checksum, do not re-hash.
                    self.skip_count += 1;
                    continue;
                }
            }

            // New or changed file: recompute the checksum.
            let result = compute_checksums_mmap(path, &self.config)?;
            let checksum = primary_digest(&result);
            let record = FileScanRecord::new(
                path.to_string_lossy().into_owned(),
                checksum.clone(),
                checksum,
                size_bytes,
                scanned_at_ms,
            )
            .with_mtime(mtime_ms);
            self.records.insert(path.clone(), record);
            self.rehash_count += 1;
        }

        Ok(())
    }

    /// Number of files re-hashed during the most recent [`Self::scan`] call.
    pub fn rehash_count(&self) -> usize {
        self.rehash_count
    }

    /// Number of files skipped (unchanged) during the most recent
    /// [`Self::scan`] call.
    pub fn skip_count(&self) -> usize {
        self.skip_count
    }

    /// Total number of cached file records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the cache holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Fetch the cached scan record for a path, if present.
    pub fn record(&self, path: &std::path::Path) -> Option<&FileScanRecord> {
        self.records.get(path)
    }

    /// Fetch the cached primary checksum for a path, if present.
    ///
    /// This is the digest stored in [`FileScanRecord::actual_checksum`] — it is
    /// retained across skips, so it equals the value computed the last time the
    /// file was actually hashed.
    pub fn checksum(&self, path: &std::path::Path) -> Option<&str> {
        self.records.get(path).map(|r| r.actual_checksum.as_str())
    }

    /// Remove a path from the cache (e.g. after a file is deleted).
    ///
    /// Returns the removed record, if any.
    pub fn forget(&mut self, path: &std::path::Path) -> Option<FileScanRecord> {
        self.records.remove(path)
    }
}

/// Pick the strongest available digest from a checksum result, preferring
/// BLAKE3, then SHA-256, then CRC32, then MD5. Falls back to an empty string if
/// no algorithm was enabled (degenerate configuration).
fn primary_digest(result: &crate::mmap_checksum::MmapChecksumResult) -> String {
    result
        .blake3
        .as_ref()
        .or(result.sha256.as_ref())
        .or(result.crc32.as_ref())
        .or(result.md5.as_ref())
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_integrity_display() {
        assert_eq!(FileIntegrity::Ok.to_string(), "OK");
        assert_eq!(FileIntegrity::Corrupted.to_string(), "CORRUPTED");
        assert_eq!(FileIntegrity::Missing.to_string(), "MISSING");
        assert_eq!(FileIntegrity::Modified.to_string(), "MODIFIED");
        assert_eq!(FileIntegrity::Unknown.to_string(), "UNKNOWN");
    }

    #[test]
    fn test_default_scan_policy() {
        let p = ScanPolicy::default();
        assert_eq!(p.full_scan_interval_hours, 168);
        assert_eq!(p.incremental_interval_hours, 24);
        assert!(!p.stop_on_first_error);
    }

    #[test]
    fn test_scan_record_ok() {
        let r = FileScanRecord::new("/a.mxf", "abc", "abc", 1024, 1000);
        assert_eq!(r.status, FileIntegrity::Ok);
        assert_eq!(r.size_bytes, 1024);
    }

    #[test]
    fn test_scan_record_corrupted() {
        let r = FileScanRecord::new("/a.mxf", "abc", "xyz", 1024, 1000);
        assert_eq!(r.status, FileIntegrity::Corrupted);
    }

    #[test]
    fn test_scan_record_missing() {
        let r = FileScanRecord::missing("/gone.mxf", 2000);
        assert_eq!(r.status, FileIntegrity::Missing);
        assert_eq!(r.size_bytes, 0);
    }

    #[test]
    fn test_new_scan_not_finished() {
        let scan = IntegrityScan::with_defaults(1000);
        assert!(!scan.is_finished());
        assert_eq!(scan.record_count(), 0);
    }

    #[test]
    fn test_scan_add_record_and_finish() {
        let mut scan = IntegrityScan::with_defaults(1000);
        scan.add_record(FileScanRecord::new("/a.mxf", "aa", "aa", 500, 1001));
        scan.finish(2000);
        assert!(scan.is_finished());
        assert_eq!(scan.record_count(), 1);
    }

    #[test]
    fn test_scan_corrupted_filter() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::new("/ok.mxf", "aa", "aa", 100, 1));
        scan.add_record(FileScanRecord::new("/bad.mxf", "aa", "bb", 200, 2));
        assert_eq!(scan.corrupted().len(), 1);
        assert_eq!(scan.corrupted()[0].path, "/bad.mxf");
    }

    #[test]
    fn test_scan_missing_filter() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::missing("/gone.mxf", 1));
        scan.add_record(FileScanRecord::new("/ok.mxf", "aa", "aa", 100, 2));
        assert_eq!(scan.missing().len(), 1);
    }

    #[test]
    fn test_metrics_all_ok() {
        let mut scan = IntegrityScan::with_defaults(100);
        scan.add_record(FileScanRecord::new("/a.mxf", "aa", "aa", 100, 101));
        scan.add_record(FileScanRecord::new("/b.mxf", "bb", "bb", 200, 102));
        scan.finish(200);
        let m = scan.metrics();
        assert_eq!(m.total_scanned, 2);
        assert_eq!(m.ok_count, 2);
        assert_eq!(m.corrupted_count, 0);
        assert!((m.health_score() - 1.0).abs() < f64::EPSILON);
        assert_eq!(m.duration_ms, 100);
        assert_eq!(m.total_bytes_scanned, 300);
    }

    #[test]
    fn test_metrics_with_corruption() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::new("/a.mxf", "aa", "aa", 100, 1));
        scan.add_record(FileScanRecord::new("/b.mxf", "bb", "xx", 200, 2));
        let m = scan.metrics();
        assert_eq!(m.ok_count, 1);
        assert_eq!(m.corrupted_count, 1);
        assert!((m.health_score() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_empty_scan_healthy() {
        let scan = IntegrityScan::with_defaults(0);
        let m = scan.metrics();
        assert!((m.health_score() - 1.0).abs() < f64::EPSILON);
        assert!(m.is_healthy(0.99));
    }

    #[test]
    fn test_is_healthy_threshold() {
        let mut scan = IntegrityScan::with_defaults(0);
        for i in 0..9 {
            scan.add_record(FileScanRecord::new(
                format!("/ok_{i}.mxf"),
                "aa",
                "aa",
                100,
                1,
            ));
        }
        scan.add_record(FileScanRecord::new("/bad.mxf", "aa", "xx", 100, 1));
        let m = scan.metrics();
        assert!(m.is_healthy(0.8));
        assert!(!m.is_healthy(0.95));
    }

    #[test]
    fn test_group_by_status() {
        let mut scan = IntegrityScan::with_defaults(0);
        scan.add_record(FileScanRecord::new("/ok.mxf", "aa", "aa", 100, 1));
        scan.add_record(FileScanRecord::new("/bad.mxf", "aa", "xx", 100, 2));
        scan.add_record(FileScanRecord::missing("/gone.mxf", 3));
        let groups = scan.group_by_status();
        assert_eq!(groups.get("OK").map(|v| v.len()), Some(1));
        assert_eq!(groups.get("CORRUPTED").map(|v| v.len()), Some(1));
        assert_eq!(groups.get("MISSING").map(|v| v.len()), Some(1));
    }

    // -----------------------------------------------------------------------
    // ScanCache — incremental (mtime+size skip) integrity scan
    // -----------------------------------------------------------------------

    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Process-unique counter so concurrent tests never collide on filenames.
    static UNIQ: AtomicU64 = AtomicU64::new(0);

    /// Create a unique temp directory under the system temp dir.
    fn unique_dir(tag: &str) -> PathBuf {
        let n = UNIQ.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("oximedia_scancache_{tag}_{pid}_{n}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Write `content` to `path`, replacing any existing file.
    fn write_file(path: &Path, content: &[u8]) {
        let mut f = std::fs::File::create(path).expect("create temp file");
        f.write_all(content).expect("write temp file");
        f.flush().expect("flush temp file");
    }

    /// Pin a file's mtime to a deterministic epoch-seconds value using the
    /// std-only `File::set_modified` API (stable since Rust 1.75). Pinning the
    /// timestamp makes skip-vs-rehash decisions fully deterministic — no
    /// sleeping or wall-clock dependence.
    fn set_mtime_secs(path: &Path, secs: u64) {
        let when = UNIX_EPOCH + std::time::Duration::from_secs(secs);
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for set mtime");
        f.set_modified(when).expect("set modified time");
    }

    /// Read a file's mtime back as epoch ms (test helper).
    fn mtime_ms(path: &Path) -> u64 {
        let md = std::fs::metadata(path).expect("metadata");
        let t = md.modified().expect("modified");
        let d = t.duration_since(UNIX_EPOCH).expect("after epoch");
        u64::try_from(d.as_millis()).expect("fits u64")
    }

    #[test]
    fn test_record_with_mtime_builder() {
        let r = FileScanRecord::new("/a.mxf", "aa", "aa", 10, 1);
        assert_eq!(r.mtime_ms, None);
        let r2 = r.with_mtime(123_456);
        assert_eq!(r2.mtime_ms, Some(123_456));
        // matches_metadata requires BOTH mtime and size to agree.
        assert!(r2.matches_metadata(10, 123_456));
        assert!(!r2.matches_metadata(11, 123_456));
        assert!(!r2.matches_metadata(10, 999));
    }

    #[test]
    fn test_record_without_mtime_never_matches() {
        let r = FileScanRecord::new("/a.mxf", "aa", "aa", 10, 1);
        // No recorded mtime → cannot prove unchanged.
        assert!(!r.matches_metadata(10, 0));
    }

    #[test]
    fn test_scancache_first_scan_hashes_then_skip_unchanged() {
        let dir = unique_dir("first_then_skip");
        let path = dir.join("file.bin");
        write_file(&path, b"incremental scan cache content");
        // Pin mtime so it is stable across both scans.
        set_mtime_secs(&path, 1_700_000_000);

        let paths = vec![path.clone()];
        let mut cache = ScanCache::new();

        // First scan: brand-new path -> must be hashed.
        cache.scan(&paths).expect("first scan");
        assert_eq!(cache.rehash_count(), 1, "first scan must hash the file");
        assert_eq!(cache.skip_count(), 0);
        let first_checksum = cache.checksum(&path).expect("checksum present").to_string();
        assert!(!first_checksum.is_empty());

        // Second scan, file untouched -> skip, no re-hash, checksum retained.
        cache.scan(&paths).expect("second scan");
        assert_eq!(cache.skip_count(), 1, "unchanged file must be skipped");
        assert_eq!(cache.rehash_count(), 0, "no re-hash on unchanged file");
        let skipped_checksum = cache.checksum(&path).expect("checksum still present");
        assert_eq!(
            skipped_checksum, first_checksum,
            "checksum after skip must equal first-hash checksum"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_mtime_change_triggers_rehash() {
        let dir = unique_dir("mtime_change");
        let path = dir.join("file.bin");
        write_file(&path, b"same size payload AAAA");
        set_mtime_secs(&path, 1_700_000_000);

        let paths = vec![path.clone()];
        let mut cache = ScanCache::new();
        cache.scan(&paths).expect("first scan");
        assert_eq!(cache.rehash_count(), 1);

        // Rewrite with DIFFERENT content of the SAME length, then set a NEW
        // mtime. The size key is unchanged; only mtime differs -> must rehash.
        write_file(&path, b"same size payload BBBB");
        set_mtime_secs(&path, 1_700_000_500);
        let before = cache.checksum(&path).expect("checksum").to_string();

        cache.scan(&paths).expect("second scan");
        assert_eq!(
            cache.rehash_count(),
            1,
            "changed mtime (+changed content) must trigger rehash"
        );
        assert_eq!(cache.skip_count(), 0);
        let after = cache.checksum(&path).expect("checksum");
        assert_ne!(after, before, "rehash must update the stored checksum");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_size_change_triggers_rehash() {
        let dir = unique_dir("size_change");
        let path = dir.join("file.bin");
        write_file(&path, b"original");
        // Pin the SAME mtime for both scans so ONLY the size differs.
        set_mtime_secs(&path, 1_700_111_111);

        let paths = vec![path.clone()];
        let mut cache = ScanCache::new();
        cache.scan(&paths).expect("first scan");
        assert_eq!(cache.rehash_count(), 1);

        // Append bytes -> size grows. Re-stamp the SAME mtime so the only
        // changed component of the change-key is the size.
        write_file(&path, b"original_plus_more_bytes");
        set_mtime_secs(&path, 1_700_111_111);

        cache.scan(&paths).expect("second scan");
        assert_eq!(
            cache.rehash_count(),
            1,
            "size change alone must trigger rehash"
        );
        assert_eq!(cache.skip_count(), 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_new_path_always_hashed() {
        let dir = unique_dir("new_path");
        let path_a = dir.join("a.bin");
        let path_b = dir.join("b.bin");
        write_file(&path_a, b"file a contents");
        write_file(&path_b, b"file b contents");
        set_mtime_secs(&path_a, 1_700_222_000);
        set_mtime_secs(&path_b, 1_700_222_000);

        let mut cache = ScanCache::new();

        // Scan only A first.
        cache.scan(std::slice::from_ref(&path_a)).expect("scan a");
        assert_eq!(cache.rehash_count(), 1);
        assert_eq!(cache.skip_count(), 0);

        // Now scan BOTH: A is cached+unchanged (skip), B is brand-new (hash).
        cache
            .scan(&[path_a.clone(), path_b.clone()])
            .expect("scan a+b");
        assert_eq!(cache.skip_count(), 1, "A unchanged -> skip");
        assert_eq!(cache.rehash_count(), 1, "B new -> hash");
        assert!(cache.checksum(&path_b).is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_skipped_checksum_equals_first_hash_large_file() {
        // Exercise the mmap path (>64 KiB) to prove skip retention holds there
        // too, and that the retained digest matches an independent computation.
        let dir = unique_dir("large_skip");
        let path = dir.join("large.bin");
        let content: Vec<u8> = (0u8..=255).cycle().take(256 * 1024).collect();
        write_file(&path, &content);
        set_mtime_secs(&path, 1_700_333_000);

        let paths = vec![path.clone()];
        let mut cache = ScanCache::new();
        cache.scan(&paths).expect("first scan");
        assert_eq!(cache.rehash_count(), 1);

        let independent = blake3::hash(&content).to_hex().to_string();
        assert_eq!(
            cache.checksum(&path),
            Some(independent.as_str()),
            "first-hash digest must equal independent BLAKE3"
        );

        cache.scan(&paths).expect("second scan");
        assert_eq!(cache.skip_count(), 1);
        assert_eq!(cache.rehash_count(), 0);
        assert_eq!(
            cache.checksum(&path),
            Some(independent.as_str()),
            "skipped digest must still equal the original"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_record_carries_size_and_mtime() {
        let dir = unique_dir("record_fields");
        let path = dir.join("file.bin");
        let content = b"record metadata payload";
        write_file(&path, content);
        set_mtime_secs(&path, 1_700_444_000);

        let mut cache = ScanCache::new();
        cache.scan(std::slice::from_ref(&path)).expect("scan");

        let rec = cache.record(&path).expect("record present");
        assert_eq!(rec.size_bytes, content.len() as u64);
        assert_eq!(rec.mtime_ms, Some(mtime_ms(&path)));
        assert_eq!(rec.status, FileIntegrity::Ok);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_missing_file_errors() {
        let dir = unique_dir("missing");
        let missing = dir.join("does_not_exist.bin");
        let mut cache = ScanCache::new();
        let result = cache.scan(&[missing]);
        assert!(result.is_err(), "scanning a missing file must error");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_forget_removes_entry() {
        let dir = unique_dir("forget");
        let path = dir.join("file.bin");
        write_file(&path, b"to be forgotten");
        set_mtime_secs(&path, 1_700_555_000);

        let mut cache = ScanCache::new();
        cache.scan(std::slice::from_ref(&path)).expect("scan");
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let removed = cache.forget(&path);
        assert!(removed.is_some());
        assert!(cache.is_empty());
        assert!(cache.checksum(&path).is_none());

        // After forgetting, the same unchanged file must be hashed again.
        cache
            .scan(std::slice::from_ref(&path))
            .expect("re-scan after forget");
        assert_eq!(cache.rehash_count(), 1, "forgotten path treated as new");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scancache_counters_reset_each_call() {
        let dir = unique_dir("reset");
        let path = dir.join("file.bin");
        write_file(&path, b"counter reset check");
        set_mtime_secs(&path, 1_700_666_000);

        let paths = vec![path.clone()];
        let mut cache = ScanCache::new();

        cache.scan(&paths).expect("scan 1");
        assert_eq!(cache.rehash_count(), 1);
        cache.scan(&paths).expect("scan 2");
        // Counters describe ONLY the latest call, not a running total.
        assert_eq!(cache.rehash_count(), 0);
        assert_eq!(cache.skip_count(), 1);
        cache.scan(&paths).expect("scan 3");
        assert_eq!(cache.rehash_count(), 0);
        assert_eq!(cache.skip_count(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }
}
