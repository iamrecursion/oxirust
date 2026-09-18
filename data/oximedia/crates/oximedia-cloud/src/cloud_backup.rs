#![allow(dead_code)]
//! Cloud backup strategies for media assets.
//!
//! Provides incremental, differential, and full backup planning, versioned
//! backup management, retention policies, and backup verification for
//! cloud-stored media files.

use std::collections::HashMap;

/// Backup strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupStrategy {
    /// Full backup — copies all data every time.
    Full,
    /// Incremental — copies only data changed since the last backup (of any type).
    Incremental,
    /// Differential — copies data changed since the last full backup.
    Differential,
    /// Mirror — exact copy with no versioning.
    Mirror,
}

impl std::fmt::Display for BackupStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Incremental => write!(f, "incremental"),
            Self::Differential => write!(f, "differential"),
            Self::Mirror => write!(f, "mirror"),
        }
    }
}

/// Backup status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackupStatus {
    /// Backup is pending / not started.
    Pending,
    /// Backup is currently running.
    Running,
    /// Backup completed successfully.
    Completed,
    /// Backup failed.
    Failed,
    /// Backup was cancelled.
    Cancelled,
}

impl std::fmt::Display for BackupStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// A single backed-up file entry.
#[derive(Debug, Clone)]
pub struct BackupFileEntry {
    /// Relative path of the file.
    pub path: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// SHA-256 checksum as hex string.
    pub checksum: String,
    /// Last modified timestamp (Unix epoch seconds).
    pub modified_epoch: u64,
}

impl BackupFileEntry {
    /// Creates a new backup file entry.
    #[must_use]
    pub fn new(path: &str, size_bytes: u64, checksum: &str, modified_epoch: u64) -> Self {
        Self {
            path: path.to_string(),
            size_bytes,
            checksum: checksum.to_string(),
            modified_epoch,
        }
    }
}

/// A backup snapshot representing one backup run.
#[derive(Debug, Clone)]
pub struct BackupSnapshot {
    /// Unique identifier for this snapshot.
    pub snapshot_id: String,
    /// Strategy used for this backup.
    pub strategy: BackupStrategy,
    /// Timestamp when the backup started (Unix epoch seconds).
    pub started_epoch: u64,
    /// Timestamp when the backup completed (0 if not done).
    pub completed_epoch: u64,
    /// Current status.
    pub status: BackupStatus,
    /// Files included in this snapshot.
    pub files: Vec<BackupFileEntry>,
    /// Total bytes in this snapshot.
    pub total_bytes: u64,
}

impl BackupSnapshot {
    /// Creates a new pending backup snapshot.
    #[must_use]
    pub fn new(snapshot_id: &str, strategy: BackupStrategy, started_epoch: u64) -> Self {
        Self {
            snapshot_id: snapshot_id.to_string(),
            strategy,
            started_epoch,
            completed_epoch: 0,
            status: BackupStatus::Pending,
            files: Vec::new(),
            total_bytes: 0,
        }
    }

    /// Adds a file to this snapshot.
    pub fn add_file(&mut self, entry: BackupFileEntry) {
        self.total_bytes += entry.size_bytes;
        self.files.push(entry);
    }

    /// Marks the snapshot as completed.
    pub fn complete(&mut self, completed_epoch: u64) {
        self.status = BackupStatus::Completed;
        self.completed_epoch = completed_epoch;
    }

    /// Marks the snapshot as failed.
    pub fn fail(&mut self) {
        self.status = BackupStatus::Failed;
    }

    /// Returns the number of files in this snapshot.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Returns the duration of the backup in seconds (0 if not completed).
    #[must_use]
    pub fn duration_secs(&self) -> u64 {
        self.completed_epoch.saturating_sub(self.started_epoch)
    }
}

/// Retention policy for backup snapshots.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum number of full backups to keep.
    pub max_full_backups: usize,
    /// Maximum number of incremental backups to keep.
    pub max_incremental_backups: usize,
    /// Maximum age of any backup in days (0 = no limit).
    pub max_age_days: u64,
    /// Minimum number of backups to always keep regardless of age.
    pub min_keep_count: usize,
}

impl RetentionPolicy {
    /// Creates a new retention policy.
    #[must_use]
    pub fn new(max_full: usize, max_incremental: usize) -> Self {
        Self {
            max_full_backups: max_full,
            max_incremental_backups: max_incremental,
            max_age_days: 0,
            min_keep_count: 1,
        }
    }

    /// Sets the maximum age.
    #[must_use]
    pub fn with_max_age_days(mut self, days: u64) -> Self {
        self.max_age_days = days;
        self
    }

    /// Sets the minimum keep count.
    #[must_use]
    pub fn with_min_keep(mut self, count: usize) -> Self {
        self.min_keep_count = count;
        self
    }

    /// Returns the snapshot IDs that should be removed based on this policy.
    #[must_use]
    pub fn snapshots_to_remove<'a>(
        &self,
        snapshots: &'a [BackupSnapshot],
        current_epoch: u64,
    ) -> Vec<&'a str> {
        let mut to_remove = Vec::new();
        let max_age_secs = self.max_age_days * 86_400;

        // Separate by strategy
        let mut full_ids: Vec<(usize, &BackupSnapshot)> = Vec::new();
        let mut incr_ids: Vec<(usize, &BackupSnapshot)> = Vec::new();

        for (i, snap) in snapshots.iter().enumerate() {
            match snap.strategy {
                BackupStrategy::Full => full_ids.push((i, snap)),
                BackupStrategy::Incremental | BackupStrategy::Differential => {
                    incr_ids.push((i, snap));
                }
                BackupStrategy::Mirror => {}
            }
        }

        // Remove excess full backups (keep newest)
        if full_ids.len() > self.max_full_backups {
            let excess = full_ids.len() - self.max_full_backups;
            // Sort by started_epoch ascending (oldest first)
            full_ids.sort_by_key(|(_, s)| s.started_epoch);
            for (_, snap) in full_ids.iter().take(excess) {
                to_remove.push(snap.snapshot_id.as_str());
            }
        }

        // Remove excess incremental backups
        if incr_ids.len() > self.max_incremental_backups {
            let excess = incr_ids.len() - self.max_incremental_backups;
            incr_ids.sort_by_key(|(_, s)| s.started_epoch);
            for (_, snap) in incr_ids.iter().take(excess) {
                to_remove.push(snap.snapshot_id.as_str());
            }
        }

        // Remove by age
        if max_age_secs > 0 && snapshots.len() > self.min_keep_count {
            for snap in snapshots {
                if current_epoch > snap.started_epoch
                    && (current_epoch - snap.started_epoch) > max_age_secs
                    && !to_remove.contains(&snap.snapshot_id.as_str())
                {
                    to_remove.push(snap.snapshot_id.as_str());
                }
            }
        }

        to_remove
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::new(5, 30)
    }
}

/// Incremental backup planner.
///
/// Compares current file state against the last backup snapshot to determine
/// which files need to be backed up.
#[derive(Debug)]
pub struct IncrementalPlanner {
    /// Previous snapshot's file checksums indexed by path.
    previous_checksums: HashMap<String, String>,
}

impl IncrementalPlanner {
    /// Creates a planner with no previous state (equivalent to full backup).
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous_checksums: HashMap::new(),
        }
    }

    /// Creates a planner from a previous snapshot.
    #[must_use]
    pub fn from_snapshot(snapshot: &BackupSnapshot) -> Self {
        let mut checksums = HashMap::new();
        for file in &snapshot.files {
            checksums.insert(file.path.clone(), file.checksum.clone());
        }
        Self {
            previous_checksums: checksums,
        }
    }

    /// Returns the files that need to be backed up (changed or new).
    #[must_use]
    pub fn plan<'a>(&self, current_files: &'a [BackupFileEntry]) -> Vec<&'a BackupFileEntry> {
        current_files
            .iter()
            .filter(|f| {
                match self.previous_checksums.get(&f.path) {
                    Some(prev_checksum) => prev_checksum != &f.checksum, // Changed
                    None => true,                                        // New file
                }
            })
            .collect()
    }

    /// Returns the count of files in the previous snapshot.
    #[must_use]
    pub fn previous_file_count(&self) -> usize {
        self.previous_checksums.len()
    }
}

impl Default for IncrementalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Backup verification result.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether verification passed.
    pub passed: bool,
    /// Number of files verified.
    pub files_verified: usize,
    /// Number of files with checksum mismatches.
    pub mismatches: usize,
    /// Number of files missing from the backup.
    pub missing: usize,
    /// Paths with issues.
    pub issue_paths: Vec<String>,
}

impl VerificationResult {
    /// Creates a new passing verification result.
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            files_verified: 0,
            mismatches: 0,
            missing: 0,
            issue_paths: Vec::new(),
        }
    }

    /// Records a mismatch.
    pub fn add_mismatch(&mut self, path: &str) {
        self.mismatches += 1;
        self.passed = false;
        self.issue_paths.push(path.to_string());
    }

    /// Records a missing file.
    pub fn add_missing(&mut self, path: &str) {
        self.missing += 1;
        self.passed = false;
        self.issue_paths.push(path.to_string());
    }
}

impl Default for VerificationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifies a backup snapshot against expected file entries.
#[must_use]
pub fn verify_backup(
    snapshot: &BackupSnapshot,
    expected: &[BackupFileEntry],
) -> VerificationResult {
    let mut result = VerificationResult::new();

    let snap_map: HashMap<&str, &BackupFileEntry> = snapshot
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();

    for expected_file in expected {
        result.files_verified += 1;
        match snap_map.get(expected_file.path.as_str()) {
            Some(snap_file) => {
                if snap_file.checksum != expected_file.checksum {
                    result.add_mismatch(&expected_file.path);
                }
            }
            None => {
                result.add_missing(&expected_file.path);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_strategy_display() {
        assert_eq!(BackupStrategy::Full.to_string(), "full");
        assert_eq!(BackupStrategy::Incremental.to_string(), "incremental");
        assert_eq!(BackupStrategy::Differential.to_string(), "differential");
        assert_eq!(BackupStrategy::Mirror.to_string(), "mirror");
    }

    #[test]
    fn test_backup_status_display() {
        assert_eq!(BackupStatus::Pending.to_string(), "pending");
        assert_eq!(BackupStatus::Running.to_string(), "running");
        assert_eq!(BackupStatus::Completed.to_string(), "completed");
        assert_eq!(BackupStatus::Failed.to_string(), "failed");
        assert_eq!(BackupStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_snapshot_new() {
        let snap = BackupSnapshot::new("snap-001", BackupStrategy::Full, 1000);
        assert_eq!(snap.snapshot_id, "snap-001");
        assert_eq!(snap.strategy, BackupStrategy::Full);
        assert_eq!(snap.status, BackupStatus::Pending);
        assert_eq!(snap.file_count(), 0);
        assert_eq!(snap.total_bytes, 0);
    }

    #[test]
    fn test_snapshot_add_file() {
        let mut snap = BackupSnapshot::new("snap-001", BackupStrategy::Full, 1000);
        snap.add_file(BackupFileEntry::new("a.mp4", 1000, "abc123", 900));
        snap.add_file(BackupFileEntry::new("b.mp4", 2000, "def456", 950));
        assert_eq!(snap.file_count(), 2);
        assert_eq!(snap.total_bytes, 3000);
    }

    #[test]
    fn test_snapshot_complete() {
        let mut snap = BackupSnapshot::new("snap-001", BackupStrategy::Full, 1000);
        snap.complete(1100);
        assert_eq!(snap.status, BackupStatus::Completed);
        assert_eq!(snap.duration_secs(), 100);
    }

    #[test]
    fn test_snapshot_fail() {
        let mut snap = BackupSnapshot::new("snap-001", BackupStrategy::Full, 1000);
        snap.fail();
        assert_eq!(snap.status, BackupStatus::Failed);
    }

    #[test]
    fn test_snapshot_duration_incomplete() {
        let snap = BackupSnapshot::new("snap-001", BackupStrategy::Full, 1000);
        assert_eq!(snap.duration_secs(), 0);
    }

    #[test]
    fn test_incremental_planner_no_previous() {
        let planner = IncrementalPlanner::new();
        let files = vec![
            BackupFileEntry::new("a.mp4", 100, "aaa", 1),
            BackupFileEntry::new("b.mp4", 200, "bbb", 2),
        ];
        let plan = planner.plan(&files);
        assert_eq!(plan.len(), 2); // All files are new
    }

    #[test]
    fn test_incremental_planner_with_previous() {
        let mut prev_snap = BackupSnapshot::new("prev", BackupStrategy::Full, 1000);
        prev_snap.add_file(BackupFileEntry::new("a.mp4", 100, "aaa", 1));
        prev_snap.add_file(BackupFileEntry::new("b.mp4", 200, "bbb", 2));
        prev_snap.complete(1010);

        let planner = IncrementalPlanner::from_snapshot(&prev_snap);
        assert_eq!(planner.previous_file_count(), 2);

        let files = vec![
            BackupFileEntry::new("a.mp4", 100, "aaa", 1), // Unchanged
            BackupFileEntry::new("b.mp4", 250, "bbb2", 3), // Changed
            BackupFileEntry::new("c.mp4", 300, "ccc", 4), // New
        ];
        let plan = planner.plan(&files);
        assert_eq!(plan.len(), 2); // b.mp4 changed, c.mp4 new
        let paths: Vec<&str> = plan.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"b.mp4"));
        assert!(paths.contains(&"c.mp4"));
    }

    #[test]
    fn test_retention_policy_excess_full() {
        let policy = RetentionPolicy::new(2, 10);
        let snapshots = vec![
            BackupSnapshot::new("full-1", BackupStrategy::Full, 100),
            BackupSnapshot::new("full-2", BackupStrategy::Full, 200),
            BackupSnapshot::new("full-3", BackupStrategy::Full, 300),
        ];
        let to_remove = policy.snapshots_to_remove(&snapshots, 400);
        assert_eq!(to_remove.len(), 1);
        assert!(to_remove.contains(&"full-1")); // Oldest removed
    }

    #[test]
    fn test_retention_policy_excess_incremental() {
        let policy = RetentionPolicy::new(10, 2);
        let snapshots = vec![
            BackupSnapshot::new("incr-1", BackupStrategy::Incremental, 100),
            BackupSnapshot::new("incr-2", BackupStrategy::Incremental, 200),
            BackupSnapshot::new("incr-3", BackupStrategy::Incremental, 300),
        ];
        let to_remove = policy.snapshots_to_remove(&snapshots, 400);
        assert_eq!(to_remove.len(), 1);
        assert!(to_remove.contains(&"incr-1"));
    }

    #[test]
    fn test_retention_policy_age_based() {
        let policy = RetentionPolicy::new(10, 10)
            .with_max_age_days(30)
            .with_min_keep(0);
        let day_secs: u64 = 86_400;
        let snapshots = vec![
            BackupSnapshot::new("old", BackupStrategy::Full, 0), // Very old
            BackupSnapshot::new("recent", BackupStrategy::Full, 100 * day_secs), // Recent
        ];
        let current = 110 * day_secs;
        let to_remove = policy.snapshots_to_remove(&snapshots, current);
        assert!(to_remove.contains(&"old"));
        assert!(!to_remove.contains(&"recent"));
    }

    #[test]
    fn test_verification_pass() {
        let mut snap = BackupSnapshot::new("snap", BackupStrategy::Full, 1000);
        snap.add_file(BackupFileEntry::new("a.mp4", 100, "aaa", 1));
        snap.add_file(BackupFileEntry::new("b.mp4", 200, "bbb", 2));

        let expected = vec![
            BackupFileEntry::new("a.mp4", 100, "aaa", 1),
            BackupFileEntry::new("b.mp4", 200, "bbb", 2),
        ];
        let result = verify_backup(&snap, &expected);
        assert!(result.passed);
        assert_eq!(result.files_verified, 2);
        assert_eq!(result.mismatches, 0);
        assert_eq!(result.missing, 0);
    }

    #[test]
    fn test_verification_mismatch() {
        let mut snap = BackupSnapshot::new("snap", BackupStrategy::Full, 1000);
        snap.add_file(BackupFileEntry::new("a.mp4", 100, "aaa", 1));

        let expected = vec![BackupFileEntry::new("a.mp4", 100, "WRONG", 1)];
        let result = verify_backup(&snap, &expected);
        assert!(!result.passed);
        assert_eq!(result.mismatches, 1);
    }

    #[test]
    fn test_verification_missing() {
        let snap = BackupSnapshot::new("snap", BackupStrategy::Full, 1000);
        let expected = vec![BackupFileEntry::new("a.mp4", 100, "aaa", 1)];
        let result = verify_backup(&snap, &expected);
        assert!(!result.passed);
        assert_eq!(result.missing, 1);
    }

    #[test]
    fn test_default_retention_policy() {
        let policy = RetentionPolicy::default();
        assert_eq!(policy.max_full_backups, 5);
        assert_eq!(policy.max_incremental_backups, 30);
    }
}
