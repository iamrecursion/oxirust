//! Asset version management
//!
//! Provides tracking and management of asset versions including:
//! - Version history with metadata diffs
//! - Branching from any version
//! - Rollback capability
//! - Retention policies for automated cleanup

#![allow(dead_code)]

use std::collections::HashMap;

/// A single version of a media asset
#[derive(Debug, Clone)]
pub struct AssetVersion {
    /// Unique version identifier
    pub version_id: String,
    /// ID of the parent asset
    pub asset_id: String,
    /// Sequential version number
    pub version_num: u32,
    /// Creation timestamp in milliseconds since epoch
    pub created_at_ms: u64,
    /// User who created this version
    pub created_by: String,
    /// Notes describing what changed in this version
    pub change_notes: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA-256 or similar checksum of the file
    pub checksum: String,
    /// Optional metadata associated with this version
    pub metadata: HashMap<String, String>,
}

impl AssetVersion {
    /// Create a new asset version
    #[must_use]
    pub fn new(
        version_id: String,
        asset_id: String,
        version_num: u32,
        created_at_ms: u64,
        created_by: String,
        change_notes: String,
        size_bytes: u64,
        checksum: String,
    ) -> Self {
        Self {
            version_id,
            asset_id,
            version_num,
            created_at_ms,
            created_by,
            change_notes,
            size_bytes,
            checksum,
            metadata: HashMap::new(),
        }
    }
}

/// A tree of versions for a single asset
#[derive(Debug, Clone)]
pub struct VersionTree {
    /// ID of the asset this tree belongs to
    pub asset_id: String,
    /// All versions in chronological order
    pub versions: Vec<AssetVersion>,
    /// Version number of the current (active) version
    pub current: u32,
}

impl VersionTree {
    /// Create a new version tree for an asset
    #[must_use]
    pub fn new(asset_id: String) -> Self {
        Self {
            asset_id,
            versions: Vec::new(),
            current: 0,
        }
    }

    /// Add a version to the tree
    pub fn add_version(&mut self, version: AssetVersion) {
        if version.version_num > self.current {
            self.current = version.version_num;
        }
        self.versions.push(version);
    }

    /// Create a branch from the specified version number
    ///
    /// Returns a new `AssetVersion` with a new ID derived from `new_id`,
    /// branched from the given `version_num`. The new version gets the next
    /// available version number.
    ///
    /// Returns `None` if the source version is not found.
    #[must_use]
    pub fn branch_from(&self, version_num: u32, new_id: &str) -> Option<AssetVersion> {
        let source = self
            .versions
            .iter()
            .find(|v| v.version_num == version_num)?;

        let next_num = self
            .versions
            .iter()
            .map(|v| v.version_num)
            .max()
            .unwrap_or(0)
            + 1;

        let mut branched = AssetVersion {
            version_id: new_id.to_string(),
            asset_id: source.asset_id.clone(),
            version_num: next_num,
            created_at_ms: source.created_at_ms,
            created_by: source.created_by.clone(),
            change_notes: format!("Branched from version {version_num}"),
            size_bytes: source.size_bytes,
            checksum: source.checksum.clone(),
            metadata: source.metadata.clone(),
        };
        branched
            .metadata
            .insert("branch_source".to_string(), version_num.to_string());

        Some(branched)
    }

    /// Get the current active version
    #[must_use]
    pub fn current_version(&self) -> Option<&AssetVersion> {
        self.versions.iter().find(|v| v.version_num == self.current)
    }
}

/// Difference between two asset versions
#[derive(Debug, Clone)]
pub struct VersionDiff {
    /// Metadata keys and values added or changed in version b vs a
    pub added_metadata: HashMap<String, String>,
    /// Metadata keys removed in version b vs a
    pub removed_metadata: Vec<String>,
    /// Change in file size (positive = grew, negative = shrank)
    pub size_delta: i64,
}

impl VersionDiff {
    /// Compute the diff between two versions (a = older, b = newer)
    #[must_use]
    pub fn compute(a: &AssetVersion, b: &AssetVersion) -> Self {
        let mut added_metadata = HashMap::new();
        let mut removed_metadata = Vec::new();

        // Find added/changed keys
        for (k, v) in &b.metadata {
            match a.metadata.get(k) {
                None => {
                    added_metadata.insert(k.clone(), v.clone());
                }
                Some(old_v) if old_v != v => {
                    added_metadata.insert(k.clone(), v.clone());
                }
                _ => {}
            }
        }

        // Find removed keys
        for k in a.metadata.keys() {
            if !b.metadata.contains_key(k) {
                removed_metadata.push(k.clone());
            }
        }

        let size_delta = b.size_bytes as i64 - a.size_bytes as i64;

        Self {
            added_metadata,
            removed_metadata,
            size_delta,
        }
    }

    /// Returns true if there are no differences
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added_metadata.is_empty() && self.removed_metadata.is_empty() && self.size_delta == 0
    }
}

/// Storage for asset versions across multiple assets
#[derive(Debug, Default)]
pub struct VersionStore {
    /// Map of asset_id -> VersionTree
    trees: HashMap<String, VersionTree>,
}

impl VersionStore {
    /// Create a new empty version store
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new version for an asset
    pub fn add_version(&mut self, version: AssetVersion) {
        let tree = self
            .trees
            .entry(version.asset_id.clone())
            .or_insert_with(|| VersionTree::new(version.asset_id.clone()));
        tree.add_version(version);
    }

    /// Get a specific version by asset ID and version number
    #[must_use]
    pub fn get_version(&self, asset_id: &str, num: u32) -> Option<&AssetVersion> {
        self.trees
            .get(asset_id)?
            .versions
            .iter()
            .find(|v| v.version_num == num)
    }

    /// List all versions for an asset, sorted by version number ascending
    #[must_use]
    pub fn list_versions(&self, asset_id: &str) -> Vec<&AssetVersion> {
        match self.trees.get(asset_id) {
            None => Vec::new(),
            Some(tree) => {
                let mut versions: Vec<&AssetVersion> = tree.versions.iter().collect();
                versions.sort_by_key(|v| v.version_num);
                versions
            }
        }
    }

    /// Rollback an asset to a specific version number.
    ///
    /// Returns `true` if the rollback succeeded (the version exists).
    /// This sets the `current` pointer in the version tree to the target version.
    pub fn rollback(&mut self, asset_id: &str, version_num: u32) -> bool {
        match self.trees.get_mut(asset_id) {
            None => false,
            Some(tree) => {
                if tree.versions.iter().any(|v| v.version_num == version_num) {
                    tree.current = version_num;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Get the current version tree for an asset
    #[must_use]
    pub fn get_tree(&self, asset_id: &str) -> Option<&VersionTree> {
        self.trees.get(asset_id)
    }
}

/// Policy for how many versions to retain
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Always keep the most recent N versions
    pub keep_last_n: usize,
    /// Always keep versions whose version_num is a multiple of 10 (major versions)
    pub keep_major_versions: bool,
    /// Do not delete versions younger than this many days
    pub min_age_days: u32,
}

impl RetentionPolicy {
    /// Create a default retention policy
    #[must_use]
    pub fn new(keep_last_n: usize, keep_major_versions: bool, min_age_days: u32) -> Self {
        Self {
            keep_last_n,
            keep_major_versions,
            min_age_days,
        }
    }

    /// Apply the policy and return the list of version IDs that should be deleted.
    ///
    /// `current_time_ms` is the current time in milliseconds since epoch, used
    /// to evaluate the `min_age_days` constraint.
    #[must_use]
    pub fn apply(&self, versions: &[AssetVersion]) -> Vec<String> {
        self.apply_with_time(versions, current_time_ms())
    }

    /// Apply the policy with an explicit current time (useful for tests)
    #[must_use]
    pub fn apply_with_time(&self, versions: &[AssetVersion], current_ms: u64) -> Vec<String> {
        if versions.is_empty() {
            return Vec::new();
        }

        let min_age_ms = self.min_age_days as u64 * 24 * 60 * 60 * 1000;
        let cutoff_ms = current_ms.saturating_sub(min_age_ms);

        // Sort by version_num descending to find the "last N"
        let mut sorted: Vec<&AssetVersion> = versions.iter().collect();
        sorted.sort_by(|a, b| b.version_num.cmp(&a.version_num));

        let keep_set: std::collections::HashSet<&str> = sorted
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                // Keep last N
                if idx < self.keep_last_n {
                    return Some(v.version_id.as_str());
                }
                // Keep major versions (multiples of 10, non-zero)
                if self.keep_major_versions && v.version_num > 0 && v.version_num % 10 == 0 {
                    return Some(v.version_id.as_str());
                }
                // Keep if too young to delete
                if v.created_at_ms > cutoff_ms {
                    return Some(v.version_id.as_str());
                }
                None
            })
            .collect();

        versions
            .iter()
            .filter(|v| !keep_set.contains(v.version_id.as_str()))
            .map(|v| v.version_id.clone())
            .collect()
    }
}

/// Returns current time in milliseconds since UNIX epoch
fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version(asset_id: &str, num: u32, vid: &str, size: u64) -> AssetVersion {
        AssetVersion::new(
            vid.to_string(),
            asset_id.to_string(),
            num,
            1_000_000 + num as u64 * 1000,
            "user1".to_string(),
            format!("Version {num}"),
            size,
            format!("checksum{num}"),
        )
    }

    #[test]
    fn test_asset_version_creation() {
        let v = make_version("asset-1", 1, "v1", 1024);
        assert_eq!(v.version_num, 1);
        assert_eq!(v.asset_id, "asset-1");
        assert_eq!(v.size_bytes, 1024);
    }

    #[test]
    fn test_version_store_add_and_get() {
        let mut store = VersionStore::new();
        store.add_version(make_version("a1", 1, "v1", 100));
        store.add_version(make_version("a1", 2, "v2", 200));

        assert!(store.get_version("a1", 1).is_some());
        assert!(store.get_version("a1", 2).is_some());
        assert!(store.get_version("a1", 99).is_none());
    }

    #[test]
    fn test_version_store_list_versions_sorted() {
        let mut store = VersionStore::new();
        store.add_version(make_version("a1", 3, "v3", 300));
        store.add_version(make_version("a1", 1, "v1", 100));
        store.add_version(make_version("a1", 2, "v2", 200));

        let versions = store.list_versions("a1");
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version_num, 1);
        assert_eq!(versions[1].version_num, 2);
        assert_eq!(versions[2].version_num, 3);
    }

    #[test]
    fn test_version_store_list_unknown_asset() {
        let store = VersionStore::new();
        assert!(store.list_versions("unknown").is_empty());
    }

    #[test]
    fn test_rollback_success() {
        let mut store = VersionStore::new();
        store.add_version(make_version("a1", 1, "v1", 100));
        store.add_version(make_version("a1", 2, "v2", 200));
        store.add_version(make_version("a1", 3, "v3", 300));

        assert!(store.rollback("a1", 1));
        assert_eq!(
            store
                .get_tree("a1")
                .expect("should succeed in test")
                .current,
            1
        );
    }

    #[test]
    fn test_rollback_nonexistent_version() {
        let mut store = VersionStore::new();
        store.add_version(make_version("a1", 1, "v1", 100));
        assert!(!store.rollback("a1", 99));
    }

    #[test]
    fn test_rollback_unknown_asset() {
        let mut store = VersionStore::new();
        assert!(!store.rollback("unknown", 1));
    }

    #[test]
    fn test_version_tree_branch_from() {
        let mut tree = VersionTree::new("a1".to_string());
        tree.add_version(make_version("a1", 1, "v1", 500));
        tree.add_version(make_version("a1", 2, "v2", 600));

        let branched = tree
            .branch_from(1, "v-branch")
            .expect("should succeed in test");
        assert_eq!(branched.version_id, "v-branch");
        assert_eq!(branched.version_num, 3); // next after max=2
        assert!(branched.change_notes.contains("1"));
        assert_eq!(branched.size_bytes, 500);
    }

    #[test]
    fn test_version_tree_branch_from_missing() {
        let tree = VersionTree::new("a1".to_string());
        assert!(tree.branch_from(99, "v-new").is_none());
    }

    #[test]
    fn test_version_diff_compute() {
        let mut a = make_version("a1", 1, "v1", 1000);
        a.metadata
            .insert("title".to_string(), "Old Title".to_string());
        a.metadata.insert("genre".to_string(), "Drama".to_string());

        let mut b = make_version("a1", 2, "v2", 1500);
        b.metadata
            .insert("title".to_string(), "New Title".to_string());
        b.metadata.insert("rating".to_string(), "PG".to_string());
        // "genre" is removed

        let diff = VersionDiff::compute(&a, &b);
        assert_eq!(diff.size_delta, 500);
        assert!(diff.added_metadata.contains_key("title"));
        assert!(diff.added_metadata.contains_key("rating"));
        assert!(diff.removed_metadata.contains(&"genre".to_string()));
    }

    #[test]
    fn test_version_diff_empty() {
        let a = make_version("a1", 1, "v1", 1000);
        let b = make_version("a1", 2, "v2", 1000);
        let diff = VersionDiff::compute(&a, &b);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_retention_policy_keep_last_n() {
        // Create 5 old versions (older than 30 days)
        let old_ms = 0u64; // epoch start - very old
        let mut versions = Vec::new();
        for i in 1u32..=5 {
            let mut v = make_version("a1", i, &format!("v{i}"), 100);
            v.created_at_ms = old_ms;
            versions.push(v);
        }

        let policy = RetentionPolicy::new(2, false, 30);
        // Use a large "current time" so all are older than 30 days
        let current_ms = 31u64 * 24 * 60 * 60 * 1000;
        let to_delete = policy.apply_with_time(&versions, current_ms);

        // Should keep last 2 (v4, v5), delete v1, v2, v3
        assert_eq!(to_delete.len(), 3);
        assert!(to_delete.contains(&"v1".to_string()));
        assert!(to_delete.contains(&"v2".to_string()));
        assert!(to_delete.contains(&"v3".to_string()));
    }

    #[test]
    fn test_retention_policy_keep_major_versions() {
        let old_ms = 0u64;
        let mut versions = Vec::new();
        for i in [1u32, 5, 10, 15, 20] {
            let mut v = make_version("a1", i, &format!("v{i}"), 100);
            v.created_at_ms = old_ms;
            versions.push(v);
        }

        let policy = RetentionPolicy::new(1, true, 30);
        let current_ms = 31u64 * 24 * 60 * 60 * 1000;
        let to_delete = policy.apply_with_time(&versions, current_ms);

        // Keep: v20 (last 1), v10 (major), v20 (major) -> keep v10, v20
        // Delete: v1, v5, v15
        assert!(!to_delete.contains(&"v10".to_string()));
        assert!(!to_delete.contains(&"v20".to_string()));
        assert!(to_delete.contains(&"v1".to_string()));
        assert!(to_delete.contains(&"v5".to_string()));
        assert!(to_delete.contains(&"v15".to_string()));
    }
}
