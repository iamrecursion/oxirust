//! Asset version control
//!
//! Provides lightweight, in-memory version history tracking for MAM assets:
//! - Recording discrete version actions (create, export, transcode, etc.)
//! - Querying the latest version of any asset
//! - Listing all historical versions for an asset

#![allow(dead_code)]

/// The action that caused a new version entry to be recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionAction {
    /// Asset was first created / ingested
    Created,
    /// Asset was exported to an external system or file
    Exported,
    /// Asset was transcoded to a new format or codec
    Transcoded,
    /// Asset metadata or content was edited
    Edited,
    /// Asset was approved by a reviewer
    Approved,
    /// Asset was rejected by a reviewer
    Rejected,
    /// Asset was moved to the archive tier
    Archived,
}

impl VersionAction {
    /// Returns `true` if this action is typically the result of creating
    /// a new derivative or materially changed version of the asset.
    #[must_use]
    pub fn creates_new_version(&self) -> bool {
        matches!(
            self,
            VersionAction::Created | VersionAction::Transcoded | VersionAction::Edited
        )
    }
}

/// A single version record in the version history of an asset.
#[derive(Debug, Clone)]
pub struct AssetVersion {
    /// Unique identifier for this version record
    pub version_id: u64,
    /// ID of the owning asset
    pub asset_id: u64,
    /// Monotonically increasing version number (per asset)
    pub version_num: u32,
    /// The action that produced this version
    pub action: VersionAction,
    /// Unix epoch timestamp (seconds) when this version was recorded
    pub timestamp_epoch: u64,
    /// Size of the asset file at this version in bytes
    pub size_bytes: u64,
    /// Storage path for the asset file at this version
    pub path: String,
}

impl AssetVersion {
    /// Returns `true` if this version was produced by an `Approved` action.
    #[must_use]
    pub fn is_approved(&self) -> bool {
        self.action == VersionAction::Approved
    }
}

/// In-memory store for the complete version history across all assets.
#[derive(Debug, Default)]
pub struct VersionHistory {
    /// All recorded version entries
    pub versions: Vec<AssetVersion>,
    /// Counter used to assign unique version IDs
    pub next_id: u64,
}

impl VersionHistory {
    /// Create a new, empty version history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new version entry and return its assigned `version_id`.
    ///
    /// The `version_num` is automatically computed as one more than the
    /// current highest version number for `asset_id`.
    pub fn add_version(
        &mut self,
        asset_id: u64,
        action: VersionAction,
        epoch: u64,
        size: u64,
        path: impl Into<String>,
    ) -> u64 {
        let version_num = self
            .versions
            .iter()
            .filter(|v| v.asset_id == asset_id)
            .map(|v| v.version_num)
            .max()
            .map_or(1, |n| n + 1);

        let version_id = self.next_id;
        self.next_id += 1;
        self.versions.push(AssetVersion {
            version_id,
            asset_id,
            version_num,
            action,
            timestamp_epoch: epoch,
            size_bytes: size,
            path: path.into(),
        });
        version_id
    }

    /// Return the most recently added version for the given asset, or `None`.
    #[must_use]
    pub fn latest_version(&self, asset_id: u64) -> Option<&AssetVersion> {
        self.versions
            .iter()
            .filter(|v| v.asset_id == asset_id)
            .max_by_key(|v| v.version_num)
    }

    /// Return the total number of version records for the given asset.
    #[must_use]
    pub fn version_count(&self, asset_id: u64) -> usize {
        self.versions
            .iter()
            .filter(|v| v.asset_id == asset_id)
            .count()
    }

    /// Return references to all version records for the given asset,
    /// ordered by `version_num` ascending.
    #[must_use]
    pub fn all_versions_for(&self, asset_id: u64) -> Vec<&AssetVersion> {
        let mut result: Vec<&AssetVersion> = self
            .versions
            .iter()
            .filter(|v| v.asset_id == asset_id)
            .collect();
        result.sort_by_key(|v| v.version_num);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_action_creates_new_version_created() {
        assert!(VersionAction::Created.creates_new_version());
    }

    #[test]
    fn test_version_action_creates_new_version_transcoded() {
        assert!(VersionAction::Transcoded.creates_new_version());
    }

    #[test]
    fn test_version_action_creates_new_version_edited() {
        assert!(VersionAction::Edited.creates_new_version());
    }

    #[test]
    fn test_version_action_does_not_create_version_approved() {
        assert!(!VersionAction::Approved.creates_new_version());
    }

    #[test]
    fn test_version_action_does_not_create_version_rejected() {
        assert!(!VersionAction::Rejected.creates_new_version());
    }

    #[test]
    fn test_version_action_does_not_create_version_archived() {
        assert!(!VersionAction::Archived.creates_new_version());
    }

    #[test]
    fn test_version_action_does_not_create_version_exported() {
        assert!(!VersionAction::Exported.creates_new_version());
    }

    #[test]
    fn test_add_version_returns_id() {
        let mut history = VersionHistory::new();
        let id = history.add_version(1, VersionAction::Created, 1000, 512, "/mnt/file.mxf");
        assert_eq!(id, 0);
    }

    #[test]
    fn test_add_version_increments_version_num() {
        let mut history = VersionHistory::new();
        history.add_version(1, VersionAction::Created, 1000, 512, "/a.mxf");
        history.add_version(1, VersionAction::Edited, 2000, 512, "/b.mxf");
        let versions = history.all_versions_for(1);
        assert_eq!(versions[0].version_num, 1);
        assert_eq!(versions[1].version_num, 2);
    }

    #[test]
    fn test_latest_version_correct() {
        let mut history = VersionHistory::new();
        history.add_version(1, VersionAction::Created, 1000, 512, "/a.mxf");
        history.add_version(1, VersionAction::Transcoded, 2000, 256, "/b.mp4");
        let latest = history.latest_version(1).expect("should succeed in test");
        assert_eq!(latest.version_num, 2);
        assert_eq!(latest.path, "/b.mp4");
    }

    #[test]
    fn test_latest_version_missing_asset() {
        let history = VersionHistory::new();
        assert!(history.latest_version(999).is_none());
    }

    #[test]
    fn test_version_count() {
        let mut history = VersionHistory::new();
        history.add_version(5, VersionAction::Created, 100, 100, "/x");
        history.add_version(5, VersionAction::Edited, 200, 100, "/y");
        history.add_version(5, VersionAction::Approved, 300, 100, "/z");
        assert_eq!(history.version_count(5), 3);
    }

    #[test]
    fn test_version_count_different_assets() {
        let mut history = VersionHistory::new();
        history.add_version(1, VersionAction::Created, 100, 100, "/a");
        history.add_version(2, VersionAction::Created, 200, 100, "/b");
        assert_eq!(history.version_count(1), 1);
        assert_eq!(history.version_count(2), 1);
    }

    #[test]
    fn test_all_versions_ordered() {
        let mut history = VersionHistory::new();
        history.add_version(10, VersionAction::Created, 100, 100, "/1");
        history.add_version(10, VersionAction::Edited, 200, 100, "/2");
        history.add_version(10, VersionAction::Approved, 300, 100, "/3");
        let versions = history.all_versions_for(10);
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].version_num, 1);
        assert_eq!(versions[2].version_num, 3);
    }

    #[test]
    fn test_is_approved() {
        let mut history = VersionHistory::new();
        let id = history.add_version(1, VersionAction::Approved, 1000, 512, "/ok.mxf");
        let ver = history
            .versions
            .iter()
            .find(|v| v.version_id == id)
            .expect("should succeed in test");
        assert!(ver.is_approved());
    }

    #[test]
    fn test_is_not_approved() {
        let mut history = VersionHistory::new();
        let id = history.add_version(1, VersionAction::Rejected, 1000, 512, "/fail.mxf");
        let ver = history
            .versions
            .iter()
            .find(|v| v.version_id == id)
            .expect("should succeed in test");
        assert!(!ver.is_approved());
    }

    #[test]
    fn test_all_versions_empty_asset() {
        let history = VersionHistory::new();
        assert!(history.all_versions_for(42).is_empty());
    }
}
