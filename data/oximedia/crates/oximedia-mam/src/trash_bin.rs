//! Soft-delete trash bin for media assets.
//!
//! Provides a `TrashBin` that holds soft-deleted assets with configurable
//! auto-purge TTL per project.  Assets can be restored before the TTL expires
//! or explicitly purged.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Trash entry
// ---------------------------------------------------------------------------

/// A single item in the trash bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrashEntry {
    /// Unique trash entry id.
    pub id: Uuid,
    /// Original asset id.
    pub asset_id: Uuid,
    /// Project the asset belonged to (if any).
    pub project_id: Option<Uuid>,
    /// Who deleted the asset.
    pub deleted_by: Option<String>,
    /// When the asset was moved to trash.
    pub deleted_at: DateTime<Utc>,
    /// When the entry will be auto-purged.
    pub expires_at: DateTime<Utc>,
    /// Original asset name / path for display.
    pub original_name: String,
    /// Size of the asset in bytes.
    pub size_bytes: u64,
    /// Optional metadata snapshot at time of deletion.
    pub metadata: Option<serde_json::Value>,
}

impl TrashEntry {
    /// Returns `true` if this entry has expired according to `now`.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.expires_at
    }

    /// Remaining time until expiry.  Returns zero duration if already expired.
    #[must_use]
    pub fn time_remaining(&self, now: DateTime<Utc>) -> Duration {
        let diff = self.expires_at.signed_duration_since(now);
        if diff < Duration::zero() {
            Duration::zero()
        } else {
            diff
        }
    }
}

// ---------------------------------------------------------------------------
// TTL configuration
// ---------------------------------------------------------------------------

/// Per-project TTL configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTtlConfig {
    pub project_id: Uuid,
    /// TTL in days before auto-purge.
    pub ttl_days: u32,
}

// ---------------------------------------------------------------------------
// Trash bin
// ---------------------------------------------------------------------------

/// A soft-delete trash bin with per-project auto-purge TTL.
#[derive(Debug)]
pub struct TrashBin {
    /// Default TTL in days for projects without an explicit setting.
    default_ttl_days: u32,
    /// Per-project TTL overrides.
    project_ttls: HashMap<Uuid, u32>,
    /// All entries currently in the trash.
    entries: Vec<TrashEntry>,
}

impl TrashBin {
    /// Create a new trash bin with the given default TTL (in days).
    #[must_use]
    pub fn new(default_ttl_days: u32) -> Self {
        Self {
            default_ttl_days,
            project_ttls: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Set a per-project TTL override.
    pub fn set_project_ttl(&mut self, project_id: Uuid, ttl_days: u32) {
        self.project_ttls.insert(project_id, ttl_days);
    }

    /// Get the effective TTL for a project.
    #[must_use]
    pub fn effective_ttl_days(&self, project_id: Option<Uuid>) -> u32 {
        project_id
            .and_then(|pid| self.project_ttls.get(&pid).copied())
            .unwrap_or(self.default_ttl_days)
    }

    /// Move an asset to the trash bin.
    ///
    /// Returns the `TrashEntry` that was created.
    pub fn move_to_trash(
        &mut self,
        asset_id: Uuid,
        project_id: Option<Uuid>,
        original_name: impl Into<String>,
        size_bytes: u64,
        deleted_by: Option<String>,
        metadata: Option<serde_json::Value>,
    ) -> TrashEntry {
        let ttl = self.effective_ttl_days(project_id);
        let now = Utc::now();
        let entry = TrashEntry {
            id: Uuid::new_v4(),
            asset_id,
            project_id,
            deleted_by,
            deleted_at: now,
            expires_at: now + Duration::days(i64::from(ttl)),
            original_name: original_name.into(),
            size_bytes,
            metadata,
        };
        self.entries.push(entry.clone());
        entry
    }

    /// Restore an asset from the trash by its trash entry id.
    ///
    /// Returns `Some(TrashEntry)` if found, `None` otherwise.
    pub fn restore(&mut self, entry_id: Uuid) -> Option<TrashEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.id == entry_id) {
            Some(self.entries.remove(pos))
        } else {
            None
        }
    }

    /// Restore an asset from the trash by its original asset id.
    ///
    /// If multiple entries exist for the same asset, the most recently deleted
    /// one is restored.
    pub fn restore_by_asset_id(&mut self, asset_id: Uuid) -> Option<TrashEntry> {
        // Find the most recent entry for this asset.
        let pos = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.asset_id == asset_id)
            .max_by_key(|(_, e)| e.deleted_at)
            .map(|(i, _)| i);

        pos.map(|i| self.entries.remove(i))
    }

    /// Purge all expired entries.  Returns the list of purged entries.
    pub fn purge_expired(&mut self) -> Vec<TrashEntry> {
        self.purge_expired_at(Utc::now())
    }

    /// Purge entries expired as of the given timestamp (useful for testing).
    pub fn purge_expired_at(&mut self, now: DateTime<Utc>) -> Vec<TrashEntry> {
        let mut purged = Vec::new();
        let mut kept = Vec::new();
        for entry in self.entries.drain(..) {
            if entry.is_expired(now) {
                purged.push(entry);
            } else {
                kept.push(entry);
            }
        }
        self.entries = kept;
        purged
    }

    /// Forcefully purge a specific entry by its id.
    pub fn force_purge(&mut self, entry_id: Uuid) -> Option<TrashEntry> {
        if let Some(pos) = self.entries.iter().position(|e| e.id == entry_id) {
            Some(self.entries.remove(pos))
        } else {
            None
        }
    }

    /// List all entries currently in the trash.
    #[must_use]
    pub fn list(&self) -> &[TrashEntry] {
        &self.entries
    }

    /// List entries for a specific project.
    #[must_use]
    pub fn list_for_project(&self, project_id: Uuid) -> Vec<&TrashEntry> {
        self.entries
            .iter()
            .filter(|e| e.project_id == Some(project_id))
            .collect()
    }

    /// Number of entries in the trash.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the trash is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total size of all entries in the trash (bytes).
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Get an entry by its id.
    #[must_use]
    pub fn get(&self, entry_id: Uuid) -> Option<&TrashEntry> {
        self.entries.iter().find(|e| e.id == entry_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bin() -> TrashBin {
        TrashBin::new(30) // 30-day default TTL
    }

    #[test]
    fn test_trash_bin_new() {
        let bin = make_bin();
        assert!(bin.is_empty());
        assert_eq!(bin.len(), 0);
    }

    #[test]
    fn test_move_to_trash() {
        let mut bin = make_bin();
        let asset_id = Uuid::new_v4();
        let entry = bin.move_to_trash(asset_id, None, "clip.mp4", 1024, None, None);

        assert_eq!(entry.asset_id, asset_id);
        assert_eq!(entry.original_name, "clip.mp4");
        assert_eq!(entry.size_bytes, 1024);
        assert_eq!(bin.len(), 1);
    }

    #[test]
    fn test_restore_by_entry_id() {
        let mut bin = make_bin();
        let entry = bin.move_to_trash(Uuid::new_v4(), None, "a.mp4", 100, None, None);
        let eid = entry.id;

        let restored = bin.restore(eid);
        assert!(restored.is_some());
        assert!(bin.is_empty());
    }

    #[test]
    fn test_restore_nonexistent() {
        let mut bin = make_bin();
        assert!(bin.restore(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_restore_by_asset_id() {
        let mut bin = make_bin();
        let asset_id = Uuid::new_v4();
        bin.move_to_trash(asset_id, None, "v1.mp4", 100, None, None);
        bin.move_to_trash(asset_id, None, "v2.mp4", 200, None, None);

        let restored = bin.restore_by_asset_id(asset_id);
        assert!(restored.is_some());
        // Should restore the most recent (v2)
        assert_eq!(
            restored.as_ref().map(|e| e.original_name.as_str()),
            Some("v2.mp4")
        );
        assert_eq!(bin.len(), 1);
    }

    #[test]
    fn test_purge_expired() {
        let mut bin = make_bin();
        let asset_id = Uuid::new_v4();
        bin.move_to_trash(asset_id, None, "old.mp4", 100, None, None);

        // Not expired yet
        let purged = bin.purge_expired();
        assert!(purged.is_empty());
        assert_eq!(bin.len(), 1);

        // Simulate time passing
        let future = Utc::now() + Duration::days(31);
        let purged = bin.purge_expired_at(future);
        assert_eq!(purged.len(), 1);
        assert!(bin.is_empty());
    }

    #[test]
    fn test_project_ttl_override() {
        let mut bin = make_bin();
        let project_id = Uuid::new_v4();
        bin.set_project_ttl(project_id, 7); // 7 days

        assert_eq!(bin.effective_ttl_days(Some(project_id)), 7);
        assert_eq!(bin.effective_ttl_days(None), 30);
        assert_eq!(bin.effective_ttl_days(Some(Uuid::new_v4())), 30);
    }

    #[test]
    fn test_project_ttl_affects_expiry() {
        let mut bin = make_bin();
        let project_id = Uuid::new_v4();
        bin.set_project_ttl(project_id, 7);

        let entry = bin.move_to_trash(
            Uuid::new_v4(),
            Some(project_id),
            "short.mp4",
            50,
            None,
            None,
        );

        // Should expire in 7 days, not 30
        let in_8_days = Utc::now() + Duration::days(8);
        assert!(entry.is_expired(in_8_days));

        let in_5_days = Utc::now() + Duration::days(5);
        assert!(!entry.is_expired(in_5_days));
    }

    #[test]
    fn test_force_purge() {
        let mut bin = make_bin();
        let entry = bin.move_to_trash(Uuid::new_v4(), None, "x.mp4", 100, None, None);
        let eid = entry.id;

        let purged = bin.force_purge(eid);
        assert!(purged.is_some());
        assert!(bin.is_empty());
    }

    #[test]
    fn test_force_purge_nonexistent() {
        let mut bin = make_bin();
        assert!(bin.force_purge(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_total_size_bytes() {
        let mut bin = make_bin();
        bin.move_to_trash(Uuid::new_v4(), None, "a.mp4", 100, None, None);
        bin.move_to_trash(Uuid::new_v4(), None, "b.mp4", 200, None, None);
        assert_eq!(bin.total_size_bytes(), 300);
    }

    #[test]
    fn test_list_for_project() {
        let mut bin = make_bin();
        let proj_a = Uuid::new_v4();
        let proj_b = Uuid::new_v4();

        bin.move_to_trash(Uuid::new_v4(), Some(proj_a), "pa.mp4", 100, None, None);
        bin.move_to_trash(Uuid::new_v4(), Some(proj_b), "pb.mp4", 200, None, None);
        bin.move_to_trash(Uuid::new_v4(), Some(proj_a), "pa2.mp4", 150, None, None);

        let list_a = bin.list_for_project(proj_a);
        assert_eq!(list_a.len(), 2);

        let list_b = bin.list_for_project(proj_b);
        assert_eq!(list_b.len(), 1);
    }

    #[test]
    fn test_trash_entry_time_remaining() {
        let entry = TrashEntry {
            id: Uuid::new_v4(),
            asset_id: Uuid::new_v4(),
            project_id: None,
            deleted_by: None,
            deleted_at: Utc::now(),
            expires_at: Utc::now() + Duration::days(10),
            original_name: "test.mp4".to_string(),
            size_bytes: 0,
            metadata: None,
        };

        let remaining = entry.time_remaining(Utc::now());
        assert!(remaining > Duration::days(9));

        let past_expiry = entry.expires_at + Duration::days(1);
        assert_eq!(entry.time_remaining(past_expiry), Duration::zero());
    }

    #[test]
    fn test_trash_entry_serialization() {
        let entry = TrashEntry {
            id: Uuid::new_v4(),
            asset_id: Uuid::new_v4(),
            project_id: None,
            deleted_by: Some("admin".to_string()),
            deleted_at: Utc::now(),
            expires_at: Utc::now() + Duration::days(30),
            original_name: "ser_test.mp4".to_string(),
            size_bytes: 999,
            metadata: Some(serde_json::json!({"key": "value"})),
        };

        let json = serde_json::to_string(&entry).expect("serialize");
        let deser: TrashEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.original_name, "ser_test.mp4");
        assert_eq!(deser.size_bytes, 999);
    }

    #[test]
    fn test_get_entry() {
        let mut bin = make_bin();
        let entry = bin.move_to_trash(Uuid::new_v4(), None, "find_me.mp4", 50, None, None);
        let eid = entry.id;

        assert!(bin.get(eid).is_some());
        assert!(bin.get(Uuid::new_v4()).is_none());
    }

    #[test]
    fn test_deleted_by_field() {
        let mut bin = make_bin();
        let entry = bin.move_to_trash(
            Uuid::new_v4(),
            None,
            "x.mp4",
            10,
            Some("admin".to_string()),
            None,
        );
        assert_eq!(entry.deleted_by.as_deref(), Some("admin"));
    }
}
