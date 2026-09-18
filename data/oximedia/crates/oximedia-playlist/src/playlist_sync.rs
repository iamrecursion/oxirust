//! Playlist synchronisation between local and remote / replica sources.
//!
//! Provides `SyncDirection`, `SyncConflict`, `PlaylistSyncConfig`, and a
//! `PlaylistSyncer` that reconciles two playlist item lists.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Direction of a playlist synchronisation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
    /// Push local changes to the remote; remote is overwritten.
    LocalToRemote,
    /// Pull remote changes to local; local is overwritten.
    RemoteToLocal,
    /// Merge both sides; conflicts must be resolved.
    Bidirectional,
}

impl SyncDirection {
    /// Short label for this direction.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::LocalToRemote => "local_to_remote",
            Self::RemoteToLocal => "remote_to_local",
            Self::Bidirectional => "bidirectional",
        }
    }

    /// Returns `true` for `Bidirectional` merges.
    #[must_use]
    pub fn is_merge(&self) -> bool {
        *self == Self::Bidirectional
    }
}

/// A conflict detected during bidirectional synchronisation.
#[derive(Debug, Clone)]
pub struct SyncConflict {
    /// ID of the conflicting playlist item.
    pub item_id: String,
    /// Value on the local side.
    pub local_value: String,
    /// Value on the remote side.
    pub remote_value: String,
    /// Which field triggered the conflict.
    pub field: String,
}

impl SyncConflict {
    /// Create a new sync conflict record.
    #[must_use]
    pub fn new(
        item_id: impl Into<String>,
        field: impl Into<String>,
        local_value: impl Into<String>,
        remote_value: impl Into<String>,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            field: field.into(),
            local_value: local_value.into(),
            remote_value: remote_value.into(),
        }
    }

    /// Human-readable description of this conflict.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "Conflict on '{}' for item '{}': local='{}' remote='{}'",
            self.field, self.item_id, self.local_value, self.remote_value
        )
    }

    /// Whether local and remote values differ.
    #[must_use]
    pub fn has_difference(&self) -> bool {
        self.local_value != self.remote_value
    }
}

/// A single playlist item used in sync operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncItem {
    /// Unique item identifier.
    pub id: String,
    /// Item title.
    pub title: String,
    /// Duration string (e.g. "PT3600S").
    pub duration: String,
    /// Order index within the playlist.
    pub order: u32,
}

impl SyncItem {
    /// Create a new sync item.
    #[must_use]
    pub fn new(id: &str, title: &str, duration: &str, order: u32) -> Self {
        Self {
            id: id.to_owned(),
            title: title.to_owned(),
            duration: duration.to_owned(),
            order,
        }
    }
}

/// Configuration for a playlist sync operation.
#[derive(Debug, Clone)]
pub struct PlaylistSyncConfig {
    /// Direction of the sync.
    pub direction: SyncDirection,
    /// If `true`, items present only on the source are deleted from the destination.
    pub delete_missing: bool,
    /// If `true`, keep the local version when a conflict is detected.
    pub prefer_local_on_conflict: bool,
    /// Maximum number of items the sync will process in one pass.
    pub max_items: usize,
}

impl Default for PlaylistSyncConfig {
    fn default() -> Self {
        Self {
            direction: SyncDirection::Bidirectional,
            delete_missing: false,
            prefer_local_on_conflict: true,
            max_items: 10_000,
        }
    }
}

impl PlaylistSyncConfig {
    /// Returns `true` if this is a merge (bidirectional) sync.
    #[must_use]
    pub fn is_merge(&self) -> bool {
        self.direction.is_merge()
    }
}

/// Result returned by a sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    /// Items added to the destination.
    pub added: Vec<SyncItem>,
    /// Items updated in the destination.
    pub updated: Vec<SyncItem>,
    /// Items removed from the destination.
    pub removed: Vec<String>,
    /// Conflicts detected (for bidirectional sync).
    pub conflicts: Vec<SyncConflict>,
}

impl SyncResult {
    /// Total number of changes applied.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.added.len() + self.updated.len() + self.removed.len()
    }
}

/// Synchronises two playlist item lists according to a `PlaylistSyncConfig`.
#[derive(Debug, Clone)]
pub struct PlaylistSyncer {
    config: PlaylistSyncConfig,
    conflict_count: usize,
}

impl PlaylistSyncer {
    /// Create a new syncer with the given configuration.
    #[must_use]
    pub fn new(config: PlaylistSyncConfig) -> Self {
        Self {
            config,
            conflict_count: 0,
        }
    }

    /// Create a syncer with default configuration.
    #[must_use]
    pub fn default_syncer() -> Self {
        Self::new(PlaylistSyncConfig::default())
    }

    /// Perform a sync from `source` to `destination`.
    ///
    /// * For `LocalToRemote` and `RemoteToLocal`, the destination is updated
    ///   to match the source.
    /// * For `Bidirectional`, items unique to either side are merged; items
    ///   that differ on both sides generate a `SyncConflict`.
    ///
    /// Returns a `SyncResult` describing all changes made.
    pub fn sync(&mut self, local: &[SyncItem], remote: &[SyncItem]) -> SyncResult {
        self.conflict_count = 0;
        match self.config.direction {
            SyncDirection::LocalToRemote => self.one_way_sync(local, remote),
            SyncDirection::RemoteToLocal => self.one_way_sync(remote, local),
            SyncDirection::Bidirectional => self.merge_sync(local, remote),
        }
    }

    /// Return the number of conflicts encountered during the last `sync()` call.
    #[must_use]
    pub fn conflict_count(&self) -> usize {
        self.conflict_count
    }

    // ---- internal helpers ----

    /// One-way sync: make `dest` match `src`.
    fn one_way_sync(&self, src: &[SyncItem], dest: &[SyncItem]) -> SyncResult {
        let mut result = SyncResult::default();

        let dest_map: std::collections::HashMap<&str, &SyncItem> =
            dest.iter().map(|i| (i.id.as_str(), i)).collect();

        for item in src.iter().take(self.config.max_items) {
            match dest_map.get(item.id.as_str()) {
                None => result.added.push(item.clone()),
                Some(existing) if *existing != item => result.updated.push(item.clone()),
                _ => {}
            }
        }

        if self.config.delete_missing {
            let src_ids: std::collections::HashSet<&str> =
                src.iter().map(|i| i.id.as_str()).collect();
            for d in dest {
                if !src_ids.contains(d.id.as_str()) {
                    result.removed.push(d.id.clone());
                }
            }
        }

        result
    }

    /// Bidirectional merge sync.
    fn merge_sync(&mut self, local: &[SyncItem], remote: &[SyncItem]) -> SyncResult {
        let mut result = SyncResult::default();

        let local_map: std::collections::HashMap<&str, &SyncItem> =
            local.iter().map(|i| (i.id.as_str(), i)).collect();
        let remote_map: std::collections::HashMap<&str, &SyncItem> =
            remote.iter().map(|i| (i.id.as_str(), i)).collect();

        // Items in local but not remote → add to remote
        for item in local.iter().take(self.config.max_items) {
            if !remote_map.contains_key(item.id.as_str()) {
                result.added.push(item.clone());
            }
        }

        // Items in remote but not local → add to local
        for item in remote.iter().take(self.config.max_items) {
            if !local_map.contains_key(item.id.as_str()) {
                result.added.push(item.clone());
            }
        }

        // Items in both but differing → conflict
        for item in local.iter().take(self.config.max_items) {
            if let Some(remote_item) = remote_map.get(item.id.as_str()) {
                if *remote_item != item {
                    let conflict =
                        SyncConflict::new(&item.id, "content", &item.title, &remote_item.title);
                    self.conflict_count += 1;
                    result.conflicts.push(conflict);

                    // Resolve according to preference
                    let resolved = if self.config.prefer_local_on_conflict {
                        item.clone()
                    } else {
                        (*remote_item).clone()
                    };
                    result.updated.push(resolved);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, order: u32) -> SyncItem {
        SyncItem::new(id, &format!("Track {id}"), "PT210S", order)
    }

    fn item_titled(id: &str, title: &str, order: u32) -> SyncItem {
        SyncItem::new(id, title, "PT210S", order)
    }

    // SyncDirection tests

    #[test]
    fn test_direction_labels() {
        assert!(!SyncDirection::LocalToRemote.label().is_empty());
        assert!(!SyncDirection::RemoteToLocal.label().is_empty());
        assert!(!SyncDirection::Bidirectional.label().is_empty());
    }

    #[test]
    fn test_is_merge_only_bidirectional() {
        assert!(SyncDirection::Bidirectional.is_merge());
        assert!(!SyncDirection::LocalToRemote.is_merge());
        assert!(!SyncDirection::RemoteToLocal.is_merge());
    }

    // SyncConflict tests

    #[test]
    fn test_conflict_description_non_empty() {
        let c = SyncConflict::new("id1", "title", "Local Title", "Remote Title");
        assert!(!c.description().is_empty());
    }

    #[test]
    fn test_conflict_has_difference_true() {
        let c = SyncConflict::new("id1", "title", "A", "B");
        assert!(c.has_difference());
    }

    #[test]
    fn test_conflict_has_difference_false_when_equal() {
        let c = SyncConflict::new("id1", "title", "Same", "Same");
        assert!(!c.has_difference());
    }

    // PlaylistSyncConfig tests

    #[test]
    fn test_config_is_merge_default() {
        let cfg = PlaylistSyncConfig::default();
        assert!(cfg.is_merge());
    }

    #[test]
    fn test_config_not_merge_one_way() {
        let cfg = PlaylistSyncConfig {
            direction: SyncDirection::LocalToRemote,
            ..Default::default()
        };
        assert!(!cfg.is_merge());
    }

    // PlaylistSyncer tests

    #[test]
    fn test_sync_local_to_remote_adds_new_item() {
        let local = vec![item("a", 0), item("b", 1)];
        let remote = vec![item("a", 0)];
        let mut syncer = PlaylistSyncer::new(PlaylistSyncConfig {
            direction: SyncDirection::LocalToRemote,
            ..Default::default()
        });
        let result = syncer.sync(&local, &remote);
        assert_eq!(result.added.len(), 1);
        assert_eq!(result.added[0].id, "b");
    }

    #[test]
    fn test_sync_remote_to_local_pulls_new_item() {
        let local = vec![item("a", 0)];
        let remote = vec![item("a", 0), item("c", 1)];
        let mut syncer = PlaylistSyncer::new(PlaylistSyncConfig {
            direction: SyncDirection::RemoteToLocal,
            ..Default::default()
        });
        let result = syncer.sync(&local, &remote);
        assert_eq!(result.added.len(), 1);
        assert_eq!(result.added[0].id, "c");
    }

    #[test]
    fn test_sync_no_changes_when_identical() {
        let items = vec![item("a", 0), item("b", 1)];
        let mut syncer = PlaylistSyncer::new(PlaylistSyncConfig {
            direction: SyncDirection::LocalToRemote,
            ..Default::default()
        });
        let result = syncer.sync(&items, &items);
        assert_eq!(result.change_count(), 0);
    }

    #[test]
    fn test_bidirectional_merge_unique_items() {
        let local = vec![item("a", 0)];
        let remote = vec![item("b", 0)];
        let mut syncer = PlaylistSyncer::default_syncer();
        let result = syncer.sync(&local, &remote);
        // "a" added to remote, "b" added to local
        assert_eq!(result.added.len(), 2);
    }

    #[test]
    fn test_bidirectional_conflict_detected() {
        let local = vec![item_titled("a", "Local Title", 0)];
        let remote = vec![item_titled("a", "Remote Title", 0)];
        let mut syncer = PlaylistSyncer::default_syncer();
        let result = syncer.sync(&local, &remote);
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(syncer.conflict_count(), 1);
    }

    #[test]
    fn test_conflict_prefers_local_by_default() {
        let local = vec![item_titled("a", "Local Title", 0)];
        let remote = vec![item_titled("a", "Remote Title", 0)];
        let mut syncer = PlaylistSyncer::default_syncer();
        let result = syncer.sync(&local, &remote);
        assert_eq!(result.updated[0].title, "Local Title");
    }

    #[test]
    fn test_conflict_prefers_remote_when_configured() {
        let local = vec![item_titled("a", "Local Title", 0)];
        let remote = vec![item_titled("a", "Remote Title", 0)];
        let mut syncer = PlaylistSyncer::new(PlaylistSyncConfig {
            prefer_local_on_conflict: false,
            ..Default::default()
        });
        let result = syncer.sync(&local, &remote);
        assert_eq!(result.updated[0].title, "Remote Title");
    }

    #[test]
    fn test_delete_missing_removes_stale_items() {
        let local = vec![item("a", 0)];
        let remote = vec![item("a", 0), item("stale", 1)];
        let mut syncer = PlaylistSyncer::new(PlaylistSyncConfig {
            direction: SyncDirection::LocalToRemote,
            delete_missing: true,
            ..Default::default()
        });
        let result = syncer.sync(&local, &remote);
        assert!(result.removed.contains(&"stale".to_string()));
    }
}
