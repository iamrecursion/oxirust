#![allow(dead_code)]
//! Metadata change history and audit trail.
//!
//! Tracks every modification to metadata fields, providing a full audit trail
//! that supports undo, diffing between revisions, and compliance reporting.

use std::collections::HashMap;
use std::fmt;

/// The kind of change applied to a metadata field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// A field was added.
    Add,
    /// A field value was modified.
    Modify,
    /// A field was removed.
    Remove,
}

impl fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "Add"),
            Self::Modify => write!(f, "Modify"),
            Self::Remove => write!(f, "Remove"),
        }
    }
}

/// A single change entry in the history log.
#[derive(Debug, Clone)]
pub struct ChangeEntry {
    /// Monotonically increasing revision number.
    pub revision: u64,
    /// Timestamp in milliseconds since UNIX epoch.
    pub timestamp_ms: u64,
    /// Key of the changed field.
    pub field_key: String,
    /// Kind of change.
    pub kind: ChangeKind,
    /// Previous value (None for Add).
    pub old_value: Option<String>,
    /// New value (None for Remove).
    pub new_value: Option<String>,
    /// Name of the user or system that made the change.
    pub author: String,
    /// Optional reason or comment.
    pub comment: String,
}

/// A full metadata history log for a single asset.
#[derive(Debug, Clone)]
pub struct MetadataHistory {
    /// Asset identifier this history belongs to.
    asset_id: String,
    /// Ordered list of changes (oldest first).
    entries: Vec<ChangeEntry>,
    /// Next revision number to assign.
    next_revision: u64,
}

impl MetadataHistory {
    /// Create a new empty history for an asset.
    pub fn new(asset_id: impl Into<String>) -> Self {
        Self {
            asset_id: asset_id.into(),
            entries: Vec::new(),
            next_revision: 1,
        }
    }

    /// Get the asset identifier.
    pub fn asset_id(&self) -> &str {
        &self.asset_id
    }

    /// Record an addition of a field.
    pub fn record_add(
        &mut self,
        field_key: impl Into<String>,
        new_value: impl Into<String>,
        author: impl Into<String>,
        timestamp_ms: u64,
    ) -> u64 {
        self.push_entry(
            field_key.into(),
            ChangeKind::Add,
            None,
            Some(new_value.into()),
            author.into(),
            String::new(),
            timestamp_ms,
        )
    }

    /// Record a modification of a field.
    pub fn record_modify(
        &mut self,
        field_key: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
        author: impl Into<String>,
        timestamp_ms: u64,
    ) -> u64 {
        self.push_entry(
            field_key.into(),
            ChangeKind::Modify,
            Some(old_value.into()),
            Some(new_value.into()),
            author.into(),
            String::new(),
            timestamp_ms,
        )
    }

    /// Record a removal of a field.
    pub fn record_remove(
        &mut self,
        field_key: impl Into<String>,
        old_value: impl Into<String>,
        author: impl Into<String>,
        timestamp_ms: u64,
    ) -> u64 {
        self.push_entry(
            field_key.into(),
            ChangeKind::Remove,
            Some(old_value.into()),
            None,
            author.into(),
            String::new(),
            timestamp_ms,
        )
    }

    /// Record a change with a comment.
    #[allow(clippy::too_many_arguments)]
    pub fn record_with_comment(
        &mut self,
        field_key: impl Into<String>,
        kind: ChangeKind,
        old_value: Option<String>,
        new_value: Option<String>,
        author: impl Into<String>,
        comment: impl Into<String>,
        timestamp_ms: u64,
    ) -> u64 {
        self.push_entry(
            field_key.into(),
            kind,
            old_value,
            new_value,
            author.into(),
            comment.into(),
            timestamp_ms,
        )
    }

    /// Get all entries.
    pub fn entries(&self) -> &[ChangeEntry] {
        &self.entries
    }

    /// Get the total number of changes.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no changes recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the latest revision number (0 if empty).
    pub fn latest_revision(&self) -> u64 {
        if self.entries.is_empty() {
            0
        } else {
            self.next_revision - 1
        }
    }

    /// Get a specific entry by revision number.
    pub fn get_revision(&self, revision: u64) -> Option<&ChangeEntry> {
        self.entries.iter().find(|e| e.revision == revision)
    }

    /// Get all changes for a specific field.
    pub fn field_history(&self, field_key: &str) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.field_key == field_key)
            .collect()
    }

    /// Get all changes by a specific author.
    pub fn by_author(&self, author: &str) -> Vec<&ChangeEntry> {
        self.entries.iter().filter(|e| e.author == author).collect()
    }

    /// Get changes within a time range (inclusive).
    pub fn in_time_range(&self, start_ms: u64, end_ms: u64) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Reconstruct the metadata state at a given revision.
    ///
    /// Replays all changes up to and including `revision` and returns the
    /// resulting key-value map.
    pub fn state_at_revision(&self, revision: u64) -> HashMap<String, String> {
        let mut state = HashMap::new();
        for entry in &self.entries {
            if entry.revision > revision {
                break;
            }
            match entry.kind {
                ChangeKind::Add | ChangeKind::Modify => {
                    if let Some(ref val) = entry.new_value {
                        state.insert(entry.field_key.clone(), val.clone());
                    }
                }
                ChangeKind::Remove => {
                    state.remove(&entry.field_key);
                }
            }
        }
        state
    }

    /// Compute a diff between two revisions (fields that changed).
    pub fn diff_revisions(&self, from: u64, to: u64) -> Vec<FieldDiff> {
        let state_from = self.state_at_revision(from);
        let state_to = self.state_at_revision(to);
        let mut diffs = Vec::new();

        // Fields in state_to that differ or are new
        for (key, new_val) in &state_to {
            match state_from.get(key) {
                Some(old_val) if old_val != new_val => {
                    diffs.push(FieldDiff {
                        key: key.clone(),
                        kind: ChangeKind::Modify,
                        old_value: Some(old_val.clone()),
                        new_value: Some(new_val.clone()),
                    });
                }
                None => {
                    diffs.push(FieldDiff {
                        key: key.clone(),
                        kind: ChangeKind::Add,
                        old_value: None,
                        new_value: Some(new_val.clone()),
                    });
                }
                _ => {}
            }
        }

        // Fields removed
        for (key, old_val) in &state_from {
            if !state_to.contains_key(key) {
                diffs.push(FieldDiff {
                    key: key.clone(),
                    kind: ChangeKind::Remove,
                    old_value: Some(old_val.clone()),
                    new_value: None,
                });
            }
        }

        diffs.sort_by(|a, b| a.key.cmp(&b.key));
        diffs
    }

    /// Internal helper to push an entry and advance the revision counter.
    #[allow(clippy::too_many_arguments)]
    fn push_entry(
        &mut self,
        field_key: String,
        kind: ChangeKind,
        old_value: Option<String>,
        new_value: Option<String>,
        author: String,
        comment: String,
        timestamp_ms: u64,
    ) -> u64 {
        let rev = self.next_revision;
        self.entries.push(ChangeEntry {
            revision: rev,
            timestamp_ms,
            field_key,
            kind,
            old_value,
            new_value,
            author,
            comment,
        });
        self.next_revision += 1;
        rev
    }
}

/// A diff entry between two revisions.
#[derive(Debug, Clone)]
pub struct FieldDiff {
    /// Field key.
    pub key: String,
    /// Kind of change.
    pub kind: ChangeKind,
    /// Old value (if any).
    pub old_value: Option<String>,
    /// New value (if any).
    pub new_value: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_history_is_empty() {
        let h = MetadataHistory::new("asset-001");
        assert_eq!(h.asset_id(), "asset-001");
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.latest_revision(), 0);
    }

    #[test]
    fn test_record_add() {
        let mut h = MetadataHistory::new("a1");
        let rev = h.record_add("title", "Hello", "admin", 1000);
        assert_eq!(rev, 1);
        assert_eq!(h.len(), 1);
        assert_eq!(h.latest_revision(), 1);
    }

    #[test]
    fn test_record_modify() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "Old", "admin", 1000);
        let rev = h.record_modify("title", "Old", "New", "editor", 2000);
        assert_eq!(rev, 2);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_record_remove() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "V", "admin", 1000);
        let rev = h.record_remove("title", "V", "admin", 2000);
        assert_eq!(rev, 2);
    }

    #[test]
    fn test_get_revision() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("a", "1", "u", 100);
        h.record_add("b", "2", "u", 200);
        let entry = h.get_revision(2).expect("should succeed in test");
        assert_eq!(entry.field_key, "b");
        assert!(h.get_revision(99).is_none());
    }

    #[test]
    fn test_field_history() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "V1", "u", 100);
        h.record_add("artist", "A", "u", 200);
        h.record_modify("title", "V1", "V2", "u", 300);
        let title_hist = h.field_history("title");
        assert_eq!(title_hist.len(), 2);
        assert_eq!(title_hist[0].kind, ChangeKind::Add);
        assert_eq!(title_hist[1].kind, ChangeKind::Modify);
    }

    #[test]
    fn test_by_author() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("a", "1", "alice", 100);
        h.record_add("b", "2", "bob", 200);
        h.record_add("c", "3", "alice", 300);
        assert_eq!(h.by_author("alice").len(), 2);
        assert_eq!(h.by_author("bob").len(), 1);
    }

    #[test]
    fn test_in_time_range() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("a", "1", "u", 100);
        h.record_add("b", "2", "u", 200);
        h.record_add("c", "3", "u", 300);
        let range = h.in_time_range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].field_key, "b");
    }

    #[test]
    fn test_state_at_revision() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "T1", "u", 100);
        h.record_add("artist", "A1", "u", 200);
        h.record_modify("title", "T1", "T2", "u", 300);

        let s1 = h.state_at_revision(1);
        assert_eq!(s1.get("title").expect("should succeed in test"), "T1");
        assert!(s1.get("artist").is_none());

        let s3 = h.state_at_revision(3);
        assert_eq!(s3.get("title").expect("should succeed in test"), "T2");
        assert_eq!(s3.get("artist").expect("should succeed in test"), "A1");
    }

    #[test]
    fn test_state_at_revision_with_remove() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "T1", "u", 100);
        h.record_remove("title", "T1", "u", 200);
        let s = h.state_at_revision(2);
        assert!(s.get("title").is_none());
    }

    #[test]
    fn test_diff_revisions_add() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "T1", "u", 100);
        h.record_add("artist", "A1", "u", 200);
        let diffs = h.diff_revisions(1, 2);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].key, "artist");
        assert_eq!(diffs[0].kind, ChangeKind::Add);
    }

    #[test]
    fn test_diff_revisions_modify() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "Old", "u", 100);
        h.record_modify("title", "Old", "New", "u", 200);
        let diffs = h.diff_revisions(1, 2);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].kind, ChangeKind::Modify);
        assert_eq!(diffs[0].old_value.as_deref(), Some("Old"));
        assert_eq!(diffs[0].new_value.as_deref(), Some("New"));
    }

    #[test]
    fn test_diff_revisions_remove() {
        let mut h = MetadataHistory::new("a1");
        h.record_add("title", "T1", "u", 100);
        h.record_remove("title", "T1", "u", 200);
        let diffs = h.diff_revisions(1, 2);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].kind, ChangeKind::Remove);
    }

    #[test]
    fn test_record_with_comment() {
        let mut h = MetadataHistory::new("a1");
        let rev = h.record_with_comment(
            "title",
            ChangeKind::Add,
            None,
            Some("T1".into()),
            "admin",
            "Initial import",
            1000,
        );
        assert_eq!(rev, 1);
        assert_eq!(h.entries()[0].comment, "Initial import");
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Add.to_string(), "Add");
        assert_eq!(ChangeKind::Modify.to_string(), "Modify");
        assert_eq!(ChangeKind::Remove.to_string(), "Remove");
    }
}
