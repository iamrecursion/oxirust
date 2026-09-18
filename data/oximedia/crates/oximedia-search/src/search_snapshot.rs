#![allow(dead_code)]

//! Point-in-time search index snapshots.
//!
//! This module enables creating, comparing, and restoring lightweight snapshots
//! of search index state. Snapshots record document counts, field counts, and
//! checksum-like digests so operators can verify index integrity over time.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Compact summary of the state of a single index field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSnapshot {
    /// Name of the field.
    pub name: String,
    /// Number of documents that contain this field.
    pub doc_count: u64,
    /// Total number of indexed terms in this field.
    pub term_count: u64,
}

/// A point-in-time snapshot of the entire search index.
#[derive(Debug, Clone)]
pub struct IndexSnapshot {
    /// Snapshot identifier (monotonically increasing).
    pub id: u64,
    /// Unix-epoch timestamp (seconds) when the snapshot was taken.
    pub timestamp: u64,
    /// Total number of documents in the index.
    pub total_docs: u64,
    /// Per-field snapshots.
    pub fields: Vec<FieldSnapshot>,
    /// A simple checksum derived from field data.
    pub checksum: u64,
    /// Optional human-readable label.
    pub label: Option<String>,
}

/// Diff between two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Snapshot id of the "before" state.
    pub before_id: u64,
    /// Snapshot id of the "after" state.
    pub after_id: u64,
    /// Change in total documents.
    pub doc_delta: i64,
    /// Fields added since the before snapshot.
    pub fields_added: Vec<String>,
    /// Fields removed since the before snapshot.
    pub fields_removed: Vec<String>,
    /// Fields whose doc_count or term_count changed.
    pub fields_changed: Vec<String>,
}

/// Manager that stores a sequence of snapshots.
#[derive(Debug, Default)]
pub struct SnapshotStore {
    /// Ordered list of snapshots.
    snapshots: Vec<IndexSnapshot>,
    /// Next id to assign.
    next_id: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a simple additive checksum from field data.
fn compute_checksum(fields: &[FieldSnapshot]) -> u64 {
    let mut ck: u64 = 0;
    for f in fields {
        ck = ck.wrapping_add(f.doc_count);
        ck = ck.wrapping_add(f.term_count);
        for b in f.name.as_bytes() {
            ck = ck.wrapping_add(u64::from(*b));
        }
    }
    ck
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl IndexSnapshot {
    /// Build a new snapshot from raw field data.
    pub fn build(id: u64, timestamp: u64, total_docs: u64, fields: Vec<FieldSnapshot>) -> Self {
        let checksum = compute_checksum(&fields);
        Self {
            id,
            timestamp,
            total_docs,
            fields,
            checksum,
            label: None,
        }
    }

    /// Verify that the stored checksum matches re-computation.
    pub fn verify_checksum(&self) -> bool {
        compute_checksum(&self.fields) == self.checksum
    }

    /// Number of fields recorded.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

impl SnapshotStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a snapshot and add it to the store. Returns the assigned id.
    pub fn take_snapshot(
        &mut self,
        timestamp: u64,
        total_docs: u64,
        fields: Vec<FieldSnapshot>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let snap = IndexSnapshot::build(id, timestamp, total_docs, fields);
        self.snapshots.push(snap);
        id
    }

    /// Number of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Retrieve a snapshot by id.
    pub fn get(&self, id: u64) -> Option<&IndexSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Retrieve the most recent snapshot.
    pub fn latest(&self) -> Option<&IndexSnapshot> {
        self.snapshots.last()
    }

    /// Compute a diff between two snapshots identified by id.
    pub fn diff(&self, before_id: u64, after_id: u64) -> Option<SnapshotDiff> {
        let before = self.get(before_id)?;
        let after = self.get(after_id)?;

        let before_map: HashMap<&str, &FieldSnapshot> =
            before.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let after_map: HashMap<&str, &FieldSnapshot> =
            after.fields.iter().map(|f| (f.name.as_str(), f)).collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut changed = Vec::new();

        for (name, af) in &after_map {
            match before_map.get(name) {
                Some(bf) => {
                    if af.doc_count != bf.doc_count || af.term_count != bf.term_count {
                        changed.push((*name).to_string());
                    }
                }
                None => added.push((*name).to_string()),
            }
        }
        for name in before_map.keys() {
            if !after_map.contains_key(name) {
                removed.push((*name).to_string());
            }
        }

        Some(SnapshotDiff {
            before_id,
            after_id,
            doc_delta: after.total_docs as i64 - before.total_docs as i64,
            fields_added: added,
            fields_removed: removed,
            fields_changed: changed,
        })
    }

    /// Remove all snapshots older than the given timestamp.
    pub fn prune_before(&mut self, timestamp: u64) {
        self.snapshots.retain(|s| s.timestamp >= timestamp);
    }

    /// Clear the entire store.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Label a specific snapshot.
    pub fn label(&mut self, id: u64, label: &str) {
        if let Some(snap) = self.snapshots.iter_mut().find(|s| s.id == id) {
            snap.label = Some(label.to_string());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fields() -> Vec<FieldSnapshot> {
        vec![
            FieldSnapshot {
                name: "title".into(),
                doc_count: 100,
                term_count: 500,
            },
            FieldSnapshot {
                name: "body".into(),
                doc_count: 90,
                term_count: 3000,
            },
        ]
    }

    #[test]
    fn test_store_new_is_empty() {
        let s = SnapshotStore::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_take_snapshot() {
        let mut s = SnapshotStore::new();
        let id = s.take_snapshot(1000, 42, sample_fields());
        assert_eq!(id, 0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_get_snapshot() {
        let mut s = SnapshotStore::new();
        let id = s.take_snapshot(1000, 42, sample_fields());
        let snap = s.get(id).expect("should succeed in test");
        assert_eq!(snap.total_docs, 42);
        assert_eq!(snap.field_count(), 2);
    }

    #[test]
    fn test_latest() {
        let mut s = SnapshotStore::new();
        s.take_snapshot(1000, 10, sample_fields());
        s.take_snapshot(2000, 20, sample_fields());
        let latest = s.latest().expect("should succeed in test");
        assert_eq!(latest.total_docs, 20);
    }

    #[test]
    fn test_verify_checksum() {
        let mut s = SnapshotStore::new();
        let id = s.take_snapshot(1000, 42, sample_fields());
        let snap = s.get(id).expect("should succeed in test");
        assert!(snap.verify_checksum());
    }

    #[test]
    fn test_diff_doc_delta() {
        let mut s = SnapshotStore::new();
        let id1 = s.take_snapshot(1000, 10, sample_fields());
        let id2 = s.take_snapshot(2000, 25, sample_fields());
        let diff = s.diff(id1, id2).expect("should succeed in test");
        assert_eq!(diff.doc_delta, 15);
    }

    #[test]
    fn test_diff_field_added() {
        let mut s = SnapshotStore::new();
        let id1 = s.take_snapshot(
            1000,
            10,
            vec![FieldSnapshot {
                name: "title".into(),
                doc_count: 1,
                term_count: 1,
            }],
        );
        let mut fields = sample_fields();
        fields.push(FieldSnapshot {
            name: "tags".into(),
            doc_count: 5,
            term_count: 20,
        });
        let id2 = s.take_snapshot(2000, 20, fields);
        let diff = s.diff(id1, id2).expect("should succeed in test");
        assert!(diff.fields_added.contains(&"body".to_string()));
        assert!(diff.fields_added.contains(&"tags".to_string()));
    }

    #[test]
    fn test_diff_field_removed() {
        let mut s = SnapshotStore::new();
        let id1 = s.take_snapshot(1000, 10, sample_fields());
        let id2 = s.take_snapshot(
            2000,
            10,
            vec![FieldSnapshot {
                name: "title".into(),
                doc_count: 100,
                term_count: 500,
            }],
        );
        let diff = s.diff(id1, id2).expect("should succeed in test");
        assert!(diff.fields_removed.contains(&"body".to_string()));
    }

    #[test]
    fn test_diff_field_changed() {
        let mut s = SnapshotStore::new();
        let id1 = s.take_snapshot(1000, 10, sample_fields());
        let id2 = s.take_snapshot(
            2000,
            10,
            vec![
                FieldSnapshot {
                    name: "title".into(),
                    doc_count: 200,
                    term_count: 500,
                },
                FieldSnapshot {
                    name: "body".into(),
                    doc_count: 90,
                    term_count: 3000,
                },
            ],
        );
        let diff = s.diff(id1, id2).expect("should succeed in test");
        assert!(diff.fields_changed.contains(&"title".to_string()));
    }

    #[test]
    fn test_prune_before() {
        let mut s = SnapshotStore::new();
        s.take_snapshot(100, 1, sample_fields());
        s.take_snapshot(200, 2, sample_fields());
        s.take_snapshot(300, 3, sample_fields());
        s.prune_before(200);
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut s = SnapshotStore::new();
        s.take_snapshot(100, 1, sample_fields());
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn test_label_snapshot() {
        let mut s = SnapshotStore::new();
        let id = s.take_snapshot(100, 1, sample_fields());
        s.label(id, "release-v1");
        let snap = s.get(id).expect("should succeed in test");
        assert_eq!(snap.label.as_deref(), Some("release-v1"));
    }

    #[test]
    fn test_diff_nonexistent_returns_none() {
        let s = SnapshotStore::new();
        assert!(s.diff(0, 1).is_none());
    }
}
