#![allow(dead_code)]
//! Diff engine for comparing two timeline states.
//!
//! This module enables undo/redo by computing the minimal set of changes
//! between timeline snapshots.  Each change is represented as a [`DiffEntry`]
//! which can be applied forward (redo) or reversed (undo).

use std::collections::HashMap;

/// Identifies which aspect of the timeline changed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiffTarget {
    /// A clip was affected.
    Clip(u64),
    /// A track was affected.
    Track(u64),
    /// A transition between two clips.
    Transition(u64),
    /// A marker on the timeline.
    Marker(u64),
    /// A keyframe on an effect parameter.
    Keyframe(u64),
    /// Timeline-level property (name, frame-rate, etc.).
    TimelineProperty(String),
}

/// The kind of mutation that occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffKind {
    /// An item was added.
    Added,
    /// An item was removed.
    Removed,
    /// An item was modified.
    Modified,
    /// An item was moved from one position to another.
    Moved,
    /// An item was reordered within its container.
    Reordered,
}

/// A single serialisable property value used in before/after snapshots.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    #[allow(clippy::cast_precision_loss)]
    Float(f64),
    /// String value.
    Text(String),
    /// Boolean value.
    Bool(bool),
    /// Absent / null value.
    None,
}

/// One atomic difference between two timeline states.
#[derive(Debug, Clone)]
pub struct DiffEntry {
    /// What was affected.
    pub target: DiffTarget,
    /// Kind of change.
    pub kind: DiffKind,
    /// Property name that changed (empty for add/remove of whole items).
    pub property: String,
    /// Value before the change.
    pub old_value: PropValue,
    /// Value after the change.
    pub new_value: PropValue,
}

impl DiffEntry {
    /// Create a new diff entry.
    #[must_use]
    pub fn new(
        target: DiffTarget,
        kind: DiffKind,
        property: impl Into<String>,
        old_value: PropValue,
        new_value: PropValue,
    ) -> Self {
        Self {
            target,
            kind,
            property: property.into(),
            old_value,
            new_value,
        }
    }

    /// Create a diff entry for adding an item.
    #[must_use]
    pub fn added(target: DiffTarget) -> Self {
        Self::new(
            target,
            DiffKind::Added,
            "",
            PropValue::None,
            PropValue::None,
        )
    }

    /// Create a diff entry for removing an item.
    #[must_use]
    pub fn removed(target: DiffTarget) -> Self {
        Self::new(
            target,
            DiffKind::Removed,
            "",
            PropValue::None,
            PropValue::None,
        )
    }

    /// Produce the inverse of this diff entry (for undo).
    #[must_use]
    pub fn invert(&self) -> Self {
        let inv_kind = match self.kind {
            DiffKind::Added => DiffKind::Removed,
            DiffKind::Removed => DiffKind::Added,
            DiffKind::Modified | DiffKind::Moved | DiffKind::Reordered => self.kind.clone(),
        };
        Self {
            target: self.target.clone(),
            kind: inv_kind,
            property: self.property.clone(),
            old_value: self.new_value.clone(),
            new_value: self.old_value.clone(),
        }
    }
}

/// A complete diff between two timeline states.
#[derive(Debug, Clone)]
pub struct TimelineDiff {
    /// Ordered list of changes.
    pub entries: Vec<DiffEntry>,
    /// Human-readable description of the edit action.
    pub description: String,
    /// Monotonic revision number of the "after" state.
    pub revision: u64,
}

impl TimelineDiff {
    /// Create a new empty diff with a description.
    #[must_use]
    pub fn new(description: impl Into<String>, revision: u64) -> Self {
        Self {
            entries: Vec::new(),
            description: description.into(),
            revision,
        }
    }

    /// Add an entry to this diff.
    pub fn push(&mut self, entry: DiffEntry) {
        self.entries.push(entry);
    }

    /// Return the number of individual changes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the diff is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Produce the inverse diff (for undo).
    #[must_use]
    pub fn invert(&self) -> Self {
        let inverted: Vec<DiffEntry> = self.entries.iter().rev().map(DiffEntry::invert).collect();
        Self {
            entries: inverted,
            description: format!("Undo: {}", self.description),
            revision: self.revision.wrapping_sub(1),
        }
    }

    /// Filter entries by target kind.
    #[must_use]
    pub fn clip_changes(&self) -> Vec<&DiffEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.target, DiffTarget::Clip(_)))
            .collect()
    }

    /// Filter entries by diff kind.
    #[must_use]
    pub fn entries_of_kind(&self, kind: &DiffKind) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.kind == *kind).collect()
    }
}

/// Simple key-value snapshot of timeline state for diffing.
#[derive(Debug, Clone, Default)]
pub struct TimelineSnapshot {
    /// Properties keyed by (target, `property_name`).
    properties: HashMap<(DiffTarget, String), PropValue>,
}

impl TimelineSnapshot {
    /// Create a new empty snapshot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a property in the snapshot.
    pub fn set(&mut self, target: DiffTarget, property: impl Into<String>, value: PropValue) {
        self.properties.insert((target, property.into()), value);
    }

    /// Get a property from the snapshot.
    #[must_use]
    pub fn get(&self, target: &DiffTarget, property: &str) -> Option<&PropValue> {
        self.properties.get(&(target.clone(), property.to_string()))
    }

    /// Return the number of properties.
    #[must_use]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Compare this snapshot against another and produce a diff.
    #[must_use]
    pub fn diff(
        &self,
        other: &Self,
        description: impl Into<String>,
        revision: u64,
    ) -> TimelineDiff {
        let mut diff = TimelineDiff::new(description, revision);

        // Find modified and removed properties
        for ((target, prop), old_val) in &self.properties {
            match other.properties.get(&(target.clone(), prop.clone())) {
                Some(new_val) if new_val != old_val => {
                    diff.push(DiffEntry::new(
                        target.clone(),
                        DiffKind::Modified,
                        prop.clone(),
                        old_val.clone(),
                        new_val.clone(),
                    ));
                }
                None => {
                    diff.push(DiffEntry::new(
                        target.clone(),
                        DiffKind::Removed,
                        prop.clone(),
                        old_val.clone(),
                        PropValue::None,
                    ));
                }
                _ => {}
            }
        }

        // Find added properties
        for ((target, prop), new_val) in &other.properties {
            if !self
                .properties
                .contains_key(&(target.clone(), prop.clone()))
            {
                diff.push(DiffEntry::new(
                    target.clone(),
                    DiffKind::Added,
                    prop.clone(),
                    PropValue::None,
                    new_val.clone(),
                ));
            }
        }

        diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_entry_creation() {
        let entry = DiffEntry::new(
            DiffTarget::Clip(1),
            DiffKind::Modified,
            "in_point",
            PropValue::Int(0),
            PropValue::Int(10),
        );
        assert_eq!(entry.property, "in_point");
        assert_eq!(entry.old_value, PropValue::Int(0));
        assert_eq!(entry.new_value, PropValue::Int(10));
    }

    #[test]
    fn test_diff_entry_added_removed() {
        let added = DiffEntry::added(DiffTarget::Clip(5));
        assert_eq!(added.kind, DiffKind::Added);
        let removed = DiffEntry::removed(DiffTarget::Track(3));
        assert_eq!(removed.kind, DiffKind::Removed);
    }

    #[test]
    fn test_diff_entry_invert_modified() {
        let entry = DiffEntry::new(
            DiffTarget::Clip(1),
            DiffKind::Modified,
            "gain",
            PropValue::Float(0.5),
            PropValue::Float(1.0),
        );
        let inv = entry.invert();
        assert_eq!(inv.old_value, PropValue::Float(1.0));
        assert_eq!(inv.new_value, PropValue::Float(0.5));
    }

    #[test]
    fn test_diff_entry_invert_added_becomes_removed() {
        let entry = DiffEntry::added(DiffTarget::Marker(7));
        let inv = entry.invert();
        assert_eq!(inv.kind, DiffKind::Removed);
    }

    #[test]
    fn test_diff_entry_invert_removed_becomes_added() {
        let entry = DiffEntry::removed(DiffTarget::Keyframe(9));
        let inv = entry.invert();
        assert_eq!(inv.kind, DiffKind::Added);
    }

    #[test]
    fn test_timeline_diff_push_and_len() {
        let mut diff = TimelineDiff::new("test edit", 1);
        assert!(diff.is_empty());
        diff.push(DiffEntry::added(DiffTarget::Clip(1)));
        diff.push(DiffEntry::added(DiffTarget::Clip(2)));
        assert_eq!(diff.len(), 2);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_timeline_diff_invert() {
        let mut diff = TimelineDiff::new("add clips", 5);
        diff.push(DiffEntry::added(DiffTarget::Clip(1)));
        diff.push(DiffEntry::added(DiffTarget::Clip(2)));
        let inv = diff.invert();
        assert_eq!(inv.len(), 2);
        assert_eq!(inv.revision, 4);
        assert!(inv.description.starts_with("Undo:"));
        // Inverted diff should be in reverse order
        assert_eq!(inv.entries[0].kind, DiffKind::Removed);
        assert_eq!(inv.entries[0].target, DiffTarget::Clip(2));
    }

    #[test]
    fn test_timeline_diff_clip_changes() {
        let mut diff = TimelineDiff::new("mixed edit", 1);
        diff.push(DiffEntry::added(DiffTarget::Clip(1)));
        diff.push(DiffEntry::added(DiffTarget::Track(2)));
        diff.push(DiffEntry::removed(DiffTarget::Clip(3)));
        let clip_changes = diff.clip_changes();
        assert_eq!(clip_changes.len(), 2);
    }

    #[test]
    fn test_timeline_diff_entries_of_kind() {
        let mut diff = TimelineDiff::new("test", 1);
        diff.push(DiffEntry::added(DiffTarget::Clip(1)));
        diff.push(DiffEntry::removed(DiffTarget::Clip(2)));
        diff.push(DiffEntry::added(DiffTarget::Track(3)));
        let added = diff.entries_of_kind(&DiffKind::Added);
        assert_eq!(added.len(), 2);
    }

    #[test]
    fn test_snapshot_set_get() {
        let mut snap = TimelineSnapshot::new();
        snap.set(
            DiffTarget::Clip(1),
            "name",
            PropValue::Text("Clip A".into()),
        );
        let val = snap
            .get(&DiffTarget::Clip(1), "name")
            .expect("should succeed in test");
        assert_eq!(*val, PropValue::Text("Clip A".into()));
        assert_eq!(snap.property_count(), 1);
    }

    #[test]
    fn test_snapshot_diff_modified() {
        let mut before = TimelineSnapshot::new();
        before.set(DiffTarget::Clip(1), "in_point", PropValue::Int(0));
        let mut after = TimelineSnapshot::new();
        after.set(DiffTarget::Clip(1), "in_point", PropValue::Int(10));
        let diff = before.diff(&after, "trim clip", 2);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff.entries[0].kind, DiffKind::Modified);
    }

    #[test]
    fn test_snapshot_diff_added_and_removed() {
        let mut before = TimelineSnapshot::new();
        before.set(DiffTarget::Clip(1), "name", PropValue::Text("A".into()));
        let mut after = TimelineSnapshot::new();
        after.set(DiffTarget::Clip(2), "name", PropValue::Text("B".into()));
        let diff = before.diff(&after, "replace clip", 3);
        // One removed (clip 1) and one added (clip 2)
        assert_eq!(diff.len(), 2);
        let removed = diff.entries_of_kind(&DiffKind::Removed);
        let added = diff.entries_of_kind(&DiffKind::Added);
        assert_eq!(removed.len(), 1);
        assert_eq!(added.len(), 1);
    }

    #[test]
    fn test_snapshot_diff_unchanged() {
        let mut before = TimelineSnapshot::new();
        before.set(DiffTarget::Track(1), "name", PropValue::Text("V1".into()));
        let mut after = TimelineSnapshot::new();
        after.set(DiffTarget::Track(1), "name", PropValue::Text("V1".into()));
        let diff = before.diff(&after, "no change", 1);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_prop_value_equality() {
        assert_eq!(PropValue::Int(42), PropValue::Int(42));
        assert_ne!(PropValue::Int(1), PropValue::Int(2));
        assert_eq!(PropValue::Bool(true), PropValue::Bool(true));
        assert_ne!(PropValue::Bool(true), PropValue::Bool(false));
        assert_eq!(PropValue::None, PropValue::None);
    }
}
