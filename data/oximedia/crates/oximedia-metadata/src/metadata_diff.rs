//! Metadata diff computation, application, and three-way merge.
//!
//! Provides tools to compare two metadata states and produce a structured
//! diff that can be applied or inspected. Also supports three-way merge
//! for resolving concurrent metadata edits against a common base.

use std::collections::{HashMap, HashSet};

/// Describes a single change to a metadata field.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataChange {
    /// A new field was added with the given value.
    Added(String),
    /// A field was removed; the old value is stored.
    Removed(String),
    /// A field's value changed from old to new.
    Modified { old: String, new: String },
}

impl MetadataChange {
    /// Returns `true` if this change is considered destructive (removes or modifies data).
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Removed(_) | Self::Modified { .. })
    }

    /// Returns the new value string if one exists after the change.
    pub fn new_value(&self) -> Option<&str> {
        match self {
            Self::Added(v) | Self::Modified { new: v, .. } => Some(v),
            Self::Removed(_) => None,
        }
    }

    /// Returns the old value string if one existed before the change.
    pub fn old_value(&self) -> Option<&str> {
        match self {
            Self::Removed(v) | Self::Modified { old: v, .. } => Some(v),
            Self::Added(_) => None,
        }
    }
}

/// A complete diff between two metadata snapshots.
#[derive(Debug, Clone, Default)]
pub struct MetadataDiff {
    changes: HashMap<String, MetadataChange>,
}

impl MetadataDiff {
    /// Create an empty diff.
    pub fn new() -> Self {
        Self {
            changes: HashMap::new(),
        }
    }

    /// Compute the diff between `before` and `after` snapshots.
    ///
    /// Both arguments are `&HashMap<String, String>` representing field name -> text value.
    pub fn compute(before: &HashMap<String, String>, after: &HashMap<String, String>) -> Self {
        let mut changes = HashMap::new();

        // Detect added and modified fields.
        for (key, new_val) in after {
            match before.get(key) {
                None => {
                    changes.insert(key.clone(), MetadataChange::Added(new_val.clone()));
                }
                Some(old_val) if old_val != new_val => {
                    changes.insert(
                        key.clone(),
                        MetadataChange::Modified {
                            old: old_val.clone(),
                            new: new_val.clone(),
                        },
                    );
                }
                _ => {}
            }
        }

        // Detect removed fields.
        for key in before.keys() {
            if !after.contains_key(key) {
                let old_val = before[key].clone();
                changes.insert(key.clone(), MetadataChange::Removed(old_val));
            }
        }

        Self { changes }
    }

    /// Returns the names of all fields that were added.
    pub fn added_fields(&self) -> Vec<&str> {
        self.changes
            .iter()
            .filter_map(|(k, v)| {
                if matches!(v, MetadataChange::Added(_)) {
                    Some(k.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the names of all fields that were removed.
    pub fn removed_fields(&self) -> Vec<&str> {
        self.changes
            .iter()
            .filter_map(|(k, v)| {
                if matches!(v, MetadataChange::Removed(_)) {
                    Some(k.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the names of all fields whose values changed.
    pub fn changed_fields(&self) -> Vec<&str> {
        self.changes
            .iter()
            .filter_map(|(k, v)| {
                if matches!(v, MetadataChange::Modified { .. }) {
                    Some(k.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns the full change map.
    pub fn changes(&self) -> &HashMap<String, MetadataChange> {
        &self.changes
    }

    /// Returns `true` if there are no changes.
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Returns the total number of changes.
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Returns `true` if any change in the diff is destructive.
    pub fn has_destructive_changes(&self) -> bool {
        self.changes.values().any(MetadataChange::is_destructive)
    }
}

/// Applies a [`MetadataDiff`] to a metadata snapshot.
pub struct MetadataDiffApplier;

impl MetadataDiffApplier {
    /// Apply the diff to `target`, returning a new snapshot.
    pub fn apply(target: &HashMap<String, String>, diff: &MetadataDiff) -> HashMap<String, String> {
        let mut result = target.clone();

        for (key, change) in diff.changes() {
            match change {
                MetadataChange::Added(val) => {
                    result.insert(key.clone(), val.clone());
                }
                MetadataChange::Removed(_) => {
                    result.remove(key);
                }
                MetadataChange::Modified { new, .. } => {
                    result.insert(key.clone(), new.clone());
                }
            }
        }

        result
    }

    /// Apply the diff in-place to `target`.
    pub fn apply_in_place(target: &mut HashMap<String, String>, diff: &MetadataDiff) {
        for (key, change) in diff.changes() {
            match change {
                MetadataChange::Added(val) | MetadataChange::Modified { new: val, .. } => {
                    target.insert(key.clone(), val.clone());
                }
                MetadataChange::Removed(_) => {
                    target.remove(key);
                }
            }
        }
    }
}

// ========================
// Three-Way Merge
// ========================

/// Describes a conflict where two branches made different changes to the same field.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeConflict {
    /// The field key that has a conflict.
    pub key: String,
    /// The value in the base (common ancestor). `None` if the field did not exist in base.
    pub base_value: Option<String>,
    /// The value in branch A (ours). `None` if the field was removed.
    pub ours_value: Option<String>,
    /// The value in branch B (theirs). `None` if the field was removed.
    pub theirs_value: Option<String>,
    /// The type of conflict.
    pub conflict_type: ConflictType,
}

/// Types of three-way merge conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// Both branches modified the same field to different values.
    BothModified,
    /// One branch modified the field while the other removed it.
    ModifyDelete,
    /// Both branches added the same key with different values.
    BothAdded,
    /// Both branches removed the field (not really a conflict, but noted).
    BothRemoved,
}

/// Strategy for resolving three-way merge conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeWayStrategy {
    /// Prefer "ours" (branch A) in case of conflict.
    PreferOurs,
    /// Prefer "theirs" (branch B) in case of conflict.
    PreferTheirs,
    /// Keep the base value (reject both changes).
    PreferBase,
    /// Mark conflicts but do not resolve them automatically.
    MarkConflicts,
    /// Keep the longer value.
    KeepLonger,
    /// Keep the shorter value.
    KeepShorter,
}

/// Result of a three-way merge operation.
#[derive(Debug, Clone)]
pub struct ThreeWayMergeResult {
    /// The merged metadata fields.
    pub merged: HashMap<String, String>,
    /// Any conflicts that were encountered.
    pub conflicts: Vec<MergeConflict>,
    /// Keys where both branches made identical changes (clean merge).
    pub clean_merges: Vec<String>,
    /// Keys that only one branch changed (no conflict possible).
    pub one_sided: Vec<String>,
}

impl ThreeWayMergeResult {
    /// Whether the merge completed without conflicts.
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    /// Number of conflicts.
    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    /// Get a merged field value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.merged.get(key).map(|s| s.as_str())
    }

    /// Get conflicts for a specific key.
    pub fn conflict_for(&self, key: &str) -> Option<&MergeConflict> {
        self.conflicts.iter().find(|c| c.key == key)
    }
}

/// Three-way merge engine for metadata.
///
/// Given a common base version and two divergent branches (ours and theirs),
/// produces a merged result. Conflicts are detected when both branches
/// change the same field differently.
pub struct ThreeWayMerge {
    strategy: ThreeWayStrategy,
    /// Per-key strategy overrides.
    key_strategies: HashMap<String, ThreeWayStrategy>,
}

impl ThreeWayMerge {
    /// Create a new three-way merge engine with the given default strategy.
    pub fn new(strategy: ThreeWayStrategy) -> Self {
        Self {
            strategy,
            key_strategies: HashMap::new(),
        }
    }

    /// Set a per-key conflict resolution strategy.
    pub fn set_key_strategy(&mut self, key: impl Into<String>, strategy: ThreeWayStrategy) {
        self.key_strategies.insert(key.into(), strategy);
    }

    /// Get the effective strategy for a given key.
    pub fn strategy_for(&self, key: &str) -> ThreeWayStrategy {
        self.key_strategies
            .get(key)
            .copied()
            .unwrap_or(self.strategy)
    }

    /// Perform a three-way merge.
    ///
    /// - `base`: the common ancestor state
    /// - `ours`: our branch's state
    /// - `theirs`: their branch's state
    pub fn merge(
        &self,
        base: &HashMap<String, String>,
        ours: &HashMap<String, String>,
        theirs: &HashMap<String, String>,
    ) -> ThreeWayMergeResult {
        let mut result = ThreeWayMergeResult {
            merged: HashMap::new(),
            conflicts: Vec::new(),
            clean_merges: Vec::new(),
            one_sided: Vec::new(),
        };

        // Collect all keys from all three sources
        let all_keys: HashSet<&String> = base
            .keys()
            .chain(ours.keys())
            .chain(theirs.keys())
            .collect();

        for key in all_keys {
            let base_val = base.get(key);
            let ours_val = ours.get(key);
            let theirs_val = theirs.get(key);

            let ours_changed = base_val != ours_val;
            let theirs_changed = base_val != theirs_val;

            match (ours_changed, theirs_changed) {
                (false, false) => {
                    // Neither changed: keep base value (if it exists)
                    if let Some(val) = base_val {
                        result.merged.insert(key.clone(), val.clone());
                    }
                }
                (true, false) => {
                    // Only ours changed: take ours
                    result.one_sided.push(key.clone());
                    if let Some(val) = ours_val {
                        result.merged.insert(key.clone(), val.clone());
                    }
                    // If ours_val is None, the field was removed by ours
                }
                (false, true) => {
                    // Only theirs changed: take theirs
                    result.one_sided.push(key.clone());
                    if let Some(val) = theirs_val {
                        result.merged.insert(key.clone(), val.clone());
                    }
                    // If theirs_val is None, the field was removed by theirs
                }
                (true, true) => {
                    // Both changed: check if they agree
                    if ours_val == theirs_val {
                        // Both made the same change: clean merge
                        result.clean_merges.push(key.clone());
                        if let Some(val) = ours_val {
                            result.merged.insert(key.clone(), val.clone());
                        }
                    } else {
                        // Conflict!
                        let conflict_type = classify_conflict(base_val, ours_val, theirs_val);
                        let conflict = MergeConflict {
                            key: key.clone(),
                            base_value: base_val.cloned(),
                            ours_value: ours_val.cloned(),
                            theirs_value: theirs_val.cloned(),
                            conflict_type,
                        };
                        result.conflicts.push(conflict);

                        // Resolve according to strategy
                        let resolved = self.resolve_conflict(key, base_val, ours_val, theirs_val);
                        if let Some(val) = resolved {
                            result.merged.insert(key.clone(), val);
                        }
                    }
                }
            }
        }

        result
    }

    /// Resolve a single conflict according to the configured strategy.
    fn resolve_conflict(
        &self,
        key: &str,
        base_val: Option<&String>,
        ours_val: Option<&String>,
        theirs_val: Option<&String>,
    ) -> Option<String> {
        let strategy = self.strategy_for(key);
        match strategy {
            ThreeWayStrategy::PreferOurs => ours_val.cloned(),
            ThreeWayStrategy::PreferTheirs => theirs_val.cloned(),
            ThreeWayStrategy::PreferBase => base_val.cloned(),
            ThreeWayStrategy::MarkConflicts => {
                // Create a conflict marker string
                let ours_str = ours_val.map_or("<removed>", |s| s.as_str());
                let theirs_str = theirs_val.map_or("<removed>", |s| s.as_str());
                Some(format!("<<<OURS:{ours_str}===THEIRS:{theirs_str}>>>"))
            }
            ThreeWayStrategy::KeepLonger => {
                let ours_len = ours_val.map_or(0, |s| s.len());
                let theirs_len = theirs_val.map_or(0, |s| s.len());
                if ours_len >= theirs_len {
                    ours_val.cloned()
                } else {
                    theirs_val.cloned()
                }
            }
            ThreeWayStrategy::KeepShorter => match (ours_val, theirs_val) {
                (Some(o), Some(t)) => {
                    if o.len() <= t.len() {
                        Some(o.clone())
                    } else {
                        Some(t.clone())
                    }
                }
                (Some(o), None) => Some(o.clone()),
                (None, Some(t)) => Some(t.clone()),
                (None, None) => None,
            },
        }
    }
}

impl Default for ThreeWayMerge {
    fn default() -> Self {
        Self::new(ThreeWayStrategy::PreferOurs)
    }
}

/// Classify the type of conflict between two branches.
fn classify_conflict(
    base: Option<&String>,
    ours: Option<&String>,
    theirs: Option<&String>,
) -> ConflictType {
    match (base, ours, theirs) {
        (Some(_), Some(_), Some(_)) => ConflictType::BothModified,
        (Some(_), None, None) => ConflictType::BothRemoved,
        (Some(_), Some(_), None) | (Some(_), None, Some(_)) => ConflictType::ModifyDelete,
        (None, Some(_), Some(_)) => ConflictType::BothAdded,
        // Edge cases
        (None, None, Some(_)) | (None, Some(_), None) | (None, None, None) => {
            ConflictType::BothModified
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ---- Two-way diff tests ----

    #[test]
    fn test_change_added_not_destructive() {
        let c = MetadataChange::Added("val".to_string());
        assert!(!c.is_destructive());
    }

    #[test]
    fn test_change_removed_is_destructive() {
        let c = MetadataChange::Removed("val".to_string());
        assert!(c.is_destructive());
    }

    #[test]
    fn test_change_modified_is_destructive() {
        let c = MetadataChange::Modified {
            old: "a".to_string(),
            new: "b".to_string(),
        };
        assert!(c.is_destructive());
    }

    #[test]
    fn test_change_new_value() {
        let c = MetadataChange::Added("hello".to_string());
        assert_eq!(c.new_value(), Some("hello"));
        let r = MetadataChange::Removed("old".to_string());
        assert_eq!(r.new_value(), None);
    }

    #[test]
    fn test_change_old_value() {
        let c = MetadataChange::Modified {
            old: "prev".to_string(),
            new: "next".to_string(),
        };
        assert_eq!(c.old_value(), Some("prev"));
        let a = MetadataChange::Added("x".to_string());
        assert_eq!(a.old_value(), None);
    }

    #[test]
    fn test_diff_compute_added() {
        let before = map(&[]);
        let after = map(&[("title", "Song")]);
        let diff = MetadataDiff::compute(&before, &after);
        assert_eq!(diff.added_fields(), vec!["title"]);
        assert!(diff.removed_fields().is_empty());
        assert!(diff.changed_fields().is_empty());
    }

    #[test]
    fn test_diff_compute_removed() {
        let before = map(&[("title", "Song")]);
        let after = map(&[]);
        let diff = MetadataDiff::compute(&before, &after);
        assert_eq!(diff.removed_fields(), vec!["title"]);
        assert!(diff.added_fields().is_empty());
    }

    #[test]
    fn test_diff_compute_modified() {
        let before = map(&[("title", "Old")]);
        let after = map(&[("title", "New")]);
        let diff = MetadataDiff::compute(&before, &after);
        assert_eq!(diff.changed_fields(), vec!["title"]);
    }

    #[test]
    fn test_diff_compute_no_change() {
        let before = map(&[("title", "Same")]);
        let after = map(&[("title", "Same")]);
        let diff = MetadataDiff::compute(&before, &after);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_len() {
        let before = map(&[("a", "1"), ("b", "2")]);
        let after = map(&[("b", "3"), ("c", "4")]);
        let diff = MetadataDiff::compute(&before, &after);
        assert_eq!(diff.len(), 3); // a removed, b modified, c added
    }

    #[test]
    fn test_diff_has_destructive_changes() {
        let before = map(&[("a", "1")]);
        let after = map(&[]);
        let diff = MetadataDiff::compute(&before, &after);
        assert!(diff.has_destructive_changes());
    }

    #[test]
    fn test_applier_apply() {
        let before = map(&[("title", "Old"), ("artist", "A")]);
        let after = map(&[("title", "New"), ("album", "X")]);
        let diff = MetadataDiff::compute(&before, &after);
        let result = MetadataDiffApplier::apply(&before, &diff);
        assert_eq!(result.get("title").map(String::as_str), Some("New"));
        assert_eq!(result.get("album").map(String::as_str), Some("X"));
        assert!(!result.contains_key("artist"));
    }

    #[test]
    fn test_applier_apply_in_place() {
        let mut target = map(&[("x", "1")]);
        let before = target.clone();
        let after = map(&[("x", "2"), ("y", "3")]);
        let diff = MetadataDiff::compute(&before, &after);
        MetadataDiffApplier::apply_in_place(&mut target, &diff);
        assert_eq!(target.get("x").map(String::as_str), Some("2"));
        assert_eq!(target.get("y").map(String::as_str), Some("3"));
    }

    #[test]
    fn test_diff_default_is_empty() {
        let diff = MetadataDiff::default();
        assert!(diff.is_empty());
        assert_eq!(diff.len(), 0);
    }

    // ---- Three-way merge tests ----

    #[test]
    fn test_three_way_no_changes() {
        let base = map(&[("title", "Song"), ("artist", "Band")]);
        let ours = base.clone();
        let theirs = base.clone();

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.conflict_count(), 0);
        assert_eq!(result.get("title"), Some("Song"));
        assert_eq!(result.get("artist"), Some("Band"));
    }

    #[test]
    fn test_three_way_one_sided_ours() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "New Song")]);
        let theirs = base.clone();

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("New Song"));
        assert!(result.one_sided.contains(&"title".to_string()));
    }

    #[test]
    fn test_three_way_one_sided_theirs() {
        let base = map(&[("title", "Song")]);
        let ours = base.clone();
        let theirs = map(&[("title", "Their Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("Their Song"));
        assert!(result.one_sided.contains(&"title".to_string()));
    }

    #[test]
    fn test_three_way_both_same_change() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "New Song")]);
        let theirs = map(&[("title", "New Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("New Song"));
        assert!(result.clean_merges.contains(&"title".to_string()));
    }

    #[test]
    fn test_three_way_conflict_both_modified() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "Our Song")]);
        let theirs = map(&[("title", "Their Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(!result.is_clean());
        assert_eq!(result.conflict_count(), 1);
        let conflict = result.conflict_for("title").expect("should have conflict");
        assert_eq!(conflict.conflict_type, ConflictType::BothModified);
        assert_eq!(conflict.base_value.as_deref(), Some("Song"));
        assert_eq!(conflict.ours_value.as_deref(), Some("Our Song"));
        assert_eq!(conflict.theirs_value.as_deref(), Some("Their Song"));
        // PreferOurs resolves to our value
        assert_eq!(result.get("title"), Some("Our Song"));
    }

    #[test]
    fn test_three_way_conflict_prefer_theirs() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "Our Song")]);
        let theirs = map(&[("title", "Their Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferTheirs);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.get("title"), Some("Their Song"));
    }

    #[test]
    fn test_three_way_conflict_prefer_base() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "Our Song")]);
        let theirs = map(&[("title", "Their Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferBase);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.get("title"), Some("Song"));
    }

    #[test]
    fn test_three_way_conflict_mark_conflicts() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "Our Song")]);
        let theirs = map(&[("title", "Their Song")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::MarkConflicts);
        let result = merger.merge(&base, &ours, &theirs);

        let merged_title = result.get("title").expect("should have title");
        assert!(merged_title.contains("<<<OURS:"));
        assert!(merged_title.contains("===THEIRS:"));
        assert!(merged_title.contains(">>>"));
    }

    #[test]
    fn test_three_way_modify_delete_conflict() {
        let base = map(&[("title", "Song"), ("artist", "Band")]);
        let ours = map(&[("title", "New Song")]); // removed "artist"
        let theirs = map(&[("title", "Song"), ("artist", "New Band")]); // modified "artist"

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        // "artist" is a conflict: ours removed, theirs modified
        assert!(!result.is_clean());
        let conflict = result.conflict_for("artist").expect("should have conflict");
        assert_eq!(conflict.conflict_type, ConflictType::ModifyDelete);
        assert_eq!(conflict.ours_value, None);
        assert_eq!(conflict.theirs_value.as_deref(), Some("New Band"));
        // PreferOurs: removed (not in merged)
        assert!(result.get("artist").is_none());
    }

    #[test]
    fn test_three_way_both_added_same_key() {
        let base = map(&[]);
        let ours = map(&[("genre", "Rock")]);
        let theirs = map(&[("genre", "Pop")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(!result.is_clean());
        let conflict = result.conflict_for("genre").expect("should have conflict");
        assert_eq!(conflict.conflict_type, ConflictType::BothAdded);
        assert_eq!(result.get("genre"), Some("Rock"));
    }

    #[test]
    fn test_three_way_both_removed() {
        let base = map(&[("title", "Song"), ("comment", "Old")]);
        let ours = map(&[("title", "Song")]); // removed "comment"
        let theirs = map(&[("title", "Song")]); // also removed "comment"

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        // Both removed same field: clean merge (field stays removed)
        assert!(result.get("comment").is_none());
        // This is a clean merge since both branches agree
        assert!(result.clean_merges.contains(&"comment".to_string()));
    }

    #[test]
    fn test_three_way_disjoint_additions() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "Song"), ("artist", "Band")]);
        let theirs = map(&[("title", "Song"), ("genre", "Rock")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("Song"));
        assert_eq!(result.get("artist"), Some("Band"));
        assert_eq!(result.get("genre"), Some("Rock"));
    }

    #[test]
    fn test_three_way_per_key_strategy() {
        let base = map(&[("title", "Song"), ("artist", "Band")]);
        let ours = map(&[("title", "Our Title"), ("artist", "Our Artist")]);
        let theirs = map(&[("title", "Their Title"), ("artist", "Their Artist")]);

        let mut merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        merger.set_key_strategy("artist", ThreeWayStrategy::PreferTheirs);

        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.get("title"), Some("Our Title"));
        assert_eq!(result.get("artist"), Some("Their Artist"));
    }

    #[test]
    fn test_three_way_keep_longer_strategy() {
        let base = map(&[("title", "X")]);
        let ours = map(&[("title", "Short")]);
        let theirs = map(&[("title", "Much Longer Title")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::KeepLonger);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.get("title"), Some("Much Longer Title"));
    }

    #[test]
    fn test_three_way_keep_shorter_strategy() {
        let base = map(&[("title", "X")]);
        let ours = map(&[("title", "Short")]);
        let theirs = map(&[("title", "Much Longer Title")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::KeepShorter);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.get("title"), Some("Short"));
    }

    #[test]
    fn test_three_way_complex_scenario() {
        let base = map(&[
            ("title", "Song"),
            ("artist", "Band"),
            ("year", "2020"),
            ("genre", "Rock"),
            ("comment", "old comment"),
        ]);

        let ours = map(&[
            ("title", "Song"),      // unchanged
            ("artist", "New Band"), // modified
            ("year", "2021"),       // modified
            // "genre" removed
            ("comment", "our comment"), // modified
            ("bpm", "120"),             // added
        ]);

        let theirs = map(&[
            ("title", "New Song"),        // modified
            ("artist", "Their Band"),     // modified (conflict with ours)
            ("year", "2020"),             // unchanged
            ("genre", "Pop"),             // modified
            ("comment", "their comment"), // modified (conflict with ours)
            ("key", "Cmaj"),              // added
        ]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        // title: only theirs changed -> "New Song"
        assert_eq!(result.get("title"), Some("New Song"));
        // artist: conflict -> prefer ours -> "New Band"
        assert_eq!(result.get("artist"), Some("New Band"));
        // year: only ours changed -> "2021"
        assert_eq!(result.get("year"), Some("2021"));
        // genre: ours removed, theirs modified -> modify-delete conflict -> prefer ours (removed)
        assert!(result.get("genre").is_none());
        // comment: both modified differently -> conflict -> prefer ours
        assert_eq!(result.get("comment"), Some("our comment"));
        // bpm: only ours added -> "120"
        assert_eq!(result.get("bpm"), Some("120"));
        // key: only theirs added -> "Cmaj"
        assert_eq!(result.get("key"), Some("Cmaj"));

        // Should have 3 conflicts: artist, genre, comment
        assert_eq!(result.conflict_count(), 3);
    }

    #[test]
    fn test_three_way_merge_default() {
        let merger = ThreeWayMerge::default();
        assert_eq!(merger.strategy, ThreeWayStrategy::PreferOurs);
    }

    #[test]
    fn test_three_way_empty_base() {
        let base = map(&[]);
        let ours = map(&[("title", "A")]);
        let theirs = map(&[("artist", "B")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.get("title"), Some("A"));
        assert_eq!(result.get("artist"), Some("B"));
    }

    #[test]
    fn test_three_way_all_empty() {
        let base = map(&[]);
        let ours = map(&[]);
        let theirs = map(&[]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert!(result.merged.is_empty());
    }

    #[test]
    fn test_conflict_type_classification() {
        let s = "val".to_string();
        assert_eq!(
            classify_conflict(Some(&s), Some(&"a".to_string()), Some(&"b".to_string())),
            ConflictType::BothModified
        );
        assert_eq!(
            classify_conflict(Some(&s), None, None),
            ConflictType::BothRemoved
        );
        assert_eq!(
            classify_conflict(Some(&s), Some(&"a".to_string()), None),
            ConflictType::ModifyDelete
        );
        assert_eq!(
            classify_conflict(Some(&s), None, Some(&"b".to_string())),
            ConflictType::ModifyDelete
        );
        assert_eq!(
            classify_conflict(None, Some(&"a".to_string()), Some(&"b".to_string())),
            ConflictType::BothAdded
        );
    }

    #[test]
    fn test_three_way_merge_result_helpers() {
        let base = map(&[("x", "1")]);
        let ours = map(&[("x", "2")]);
        let theirs = map(&[("x", "3")]);

        let merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(!result.is_clean());
        assert_eq!(result.conflict_count(), 1);
        assert!(result.conflict_for("x").is_some());
        assert!(result.conflict_for("y").is_none());
    }

    #[test]
    fn test_three_way_strategy_for_key() {
        let mut merger = ThreeWayMerge::new(ThreeWayStrategy::PreferOurs);
        merger.set_key_strategy("title", ThreeWayStrategy::PreferTheirs);

        assert_eq!(merger.strategy_for("title"), ThreeWayStrategy::PreferTheirs);
        assert_eq!(merger.strategy_for("artist"), ThreeWayStrategy::PreferOurs);
    }

    #[test]
    fn test_three_way_mark_conflicts_modify_delete() {
        let base = map(&[("title", "Song")]);
        let ours = map(&[("title", "New Song")]);
        let theirs = map(&[]); // removed

        let merger = ThreeWayMerge::new(ThreeWayStrategy::MarkConflicts);
        let result = merger.merge(&base, &ours, &theirs);

        let val = result.get("title").expect("should have title");
        assert!(val.contains("New Song"));
        assert!(val.contains("<removed>"));
    }

    #[test]
    fn test_three_way_keep_shorter_both_removed() {
        let base = map(&[("x", "val")]);
        let ours = map(&[("x", "a")]);
        let theirs = map(&[]); // removed

        let merger = ThreeWayMerge::new(ThreeWayStrategy::KeepShorter);
        let result = merger.merge(&base, &ours, &theirs);

        // ours="a", theirs=None -> KeepShorter picks ours since theirs is None
        assert_eq!(result.get("x"), Some("a"));
    }
}
