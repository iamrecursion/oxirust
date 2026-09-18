//! Merge strategies for collaborative document edits.
//!
//! Provides operational transformation helpers, three-way merge,
//! and configurable merge policies for timeline edit sessions.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A property value that can be merged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PropValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
}

impl PropValue {
    /// Returns true if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, PropValue::Null)
    }
}

/// A document state: a flat map of property -> value.
pub type Document = HashMap<String, PropValue>;

/// A diff between two document states.
#[derive(Debug, Clone, Default)]
pub struct DocumentDiff {
    /// Properties added or changed.
    pub modified: HashMap<String, PropValue>,
    /// Properties removed.
    pub removed: Vec<String>,
}

impl DocumentDiff {
    /// Compute a diff from base to target.
    pub fn compute(base: &Document, target: &Document) -> Self {
        let mut modified = HashMap::new();
        let mut removed = Vec::new();

        // Find added/changed
        for (key, val) in target {
            if base.get(key) != Some(val) {
                modified.insert(key.clone(), val.clone());
            }
        }

        // Find removed
        for key in base.keys() {
            if !target.contains_key(key) {
                removed.push(key.clone());
            }
        }

        Self { modified, removed }
    }

    /// Returns true if no changes.
    pub fn is_empty(&self) -> bool {
        self.modified.is_empty() && self.removed.is_empty()
    }

    /// Apply this diff to a base document.
    pub fn apply(&self, base: &Document) -> Document {
        let mut result = base.clone();
        for (k, v) in &self.modified {
            result.insert(k.clone(), v.clone());
        }
        for k in &self.removed {
            result.remove(k);
        }
        result
    }
}

/// Conflict type in a three-way merge.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeConflict {
    /// Both sides modified the same property with different values.
    BothModified {
        key: String,
        ours: PropValue,
        theirs: PropValue,
    },
    /// Both sides deleted the same property.
    BothDeleted { key: String },
    /// One side deleted while the other modified.
    DeleteModify {
        key: String,
        modified: PropValue,
        deleted_by_ours: bool,
    },
}

/// Result of a three-way merge.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Merged document (may be partial if there are conflicts).
    pub document: Document,
    /// Conflicts that could not be automatically resolved.
    pub conflicts: Vec<MergeConflict>,
}

impl MergeResult {
    /// Returns true if the merge completed without conflicts.
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }
}

/// Merge policy for resolving conflicts automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePolicy {
    /// Prefer the "ours" side when both sides conflict.
    PreferOurs,
    /// Prefer the "theirs" side when both sides conflict.
    PreferTheirs,
    /// Keep conflicting properties as they are in the base.
    KeepBase,
    /// Mark conflicts and leave them for manual resolution.
    ManualReview,
}

/// Three-way merge engine.
#[derive(Debug)]
pub struct ThreeWayMerger {
    policy: MergePolicy,
}

impl ThreeWayMerger {
    /// Create a new merger with the specified policy.
    pub fn new(policy: MergePolicy) -> Self {
        Self { policy }
    }

    /// Perform a three-way merge.
    ///
    /// - `base`: the common ancestor document
    /// - `ours`: our modified document
    /// - `theirs`: their modified document
    pub fn merge(&self, base: &Document, ours: &Document, theirs: &Document) -> MergeResult {
        let diff_ours = DocumentDiff::compute(base, ours);
        let diff_theirs = DocumentDiff::compute(base, theirs);

        let mut result = base.clone();
        let mut conflicts = Vec::new();

        // Apply non-conflicting ours changes
        for (key, val) in &diff_ours.modified {
            if !diff_theirs.modified.contains_key(key) && !diff_theirs.removed.contains(key) {
                result.insert(key.clone(), val.clone());
            }
        }
        for key in &diff_ours.removed {
            if !diff_theirs.modified.contains_key(key) {
                result.remove(key);
            }
        }

        // Apply non-conflicting theirs changes
        for (key, val) in &diff_theirs.modified {
            if !diff_ours.modified.contains_key(key) && !diff_ours.removed.contains(key) {
                result.insert(key.clone(), val.clone());
            }
        }
        for key in &diff_theirs.removed {
            if !diff_ours.modified.contains_key(key) {
                result.remove(key);
            }
        }

        // Handle both-modified-same-value (not a conflict, just apply)
        for (key, our_val) in &diff_ours.modified {
            if let Some(their_val) = diff_theirs.modified.get(key) {
                if our_val == their_val {
                    // Both sides made the same change — apply it
                    result.insert(key.clone(), our_val.clone());
                    continue;
                }
            }
        }

        // Detect and handle conflicts
        for (key, our_val) in &diff_ours.modified {
            if let Some(their_val) = diff_theirs.modified.get(key) {
                if our_val == their_val {
                    // Already handled above
                    continue;
                }
                if our_val != their_val {
                    let conflict = MergeConflict::BothModified {
                        key: key.clone(),
                        ours: our_val.clone(),
                        theirs: their_val.clone(),
                    };
                    match self.policy {
                        MergePolicy::PreferOurs => {
                            result.insert(key.clone(), our_val.clone());
                        }
                        MergePolicy::PreferTheirs => {
                            result.insert(key.clone(), their_val.clone());
                        }
                        MergePolicy::KeepBase => {
                            if let Some(base_val) = base.get(key) {
                                result.insert(key.clone(), base_val.clone());
                            }
                        }
                        MergePolicy::ManualReview => {
                            // Remove from result to mark as unresolved
                            result.remove(key);
                            conflicts.push(conflict);
                            continue;
                        }
                    }
                    if !matches!(self.policy, MergePolicy::ManualReview) {
                        conflicts.push(conflict);
                    }
                }
            }

            if diff_theirs.removed.contains(key) {
                let conflict = MergeConflict::DeleteModify {
                    key: key.clone(),
                    modified: our_val.clone(),
                    deleted_by_ours: false,
                };
                match self.policy {
                    MergePolicy::PreferOurs => {
                        result.insert(key.clone(), our_val.clone());
                    }
                    MergePolicy::PreferTheirs | MergePolicy::KeepBase => {
                        result.remove(key);
                    }
                    MergePolicy::ManualReview => {
                        result.remove(key);
                        conflicts.push(conflict);
                        continue;
                    }
                }
                if !matches!(self.policy, MergePolicy::ManualReview) {
                    conflicts.push(conflict);
                }
            }
        }

        // Both-deleted conflicts
        for key in &diff_ours.removed {
            if diff_theirs.removed.contains(key) && base.contains_key(key) {
                conflicts.push(MergeConflict::BothDeleted { key: key.clone() });
            }
        }

        MergeResult {
            document: result,
            conflicts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(pairs: &[(&str, PropValue)]) -> Document {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_diff_compute_added() {
        let base = doc(&[("a", PropValue::Number(1.0))]);
        let target = doc(&[("a", PropValue::Number(1.0)), ("b", PropValue::Bool(true))]);
        let diff = DocumentDiff::compute(&base, &target);
        assert!(diff.modified.contains_key("b"));
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn test_diff_compute_removed() {
        let base = doc(&[("a", PropValue::Number(1.0)), ("b", PropValue::Bool(true))]);
        let target = doc(&[("a", PropValue::Number(1.0))]);
        let diff = DocumentDiff::compute(&base, &target);
        assert!(diff.removed.contains(&"b".to_string()));
    }

    #[test]
    fn test_diff_compute_changed() {
        let base = doc(&[("a", PropValue::Number(1.0))]);
        let target = doc(&[("a", PropValue::Number(2.0))]);
        let diff = DocumentDiff::compute(&base, &target);
        assert_eq!(diff.modified.get("a"), Some(&PropValue::Number(2.0)));
    }

    #[test]
    fn test_diff_is_empty_for_identical_docs() {
        let base = doc(&[("a", PropValue::String("x".to_string()))]);
        let diff = DocumentDiff::compute(&base, &base);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_apply_roundtrip() {
        let base = doc(&[("a", PropValue::Number(1.0))]);
        let target = doc(&[("b", PropValue::Bool(true))]);
        let diff = DocumentDiff::compute(&base, &target);
        let applied = diff.apply(&base);
        assert_eq!(applied, target);
    }

    #[test]
    fn test_three_way_merge_clean() {
        let base = doc(&[("a", PropValue::Number(1.0)), ("b", PropValue::Bool(true))]);
        let ours = doc(&[("a", PropValue::Number(2.0)), ("b", PropValue::Bool(true))]);
        let theirs = doc(&[("a", PropValue::Number(1.0)), ("b", PropValue::Bool(false))]);

        let merger = ThreeWayMerger::new(MergePolicy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert_eq!(result.document.get("a"), Some(&PropValue::Number(2.0)));
        assert_eq!(result.document.get("b"), Some(&PropValue::Bool(false)));
    }

    #[test]
    fn test_three_way_merge_prefer_ours_on_conflict() {
        let base = doc(&[("x", PropValue::Number(0.0))]);
        let ours = doc(&[("x", PropValue::Number(1.0))]);
        let theirs = doc(&[("x", PropValue::Number(2.0))]);

        let merger = ThreeWayMerger::new(MergePolicy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.document.get("x"), Some(&PropValue::Number(1.0)));
    }

    #[test]
    fn test_three_way_merge_prefer_theirs_on_conflict() {
        let base = doc(&[("x", PropValue::Number(0.0))]);
        let ours = doc(&[("x", PropValue::Number(1.0))]);
        let theirs = doc(&[("x", PropValue::Number(2.0))]);

        let merger = ThreeWayMerger::new(MergePolicy::PreferTheirs);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.document.get("x"), Some(&PropValue::Number(2.0)));
    }

    #[test]
    fn test_three_way_merge_keep_base_on_conflict() {
        let base = doc(&[("x", PropValue::Number(0.0))]);
        let ours = doc(&[("x", PropValue::Number(1.0))]);
        let theirs = doc(&[("x", PropValue::Number(2.0))]);

        let merger = ThreeWayMerger::new(MergePolicy::KeepBase);
        let result = merger.merge(&base, &ours, &theirs);

        assert_eq!(result.document.get("x"), Some(&PropValue::Number(0.0)));
    }

    #[test]
    fn test_three_way_merge_manual_review_marks_conflict() {
        let base = doc(&[("x", PropValue::Number(0.0))]);
        let ours = doc(&[("x", PropValue::Number(1.0))]);
        let theirs = doc(&[("x", PropValue::Number(2.0))]);

        let merger = ThreeWayMerger::new(MergePolicy::ManualReview);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(!result.is_clean());
        assert!(!result.conflicts.is_empty());
    }

    #[test]
    fn test_prop_value_is_null() {
        assert!(PropValue::Null.is_null());
        assert!(!PropValue::Number(1.0).is_null());
    }

    #[test]
    fn test_merge_both_same_change_no_conflict() {
        let base = doc(&[("x", PropValue::Number(0.0))]);
        let ours = doc(&[("x", PropValue::Number(5.0))]);
        let theirs = doc(&[("x", PropValue::Number(5.0))]);

        let merger = ThreeWayMerger::new(MergePolicy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        // Same value on both sides, no real conflict
        assert_eq!(result.document.get("x"), Some(&PropValue::Number(5.0)));
    }

    #[test]
    fn test_merge_adds_from_both_sides() {
        let base: Document = HashMap::new();
        let ours = doc(&[("a", PropValue::Number(1.0))]);
        let theirs = doc(&[("b", PropValue::Bool(true))]);

        let merger = ThreeWayMerger::new(MergePolicy::PreferOurs);
        let result = merger.merge(&base, &ours, &theirs);

        assert!(result.is_clean());
        assert!(result.document.contains_key("a"));
        assert!(result.document.contains_key("b"));
    }
}
