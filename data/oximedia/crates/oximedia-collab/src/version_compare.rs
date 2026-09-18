//! Compare two versions of a project edit and report structural differences.

#![allow(dead_code)]

/// The kind of difference between two edit versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffKind {
    /// A clip present in version B but not in version A.
    Added,
    /// A clip present in version A but not in version B.
    Removed,
    /// A clip present in both versions but with changed in/out points.
    Modified,
    /// A clip whose position in the timeline changed (same id, same in/out).
    Moved,
}

impl DiffKind {
    /// Returns `true` for diff kinds that affect the structural edit (added, removed, moved).
    pub fn is_structural(&self) -> bool {
        matches!(self, DiffKind::Added | DiffKind::Removed | DiffKind::Moved)
    }
}

/// Describes the difference for a single clip between two edit versions.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipDiff {
    /// The clip identifier.
    pub clip_id: u64,
    /// Nature of the difference.
    pub kind: DiffKind,
    /// Human-readable description of what changed.
    pub description: String,
}

impl ClipDiff {
    /// Returns `true` when this diff represents an added clip.
    pub fn is_addition(&self) -> bool {
        matches!(self.kind, DiffKind::Added)
    }
}

/// Aggregated diff between two named edit versions.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionDiff {
    /// Label for the first (reference) version.
    pub version_a: String,
    /// Label for the second (comparison) version.
    pub version_b: String,
    /// All clip-level differences.
    pub diffs: Vec<ClipDiff>,
}

impl VersionDiff {
    /// Number of added clips.
    pub fn added_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| matches!(d.kind, DiffKind::Added))
            .count()
    }

    /// Number of removed clips.
    pub fn removed_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| matches!(d.kind, DiffKind::Removed))
            .count()
    }

    /// Returns `true` when there is at least one difference.
    pub fn has_changes(&self) -> bool {
        !self.diffs.is_empty()
    }

    /// Total number of individual clip differences.
    pub fn change_count(&self) -> usize {
        self.diffs.len()
    }
}

/// Compare two clip lists and produce a `VersionDiff`.
///
/// Each clip is a tuple `(id, in_point, out_point)`.
///
/// Rules applied (in order):
/// 1. Clips whose id is in B but not A → `Added`.
/// 2. Clips whose id is in A but not B → `Removed`.
/// 3. Clips present in both:
///    - If `in_point` or `out_point` differ → `Modified`.
///    - Otherwise (same id and same in/out but conceptually reordered) → ignored (no diff).
pub fn compare_clip_lists(clips_a: &[(u64, u64, u64)], clips_b: &[(u64, u64, u64)]) -> VersionDiff {
    use std::collections::HashMap;

    let map_a: HashMap<u64, (u64, u64)> = clips_a.iter().map(|&(id, i, o)| (id, (i, o))).collect();
    let map_b: HashMap<u64, (u64, u64)> = clips_b.iter().map(|&(id, i, o)| (id, (i, o))).collect();

    let mut diffs: Vec<ClipDiff> = Vec::new();

    // Added (in B, not in A)
    for (&id, &(ib, ob)) in &map_b {
        if !map_a.contains_key(&id) {
            diffs.push(ClipDiff {
                clip_id: id,
                kind: DiffKind::Added,
                description: format!("clip {id} added (in={ib}, out={ob})"),
            });
        }
    }

    // Removed (in A, not in B)
    for (&id, &(ia, oa)) in &map_a {
        if !map_b.contains_key(&id) {
            diffs.push(ClipDiff {
                clip_id: id,
                kind: DiffKind::Removed,
                description: format!("clip {id} removed (was in={ia}, out={oa})"),
            });
        }
    }

    // Modified (in both but different points)
    for (&id, &(ia, oa)) in &map_a {
        if let Some(&(ib, ob)) = map_b.get(&id) {
            if ia != ib || oa != ob {
                diffs.push(ClipDiff {
                    clip_id: id,
                    kind: DiffKind::Modified,
                    description: format!("clip {id} modified: ({ia}, {oa}) → ({ib}, {ob})"),
                });
            }
        }
    }

    // Sort for deterministic output
    diffs.sort_by_key(|d| (d.clip_id, format!("{:?}", d.kind)));

    VersionDiff {
        version_a: "A".to_string(),
        version_b: "B".to_string(),
        diffs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- DiffKind tests ----

    #[test]
    fn test_added_is_structural() {
        assert!(DiffKind::Added.is_structural());
    }

    #[test]
    fn test_removed_is_structural() {
        assert!(DiffKind::Removed.is_structural());
    }

    #[test]
    fn test_moved_is_structural() {
        assert!(DiffKind::Moved.is_structural());
    }

    #[test]
    fn test_modified_not_structural() {
        assert!(!DiffKind::Modified.is_structural());
    }

    // ---- ClipDiff tests ----

    #[test]
    fn test_is_addition_true() {
        let d = ClipDiff {
            clip_id: 1,
            kind: DiffKind::Added,
            description: String::new(),
        };
        assert!(d.is_addition());
    }

    #[test]
    fn test_is_addition_false_for_removed() {
        let d = ClipDiff {
            clip_id: 1,
            kind: DiffKind::Removed,
            description: String::new(),
        };
        assert!(!d.is_addition());
    }

    // ---- VersionDiff tests ----

    #[test]
    fn test_no_changes_empty_lists() {
        let vd = compare_clip_lists(&[], &[]);
        assert!(!vd.has_changes());
        assert_eq!(vd.change_count(), 0);
    }

    #[test]
    fn test_identical_lists_no_diff() {
        let clips = [(1, 0, 100), (2, 100, 200)];
        let vd = compare_clip_lists(&clips, &clips);
        assert!(!vd.has_changes());
    }

    #[test]
    fn test_added_clip_detected() {
        let a = [(1u64, 0u64, 100u64)];
        let b = [(1, 0, 100), (2, 200, 300)];
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.added_count(), 1);
        assert_eq!(vd.removed_count(), 0);
    }

    #[test]
    fn test_removed_clip_detected() {
        let a = [(1u64, 0u64, 100u64), (2, 200, 300)];
        let b = [(1, 0, 100)];
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.removed_count(), 1);
        assert_eq!(vd.added_count(), 0);
    }

    #[test]
    fn test_modified_clip_detected() {
        let a = [(1u64, 0u64, 100u64)];
        let b = [(1, 0, 90)]; // out_point changed
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.change_count(), 1);
        assert!(matches!(vd.diffs[0].kind, DiffKind::Modified));
    }

    #[test]
    fn test_unmodified_clip_produces_no_diff() {
        let clips = [(1u64, 0u64, 100u64)];
        let vd = compare_clip_lists(&clips, &clips);
        assert_eq!(vd.change_count(), 0);
    }

    #[test]
    fn test_has_changes_true() {
        let a: [(u64, u64, u64); 0] = [];
        let b = [(1u64, 0u64, 50u64)];
        let vd = compare_clip_lists(&a, &b);
        assert!(vd.has_changes());
    }

    #[test]
    fn test_added_count_multiple() {
        let a: [(u64, u64, u64); 0] = [];
        let b = [(1u64, 0u64, 50u64), (2, 50, 100), (3, 100, 150)];
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.added_count(), 3);
    }

    #[test]
    fn test_removed_count_multiple() {
        let a = [(1u64, 0u64, 50u64), (2, 50, 100), (3, 100, 150)];
        let b: [(u64, u64, u64); 0] = [];
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.removed_count(), 3);
    }

    #[test]
    fn test_combined_add_remove_modify() {
        let a = [(1u64, 0u64, 100u64), (2, 100, 200)];
        let b = [(1, 0, 90), (3, 300, 400)]; // 1 modified, 2 removed, 3 added
        let vd = compare_clip_lists(&a, &b);
        assert_eq!(vd.added_count(), 1);
        assert_eq!(vd.removed_count(), 1);
        // 1 modified entry
        assert!(vd
            .diffs
            .iter()
            .any(|d| matches!(d.kind, DiffKind::Modified)));
        assert_eq!(vd.change_count(), 3);
    }
}
