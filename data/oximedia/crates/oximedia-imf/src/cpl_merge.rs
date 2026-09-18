#![allow(dead_code)]
//! CPL merging and conflict resolution for IMF packages.
//!
//! When building complex IMF deliveries, it is common to merge multiple
//! Composition Playlists (CPLs) together -- for example, combining a
//! supplemental package with a base package, or merging locale-specific
//! audio/subtitle tracks into a single CPL.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for a CPL or segment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CplId(pub String);

impl CplId {
    /// Creates a new CPL identifier.
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }

    /// Returns the identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CplId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strategy for resolving conflicts when merging CPLs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConflictStrategy {
    /// Keep the version from the first (base) CPL.
    KeepBase,
    /// Keep the version from the second (supplemental) CPL.
    KeepSupplemental,
    /// Fail on any conflict.
    Fail,
    /// Concatenate conflicting segments in order.
    Concatenate,
}

/// Describes a conflict found during CPL merge.
#[derive(Clone, Debug)]
pub struct MergeConflict {
    /// The segment or resource identifier where the conflict occurred.
    pub resource_id: String,
    /// Description of the conflict.
    pub description: String,
    /// Severity level.
    pub severity: ConflictSeverity,
}

impl fmt::Display for MergeConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] {}: {}",
            self.severity, self.resource_id, self.description
        )
    }
}

/// Severity of a merge conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConflictSeverity {
    /// Informational only; merge can proceed.
    Info,
    /// Warning; merge can proceed but result may be unexpected.
    Warning,
    /// Error; merge cannot proceed without resolution.
    Error,
}

/// Represents a virtual segment in a CPL for merging purposes.
#[derive(Clone, Debug)]
pub struct MergeSegment {
    /// Unique segment identifier.
    pub id: String,
    /// Sequence number within the CPL.
    pub sequence_number: u32,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Duration in edit units.
    pub duration: u64,
    /// Track resource IDs contained in this segment.
    pub resource_ids: Vec<String>,
}

impl MergeSegment {
    /// Creates a new merge segment.
    pub fn new(id: &str, seq: u32, rate_num: u32, rate_den: u32, duration: u64) -> Self {
        Self {
            id: id.to_string(),
            sequence_number: seq,
            edit_rate_num: rate_num,
            edit_rate_den: rate_den,
            duration,
            resource_ids: Vec::new(),
        }
    }

    /// Adds a resource ID to this segment.
    pub fn add_resource(&mut self, resource_id: &str) {
        self.resource_ids.push(resource_id.to_string());
    }

    /// Returns the duration in seconds as f64.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        if self.edit_rate_num == 0 {
            return 0.0;
        }
        self.duration as f64 * self.edit_rate_den as f64 / self.edit_rate_num as f64
    }
}

/// A virtual CPL structure used for merge operations.
#[derive(Clone, Debug)]
pub struct MergeCpl {
    /// CPL identifier.
    pub id: CplId,
    /// Content title.
    pub title: String,
    /// Edit rate numerator.
    pub edit_rate_num: u32,
    /// Edit rate denominator.
    pub edit_rate_den: u32,
    /// Segments in this CPL.
    pub segments: Vec<MergeSegment>,
}

impl MergeCpl {
    /// Creates a new merge CPL.
    pub fn new(id: &str, title: &str, rate_num: u32, rate_den: u32) -> Self {
        Self {
            id: CplId::new(id),
            title: title.to_string(),
            edit_rate_num: rate_num,
            edit_rate_den: rate_den,
            segments: Vec::new(),
        }
    }

    /// Adds a segment to this CPL.
    pub fn add_segment(&mut self, segment: MergeSegment) {
        self.segments.push(segment);
    }

    /// Returns the total duration in edit units.
    pub fn total_duration(&self) -> u64 {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Returns the total duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_duration_seconds(&self) -> f64 {
        if self.edit_rate_num == 0 {
            return 0.0;
        }
        self.total_duration() as f64 * self.edit_rate_den as f64 / self.edit_rate_num as f64
    }

    /// Returns all unique resource IDs across segments.
    pub fn all_resource_ids(&self) -> HashSet<String> {
        let mut ids = HashSet::new();
        for seg in &self.segments {
            for rid in &seg.resource_ids {
                ids.insert(rid.clone());
            }
        }
        ids
    }
}

/// Result of a CPL merge operation.
#[derive(Clone, Debug)]
pub struct MergeResult {
    /// The merged CPL.
    pub merged: MergeCpl,
    /// Conflicts encountered during the merge.
    pub conflicts: Vec<MergeConflict>,
    /// Number of segments from the base CPL.
    pub base_segments: usize,
    /// Total number of segments in the supplemental CPL (including any whose `id` overlapped a base segment).
    pub supplemental_segments: usize,
}

impl MergeResult {
    /// Returns true if the merge had no errors.
    pub fn is_ok(&self) -> bool {
        !self
            .conflicts
            .iter()
            .any(|c| c.severity == ConflictSeverity::Error)
    }

    /// Returns only error-level conflicts.
    pub fn errors(&self) -> Vec<&MergeConflict> {
        self.conflicts
            .iter()
            .filter(|c| c.severity == ConflictSeverity::Error)
            .collect()
    }

    /// Returns only warning-level conflicts.
    pub fn warnings(&self) -> Vec<&MergeConflict> {
        self.conflicts
            .iter()
            .filter(|c| c.severity == ConflictSeverity::Warning)
            .collect()
    }
}

// ── CPL Diff ─────────────────────────────────────────────────────────────────

/// The structural difference between two [`MergeCpl`] compositions.
///
/// `CplDiff::compute(a, b)` identifies:
/// - segments present only in `b` (additions),
/// - segments present only in `a` (removals), and
/// - segments present in both but with differing content (modifications).
///
/// Segment identity is determined by the [`MergeSegment::id`] field.
#[derive(Clone, Debug)]
pub struct CplDiff {
    /// Segments that exist in `b` but not in `a`.
    pub added_segments: Vec<MergeSegment>,
    /// Segments that exist in `a` but not in `b`.
    pub removed_segments: Vec<MergeSegment>,
    /// Segments present in both `a` and `b` but with different content.
    ///
    /// Each tuple is `(segment_in_a, segment_in_b)`.
    pub modified_segments: Vec<(MergeSegment, MergeSegment)>,
}

impl CplDiff {
    /// Compute the diff between two compositions.
    ///
    /// Two segments are considered the same when their `id` strings match.
    /// They are considered *modified* when the id matches but any of
    /// `duration`, `edit_rate_num`, `edit_rate_den`, or `resource_ids` differ.
    pub fn compute(a: &MergeCpl, b: &MergeCpl) -> Self {
        // Build lookup maps keyed by segment ID.
        let a_map: HashMap<&str, &MergeSegment> =
            a.segments.iter().map(|s| (s.id.as_str(), s)).collect();
        let b_map: HashMap<&str, &MergeSegment> =
            b.segments.iter().map(|s| (s.id.as_str(), s)).collect();

        let mut added_segments = Vec::new();
        let mut removed_segments = Vec::new();
        let mut modified_segments = Vec::new();

        // Walk segments in `b`: find added and modified.
        for b_seg in &b.segments {
            match a_map.get(b_seg.id.as_str()) {
                None => added_segments.push(b_seg.clone()),
                Some(a_seg) => {
                    if !segments_equal(a_seg, b_seg) {
                        modified_segments.push(((*a_seg).clone(), b_seg.clone()));
                    }
                }
            }
        }

        // Walk segments in `a`: find removed.
        for a_seg in &a.segments {
            if !b_map.contains_key(a_seg.id.as_str()) {
                removed_segments.push(a_seg.clone());
            }
        }

        Self {
            added_segments,
            removed_segments,
            modified_segments,
        }
    }

    /// Returns `true` when there are no differences between the two CPLs.
    pub fn is_empty(&self) -> bool {
        self.added_segments.is_empty()
            && self.removed_segments.is_empty()
            && self.modified_segments.is_empty()
    }

    /// Total number of changed segments (additions + removals + modifications).
    pub fn change_count(&self) -> usize {
        self.added_segments.len() + self.removed_segments.len() + self.modified_segments.len()
    }
}

/// Returns `true` when two segments with the same ID have identical content.
fn segments_equal(a: &MergeSegment, b: &MergeSegment) -> bool {
    a.duration == b.duration
        && a.edit_rate_num == b.edit_rate_num
        && a.edit_rate_den == b.edit_rate_den
        && a.resource_ids == b.resource_ids
}

// ── Merge ─────────────────────────────────────────────────────────────────────

/// Merges two CPLs according to the given conflict strategy.
///
/// The `base` CPL is treated as the primary, and `supplemental` provides
/// additional or replacement content.
pub fn merge_cpls(
    base: &MergeCpl,
    supplemental: &MergeCpl,
    strategy: ConflictStrategy,
) -> MergeResult {
    let mut conflicts = Vec::new();
    let mut merged_segments = Vec::new();

    // Check edit rate compatibility
    if base.edit_rate_num != supplemental.edit_rate_num
        || base.edit_rate_den != supplemental.edit_rate_den
    {
        conflicts.push(MergeConflict {
            resource_id: "edit_rate".to_string(),
            description: format!(
                "Edit rate mismatch: {}/{} vs {}/{}",
                base.edit_rate_num,
                base.edit_rate_den,
                supplemental.edit_rate_num,
                supplemental.edit_rate_den
            ),
            severity: ConflictSeverity::Error,
        });
    }

    // Build a map of supplemental segments by ID
    let supp_map: HashMap<&str, &MergeSegment> = supplemental
        .segments
        .iter()
        .map(|s| (s.id.as_str(), s))
        .collect();

    let mut used_supp_ids = HashSet::new();

    for base_seg in &base.segments {
        if let Some(supp_seg) = supp_map.get(base_seg.id.as_str()) {
            // Conflict: same segment ID in both
            used_supp_ids.insert(base_seg.id.as_str());
            match strategy {
                ConflictStrategy::KeepBase => {
                    conflicts.push(MergeConflict {
                        resource_id: base_seg.id.clone(),
                        description: "Duplicate segment; keeping base version".to_string(),
                        severity: ConflictSeverity::Info,
                    });
                    merged_segments.push(base_seg.clone());
                }
                ConflictStrategy::KeepSupplemental => {
                    conflicts.push(MergeConflict {
                        resource_id: base_seg.id.clone(),
                        description: "Duplicate segment; keeping supplemental version".to_string(),
                        severity: ConflictSeverity::Info,
                    });
                    merged_segments.push((*supp_seg).clone());
                }
                ConflictStrategy::Fail => {
                    conflicts.push(MergeConflict {
                        resource_id: base_seg.id.clone(),
                        description: "Duplicate segment; merge aborted".to_string(),
                        severity: ConflictSeverity::Error,
                    });
                    merged_segments.push(base_seg.clone());
                }
                ConflictStrategy::Concatenate => {
                    conflicts.push(MergeConflict {
                        resource_id: base_seg.id.clone(),
                        description: "Duplicate segment; concatenating both versions".to_string(),
                        severity: ConflictSeverity::Warning,
                    });
                    merged_segments.push(base_seg.clone());
                    merged_segments.push((*supp_seg).clone());
                }
            }
        } else {
            merged_segments.push(base_seg.clone());
        }
    }

    // Add supplemental-only segments
    for supp_seg in &supplemental.segments {
        if !used_supp_ids.contains(supp_seg.id.as_str()) {
            merged_segments.push(supp_seg.clone());
        }
    }

    // Edit-rate selection honors the conflict strategy. `KeepSupplemental` adopts the
    // supplemental rate; every other strategy keeps the base rate. (A `MergeConflict` is still
    // pushed above when the rates differ, so the mismatch stays visible regardless.)
    let (merged_rate_num, merged_rate_den) = match strategy {
        ConflictStrategy::KeepSupplemental => {
            (supplemental.edit_rate_num, supplemental.edit_rate_den)
        }
        ConflictStrategy::KeepBase | ConflictStrategy::Fail | ConflictStrategy::Concatenate => {
            (base.edit_rate_num, base.edit_rate_den)
        }
    };

    let merged = MergeCpl {
        id: base.id.clone(),
        title: base.title.clone(),
        edit_rate_num: merged_rate_num,
        edit_rate_den: merged_rate_den,
        segments: merged_segments,
    };

    MergeResult {
        base_segments: base.segments.len(),
        supplemental_segments: supplemental.segments.len(),
        merged,
        conflicts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base_cpl() -> MergeCpl {
        let mut cpl = MergeCpl::new("cpl-base-001", "Base CPL", 24, 1);
        let mut seg1 = MergeSegment::new("seg-001", 1, 24, 1, 240);
        seg1.add_resource("video-001");
        seg1.add_resource("audio-001");
        cpl.add_segment(seg1);

        let mut seg2 = MergeSegment::new("seg-002", 2, 24, 1, 480);
        seg2.add_resource("video-002");
        cpl.add_segment(seg2);
        cpl
    }

    fn make_supp_cpl() -> MergeCpl {
        let mut cpl = MergeCpl::new("cpl-supp-001", "Supplemental CPL", 24, 1);
        let mut seg1 = MergeSegment::new("seg-002", 1, 24, 1, 480);
        seg1.add_resource("video-002-patched");
        cpl.add_segment(seg1);

        let mut seg3 = MergeSegment::new("seg-003", 2, 24, 1, 120);
        seg3.add_resource("subtitle-001");
        cpl.add_segment(seg3);
        cpl
    }

    #[test]
    fn test_cpl_id_display() {
        let id = CplId::new("urn:uuid:12345");
        assert_eq!(id.to_string(), "urn:uuid:12345");
        assert_eq!(id.as_str(), "urn:uuid:12345");
    }

    #[test]
    fn test_merge_segment_duration_seconds() {
        let seg = MergeSegment::new("seg-1", 1, 24, 1, 48);
        assert!((seg.duration_seconds() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_merge_segment_zero_rate() {
        let seg = MergeSegment::new("seg-1", 1, 0, 1, 48);
        assert_eq!(seg.duration_seconds(), 0.0);
    }

    #[test]
    fn test_cpl_total_duration() {
        let cpl = make_base_cpl();
        assert_eq!(cpl.total_duration(), 720); // 240 + 480
    }

    #[test]
    fn test_cpl_total_duration_seconds() {
        let cpl = make_base_cpl();
        assert!((cpl.total_duration_seconds() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpl_all_resource_ids() {
        let cpl = make_base_cpl();
        let ids = cpl.all_resource_ids();
        assert!(ids.contains("video-001"));
        assert!(ids.contains("audio-001"));
        assert!(ids.contains("video-002"));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_merge_keep_base() {
        let base = make_base_cpl();
        let supp = make_supp_cpl();
        let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);
        assert!(result.is_ok());
        // base has seg-001, seg-002; supp has seg-002 (conflict), seg-003 (new)
        // Result: seg-001 (base), seg-002 (base kept), seg-003 (supp)
        assert_eq!(result.merged.segments.len(), 3);
        assert_eq!(result.merged.segments[1].resource_ids[0], "video-002");
    }

    #[test]
    fn test_merge_keep_supplemental() {
        let base = make_base_cpl();
        let supp = make_supp_cpl();
        let result = merge_cpls(&base, &supp, ConflictStrategy::KeepSupplemental);
        assert!(result.is_ok());
        assert_eq!(result.merged.segments.len(), 3);
        assert_eq!(
            result.merged.segments[1].resource_ids[0],
            "video-002-patched"
        );
    }

    #[test]
    fn test_merge_fail_on_conflict() {
        let base = make_base_cpl();
        let supp = make_supp_cpl();
        let result = merge_cpls(&base, &supp, ConflictStrategy::Fail);
        assert!(!result.is_ok());
        assert_eq!(result.errors().len(), 1);
    }

    #[test]
    fn test_merge_concatenate() {
        let base = make_base_cpl();
        let supp = make_supp_cpl();
        let result = merge_cpls(&base, &supp, ConflictStrategy::Concatenate);
        assert!(result.is_ok());
        // seg-001 + seg-002 (base) + seg-002 (supp) + seg-003
        assert_eq!(result.merged.segments.len(), 4);
    }

    #[test]
    fn test_merge_no_overlap() {
        let mut base = MergeCpl::new("b", "Base", 24, 1);
        base.add_segment(MergeSegment::new("a", 1, 24, 1, 100));

        let mut supp = MergeCpl::new("s", "Supp", 24, 1);
        supp.add_segment(MergeSegment::new("b", 1, 24, 1, 200));

        let result = merge_cpls(&base, &supp, ConflictStrategy::Fail);
        assert!(result.is_ok());
        assert_eq!(result.merged.segments.len(), 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_merge_edit_rate_mismatch() {
        let base = MergeCpl::new("b", "Base", 24, 1);
        let supp = MergeCpl::new("s", "Supp", 25, 1);
        let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);
        assert!(!result.is_ok());
        assert_eq!(result.errors().len(), 1);
    }

    #[test]
    fn test_merge_result_warnings() {
        let base = make_base_cpl();
        let supp = make_supp_cpl();
        let result = merge_cpls(&base, &supp, ConflictStrategy::Concatenate);
        assert_eq!(result.warnings().len(), 1);
    }

    #[test]
    fn test_conflict_display() {
        let c = MergeConflict {
            resource_id: "seg-001".to_string(),
            description: "test conflict".to_string(),
            severity: ConflictSeverity::Warning,
        };
        let s = format!("{c}");
        assert!(s.contains("Warning"));
        assert!(s.contains("seg-001"));
    }

    // ── CplDiff tests ─────────────────────────────────────────────────────

    #[test]
    fn test_diff_identical_cpls_no_changes() {
        let base = make_base_cpl();
        let diff = CplDiff::compute(&base, &base);
        assert!(diff.is_empty());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_diff_added_segment() {
        let base = make_base_cpl();
        let mut extended = make_base_cpl();
        extended.add_segment(MergeSegment::new("seg-999", 3, 24, 1, 120));

        let diff = CplDiff::compute(&base, &extended);
        assert_eq!(diff.added_segments.len(), 1);
        assert_eq!(diff.added_segments[0].id, "seg-999");
        assert!(diff.removed_segments.is_empty());
        assert!(diff.modified_segments.is_empty());
    }

    #[test]
    fn test_diff_removed_segment() {
        let mut a = MergeCpl::new("cpl-a", "CPL A", 24, 1);
        a.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 240));
        a.add_segment(MergeSegment::new("seg-002", 2, 24, 1, 480));

        let mut b = MergeCpl::new("cpl-b", "CPL B", 24, 1);
        b.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 240));
        // seg-002 is removed

        let diff = CplDiff::compute(&a, &b);
        assert!(diff.added_segments.is_empty());
        assert_eq!(diff.removed_segments.len(), 1);
        assert_eq!(diff.removed_segments[0].id, "seg-002");
        assert!(diff.modified_segments.is_empty());
    }

    #[test]
    fn test_diff_modified_segment_duration_change() {
        let mut a = MergeCpl::new("cpl-a", "A", 24, 1);
        a.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 240));

        let mut b = MergeCpl::new("cpl-b", "B", 24, 1);
        b.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 480)); // different duration

        let diff = CplDiff::compute(&a, &b);
        assert!(diff.added_segments.is_empty());
        assert!(diff.removed_segments.is_empty());
        assert_eq!(diff.modified_segments.len(), 1);
        let (seg_a, seg_b) = &diff.modified_segments[0];
        assert_eq!(seg_a.duration, 240);
        assert_eq!(seg_b.duration, 480);
    }

    #[test]
    fn test_diff_modified_segment_resources_change() {
        let mut a = MergeCpl::new("a", "A", 24, 1);
        let mut seg_a = MergeSegment::new("seg-001", 1, 24, 1, 240);
        seg_a.add_resource("video-v1");
        a.add_segment(seg_a);

        let mut b = MergeCpl::new("b", "B", 24, 1);
        let mut seg_b = MergeSegment::new("seg-001", 1, 24, 1, 240);
        seg_b.add_resource("video-v2"); // different resource
        b.add_segment(seg_b);

        let diff = CplDiff::compute(&a, &b);
        assert_eq!(diff.modified_segments.len(), 1);
    }

    #[test]
    fn test_diff_complex_changes() {
        let base = make_base_cpl(); // seg-001, seg-002

        let mut modified = MergeCpl::new("m", "Modified", 24, 1);
        // seg-001: same
        let mut seg1 = MergeSegment::new("seg-001", 1, 24, 1, 240);
        seg1.add_resource("video-001");
        seg1.add_resource("audio-001");
        modified.add_segment(seg1);
        // seg-002: changed duration
        modified.add_segment(MergeSegment::new("seg-002", 2, 24, 1, 960));
        // seg-003: new
        modified.add_segment(MergeSegment::new("seg-003", 3, 24, 1, 120));

        let diff = CplDiff::compute(&base, &modified);
        assert_eq!(diff.added_segments.len(), 1);
        assert_eq!(diff.added_segments[0].id, "seg-003");
        assert!(diff.removed_segments.is_empty());
        assert_eq!(diff.modified_segments.len(), 1);
        assert_eq!(diff.modified_segments[0].0.id, "seg-002");
        assert_eq!(diff.change_count(), 2);
    }

    #[test]
    fn test_diff_completely_replaced() {
        let mut a = MergeCpl::new("a", "A", 24, 1);
        a.add_segment(MergeSegment::new("seg-a1", 1, 24, 1, 100));
        a.add_segment(MergeSegment::new("seg-a2", 2, 24, 1, 200));

        let mut b = MergeCpl::new("b", "B", 24, 1);
        b.add_segment(MergeSegment::new("seg-b1", 1, 24, 1, 100));
        b.add_segment(MergeSegment::new("seg-b2", 2, 24, 1, 200));

        let diff = CplDiff::compute(&a, &b);
        assert_eq!(diff.added_segments.len(), 2);
        assert_eq!(diff.removed_segments.len(), 2);
        assert!(diff.modified_segments.is_empty());
        assert_eq!(diff.change_count(), 4);
    }
}
