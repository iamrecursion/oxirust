//! Conflict-resolution tests for CPL merging.
//!
//! These exercise [`oximedia_imf::cpl_merge::merge_cpls`], which NEVER returns
//! an error — conflicts are returned as data inside the [`MergeResult`]. The
//! tests pin the four [`ConflictStrategy`] resolutions plus the
//! edit-rate-mismatch behaviour (which always merges on the base rate while
//! still flagging an Error-severity conflict).

use oximedia_imf::cpl_merge::{merge_cpls, ConflictStrategy, MergeCpl, MergeSegment};

/// Conflicting edit rates must surface a single blocking (Error) conflict whose
/// `resource_id` is `"edit_rate"` and whose description names both rates.
#[test]
fn conflicting_edit_rates_report_blocking_error() {
    let base = MergeCpl::new("b", "Base", 24, 1);
    let supp = MergeCpl::new("s", "Supp", 25, 1);

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);

    assert!(
        !result.is_ok(),
        "edit-rate mismatch must produce a blocking error"
    );
    let errors = result.errors();
    assert_eq!(errors.len(), 1, "exactly one Error-severity conflict");
    assert_eq!(errors[0].resource_id, "edit_rate");
    assert!(
        errors[0].description.contains("24/1"),
        "description should name the base rate 24/1: {}",
        errors[0].description
    );
    assert!(
        errors[0].description.contains("25/1"),
        "description should name the supplemental rate 25/1: {}",
        errors[0].description
    );
}

/// Even with conflicting edit rates the merge still proceeds, adopting the BASE
/// edit rate and concatenating the non-overlapping segments from both CPLs.
#[test]
fn conflicting_edit_rates_still_merges_on_base_rate() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    base.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 240));

    let mut supp = MergeCpl::new("s", "Supp", 30, 1);
    supp.add_segment(MergeSegment::new("seg-002", 1, 30, 1, 300));

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);

    assert_eq!(
        result.merged.edit_rate_num, 24,
        "merged CPL must adopt the base edit-rate numerator"
    );
    assert_eq!(
        result.merged.edit_rate_den, 1,
        "merged CPL must adopt the base edit-rate denominator"
    );
    assert_eq!(
        result.merged.segments.len(),
        2,
        "both distinct segments survive the merge"
    );
    assert!(
        !result.is_ok(),
        "the edit-rate conflict is still flagged as blocking"
    );
}

/// Two segments sharing the same `id` under the `Fail` strategy must produce a
/// single blocking conflict keyed by that segment id.
#[test]
fn overlapping_segment_id_fail_strategy_blocks() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    base.add_segment(MergeSegment::new("seg-X", 1, 24, 1, 100));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-X", 1, 24, 1, 200));

    let result = merge_cpls(&base, &supp, ConflictStrategy::Fail);

    assert!(!result.is_ok(), "Fail strategy must block on overlap");
    let errors = result.errors();
    assert_eq!(
        errors.len(),
        1,
        "one Error-severity conflict for the overlap"
    );
    assert_eq!(errors[0].resource_id, "seg-X");
}

/// Under `Concatenate`, an overlapping segment id is NON-blocking (Warning):
/// both versions are kept, in base-then-supplemental order.
#[test]
fn overlapping_segment_id_concatenate_emits_both() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    base.add_segment(MergeSegment::new("seg-X", 1, 24, 1, 100));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-X", 1, 24, 1, 200));

    let result = merge_cpls(&base, &supp, ConflictStrategy::Concatenate);

    assert!(
        result.is_ok(),
        "Concatenate emits a Warning, not a blocking Error"
    );
    assert_eq!(
        result.merged.segments.len(),
        2,
        "both versions of the overlapping segment are retained"
    );
    assert_eq!(result.warnings().len(), 1, "exactly one Warning conflict");
    let durations: Vec<u64> = result.merged.segments.iter().map(|s| s.duration).collect();
    assert_eq!(
        durations,
        vec![100, 200],
        "base version precedes the supplemental version"
    );
}

/// Under `KeepSupplemental`, the supplemental segment's resources replace the
/// base segment's resources for an overlapping id, and the merge is clean.
#[test]
fn keep_supplemental_replaces_overlapping_resource() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    let mut base_seg = MergeSegment::new("seg-X", 1, 24, 1, 100);
    base_seg.add_resource("v1");
    base.add_segment(base_seg);

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    let mut supp_seg = MergeSegment::new("seg-X", 1, 24, 1, 100);
    supp_seg.add_resource("v2");
    supp.add_segment(supp_seg);

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepSupplemental);

    assert!(result.is_ok(), "KeepSupplemental is a clean resolution");
    assert_eq!(
        result.merged.segments.len(),
        1,
        "the overlap collapses to one segment"
    );
    assert_eq!(
        result.merged.segments[0].resource_ids[0], "v2",
        "supplemental resource replaces the base resource"
    );
}
