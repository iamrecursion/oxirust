//! Strategy-aware edit-rate selection and supplemental-segment counting tests
//! for [`oximedia_imf::cpl_merge::merge_cpls`].
//!
//! These pin two production behaviours:
//!
//! 1. The merged CPL's edit rate is chosen according to the [`ConflictStrategy`]:
//!    `KeepSupplemental` adopts the supplemental rate, while `KeepBase`, `Fail`,
//!    and `Concatenate` retain the base rate. When the rates differ, an
//!    Error-severity `edit_rate` conflict is still surfaced regardless of strategy.
//! 2. [`MergeResult::supplemental_segments`] reports the TOTAL number of segments
//!    in the supplemental CPL — including any whose `id` overlapped a base segment.

use oximedia_imf::cpl_merge::{merge_cpls, ConflictStrategy, MergeCpl, MergeSegment};

/// `KeepBase` keeps the base edit rate while still flagging the rate mismatch as
/// a single blocking (Error) conflict keyed `"edit_rate"`.
#[test]
fn keep_base_adopts_base_rate_and_flags_conflict() {
    let mut base = MergeCpl::new("b", "Base", 24000, 1001);
    base.add_segment(MergeSegment::new("seg-001", 1, 24000, 1001, 240));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-002", 1, 24, 1, 300));

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);

    assert_eq!(
        result.merged.edit_rate_num, 24000,
        "KeepBase must retain the base edit-rate numerator"
    );
    assert_eq!(
        result.merged.edit_rate_den, 1001,
        "KeepBase must retain the base edit-rate denominator"
    );
    assert!(
        !result.is_ok(),
        "differing edit rates must still produce a blocking error"
    );
    let errors = result.errors();
    assert_eq!(errors.len(), 1, "exactly one Error-severity conflict");
    assert_eq!(errors[0].resource_id, "edit_rate");
}

/// `KeepSupplemental` adopts the supplemental edit rate while still flagging the
/// rate mismatch as a single blocking (Error) conflict keyed `"edit_rate"`.
#[test]
fn keep_supplemental_adopts_supplemental_rate_and_flags_conflict() {
    let mut base = MergeCpl::new("b", "Base", 24000, 1001);
    base.add_segment(MergeSegment::new("seg-001", 1, 24000, 1001, 240));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-002", 1, 24, 1, 300));

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepSupplemental);

    assert_eq!(
        result.merged.edit_rate_num, 24,
        "KeepSupplemental must adopt the supplemental edit-rate numerator"
    );
    assert_eq!(
        result.merged.edit_rate_den, 1,
        "KeepSupplemental must adopt the supplemental edit-rate denominator"
    );
    assert!(
        !result.is_ok(),
        "differing edit rates must still produce a blocking error"
    );
    let errors = result.errors();
    assert_eq!(errors.len(), 1, "exactly one Error-severity conflict");
    assert_eq!(errors[0].resource_id, "edit_rate");
}

/// Both `Fail` and `Concatenate` keep the base edit rate (only `KeepSupplemental`
/// switches to the supplemental rate).
#[test]
fn fail_and_concatenate_keep_base_rate() {
    for strategy in [ConflictStrategy::Fail, ConflictStrategy::Concatenate] {
        let mut base = MergeCpl::new("b", "Base", 24000, 1001);
        base.add_segment(MergeSegment::new("seg-001", 1, 24000, 1001, 240));

        let mut supp = MergeCpl::new("s", "Supp", 24, 1);
        supp.add_segment(MergeSegment::new("seg-002", 1, 24, 1, 300));

        let result = merge_cpls(&base, &supp, strategy);

        assert_eq!(
            result.merged.edit_rate_num, 24000,
            "{strategy:?} must retain the base edit-rate numerator"
        );
        assert_eq!(
            result.merged.edit_rate_den, 1001,
            "{strategy:?} must retain the base edit-rate denominator"
        );
    }
}

/// When the rates already match, `KeepSupplemental` yields the (identical) rate
/// and no `edit_rate` conflict is raised.
#[test]
fn matching_rates_unaffected_by_strategy() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    base.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 240));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-002", 1, 24, 1, 300));

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepSupplemental);

    assert_eq!(
        result.merged.edit_rate_num, 24,
        "matching rates yield the shared numerator"
    );
    assert_eq!(
        result.merged.edit_rate_den, 1,
        "matching rates yield the shared denominator"
    );
    assert!(
        !result
            .conflicts
            .iter()
            .any(|c| c.resource_id == "edit_rate"),
        "no edit_rate conflict when the rates already match"
    );
}

/// `supplemental_segments` is the TOTAL supplemental count, including a segment
/// whose `id` overlaps a base segment (pre-fix this reported only the
/// non-overlapping count, i.e. 1).
#[test]
fn supplemental_segments_counts_total_including_overlap() {
    let mut base = MergeCpl::new("b", "Base", 24, 1);
    base.add_segment(MergeSegment::new("seg-001", 1, 24, 1, 100));
    base.add_segment(MergeSegment::new("seg-002", 2, 24, 1, 100));

    let mut supp = MergeCpl::new("s", "Supp", 24, 1);
    supp.add_segment(MergeSegment::new("seg-002", 1, 24, 1, 200)); // overlaps base
    supp.add_segment(MergeSegment::new("seg-003", 2, 24, 1, 300)); // new

    let result = merge_cpls(&base, &supp, ConflictStrategy::KeepBase);

    assert_eq!(
        result.base_segments, 2,
        "base_segments is the total base segment count"
    );
    assert_eq!(
        result.supplemental_segments, 2,
        "supplemental_segments is the total supplemental count, including the overlap"
    );
}
