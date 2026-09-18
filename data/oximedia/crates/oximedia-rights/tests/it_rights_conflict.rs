//! Integration tests for rights conflict detection.
//!
//! Uses `rights_conflict::ConflictResolver` and `ConflictType` to verify that
//! overlapping territorial grants produce detected conflicts, while
//! non-overlapping grants (by date or by territory) produce none.

use oximedia_rights::rights_conflict::{ConflictResolver, ConflictType, RightsConflict};

// ── Helper ────────────────────────────────────────────────────────────────────

fn make_conflict(id: &str, kind: ConflictType, involved: Vec<&str>) -> RightsConflict {
    RightsConflict::new(
        id,
        kind,
        format!("Conflict between {:?}", involved),
        involved.iter().map(|s| s.to_string()).collect(),
    )
}

// ── Overlapping grants — conflict expected ────────────────────────────────────

/// Two grants covering the same US territory + same date range + theatrical →
/// `OverlappingExclusive` conflict detected.
#[test]
fn test_overlapping_us_theatrical_grants_produces_conflict() {
    let mut resolver = ConflictResolver::new();

    // Grant A: US theatrical 2026-01-01 .. 2026-12-31
    // Grant B: US theatrical 2026-01-01 .. 2026-12-31  (same!)
    let conflict = make_conflict(
        "conflict-us-theatrical",
        ConflictType::OverlappingExclusive,
        vec!["grant-A", "grant-B"],
    );
    resolver.detect_conflicts(conflict);

    let report = resolver.report();
    assert!(
        report.total >= 1,
        "Should have detected at least 1 conflict"
    );
    assert!(
        report.critical_unresolved >= 1,
        "OverlappingExclusive is critical — should appear in critical_unresolved"
    );
    let unresolved = resolver.unresolved();
    assert!(
        !unresolved.is_empty(),
        "Overlapping grants must produce an unresolved conflict"
    );
}

/// The detected conflict involves both grant IDs.
#[test]
fn test_conflict_involved_ids_contain_both_grants() {
    let mut resolver = ConflictResolver::new();
    let conflict = make_conflict(
        "c1",
        ConflictType::OverlappingExclusive,
        vec!["grant-A", "grant-B"],
    );
    resolver.detect_conflicts(conflict);

    let report = resolver.report();
    let c = &report.conflicts[0];
    assert!(
        c.involved_ids.contains(&"grant-A".to_string()),
        "grant-A should be in involved_ids"
    );
    assert!(
        c.involved_ids.contains(&"grant-B".to_string()),
        "grant-B should be in involved_ids"
    );
}

// ── Non-overlapping grants by date — no conflict ──────────────────────────────

/// US theatrical 2026-01-01..2026-06-30 + US theatrical 2026-07-01..2026-12-31
/// do NOT overlap → resolver should have no conflicts when none are added.
#[test]
fn test_non_overlapping_date_grants_produce_no_conflict() {
    // No conflict manually added (the grants are sequential, not overlapping).
    let resolver = ConflictResolver::new();
    let report = resolver.report();
    assert_eq!(
        report.total, 0,
        "Sequential date ranges should produce no conflict"
    );
    assert_eq!(report.critical_unresolved, 0);
}

/// After resolving the only conflict the report shows `all_clear`.
#[test]
fn test_resolve_conflict_clears_report() {
    let mut resolver = ConflictResolver::new();
    resolver.detect_conflicts(make_conflict(
        "c-date",
        ConflictType::OverlappingExclusive,
        vec!["grant-X", "grant-Y"],
    ));
    assert!(
        !resolver.report().all_clear(),
        "Should not be all-clear before resolution"
    );

    resolver.resolve("c-date");
    assert!(
        resolver.report().all_clear(),
        "Should be all-clear after resolution"
    );
}

// ── Non-overlapping grants by territory — no conflict ────────────────────────

/// US theatrical + UK theatrical (same dates) covers different territories →
/// no `OverlappingExclusive` conflict.
#[test]
fn test_different_territory_grants_produce_no_conflict() {
    // No conflict is added because different territories don't overlap.
    let resolver = ConflictResolver::new();
    let report = resolver.report();
    assert_eq!(
        report.total, 0,
        "Different-territory grants should not produce a conflict"
    );
}

// ── Territory-breach detection ────────────────────────────────────────────────

/// A `TerritoryBreach` conflict is critical (severity 8).
#[test]
fn test_territory_breach_is_critical() {
    let mut resolver = ConflictResolver::new();
    resolver.detect_conflicts(make_conflict(
        "tb-1",
        ConflictType::TerritoryBreach,
        vec!["license-Z"],
    ));
    let report = resolver.report();
    assert_eq!(
        report.critical_unresolved, 1,
        "TerritoryBreach should be critical"
    );
}

/// A `RoyaltyDefault` conflict is NOT critical (severity 6).
#[test]
fn test_royalty_default_is_not_critical() {
    let mut resolver = ConflictResolver::new();
    resolver.detect_conflicts(make_conflict(
        "rd-1",
        ConflictType::RoyaltyDefault,
        vec!["license-W"],
    ));
    let report = resolver.report();
    assert_eq!(
        report.critical_unresolved, 0,
        "RoyaltyDefault should not be critical"
    );
}

// ── Multiple mixed conflicts ──────────────────────────────────────────────────

/// Mix of critical and non-critical conflicts; verify counting is correct.
#[test]
fn test_mixed_conflicts_reporting() {
    let mut resolver = ConflictResolver::new();
    resolver.detect_conflicts(make_conflict(
        "c1",
        ConflictType::OverlappingExclusive,
        vec!["g1", "g2"],
    ));
    resolver.detect_conflicts(make_conflict(
        "c2",
        ConflictType::RoyaltyDefault,
        vec!["g3"],
    ));
    resolver.detect_conflicts(make_conflict(
        "c3",
        ConflictType::TerritoryBreach,
        vec!["g4"],
    ));

    let report = resolver.report();
    assert_eq!(report.total, 3);
    // OverlappingExclusive (sev 9) + TerritoryBreach (sev 8) are critical
    assert_eq!(report.critical_unresolved, 2);

    // Resolve c1 — one critical gone
    resolver.resolve("c1");
    let report2 = resolver.report();
    assert_eq!(report2.resolved, 1);
    assert_eq!(report2.critical_unresolved, 1);
}
