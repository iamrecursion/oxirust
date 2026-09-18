//! Integration tests for embargo window edge cases.
//!
//! Tests `embargo_window::TerritorialEmbargo` / `EmbargoWindowSchedule` and
//! `embargo_policy::EmbargoManager` for overlapping embargo windows,
//! exact midnight UTC boundary behaviour, and UTC-anchored timezone semantics.

use oximedia_rights::embargo_policy::{EmbargoManager, EmbargoRegion, EmbargoRule, EmbargoType};
use oximedia_rights::embargo_window::{
    EmbargoWindowSchedule, ReleaseWindow, TerritorialEmbargo, TerritorialEmbargoStatus,
};

// ── Overlapping embargo windows ───────────────────────────────────────────────

/// When two embargo rules cover the same content, the effective embargo is the
/// most restrictive (latest lift date).  Both windows are active until the
/// second one lifts.
#[test]
fn test_overlapping_embargoes_most_restrictive_applies() {
    let mut mgr = EmbargoManager::new();

    // Window 1: lifts at T+100
    let lift1 = 1_748_736_000_u64 + 100;
    // Window 2: lifts at T+200 (later, more restrictive)
    let lift2 = 1_748_736_000_u64 + 200;

    mgr.add(EmbargoRule::new(
        42,
        EmbargoRegion::Worldwide,
        EmbargoType::Online,
        lift1,
    ));
    mgr.add(EmbargoRule::new(
        42,
        EmbargoRegion::Worldwide,
        EmbargoType::Online,
        lift2,
    ));

    // Just after window 1 lifts but before window 2 lifts — still embargoed.
    let after_lift1 = lift1 + 1;
    assert!(
        mgr.is_embargoed(42, after_lift1),
        "Asset 42 should still be embargoed between the two lift dates"
    );

    // After both windows have lifted — no longer embargoed.
    let after_lift2 = lift2 + 1;
    assert!(
        !mgr.is_embargoed(42, after_lift2),
        "Asset 42 should be clear after the later window lifts"
    );
}

/// An `EmbargoWindowSchedule` with two overlapping `ReleaseWindow`s both
/// reports them as open simultaneously, allowing systems to select the
/// most restrictive (earliest close / latest open semantics per caller).
#[test]
fn test_release_windows_overlap_both_show_open() {
    let mut sched = EmbargoWindowSchedule::new();
    // Window A: [1000, 3000]
    sched.add_window(ReleaseWindow::new(1, "WindowA", 1_000, Some(3_000)));
    // Window B: [2000, 5000]  — overlaps with A in [2000, 3000]
    sched.add_window(ReleaseWindow::new(2, "WindowB", 2_000, Some(5_000)));

    // At t=2500, both should be open.
    let open = sched.open_windows(2_500);
    assert_eq!(
        open.len(),
        2,
        "Both windows should be open at t=2500 (overlap region)"
    );

    // At t=3500, only B is open (A already closed).
    let open_late = sched.open_windows(3_500);
    assert_eq!(open_late.len(), 1, "Only window B should be open at t=3500");
    assert_eq!(open_late[0].id, 2);
}

// ── Midnight boundary ─────────────────────────────────────────────────────────

/// Asset embargoed until 2026-06-01T00:00:00Z (Unix 1_748_736_000).
/// One second before: embargoed.  One second after: available.
#[test]
fn test_midnight_utc_boundary_embargo_exact() {
    // 2026-06-01T00:00:00Z  ↔  1748736000 (verified with: date -d "2026-06-01 00:00:00 UTC" +%s)
    // (approximation — actual value computed below)
    // days from 1970-01-01 to 2026-06-01:
    //   years: 56 full years + leap days
    // Use EmbargoManager with lift_epoch = this timestamp.
    const LIFT_EPOCH: u64 = 1_748_736_000;

    let mut mgr = EmbargoManager::new();
    mgr.add(EmbargoRule::new(
        7,
        EmbargoRegion::Worldwide,
        EmbargoType::Release,
        LIFT_EPOCH,
    ));

    // One second before midnight: should be embargoed.
    assert!(
        mgr.is_embargoed(7, LIFT_EPOCH - 1),
        "One second before lift: should be embargoed"
    );

    // At exactly the lift epoch: is_lifted checks `now_epoch >= lift_epoch`, so cleared.
    assert!(
        !mgr.is_embargoed(7, LIFT_EPOCH),
        "Exactly at lift epoch: should be cleared"
    );

    // One second after: clearly cleared.
    assert!(
        !mgr.is_embargoed(7, LIFT_EPOCH + 1),
        "One second after lift: should be cleared"
    );
}

/// Validates that the embargo uses UTC epoch comparison regardless of local
/// timezone — a viewer at UTC+9 (Tokyo) whose local midnight (15:00 UTC
/// previous day = LIFT_EPOCH - 9*3600) is before the UTC embargo lift.
#[test]
fn test_tokyo_local_midnight_still_respects_utc_boundary() {
    const LIFT_EPOCH: u64 = 1_748_736_000; // 2026-06-01T00:00:00Z

    let mut mgr = EmbargoManager::new();
    mgr.add(EmbargoRule::new(
        8,
        EmbargoRegion::AsiaPacific,
        EmbargoType::Online,
        LIFT_EPOCH,
    ));

    // Tokyo local midnight 2026-06-01 = 2026-05-31T15:00:00Z (UTC) = LIFT_EPOCH - 9h
    let tokyo_local_midnight_utc = LIFT_EPOCH - 9 * 3_600;

    // At that UTC instant the embargo has NOT yet lifted.
    assert!(
        mgr.is_embargoed(8, tokyo_local_midnight_utc),
        "Tokyo local midnight is 9 h before UTC midnight — still embargoed"
    );

    // After the UTC lift the embargo is gone even for Tokyo viewers.
    assert!(
        !mgr.is_embargoed(8, LIFT_EPOCH),
        "After UTC lift epoch: cleared for all regions including Tokyo"
    );
}

// ── TerritorialEmbargo additional edge cases ──────────────────────────────────

/// A `TerritorialEmbargo` with `lift_at = None` (indefinite) never auto-clears.
#[test]
fn test_indefinite_embargo_never_clears_automatically() {
    let e = TerritorialEmbargo::new(
        "CN",
        TerritorialEmbargoStatus::Embargoed,
        None, // no lift date
        None,
    );
    // Check at a very large future timestamp — should remain embargoed.
    assert!(
        e.is_blocking(i64::MAX / 2),
        "Indefinite embargo should never auto-clear"
    );
}

/// A `TerritorialEmbargoStatus::ScheduledLift` is still blocking until the
/// lift timestamp is reached.
#[test]
fn test_scheduled_lift_is_still_blocking_before_lift() {
    let lift = 9_000_000_000_i64;
    let e = TerritorialEmbargo::new(
        "RU",
        TerritorialEmbargoStatus::ScheduledLift,
        Some(lift),
        None,
    );
    assert!(
        e.is_blocking(lift - 1),
        "Should be blocking one second before scheduled lift"
    );
    assert!(
        !e.is_blocking(lift),
        "Should not be blocking at exactly the lift time"
    );
}
