//! Integration tests for the clearance workflow state machine.
//!
//! Covers: initial Pending state, approval, rejection, overdue detection via
//! `created_at` + elapsed-time logic (the `ClearanceWorkflow` does not expose a
//! dedicated `find_overdue_clearances` fn, so we implement the check inline).

use oximedia_rights::clearance_workflow::{
    ClearanceRequest, ClearanceStatus, ClearanceWorkflow, UsageEventType,
};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Build a minimal `ClearanceRequest` with the given `id` and `created_at`.
fn make_request(id: &str, created_at: u64) -> ClearanceRequest {
    ClearanceRequest::new(
        id,
        "asset-001",
        "requester-corp",
        "rights-holder-inc",
        UsageEventType::Stream,
        vec![], // worldwide
        created_at,
        None, // perpetual
        created_at,
        Some(250.0),
    )
}

// ── Test 1: New clearance starts as Pending ───────────────────────────────────

#[test]
fn test_new_clearance_is_pending() {
    let mut wf = ClearanceWorkflow::new();
    let req = make_request("r-pending", 1_000);
    wf.submit_request(req);

    let found = wf
        .lookup("r-pending")
        .expect("request should exist after submit");
    assert_eq!(
        found.status,
        ClearanceStatus::Pending,
        "Freshly submitted clearance must be in Pending state"
    );
}

// ── Test 2: Approve transitions to Approved ───────────────────────────────────

#[test]
fn test_approve_transitions_to_approved() {
    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-approve", 2_000));
    wf.approve("r-approve", None, "All good")
        .expect("approve should succeed");

    let found = wf.lookup("r-approve").expect("request should exist");
    assert_eq!(
        found.status,
        ClearanceStatus::Approved,
        "After approval the status must be Approved"
    );
}

// ── Test 3: Reject transitions to Rejected ────────────────────────────────────

#[test]
fn test_reject_transitions_to_rejected() {
    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-reject", 3_000));
    wf.reject("r-reject", "Terms not acceptable")
        .expect("reject should succeed");

    let found = wf.lookup("r-reject").expect("request should exist");
    assert_eq!(
        found.status,
        ClearanceStatus::Rejected,
        "After rejection the status must be Rejected"
    );
}

// ── Test 4: Overdue detection ─────────────────────────────────────────────────

/// A clearance created 200 days ago (in seconds) and still Pending is
/// considered overdue when the timeout is 30 days.
///
/// Since `ClearanceWorkflow` does not expose a built-in `find_overdue_clearances`
/// helper we implement the check directly — this mirrors what a real caller
/// would do and verifies that `created_at` is correctly persisted.
#[test]
fn test_overdue_clearance_detected_after_200_days() {
    const SECS_PER_DAY: u64 = 86_400;
    const TIMEOUT_DAYS: u64 = 30;
    const ELAPSED_DAYS: u64 = 200;

    // Simulate "now" as a large round number.
    let now: u64 = 1_748_736_000; // 2026-06-01T00:00:00Z (approx)
    let created_at = now - ELAPSED_DAYS * SECS_PER_DAY;

    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-overdue", created_at));

    // Find overdue: still Pending + (now - created_at) > timeout
    let overdue: Vec<_> = std::iter::once(wf.lookup("r-overdue"))
        .flatten()
        .filter(|r| {
            r.status == ClearanceStatus::Pending
                && now.saturating_sub(r.created_at) > TIMEOUT_DAYS * SECS_PER_DAY
        })
        .collect();

    assert!(
        !overdue.is_empty(),
        "Clearance created {ELAPSED_DAYS} days ago (timeout={TIMEOUT_DAYS} days) should be overdue"
    );
    assert_eq!(overdue[0].id, "r-overdue");
}

/// A clearance created only 10 days ago is NOT overdue under a 30-day timeout.
#[test]
fn test_recent_clearance_not_overdue() {
    const SECS_PER_DAY: u64 = 86_400;
    const TIMEOUT_DAYS: u64 = 30;

    let now: u64 = 1_748_736_000;
    let created_at = now - 10 * SECS_PER_DAY; // only 10 days ago

    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-recent", created_at));

    let overdue: Vec<_> = std::iter::once(wf.lookup("r-recent"))
        .flatten()
        .filter(|r| {
            r.status == ClearanceStatus::Pending
                && now.saturating_sub(r.created_at) > TIMEOUT_DAYS * SECS_PER_DAY
        })
        .collect();

    assert!(
        overdue.is_empty(),
        "Clearance only 10 days old should not be flagged as overdue (30-day timeout)"
    );
}

// ── Additional state machine transitions ──────────────────────────────────────

/// Counter-offer path: Pending → UnderNegotiation → Approved.
#[test]
fn test_counter_offer_then_accept_approved() {
    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-counter", 4_000));
    wf.counter_offer("r-counter", 999.0, "counter proposal")
        .expect("counter should work");

    let mid = wf.lookup("r-counter").expect("exists");
    assert_eq!(mid.status, ClearanceStatus::UnderNegotiation);

    wf.accept_counter("r-counter").expect("accept should work");
    let final_req = wf.lookup("r-counter").expect("exists");
    assert_eq!(final_req.status, ClearanceStatus::Approved);
}

/// Cannot approve a Rejected request.
#[test]
fn test_cannot_approve_rejected_request() {
    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-no-reopen", 5_000));
    wf.reject("r-no-reopen", "denied").expect("reject");
    let result = wf.approve("r-no-reopen", None, "late approval");
    assert!(
        result.is_err(),
        "Should not be able to approve a Rejected request"
    );
}

/// `is_cleared` returns false while status is Pending.
#[test]
fn test_is_cleared_returns_false_while_pending() {
    let mut wf = ClearanceWorkflow::new();
    wf.submit_request(make_request("r-check", 6_000));
    assert!(
        !wf.is_cleared("asset-001", &UsageEventType::Stream, "US", 7_000),
        "Pending clearance should not report as cleared"
    );
}
