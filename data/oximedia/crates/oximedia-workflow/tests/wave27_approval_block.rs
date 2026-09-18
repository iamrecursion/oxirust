//! Wave 27 — workflow blocks on an approval gate, then resumes on approval.
//!
//! This models a workflow step that must pause until a human approval arrives
//! and then continues. The `ApprovalGate` itself is synchronous; the async
//! block/resume coordination is built in the test from a tokio `Notify` plus a
//! `Mutex<ApprovalGate>`, with NO production async added to `approval_gate`.
//!
//! Determinism: there are NO sleeps. A second `Notify` (`blocked`) lets the
//! approver wait until the workflow has actually entered its blocked state
//! before it approves, pinning the observed event order to
//! `["blocked", "approved", "resumed"]`. `tokio::time::timeout` is used ONLY as
//! a safety net so a logic bug fails fast instead of hanging the suite — it is
//! never the assertion.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use oximedia_workflow::approval_gate::{
    ApprovalDecision, ApprovalGate, ApprovalGateConfig, ApprovalPolicy, GateId,
};
use tokio::sync::Notify;

#[tokio::test]
async fn test_workflow_blocks_until_approval_then_resumes() {
    // Approval gate requiring a single approval from "alice".
    let config = ApprovalGateConfig::new(
        "release-gate",
        vec!["alice".to_string()],
        ApprovalPolicy::Any,
    );
    let gate = Arc::new(tokio::sync::Mutex::new(ApprovalGate::new(
        GateId::new(1),
        config,
    )));

    // `approved` wakes the blocked workflow once a decision lands; `blocked`
    // tells the approver that the workflow has parked itself.
    let approved = Arc::new(Notify::new());
    let blocked = Arc::new(Notify::new());

    // Ordered event log shared between the two halves.
    let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

    // ---- Workflow half: park until approved, then resume. ----
    let workflow = {
        let gate = gate.clone();
        let approved = approved.clone();
        let blocked = blocked.clone();
        let order = order.clone();
        tokio::spawn(async move {
            // Gate starts pending — record that we are blocked.
            assert!(
                gate.lock().await.is_pending(),
                "gate should start pending before any decision"
            );
            order
                .lock()
                .expect("order log mutex poisoned")
                .push("blocked");

            // Announce we are parked so the approver can proceed; this pins the
            // event order without a sleep.
            blocked.notify_one();

            // Block until the gate is approved.
            loop {
                if gate.lock().await.is_approved() {
                    break;
                }
                approved.notified().await;
            }

            order
                .lock()
                .expect("order log mutex poisoned")
                .push("resumed");
        })
    };

    // ---- Approver half: wait for the block, then approve. ----
    let approver = {
        let gate = gate.clone();
        let approved = approved.clone();
        let blocked = blocked.clone();
        let order = order.clone();
        tokio::spawn(async move {
            // Wait until the workflow has actually blocked.
            blocked.notified().await;

            order
                .lock()
                .expect("order log mutex poisoned")
                .push("approved");

            gate.lock().await.submit_decision(ApprovalDecision {
                approver: "alice".to_string(),
                approved: true,
                comment: None,
                decided_at: Instant::now(),
            });

            // Wake the blocked workflow.
            approved.notify_one();
        })
    };

    // Safety net only: a correct run completes near-instantly. A deadlock fails
    // here instead of hanging the whole test binary.
    tokio::time::timeout(Duration::from_secs(5), async {
        workflow.await.expect("workflow task panicked");
        approver.await.expect("approver task panicked");
    })
    .await
    .expect("block/resume must complete promptly — a hang indicates a bug");

    // The gate must end up approved, and the events must be strictly ordered.
    assert!(gate.lock().await.is_approved(), "gate must be approved");

    let log = order.lock().expect("order log mutex poisoned").clone();
    assert_eq!(
        log,
        vec!["blocked", "approved", "resumed"],
        "workflow must block, then be approved, then resume — in that order"
    );
}
