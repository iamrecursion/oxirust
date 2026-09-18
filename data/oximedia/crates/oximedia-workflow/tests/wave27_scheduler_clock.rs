//! Wave 27 — deterministic cron scheduling via the injectable [`Clock`] seam.
//!
//! These tests drive [`WorkflowScheduler`] with a `FakeClock` so cron firing is
//! evaluated at *logical* times with zero wall-clock sleeping:
//!
//! * `test_cron_fires_at_logical_time` — a per-minute cron workflow does NOT
//!   fire while the (fake) clock is before `next_execution`, fires exactly once
//!   when the clock reaches/passes it, then the queue empties.
//! * `test_cron_recomputes_next_after_fire` — after firing, `next_execution`
//!   advances by one period (the next minute boundary).
//!
//! NOTE: `ScheduledWorkflow::new` (invoked inside `add_schedule`) computes the
//! INITIAL `next_execution` from the real system clock, not the injected one
//! (this keeps `add_schedule`'s signature byte-compatible). The tests therefore
//! read the real `next_execution` back via `get_next_execution` and anchor the
//! FakeClock relative to it — fully deterministic regardless of when the test
//! actually runs.

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};
use oximedia_workflow::{Clock, Trigger, Workflow, WorkflowScheduler};

/// A test clock whose time is fully controlled by the test via [`Self::advance`]
/// / [`Self::set`]. Reads never touch the wall clock, so cron evaluation is
/// deterministic.
#[derive(Debug)]
struct FakeClock {
    now: Mutex<DateTime<Utc>>,
}

impl FakeClock {
    fn new(start: DateTime<Utc>) -> Self {
        Self {
            now: Mutex::new(start),
        }
    }

    /// Advance the clock by `delta`.
    fn advance(&self, delta: Duration) {
        let mut guard = self.now.lock().expect("FakeClock mutex poisoned in test");
        *guard += delta;
    }

    /// Set the clock to an absolute instant.
    fn set(&self, when: DateTime<Utc>) {
        let mut guard = self.now.lock().expect("FakeClock mutex poisoned in test");
        *guard = when;
    }
}

impl Clock for FakeClock {
    fn now(&self) -> DateTime<Utc> {
        *self.now.lock().expect("FakeClock mutex poisoned in test")
    }
}

/// Per-minute cron trigger: fires at second 0 of every minute (6-field cron:
/// sec min hour dom month dow).
fn per_minute_trigger() -> Trigger {
    Trigger::Cron {
        expression: "0 * * * * *".to_string(),
        timezone: "UTC".to_string(),
    }
}

#[tokio::test]
async fn test_cron_fires_at_logical_time() {
    let fake = Arc::new(FakeClock::new(Utc::now()));
    let scheduler = WorkflowScheduler::with_clock(fake.clone());
    scheduler.start().await.expect("scheduler should start");

    let workflow = Workflow::new("cron-workflow");
    let workflow_id = scheduler
        .add_schedule(workflow, per_minute_trigger())
        .await
        .expect("add_schedule should succeed");

    // The initial next_execution is computed from the real clock inside
    // add_schedule; read it back so we can anchor the FakeClock around it.
    let next = scheduler
        .get_next_execution(workflow_id)
        .await
        .expect("cron schedule must have a next_execution");

    // BEFORE the boundary: pin the fake clock one second prior — nothing fires.
    fake.set(next - Duration::seconds(1));
    let ready = scheduler.check_schedules().await;
    assert!(
        ready.is_empty(),
        "workflow must NOT fire before its next_execution (now < next)"
    );

    // AT/AFTER the boundary: advance the one remaining second so the clock
    // lands exactly on `next` — it fires once.
    fake.advance(Duration::seconds(1));
    assert_eq!(fake.now(), next, "clock should now sit on the boundary");
    let ready = scheduler.check_schedules().await;
    assert_eq!(
        ready.len(),
        1,
        "workflow must fire exactly when now >= next_execution"
    );
    assert_eq!(ready[0].id, workflow_id);

    // Immediately re-checking at the same instant must NOT re-fire: the
    // schedule recomputed a fresh next_execution strictly in the future.
    let ready_again = scheduler.check_schedules().await;
    assert!(
        ready_again.is_empty(),
        "workflow must not fire twice for the same period — queue empties"
    );
}

#[tokio::test]
async fn test_cron_recomputes_next_after_fire() {
    let fake = Arc::new(FakeClock::new(Utc::now()));
    let scheduler = WorkflowScheduler::with_clock(fake.clone());
    scheduler.start().await.expect("scheduler should start");

    let workflow = Workflow::new("cron-recompute");
    let workflow_id = scheduler
        .add_schedule(workflow, per_minute_trigger())
        .await
        .expect("add_schedule should succeed");

    let first_next = scheduler
        .get_next_execution(workflow_id)
        .await
        .expect("cron schedule must have a next_execution");

    // Fire it by moving the fake clock to the boundary.
    fake.set(first_next);
    let ready = scheduler.check_schedules().await;
    assert_eq!(ready.len(), 1, "workflow should fire at the boundary");

    // After firing, next_execution must have advanced to a strictly later
    // instant (the following minute boundary, ~60s later for a per-minute cron).
    let second_next = scheduler
        .get_next_execution(workflow_id)
        .await
        .expect("next_execution must still be present after firing");

    assert!(
        second_next > first_next,
        "next_execution must advance after firing: {second_next} should be > {first_next}"
    );

    // For a per-minute cron the period is 60s; the recomputed boundary is the
    // next minute strictly after `first_next`, i.e. within (0, 60] seconds.
    let delta = second_next - first_next;
    assert!(
        delta > Duration::zero() && delta <= Duration::seconds(60),
        "per-minute cron must advance by one period (<=60s), got {delta}"
    );
}
