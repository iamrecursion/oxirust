//! Integration tests for `KeyRotationSchedule` time-based rotation triggers.
//!
//! Smoke-tests the public API at `oximedia_drm::key_rotation_schedule` by
//! simulating a logical clock that advances in fixed steps. The schedule is
//! checked against a rotation period (5 seconds) over a 30-second window and
//! the test asserts both the count and the boundary alignment of rotation
//! events.
//!
//! Because the API takes Unix epoch seconds as input (no real-time clock
//! dependency), we drive it with a deterministic counter rather than
//! `SystemTime::now()`.

use oximedia_drm::key_rotation_schedule::{
    KeyRotationSchedule, RotationInterval, TransitionWindow, VersionedKey,
};

/// Simulated logical clock with a fixed rotation period.
fn simulate_rotations(start: i64, period_secs: i64, window_secs: i64) -> Vec<i64> {
    let mut sched = KeyRotationSchedule::new(
        "content-rotation-test",
        RotationInterval::EverySeconds(period_secs as u64),
        TransitionWindow::of_secs(0),
    )
    .with_max_history(100);

    let mut events: Vec<i64> = Vec::new();
    let end = start + window_secs;
    let mut now = start;

    while now <= end {
        if sched.rotation_due(now) {
            let v = sched.active_version().unwrap_or(0) + 1;
            sched.add_key(vec![v as u8; 16], vec![v as u8; 16], now);
            events.push(now);
        }
        now += 1;
    }

    events
}

#[test]
fn five_second_period_over_thirty_seconds_emits_seven_keys() {
    // Window includes t=0, 5, 10, 15, 20, 25, 30 → 7 rotations
    // (first one at start because last_rotation is None).
    let events = simulate_rotations(0, 5, 30);
    assert_eq!(
        events.len(),
        7,
        "expected 7 rotation events (t=0..30 step 5)"
    );
    assert_eq!(events, vec![0, 5, 10, 15, 20, 25, 30]);
}

#[test]
fn rotation_boundary_alignment_is_exact() {
    // Verify rotation_due fires exactly at the boundary, not before.
    let mut sched = KeyRotationSchedule::new(
        "alignment-test",
        RotationInterval::EverySeconds(5),
        TransitionWindow::none(),
    );

    sched.add_key(vec![1u8; 16], vec![1u8; 16], 100);
    assert!(!sched.rotation_due(101), "not due 1s after rotation");
    assert!(!sched.rotation_due(104), "not due 4s after rotation");
    assert!(sched.rotation_due(105), "due exactly 5s after rotation");
    assert!(sched.rotation_due(200), "still due 100s after rotation");
}

#[test]
fn graceful_transition_overlap_keeps_old_key_valid() {
    // 5s rotation, 3s overlap. After 5s rotation, both old and new keys
    // should be valid for the next 3 seconds, then only the new one.
    let mut sched = KeyRotationSchedule::new(
        "overlap-test",
        RotationInterval::EverySeconds(5),
        TransitionWindow::of_secs(3),
    );

    sched.add_key(vec![1u8; 16], vec![1u8; 16], 1000);
    sched.add_key(vec![2u8; 16], vec![2u8; 16], 1005);

    // Both keys valid during overlap window (1005..=1008)
    let during = sched.keys_valid_at(1006);
    assert_eq!(during.len(), 2, "both keys valid during overlap");

    // Only new key valid after overlap expires
    let after = sched.keys_valid_at(1009);
    assert_eq!(after.len(), 1, "only new key valid after overlap");
    assert_eq!(after[0].version, 2);
}

#[test]
fn manual_rotation_never_triggers_automatically() {
    let sched = KeyRotationSchedule::new(
        "manual-only",
        RotationInterval::Manual,
        TransitionWindow::none(),
    );
    assert!(!sched.rotation_due(0));
    assert!(!sched.rotation_due(i64::MAX));
}

#[test]
fn versioned_keys_strictly_monotonic() {
    let mut sched = KeyRotationSchedule::new(
        "version-monotonic",
        RotationInterval::EverySeconds(10),
        TransitionWindow::none(),
    )
    .with_max_history(100);

    for i in 0i64..5 {
        sched.add_key(vec![i as u8; 16], vec![i as u8; 16], 1000 + i * 10);
    }

    // History retains all 5 keys; versions increase 1..=5
    assert_eq!(sched.key_count(), 5);
    assert_eq!(sched.active_version(), Some(5));

    // The "valid_at" snapshot pulls a single key once overlap expires.
    let one_key = sched.keys_valid_at(1041);
    let versions: Vec<u32> = one_key.iter().map(|k: &&VersionedKey| k.version).collect();
    assert!(versions.contains(&5), "active version-5 always valid");
}

#[test]
fn max_history_caps_retained_keys() {
    let mut sched = KeyRotationSchedule::new(
        "history-cap",
        RotationInterval::Manual,
        TransitionWindow::none(),
    )
    .with_max_history(3);

    for i in 0u8..6 {
        sched.add_key(vec![i; 16], vec![i; 16], i64::from(i) * 1000);
    }
    assert_eq!(sched.key_count(), 3, "only the 3 most recent keys remain");
}
