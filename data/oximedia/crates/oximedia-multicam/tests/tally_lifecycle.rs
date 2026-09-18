//! Integration tests for the tally light controller lifecycle.
//!
//! Wave 29 / Slice 8 — PURE test-hardening of `tally_system`.
//!
//! Verifies the preview → program → off transition lifecycle, history
//! capture, the single-program bus invariant, and clock-driven timestamps.
//! Oracles are derived directly from `src/tally_system.rs`.

use oximedia_multicam::tally_system::{TallyColor, TallyController, TallyState};

/// A freshly registered angle starts in the Off state.
#[test]
fn test_registered_angle_starts_off() {
    let mut ctrl = TallyController::new();
    ctrl.register_angle(0);

    assert_eq!(ctrl.get_state(0), TallyState::off());
    assert_eq!(ctrl.get_state(0).front, TallyColor::Off);
    assert!(!ctrl.get_state(0).is_active());
    assert!(ctrl.program_angles().is_empty());
    assert!(ctrl.preview_angles().is_empty());
}

/// Full preview → program → off lifecycle for a single angle, with the bus
/// membership and history invariants checked at each step.
#[test]
fn test_preview_program_off_lifecycle() {
    let mut ctrl = TallyController::new();
    ctrl.register_angle(0);

    // --- preview ---
    ctrl.set_preview(0);
    assert_eq!(ctrl.get_state(0), TallyState::preview());
    assert_eq!(ctrl.get_state(0).front, TallyColor::Green);
    assert_eq!(ctrl.preview_angles(), vec![0]);
    assert!(ctrl.program_angles().is_empty());

    // --- program ---
    ctrl.set_program(0);
    assert_eq!(ctrl.get_state(0), TallyState::program());
    assert_eq!(ctrl.get_state(0).front, TallyColor::Red);
    assert_eq!(ctrl.program_angles(), vec![0]);
    assert!(ctrl.preview_angles().is_empty());

    // --- off ---
    ctrl.set_off(0);
    assert_eq!(ctrl.get_state(0), TallyState::off());
    assert!(!ctrl.get_state(0).is_active());
    assert!(ctrl.program_angles().is_empty());
    assert!(ctrl.preview_angles().is_empty());

    // --- history: three distinct transitions ---
    let history = ctrl.history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].current, TallyState::preview());
    assert_eq!(history[1].current, TallyState::program());
    assert_eq!(history[2].current, TallyState::off());
    // The last transition's previous state must be the program state.
    assert_eq!(history[2].previous, TallyState::program());
    // angle_id is propagated.
    assert!(history.iter().all(|e| e.angle_id == 0));
}

/// Setting the same state twice does not append a redundant history event.
#[test]
fn test_no_history_on_redundant_state() {
    let mut ctrl = TallyController::new();
    ctrl.register_angle(0);

    ctrl.set_program(0);
    ctrl.set_program(0); // identical → no new event
    assert_eq!(ctrl.history().len(), 1);
}

/// The bus update enforces exactly one program and one preview angle; all
/// other registered angles are forced off.
#[test]
fn test_bus_single_program_invariant() {
    let mut ctrl = TallyController::new();
    for a in 0..4 {
        ctrl.register_angle(a);
    }

    ctrl.update_from_buses(2, 3);

    // Exactly one program (angle 2) and one preview (angle 3).
    assert_eq!(ctrl.program_angles(), vec![2]);
    assert_eq!(ctrl.preview_angles(), vec![3]);
    assert_eq!(ctrl.program_angles().len(), 1);
    assert_eq!(ctrl.preview_angles().len(), 1);
    assert_eq!(ctrl.get_state(0), TallyState::off());
    assert_eq!(ctrl.get_state(1), TallyState::off());
    assert_eq!(ctrl.get_state(2), TallyState::program());
    assert_eq!(ctrl.get_state(3), TallyState::preview());
}

/// A second bus update re-points program/preview and clears the prior program.
#[test]
fn test_bus_update_repoints_and_clears_prior() {
    let mut ctrl = TallyController::new();
    for a in 0..4 {
        ctrl.register_angle(a);
    }

    ctrl.update_from_buses(2, 3);
    ctrl.update_from_buses(3, 0);

    // Program moves to 3, preview to 0; angle 2 (old program) is now off.
    assert_eq!(ctrl.program_angles(), vec![3]);
    assert_eq!(ctrl.preview_angles(), vec![0]);
    assert_eq!(ctrl.program_angles().len(), 1);
    assert_eq!(ctrl.preview_angles().len(), 1);
    assert_eq!(ctrl.get_state(2), TallyState::off());
    assert_eq!(ctrl.get_state(1), TallyState::off());
    assert_eq!(ctrl.get_state(3), TallyState::program());
    assert_eq!(ctrl.get_state(0), TallyState::preview());
}

/// Events are timestamped with the advancing monotonic clock.
#[test]
fn test_clock_driven_timestamps() {
    let mut ctrl = TallyController::new();
    ctrl.register_angle(0);

    ctrl.advance_clock(5000);
    ctrl.set_program(0); // event 0 @ 5000us
    ctrl.advance_clock(1000);
    ctrl.set_off(0); // event 1 @ 6000us

    let history = ctrl.history();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].timestamp_us, 5000);
    assert_eq!(history[1].timestamp_us, 6000);
}
