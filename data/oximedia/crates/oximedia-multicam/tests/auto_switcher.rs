//! Integration tests pinning the automatic angle-selection logic.
//!
//! These tests exercise [`oximedia_multicam::auto::AutoSwitcher`] end-to-end:
//! score → switch decision, the confidence gate, and the minimum-hold
//! (hysteresis / cooldown) behaviour. All inputs are explicit and
//! deterministic — no RNG — and the expected confidence values are derived
//! directly from the documented scoring math.
//!
//! ## Reference scoring math (default weights)
//!
//! Default [`ScoringWeights`]: face = 1.0, comp = 0.8, audio = 1.2,
//! motion = 0.6, speaker = 1.5 (weight sum = 5.1). `calculate_total` divides
//! the weighted sum by the weight sum, so every total lands in `[0, 1]`.
//!
//! * A **dataless** angle (no face / audio / motion data injected) scores the
//!   per-criterion defaults face 0.5, comp 0.5, audio 0.3, motion 0.3,
//!   speaker 0.3, giving
//!   `(0.5·1.0 + 0.5·0.8 + 0.3·1.2 + 0.3·0.6 + 0.3·1.5) / 5.1`
//!   `= 1.89 / 5.1 ≈ 0.37058824` — below the 0.7 default `min_confidence`, so a
//!   dataless angle is never eligible to be switched to on its own.
//! * An angle with strong audio + a known speaker, i.e.
//!   `add_audio_data(angle, -10.0, true, Some(0))` (and no face/motion data, so
//!   those keep their defaults), scores
//!   audio = `((-10 + 60)/60 + 1.0)/2 = 0.91666669`, speaker = 1.0, face 0.5,
//!   comp 0.5, motion 0.3, giving
//!   `(0.5·1.0 + 0.5·0.8 + 0.91666669·1.2 + 0.3·0.6 + 1.0·1.5) / 5.1`
//!   `= 3.68 / 5.1 ≈ 0.72156864` — above 0.7, so it is eligible.
//!
//! ## Tie-break behaviour
//!
//! The switcher picks the best angle with `iter().enumerate().max_by(..)`.
//! `Iterator::max_by` returns the **last** element among equal maxima, so when
//! two angles carry identical strong data the switcher selects the
//! **highest-index** angle, not the lowest. `switcher_tie_breaks_to_highest_index`
//! pins exactly that.

use oximedia_multicam::auto::{AutoSwitcher, SelectionCriteria};

/// Expected total score for an angle carrying only strong audio + speaker data,
/// computed from the default scoring weights (see module docs).
///
/// `(0.5·1.0 + 0.5·0.8 + 0.91666669·1.2 + 0.3·0.6 + 1.0·1.5) / 5.1`.
const STRONG_AUDIO_TOTAL: f32 = {
    let weighted =
        0.5 * 1.0 + 0.5 * 0.8 + ((-10.0 + 60.0) / 60.0 + 1.0) / 2.0 * 1.2 + 0.3 * 0.6 + 1.0 * 1.5;
    weighted / 5.1
};

/// Expected total score for a fully dataless angle (see module docs):
/// `(0.5·1.0 + 0.5·0.8 + 0.3·1.2 + 0.3·0.6 + 0.3·1.5) / 5.1`.
const DATALESS_TOTAL: f32 = (0.5 * 1.0 + 0.5 * 0.8 + 0.3 * 1.2 + 0.3 * 0.6 + 0.3 * 1.5) / 5.1;

/// Inject strong audio + a known speaker on `angle` of `switcher`'s scorer.
fn make_angle_strong(switcher: &mut AutoSwitcher, angle: usize) {
    switcher
        .scorer_mut()
        .add_audio_data(angle, -10.0, true, Some(0));
}

/// With the hold cooldown disabled, the switcher must pick the single angle
/// whose score clears the confidence gate, update `current_angle`, and record
/// the exact computed confidence.
#[test]
fn switcher_selects_highest_scoring_angle() {
    let mut switcher = AutoSwitcher::new();
    switcher.set_min_hold_frames(0);

    // Angle 2 is the only one with strong data; 0 and 1 stay dataless.
    make_angle_strong(&mut switcher, 2);

    let criteria = SelectionCriteria::default();
    let selected = switcher
        .select_angle(0, 3, &criteria)
        .expect("select_angle should succeed");

    assert_eq!(selected, 2, "strongest angle (2) must be selected");
    assert_eq!(
        switcher.current_angle(),
        2,
        "current_angle must follow the selection"
    );

    // The recorded confidence is the winning angle's total score.
    let record = switcher
        .history()
        .last()
        .expect("a selection record must be present");
    assert_eq!(record.angle, 2, "recorded angle must be 2");
    assert!(
        (record.confidence - STRONG_AUDIO_TOTAL).abs() < 1e-4,
        "recorded confidence {} should match computed {STRONG_AUDIO_TOTAL}",
        record.confidence
    );
    // Sanity: the computed strong total really is above the gate, and the
    // dataless total really is below it.
    assert!(
        STRONG_AUDIO_TOTAL >= criteria.min_confidence,
        "strong total {STRONG_AUDIO_TOTAL} must clear the {} gate",
        criteria.min_confidence
    );
    assert!(
        DATALESS_TOTAL < criteria.min_confidence,
        "dataless total {DATALESS_TOTAL} must be below the {} gate",
        criteria.min_confidence
    );
}

/// When no angle clears the confidence gate (all dataless), the switcher must
/// hold on the initial angle 0 even though the cooldown is disabled.
#[test]
fn switcher_holds_when_no_angle_meets_confidence() {
    let mut switcher = AutoSwitcher::new();
    switcher.set_min_hold_frames(0);

    let criteria = SelectionCriteria::default();
    let selected = switcher
        .select_angle(0, 3, &criteria)
        .expect("select_angle should succeed");

    assert_eq!(
        selected, 0,
        "no angle clears the gate → hold on initial angle 0"
    );
    assert_eq!(switcher.current_angle(), 0);

    // The best dataless score is below the gate, so confidence is recorded but
    // no switch is committed.
    let record = switcher
        .history()
        .last()
        .expect("a selection record must be present");
    assert!(
        record.confidence < criteria.min_confidence,
        "best dataless confidence {} must be below the {} gate",
        record.confidence,
        criteria.min_confidence
    );
}

/// The default 50-frame minimum-hold must short-circuit scoring before the hold
/// expires, then permit the switch once it does, and finally hold the new angle
/// for its own cooldown window.
#[test]
fn switcher_respects_hold_cooldown() {
    let mut switcher = AutoSwitcher::new();
    // Keep the default min_hold_frames (50).
    make_angle_strong(&mut switcher, 2);

    let criteria = SelectionCriteria::default();

    // Frame 10 < last_selection_frame(0) + 50 → short-circuit, returns current
    // angle 0 WITHOUT scoring. No history is recorded on a short-circuit.
    let at_10 = switcher
        .select_angle(10, 3, &criteria)
        .expect("select_angle should succeed");
    assert_eq!(at_10, 0, "frame 10 is inside the hold window → stay on 0");
    assert!(
        switcher.history().is_empty(),
        "a short-circuited call records no history"
    );

    // Frame 60 >= 0 + 50 → scoring runs and the switch to angle 2 commits.
    let at_60 = switcher
        .select_angle(60, 3, &criteria)
        .expect("select_angle should succeed");
    assert_eq!(at_60, 2, "frame 60 clears the hold → switch to 2");
    assert_eq!(switcher.current_angle(), 2);

    // Frame 70 < last_selection_frame(60) + 50 → short-circuit, holds on 2.
    let at_70 = switcher
        .select_angle(70, 3, &criteria)
        .expect("select_angle should succeed");
    assert_eq!(at_70, 2, "frame 70 is inside the new hold → hold on 2");
    assert_eq!(switcher.current_angle(), 2);
}

/// When two angles carry identical strong data, the switcher selects the
/// **highest** index, because `Iterator::max_by` returns the last element among
/// equal maxima. Documented here so a future change to the tie-break is caught.
#[test]
fn switcher_tie_breaks_to_highest_index() {
    let mut switcher = AutoSwitcher::new();
    switcher.set_min_hold_frames(0);

    // Angles 1 AND 2 are equally strong; angle 0 stays dataless.
    make_angle_strong(&mut switcher, 1);
    make_angle_strong(&mut switcher, 2);

    let criteria = SelectionCriteria::default();
    let selected = switcher
        .select_angle(0, 3, &criteria)
        .expect("select_angle should succeed");

    assert_eq!(
        selected, 2,
        "equal maxima → max_by keeps the last (highest-index) angle"
    );
    assert_eq!(switcher.current_angle(), 2);

    // Both candidates scored the same strong total.
    let record = switcher
        .history()
        .last()
        .expect("a selection record must be present");
    assert!(
        (record.scores[1].total_score - record.scores[2].total_score).abs() < 1e-6,
        "angles 1 and 2 must score identically (got {} vs {})",
        record.scores[1].total_score,
        record.scores[2].total_score
    );
    assert!(
        (record.confidence - STRONG_AUDIO_TOTAL).abs() < 1e-4,
        "tie confidence {} should match computed {STRONG_AUDIO_TOTAL}",
        record.confidence
    );
}
