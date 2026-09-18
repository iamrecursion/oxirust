//! Integration tests for NestedSequence::duration_in_outer_timebase (ConformMethod).

use oximedia_core::Rational;
use oximedia_edit::nested_sequence::{NestedSequence, SequenceId};

#[test]
fn test_24fps_in_60fps_outer_duration() {
    // inner: 24 frames at 24 fps = 1 second
    // outer: 60 fps → expect 60 outer frames
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(24, 1)).with_duration(24);
    let outer = seq.duration_in_outer_timebase(Rational::new(60, 1));
    assert_eq!(outer, 60);
}

#[test]
fn test_duration_unchanged_when_same_rate() {
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(25, 1)).with_duration(100);
    let outer = seq.duration_in_outer_timebase(Rational::new(25, 1));
    assert_eq!(outer, 100);
}

#[test]
fn test_30fps_to_24fps() {
    // 30 frames at 30 fps = 1 second → 24 outer frames at 24 fps
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(30, 1)).with_duration(30);
    let outer = seq.duration_in_outer_timebase(Rational::new(24, 1));
    assert_eq!(outer, 24);
}

#[test]
fn test_25fps_to_50fps() {
    // 25 frames at 25 fps = 1 second → 50 outer frames at 50 fps
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(25, 1)).with_duration(25);
    let outer = seq.duration_in_outer_timebase(Rational::new(50, 1));
    assert_eq!(outer, 50);
}

#[test]
fn test_two_seconds_at_different_rates() {
    // 2 seconds of 24 fps content = 48 inner frames → at 30 fps = 60 outer frames
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(24, 1)).with_duration(48);
    let outer = seq.duration_in_outer_timebase(Rational::new(30, 1));
    assert_eq!(outer, 60);
}

#[test]
fn test_zero_inner_duration() {
    let seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(24, 1)).with_duration(0);
    assert_eq!(seq.duration_in_outer_timebase(Rational::new(60, 1)), 0);
}
