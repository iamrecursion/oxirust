//! Integration tests for i64 position fields in NestedSequenceRef.

use oximedia_core::Rational;
use oximedia_edit::nested_sequence::{NestedSequence, NestedSequenceRef, SequenceId};

#[test]
fn test_negative_position_allowed() {
    // i64 positions allow negative values (pre-roll before timeline start).
    let mut r = NestedSequenceRef::new(SequenceId(1), 0, 0, 100);
    r.position = -24; // 1 second before timeline start at 24 fps
    assert_eq!(r.position, -24_i64);
}

#[test]
fn test_negative_position_end_position() {
    // pre-roll: placed at -24, duration 48 → ends at 24
    let r = NestedSequenceRef::new(SequenceId(1), -24, 0, 48);
    assert_eq!(r.end_position(), 24_i64);
}

#[test]
fn test_zero_position() {
    let r = NestedSequenceRef::new(SequenceId(1), 0, 0, 100);
    assert_eq!(r.position, 0_i64);
    assert_eq!(r.end_position(), 100_i64);
}

#[test]
fn test_large_positive_position() {
    // Timeline positions can be large (e.g., late in a long project)
    let r = NestedSequenceRef::new(SequenceId(1), 86_400 * 30, 0, 1000);
    assert_eq!(r.position, 86_400 * 30_i64);
}

#[test]
fn test_nested_sequence_duration_field_is_i64() {
    let seq =
        NestedSequence::new(SequenceId(1), "S", Rational::new(24, 1)).with_duration(i64::MAX / 2);
    assert_eq!(seq.duration, i64::MAX / 2);
}

#[test]
fn test_in_out_points_i64() {
    let r = NestedSequenceRef::new(SequenceId(1), 0, 100, 200);
    assert_eq!(r.in_point, 100_i64);
    assert_eq!(r.out_point, 200_i64);
    assert_eq!(r.duration(), 100_i64);
}
