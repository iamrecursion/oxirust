//! Integration tests for NestedSequence resolution field.

use oximedia_core::Rational;
use oximedia_edit::nested_sequence::{NestedSequence, SequenceId};

#[test]
fn test_nested_sequence_has_resolution_field() {
    let mut seq = NestedSequence::new(SequenceId(1), "Clip", Rational::new(24, 1));
    seq.resolution = (1280, 720);
    assert_eq!(seq.resolution, (1280, 720));
}

#[test]
fn test_nested_sequence_default_resolution_is_1080p() {
    let seq = NestedSequence::new(SequenceId(2), "HD Clip", Rational::new(25, 1));
    assert_eq!(seq.resolution, (1920, 1080));
}

#[test]
fn test_nested_sequence_with_resolution_builder() {
    let seq = NestedSequence::new(SequenceId(3), "UHD Clip", Rational::new(60, 1))
        .with_resolution(3840, 2160);
    assert_eq!(seq.resolution, (3840, 2160));
}

#[test]
fn test_nested_sequence_resolution_4k() {
    let seq =
        NestedSequence::new(SequenceId(4), "4K", Rational::new(24, 1)).with_resolution(4096, 2160);
    assert_eq!(seq.resolution.0, 4096);
    assert_eq!(seq.resolution.1, 2160);
}
