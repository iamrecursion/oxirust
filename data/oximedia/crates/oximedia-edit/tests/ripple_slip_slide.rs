//! Integration tests: ripple editing and slip/slide operations.
//!
//! These tests exercise the free functions in `oximedia_edit::ripple` and the
//! structs in `oximedia_edit::slip_slide`.  Both APIs operate on
//! `Vec<TimelineClip>` / dedicated structs rather than on the main `Timeline`.

use oximedia_edit::ripple::{
    ripple_delete, ripple_insert, ripple_trim_left, ripple_trim_right, three_point_edit,
    RippleMode, TimelineClip,
};
use oximedia_edit::slip_slide::{EditValidator, ExtendEdit, SlideEdit, SlipEdit};

// ── Helper ────────────────────────────────────────────────────────────────────

fn make(id: u64, start: u64, duration: u64, track: u32) -> TimelineClip {
    TimelineClip::new(id, start, duration, 0, track)
}

// ── ripple_delete ─────────────────────────────────────────────────────────────

#[test]
fn test_ripple_delete_shifts_downstream_clips() {
    let mut clips = vec![
        make(1, 0, 100, 0),
        make(2, 200, 100, 0),
        make(3, 400, 100, 0),
    ];
    // Delete [100, 200): clip 1 ends at 100 so untouched; clips 2,3 shift left by 100.
    ripple_delete(&mut clips, 100, 200, 0);
    assert_eq!(clips[0].start, 0, "clip 1 must not move");
    // Clip 2 starts at 200 which is ≥ end(200) so shifted left by 100 → 100.
    assert_eq!(clips[1].start, 100, "clip 2 must shift left to 100");
    assert_eq!(clips[2].start, 300, "clip 3 must shift left to 300");
}

#[test]
fn test_ripple_delete_removes_clips_inside_range() {
    let mut clips = vec![
        make(1, 0, 50, 0),
        make(2, 50, 100, 0), // fully inside [50, 150)
        make(3, 150, 50, 0),
    ];
    ripple_delete(&mut clips, 50, 150, 0);
    assert!(
        !clips.iter().any(|c| c.id == 2),
        "clip inside range must be removed"
    );
    assert_eq!(clips.len(), 2);
}

#[test]
fn test_ripple_delete_other_track_unaffected() {
    let mut clips = vec![
        make(1, 100, 100, 0), // track 0
        make(2, 100, 100, 1), // track 1 — must not be touched
    ];
    ripple_delete(&mut clips, 0, 50, 0);
    let c2 = clips.iter().find(|c| c.id == 2).expect("clip 2 must exist");
    assert_eq!(c2.start, 100, "track-1 clip must not be shifted");
}

#[test]
fn test_ripple_delete_returns_deleted_duration() {
    let mut clips = vec![make(1, 500, 200, 0)];
    let deleted = ripple_delete(&mut clips, 0, 75, 0);
    assert_eq!(deleted, 75);
}

// ── ripple_insert ─────────────────────────────────────────────────────────────

#[test]
fn test_ripple_insert_pushes_clips_right() {
    let mut clips = vec![make(1, 0, 100, 0), make(2, 100, 100, 0)];
    // Insert 50 units at position 100.
    ripple_insert(&mut clips, 100, 50, 0);
    assert_eq!(clips[0].start, 0, "clip before insert_at must not move");
    assert_eq!(
        clips[1].start, 150,
        "clip at insert_at must shift right by 50"
    );
}

#[test]
fn test_ripple_insert_at_timeline_start() {
    // Inserting at position 0 must shift ALL clips on the same track.
    let mut clips = vec![make(1, 0, 100, 0), make(2, 100, 100, 0)];
    ripple_insert(&mut clips, 0, 200, 0);
    assert_eq!(clips[0].start, 200);
    assert_eq!(clips[1].start, 300);
}

// ── ripple_trim_right ─────────────────────────────────────────────────────────

#[test]
fn test_ripple_trim_right_shortens_and_shifts_followers() {
    let mut clips = vec![make(1, 0, 200, 0), make(2, 200, 100, 0)];
    // Trim clip 1's right edge from 200 → 150 (shrink by 50).
    ripple_trim_right(&mut clips, 1, 150).expect("ripple_trim_right failed");
    assert_eq!(clips[0].duration, 150, "clip 1 must be 150 units long");
    assert_eq!(clips[1].start, 150, "clip 2 must shift left by 50");
}

#[test]
fn test_ripple_trim_right_extend_pushes_followers() {
    let mut clips = vec![make(1, 0, 100, 0), make(2, 100, 100, 0)];
    // Extend clip 1's right edge from 100 → 150.
    ripple_trim_right(&mut clips, 1, 150).expect("ripple_trim_right failed");
    assert_eq!(clips[0].duration, 150);
    assert_eq!(clips[1].start, 150, "clip 2 must be pushed right by 50");
}

#[test]
fn test_ripple_trim_right_unknown_clip_errors() {
    let mut clips = vec![make(1, 0, 100, 0)];
    let result = ripple_trim_right(&mut clips, 99, 50);
    assert!(result.is_err());
}

// ── ripple_trim_left ──────────────────────────────────────────────────────────

#[test]
fn test_ripple_trim_left_trims_in_point() {
    let mut clips = vec![make(1, 0, 200, 0)];
    ripple_trim_left(&mut clips, 1, 50).expect("ripple_trim_left failed");
    assert_eq!(clips[0].start, 50, "clip must start later");
    assert_eq!(clips[0].duration, 150, "clip must be shorter");
    assert_eq!(clips[0].source_in, 50, "source in-point must advance by 50");
}

#[test]
fn test_ripple_trim_left_unknown_clip_errors() {
    let mut clips = vec![make(1, 0, 100, 0)];
    let result = ripple_trim_left(&mut clips, 99, 10);
    assert!(result.is_err());
}

// ── three_point_edit ──────────────────────────────────────────────────────────

#[test]
fn test_three_point_edit_insert_shifts_existing() {
    let mut clips = vec![make(1, 100, 100, 0)];
    let new_clip = three_point_edit(&mut clips, 0, 50, 0, RippleMode::Insert)
        .expect("three_point_edit failed");
    assert_eq!(new_clip.duration, 50);
    let orig = clips
        .iter()
        .find(|c| c.id == 1)
        .expect("original must exist");
    assert_eq!(orig.start, 150, "existing clip must shift right by 50");
}

#[test]
fn test_three_point_edit_overwrite_no_shift() {
    let mut clips = vec![make(1, 100, 100, 0)];
    three_point_edit(&mut clips, 0, 50, 0, RippleMode::Overwrite).expect("three_point_edit failed");
    let orig = clips
        .iter()
        .find(|c| c.id == 1)
        .expect("original must exist");
    assert_eq!(orig.start, 100, "overwrite must not shift existing clip");
}

#[test]
fn test_three_point_edit_invalid_range_errors() {
    let mut clips: Vec<TimelineClip> = Vec::new();
    let result = three_point_edit(&mut clips, 100, 50, 0, RippleMode::Insert);
    assert!(result.is_err(), "src_out < src_in must error");
}

// ── SlipEdit ──────────────────────────────────────────────────────────────────

#[test]
fn test_slip_edit_forward_shifts_in_out() {
    let edit = SlipEdit::new(1, 10);
    let result = edit.apply_to(20, 80, 100);
    assert_eq!(result, Some((30, 90)));
}

#[test]
fn test_slip_edit_backward_shifts_in_out() {
    let edit = SlipEdit::new(1, -10);
    let result = edit.apply_to(20, 80, 100);
    assert_eq!(result, Some((10, 70)));
}

#[test]
fn test_slip_edit_out_of_bounds_returns_none() {
    // new_out would be 110 which exceeds duration 100.
    let edit = SlipEdit::new(1, 30);
    assert!(edit.apply_to(20, 80, 100).is_none());
}

#[test]
fn test_slip_edit_negative_in_returns_none() {
    let edit = SlipEdit::new(1, -25);
    assert!(edit.apply_to(20, 80, 100).is_none());
}

#[test]
fn test_slip_edit_zero_offset_identity() {
    let edit = SlipEdit::new(42, 0);
    let result = edit.apply_to(10, 50, 100);
    assert_eq!(result, Some((10, 50)));
}

// ── SlideEdit ────────────────────────────────────────────────────────────────

#[test]
fn test_slide_edit_forward_is_positive() {
    let edit = SlideEdit::new(5, 20);
    assert!(edit.is_forward());
}

#[test]
fn test_slide_edit_backward_not_forward() {
    let edit = SlideEdit::new(5, -20);
    assert!(!edit.is_forward());
}

#[test]
fn test_edit_validator_slip_valid() {
    let edit = SlipEdit::new(1, 5);
    assert!(EditValidator::validate_slip(&edit, 10, 50, 100));
}

#[test]
fn test_edit_validator_slip_invalid_exceeds_duration() {
    let edit = SlipEdit::new(1, 60);
    assert!(!EditValidator::validate_slip(&edit, 10, 50, 100));
}

#[test]
fn test_edit_validator_slide_valid() {
    let edit = SlideEdit::new(1, 50);
    assert!(EditValidator::validate_slide(&edit, 0, 1000));
}

#[test]
fn test_edit_validator_slide_negative_start_invalid() {
    let edit = SlideEdit::new(1, -200);
    assert!(!EditValidator::validate_slide(&edit, 0, 1000));
}

// ── ExtendEdit ───────────────────────────────────────────────────────────────

#[test]
fn test_extend_edit_positive_change() {
    let edit = ExtendEdit::new(1, 200);
    assert_eq!(edit.duration_change(150), 50);
}

#[test]
fn test_extend_edit_negative_change() {
    let edit = ExtendEdit::new(1, 100);
    assert_eq!(edit.duration_change(150), -50);
}

#[test]
fn test_extend_edit_zero_change() {
    let edit = ExtendEdit::new(1, 100);
    assert_eq!(edit.duration_change(100), 0);
}
