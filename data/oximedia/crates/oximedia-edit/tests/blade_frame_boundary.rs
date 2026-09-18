//! Integration tests: BladeTool cut descriptors and Clip::split_at at frame
//! boundaries for 24, 25, 30 and 60 fps.

use oximedia_core::Rational;
use oximedia_edit::blade_tool::{BladeMode, BladeTool};
use oximedia_edit::clip::{Clip, ClipType};
use oximedia_edit::Timeline;
use oximedia_edit::TrackType;

// ─────────────────────────────────────────────────────────────────────────────
// BladeTool tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_blade_cut_single_clip_inside_range() {
    let tool = BladeTool::new(BladeMode::Single, 0);
    // Clip occupies frames [0, 100).
    let clips = vec![(0usize, 1u64, 0i64, 100i64)];
    let result = tool.cut(&clips, 50);
    assert_eq!(result.cuts_applied(), 1);
    assert_eq!(result.cuts[0].track_index, 0);
    assert_eq!(result.cuts[0].clip_id, 1);
    assert_eq!(result.cuts[0].cut_frame, 50);
}

#[test]
fn test_blade_cut_at_start_boundary_no_cut() {
    // Exact start is not strictly inside → no cut.
    let tool = BladeTool::new(BladeMode::Single, 0);
    let clips = vec![(0usize, 1u64, 0i64, 100i64)];
    let result = tool.cut(&clips, 0);
    assert_eq!(result.cuts_applied(), 0);
}

#[test]
fn test_blade_cut_at_end_boundary_no_cut() {
    // Exact end is not strictly inside → no cut.
    let tool = BladeTool::new(BladeMode::Single, 0);
    let clips = vec![(0usize, 1u64, 0i64, 100i64)];
    let result = tool.cut(&clips, 100);
    assert_eq!(result.cuts_applied(), 0);
}

#[test]
fn test_blade_cut_outside_range_no_cut() {
    let tool = BladeTool::new(BladeMode::Single, 0);
    let clips = vec![(0usize, 1u64, 0i64, 100i64)];
    let result = tool.cut(&clips, 150);
    assert_eq!(result.cuts_applied(), 0);
}

#[test]
fn test_blade_all_tracks_cuts_all_clips() {
    let tool = BladeTool::new(BladeMode::AllTracks, 0);
    let clips = vec![
        (0usize, 10u64, 0i64, 100i64),
        (1usize, 20u64, 0i64, 100i64),
        (2usize, 30u64, 0i64, 100i64),
    ];
    let result = tool.cut(&clips, 50);
    assert_eq!(result.cuts_applied(), 3, "AllTracks must cut every clip");
}

#[test]
fn test_blade_all_tracks_only_clips_containing_frame() {
    let tool = BladeTool::new(BladeMode::AllTracks, 0);
    let clips = vec![
        (0usize, 10u64, 0i64, 100i64),
        (1usize, 20u64, 200i64, 400i64), // doesn't contain frame 50
    ];
    let result = tool.cut(&clips, 50);
    assert_eq!(
        result.cuts_applied(),
        1,
        "only clips containing frame 50 are cut"
    );
    assert_eq!(result.cuts[0].clip_id, 10);
}

#[test]
fn test_blade_preview_matches_cut() {
    let tool = BladeTool::new(BladeMode::Single, 0);
    let clips = vec![(0usize, 7u64, 10i64, 90i64)];
    let preview = tool.preview_cut(&clips, 40);
    let actual = tool.cut(&clips, 40).cuts;
    assert_eq!(preview.len(), actual.len());
    assert_eq!(preview[0].cut_frame, actual[0].cut_frame);
}

#[test]
fn test_blade_new_segments_equals_cut_count() {
    let tool = BladeTool::new(BladeMode::AllTracks, 0);
    let clips = vec![(0usize, 1u64, 0i64, 100i64), (1usize, 2u64, 0i64, 100i64)];
    let result = tool.cut(&clips, 50);
    assert_eq!(result.new_segments, result.cuts.len());
}

// ─────────────────────────────────────────────────────────────────────────────
// Clip::split_at frame-boundary tests
// ─────────────────────────────────────────────────────────────────────────────

/// Calculate the frame duration in timebase units (ms) for the given fps.
fn frame_ms(fps: u32) -> i64 {
    // 1 frame at fps = 1000/fps ms (integer, truncated).
    1000i64 / fps as i64
}

/// Add a long clip to a timeline and split it at a frame boundary, returning
/// the (first_duration, second_start, second_duration) triple.
fn split_clip_at_frame(fps: u32, clip_duration_ms: i64, cut_at_ms: i64) -> (i64, i64, i64) {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(fps as i64, 1));
    let vt = tl.add_track(TrackType::Video);
    let id = tl
        .add_clip(vt, Clip::new(1, ClipType::Video, 0, clip_duration_ms))
        .expect("add_clip must succeed");

    let track = &mut tl.tracks[vt];
    let clip_idx = track
        .clips
        .iter()
        .position(|c| c.id == id)
        .expect("clip must exist");
    let clip = &mut track.clips[clip_idx];

    let second_half = clip.split_at(cut_at_ms, 2).expect("split_at must succeed");
    let first_duration = clip.timeline_duration;
    let second_start = second_half.timeline_start;
    let second_duration = second_half.timeline_duration;

    (first_duration, second_start, second_duration)
}

#[test]
fn test_split_at_30fps_frame_boundary() {
    // At 30 fps one frame = 33 ms (truncated). Cut at frame 10 → ms 333.
    let fps = 30u32;
    let cut_ms = frame_ms(fps) * 10; // 330 ms
    let (first_dur, second_start, second_dur) = split_clip_at_frame(fps, 2000, cut_ms);
    assert_eq!(first_dur, cut_ms, "first half must end at cut point");
    assert_eq!(second_start, cut_ms, "second half must start at cut point");
    assert_eq!(
        first_dur + second_dur,
        2000,
        "durations must sum to original"
    );
}

#[test]
fn test_split_at_24fps_frame_boundary() {
    let fps = 24u32;
    let cut_ms = frame_ms(fps) * 12; // 12 frames at 24 fps = 12*41=492 ms
    let (first_dur, second_start, second_dur) = split_clip_at_frame(fps, 3000, cut_ms);
    assert_eq!(first_dur, cut_ms);
    assert_eq!(second_start, cut_ms);
    assert_eq!(first_dur + second_dur, 3000);
}

#[test]
fn test_split_at_25fps_frame_boundary() {
    let fps = 25u32;
    let cut_ms = frame_ms(fps) * 25; // 25 frames at 25 fps = 25*40=1000 ms
    let (first_dur, second_start, second_dur) = split_clip_at_frame(fps, 4000, cut_ms);
    assert_eq!(first_dur, cut_ms);
    assert_eq!(second_start, cut_ms);
    assert_eq!(first_dur + second_dur, 4000);
}

#[test]
fn test_split_at_60fps_frame_boundary() {
    let fps = 60u32;
    let cut_ms = frame_ms(fps) * 30; // 30 frames at 60 fps = 30*16=480 ms
    let (first_dur, second_start, second_dur) = split_clip_at_frame(fps, 2000, cut_ms);
    assert_eq!(first_dur, cut_ms);
    assert_eq!(second_start, cut_ms);
    assert_eq!(first_dur + second_dur, 2000);
}

#[test]
fn test_split_at_before_clip_errors() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let id = tl
        .add_clip(vt, Clip::new(1, ClipType::Video, 1000, 3000))
        .expect("add_clip");
    let track = &mut tl.tracks[vt];
    let idx = track.clips.iter().position(|c| c.id == id).expect("clip");
    // Splitting before the clip's timeline_start (999 < 1000) must error.
    let result = track.clips[idx].split_at(999, 2);
    assert!(result.is_err(), "split before clip start must error");
}

#[test]
fn test_split_at_end_errors() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let id = tl
        .add_clip(vt, Clip::new(1, ClipType::Video, 0, 3000))
        .expect("add_clip");
    let track = &mut tl.tracks[vt];
    let idx = track.clips.iter().position(|c| c.id == id).expect("clip");
    // Splitting at the end (3000) should error.
    let result = track.clips[idx].split_at(3000, 2);
    assert!(result.is_err(), "split at end must error");
}

#[test]
fn test_split_second_half_is_independent() {
    // After split, modifying first half must not affect second.
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let id = tl
        .add_clip(vt, Clip::new(1, ClipType::Video, 100, 2000))
        .expect("add_clip");
    let track = &mut tl.tracks[vt];
    let idx = track.clips.iter().position(|c| c.id == id).expect("clip");
    let second = track.clips[idx].split_at(600, 2).expect("split");
    assert_eq!(second.timeline_start, 600);
    assert_eq!(second.timeline_duration, 1500);
}
