//! Tests for the seek infrastructure.

use super::*;

// ── SeekFlags tests ─────────────────────────────────────────────────

#[test]
fn test_seek_flags() {
    let flags = SeekFlags::BACKWARD | SeekFlags::KEYFRAME;
    assert!(flags.contains(SeekFlags::BACKWARD));
    assert!(flags.contains(SeekFlags::KEYFRAME));
    assert!(!flags.contains(SeekFlags::ANY));
}

// ── SeekTarget tests ────────────────────────────────────────────────

#[test]
fn test_seek_target_time() {
    let target = SeekTarget::time(10.5);
    assert_eq!(target.position, 10.5);
    assert!(target.is_keyframe());
    assert!(!target.is_byte());
    assert_eq!(target.stream_index, None);
}

#[test]
fn test_seek_target_byte() {
    let target = SeekTarget::byte(1024);
    assert_eq!(target.position, 1024.0);
    assert!(target.is_byte());
    assert_eq!(target.stream_index, None);
}

#[test]
fn test_seek_target_sample_accurate() {
    let target = SeekTarget::sample_accurate(5.0);
    assert!(target.is_frame_accurate());
    assert!(target.is_backward());
    assert!(!target.is_keyframe());
}

#[test]
fn test_seek_target_with_stream() {
    let target = SeekTarget::time(5.0).with_stream(1);
    assert_eq!(target.stream_index, Some(1));
    assert_eq!(target.position, 5.0);
}

#[test]
fn test_seek_target_with_flags() {
    let target = SeekTarget::time(3.0)
        .with_flags(SeekFlags::BACKWARD)
        .add_flags(SeekFlags::ANY);

    assert!(target.is_backward());
    assert!(target.is_any());
}

#[test]
fn test_seek_target_predicates() {
    let target = SeekTarget::time(1.0).add_flags(SeekFlags::BACKWARD | SeekFlags::FRAME_ACCURATE);

    assert!(target.is_backward());
    assert!(!target.is_any());
    assert!(target.is_keyframe());
    assert!(!target.is_byte());
    assert!(target.is_frame_accurate());
}

// ── SeekIndexEntry tests ────────────────────────────────────────────

#[test]
fn test_entry_keyframe() {
    let e = SeekIndexEntry::keyframe(0, 0, 100, 500, 3000, 0);
    assert!(e.is_keyframe);
    assert_eq!(e.pts, 0);
    assert_eq!(e.file_offset, 100);
    assert_eq!(e.end_pts(), 3000);
}

#[test]
fn test_entry_non_keyframe() {
    let e = SeekIndexEntry::non_keyframe(3000, 3000, 600, 200, 3000, 1);
    assert!(!e.is_keyframe);
    assert_eq!(e.sample_number, 1);
    assert_eq!(e.end_pts(), 6000);
}

// ── SeekIndex basic tests ───────────────────────────────────────────

fn build_test_index() -> SeekIndex {
    // 90kHz timescale, 30fps video (3000 ticks per frame)
    // GOP size = 5 frames (keyframe every 5th frame)
    let mut index = SeekIndex::new(90000);
    for i in 0u32..20 {
        let pts = i64::from(i) * 3000;
        let is_kf = i % 5 == 0;
        let offset = u64::from(i) * 500 + 1000; // arbitrary offsets
        if is_kf {
            index.add_entry(SeekIndexEntry::keyframe(pts, pts, offset, 500, 3000, i));
        } else {
            index.add_entry(SeekIndexEntry::non_keyframe(pts, pts, offset, 200, 3000, i));
        }
    }
    index
}

#[test]
fn test_index_new() {
    let index = SeekIndex::new(48000);
    assert_eq!(index.timescale(), 48000);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.keyframe_count(), 0);
}

#[test]
fn test_index_add_entries() {
    let index = build_test_index();
    assert_eq!(index.len(), 20);
    assert_eq!(index.keyframe_count(), 4); // frames 0, 5, 10, 15
}

#[test]
fn test_seconds_to_ticks() {
    let index = SeekIndex::new(90000);
    assert_eq!(index.seconds_to_ticks(1.0), 90000);
    assert_eq!(index.seconds_to_ticks(0.5), 45000);
}

#[test]
fn test_ticks_to_seconds() {
    let index = SeekIndex::new(90000);
    let s = index.ticks_to_seconds(90000);
    assert!((s - 1.0).abs() < 1e-10);
}

#[test]
fn test_ticks_to_seconds_zero_timescale() {
    let index = SeekIndex::new(0);
    assert_eq!(index.ticks_to_seconds(12345), 0.0);
}

// ── Keyframe search tests ───────────────────────────────────────────

#[test]
fn test_find_keyframe_before_exact() {
    let index = build_test_index();
    // Keyframes at pts: 0, 15000, 30000, 45000
    let kf = index.find_keyframe_before(15000).expect("should find");
    assert_eq!(kf.pts, 15000);
    assert!(kf.is_keyframe);
}

#[test]
fn test_find_keyframe_before_between() {
    let index = build_test_index();
    // Target at 20000 (between kf@15000 and kf@30000)
    let kf = index.find_keyframe_before(20000).expect("should find");
    assert_eq!(kf.pts, 15000);
}

#[test]
fn test_find_keyframe_before_start() {
    let index = build_test_index();
    let kf = index.find_keyframe_before(0).expect("should find");
    assert_eq!(kf.pts, 0);
}

#[test]
fn test_find_keyframe_before_none() {
    let index = build_test_index();
    // Target before all entries
    let kf = index.find_keyframe_before(-1);
    assert!(kf.is_none());
}

#[test]
fn test_find_keyframe_after() {
    let index = build_test_index();
    let kf = index.find_keyframe_after(20000).expect("should find");
    assert_eq!(kf.pts, 30000);
}

#[test]
fn test_find_keyframe_after_exact() {
    let index = build_test_index();
    let kf = index.find_keyframe_after(15000).expect("should find");
    assert_eq!(kf.pts, 15000);
}

#[test]
fn test_find_keyframe_after_past_end() {
    let index = build_test_index();
    let kf = index.find_keyframe_after(99999);
    assert!(kf.is_none());
}

#[test]
fn test_find_nearest_keyframe_closer_before() {
    let index = build_test_index();
    // 16000 is closer to kf@15000 than kf@30000
    let kf = index.find_nearest_keyframe(16000).expect("should find");
    assert_eq!(kf.pts, 15000);
}

#[test]
fn test_find_nearest_keyframe_closer_after() {
    let index = build_test_index();
    // 28000 is closer to kf@30000 than kf@15000
    let kf = index.find_nearest_keyframe(28000).expect("should find");
    assert_eq!(kf.pts, 30000);
}

#[test]
fn test_find_nearest_keyframe_equidistant() {
    let index = build_test_index();
    // 22500 is equidistant between kf@15000 and kf@30000
    // Should prefer the earlier one (backward preference)
    let kf = index.find_nearest_keyframe(22500).expect("should find");
    assert_eq!(kf.pts, 15000);
}

// ── Sample-at tests ─────────────────────────────────────────────────

#[test]
fn test_find_sample_at_exact_start() {
    let index = build_test_index();
    let sample = index.find_sample_at(3000).expect("should find");
    assert_eq!(sample.pts, 3000);
}

#[test]
fn test_find_sample_at_mid_frame() {
    let index = build_test_index();
    // 4000 is within frame at pts=3000 (duration=3000, so [3000..6000))
    let sample = index.find_sample_at(4000).expect("should find");
    assert_eq!(sample.pts, 3000);
}

#[test]
fn test_find_sample_at_last() {
    let index = build_test_index();
    // Last frame: pts=57000
    let sample = index.find_sample_at(58000).expect("should find");
    assert_eq!(sample.pts, 57000);
}

// ── Seek Plan tests ─────────────────────────────────────────────────

#[test]
fn test_plan_seek_keyframe() {
    let index = build_test_index();
    let plan = index
        .plan_seek(20000, SeekAccuracy::Keyframe)
        .expect("should plan");
    // Should go to keyframe at 15000
    assert_eq!(plan.keyframe_entry.pts, 15000);
    assert_eq!(plan.discard_count, 0);
    assert_eq!(plan.target_entry.pts, 15000);
}

#[test]
fn test_plan_seek_sample_accurate() {
    let index = build_test_index();
    // Target: pts=21000 (frame 7 at pts=21000)
    let plan = index
        .plan_seek(21000, SeekAccuracy::SampleAccurate)
        .expect("should plan");

    // Keyframe should be at pts=15000 (frame 5)
    assert_eq!(plan.keyframe_entry.pts, 15000);
    // Target should be the frame at pts=21000
    assert_eq!(plan.target_entry.pts, 21000);
    // Discard count: frames 6 (18000) between keyframe 5 (15000) and target 7 (21000)
    assert_eq!(plan.discard_count, 1); // frame at 18000
    assert!(plan.is_exact);
}

#[test]
fn test_plan_seek_sample_accurate_on_keyframe() {
    let index = build_test_index();
    let plan = index
        .plan_seek(15000, SeekAccuracy::SampleAccurate)
        .expect("should plan");
    assert_eq!(plan.keyframe_entry.pts, 15000);
    assert_eq!(plan.discard_count, 0);
    assert!(plan.is_exact);
}

#[test]
fn test_plan_seek_sample_accurate_first_frame() {
    let index = build_test_index();
    let plan = index
        .plan_seek(0, SeekAccuracy::SampleAccurate)
        .expect("should plan");
    assert_eq!(plan.keyframe_entry.pts, 0);
    assert_eq!(plan.target_entry.pts, 0);
    assert_eq!(plan.discard_count, 0);
}

#[test]
fn test_plan_seek_within_tolerance_exact() {
    let index = build_test_index();
    // Within 5000 ticks of a keyframe at 15000
    let plan = index
        .plan_seek(16000, SeekAccuracy::WithinTolerance(5000))
        .expect("should plan");
    assert_eq!(plan.target_entry.pts, 15000);
}

#[test]
fn test_plan_seek_within_tolerance_out_of_range() {
    let index = build_test_index();
    // Only 1 tick tolerance, and 20000 is not within 1 tick of any keyframe
    let plan = index.plan_seek(20000, SeekAccuracy::WithinTolerance(1));
    assert!(plan.is_none());
}

#[test]
fn test_plan_seek_empty_index() {
    let index = SeekIndex::new(90000);
    let plan = index.plan_seek(0, SeekAccuracy::Keyframe);
    assert!(plan.is_none());
}

// ── Duration and statistics tests ───────────────────────────────────

#[test]
fn test_duration_ticks() {
    let index = build_test_index();
    // 20 frames, each 3000 ticks, last frame ends at 20*3000 = 60000
    assert_eq!(index.duration_ticks(), 60000);
}

#[test]
fn test_duration_seconds() {
    let index = build_test_index();
    let dur = index.duration_seconds();
    // 60000 ticks at 90kHz = 0.6667 seconds
    assert!((dur - 60000.0 / 90000.0).abs() < 1e-6);
}

#[test]
fn test_average_keyframe_interval() {
    let index = build_test_index();
    let avg = index.average_keyframe_interval().expect("should calculate");
    // Keyframes at 0, 15000, 30000, 45000 -> intervals: 15000, 15000, 15000
    assert!((avg - 15000.0).abs() < 1e-6);
}

#[test]
fn test_average_keyframe_interval_single() {
    let mut index = SeekIndex::new(90000);
    index.add_entry(SeekIndexEntry::keyframe(0, 0, 0, 100, 3000, 0));
    assert!(index.average_keyframe_interval().is_none());
}

// ── Sort test ───────────────────────────────────────────────────────

#[test]
fn test_sort_reorders_entries() {
    let mut index = SeekIndex::new(90000);
    // Add out of order
    index.add_entry(SeekIndexEntry::non_keyframe(6000, 6000, 300, 100, 3000, 2));
    index.add_entry(SeekIndexEntry::keyframe(0, 0, 100, 100, 3000, 0));
    index.add_entry(SeekIndexEntry::non_keyframe(3000, 3000, 200, 100, 3000, 1));

    index.sort();

    assert_eq!(index.entries()[0].pts, 0);
    assert_eq!(index.entries()[1].pts, 3000);
    assert_eq!(index.entries()[2].pts, 6000);
    assert_eq!(index.keyframe_count(), 1);
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn test_find_keyframe_empty() {
    let index = SeekIndex::new(90000);
    assert!(index.find_keyframe_before(0).is_none());
    assert!(index.find_keyframe_after(0).is_none());
    assert!(index.find_nearest_keyframe(0).is_none());
}

#[test]
fn test_single_keyframe_index() {
    let mut index = SeekIndex::new(90000);
    index.add_entry(SeekIndexEntry::keyframe(0, 0, 0, 100, 90000, 0));

    let kf = index.find_keyframe_before(45000).expect("should find");
    assert_eq!(kf.pts, 0);

    let plan = index
        .plan_seek(45000, SeekAccuracy::SampleAccurate)
        .expect("should plan");
    assert_eq!(plan.keyframe_entry.pts, 0);
}

#[test]
fn test_all_keyframes_index() {
    let mut index = SeekIndex::new(48000);
    // Audio: every frame is a keyframe
    for i in 0u32..100 {
        let pts = i64::from(i) * 960;
        index.add_entry(SeekIndexEntry::keyframe(
            pts,
            pts,
            u64::from(i) * 100,
            100,
            960,
            i,
        ));
    }

    assert_eq!(index.keyframe_count(), 100);

    let plan = index
        .plan_seek(48000, SeekAccuracy::SampleAccurate)
        .expect("should plan");
    // Should find the exact frame
    assert_eq!(plan.keyframe_entry.pts, 48000);
    assert_eq!(plan.discard_count, 0);
}

#[test]
fn test_with_capacity() {
    let index = SeekIndex::with_capacity(90000, 1000);
    assert_eq!(index.timescale(), 90000);
    assert!(index.is_empty());
}

// ── SampleAccurateSeeker tests ───────────────────────────────────────

fn build_seeker_index() -> SeekIndex {
    // 90kHz, 30fps (3000 ticks/frame), GOP=5
    let mut index = SeekIndex::new(90000);
    for i in 0u32..20 {
        let pts = i64::from(i) * 3000;
        let is_kf = i % 5 == 0;
        let offset = u64::from(i) * 500 + 1000;
        if is_kf {
            index.add_entry(SeekIndexEntry::keyframe(pts, pts, offset, 500, 3000, i));
        } else {
            index.add_entry(SeekIndexEntry::non_keyframe(pts, pts, offset, 200, 3000, i));
        }
    }
    index
}

#[test]
fn test_sample_accurate_seeker_on_keyframe() {
    let track = TrackIndex::new(build_seeker_index());
    let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
    let result = seeker.seek_to_sample(15000, &track).expect("should find");
    assert_eq!(result.keyframe_pts, 15000);
    assert_eq!(result.preroll_samples, 0);
    assert_eq!(result.sample_offset, 1000 + 5 * 500); // frame 5 offset
}

#[test]
fn test_sample_accurate_seeker_between_keyframes() {
    let track = TrackIndex::new(build_seeker_index());
    let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
    // Target pts=21000 (frame 7) — keyframe is at pts=15000 (frame 5)
    // Frames 6 (pts=18000) must be discarded → preroll = 1
    let result = seeker.seek_to_sample(21000, &track).expect("should find");
    assert_eq!(result.keyframe_pts, 15000);
    assert_eq!(result.preroll_samples, 1);
}

#[test]
fn test_sample_accurate_seeker_codec_delay_added() {
    let track = TrackIndex::with_codec_delay(build_seeker_index(), 512);
    let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
    // On a keyframe: preroll = 0 inter-frame + 512 codec_delay = 512
    let result = seeker.seek_to_sample(0, &track).expect("should find");
    assert_eq!(result.keyframe_pts, 0);
    assert_eq!(result.preroll_samples, 512);
}

#[test]
fn test_sample_accurate_seeker_empty_index() {
    let track = TrackIndex::new(SeekIndex::new(90000));
    let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(SeekIndex::new(90000)));
    let result = seeker.seek_to_sample(0, &track);
    assert!(result.is_none());
}

#[test]
fn test_track_index_default_codec_delay() {
    let idx = SeekIndex::new(90000);
    let track = TrackIndex::new(idx);
    assert_eq!(track.codec_delay_samples, 0);
}

#[test]
fn test_seek_result_fields() {
    let track = TrackIndex::new(build_seeker_index());
    let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
    let result = seeker.seek_to_sample(0, &track).expect("should find");
    // frame 0 is a keyframe at file offset 1000
    assert_eq!(result.keyframe_pts, 0);
    assert_eq!(result.sample_offset, 1000);
    assert_eq!(result.preroll_samples, 0);
}

// ── MultiTrackSeeker tests ───────────────────────────────────────────

fn build_multi_track_samples() -> Vec<SampleIndexEntry> {
    // 30fps video at 90kHz: keyframe every 5 frames (GOP=5)
    (0u32..20)
        .map(|i| {
            let pts = i64::from(i) * 3000;
            let offset = 1000 + u64::from(i) * 500;
            if i % 5 == 0 {
                SampleIndexEntry::keyframe(pts, offset)
            } else {
                SampleIndexEntry::delta(pts, offset)
            }
        })
        .collect()
}

#[test]
fn test_multi_track_seeker_new() {
    let seeker = MultiTrackSeeker::new();
    assert_eq!(seeker.indexed_track_count(), 0);
}

#[test]
fn test_multi_track_build_index() {
    let mut seeker = MultiTrackSeeker::new();
    let samples = build_multi_track_samples();
    seeker.build_index(1, &samples).expect("build ok");
    assert_eq!(seeker.indexed_track_count(), 1);
    assert_eq!(seeker.sample_count(1), Some(20));
}

#[test]
fn test_multi_track_entries_sorted() {
    let mut seeker = MultiTrackSeeker::new();
    // Insert out of order
    let samples = vec![
        SampleIndexEntry::delta(9000, 3000),
        SampleIndexEntry::keyframe(0, 1000),
        SampleIndexEntry::delta(3000, 1500),
        SampleIndexEntry::delta(6000, 2000),
    ];
    seeker.build_index(1, &samples).expect("ok");
    let entries = seeker.entries(1).expect("entries exist");
    assert_eq!(entries[0].pts, 0);
    assert_eq!(entries[1].pts, 3000);
    assert_eq!(entries[2].pts, 6000);
    assert_eq!(entries[3].pts, 9000);
}

#[test]
fn test_seek_to_pts_exact() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    // Seek exactly to frame 5 (pts=15000)
    let result = seeker.seek_to_pts(1, 15000).expect("seek ok");
    assert_eq!(result.found_pts, 15000);
    assert_eq!(result.byte_offset, 1000 + 5 * 500);
    assert_eq!(result.sample_idx, 5);
}

#[test]
fn test_seek_to_pts_between_samples() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    // PTS 16000 falls between frame 5 (15000) and frame 6 (18000)
    let result = seeker.seek_to_pts(1, 16000).expect("seek ok");
    assert_eq!(
        result.found_pts, 15000,
        "should return the preceding sample"
    );
    assert_eq!(result.sample_idx, 5);
}

#[test]
fn test_seek_to_pts_first_sample() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    let result = seeker.seek_to_pts(1, 0).expect("seek ok");
    assert_eq!(result.found_pts, 0);
    assert_eq!(result.sample_idx, 0);
}

#[test]
fn test_seek_to_pts_last_sample() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    // PTS beyond the last sample should return the last sample
    let result = seeker.seek_to_pts(1, 99999).expect("seek ok");
    assert_eq!(result.found_pts, 19 * 3000); // last sample
    assert_eq!(result.sample_idx, 19);
}

#[test]
fn test_seek_to_pts_before_first_sample() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    let err = seeker.seek_to_pts(1, -1);
    assert!(matches!(
        err,
        Err(MultiTrackSeekerError::BeforeFirstSample(-1, 1))
    ));
}

#[test]
fn test_seek_to_pts_no_index() {
    let seeker = MultiTrackSeeker::new();
    let err = seeker.seek_to_pts(42, 0);
    assert!(matches!(err, Err(MultiTrackSeekerError::NoIndex(42))));
}

#[test]
fn test_seek_to_pts_empty_index() {
    let mut seeker = MultiTrackSeeker::new();
    seeker.build_index(1, &[]).expect("ok");
    let err = seeker.seek_to_pts(1, 0);
    assert!(matches!(err, Err(MultiTrackSeekerError::EmptyIndex(1))));
}

#[test]
fn test_multi_track_multiple_tracks() {
    let mut seeker = MultiTrackSeeker::new();
    let video = build_multi_track_samples();
    // 51 audio frames: pts 0, 960, 1920, ..., 50*960=48000
    let audio: Vec<SampleIndexEntry> = (0u32..=50)
        .map(|i| SampleIndexEntry::keyframe(i64::from(i) * 960, u64::from(i) * 100 + 500))
        .collect();

    seeker.build_index(1, &video).expect("video ok");
    seeker.build_index(2, &audio).expect("audio ok");

    assert_eq!(seeker.indexed_track_count(), 2);

    let v_result = seeker.seek_to_pts(1, 15000).expect("video seek ok");
    let a_result = seeker.seek_to_pts(2, 48000).expect("audio seek ok");

    assert_eq!(v_result.found_pts, 15000);
    assert_eq!(a_result.found_pts, 48000);
}

#[test]
fn test_clear_index() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &build_multi_track_samples())
        .expect("ok");
    assert_eq!(seeker.indexed_track_count(), 1);
    seeker.clear_index(1);
    assert_eq!(seeker.indexed_track_count(), 0);
    let err = seeker.seek_to_pts(1, 0);
    assert!(matches!(err, Err(MultiTrackSeekerError::NoIndex(1))));
}

#[test]
fn test_build_index_replaces_existing() {
    let mut seeker = MultiTrackSeeker::new();
    let old: Vec<SampleIndexEntry> = vec![SampleIndexEntry::keyframe(0, 100)];
    let new: Vec<SampleIndexEntry> = vec![
        SampleIndexEntry::keyframe(0, 200),
        SampleIndexEntry::keyframe(3000, 300),
    ];
    seeker.build_index(1, &old).expect("ok");
    seeker.build_index(1, &new).expect("replace ok");
    assert_eq!(seeker.sample_count(1), Some(2));
    let result = seeker.seek_to_pts(1, 0).expect("ok");
    assert_eq!(result.byte_offset, 200);
}

#[test]
fn test_sample_index_entry_constructors() {
    let kf = SampleIndexEntry::keyframe(1000, 9999);
    assert!(kf.is_sync);
    assert_eq!(kf.pts, 1000);
    assert_eq!(kf.byte_offset, 9999);

    let df = SampleIndexEntry::delta(2000, 8888);
    assert!(!df.is_sync);
    assert_eq!(df.pts, 2000);
}

#[test]
fn test_pts_seek_result_fields() {
    let mut seeker = MultiTrackSeeker::new();
    seeker
        .build_index(1, &[SampleIndexEntry::keyframe(5000, 12345)])
        .expect("ok");
    let r = seeker.seek_to_pts(1, 5000).expect("ok");
    assert_eq!(r.found_pts, 5000);
    assert_eq!(r.byte_offset, 12345);
    assert_eq!(r.sample_idx, 0);
}

// ── SeekMode + seek_to_pts tests ─────────────────────────────────────────

#[test]
fn test_sample_accurate_seek_at_keyframe() {
    // Target is exactly a keyframe PTS (frame 0, gop=10, step=1).
    // seek_fn returns 0, first decode_fn call returns 0 >= 0, so 0 preroll.
    let seeker = SampleAccurateSeeker::with_max_preroll(200);
    let seek_fn = |_target: u64| -> Result<u64, String> { Ok(0) };
    let mut cursor: u64 = 0;
    let decode_fn = move || -> Result<Option<u64>, String> {
        let pts = cursor;
        cursor += 1;
        Ok(Some(pts))
    };
    let preroll = seeker.seek_to_pts(seek_fn, decode_fn, 0).expect("seek ok");
    assert_eq!(preroll, 0, "no preroll when target == keyframe PTS");
}

#[test]
fn test_sample_accurate_seek_between_keyframes() {
    // Keyframe every 10 ticks, step=1. Target=15 → keyframe at 10.
    // seek_fn returns 10; decode_fn delivers PTS 10,11,12,13,14 (5 preroll
    // frames, all < 15) then PTS 15 ≥ 15 → returns 5.
    let seeker = SampleAccurateSeeker::with_max_preroll(200);
    let seek_fn = |target: u64| -> Result<u64, String> { Ok((target / 10) * 10) };
    // After seeking to keyframe 10, decoder delivers 10, 11, 12, 13, 14, 15, …
    let mut cursor: u64 = 10;
    let decode_fn = move || -> Result<Option<u64>, String> {
        let pts = cursor;
        cursor += 1;
        Ok(Some(pts))
    };

    let preroll = seeker.seek_to_pts(seek_fn, decode_fn, 15).expect("seek ok");
    // PTS 10,11,12,13,14 are < 15 → 5 preroll frames decoded before PTS 15 hits
    assert_eq!(preroll, 5, "should decode 5 preroll frames to reach PTS 15");
}

#[test]
fn test_sample_accurate_seek_max_preroll() {
    // max_preroll=3 but target requires 10 frames of preroll → error.
    let seeker = SampleAccurateSeeker::with_max_preroll(3);
    let seek_fn = |_target: u64| -> Result<u64, String> { Ok(0) };
    // decode_fn always returns PTS 0 (never reaches target=99)
    let decode_fn = || -> Result<Option<u64>, String> { Ok(Some(0)) };
    let err = seeker.seek_to_pts(seek_fn, decode_fn, 99);
    assert!(
        matches!(
            err,
            Err(ClosedLoopSeekError::MaxPrerollExceeded { limit: 3, .. })
        ),
        "expected MaxPrerollExceeded, got {:?}",
        err,
    );
}
