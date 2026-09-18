//! Smoke tests for newly-wired orphan modules in oximedia-playout.

// ─── branding_inserter ───────────────────────────────────────────────────────
#[test]
fn test_branding_inserter_schedule_and_advance() {
    use oximedia_playout::branding_inserter::{
        BrandingInserter, BrandingSlot, BrandingZone, InsertionPolicy,
    };

    let mut inserter = BrandingInserter::new(InsertionPolicy::ReplaceOnConflict);
    inserter
        .schedule(BrandingSlot {
            id: 1,
            asset_id: 10,
            zone: BrandingZone::BugBottomRight,
            start_frame: 0,
            end_frame: Some(500),
            priority: 1,
            label: "channel-bug".to_string(),
        })
        .expect("schedule should succeed");

    // Advance to frame 100 — should fire an Activate event
    let events = inserter.advance_to_frame(100);
    assert!(
        !events.is_empty(),
        "branding slot should fire an activate event"
    );
    assert_eq!(inserter.active_slot_ids().len(), 1);
}

// ─── compliance_recorder ─────────────────────────────────────────────────────
#[test]
fn test_compliance_recorder_segment_lifecycle() {
    use oximedia_playout::compliance_recorder::{ComplianceRecorder, RecorderConfig};
    use std::time::SystemTime;

    let config = RecorderConfig::default();
    let mut recorder = ComplianceRecorder::new(config);

    let now = SystemTime::UNIX_EPOCH;
    recorder
        .start_segment("seg-001", "Morning News", 0, now)
        .expect("start segment should succeed");
    recorder
        .write_frame(1_000_000, &[0xAB; 188])
        .expect("write frame");
    recorder
        .write_frame(2_000_000, &[0xCD; 188])
        .expect("write frame");
    let _seg = recorder
        .close_segment(now)
        .expect("close segment should succeed");

    assert_eq!(
        recorder.archive_len(),
        1,
        "archive should contain the closed segment"
    );
}

// ─── cue_trigger ─────────────────────────────────────────────────────────────
#[test]
fn test_cue_trigger_fires_at_frame() {
    use oximedia_playout::cue_trigger::{CueAction, CuePoint, CueTrigger};

    let mut trigger = CueTrigger::new(25.0);
    trigger.add_cue(CuePoint {
        id: "intro-slate".to_string(),
        timecode_frames: 25, // 1 second in
        action: CueAction::ShowGraphic("slate.png".to_string()),
        triggered: false,
    });

    // Frame 24 — not yet
    let fired = trigger.check_frame(24);
    assert!(fired.is_empty(), "cue should not fire before its timecode");

    // Frame 25 — fires
    let fired = trigger.check_frame(25);
    assert_eq!(fired.len(), 1, "cue should fire at its timecode");
    // fired returns CueAction, verify it's the expected graphic
    assert!(
        matches!(&fired[0], oximedia_playout::cue_trigger::CueAction::ShowGraphic(name) if name == "slate.png"),
        "fired action should be ShowGraphic(slate.png)"
    );
}

// ─── emergency_alert ─────────────────────────────────────────────────────────
#[test]
fn test_emergency_alert_parse_same_header() {
    use oximedia_playout::emergency_alert::SameHeader;

    // Minimal single-location SAME header (WXR-TOR, 1 location, purge 0030, issued 3011655, station KSVR/NWS)
    let header_str = "ZCZC-WXR-TOR-020103+0030-3011655-KSVR/NWS-";
    let result = SameHeader::parse(header_str);
    assert!(
        result.is_ok(),
        "valid SAME header should parse without error: {:?}",
        result.err()
    );

    let hdr = result.unwrap();
    assert_eq!(hdr.originator, "WXR");
    assert_eq!(hdr.event.code, "TOR");
}

#[test]
fn test_emergency_alert_invalid_header() {
    use oximedia_playout::emergency_alert::SameHeader;

    let result = SameHeader::parse("not a valid SAME header");
    assert!(result.is_err(), "invalid header should fail to parse");
}

// ─── health ──────────────────────────────────────────────────────────────────
#[test]
fn test_health_monitor_within_threshold() {
    use oximedia_playout::health::PlayoutHealth;

    let mut health = PlayoutHealth::new();
    health.update(0, 0);
    assert!(health.is_healthy(5), "zero drops should be healthy");

    health.update(3, 2);
    assert!(
        health.is_healthy(5),
        "3 drops ≤ 5 threshold should still be healthy"
    );
}

#[test]
fn test_health_monitor_exceeds_threshold() {
    use oximedia_playout::health::PlayoutHealth;

    let mut health = PlayoutHealth::new();
    health.update(10, 0);
    assert!(
        !health.is_healthy(5),
        "10 drops > 5 threshold should be unhealthy"
    );
}

// ─── hls_catchup ─────────────────────────────────────────────────────────────
#[test]
fn test_hls_catchup_manifest_builder() {
    use oximedia_playout::hls_catchup::{HlsManifestBuilder, HlsManifestConfig, VodSegment};

    let segments = vec![
        VodSegment::new(0, "seg001.ts", 6.006),
        VodSegment::new(1, "seg002.ts", 6.006),
    ];

    let config = HlsManifestConfig::default();
    let builder = HlsManifestBuilder::new(config);
    let manifest = builder.build_media_playlist(&segments);
    assert!(
        manifest.contains("#EXTM3U"),
        "HLS manifest should start with #EXTM3U"
    );
    assert!(
        manifest.contains("seg001.ts"),
        "should contain first segment filename"
    );
}

// ─── lockfree_frame_ring ─────────────────────────────────────────────────────
#[test]
fn test_lockfree_frame_ring_push_pop() {
    use oximedia_playout::lockfree_frame_ring::{LockfreeFrameRing, VideoFrame};

    let ring = LockfreeFrameRing::new(8);
    let frame = VideoFrame::blank(1, 1_000_000_000, 8, 8);
    let pushed = ring.push(frame);
    assert!(pushed.is_ok(), "push into empty ring should succeed");

    let popped = ring.pop();
    assert!(
        popped.is_some(),
        "pop from non-empty ring should return a frame"
    );
    assert_eq!(popped.unwrap().sequence, 1);
}

#[test]
fn test_lockfree_frame_ring_overflow_counted() {
    use oximedia_playout::lockfree_frame_ring::{LockfreeFrameRing, VideoFrame};

    let ring = LockfreeFrameRing::new(2);
    let mk = |seq: u64| VideoFrame::blank(seq, seq * 1_000_000, 4, 4);

    // Fill to capacity
    assert!(ring.push(mk(1)).is_ok());
    assert!(ring.push(mk(2)).is_ok());
    // This push should overflow
    let result = ring.push(mk(3));
    assert!(result.is_err(), "push onto full ring should return Err");
    assert!(
        ring.overflow_count() >= 1,
        "overflow counter should be incremented"
    );
}

// ─── lower_third ─────────────────────────────────────────────────────────────
#[test]
fn test_lower_third_news_standard_template() {
    use oximedia_playout::lower_third::LowerThirdTemplate;

    let lt = LowerThirdTemplate::news_standard();
    // Should have a primary_text and secondary_text slot
    let (w, h) = (1920_u32, 1080_u32);
    let rect = lt.compute_rect(w, h);
    // x, y, width, height — all must be within frame bounds
    assert!(rect.0 < w, "x must be within frame width");
    assert!(rect.1 < h, "y must be within frame height");
    assert!(rect.2 <= w, "template width must not exceed frame width");
}

// ─── rundown_editor ──────────────────────────────────────────────────────────
#[test]
fn test_rundown_editor_insert_and_undo() {
    use oximedia_playout::rundown::{ItemType, Rundown, RundownItem};
    use oximedia_playout::rundown_editor::RundownEditor;

    let rundown = Rundown::new("Morning Show", 3600.0);
    let mut editor = RundownEditor::new(rundown);

    let id = editor.alloc_id();
    let item = RundownItem::new(id, "Opening Titles", ItemType::Story, 150.0);
    editor.insert_at(0, item).expect("insert should succeed");
    assert_eq!(editor.len(), 1);

    editor.undo().expect("undo should succeed");
    assert_eq!(editor.len(), 0, "after undo rundown should be empty");
}

// ─── scene_highlight ─────────────────────────────────────────────────────────
#[test]
fn test_scene_highlight_motion_energy_detector() {
    use oximedia_playout::scene_highlight::{GreyscaleFrame, MotionEnergyDetector};

    let mut detector = MotionEnergyDetector::new(0.3);
    let black = GreyscaleFrame::filled(64, 64, 0, 0, 0);
    let white = GreyscaleFrame::filled(64, 64, 1, 40_000_000, 255);

    // First frame — no previous, submit returns None
    let first = detector.submit(&black);
    assert!(first.is_none(), "first frame has no previous to compare");

    // Second frame — large MAD between black and white
    let score = detector
        .submit(&white)
        .expect("second frame should return a score");
    assert!(
        score > 0.5,
        "black→white should have high motion energy score, got {score}"
    );
    assert!(detector.is_high_motion(score), "should detect high motion");
}

// ─── schedule_convert ────────────────────────────────────────────────────────
#[test]
fn test_schedule_convert_two_items() {
    use oximedia_playout::schedule_convert::{PlaylistToSchedule, SchedulePlaylistItem};

    let items = vec![
        SchedulePlaylistItem {
            name: "Clip A".into(),
            duration_frames: Some(250),
            fps: 25.0,
        },
        SchedulePlaylistItem {
            name: "Clip B".into(),
            duration_frames: Some(500),
            fps: 25.0,
        },
    ];
    let scheduled = PlaylistToSchedule::convert(&items, 1_700_000_000_000);
    assert_eq!(scheduled.len(), 2);
    assert_eq!(scheduled[0].start_ts_ms, 1_700_000_000_000);
    // Clip A is 250 frames at 25fps = 10 seconds = 10_000 ms
    assert_eq!(scheduled[1].start_ts_ms, 1_700_000_010_000);
}
