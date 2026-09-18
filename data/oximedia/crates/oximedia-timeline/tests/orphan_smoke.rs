//! Smoke tests for newly-wired orphan modules in oximedia-timeline.

// ── auto_sequence / beat_markers ─────────────────────────────────────────────

#[test]
fn test_auto_sequencer_returns_error_on_empty_input() {
    use oximedia_timeline::auto_sequence::{AutoSequenceConfig, AutoSequencer};
    use oximedia_timeline::track::TrackId;

    let tid = TrackId::new();
    let config = AutoSequenceConfig::new(tid);
    let sequencer = AutoSequencer::new();
    // empty markers → NoMarkers error (min_rating=0 still requires ≥1 marker)
    let result = sequencer.assemble(&[], &config);
    assert!(result.is_err(), "empty input should produce an error");
}

#[test]
fn test_beat_marker_position_string() {
    use oximedia_timeline::beat_markers::BeatMarker;

    let marker = BeatMarker::new(48, 2, 3).with_label("verse");
    let s = marker.position_string();
    assert!(!s.is_empty());
    assert_eq!(marker.beat, 2);
    assert_eq!(marker.bar, 3);
}

#[test]
fn test_tempo_map_generates_markers_at_120bpm() {
    use oximedia_timeline::beat_markers::TempoMap;

    let map = TempoMap::new(24.0, 120.0);
    // 120 bpm at 24 fps → 2 beats/sec → 1 beat every 12 frames → ~4 beats in 48 frames
    let markers = map.generate_markers(0, 48);
    assert!(!markers.is_empty());
}

// ── cache_warming ─────────────────────────────────────────────────────────────

#[test]
fn test_cache_warmer_hit_ratio_valid_range() {
    use oximedia_timeline::cache_warming::CacheWarmer;
    use oximedia_timeline::types::Position;

    let mut warmer = CacheWarmer::new();
    warmer.update(Position::new(0));
    warmer.record_hit();
    warmer.record_hit();
    let ratio = warmer.hit_ratio();
    assert!((0.0..=1.0).contains(&ratio));
}

// ── speed_ramp ────────────────────────────────────────────────────────────────

#[test]
fn test_speed_ramp_from_keyframes() {
    use oximedia_timeline::speed_ramp::{SpeedEasing, SpeedKeyframe, SpeedRamp};
    use oximedia_timeline::types::Position;

    let kf =
        SpeedKeyframe::new(Position::new(0), 1.5, SpeedEasing::Linear).expect("valid keyframe");
    let ramp = SpeedRamp::from_keyframes(vec![kf]).expect("valid ramp");
    assert_eq!(ramp.keyframe_count(), 1);
    assert!(ramp.average_speed() > 0.0);
}

// ── proxy_workflow ────────────────────────────────────────────────────────────

#[test]
fn test_proxy_manager_register_and_get() {
    use oximedia_timeline::clip::ClipId;
    use oximedia_timeline::proxy_workflow::{ProxyEntry, ProxyManager};

    let id = ClipId::new();
    let entry = ProxyEntry::new(id, "/src/clip.mov", "/proxy/clip_proxy.mov");
    let mut manager = ProxyManager::new();
    manager.register(entry);
    assert!(manager.get(id).is_some());
}

// ── timeline_compare ──────────────────────────────────────────────────────────

#[test]
fn test_timeline_snapshot_add_track() {
    use oximedia_timeline::timeline_compare::{TimelineSnapshot, TrackSnapshot};
    use oximedia_timeline::track::TrackId;

    let mut snapshot = TimelineSnapshot::new("v1", 24, 1);
    let tid = TrackId::new();
    snapshot.add_track(TrackSnapshot {
        id: tid,
        name: "V1".into(),
        clips: vec![],
        muted: false,
        locked: false,
    });
    assert_eq!(snapshot.tracks.len(), 1);
}

// ── multi_select ──────────────────────────────────────────────────────────────

#[test]
fn test_selection_empty_initially() {
    use oximedia_timeline::multi_select::Selection;

    let sel = Selection::new();
    assert!(sel.is_empty());
    assert_eq!(sel.len(), 0);
}

// ── lazy_clip ─────────────────────────────────────────────────────────────────

#[test]
fn test_lazy_clip_entry_is_unprobed_initially() {
    use oximedia_timeline::clip::ClipId;
    use oximedia_timeline::lazy_clip::LazyClipEntry;

    let id = ClipId::new();
    let entry = LazyClipEntry::new(id, "/media/test.mov");
    assert!(entry.state.is_unprobed());
}

// ── linked_clips ──────────────────────────────────────────────────────────────

#[test]
fn test_link_group_is_trivial_when_empty() {
    use oximedia_timeline::linked_clips::LinkGroup;

    let group = LinkGroup::new("dialogue");
    assert_eq!(group.member_count(), 0);
    assert!(group.is_trivial());
}

// ── incremental_render ────────────────────────────────────────────────────────

#[test]
fn test_change_tracker_no_changes_initially() {
    use oximedia_timeline::incremental_render::ChangeTracker;

    let tracker = ChangeTracker::new();
    assert!(!tracker.has_changes());
    assert_eq!(tracker.track_count(), 0);
}

// ── point_edit ────────────────────────────────────────────────────────────────

#[test]
fn test_three_point_edit_source_range_and_record_in() {
    use oximedia_timeline::point_edit::ThreePointEdit;
    use oximedia_timeline::types::Position;

    let ep = ThreePointEdit::with_source_range_and_record_in(
        Position::new(0),
        Position::new(100),
        Position::new(50),
    )
    .expect("valid three-point edit");
    assert_eq!(ep.source_in(), Position::new(0));
    assert_eq!(ep.source_out(), Position::new(100));
}

// ── fcpxml_export / otio_export (struct-level smoke) ─────────────────────────

#[test]
fn test_fcpxml_exporter_struct_exists() {
    use oximedia_timeline::fcpxml_export::FcpxmlExporter;
    let _: std::marker::PhantomData<FcpxmlExporter> = std::marker::PhantomData;
}

#[test]
fn test_otio_exporter_struct_exists() {
    use oximedia_timeline::otio_export::OtioExporter;
    let _: std::marker::PhantomData<OtioExporter> = std::marker::PhantomData;
}
