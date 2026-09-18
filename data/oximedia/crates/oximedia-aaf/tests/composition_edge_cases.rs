//! Composition edge-case tests for CMX3600 EDL export.
//!
//! Exercises the `Cmx3600Exporter` against degenerate and boundary
//! compositions: empty mobs, single-frame clips, nested effects (which the
//! exporter intentionally skips), and filler-driven record-position advance.
//!
//! These tests work over the `Uuid`-keyed `CompositionMob` model and assert
//! the precise timecodes the exporter computes at 25 fps non-drop-frame.

use oximedia_aaf::composition::{
    CompositionMob, Effect, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
};
use oximedia_aaf::dictionary::Auid;
use oximedia_aaf::edl_export::{emit_edl, Cmx3600Exporter};
use oximedia_aaf::timeline::{EditRate, Position};
use uuid::Uuid;

/// An empty composition produces no events and still emits a valid EDL header.
#[test]
fn empty_composition_exports_empty_ok() {
    let comp = CompositionMob::new(Uuid::new_v4(), "Empty");
    let exporter = Cmx3600Exporter::new(25.0, false);

    let events = exporter.export_composition_to_cmx3600(&comp);
    assert!(
        events.is_empty(),
        "empty composition must yield zero events, got {}",
        events.len()
    );

    let edl = emit_edl(&events, "Empty");
    assert!(
        edl.contains("TITLE: Empty"),
        "EDL header must carry the title, got:\n{edl}"
    );
}

/// A single one-frame source clip becomes exactly one cut event on the video
/// track, spanning frame 0 to frame 1 (00:00:00:00 → 00:00:00:01 at 25 fps).
#[test]
fn single_frame_clip_one_event() {
    let mut comp = CompositionMob::new(Uuid::new_v4(), "SingleFrame");
    let mut video_track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
    let mut seq = Sequence::new(Auid::PICTURE);
    seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
        1,
        Position::new(0),
        Uuid::new_v4(),
        1,
    )));
    video_track.set_sequence(seq);
    comp.add_track(video_track);

    let exporter = Cmx3600Exporter::new(25.0, false);
    let events = exporter.export_composition_to_cmx3600(&comp);

    assert_eq!(
        events.len(),
        1,
        "single clip must produce exactly one event"
    );
    assert_eq!(events[0].track_type, "V", "picture track designates as V");
    assert_eq!(events[0].transition, "C", "plain source clip is a cut");
    assert_eq!(events[0].src_in, "00:00:00:00", "src_in at frame 0");
    assert_eq!(
        events[0].src_out, "00:00:00:01",
        "one frame @25fps ends at frame 1"
    );
}

/// An effect with a nested source-clip input resolves at the data level
/// (the input clip is retained in `eff.inputs`), but the exporter SILENTLY
/// SKIPS effect components, so the composition exports to zero events.
#[test]
fn nested_effect_resolves_at_data_level() {
    let mut eff = Effect::new(Auid::PICTURE);
    eff.add_input(SequenceComponent::SourceClip(SourceClip::new(
        50,
        Position::new(0),
        Uuid::new_v4(),
        1,
    )));
    assert_eq!(
        eff.inputs.len(),
        1,
        "effect must retain its single nested input at the data level"
    );

    let mut comp = CompositionMob::new(Uuid::new_v4(), "EffectOnly");
    let mut video_track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
    let mut seq = Sequence::new(Auid::PICTURE);
    seq.add_component(SequenceComponent::Effect(eff));
    video_track.set_sequence(seq);
    comp.add_track(video_track);

    let exporter = Cmx3600Exporter::new(25.0, false);
    let events = exporter.export_composition_to_cmx3600(&comp);
    assert!(
        events.is_empty(),
        "exporter must skip effect components, got {} event(s)",
        events.len()
    );
}

/// A filler preceding a clip advances the record position WITHOUT emitting an
/// event of its own: a 50-frame filler (2s @25fps) pushes the following clip's
/// record-in to 00:00:02:00.
#[test]
fn filler_advances_record_without_event() {
    let mut comp = CompositionMob::new(Uuid::new_v4(), "FillerThenClip");
    let mut video_track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
    let mut seq = Sequence::new(Auid::PICTURE);
    seq.add_component(SequenceComponent::Filler(Filler::new(50)));
    seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
        25,
        Position::new(0),
        Uuid::new_v4(),
        1,
    )));
    video_track.set_sequence(seq);
    comp.add_track(video_track);

    let exporter = Cmx3600Exporter::new(25.0, false);
    let events = exporter.export_composition_to_cmx3600(&comp);

    assert_eq!(
        events.len(),
        1,
        "filler emits no event of its own; only the clip does"
    );
    assert_eq!(
        events[0].rec_in, "00:00:02:00",
        "50-frame filler (2s @25fps) advances record-in to 2s"
    );
}
