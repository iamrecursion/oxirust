//! Integration tests: Timeline → EDL export → EDL parse back, and FCP XML export.
//!
//! These tests verify that `TimelineExporter` produces well-formed output and
//! that the `edl::Edl` parser can round-trip through CMX-3600.  Note: there is
//! no EDL→`Timeline` importer, so we validate the parsed `Edl` struct instead
//! of a full `Timeline` equivalence.

use oximedia_core::Rational;
use oximedia_edit::edl::{Edl, EdlFormat};
use oximedia_edit::timeline_export::TimelineExporter;
use oximedia_edit::{Clip, ClipType, Timeline, TrackType};

/// Build a simple two-clip video timeline and return it.
fn make_two_clip_timeline() -> Timeline {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let vt = tl.add_track(TrackType::Video);
    let _id1 = tl
        .add_clip(vt, Clip::new(1, ClipType::Video, 0, 3000))
        .expect("add clip 1");
    let _id2 = tl
        .add_clip(vt, Clip::new(2, ClipType::Video, 3000, 2000))
        .expect("add clip 2");
    tl
}

// ── EDL export tests ─────────────────────────────────────────────────────────

#[test]
fn test_edl_contains_title_header() {
    let tl = make_two_clip_timeline();
    let edl = TimelineExporter::to_edl(&tl);
    assert!(edl.contains("TITLE:"), "EDL must start with TITLE:");
}

#[test]
fn test_edl_contains_fcm_line() {
    let tl = make_two_clip_timeline();
    let edl = TimelineExporter::to_edl(&tl);
    assert!(edl.contains("FCM:"), "EDL must contain FCM:");
}

#[test]
fn test_edl_has_two_events_for_two_clips() {
    let tl = make_two_clip_timeline();
    let edl = TimelineExporter::to_edl(&tl);
    // Each event line starts with a 3-digit event number followed by two spaces.
    let event_count = edl
        .lines()
        .filter(|l| l.starts_with("001") || l.starts_with("002"))
        .count();
    assert_eq!(
        event_count, 2,
        "Expected exactly 2 event lines, got:\n{edl}"
    );
}

#[test]
fn test_edl_timecodes_are_nonzero_for_second_clip() {
    let tl = make_two_clip_timeline();
    let edl = TimelineExporter::to_edl(&tl);
    // The second clip starts at 3000 ms which at 30 fps = 90 frames = "00:00:03:00".
    assert!(
        edl.contains("00:00:03:00"),
        "Record-in timecode for clip 2 missing:\n{edl}"
    );
}

#[test]
fn test_edl_audio_track_code() {
    let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let at = tl.add_track(TrackType::Audio);
    let _id = tl
        .add_clip(at, Clip::new(1, ClipType::Audio, 0, 5000))
        .expect("add audio clip");
    let edl = TimelineExporter::to_edl(&tl);
    assert!(
        edl.contains(" A     C"),
        "Audio track must produce 'A' track code:\n{edl}"
    );
}

#[test]
fn test_edl_empty_timeline_contains_only_header() {
    let tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
    let edl = TimelineExporter::to_edl(&tl);
    assert!(edl.contains("TITLE:"));
    // No event lines expected.
    let event_lines: Vec<&str> = edl
        .lines()
        .filter(|l| l.len() > 3 && l.chars().take(3).all(|c| c.is_ascii_digit()))
        .collect();
    assert!(
        event_lines.is_empty(),
        "Empty timeline must have no events:\n{edl}"
    );
}

// ── EDL parse-back tests ─────────────────────────────────────────────────────

#[test]
fn test_edl_parse_back_event_count_matches() {
    let tl = make_two_clip_timeline();
    let edl_str = TimelineExporter::to_edl(&tl);
    let parsed = Edl::parse(&edl_str, EdlFormat::Cmx3600).expect("EDL parse should succeed");
    assert_eq!(
        parsed.events.len(),
        2,
        "Parsed EDL must have 2 events, got {}; source:\n{edl_str}",
        parsed.events.len()
    );
}

#[test]
fn test_edl_parse_back_event_numbers_sequential() {
    let tl = make_two_clip_timeline();
    let edl_str = TimelineExporter::to_edl(&tl);
    let parsed = Edl::parse(&edl_str, EdlFormat::Cmx3600).expect("EDL parse should succeed");
    assert_eq!(parsed.events[0].number, 1);
    assert_eq!(parsed.events[1].number, 2);
}

#[test]
fn test_edl_parse_back_record_out_greater_than_record_in() {
    let tl = make_two_clip_timeline();
    let edl_str = TimelineExporter::to_edl(&tl);
    let parsed = Edl::parse(&edl_str, EdlFormat::Cmx3600).expect("EDL parse should succeed");
    for evt in &parsed.events {
        let rec_in = evt.record_in.to_frames();
        let rec_out = evt.record_out.to_frames();
        assert!(
            rec_out > rec_in,
            "record_out ({rec_out}) must be > record_in ({rec_in}) for event {}",
            evt.number
        );
    }
}

#[test]
fn test_edl_parse_back_no_gaps_between_events() {
    let tl = make_two_clip_timeline();
    let edl_str = TimelineExporter::to_edl(&tl);
    let mut parsed = Edl::parse(&edl_str, EdlFormat::Cmx3600).expect("EDL parse should succeed");
    parsed.sort_events();
    // First event starts at record_in=0 frames.
    assert_eq!(parsed.events[0].record_in.to_frames(), 0);
    // Events are consecutive (rec_in[n] == rec_out[n-1]).
    let first_rec_out = parsed.events[0].record_out.to_frames();
    let second_rec_in = parsed.events[1].record_in.to_frames();
    assert_eq!(
        first_rec_out, second_rec_in,
        "Gap between events: event 1 rec_out={first_rec_out} event 2 rec_in={second_rec_in}"
    );
}

// ── FCP XML export tests ─────────────────────────────────────────────────────

#[test]
fn test_xml_contains_sequence_element() {
    let tl = make_two_clip_timeline();
    let xml = TimelineExporter::to_xml(&tl);
    assert!(
        xml.contains("<sequence"),
        "XML must contain <sequence element:\n{xml}"
    );
}

#[test]
fn test_xml_contains_xml_declaration() {
    let tl = make_two_clip_timeline();
    let xml = TimelineExporter::to_xml(&tl);
    assert!(
        xml.starts_with("<?xml"),
        "XML must start with declaration:\n{xml}"
    );
}

#[test]
fn test_xml_contains_video_element() {
    let tl = make_two_clip_timeline();
    let xml = TimelineExporter::to_xml(&tl);
    assert!(
        xml.contains("<video>"),
        "XML must contain <video> element:\n{xml}"
    );
}

#[test]
fn test_xml_contains_clipitem_elements() {
    let tl = make_two_clip_timeline();
    let xml = TimelineExporter::to_xml(&tl);
    assert!(
        xml.contains("<clipitem"),
        "XML must contain <clipitem elements:\n{xml}"
    );
}

#[test]
fn test_xml_contains_frame_rate() {
    let tl = make_two_clip_timeline();
    let xml = TimelineExporter::to_xml(&tl);
    assert!(
        xml.contains("<timebase>30</timebase>"),
        "XML must embed 30 fps timebase:\n{xml}"
    );
}
