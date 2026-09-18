//! Integration tests for ISO (isolated) per-angle recording management.
//!
//! Wave 29 / Slice 8 — PURE test-hardening of `iso_recording`.
//!
//! Verifies per-angle stream configuration, the start/stop recording
//! lifecycle, that every angle survives into the resulting session, distinct
//! per-angle file names, multi-quality ordering, and the recording guard
//! conditions. The module is purely in-memory (it does not write files), so
//! file-name assertions check prefix/suffix/substring only — never the full
//! string, which embeds today's UTC date.

use std::collections::HashSet;

use oximedia_multicam::iso_recording::{IsoFileNaming, IsoQuality, IsoRecorder, IsoStream};

/// Four camera angles each configured at FullRes produce four streams, and a
/// full record lifecycle preserves every angle in the resulting session.
#[test]
fn test_four_angle_iso_lifecycle() {
    let mut recorder = IsoRecorder::new();
    for cam in 0..4u32 {
        recorder.add_stream(IsoStream::new(cam, IsoQuality::FullRes, "ProRes422", 2));
    }
    assert_eq!(recorder.streams().len(), 4);
    assert!(!recorder.is_recording());

    // Start recording.
    assert!(recorder.start_recording("SESS001"));
    assert!(recorder.is_recording());

    // Stop and collect the session summary.
    let session = recorder
        .stop_recording()
        .expect("stop_recording should return a session while recording");
    assert!(!recorder.is_recording());
    assert_eq!(session.session_id, "SESS001");

    // Every angle survives into the session — no angle dropped.
    assert_eq!(session.streams.len(), 4);
    let mut cam_ids: Vec<u32> = session.streams.iter().map(|s| s.camera_id).collect();
    cam_ids.sort_unstable();
    assert_eq!(cam_ids, vec![0, 1, 2, 3]);
}

/// Generated per-angle file names share the ISO/quality envelope, embed the
/// session id, and are all distinct across angles.
#[test]
fn test_iso_per_angle_file_names() {
    let session_id = "SESS001";
    let names: Vec<String> = (0..4u32)
        .map(|cam| IsoFileNaming::generate(cam, session_id, &IsoQuality::FullRes))
        .collect();

    for (cam, name) in names.iter().enumerate() {
        // camera_id is zero-padded to two digits: CAM00..CAM03.
        let prefix = format!("ISO_CAM0{cam}_");
        assert!(
            name.starts_with(&prefix),
            "name {name:?} should start with {prefix:?}"
        );
        assert!(
            name.ends_with("_FULL.mxf"),
            "name {name:?} should end with _FULL.mxf"
        );
        assert!(
            name.contains("SESS001"),
            "name {name:?} should contain the uppercased session id"
        );
    }

    // All four file names are unique.
    let unique: HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), 4);
}

/// Adding multiple quality levels for one camera yields one stream per quality,
/// and `streams_by_quality` orders them by ascending bitrate.
#[test]
fn test_iso_multi_quality_ordering() {
    let mut recorder = IsoRecorder::new();
    recorder.add_stream(IsoStream::new(0, IsoQuality::Raw, "RAW", 2));
    recorder.add_stream(IsoStream::new(0, IsoQuality::Proxy, "H.264", 2));
    recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));
    assert_eq!(recorder.streams().len(), 3);

    assert!(recorder.start_recording("MULTIQ"));
    let session = recorder
        .stop_recording()
        .expect("stop_recording should return a session while recording");
    assert_eq!(session.streams.len(), 3);

    // Ascending bitrate: Proxy (0.1) < FullRes (1.0) < Raw (5.0).
    let ordered: Vec<IsoQuality> = session
        .streams_by_quality()
        .iter()
        .map(|s| s.quality)
        .collect();
    assert_eq!(
        ordered,
        vec![IsoQuality::Proxy, IsoQuality::FullRes, IsoQuality::Raw]
    );
}

/// Adding a stream with the same camera_id + quality replaces (dedups) rather
/// than appends, so each (camera, quality) pair maps to one stream.
#[test]
fn test_iso_dedup_on_camera_and_quality() {
    let mut recorder = IsoRecorder::new();
    recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));
    recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "DNxHD", 4));
    // Same (camera 0, FullRes) → replaced, not duplicated.
    assert_eq!(recorder.streams().len(), 1);
    assert_eq!(recorder.streams()[0].codec, "DNxHD");
    assert_eq!(recorder.streams()[0].audio_channels, 4);

    // A different quality for the same camera is a distinct stream.
    recorder.add_stream(IsoStream::new(0, IsoQuality::Proxy, "H.264", 2));
    assert_eq!(recorder.streams().len(), 2);
}

/// A second `start_recording` while already recording returns false and does
/// not change the active session.
#[test]
fn test_iso_double_start_rejected() {
    let mut recorder = IsoRecorder::new();
    recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));

    assert!(recorder.start_recording("FIRST"));
    assert!(!recorder.start_recording("SECOND"));
    assert!(recorder.is_recording());

    // The original session id wins.
    let session = recorder
        .stop_recording()
        .expect("stop_recording should return a session while recording");
    assert_eq!(session.session_id, "FIRST");
}

/// Stopping when not recording (and stopping twice) returns None.
#[test]
fn test_iso_stop_when_not_recording() {
    let mut recorder = IsoRecorder::new();
    recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));

    // Never started.
    assert!(recorder.stop_recording().is_none());

    // Start, stop once (ok), stop again (None).
    assert!(recorder.start_recording("S1"));
    assert!(recorder.stop_recording().is_some());
    assert!(recorder.stop_recording().is_none());
    assert!(!recorder.is_recording());
}
