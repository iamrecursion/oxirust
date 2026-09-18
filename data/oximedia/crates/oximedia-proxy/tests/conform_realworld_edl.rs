//! Tests `ConformEngine` against real-world-shaped EDL samples: CMX 3600 and
//! Final Cut Pro XML.
//!
//! Two layers are exercised, matching what is actually implemented today:
//!
//! 1. **Real parsing + real merge logic.** CMX 3600 text (as two independent
//!    editors' cut lists for the same reel would look) is parsed with the
//!    production `oximedia_edl::parse_edl` parser, then fed into
//!    `ConformEngine::batch_conform` — a fully implemented merge algorithm —
//!    to verify correct overlap resolution and provenance tracking against
//!    realistic multi-event, multi-reel, commented EDL text (not just
//!    synthetic single-event fixtures).
//! 2. **The file-based conform seam.** `ConformEngine::conform_from_edl`
//!    (CMX 3600 path) and `XmlConformer` (FCP XML path) are exercised
//!    directly against real-world-shaped fixture files on disk. As of this
//!    writing both are existence-check-only placeholders (see
//!    `src/conform/edl.rs` and `src/conform/xml.rs`), so these tests pin the
//!    honest current contract (file must exist; fixed `ConformResult` /
//!    format-detection is returned) rather than fabricating relink counts.
//!
//! Final Cut Pro XML parsing itself (`oximedia_edl::fcpxml::parse_fcpxml`) is
//! validated against a realistic multi-clip-with-transition sequence to prove
//! the "real-world EDL sample" claim for the FCP XML side too.

use oximedia_edl::fcpxml::parse_fcpxml;
use oximedia_edl::parse_edl;
use oximedia_proxy::conform::engine::MergeStrategy;
use oximedia_proxy::conform::XmlFormat;
use oximedia_proxy::{ConformEngine, ProxyLinkManager, XmlConformer};

// ── Real-world-shaped CMX 3600 fixtures ─────────────────────────────────────

/// Editor A's rough cut: two reels, wide shot then close-up, back to back.
const CMX_EDITOR_A: &str = r#"TITLE: Interview Scene - Editor A Cut
FCM: NON-DROP FRAME

001  A001     V     C        01:00:00:00 01:00:10:00 01:00:00:00 01:00:10:00
* FROM CLIP NAME: interview_wide.mov
002  A002     V     C        01:00:00:00 01:00:08:00 01:00:10:00 01:00:18:00
* FROM CLIP NAME: interview_cu.mov
"#;

/// Editor B's alternate cut of the same scene: a longer close-up take that
/// overlaps Editor A's second event (both start their close-up at 10s).
const CMX_EDITOR_B: &str = r#"TITLE: Interview Scene - Editor B Cut
FCM: NON-DROP FRAME

001  B001     V     C        01:00:00:00 01:00:12:00 01:00:10:00 01:00:22:00
* FROM CLIP NAME: interview_cu_alt_angle.mov
"#;

#[test]
fn test_parse_real_world_cmx3600_editor_a() {
    let edl = parse_edl(CMX_EDITOR_A).expect("real-world CMX 3600 sample must parse");
    assert_eq!(
        edl.title,
        Some("Interview Scene - Editor A Cut".to_string())
    );
    assert_eq!(edl.events.len(), 2);
    assert_eq!(edl.events[0].reel, "A001");
    assert_eq!(edl.events[1].reel, "A002");
}

#[test]
fn test_parse_real_world_cmx3600_editor_b() {
    let edl = parse_edl(CMX_EDITOR_B).expect("real-world CMX 3600 sample must parse");
    assert_eq!(edl.events.len(), 1);
    assert_eq!(edl.events[0].reel, "B001");
}

/// Feed both real, independently-parsed CMX 3600 EDLs into the real
/// `batch_conform` merge logic and verify overlap resolution + provenance
/// against realistic (not single-event-toy) data.
#[tokio::test]
async fn test_batch_conform_real_world_cmx3600_prefer_earlier() {
    let edl_a = parse_edl(CMX_EDITOR_A).expect("editor A EDL must parse");
    let edl_b = parse_edl(CMX_EDITOR_B).expect("editor B EDL must parse");

    let db = std::env::temp_dir().join(format!(
        "conform_realworld_edl_test_a_{}.json",
        std::process::id()
    ));
    let engine = ConformEngine::new(&db)
        .await
        .expect("conform engine must open");
    let _ = std::fs::remove_file(&db);

    let result = engine.batch_conform(&[edl_a, edl_b], MergeStrategy::PreferEarlier);

    // Editor A's wide shot (0-10s) does not overlap anything and survives.
    // Editor A's and B's close-ups both start at 10s and overlap; with
    // PreferEarlier and equal record-in, the first-encountered (source EDL 0)
    // wins.
    assert_eq!(result.events.len(), 2);
    assert!(result
        .provenance
        .iter()
        .any(|p| p.source_edl_index == 0 && p.event_index == 0));
    let overlap_winner = &result.provenance[1];
    assert_eq!(
        overlap_winner.source_edl_index, 0,
        "PreferEarlier must keep editor A's close-up when record-in times tie"
    );
}

/// Same real-world data, `PreferLonger` strategy: Editor B's alternate-angle
/// close-up (12s) is longer than Editor A's (8s) and must win the overlap.
#[tokio::test]
async fn test_batch_conform_real_world_cmx3600_prefer_longer() {
    let edl_a = parse_edl(CMX_EDITOR_A).expect("editor A EDL must parse");
    let edl_b = parse_edl(CMX_EDITOR_B).expect("editor B EDL must parse");

    let db = std::env::temp_dir().join(format!(
        "conform_realworld_edl_test_b_{}.json",
        std::process::id()
    ));
    let engine = ConformEngine::new(&db)
        .await
        .expect("conform engine must open");
    let _ = std::fs::remove_file(&db);

    let result = engine.batch_conform(&[edl_a, edl_b], MergeStrategy::PreferLonger);

    assert_eq!(result.events.len(), 2);
    let overlap_winner = &result.provenance[1];
    assert_eq!(
        overlap_winner.source_edl_index, 1,
        "PreferLonger must keep editor B's longer close-up take"
    );
}

// ── File-based conform seam: CMX 3600 EDL, real-world-shaped fixture ───────

#[tokio::test]
async fn test_conform_from_edl_real_world_cmx3600_file() {
    let dir = std::env::temp_dir().join(format!(
        "conform_realworld_edl_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let edl_path = dir.join("editor_a_cut.edl");
    std::fs::write(&edl_path, CMX_EDITOR_A).expect("write real-world EDL fixture");

    let db_path = dir.join("links.json");
    let engine = ConformEngine::new(&db_path)
        .await
        .expect("conform engine must open");

    let output = dir.join("conformed.mov");
    let result = engine
        .conform_from_edl(&edl_path, &output)
        .await
        .expect("conform_from_edl must succeed for an existing real-world EDL file");

    // Current honest contract: existence-check-only (see module doc).
    assert_eq!(result.output_path, output);
    assert!(result.frame_accurate);

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Real-world-shaped Final Cut Pro XML fixture ────────────────────────────

/// A realistic 3-clip FCP XML sequence with a cross-dissolve transition,
/// modeled on genuine FCP XML export structure (sequence/rate/media/video
/// track/clipitem/transitionitem).
const FCPXML_REAL_WORLD: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <sequence>
    <name>Documentary Rough Cut v3</name>
    <duration>625</duration>
    <rate>
      <timebase>25</timebase>
      <ntsc>FALSE</ntsc>
    </rate>
    <media>
      <video>
        <track>
          <clipitem id="clipitem-1">
            <name>b-roll_harbor_sunrise.mov</name>
            <file>b-roll_harbor_sunrise.mov</file>
            <in>0</in>
            <out>250</out>
            <start>0</start>
            <end>250</end>
            <reel>A001</reel>
          </clipitem>
          <transitionitem>
            <name>Cross Dissolve</name>
            <duration>25</duration>
            <start>238</start>
          </transitionitem>
          <clipitem id="clipitem-2">
            <name>interview_captain_wide.mov</name>
            <file>interview_captain_wide.mov</file>
            <in>500</in>
            <out>875</out>
            <start>250</start>
            <end>625</end>
            <reel>A002</reel>
          </clipitem>
        </track>
      </video>
    </media>
  </sequence>
</xmeml>"#;

#[test]
fn test_parse_real_world_fcpxml_sequence() {
    let seq = parse_fcpxml(FCPXML_REAL_WORLD).expect("real-world FCP XML sample must parse");
    assert_eq!(seq.name, "Documentary Rough Cut v3");
    assert_eq!(seq.timebase, 25);
    assert!(!seq.ntsc);
    assert_eq!(seq.duration_frames, 625);
    assert_eq!(seq.clip_count(), 2);

    assert_eq!(seq.clips[0].name, "b-roll_harbor_sunrise.mov");
    assert_eq!(seq.clips[0].reel, Some("A001".to_string()));
    assert_eq!(seq.clips[0].timeline_in, 0);
    assert_eq!(seq.clips[0].timeline_out, 250);

    assert_eq!(seq.clips[1].name, "interview_captain_wide.mov");
    assert_eq!(seq.clips[1].reel, Some("A002".to_string()));
    assert_eq!(seq.clips[1].source_in, 500);
    assert_eq!(seq.clips[1].source_out, 875);
    assert_eq!(seq.clips[1].duration_frames(), 375);

    assert_eq!(seq.transitions.len(), 1);
    assert_eq!(seq.transitions[0].name, "Cross Dissolve");
    assert_eq!(seq.transitions[0].duration_frames, 25);
}

// ── File-based conform seam: FCP XML, real-world-shaped fixture ────────────

#[tokio::test]
async fn test_xml_conformer_real_world_fcpxml_file() {
    let dir = std::env::temp_dir().join(format!(
        "conform_realworld_fcpxml_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let xml_path = dir.join("rough_cut_v3.xml");
    std::fs::write(&xml_path, FCPXML_REAL_WORLD).expect("write real-world FCP XML fixture");

    let db_path = dir.join("links.json");
    let manager = ProxyLinkManager::new(&db_path)
        .await
        .expect("link manager must open");
    let conformer = XmlConformer::new(&manager);

    // Current honest contract: `detect_format` always reports FinalCutPro
    // (see `src/conform/xml.rs`) rather than sniffing the XML structure —
    // pin that contract explicitly.
    let format = conformer
        .detect_format(&xml_path)
        .expect("detect_format must succeed for an existing file");
    assert_eq!(format, XmlFormat::FinalCutPro);

    let output = dir.join("conformed.xml");
    let result = conformer
        .conform(&xml_path, &output)
        .await
        .expect("conform must succeed for an existing real-world FCP XML file");
    assert_eq!(result.output_path, output);
    assert!(result.frame_accurate);

    let _ = std::fs::remove_dir_all(&dir);
}
