//! EDL/XML timeline export for interchange with external NLEs.
//!
//! [`TimelineExporter`] converts an in-memory [`Timeline`] to:
//!
//! - **CMX-3600 EDL** — the industry-standard Edit Decision List format used
//!   by Avid, DaVinci Resolve, Final Cut Pro, and most broadcast systems.
//! - **Simple FCP XML** — a minimal Final Cut Pro 7 / FCP X-style XML skeleton
//!   that downstream tools can import.
//!
//! # Example
//!
//! ```
//! use oximedia_edit::{Timeline, TrackType};
//! use oximedia_edit::clip::{Clip, ClipType};
//! use oximedia_edit::timeline_export::TimelineExporter;
//! use oximedia_core::Rational;
//!
//! let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
//! let vt = tl.add_track(TrackType::Video);
//! let _ = tl.add_clip(vt, Clip::new(1, ClipType::Video, 0, 5000));
//!
//! let edl = TimelineExporter::to_edl(&tl);
//! assert!(edl.contains("TITLE:"));
//!
//! let xml = TimelineExporter::to_xml(&tl);
//! assert!(xml.contains("<sequence"));
//! ```

use crate::timeline::Timeline;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a timeline position (in timebase units, e.g. milliseconds at 1/1000)
/// to a frame count at the timeline's declared frame rate.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn ms_to_frames(position_ms: i64, fps: f64) -> u64 {
    let secs = position_ms as f64 / 1000.0;
    (secs * fps).round() as u64
}

/// Format a frame count as a CMX-3600 timecode string `HH:MM:SS:FF`.
fn frames_to_timecode(frames: u64, fps: u32) -> String {
    let fps = fps.max(1) as u64;
    let total_seconds = frames / fps;
    let ff = frames % fps;
    let hh = total_seconds / 3600;
    let mm = (total_seconds % 3600) / 60;
    let ss = total_seconds % 60;
    format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelineExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a [`Timeline`] to standard interchange formats.
pub struct TimelineExporter;

impl TimelineExporter {
    /// Export `timeline` as a **CMX-3600 EDL** string.
    ///
    /// Each video clip on each video track becomes one edit event.  Audio clips
    /// produce events with track code `A`.  Clips without an assigned source
    /// path use the reel name `AX` (the CMX-3600 convention for "any source").
    ///
    /// Timecodes are derived from the timeline's timebase (assumed to be 1/1000,
    /// i.e. milliseconds) and converted to the frame rate declared on the timeline.
    #[must_use]
    pub fn to_edl(timeline: &Timeline) -> String {
        let fps_rational = timeline.frame_rate;
        let fps_f64 = fps_rational.to_f64();
        let fps_u32 = fps_f64.round() as u32;

        let mut output = String::new();

        // Header
        output.push_str("TITLE: OxiMedia Timeline\n");
        output.push_str("FCM: NON-DROP FRAME\n\n");

        let mut event_number: u32 = 1;

        for track in &timeline.tracks {
            let track_code = match track.track_type {
                crate::timeline::TrackType::Video => "V",
                crate::timeline::TrackType::Audio => "A",
                crate::timeline::TrackType::Subtitle => "T",
            };

            for clip in &track.clips {
                // Reel name: use source file stem or "AX".
                let reel = clip
                    .source
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .and_then(|s| s.to_str())
                    .map(|s| {
                        // CMX reel names must be ≤ 8 chars, uppercase.
                        let truncated: String = s.chars().take(8).collect();
                        truncated.to_uppercase()
                    })
                    .unwrap_or_else(|| "AX".to_string());

                // Source timecodes (in/out of the source media).
                let src_in_frames = ms_to_frames(clip.source_in, fps_f64);
                let src_out_frames = ms_to_frames(clip.source_out, fps_f64);

                // Record timecodes (position on the timeline).
                let rec_in_frames = ms_to_frames(clip.timeline_start, fps_f64);
                let rec_out_frames = ms_to_frames(clip.timeline_end(), fps_f64);

                let src_in_tc = frames_to_timecode(src_in_frames, fps_u32);
                let src_out_tc = frames_to_timecode(src_out_frames, fps_u32);
                let rec_in_tc = frames_to_timecode(rec_in_frames, fps_u32);
                let rec_out_tc = frames_to_timecode(rec_out_frames, fps_u32);

                // CMX-3600 event line.
                output.push_str(&format!(
                    "{event_number:03}  {reel:<8} {track_code:<5} C        \
                     {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}\n"
                ));

                // Optional: clip name comment.
                if let Some(name) = &clip.metadata.name {
                    output.push_str(&format!("* FROM CLIP NAME: {name}\n"));
                } else if let Some(src) = &clip.source {
                    if let Some(name) = src.file_name().and_then(|n| n.to_str()) {
                        output.push_str(&format!("* FROM CLIP NAME: {name}\n"));
                    }
                }

                output.push('\n');
                event_number += 1;
            }
        }

        output
    }

    /// Export `timeline` as a **simple FCP XML** skeleton.
    ///
    /// The generated XML is compatible with Final Cut Pro 7 and can serve as a
    /// starting point for import into DaVinci Resolve, Premiere Pro (via the
    /// "Import FCP XML" workflow), and other NLEs.
    ///
    /// The skeleton includes:
    /// - A `<sequence>` element with frame-rate and duration attributes.
    /// - A `<media>` element containing `<video>` and `<audio>` spine tracks.
    /// - One `<clipitem>` per clip with source in/out and timeline in/out.
    #[must_use]
    pub fn to_xml(timeline: &Timeline) -> String {
        let fps_f64 = timeline.frame_rate.to_f64();
        let fps_u32 = fps_f64.round() as u32;

        let total_frames = ms_to_frames(timeline.duration, fps_f64);

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<!DOCTYPE xmeml>\n");
        xml.push_str("<xmeml version=\"5\">\n");
        xml.push_str("  <sequence>\n");
        xml.push_str(&format!(
            "    <name>OxiMedia Timeline</name>\n\
                 <duration>{total_frames}</duration>\n\
                 <rate>\n\
                   <timebase>{fps_u32}</timebase>\n\
                   <ntsc>FALSE</ntsc>\n\
                 </rate>\n"
        ));
        xml.push_str("    <media>\n");

        // ── Video tracks ────────────────────────────────────────────────────
        xml.push_str("      <video>\n");
        for (track_idx, track) in timeline.tracks.iter().enumerate() {
            if !matches!(track.track_type, crate::timeline::TrackType::Video) {
                continue;
            }
            let track_name = track
                .name
                .as_deref()
                .unwrap_or(&format!("Video {}", track_idx + 1))
                .to_string();

            xml.push_str("        <track>\n");
            xml.push_str(&format!("          <name>{track_name}</name>\n"));

            for (clip_idx, clip) in track.clips.iter().enumerate() {
                let clip_name = clip
                    .metadata
                    .name
                    .as_deref()
                    .map(String::from)
                    .or_else(|| {
                        clip.source
                            .as_ref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .map(String::from)
                    })
                    .unwrap_or_else(|| format!("clip_{}", clip_idx + 1));

                let src_in = ms_to_frames(clip.source_in, fps_f64);
                let src_out = ms_to_frames(clip.source_out, fps_f64);
                let in_point = ms_to_frames(clip.timeline_start, fps_f64);
                let out_point = ms_to_frames(clip.timeline_end(), fps_f64);

                xml.push_str(&format!(
                    "          <clipitem id=\"clip-{track_idx}-{clip_idx}\">\n\
                     \t            <name>{clip_name}</name>\n\
                     \t            <start>{in_point}</start>\n\
                     \t            <end>{out_point}</end>\n\
                     \t            <in>{src_in}</in>\n\
                     \t            <out>{src_out}</out>\n\
                     \t          </clipitem>\n"
                ));
            }

            xml.push_str("        </track>\n");
        }
        xml.push_str("      </video>\n");

        // ── Audio tracks ────────────────────────────────────────────────────
        xml.push_str("      <audio>\n");
        for (track_idx, track) in timeline.tracks.iter().enumerate() {
            if !matches!(track.track_type, crate::timeline::TrackType::Audio) {
                continue;
            }
            let track_name = track
                .name
                .as_deref()
                .unwrap_or(&format!("Audio {}", track_idx + 1))
                .to_string();

            xml.push_str("        <track>\n");
            xml.push_str(&format!("          <name>{track_name}</name>\n"));

            for (clip_idx, clip) in track.clips.iter().enumerate() {
                let clip_name = clip
                    .metadata
                    .name
                    .as_deref()
                    .map(String::from)
                    .or_else(|| {
                        clip.source
                            .as_ref()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            .map(String::from)
                    })
                    .unwrap_or_else(|| format!("audioclip_{}", clip_idx + 1));

                let src_in = ms_to_frames(clip.source_in, fps_f64);
                let src_out = ms_to_frames(clip.source_out, fps_f64);
                let in_point = ms_to_frames(clip.timeline_start, fps_f64);
                let out_point = ms_to_frames(clip.timeline_end(), fps_f64);

                xml.push_str(&format!(
                    "          <clipitem id=\"audioclip-{track_idx}-{clip_idx}\">\n\
                     \t            <name>{clip_name}</name>\n\
                     \t            <start>{in_point}</start>\n\
                     \t            <end>{out_point}</end>\n\
                     \t            <in>{src_in}</in>\n\
                     \t            <out>{src_out}</out>\n\
                     \t          </clipitem>\n"
                ));
            }

            xml.push_str("        </track>\n");
        }
        xml.push_str("      </audio>\n");

        xml.push_str("    </media>\n");
        xml.push_str("  </sequence>\n");
        xml.push_str("</xmeml>\n");

        xml
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, ClipType};
    use crate::timeline::{Timeline, TrackType};
    use oximedia_core::Rational;

    fn make_test_timeline() -> Timeline {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let at = tl.add_track(TrackType::Audio);

        // Video clip: 0–5000 ms (0–150 frames at 30fps)
        let _ = tl.add_clip(vt, Clip::new(1, ClipType::Video, 0, 5000));
        // Another video clip: 5000–8000 ms
        let _ = tl.add_clip(vt, Clip::new(2, ClipType::Video, 5000, 3000));
        // Audio clip: 0–8000 ms
        let _ = tl.add_clip(at, Clip::new(3, ClipType::Audio, 0, 8000));

        tl
    }

    // ── EDL tests ───────────────────────────────────────────────────────────

    /// Required test: EDL output must contain at least one event entry.
    #[test]
    fn test_timeline_edl_export_has_events() {
        let tl = make_test_timeline();
        let edl = TimelineExporter::to_edl(&tl);

        // Must have the standard CMX-3600 header.
        assert!(edl.contains("TITLE:"), "EDL must have a TITLE line");
        assert!(edl.contains("FCM:"), "EDL must have an FCM line");

        // Must contain at least one numbered event line (e.g. "001  AX       V").
        assert!(
            edl.contains("001  "),
            "EDL must have at least one event: got:\n{edl}"
        );

        // Timecodes must appear in the correct HH:MM:SS:FF format.
        assert!(
            edl.contains("00:00:00:00"),
            "First clip should start at 00:00:00:00"
        );

        // Video events use 'V', audio events use 'A'.
        assert!(edl.contains(" V     "), "EDL must have a video event");
        assert!(edl.contains(" A     "), "EDL must have an audio event");
    }

    #[test]
    fn test_edl_empty_timeline() {
        let tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let edl = TimelineExporter::to_edl(&tl);
        assert!(edl.contains("TITLE:"));
        assert!(edl.contains("FCM:"));
        // No event lines.
        assert!(
            !edl.contains("001  "),
            "empty timeline should have no events"
        );
    }

    #[test]
    fn test_edl_timecode_accuracy() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        // 5000 ms = 5 s × 30fps = 150 frames → 00:00:05:00
        let _ = tl.add_clip(vt, Clip::new(1, ClipType::Video, 0, 5000));

        let edl = TimelineExporter::to_edl(&tl);
        // Record out should be 00:00:05:00.
        assert!(
            edl.contains("00:00:05:00"),
            "5000 ms at 30fps should produce timecode 00:00:05:00; got:\n{edl}"
        );
    }

    #[test]
    fn test_edl_source_name_from_metadata() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);

        let mut clip = Clip::new(1, ClipType::Video, 0, 3000);
        clip.metadata.name = Some("Interview".to_string());
        let _ = tl.add_clip(vt, clip);

        let edl = TimelineExporter::to_edl(&tl);
        assert!(
            edl.contains("* FROM CLIP NAME: Interview"),
            "Clip name comment should appear; got:\n{edl}"
        );
    }

    #[test]
    fn test_edl_event_count_matches_clips() {
        let tl = make_test_timeline();
        let edl = TimelineExporter::to_edl(&tl);

        // 3 clips → events 001, 002, 003.
        assert!(edl.contains("001  "), "event 001 missing");
        assert!(edl.contains("002  "), "event 002 missing");
        assert!(edl.contains("003  "), "event 003 missing");
        assert!(!edl.contains("004  "), "unexpected event 004 found");
    }

    // ── XML tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_xml_structure() {
        let tl = make_test_timeline();
        let xml = TimelineExporter::to_xml(&tl);

        assert!(xml.contains("<sequence"), "XML must contain <sequence");
        assert!(xml.contains("</sequence>"), "XML must contain </sequence>");
        assert!(xml.contains("<video>"), "XML must contain <video>");
        assert!(xml.contains("<audio>"), "XML must contain <audio>");
        assert!(xml.contains("<clipitem"), "XML must contain <clipitem");
        assert!(xml.contains("<rate>"), "XML must contain <rate>");
        assert!(
            xml.contains("<timebase>30</timebase>"),
            "timebase should be 30"
        );
    }

    #[test]
    fn test_xml_duration_field() {
        let tl = make_test_timeline();
        let xml = TimelineExporter::to_xml(&tl);

        // Timeline ends at 8000 ms = 240 frames at 30fps.
        assert!(
            xml.contains("<duration>240</duration>"),
            "duration should be 240 frames; got:\n{xml}"
        );
    }

    #[test]
    fn test_xml_valid_xml_like_structure() {
        let tl = make_test_timeline();
        let xml = TimelineExporter::to_xml(&tl);

        // Every opened tag should have a matching close.
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("</xmeml>"));
        assert!(xml.starts_with("<?xml"));
        assert!(xml.ends_with("</xmeml>\n"));
    }

    // ── Helper function tests ────────────────────────────────────────────────

    #[test]
    fn test_frames_to_timecode() {
        // 150 frames at 30fps = 00:00:05:00.
        assert_eq!(frames_to_timecode(150, 30), "00:00:05:00");
        // 0 frames.
        assert_eq!(frames_to_timecode(0, 30), "00:00:00:00");
        // 108000 frames at 30fps = exactly 1 hour.
        assert_eq!(frames_to_timecode(108000, 30), "01:00:00:00");
        // 32 frames at 30fps = 1 second + 2 frames.
        assert_eq!(frames_to_timecode(32, 30), "00:00:01:02");
    }

    #[test]
    fn test_ms_to_frames() {
        // 1000 ms at 30fps = 30 frames.
        assert_eq!(ms_to_frames(1000, 30.0), 30);
        // 5000 ms at 30fps = 150 frames.
        assert_eq!(ms_to_frames(5000, 30.0), 150);
        // 0 ms = 0 frames.
        assert_eq!(ms_to_frames(0, 30.0), 0);
    }
}
