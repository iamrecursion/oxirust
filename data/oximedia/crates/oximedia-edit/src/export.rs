//! Timeline export: CMX3600 EDL, FCP-compatible XML, and CSV clip-list.
//!
//! [`TimelineExporter`] wraps a [`Timeline`] reference and provides three
//! serialisation methods that do not require I/O — they return `String` so the
//! caller can write to a file, send over a network, etc.
//!
//! # Supported formats
//!
//! | Method | Format |
//! |---|---|
//! | [`export_edl`] | CMX 3600 (industry-standard linear EDL) |
//! | [`export_xml`] | Basic FCP 7-style XML |
//! | [`export_csv`] | Simple CSV clip list |
//!
//! [`export_edl`]: TimelineExporter::export_edl
//! [`export_xml`]: TimelineExporter::export_xml
//! [`export_csv`]: TimelineExporter::export_csv

use crate::clip::Clip;
use crate::timeline::{Timeline, TrackType};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format a raw frame count as a SMPTE timecode string `HH:MM:SS:FF`.
///
/// `fps` is the frames-per-second denominator (e.g. 30).
#[must_use]
fn frames_to_tc(frames: i64, fps: i64) -> String {
    let fps = fps.max(1);
    let total_frames = frames.max(0);
    let ff = total_frames % fps;
    let total_secs = total_frames / fps;
    let ss = total_secs % 60;
    let total_mins = total_secs / 60;
    let mm = total_mins % 60;
    let hh = total_mins / 60;
    format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
}

/// Convert timeline units (milliseconds at default timebase) to frame count.
///
/// `timebase_num / timebase_den` gives seconds-per-unit.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn units_to_frames(units: i64, timebase_num: i64, timebase_den: i64, fps: f64) -> i64 {
    // seconds = units * (timebase_num / timebase_den)
    let secs = units as f64 * (timebase_num as f64 / timebase_den as f64);
    (secs * fps).round() as i64
}

// ─────────────────────────────────────────────────────────────────────────────
// ExportClipInfo  (flattened view of a Clip used by all exporters)
// ─────────────────────────────────────────────────────────────────────────────

/// Flattened, serialisable view of one clip suitable for all export formats.
#[derive(Debug, Clone)]
pub struct ExportClipInfo {
    /// Sequential 1-based event number.
    pub event_number: u32,
    /// Track index (0-based).
    pub track_index: usize,
    /// Track type label: `"V"`, `"A"`, or `"SUB"`.
    pub track_label: String,
    /// Reel / source name (file stem or "AX" if unknown).
    pub reel_name: String,
    /// Clip display name.
    pub clip_name: String,
    /// Source in-point as SMPTE timecode.
    pub source_in_tc: String,
    /// Source out-point as SMPTE timecode.
    pub source_out_tc: String,
    /// Record in-point (timeline) as SMPTE timecode.
    pub record_in_tc: String,
    /// Record out-point (timeline) as SMPTE timecode.
    pub record_out_tc: String,
    /// Speed multiplier (1.0 = normal).
    pub speed: f64,
    /// Reverse flag.
    pub reverse: bool,
    /// Clip opacity / volume (0.0–1.0).
    pub opacity: f32,
    /// Muted flag.
    pub muted: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelineExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Exports a [`Timeline`] to various interchange formats.
pub struct TimelineExporter<'a> {
    timeline: &'a Timeline,
    /// Title embedded in EDL and XML headers.
    pub title: String,
    /// Nominal frame rate used for timecode arithmetic (default: 30).
    pub fps: f64,
    /// Drop-frame mode for CMX 3600 timecodes.
    pub drop_frame: bool,
}

impl<'a> TimelineExporter<'a> {
    /// Create an exporter for `timeline`.
    ///
    /// The frame rate is derived from `timeline.frame_rate`; the title
    /// defaults to `"Untitled"`.
    #[must_use]
    pub fn new(timeline: &'a Timeline) -> Self {
        let fps = timeline.frame_rate.to_f64().max(1.0);
        Self {
            timeline,
            title: "Untitled".to_string(),
            fps,
            drop_frame: false,
        }
    }

    /// Override the title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Override the frames-per-second value.
    #[must_use]
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = fps.max(1.0);
        self
    }

    /// Enable drop-frame mode in EDL output.
    #[must_use]
    pub fn with_drop_frame(mut self, drop_frame: bool) -> Self {
        self.drop_frame = drop_frame;
        self
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn clip_to_tc(&self, units: i64) -> String {
        let tb = &self.timeline.timebase;
        let f = units_to_frames(units, tb.num, tb.den, self.fps);
        frames_to_tc(f, self.fps.round() as i64)
    }

    fn clip_reel_name(clip: &Clip) -> String {
        clip.source
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("AX")
            .to_string()
    }

    fn clip_display_name(clip: &Clip) -> String {
        clip.metadata
            .name
            .clone()
            .unwrap_or_else(|| format!("clip_{}", clip.id))
    }

    fn track_label(track_type: TrackType) -> &'static str {
        match track_type {
            TrackType::Video => "V",
            TrackType::Audio => "A",
            TrackType::Subtitle => "SUB",
        }
    }

    /// Collect all clips from the timeline in chronological order per track,
    /// assigning sequential event numbers.
    fn collect_clips(&self) -> Vec<ExportClipInfo> {
        let mut infos = Vec::new();
        let mut event_num: u32 = 1;

        for track in &self.timeline.tracks {
            // Clips are already sorted by timeline_start
            for clip in &track.clips {
                let reel = Self::clip_reel_name(clip);
                let name = Self::clip_display_name(clip);
                let label = Self::track_label(track.track_type);

                let src_in_tc = self.clip_to_tc(clip.source_in);
                let src_out_tc = self.clip_to_tc(clip.source_out);
                let rec_in_tc = self.clip_to_tc(clip.timeline_start);
                let rec_out_tc = self.clip_to_tc(clip.timeline_end());

                infos.push(ExportClipInfo {
                    event_number: event_num,
                    track_index: track.index,
                    track_label: label.to_string(),
                    reel_name: reel,
                    clip_name: name,
                    source_in_tc: src_in_tc,
                    source_out_tc: src_out_tc,
                    record_in_tc: rec_in_tc,
                    record_out_tc: rec_out_tc,
                    speed: clip.speed,
                    reverse: clip.reverse,
                    opacity: clip.opacity,
                    muted: clip.muted,
                });
                event_num += 1;
            }
        }

        infos
    }

    // ── Public export methods ─────────────────────────────────────────────

    /// Export as CMX 3600 EDL.
    ///
    /// The output follows the standard header-then-events layout:
    ///
    /// ```text
    /// TITLE: My Project
    /// FCM: NON-DROP FRAME
    ///
    /// 001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
    /// * FROM CLIP NAME: clip_1
    /// ```
    #[must_use]
    pub fn export_edl(&self) -> String {
        let clips = self.collect_clips();
        let mut out = String::new();

        // Header
        out.push_str(&format!("TITLE: {}\n", self.title));
        let fcm = if self.drop_frame {
            "DROP FRAME"
        } else {
            "NON-DROP FRAME"
        };
        out.push_str(&format!("FCM: {fcm}\n\n"));

        for info in &clips {
            // Event line
            out.push_str(&format!(
                "{:03}  {:<8} {:<5} C        {} {} {} {}\n",
                info.event_number,
                info.reel_name,
                info.track_label,
                info.source_in_tc,
                info.source_out_tc,
                info.record_in_tc,
                info.record_out_tc,
            ));

            // Clip name comment
            out.push_str(&format!("* FROM CLIP NAME: {}\n", info.clip_name));

            // Speed / motion effects
            if (info.speed - 1.0).abs() > 1e-6 || info.reverse {
                let speed_code = if info.reverse {
                    -info.speed.abs()
                } else {
                    info.speed
                };
                out.push_str(&format!(
                    "M2   {:<8} {:03}   {}\n",
                    info.reel_name,
                    (speed_code * 100.0).round() as i32,
                    info.record_in_tc,
                ));
                if info.reverse {
                    out.push_str("* REVERSE MOTION\n");
                }
            }

            // Mute comment
            if info.muted {
                out.push_str("* MUTED\n");
            }

            out.push('\n');
        }

        out
    }

    /// Export as a basic FCP 7-compatible XML string.
    ///
    /// The structure is intentionally minimal: `<xmeml>` → `<sequence>` →
    /// `<media>` → one `<video>` block and one `<audio>` block each containing
    /// `<track>` elements.  The format is readable by most NLEs that support
    /// FCP XML.
    #[must_use]
    pub fn export_xml(&self) -> String {
        let clips = self.collect_clips();
        let fps_int = self.fps.round() as u32;
        let tb = &self.timeline.timebase;
        // timebase denominator (e.g. 1000 for ms)
        let tb_den = tb.den;
        let total_tc = self.clip_to_tc(self.timeline.duration);

        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<!DOCTYPE xmeml>\n");
        out.push_str("<xmeml version=\"5\">\n");
        out.push_str("  <sequence>\n");
        out.push_str(&format!("    <name>{}</name>\n", xml_escape(&self.title)));
        out.push_str(&format!("    <duration>{tb_den}</duration>\n"));
        out.push_str("    <rate>\n");
        out.push_str(&format!("      <timebase>{fps_int}</timebase>\n"));
        out.push_str(&format!(
            "      <ntsc>{}</ntsc>\n",
            if self.drop_frame { "TRUE" } else { "FALSE" }
        ));
        out.push_str("    </rate>\n");
        out.push_str("    <timecode>\n");
        out.push_str(&format!("      <string>{total_tc}</string>\n"));
        out.push_str("    </timecode>\n");
        out.push_str("    <media>\n");

        // ── video ─────────────────────────────────────────────────────────
        let video_clips: Vec<&ExportClipInfo> =
            clips.iter().filter(|c| c.track_label == "V").collect();

        if !video_clips.is_empty() {
            out.push_str("      <video>\n");
            out.push_str("        <track>\n");
            for info in &video_clips {
                write_xml_clip_item(&mut out, info, self.fps, tb.num, tb.den);
            }
            out.push_str("        </track>\n");
            out.push_str("      </video>\n");
        }

        // ── audio ─────────────────────────────────────────────────────────
        let audio_clips: Vec<&ExportClipInfo> =
            clips.iter().filter(|c| c.track_label == "A").collect();

        if !audio_clips.is_empty() {
            out.push_str("      <audio>\n");
            out.push_str("        <track>\n");
            for info in &audio_clips {
                write_xml_clip_item(&mut out, info, self.fps, tb.num, tb.den);
            }
            out.push_str("        </track>\n");
            out.push_str("      </audio>\n");
        }

        out.push_str("    </media>\n");
        out.push_str("  </sequence>\n");
        out.push_str("</xmeml>\n");

        out
    }

    /// Export as a CSV clip list.
    ///
    /// The header row is:
    /// `Event,Track,Reel,Name,SourceIn,SourceOut,RecordIn,RecordOut,Speed,Reverse,Opacity,Muted`
    #[must_use]
    pub fn export_csv(&self) -> String {
        let clips = self.collect_clips();
        let mut out = String::new();

        // Header
        out.push_str(
            "Event,Track,Reel,Name,SourceIn,SourceOut,RecordIn,RecordOut,Speed,Reverse,Opacity,Muted\n",
        );

        for info in &clips {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.6},{},{:.4},{}\n",
                info.event_number,
                info.track_label,
                csv_escape(&info.reel_name),
                csv_escape(&info.clip_name),
                info.source_in_tc,
                info.source_out_tc,
                info.record_in_tc,
                info.record_out_tc,
                info.speed,
                info.reverse,
                info.opacity,
                info.muted,
            ));
        }

        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// XML helpers
// ─────────────────────────────────────────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn write_xml_clip_item(
    out: &mut String,
    info: &ExportClipInfo,
    fps: f64,
    tb_num: i64,
    tb_den: i64,
) {
    // Convert SMPTE timecodes back to raw frame counts for `<in>` / `<out>` tags.
    // We use the record timecodes as a start/end pair in the sequence.
    let rec_in_frames = tc_to_frames(&info.record_in_tc, fps.round() as i64);
    let rec_out_frames = tc_to_frames(&info.record_out_tc, fps.round() as i64);
    let src_in_frames = tc_to_frames(&info.source_in_tc, fps.round() as i64);
    let src_out_frames = tc_to_frames(&info.source_out_tc, fps.round() as i64);
    let duration_frames = (rec_out_frames - rec_in_frames).max(0);

    out.push_str("          <clipitem>\n");
    out.push_str(&format!(
        "            <name>{}</name>\n",
        xml_escape(&info.clip_name)
    ));
    out.push_str(&format!(
        "            <duration>{duration_frames}</duration>\n"
    ));
    out.push_str(&format!("            <in>{src_in_frames}</in>\n"));
    out.push_str(&format!("            <out>{src_out_frames}</out>\n"));
    out.push_str(&format!("            <start>{rec_in_frames}</start>\n"));
    out.push_str(&format!("            <end>{rec_out_frames}</end>\n"));
    out.push_str(&format!("            <speed>{:.6}</speed>\n", info.speed));
    if info.reverse {
        out.push_str("            <reverse>TRUE</reverse>\n");
    }
    out.push_str(&format!(
        "            <opacity>{:.4}</opacity>\n",
        info.opacity
    ));
    if info.muted {
        out.push_str("            <enabled>FALSE</enabled>\n");
    }
    // Reel / file reference
    out.push_str("            <file>\n");
    out.push_str(&format!(
        "              <name>{}</name>\n",
        xml_escape(&info.reel_name)
    ));
    out.push_str("            </file>\n");
    out.push_str("          </clipitem>\n");

    let _ = (fps, tb_num, tb_den); // suppress unused warnings
}

/// Parse a SMPTE timecode string `HH:MM:SS:FF` (colon or semicolon) to frames.
#[must_use]
fn tc_to_frames(tc: &str, fps: i64) -> i64 {
    let parts: Vec<&str> = tc.split(&[':', ';'][..]).collect();
    if parts.len() != 4 {
        return 0;
    }
    let hh: i64 = parts[0].parse().unwrap_or(0);
    let mm: i64 = parts[1].parse().unwrap_or(0);
    let ss: i64 = parts[2].parse().unwrap_or(0);
    let ff: i64 = parts[3].parse().unwrap_or(0);
    hh * 3600 * fps + mm * 60 * fps + ss * fps + ff
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

    fn build_test_timeline() -> Timeline {
        let mut tl = Timeline::new(
            Rational::new(1, 1000), // 1ms timebase
            Rational::new(30, 1),   // 30 fps
        );

        // video track
        let vt = tl.add_track(TrackType::Video);
        let c1 = Clip::new(0, ClipType::Video, 0, 5000); // 0..5s
        let c2 = Clip::new(0, ClipType::Video, 5000, 3000); // 5..8s
        let _ = tl.add_clip(vt, c1);
        let _ = tl.add_clip(vt, c2);

        // audio track
        let at = tl.add_track(TrackType::Audio);
        let a1 = Clip::new(0, ClipType::Audio, 0, 8000); // 0..8s
        let _ = tl.add_clip(at, a1);

        tl
    }

    // ── frames_to_tc ─────────────────────────────────────────────────────

    #[test]
    fn test_frames_to_tc_zero() {
        assert_eq!(frames_to_tc(0, 30), "00:00:00:00");
    }

    #[test]
    fn test_frames_to_tc_one_hour() {
        // 1h = 3600 s * 30 fps = 108_000 frames
        assert_eq!(frames_to_tc(108_000, 30), "01:00:00:00");
    }

    #[test]
    fn test_frames_to_tc_compound() {
        // 01:02:03:04 → 1*3600*30 + 2*60*30 + 3*30 + 4 = 108_000 + 3_600 + 90 + 4 = 111_694
        let f = 111_694_i64;
        assert_eq!(frames_to_tc(f, 30), "01:02:03:04");
    }

    #[test]
    fn test_frames_to_tc_roundtrip() {
        let tc = "00:10:30:15";
        let f = tc_to_frames(tc, 30);
        assert_eq!(frames_to_tc(f, 30), tc);
    }

    // ── export_edl ────────────────────────────────────────────────────────

    #[test]
    fn test_export_edl_has_title() {
        let tl = build_test_timeline();
        let exporter = TimelineExporter::new(&tl).with_title("TestProject");
        let edl = exporter.export_edl();
        assert!(edl.contains("TITLE: TestProject"), "missing TITLE");
    }

    #[test]
    fn test_export_edl_has_fcm() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl).export_edl();
        assert!(edl.contains("FCM:"), "missing FCM line");
    }

    #[test]
    fn test_export_edl_event_count() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl).export_edl();
        // 2 video clips + 1 audio clip = 3 events
        let event_count = edl
            .lines()
            .filter(|l| l.starts_with("001") || l.starts_with("002") || l.starts_with("003"))
            .count();
        assert_eq!(event_count, 3, "expected 3 events");
    }

    #[test]
    fn test_export_edl_drop_frame_mode() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl)
            .with_drop_frame(true)
            .export_edl();
        assert!(edl.contains("FCM: DROP FRAME"));
    }

    #[test]
    fn test_export_edl_clip_name_comments() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl).export_edl();
        assert!(
            edl.contains("* FROM CLIP NAME:"),
            "missing clip name comment"
        );
    }

    #[test]
    fn test_export_edl_track_label_video() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl).export_edl();
        assert!(
            edl.contains(" V     ") || edl.contains(" V "),
            "missing video track label"
        );
    }

    #[test]
    fn test_export_edl_track_label_audio() {
        let tl = build_test_timeline();
        let edl = TimelineExporter::new(&tl).export_edl();
        assert!(
            edl.contains(" A     ") || edl.contains(" A "),
            "missing audio track label"
        );
    }

    // ── export_xml ────────────────────────────────────────────────────────

    #[test]
    fn test_export_xml_has_xmeml_root() {
        let tl = build_test_timeline();
        let xml = TimelineExporter::new(&tl).export_xml();
        assert!(xml.contains("<xmeml"), "missing <xmeml> root");
        assert!(xml.contains("</xmeml>"), "missing </xmeml>");
    }

    #[test]
    fn test_export_xml_has_sequence_name() {
        let tl = build_test_timeline();
        let xml = TimelineExporter::new(&tl).with_title("MySeq").export_xml();
        assert!(xml.contains("<name>MySeq</name>"), "missing sequence name");
    }

    #[test]
    fn test_export_xml_has_video_block() {
        let tl = build_test_timeline();
        let xml = TimelineExporter::new(&tl).export_xml();
        assert!(xml.contains("<video>"), "missing <video> block");
    }

    #[test]
    fn test_export_xml_has_audio_block() {
        let tl = build_test_timeline();
        let xml = TimelineExporter::new(&tl).export_xml();
        assert!(xml.contains("<audio>"), "missing <audio> block");
    }

    #[test]
    fn test_export_xml_clipitem_count() {
        let tl = build_test_timeline();
        let xml = TimelineExporter::new(&tl).export_xml();
        let count = xml.matches("<clipitem>").count();
        assert_eq!(count, 3, "expected 3 clipitems");
    }

    #[test]
    fn test_export_xml_escapes_special_chars() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let mut clip = Clip::new(0, ClipType::Video, 0, 1000);
        clip.metadata.name = Some("Clip & <test>".to_string());
        let _ = tl.add_clip(vt, clip);

        let xml = TimelineExporter::new(&tl).export_xml();
        assert!(xml.contains("&amp;"), "ampersand not escaped");
        assert!(xml.contains("&lt;"), "< not escaped");
    }

    // ── export_csv ────────────────────────────────────────────────────────

    #[test]
    fn test_export_csv_has_header() {
        let tl = build_test_timeline();
        let csv = TimelineExporter::new(&tl).export_csv();
        let first_line = csv.lines().next().unwrap_or("");
        assert!(
            first_line.starts_with("Event,Track,Reel,Name,"),
            "bad CSV header"
        );
    }

    #[test]
    fn test_export_csv_row_count() {
        let tl = build_test_timeline();
        let csv = TimelineExporter::new(&tl).export_csv();
        // 1 header + 3 clip rows
        let rows: Vec<&str> = csv.lines().collect();
        assert_eq!(rows.len(), 4, "expected 4 rows (header + 3 clips)");
    }

    #[test]
    fn test_export_csv_speed_column() {
        let tl = build_test_timeline();
        let csv = TimelineExporter::new(&tl).export_csv();
        // default speed is 1.0
        assert!(csv.contains("1.000000"), "speed column should contain 1.0");
    }

    #[test]
    fn test_export_csv_muted_flag() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let mut c = Clip::new(0, ClipType::Video, 0, 1000);
        c.muted = true;
        let _ = tl.add_clip(vt, c);

        let csv = TimelineExporter::new(&tl).export_csv();
        assert!(csv.contains(",true"), "muted=true should appear in CSV");
    }

    #[test]
    fn test_export_csv_escapes_comma_in_name() {
        let mut tl = Timeline::new(Rational::new(1, 1000), Rational::new(30, 1));
        let vt = tl.add_track(TrackType::Video);
        let mut c = Clip::new(0, ClipType::Video, 0, 1000);
        c.metadata.name = Some("hello, world".to_string());
        let _ = tl.add_clip(vt, c);

        let csv = TimelineExporter::new(&tl).export_csv();
        assert!(csv.contains("\"hello, world\""), "comma in name not quoted");
    }

    // ── collect_clips ordering ────────────────────────────────────────────

    #[test]
    fn test_collect_clips_event_numbers_sequential() {
        let tl = build_test_timeline();
        let exporter = TimelineExporter::new(&tl);
        let clips = exporter.collect_clips();
        for (i, c) in clips.iter().enumerate() {
            assert_eq!(c.event_number, (i + 1) as u32);
        }
    }

    // ── xml_escape helper ─────────────────────────────────────────────────

    #[test]
    fn test_xml_escape_all_chars() {
        let s = r#"<"'>&"#;
        let escaped = xml_escape(s);
        assert!(!escaped.contains('<'));
        assert!(!escaped.contains('>'));
        assert!(!escaped.contains('"'));
        assert!(!escaped.contains('\''));
        assert!(!escaped.contains('&') || escaped.contains("&amp;"));
    }
}
