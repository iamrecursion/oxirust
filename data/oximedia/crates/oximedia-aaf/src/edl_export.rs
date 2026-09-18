//! Export AAF compositions to CMX3600 EDL format
//!
//! Implements conversion of AAF `CompositionMob` to CMX3600 Edit Decision List
//! events, full EDL text generation and parsing, and timecode arithmetic.
//!
//! References:
//! - CMX 3600 EDL format specification
//! - SMPTE ST 12-1 (timecode)

use crate::composition::{CompositionMob, SequenceComponent};
use crate::{AafError, Result};

// ─── CMX3600 Event ────────────────────────────────────────────────────────────

/// A single event in a CMX3600 EDL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmx3600Event {
    /// Event (edit) number (1-based, displayed zero-padded to 3 digits).
    pub event_number: u32,
    /// Reel or tape name (up to 8 characters in strict CMX3600).
    pub reel_name: String,
    /// Track designator: `"V"`, `"A"`, `"A2"`, `"AA"`, etc.
    pub track_type: String,
    /// Edit type / transition: `"C"` (cut), `"D"` (dissolve), `"W"` (wipe), etc.
    pub transition: String,
    /// Source in timecode (HH:MM:SS:FF or HH:MM:SS;FF).
    pub src_in: String,
    /// Source out timecode.
    pub src_out: String,
    /// Record in timecode.
    pub rec_in: String,
    /// Record out timecode.
    pub rec_out: String,
    /// Optional comment line (prefixed with `*` in EDL output).
    pub comment: Option<String>,
}

/// Transition type constants for CMX3600 events.
pub mod transitions {
    /// Cut (straight cut).
    pub const CUT: &str = "C";
    /// Dissolve.
    pub const DISSOLVE: &str = "D";
    /// Wipe.
    pub const WIPE: &str = "W";
    /// Key.
    pub const KEY: &str = "K";
    /// Background key.
    pub const BG_KEY: &str = "B";
}

/// Track type designators for CMX3600 events.
pub mod track_types {
    /// Video track.
    pub const VIDEO: &str = "V";
    /// First audio track.
    pub const AUDIO: &str = "A";
    /// Audio track 2.
    pub const AUDIO2: &str = "A2";
    /// Combined audio/video.
    pub const BOTH: &str = "B";
    /// Audio tracks 1 and 2 combined.
    pub const AA: &str = "AA";
}

impl Cmx3600Event {
    /// Create a new CMX3600 event with all required fields.
    #[must_use]
    pub fn new(
        event_number: u32,
        reel_name: impl Into<String>,
        track_type: impl Into<String>,
        transition: impl Into<String>,
        src_in: impl Into<String>,
        src_out: impl Into<String>,
        rec_in: impl Into<String>,
        rec_out: impl Into<String>,
    ) -> Self {
        Self {
            event_number,
            reel_name: reel_name.into(),
            track_type: track_type.into(),
            transition: transition.into(),
            src_in: src_in.into(),
            src_out: src_out.into(),
            rec_in: rec_in.into(),
            rec_out: rec_out.into(),
            comment: None,
        }
    }

    /// Attach a comment to this event.
    #[must_use]
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Check whether this event is a cut.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.transition == transitions::CUT
    }

    /// Check whether this event is a dissolve.
    #[must_use]
    pub fn is_dissolve(&self) -> bool {
        self.transition == transitions::DISSOLVE
    }

    /// Check whether this event is on a video track.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.track_type == track_types::VIDEO
    }

    /// Check whether this event is on an audio track.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.track_type.starts_with('A')
    }

    /// Compute the source duration in frames (src_out - src_in) given fps.
    pub fn source_duration_frames(&self, fps: f32) -> Result<u64> {
        let src_in = parse_cmx3600_timecode(&self.src_in, fps)?;
        let src_out = parse_cmx3600_timecode(&self.src_out, fps)?;
        Ok(src_out.saturating_sub(src_in))
    }

    /// Compute the record duration in frames (rec_out - rec_in) given fps.
    pub fn record_duration_frames(&self, fps: f32) -> Result<u64> {
        let rec_in = parse_cmx3600_timecode(&self.rec_in, fps)?;
        let rec_out = parse_cmx3600_timecode(&self.rec_out, fps)?;
        Ok(rec_out.saturating_sub(rec_in))
    }

    /// Format this event as a single CMX3600 text line (without comment).
    #[must_use]
    pub fn to_line(&self) -> String {
        format!(
            "{:03}  {:<8}  {:<4}  {:<3}  {}  {}  {}  {}",
            self.event_number,
            self.reel_name,
            self.track_type,
            self.transition,
            self.src_in,
            self.src_out,
            self.rec_in,
            self.rec_out,
        )
    }
}

/// Validate a timecode string format (does not check numeric ranges).
///
/// Returns `true` if the string looks like `HH:MM:SS:FF` or `HH:MM:SS;FF`.
#[must_use]
pub fn is_valid_timecode_format(tc: &str) -> bool {
    if tc.len() != 11 {
        return false;
    }
    let bytes = tc.as_bytes();
    // Check separators: positions 2, 5, 8
    if bytes[2] != b':' || bytes[5] != b':' {
        return false;
    }
    if bytes[8] != b':' && bytes[8] != b';' {
        return false;
    }
    // Check digits
    for &pos in &[0, 1, 3, 4, 6, 7, 9, 10] {
        if !bytes[pos].is_ascii_digit() {
            return false;
        }
    }
    true
}

/// Convert a frame count to a duration string (HH:MM:SS.mmm).
#[must_use]
pub fn frames_to_duration_string(frames: u64, fps: f32) -> String {
    let fps_f64 = f64::from(fps);
    if fps_f64 <= 0.0 {
        return "00:00:00.000".to_string();
    }
    let total_secs = frames as f64 / fps_f64;
    let hours = (total_secs / 3600.0) as u64;
    let mins = ((total_secs % 3600.0) / 60.0) as u64;
    let secs = total_secs % 60.0;
    let whole_secs = secs as u64;
    let millis = ((secs - whole_secs as f64) * 1000.0) as u64;
    format!("{hours:02}:{mins:02}:{whole_secs:02}.{millis:03}")
}

// ─── Timecode helpers ─────────────────────────────────────────────────────────

/// Format a frame count as an SMPTE timecode string.
///
/// Uses `;` (semicolon) as the frame separator for drop-frame timecode,
/// `:` (colon) for non-drop-frame.
///
/// # Examples
///
/// ```
/// use oximedia_aaf::edl_export::format_timecode;
/// assert_eq!(format_timecode(0, 25.0, false), "00:00:00:00");
/// assert_eq!(format_timecode(25, 25.0, false), "00:00:01:00");
/// ```
#[must_use]
pub fn format_timecode(frame_count: u64, fps: f32, drop_frame: bool) -> String {
    let fps_u = fps.round() as u64;
    if fps_u == 0 {
        return "00:00:00:00".to_string();
    }

    let frames = frame_count % fps_u;
    let total_secs = frame_count / fps_u;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    let sep = if drop_frame { ';' } else { ':' };
    format!("{hours:02}:{mins:02}:{secs:02}{sep}{frames:02}")
}

/// Parse a CMX3600 timecode string into a frame count.
///
/// Accepts both `HH:MM:SS:FF` (non-drop-frame) and `HH:MM:SS;FF` (drop-frame)
/// with the `fps` implied by the exporter configuration.
///
/// # Errors
///
/// Returns `AafError::ParseError` if the string is not in a recognised format.
pub fn parse_cmx3600_timecode(tc: &str, fps: f32) -> Result<u64> {
    if tc.len() < 11 {
        return Err(AafError::ParseError(format!("Timecode too short: '{tc}'")));
    }

    // Accept HH:MM:SS:FF and HH:MM:SS;FF
    let parts: Vec<&str> = tc.splitn(4, [':', ';']).collect();
    if parts.len() != 4 {
        return Err(AafError::ParseError(format!(
            "Invalid timecode format: '{tc}'"
        )));
    }

    let parse_u64 = |s: &str| {
        s.parse::<u64>()
            .map_err(|_| AafError::ParseError(format!("Non-numeric timecode component: '{s}'")))
    };

    let hours = parse_u64(parts[0])?;
    let mins = parse_u64(parts[1])?;
    let secs = parse_u64(parts[2])?;
    let frames = parse_u64(parts[3])?;

    let fps_u = fps.round() as u64;
    let total = hours * 3600 * fps_u + mins * 60 * fps_u + secs * fps_u + frames;
    Ok(total)
}

// ─── CMX3600 Exporter ─────────────────────────────────────────────────────────

/// Exports an AAF `CompositionMob` to a list of CMX3600 events.
#[derive(Debug, Clone)]
pub struct Cmx3600Exporter {
    /// Frame rate used for timecode formatting.
    pub fps: f32,
    /// Whether to use drop-frame notation.
    pub drop_frame: bool,
}

impl Cmx3600Exporter {
    /// Create a new exporter with the given frame rate.
    #[must_use]
    pub fn new(fps: f32, drop_frame: bool) -> Self {
        Self { fps, drop_frame }
    }

    /// Create an exporter preset for PAL (25 fps, non-drop).
    #[must_use]
    pub fn pal() -> Self {
        Self::new(25.0, false)
    }

    /// Create an exporter preset for NTSC (29.97 fps, drop-frame).
    #[must_use]
    pub fn ntsc() -> Self {
        Self::new(29.97, true)
    }

    /// Format a frame count using this exporter's fps / drop-frame settings.
    #[must_use]
    pub fn fmt_tc(&self, frames: u64) -> String {
        format_timecode(frames, self.fps, self.drop_frame)
    }

    /// Convert a `CompositionMob` into a flat list of `Cmx3600Event`s.
    ///
    /// Each source clip in each track becomes one CMX3600 event.  Picture
    /// tracks are designated `"V"`, sound tracks `"A"` (or `"A2"`, `"A3"` …
    /// for additional audio tracks).  Transitions produce events with type
    /// `"D"` (dissolve); all other components produce cut events (`"C"`).
    #[must_use]
    pub fn export_composition_to_cmx3600(&self, comp: &CompositionMob) -> Vec<Cmx3600Event> {
        let mut events = Vec::new();
        let mut event_number = 1u32;
        let mut audio_track_counter = 0u32;

        for track in comp.tracks() {
            let track_type = if track.is_picture() {
                "V".to_string()
            } else if track.is_sound() {
                audio_track_counter += 1;
                if audio_track_counter == 1 {
                    "A".to_string()
                } else {
                    format!("A{audio_track_counter}")
                }
            } else {
                // Skip timecode / data tracks
                continue;
            };

            let seq = match &track.sequence {
                Some(s) => s,
                None => continue,
            };

            let mut rec_position: u64 = 0;

            for component in &seq.components {
                match component {
                    SequenceComponent::SourceClip(clip) => {
                        let src_in = clip.start_time.0.max(0) as u64;
                        let src_out = src_in + clip.length.max(0) as u64;
                        let rec_in = rec_position;
                        let rec_out = rec_position + clip.length.max(0) as u64;

                        // Derive a short reel name from the mob ID
                        let mob_str = clip.source_mob_id.to_string();
                        let reel = mob_str[..8.min(mob_str.len())].to_string();

                        let event = Cmx3600Event::new(
                            event_number,
                            reel,
                            &track_type,
                            "C",
                            self.fmt_tc(src_in),
                            self.fmt_tc(src_out),
                            self.fmt_tc(rec_in),
                            self.fmt_tc(rec_out),
                        );
                        events.push(event);
                        event_number += 1;
                        rec_position = rec_out;
                    }
                    SequenceComponent::Transition(trans) => {
                        let length = trans.length.max(0) as u64;
                        let src_in = trans.cut_point.0.max(0) as u64;
                        let src_out = src_in + length;
                        let rec_in = rec_position;
                        let rec_out = rec_position + length;

                        let event = Cmx3600Event::new(
                            event_number,
                            "BL",
                            &track_type,
                            "D",
                            self.fmt_tc(src_in),
                            self.fmt_tc(src_out),
                            self.fmt_tc(rec_in),
                            self.fmt_tc(rec_out),
                        );
                        events.push(event);
                        event_number += 1;
                        rec_position = rec_out;
                    }
                    SequenceComponent::Filler(filler) => {
                        // Advance record position over the filler gap
                        rec_position += filler.length.max(0) as u64;
                    }
                    SequenceComponent::Effect(_) => {
                        // Skip effects — they are encoded in separate passes
                    }
                }
            }
        }

        events
    }
}

impl Default for Cmx3600Exporter {
    fn default() -> Self {
        Self::pal()
    }
}

// ─── EDL text emitter ─────────────────────────────────────────────────────────

/// Emit a complete EDL text string from a slice of `Cmx3600Event`s.
///
/// The output starts with `TITLE:` and `FCM:` headers followed by one line
/// per event in standard CMX3600 column layout.
#[must_use]
pub fn emit_edl(events: &[Cmx3600Event], title: &str) -> String {
    let mut out = String::new();

    // Header lines
    out.push_str(&format!("TITLE: {title}\n"));
    out.push_str("FCM: NON-DROP FRAME\n");
    out.push('\n');

    for event in events {
        // Optional comment
        if let Some(ref comment) = event.comment {
            out.push_str(&format!("* {comment}\n"));
        }

        // Event line: NNN  REEL  TRACK  TRANS  SRC_IN  SRC_OUT  REC_IN  REC_OUT
        out.push_str(&format!(
            "{:03}  {:<8}  {:<4}  {:<3}  {}  {}  {}  {}\n",
            event.event_number,
            event.reel_name,
            event.track_type,
            event.transition,
            event.src_in,
            event.src_out,
            event.rec_in,
            event.rec_out,
        ));
    }

    out
}

// ─── CMX3600 Importer ─────────────────────────────────────────────────────────

/// Parses a CMX3600 EDL text string into a list of `Cmx3600Event`s.
pub struct Cmx3600Importer;

impl Cmx3600Importer {
    /// Create a new importer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse an EDL text string.
    ///
    /// Lines starting with `*` are treated as comments and attached to the
    /// immediately following event.  Lines starting with `TITLE:` or `FCM:`
    /// are skipped.
    ///
    /// # Errors
    ///
    /// Returns `AafError::ParseError` if a data line cannot be parsed.
    pub fn parse(&self, edl_text: &str) -> Result<Vec<Cmx3600Event>> {
        let mut events = Vec::new();
        let mut pending_comment: Option<String> = None;

        for (line_num, raw_line) in edl_text.lines().enumerate() {
            let line = raw_line.trim();

            if line.is_empty() || line.starts_with("TITLE:") || line.starts_with("FCM:") {
                continue;
            }

            if let Some(comment_text) = line.strip_prefix("* ").or_else(|| line.strip_prefix('*')) {
                pending_comment = Some(comment_text.to_string());
                continue;
            }

            // Expect an event data line
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 8 {
                return Err(AafError::ParseError(format!(
                    "EDL line {}: expected ≥ 8 fields, found {}: '{line}'",
                    line_num + 1,
                    cols.len()
                )));
            }

            let event_number = cols[0].parse::<u32>().map_err(|_| {
                AafError::ParseError(format!(
                    "EDL line {}: invalid event number '{}'",
                    line_num + 1,
                    cols[0]
                ))
            })?;

            let mut event = Cmx3600Event::new(
                event_number,
                cols[1],
                cols[2],
                cols[3],
                cols[4],
                cols[5],
                cols[6],
                cols[7],
            );

            if let Some(comment) = pending_comment.take() {
                event.comment = Some(comment);
            }

            events.push(event);
        }

        Ok(events)
    }
}

impl Default for Cmx3600Importer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
        Transition,
    };
    use crate::dictionary::Auid;
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    fn make_comp_with_clip(fps: f32) -> (CompositionMob, Cmx3600Exporter) {
        let rate = EditRate::new(fps.round() as i32, 1);
        let mob_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(mob_id, "TestComp");

        let mut video_track = Track::new(1, "V1", rate, TrackType::Picture);
        let mut seq = Sequence::new(Auid::PICTURE);
        let src_id =
            Uuid::parse_str("12345678-0000-0000-0000-000000000001").expect("valid UUID literal");
        let clip = SourceClip::new(50, Position::zero(), src_id, 1);
        seq.add_component(SequenceComponent::SourceClip(clip));
        video_track.set_sequence(seq);
        comp.add_track(video_track);

        let exporter = Cmx3600Exporter::new(fps, false);
        (comp, exporter)
    }

    // ── format_timecode ─────────────────────────────────────────────────────

    #[test]
    fn test_format_timecode_zero() {
        assert_eq!(format_timecode(0, 25.0, false), "00:00:00:00");
    }

    #[test]
    fn test_format_timecode_one_second_25fps() {
        assert_eq!(format_timecode(25, 25.0, false), "00:00:01:00");
    }

    #[test]
    fn test_format_timecode_one_minute_25fps() {
        assert_eq!(format_timecode(25 * 60, 25.0, false), "00:01:00:00");
    }

    #[test]
    fn test_format_timecode_one_hour_25fps() {
        assert_eq!(format_timecode(25 * 3600, 25.0, false), "01:00:00:00");
    }

    #[test]
    fn test_format_timecode_drop_frame_separator() {
        let tc = format_timecode(100, 30.0, true);
        assert!(tc.contains(';'), "Drop-frame must use ';', got: {tc}");
    }

    #[test]
    fn test_format_timecode_non_drop_separator() {
        let tc = format_timecode(100, 25.0, false);
        assert!(
            tc.contains(':') && !tc.contains(';'),
            "Non-drop must use ':', got: {tc}"
        );
    }

    #[test]
    fn test_format_timecode_frame_remainder() {
        // 26 frames at 25fps = 1 second + 1 frame
        assert_eq!(format_timecode(26, 25.0, false), "00:00:01:01");
    }

    // ── parse_cmx3600_timecode ───────────────────────────────────────────────

    #[test]
    fn test_parse_tc_zero() {
        assert_eq!(
            parse_cmx3600_timecode("00:00:00:00", 25.0).expect("parse zero TC"),
            0
        );
    }

    #[test]
    fn test_parse_tc_one_second() {
        assert_eq!(
            parse_cmx3600_timecode("00:00:01:00", 25.0).expect("parse one-second TC"),
            25
        );
    }

    #[test]
    fn test_parse_tc_drop_frame() {
        assert_eq!(
            parse_cmx3600_timecode("00:00:01;00", 30.0).expect("parse drop-frame TC"),
            30
        );
    }

    #[test]
    fn test_parse_tc_invalid_format() {
        assert!(parse_cmx3600_timecode("abc", 25.0).is_err());
    }

    #[test]
    fn test_parse_tc_roundtrip() {
        let frames = 25 * 3600 + 25 * 60 + 25 * 13 + 7; // 01:01:13:07
        let tc = format_timecode(frames, 25.0, false);
        let back = parse_cmx3600_timecode(&tc, 25.0).expect("roundtrip parse TC");
        assert_eq!(back, frames);
    }

    // ── Exporter ─────────────────────────────────────────────────────────────

    #[test]
    fn test_export_single_clip() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events.len(), 1, "Should produce exactly one event");
    }

    #[test]
    fn test_export_event_number_starts_at_1() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events[0].event_number, 1);
    }

    #[test]
    fn test_export_video_track_type() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events[0].track_type, "V");
    }

    #[test]
    fn test_export_cut_transition() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events[0].transition, "C");
    }

    #[test]
    fn test_export_src_in_timecode() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events[0].src_in, "00:00:00:00");
    }

    #[test]
    fn test_export_rec_in_advances() {
        let rate = EditRate::new(25, 1);
        let mob_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(mob_id, "Multi");
        let mut track = Track::new(1, "V1", rate, TrackType::Picture);
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            25,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            25,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        track.set_sequence(seq);
        comp.add_track(track);

        let exp = Cmx3600Exporter::pal();
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].rec_in, "00:00:00:00");
        assert_eq!(events[1].rec_in, "00:00:01:00", "second clip starts at 1s");
    }

    #[test]
    fn test_export_audio_track_designation() {
        let rate = EditRate::new(25, 1);
        let mob_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(mob_id, "Audio");
        let mut atrack = Track::new(2, "A1", rate, TrackType::Sound);
        let mut seq = Sequence::new(Auid::SOUND);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            25,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        atrack.set_sequence(seq);
        comp.add_track(atrack);

        let exp = Cmx3600Exporter::pal();
        let events = exp.export_composition_to_cmx3600(&comp);
        assert_eq!(events[0].track_type, "A");
    }

    #[test]
    fn test_export_filler_advances_record() {
        let rate = EditRate::new(25, 1);
        let mob_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(mob_id, "Filler");
        let mut track = Track::new(1, "V1", rate, TrackType::Picture);
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::Filler(Filler::new(50)));
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            25,
            Position::zero(),
            Uuid::new_v4(),
            1,
        )));
        track.set_sequence(seq);
        comp.add_track(track);

        let exp = Cmx3600Exporter::pal();
        let events = exp.export_composition_to_cmx3600(&comp);
        // The filler = 50 frames = 2 seconds at 25fps
        assert_eq!(events[0].rec_in, "00:00:02:00");
    }

    #[test]
    fn test_export_dissolve_transition() {
        let rate = EditRate::new(25, 1);
        let mob_id = Uuid::new_v4();
        let mut comp = CompositionMob::new(mob_id, "Trans");
        let mut track = Track::new(1, "V1", rate, TrackType::Picture);
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::Transition(Transition::new(
            12,
            Position::new(6),
        )));
        track.set_sequence(seq);
        comp.add_track(track);

        let exp = Cmx3600Exporter::pal();
        let events = exp.export_composition_to_cmx3600(&comp);
        assert!(!events.is_empty());
        assert_eq!(events[0].transition, "D");
    }

    // ── emit_edl ──────────────────────────────────────────────────────────────

    #[test]
    fn test_emit_edl_title_header() {
        let events = vec![];
        let edl = emit_edl(&events, "MY SHOW");
        assert!(edl.starts_with("TITLE: MY SHOW"), "edl={edl}");
    }

    #[test]
    fn test_emit_edl_fcm_header() {
        let edl = emit_edl(&[], "X");
        assert!(edl.contains("FCM: NON-DROP FRAME"));
    }

    #[test]
    fn test_emit_edl_event_line_format() {
        let event = Cmx3600Event::new(
            1,
            "TAPE001",
            "V",
            "C",
            "00:00:00:00",
            "00:00:02:00",
            "00:00:00:00",
            "00:00:02:00",
        );
        let edl = emit_edl(&[event], "T");
        assert!(edl.contains("001"), "event number formatted");
        assert!(edl.contains("TAPE001"), "reel name present");
        assert!(edl.contains("00:00:00:00"));
    }

    #[test]
    fn test_emit_edl_comment_line() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        )
        .with_comment("From Scene 1");
        let edl = emit_edl(&[event], "T");
        assert!(edl.contains("* From Scene 1"), "edl={edl}");
    }

    // ── Importer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_importer_round_trip() {
        let (comp, exp) = make_comp_with_clip(25.0);
        let events = exp.export_composition_to_cmx3600(&comp);
        let edl = emit_edl(&events, "ROUND TRIP");

        let importer = Cmx3600Importer::new();
        let parsed = importer.parse(&edl).expect("round-trip import");
        assert_eq!(parsed.len(), events.len());
        assert_eq!(parsed[0].event_number, events[0].event_number);
        assert_eq!(parsed[0].src_in, events[0].src_in);
    }

    #[test]
    fn test_importer_skips_header_lines() {
        let edl = "TITLE: MyShow\nFCM: NON-DROP FRAME\n\n001  REEL  V    C    00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let importer = Cmx3600Importer::new();
        let events = importer.parse(edl).expect("parse");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].reel_name, "REEL");
    }

    #[test]
    fn test_importer_comment_attached_to_next_event() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\n* My Comment\n001  R  V  C  00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let importer = Cmx3600Importer::new();
        let events = importer.parse(edl).expect("parse with comment");
        assert_eq!(events[0].comment.as_deref(), Some("My Comment"));
    }

    #[test]
    fn test_importer_error_on_bad_event_number() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\nXXX  R  V  C  00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let importer = Cmx3600Importer::new();
        assert!(importer.parse(edl).is_err());
    }

    #[test]
    fn test_importer_error_on_too_few_columns() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\n001  R  V\n";
        let importer = Cmx3600Importer::new();
        assert!(importer.parse(edl).is_err());
    }

    // ── Event methods ─────────────────────────────────────────────────────

    #[test]
    fn test_event_is_cut() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        );
        assert!(event.is_cut());
        assert!(!event.is_dissolve());
    }

    #[test]
    fn test_event_is_dissolve() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "D",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        );
        assert!(event.is_dissolve());
        assert!(!event.is_cut());
    }

    #[test]
    fn test_event_is_video() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        );
        assert!(event.is_video());
        assert!(!event.is_audio());
    }

    #[test]
    fn test_event_is_audio() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "A",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        );
        assert!(event.is_audio());
        assert!(!event.is_video());
    }

    #[test]
    fn test_event_is_audio_a2() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "A2",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        );
        assert!(event.is_audio());
    }

    #[test]
    fn test_event_source_duration_frames() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "C",
            "00:00:00:00",
            "00:00:02:00",
            "00:00:00:00",
            "00:00:02:00",
        );
        let dur = event.source_duration_frames(25.0).expect("duration");
        assert_eq!(dur, 50);
    }

    #[test]
    fn test_event_record_duration_frames() {
        let event = Cmx3600Event::new(
            1,
            "R",
            "V",
            "C",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:03:00",
        );
        let dur = event.record_duration_frames(25.0).expect("duration");
        assert_eq!(dur, 75);
    }

    #[test]
    fn test_event_to_line() {
        let event = Cmx3600Event::new(
            1,
            "TAPE001",
            "V",
            "C",
            "00:00:00:00",
            "00:00:02:00",
            "00:00:00:00",
            "00:00:02:00",
        );
        let line = event.to_line();
        assert!(line.starts_with("001"));
        assert!(line.contains("TAPE001"));
    }

    // ── Timecode validation ───────────────────────────────────────────────

    #[test]
    fn test_valid_timecode_format() {
        assert!(is_valid_timecode_format("00:00:00:00"));
        assert!(is_valid_timecode_format("23:59:59:29"));
        assert!(is_valid_timecode_format("01:02:03;04"));
    }

    #[test]
    fn test_invalid_timecode_format() {
        assert!(!is_valid_timecode_format(""));
        assert!(!is_valid_timecode_format("00:00:00"));
        assert!(!is_valid_timecode_format("abcdefghijk"));
        assert!(!is_valid_timecode_format("00-00-00-00"));
    }

    // ── Duration string ───────────────────────────────────────────────────

    #[test]
    fn test_frames_to_duration_string_zero() {
        assert_eq!(frames_to_duration_string(0, 25.0), "00:00:00.000");
    }

    #[test]
    fn test_frames_to_duration_string_one_second() {
        let dur = frames_to_duration_string(25, 25.0);
        assert_eq!(dur, "00:00:01.000");
    }

    #[test]
    fn test_frames_to_duration_string_one_hour() {
        let dur = frames_to_duration_string(25 * 3600, 25.0);
        assert_eq!(dur, "01:00:00.000");
    }

    #[test]
    fn test_frames_to_duration_string_zero_fps() {
        assert_eq!(frames_to_duration_string(100, 0.0), "00:00:00.000");
    }

    // ── Transition/track constants ────────────────────────────────────────

    #[test]
    fn test_transition_constants() {
        assert_eq!(transitions::CUT, "C");
        assert_eq!(transitions::DISSOLVE, "D");
        assert_eq!(transitions::WIPE, "W");
    }

    #[test]
    fn test_track_type_constants() {
        assert_eq!(track_types::VIDEO, "V");
        assert_eq!(track_types::AUDIO, "A");
        assert_eq!(track_types::AUDIO2, "A2");
    }

    // ── Multi-event EDL roundtrip ─────────────────────────────────────────

    #[test]
    fn test_multi_event_edl_roundtrip() {
        let events = vec![
            Cmx3600Event::new(
                1,
                "REEL1",
                "V",
                "C",
                "00:00:00:00",
                "00:00:02:00",
                "00:00:00:00",
                "00:00:02:00",
            )
            .with_comment("First clip"),
            Cmx3600Event::new(
                2,
                "REEL2",
                "V",
                "D",
                "00:00:01:00",
                "00:00:03:00",
                "00:00:02:00",
                "00:00:04:00",
            ),
            Cmx3600Event::new(
                3,
                "REEL1",
                "A",
                "C",
                "00:00:00:00",
                "00:00:04:00",
                "00:00:00:00",
                "00:00:04:00",
            )
            .with_comment("Audio track"),
        ];

        let edl = emit_edl(&events, "MULTI TEST");
        let importer = Cmx3600Importer::new();
        let parsed = importer.parse(&edl).expect("parse multi");

        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].event_number, 1);
        assert_eq!(parsed[0].reel_name, "REEL1");
        assert_eq!(parsed[0].comment.as_deref(), Some("First clip"));
        assert_eq!(parsed[1].event_number, 2);
        assert_eq!(parsed[1].transition, "D");
        assert_eq!(parsed[2].track_type, "A");
        assert_eq!(parsed[2].comment.as_deref(), Some("Audio track"));
    }

    #[test]
    fn test_parse_tc_complex_value() {
        // 1 hour + 30 minutes + 15 seconds + 12 frames at 25fps
        let expected = 25 * 3600 + 25 * 30 * 60 + 25 * 15 + 12;
        let result = parse_cmx3600_timecode("01:30:15:12", 25.0).expect("parse");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_format_timecode_zero_fps() {
        assert_eq!(format_timecode(100, 0.0, false), "00:00:00:00");
    }
}
