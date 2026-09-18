//! DaVinci Resolve EDL dialect support
//!
//! DaVinci Resolve uses a CMX3600-compatible EDL format with several
//! Resolve-specific extensions:
//!
//! - `* FROM CLIP NAME:` comments carrying clip names
//! - `* TO CLIP NAME:` comments for the outgoing clip in dissolves
//! - `* SOURCE FILE:` lines with file paths
//! - `* ASC_SOP` and `* ASC_SAT` colour grading parameters
//! - Reel names derived from clip names rather than tape names
//! - `* COMMENT:` free-form event comments
//!
//! This module can both **export** AAF compositions to Resolve EDL and
//! **parse** Resolve EDL back to a list of `ResolveEvent`s.

use crate::composition::{CompositionMob, SequenceComponent};
use crate::edl_export::format_timecode;
use crate::{AafError, Result};

// ─── Resolve event ────────────────────────────────────────────────────────────

/// A single edit event in a DaVinci Resolve EDL.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveEvent {
    /// CMX event number (1-based).
    pub event_number: u32,
    /// Reel / tape name.
    pub reel_name: String,
    /// Track designator (`"V"`, `"A"`, `"A2"`, …).
    pub track_type: String,
    /// Transition type (`"C"` cut, `"D"` dissolve, …).
    pub transition: String,
    /// Source in timecode.
    pub src_in: String,
    /// Source out timecode.
    pub src_out: String,
    /// Record in timecode.
    pub rec_in: String,
    /// Record out timecode.
    pub rec_out: String,
    /// `FROM CLIP NAME` annotation.
    pub from_clip_name: Option<String>,
    /// `TO CLIP NAME` annotation (for dissolves).
    pub to_clip_name: Option<String>,
    /// `SOURCE FILE` path annotation.
    pub source_file: Option<String>,
    /// `ASC_SOP` colour grading (slope, offset, power per RGB channel).
    pub asc_sop: Option<AscSop>,
    /// `ASC_SAT` colour saturation scalar.
    pub asc_sat: Option<f64>,
    /// Free-form `COMMENT` text.
    pub comment: Option<String>,
}

/// ASC CDL SOP (Slope/Offset/Power) parameters for a single event.
#[derive(Debug, Clone, PartialEq)]
pub struct AscSop {
    /// Slope for R, G, B channels
    pub slope: [f64; 3],
    /// Offset for R, G, B channels
    pub offset: [f64; 3],
    /// Power (gamma) for R, G, B channels
    pub power: [f64; 3],
}

impl AscSop {
    /// Create an identity (no-op) SOP.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            slope: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            power: [1.0, 1.0, 1.0],
        }
    }

    /// Format as the `* ASC_SOP` EDL annotation string.
    #[must_use]
    pub fn to_annotation(&self) -> String {
        format!(
            "* ASC_SOP ({:.6} {:.6} {:.6})({:.6} {:.6} {:.6})({:.6} {:.6} {:.6})",
            self.slope[0],
            self.slope[1],
            self.slope[2],
            self.offset[0],
            self.offset[1],
            self.offset[2],
            self.power[0],
            self.power[1],
            self.power[2],
        )
    }
}

impl ResolveEvent {
    /// Create a basic cut event.
    #[must_use]
    pub fn new_cut(
        event_number: u32,
        reel_name: impl Into<String>,
        track_type: impl Into<String>,
        src_in: impl Into<String>,
        src_out: impl Into<String>,
        rec_in: impl Into<String>,
        rec_out: impl Into<String>,
    ) -> Self {
        Self {
            event_number,
            reel_name: reel_name.into(),
            track_type: track_type.into(),
            transition: "C".to_string(),
            src_in: src_in.into(),
            src_out: src_out.into(),
            rec_in: rec_in.into(),
            rec_out: rec_out.into(),
            from_clip_name: None,
            to_clip_name: None,
            source_file: None,
            asc_sop: None,
            asc_sat: None,
            comment: None,
        }
    }

    /// Set the FROM CLIP NAME annotation.
    #[must_use]
    pub fn with_from_clip(mut self, name: impl Into<String>) -> Self {
        self.from_clip_name = Some(name.into());
        self
    }

    /// Set the SOURCE FILE annotation.
    #[must_use]
    pub fn with_source_file(mut self, path: impl Into<String>) -> Self {
        self.source_file = Some(path.into());
        self
    }

    /// Set ASC_SOP colour grading.
    #[must_use]
    pub fn with_asc_sop(mut self, sop: AscSop) -> Self {
        self.asc_sop = Some(sop);
        self
    }

    /// Set ASC_SAT.
    #[must_use]
    pub fn with_asc_sat(mut self, sat: f64) -> Self {
        self.asc_sat = Some(sat);
        self
    }

    /// Format the event as CMX3600 lines with Resolve annotations.
    #[must_use]
    pub fn to_lines(&self) -> String {
        let mut out = String::new();

        // Optional annotations before the event line
        if let Some(ref name) = self.from_clip_name {
            out.push_str(&format!("* FROM CLIP NAME: {name}\n"));
        }
        if let Some(ref name) = self.to_clip_name {
            out.push_str(&format!("* TO CLIP NAME: {name}\n"));
        }
        if let Some(ref path) = self.source_file {
            out.push_str(&format!("* SOURCE FILE: {path}\n"));
        }
        if let Some(ref sop) = self.asc_sop {
            out.push_str(&format!("{}\n", sop.to_annotation()));
        }
        if let Some(sat) = self.asc_sat {
            out.push_str(&format!("* ASC_SAT {sat:.6}\n"));
        }
        if let Some(ref comment) = self.comment {
            out.push_str(&format!("* COMMENT: {comment}\n"));
        }

        // Event line
        out.push_str(&format!(
            "{:03}  {:<8}  {:<4}  {:<3}  {}  {}  {}  {}\n",
            self.event_number,
            self.reel_name,
            self.track_type,
            self.transition,
            self.src_in,
            self.src_out,
            self.rec_in,
            self.rec_out,
        ));

        out
    }
}

// ─── Exporter ─────────────────────────────────────────────────────────────────

/// Exporter that generates a DaVinci Resolve-style EDL from a `CompositionMob`.
#[derive(Debug, Clone)]
pub struct ResolveEdlExporter {
    /// Frame rate for timecode formatting.
    pub fps: f32,
    /// Whether to use drop-frame notation.
    pub drop_frame: bool,
    /// Whether to embed `FROM CLIP NAME` annotations.
    pub include_clip_names: bool,
}

impl ResolveEdlExporter {
    /// Create a new exporter at the given frame rate.
    #[must_use]
    pub fn new(fps: f32, drop_frame: bool) -> Self {
        Self {
            fps,
            drop_frame,
            include_clip_names: true,
        }
    }

    /// PAL preset (25 fps, non-drop).
    #[must_use]
    pub fn pal() -> Self {
        Self::new(25.0, false)
    }

    /// NTSC preset (29.97 fps, drop-frame).
    #[must_use]
    pub fn ntsc() -> Self {
        Self::new(29.97, true)
    }

    /// Format a frame count as timecode.
    #[must_use]
    fn fmt_tc(&self, frames: u64) -> String {
        format_timecode(frames, self.fps, self.drop_frame)
    }

    /// Export a `CompositionMob` to a list of `ResolveEvent`s.
    #[must_use]
    pub fn export(&self, comp: &CompositionMob) -> Vec<ResolveEvent> {
        let mut events = Vec::new();
        let mut event_number = 1u32;
        let mut audio_counter = 0u32;

        for track in comp.tracks() {
            let track_type = if track.is_picture() {
                "V".to_string()
            } else if track.is_sound() {
                audio_counter += 1;
                if audio_counter == 1 {
                    "A".to_string()
                } else {
                    format!("A{audio_counter}")
                }
            } else {
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

                        let mob_str = clip.source_mob_id.to_string();
                        // Resolve convention: use full mob ID as reel name
                        let reel = mob_str[..8.min(mob_str.len())].to_string();

                        let mut ev = ResolveEvent::new_cut(
                            event_number,
                            &reel,
                            &track_type,
                            self.fmt_tc(src_in),
                            self.fmt_tc(src_out),
                            self.fmt_tc(rec_in),
                            self.fmt_tc(rec_out),
                        );

                        if self.include_clip_names {
                            ev.from_clip_name = Some(format!("mob_{}", &mob_str[..8]));
                        }

                        events.push(ev);
                        event_number += 1;
                        rec_position = rec_out;
                    }
                    SequenceComponent::Filler(filler) => {
                        rec_position += filler.length.max(0) as u64;
                    }
                    SequenceComponent::Transition(trans) => {
                        let length = trans.length.max(0) as u64;
                        let src_in = trans.cut_point.0.max(0) as u64;
                        let src_out = src_in + length;
                        let rec_in = rec_position;
                        let rec_out = rec_position + length;

                        events.push(ResolveEvent {
                            event_number,
                            reel_name: "BL".to_string(),
                            track_type: track_type.clone(),
                            transition: "D".to_string(),
                            src_in: self.fmt_tc(src_in),
                            src_out: self.fmt_tc(src_out),
                            rec_in: self.fmt_tc(rec_in),
                            rec_out: self.fmt_tc(rec_out),
                            from_clip_name: None,
                            to_clip_name: None,
                            source_file: None,
                            asc_sop: None,
                            asc_sat: None,
                            comment: None,
                        });
                        event_number += 1;
                        rec_position = rec_out;
                    }
                    SequenceComponent::Effect(_) => {}
                }
            }
        }

        events
    }

    /// Emit the events as a complete Resolve EDL text.
    #[must_use]
    pub fn emit_edl(&self, events: &[ResolveEvent], title: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("TITLE: {title}\n"));
        let fcm = if self.drop_frame {
            "DROP FRAME"
        } else {
            "NON-DROP FRAME"
        };
        out.push_str(&format!("FCM: {fcm}\n\n"));
        for event in events {
            out.push_str(&event.to_lines());
        }
        out
    }
}

impl Default for ResolveEdlExporter {
    fn default() -> Self {
        Self::pal()
    }
}

// ─── Importer ─────────────────────────────────────────────────────────────────

/// Parser for DaVinci Resolve EDL files.
pub struct ResolveEdlImporter;

impl ResolveEdlImporter {
    /// Create a new importer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Parse a Resolve EDL text into `ResolveEvent`s.
    ///
    /// Handles standard CMX3600 event lines plus all Resolve annotation comments.
    ///
    /// # Errors
    ///
    /// Returns `AafError::ParseError` if a data line cannot be parsed.
    pub fn parse(&self, edl_text: &str) -> Result<Vec<ResolveEvent>> {
        let mut events: Vec<ResolveEvent> = Vec::new();

        // Pending annotation state
        let mut pending_from_clip: Option<String> = None;
        let mut pending_to_clip: Option<String> = None;
        let mut pending_source_file: Option<String> = None;
        let mut pending_asc_sop: Option<AscSop> = None;
        let mut pending_asc_sat: Option<f64> = None;
        let mut pending_comment: Option<String> = None;

        for (line_num, raw) in edl_text.lines().enumerate() {
            let line = raw.trim();

            if line.is_empty() || line.starts_with("TITLE:") || line.starts_with("FCM:") {
                continue;
            }

            if let Some(rest) = line.strip_prefix("* FROM CLIP NAME:") {
                pending_from_clip = Some(rest.trim().to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("* TO CLIP NAME:") {
                pending_to_clip = Some(rest.trim().to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("* SOURCE FILE:") {
                pending_source_file = Some(rest.trim().to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("* ASC_SOP") {
                pending_asc_sop = parse_asc_sop(rest.trim());
                continue;
            }
            if let Some(rest) = line.strip_prefix("* ASC_SAT") {
                if let Ok(sat) = rest.trim().parse::<f64>() {
                    pending_asc_sat = Some(sat);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("* COMMENT:") {
                pending_comment = Some(rest.trim().to_string());
                continue;
            }
            // Generic comment line
            if line.starts_with('*') {
                // Keep as pending comment if not already set
                if pending_comment.is_none() {
                    pending_comment = Some(
                        line.strip_prefix("* ")
                            .or_else(|| line.strip_prefix('*'))
                            .unwrap_or(line)
                            .to_string(),
                    );
                }
                continue;
            }

            // Data line
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 8 {
                return Err(AafError::ParseError(format!(
                    "Resolve EDL line {}: expected ≥ 8 fields, found {}: '{line}'",
                    line_num + 1,
                    cols.len()
                )));
            }

            let event_number = cols[0].parse::<u32>().map_err(|_| {
                AafError::ParseError(format!(
                    "Resolve EDL line {}: invalid event number '{}'",
                    line_num + 1,
                    cols[0]
                ))
            })?;

            let ev = ResolveEvent {
                event_number,
                reel_name: cols[1].to_string(),
                track_type: cols[2].to_string(),
                transition: cols[3].to_string(),
                src_in: cols[4].to_string(),
                src_out: cols[5].to_string(),
                rec_in: cols[6].to_string(),
                rec_out: cols[7].to_string(),
                from_clip_name: pending_from_clip.take(),
                to_clip_name: pending_to_clip.take(),
                source_file: pending_source_file.take(),
                asc_sop: pending_asc_sop.take(),
                asc_sat: pending_asc_sat.take(),
                comment: pending_comment.take(),
            };

            events.push(ev);
        }

        Ok(events)
    }
}

impl Default for ResolveEdlImporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse an `ASC_SOP` annotation string of the form
/// `(s1 s2 s3)(o1 o2 o3)(p1 p2 p3)`
fn parse_asc_sop(s: &str) -> Option<AscSop> {
    // Extract three parenthesised groups
    let groups: Vec<&str> = s
        .split('(')
        .skip(1)
        .filter_map(|g| g.split(')').next())
        .collect();
    if groups.len() < 3 {
        return None;
    }

    let parse_triple = |group: &str| -> Option<[f64; 3]> {
        let parts: Vec<f64> = group
            .split_whitespace()
            .filter_map(|v| v.parse::<f64>().ok())
            .collect();
        if parts.len() == 3 {
            Some([parts[0], parts[1], parts[2]])
        } else {
            None
        }
    };

    let slope = parse_triple(groups[0])?;
    let offset = parse_triple(groups[1])?;
    let power = parse_triple(groups[2])?;

    Some(AscSop {
        slope,
        offset,
        power,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    fn make_comp() -> CompositionMob {
        let src = Uuid::parse_str("12345678-0000-0000-0000-000000000001").expect("valid uuid");
        let mut comp = CompositionMob::new(Uuid::new_v4(), "TestComp");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::zero(),
            src,
            1,
        )));
        let mut track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        comp
    }

    #[test]
    fn test_export_single_clip() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::pal();
        let events = exp.export(&comp);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].transition, "C");
        assert_eq!(events[0].track_type, "V");
    }

    #[test]
    fn test_export_includes_from_clip_name() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::pal();
        let events = exp.export(&comp);
        assert!(events[0].from_clip_name.is_some());
    }

    #[test]
    fn test_emit_edl_title() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::pal();
        let events = exp.export(&comp);
        let edl = exp.emit_edl(&events, "MY SHOW");
        assert!(edl.starts_with("TITLE: MY SHOW"));
        assert!(edl.contains("FCM: NON-DROP FRAME"));
    }

    #[test]
    fn test_emit_edl_from_clip_name_annotation() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::pal();
        let events = exp.export(&comp);
        let edl = exp.emit_edl(&events, "T");
        assert!(edl.contains("* FROM CLIP NAME:"));
    }

    #[test]
    fn test_roundtrip_export_import() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::pal();
        let events = exp.export(&comp);
        let edl = exp.emit_edl(&events, "ROUNDTRIP");

        let imp = ResolveEdlImporter::new();
        let parsed = imp.parse(&edl).expect("parse should succeed");
        assert_eq!(parsed.len(), events.len());
        assert_eq!(parsed[0].event_number, events[0].event_number);
        assert_eq!(parsed[0].src_in, events[0].src_in);
        assert_eq!(parsed[0].from_clip_name, events[0].from_clip_name);
    }

    #[test]
    fn test_parse_from_clip_name() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\n* FROM CLIP NAME: MyClip\n001  REEL1  V  C  00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let imp = ResolveEdlImporter::new();
        let parsed = imp.parse(edl).expect("parse");
        assert_eq!(parsed[0].from_clip_name.as_deref(), Some("MyClip"));
    }

    #[test]
    fn test_parse_source_file() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\n* SOURCE FILE: /path/to/clip.mov\n001  R  V  C  00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let imp = ResolveEdlImporter::new();
        let parsed = imp.parse(edl).expect("parse");
        assert_eq!(parsed[0].source_file.as_deref(), Some("/path/to/clip.mov"));
    }

    #[test]
    fn test_parse_asc_sop_annotation() {
        let sop_line =
            "(1.000000 1.000000 1.000000)(0.000000 0.000000 0.000000)(1.000000 1.000000 1.000000)";
        let sop = parse_asc_sop(sop_line).expect("parse ASC_SOP");
        assert_eq!(sop.slope, [1.0, 1.0, 1.0]);
        assert_eq!(sop.offset, [0.0, 0.0, 0.0]);
        assert_eq!(sop.power, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_parse_asc_sat() {
        let edl = "TITLE: X\nFCM: NON-DROP FRAME\n\n* ASC_SAT 0.850000\n001  R  V  C  00:00:00:00  00:00:01:00  00:00:00:00  00:00:01:00\n";
        let imp = ResolveEdlImporter::new();
        let parsed = imp.parse(edl).expect("parse");
        let sat = parsed[0].asc_sat.expect("asc_sat");
        assert!((sat - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_asc_sop_identity_annotation() {
        let sop = AscSop::identity();
        let ann = sop.to_annotation();
        assert!(ann.starts_with("* ASC_SOP"));
        assert!(ann.contains("1.000000"));
    }

    #[test]
    fn test_asc_sop_roundtrip_in_event() {
        let sop = AscSop {
            slope: [1.1, 1.0, 0.9],
            offset: [0.05, 0.0, -0.05],
            power: [1.0, 1.0, 1.0],
        };
        let event = ResolveEvent::new_cut(
            1,
            "REEL",
            "V",
            "00:00:00:00",
            "00:00:01:00",
            "00:00:00:00",
            "00:00:01:00",
        )
        .with_asc_sop(sop.clone());

        let lines = event.to_lines();
        assert!(lines.contains("* ASC_SOP"));

        // Parse back
        let edl = format!("TITLE: T\nFCM: NON-DROP FRAME\n\n{lines}");
        let imp = ResolveEdlImporter::new();
        let parsed = imp.parse(&edl).expect("parse");
        let recovered = parsed[0].asc_sop.as_ref().expect("asc_sop recovered");
        for i in 0..3 {
            assert!((recovered.slope[i] - sop.slope[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_drop_frame_fcm() {
        let comp = make_comp();
        let exp = ResolveEdlExporter::ntsc();
        let events = exp.export(&comp);
        let edl = exp.emit_edl(&events, "NTSC");
        assert!(edl.contains("FCM: DROP FRAME"));
    }

    #[test]
    fn test_importer_error_bad_line() {
        let edl = "TITLE: T\nFCM: NON-DROP FRAME\n\nBAD LINE\n";
        let imp = ResolveEdlImporter::new();
        let result = imp.parse(edl);
        assert!(result.is_err());
    }
}
