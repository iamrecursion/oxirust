//! AAF (Advanced Authoring Format) round-trip support.
//!
//! This module provides import and export of EDL data in a simplified AAF-compatible
//! representation. True AAF files use a structured storage format (COM/OLE), which
//! is not practical to implement in pure Rust without significant effort. Instead,
//! this module provides:
//!
//! 1. An `AafComposition` data model that mirrors the key AAF constructs
//!    (Composition, MobSlot, TimelineMobSlot, SourceClip, etc.)
//! 2. A text-based serialisation/deserialisation using a simple XML-like format
//!    that can be processed by conversion tools such as `aaf2xml`.
//! 3. Round-trip conversion between `AafComposition` and the crate's `Edl` type.

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent, TrackType};
use crate::timecode::{EdlFrameRate, EdlTimecode};
use crate::Edl;
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// Data model
// ─────────────────────────────────────────────────────────────────────────────

/// Track kind in an AAF composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AafTrackKind {
    /// Video track (picture essence).
    Video,
    /// Audio track (sound essence).
    Audio,
    /// Timecode track.
    Timecode,
}

impl AafTrackKind {
    /// Convert to a short label string.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
            Self::Timecode => "Timecode",
        }
    }
}

impl std::fmt::Display for AafTrackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// A source clip within an AAF slot.
#[derive(Debug, Clone)]
pub struct AafSourceClip {
    /// Human-readable clip name.
    pub name: String,
    /// Tape/reel identifier.
    pub reel: String,
    /// Source in point (frames from clip origin).
    pub source_in: u64,
    /// Source out point (frames from clip origin).
    pub source_out: u64,
    /// Record in point (frames from timeline origin).
    pub record_in: u64,
    /// Record out point (frames from timeline origin).
    pub record_out: u64,
    /// Edit type (cut, dissolve, wipe, key).
    pub edit_type: String,
    /// Transition duration in frames (0 for cuts).
    pub transition_duration: u32,
}

impl AafSourceClip {
    /// Duration in frames on the record timeline.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.record_out.saturating_sub(self.record_in)
    }
}

/// An AAF timeline mob slot (one track on the composition timeline).
#[derive(Debug, Clone)]
pub struct AafSlot {
    /// Slot identifier (1-based).
    pub slot_id: u32,
    /// Human-readable slot name.
    pub name: String,
    /// Track kind.
    pub kind: AafTrackKind,
    /// Source clips in this slot, in timeline order.
    pub clips: Vec<AafSourceClip>,
    /// Edit rate (frame rate).
    pub edit_rate_numerator: u32,
    /// Edit rate denominator.
    pub edit_rate_denominator: u32,
}

impl AafSlot {
    /// Create a new empty slot.
    #[must_use]
    pub fn new(slot_id: u32, name: impl Into<String>, kind: AafTrackKind) -> Self {
        Self {
            slot_id,
            name: name.into(),
            kind,
            clips: Vec::new(),
            edit_rate_numerator: 25,
            edit_rate_denominator: 1,
        }
    }

    /// Total duration of this slot in frames.
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.clips.iter().map(|c| c.duration_frames()).sum()
    }

    /// Clip count.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

/// An AAF Composition Mob — the top-level timeline object.
#[derive(Debug, Clone)]
pub struct AafComposition {
    /// Composition name (maps to EDL title).
    pub name: String,
    /// Unique identifier (UUID-like string).
    pub mob_id: String,
    /// Edit rate numerator.
    pub edit_rate_numerator: u32,
    /// Edit rate denominator.
    pub edit_rate_denominator: u32,
    /// Whether timecodes are drop-frame.
    pub drop_frame: bool,
    /// Timeline slots (tracks).
    pub slots: Vec<AafSlot>,
}

impl AafComposition {
    /// Create a new empty composition.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mob_id: String::new(),
            edit_rate_numerator: 25,
            edit_rate_denominator: 1,
            drop_frame: false,
            slots: Vec::new(),
        }
    }

    /// Add a slot to the composition.
    pub fn add_slot(&mut self, slot: AafSlot) {
        self.slots.push(slot);
    }

    /// Get the nominal frame rate as a float.
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        if self.edit_rate_denominator == 0 {
            return 0.0;
        }
        self.edit_rate_numerator as f64 / self.edit_rate_denominator as f64
    }

    /// Total number of clips across all slots.
    #[must_use]
    pub fn total_clip_count(&self) -> usize {
        self.slots.iter().map(|s| s.clip_count()).sum()
    }

    /// Map frame rate to the crate's `EdlFrameRate` type.
    #[must_use]
    pub fn edl_frame_rate(&self) -> EdlFrameRate {
        let num = self.edit_rate_numerator;
        let den = self.edit_rate_denominator;
        match (num, den) {
            (24, 1) => EdlFrameRate::Fps24,
            (25, 1) => EdlFrameRate::Fps25,
            (30, 1) => EdlFrameRate::Fps30,
            (60, 1) => EdlFrameRate::Fps60,
            (30000, 1001) if self.drop_frame => EdlFrameRate::Fps2997DF,
            (30000, 1001) => EdlFrameRate::Fps2997NDF,
            _ => EdlFrameRate::Fps25,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion: Edl → AafComposition
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an `Edl` into an `AafComposition`.
///
/// The EDL's video events are placed in slot 1, and audio events in slot 2.
/// Transition information is carried on the clips.
///
/// # Errors
///
/// Currently infallible, but returns `EdlResult` for future API compatibility.
pub fn edl_to_aaf(edl: &Edl) -> EdlResult<AafComposition> {
    let name = edl.title.clone().unwrap_or_else(|| "Untitled".to_string());
    let mut comp = AafComposition::new(name);

    // Set edit rate from EDL frame rate
    let fps = edl.frame_rate;
    (
        comp.edit_rate_numerator,
        comp.edit_rate_denominator,
        comp.drop_frame,
    ) = match fps {
        EdlFrameRate::Fps24 => (24, 1, false),
        EdlFrameRate::Fps25 => (25, 1, false),
        EdlFrameRate::Fps30 => (30, 1, false),
        EdlFrameRate::Fps60 => (60, 1, false),
        EdlFrameRate::Fps2997DF => (30_000, 1001, true),
        EdlFrameRate::Fps2997NDF => (30_000, 1001, false),
        EdlFrameRate::Fps23_976 => (24_000, 1001, false),
        EdlFrameRate::Fps59_94 => (60_000, 1001, false),
        EdlFrameRate::Fps23976 => (24, 1, false),
        EdlFrameRate::Fps50 => (50, 1, false),
        EdlFrameRate::Fps5994 => (60_000, 1001, false),
    };

    let mut video_slot = AafSlot::new(1, "V1", AafTrackKind::Video);
    video_slot.edit_rate_numerator = comp.edit_rate_numerator;
    video_slot.edit_rate_denominator = comp.edit_rate_denominator;

    let mut audio_slot = AafSlot::new(2, "A1", AafTrackKind::Audio);
    audio_slot.edit_rate_numerator = comp.edit_rate_numerator;
    audio_slot.edit_rate_denominator = comp.edit_rate_denominator;

    for event in &edl.events {
        let clip = AafSourceClip {
            name: event
                .clip_name
                .clone()
                .unwrap_or_else(|| event.reel.clone()),
            reel: event.reel.clone(),
            source_in: event.source_in.to_frames(),
            source_out: event.source_out.to_frames(),
            record_in: event.record_in.to_frames(),
            record_out: event.record_out.to_frames(),
            edit_type: format!("{:?}", event.edit_type),
            transition_duration: event.transition_duration.unwrap_or(0),
        };

        match event.track {
            TrackType::Video
            | TrackType::AudioWithVideo
            | TrackType::AudioPairWithVideo
            | TrackType::VideoWithAudioMulti(_) => {
                video_slot.clips.push(clip.clone());
            }
            _ => {}
        }
        match &event.track {
            TrackType::Audio(_)
            | TrackType::AudioPair
            | TrackType::AudioWithVideo
            | TrackType::AudioPairWithVideo
            | TrackType::AudioMulti(_)
            | TrackType::VideoWithAudioMulti(_) => {
                audio_slot.clips.push(clip);
            }
            _ => {}
        }
    }

    if !video_slot.clips.is_empty() {
        comp.add_slot(video_slot);
    }
    if !audio_slot.clips.is_empty() {
        comp.add_slot(audio_slot);
    }

    Ok(comp)
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion: AafComposition → Edl
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an `AafComposition` back into an `Edl`.
///
/// Events are reconstructed from the video slot (slot kind `Video`).
/// If no video slot is found, audio clips are used instead.
///
/// # Errors
///
/// Returns an error if timecodes cannot be computed from the stored frame counts.
pub fn aaf_to_edl(comp: &AafComposition) -> EdlResult<Edl> {
    use crate::EdlFormat;

    let mut edl = Edl::new(EdlFormat::Cmx3600);
    if !comp.name.is_empty() && comp.name != "Untitled" {
        edl.set_title(comp.name.clone());
    }

    let fps = comp.edl_frame_rate();
    edl.set_frame_rate(fps);

    // Find the primary (video) slot
    let primary_slot = comp
        .slots
        .iter()
        .find(|s| s.kind == AafTrackKind::Video)
        .or_else(|| comp.slots.first());

    let Some(slot) = primary_slot else {
        return Ok(edl);
    };

    use crate::audio::AudioChannel;
    let track_type = match slot.kind {
        AafTrackKind::Video => TrackType::Video,
        AafTrackKind::Audio => TrackType::Audio(AudioChannel::A1),
        AafTrackKind::Timecode => TrackType::Video,
    };

    for (idx, clip) in slot.clips.iter().enumerate() {
        let event_num = (idx + 1) as u32;

        let source_in = EdlTimecode::from_frames(clip.source_in, fps)?;
        let source_out = EdlTimecode::from_frames(clip.source_out, fps)?;
        let record_in = EdlTimecode::from_frames(clip.record_in, fps)?;
        let record_out = EdlTimecode::from_frames(clip.record_out, fps)?;

        let edit_type = match clip.edit_type.as_str() {
            "Dissolve" => EditType::Dissolve,
            "Wipe" => EditType::Wipe,
            "Key" => EditType::Key,
            _ => EditType::Cut,
        };

        let mut event = EdlEvent::new(
            event_num,
            clip.reel.clone(),
            track_type.clone(),
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
        );

        if clip.name != clip.reel {
            event.set_clip_name(clip.name.clone());
        }
        if clip.transition_duration > 0 {
            event.transition_duration = Some(clip.transition_duration);
        }

        edl.add_event(event)?;
    }

    Ok(edl)
}

// ─────────────────────────────────────────────────────────────────────────────
// Text serialisation (AAF XML-like export)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a text-based AAF-XML representation of the composition.
///
/// The format is a simplified XML that mirrors the AAF object model.
/// It is not a valid AAF binary file, but can be processed by conversion tools.
///
/// # Errors
///
/// Returns an error if string formatting fails.
pub fn generate_aaf_xml(comp: &AafComposition) -> EdlResult<String> {
    let mut out = String::new();
    let map_err = |e: std::fmt::Error| EdlError::ValidationError(format!("Write error: {e}"));

    writeln!(out, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>").map_err(map_err)?;
    writeln!(out, "<aaf>").map_err(map_err)?;
    writeln!(out, "  <CompositionMob>").map_err(map_err)?;
    writeln!(out, "    <Name>{}</Name>", escape_xml(&comp.name)).map_err(map_err)?;
    writeln!(out, "    <MobID>{}</MobID>", comp.mob_id).map_err(map_err)?;
    writeln!(
        out,
        "    <EditRate>{}/{}</EditRate>",
        comp.edit_rate_numerator, comp.edit_rate_denominator
    )
    .map_err(map_err)?;
    writeln!(
        out,
        "    <DropFrame>{}</DropFrame>",
        if comp.drop_frame { "true" } else { "false" }
    )
    .map_err(map_err)?;

    for slot in &comp.slots {
        writeln!(out, "    <TimelineMobSlot id=\"{}\">", slot.slot_id).map_err(map_err)?;
        writeln!(out, "      <SlotName>{}</SlotName>", escape_xml(&slot.name)).map_err(map_err)?;
        writeln!(out, "      <TrackKind>{}</TrackKind>", slot.kind).map_err(map_err)?;
        writeln!(
            out,
            "      <EditRate>{}/{}</EditRate>",
            slot.edit_rate_numerator, slot.edit_rate_denominator
        )
        .map_err(map_err)?;
        writeln!(out, "      <Sequence>").map_err(map_err)?;

        for clip in &slot.clips {
            writeln!(out, "        <SourceClip>").map_err(map_err)?;
            writeln!(out, "          <Name>{}</Name>", escape_xml(&clip.name)).map_err(map_err)?;
            writeln!(out, "          <Reel>{}</Reel>", escape_xml(&clip.reel)).map_err(map_err)?;
            writeln!(out, "          <SourceIn>{}</SourceIn>", clip.source_in).map_err(map_err)?;
            writeln!(out, "          <SourceOut>{}</SourceOut>", clip.source_out)
                .map_err(map_err)?;
            writeln!(out, "          <RecordIn>{}</RecordIn>", clip.record_in).map_err(map_err)?;
            writeln!(out, "          <RecordOut>{}</RecordOut>", clip.record_out)
                .map_err(map_err)?;
            writeln!(
                out,
                "          <EditType>{}</EditType>",
                escape_xml(&clip.edit_type)
            )
            .map_err(map_err)?;
            writeln!(
                out,
                "          <TransitionDuration>{}</TransitionDuration>",
                clip.transition_duration
            )
            .map_err(map_err)?;
            writeln!(out, "        </SourceClip>").map_err(map_err)?;
        }

        writeln!(out, "      </Sequence>").map_err(map_err)?;
        writeln!(out, "    </TimelineMobSlot>").map_err(map_err)?;
    }

    writeln!(out, "  </CompositionMob>").map_err(map_err)?;
    writeln!(out, "</aaf>").map_err(map_err)?;

    Ok(out)
}

/// Parse an AAF XML string back into an `AafComposition`.
///
/// # Errors
///
/// Returns an error if required XML structure is missing or invalid.
pub fn parse_aaf_xml(xml: &str) -> EdlResult<AafComposition> {
    let name = extract_xml_text(xml, "Name").unwrap_or_else(|| "Untitled".to_string());
    let mob_id = extract_xml_text(xml, "MobID").unwrap_or_default();
    let edit_rate_str = extract_xml_text(xml, "EditRate").unwrap_or_else(|| "25/1".to_string());
    let drop_frame =
        extract_xml_text(xml, "DropFrame").is_some_and(|v| v.trim().eq_ignore_ascii_case("true"));

    let (edit_rate_num, edit_rate_den) = parse_rate(&edit_rate_str);

    let mut comp = AafComposition::new(name);
    comp.mob_id = mob_id;
    comp.edit_rate_numerator = edit_rate_num;
    comp.edit_rate_denominator = edit_rate_den;
    comp.drop_frame = drop_frame;

    // Parse slots
    let mut search = 0_usize;
    while let Some(rel) = xml[search..].find("<TimelineMobSlot") {
        let abs = search + rel;
        let Some(rel_end) = xml[abs..].find("</TimelineMobSlot>") else {
            break;
        };
        let slot_xml = &xml[abs..abs + rel_end + "</TimelineMobSlot>".len()];

        let slot_id = extract_xml_attr(slot_xml, "TimelineMobSlot", "id")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1);
        let slot_name = extract_xml_text(slot_xml, "SlotName").unwrap_or_default();
        let kind_str = extract_xml_text(slot_xml, "TrackKind").unwrap_or_default();
        let kind = match kind_str.trim() {
            "Audio" => AafTrackKind::Audio,
            "Timecode" => AafTrackKind::Timecode,
            _ => AafTrackKind::Video,
        };
        let slot_rate_str =
            extract_xml_text(slot_xml, "EditRate").unwrap_or_else(|| "25/1".to_string());
        let (slot_num, slot_den) = parse_rate(&slot_rate_str);

        let mut slot = AafSlot::new(slot_id, slot_name, kind);
        slot.edit_rate_numerator = slot_num;
        slot.edit_rate_denominator = slot_den;

        // Parse clips
        let mut clip_search = 0_usize;
        while let Some(rel_clip) = slot_xml[clip_search..].find("<SourceClip>") {
            let abs_clip = clip_search + rel_clip;
            let Some(rel_clip_end) = slot_xml[abs_clip..].find("</SourceClip>") else {
                break;
            };
            let clip_xml = &slot_xml[abs_clip..abs_clip + rel_clip_end + "</SourceClip>".len()];

            let clip_name = extract_xml_text(clip_xml, "Name").unwrap_or_default();
            let reel = extract_xml_text(clip_xml, "Reel").unwrap_or_default();
            let source_in = extract_xml_text(clip_xml, "SourceIn")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let source_out = extract_xml_text(clip_xml, "SourceOut")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let record_in = extract_xml_text(clip_xml, "RecordIn")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let record_out = extract_xml_text(clip_xml, "RecordOut")
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let edit_type = extract_xml_text(clip_xml, "EditType").unwrap_or_default();
            let transition_duration = extract_xml_text(clip_xml, "TransitionDuration")
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(0);

            slot.clips.push(AafSourceClip {
                name: clip_name,
                reel,
                source_in,
                source_out,
                record_in,
                record_out,
                edit_type,
                transition_duration,
            });

            clip_search = abs_clip + rel_clip_end + "</SourceClip>".len();
        }

        comp.add_slot(slot);
        search = abs + rel_end + "</TimelineMobSlot>".len();
    }

    Ok(comp)
}

// ─────────────────────────────────────────────────────────────────────────────
// XML helpers
// ─────────────────────────────────────────────────────────────────────────────

fn extract_xml_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let content_start = start + open.len();
    let end = xml[content_start..].find(&close)?;
    Some(xml[content_start..content_start + end].to_string())
}

fn extract_xml_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let tag_start = xml.find(&format!("<{tag}"))?;
    let tag_end = xml[tag_start..].find('>')?;
    let tag_text = &xml[tag_start..tag_start + tag_end];
    let attr_pattern = format!("{attr}=\"");
    let attr_start = tag_text.find(&attr_pattern)?;
    let value_start = attr_start + attr_pattern.len();
    let value_end = tag_text[value_start..].find('"')?;
    Some(tag_text[value_start..value_start + value_end].to_string())
}

fn parse_rate(s: &str) -> (u32, u32) {
    let parts: Vec<&str> = s.splitn(2, '/').collect();
    let num = parts
        .first()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(25);
    let den = parts
        .get(1)
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(1);
    (num, den)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::EdlTimecode;
    use crate::{Edl, EdlFormat};

    fn make_edl_with_events() -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title("AAF Test".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc_in");
        let tc_out = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc_out");

        let mut ev = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc_in,
            tc_out,
            tc_in,
            tc_out,
        );
        ev.set_clip_name("shot001.mov".to_string());
        edl.add_event(ev).expect("add_event");

        let tc_in2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc_in2");
        let tc_out2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc_out2");
        let ev2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc_in2,
            tc_out2,
            tc_in2,
            tc_out2,
        );
        edl.add_event(ev2).expect("add_event2");

        edl
    }

    #[test]
    fn test_edl_to_aaf_composition() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        assert_eq!(comp.name, "AAF Test");
        assert_eq!(comp.edit_rate_numerator, 25);
        assert_eq!(comp.edit_rate_denominator, 1);
        assert!(!comp.drop_frame);
    }

    #[test]
    fn test_edl_to_aaf_video_slot() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");

        let video_slot = comp.slots.iter().find(|s| s.kind == AafTrackKind::Video);
        assert!(video_slot.is_some());
        let video_slot = video_slot.expect("video slot should exist");
        assert_eq!(video_slot.clip_count(), 2);
        assert_eq!(video_slot.clips[0].reel, "A001");
        assert_eq!(video_slot.clips[1].reel, "A002");
    }

    #[test]
    fn test_edl_to_aaf_clip_name_preserved() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        let video_slot = comp
            .slots
            .iter()
            .find(|s| s.kind == AafTrackKind::Video)
            .expect("video slot");
        assert_eq!(video_slot.clips[0].name, "shot001.mov");
    }

    #[test]
    fn test_aaf_composition_frame_rate() {
        let mut comp = AafComposition::new("test");
        comp.edit_rate_numerator = 25;
        comp.edit_rate_denominator = 1;
        assert!((comp.frame_rate() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aaf_composition_zero_denominator() {
        let mut comp = AafComposition::new("test");
        comp.edit_rate_denominator = 0;
        assert_eq!(comp.frame_rate(), 0.0);
    }

    #[test]
    fn test_aaf_generate_xml() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        let xml = generate_aaf_xml(&comp).expect("xml generation should succeed");

        assert!(xml.contains("<Name>AAF Test</Name>"));
        assert!(xml.contains("<EditRate>25/1</EditRate>"));
        assert!(xml.contains("<SourceClip>"));
        assert!(xml.contains("<Reel>A001</Reel>"));
    }

    #[test]
    fn test_aaf_xml_roundtrip_composition() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        let xml = generate_aaf_xml(&comp).expect("xml generation should succeed");
        let reparsed = parse_aaf_xml(&xml).expect("xml parsing should succeed");

        assert_eq!(reparsed.name, comp.name);
        assert_eq!(reparsed.edit_rate_numerator, comp.edit_rate_numerator);
        assert_eq!(reparsed.edit_rate_denominator, comp.edit_rate_denominator);
    }

    #[test]
    fn test_aaf_xml_roundtrip_slot_count() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        let xml = generate_aaf_xml(&comp).expect("xml generation should succeed");
        let reparsed = parse_aaf_xml(&xml).expect("xml parsing should succeed");

        assert_eq!(reparsed.slots.len(), comp.slots.len());
    }

    #[test]
    fn test_aaf_xml_roundtrip_clips() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion should succeed");
        let xml = generate_aaf_xml(&comp).expect("xml generation should succeed");
        let reparsed = parse_aaf_xml(&xml).expect("xml parsing should succeed");

        let orig_video = comp
            .slots
            .iter()
            .find(|s| s.kind == AafTrackKind::Video)
            .expect("orig video slot");
        let rep_video = reparsed
            .slots
            .iter()
            .find(|s| s.kind == AafTrackKind::Video)
            .expect("reparsed video slot");

        assert_eq!(orig_video.clip_count(), rep_video.clip_count());
        for (a, b) in orig_video.clips.iter().zip(rep_video.clips.iter()) {
            assert_eq!(a.reel, b.reel);
            assert_eq!(a.source_in, b.source_in);
            assert_eq!(a.record_in, b.record_in);
        }
    }

    #[test]
    fn test_aaf_to_edl_roundtrip() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("edl->aaf should succeed");
        let edl2 = aaf_to_edl(&comp).expect("aaf->edl should succeed");

        assert_eq!(edl2.events.len(), edl.events.len());
        for (a, b) in edl.events.iter().zip(edl2.events.iter()) {
            assert_eq!(a.reel, b.reel);
            assert_eq!(a.source_in.to_frames(), b.source_in.to_frames());
            assert_eq!(a.record_in.to_frames(), b.record_in.to_frames());
        }
    }

    #[test]
    fn test_aaf_to_edl_empty_composition() {
        let comp = AafComposition::new("Empty");
        let edl = aaf_to_edl(&comp).expect("aaf->edl should succeed");
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_aaf_source_clip_duration() {
        let clip = AafSourceClip {
            name: "test".to_string(),
            reel: "A001".to_string(),
            source_in: 0,
            source_out: 125,
            record_in: 0,
            record_out: 125,
            edit_type: "Cut".to_string(),
            transition_duration: 0,
        };
        assert_eq!(clip.duration_frames(), 125);
    }

    #[test]
    fn test_aaf_slot_total_duration() {
        let mut slot = AafSlot::new(1, "V1", AafTrackKind::Video);
        slot.clips.push(AafSourceClip {
            name: "c1".to_string(),
            reel: "A1".to_string(),
            source_in: 0,
            source_out: 50,
            record_in: 0,
            record_out: 50,
            edit_type: "Cut".to_string(),
            transition_duration: 0,
        });
        slot.clips.push(AafSourceClip {
            name: "c2".to_string(),
            reel: "A2".to_string(),
            source_in: 0,
            source_out: 75,
            record_in: 50,
            record_out: 125,
            edit_type: "Cut".to_string(),
            transition_duration: 0,
        });
        assert_eq!(slot.total_duration(), 125);
    }

    #[test]
    fn test_aaf_track_kind_label() {
        assert_eq!(AafTrackKind::Video.label(), "Video");
        assert_eq!(AafTrackKind::Audio.label(), "Audio");
        assert_eq!(AafTrackKind::Timecode.label(), "Timecode");
    }

    #[test]
    fn test_aaf_xml_drop_frame() {
        let mut comp = AafComposition::new("DF Test");
        comp.edit_rate_numerator = 30_000;
        comp.edit_rate_denominator = 1001;
        comp.drop_frame = true;

        let xml = generate_aaf_xml(&comp).expect("generate");
        assert!(xml.contains("<DropFrame>true</DropFrame>"));

        let reparsed = parse_aaf_xml(&xml).expect("parse");
        assert!(reparsed.drop_frame);
    }

    #[test]
    fn test_aaf_edl_frame_rate_mapping() {
        let mut comp = AafComposition::new("test");
        comp.edit_rate_numerator = 24;
        comp.edit_rate_denominator = 1;
        assert_eq!(comp.edl_frame_rate(), EdlFrameRate::Fps24);

        comp.edit_rate_numerator = 25;
        assert_eq!(comp.edl_frame_rate(), EdlFrameRate::Fps25);

        comp.edit_rate_numerator = 30_000;
        comp.edit_rate_denominator = 1001;
        comp.drop_frame = true;
        assert_eq!(comp.edl_frame_rate(), EdlFrameRate::Fps2997DF);
    }

    #[test]
    fn test_aaf_total_clip_count() {
        let edl = make_edl_with_events();
        let comp = edl_to_aaf(&edl).expect("conversion");
        assert_eq!(comp.total_clip_count(), 2); // 2 video clips, no audio
    }

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(escape_xml("A&B"), "A&amp;B");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
    }
}
