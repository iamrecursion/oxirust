//! OTIO (OpenTimelineIO) format import/export.
//!
//! OpenTimelineIO is an open-source format for representing and exchanging
//! editorial timeline information between NLE systems, VFX pipelines, and
//! editorial tools.
//!
//! This module implements a pure-Rust JSON-based serialiser and deserialiser
//! for the OTIO JSON representation, covering the core object types:
//! - `Timeline` → root container
//! - `Stack` / `Track` → container tracks
//! - `Clip` → a single source reference
//! - `Gap` → empty space on the timeline
//! - `Transition` → dissolve/wipe/etc.
//! - `TimeRange` + `RationalTime` → time representation
//!
//! The implementation does not require the `otio` C library; it operates
//! entirely on the JSON wire format.

#![allow(dead_code)]

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent, TrackType};
use crate::timecode::{EdlFrameRate, EdlTimecode};
use crate::Edl;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

// ─────────────────────────────────────────────────────────────────────────────
// RationalTime
// ─────────────────────────────────────────────────────────────────────────────

/// A time value represented as a rational number (value / rate).
///
/// Mirrors OTIO's `RationalTime` type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OtioRationalTime {
    /// Numerator value.
    pub value: f64,
    /// Rate (frames per second denominator).
    pub rate: f64,
}

impl OtioRationalTime {
    /// Create a new rational time.
    #[must_use]
    pub fn new(value: f64, rate: f64) -> Self {
        Self { value, rate }
    }

    /// Create from a frame count at a given frame rate.
    #[must_use]
    pub fn from_frames(frames: u64, fps: f64) -> Self {
        Self {
            value: frames as f64,
            rate: fps,
        }
    }

    /// Convert to seconds.
    #[must_use]
    pub fn to_seconds(&self) -> f64 {
        if self.rate == 0.0 {
            return 0.0;
        }
        self.value / self.rate
    }

    /// Convert to integer frame count.
    #[must_use]
    pub fn to_frames(&self) -> u64 {
        self.value as u64
    }

    /// Rescale to a different rate.
    #[must_use]
    pub fn rescaled_to(&self, new_rate: f64) -> Self {
        if self.rate == 0.0 {
            return Self::new(0.0, new_rate);
        }
        let new_value = self.value * new_rate / self.rate;
        Self::new(new_value, new_rate)
    }
}

impl std::fmt::Display for OtioRationalTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @ {}fps", self.value, self.rate)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeRange
// ─────────────────────────────────────────────────────────────────────────────

/// A time range with start time and duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OtioTimeRange {
    /// Start time.
    pub start_time: OtioRationalTime,
    /// Duration.
    pub duration: OtioRationalTime,
}

impl OtioTimeRange {
    /// Create a new time range.
    #[must_use]
    pub fn new(start_time: OtioRationalTime, duration: OtioRationalTime) -> Self {
        Self {
            start_time,
            duration,
        }
    }

    /// End time (start + duration).
    #[must_use]
    pub fn end_time_exclusive(&self) -> OtioRationalTime {
        OtioRationalTime::new(
            self.start_time.value + self.duration.rescaled_to(self.start_time.rate).value,
            self.start_time.rate,
        )
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.duration.to_seconds()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OTIO data model
// ─────────────────────────────────────────────────────────────────────────────

/// Kind of an OTIO track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtioTrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
}

impl OtioTrackKind {
    /// String representation used in OTIO JSON.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
        }
    }
}

/// An external reference to a source file.
#[derive(Debug, Clone)]
pub struct OtioExternalReference {
    /// File path or URI.
    pub target_url: String,
    /// Optional available range of the source file.
    pub available_range: Option<OtioTimeRange>,
}

/// A clip item on the timeline.
#[derive(Debug, Clone)]
pub struct OtioClip {
    /// Clip name.
    pub name: String,
    /// Optional external reference.
    pub media_reference: Option<OtioExternalReference>,
    /// Source range (what portion of the source to use).
    pub source_range: Option<OtioTimeRange>,
    /// User metadata (arbitrary key/value).
    pub metadata: HashMap<String, String>,
}

impl OtioClip {
    /// Create a new clip.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            media_reference: None,
            source_range: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the source range.
    #[must_use]
    pub fn with_source_range(mut self, range: OtioTimeRange) -> Self {
        self.source_range = Some(range);
        self
    }

    /// Set an external reference.
    #[must_use]
    pub fn with_reference(mut self, url: impl Into<String>) -> Self {
        self.media_reference = Some(OtioExternalReference {
            target_url: url.into(),
            available_range: None,
        });
        self
    }

    /// Duration in seconds from source_range (0.0 if no range).
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.source_range
            .map(|r| r.duration_seconds())
            .unwrap_or(0.0)
    }
}

/// A gap (black / silence) on the timeline.
#[derive(Debug, Clone, Copy)]
pub struct OtioGap {
    /// Duration of the gap.
    pub duration: OtioRationalTime,
}

/// A transition between two clips.
#[derive(Debug, Clone)]
pub struct OtioTransition {
    /// Transition name (e.g. "SMPTE_Dissolve").
    pub name: String,
    /// Duration.
    pub duration: OtioRationalTime,
    /// Overlap into the previous clip.
    pub in_offset: OtioRationalTime,
    /// Overlap into the next clip.
    pub out_offset: OtioRationalTime,
}

/// An item on an OTIO track.
#[derive(Debug, Clone)]
pub enum OtioItem {
    /// A source clip.
    Clip(OtioClip),
    /// An empty gap.
    Gap(OtioGap),
    /// A transition.
    Transition(OtioTransition),
}

impl OtioItem {
    /// Duration in frames at a given rate (best-effort).
    #[must_use]
    pub fn duration_frames(&self, fps: f64) -> u64 {
        match self {
            Self::Clip(c) => {
                if let Some(range) = c.source_range {
                    range.duration.rescaled_to(fps).to_frames()
                } else {
                    0
                }
            }
            Self::Gap(g) => g.duration.rescaled_to(fps).to_frames(),
            Self::Transition(t) => t.duration.rescaled_to(fps).to_frames(),
        }
    }
}

/// An OTIO track (sequence of items).
#[derive(Debug, Clone)]
pub struct OtioTrack {
    /// Track name.
    pub name: String,
    /// Track kind (video / audio).
    pub kind: OtioTrackKind,
    /// Items on this track.
    pub children: Vec<OtioItem>,
}

impl OtioTrack {
    /// Create a new empty track.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: OtioTrackKind) -> Self {
        Self {
            name: name.into(),
            kind,
            children: Vec::new(),
        }
    }

    /// Add an item.
    pub fn push(&mut self, item: OtioItem) {
        self.children.push(item);
    }

    /// Number of clips (excluding gaps and transitions).
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.children
            .iter()
            .filter(|i| matches!(i, OtioItem::Clip(_)))
            .count()
    }
}

/// An OTIO Stack (parallel container of tracks).
#[derive(Debug, Clone)]
pub struct OtioStack {
    /// Stack name.
    pub name: String,
    /// Tracks within the stack.
    pub tracks: Vec<OtioTrack>,
}

impl OtioStack {
    /// Create a new empty stack.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tracks: Vec::new(),
        }
    }

    /// Add a track.
    pub fn add_track(&mut self, track: OtioTrack) {
        self.tracks.push(track);
    }

    /// Total number of clips across all tracks.
    #[must_use]
    pub fn total_clip_count(&self) -> usize {
        self.tracks.iter().map(|t| t.clip_count()).sum()
    }
}

/// The root OTIO Timeline object.
#[derive(Debug, Clone)]
pub struct OtioTimeline {
    /// Timeline name (maps to EDL title).
    pub name: String,
    /// Global start time.
    pub global_start_time: Option<OtioRationalTime>,
    /// The tracks container.
    pub tracks: OtioStack,
    /// User metadata.
    pub metadata: HashMap<String, String>,
}

impl OtioTimeline {
    /// Create a new empty timeline.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        let name_str = name.into();
        Self {
            tracks: OtioStack::new(name_str.clone()),
            name: name_str,
            global_start_time: None,
            metadata: HashMap::new(),
        }
    }

    /// Total clip count across all tracks.
    #[must_use]
    pub fn total_clip_count(&self) -> usize {
        self.tracks.total_clip_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion: Edl → OtioTimeline
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an `Edl` into an `OtioTimeline`.
///
/// # Errors
///
/// Currently infallible, but returns `EdlResult` for future API compatibility.
pub fn edl_to_otio(edl: &Edl) -> EdlResult<OtioTimeline> {
    let title = edl.title.clone().unwrap_or_else(|| "Untitled".to_string());
    let mut timeline = OtioTimeline::new(&title);

    let fps = edl.frame_rate.fps() as f64;

    // Set global start time from first event's record_in
    if let Some(first) = edl.events.first() {
        timeline.global_start_time = Some(OtioRationalTime::from_frames(
            first.record_in.to_frames(),
            fps,
        ));
    }

    let mut video_track = OtioTrack::new("V1", OtioTrackKind::Video);
    let mut audio_track = OtioTrack::new("A1", OtioTrackKind::Audio);

    let mut prev_record_out: u64 = edl
        .events
        .first()
        .map(|e| e.record_in.to_frames())
        .unwrap_or(0);

    for event in &edl.events {
        let record_in_frames = event.record_in.to_frames();
        let record_out_frames = event.record_out.to_frames();
        let source_in_frames = event.source_in.to_frames();
        let source_out_frames = event.source_out.to_frames();

        // Insert gap if there's a hole in the timeline
        if record_in_frames > prev_record_out {
            let gap_duration = record_in_frames - prev_record_out;
            let gap = OtioGap {
                duration: OtioRationalTime::from_frames(gap_duration, fps),
            };
            video_track.push(OtioItem::Gap(gap));
            audio_track.push(OtioItem::Gap(gap));
        }

        // Build transition if needed
        if event.edit_type != EditType::Cut {
            let trans_dur = event.transition_duration.unwrap_or(0) as u64;
            let trans = OtioTransition {
                name: match event.edit_type {
                    EditType::Dissolve => "SMPTE_Dissolve".to_string(),
                    EditType::Wipe => "SMPTE_Wipe".to_string(),
                    EditType::Key => "SMPTE_Key".to_string(),
                    _ => "Unknown".to_string(),
                },
                duration: OtioRationalTime::from_frames(trans_dur, fps),
                in_offset: OtioRationalTime::from_frames(trans_dur / 2, fps),
                out_offset: OtioRationalTime::from_frames(trans_dur - trans_dur / 2, fps),
            };
            video_track.push(OtioItem::Transition(trans));
        }

        // Build clip
        let source_start = OtioRationalTime::from_frames(source_in_frames, fps);
        let source_dur =
            OtioRationalTime::from_frames(source_out_frames.saturating_sub(source_in_frames), fps);
        let source_range = OtioTimeRange::new(source_start, source_dur);

        let clip_name = event
            .clip_name
            .clone()
            .unwrap_or_else(|| event.reel.clone());
        let mut clip = OtioClip::new(&clip_name);
        clip.source_range = Some(source_range);
        clip.media_reference = Some(OtioExternalReference {
            target_url: format!("{}.mov", event.reel),
            available_range: None,
        });
        clip.metadata.insert("reel".to_string(), event.reel.clone());
        clip.metadata
            .insert("event_number".to_string(), event.number.to_string());

        match event.track {
            TrackType::Video
            | TrackType::AudioWithVideo
            | TrackType::AudioPairWithVideo
            | TrackType::VideoWithAudioMulti(_) => {
                video_track.push(OtioItem::Clip(clip.clone()));
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
                audio_track.push(OtioItem::Clip(clip));
            }
            _ => {}
        }

        prev_record_out = record_out_frames;
    }

    if video_track.clip_count() > 0 {
        timeline.tracks.add_track(video_track);
    }
    if audio_track.clip_count() > 0 {
        timeline.tracks.add_track(audio_track);
    }

    Ok(timeline)
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion: OtioTimeline → Edl
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an `OtioTimeline` back to an `Edl`.
///
/// Events are reconstructed from the first video track. If none exists, the
/// first audio track is used.
///
/// # Errors
///
/// Returns an error if timecodes cannot be constructed from the stored frame counts.
pub fn otio_to_edl(timeline: &OtioTimeline, fps: EdlFrameRate) -> EdlResult<Edl> {
    use crate::EdlFormat;

    let mut edl = Edl::new(EdlFormat::Cmx3600);
    if !timeline.name.is_empty() && timeline.name != "Untitled" {
        edl.set_title(timeline.name.clone());
    }
    edl.set_frame_rate(fps);

    let native_fps = fps.fps() as f64;

    // Find primary track
    let primary = timeline
        .tracks
        .tracks
        .iter()
        .find(|t| t.kind == OtioTrackKind::Video)
        .or_else(|| timeline.tracks.tracks.first());

    let Some(track) = primary else {
        return Ok(edl);
    };

    use crate::audio::AudioChannel;
    let track_type = match track.kind {
        OtioTrackKind::Video => TrackType::Video,
        OtioTrackKind::Audio => TrackType::Audio(AudioChannel::A1),
    };

    let mut event_num: u32 = 1;
    let mut record_cursor: u64 = timeline
        .global_start_time
        .map(|t| t.rescaled_to(native_fps).to_frames())
        .unwrap_or(0);

    for item in &track.children {
        match item {
            OtioItem::Gap(gap) => {
                record_cursor += gap.duration.rescaled_to(native_fps).to_frames();
            }
            OtioItem::Transition(_) => {
                // Transitions are informational here; the following clip will carry the edit type
            }
            OtioItem::Clip(clip) => {
                let source_range = clip.source_range.unwrap_or(OtioTimeRange::new(
                    OtioRationalTime::new(0.0, native_fps),
                    OtioRationalTime::new(0.0, native_fps),
                ));

                let source_in_frames = source_range.start_time.rescaled_to(native_fps).to_frames();
                let source_dur_frames = source_range.duration.rescaled_to(native_fps).to_frames();
                let source_out_frames = source_in_frames + source_dur_frames;

                let record_in_frames = record_cursor;
                let record_out_frames = record_cursor + source_dur_frames;

                let reel = clip
                    .metadata
                    .get("reel")
                    .cloned()
                    .unwrap_or_else(|| clip.name.clone());

                let source_in = EdlTimecode::from_frames(source_in_frames, fps)?;
                let source_out = EdlTimecode::from_frames(source_out_frames, fps)?;
                let record_in = EdlTimecode::from_frames(record_in_frames, fps)?;
                let record_out = EdlTimecode::from_frames(record_out_frames, fps)?;

                let mut event = EdlEvent::new(
                    event_num,
                    reel.clone(),
                    track_type.clone(),
                    EditType::Cut,
                    source_in,
                    source_out,
                    record_in,
                    record_out,
                );

                if clip.name != reel {
                    event.set_clip_name(clip.name.clone());
                }

                edl.add_event(event)?;
                record_cursor = record_out_frames;
                event_num += 1;
            }
        }
    }

    Ok(edl)
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON serialisation (OTIO wire format)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise an `OtioTimeline` to OTIO JSON format.
///
/// The format follows the OTIO JSON schema used by the Python `opentimelineio`
/// library for interchange with other tools.
///
/// # Errors
///
/// Returns an error if string formatting fails.
pub fn generate_otio_json(timeline: &OtioTimeline) -> EdlResult<String> {
    let mut out = String::new();
    let map_err = |e: std::fmt::Error| EdlError::ValidationError(format!("Write error: {e}"));

    writeln!(out, "{{").map_err(map_err)?;
    writeln!(out, "  \"OTIO_SCHEMA\": \"Timeline.1\",").map_err(map_err)?;
    writeln!(out, "  \"name\": \"{}\",", escape_json(&timeline.name)).map_err(map_err)?;

    if let Some(gst) = timeline.global_start_time {
        writeln!(out, "  \"global_start_time\": {{").map_err(map_err)?;
        writeln!(out, "    \"OTIO_SCHEMA\": \"RationalTime.1\",").map_err(map_err)?;
        writeln!(out, "    \"value\": {},", gst.value).map_err(map_err)?;
        writeln!(out, "    \"rate\": {}", gst.rate).map_err(map_err)?;
        writeln!(out, "  }},").map_err(map_err)?;
    }

    writeln!(out, "  \"tracks\": {{").map_err(map_err)?;
    writeln!(out, "    \"OTIO_SCHEMA\": \"Stack.1\",").map_err(map_err)?;
    writeln!(
        out,
        "    \"name\": \"{}\",",
        escape_json(&timeline.tracks.name)
    )
    .map_err(map_err)?;
    writeln!(out, "    \"children\": [").map_err(map_err)?;

    for (track_idx, track) in timeline.tracks.tracks.iter().enumerate() {
        let is_last_track = track_idx + 1 == timeline.tracks.tracks.len();
        writeln!(out, "      {{").map_err(map_err)?;
        writeln!(out, "        \"OTIO_SCHEMA\": \"Track.1\",").map_err(map_err)?;
        writeln!(out, "        \"name\": \"{}\",", escape_json(&track.name)).map_err(map_err)?;
        writeln!(out, "        \"kind\": \"{}\",", track.kind.as_str()).map_err(map_err)?;
        writeln!(out, "        \"children\": [").map_err(map_err)?;

        for (child_idx, child) in track.children.iter().enumerate() {
            let is_last_child = child_idx + 1 == track.children.len();
            write_otio_item(&mut out, child).map_err(map_err)?;
            if !is_last_child {
                writeln!(out, ",").map_err(map_err)?;
            } else {
                writeln!(out).map_err(map_err)?;
            }
        }

        writeln!(out, "        ]").map_err(map_err)?;
        if is_last_track {
            writeln!(out, "      }}").map_err(map_err)?;
        } else {
            writeln!(out, "      }},").map_err(map_err)?;
        }
    }

    writeln!(out, "    ]").map_err(map_err)?;
    writeln!(out, "  }}").map_err(map_err)?;
    writeln!(out, "}}").map_err(map_err)?;

    Ok(out)
}

fn write_otio_item(out: &mut String, item: &OtioItem) -> Result<(), std::fmt::Error> {
    match item {
        OtioItem::Clip(clip) => {
            writeln!(out, "          {{")?;
            writeln!(out, "            \"OTIO_SCHEMA\": \"Clip.1\",")?;
            writeln!(
                out,
                "            \"name\": \"{}\",",
                escape_json(&clip.name)
            )?;
            if let Some(range) = clip.source_range {
                writeln!(out, "            \"source_range\": {{")?;
                writeln!(out, "              \"OTIO_SCHEMA\": \"TimeRange.1\",")?;
                writeln!(out, "              \"start_time\": {{")?;
                writeln!(out, "                \"OTIO_SCHEMA\": \"RationalTime.1\",")?;
                writeln!(
                    out,
                    "                \"value\": {},",
                    range.start_time.value
                )?;
                writeln!(out, "                \"rate\": {}", range.start_time.rate)?;
                writeln!(out, "              }},")?;
                writeln!(out, "              \"duration\": {{")?;
                writeln!(out, "                \"OTIO_SCHEMA\": \"RationalTime.1\",")?;
                writeln!(out, "                \"value\": {},", range.duration.value)?;
                writeln!(out, "                \"rate\": {}", range.duration.rate)?;
                writeln!(out, "              }}")?;
                writeln!(out, "            }},")?;
            }
            if let Some(ref mref) = clip.media_reference {
                writeln!(out, "            \"media_reference\": {{")?;
                writeln!(
                    out,
                    "              \"OTIO_SCHEMA\": \"ExternalReference.1\","
                )?;
                writeln!(
                    out,
                    "              \"target_url\": \"{}\"",
                    escape_json(&mref.target_url)
                )?;
                writeln!(out, "            }},")?;
            }
            // metadata
            writeln!(out, "            \"metadata\": {{")?;
            let entries: Vec<_> = clip.metadata.iter().collect();
            for (i, (k, v)) in entries.iter().enumerate() {
                if i + 1 < entries.len() {
                    writeln!(
                        out,
                        "              \"{}\": \"{}\",",
                        escape_json(k),
                        escape_json(v)
                    )?;
                } else {
                    writeln!(
                        out,
                        "              \"{}\": \"{}\"",
                        escape_json(k),
                        escape_json(v)
                    )?;
                }
            }
            writeln!(out, "            }}")?;
            write!(out, "          }}")?;
        }
        OtioItem::Gap(gap) => {
            writeln!(out, "          {{")?;
            writeln!(out, "            \"OTIO_SCHEMA\": \"Gap.1\",")?;
            writeln!(out, "            \"duration\": {{")?;
            writeln!(out, "              \"OTIO_SCHEMA\": \"RationalTime.1\",")?;
            writeln!(out, "              \"value\": {},", gap.duration.value)?;
            writeln!(out, "              \"rate\": {}", gap.duration.rate)?;
            writeln!(out, "            }}")?;
            write!(out, "          }}")?;
        }
        OtioItem::Transition(trans) => {
            writeln!(out, "          {{")?;
            writeln!(out, "            \"OTIO_SCHEMA\": \"Transition.1\",")?;
            writeln!(
                out,
                "            \"name\": \"{}\",",
                escape_json(&trans.name)
            )?;
            writeln!(out, "            \"duration\": {{")?;
            writeln!(out, "              \"OTIO_SCHEMA\": \"RationalTime.1\",")?;
            writeln!(out, "              \"value\": {},", trans.duration.value)?;
            writeln!(out, "              \"rate\": {}", trans.duration.rate)?;
            writeln!(out, "            }}")?;
            write!(out, "          }}")?;
        }
    }
    Ok(())
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Parse an OTIO JSON string into an `OtioTimeline`.
///
/// This is a simplified parser that handles the common OTIO JSON structures.
/// It does not require a full JSON parser library.
///
/// # Errors
///
/// Returns an error if required fields are missing or malformed.
pub fn parse_otio_json(json: &str) -> EdlResult<OtioTimeline> {
    // Basic structural check
    if !json.contains("\"OTIO_SCHEMA\": \"Timeline.1\"")
        && !json.contains("\"OTIO_SCHEMA\":\"Timeline.1\"")
    {
        return Err(EdlError::parse(
            0,
            "Not a valid OTIO Timeline JSON (missing OTIO_SCHEMA: Timeline.1)",
        ));
    }

    let name = extract_json_string(json, "name").unwrap_or_else(|| "Untitled".to_string());
    let mut timeline = OtioTimeline::new(&name);

    // Parse global start time
    if let Some(gst_block) = extract_json_block(json, "global_start_time") {
        if let (Some(v), Some(r)) = (
            extract_json_number(&gst_block, "value"),
            extract_json_number(&gst_block, "rate"),
        ) {
            timeline.global_start_time = Some(OtioRationalTime::new(v, r));
        }
    }

    // Parse tracks block
    let tracks_block = extract_json_block(json, "tracks")
        .ok_or_else(|| EdlError::parse(0, "Missing 'tracks' in OTIO JSON"))?;

    // Parse children (individual tracks)
    let children_content =
        extract_json_array_content(&tracks_block, "children").unwrap_or_default();

    // Parse each Track
    let mut search = 0_usize;
    while let Some(rel) = children_content[search..]
        .find("\"OTIO_SCHEMA\": \"Track.1\"")
        .or_else(|| children_content[search..].find("\"OTIO_SCHEMA\":\"Track.1\""))
    {
        // Find the enclosing object for this track
        let abs = search + rel;
        // Walk back to find the opening brace
        let track_start = find_object_start(&children_content, abs);
        let track_end =
            find_matching_brace(&children_content, track_start).unwrap_or(children_content.len());
        let track_json = &children_content[track_start..track_end];

        let track_name =
            extract_json_string(track_json, "name").unwrap_or_else(|| "Track".to_string());
        let kind_str = extract_json_string(track_json, "kind").unwrap_or_default();
        let kind = if kind_str == "Audio" {
            OtioTrackKind::Audio
        } else {
            OtioTrackKind::Video
        };

        let mut track = OtioTrack::new(track_name, kind);

        // Parse children of this track
        let track_children = extract_json_array_content(track_json, "children").unwrap_or_default();
        let mut clip_search = 0_usize;
        while clip_search < track_children.len() {
            if let Some(schema_rel) = track_children[clip_search..].find("\"OTIO_SCHEMA\"") {
                let schema_abs = clip_search + schema_rel;
                let obj_start = find_object_start(&track_children, schema_abs);
                let obj_end =
                    find_matching_brace(&track_children, obj_start).unwrap_or(track_children.len());
                let obj_json = &track_children[obj_start..obj_end];

                if obj_json.contains("\"Clip.1\"") {
                    let clip = parse_otio_clip(obj_json);
                    track.push(OtioItem::Clip(clip));
                } else if obj_json.contains("\"Gap.1\"") {
                    if let Some(dur_block) = extract_json_block(obj_json, "duration") {
                        let value = extract_json_number(&dur_block, "value").unwrap_or(0.0);
                        let rate = extract_json_number(&dur_block, "rate").unwrap_or(25.0);
                        track.push(OtioItem::Gap(OtioGap {
                            duration: OtioRationalTime::new(value, rate),
                        }));
                    }
                } else if obj_json.contains("\"Transition.1\"") {
                    let trans_name = extract_json_string(obj_json, "name").unwrap_or_default();
                    let dur = extract_json_block(obj_json, "duration")
                        .map(|b| {
                            let v = extract_json_number(&b, "value").unwrap_or(0.0);
                            let r = extract_json_number(&b, "rate").unwrap_or(25.0);
                            OtioRationalTime::new(v, r)
                        })
                        .unwrap_or(OtioRationalTime::new(0.0, 25.0));
                    track.push(OtioItem::Transition(OtioTransition {
                        name: trans_name,
                        duration: dur,
                        in_offset: OtioRationalTime::new(0.0, 25.0),
                        out_offset: OtioRationalTime::new(0.0, 25.0),
                    }));
                }

                clip_search = obj_end;
            } else {
                break;
            }
        }

        timeline.tracks.add_track(track);
        search = track_start + (track_end - track_start);
    }

    Ok(timeline)
}

fn parse_otio_clip(json: &str) -> OtioClip {
    let name = extract_json_string(json, "name").unwrap_or_default();
    let mut clip = OtioClip::new(name);

    if let Some(range_block) = extract_json_block(json, "source_range") {
        let start_block = extract_json_block(&range_block, "start_time");
        let dur_block = extract_json_block(&range_block, "duration");
        if let (Some(sb), Some(db)) = (start_block, dur_block) {
            let sv = extract_json_number(&sb, "value").unwrap_or(0.0);
            let sr = extract_json_number(&sb, "rate").unwrap_or(25.0);
            let dv = extract_json_number(&db, "value").unwrap_or(0.0);
            let dr = extract_json_number(&db, "rate").unwrap_or(25.0);
            clip.source_range = Some(OtioTimeRange::new(
                OtioRationalTime::new(sv, sr),
                OtioRationalTime::new(dv, dr),
            ));
        }
    }

    if let Some(mref_block) = extract_json_block(json, "media_reference") {
        if let Some(url) = extract_json_string(&mref_block, "target_url") {
            clip.media_reference = Some(OtioExternalReference {
                target_url: url,
                available_range: None,
            });
        }
    }

    // Parse metadata as key/value pairs (simplified)
    if let Some(meta_block) = extract_json_block(json, "metadata") {
        // Extract simple string key/value pairs
        let mut pos = 0_usize;
        while let Some(quote_pos) = meta_block[pos..].find('"') {
            let key_start = pos + quote_pos + 1;
            let Some(key_end_rel) = meta_block[key_start..].find('"') else {
                break;
            };
            let key = &meta_block[key_start..key_start + key_end_rel];
            if key.starts_with("OTIO") {
                pos = key_start + key_end_rel + 1;
                continue;
            }
            // Find the colon and then the value
            let after_key = key_start + key_end_rel + 1;
            if let Some(colon_rel) = meta_block[after_key..].find(':') {
                let after_colon = after_key + colon_rel + 1;
                let trimmed = meta_block[after_colon..].trim_start();
                if trimmed.starts_with('"') {
                    let val_start =
                        after_colon + meta_block[after_colon..].find('"').unwrap_or(0) + 1;
                    if let Some(val_end_rel) = meta_block[val_start..].find('"') {
                        let val = &meta_block[val_start..val_start + val_end_rel];
                        clip.metadata.insert(key.to_string(), val.to_string());
                        pos = val_start + val_end_rel + 1;
                        continue;
                    }
                }
            }
            pos = key_start + key_end_rel + 1;
        }
    }

    clip
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON helpers (no external dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a JSON string value for a given key.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after_key = pos + pattern.len();
    let colon_pos = json[after_key..].find(':')? + after_key;
    let after_colon = &json[colon_pos + 1..];
    let trimmed = after_colon.trim_start();
    if !trimmed.starts_with('"') {
        return None;
    }
    let val_start = after_colon.find('"')? + colon_pos + 2;
    let val_content = &json[val_start..];
    let mut end = 0;
    let mut escaped = false;
    for (i, ch) in val_content.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            end = i;
            break;
        }
    }
    Some(val_content[..end].to_string())
}

/// Extract a JSON number for a given key.
fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after_key = pos + pattern.len();
    let colon_pos = json[after_key..].find(':')? + after_key;
    let after_colon = json[colon_pos + 1..].trim_start();
    // Read until non-numeric
    let end = after_colon
        .find(|c: char| {
            !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
        })
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse::<f64>().ok()
}

/// Find the start of the JSON object (the `{`) that contains the given position.
fn find_object_start(json: &str, near: usize) -> usize {
    // Walk backwards to find the enclosing `{`
    let bytes = json.as_bytes();
    let mut depth = 0_i32;
    let start = near.min(bytes.len().saturating_sub(1));
    for i in (0..=start).rev() {
        match bytes[i] {
            b'}' => depth += 1,
            b'{' => {
                if depth == 0 {
                    return i;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    0
}

/// Find the matching closing brace for an opening brace at `start`.
fn find_matching_brace(json: &str, start: usize) -> Option<usize> {
    let bytes = json.as_bytes();
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    for i in start..bytes.len() {
        let ch = bytes[i];
        if escape {
            escape = false;
            continue;
        }
        if ch == b'\\' && in_string {
            escape = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract the content of the first JSON object value for a given key.
fn extract_json_block(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after_key = pos + pattern.len();
    let colon_pos = json[after_key..].find(':')? + after_key;
    let trimmed_start = json[colon_pos + 1..].trim_start();
    if !trimmed_start.starts_with('{') {
        return None;
    }
    let brace_start = colon_pos + 1 + json[colon_pos + 1..].find('{').unwrap_or(0);
    let end = find_matching_brace(json, brace_start)?;
    Some(json[brace_start..end].to_string())
}

/// Extract the raw content between the `[` and `]` for a given key.
fn extract_json_array_content(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let pos = json.find(&pattern)?;
    let after_key = pos + pattern.len();
    let colon_pos = json[after_key..].find(':')? + after_key;
    let after_colon = &json[colon_pos + 1..];
    let bracket_rel = after_colon.find('[')?;
    let bracket_abs = colon_pos + 1 + bracket_rel;
    // Find matching ]
    let bytes = json.as_bytes();
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escape = false;
    for i in bracket_abs..bytes.len() {
        let ch = bytes[i];
        if escape {
            escape = false;
            continue;
        }
        if ch == b'\\' && in_string {
            escape = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(json[bracket_abs..i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
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

    fn make_test_edl() -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title("OTIO Test".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc_in");
        let tc_out = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc_out");

        let ev = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc_in,
            tc_out,
            tc_in,
            tc_out,
        );
        edl.add_event(ev).expect("add_event");

        let tc_in2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc_in2");
        let tc_out2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc_out2");
        let ev2 = EdlEvent::new(
            2,
            "B001".to_string(),
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
    fn test_rational_time_to_seconds() {
        let rt = OtioRationalTime::new(25.0, 25.0);
        assert!((rt.to_seconds() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rational_time_rescale() {
        let rt = OtioRationalTime::new(24.0, 24.0);
        let rescaled = rt.rescaled_to(25.0);
        assert!((rescaled.value - 25.0).abs() < 0.01);
        assert!((rescaled.rate - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rational_time_zero_rate() {
        let rt = OtioRationalTime::new(10.0, 0.0);
        assert_eq!(rt.to_seconds(), 0.0);
    }

    #[test]
    fn test_time_range_end_time() {
        let start = OtioRationalTime::new(0.0, 25.0);
        let dur = OtioRationalTime::new(125.0, 25.0);
        let range = OtioTimeRange::new(start, dur);
        let end = range.end_time_exclusive();
        assert!((end.value - 125.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edl_to_otio_title() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        assert_eq!(timeline.name, "OTIO Test");
    }

    #[test]
    fn test_edl_to_otio_clip_count() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        assert_eq!(timeline.total_clip_count(), 2);
    }

    #[test]
    fn test_edl_to_otio_track_count() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        assert_eq!(timeline.tracks.tracks.len(), 1); // video only
    }

    #[test]
    fn test_otio_clip_duration_seconds() {
        let start = OtioRationalTime::new(0.0, 25.0);
        let dur = OtioRationalTime::new(50.0, 25.0);
        let clip = OtioClip::new("test").with_source_range(OtioTimeRange::new(start, dur));
        assert!((clip.duration_seconds() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_otio_track_clip_count() {
        let mut track = OtioTrack::new("V1", OtioTrackKind::Video);
        track.push(OtioItem::Clip(OtioClip::new("c1")));
        track.push(OtioItem::Gap(OtioGap {
            duration: OtioRationalTime::new(10.0, 25.0),
        }));
        track.push(OtioItem::Clip(OtioClip::new("c2")));
        assert_eq!(track.clip_count(), 2);
    }

    #[test]
    fn test_generate_otio_json_contains_schema() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        let json = generate_otio_json(&timeline).expect("json generation should succeed");
        assert!(json.contains("\"OTIO_SCHEMA\": \"Timeline.1\""));
        assert!(json.contains("\"OTIO_SCHEMA\": \"Track.1\""));
        assert!(json.contains("\"OTIO_SCHEMA\": \"Clip.1\""));
    }

    #[test]
    fn test_generate_otio_json_contains_name() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        let json = generate_otio_json(&timeline).expect("json generation should succeed");
        assert!(json.contains("\"OTIO Test\""));
    }

    #[test]
    fn test_parse_otio_json_roundtrip_title() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("conversion should succeed");
        let json = generate_otio_json(&timeline).expect("json generation should succeed");
        let parsed = parse_otio_json(&json).expect("parsing should succeed");
        assert_eq!(parsed.name, timeline.name);
    }

    #[test]
    fn test_parse_otio_json_invalid() {
        let result = parse_otio_json("{\"not_otio\": true}");
        assert!(result.is_err());
    }

    #[test]
    fn test_otio_roundtrip_to_edl() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("edl->otio");
        let edl2 = otio_to_edl(&timeline, EdlFrameRate::Fps25).expect("otio->edl");

        assert_eq!(edl2.events.len(), edl.events.len());
        for (a, b) in edl.events.iter().zip(edl2.events.iter()) {
            assert_eq!(a.source_in.to_frames(), b.source_in.to_frames());
            assert_eq!(a.source_out.to_frames(), b.source_out.to_frames());
            assert_eq!(a.record_in.to_frames(), b.record_in.to_frames());
            assert_eq!(a.record_out.to_frames(), b.record_out.to_frames());
        }
    }

    #[test]
    fn test_otio_to_edl_empty_timeline() {
        let timeline = OtioTimeline::new("Empty");
        let edl = otio_to_edl(&timeline, EdlFrameRate::Fps25).expect("otio->edl");
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_otio_item_duration_frames() {
        let start = OtioRationalTime::new(0.0, 25.0);
        let dur = OtioRationalTime::new(50.0, 25.0);
        let clip = OtioClip::new("c").with_source_range(OtioTimeRange::new(start, dur));
        let item = OtioItem::Clip(clip);
        assert_eq!(item.duration_frames(25.0), 50);
    }

    #[test]
    fn test_otio_gap_item_duration() {
        let gap = OtioItem::Gap(OtioGap {
            duration: OtioRationalTime::new(30.0, 25.0),
        });
        assert_eq!(gap.duration_frames(25.0), 30);
    }

    #[test]
    fn test_otio_track_kind_str() {
        assert_eq!(OtioTrackKind::Video.as_str(), "Video");
        assert_eq!(OtioTrackKind::Audio.as_str(), "Audio");
    }

    #[test]
    fn test_otio_json_full_roundtrip_clip_count() {
        let edl = make_test_edl();
        let timeline = edl_to_otio(&edl).expect("edl->otio");
        let json = generate_otio_json(&timeline).expect("json generation");
        let parsed = parse_otio_json(&json).expect("parse json");
        assert_eq!(parsed.total_clip_count(), timeline.total_clip_count());
    }

    #[test]
    fn test_escape_json_quotes() {
        assert_eq!(escape_json("say \"hello\""), "say \\\"hello\\\"");
    }

    #[test]
    fn test_escape_json_backslash() {
        assert_eq!(escape_json("path\\to\\file"), "path\\\\to\\\\file");
    }
}
