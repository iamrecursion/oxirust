//! OpenTimelineIO (OTIO) importer for conform sessions.
//!
//! Parses a subset of the OpenTimelineIO JSON format:
//! - `Timeline`, `Track`, `Stack`, `Clip`, `Gap` schema objects
//! - Rational time and time range representations
//! - Source clip metadata extraction

use crate::error::{ConformError, ConformResult};
use crate::types::{ClipReference, FrameRate, Timecode, TrackType};
use serde::Deserialize;
use std::collections::HashMap;

// ─── Raw OTIO JSON deserialization structs ────────────────────────────────────

/// Raw OTIO rational-time value: `{"value": 0.0, "rate": 24.0}`.
#[derive(Debug, Clone, Deserialize)]
pub struct OtioRationalTime {
    /// Numeric value (frame count at the given rate).
    pub value: f64,
    /// Frames-per-second denominator.
    pub rate: f64,
}

impl OtioRationalTime {
    /// Convert to seconds.
    #[must_use]
    pub fn to_seconds(&self) -> f64 {
        if self.rate == 0.0 {
            return 0.0;
        }
        self.value / self.rate
    }

    /// Convert to a `Timecode` using the embedded rate.
    #[must_use]
    pub fn to_timecode(&self) -> Timecode {
        let total_frames = self.value.round() as u64;
        let fps_u = self.rate.round().max(1.0) as u64;
        let hours = (total_frames / (3600 * fps_u)) as u8;
        let rem = total_frames % (3600 * fps_u);
        let minutes = (rem / (60 * fps_u)) as u8;
        let rem2 = rem % (60 * fps_u);
        let seconds = (rem2 / fps_u) as u8;
        let frames = (rem2 % fps_u) as u8;
        Timecode::new(hours, minutes, seconds, frames)
    }

    /// Map the rate to the nearest `FrameRate` variant.
    #[must_use]
    pub fn to_frame_rate(&self) -> FrameRate {
        rate_to_frame_rate(self.rate)
    }
}

/// Raw OTIO time-range: `{"start_time": {...}, "duration": {...}}`.
#[derive(Debug, Clone, Deserialize)]
pub struct OtioTimeRange {
    /// Start of the range.
    pub start_time: OtioRationalTime,
    /// Duration of the range.
    pub duration: OtioRationalTime,
}

impl OtioTimeRange {
    /// Compute the end-time rational.
    #[must_use]
    pub fn end_time(&self) -> OtioRationalTime {
        OtioRationalTime {
            value: self.start_time.value + self.duration.value,
            rate: self.start_time.rate,
        }
    }
}

/// A generic OTIO JSON object — may be `Timeline`, `Track`, `Clip`, `Gap`, `Stack`, etc.
#[derive(Debug, Clone, Deserialize)]
pub struct OtioObject {
    /// OTIO schema string, e.g. `"Clip.1"` or `"Timeline.1"`.
    #[serde(rename = "OTIO_SCHEMA")]
    pub schema: String,
    /// Human-readable name.
    #[serde(default)]
    pub name: String,
    /// Source range (for Clip / Gap).
    pub source_range: Option<OtioTimeRange>,
    /// Children objects (for Stack / Track).
    pub children: Option<Vec<OtioObject>>,
    /// Track kind hint: `"Video"` | `"Audio"`.
    pub kind: Option<String>,
    /// Tracks container (for Timeline — wraps a Stack).
    pub tracks: Option<Box<OtioObject>>,
    /// Metadata object (free-form).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl OtioObject {
    /// Extract the schema type name (before the dot).
    #[must_use]
    pub fn schema_type(&self) -> &str {
        self.schema.split('.').next().unwrap_or(&self.schema)
    }
}

// ─── Public OTIO importer ─────────────────────────────────────────────────────

/// Imports a conform session from an OpenTimelineIO JSON string.
pub struct OtioImporter {
    /// Default frame rate to use when none is specified.
    default_fps: f64,
}

impl OtioImporter {
    /// Create a new importer with default fps = 25.
    #[must_use]
    pub fn new() -> Self {
        Self { default_fps: 25.0 }
    }

    /// Create a new importer with an explicit default fps.
    #[must_use]
    pub fn with_default_fps(fps: f64) -> Self {
        Self { default_fps: fps }
    }

    /// Parse an OTIO JSON string and return the list of `ClipReference`s.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed or the OTIO schema is unsupported.
    pub fn import(&self, json: &str) -> ConformResult<Vec<ClipReference>> {
        let root: OtioObject =
            serde_json::from_str(json).map_err(|e| ConformError::Other(e.to_string()))?;

        let schema_type = root.schema_type();
        if schema_type != "Timeline" {
            return Err(ConformError::UnsupportedFormat(format!(
                "Expected OTIO Timeline schema, got '{}'",
                root.schema
            )));
        }

        let mut clips = Vec::new();
        self.collect_from_timeline(&root, &mut clips)?;
        Ok(clips)
    }

    // ── Internal traversal ────────────────────────────────────────────────────

    fn collect_from_timeline(
        &self,
        timeline: &OtioObject,
        clips: &mut Vec<ClipReference>,
    ) -> ConformResult<()> {
        // The timeline has a `tracks` field which is a Stack.
        if let Some(tracks_obj) = &timeline.tracks {
            self.collect_from_object(tracks_obj, clips)?;
        }
        Ok(())
    }

    fn collect_from_object(
        &self,
        obj: &OtioObject,
        clips: &mut Vec<ClipReference>,
    ) -> ConformResult<()> {
        match obj.schema_type() {
            "Stack" => {
                // A Stack contains a list of Tracks (children).
                if let Some(children) = &obj.children {
                    for child in children {
                        self.collect_from_object(child, clips)?;
                    }
                }
            }
            "Track" => {
                let track_type = parse_track_kind(obj.kind.as_deref());
                if let Some(children) = &obj.children {
                    for child in children {
                        self.collect_from_item(child, track_type, clips)?;
                    }
                }
            }
            "Timeline" => {
                self.collect_from_timeline(obj, clips)?;
            }
            other => {
                return Err(ConformError::UnsupportedFormat(format!(
                    "Unexpected OTIO container schema: '{other}'"
                )));
            }
        }
        Ok(())
    }

    fn collect_from_item(
        &self,
        obj: &OtioObject,
        track_type: TrackType,
        clips: &mut Vec<ClipReference>,
    ) -> ConformResult<()> {
        match obj.schema_type() {
            "Clip" => {
                let clip = self.clip_reference_from(obj, track_type)?;
                clips.push(clip);
            }
            "Gap" => {
                // Gaps are skipped — they represent silence / black.
            }
            "Stack" | "Track" => {
                // Nested tracks/stacks — recurse
                self.collect_from_object(obj, clips)?;
            }
            other => {
                tracing::warn!("Unknown OTIO item schema '{}', skipping", other);
            }
        }
        Ok(())
    }

    fn clip_reference_from(
        &self,
        clip_obj: &OtioObject,
        track_type: TrackType,
    ) -> ConformResult<ClipReference> {
        let source_range = clip_obj.source_range.as_ref();

        // Determine fps from the source_range or fall back to default
        let fps_value = source_range
            .map(|r| r.start_time.rate)
            .unwrap_or(self.default_fps);
        let fps = rate_to_frame_rate(fps_value);

        // Source in / out
        let (source_in, source_out) = if let Some(range) = source_range {
            (
                range.start_time.to_timecode(),
                range.end_time().to_timecode(),
            )
        } else {
            (Timecode::new(0, 0, 0, 0), Timecode::new(0, 0, 0, 0))
        };

        // Source file from metadata or clip name
        let source_file = extract_source_file(clip_obj);

        // Unique ID — prefer metadata "reel" or clip name
        let id = if !clip_obj.name.is_empty() {
            clip_obj.name.clone()
        } else {
            uuid::Uuid::new_v4().to_string()
        };

        // Build metadata map
        let mut metadata: HashMap<String, String> = HashMap::new();
        for (k, v) in &clip_obj.metadata {
            let value_str = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            metadata.insert(k.clone(), value_str);
        }
        if let Some(ref sf) = source_file {
            metadata.insert("otio_source_file".to_string(), sf.clone());
        }
        metadata.insert("otio_schema".to_string(), clip_obj.schema.clone());

        Ok(ClipReference {
            id,
            source_file,
            source_in,
            source_out,
            record_in: source_in,
            record_out: source_out,
            track: track_type,
            fps,
            metadata,
        })
    }
}

impl Default for OtioImporter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn rate_to_frame_rate(rate: f64) -> FrameRate {
    // Use tight tolerances for integer rates (0.01) and looser for fractional (0.02).
    // Integer rates are checked first so e.g. 24.0 never accidentally maps to 23.976.
    if (rate - 24.0).abs() < 0.01 {
        FrameRate::Fps24
    } else if (rate - 25.0).abs() < 0.01 {
        FrameRate::Fps25
    } else if (rate - 30.0).abs() < 0.01 {
        FrameRate::Fps30
    } else if (rate - 50.0).abs() < 0.01 {
        FrameRate::Fps50
    } else if (rate - 60.0).abs() < 0.01 {
        FrameRate::Fps60
    } else if (rate - 23.976).abs() < 0.02 {
        FrameRate::Fps23976
    } else if (rate - 29.97).abs() < 0.02 {
        FrameRate::Fps2997NDF
    } else if (rate - 59.94).abs() < 0.02 {
        FrameRate::Fps5994
    } else {
        FrameRate::Custom(rate)
    }
}

fn parse_track_kind(kind: Option<&str>) -> TrackType {
    match kind {
        Some("Video") | Some("video") => TrackType::Video,
        Some("Audio") | Some("audio") => TrackType::Audio,
        _ => TrackType::Video, // default
    }
}

fn extract_source_file(clip_obj: &OtioObject) -> Option<String> {
    // Check common metadata paths used by OTIO exporters
    // 1. metadata["media_references"]["DEFAULT_MEDIA"]["target_url"]
    if let Some(serde_json::Value::Object(media_refs)) = clip_obj.metadata.get("media_references") {
        for (_key, media_ref) in media_refs {
            if let serde_json::Value::Object(ref_map) = media_ref {
                if let Some(serde_json::Value::String(url)) = ref_map.get("target_url") {
                    return Some(url.clone());
                }
            }
        }
    }
    // 2. metadata["source_file"]
    if let Some(serde_json::Value::String(sf)) = clip_obj.metadata.get("source_file") {
        return Some(sf.clone());
    }
    // 3. Fall back to clip name as source hint
    if !clip_obj.name.is_empty() {
        return Some(clip_obj.name.clone());
    }
    None
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_timeline_json(name: &str, clips: &[(&str, f64, f64, f64)]) -> String {
        // clips: (name, start_value, duration_value, rate)
        let clip_jsons: Vec<String> = clips
            .iter()
            .map(|(n, start, dur, rate)| {
                format!(
                    r#"{{
                        "OTIO_SCHEMA": "Clip.1",
                        "name": "{n}",
                        "source_range": {{
                            "start_time": {{"value": {start}, "rate": {rate}}},
                            "duration": {{"value": {dur}, "rate": {rate}}}
                        }},
                        "metadata": {{}}
                    }}"#
                )
            })
            .collect();
        let clips_str = clip_jsons.join(",");
        format!(
            r#"{{
                "OTIO_SCHEMA": "Timeline.1",
                "name": "{name}",
                "tracks": {{
                    "OTIO_SCHEMA": "Stack.1",
                    "name": "tracks",
                    "children": [
                        {{
                            "OTIO_SCHEMA": "Track.1",
                            "name": "V1",
                            "kind": "Video",
                            "children": [{clips_str}]
                        }}
                    ]
                }}
            }}"#
        )
    }

    #[test]
    fn test_import_empty_timeline() {
        let json = minimal_timeline_json("Empty", &[]);
        let importer = OtioImporter::new();
        let clips = importer.import(&json).expect("should parse empty timeline");
        assert_eq!(clips.len(), 0);
    }

    #[test]
    fn test_import_single_clip() {
        let json = minimal_timeline_json("Test", &[("clip_A", 0.0, 25.0, 25.0)]);
        let importer = OtioImporter::new();
        let clips = importer.import(&json).expect("should parse single clip");
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].id, "clip_A");
    }

    #[test]
    fn test_import_multiple_clips() {
        let json = minimal_timeline_json(
            "Multi",
            &[
                ("clip_A", 0.0, 24.0, 24.0),
                ("clip_B", 24.0, 48.0, 24.0),
                ("clip_C", 72.0, 24.0, 24.0),
            ],
        );
        let importer = OtioImporter::new();
        let clips = importer
            .import(&json)
            .expect("should parse multi-clip timeline");
        assert_eq!(clips.len(), 3);
        assert_eq!(clips[0].id, "clip_A");
        assert_eq!(clips[2].id, "clip_C");
    }

    #[test]
    fn test_import_fps_detection() {
        let json = minimal_timeline_json("FPS", &[("c", 0.0, 2997.0, 29.97)]);
        let importer = OtioImporter::new();
        let clips = importer.import(&json).expect("should parse 29.97 fps");
        assert_eq!(clips[0].fps, FrameRate::Fps2997NDF);
    }

    #[test]
    fn test_import_track_type_video() {
        let json = minimal_timeline_json("VT", &[("v", 0.0, 25.0, 25.0)]);
        let importer = OtioImporter::new();
        let clips = importer.import(&json).expect("should parse video track");
        assert_eq!(clips[0].track, TrackType::Video);
    }

    #[test]
    fn test_import_audio_track() {
        let json = r#"{
            "OTIO_SCHEMA": "Timeline.1",
            "name": "Audio",
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": [{
                    "OTIO_SCHEMA": "Track.1",
                    "name": "A1",
                    "kind": "Audio",
                    "children": [{
                        "OTIO_SCHEMA": "Clip.1",
                        "name": "audio_clip",
                        "source_range": {
                            "start_time": {"value": 0.0, "rate": 48000.0},
                            "duration": {"value": 96000.0, "rate": 48000.0}
                        },
                        "metadata": {}
                    }]
                }]
            }
        }"#;
        let importer = OtioImporter::new();
        let clips = importer.import(json).expect("should parse audio track");
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].track, TrackType::Audio);
    }

    #[test]
    fn test_import_gap_ignored() {
        let json = r#"{
            "OTIO_SCHEMA": "Timeline.1",
            "name": "WithGap",
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": [{
                    "OTIO_SCHEMA": "Track.1",
                    "name": "V1",
                    "kind": "Video",
                    "children": [
                        {"OTIO_SCHEMA": "Clip.1", "name": "c1",
                         "source_range": {"start_time": {"value": 0.0, "rate": 25.0}, "duration": {"value": 25.0, "rate": 25.0}},
                         "metadata": {}},
                        {"OTIO_SCHEMA": "Gap.1", "name": "black",
                         "source_range": {"start_time": {"value": 0.0, "rate": 25.0}, "duration": {"value": 25.0, "rate": 25.0}},
                         "metadata": {}},
                        {"OTIO_SCHEMA": "Clip.1", "name": "c2",
                         "source_range": {"start_time": {"value": 25.0, "rate": 25.0}, "duration": {"value": 25.0, "rate": 25.0}},
                         "metadata": {}}
                    ]
                }]
            }
        }"#;
        let importer = OtioImporter::new();
        let clips = importer.import(json).expect("should parse with gap");
        // Gap must be ignored → only 2 Clips
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_import_timecode_conversion() {
        // 1 second at 25fps = frames 0..25
        let json = minimal_timeline_json("TC", &[("tc_clip", 25.0, 25.0, 25.0)]);
        let importer = OtioImporter::new();
        let clips = importer.import(&json).expect("should parse timecode clip");
        let c = &clips[0];
        assert_eq!(c.source_in, Timecode::new(0, 0, 1, 0));
        assert_eq!(c.source_out, Timecode::new(0, 0, 2, 0));
    }

    #[test]
    fn test_import_source_file_from_metadata() {
        let json = r#"{
            "OTIO_SCHEMA": "Timeline.1",
            "name": "Src",
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": [{
                    "OTIO_SCHEMA": "Track.1",
                    "name": "V1",
                    "kind": "Video",
                    "children": [{
                        "OTIO_SCHEMA": "Clip.1",
                        "name": "my_clip",
                        "source_range": {
                            "start_time": {"value": 0.0, "rate": 25.0},
                            "duration": {"value": 25.0, "rate": 25.0}
                        },
                        "metadata": {"source_file": "/media/raw/shot_001.mov"}
                    }]
                }]
            }
        }"#;
        let importer = OtioImporter::new();
        let clips = importer
            .import(json)
            .expect("should parse clip with source metadata");
        assert_eq!(
            clips[0].source_file.as_deref(),
            Some("/media/raw/shot_001.mov")
        );
    }

    #[test]
    fn test_import_invalid_json_returns_error() {
        let importer = OtioImporter::new();
        let result = importer.import("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_import_wrong_schema_returns_error() {
        let json = r#"{"OTIO_SCHEMA": "Clip.1", "name": "oops", "metadata": {}}"#;
        let importer = OtioImporter::new();
        let result = importer.import(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_import_multi_track_timeline() {
        let json = r#"{
            "OTIO_SCHEMA": "Timeline.1",
            "name": "MultiTrack",
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": [
                    {
                        "OTIO_SCHEMA": "Track.1",
                        "name": "V1",
                        "kind": "Video",
                        "children": [
                            {"OTIO_SCHEMA": "Clip.1", "name": "v_clip",
                             "source_range": {"start_time": {"value": 0.0, "rate": 25.0}, "duration": {"value": 25.0, "rate": 25.0}},
                             "metadata": {}}
                        ]
                    },
                    {
                        "OTIO_SCHEMA": "Track.1",
                        "name": "A1",
                        "kind": "Audio",
                        "children": [
                            {"OTIO_SCHEMA": "Clip.1", "name": "a_clip",
                             "source_range": {"start_time": {"value": 0.0, "rate": 48000.0}, "duration": {"value": 96000.0, "rate": 48000.0}},
                             "metadata": {}}
                        ]
                    }
                ]
            }
        }"#;
        let importer = OtioImporter::new();
        let clips = importer.import(json).expect("should parse multi-track");
        assert_eq!(clips.len(), 2);
        let video_clips: Vec<_> = clips
            .iter()
            .filter(|c| c.track == TrackType::Video)
            .collect();
        let audio_clips: Vec<_> = clips
            .iter()
            .filter(|c| c.track == TrackType::Audio)
            .collect();
        assert_eq!(video_clips.len(), 1);
        assert_eq!(audio_clips.len(), 1);
    }

    #[test]
    fn test_import_default_fps_fallback() {
        let json = r#"{
            "OTIO_SCHEMA": "Timeline.1",
            "name": "NoRange",
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": [{
                    "OTIO_SCHEMA": "Track.1",
                    "name": "V1",
                    "kind": "Video",
                    "children": [{
                        "OTIO_SCHEMA": "Clip.1",
                        "name": "no_range_clip",
                        "metadata": {}
                    }]
                }]
            }
        }"#;
        let importer = OtioImporter::with_default_fps(30.0);
        let clips = importer
            .import(json)
            .expect("should parse clip without source_range");
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].fps, FrameRate::Fps30);
    }

    #[test]
    fn test_rational_time_to_seconds() {
        let rt = OtioRationalTime {
            value: 50.0,
            rate: 25.0,
        };
        assert!((rt.to_seconds() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rational_time_zero_rate() {
        let rt = OtioRationalTime {
            value: 10.0,
            rate: 0.0,
        };
        assert_eq!(rt.to_seconds(), 0.0);
    }
}
