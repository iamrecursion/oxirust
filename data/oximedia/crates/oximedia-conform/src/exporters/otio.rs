//! OpenTimelineIO (OTIO) exporter.
//!
//! Serializes a list of `ClipReference`s into valid OpenTimelineIO JSON.
//! The output follows the OTIO spec subset:
//!   `Timeline.1` → `Stack.1` (tracks) → `Track.1` (per track kind) → `Clip.1` / `Gap.1`

use crate::types::{ClipReference, FrameRate, Timecode, TrackType};
use serde_json::{json, Value};

// ─── Public exporter ──────────────────────────────────────────────────────────

/// Exports clip references as an OpenTimelineIO JSON string.
pub struct OtioExporter {
    /// Timeline name embedded in the output.
    timeline_name: String,
    /// Pretty-print the JSON output.
    pretty_print: bool,
}

impl OtioExporter {
    /// Create a new OTIO exporter.
    #[must_use]
    pub fn new(timeline_name: impl Into<String>) -> Self {
        Self {
            timeline_name: timeline_name.into(),
            pretty_print: false,
        }
    }

    /// Enable/disable pretty-printed JSON (indented).
    #[must_use]
    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty_print = pretty;
        self
    }

    /// Export the given clips as an OTIO JSON string.
    ///
    /// Clips are grouped by their `TrackType`; each unique track type produces
    /// one `Track.1` object inside the top-level `Stack.1`.
    #[must_use]
    pub fn export(&self, clips: &[ClipReference]) -> String {
        let tracks_value = self.build_tracks(clips);

        let timeline = json!({
            "OTIO_SCHEMA": "Timeline.1",
            "name": self.timeline_name,
            "metadata": {},
            "tracks": tracks_value,
        });

        if self.pretty_print {
            serde_json::to_string_pretty(&timeline).unwrap_or_else(|_| "{}".to_string())
        } else {
            serde_json::to_string(&timeline).unwrap_or_else(|_| "{}".to_string())
        }
    }

    // ── Internal builders ────────────────────────────────────────────────────

    fn build_tracks(&self, clips: &[ClipReference]) -> Value {
        // Collect unique track types in deterministic order
        let mut track_kinds: Vec<TrackType> = Vec::new();
        for clip in clips {
            let kind = match clip.track {
                TrackType::Video | TrackType::AudioVideo => TrackType::Video,
                TrackType::Audio => TrackType::Audio,
            };
            if !track_kinds.contains(&kind) {
                track_kinds.push(kind);
            }
        }
        // If a clip is AudioVideo, add both tracks
        for clip in clips {
            if clip.track == TrackType::AudioVideo && !track_kinds.contains(&TrackType::Audio) {
                track_kinds.push(TrackType::Audio);
            }
        }

        let track_objects: Vec<Value> = track_kinds
            .iter()
            .enumerate()
            .map(|(idx, &kind)| {
                let kind_str = match kind {
                    TrackType::Video | TrackType::AudioVideo => "Video",
                    TrackType::Audio => "Audio",
                };
                let track_name = format!("{kind_str}{}", idx + 1);

                // Collect clips belonging to this track
                let track_clips: Vec<Value> = clips
                    .iter()
                    .filter(|c| {
                        matches!(
                            (c.track, kind),
                            (TrackType::Video, TrackType::Video)
                                | (TrackType::AudioVideo, TrackType::Video)
                                | (TrackType::Audio, TrackType::Audio)
                                | (TrackType::AudioVideo, TrackType::Audio)
                        )
                    })
                    .map(|c| self.build_clip(c))
                    .collect();

                json!({
                    "OTIO_SCHEMA": "Track.1",
                    "name": track_name,
                    "kind": kind_str,
                    "metadata": {},
                    "children": track_clips,
                })
            })
            .collect();

        json!({
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "metadata": {},
            "children": track_objects,
        })
    }

    fn build_clip(&self, clip: &ClipReference) -> Value {
        let rate = fps_to_rate(clip.fps);
        let source_range = build_time_range(&clip.source_in, &clip.source_out, rate);

        let mut metadata: serde_json::Map<String, Value> = serde_json::Map::new();
        if let Some(ref sf) = clip.source_file {
            metadata.insert("source_file".to_string(), json!(sf));
        }
        for (k, v) in &clip.metadata {
            // Skip internal OTIO roundtrip keys
            if k != "otio_schema" && k != "otio_source_file" {
                metadata.insert(k.clone(), json!(v));
            }
        }

        json!({
            "OTIO_SCHEMA": "Clip.1",
            "name": clip.id,
            "metadata": metadata,
            "source_range": source_range,
        })
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn fps_to_rate(fps: FrameRate) -> f64 {
    fps.as_f64()
}

fn timecode_to_frame_value(tc: &Timecode, rate: f64) -> f64 {
    let rate_u = rate.round().max(1.0) as u64;
    let total = u64::from(tc.hours) * 3600 * rate_u
        + u64::from(tc.minutes) * 60 * rate_u
        + u64::from(tc.seconds) * rate_u
        + u64::from(tc.frames);
    total as f64
}

fn build_rational_time(tc: &Timecode, rate: f64) -> Value {
    json!({
        "value": timecode_to_frame_value(tc, rate),
        "rate": rate,
    })
}

fn build_time_range(start_tc: &Timecode, end_tc: &Timecode, rate: f64) -> Value {
    let start_val = timecode_to_frame_value(start_tc, rate);
    let end_val = timecode_to_frame_value(end_tc, rate);
    let duration_val = (end_val - start_val).max(0.0);

    json!({
        "start_time": build_rational_time(start_tc, rate),
        "duration": {"value": duration_val, "rate": rate},
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::importers::otio::OtioImporter;
    use std::collections::HashMap;

    fn make_clip(id: &str, source_file: &str, fps: FrameRate, track: TrackType) -> ClipReference {
        ClipReference {
            id: id.to_string(),
            source_file: Some(source_file.to_string()),
            source_in: Timecode::new(0, 0, 0, 0),
            source_out: Timecode::new(0, 0, 2, 0),
            record_in: Timecode::new(0, 0, 0, 0),
            record_out: Timecode::new(0, 0, 2, 0),
            track,
            fps,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_export_produces_valid_json() {
        let clips = vec![make_clip(
            "c1",
            "test.mov",
            FrameRate::Fps25,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("TestTimeline");
        let json_str = exporter.export(&clips);
        assert!(serde_json::from_str::<serde_json::Value>(&json_str).is_ok());
    }

    #[test]
    fn test_export_timeline_schema() {
        let clips = vec![make_clip(
            "c1",
            "test.mov",
            FrameRate::Fps25,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("MyEdit");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        assert_eq!(v["OTIO_SCHEMA"], "Timeline.1");
        assert_eq!(v["name"], "MyEdit");
    }

    #[test]
    fn test_export_stack_structure() {
        let clips = vec![make_clip(
            "c1",
            "test.mov",
            FrameRate::Fps25,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("T");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        assert_eq!(v["tracks"]["OTIO_SCHEMA"], "Stack.1");
    }

    #[test]
    fn test_export_clip_count() {
        let clips = vec![
            make_clip("c1", "a.mov", FrameRate::Fps25, TrackType::Video),
            make_clip("c2", "b.mov", FrameRate::Fps25, TrackType::Video),
            make_clip("c3", "c.mov", FrameRate::Fps25, TrackType::Video),
        ];
        let exporter = OtioExporter::new("T");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        let children = v["tracks"]["children"][0]["children"]
            .as_array()
            .expect("array");
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_export_empty_clips() {
        let exporter = OtioExporter::new("Empty");
        let json_str = exporter.export(&[]);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        assert_eq!(v["OTIO_SCHEMA"], "Timeline.1");
        // Stack with no track children
        let children = v["tracks"]["children"].as_array().expect("array");
        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_export_audio_track_kind() {
        let clips = vec![make_clip(
            "a1",
            "audio.wav",
            FrameRate::Fps25,
            TrackType::Audio,
        )];
        let exporter = OtioExporter::new("A");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        let track = &v["tracks"]["children"][0];
        assert_eq!(track["kind"], "Audio");
    }

    #[test]
    fn test_export_clip_schema() {
        let clips = vec![make_clip(
            "shot_001",
            "shot.mov",
            FrameRate::Fps24,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("T");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        let clip = &v["tracks"]["children"][0]["children"][0];
        assert_eq!(clip["OTIO_SCHEMA"], "Clip.1");
        assert_eq!(clip["name"], "shot_001");
    }

    #[test]
    fn test_export_source_file_in_metadata() {
        let clips = vec![make_clip(
            "c1",
            "/media/raw.mov",
            FrameRate::Fps25,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("T");
        let json_str = exporter.export(&clips);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        let clip = &v["tracks"]["children"][0]["children"][0];
        assert_eq!(clip["metadata"]["source_file"], "/media/raw.mov");
    }

    #[test]
    fn test_export_rational_time_values() {
        let clip = ClipReference {
            id: "tc_clip".to_string(),
            source_file: None,
            source_in: Timecode::new(0, 0, 1, 0), // 1 second = 25 frames
            source_out: Timecode::new(0, 0, 2, 0), // 2 seconds = 50 frames
            record_in: Timecode::new(0, 0, 0, 0),
            record_out: Timecode::new(0, 0, 1, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: HashMap::new(),
        };
        let exporter = OtioExporter::new("T");
        let json_str = exporter.export(&[clip]);
        let v: serde_json::Value = serde_json::from_str(&json_str).expect("valid json");
        let sr = &v["tracks"]["children"][0]["children"][0]["source_range"];
        assert_eq!(sr["start_time"]["value"], 25.0);
        assert_eq!(sr["duration"]["value"], 25.0);
        assert_eq!(sr["start_time"]["rate"], 25.0);
    }

    /// Round-trip: export then import and verify clip identity.
    #[test]
    fn test_roundtrip_clip_identity() {
        let original_clips = vec![
            make_clip("shot_A", "/media/A.mov", FrameRate::Fps25, TrackType::Video),
            make_clip("shot_B", "/media/B.mov", FrameRate::Fps25, TrackType::Video),
        ];
        let exporter = OtioExporter::new("Roundtrip");
        let json_str = exporter.export(&original_clips);

        let importer = OtioImporter::new();
        let reimported = importer.import(&json_str).expect("should reimport");

        assert_eq!(reimported.len(), original_clips.len());
        for (orig, reimp) in original_clips.iter().zip(reimported.iter()) {
            assert_eq!(orig.id, reimp.id);
        }
    }

    /// Round-trip: source timecodes survive export → import.
    #[test]
    fn test_roundtrip_timecode_preservation() {
        let clip = ClipReference {
            id: "tc_rt".to_string(),
            source_file: Some("tc.mov".to_string()),
            source_in: Timecode::new(0, 0, 2, 0),
            source_out: Timecode::new(0, 0, 5, 0),
            record_in: Timecode::new(0, 0, 0, 0),
            record_out: Timecode::new(0, 0, 3, 0),
            track: TrackType::Video,
            fps: FrameRate::Fps25,
            metadata: HashMap::new(),
        };
        let exporter = OtioExporter::new("TCRoundtrip");
        let json_str = exporter.export(&[clip.clone()]);

        let importer = OtioImporter::new();
        let reimported = importer
            .import(&json_str)
            .expect("should reimport timecode clip");
        assert_eq!(reimported[0].source_in, clip.source_in);
        assert_eq!(reimported[0].source_out, clip.source_out);
    }

    /// Round-trip: fps survives export → import.
    #[test]
    fn test_roundtrip_fps_preservation() {
        let clip = make_clip("fps_clip", "file.mov", FrameRate::Fps24, TrackType::Video);
        let exporter = OtioExporter::new("FPSRoundtrip");
        let json_str = exporter.export(&[clip]);

        let importer = OtioImporter::new();
        let reimported = importer
            .import(&json_str)
            .expect("should reimport fps clip");
        assert_eq!(reimported[0].fps, FrameRate::Fps24);
    }

    #[test]
    fn test_pretty_print_produces_newlines() {
        let clips = vec![make_clip(
            "c1",
            "test.mov",
            FrameRate::Fps25,
            TrackType::Video,
        )];
        let exporter = OtioExporter::new("T").pretty(true);
        let json_str = exporter.export(&clips);
        assert!(json_str.contains('\n'));
    }

    /// Round-trip multi-track test.
    #[test]
    fn test_roundtrip_multi_track() {
        let clips = vec![
            make_clip("v1", "video.mov", FrameRate::Fps25, TrackType::Video),
            make_clip("a1", "audio.wav", FrameRate::Fps25, TrackType::Audio),
        ];
        let exporter = OtioExporter::new("MultiTrack");
        let json_str = exporter.export(&clips);

        let importer = OtioImporter::new();
        let reimported = importer
            .import(&json_str)
            .expect("should reimport multi-track");
        assert_eq!(reimported.len(), 2);
        assert!(reimported.iter().any(|c| c.track == TrackType::Video));
        assert!(reimported.iter().any(|c| c.track == TrackType::Audio));
    }
}
