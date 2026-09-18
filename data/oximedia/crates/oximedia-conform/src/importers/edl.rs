//! EDL importer for conforming.
//!
//! Supports CMX 3600, CMX 3400, CMX 340, and File128 EDL variants.

use crate::error::{ConformError, ConformResult};
use crate::importers::TimelineImporter;
use crate::types::{ClipReference, FrameRate, Timecode, TrackType};
use oximedia_edl::event::EdlEvent;
use oximedia_edl::Edl;
use std::path::Path;

/// EDL variant type for differentiating between standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdlVariant {
    /// CMX 3600 — the most common EDL format (default).
    #[default]
    Cmx3600,
    /// CMX 3400 — older 16-channel audio variant.
    Cmx3400,
    /// CMX 340 — limited-field older format (clip names up to 8 chars).
    Cmx340,
    /// File128 — extended format supporting 128-character reel/clip names.
    File128,
}

impl EdlVariant {
    /// Detect the EDL variant from the file content.
    ///
    /// Heuristics:
    /// - A `FCM` header line containing `NON-DROP` or `DROP FRAME` alone indicates CMX 3600.
    /// - A `FORMAT:` header line with `CMX 340` indicates CMX 340.
    /// - A `FORMAT:` header line with `FILE128` indicates File128.
    /// - A `FORMAT:` header line with `CMX 3400` indicates CMX 3400.
    #[must_use]
    pub fn detect(content: &str) -> Self {
        for line in content.lines().take(20) {
            let upper = line.trim().to_uppercase();
            if upper.starts_with("FORMAT:") {
                if upper.contains("FILE128") {
                    return Self::File128;
                }
                if upper.contains("CMX 3400") {
                    return Self::Cmx3400;
                }
                if upper.contains("CMX 340") {
                    return Self::Cmx340;
                }
                if upper.contains("CMX 3600") {
                    return Self::Cmx3600;
                }
            }
        }
        Self::Cmx3600
    }

    /// Maximum reel/clip name length for this variant.
    #[must_use]
    pub const fn max_name_length(&self) -> usize {
        match self {
            Self::Cmx340 => 8,
            Self::Cmx3400 | Self::Cmx3600 => 32,
            Self::File128 => 128,
        }
    }

    /// Whether this variant supports extended audio channel routing.
    #[must_use]
    pub const fn supports_extended_audio(&self) -> bool {
        matches!(self, Self::Cmx3400)
    }
}

/// EDL importer with EDL variant awareness.
pub struct EdlImporter {
    /// The EDL variant to use (or auto-detected from content).
    variant: Option<EdlVariant>,
}

impl EdlImporter {
    /// Create a new EDL importer with auto-detection of EDL variant.
    #[must_use]
    pub const fn new() -> Self {
        Self { variant: None }
    }

    /// Create a new EDL importer for a specific EDL variant.
    #[must_use]
    pub const fn with_variant(variant: EdlVariant) -> Self {
        Self {
            variant: Some(variant),
        }
    }

    /// Get the effective EDL variant, detecting from content if not set.
    fn effective_variant(content: &str, hint: Option<EdlVariant>) -> EdlVariant {
        hint.unwrap_or_else(|| EdlVariant::detect(content))
    }

    /// Truncate a reel or clip name according to the variant's length limit.
    #[must_use]
    fn truncate_name(name: &str, variant: EdlVariant) -> String {
        let max = variant.max_name_length();
        if name.len() <= max {
            name.to_string()
        } else {
            name[..max].to_string()
        }
    }

    /// Convert EDL frame rate to conform frame rate.
    fn convert_frame_rate(edl_fps: oximedia_edl::timecode::EdlFrameRate) -> FrameRate {
        match edl_fps {
            oximedia_edl::timecode::EdlFrameRate::Fps24 => FrameRate::Fps24,
            oximedia_edl::timecode::EdlFrameRate::Fps25 => FrameRate::Fps25,
            oximedia_edl::timecode::EdlFrameRate::Fps2997DF => FrameRate::Fps2997DF,
            oximedia_edl::timecode::EdlFrameRate::Fps2997NDF => FrameRate::Fps2997NDF,
            oximedia_edl::timecode::EdlFrameRate::Fps30 => FrameRate::Fps30,
            _ => FrameRate::Fps25, // Default fallback
        }
    }

    /// Convert EDL timecode to conform timecode.
    fn convert_timecode(edl_tc: &oximedia_edl::timecode::EdlTimecode) -> Timecode {
        Timecode::new(
            edl_tc.hours(),
            edl_tc.minutes(),
            edl_tc.seconds(),
            edl_tc.frames(),
        )
    }

    /// Convert EDL track type to conform track type.
    fn convert_track_type(edl_track: &oximedia_edl::event::TrackType) -> TrackType {
        if edl_track.has_video() && edl_track.has_audio() {
            TrackType::AudioVideo
        } else if edl_track.has_video() {
            TrackType::Video
        } else {
            TrackType::Audio
        }
    }

    /// Convert an EDL event to a clip reference.
    fn event_to_clip(event: &EdlEvent, fps: FrameRate) -> ClipReference {
        let mut metadata: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        metadata.insert("reel".to_string(), event.reel.clone());
        metadata.insert("event_number".to_string(), event.number.to_string());
        metadata.insert("edit_type".to_string(), event.edit_type.to_string());

        if let Some(clip_name) = &event.clip_name {
            metadata.insert("clip_name".to_string(), clip_name.clone());
        }

        ClipReference {
            id: format!("event_{}", event.number),
            source_file: event.clip_name.clone(),
            source_in: Self::convert_timecode(&event.source_in),
            source_out: Self::convert_timecode(&event.source_out),
            record_in: Self::convert_timecode(&event.record_in),
            record_out: Self::convert_timecode(&event.record_out),
            track: Self::convert_track_type(&event.track),
            fps,
            metadata,
        }
    }

    /// Import from EDL object.
    pub fn import_from_edl(edl: &Edl) -> ConformResult<Vec<ClipReference>> {
        let fps = Self::convert_frame_rate(edl.frame_rate);
        let clips: Vec<ClipReference> = edl
            .events
            .iter()
            .map(|event| Self::event_to_clip(event, fps))
            .collect();
        Ok(clips)
    }
}

impl Default for EdlImporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineImporter for EdlImporter {
    fn import<P: AsRef<Path>>(&self, path: P) -> ConformResult<Vec<ClipReference>> {
        // Read content first so we can detect the variant before parsing
        let content = std::fs::read_to_string(path.as_ref()).map_err(ConformError::Io)?;
        let variant = Self::effective_variant(&content, self.variant);

        let edl = Edl::from_file(path.as_ref()).map_err(|e| ConformError::Edl(e.to_string()))?;
        let mut clips = Self::import_from_edl(&edl)?;

        // Post-process reel/clip names according to the variant's length limit
        for clip in &mut clips {
            if let Some(reel) = clip.metadata.get("reel").cloned() {
                clip.metadata
                    .insert("reel".to_string(), Self::truncate_name(&reel, variant));
            }
            if let Some(clip_name) = clip.metadata.get("clip_name").cloned() {
                clip.metadata.insert(
                    "clip_name".to_string(),
                    Self::truncate_name(&clip_name, variant),
                );
            }
            // Store the variant name as metadata
            let variant_name = match variant {
                EdlVariant::Cmx3600 => "CMX3600",
                EdlVariant::Cmx3400 => "CMX3400",
                EdlVariant::Cmx340 => "CMX340",
                EdlVariant::File128 => "File128",
            };
            clip.metadata
                .insert("edl_variant".to_string(), variant_name.to_string());
        }

        Ok(clips)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_edl::event::{EditType, TrackType as EdlTrackType};
    use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};

    #[test]
    fn test_convert_frame_rate() {
        assert_eq!(
            EdlImporter::convert_frame_rate(EdlFrameRate::Fps25),
            FrameRate::Fps25
        );
        assert_eq!(
            EdlImporter::convert_frame_rate(EdlFrameRate::Fps2997DF),
            FrameRate::Fps2997DF
        );
    }

    #[test]
    fn test_convert_timecode() {
        let edl_tc =
            EdlTimecode::new(1, 23, 45, 12, EdlFrameRate::Fps25).expect("edl_tc should be valid");
        let tc = EdlImporter::convert_timecode(&edl_tc);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_event_to_clip() {
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc1 should be valid");
        let tc2 = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc2 should be valid");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let clip = EdlImporter::event_to_clip(&event, FrameRate::Fps25);
        assert_eq!(clip.id, "event_1");
        assert_eq!(clip.track, TrackType::Video);
    }

    #[test]
    fn test_edl_variant_detect_default() {
        let content =
            "TITLE: Test\n001  REEL1  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let variant = EdlVariant::detect(content);
        assert_eq!(variant, EdlVariant::Cmx3600);
    }

    #[test]
    fn test_edl_variant_detect_file128() {
        let content = "TITLE: Test\nFORMAT: FILE128\n001  REEL1  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let variant = EdlVariant::detect(content);
        assert_eq!(variant, EdlVariant::File128);
    }

    #[test]
    fn test_edl_variant_detect_cmx340() {
        let content = "TITLE: Test\nFORMAT: CMX 340\n001  REEL1  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let variant = EdlVariant::detect(content);
        assert_eq!(variant, EdlVariant::Cmx340);
    }

    #[test]
    fn test_edl_variant_detect_cmx3400() {
        let content = "TITLE: Test\nFORMAT: CMX 3400\n001  REEL1  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let variant = EdlVariant::detect(content);
        assert_eq!(variant, EdlVariant::Cmx3400);
    }

    #[test]
    fn test_edl_variant_name_length() {
        assert_eq!(EdlVariant::Cmx340.max_name_length(), 8);
        assert_eq!(EdlVariant::Cmx3600.max_name_length(), 32);
        assert_eq!(EdlVariant::File128.max_name_length(), 128);
    }

    #[test]
    fn test_edl_truncate_name_cmx340() {
        let long_name = "very_long_clip_name";
        let truncated = EdlImporter::truncate_name(long_name, EdlVariant::Cmx340);
        assert_eq!(truncated.len(), 8);
        assert_eq!(truncated, "very_lon");
    }

    #[test]
    fn test_edl_truncate_name_file128_no_truncation() {
        let name = "A".repeat(100);
        let result = EdlImporter::truncate_name(&name, EdlVariant::File128);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_edl_importer_with_variant() {
        let importer = EdlImporter::with_variant(EdlVariant::Cmx340);
        assert_eq!(importer.variant, Some(EdlVariant::Cmx340));
    }

    #[test]
    fn test_edl_variant_extended_audio() {
        assert!(EdlVariant::Cmx3400.supports_extended_audio());
        assert!(!EdlVariant::Cmx3600.supports_extended_audio());
        assert!(!EdlVariant::File128.supports_extended_audio());
    }

    /// CMX 340 fixture round-trip: build an EDL, detect it as CMX 340 when the
    /// header is present, verify that clip names are truncated to 8 characters
    /// and reel names to 8 characters per the variant's column-width rules.
    #[test]
    fn test_cmx340_parse_fixture_round_trip() {
        use crate::types::Timecode as ConformTimecode;
        use oximedia_edl::{Edl, EdlFormat, EdlGenerator};

        // Construct an EDL with a long reel name and a long clip name to verify
        // that the CMX 340 name-truncation logic fires correctly.
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title("CMX340RoundTrip".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let src_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let src_out = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let rec_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let rec_out = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc must be valid");

        let mut event = EdlEvent::new(
            1,
            // 16-char reel name — CMX 340 allows max 8 chars → expect truncation
            "LONGREELAB123456".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            src_in,
            src_out,
            rec_in,
            rec_out,
        );
        // 20-char clip name — CMX 340 max 8 → expect truncation
        event.clip_name = Some("very_long_clip_name!".to_string());
        edl.add_event(event).expect("add_event must succeed");

        // Generate EDL string and prepend a CMX 340 FORMAT header so
        // detection returns CMX340.
        let generator = EdlGenerator::new();
        let raw_edl = generator.generate(&edl).expect("generate must succeed");
        let edl_string = format!("FORMAT: CMX 340\n{raw_edl}");

        // Verify variant detection
        let variant = EdlVariant::detect(&edl_string);
        assert_eq!(
            variant,
            EdlVariant::Cmx340,
            "CMX 340 FORMAT header must be detected correctly"
        );

        // Re-parse the EDL and import clips
        let parsed = oximedia_edl::parse_edl(&raw_edl)
            .expect("parse_edl must succeed for the base EDL content");
        let clips = EdlImporter::import_from_edl(&parsed).expect("import_from_edl must succeed");
        assert_eq!(clips.len(), 1, "fixture must produce exactly one clip");

        // Apply CMX 340 name truncation the same way the importer does
        let clip = &clips[0];
        let truncated_reel = EdlImporter::truncate_name(
            clip.metadata.get("reel").map(String::as_str).unwrap_or(""),
            EdlVariant::Cmx340,
        );
        let truncated_clip = EdlImporter::truncate_name(
            clip.metadata
                .get("clip_name")
                .map(String::as_str)
                .unwrap_or(""),
            EdlVariant::Cmx340,
        );

        // Reel and clip names must be ≤ 8 chars after truncation
        assert!(
            truncated_reel.len() <= 8,
            "CMX 340 reel name must be ≤ 8 chars, got {}: '{truncated_reel}'",
            truncated_reel.len()
        );
        assert!(
            truncated_clip.len() <= 8,
            "CMX 340 clip name must be ≤ 8 chars, got {}: '{truncated_clip}'",
            truncated_clip.len()
        );

        // Timecodes must round-trip correctly
        assert_eq!(
            clip.source_in,
            ConformTimecode::new(1, 0, 0, 0),
            "source_in must round-trip"
        );
        assert_eq!(
            clip.source_out,
            ConformTimecode::new(1, 0, 5, 0),
            "source_out must round-trip"
        );
    }

    /// File128 fixture: the variant is detected correctly from a FORMAT: FILE128
    /// header and the `max_name_length` is 128.  Names within the 128-char limit
    /// must not be truncated by `truncate_name`.  A standard-width reel/clip name
    /// is used so that the underlying oximedia_edl generator + parser round-trip
    /// succeeds (the generator targets CMX 3600 column widths internally).
    #[test]
    fn test_file128_parse_fixture() {
        use crate::types::Timecode as ConformTimecode;
        use oximedia_edl::{Edl, EdlFormat, EdlGenerator};

        // ── Part 1: variant detection ─────────────────────────────────────────
        let header_content = "FORMAT: FILE128\nTITLE: File128Test\n";
        let variant = EdlVariant::detect(header_content);
        assert_eq!(
            variant,
            EdlVariant::File128,
            "FILE128 FORMAT header must be detected as File128"
        );
        assert_eq!(
            EdlVariant::File128.max_name_length(),
            128,
            "File128 max_name_length must be 128"
        );

        // ── Part 2: truncation leaves ≤128-char names untouched ───────────────
        // A 100-char name is within the 128-char limit and must not be truncated.
        let name_100 = "A".repeat(100);
        let after_trunc = EdlImporter::truncate_name(&name_100, EdlVariant::File128);
        assert_eq!(
            after_trunc.len(),
            100,
            "File128: 100-char name must not be truncated"
        );

        // A 128-char name is at the boundary and must not be truncated.
        let name_128 = "B".repeat(128);
        let after_trunc_128 = EdlImporter::truncate_name(&name_128, EdlVariant::File128);
        assert_eq!(
            after_trunc_128.len(),
            128,
            "File128: 128-char name must not be truncated"
        );

        // A 200-char name exceeds the limit and must be truncated to 128.
        let name_200 = "C".repeat(200);
        let after_trunc_200 = EdlImporter::truncate_name(&name_200, EdlVariant::File128);
        assert_eq!(
            after_trunc_200.len(),
            128,
            "File128: 200-char name must be truncated to 128"
        );

        // ── Part 3: EDL round-trip with a standard-length reel name ──────────
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title("File128RoundTrip".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let src_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let src_out = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let rec_in = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc must be valid");
        let rec_out = EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc must be valid");

        // Use a standard short reel name so the CMX 3600 generator + parser
        // round-trip succeeds.  File128 extends the *allowable* length but
        // the generator and parser target the CMX 3600 fixed-field columns.
        let mut event = EdlEvent::new(
            1,
            "FILE128R".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            src_in,
            src_out,
            rec_in,
            rec_out,
        );
        event.clip_name = Some("shot_f128.mov".to_string());
        edl.add_event(event).expect("add_event must succeed");

        let generator = EdlGenerator::new();
        let raw_edl = generator.generate(&edl).expect("generate must succeed");

        // Parse and import clips
        let parsed = oximedia_edl::parse_edl(&raw_edl).expect("parse_edl must succeed");
        let clips = EdlImporter::import_from_edl(&parsed).expect("import_from_edl must succeed");
        assert_eq!(clips.len(), 1, "fixture must produce exactly one clip");

        let clip = &clips[0];

        // Timecodes must round-trip correctly
        assert_eq!(
            clip.source_in,
            ConformTimecode::new(1, 0, 0, 0),
            "source_in must round-trip"
        );
        assert_eq!(
            clip.source_out,
            ConformTimecode::new(1, 0, 10, 0),
            "source_out must round-trip"
        );

        // The reel name must survive without truncation (it's within 128 chars)
        let reel = clip.metadata.get("reel").map(String::as_str).unwrap_or("");
        assert!(
            reel.len() <= EdlVariant::File128.max_name_length(),
            "reel name must be within File128 max_name_length"
        );
    }

    /// Round-trip test: build an EDL in memory → generate EDL string → parse
    /// the string back → import clips → verify clip count and timecodes.
    #[test]
    fn test_edl_import_export_roundtrip() {
        use crate::types::Timecode as ConformTimecode;
        use oximedia_edl::{Edl, EdlFormat, EdlGenerator};

        // ── Step 1: construct an EDL with two cuts ─────────────────────────
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title("RoundTripTest".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);

        // Event 1: 01:00:00:00 → 01:00:05:00
        let src_in_1 =
            EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let src_out_1 =
            EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let rec_in_1 =
            EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let rec_out_1 =
            EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");

        let mut event1 = EdlEvent::new(
            1,
            "REEL_A".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            src_in_1,
            src_out_1,
            rec_in_1,
            rec_out_1,
        );
        event1.clip_name = Some("shot_001.mov".to_string());
        edl.add_event(event1).expect("add_event must succeed");

        // Event 2: 01:00:10:00 → 01:00:20:12
        let src_in_2 =
            EdlTimecode::new(1, 0, 10, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let src_out_2 =
            EdlTimecode::new(1, 0, 20, 12, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let rec_in_2 =
            EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("tc creation must succeed");
        let rec_out_2 =
            EdlTimecode::new(1, 0, 15, 12, EdlFrameRate::Fps25).expect("tc creation must succeed");

        let mut event2 = EdlEvent::new(
            2,
            "REEL_B".to_string(),
            EdlTrackType::Video,
            EditType::Cut,
            src_in_2,
            src_out_2,
            rec_in_2,
            rec_out_2,
        );
        event2.clip_name = Some("shot_002.mov".to_string());
        edl.add_event(event2).expect("add_event must succeed");

        // ── Step 2: export to EDL string ───────────────────────────────────
        let generator = EdlGenerator::new();
        let edl_string = generator.generate(&edl).expect("generate must succeed");

        assert!(
            edl_string.contains("RoundTripTest"),
            "generated EDL must contain the title"
        );

        // ── Step 3: re-parse the generated string ──────────────────────────
        let parsed = oximedia_edl::parse_edl(&edl_string)
            .expect("re-parsing the generated EDL must succeed");

        // ── Step 4: import clips via EdlImporter ───────────────────────────
        let clips = EdlImporter::import_from_edl(&parsed).expect("import_from_edl must succeed");

        // ── Step 5: verify clip count ──────────────────────────────────────
        assert_eq!(
            clips.len(),
            2,
            "round-trip must preserve the two-clip timeline, got {} clips",
            clips.len()
        );

        // ── Step 6: verify timecodes of each clip ──────────────────────────
        // Find clip for event 1 (id = "event_1")
        let clip1 = clips
            .iter()
            .find(|c| c.id == "event_1")
            .expect("event_1 must be present after round-trip");

        assert_eq!(
            clip1.source_in,
            ConformTimecode::new(1, 0, 0, 0),
            "event_1 source_in mismatch"
        );
        assert_eq!(
            clip1.source_out,
            ConformTimecode::new(1, 0, 5, 0),
            "event_1 source_out mismatch"
        );

        // Find clip for event 2 (id = "event_2")
        let clip2 = clips
            .iter()
            .find(|c| c.id == "event_2")
            .expect("event_2 must be present after round-trip");

        assert_eq!(
            clip2.source_in,
            ConformTimecode::new(1, 0, 10, 0),
            "event_2 source_in mismatch"
        );
        assert_eq!(
            clip2.source_out,
            ConformTimecode::new(1, 0, 20, 12),
            "event_2 source_out mismatch"
        );
    }
}
