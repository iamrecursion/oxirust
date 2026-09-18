// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Subtitle track packaging for adaptive streaming.
//!
//! Supports conversion between common subtitle formats (SRT, `WebVTT`, TTML,
//! SMPTE 2052) and provides segmentation of subtitle tracks for HLS / DASH.

// ---------------------------------------------------------------------------
// SubtitleFormat
// ---------------------------------------------------------------------------

/// Supported subtitle container / encoding formats.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SubtitleFormat {
    /// W3C `WebVTT`.
    WebVtt,
    /// TTML (Timed Text Markup Language).
    Ttml,
    /// `SubRip` text format.
    Srt,
    /// SMPTE-TT / SMPTE 2052.
    Smpte2052,
}

// ---------------------------------------------------------------------------
// SubtitleCue
// ---------------------------------------------------------------------------

/// A single subtitle cue (one caption entry).
#[derive(Debug, Clone)]
pub struct SubtitleCue {
    /// Start time in milliseconds from the beginning of the presentation.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Subtitle text (may contain newlines for multi-line captions).
    pub text: String,
    /// Optional horizontal position as a percentage (0.0–100.0).
    pub position_x: Option<f64>,
    /// Optional vertical position as a percentage (0.0–100.0).
    pub position_y: Option<f64>,
}

impl SubtitleCue {
    /// Create a simple cue without position overrides.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            text: text.into(),
            position_x: None,
            position_y: None,
        }
    }

    /// Duration of this cue in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Render this cue in `WebVTT` format.
    ///
    /// Example output:
    /// ```text
    /// 00:00:01.000 --> 00:00:04.000
    /// Hello, world!
    /// ```
    #[must_use]
    pub fn to_webvtt(&self) -> String {
        let mut cue = format!(
            "{} --> {}",
            ms_to_webvtt_timestamp(self.start_ms),
            ms_to_webvtt_timestamp(self.end_ms),
        );

        // Append optional position settings
        if let (Some(x), Some(y)) = (self.position_x, self.position_y) {
            cue.push_str(&format!(" position:{x:.0}% line:{y:.0}%"));
        }

        cue.push('\n');
        cue.push_str(&self.text);
        cue
    }

    /// Render this cue in SRT format.
    ///
    /// `index` is 1-based sequence number required by the SRT specification.
    #[must_use]
    pub fn to_srt(&self, index: usize) -> String {
        format!(
            "{}\n{} --> {}\n{}\n",
            index,
            ms_to_srt_timestamp(self.start_ms),
            ms_to_srt_timestamp(self.end_ms),
            self.text,
        )
    }
}

// ---------------------------------------------------------------------------
// SubtitleTrack
// ---------------------------------------------------------------------------

/// A collection of subtitle cues for a single language / track.
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// BCP-47 language tag (e.g. `"en"`, `"fr-CA"`).
    pub language: String,
    /// Container format for this track.
    pub format: SubtitleFormat,
    /// Ordered list of cues (should be sorted by `start_ms`).
    pub cues: Vec<SubtitleCue>,
}

impl SubtitleTrack {
    /// Create an empty subtitle track.
    #[must_use]
    pub fn new(language: &str, format: SubtitleFormat) -> Self {
        Self {
            language: language.to_string(),
            format,
            cues: Vec::new(),
        }
    }

    /// Append a cue to this track.
    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.cues.push(cue);
    }

    /// Render the entire track as a `WebVTT` document.
    #[must_use]
    pub fn to_webvtt(&self) -> String {
        let mut out = String::from("WEBVTT\n\n");
        for (i, cue) in self.cues.iter().enumerate() {
            out.push_str(&format!("{}\n", i + 1)); // cue identifier
            out.push_str(&cue.to_webvtt());
            out.push_str("\n\n");
        }
        out
    }

    /// Total duration covered by this track in milliseconds (end of the last
    /// cue), or `0` if the track is empty.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.cues.iter().map(|c| c.end_ms).max().unwrap_or(0)
    }

    /// Split the track into `segment_duration_ms`-long segments.
    ///
    /// Each returned `SubtitleTrack` contains only the cues whose *start*
    /// time falls within that segment's time range.  Cues that overlap a
    /// segment boundary are included in the segment where they start.
    #[must_use]
    pub fn segment(self, segment_duration_ms: u64) -> Vec<SubtitleTrack> {
        if segment_duration_ms == 0 || self.cues.is_empty() {
            return vec![self];
        }

        let duration = self.duration_ms();
        let num_segments = (duration / segment_duration_ms + 1) as usize;
        let mut segments: Vec<SubtitleTrack> = (0..num_segments)
            .map(|_| SubtitleTrack::new(&self.language, self.format.clone()))
            .collect();

        for cue in self.cues {
            let seg_index = (cue.start_ms / segment_duration_ms) as usize;
            let idx = seg_index.min(segments.len() - 1);
            segments[idx].add_cue(cue);
        }

        // Discard trailing empty segments
        while segments.last().is_some_and(|s| s.cues.is_empty()) {
            segments.pop();
        }

        if segments.is_empty() {
            // Return a single empty track to avoid returning nothing
            segments.push(SubtitleTrack::new(&self.language, self.format));
        }

        segments
    }
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/// Convert milliseconds to `WebVTT` timestamp format `HH:MM:SS.mmm`.
#[must_use]
pub fn ms_to_webvtt_timestamp(ms: u64) -> String {
    let total_secs = ms / 1_000;
    let millis = ms % 1_000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3_600;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Convert milliseconds to SRT timestamp format `HH:MM:SS,mmm`.
fn ms_to_srt_timestamp(ms: u64) -> String {
    let total_secs = ms / 1_000;
    let millis = ms % 1_000;
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3_600;
    format!("{hours:02}:{mins:02}:{secs:02},{millis:03}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ms_to_webvtt_timestamp_zero() {
        assert_eq!(ms_to_webvtt_timestamp(0), "00:00:00.000");
    }

    #[test]
    fn test_ms_to_webvtt_timestamp_seconds() {
        assert_eq!(ms_to_webvtt_timestamp(1_500), "00:00:01.500");
    }

    #[test]
    fn test_ms_to_webvtt_timestamp_minutes() {
        assert_eq!(ms_to_webvtt_timestamp(90_000), "00:01:30.000");
    }

    #[test]
    fn test_ms_to_webvtt_timestamp_hours() {
        assert_eq!(ms_to_webvtt_timestamp(3_661_001), "01:01:01.001");
    }

    #[test]
    fn test_cue_duration_ms() {
        let cue = SubtitleCue::new(1_000, 4_500, "hello");
        assert_eq!(cue.duration_ms(), 3_500);
    }

    #[test]
    fn test_cue_duration_ms_zero_when_inverted() {
        // saturating sub should return 0 not underflow
        let cue = SubtitleCue {
            start_ms: 5_000,
            end_ms: 3_000,
            text: "oops".to_string(),
            position_x: None,
            position_y: None,
        };
        assert_eq!(cue.duration_ms(), 0);
    }

    #[test]
    fn test_cue_to_webvtt_basic() {
        let cue = SubtitleCue::new(0, 2_000, "Hello!");
        let vtt = cue.to_webvtt();
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.000"));
        assert!(vtt.contains("Hello!"));
    }

    #[test]
    fn test_cue_to_srt_basic() {
        let cue = SubtitleCue::new(1_000, 3_000, "World");
        let srt = cue.to_srt(1);
        assert!(srt.starts_with('1'));
        assert!(srt.contains("00:00:01,000 --> 00:00:03,000"));
        assert!(srt.contains("World"));
    }

    #[test]
    fn test_subtitle_track_new_empty() {
        let track = SubtitleTrack::new("en", SubtitleFormat::WebVtt);
        assert_eq!(track.language, "en");
        assert!(track.cues.is_empty());
        assert_eq!(track.duration_ms(), 0);
    }

    #[test]
    fn test_subtitle_track_add_cue_and_duration() {
        let mut track = SubtitleTrack::new("fr", SubtitleFormat::Srt);
        track.add_cue(SubtitleCue::new(0, 2_000, "Bonjour"));
        track.add_cue(SubtitleCue::new(3_000, 6_000, "Monde"));
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.duration_ms(), 6_000);
    }

    #[test]
    fn test_subtitle_track_to_webvtt_header() {
        let track = SubtitleTrack::new("en", SubtitleFormat::WebVtt);
        assert!(track.to_webvtt().starts_with("WEBVTT"));
    }

    #[test]
    fn test_subtitle_track_to_webvtt_with_cues() {
        let mut track = SubtitleTrack::new("en", SubtitleFormat::WebVtt);
        track.add_cue(SubtitleCue::new(0, 1_000, "Line one"));
        track.add_cue(SubtitleCue::new(2_000, 3_000, "Line two"));
        let vtt = track.to_webvtt();
        assert!(vtt.contains("Line one"));
        assert!(vtt.contains("Line two"));
    }

    #[test]
    fn test_subtitle_track_segment_splits_cues() {
        let mut track = SubtitleTrack::new("en", SubtitleFormat::WebVtt);
        // 3 cues across 0-9 seconds; segment every 5 seconds
        track.add_cue(SubtitleCue::new(0, 2_000, "A")); // seg 0
        track.add_cue(SubtitleCue::new(3_000, 4_000, "B")); // seg 0
        track.add_cue(SubtitleCue::new(6_000, 8_000, "C")); // seg 1
        let segments = track.segment(5_000);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].cues.len(), 2);
        assert_eq!(segments[1].cues.len(), 1);
    }

    #[test]
    fn test_subtitle_track_segment_empty_track() {
        let track = SubtitleTrack::new("de", SubtitleFormat::Ttml);
        let segments = track.segment(5_000);
        // Returns one (empty) track
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn test_subtitle_format_variants_exist() {
        let _a = SubtitleFormat::WebVtt;
        let _b = SubtitleFormat::Ttml;
        let _c = SubtitleFormat::Srt;
        let _d = SubtitleFormat::Smpte2052;
    }
}
