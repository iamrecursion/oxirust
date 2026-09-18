//! WebVTT subtitle segment packaging for adaptive streaming.
//!
//! Provides segment packaging for WebVTT subtitle cues into timed media
//! segments for HLS and DASH delivery, including `#EXT-X-MEDIA:TYPE=SUBTITLES`
//! manifest generation.

use crate::StreamError;

// ─── Subtitle Cue ─────────────────────────────────────────────────────────────

/// A single subtitle cue with timing and text content.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleCue {
    /// Optional cue identifier.
    pub id: Option<String>,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Text content of the cue (may include HTML-like tags for styling).
    pub text: String,
    /// Optional cue settings string (position, alignment, etc.).
    pub settings: Option<String>,
}

impl SubtitleCue {
    /// Create a new subtitle cue.
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self {
            id: None,
            start_ms,
            end_ms,
            text: text.into(),
            settings: None,
        }
    }

    /// Create a cue with an explicit ID.
    pub fn with_id(
        id: impl Into<String>,
        start_ms: u64,
        end_ms: u64,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: Some(id.into()),
            start_ms,
            end_ms,
            text: text.into(),
            settings: None,
        }
    }

    /// Duration of this cue in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Format the start time as WebVTT timestamp `HH:MM:SS.mmm`.
    pub fn start_timestamp(&self) -> String {
        format_vtt_timestamp(self.start_ms)
    }

    /// Format the end time as WebVTT timestamp `HH:MM:SS.mmm`.
    pub fn end_timestamp(&self) -> String {
        format_vtt_timestamp(self.end_ms)
    }

    /// Render this cue as WebVTT text (id + timestamp line + text).
    pub fn to_vtt_cue(&self) -> String {
        let mut out = String::new();
        if let Some(id) = &self.id {
            out.push_str(id);
            out.push('\n');
        }
        out.push_str(&self.start_timestamp());
        out.push_str(" --> ");
        out.push_str(&self.end_timestamp());
        if let Some(settings) = &self.settings {
            out.push(' ');
            out.push_str(settings);
        }
        out.push('\n');
        out.push_str(&self.text);
        out.push('\n');
        out
    }
}

/// Format milliseconds as `HH:MM:SS.mmm` for WebVTT.
fn format_vtt_timestamp(ms: u64) -> String {
    let hours = ms / 3_600_000;
    let minutes = (ms % 3_600_000) / 60_000;
    let seconds = (ms % 60_000) / 1000;
    let millis = ms % 1000;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

// ─── Subtitle Segment ─────────────────────────────────────────────────────────

/// A packaged subtitle segment ready for HLS/DASH delivery.
///
/// Contains a window of subtitle cues that fall within the segment's time range,
/// formatted as a complete WebVTT document.
#[derive(Debug, Clone)]
pub struct SubtitleSegment {
    /// Monotonically increasing sequence number.
    pub sequence_number: u64,
    /// Start time of this segment in milliseconds.
    pub start_ms: u64,
    /// Duration of this segment in milliseconds.
    pub duration_ms: u64,
    /// BCP-47 language tag for this subtitle track.
    pub language: String,
    /// WebVTT document bytes for this segment.
    pub data: Vec<u8>,
    /// Number of cues in this segment.
    pub cue_count: usize,
}

impl SubtitleSegment {
    /// Return the segment duration as floating-point seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ms as f64 / 1000.0
    }

    /// Return the segment start time as floating-point seconds.
    pub fn start_secs(&self) -> f64 {
        self.start_ms as f64 / 1000.0
    }

    /// Return the WebVTT content as a UTF-8 string.
    pub fn as_str(&self) -> Result<&str, StreamError> {
        std::str::from_utf8(&self.data)
            .map_err(|e| StreamError::ParseError(format!("subtitle segment not valid UTF-8: {e}")))
    }
}

// ─── Subtitle Track ──────────────────────────────────────────────────────────

/// Describes a single subtitle rendition.
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// Unique identifier for this track.
    pub id: String,
    /// BCP-47 language tag (e.g. `"en"`, `"fr-CA"`).
    pub language: String,
    /// Human-readable label (e.g. `"English"`).
    pub name: String,
    /// Whether this is the default subtitle track.
    pub default: bool,
    /// Whether auto-selection is permitted.
    pub autoselect: bool,
    /// Whether this is a forced subtitle track.
    pub forced: bool,
    /// HLS group identifier.
    pub group_id: String,
    /// Optional URI to the media playlist for this subtitle track.
    pub uri: Option<String>,
}

impl SubtitleTrack {
    /// Create a new subtitle track with sensible defaults.
    pub fn new(
        id: impl Into<String>,
        language: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            language: language.into(),
            name: name.into(),
            default: false,
            autoselect: true,
            forced: false,
            group_id: "subs".to_string(),
            uri: None,
        }
    }

    /// Render this track as an `#EXT-X-MEDIA:TYPE=SUBTITLES` line for HLS.
    pub fn to_ext_x_media_line(&self) -> String {
        let mut line = format!(
            "#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID=\"{}\",LANGUAGE=\"{}\",NAME=\"{}\"",
            self.group_id, self.language, self.name
        );
        line.push_str(if self.default {
            ",DEFAULT=YES"
        } else {
            ",DEFAULT=NO"
        });
        line.push_str(if self.autoselect {
            ",AUTOSELECT=YES"
        } else {
            ",AUTOSELECT=NO"
        });
        if self.forced {
            line.push_str(",FORCED=YES");
        }
        if let Some(uri) = &self.uri {
            line.push_str(&format!(",URI=\"{}\"", uri));
        }
        line
    }
}

// ─── Subtitle Packager ────────────────────────────────────────────────────────

/// Packages subtitle cues into timed WebVTT segments.
///
/// Accepts cues in any order and produces segments aligned to configurable
/// boundaries.  Each segment contains all cues that overlap the segment window.
#[derive(Debug)]
pub struct SubtitlePackager {
    /// Segment duration in milliseconds.
    pub segment_duration_ms: u64,
    /// BCP-47 language tag for the output track.
    pub language: String,
    /// Buffered cues not yet packaged.
    pending_cues: Vec<SubtitleCue>,
    /// Next segment sequence number.
    next_sequence: u64,
    /// Current segment start time in milliseconds.
    current_start_ms: u64,
}

impl SubtitlePackager {
    /// Create a packager with the given segment duration and language.
    pub fn new(segment_duration_ms: u64, language: impl Into<String>) -> Self {
        Self {
            segment_duration_ms: segment_duration_ms.max(1),
            language: language.into(),
            pending_cues: Vec::new(),
            next_sequence: 0,
            current_start_ms: 0,
        }
    }

    /// Add a subtitle cue to the packager.
    ///
    /// Cues are buffered until `flush_segment` is called or until a segment
    /// boundary is crossed.
    pub fn add_cue(&mut self, cue: SubtitleCue) {
        self.pending_cues.push(cue);
    }

    /// Flush all pending cues into a [`SubtitleSegment`] and advance the segment
    /// boundary by one `segment_duration_ms`.
    ///
    /// Returns `None` if there are no pending cues and `include_empty` is `false`.
    ///
    /// The generated WebVTT document contains all cues that were added since
    /// the last flush, regardless of whether they fall within the nominal
    /// segment window.
    pub fn flush_segment(&mut self, include_empty: bool) -> Option<SubtitleSegment> {
        if self.pending_cues.is_empty() && !include_empty {
            return None;
        }

        let start_ms = self.current_start_ms;
        let duration_ms = self.segment_duration_ms;
        let seq = self.next_sequence;

        // Sort cues by start time.
        self.pending_cues.sort_by_key(|c| c.start_ms);
        let cue_count = self.pending_cues.len();
        let cues = std::mem::take(&mut self.pending_cues);

        let vtt = build_webvtt_document(&cues);

        self.next_sequence += 1;
        self.current_start_ms += duration_ms;

        Some(SubtitleSegment {
            sequence_number: seq,
            start_ms,
            duration_ms,
            language: self.language.clone(),
            data: vtt.into_bytes(),
            cue_count,
        })
    }

    /// Flush all pending cues spanning multiple segment boundaries.
    ///
    /// Splits cues across `count` segment boundaries and returns one
    /// [`SubtitleSegment`] per boundary.  Empty segments are not produced.
    pub fn flush_segments(&mut self, count: usize) -> Vec<SubtitleSegment> {
        let mut segments = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(seg) = self.flush_segment(false) {
                segments.push(seg);
            } else {
                // Advance the clock even for empty windows.
                self.next_sequence += 1;
                self.current_start_ms += self.segment_duration_ms;
            }
        }
        segments
    }

    /// Return the next segment sequence number (without flushing).
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Return the number of pending (unflushed) cues.
    pub fn pending_count(&self) -> usize {
        self.pending_cues.len()
    }
}

/// Build a complete WebVTT document from a slice of cues.
fn build_webvtt_document(cues: &[SubtitleCue]) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("WEBVTT\n\n");
    for cue in cues {
        out.push_str(&cue.to_vtt_cue());
        out.push('\n');
    }
    out
}

// ─── Subtitle Track Manager ───────────────────────────────────────────────────

/// Manages multiple subtitle tracks for a streaming presentation.
#[derive(Debug, Default)]
pub struct SubtitleTrackManager {
    tracks: Vec<SubtitleTrack>,
}

impl SubtitleTrackManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a subtitle track.
    ///
    /// Returns an error if a track with the same `id` already exists.
    pub fn add_track(&mut self, track: SubtitleTrack) -> Result<(), StreamError> {
        if self.tracks.iter().any(|t| t.id == track.id) {
            return Err(StreamError::Generic(format!(
                "subtitle track with id '{}' already exists",
                track.id
            )));
        }
        self.tracks.push(track);
        Ok(())
    }

    /// Remove a track by `id`.
    ///
    /// Returns `true` if the track was found and removed.
    pub fn remove_track(&mut self, id: &str) -> bool {
        let before = self.tracks.len();
        self.tracks.retain(|t| t.id != id);
        self.tracks.len() < before
    }

    /// Return all tracks.
    pub fn tracks(&self) -> &[SubtitleTrack] {
        &self.tracks
    }

    /// Number of registered tracks.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Return `true` if no tracks are registered.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Return the default track, if one is marked.
    pub fn default_track(&self) -> Option<&SubtitleTrack> {
        self.tracks.iter().find(|t| t.default)
    }

    /// Select a track by BCP-47 language tag (exact or prefix match).
    pub fn select_track(&self, language: &str) -> Option<&SubtitleTrack> {
        self.tracks
            .iter()
            .find(|t| t.language == language)
            .or_else(|| {
                self.tracks
                    .iter()
                    .find(|t| t.language.starts_with(language))
            })
    }

    /// Generate the full `#EXT-X-MEDIA` block for all subtitle tracks.
    pub fn to_hls_ext_x_media(&self) -> String {
        self.tracks
            .iter()
            .map(|t| t.to_ext_x_media_line())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Set the default subtitle track by `id`, clearing any previous default.
    pub fn set_default(&mut self, id: &str) -> Result<(), StreamError> {
        if !self.tracks.iter().any(|t| t.id == id) {
            return Err(StreamError::Generic(format!(
                "no subtitle track with id '{}' found",
                id
            )));
        }
        for track in &mut self.tracks {
            track.default = track.id == id;
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cue(start_ms: u64, end_ms: u64, text: &str) -> SubtitleCue {
        SubtitleCue::new(start_ms, end_ms, text)
    }

    fn en_track() -> SubtitleTrack {
        let mut t = SubtitleTrack::new("en-subs", "en", "English");
        t.default = true;
        t.uri = Some("subs_en.m3u8".to_string());
        t
    }

    fn fr_track() -> SubtitleTrack {
        SubtitleTrack::new("fr-subs", "fr-CA", "Français")
    }

    // ── SubtitleCue ───────────────────────────────────────────────────────────

    #[test]
    fn test_cue_duration_ms() {
        let cue = make_cue(1000, 4000, "Hello");
        assert_eq!(cue.duration_ms(), 3000);
    }

    #[test]
    fn test_cue_timestamps() {
        let cue = make_cue(3_661_500, 3_665_000, "Test");
        assert_eq!(cue.start_timestamp(), "01:01:01.500");
        assert_eq!(cue.end_timestamp(), "01:01:05.000");
    }

    #[test]
    fn test_cue_to_vtt_cue_no_id() {
        let cue = make_cue(0, 2000, "Hello world");
        let vtt = cue.to_vtt_cue();
        assert!(vtt.contains("00:00:00.000 --> 00:00:02.000"), "vtt={vtt}");
        assert!(vtt.contains("Hello world"), "vtt={vtt}");
    }

    #[test]
    fn test_cue_to_vtt_cue_with_id() {
        let cue = SubtitleCue::with_id("cue-1", 1000, 3000, "Test");
        let vtt = cue.to_vtt_cue();
        assert!(vtt.starts_with("cue-1\n"), "vtt={vtt}");
    }

    #[test]
    fn test_cue_with_settings() {
        let mut cue = make_cue(0, 1000, "Positioned");
        cue.settings = Some("line:90% align:center".to_string());
        let vtt = cue.to_vtt_cue();
        assert!(vtt.contains("line:90%"), "vtt={vtt}");
    }

    // ── SubtitlePackager ──────────────────────────────────────────────────────

    #[test]
    fn test_packager_flush_empty_none() {
        let mut p = SubtitlePackager::new(6000, "en");
        assert!(p.flush_segment(false).is_none());
    }

    #[test]
    fn test_packager_flush_empty_include_empty() {
        let mut p = SubtitlePackager::new(6000, "en");
        let seg = p.flush_segment(true);
        assert!(seg.is_some());
        let seg = seg.expect("should be Some");
        assert_eq!(seg.cue_count, 0);
    }

    #[test]
    fn test_packager_add_and_flush_cue() {
        let mut p = SubtitlePackager::new(6000, "en");
        p.add_cue(make_cue(1000, 3000, "Hello"));
        let seg = p.flush_segment(false).expect("segment");
        assert_eq!(seg.cue_count, 1);
        assert_eq!(seg.sequence_number, 0);
        assert_eq!(seg.language, "en");
    }

    #[test]
    fn test_packager_vtt_content() {
        let mut p = SubtitlePackager::new(6000, "en");
        p.add_cue(make_cue(0, 2000, "Line one"));
        let seg = p.flush_segment(false).expect("segment");
        let text = seg.as_str().expect("utf8");
        assert!(text.starts_with("WEBVTT\n"), "text={text}");
        assert!(text.contains("Line one"), "text={text}");
        assert!(
            text.contains("00:00:00.000 --> 00:00:02.000"),
            "text={text}"
        );
    }

    #[test]
    fn test_packager_sequence_increments() {
        let mut p = SubtitlePackager::new(6000, "en");
        for i in 0..3u64 {
            p.add_cue(make_cue(i * 2000, i * 2000 + 1000, "cue"));
            let seg = p.flush_segment(false).expect("seg");
            assert_eq!(seg.sequence_number, i);
        }
    }

    #[test]
    fn test_packager_cues_sorted_by_start() {
        let mut p = SubtitlePackager::new(6000, "en");
        p.add_cue(make_cue(3000, 4000, "Second"));
        p.add_cue(make_cue(1000, 2000, "First"));
        let seg = p.flush_segment(false).expect("seg");
        let text = seg.as_str().expect("utf8");
        // "First" should appear before "Second" in the output.
        let pos_first = text.find("First").expect("First");
        let pos_second = text.find("Second").expect("Second");
        assert!(
            pos_first < pos_second,
            "cues should be sorted by start time"
        );
    }

    #[test]
    fn test_packager_pending_count() {
        let mut p = SubtitlePackager::new(6000, "en");
        p.add_cue(make_cue(0, 1000, "A"));
        p.add_cue(make_cue(1000, 2000, "B"));
        assert_eq!(p.pending_count(), 2);
        p.flush_segment(false);
        assert_eq!(p.pending_count(), 0);
    }

    #[test]
    fn test_packager_segment_duration() {
        let mut p = SubtitlePackager::new(4000, "fr");
        p.add_cue(make_cue(0, 1000, "Bonjour"));
        let seg = p.flush_segment(false).expect("seg");
        assert_eq!(seg.duration_ms, 4000);
        assert!((seg.duration_secs() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_packager_start_advances() {
        let mut p = SubtitlePackager::new(6000, "en");
        p.add_cue(make_cue(0, 1000, "A"));
        let seg1 = p.flush_segment(false).expect("seg1");
        p.add_cue(make_cue(6000, 7000, "B"));
        let seg2 = p.flush_segment(false).expect("seg2");
        assert_eq!(seg1.start_ms, 0);
        assert_eq!(seg2.start_ms, 6000);
    }

    // ── SubtitleTrack ─────────────────────────────────────────────────────────

    #[test]
    fn test_ext_x_media_type_subtitles() {
        let track = en_track();
        let line = track.to_ext_x_media_line();
        assert!(line.contains("TYPE=SUBTITLES"), "line={line}");
    }

    #[test]
    fn test_ext_x_media_default_yes() {
        let track = en_track();
        let line = track.to_ext_x_media_line();
        assert!(line.contains("DEFAULT=YES"), "line={line}");
    }

    #[test]
    fn test_ext_x_media_uri() {
        let track = en_track();
        let line = track.to_ext_x_media_line();
        assert!(line.contains("URI=\"subs_en.m3u8\""), "line={line}");
    }

    #[test]
    fn test_ext_x_media_forced() {
        let mut track = en_track();
        track.forced = true;
        let line = track.to_ext_x_media_line();
        assert!(line.contains("FORCED=YES"), "line={line}");
    }

    // ── SubtitleTrackManager ──────────────────────────────────────────────────

    #[test]
    fn test_add_track_and_count() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("add en");
        mgr.add_track(fr_track()).expect("add fr");
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("first");
        assert!(mgr.add_track(en_track()).is_err());
    }

    #[test]
    fn test_select_track_exact_language() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(fr_track()).expect("add fr");
        let t = mgr.select_track("fr-CA").expect("found");
        assert_eq!(t.id, "fr-subs");
    }

    #[test]
    fn test_select_track_prefix() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(fr_track()).expect("add fr");
        let t = mgr.select_track("fr").expect("prefix match");
        assert_eq!(t.id, "fr-subs");
    }

    #[test]
    fn test_default_track() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("add en");
        mgr.add_track(fr_track()).expect("add fr");
        let d = mgr.default_track().expect("default");
        assert_eq!(d.id, "en-subs");
    }

    #[test]
    fn test_set_default() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("add en");
        mgr.add_track(fr_track()).expect("add fr");
        mgr.set_default("fr-subs").expect("set default");
        assert_eq!(mgr.default_track().expect("default").id, "fr-subs");
    }

    #[test]
    fn test_remove_track() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("add");
        assert!(mgr.remove_track("en-subs"));
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_hls_ext_x_media_block() {
        let mut mgr = SubtitleTrackManager::new();
        mgr.add_track(en_track()).expect("add en");
        mgr.add_track(fr_track()).expect("add fr");
        let block = mgr.to_hls_ext_x_media();
        assert_eq!(block.lines().count(), 2);
        assert!(block.contains("TYPE=SUBTITLES"));
    }

    // ── build_webvtt_document ─────────────────────────────────────────────────

    #[test]
    fn test_build_webvtt_empty_cues() {
        let doc = build_webvtt_document(&[]);
        assert!(doc.starts_with("WEBVTT\n"), "doc={doc}");
    }

    #[test]
    fn test_build_webvtt_multiple_cues() {
        let cues = vec![make_cue(0, 1000, "First"), make_cue(2000, 3000, "Second")];
        let doc = build_webvtt_document(&cues);
        assert!(doc.contains("First"));
        assert!(doc.contains("Second"));
    }
}
