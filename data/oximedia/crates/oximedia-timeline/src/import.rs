//! Import timeline from various formats.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use quick_xml::Reader;

use crate::clip::{Clip, MediaSource};
use crate::error::{TimelineError, TimelineResult};
use crate::timeline::Timeline;
use crate::types::Position;

/// Import format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportFormat {
    /// EDL (Edit Decision List) - CMX 3600.
    Edl,
    /// FCPXML (Final Cut Pro XML).
    Fcpxml,
    /// Adobe Premiere XML.
    PremiereXml,
    /// AAF (Advanced Authoring Format).
    Aaf,
    /// `DaVinci` Resolve XML.
    ResolveXml,
}

impl ImportFormat {
    /// Detects format from file extension.
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()?
            .to_str()
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "edl" => Some(Self::Edl),
                "fcpxml" => Some(Self::Fcpxml),
                "xml" => Some(Self::PremiereXml), // Ambiguous, might need content detection
                "aaf" => Some(Self::Aaf),
                _ => None,
            })
    }
}

/// Import options.
#[derive(Clone, Debug)]
pub struct ImportOptions {
    /// Whether to import audio tracks.
    pub import_audio: bool,
    /// Whether to import video tracks.
    pub import_video: bool,
    /// Whether to import markers.
    pub import_markers: bool,
    /// Whether to import effects.
    pub import_effects: bool,
    /// Whether to import transitions.
    pub import_transitions: bool,
    /// Frame rate override (if None, use from source).
    pub frame_rate_override: Option<oximedia_core::Rational>,
    /// Sample rate override (if None, use from source).
    pub sample_rate_override: Option<u32>,
}

impl ImportOptions {
    /// Creates default import options (import everything).
    #[must_use]
    pub fn default_all() -> Self {
        Self {
            import_audio: true,
            import_video: true,
            import_markers: true,
            import_effects: true,
            import_transitions: true,
            frame_rate_override: None,
            sample_rate_override: None,
        }
    }

    /// Creates options for video only.
    #[must_use]
    pub fn video_only() -> Self {
        Self {
            import_audio: false,
            import_video: true,
            import_markers: true,
            import_effects: true,
            import_transitions: true,
            frame_rate_override: None,
            sample_rate_override: None,
        }
    }

    /// Creates options for audio only.
    #[must_use]
    pub fn audio_only() -> Self {
        Self {
            import_audio: true,
            import_video: false,
            import_markers: true,
            import_effects: true,
            import_transitions: true,
            frame_rate_override: None,
            sample_rate_override: None,
        }
    }
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self::default_all()
    }
}

/// Import statistics.
#[derive(Clone, Debug, Default)]
pub struct ImportStats {
    /// Number of video tracks imported.
    pub video_tracks: usize,
    /// Number of audio tracks imported.
    pub audio_tracks: usize,
    /// Number of clips imported.
    pub clips: usize,
    /// Number of transitions imported.
    pub transitions: usize,
    /// Number of effects imported.
    pub effects: usize,
    /// Number of markers imported.
    pub markers: usize,
    /// Warnings encountered during import.
    pub warnings: Vec<String>,
}

/// Timeline importer.
#[allow(dead_code)]
pub struct TimelineImporter {
    options: ImportOptions,
}

// ---------------------------------------------------------------------------
// Internal FCPXML parsing helpers
// ---------------------------------------------------------------------------

/// Parsed information extracted from FCPXML resources and spine.
#[derive(Debug, Default)]
struct FcpxmlData {
    /// Project/sequence name.
    name: String,
    /// Frame duration string (e.g. "100/2400s").
    frame_duration: String,
    /// Width and height from format element.
    width: u32,
    height: u32,
    /// Map from asset id to file path.
    assets: HashMap<String, PathBuf>,
    /// Clips collected from spine.
    clips: Vec<FcpxmlClip>,
}

/// A single clip entry parsed from an FCPXML spine.
#[derive(Debug)]
struct FcpxmlClip {
    name: String,
    asset_ref: String,
    /// Timeline offset in the FCPXML rational-time format.
    offset: String,
    /// Clip duration in rational-time format.
    duration: String,
    /// Source in-point.
    start: String,
}

/// Parse an FCPXML rational-time string (e.g. "100/2400s", "48s", "0s") into
/// frames given the sequence frame rate (num/den).  Returns 0 on parse
/// failure so callers stay resilient.
fn fcpxml_time_to_frames(time_str: &str, fps_num: i64, fps_den: i64) -> i64 {
    // Strip trailing 's'
    let s = time_str.trim_end_matches('s');
    if s.is_empty() {
        return 0;
    }

    // Either "num/den" or plain integer seconds
    let (num, den): (i64, i64) = if let Some(pos) = s.find('/') {
        let n = s[..pos].parse::<i64>().unwrap_or(0);
        let d = s[pos + 1..].parse::<i64>().unwrap_or(1);
        (n, d)
    } else {
        let n = s.parse::<i64>().unwrap_or(0);
        (n, 1)
    };

    // Convert seconds (num/den) → frames (fps_num / fps_den).
    // frames = (num/den) * (fps_num/fps_den)
    // Use integer arithmetic: frames = num * fps_num / (den * fps_den)
    if den == 0 || fps_den == 0 {
        return 0;
    }
    num * fps_num / (den * fps_den)
}

/// Parse the FCPXML frameDuration string (e.g. "100/2400s") and return
/// (`fps_num`, `fps_den`) where fps = `fps_num` / `fps_den` (e.g. 24/1).
fn parse_frame_duration(fd: &str) -> (i64, i64) {
    // frameDuration is the duration of ONE frame expressed as "num/denom s".
    // So fps = denom / num.
    let s = fd.trim_end_matches('s');
    if let Some(pos) = s.find('/') {
        let num = s[..pos].parse::<i64>().unwrap_or(1);
        let den = s[pos + 1..].parse::<i64>().unwrap_or(1);
        // fps = den / num  (e.g. 2400/100 = 24)
        (den, num)
    } else {
        // Plain "Ns" means 1/N seconds per frame → fps = N
        let n = s.parse::<i64>().unwrap_or(1);
        (n, 1)
    }
}

/// Extract a named attribute value from a quick-xml `BytesStart`.
fn attr_val(e: &quick_xml::events::BytesStart<'_>, name: &[u8]) -> Option<String> {
    e.attributes()
        .filter_map(std::result::Result::ok)
        .find(|a| a.key.as_ref() == name)
        .and_then(|a| String::from_utf8(a.value.into_owned()).ok())
}

/// Parse FCPXML file and return intermediate `FcpxmlData`.
fn parse_fcpxml_file(path: &Path) -> TimelineResult<FcpxmlData> {
    let content = std::fs::read_to_string(path)?;
    let mut reader = Reader::from_str(&content);
    reader.config_mut().trim_text(true);

    let mut data = FcpxmlData::default();
    // Track context for nested text accumulation (Premiere/Resolve style)
    let mut buf = Vec::new();

    // We use a simple state machine to know where we are in the document.
    #[derive(PartialEq)]
    enum Context {
        Root,
        Spine,
    }
    let mut ctx = Context::Root;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e) | Event::Empty(ref e)) => {
                match e.name().as_ref() {
                    b"project" | b"sequence" if ctx == Context::Root => {
                        if let Some(n) = attr_val(e, b"name") {
                            if data.name.is_empty() {
                                data.name = n;
                            }
                        }
                    }
                    b"format" => {
                        if let Some(fd) = attr_val(e, b"frameDuration") {
                            data.frame_duration = fd;
                        }
                        if let Some(w) = attr_val(e, b"width") {
                            data.width = w.parse().unwrap_or(1920);
                        }
                        if let Some(h) = attr_val(e, b"height") {
                            data.height = h.parse().unwrap_or(1080);
                        }
                    }
                    b"asset" => {
                        if let (Some(id), Some(src)) = (attr_val(e, b"id"), attr_val(e, b"src")) {
                            // FCPXML uses "file:///path/to/file" URIs
                            let path_str = src.strip_prefix("file://").unwrap_or(&src).to_string();
                            data.assets.insert(id, PathBuf::from(path_str));
                        }
                    }
                    b"spine" => {
                        ctx = Context::Spine;
                    }
                    b"asset-clip" if ctx == Context::Spine => {
                        let clip = FcpxmlClip {
                            name: attr_val(e, b"name").unwrap_or_default(),
                            asset_ref: attr_val(e, b"ref").unwrap_or_default(),
                            offset: attr_val(e, b"offset").unwrap_or_else(|| "0s".to_string()),
                            duration: attr_val(e, b"duration").unwrap_or_else(|| "0s".to_string()),
                            start: attr_val(e, b"start").unwrap_or_else(|| "0s".to_string()),
                        };
                        data.clips.push(clip);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.name().as_ref() == b"spine" {
                    ctx = Context::Root;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(TimelineError::ImportError(format!(
                    "FCPXML parse error: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    // Default name if not found.
    if data.name.is_empty() {
        data.name = "Imported Timeline".to_string();
    }

    // Default frame duration: 24fps
    if data.frame_duration.is_empty() {
        data.frame_duration = "100/2400s".to_string();
    }

    Ok(data)
}

// ---------------------------------------------------------------------------
// Internal xmeml (Premiere / Resolve) parsing helpers
// ---------------------------------------------------------------------------

/// Accumulated data while parsing an xmeml clip item.
#[derive(Debug, Default, Clone)]
struct XmemlClipItem {
    name: String,
    path_url: String,
    clip_in: i64,
    clip_out: i64,
    start: i64,
    end: i64,
}

/// Parse an xmeml XML file (Premiere or Resolve format).
/// Returns (`sequence_name`, `timebase_fps`, `video_clips`, `audio_clips`).
#[allow(clippy::type_complexity)]
fn parse_xmeml_file(
    path: &Path,
) -> TimelineResult<(String, i64, Vec<XmemlClipItem>, Vec<XmemlClipItem>)> {
    let content = std::fs::read_to_string(path)?;
    let mut reader = Reader::from_str(&content);
    reader.config_mut().trim_text(true);

    let mut seq_name = String::new();
    let mut timebase: i64 = 25;
    let mut video_clips: Vec<XmemlClipItem> = Vec::new();
    let mut audio_clips: Vec<XmemlClipItem> = Vec::new();
    let mut buf = Vec::new();

    // State
    #[derive(PartialEq, Clone, Copy)]
    enum Section {
        Root,
        Sequence,
        Video,
        Audio,
        VideoTrack,
        AudioTrack,
        VideoClipItem,
        AudioClipItem,
    }

    let mut section = Section::Root;
    // Stack for text accumulation
    let mut current_tag: Vec<u8> = Vec::new();
    let mut current_clip = XmemlClipItem::default();
    // For timebase / name collection inside <sequence>
    let mut collecting_seq_name = false;
    let mut collecting_timebase = false;
    // For clip field collection
    let mut collecting_clip_name = false;
    let mut collecting_path_url = false;
    let mut collecting_in = false;
    let mut collecting_out = false;
    let mut collecting_start = false;
    let mut collecting_end = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                current_tag = e.name().as_ref().to_vec();
                match (section, e.name().as_ref()) {
                    (Section::Root, b"sequence") => {
                        section = Section::Sequence;
                    }
                    (Section::Sequence, b"name") => {
                        collecting_seq_name = true;
                    }
                    (Section::Sequence, b"timebase") => {
                        collecting_timebase = true;
                    }
                    (Section::Sequence, b"video") => {
                        section = Section::Video;
                    }
                    (Section::Sequence, b"audio") => {
                        section = Section::Audio;
                    }
                    (Section::Video, b"track") => {
                        section = Section::VideoTrack;
                    }
                    (Section::Audio, b"track") => {
                        section = Section::AudioTrack;
                    }
                    (Section::VideoTrack, b"clipitem") => {
                        section = Section::VideoClipItem;
                        current_clip = XmemlClipItem::default();
                    }
                    (Section::AudioTrack, b"clipitem") => {
                        section = Section::AudioClipItem;
                        current_clip = XmemlClipItem::default();
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"name") => {
                        collecting_clip_name = true;
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"pathurl") => {
                        collecting_path_url = true;
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"in") => {
                        collecting_in = true;
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"out") => {
                        collecting_out = true;
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"start") => {
                        collecting_start = true;
                    }
                    (Section::VideoClipItem | Section::AudioClipItem, b"end") => {
                        collecting_end = true;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                let text = String::from_utf8_lossy(e.as_ref()).into_owned();
                if collecting_seq_name && seq_name.is_empty() {
                    seq_name = text;
                    collecting_seq_name = false;
                } else if collecting_timebase {
                    timebase = text.trim().parse::<i64>().unwrap_or(25);
                    collecting_timebase = false;
                } else if collecting_clip_name {
                    current_clip.name = text;
                    collecting_clip_name = false;
                } else if collecting_path_url {
                    current_clip.path_url = text;
                    collecting_path_url = false;
                } else if collecting_in {
                    current_clip.clip_in = text.trim().parse::<i64>().unwrap_or(0);
                    collecting_in = false;
                } else if collecting_out {
                    current_clip.clip_out = text.trim().parse::<i64>().unwrap_or(0);
                    collecting_out = false;
                } else if collecting_start {
                    current_clip.start = text.trim().parse::<i64>().unwrap_or(0);
                    collecting_start = false;
                } else if collecting_end {
                    current_clip.end = text.trim().parse::<i64>().unwrap_or(0);
                    collecting_end = false;
                }
            }
            Ok(Event::End(ref e)) => {
                match (section, e.name().as_ref()) {
                    (Section::VideoClipItem, b"clipitem") => {
                        video_clips.push(current_clip.clone());
                        current_clip = XmemlClipItem::default();
                        section = Section::VideoTrack;
                    }
                    (Section::AudioClipItem, b"clipitem") => {
                        audio_clips.push(current_clip.clone());
                        current_clip = XmemlClipItem::default();
                        section = Section::AudioTrack;
                    }
                    (Section::VideoTrack, b"track") => {
                        section = Section::Video;
                    }
                    (Section::AudioTrack, b"track") => {
                        section = Section::Audio;
                    }
                    (Section::Video, b"video") => {
                        section = Section::Sequence;
                    }
                    (Section::Audio, b"audio") => {
                        section = Section::Sequence;
                    }
                    (Section::Sequence, b"sequence") => {
                        section = Section::Root;
                    }
                    _ => {}
                }
                current_tag.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(TimelineError::ImportError(format!(
                    "xmeml parse error: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }
    let _ = current_tag; // suppress unused warning

    if seq_name.is_empty() {
        seq_name = "Imported Timeline".to_string();
    }

    Ok((seq_name, timebase, video_clips, audio_clips))
}

/// Convert a `file:///path` URL to a `PathBuf`.
fn pathurl_to_path(url: &str) -> PathBuf {
    let stripped = url
        .strip_prefix("file:///")
        .or_else(|| url.strip_prefix("file://"))
        .unwrap_or(url);
    PathBuf::from(stripped)
}

/// Build a `Timeline` from parsed xmeml data, respecting the import options.
fn build_timeline_from_xmeml(
    seq_name: String,
    timebase: i64,
    video_items: Vec<XmemlClipItem>,
    audio_items: Vec<XmemlClipItem>,
    options: &ImportOptions,
) -> TimelineResult<(Timeline, ImportStats)> {
    use oximedia_core::Rational;

    let fps = if let Some(ovr) = options.frame_rate_override {
        ovr
    } else if timebase > 0 {
        Rational::new(timebase, 1)
    } else {
        Rational::new(25, 1)
    };

    let sample_rate = options.sample_rate_override.unwrap_or(48000);
    let mut timeline = Timeline::new(seq_name, fps, sample_rate)?;
    let mut stats = ImportStats::default();

    if options.import_video && !video_items.is_empty() {
        let track_id = timeline.add_video_track("V1")?;
        stats.video_tracks += 1;
        for item in &video_items {
            let src_in = Position::new(item.clip_in);
            let src_out = Position::new(item.clip_out.max(item.clip_in + 1));
            let tl_in = Position::new(item.start);
            let path = pathurl_to_path(&item.path_url);
            let clip = Clip::new(
                item.name.clone(),
                MediaSource::file(path),
                src_in,
                src_out,
                tl_in,
            )?;
            timeline.add_clip(track_id, clip)?;
            stats.clips += 1;
        }
    }

    if options.import_audio && !audio_items.is_empty() {
        let track_id = timeline.add_audio_track("A1")?;
        stats.audio_tracks += 1;
        for item in &audio_items {
            let src_in = Position::new(item.clip_in);
            let src_out = Position::new(item.clip_out.max(item.clip_in + 1));
            let tl_in = Position::new(item.start);
            let path = pathurl_to_path(&item.path_url);
            let clip = Clip::new(
                item.name.clone(),
                MediaSource::file(path),
                src_in,
                src_out,
                tl_in,
            )?;
            timeline.add_clip(track_id, clip)?;
            stats.clips += 1;
        }
    }

    Ok((timeline, stats))
}

// ---------------------------------------------------------------------------
// TimelineImporter implementation
// ---------------------------------------------------------------------------

impl TimelineImporter {
    /// Creates a new importer with options.
    #[must_use]
    pub fn new(options: ImportOptions) -> Self {
        Self { options }
    }

    /// Creates a new importer with default options.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ImportOptions::default())
    }

    /// Imports a timeline from a file.
    ///
    /// # Errors
    ///
    /// Returns error if import fails.
    pub fn import_file(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        let format = ImportFormat::from_path(path).ok_or_else(|| {
            TimelineError::ImportError(format!("Unsupported file format: {}", path.display()))
        })?;

        match format {
            ImportFormat::Edl => self.import_edl(path),
            ImportFormat::Fcpxml => self.import_fcpxml(path),
            ImportFormat::PremiereXml => self.import_premiere_xml(path),
            ImportFormat::Aaf => self.import_aaf(path),
            ImportFormat::ResolveXml => self.import_resolve_xml(path),
        }
    }

    /// Imports from EDL format (CMX 3600) using the oximedia-edl crate.
    fn import_edl(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        use crate::clip::{Clip, MediaSource};
        use oximedia_core::Rational;
        use oximedia_edl::Edl;

        let edl = Edl::from_file(path)
            .map_err(|e| TimelineError::ImportError(format!("EDL parse error: {e}")))?;

        // Map EdlFrameRate to fps rational.
        let fps = if let Some(ref ovr) = self.options.frame_rate_override {
            *ovr
        } else {
            use oximedia_edl::timecode::EdlFrameRate;
            match edl.frame_rate {
                EdlFrameRate::Fps23976 => Rational::new(24000, 1001),
                EdlFrameRate::Fps23_976 => Rational::new(24000, 1001),
                EdlFrameRate::Fps24 => Rational::new(24, 1),
                EdlFrameRate::Fps25 => Rational::new(25, 1),
                EdlFrameRate::Fps2997DF | EdlFrameRate::Fps2997NDF => Rational::new(30000, 1001),
                EdlFrameRate::Fps30 => Rational::new(30, 1),
                EdlFrameRate::Fps50 => Rational::new(50, 1),
                EdlFrameRate::Fps5994 => Rational::new(60000, 1001),
                EdlFrameRate::Fps59_94 => Rational::new(60000, 1001),
                EdlFrameRate::Fps60 => Rational::new(60, 1),
            }
        };

        let sample_rate = self.options.sample_rate_override.unwrap_or(48000);
        let title = edl.title.as_deref().unwrap_or("Imported EDL").to_string();
        let mut timeline = Timeline::new(title, fps, sample_rate)?;
        let mut stats = ImportStats::default();

        if self.options.import_video && !edl.events.is_empty() {
            let video_events: Vec<_> = edl.events.iter().filter(|e| e.track.has_video()).collect();

            if !video_events.is_empty() {
                let track_id = timeline.add_video_track("V1")?;
                stats.video_tracks += 1;

                for event in &video_events {
                    let src_in_frames = event.source_in.to_frames() as i64;
                    let src_out_frames = event.source_out.to_frames() as i64;
                    let rec_in_frames = event.record_in.to_frames() as i64;

                    let src_in = Position::new(src_in_frames);
                    let src_out = Position::new(src_out_frames.max(src_in_frames + 1));
                    let tl_in = Position::new(rec_in_frames);

                    let clip_name = event
                        .clip_name
                        .as_deref()
                        .unwrap_or(event.reel.as_str())
                        .to_string();

                    // Use reel name as a path hint (no real file to open).
                    let source = MediaSource::file(std::path::PathBuf::from(event.reel.as_str()));

                    let clip = Clip::new(clip_name, source, src_in, src_out, tl_in)?;
                    timeline.add_clip(track_id, clip)?;
                    stats.clips += 1;
                }
            }
        }

        if self.options.import_audio && !edl.events.is_empty() {
            let audio_events: Vec<_> = edl.events.iter().filter(|e| e.track.has_audio()).collect();

            if !audio_events.is_empty() {
                let track_id = timeline.add_audio_track("A1")?;
                stats.audio_tracks += 1;

                for event in &audio_events {
                    let src_in_frames = event.source_in.to_frames() as i64;
                    let src_out_frames = event.source_out.to_frames() as i64;
                    let rec_in_frames = event.record_in.to_frames() as i64;

                    let src_in = Position::new(src_in_frames);
                    let src_out = Position::new(src_out_frames.max(src_in_frames + 1));
                    let tl_in = Position::new(rec_in_frames);

                    let clip_name = event
                        .clip_name
                        .as_deref()
                        .unwrap_or(event.reel.as_str())
                        .to_string();

                    let source = MediaSource::file(std::path::PathBuf::from(event.reel.as_str()));

                    let clip = Clip::new(clip_name, source, src_in, src_out, tl_in)?;
                    timeline.add_clip(track_id, clip)?;
                    stats.clips += 1;
                }
            }
        }

        Ok((timeline, stats))
    }

    /// Imports from FCPXML format.
    ///
    /// Parses Final Cut Pro XML (version 1.9) and builds a `Timeline`.
    fn import_fcpxml(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        use oximedia_core::Rational;

        let data = parse_fcpxml_file(path)?;

        let (fps_num, fps_den) = if data.frame_duration.is_empty() {
            (24, 1)
        } else {
            parse_frame_duration(&data.frame_duration)
        };

        let fps = if let Some(ovr) = self.options.frame_rate_override {
            ovr
        } else {
            Rational::new(fps_num, fps_den)
        };

        let sample_rate = self.options.sample_rate_override.unwrap_or(48000);
        let mut timeline = Timeline::new(data.name, fps, sample_rate)?;
        let mut stats = ImportStats::default();

        // Add default video track if there are spine clips and video import enabled.
        if self.options.import_video && !data.clips.is_empty() {
            let track_id = timeline.add_video_track("V1")?;
            stats.video_tracks += 1;

            for clip_entry in &data.clips {
                let tl_offset = fcpxml_time_to_frames(&clip_entry.offset, fps_num, fps_den);
                let dur = fcpxml_time_to_frames(&clip_entry.duration, fps_num, fps_den);
                let src_start = fcpxml_time_to_frames(&clip_entry.start, fps_num, fps_den);

                let dur_positive = dur.max(1);
                let src_in = Position::new(src_start);
                let src_out = Position::new(src_start + dur_positive);
                let tl_in = Position::new(tl_offset);

                // Resolve the file path from assets map.
                let media_path = data
                    .assets
                    .get(&clip_entry.asset_ref)
                    .cloned()
                    .unwrap_or_else(|| PathBuf::from(&clip_entry.asset_ref));

                let clip = Clip::new(
                    clip_entry.name.clone(),
                    MediaSource::file(media_path),
                    src_in,
                    src_out,
                    tl_in,
                )?;
                timeline.add_clip(track_id, clip)?;
                stats.clips += 1;
            }
        }

        Ok((timeline, stats))
    }

    /// Imports from Premiere XML format (xmeml v4).
    fn import_premiere_xml(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        let (name, timebase, video_items, audio_items) = parse_xmeml_file(path)?;
        build_timeline_from_xmeml(name, timebase, video_items, audio_items, &self.options)
    }

    /// Imports from AAF format using the oximedia-aaf crate.
    ///
    /// AAF files store compositions (analogous to sequences) with tracks
    /// and clips. We use `TimelineConverter` to extract the simplified
    /// representation and build a `Timeline` from it.
    fn import_aaf(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        use crate::clip::{Clip, MediaSource};
        use oximedia_aaf::{AafReader, TimelineConverter};
        use oximedia_core::Rational;

        let mut reader = AafReader::open(path)
            .map_err(|e| TimelineError::ImportError(format!("AAF open error: {e}")))?;
        let aaf = reader
            .read()
            .map_err(|e| TimelineError::ImportError(format!("AAF read error: {e}")))?;

        let aaf_tl = TimelineConverter::convert(&aaf)
            .map_err(|e| TimelineError::ImportError(format!("AAF convert error: {e}")))?;

        // Derive frame rate from the AAF edit rate (num/den).
        let fps = if let Some(ref ovr) = self.options.frame_rate_override {
            *ovr
        } else if let Some(ref er) = aaf_tl.edit_rate {
            Rational::new(i64::from(er.numerator), i64::from(er.denominator))
        } else {
            Rational::new(25, 1)
        };

        let sample_rate = self.options.sample_rate_override.unwrap_or(48000);
        let name = if aaf_tl.name.is_empty() {
            "Imported AAF".to_string()
        } else {
            aaf_tl.name.clone()
        };

        let mut timeline = Timeline::new(name, fps, sample_rate)?;
        let mut stats = ImportStats::default();

        for aaf_track in &aaf_tl.tracks {
            if aaf_track.clips.is_empty() {
                continue;
            }

            let is_video = aaf_track.track_type == "picture";
            let is_audio = aaf_track.track_type == "sound";

            if is_video && self.options.import_video {
                let track_id = timeline.add_video_track(aaf_track.name.as_str())?;
                stats.video_tracks += 1;

                for aaf_clip in &aaf_track.clips {
                    let tl_in = Position::new(aaf_clip.start.0);
                    let src_in = Position::new(aaf_clip.source_start.0);
                    let src_out = Position::new(
                        (aaf_clip.source_start.0 + aaf_clip.duration)
                            .max(aaf_clip.source_start.0 + 1),
                    );

                    let source =
                        MediaSource::file(std::path::PathBuf::from(aaf_clip.source_id.as_str()));
                    let clip =
                        Clip::new(aaf_clip.source_id.clone(), source, src_in, src_out, tl_in)?;
                    timeline.add_clip(track_id, clip)?;
                    stats.clips += 1;
                }
            } else if is_audio && self.options.import_audio {
                let track_id = timeline.add_audio_track(aaf_track.name.as_str())?;
                stats.audio_tracks += 1;

                for aaf_clip in &aaf_track.clips {
                    let tl_in = Position::new(aaf_clip.start.0);
                    let src_in = Position::new(aaf_clip.source_start.0);
                    let src_out = Position::new(
                        (aaf_clip.source_start.0 + aaf_clip.duration)
                            .max(aaf_clip.source_start.0 + 1),
                    );

                    let source =
                        MediaSource::file(std::path::PathBuf::from(aaf_clip.source_id.as_str()));
                    let clip =
                        Clip::new(aaf_clip.source_id.clone(), source, src_in, src_out, tl_in)?;
                    timeline.add_clip(track_id, clip)?;
                    stats.clips += 1;
                }
            }
        }

        Ok((timeline, stats))
    }

    /// Imports from `DaVinci` Resolve XML format (xmeml v5, structurally identical to Premiere).
    fn import_resolve_xml(&self, path: &Path) -> TimelineResult<(Timeline, ImportStats)> {
        let (name, timebase, video_items, audio_items) = parse_xmeml_file(path)?;
        build_timeline_from_xmeml(name, timebase, video_items, audio_items, &self.options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_import_format_from_path() {
        assert_eq!(
            ImportFormat::from_path(&PathBuf::from("test.edl")),
            Some(ImportFormat::Edl)
        );
        assert_eq!(
            ImportFormat::from_path(&PathBuf::from("test.fcpxml")),
            Some(ImportFormat::Fcpxml)
        );
        assert_eq!(
            ImportFormat::from_path(&PathBuf::from("test.aaf")),
            Some(ImportFormat::Aaf)
        );
        assert_eq!(ImportFormat::from_path(&PathBuf::from("test.txt")), None);
    }

    #[test]
    fn test_import_options_default() {
        let opts = ImportOptions::default();
        assert!(opts.import_audio);
        assert!(opts.import_video);
        assert!(opts.import_markers);
    }

    #[test]
    fn test_import_options_video_only() {
        let opts = ImportOptions::video_only();
        assert!(!opts.import_audio);
        assert!(opts.import_video);
    }

    #[test]
    fn test_import_options_audio_only() {
        let opts = ImportOptions::audio_only();
        assert!(opts.import_audio);
        assert!(!opts.import_video);
    }

    #[test]
    fn test_import_stats() {
        let stats = ImportStats::default();
        assert_eq!(stats.video_tracks, 0);
        assert_eq!(stats.audio_tracks, 0);
        assert_eq!(stats.clips, 0);
    }

    #[test]
    fn test_timeline_importer_creation() {
        let importer = TimelineImporter::with_defaults();
        assert!(importer.options.import_audio);
        assert!(importer.options.import_video);
    }

    #[test]
    fn test_parse_frame_duration_fractional() {
        let (num, den) = parse_frame_duration("100/2400s");
        assert_eq!(num, 2400);
        assert_eq!(den, 100);
    }

    #[test]
    fn test_parse_frame_duration_integer() {
        let (num, den) = parse_frame_duration("25s");
        assert_eq!(num, 25);
        assert_eq!(den, 1);
    }

    #[test]
    fn test_fcpxml_time_to_frames_fractional() {
        // "100/2400s" at 24fps → 1 frame
        let frames = fcpxml_time_to_frames("100/2400s", 24, 1);
        assert_eq!(frames, 1);
    }

    #[test]
    fn test_fcpxml_time_to_frames_seconds() {
        // "1s" at 24fps → 24 frames
        let frames = fcpxml_time_to_frames("1s", 24, 1);
        assert_eq!(frames, 24);
    }

    #[test]
    fn test_fcpxml_time_zero() {
        let frames = fcpxml_time_to_frames("0s", 24, 1);
        assert_eq!(frames, 0);
    }

    #[test]
    fn test_import_fcpxml_from_string() {
        // Write a minimal FCPXML to a temp file and import it.
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="FFVideoFormat1080p24" frameDuration="100/2400s" width="1920" height="1080"/>
    <asset id="r2" name="clip1" src="file:///tmp/clip1.mov" duration="2400/2400s"/>
  </resources>
  <library>
    <event name="MyEvent">
      <project name="My Project">
        <sequence duration="2400/2400s" tcFormat="NDF" tcStart="0s" format="r1">
          <spine>
            <asset-clip ref="r2" name="clip1" offset="0s" duration="1200/2400s" start="0s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;

        let tmp = std::env::temp_dir().join("test_import.fcpxml");
        std::fs::write(&tmp, xml).expect("should succeed in test");

        let importer = TimelineImporter::with_defaults();
        let result = importer.import_fcpxml(&tmp);
        assert!(result.is_ok(), "import failed: {result:?}");
        let (timeline, stats) = result.expect("should succeed in test");
        assert_eq!(timeline.name, "My Project");
        assert_eq!(stats.clips, 1);
        assert_eq!(stats.video_tracks, 1);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_import_premiere_xml_from_string() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
  <sequence>
    <name>Test Sequence</name>
    <timebase>24</timebase>
    <media>
      <video>
        <track>
          <clipitem>
            <name>My Clip</name>
            <file><pathurl>file:///tmp/myclip.mov</pathurl></file>
            <in>0</in>
            <out>48</out>
            <start>0</start>
            <end>48</end>
          </clipitem>
        </track>
      </video>
    </media>
  </sequence>
</xmeml>"#;

        let tmp = std::env::temp_dir().join("test_import.xml");
        std::fs::write(&tmp, xml).expect("should succeed in test");

        let importer = TimelineImporter::with_defaults();
        let result = importer.import_premiere_xml(&tmp);
        assert!(result.is_ok(), "import failed: {result:?}");
        let (timeline, stats) = result.expect("should succeed in test");
        assert_eq!(timeline.name, "Test Sequence");
        assert_eq!(stats.clips, 1);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_import_resolve_xml_from_string() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>Resolve Timeline</name>
    <timebase>25</timebase>
    <media>
      <video>
        <track>
          <clipitem>
            <name>Shot 01</name>
            <file><pathurl>file:///tmp/shot01.mov</pathurl></file>
            <in>0</in>
            <out>50</out>
            <start>0</start>
            <end>50</end>
          </clipitem>
        </track>
      </video>
    </media>
  </sequence>
</xmeml>"#;

        let tmp = std::env::temp_dir().join("test_resolve_import.xml");
        std::fs::write(&tmp, xml).expect("should succeed in test");

        let importer = TimelineImporter::with_defaults();
        let result = importer.import_resolve_xml(&tmp);
        assert!(result.is_ok(), "import failed: {result:?}");
        let (timeline, stats) = result.expect("should succeed in test");
        assert_eq!(timeline.name, "Resolve Timeline");
        assert_eq!(stats.clips, 1);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_import_edl_from_file() {
        // Write a minimal CMX 3600 EDL to a temp file and import it.
        let edl_text = "TITLE: Test EDL\nFCM: NON-DROP FRAME\n\n\
001  A001     V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n\
* FROM CLIP NAME: shot001.mov\n\
002  A002     V     C        01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00\n\
* FROM CLIP NAME: shot002.mov\n";

        let tmp = std::env::temp_dir().join("test_import.edl");
        std::fs::write(&tmp, edl_text).expect("should succeed in test");

        let importer = TimelineImporter::with_defaults();
        let result = importer.import_edl(&tmp);
        assert!(result.is_ok(), "EDL import failed: {result:?}");
        let (timeline, stats) = result.expect("should succeed in test");
        assert_eq!(timeline.name, "Test EDL");
        assert_eq!(stats.clips, 2, "Expected 2 clips");
        assert_eq!(stats.video_tracks, 1, "Expected 1 video track");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_import_unsupported_format() {
        let importer = TimelineImporter::with_defaults();
        let result = importer.import_file(&PathBuf::from("test.xyz"));
        assert!(result.is_err(), "Should fail for unsupported format");
    }
}
