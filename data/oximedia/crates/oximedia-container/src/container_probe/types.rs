//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Detailed structural information about a container, produced after a
/// more thorough header scan than a simple magic-byte probe.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Short format name (e.g. `"matroska"`, `"mp4"`, `"ogg"`).
    format_name: String,
    /// Total number of tracks (all types).
    total_tracks: usize,
    /// Number of video tracks.
    video_count: usize,
    /// Number of audio tracks.
    audio_count: usize,
    /// Total container duration in milliseconds, if signalled.
    duration_ms: Option<u64>,
    /// Container file size in bytes, if known.
    file_size: Option<u64>,
}
impl ContainerInfo {
    /// Creates a new `ContainerInfo`.
    #[must_use]
    pub fn new(format_name: impl Into<String>) -> Self {
        Self {
            format_name: format_name.into(),
            total_tracks: 0,
            video_count: 0,
            audio_count: 0,
            duration_ms: None,
            file_size: None,
        }
    }
    /// Sets video and audio track counts, automatically deriving `total_tracks`.
    #[must_use]
    pub fn with_tracks(mut self, video: usize, audio: usize) -> Self {
        self.video_count = video;
        self.audio_count = audio;
        self.total_tracks = video + audio;
        self
    }
    /// Sets the duration.
    #[must_use]
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }
    /// Sets the file size.
    #[must_use]
    pub fn with_file_size(mut self, bytes: u64) -> Self {
        self.file_size = Some(bytes);
        self
    }
    /// Returns the short format name.
    #[must_use]
    pub fn format_name(&self) -> &str {
        &self.format_name
    }
    /// Returns the total track count (all types).
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.total_tracks
    }
    /// Returns the number of video tracks.
    #[must_use]
    pub fn video_count(&self) -> usize {
        self.video_count
    }
    /// Returns the number of audio tracks.
    #[must_use]
    pub fn audio_count(&self) -> usize {
        self.audio_count
    }
    /// Returns the duration in milliseconds, if known.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }
    /// Estimates the average bit rate in kbps from file size and duration.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_bitrate_kbps(&self) -> Option<f64> {
        match (self.file_size, self.duration_ms) {
            (Some(bytes), Some(ms)) if ms > 0 => Some((bytes as f64 * 8.0) / (ms as f64)),
            _ => None,
        }
    }
}
/// Summary flags produced by probing a container's header region.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerProbeResult {
    /// Whether at least one video track was detected.
    pub video_present: bool,
    /// Whether at least one audio track was detected.
    pub audio_present: bool,
    /// Whether at least one subtitle track was detected.
    pub subtitle_present: bool,
    /// Confidence of the format detection in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Raw format name string as reported by the container layer.
    pub format_label: String,
}
impl ContainerProbeResult {
    /// Creates a new probe result with default confidence of 1.0.
    #[must_use]
    pub fn new(format_label: impl Into<String>) -> Self {
        Self {
            video_present: false,
            audio_present: false,
            subtitle_present: false,
            confidence: 1.0,
            format_label: format_label.into(),
        }
    }
    /// Returns `true` when at least one video track was detected.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video_present
    }
    /// Returns `true` when at least one audio track was detected.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_present
    }
    /// Returns `true` for multimedia containers that have both video and audio.
    #[must_use]
    pub fn is_av(&self) -> bool {
        self.video_present && self.audio_present
    }
    /// Returns `true` when confidence is at or above `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}
/// A thin prober that inspects raw bytes and fills a `ContainerInfo`.
#[derive(Debug, Default)]
pub struct ContainerProber {
    probed_count: usize,
}
impl ContainerProber {
    /// Creates a new `ContainerProber`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of containers probed so far.
    #[must_use]
    pub fn probed_count(&self) -> usize {
        self.probed_count
    }
    /// Inspects the first bytes of a container and returns a
    /// `ContainerProbeResult`.
    ///
    /// Detection is based on well-known magic sequences:
    /// - `[0x1A, 0x45, 0xDF, 0xA3]` → Matroska / `WebM`
    /// - `[0x66, 0x4C, 0x61, 0x43]` (`fLaC`) → FLAC
    /// - `[0x4F, 0x67, 0x67, 0x53]` (`OggS`) → Ogg
    /// - `[0x52, 0x49, 0x46, 0x46]` (`RIFF`) → WAV
    /// - `[0x00, 0x00, 0x00, _, 0x66, 0x74, 0x79, 0x70]` → MP4/ftyp
    pub fn probe_header(&mut self, header: &[u8]) -> ContainerProbeResult {
        self.probed_count += 1;
        if header.len() >= 4 && header[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            let mut r = ContainerProbeResult::new("matroska");
            r.video_present = true;
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"fLaC" {
            let mut r = ContainerProbeResult::new("flac");
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"OggS" {
            let mut r = ContainerProbeResult::new("ogg");
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"RIFF" {
            let mut r = ContainerProbeResult::new("wav");
            r.audio_present = true;
            return r;
        }
        if header.len() >= 8 && &header[4..8] == b"ftyp" {
            let mut r = ContainerProbeResult::new("mp4");
            r.video_present = true;
            r.audio_present = true;
            return r;
        }
        let mut r = ContainerProbeResult::new("unknown");
        r.confidence = 0.0;
        r
    }
}
/// Rich container information returned by [`super::multi_format::MultiFormatProber`].
#[derive(Debug, Clone, Default)]
pub struct DetailedContainerInfo {
    /// Short format name (`"mp4"`, `"mkv"`, `"mpeg-ts"`, `"webm"`, `"ogg"`,
    /// `"wav"`, `"flac"`, `"unknown"`).
    pub format: String,
    /// Total duration in milliseconds, if signalled.
    pub duration_ms: Option<u64>,
    /// Overall bitrate in kbps, if estimable from file_size_bytes + duration_ms.
    pub bitrate_kbps: Option<u32>,
    /// Discovered streams.
    pub streams: Vec<DetailedStreamInfo>,
    /// Key/value metadata extracted from the container header.
    pub metadata: std::collections::HashMap<String, String>,
    /// Byte length of the input slice.
    pub file_size_bytes: u64,
}
/// Detailed information about one media stream found inside a container.
#[derive(Debug, Clone, Default)]
pub struct DetailedStreamInfo {
    /// Zero-based stream index.
    pub index: u32,
    /// Stream type: `"video"`, `"audio"`, `"subtitle"`, or `"data"`.
    pub stream_type: String,
    /// Short codec name (e.g. `"av1"`, `"opus"`, `"flac"`).
    pub codec: String,
    /// ISO 639-2 language tag, if present.
    pub language: Option<String>,
    /// Stream duration in milliseconds, if known.
    pub duration_ms: Option<u64>,
    /// Average bitrate in kbps, if estimable.
    pub bitrate_kbps: Option<u32>,
    /// Frame width in pixels.
    pub width: Option<u32>,
    /// Frame height in pixels.
    pub height: Option<u32>,
    /// Frames per second.
    pub fps: Option<f32>,
    /// Pixel format string (e.g. `"yuv420p"`).
    pub pixel_format: Option<String>,
    /// Audio sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    pub channels: Option<u8>,
    /// Sample format string (e.g. `"s16"`).
    pub sample_format: Option<String>,
}
/// Per-second bitrate window statistics for one stream.
///
/// Produced by [`super::stats::probe_detailed`].  The histogram divides the stream into
/// 1-second windows and records the number of bits observed in each window.
#[derive(Debug, Clone)]
pub struct DetailedStreamStats {
    /// Zero-based stream index (matching the ordering in [`DetailedContainerInfo::streams`]).
    pub stream_index: usize,
    /// Short codec name (e.g. `"av1"`, `"opus"`, `"flac"`).
    pub codec_id: String,
    /// Stream duration in fractional seconds (0.0 if unknown).
    pub duration_s: f64,
    /// Bitrate window size in seconds (always 1.0 in current implementation).
    pub bitrate_window_s: f64,
    /// Number of bits per window, one entry per complete second.
    pub bitrate_histogram: Vec<u64>,
    /// Mean bitrate across all windows (bits per second).
    pub bitrate_mean: f64,
    /// Median (P50) bitrate (bits per second).
    pub bitrate_p50: f64,
    /// 95th-percentile bitrate (bits per second).
    pub bitrate_p95: f64,
    /// Peak bitrate across all windows (bits per second).
    pub bitrate_max: f64,
    /// Sorted list of inter-keyframe intervals in seconds.  `None` for
    /// audio/data streams where keyframes are not meaningful.
    pub keyframe_intervals_s: Option<Vec<f64>>,
    /// Mean inter-keyframe interval (seconds).  `None` when `keyframe_intervals_s` is `None`
    /// or empty (fewer than two keyframes observed).
    pub keyframe_interval_mean: Option<f64>,
    /// Median (P50) inter-keyframe interval (seconds).
    pub keyframe_interval_p50: Option<f64>,
    /// 95th-percentile inter-keyframe interval (seconds).
    pub keyframe_interval_p95: Option<f64>,
    /// Maximum inter-keyframe interval (seconds).
    pub keyframe_interval_max: Option<f64>,
}
/// Result of a container integrity check.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegrityCheckResult {
    /// Whether the container passes structural validation.
    pub valid: bool,
    /// List of issues found during validation.
    pub issues: Vec<String>,
    /// Overall integrity score (0.0 = completely corrupted, 1.0 = perfect).
    pub score: f64,
}
impl IntegrityCheckResult {
    /// Creates a new passing result.
    #[must_use]
    pub fn ok() -> Self {
        Self {
            valid: true,
            issues: Vec::new(),
            score: 1.0,
        }
    }
    /// Adds an issue and adjusts the score.
    pub fn add_issue(&mut self, issue: impl Into<String>, severity: f64) {
        self.issues.push(issue.into());
        self.score = (self.score - severity).max(0.0);
        if self.score < 0.5 {
            self.valid = false;
        }
    }
}
