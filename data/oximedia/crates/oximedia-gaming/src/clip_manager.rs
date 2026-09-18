//! Gaming clip and highlight management.
//!
//! Provides types for defining how clips are triggered, assembling highlight
//! reels, choosing export formats, and a real-time clip buffer that retains
//! the last N seconds of encoded frames for instant clip saving.

use crate::{GamingError, GamingResult};
use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ClipTrigger
// ---------------------------------------------------------------------------

/// What caused a gaming clip to be saved.
#[derive(Debug, Clone, PartialEq)]
pub enum ClipTrigger {
    /// Manually triggered by the user.
    Manual,
    /// An in-game achievement was unlocked.
    Achievement,
    /// A kill-streak of the given count was reached.
    KillStreak(u32),
    /// Player health dropped at or below the given threshold (0.0-1.0).
    HealthThreshold(f32),
    /// A custom/plugin-defined trigger.
    Custom(String),
}

impl ClipTrigger {
    /// Returns `true` for triggers that fire automatically without user input
    /// (`Achievement`, `KillStreak`, `HealthThreshold`).
    #[must_use]
    pub fn is_automatic(&self) -> bool {
        matches!(
            self,
            Self::Achievement | Self::KillStreak(_) | Self::HealthThreshold(_)
        )
    }
}

// ---------------------------------------------------------------------------
// GamingClip
// ---------------------------------------------------------------------------

/// A saved gaming clip with timing metadata.
#[derive(Debug, Clone)]
pub struct GamingClip {
    /// Unique clip identifier.
    pub id: u64,
    /// What caused this clip to be saved.
    pub triggered_by: ClipTrigger,
    /// Milliseconds of footage before the trigger point.
    pub pre_roll_ms: u32,
    /// Milliseconds of footage after the trigger point.
    pub post_roll_ms: u32,
    /// Clip start time in milliseconds since recording began.
    pub start_ms: u64,
    /// Clip end time in milliseconds since recording began.
    pub end_ms: u64,
}

impl GamingClip {
    /// Total clip duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` when the clip has a non-zero pre-roll buffer.
    #[must_use]
    pub fn has_pre_roll(&self) -> bool {
        self.pre_roll_ms > 0
    }
}

// ---------------------------------------------------------------------------
// HighlightReel
// ---------------------------------------------------------------------------

/// An ordered collection of clips assembled into a highlight reel.
#[derive(Debug, Default)]
pub struct HighlightReel {
    /// All clips in the reel.
    pub clips: Vec<GamingClip>,
    /// Desired maximum reel duration in milliseconds.
    pub target_duration_ms: u64,
}

impl HighlightReel {
    /// Create a new highlight reel with a target maximum duration.
    #[must_use]
    pub fn new(target_duration_ms: u64) -> Self {
        Self {
            clips: Vec::new(),
            target_duration_ms,
        }
    }

    /// Append a clip to the reel.
    pub fn add_clip(&mut self, clip: GamingClip) {
        self.clips.push(clip);
    }

    /// Select the `max_clips` most-recently-added clips.
    ///
    /// The returned vector is in chronological order (oldest first).
    #[must_use]
    pub fn auto_select(&self, max_clips: usize) -> Vec<&GamingClip> {
        let skip = self.clips.len().saturating_sub(max_clips);
        self.clips.iter().skip(skip).collect()
    }

    /// Sum of all clip durations in milliseconds.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.clips.iter().map(GamingClip::duration_ms).sum()
    }

    /// Returns `true` when the total duration exceeds the target.
    #[must_use]
    pub fn needs_trim(&self) -> bool {
        self.total_duration_ms() > self.target_duration_ms
    }
}

// ---------------------------------------------------------------------------
// ClipExportFormat
// ---------------------------------------------------------------------------

/// Container / codec combination used when exporting a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipExportFormat {
    /// MP4 container with H.264 video.
    Mp4H264,
    /// MP4 container with H.265 / HEVC video.
    Mp4Hevc,
    /// `WebM` container (VP8/VP9).
    WebM,
    /// Animated GIF (no audio, limited colour depth).
    Gif,
}

impl ClipExportFormat {
    /// Canonical file extension for this format (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Mp4H264 | Self::Mp4Hevc => "mp4",
            Self::WebM => "webm",
            Self::Gif => "gif",
        }
    }

    /// Returns `true` when the format can carry HDR metadata.
    ///
    /// Only `Mp4Hevc` supports HDR in this implementation.
    #[must_use]
    pub fn supports_hdr(&self) -> bool {
        *self == Self::Mp4Hevc
    }
}

// ---------------------------------------------------------------------------
// ClipFrame -- a frame stored in the clip ring buffer
// ---------------------------------------------------------------------------

/// A single encoded frame stored in the clip buffer.
#[derive(Debug, Clone)]
pub struct ClipFrame {
    /// Encoded data (from the encoder pipeline).
    pub data: Vec<u8>,
    /// Presentation timestamp relative to session start.
    pub timestamp: Duration,
    /// Whether this frame is a keyframe.
    pub is_keyframe: bool,
    /// Sequence number.
    pub sequence: u64,
}

// ---------------------------------------------------------------------------
// ClipMetadata
// ---------------------------------------------------------------------------

/// Metadata attached to a saved clip.
#[derive(Debug, Clone)]
pub struct ClipMetadata {
    /// Clip title / label.
    pub title: String,
    /// Game name.
    pub game: String,
    /// Resolution at time of capture.
    pub resolution: (u32, u32),
    /// Framerate at time of capture.
    pub framerate: u32,
    /// Duration of the clip.
    pub duration: Duration,
    /// What triggered the clip.
    pub trigger: ClipTrigger,
    /// Number of frames in the clip.
    pub frame_count: usize,
    /// Total bytes of encoded data.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// SavedClip -- result of extracting from the buffer
// ---------------------------------------------------------------------------

/// A clip extracted from the ring buffer, ready for export.
#[derive(Debug)]
pub struct SavedClip {
    /// The encoded frames.
    pub frames: Vec<ClipFrame>,
    /// Metadata about the clip.
    pub metadata: ClipMetadata,
}

// ---------------------------------------------------------------------------
// ClipBufferConfig
// ---------------------------------------------------------------------------

/// Configuration for the clip ring buffer.
#[derive(Debug, Clone)]
pub struct ClipBufferConfig {
    /// Maximum duration to retain (seconds).
    pub max_duration_secs: u32,
    /// Estimated framerate (for capacity).
    pub framerate: u32,
    /// Maximum memory budget in bytes.
    pub max_bytes: usize,
    /// Game name for metadata.
    pub game_name: String,
    /// Capture resolution for metadata.
    pub resolution: (u32, u32),
}

impl Default for ClipBufferConfig {
    fn default() -> Self {
        Self {
            max_duration_secs: 30,
            framerate: 60,
            max_bytes: 50 * 1024 * 1024, // 50 MB
            game_name: String::new(),
            resolution: (1920, 1080),
        }
    }
}

// ---------------------------------------------------------------------------
// ClipBuffer -- the actual ring buffer
// ---------------------------------------------------------------------------

/// A ring buffer that stores the last N seconds of encoded frames, allowing
/// instant clip creation on demand.
pub struct ClipBuffer {
    config: ClipBufferConfig,
    frames: VecDeque<ClipFrame>,
    max_frames: usize,
    total_bytes: usize,
    next_sequence: u64,
    enabled: bool,
    /// Saved clips metadata (ID counter).
    next_clip_id: u64,
}

impl ClipBuffer {
    /// Create a new clip buffer.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: ClipBufferConfig) -> GamingResult<Self> {
        if config.max_duration_secs == 0 || config.max_duration_secs > 600 {
            return Err(GamingError::InvalidConfig(
                "Clip buffer duration must be 1-600 seconds".into(),
            ));
        }
        if config.framerate == 0 {
            return Err(GamingError::InvalidConfig(
                "Clip buffer framerate must be non-zero".into(),
            ));
        }

        let max_frames = (config.framerate as usize) * (config.max_duration_secs as usize);

        Ok(Self {
            config,
            frames: VecDeque::with_capacity(max_frames.min(4096)),
            max_frames,
            total_bytes: 0,
            next_sequence: 0,
            enabled: false,
            next_clip_id: 1,
        })
    }

    /// Enable the clip buffer.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the clip buffer and clear stored frames.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.clear();
    }

    /// Whether the buffer is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Push an encoded frame into the buffer.
    ///
    /// If the buffer is full (by frame count or byte budget), oldest frames
    /// are evicted. No-op if not enabled.
    pub fn push_frame(&mut self, data: Vec<u8>, timestamp: Duration, is_keyframe: bool) {
        if !self.enabled {
            return;
        }

        let frame_bytes = data.len();

        // Evict by frame count
        while self.frames.len() >= self.max_frames {
            if let Some(old) = self.frames.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
            }
        }

        // Evict by byte budget
        while self.total_bytes + frame_bytes > self.config.max_bytes && !self.frames.is_empty() {
            if let Some(old) = self.frames.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(old.data.len());
            }
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.total_bytes += frame_bytes;

        self.frames.push_back(ClipFrame {
            data,
            timestamp,
            is_keyframe,
            sequence: seq,
        });
    }

    /// Number of frames in the buffer.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Total bytes stored.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Actual duration covered by the buffer.
    #[must_use]
    pub fn buffered_duration(&self) -> Duration {
        if self.frames.len() < 2 {
            return Duration::ZERO;
        }
        let oldest = self.frames[0].timestamp;
        let newest = self
            .frames
            .back()
            .map(|f| f.timestamp)
            .unwrap_or(Duration::ZERO);
        newest.saturating_sub(oldest)
    }

    /// Save a clip of the last `duration` seconds with the given trigger and title.
    ///
    /// Returns a `SavedClip` containing all frames from the nearest keyframe
    /// before the requested window, along with metadata.
    ///
    /// # Errors
    ///
    /// Returns error if no frames are available.
    pub fn save_clip(
        &mut self,
        duration: Duration,
        trigger: ClipTrigger,
        title: &str,
    ) -> GamingResult<SavedClip> {
        if self.frames.is_empty() {
            return Err(GamingError::ReplayBufferError(
                "No frames in clip buffer".into(),
            ));
        }

        let newest_ts = self
            .frames
            .back()
            .map(|f| f.timestamp)
            .unwrap_or(Duration::ZERO);
        let cutoff = newest_ts.saturating_sub(duration);

        // Find first frame in the time window
        let first_in_window = self
            .frames
            .iter()
            .position(|f| f.timestamp >= cutoff)
            .unwrap_or(0);

        // Walk backwards to find nearest keyframe
        let mut start = first_in_window;
        for i in (0..=first_in_window).rev() {
            if self.frames[i].is_keyframe {
                start = i;
                break;
            }
        }

        let frames: Vec<ClipFrame> = self.frames.iter().skip(start).cloned().collect();
        let frame_count = frames.len();
        let clip_bytes: usize = frames.iter().map(|f| f.data.len()).sum();
        let clip_duration = if frame_count >= 2 {
            let first_ts = frames[0].timestamp;
            let last_ts = frames[frame_count - 1].timestamp;
            last_ts.saturating_sub(first_ts)
        } else {
            Duration::ZERO
        };

        let clip_id = self.next_clip_id;
        self.next_clip_id += 1;

        let metadata = ClipMetadata {
            title: if title.is_empty() {
                format!("clip_{clip_id}")
            } else {
                title.to_string()
            },
            game: self.config.game_name.clone(),
            resolution: self.config.resolution,
            framerate: self.config.framerate,
            duration: clip_duration,
            trigger,
            frame_count,
            total_bytes: clip_bytes,
        };

        Ok(SavedClip { frames, metadata })
    }

    /// Export a saved clip as raw bytes (header + frame data).
    ///
    /// Format: `[8 bytes: "OxiClip\0"] [metadata section] [frame data]`
    ///
    /// This is a simple binary format suitable for later transcoding.
    #[must_use]
    pub fn export_clip_bytes(clip: &SavedClip) -> Vec<u8> {
        let mut output = Vec::new();

        // Magic header
        output.extend_from_slice(b"OxiClip\0");

        // Metadata: resolution, framerate, frame count, total bytes
        output.extend_from_slice(&clip.metadata.resolution.0.to_le_bytes());
        output.extend_from_slice(&clip.metadata.resolution.1.to_le_bytes());
        output.extend_from_slice(&clip.metadata.framerate.to_le_bytes());
        output.extend_from_slice(&(clip.metadata.frame_count as u32).to_le_bytes());
        output.extend_from_slice(&(clip.metadata.total_bytes as u64).to_le_bytes());
        output.extend_from_slice(&clip.metadata.duration.as_millis().to_le_bytes());

        // Frame data: for each frame [4 bytes: size] [1 byte: flags] [data]
        for frame in &clip.frames {
            let size = frame.data.len() as u32;
            output.extend_from_slice(&size.to_le_bytes());
            let flags: u8 = if frame.is_keyframe { 0x01 } else { 0x00 };
            output.push(flags);
            output.extend_from_slice(&frame.data);
        }

        output
    }

    /// Clear all buffered frames.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.total_bytes = 0;
    }

    /// Returns comprehensive buffer statistics.
    #[must_use]
    pub fn statistics(&self) -> BufferStatistics {
        let keyframe_count = self.frames.iter().filter(|f| f.is_keyframe).count();
        let avg_frame_bytes = if self.frames.is_empty() {
            0
        } else {
            self.total_bytes / self.frames.len()
        };
        let oldest_timestamp = self.frames.front().map(|f| f.timestamp);
        let newest_timestamp = self.frames.back().map(|f| f.timestamp);

        BufferStatistics {
            frame_count: self.frames.len(),
            keyframe_count,
            total_bytes: self.total_bytes,
            avg_frame_bytes,
            max_bytes_budget: self.config.max_bytes,
            memory_usage_ratio: if self.config.max_bytes > 0 {
                self.total_bytes as f64 / self.config.max_bytes as f64
            } else {
                0.0
            },
            buffered_duration: self.buffered_duration(),
            max_duration: Duration::from_secs(u64::from(self.config.max_duration_secs)),
            duration_fill_ratio: {
                let max_dur = Duration::from_secs(u64::from(self.config.max_duration_secs));
                let buffered = self.buffered_duration();
                if max_dur.as_secs_f64() > 0.0 {
                    buffered.as_secs_f64() / max_dur.as_secs_f64()
                } else {
                    0.0
                }
            },
            max_frames: self.max_frames,
            frame_fill_ratio: if self.max_frames > 0 {
                self.frames.len() as f64 / self.max_frames as f64
            } else {
                0.0
            },
            oldest_timestamp,
            newest_timestamp,
            next_sequence: self.next_sequence,
            enabled: self.enabled,
            clips_saved: self.next_clip_id.saturating_sub(1),
        }
    }

    /// Update the buffer configuration without clearing frames.
    ///
    /// Only fields that do not affect in-flight frames are updated: `max_bytes`,
    /// `game_name`, and `resolution`. Duration/framerate changes require a new buffer.
    pub fn update_config(
        &mut self,
        max_bytes: Option<usize>,
        game_name: Option<String>,
        resolution: Option<(u32, u32)>,
    ) {
        if let Some(mb) = max_bytes {
            self.config.max_bytes = mb;
        }
        if let Some(gn) = game_name {
            self.config.game_name = gn;
        }
        if let Some(res) = resolution {
            self.config.resolution = res;
        }
    }

    /// Reconfigure buffer duration and framerate. This clears all buffered frames.
    ///
    /// # Errors
    ///
    /// Returns error if the new configuration is invalid.
    pub fn reconfigure(&mut self, max_duration_secs: u32, framerate: u32) -> GamingResult<()> {
        if max_duration_secs == 0 || max_duration_secs > 600 {
            return Err(GamingError::InvalidConfig(
                "Clip buffer duration must be 1-600 seconds".into(),
            ));
        }
        if framerate == 0 {
            return Err(GamingError::InvalidConfig(
                "Clip buffer framerate must be non-zero".into(),
            ));
        }
        self.config.max_duration_secs = max_duration_secs;
        self.config.framerate = framerate;
        self.max_frames = (framerate as usize) * (max_duration_secs as usize);
        self.clear();
        Ok(())
    }

    /// Extract frames from the buffer covering the requested duration without
    /// cloning (returns references). Useful for preview / scrubbing without
    /// allocating a full `SavedClip`.
    #[must_use]
    pub fn peek_frames(&self, duration: Duration) -> Vec<&ClipFrame> {
        if self.frames.is_empty() {
            return Vec::new();
        }
        let newest_ts = self
            .frames
            .back()
            .map(|f| f.timestamp)
            .unwrap_or(Duration::ZERO);
        let cutoff = newest_ts.saturating_sub(duration);

        self.frames
            .iter()
            .filter(|f| f.timestamp >= cutoff)
            .collect()
    }

    /// Return the number of keyframes currently in the buffer.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.frames.iter().filter(|f| f.is_keyframe).count()
    }

    /// Return the current configuration (read-only).
    #[must_use]
    pub fn config(&self) -> &ClipBufferConfig {
        &self.config
    }

    /// Return the number of frames that have been dropped (evicted) over the
    /// buffer's lifetime. Computed from the sequence counter minus frame count.
    #[must_use]
    pub fn frames_evicted(&self) -> u64 {
        self.next_sequence.saturating_sub(self.frames.len() as u64)
    }

    /// Save a clip and also return the binary export of that clip in one call.
    ///
    /// # Errors
    ///
    /// Returns error if no frames are available.
    pub fn save_and_export(
        &mut self,
        duration: Duration,
        trigger: ClipTrigger,
        title: &str,
    ) -> GamingResult<(SavedClip, Vec<u8>)> {
        let clip = self.save_clip(duration, trigger, title)?;
        let bytes = Self::export_clip_bytes(&clip);
        Ok((clip, bytes))
    }
}

// ---------------------------------------------------------------------------
// BufferStatistics
// ---------------------------------------------------------------------------

/// Comprehensive statistics about the clip ring buffer.
#[derive(Debug, Clone)]
pub struct BufferStatistics {
    /// Number of frames currently in the buffer.
    pub frame_count: usize,
    /// Number of keyframes currently in the buffer.
    pub keyframe_count: usize,
    /// Total bytes of encoded data in the buffer.
    pub total_bytes: usize,
    /// Average bytes per frame.
    pub avg_frame_bytes: usize,
    /// Maximum bytes budget.
    pub max_bytes_budget: usize,
    /// Ratio of used memory to budget (0.0 - 1.0+).
    pub memory_usage_ratio: f64,
    /// Actual duration covered by buffered frames.
    pub buffered_duration: Duration,
    /// Maximum configured duration.
    pub max_duration: Duration,
    /// Ratio of buffered duration to max duration.
    pub duration_fill_ratio: f64,
    /// Maximum number of frames the buffer can hold.
    pub max_frames: usize,
    /// Ratio of current frame count to max frames.
    pub frame_fill_ratio: f64,
    /// Timestamp of the oldest frame, if any.
    pub oldest_timestamp: Option<Duration>,
    /// Timestamp of the newest frame, if any.
    pub newest_timestamp: Option<Duration>,
    /// Next sequence number to be assigned.
    pub next_sequence: u64,
    /// Whether the buffer is enabled.
    pub enabled: bool,
    /// Number of clips saved from this buffer.
    pub clips_saved: u64,
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, start: u64, end: u64) -> GamingClip {
        GamingClip {
            id,
            triggered_by: ClipTrigger::Manual,
            pre_roll_ms: 5000,
            post_roll_ms: 3000,
            start_ms: start,
            end_ms: end,
        }
    }

    // ClipTrigger

    #[test]
    fn test_manual_not_automatic() {
        assert!(!ClipTrigger::Manual.is_automatic());
    }

    #[test]
    fn test_achievement_is_automatic() {
        assert!(ClipTrigger::Achievement.is_automatic());
    }

    #[test]
    fn test_kill_streak_is_automatic() {
        assert!(ClipTrigger::KillStreak(5).is_automatic());
    }

    #[test]
    fn test_health_threshold_is_automatic() {
        assert!(ClipTrigger::HealthThreshold(0.2).is_automatic());
    }

    #[test]
    fn test_custom_not_automatic() {
        assert!(!ClipTrigger::Custom("plugin".to_string()).is_automatic());
    }

    // GamingClip

    #[test]
    fn test_clip_duration_ms() {
        let c = make_clip(1, 10_000, 40_000);
        assert_eq!(c.duration_ms(), 30_000);
    }

    #[test]
    fn test_clip_duration_zero_end_before_start() {
        let c = make_clip(1, 50_000, 30_000);
        assert_eq!(c.duration_ms(), 0);
    }

    #[test]
    fn test_clip_has_pre_roll_true() {
        let c = make_clip(1, 0, 10_000);
        assert!(c.has_pre_roll());
    }

    #[test]
    fn test_clip_has_pre_roll_false() {
        let mut c = make_clip(1, 0, 10_000);
        c.pre_roll_ms = 0;
        assert!(!c.has_pre_roll());
    }

    // HighlightReel

    #[test]
    fn test_reel_total_duration() {
        let mut reel = HighlightReel::new(60_000);
        reel.add_clip(make_clip(1, 0, 10_000));
        reel.add_clip(make_clip(2, 10_000, 30_000));
        assert_eq!(reel.total_duration_ms(), 30_000);
    }

    #[test]
    fn test_reel_needs_trim_true() {
        let mut reel = HighlightReel::new(15_000);
        reel.add_clip(make_clip(1, 0, 10_000));
        reel.add_clip(make_clip(2, 10_000, 20_000));
        assert!(reel.needs_trim());
    }

    #[test]
    fn test_reel_needs_trim_false() {
        let mut reel = HighlightReel::new(60_000);
        reel.add_clip(make_clip(1, 0, 10_000));
        assert!(!reel.needs_trim());
    }

    #[test]
    fn test_reel_auto_select_count() {
        let mut reel = HighlightReel::new(120_000);
        for i in 0..5u64 {
            reel.add_clip(make_clip(i, i * 10_000, (i + 1) * 10_000));
        }
        let selected = reel.auto_select(3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_reel_auto_select_most_recent() {
        let mut reel = HighlightReel::new(120_000);
        reel.add_clip(make_clip(1, 0, 10_000));
        reel.add_clip(make_clip(2, 10_000, 20_000));
        reel.add_clip(make_clip(3, 20_000, 30_000));
        let selected = reel.auto_select(2);
        assert_eq!(selected[0].id, 2);
        assert_eq!(selected[1].id, 3);
    }

    #[test]
    fn test_reel_auto_select_more_than_available() {
        let mut reel = HighlightReel::new(120_000);
        reel.add_clip(make_clip(1, 0, 5_000));
        let selected = reel.auto_select(10);
        assert_eq!(selected.len(), 1);
    }

    // ClipExportFormat

    #[test]
    fn test_mp4_h264_extension() {
        assert_eq!(ClipExportFormat::Mp4H264.extension(), "mp4");
    }

    #[test]
    fn test_mp4_hevc_extension() {
        assert_eq!(ClipExportFormat::Mp4Hevc.extension(), "mp4");
    }

    #[test]
    fn test_webm_extension() {
        assert_eq!(ClipExportFormat::WebM.extension(), "webm");
    }

    #[test]
    fn test_gif_extension() {
        assert_eq!(ClipExportFormat::Gif.extension(), "gif");
    }

    #[test]
    fn test_hevc_supports_hdr() {
        assert!(ClipExportFormat::Mp4Hevc.supports_hdr());
    }

    #[test]
    fn test_h264_no_hdr() {
        assert!(!ClipExportFormat::Mp4H264.supports_hdr());
    }

    // ClipBuffer

    #[test]
    fn test_clip_buffer_creation() {
        let buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        assert!(!buf.is_enabled());
        assert_eq!(buf.frame_count(), 0);
    }

    #[test]
    fn test_clip_buffer_invalid_duration() {
        let cfg = ClipBufferConfig {
            max_duration_secs: 0,
            ..ClipBufferConfig::default()
        };
        assert!(ClipBuffer::new(cfg).is_err());
    }

    #[test]
    fn test_clip_buffer_invalid_framerate() {
        let cfg = ClipBufferConfig {
            framerate: 0,
            ..ClipBufferConfig::default()
        };
        assert!(ClipBuffer::new(cfg).is_err());
    }

    #[test]
    fn test_clip_buffer_push_when_disabled() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.push_frame(vec![0u8; 100], Duration::ZERO, true);
        assert_eq!(buf.frame_count(), 0);
    }

    #[test]
    fn test_clip_buffer_push_and_count() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        for i in 0..10 {
            buf.push_frame(vec![0u8; 500], Duration::from_millis(i * 16), i % 30 == 0);
        }
        assert_eq!(buf.frame_count(), 10);
        assert_eq!(buf.total_bytes(), 5000);
    }

    #[test]
    fn test_clip_buffer_eviction_by_count() {
        let cfg = ClipBufferConfig {
            max_duration_secs: 5,
            framerate: 2, // 2 * 5 = 10 frames max
            max_bytes: 1_000_000,
            ..ClipBufferConfig::default()
        };
        let mut buf = ClipBuffer::new(cfg).expect("valid");
        buf.enable();
        for i in 0..20u64 {
            buf.push_frame(vec![0u8; 50], Duration::from_millis(i * 500), i % 5 == 0);
        }
        assert_eq!(buf.frame_count(), 10);
    }

    #[test]
    fn test_clip_buffer_eviction_by_bytes() {
        let cfg = ClipBufferConfig {
            max_duration_secs: 60,
            framerate: 60,
            max_bytes: 3000,
            ..ClipBufferConfig::default()
        };
        let mut buf = ClipBuffer::new(cfg).expect("valid");
        buf.enable();
        for i in 0..10u64 {
            buf.push_frame(vec![0u8; 1000], Duration::from_millis(i * 16), true);
        }
        assert!(buf.total_bytes() <= 3000);
    }

    #[test]
    fn test_clip_buffer_buffered_duration() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::from_millis(100), true);
        buf.push_frame(vec![0], Duration::from_millis(600), false);
        assert_eq!(buf.buffered_duration(), Duration::from_millis(500));
    }

    #[test]
    fn test_clip_buffer_save_clip() {
        let cfg = ClipBufferConfig {
            game_name: "TestGame".into(),
            resolution: (1920, 1080),
            ..ClipBufferConfig::default()
        };
        let mut buf = ClipBuffer::new(cfg).expect("valid");
        buf.enable();

        // Push 10 frames at 100ms intervals, keyframe every 5
        for i in 0..10u64 {
            buf.push_frame(
                vec![i as u8; 200],
                Duration::from_millis(i * 100),
                i % 5 == 0,
            );
        }

        let clip = buf
            .save_clip(Duration::from_millis(300), ClipTrigger::Manual, "epic")
            .expect("save clip");

        assert!(!clip.frames.is_empty());
        assert!(clip.frames[0].is_keyframe);
        assert_eq!(clip.metadata.title, "epic");
        assert_eq!(clip.metadata.game, "TestGame");
        assert_eq!(clip.metadata.resolution, (1920, 1080));
    }

    #[test]
    fn test_clip_buffer_save_clip_empty() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        let result = buf.save_clip(Duration::from_secs(5), ClipTrigger::Manual, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_clip_buffer_save_clip_default_title() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::ZERO, true);

        let clip = buf
            .save_clip(Duration::from_secs(5), ClipTrigger::Achievement, "")
            .expect("save");
        assert!(clip.metadata.title.starts_with("clip_"));
    }

    #[test]
    fn test_clip_buffer_export_bytes() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![42u8; 100], Duration::ZERO, true);
        buf.push_frame(vec![99u8; 100], Duration::from_millis(16), false);

        let clip = buf
            .save_clip(Duration::from_secs(1), ClipTrigger::Manual, "test")
            .expect("save");
        let bytes = ClipBuffer::export_clip_bytes(&clip);

        // Should start with magic
        assert_eq!(&bytes[..8], b"OxiClip\0");
        // Should contain frame data
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_clip_buffer_clear() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0; 100], Duration::ZERO, true);
        assert_eq!(buf.frame_count(), 1);
        buf.clear();
        assert_eq!(buf.frame_count(), 0);
        assert_eq!(buf.total_bytes(), 0);
    }

    #[test]
    fn test_clip_buffer_disable_clears() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0; 100], Duration::ZERO, true);
        buf.disable();
        assert!(!buf.is_enabled());
        assert_eq!(buf.frame_count(), 0);
    }

    #[test]
    fn test_clip_metadata_trigger() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::ZERO, true);

        let clip = buf
            .save_clip(
                Duration::from_secs(1),
                ClipTrigger::KillStreak(5),
                "pentakill",
            )
            .expect("save");
        assert_eq!(clip.metadata.trigger, ClipTrigger::KillStreak(5));
    }

    #[test]
    fn test_clip_ids_increment() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::ZERO, true);

        let c1 = buf
            .save_clip(Duration::from_secs(1), ClipTrigger::Manual, "a")
            .expect("save 1");
        let c2 = buf
            .save_clip(Duration::from_secs(1), ClipTrigger::Manual, "b")
            .expect("save 2");

        // Metadata titles should differ (different clip IDs internally)
        assert_ne!(c1.metadata.title, c2.metadata.title);
    }

    // -- BufferStatistics --

    #[test]
    fn test_buffer_statistics_empty() {
        let buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        let stats = buf.statistics();
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.keyframe_count, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.avg_frame_bytes, 0);
        assert!(!stats.enabled);
        assert_eq!(stats.clips_saved, 0);
        assert!(stats.oldest_timestamp.is_none());
        assert!(stats.newest_timestamp.is_none());
    }

    #[test]
    fn test_buffer_statistics_with_frames() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        for i in 0..20u64 {
            buf.push_frame(vec![0u8; 500], Duration::from_millis(i * 16), i % 10 == 0);
        }
        let stats = buf.statistics();
        assert_eq!(stats.frame_count, 20);
        assert_eq!(stats.keyframe_count, 2); // i=0, i=10
        assert_eq!(stats.total_bytes, 10000);
        assert_eq!(stats.avg_frame_bytes, 500);
        assert!(stats.enabled);
        assert!(stats.memory_usage_ratio > 0.0);
        assert!(stats.duration_fill_ratio > 0.0);
        assert!(stats.frame_fill_ratio > 0.0);
        assert!(stats.oldest_timestamp.is_some());
        assert!(stats.newest_timestamp.is_some());
    }

    #[test]
    fn test_buffer_statistics_after_clip_save() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0u8; 100], Duration::ZERO, true);
        let _ = buf
            .save_clip(Duration::from_secs(1), ClipTrigger::Manual, "x")
            .expect("save");
        let _ = buf
            .save_clip(Duration::from_secs(1), ClipTrigger::Manual, "y")
            .expect("save");
        let stats = buf.statistics();
        assert_eq!(stats.clips_saved, 2);
    }

    #[test]
    fn test_buffer_memory_usage_ratio() {
        let cfg = ClipBufferConfig {
            max_bytes: 10000,
            ..ClipBufferConfig::default()
        };
        let mut buf = ClipBuffer::new(cfg).expect("valid");
        buf.enable();
        buf.push_frame(vec![0u8; 5000], Duration::ZERO, true);
        let stats = buf.statistics();
        assert!((stats.memory_usage_ratio - 0.5).abs() < 0.01);
    }

    // -- update_config --

    #[test]
    fn test_update_config_game_name() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.update_config(None, Some("NewGame".into()), None);
        assert_eq!(buf.config().game_name, "NewGame");
    }

    #[test]
    fn test_update_config_resolution() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.update_config(None, None, Some((3840, 2160)));
        assert_eq!(buf.config().resolution, (3840, 2160));
    }

    #[test]
    fn test_update_config_max_bytes() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.update_config(Some(100_000_000), None, None);
        assert_eq!(buf.config().max_bytes, 100_000_000);
    }

    // -- reconfigure --

    #[test]
    fn test_reconfigure_clears_frames() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0u8; 100], Duration::ZERO, true);
        assert_eq!(buf.frame_count(), 1);
        buf.reconfigure(10, 30).expect("reconfigure");
        assert_eq!(buf.frame_count(), 0);
    }

    #[test]
    fn test_reconfigure_invalid_duration() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        assert!(buf.reconfigure(0, 30).is_err());
        assert!(buf.reconfigure(700, 30).is_err());
    }

    #[test]
    fn test_reconfigure_invalid_framerate() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        assert!(buf.reconfigure(30, 0).is_err());
    }

    // -- peek_frames --

    #[test]
    fn test_peek_frames_empty() {
        let buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        assert!(buf.peek_frames(Duration::from_secs(5)).is_empty());
    }

    #[test]
    fn test_peek_frames_subset() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        for i in 0..10u64 {
            buf.push_frame(vec![i as u8], Duration::from_secs(i), i == 0);
        }
        // Last 3 seconds: ts >= 6 => frames at t=6,7,8,9 = 4 frames
        let peeked = buf.peek_frames(Duration::from_secs(3));
        assert_eq!(peeked.len(), 4);
    }

    // -- keyframe_count --

    #[test]
    fn test_keyframe_count() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::from_millis(0), true);
        buf.push_frame(vec![0], Duration::from_millis(16), false);
        buf.push_frame(vec![0], Duration::from_millis(32), false);
        buf.push_frame(vec![0], Duration::from_millis(48), true);
        assert_eq!(buf.keyframe_count(), 2);
    }

    // -- frames_evicted --

    #[test]
    fn test_frames_evicted_none() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![0], Duration::ZERO, true);
        assert_eq!(buf.frames_evicted(), 0);
    }

    #[test]
    fn test_frames_evicted_some() {
        let cfg = ClipBufferConfig {
            max_duration_secs: 1,
            framerate: 5, // max 5 frames
            max_bytes: 10_000_000,
            ..ClipBufferConfig::default()
        };
        let mut buf = ClipBuffer::new(cfg).expect("valid");
        buf.enable();
        for i in 0..10u64 {
            buf.push_frame(vec![0u8; 50], Duration::from_millis(i * 200), true);
        }
        assert_eq!(buf.frame_count(), 5);
        assert_eq!(buf.frames_evicted(), 5);
    }

    // -- save_and_export --

    #[test]
    fn test_save_and_export() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        buf.push_frame(vec![42u8; 200], Duration::ZERO, true);
        buf.push_frame(vec![99u8; 200], Duration::from_millis(16), false);

        let (clip, bytes) = buf
            .save_and_export(Duration::from_secs(1), ClipTrigger::Manual, "test_export")
            .expect("save_and_export");

        assert_eq!(clip.metadata.title, "test_export");
        assert_eq!(&bytes[..8], b"OxiClip\0");
        assert!(bytes.len() > 50);
    }

    #[test]
    fn test_save_and_export_empty_fails() {
        let mut buf = ClipBuffer::new(ClipBufferConfig::default()).expect("valid");
        buf.enable();
        assert!(buf
            .save_and_export(Duration::from_secs(1), ClipTrigger::Manual, "")
            .is_err());
    }
}
