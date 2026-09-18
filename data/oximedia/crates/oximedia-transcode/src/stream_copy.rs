//! Stream copy (passthrough) mode for transcoding.
//!
//! When the source codec matches the desired output codec and no filters
//! (resize, crop, tone-map, etc.) are required, the stream can be copied
//! bit-for-bit without re-encoding. This is dramatically faster and
//! preserves the original quality.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Describes which streams should be copied vs. re-encoded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamCopyMode {
    /// Re-encode everything (default transcoding behaviour).
    ReEncode,
    /// Copy the video stream verbatim; re-encode audio.
    CopyVideo,
    /// Copy the audio stream verbatim; re-encode video.
    CopyAudio,
    /// Copy both video and audio streams (remux only).
    CopyAll,
    /// Automatic: detect matching codecs and copy where possible.
    Auto,
}

impl Default for StreamCopyMode {
    fn default() -> Self {
        Self::ReEncode
    }
}

/// Information about a single media stream used for copy-eligibility checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    /// Codec identifier (e.g. "av1", "vp9", "opus", "flac").
    pub codec: String,
    /// Stream type.
    pub stream_type: StreamType,
    /// Width in pixels (video only).
    pub width: Option<u32>,
    /// Height in pixels (video only).
    pub height: Option<u32>,
    /// Sample rate in Hz (audio only).
    pub sample_rate: Option<u32>,
    /// Number of audio channels (audio only).
    pub channels: Option<u8>,
    /// Bitrate in bits per second (if known).
    pub bitrate: Option<u64>,
}

/// Type of a media stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data / metadata stream.
    Data,
}

impl StreamInfo {
    /// Creates a video stream info.
    #[must_use]
    pub fn video(codec: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            codec: codec.into(),
            stream_type: StreamType::Video,
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
            bitrate: None,
        }
    }

    /// Creates an audio stream info.
    #[must_use]
    pub fn audio(codec: impl Into<String>, sample_rate: u32, channels: u8) -> Self {
        Self {
            codec: codec.into(),
            stream_type: StreamType::Audio,
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            bitrate: None,
        }
    }

    /// Sets the bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }
}

/// Configuration for stream copy decisions.
#[derive(Debug, Clone)]
pub struct StreamCopyConfig {
    /// The copy mode to use.
    pub mode: StreamCopyMode,
    /// Desired output video codec (if any).
    pub target_video_codec: Option<String>,
    /// Desired output audio codec (if any).
    pub target_audio_codec: Option<String>,
    /// Desired output width (if resizing is needed).
    pub target_width: Option<u32>,
    /// Desired output height (if resizing is needed).
    pub target_height: Option<u32>,
    /// Whether any video filters are applied (forces re-encode).
    pub has_video_filters: bool,
    /// Whether any audio filters are applied (forces re-encode).
    pub has_audio_filters: bool,
}

impl Default for StreamCopyConfig {
    fn default() -> Self {
        Self {
            mode: StreamCopyMode::Auto,
            target_video_codec: None,
            target_audio_codec: None,
            target_width: None,
            target_height: None,
            has_video_filters: false,
            has_audio_filters: false,
        }
    }
}

/// Result of a stream copy eligibility check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyDecision {
    /// Whether the video stream can be copied.
    pub copy_video: bool,
    /// Whether the audio stream can be copied.
    pub copy_audio: bool,
    /// Reason video cannot be copied (if applicable).
    pub video_reason: Option<String>,
    /// Reason audio cannot be copied (if applicable).
    pub audio_reason: Option<String>,
}

impl CopyDecision {
    /// Returns `true` if at least one stream can be copied.
    #[must_use]
    pub fn any_copy(&self) -> bool {
        self.copy_video || self.copy_audio
    }

    /// Returns `true` if all streams can be copied (full remux).
    #[must_use]
    pub fn full_remux(&self) -> bool {
        self.copy_video && self.copy_audio
    }

    /// Returns the effective `StreamCopyMode` based on the decision.
    #[must_use]
    pub fn effective_mode(&self) -> StreamCopyMode {
        match (self.copy_video, self.copy_audio) {
            (true, true) => StreamCopyMode::CopyAll,
            (true, false) => StreamCopyMode::CopyVideo,
            (false, true) => StreamCopyMode::CopyAudio,
            (false, false) => StreamCopyMode::ReEncode,
        }
    }
}

/// Detects whether streams can be copied without re-encoding.
pub struct StreamCopyDetector;

impl StreamCopyDetector {
    /// Checks whether the given input stream can be copied to the output
    /// given the configuration constraints.
    #[must_use]
    pub fn evaluate(
        input_video: Option<&StreamInfo>,
        input_audio: Option<&StreamInfo>,
        config: &StreamCopyConfig,
    ) -> CopyDecision {
        // Handle explicit modes first
        match config.mode {
            StreamCopyMode::ReEncode => {
                return CopyDecision {
                    copy_video: false,
                    copy_audio: false,
                    video_reason: Some("Re-encode mode selected".to_string()),
                    audio_reason: Some("Re-encode mode selected".to_string()),
                };
            }
            StreamCopyMode::CopyAll => {
                return CopyDecision {
                    copy_video: input_video.is_some(),
                    copy_audio: input_audio.is_some(),
                    video_reason: if input_video.is_none() {
                        Some("No video stream".to_string())
                    } else {
                        None
                    },
                    audio_reason: if input_audio.is_none() {
                        Some("No audio stream".to_string())
                    } else {
                        None
                    },
                };
            }
            StreamCopyMode::CopyVideo => {
                let (copy_v, v_reason) = Self::check_video_copy(input_video, config);
                return CopyDecision {
                    copy_video: copy_v,
                    copy_audio: false,
                    video_reason: v_reason,
                    audio_reason: Some("Copy-video mode: audio will be re-encoded".to_string()),
                };
            }
            StreamCopyMode::CopyAudio => {
                let (copy_a, a_reason) = Self::check_audio_copy(input_audio, config);
                return CopyDecision {
                    copy_video: false,
                    copy_audio: copy_a,
                    video_reason: Some("Copy-audio mode: video will be re-encoded".to_string()),
                    audio_reason: a_reason,
                };
            }
            StreamCopyMode::Auto => {
                // Fall through to auto-detection below
            }
        }

        // Auto mode: check each stream independently
        let (copy_v, v_reason) = Self::check_video_copy(input_video, config);
        let (copy_a, a_reason) = Self::check_audio_copy(input_audio, config);

        CopyDecision {
            copy_video: copy_v,
            copy_audio: copy_a,
            video_reason: v_reason,
            audio_reason: a_reason,
        }
    }

    /// Checks whether the video stream is eligible for copy.
    fn check_video_copy(
        input: Option<&StreamInfo>,
        config: &StreamCopyConfig,
    ) -> (bool, Option<String>) {
        let Some(stream) = input else {
            return (false, Some("No video stream present".to_string()));
        };

        if config.has_video_filters {
            return (
                false,
                Some("Video filters are applied; re-encoding required".to_string()),
            );
        }

        // Check codec match
        if let Some(target_codec) = &config.target_video_codec {
            if !Self::codecs_match(&stream.codec, target_codec) {
                return (
                    false,
                    Some(format!(
                        "Codec mismatch: source={}, target={}",
                        stream.codec, target_codec
                    )),
                );
            }
        }

        // Check resolution match (if target resolution is specified)
        if let (Some(tw), Some(th)) = (config.target_width, config.target_height) {
            if let (Some(sw), Some(sh)) = (stream.width, stream.height) {
                if sw != tw || sh != th {
                    return (
                        false,
                        Some(format!(
                            "Resolution mismatch: source={sw}x{sh}, target={tw}x{th}"
                        )),
                    );
                }
            }
        }

        (true, None)
    }

    /// Checks whether the audio stream is eligible for copy.
    fn check_audio_copy(
        input: Option<&StreamInfo>,
        config: &StreamCopyConfig,
    ) -> (bool, Option<String>) {
        let Some(stream) = input else {
            return (false, Some("No audio stream present".to_string()));
        };

        if config.has_audio_filters {
            return (
                false,
                Some("Audio filters are applied; re-encoding required".to_string()),
            );
        }

        if let Some(target_codec) = &config.target_audio_codec {
            if !Self::codecs_match(&stream.codec, target_codec) {
                return (
                    false,
                    Some(format!(
                        "Codec mismatch: source={}, target={}",
                        stream.codec, target_codec
                    )),
                );
            }
        }

        (true, None)
    }

    /// Normalised codec name comparison.
    ///
    /// Handles common aliases like "libvpx-vp9" == "vp9", "libopus" == "opus", etc.
    fn codecs_match(a: &str, b: &str) -> bool {
        let na = Self::normalise_codec(a);
        let nb = Self::normalise_codec(b);
        na == nb
    }

    /// Normalises a codec name to a canonical form.
    fn normalise_codec(name: &str) -> &str {
        match name {
            "libvpx-vp9" | "libvpx_vp9" => "vp9",
            "libvpx" | "libvpx-vp8" => "vp8",
            "libaom-av1" | "libaom_av1" | "svt-av1" | "svt_av1" | "rav1e" => "av1",
            "libx264" | "x264" | "h.264" | "avc" => "h264",
            "libx265" | "x265" | "h.265" => "hevc",
            "libopus" => "opus",
            "libvorbis" => "vorbis",
            "pcm_s16le" | "pcm_s24le" | "pcm_s32le" | "pcm_f32le" => "pcm",
            other => other,
        }
    }
}

/// Estimated speedup factor when stream-copying vs. re-encoding.
///
/// Typically stream copy is 50-100x faster than re-encoding, since it
/// only involves demuxing and remuxing without any pixel processing.
pub const STREAM_COPY_SPEEDUP_FACTOR: f64 = 50.0;

/// Estimates the time saved by stream copying instead of re-encoding.
///
/// # Arguments
///
/// * `duration_secs` - Source duration in seconds.
/// * `encode_speed_factor` - Encoding speed factor (from `TranscodeEstimator`).
/// * `decision` - The copy decision from `StreamCopyDetector`.
///
/// # Returns
///
/// Estimated time saved in seconds.
#[must_use]
pub fn estimate_time_saved(
    duration_secs: f64,
    encode_speed_factor: f64,
    decision: &CopyDecision,
) -> f64 {
    if duration_secs <= 0.0 || encode_speed_factor <= 0.0 {
        return 0.0;
    }

    let encode_time = duration_secs * encode_speed_factor;
    let copy_time = duration_secs / STREAM_COPY_SPEEDUP_FACTOR;

    // Weight: video is typically 80% of encoding time, audio 20%
    let video_weight = 0.8;
    let audio_weight = 0.2;

    let mut saved = 0.0;
    if decision.copy_video {
        saved += (encode_time - copy_time) * video_weight;
    }
    if decision.copy_audio {
        saved += (encode_time - copy_time) * audio_weight;
    }

    saved.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamCopyMode tests ────────────────────────────────────────────

    #[test]
    fn test_default_mode_is_reencode() {
        assert_eq!(StreamCopyMode::default(), StreamCopyMode::ReEncode);
    }

    #[test]
    fn test_stream_copy_mode_equality() {
        assert_eq!(StreamCopyMode::Auto, StreamCopyMode::Auto);
        assert_ne!(StreamCopyMode::Auto, StreamCopyMode::CopyAll);
    }

    // ── StreamInfo constructors ─────────────────────────────────────────

    #[test]
    fn test_video_stream_info() {
        let info = StreamInfo::video("vp9", 1920, 1080);
        assert_eq!(info.codec, "vp9");
        assert_eq!(info.stream_type, StreamType::Video);
        assert_eq!(info.width, Some(1920));
        assert_eq!(info.height, Some(1080));
        assert!(info.sample_rate.is_none());
    }

    #[test]
    fn test_audio_stream_info() {
        let info = StreamInfo::audio("opus", 48000, 2);
        assert_eq!(info.codec, "opus");
        assert_eq!(info.stream_type, StreamType::Audio);
        assert_eq!(info.sample_rate, Some(48000));
        assert_eq!(info.channels, Some(2));
        assert!(info.width.is_none());
    }

    #[test]
    fn test_stream_info_with_bitrate() {
        let info = StreamInfo::video("av1", 3840, 2160).with_bitrate(15_000_000);
        assert_eq!(info.bitrate, Some(15_000_000));
    }

    // ── Codec normalisation ─────────────────────────────────────────────

    #[test]
    fn test_normalise_codec_vp9_aliases() {
        assert_eq!(StreamCopyDetector::normalise_codec("vp9"), "vp9");
        assert_eq!(StreamCopyDetector::normalise_codec("libvpx-vp9"), "vp9");
        assert_eq!(StreamCopyDetector::normalise_codec("libvpx_vp9"), "vp9");
    }

    #[test]
    fn test_normalise_codec_av1_aliases() {
        assert_eq!(StreamCopyDetector::normalise_codec("av1"), "av1");
        assert_eq!(StreamCopyDetector::normalise_codec("libaom-av1"), "av1");
        assert_eq!(StreamCopyDetector::normalise_codec("svt-av1"), "av1");
        assert_eq!(StreamCopyDetector::normalise_codec("rav1e"), "av1");
    }

    #[test]
    fn test_normalise_codec_h264_aliases() {
        assert_eq!(StreamCopyDetector::normalise_codec("h264"), "h264");
        assert_eq!(StreamCopyDetector::normalise_codec("libx264"), "h264");
        assert_eq!(StreamCopyDetector::normalise_codec("avc"), "h264");
    }

    #[test]
    fn test_normalise_codec_opus_aliases() {
        assert_eq!(StreamCopyDetector::normalise_codec("opus"), "opus");
        assert_eq!(StreamCopyDetector::normalise_codec("libopus"), "opus");
    }

    #[test]
    fn test_normalise_codec_unknown() {
        assert_eq!(
            StreamCopyDetector::normalise_codec("custom_codec"),
            "custom_codec"
        );
    }

    // ── CopyDecision helpers ────────────────────────────────────────────

    #[test]
    fn test_copy_decision_full_remux() {
        let d = CopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        assert!(d.full_remux());
        assert!(d.any_copy());
        assert_eq!(d.effective_mode(), StreamCopyMode::CopyAll);
    }

    #[test]
    fn test_copy_decision_video_only() {
        let d = CopyDecision {
            copy_video: true,
            copy_audio: false,
            video_reason: None,
            audio_reason: Some("mismatch".to_string()),
        };
        assert!(!d.full_remux());
        assert!(d.any_copy());
        assert_eq!(d.effective_mode(), StreamCopyMode::CopyVideo);
    }

    #[test]
    fn test_copy_decision_audio_only() {
        let d = CopyDecision {
            copy_video: false,
            copy_audio: true,
            video_reason: Some("mismatch".to_string()),
            audio_reason: None,
        };
        assert_eq!(d.effective_mode(), StreamCopyMode::CopyAudio);
    }

    #[test]
    fn test_copy_decision_reencode() {
        let d = CopyDecision {
            copy_video: false,
            copy_audio: false,
            video_reason: Some("mismatch".to_string()),
            audio_reason: Some("mismatch".to_string()),
        };
        assert!(!d.any_copy());
        assert_eq!(d.effective_mode(), StreamCopyMode::ReEncode);
    }

    // ── StreamCopyDetector evaluate ─────────────────────────────────────

    #[test]
    fn test_auto_matching_codecs_copies_both() {
        let video = StreamInfo::video("vp9", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(decision.copy_video);
        assert!(decision.copy_audio);
        assert!(decision.full_remux());
    }

    #[test]
    fn test_auto_mismatched_video_codec() {
        let video = StreamInfo::video("h264", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(!decision.copy_video);
        assert!(decision.copy_audio);
        assert!(decision.video_reason.is_some());
    }

    #[test]
    fn test_auto_mismatched_resolution() {
        let video = StreamInfo::video("vp9", 1280, 720);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_video_codec: Some("vp9".to_string()),
            target_width: Some(1920),
            target_height: Some(1080),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), None, &config);
        assert!(!decision.copy_video);
        assert!(decision
            .video_reason
            .as_deref()
            .map_or(false, |r| r.contains("Resolution")));
    }

    #[test]
    fn test_auto_with_video_filters_forces_reencode() {
        let video = StreamInfo::video("vp9", 1920, 1080);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_video_codec: Some("vp9".to_string()),
            has_video_filters: true,
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), None, &config);
        assert!(!decision.copy_video);
    }

    #[test]
    fn test_auto_with_audio_filters_forces_reencode() {
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_audio_codec: Some("opus".to_string()),
            has_audio_filters: true,
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(None, Some(&audio), &config);
        assert!(!decision.copy_audio);
    }

    #[test]
    fn test_explicit_reencode_mode() {
        let video = StreamInfo::video("vp9", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::ReEncode,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(!decision.copy_video);
        assert!(!decision.copy_audio);
    }

    #[test]
    fn test_explicit_copy_all_mode() {
        let video = StreamInfo::video("h264", 1920, 1080);
        let audio = StreamInfo::audio("aac", 44100, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::CopyAll,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        // CopyAll forces copy regardless of codec mismatch
        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(decision.copy_video);
        assert!(decision.copy_audio);
    }

    #[test]
    fn test_explicit_copy_video_mode() {
        let video = StreamInfo::video("vp9", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::CopyVideo,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(decision.copy_video);
        assert!(!decision.copy_audio);
    }

    #[test]
    fn test_explicit_copy_audio_mode() {
        let video = StreamInfo::video("vp9", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::CopyAudio,
            target_video_codec: Some("vp9".to_string()),
            target_audio_codec: Some("opus".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        assert!(!decision.copy_video);
        assert!(decision.copy_audio);
    }

    #[test]
    fn test_auto_no_target_codec_allows_copy() {
        let video = StreamInfo::video("av1", 1920, 1080);
        let audio = StreamInfo::audio("opus", 48000, 2);
        let config = StreamCopyConfig::default(); // Auto, no target codecs

        let decision = StreamCopyDetector::evaluate(Some(&video), Some(&audio), &config);
        // When no target codec is specified, copy is allowed
        assert!(decision.copy_video);
        assert!(decision.copy_audio);
    }

    #[test]
    fn test_codec_alias_matching() {
        let video = StreamInfo::video("libvpx-vp9", 1920, 1080);
        let config = StreamCopyConfig {
            mode: StreamCopyMode::Auto,
            target_video_codec: Some("vp9".to_string()),
            ..StreamCopyConfig::default()
        };

        let decision = StreamCopyDetector::evaluate(Some(&video), None, &config);
        assert!(decision.copy_video, "libvpx-vp9 should match vp9");
    }

    #[test]
    fn test_no_streams_present() {
        let config = StreamCopyConfig::default();
        let decision = StreamCopyDetector::evaluate(None, None, &config);
        assert!(!decision.copy_video);
        assert!(!decision.copy_audio);
    }

    // ── estimate_time_saved ─────────────────────────────────────────────

    #[test]
    fn test_estimate_time_saved_full_remux() {
        let decision = CopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        let saved = estimate_time_saved(60.0, 5.0, &decision);
        assert!(saved > 0.0, "Should save time with full remux");
    }

    #[test]
    fn test_estimate_time_saved_no_copy() {
        let decision = CopyDecision {
            copy_video: false,
            copy_audio: false,
            video_reason: Some("mismatch".to_string()),
            audio_reason: Some("mismatch".to_string()),
        };
        let saved = estimate_time_saved(60.0, 5.0, &decision);
        assert!(
            (saved - 0.0).abs() < f64::EPSILON,
            "No time saved without copy"
        );
    }

    #[test]
    fn test_estimate_time_saved_zero_duration() {
        let decision = CopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        assert_eq!(estimate_time_saved(0.0, 5.0, &decision), 0.0);
    }

    #[test]
    fn test_estimate_time_saved_negative_duration() {
        let decision = CopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        assert_eq!(estimate_time_saved(-10.0, 5.0, &decision), 0.0);
    }

    #[test]
    fn test_estimate_time_saved_video_only_copy() {
        let full = CopyDecision {
            copy_video: true,
            copy_audio: true,
            video_reason: None,
            audio_reason: None,
        };
        let video_only = CopyDecision {
            copy_video: true,
            copy_audio: false,
            video_reason: None,
            audio_reason: Some("mismatch".to_string()),
        };
        let full_saved = estimate_time_saved(60.0, 5.0, &full);
        let video_saved = estimate_time_saved(60.0, 5.0, &video_only);
        assert!(video_saved > 0.0);
        assert!(
            video_saved < full_saved,
            "Video-only should save less than full remux"
        );
    }
}
