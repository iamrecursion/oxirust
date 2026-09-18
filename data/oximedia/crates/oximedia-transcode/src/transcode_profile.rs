//! TranscodeProfile — shareable encoding configuration with JSON import/export.
//!
//! A `TranscodeProfile` encapsulates all the settings needed to describe a
//! complete transcode operation (codecs, bitrates, quality, filters) and can
//! be serialised to / deserialised from JSON for sharing between operators,
//! pre-sets libraries, and tooling integration.

use serde::{Deserialize, Serialize};

use crate::{LoudnessStandard, MultiPassMode, QualityMode, Result, TranscodeError};

// ─── Profile components ───────────────────────────────────────────────────────

/// Video encoding parameters within a `TranscodeProfile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoProfileParams {
    /// Codec name (e.g. `"h264"`, `"av1"`, `"vp9"`).
    pub codec: String,
    /// Target bitrate in bits per second.  `None` means CRF/quality-based.
    pub bitrate_bps: Option<u64>,
    /// Constant Rate Factor (0–63 for AV1, 0–51 for H.264).
    pub crf: Option<u8>,
    /// Encoder speed preset (e.g. `"slow"`, `"medium"`, `"fast"`).
    pub preset: Option<String>,
    /// Codec profile (e.g. `"high"`, `"main"`, `"baseline"`).
    pub profile: Option<String>,
    /// Output width in pixels.
    pub width: Option<u32>,
    /// Output height in pixels.
    pub height: Option<u32>,
    /// Output frame rate as `(numerator, denominator)`.
    pub frame_rate: Option<(u32, u32)>,
    /// Number of encoding threads (0 = auto).
    pub threads: u32,
    /// Quality mode used when CRF and bitrate are both absent.
    pub quality_mode: Option<QualityMode>,
    /// Enable row-based multi-threading (AV1 / VP9).
    pub row_mt: bool,
    /// Number of tile columns (AV1 tile-based parallel encoding, log2).
    pub tile_columns: Option<u8>,
    /// Number of tile rows (AV1 tile-based parallel encoding, log2).
    pub tile_rows: Option<u8>,
}

impl Default for VideoProfileParams {
    fn default() -> Self {
        Self {
            codec: "h264".into(),
            bitrate_bps: None,
            crf: Some(23),
            preset: Some("medium".into()),
            profile: Some("high".into()),
            width: None,
            height: None,
            frame_rate: None,
            threads: 0,
            quality_mode: None,
            row_mt: true,
            tile_columns: None,
            tile_rows: None,
        }
    }
}

/// Audio encoding parameters within a `TranscodeProfile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioProfileParams {
    /// Codec name (e.g. `"aac"`, `"opus"`, `"flac"`).
    pub codec: String,
    /// Target bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Number of output channels.
    pub channels: u8,
    /// Whether to apply integrated-loudness normalisation.
    pub normalize: bool,
    /// Loudness target standard.
    pub loudness_standard: Option<LoudnessStandard>,
    /// Target loudness in LUFS (overrides `loudness_standard` when set).
    pub target_lufs: Option<f64>,
}

impl Default for AudioProfileParams {
    fn default() -> Self {
        Self {
            codec: "aac".into(),
            bitrate_bps: 192_000,
            sample_rate: 48_000,
            channels: 2,
            normalize: false,
            loudness_standard: None,
            target_lufs: None,
        }
    }
}

/// Container / muxer settings within a `TranscodeProfile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContainerParams {
    /// Container format (e.g. `"mp4"`, `"mkv"`, `"webm"`).
    pub format: String,
    /// Whether to fast-start MP4 for web delivery (moov atom at front).
    pub mp4_fast_start: bool,
    /// Whether to preserve all metadata from the source.
    pub preserve_metadata: bool,
}

impl Default for ContainerParams {
    fn default() -> Self {
        Self {
            format: "mkv".into(),
            mp4_fast_start: false,
            preserve_metadata: true,
        }
    }
}

// ─── TranscodeProfile ─────────────────────────────────────────────────────────

/// A complete, shareable encoding configuration.
///
/// # Example (round-trip)
///
/// ```
/// use oximedia_transcode::transcode_profile::TranscodeProfile;
///
/// let profile = TranscodeProfile::youtube_1080p();
/// let json = profile.to_json().expect("serialise");
/// let loaded = TranscodeProfile::from_json(&json).expect("deserialise");
/// assert_eq!(loaded.name, profile.name);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscodeProfile {
    /// Human-readable name for this profile (e.g. `"YouTube 1080p"`).
    pub name: String,
    /// Optional description explaining intended use.
    pub description: Option<String>,
    /// Profile schema version (for future migrations).
    pub version: u32,
    /// Video encoding parameters.
    pub video: VideoProfileParams,
    /// Audio encoding parameters.
    pub audio: AudioProfileParams,
    /// Container / muxer parameters.
    pub container: ContainerParams,
    /// Multi-pass encoding mode.
    pub multi_pass: MultiPassMode,
    /// Arbitrary key/value tags for tooling metadata.
    pub tags: Vec<(String, String)>,
}

impl TranscodeProfile {
    /// Creates a blank profile with default parameters.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            version: 1,
            video: VideoProfileParams::default(),
            audio: AudioProfileParams::default(),
            container: ContainerParams::default(),
            multi_pass: MultiPassMode::SinglePass,
            tags: Vec::new(),
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Sets the video parameters.
    #[must_use]
    pub fn video(mut self, params: VideoProfileParams) -> Self {
        self.video = params;
        self
    }

    /// Sets the audio parameters.
    #[must_use]
    pub fn audio(mut self, params: AudioProfileParams) -> Self {
        self.audio = params;
        self
    }

    /// Sets the container parameters.
    #[must_use]
    pub fn container(mut self, params: ContainerParams) -> Self {
        self.container = params;
        self
    }

    /// Sets the multi-pass mode.
    #[must_use]
    pub fn multi_pass(mut self, mode: MultiPassMode) -> Self {
        self.multi_pass = mode;
        self
    }

    /// Adds a tag to the profile.
    #[must_use]
    pub fn tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push((key.into(), value.into()));
        self
    }

    // ── Built-in presets ──────────────────────────────────────────────────────

    /// YouTube 1080p upload profile (H.264 + AAC).
    #[must_use]
    pub fn youtube_1080p() -> Self {
        Self::new("YouTube 1080p")
            .description("H.264 High 1080p for YouTube upload")
            .video(VideoProfileParams {
                codec: "h264".into(),
                crf: Some(18),
                preset: Some("slow".into()),
                profile: Some("high".into()),
                width: Some(1920),
                height: Some(1080),
                frame_rate: Some((30, 1)),
                ..VideoProfileParams::default()
            })
            .audio(AudioProfileParams {
                codec: "aac".into(),
                bitrate_bps: 192_000,
                sample_rate: 48_000,
                channels: 2,
                normalize: true,
                loudness_standard: Some(LoudnessStandard::EbuR128),
                ..AudioProfileParams::default()
            })
            .container(ContainerParams {
                format: "mp4".into(),
                mp4_fast_start: true,
                preserve_metadata: true,
            })
    }

    /// Podcast / audio-only Opus profile (EBU R128 normalised).
    #[must_use]
    pub fn podcast_opus() -> Self {
        Self::new("Podcast Opus")
            .description("Opus mono/stereo for podcast distribution (EBU R128 −23 LUFS)")
            .video(VideoProfileParams {
                codec: "none".into(),
                ..VideoProfileParams::default()
            })
            .audio(AudioProfileParams {
                codec: "opus".into(),
                bitrate_bps: 96_000,
                sample_rate: 48_000,
                channels: 2,
                normalize: true,
                loudness_standard: Some(LoudnessStandard::EbuR128),
                ..AudioProfileParams::default()
            })
            .container(ContainerParams {
                format: "ogg".into(),
                mp4_fast_start: false,
                preserve_metadata: true,
            })
    }

    /// AV1 1080p archive profile (CRF 28, tile-based parallel).
    #[must_use]
    pub fn av1_1080p_archive() -> Self {
        Self::new("AV1 1080p Archive")
            .description("High-efficiency AV1 1080p for long-term archival")
            .video(VideoProfileParams {
                codec: "av1".into(),
                crf: Some(28),
                preset: Some("5".into()),
                width: Some(1920),
                height: Some(1080),
                row_mt: true,
                tile_columns: Some(2),
                tile_rows: Some(1),
                ..VideoProfileParams::default()
            })
            .audio(AudioProfileParams {
                codec: "opus".into(),
                bitrate_bps: 192_000,
                ..AudioProfileParams::default()
            })
            .container(ContainerParams {
                format: "mkv".into(),
                ..ContainerParams::default()
            })
            .multi_pass(MultiPassMode::TwoPass)
    }

    // ── JSON serialisation ────────────────────────────────────────────────────

    /// Serialises the profile to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails (should not happen for valid profiles).
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| TranscodeError::CodecError(format!("Profile serialisation failed: {e}")))
    }

    /// Serialises the profile to a compact (non-pretty) JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn to_json_compact(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| TranscodeError::CodecError(format!("Profile serialisation failed: {e}")))
    }

    /// Deserialises a profile from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed or the schema does not match.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            TranscodeError::InvalidInput(format!("Profile deserialisation failed: {e}"))
        })
    }

    /// Saves the profile to a file in JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json.as_bytes()).map_err(|e| {
            TranscodeError::IoError(format!("Cannot write profile to '{}': {e}", path.display()))
        })
    }

    /// Loads a profile from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            TranscodeError::IoError(format!(
                "Cannot read profile from '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_json(&data)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn test_profile_new() {
        let p = TranscodeProfile::new("Test");
        assert_eq!(p.name, "Test");
        assert_eq!(p.version, 1);
        assert!(p.description.is_none());
        assert!(p.tags.is_empty());
    }

    #[test]
    fn test_json_round_trip() {
        let original = TranscodeProfile::youtube_1080p();
        let json = original.to_json().expect("serialise");
        let loaded = TranscodeProfile::from_json(&json).expect("deserialise");
        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.video.codec, original.video.codec);
        assert_eq!(loaded.video.width, Some(1920));
        assert_eq!(loaded.audio.codec, "aac");
        assert_eq!(loaded.container.format, "mp4");
    }

    #[test]
    fn test_json_compact() {
        let p = TranscodeProfile::podcast_opus();
        let json = p.to_json_compact().expect("compact json");
        assert!(
            !json.contains('\n'),
            "compact json should not contain newlines"
        );
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let result = TranscodeProfile::from_json("not valid json {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_podcast_profile() {
        let p = TranscodeProfile::podcast_opus();
        assert_eq!(p.audio.codec, "opus");
        assert_eq!(p.audio.sample_rate, 48_000);
        assert!(p.audio.normalize);
        assert_eq!(p.container.format, "ogg");
    }

    #[test]
    fn test_av1_archive_profile() {
        let p = TranscodeProfile::av1_1080p_archive();
        assert_eq!(p.video.codec, "av1");
        assert_eq!(p.video.tile_columns, Some(2));
        assert_eq!(p.video.tile_rows, Some(1));
        assert!(p.video.row_mt);
        assert_eq!(p.multi_pass, MultiPassMode::TwoPass);
    }

    #[test]
    fn test_save_and_load_file() {
        let path = temp_dir().join("oximedia_test_profile.json");
        let original = TranscodeProfile::youtube_1080p().tag("env", "ci");
        original.save_to_file(&path).expect("save ok");

        let loaded = TranscodeProfile::load_from_file(&path).expect("load ok");
        assert_eq!(loaded.name, original.name);
        assert_eq!(loaded.tags, vec![("env".to_string(), "ci".to_string())]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tag_builder() {
        let p = TranscodeProfile::new("Tagged")
            .tag("author", "ci")
            .tag("project", "oximedia");
        assert_eq!(p.tags.len(), 2);
        assert_eq!(p.tags[0], ("author".into(), "ci".into()));
    }

    #[test]
    fn test_profile_description() {
        let p = TranscodeProfile::new("Desc test").description("A helpful description");
        assert_eq!(p.description.as_deref(), Some("A helpful description"));
    }

    #[test]
    fn test_video_profile_default_codec() {
        let vp = VideoProfileParams::default();
        assert_eq!(vp.codec, "h264");
        assert!(vp.row_mt);
    }

    #[test]
    fn test_audio_profile_default_codec() {
        let ap = AudioProfileParams::default();
        assert_eq!(ap.codec, "aac");
        assert_eq!(ap.channels, 2);
    }
}
