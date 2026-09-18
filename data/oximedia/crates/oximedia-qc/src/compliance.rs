//! Compliance validation for broadcast and streaming platforms.
//!
//! This module provides QC rules for validating compliance with various
//! delivery specifications including broadcast standards, streaming platform
//! requirements, and custom rule sets.

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::{CodecId, OxiResult};

/// Platform specification for compliance checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    /// YouTube streaming platform.
    YouTube,
    /// Vimeo streaming platform.
    Vimeo,
    /// Generic broadcast delivery.
    Broadcast,
    /// Custom specification.
    Custom,
}

impl Platform {
    /// Returns the platform name.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::YouTube => "YouTube",
            Self::Vimeo => "Vimeo",
            Self::Broadcast => "Broadcast",
            Self::Custom => "Custom",
        }
    }
}

/// YouTube upload specifications.
///
/// Validates file meets YouTube's recommended upload specifications.
pub struct YouTubeCompliance;

impl QcRule for YouTubeCompliance {
    fn name(&self) -> &str {
        "youtube_compliance"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "Validates compliance with YouTube upload specifications"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check container format
        let path = &context.file_path;
        let youtube_formats = [".mp4", ".mkv", ".webm", ".mov"];
        let has_valid_format = youtube_formats
            .iter()
            .any(|ext| path.to_lowercase().ends_with(ext));

        if has_valid_format {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Container format compatible with YouTube".to_string()),
            );
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Error,
                    "Container format not recommended for YouTube".to_string(),
                )
                .with_recommendation("Use MP4, MKV, or WebM container".to_string()),
            );
        }

        // Check video codec
        for stream in context.video_streams() {
            match stream.codec {
                CodecId::Av1 | CodecId::Vp9 => {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!(
                                "Video codec {} is ideal for YouTube",
                                stream.codec.name()
                            )),
                    );
                }
                CodecId::Vp8 => {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(
                                "VP8 is acceptable but VP9/AV1 preferred".to_string(),
                            ),
                    );
                }
                _ => {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Warning,
                            format!(
                                "Video codec {} not optimal for YouTube",
                                stream.codec.name()
                            ),
                        )
                        .with_stream(stream.index)
                        .with_recommendation("Use AV1 or VP9 for best quality".to_string()),
                    );
                }
            }

            // Check resolution
            if let (Some(width), Some(height)) =
                (stream.codec_params.width, stream.codec_params.height)
            {
                // YouTube supports up to 8K
                if width <= 7680 && height <= 4320 {
                    let standard = Self::classify_resolution(width, height);
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!(
                                "Resolution {width}x{height} ({standard}) is YouTube compatible"
                            )),
                    );
                } else {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Error,
                            format!("Resolution {width}x{height} exceeds YouTube maximum (8K)"),
                        )
                        .with_stream(stream.index)
                        .with_recommendation("Maximum resolution: 7680x4320 (8K)".to_string()),
                    );
                }
            }
        }

        // Check audio codec
        for stream in context.audio_streams() {
            match stream.codec {
                CodecId::Opus => {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation("Opus is ideal for YouTube".to_string()),
                    );
                }
                CodecId::Vorbis | CodecId::Flac | CodecId::Pcm => {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation("Audio codec is acceptable".to_string()),
                    );
                }
                _ => {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Warning,
                            format!(
                                "Audio codec {} not optimal for YouTube",
                                stream.codec.name()
                            ),
                        )
                        .with_stream(stream.index)
                        .with_recommendation("Use Opus for best quality".to_string()),
                    );
                }
            }
        }

        Ok(results)
    }
}

impl YouTubeCompliance {
    fn classify_resolution(width: u32, height: u32) -> &'static str {
        match (width, height) {
            (7680, 4320) => "8K",
            (3840, 2160) => "4K",
            (2560, 1440) => "1440p",
            (1920, 1080) => "1080p",
            (1280, 720) => "720p",
            (854, 480) => "480p",
            (640, 360) => "360p",
            _ => "Custom",
        }
    }
}

/// Vimeo upload specifications.
///
/// Validates file meets Vimeo's upload requirements.
pub struct VimeoCompliance;

impl QcRule for VimeoCompliance {
    fn name(&self) -> &str {
        "vimeo_compliance"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "Validates compliance with Vimeo upload specifications"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check container format
        let path = &context.file_path;
        let vimeo_formats = [".mp4", ".mkv", ".webm", ".mov"];
        let has_valid_format = vimeo_formats
            .iter()
            .any(|ext| path.to_lowercase().ends_with(ext));

        if has_valid_format {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Container format compatible with Vimeo".to_string()),
            );
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Error,
                    "Container format not recommended for Vimeo".to_string(),
                )
                .with_recommendation("Use MP4, MKV, or WebM container".to_string()),
            );
        }

        // Check video codec
        for stream in context.video_streams() {
            match stream.codec {
                CodecId::Av1 | CodecId::Vp9 => {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!(
                                "Video codec {} is compatible with Vimeo",
                                stream.codec.name()
                            )),
                    );
                }
                _ => {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Warning,
                            format!(
                                "Video codec {} may not be optimal for Vimeo",
                                stream.codec.name()
                            ),
                        )
                        .with_stream(stream.index)
                        .with_recommendation("Use AV1 or VP9 for best compatibility".to_string()),
                    );
                }
            }
        }

        Ok(results)
    }
}

/// Broadcast delivery specifications.
///
/// Validates compliance with broadcast delivery standards.
pub struct BroadcastCompliance {
    require_stereo: bool,
    require_hd: bool,
}

impl BroadcastCompliance {
    /// Creates a new broadcast compliance rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            require_stereo: true,
            require_hd: true,
        }
    }

    /// Sets whether stereo audio is required.
    #[must_use]
    pub const fn with_stereo_requirement(mut self, require: bool) -> Self {
        self.require_stereo = require;
        self
    }

    /// Sets whether HD resolution is required.
    #[must_use]
    pub const fn with_hd_requirement(mut self, require: bool) -> Self {
        self.require_hd = require;
        self
    }
}

impl Default for BroadcastCompliance {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for BroadcastCompliance {
    fn name(&self) -> &str {
        "broadcast_compliance"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "Validates compliance with broadcast delivery specifications"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check video resolution
        if self.require_hd {
            for stream in context.video_streams() {
                if let (Some(width), Some(height)) =
                    (stream.codec_params.width, stream.codec_params.height)
                {
                    if width >= 1280 && height >= 720 {
                        results.push(
                            CheckResult::pass(self.name())
                                .with_stream(stream.index)
                                .with_recommendation(format!("HD resolution: {width}x{height}")),
                        );
                    } else {
                        results.push(
                            CheckResult::fail(
                                self.name(),
                                Severity::Error,
                                format!(
                                    "Resolution {width}x{height} is below HD minimum (1280x720)"
                                ),
                            )
                            .with_stream(stream.index)
                            .with_recommendation(
                                "Broadcast delivery requires HD resolution".to_string(),
                            ),
                        );
                    }
                }
            }
        }

        // Check audio configuration
        if self.require_stereo {
            for stream in context.audio_streams() {
                if let Some(channels) = stream.codec_params.channels {
                    if channels >= 2 {
                        results.push(
                            CheckResult::pass(self.name())
                                .with_stream(stream.index)
                                .with_recommendation(format!(
                                    "Stereo/multichannel: {channels} channels"
                                )),
                        );
                    } else {
                        results.push(
                            CheckResult::fail(
                                self.name(),
                                Severity::Error,
                                "Mono audio not acceptable for broadcast".to_string(),
                            )
                            .with_stream(stream.index)
                            .with_recommendation(
                                "Broadcast delivery requires stereo audio".to_string(),
                            ),
                        );
                    }
                }
            }
        }

        // Check loudness (broadcast requires compliance)
        for stream in context.audio_streams() {
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(
                        "Verify EBU R128 loudness compliance (-23 LUFS)".to_string(),
                    ),
            );
        }

        Ok(results)
    }
}

/// Patent-free codec enforcement.
///
/// Strictly enforces use of only green list codecs.
pub struct PatentFreeEnforcement;

impl QcRule for PatentFreeEnforcement {
    fn name(&self) -> &str {
        "patent_free_enforcement"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "Enforces use of patent-free codecs only (green list)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Video codecs - green list
        const VIDEO_GREEN_LIST: &[CodecId] = &[
            CodecId::Av1,
            CodecId::Vp9,
            CodecId::Vp8,
            CodecId::Theora,
            CodecId::H263,
        ];

        // Audio codecs - green list
        const AUDIO_GREEN_LIST: &[CodecId] = &[
            CodecId::Opus,
            CodecId::Vorbis,
            CodecId::Flac,
            CodecId::Mp3,
            CodecId::Pcm,
        ];

        for stream in context.video_streams() {
            if VIDEO_GREEN_LIST.contains(&stream.codec) {
                results.push(
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Patent-free video codec: {}",
                            stream.codec.name()
                        )),
                );
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Critical,
                        format!(
                            "PATENT VIOLATION: Video codec '{}' is not on green list",
                            stream.codec.name()
                        ),
                    )
                    .with_stream(stream.index)
                    .with_recommendation("Use AV1, VP9, VP8, or Theora only".to_string()),
                );
            }
        }

        for stream in context.audio_streams() {
            if AUDIO_GREEN_LIST.contains(&stream.codec) {
                results.push(
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Patent-free audio codec: {}",
                            stream.codec.name()
                        )),
                );
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Critical,
                        format!(
                            "PATENT VIOLATION: Audio codec '{}' is not on green list",
                            stream.codec.name()
                        ),
                    )
                    .with_stream(stream.index)
                    .with_recommendation("Use Opus, Vorbis, FLAC, or PCM only".to_string()),
                );
            }
        }

        Ok(results)
    }
}

/// Custom compliance rule set.
///
/// Allows defining custom validation rules via configuration.
pub struct CustomCompliance {
    name: String,
    allowed_video_codecs: Vec<CodecId>,
    allowed_audio_codecs: Vec<CodecId>,
    min_width: Option<u32>,
    min_height: Option<u32>,
    max_width: Option<u32>,
    max_height: Option<u32>,
}

impl CustomCompliance {
    /// Creates a new custom compliance rule.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            allowed_video_codecs: vec![CodecId::Av1, CodecId::Vp9, CodecId::Vp8, CodecId::Theora],
            allowed_audio_codecs: vec![CodecId::Opus, CodecId::Vorbis, CodecId::Flac, CodecId::Pcm],
            min_width: None,
            min_height: None,
            max_width: None,
            max_height: None,
        }
    }

    /// Sets allowed video codecs.
    #[must_use]
    pub fn with_video_codecs(mut self, codecs: Vec<CodecId>) -> Self {
        self.allowed_video_codecs = codecs;
        self
    }

    /// Sets allowed audio codecs.
    #[must_use]
    pub fn with_audio_codecs(mut self, codecs: Vec<CodecId>) -> Self {
        self.allowed_audio_codecs = codecs;
        self
    }

    /// Sets minimum resolution.
    #[must_use]
    pub const fn with_min_resolution(mut self, width: u32, height: u32) -> Self {
        self.min_width = Some(width);
        self.min_height = Some(height);
        self
    }

    /// Sets maximum resolution.
    #[must_use]
    pub const fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_width = Some(width);
        self.max_height = Some(height);
        self
    }
}

impl QcRule for CustomCompliance {
    fn name(&self) -> &str {
        &self.name
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "Custom compliance rule set"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check video codecs
        for stream in context.video_streams() {
            if self.allowed_video_codecs.contains(&stream.codec) {
                results.push(
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Video codec {} is allowed",
                            stream.codec.name()
                        )),
                );
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Error,
                        format!("Video codec '{}' not in allowed list", stream.codec.name()),
                    )
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Allowed codecs: {}",
                        self.allowed_video_codecs
                            .iter()
                            .map(CodecId::name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                );
            }

            // Check resolution constraints
            if let (Some(width), Some(height)) =
                (stream.codec_params.width, stream.codec_params.height)
            {
                let mut res_ok = true;

                if let (Some(min_w), Some(min_h)) = (self.min_width, self.min_height) {
                    if width < min_w || height < min_h {
                        res_ok = false;
                        results.push(
                            CheckResult::fail(
                                self.name(),
                                Severity::Error,
                                format!(
                                    "Resolution {width}x{height} below minimum {min_w}x{min_h}"
                                ),
                            )
                            .with_stream(stream.index),
                        );
                    }
                }

                if let (Some(max_w), Some(max_h)) = (self.max_width, self.max_height) {
                    if width > max_w || height > max_h {
                        res_ok = false;
                        results.push(
                            CheckResult::fail(
                                self.name(),
                                Severity::Error,
                                format!(
                                    "Resolution {width}x{height} exceeds maximum {max_w}x{max_h}"
                                ),
                            )
                            .with_stream(stream.index),
                        );
                    }
                }

                if res_ok && (self.min_width.is_some() || self.max_width.is_some()) {
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(format!(
                                "Resolution {width}x{height} within bounds"
                            )),
                    );
                }
            }
        }

        // Check audio codecs
        for stream in context.audio_streams() {
            if self.allowed_audio_codecs.contains(&stream.codec) {
                results.push(
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Audio codec {} is allowed",
                            stream.codec.name()
                        )),
                );
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Error,
                        format!("Audio codec '{}' not in allowed list", stream.codec.name()),
                    )
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Allowed codecs: {}",
                        self.allowed_audio_codecs
                            .iter()
                            .map(CodecId::name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                );
            }
        }

        Ok(results)
    }
}
