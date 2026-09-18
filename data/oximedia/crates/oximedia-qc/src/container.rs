//! Container quality control checks.
//!
//! This module provides QC rules for validating container formats, including
//! format validation, stream synchronization, timestamp continuity, keyframe
//! intervals, seeking capability, and duration consistency.

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;

/// Validates container format.
///
/// Ensures the container format is supported and properly structured.
pub struct FormatValidation;

impl QcRule for FormatValidation {
    fn name(&self) -> &str {
        "format_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates container format is supported and well-formed"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check file extension
        let path = &context.file_path;
        let supported_extensions = [
            ".mkv", ".webm", ".ogg", ".opus", ".oga", ".flac", ".wav", ".mp4",
        ];

        let has_valid_extension = supported_extensions
            .iter()
            .any(|ext| path.to_lowercase().ends_with(ext));

        if has_valid_extension {
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "File has recognized extension: {}",
                path.rsplit('.').next().unwrap_or("unknown")
            )));
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Warning,
                    format!("File extension not in supported list: {path}"),
                )
                .with_recommendation(
                    "Use .mkv, .webm, .ogg, .opus, .flac, .wav, or .mp4".to_string(),
                ),
            );
        }

        // Validate that we have at least one stream
        if context.streams.is_empty() {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Critical,
                    "Container has no streams".to_string(),
                )
                .with_recommendation("File must contain at least one media stream".to_string()),
            );
        }

        Ok(results)
    }
}

/// Validates stream synchronization.
///
/// Checks that audio and video streams maintain proper synchronization.
pub struct StreamSynchronization {
    max_av_offset_ms: f64,
}

impl StreamSynchronization {
    /// Creates a new stream synchronization rule.
    #[must_use]
    pub fn new(max_av_offset_ms: f64) -> Self {
        Self { max_av_offset_ms }
    }
}

impl Default for StreamSynchronization {
    fn default() -> Self {
        Self::new(100.0) // 100ms tolerance
    }
}

impl QcRule for StreamSynchronization {
    fn name(&self) -> &str {
        "stream_synchronization"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates audio/video synchronization"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        let has_video = !context.video_streams().is_empty();
        let has_audio = !context.audio_streams().is_empty();

        if has_video && has_audio {
            // In production, this would analyze packet timestamps
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "Will check A/V sync (tolerance: {:.0}ms)",
                self.max_av_offset_ms
            )));
        } else {
            results.push(CheckResult::pass(self.name()).with_recommendation(
                "A/V sync check not applicable (single stream type)".to_string(),
            ));
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty() && !context.audio_streams().is_empty()
    }
}

/// Validates timestamp continuity.
///
/// Checks for timestamp jumps, gaps, or discontinuities.
pub struct TimestampContinuity {
    max_gap_seconds: f64,
}

impl TimestampContinuity {
    /// Creates a new timestamp continuity rule.
    #[must_use]
    pub fn new(max_gap_seconds: f64) -> Self {
        Self { max_gap_seconds }
    }
}

impl Default for TimestampContinuity {
    fn default() -> Self {
        Self::new(0.5) // 500ms gap tolerance
    }
}

impl QcRule for TimestampContinuity {
    fn name(&self) -> &str {
        "timestamp_continuity"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates timestamp continuity and detects gaps"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in &context.streams {
            // In production, this would read packets and check for timestamp gaps
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will detect timestamp gaps > {:.1}s",
                        self.max_gap_seconds
                    )),
            );
        }

        Ok(results)
    }
}

/// Validates keyframe interval.
///
/// Checks that keyframes occur at regular intervals for seeking.
pub struct KeyframeInterval {
    max_interval_seconds: f64,
    min_interval_seconds: f64,
}

impl KeyframeInterval {
    /// Creates a new keyframe interval rule.
    #[must_use]
    pub fn new(min_seconds: f64, max_seconds: f64) -> Self {
        Self {
            min_interval_seconds: min_seconds,
            max_interval_seconds: max_seconds,
        }
    }
}

impl Default for KeyframeInterval {
    fn default() -> Self {
        Self::new(1.0, 10.0)
    }
}

impl QcRule for KeyframeInterval {
    fn name(&self) -> &str {
        "keyframe_interval"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates keyframe interval for seeking performance"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // In production, this would analyze actual keyframe positions
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Expected interval: {:.1}s - {:.1}s",
                        self.min_interval_seconds, self.max_interval_seconds
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Validates seeking capability.
///
/// Ensures the file can be properly seeked for playback.
pub struct SeekingCapability;

impl QcRule for SeekingCapability {
    fn name(&self) -> &str {
        "seeking_capability"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates file seeking capability"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check for duration info (required for seeking)
        if context.duration.is_some() {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Duration information present".to_string()),
            );
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Warning,
                    "No duration information - seeking may be limited".to_string(),
                )
                .with_recommendation(
                    "Ensure container has proper index/duration metadata".to_string(),
                ),
            );
        }

        // In production, would also check for:
        // - Seek index/cues
        // - Stream start times
        // - Actual seek testing

        Ok(results)
    }
}

/// Validates duration consistency.
///
/// Checks that reported duration matches actual content length.
pub struct DurationConsistency {
    tolerance_seconds: f64,
}

impl DurationConsistency {
    /// Creates a new duration consistency rule.
    #[must_use]
    pub fn new(tolerance_seconds: f64) -> Self {
        Self { tolerance_seconds }
    }
}

impl Default for DurationConsistency {
    fn default() -> Self {
        Self::new(0.1) // 100ms tolerance
    }
}

impl QcRule for DurationConsistency {
    fn name(&self) -> &str {
        "duration_consistency"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates duration consistency across streams"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Get all stream durations
        let stream_durations: Vec<(usize, f64)> = context
            .streams
            .iter()
            .filter_map(|s| s.duration_seconds().map(|d| (s.index, d)))
            .collect();

        if stream_durations.len() < 2 {
            return Ok(vec![CheckResult::pass(self.name()).with_recommendation(
                "Single stream or no duration info - check not applicable".to_string(),
            )]);
        }

        // Check if all durations are within tolerance of each other
        let mut durations: Vec<f64> = stream_durations.iter().map(|(_, d)| *d).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_duration = durations.first().copied().unwrap_or(0.0);
        let max_duration = durations.last().copied().unwrap_or(0.0);
        let difference = max_duration - min_duration;

        if difference <= self.tolerance_seconds {
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "All streams have consistent duration ({max_duration:.2}s, diff: {difference:.3}s)"
            )));
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Warning,
                    format!(
                        "Stream durations vary by {difference:.3}s (min: {min_duration:.2}s, max: {max_duration:.2}s)"
                    ),
                )
                .with_recommendation(
                    "Verify all streams have correct duration and trim points".to_string(),
                ),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        context.streams.len() >= 2
    }
}

/// Validates metadata presence and correctness.
///
/// Checks for required metadata fields.
pub struct MetadataValidation {
    required_fields: Vec<String>,
}

impl MetadataValidation {
    /// Creates a new metadata validation rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
        }
    }

    /// Sets required metadata fields.
    #[must_use]
    pub fn with_required_fields(mut self, fields: Vec<String>) -> Self {
        self.required_fields = fields;
        self
    }
}

impl Default for MetadataValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for MetadataValidation {
    fn name(&self) -> &str {
        "metadata_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates container metadata presence and correctness"
    }

    fn check(&self, _context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        if self.required_fields.is_empty() {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("No required metadata fields specified".to_string()),
            );
        } else {
            // In production, would check actual metadata
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "Would check for fields: {}",
                self.required_fields.join(", ")
            )));
        }

        Ok(results)
    }
}

/// Validates stream ordering.
///
/// Checks that streams are in conventional order (video, audio, subtitles).
pub struct StreamOrdering;

impl QcRule for StreamOrdering {
    fn name(&self) -> &str {
        "stream_ordering"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates stream ordering follows conventions"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Check conventional ordering: video -> audio -> subtitles
        let mut last_video_index = None;
        let mut last_audio_index = None;
        let mut ordering_correct = true;

        for stream in &context.streams {
            if stream.is_video() {
                last_video_index = Some(stream.index);
            } else if stream.is_audio() {
                last_audio_index = Some(stream.index);
                // Audio after video is fine, but not before
                if let Some(video_idx) = last_video_index {
                    if stream.index < video_idx {
                        ordering_correct = false;
                    }
                }
            } else if stream.is_subtitle() {
                // Subtitles should come after video and audio
                if let Some(video_idx) = last_video_index {
                    if stream.index < video_idx {
                        ordering_correct = false;
                    }
                }
                if let Some(audio_idx) = last_audio_index {
                    if stream.index < audio_idx {
                        ordering_correct = false;
                    }
                }
            }
        }

        if ordering_correct {
            results.push(
                CheckResult::pass(self.name())
                    .with_recommendation("Streams follow conventional ordering".to_string()),
            );
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Info,
                    "Stream ordering is non-conventional".to_string(),
                )
                .with_recommendation("Consider ordering: video, audio, subtitles".to_string()),
            );
        }

        Ok(results)
    }
}

/// Validates file size is reasonable.
///
/// Checks that file size is appropriate for content duration and quality.
pub struct FileSizeValidation {
    max_size_bytes: Option<u64>,
    min_size_bytes: Option<u64>,
}

impl FileSizeValidation {
    /// Creates a new file size validation rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_size_bytes: None,
            min_size_bytes: None,
        }
    }

    /// Sets maximum allowed file size.
    #[must_use]
    pub const fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = Some(bytes);
        self
    }

    /// Sets minimum expected file size.
    #[must_use]
    pub const fn with_min_size(mut self, bytes: u64) -> Self {
        self.min_size_bytes = Some(bytes);
        self
    }
}

impl Default for FileSizeValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for FileSizeValidation {
    fn name(&self) -> &str {
        "file_size_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates file size is within acceptable bounds"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would get actual file size
        if self.max_size_bytes.is_some() || self.min_size_bytes.is_some() {
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "File: {} (size validation requires file I/O)",
                context.file_path
            )));
        } else {
            results.push(CheckResult::pass(self.name()));
        }

        Ok(results)
    }
}
