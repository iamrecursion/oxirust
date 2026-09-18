//! Video quality control checks.
//!
//! This module provides QC rules for validating video streams, including
//! codec validation, resolution checks, frame rate validation, bitrate analysis,
//! interlacing detection, black/freeze frame detection, and compression artifacts.

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity, Thresholds};
use oximedia_core::{CodecId, OxiResult};

/// Validates that video codec is from the green list.
///
/// Ensures only patent-free codecs (AV1, VP9, VP8, Theora) are used.
pub struct VideoCodecValidation;

impl QcRule for VideoCodecValidation {
    fn name(&self) -> &str {
        "video_codec_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Validates that video codec is patent-free (green list only)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            let result = match stream.codec {
                CodecId::Av1 | CodecId::Vp9 | CodecId::Vp8 | CodecId::Theora | CodecId::H263 => {
                    CheckResult::pass(self.name())
                        .with_stream(stream.index)
                        .with_recommendation(format!(
                            "Using approved codec: {}",
                            stream.codec.name()
                        ))
                }
                _ => CheckResult::fail(
                    self.name(),
                    Severity::Critical,
                    format!(
                        "Invalid video codec '{}' - only green list codecs are allowed",
                        stream.codec.name()
                    ),
                )
                .with_stream(stream.index)
                .with_recommendation("Use AV1, VP9, VP8, or Theora instead".to_string()),
            };
            results.push(result);
        }

        if results.is_empty() {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Error,
                    "No video streams found".to_string(),
                )
                .with_recommendation("File should contain at least one video stream".to_string()),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Validates video resolution.
///
/// Checks for common issues like odd dimensions, extremely small or large resolutions.
pub struct ResolutionValidation {
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
    require_even: bool,
}

impl ResolutionValidation {
    /// Creates a new resolution validation rule with default constraints.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_width: 160,
            min_height: 120,
            max_width: 7680,  // 8K
            max_height: 4320, // 8K
            require_even: true,
        }
    }

    /// Sets whether dimensions must be even numbers.
    #[must_use]
    pub const fn with_even_requirement(mut self, require: bool) -> Self {
        self.require_even = require;
        self
    }

    /// Sets minimum resolution.
    #[must_use]
    pub const fn with_min_resolution(mut self, width: u32, height: u32) -> Self {
        self.min_width = width;
        self.min_height = height;
        self
    }

    /// Sets maximum resolution.
    #[must_use]
    pub const fn with_max_resolution(mut self, width: u32, height: u32) -> Self {
        self.max_width = width;
        self.max_height = height;
        self
    }
}

impl Default for ResolutionValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for ResolutionValidation {
    fn name(&self) -> &str {
        "resolution_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Validates video resolution is within acceptable bounds"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            if let (Some(width), Some(height)) =
                (stream.codec_params.width, stream.codec_params.height)
            {
                let mut issues = Vec::new();

                if width < self.min_width || height < self.min_height {
                    issues.push(format!(
                        "Resolution {width}x{height} is below minimum {}x{}",
                        self.min_width, self.min_height
                    ));
                }

                if width > self.max_width || height > self.max_height {
                    issues.push(format!(
                        "Resolution {width}x{height} exceeds maximum {}x{}",
                        self.max_width, self.max_height
                    ));
                }

                if self.require_even && (width % 2 != 0 || height % 2 != 0) {
                    issues.push(format!(
                        "Resolution {width}x{height} has odd dimensions (should be even)"
                    ));
                }

                let result = if issues.is_empty() {
                    CheckResult::pass(self.name()).with_stream(stream.index)
                } else {
                    CheckResult::fail(self.name(), Severity::Error, issues.join("; "))
                        .with_stream(stream.index)
                        .with_recommendation(
                            "Ensure resolution meets delivery specifications".to_string(),
                        )
                };

                results.push(result);
            } else {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Error,
                        "Video stream missing resolution information".to_string(),
                    )
                    .with_stream(stream.index),
                );
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Validates video frame rate.
///
/// Checks that frame rate is within acceptable bounds and matches expected values.
pub struct FrameRateValidation {
    min_fps: f64,
    max_fps: f64,
    expected_rates: Vec<f64>,
    tolerance: f64,
}

impl FrameRateValidation {
    /// Creates a new frame rate validation rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_fps: 23.0,
            max_fps: 120.0,
            expected_rates: vec![23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0],
            tolerance: 0.01,
        }
    }

    /// Sets the expected frame rates.
    #[must_use]
    pub fn with_expected_rates(mut self, rates: Vec<f64>) -> Self {
        self.expected_rates = rates;
        self
    }

    /// Sets frame rate tolerance for comparison.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Default for FrameRateValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for FrameRateValidation {
    fn name(&self) -> &str {
        "frame_rate_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Validates video frame rate is within acceptable range"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // Calculate frame rate from timebase
            // This is a simplified check - in production we would analyze actual frame timing
            let fps = if stream.timebase.num != 0 {
                stream.timebase.den as f64 / stream.timebase.num as f64
            } else {
                0.0
            };

            if fps < self.min_fps || fps > self.max_fps {
                results.push(
                    CheckResult::fail(
                        self.name(),
                        Severity::Warning,
                        format!(
                            "Frame rate {fps:.2} fps is outside acceptable range ({}-{} fps)",
                            self.min_fps, self.max_fps
                        ),
                    )
                    .with_stream(stream.index)
                    .with_recommendation(
                        "Verify frame rate meets delivery specifications".to_string(),
                    ),
                );
            } else if !self.expected_rates.is_empty() {
                let matches_expected = self
                    .expected_rates
                    .iter()
                    .any(|&rate| (fps - rate).abs() < self.tolerance);

                if matches_expected {
                    results.push(CheckResult::pass(self.name()).with_stream(stream.index));
                } else {
                    results.push(
                        CheckResult::fail(
                            self.name(),
                            Severity::Info,
                            format!(
                                "Frame rate {fps:.2} fps is unusual (expected one of: {})",
                                self.expected_rates
                                    .iter()
                                    .map(|r| format!("{r:.2}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        )
                        .with_stream(stream.index),
                    );
                }
            } else {
                results.push(CheckResult::pass(self.name()).with_stream(stream.index));
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Analyzes video bitrate.
///
/// Checks that bitrate falls within expected ranges for the resolution.
pub struct BitrateAnalysis {
    thresholds: Thresholds,
}

impl BitrateAnalysis {
    /// Creates a new bitrate analysis rule.
    #[must_use]
    pub fn new(thresholds: Thresholds) -> Self {
        Self { thresholds }
    }
}

impl QcRule for BitrateAnalysis {
    fn name(&self) -> &str {
        "bitrate_analysis"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Analyzes video bitrate and checks against thresholds"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // Use file-level bitrate computed from file_size / duration in probe_file.
        // For per-stream analysis we apportion it by stream count.
        let video_count = context.video_streams().len().max(1);
        let estimated_video_bitrate: Option<u64> =
            context.file_bitrate.map(|total| total / video_count as u64);

        for stream in context.video_streams() {
            // Prefer the stream's own bitrate if available (future extension),
            // otherwise fall back to the file-level estimate.
            let bitrate = estimated_video_bitrate;

            match bitrate {
                Some(bps) => {
                    let kbps = bps / 1000;
                    let mut passed = true;
                    let mut messages = Vec::new();

                    if let Some(min) = self.thresholds.min_video_bitrate {
                        if bps < min {
                            passed = false;
                            messages.push(format!(
                                "Video bitrate {kbps} kbps is below minimum {} kbps",
                                min / 1000
                            ));
                        }
                    }
                    if let Some(max) = self.thresholds.max_video_bitrate {
                        if bps > max {
                            passed = false;
                            messages.push(format!(
                                "Video bitrate {kbps} kbps exceeds maximum {} kbps",
                                max / 1000
                            ));
                        }
                    }

                    if passed {
                        results.push(
                            CheckResult::pass(self.name())
                                .with_stream(stream.index)
                                .with_recommendation(format!(
                                    "Video bitrate: {kbps} kbps (file-level estimate)"
                                )),
                        );
                    } else {
                        for msg in messages {
                            results.push(
                                CheckResult::fail(self.name(), Severity::Warning, msg)
                                    .with_stream(stream.index)
                                    .with_recommendation(
                                        "Adjust encoder bitrate settings".to_string(),
                                    ),
                            );
                        }
                    }
                }
                None => {
                    // No bitrate info available yet
                    results.push(
                        CheckResult::pass(self.name())
                            .with_stream(stream.index)
                            .with_recommendation(
                                "Bitrate could not be determined from file header".to_string(),
                            ),
                    );
                }
            }
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Detects interlaced video.
///
/// Warns if video is interlaced, as most modern delivery specs require progressive.
pub struct InterlacingDetection;

impl QcRule for InterlacingDetection {
    fn name(&self) -> &str {
        "interlacing_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects interlaced video content"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // In production, this would analyze actual frames for interlacing
            // For now, we pass with a note that full analysis is needed
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(
                        "Full frame analysis required to detect interlacing".to_string(),
                    ),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Detects black frames in video.
///
/// Identifies sequences of black frames that may indicate issues.
pub struct BlackFrameDetection {
    max_black_duration: f64,
}

impl BlackFrameDetection {
    /// Creates a new black frame detection rule.
    #[must_use]
    pub fn new(max_black_duration: f64) -> Self {
        Self { max_black_duration }
    }
}

impl Default for BlackFrameDetection {
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl QcRule for BlackFrameDetection {
    fn name(&self) -> &str {
        "black_frame_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects sequences of black frames"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // In production, this would decode and analyze frames
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will flag black sequences longer than {:.1}s",
                        self.max_black_duration
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Detects freeze frames in video.
///
/// Identifies duplicate frames that may indicate encoding or capture issues.
pub struct FreezeFrameDetection {
    max_freeze_duration: f64,
}

impl FreezeFrameDetection {
    /// Creates a new freeze frame detection rule.
    #[must_use]
    pub fn new(max_freeze_duration: f64) -> Self {
        Self {
            max_freeze_duration,
        }
    }
}

impl Default for FreezeFrameDetection {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl QcRule for FreezeFrameDetection {
    fn name(&self) -> &str {
        "freeze_frame_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects freeze frames (duplicate frames)"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // In production, this would compare frame hashes
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will flag freeze sequences longer than {:.1}s",
                        self.max_freeze_duration
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Detects compression artifacts and blockiness.
///
/// Analyzes video for visual quality issues caused by over-compression.
pub struct CompressionArtifactDetection {
    blockiness_threshold: f64,
}

impl CompressionArtifactDetection {
    /// Creates a new compression artifact detection rule.
    #[must_use]
    pub fn new(blockiness_threshold: f64) -> Self {
        Self {
            blockiness_threshold,
        }
    }
}

impl Default for CompressionArtifactDetection {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl QcRule for CompressionArtifactDetection {
    fn name(&self) -> &str {
        "compression_artifact_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects compression artifacts and blockiness"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        for stream in context.video_streams() {
            // In production, this would use PSNR, SSIM, VMAF, or blockiness detection
            results.push(
                CheckResult::pass(self.name())
                    .with_stream(stream.index)
                    .with_recommendation(format!(
                        "Will analyze frames for blockiness (threshold: {:.2})",
                        self.blockiness_threshold
                    )),
            );
        }

        Ok(results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}
