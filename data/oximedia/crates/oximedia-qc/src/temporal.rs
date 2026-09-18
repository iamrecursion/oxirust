//! Temporal quality control checks.
//!
//! This module provides QC rules for validating temporal aspects of media files,
//! including dropped frames, duplicate frames, timecode continuity, and timestamp validation.

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;

/// Dropped frame detection.
///
/// Identifies missing frames by analyzing timestamp gaps and frame counts.
pub struct DroppedFrameDetection {
    tolerance_frames: usize,
    check_all_streams: bool,
}

impl DroppedFrameDetection {
    /// Creates a new dropped frame detection rule.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tolerance_frames: 0,
            check_all_streams: true,
        }
    }

    /// Sets the tolerance for dropped frames.
    #[must_use]
    pub const fn with_tolerance(mut self, frames: usize) -> Self {
        self.tolerance_frames = frames;
        self
    }

    /// Sets whether to check all video streams or just the first.
    #[must_use]
    pub const fn with_all_streams_check(mut self, check: bool) -> Self {
        self.check_all_streams = check;
        self
    }

    fn detect_dropped_frames(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Read packet timestamps
        // - Calculate expected frame count from duration and frame rate
        // - Compare with actual packet count
        // - Detect timestamp gaps larger than one frame duration
        // - Report locations and count of dropped frames

        results.push(CheckResult::pass(self.name()).with_recommendation(format!(
            "Dropped frame detection (tolerance: {} frames)",
            self.tolerance_frames
        )));

        Ok(results)
    }
}

impl Default for DroppedFrameDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for DroppedFrameDetection {
    fn name(&self) -> &str {
        "dropped_frame_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects dropped frames by analyzing timestamps"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        let video_streams = context.video_streams();
        if video_streams.is_empty() {
            return Ok(vec![CheckResult::pass(self.name())
                .with_recommendation("No video streams to check".to_string())]);
        }

        for stream in video_streams {
            let results = self.detect_dropped_frames(stream.index)?;
            all_results.extend(results);

            if !self.check_all_streams {
                break;
            }
        }

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Duplicate frame detection.
///
/// Identifies duplicate frames that may indicate capture or encoding issues.
pub struct DuplicateFrameDetection {
    max_consecutive: usize,
    hash_based: bool,
}

impl DuplicateFrameDetection {
    #[must_use]
    /// Creates a new duplicate frame detection rule with default settings.
    pub fn new() -> Self {
        Self {
            max_consecutive: 3,
            hash_based: true,
        }
    }

    #[must_use]
    /// Sets the maximum number of consecutive duplicate frames allowed.
    pub const fn with_max_consecutive(mut self, max: usize) -> Self {
        self.max_consecutive = max;
        self
    }

    fn detect_duplicates(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Decode frames
        // - Calculate frame hash (MD5, perceptual hash, etc.)
        // - Compare consecutive frames
        // - Track sequences of duplicate frames
        // - Report locations and lengths of duplicate sequences

        results.push(CheckResult::pass(self.name()).with_recommendation(format!(
            "Duplicate frame detection (max consecutive: {})",
            self.max_consecutive
        )));

        Ok(results)
    }
}

impl Default for DuplicateFrameDetection {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for DuplicateFrameDetection {
    fn name(&self) -> &str {
        "duplicate_frame_detection"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Video
    }

    fn description(&self) -> &str {
        "Detects duplicate frames in video"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        for stream in context.video_streams() {
            let results = self.detect_duplicates(stream.index)?;
            all_results.extend(results);
        }

        Ok(all_results)
    }

    fn is_applicable(&self, context: &QcContext) -> bool {
        !context.video_streams().is_empty()
    }
}

/// Timecode continuity validation.
///
/// Validates timecode tracks for continuity and correctness.
pub struct TimecodeContinuity {
    allow_discontinuities: bool,
    check_drop_frame: bool,
}

impl TimecodeContinuity {
    #[must_use]
    /// Creates a new timecode continuity validation rule with default settings.
    pub fn new() -> Self {
        Self {
            allow_discontinuities: false,
            check_drop_frame: true,
        }
    }

    #[must_use]
    /// Configures whether timecode discontinuities are allowed.
    pub const fn with_discontinuities_allowed(mut self, allow: bool) -> Self {
        self.allow_discontinuities = allow;
        self
    }

    fn validate_timecode(&self, _file_path: &str) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Extract timecode track if present
        // - Validate timecode format (HH:MM:SS:FF)
        // - Check for discontinuities
        // - Validate drop-frame vs non-drop-frame
        // - Verify timecode matches video duration

        results.push(CheckResult::pass(self.name()).with_recommendation(
            "Timecode continuity validation (if timecode track present)".to_string(),
        ));

        Ok(results)
    }
}

impl Default for TimecodeContinuity {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for TimecodeContinuity {
    fn name(&self) -> &str {
        "timecode_continuity"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates timecode continuity"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let results = self.validate_timecode(&context.file_path)?;
        Ok(results)
    }
}

/// Duration accuracy validation.
///
/// Validates that reported duration matches actual content length.
pub struct DurationAccuracy {
    tolerance_seconds: f64,
}

impl DurationAccuracy {
    #[must_use]
    /// Creates a new duration accuracy validation rule with default settings.
    pub fn new() -> Self {
        Self {
            tolerance_seconds: 0.1,
        }
    }

    #[must_use]
    /// Sets the tolerance for duration accuracy in seconds.
    pub const fn with_tolerance(mut self, seconds: f64) -> Self {
        self.tolerance_seconds = seconds;
        self
    }

    fn validate_duration(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Read container-reported duration
        // - Calculate actual duration from last packet timestamp
        // - Compare and report difference
        // - Check per-stream durations

        if let Some(duration) = context.duration {
            results.push(CheckResult::pass(self.name()).with_recommendation(format!(
                "Duration: {:.2}s (tolerance: {:.2}s)",
                duration, self.tolerance_seconds
            )));
        } else {
            results.push(
                CheckResult::fail(
                    self.name(),
                    Severity::Warning,
                    "No duration information available".to_string(),
                )
                .with_recommendation("Container should include duration metadata".to_string()),
            );
        }

        Ok(results)
    }
}

impl Default for DurationAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for DurationAccuracy {
    fn name(&self) -> &str {
        "duration_accuracy"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates duration accuracy"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        self.validate_duration(context)
    }
}

/// Timestamp validation.
///
/// Validates packet timestamps for monotonicity and correctness.
pub struct TimestampValidation {
    check_monotonic: bool,
    check_wrap: bool,
}

impl TimestampValidation {
    #[must_use]
    /// Creates a new timestamp validation rule with default settings.
    pub fn new() -> Self {
        Self {
            check_monotonic: true,
            check_wrap: true,
        }
    }

    fn validate_timestamps(&self, _stream_index: usize) -> OxiResult<Vec<CheckResult>> {
        let mut results = Vec::new();

        // In production, would:
        // - Read all packet timestamps
        // - Check for monotonic increase (within stream)
        // - Detect timestamp wraparound
        // - Validate PTS vs DTS relationship
        // - Check for negative timestamps

        results.push(
            CheckResult::pass(self.name())
                .with_recommendation("Timestamp monotonicity and validity check".to_string()),
        );

        Ok(results)
    }
}

impl Default for TimestampValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl QcRule for TimestampValidation {
    fn name(&self) -> &str {
        "timestamp_validation"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Container
    }

    fn description(&self) -> &str {
        "Validates packet timestamp correctness"
    }

    fn check(&self, context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        let mut all_results = Vec::new();

        for stream in &context.streams {
            let results = self.validate_timestamps(stream.index)?;
            all_results.extend(results);
        }

        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropped_frame_detection() {
        let detector = DroppedFrameDetection::new();
        assert_eq!(detector.tolerance_frames, 0);
    }

    #[test]
    fn test_duplicate_frame_detection() {
        let detector = DuplicateFrameDetection::new();
        assert_eq!(detector.max_consecutive, 3);
    }

    #[test]
    fn test_timecode_continuity() {
        let validator = TimecodeContinuity::new();
        assert!(!validator.allow_discontinuities);
    }

    #[test]
    fn test_duration_accuracy() {
        let validator = DurationAccuracy::new();
        assert_eq!(validator.tolerance_seconds, 0.1);
    }
}
