#![allow(dead_code)]
//! Segment validation and integrity checking for adaptive streaming.
//!
//! This module provides tools for validating segment files, checking
//! segment consistency within a manifest, verifying duration tolerances,
//! and detecting common packaging errors.

use std::time::Duration;

/// Result of validating a single segment.
#[derive(Debug, Clone)]
pub struct SegmentValidationResult {
    /// Segment index.
    pub index: u64,
    /// Whether the segment is valid.
    pub is_valid: bool,
    /// Validation issues found.
    pub issues: Vec<ValidationIssue>,
}

impl SegmentValidationResult {
    /// Create a passing validation result.
    #[must_use]
    pub fn pass(index: u64) -> Self {
        Self {
            index,
            is_valid: true,
            issues: Vec::new(),
        }
    }

    /// Create a validation result with issues.
    #[must_use]
    pub fn with_issues(index: u64, issues: Vec<ValidationIssue>) -> Self {
        let is_valid = issues.iter().all(|i| i.severity != IssueSeverity::Error);
        Self {
            index,
            is_valid,
            issues,
        }
    }

    /// Check if there are any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning)
    }

    /// Check if there are any errors.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Error)
    }
}

/// Severity level for a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Informational note.
    Info,
    /// Warning that may cause playback issues.
    Warning,
    /// Error that will cause playback failure.
    Error,
}

/// A specific validation issue found in a segment.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level.
    pub severity: IssueSeverity,
    /// Issue code for programmatic handling.
    pub code: ValidationCode,
    /// Human-readable description of the issue.
    pub message: String,
}

impl ValidationIssue {
    /// Create a new validation issue.
    #[must_use]
    pub fn new(severity: IssueSeverity, code: ValidationCode, message: impl Into<String>) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
        }
    }

    /// Create an error issue.
    #[must_use]
    pub fn error(code: ValidationCode, message: impl Into<String>) -> Self {
        Self::new(IssueSeverity::Error, code, message)
    }

    /// Create a warning issue.
    #[must_use]
    pub fn warning(code: ValidationCode, message: impl Into<String>) -> Self {
        Self::new(IssueSeverity::Warning, code, message)
    }

    /// Create an info issue.
    #[must_use]
    pub fn info(code: ValidationCode, message: impl Into<String>) -> Self {
        Self::new(IssueSeverity::Info, code, message)
    }
}

/// Validation issue codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationCode {
    /// Segment size is zero.
    EmptySegment,
    /// Segment duration exceeds tolerance.
    DurationOutOfRange,
    /// Segment does not start with a keyframe.
    MissingKeyframe,
    /// Segment index gap detected.
    IndexGap,
    /// Segment index is duplicated.
    DuplicateIndex,
    /// Segment size is abnormally large.
    OversizedSegment,
    /// Segment size is abnormally small.
    UndersizedSegment,
    /// Duration drift accumulated beyond threshold.
    DurationDrift,
    /// Timestamp discontinuity detected.
    TimestampDiscontinuity,
}

/// Metadata about a segment for validation purposes.
#[derive(Debug, Clone)]
pub struct SegmentMetadata {
    /// Segment index.
    pub index: u64,
    /// Segment duration.
    pub duration: Duration,
    /// Segment size in bytes.
    pub size_bytes: u64,
    /// Whether the segment starts with a keyframe.
    pub starts_with_keyframe: bool,
    /// Presentation timestamp of the segment start.
    pub pts_start_ms: u64,
}

impl SegmentMetadata {
    /// Create new segment metadata.
    #[must_use]
    pub fn new(index: u64, duration: Duration, size_bytes: u64) -> Self {
        Self {
            index,
            duration,
            size_bytes,
            starts_with_keyframe: true,
            pts_start_ms: 0,
        }
    }

    /// Set whether the segment starts with a keyframe.
    #[must_use]
    pub fn with_keyframe(mut self, starts_with_keyframe: bool) -> Self {
        self.starts_with_keyframe = starts_with_keyframe;
        self
    }

    /// Set the presentation timestamp.
    #[must_use]
    pub fn with_pts(mut self, pts_start_ms: u64) -> Self {
        self.pts_start_ms = pts_start_ms;
        self
    }
}

/// Configuration for segment validation.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Target segment duration.
    pub target_duration: Duration,
    /// Allowed duration tolerance as a fraction (e.g., 0.1 for 10%).
    pub duration_tolerance: f64,
    /// Maximum allowed segment size in bytes.
    pub max_segment_size: u64,
    /// Minimum allowed segment size in bytes.
    pub min_segment_size: u64,
    /// Maximum accumulated duration drift in milliseconds.
    pub max_drift_ms: u64,
    /// Whether keyframe-at-start is required.
    pub require_keyframe_start: bool,
    /// Minimum duration multiplier relative to target (e.g. 0.5 = 50% of target).
    pub min_duration_factor: f64,
    /// Maximum duration multiplier relative to target (e.g. 2.0 = 200% of target).
    pub max_duration_factor: f64,
}

impl ValidationConfig {
    /// Create a default validation config with a target duration.
    #[must_use]
    pub fn new(target_duration: Duration) -> Self {
        Self {
            target_duration,
            duration_tolerance: 0.1,
            max_segment_size: 50 * 1024 * 1024, // 50 MB
            min_segment_size: 100,
            max_drift_ms: 500,
            require_keyframe_start: true,
            min_duration_factor: 0.5,
            max_duration_factor: 2.0,
        }
    }

    /// Set the duration tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.duration_tolerance = tolerance.clamp(0.0, 1.0);
        self
    }

    /// Set whether keyframe-at-start is required.
    #[must_use]
    pub fn with_keyframe_required(mut self, required: bool) -> Self {
        self.require_keyframe_start = required;
        self
    }

    /// Set the minimum duration factor (e.g. 0.5 means segment must be >= 50% of target).
    #[must_use]
    pub fn with_min_duration_factor(mut self, factor: f64) -> Self {
        self.min_duration_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum duration factor (e.g. 2.0 means segment must be <= 200% of target).
    #[must_use]
    pub fn with_max_duration_factor(mut self, factor: f64) -> Self {
        self.max_duration_factor = factor.max(1.0);
        self
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self::new(Duration::from_secs(6))
    }
}

/// Segment validator that checks a sequence of segments.
pub struct SegmentValidator {
    /// Validation configuration.
    config: ValidationConfig,
}

impl SegmentValidator {
    /// Create a new segment validator.
    #[must_use]
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a single segment.
    #[must_use]
    pub fn validate_segment(&self, metadata: &SegmentMetadata) -> SegmentValidationResult {
        let mut issues = Vec::new();

        // Check empty segment
        if metadata.size_bytes == 0 {
            issues.push(ValidationIssue::error(
                ValidationCode::EmptySegment,
                format!("Segment {} has zero bytes", metadata.index),
            ));
        }

        // Check size bounds
        if metadata.size_bytes > self.config.max_segment_size {
            issues.push(ValidationIssue::warning(
                ValidationCode::OversizedSegment,
                format!(
                    "Segment {} is {} bytes, exceeds max {}",
                    metadata.index, metadata.size_bytes, self.config.max_segment_size
                ),
            ));
        }

        if metadata.size_bytes > 0 && metadata.size_bytes < self.config.min_segment_size {
            issues.push(ValidationIssue::warning(
                ValidationCode::UndersizedSegment,
                format!(
                    "Segment {} is {} bytes, below min {}",
                    metadata.index, metadata.size_bytes, self.config.min_segment_size
                ),
            ));
        }

        // Check duration tolerance
        self.check_duration(metadata, &mut issues);

        // Check keyframe
        if self.config.require_keyframe_start && !metadata.starts_with_keyframe {
            issues.push(ValidationIssue::error(
                ValidationCode::MissingKeyframe,
                format!("Segment {} does not start with a keyframe", metadata.index),
            ));
        }

        SegmentValidationResult::with_issues(metadata.index, issues)
    }

    /// Check duration is within tolerance and hard bounds.
    #[allow(clippy::cast_precision_loss)]
    fn check_duration(&self, metadata: &SegmentMetadata, issues: &mut Vec<ValidationIssue>) {
        let target_ms = self.config.target_duration.as_millis() as f64;
        let actual_ms = metadata.duration.as_millis() as f64;

        if target_ms > 0.0 {
            // Soft tolerance check (warning)
            let deviation = (actual_ms - target_ms).abs() / target_ms;
            if deviation > self.config.duration_tolerance {
                issues.push(ValidationIssue::warning(
                    ValidationCode::DurationOutOfRange,
                    format!(
                        "Segment {} duration {:.0}ms deviates {:.1}% from target {:.0}ms",
                        metadata.index,
                        actual_ms,
                        deviation * 100.0,
                        target_ms
                    ),
                ));
            }

            // Hard bounds check (error): segment must be within [min_factor, max_factor] of target
            let min_ms = target_ms * self.config.min_duration_factor;
            let max_ms = target_ms * self.config.max_duration_factor;

            if actual_ms < min_ms {
                issues.push(ValidationIssue::error(
                    ValidationCode::DurationOutOfRange,
                    format!(
                        "Segment {} duration {:.0}ms is below hard minimum {:.0}ms ({:.0}x target)",
                        metadata.index, actual_ms, min_ms, self.config.min_duration_factor,
                    ),
                ));
            } else if actual_ms > max_ms {
                issues.push(ValidationIssue::error(
                    ValidationCode::DurationOutOfRange,
                    format!(
                        "Segment {} duration {:.0}ms exceeds hard maximum {:.0}ms ({:.0}x target)",
                        metadata.index, actual_ms, max_ms, self.config.max_duration_factor,
                    ),
                ));
            }
        }
    }

    /// Validate a sequence of segments for consistency.
    #[must_use]
    pub fn validate_sequence(&self, segments: &[SegmentMetadata]) -> Vec<SegmentValidationResult> {
        let mut results = Vec::with_capacity(segments.len());

        for segment in segments {
            results.push(self.validate_segment(segment));
        }

        // Check for index gaps and duplicates
        self.check_index_continuity(segments, &mut results);

        results
    }

    /// Check index continuity across segments.
    fn check_index_continuity(
        &self,
        segments: &[SegmentMetadata],
        results: &mut [SegmentValidationResult],
    ) {
        if segments.len() < 2 {
            return;
        }

        for i in 1..segments.len() {
            let expected = segments[i - 1].index + 1;
            let actual = segments[i].index;

            if actual != expected {
                results[i].issues.push(ValidationIssue::warning(
                    ValidationCode::IndexGap,
                    format!(
                        "Index gap: expected {} after {}, got {}",
                        expected,
                        segments[i - 1].index,
                        actual
                    ),
                ));
            }
        }
    }

    /// Validate that a segment duration is within the hard bounds
    /// `[min_factor * target, max_factor * target]`.
    ///
    /// Returns `Ok(())` if within bounds, or an `Err` describing the violation.
    #[allow(clippy::cast_precision_loss)]
    pub fn validate_duration_bounds(&self, segment: &SegmentMetadata) -> Result<(), String> {
        let target_ms = self.config.target_duration.as_millis() as f64;
        let actual_ms = segment.duration.as_millis() as f64;

        if target_ms <= 0.0 {
            return Ok(());
        }

        let min_ms = target_ms * self.config.min_duration_factor;
        let max_ms = target_ms * self.config.max_duration_factor;

        if actual_ms < min_ms {
            return Err(format!(
                "Segment {} duration {:.0}ms below minimum {:.0}ms",
                segment.index, actual_ms, min_ms,
            ));
        }
        if actual_ms > max_ms {
            return Err(format!(
                "Segment {} duration {:.0}ms exceeds maximum {:.0}ms",
                segment.index, actual_ms, max_ms,
            ));
        }
        Ok(())
    }

    /// Validate duration bounds for an entire sequence.
    /// Returns a list of (segment_index, error_message) for any violations.
    #[must_use]
    pub fn validate_sequence_bounds(&self, segments: &[SegmentMetadata]) -> Vec<(u64, String)> {
        let mut violations = Vec::new();
        for seg in segments {
            if let Err(msg) = self.validate_duration_bounds(seg) {
                violations.push((seg.index, msg));
            }
        }
        violations
    }

    /// Compute total validated duration.
    #[must_use]
    pub fn total_duration(&self, segments: &[SegmentMetadata]) -> Duration {
        segments.iter().map(|s| s.duration).sum()
    }
}

impl Default for SegmentValidator {
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(index: u64, duration_ms: u64, size: u64) -> SegmentMetadata {
        SegmentMetadata::new(index, Duration::from_millis(duration_ms), size)
    }

    #[test]
    fn test_valid_segment() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_segment(&seg(0, 6000, 500_000));
        assert!(result.is_valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_empty_segment() {
        let validator = SegmentValidator::default();
        let result = validator.validate_segment(&seg(0, 6000, 0));
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_oversized_segment() {
        let validator = SegmentValidator::default();
        let result = validator.validate_segment(&seg(0, 6000, 100 * 1024 * 1024));
        assert!(result.has_warnings());
    }

    #[test]
    fn test_undersized_segment() {
        let validator = SegmentValidator::default();
        let result = validator.validate_segment(&seg(0, 6000, 50));
        assert!(result.has_warnings());
    }

    #[test]
    fn test_duration_out_of_range() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_segment(&seg(0, 10_000, 500_000));
        assert!(result.has_warnings());
        let has_dur_issue = result
            .issues
            .iter()
            .any(|i| i.code == ValidationCode::DurationOutOfRange);
        assert!(has_dur_issue);
    }

    #[test]
    fn test_missing_keyframe() {
        let validator = SegmentValidator::default();
        let meta = SegmentMetadata::new(0, Duration::from_secs(6), 500_000).with_keyframe(false);
        let result = validator.validate_segment(&meta);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validation_pass_result() {
        let result = SegmentValidationResult::pass(42);
        assert!(result.is_valid);
        assert!(!result.has_warnings());
        assert!(!result.has_errors());
    }

    #[test]
    fn test_validation_issue_constructors() {
        let err = ValidationIssue::error(ValidationCode::EmptySegment, "empty");
        assert_eq!(err.severity, IssueSeverity::Error);

        let warn = ValidationIssue::warning(ValidationCode::OversizedSegment, "big");
        assert_eq!(warn.severity, IssueSeverity::Warning);

        let info = ValidationIssue::info(ValidationCode::DurationDrift, "drift");
        assert_eq!(info.severity, IssueSeverity::Info);
    }

    #[test]
    fn test_validate_sequence() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let segments = vec![
            seg(0, 6000, 500_000),
            seg(1, 6000, 500_000),
            seg(2, 6000, 500_000),
        ];
        let results = validator.validate_sequence(&segments);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_valid));
    }

    #[test]
    fn test_validate_sequence_index_gap() {
        let validator = SegmentValidator::default();
        let segments = vec![seg(0, 6000, 500_000), seg(2, 6000, 500_000)];
        let results = validator.validate_sequence(&segments);
        let gap_issues: Vec<_> = results[1]
            .issues
            .iter()
            .filter(|i| i.code == ValidationCode::IndexGap)
            .collect();
        assert_eq!(gap_issues.len(), 1);
    }

    #[test]
    fn test_total_duration() {
        let validator = SegmentValidator::default();
        let segments = vec![seg(0, 6000, 500_000), seg(1, 6000, 500_000)];
        let total = validator.total_duration(&segments);
        assert_eq!(total, Duration::from_secs(12));
    }

    #[test]
    fn test_validation_config_builder() {
        let config = ValidationConfig::new(Duration::from_secs(4))
            .with_tolerance(0.2)
            .with_keyframe_required(false);
        assert_eq!(config.target_duration, Duration::from_secs(4));
        assert!((config.duration_tolerance - 0.2).abs() < f64::EPSILON);
        assert!(!config.require_keyframe_start);
    }

    #[test]
    fn test_segment_metadata_builder() {
        let meta = SegmentMetadata::new(5, Duration::from_secs(6), 1000)
            .with_keyframe(false)
            .with_pts(5000);
        assert_eq!(meta.index, 5);
        assert!(!meta.starts_with_keyframe);
        assert_eq!(meta.pts_start_ms, 5000);
    }

    #[test]
    fn test_default_validation_config() {
        let config = ValidationConfig::default();
        assert_eq!(config.target_duration, Duration::from_secs(6));
        assert!((config.duration_tolerance - 0.1).abs() < f64::EPSILON);
        assert!(config.require_keyframe_start);
        assert!((config.min_duration_factor - 0.5).abs() < f64::EPSILON);
        assert!((config.max_duration_factor - 2.0).abs() < f64::EPSILON);
    }

    // --- Duration bounds tests (0.5x - 2x target) ---

    #[test]
    fn test_duration_within_bounds() {
        // 6s target, segment at 6s => within [3s, 12s]
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_duration_bounds(&seg(0, 6000, 500_000));
        assert!(result.is_ok());
    }

    #[test]
    fn test_duration_at_lower_bound() {
        // 6s target, segment at 3s (0.5x) => just within bounds
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_duration_bounds(&seg(0, 3000, 500_000));
        assert!(result.is_ok());
    }

    #[test]
    fn test_duration_at_upper_bound() {
        // 6s target, segment at 12s (2.0x) => just within bounds
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_duration_bounds(&seg(0, 12000, 500_000));
        assert!(result.is_ok());
    }

    #[test]
    fn test_duration_below_lower_bound() {
        // 6s target, segment at 2s => below 3s (0.5x)
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_duration_bounds(&seg(0, 2000, 500_000));
        assert!(result.is_err());
        let msg = result.expect_err("should be err");
        assert!(msg.contains("below minimum"));
    }

    #[test]
    fn test_duration_above_upper_bound() {
        // 6s target, segment at 13s => above 12s (2.0x)
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let result = validator.validate_duration_bounds(&seg(0, 13000, 500_000));
        assert!(result.is_err());
        let msg = result.expect_err("should be err");
        assert!(msg.contains("exceeds maximum"));
    }

    #[test]
    fn test_duration_bounds_custom_factors() {
        let config = ValidationConfig::new(Duration::from_secs(6))
            .with_min_duration_factor(0.8)
            .with_max_duration_factor(1.2);
        let validator = SegmentValidator::new(config);

        // 6s * 0.8 = 4.8s min, 6s * 1.2 = 7.2s max
        assert!(validator
            .validate_duration_bounds(&seg(0, 5000, 1000))
            .is_ok());
        assert!(validator
            .validate_duration_bounds(&seg(0, 7000, 1000))
            .is_ok());
        assert!(validator
            .validate_duration_bounds(&seg(0, 4000, 1000))
            .is_err());
        assert!(validator
            .validate_duration_bounds(&seg(0, 8000, 1000))
            .is_err());
    }

    #[test]
    fn test_validate_segment_hard_bounds_error() {
        // Duration bounds violations produce errors, not just warnings
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        // 1s segment => well below 0.5x (3s)
        let result = validator.validate_segment(&seg(0, 1000, 500_000));
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_segment_over_2x_error() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        // 15s segment => above 2.0x (12s)
        let result = validator.validate_segment(&seg(0, 15000, 500_000));
        assert!(!result.is_valid);
        assert!(result.has_errors());
    }

    #[test]
    fn test_validate_sequence_bounds() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let segments = vec![
            seg(0, 6000, 500_000),
            seg(1, 2000, 500_000), // too short
            seg(2, 6000, 500_000),
            seg(3, 15000, 500_000), // too long
        ];
        let violations = validator.validate_sequence_bounds(&segments);
        assert_eq!(violations.len(), 2);
        assert_eq!(violations[0].0, 1);
        assert_eq!(violations[1].0, 3);
    }

    #[test]
    fn test_validate_sequence_bounds_all_valid() {
        let validator = SegmentValidator::new(ValidationConfig::new(Duration::from_secs(6)));
        let segments = vec![
            seg(0, 5000, 500_000),
            seg(1, 7000, 500_000),
            seg(2, 6000, 500_000),
        ];
        let violations = validator.validate_sequence_bounds(&segments);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_min_duration_factor_clamped() {
        let config = ValidationConfig::new(Duration::from_secs(6)).with_min_duration_factor(5.0); // should clamp to 1.0
        assert!((config.min_duration_factor - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_duration_factor_min_bound() {
        let config = ValidationConfig::new(Duration::from_secs(6)).with_max_duration_factor(0.5); // should clamp to 1.0
        assert!((config.max_duration_factor - 1.0).abs() < f64::EPSILON);
    }
}
