//! Preset parameter validation: bounds checking, type validation, and constraint enforcement.
//!
//! Provides a composable validation framework for checking preset parameters
//! are within acceptable ranges before encoding.

#![allow(dead_code)]

/// A validation error describing a failed constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// The name of the parameter that failed validation.
    pub parameter: String,
    /// A human-readable message explaining why validation failed.
    pub message: String,
}

impl ValidationError {
    /// Create a new validation error.
    pub fn new(parameter: &str, message: &str) -> Self {
        Self {
            parameter: parameter.to_string(),
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Validation failed for '{}': {}",
            self.parameter, self.message
        )
    }
}

/// Aggregated result of validating a full preset.
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// Errors collected during validation.
    pub errors: Vec<ValidationError>,
    /// Warnings collected during validation.
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error.
    pub fn add_error(&mut self, parameter: &str, message: &str) {
        self.errors.push(ValidationError::new(parameter, message));
    }

    /// Add a warning (non-fatal issue).
    pub fn add_warning(&mut self, message: &str) {
        self.warnings.push(message.to_string());
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Check if there are any warnings.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Total number of issues (errors + warnings).
    pub fn issue_count(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: ValidationReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Validate that a `u32` value is within an inclusive range.
pub fn validate_u32_range(
    report: &mut ValidationReport,
    parameter: &str,
    value: u32,
    min: u32,
    max: u32,
) {
    if value < min || value > max {
        report.add_error(
            parameter,
            &format!("value {} is out of bounds [{}, {}]", value, min, max),
        );
    }
}

/// Validate that a `u64` value is within an inclusive range.
pub fn validate_u64_range(
    report: &mut ValidationReport,
    parameter: &str,
    value: u64,
    min: u64,
    max: u64,
) {
    if value < min || value > max {
        report.add_error(
            parameter,
            &format!("value {} is out of bounds [{}, {}]", value, min, max),
        );
    }
}

/// Validate that an `f32` value is within an inclusive range.
pub fn validate_f32_range(
    report: &mut ValidationReport,
    parameter: &str,
    value: f32,
    min: f32,
    max: f32,
) {
    if value < min || value > max {
        report.add_error(
            parameter,
            &format!("value {} is out of bounds [{}, {}]", value, min, max),
        );
    }
}

/// Validate that a string is not empty.
pub fn validate_non_empty_string(report: &mut ValidationReport, parameter: &str, value: &str) {
    if value.trim().is_empty() {
        report.add_error(parameter, "must not be empty");
    }
}

/// Validate that a string matches one of the allowed values.
pub fn validate_enum_string(
    report: &mut ValidationReport,
    parameter: &str,
    value: &str,
    allowed: &[&str],
) {
    if !allowed.contains(&value) {
        report.add_error(
            parameter,
            &format!("'{}' is not one of {:?}", value, allowed),
        );
    }
}

/// Validate that two dimensions (width, height) are valid video dimensions.
///
/// Both must be positive and even (required by most video codecs).
pub fn validate_video_dimensions(report: &mut ValidationReport, width: u32, height: u32) {
    if width == 0 {
        report.add_error("width", "must be greater than 0");
    } else if width % 2 != 0 {
        report.add_error("width", "must be even for video encoding");
    }
    if height == 0 {
        report.add_error("height", "must be greater than 0");
    } else if height % 2 != 0 {
        report.add_error("height", "must be even for video encoding");
    }
}

/// Validate a frame rate value.
///
/// Acceptable values: 23.976, 24, 25, 29.97, 30, 50, 59.94, 60 fps.
pub fn validate_frame_rate(report: &mut ValidationReport, fps: f32) {
    let valid_fps = [23.976_f32, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0];
    let tolerance = 0.01_f32;
    let is_valid = valid_fps.iter().any(|&v| (v - fps).abs() < tolerance);
    if !is_valid {
        report.add_warning(&format!(
            "frame rate {:.3} is non-standard; consider using 24/25/30/50/60 fps",
            fps
        ));
    }
}

/// Validate a video bitrate (in bits/s).
///
/// Acceptable range: 100 kbps – 800 Mbps.
pub fn validate_video_bitrate(report: &mut ValidationReport, bitrate: u64) {
    validate_u64_range(report, "video_bitrate", bitrate, 100_000, 800_000_000);
}

/// Validate an audio bitrate (in bits/s).
///
/// Acceptable range: 32 kbps – 640 kbps.
pub fn validate_audio_bitrate(report: &mut ValidationReport, bitrate: u32) {
    validate_u32_range(report, "audio_bitrate", bitrate, 32_000, 640_000);
}

/// Validate a CRF quality value (for codecs using CRF mode).
///
/// Range varies by codec; use 0–51 as a universal safe range.
pub fn validate_crf(report: &mut ValidationReport, crf: u32) {
    validate_u32_range(report, "crf", crf, 0, 51);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::new("width", "must be even");
        let s = err.to_string();
        assert!(s.contains("width"));
        assert!(s.contains("must be even"));
    }

    #[test]
    fn test_report_starts_empty() {
        let report = ValidationReport::new();
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
        assert_eq!(report.issue_count(), 0);
    }

    #[test]
    fn test_report_add_error() {
        let mut report = ValidationReport::new();
        report.add_error("fps", "invalid frame rate");
        assert!(report.has_errors());
        assert_eq!(report.errors.len(), 1);
    }

    #[test]
    fn test_report_add_warning() {
        let mut report = ValidationReport::new();
        report.add_warning("bitrate seems low");
        assert!(report.has_warnings());
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn test_report_merge() {
        let mut r1 = ValidationReport::new();
        r1.add_error("a", "err a");
        let mut r2 = ValidationReport::new();
        r2.add_warning("warn b");
        r1.merge(r2);
        assert_eq!(r1.errors.len(), 1);
        assert_eq!(r1.warnings.len(), 1);
        assert_eq!(r1.issue_count(), 2);
    }

    #[test]
    fn test_validate_u32_range_valid() {
        let mut report = ValidationReport::new();
        validate_u32_range(&mut report, "val", 50, 0, 100);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_u32_range_out_of_bounds() {
        let mut report = ValidationReport::new();
        validate_u32_range(&mut report, "val", 150, 0, 100);
        assert!(report.has_errors());
        assert_eq!(report.errors[0].parameter, "val");
    }

    #[test]
    fn test_validate_f32_range_valid() {
        let mut report = ValidationReport::new();
        validate_f32_range(&mut report, "strength", 0.5, 0.0, 1.0);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_f32_range_invalid() {
        let mut report = ValidationReport::new();
        validate_f32_range(&mut report, "strength", 1.5, 0.0, 1.0);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validate_non_empty_string_valid() {
        let mut report = ValidationReport::new();
        validate_non_empty_string(&mut report, "name", "YouTube 1080p");
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_non_empty_string_empty() {
        let mut report = ValidationReport::new();
        validate_non_empty_string(&mut report, "name", "   ");
        assert!(report.has_errors());
    }

    #[test]
    fn test_validate_enum_string_valid() {
        let mut report = ValidationReport::new();
        validate_enum_string(&mut report, "codec", "h264", &["h264", "hevc", "av1"]);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_enum_string_invalid() {
        let mut report = ValidationReport::new();
        validate_enum_string(&mut report, "codec", "xvid", &["h264", "hevc", "av1"]);
        assert!(report.has_errors());
    }

    #[test]
    fn test_validate_video_dimensions_valid() {
        let mut report = ValidationReport::new();
        validate_video_dimensions(&mut report, 1920, 1080);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_video_dimensions_odd() {
        let mut report = ValidationReport::new();
        validate_video_dimensions(&mut report, 1921, 1081);
        assert_eq!(report.errors.len(), 2); // both width and height are odd
    }

    #[test]
    fn test_validate_video_dimensions_zero() {
        let mut report = ValidationReport::new();
        validate_video_dimensions(&mut report, 0, 0);
        assert_eq!(report.errors.len(), 2);
    }

    #[test]
    fn test_validate_frame_rate_standard() {
        let mut report = ValidationReport::new();
        validate_frame_rate(&mut report, 24.0);
        assert!(!report.has_warnings());
    }

    #[test]
    fn test_validate_frame_rate_non_standard() {
        let mut report = ValidationReport::new();
        validate_frame_rate(&mut report, 15.0);
        assert!(report.has_warnings());
    }

    #[test]
    fn test_validate_video_bitrate_valid() {
        let mut report = ValidationReport::new();
        validate_video_bitrate(&mut report, 5_000_000);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_video_bitrate_too_low() {
        let mut report = ValidationReport::new();
        validate_video_bitrate(&mut report, 10_000); // 10 kbps is below minimum
        assert!(report.has_errors());
    }

    #[test]
    fn test_validate_audio_bitrate_valid() {
        let mut report = ValidationReport::new();
        validate_audio_bitrate(&mut report, 192_000);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_crf_valid() {
        let mut report = ValidationReport::new();
        validate_crf(&mut report, 23);
        assert!(!report.has_errors());
    }

    #[test]
    fn test_validate_crf_out_of_range() {
        let mut report = ValidationReport::new();
        validate_crf(&mut report, 60);
        assert!(report.has_errors());
    }
}
