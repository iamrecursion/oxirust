#![allow(dead_code)]
//! Output file verification: constraint checking for transcode deliverables.

/// A constraint that a transcode output must satisfy.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputConstraint {
    /// Video bitrate must not exceed the given value (bps).
    MaxVideoBitrate(u64),
    /// Video bitrate must be at least the given value (bps).
    MinVideoBitrate(u64),
    /// Audio bitrate must not exceed the given value (bps).
    MaxAudioBitrate(u64),
    /// Video width must equal the given pixel count.
    ExactWidth(u32),
    /// Video height must equal the given pixel count.
    ExactHeight(u32),
    /// Output file must not exceed this size in bytes.
    MaxFileSizeBytes(u64),
    /// Duration must be within `tolerance` seconds of `expected`.
    DurationWithinTolerance {
        /// Expected duration in seconds.
        expected: f64,
        /// Allowed tolerance in seconds.
        tolerance: f64,
    },
    /// The file must carry an audio track.
    HasAudio,
    /// The file must carry a video track.
    HasVideo,
}

impl OutputConstraint {
    /// A short identifier used in reports.
    #[must_use]
    pub fn constraint_name(&self) -> &'static str {
        match self {
            Self::MaxVideoBitrate(_) => "max_video_bitrate",
            Self::MinVideoBitrate(_) => "min_video_bitrate",
            Self::MaxAudioBitrate(_) => "max_audio_bitrate",
            Self::ExactWidth(_) => "exact_width",
            Self::ExactHeight(_) => "exact_height",
            Self::MaxFileSizeBytes(_) => "max_file_size_bytes",
            Self::DurationWithinTolerance { .. } => "duration_within_tolerance",
            Self::HasAudio => "has_audio",
            Self::HasVideo => "has_video",
        }
    }

    /// Returns `true` for constraints that relate to bitrate.
    #[must_use]
    pub fn is_bitrate_constraint(&self) -> bool {
        matches!(
            self,
            Self::MaxVideoBitrate(_) | Self::MinVideoBitrate(_) | Self::MaxAudioBitrate(_)
        )
    }

    /// Returns `true` for constraints about presence of a stream.
    #[must_use]
    pub fn is_stream_presence(&self) -> bool {
        matches!(self, Self::HasAudio | Self::HasVideo)
    }
}

/// A single constraint violation found during output verification.
#[derive(Debug, Clone)]
pub struct OutputViolation {
    /// The constraint that was violated.
    pub constraint: OutputConstraint,
    /// Human-readable description of what was found vs. what was expected.
    pub description: String,
    /// Whether this violation blocks delivery (true) or is merely advisory.
    pub blocking: bool,
}

impl OutputViolation {
    /// Create a new violation.
    #[must_use]
    pub fn new(
        constraint: OutputConstraint,
        description: impl Into<String>,
        blocking: bool,
    ) -> Self {
        Self {
            constraint,
            description: description.into(),
            blocking,
        }
    }

    /// Returns `true` if this violation must be fixed before delivery.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.blocking
    }
}

/// Properties of an output file used during constraint checking.
#[derive(Debug, Clone, Default)]
pub struct OutputFileInfo {
    /// Video bitrate in bits per second.
    pub video_bitrate: u64,
    /// Audio bitrate in bits per second.
    pub audio_bitrate: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Whether an audio track is present.
    pub has_audio: bool,
    /// Whether a video track is present.
    pub has_video: bool,
}

/// Verifies that an output file satisfies a list of constraints.
#[derive(Debug, Default)]
pub struct OutputVerifier {
    constraints: Vec<OutputConstraint>,
}

impl OutputVerifier {
    /// Create an empty verifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint to be checked.
    pub fn add_constraint(&mut self, constraint: OutputConstraint) {
        self.constraints.push(constraint);
    }

    /// Check a simulated file (represented by [`OutputFileInfo`]) against all constraints.
    #[must_use]
    pub fn check_file(&self, info: &OutputFileInfo) -> OutputVerifyReport {
        let mut violations = Vec::new();

        for constraint in &self.constraints {
            if let Some(v) = self.evaluate(constraint, info) {
                violations.push(v);
            }
        }

        OutputVerifyReport { violations }
    }

    fn evaluate(
        &self,
        constraint: &OutputConstraint,
        info: &OutputFileInfo,
    ) -> Option<OutputViolation> {
        match constraint {
            OutputConstraint::MaxVideoBitrate(max) => {
                if info.video_bitrate > *max {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!(
                            "video bitrate {} bps exceeds limit {} bps",
                            info.video_bitrate, max
                        ),
                        true,
                    ))
                } else {
                    None
                }
            }
            OutputConstraint::MinVideoBitrate(min) => {
                if info.video_bitrate < *min {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!(
                            "video bitrate {} bps below minimum {} bps",
                            info.video_bitrate, min
                        ),
                        false,
                    ))
                } else {
                    None
                }
            }
            OutputConstraint::MaxAudioBitrate(max) => {
                if info.audio_bitrate > *max {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!(
                            "audio bitrate {} bps exceeds limit {} bps",
                            info.audio_bitrate, max
                        ),
                        true,
                    ))
                } else {
                    None
                }
            }
            OutputConstraint::ExactWidth(w) => {
                if info.width == *w {
                    None
                } else {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!("width {} != required {}", info.width, w),
                        true,
                    ))
                }
            }
            OutputConstraint::ExactHeight(h) => {
                if info.height == *h {
                    None
                } else {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!("height {} != required {}", info.height, h),
                        true,
                    ))
                }
            }
            OutputConstraint::MaxFileSizeBytes(max) => {
                if info.file_size_bytes > *max {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!(
                            "file size {} bytes exceeds limit {} bytes",
                            info.file_size_bytes, max
                        ),
                        true,
                    ))
                } else {
                    None
                }
            }
            OutputConstraint::DurationWithinTolerance {
                expected,
                tolerance,
            } => {
                let diff = (info.duration_seconds - expected).abs();
                if diff > *tolerance {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        format!(
                            "duration {:.3}s differs from expected {:.3}s by {:.3}s (tolerance {:.3}s)",
                            info.duration_seconds, expected, diff, tolerance
                        ),
                        false,
                    ))
                } else {
                    None
                }
            }
            OutputConstraint::HasAudio => {
                if info.has_audio {
                    None
                } else {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        "no audio track present".to_string(),
                        true,
                    ))
                }
            }
            OutputConstraint::HasVideo => {
                if info.has_video {
                    None
                } else {
                    Some(OutputViolation::new(
                        constraint.clone(),
                        "no video track present".to_string(),
                        true,
                    ))
                }
            }
        }
    }
}

/// The result of running an [`OutputVerifier`] against a file.
#[derive(Debug)]
pub struct OutputVerifyReport {
    violations: Vec<OutputViolation>,
}

impl OutputVerifyReport {
    /// All violations found.
    #[must_use]
    pub fn violations(&self) -> &[OutputViolation] {
        &self.violations
    }

    /// Only violations that block delivery.
    #[must_use]
    pub fn blocking_violations(&self) -> Vec<&OutputViolation> {
        self.violations.iter().filter(|v| v.is_critical()).collect()
    }

    /// Returns `true` when there are no violations.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    /// Returns `true` when there are no blocking violations.
    #[must_use]
    pub fn is_deliverable(&self) -> bool {
        self.blocking_violations().is_empty()
    }

    /// Total number of violations.
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_info() -> OutputFileInfo {
        OutputFileInfo {
            video_bitrate: 5_000_000,
            audio_bitrate: 128_000,
            width: 1920,
            height: 1080,
            file_size_bytes: 100_000_000,
            duration_seconds: 60.0,
            has_audio: true,
            has_video: true,
        }
    }

    #[test]
    fn test_constraint_name_max_video_bitrate() {
        assert_eq!(
            OutputConstraint::MaxVideoBitrate(5_000_000).constraint_name(),
            "max_video_bitrate"
        );
    }

    #[test]
    fn test_constraint_is_bitrate_constraint() {
        assert!(OutputConstraint::MaxVideoBitrate(0).is_bitrate_constraint());
        assert!(OutputConstraint::MinVideoBitrate(0).is_bitrate_constraint());
        assert!(!OutputConstraint::HasAudio.is_bitrate_constraint());
    }

    #[test]
    fn test_constraint_is_stream_presence() {
        assert!(OutputConstraint::HasAudio.is_stream_presence());
        assert!(OutputConstraint::HasVideo.is_stream_presence());
        assert!(!OutputConstraint::ExactWidth(1920).is_stream_presence());
    }

    #[test]
    fn test_violation_is_critical() {
        let v = OutputViolation::new(OutputConstraint::HasAudio, "no audio", true);
        assert!(v.is_critical());
        let v2 = OutputViolation::new(OutputConstraint::MinVideoBitrate(0), "low", false);
        assert!(!v2.is_critical());
    }

    #[test]
    fn test_verifier_no_violations_on_pass() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::MaxVideoBitrate(10_000_000));
        v.add_constraint(OutputConstraint::ExactWidth(1920));
        let report = v.check_file(&base_info());
        assert!(report.is_ok());
    }

    #[test]
    fn test_verifier_max_video_bitrate_violation() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::MaxVideoBitrate(4_000_000));
        let report = v.check_file(&base_info());
        assert!(!report.is_ok());
        assert_eq!(report.violation_count(), 1);
    }

    #[test]
    fn test_verifier_exact_width_violation() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::ExactWidth(3840));
        let report = v.check_file(&base_info());
        assert!(!report.is_deliverable());
    }

    #[test]
    fn test_verifier_has_audio_missing() {
        let mut info = base_info();
        info.has_audio = false;
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::HasAudio);
        let report = v.check_file(&info);
        assert!(!report.is_ok());
        assert_eq!(report.blocking_violations().len(), 1);
    }

    #[test]
    fn test_verifier_has_video_ok() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::HasVideo);
        let report = v.check_file(&base_info());
        assert!(report.is_ok());
    }

    #[test]
    fn test_verifier_file_size_violation() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::MaxFileSizeBytes(50_000_000));
        let report = v.check_file(&base_info());
        assert!(!report.is_ok());
    }

    #[test]
    fn test_verifier_duration_within_tolerance_ok() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::DurationWithinTolerance {
            expected: 60.0,
            tolerance: 1.0,
        });
        let report = v.check_file(&base_info());
        assert!(report.is_ok());
    }

    #[test]
    fn test_verifier_duration_outside_tolerance() {
        let mut info = base_info();
        info.duration_seconds = 58.0;
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::DurationWithinTolerance {
            expected: 60.0,
            tolerance: 0.5,
        });
        let report = v.check_file(&info);
        assert!(!report.is_ok());
        // duration violation is non-blocking
        assert!(report.is_deliverable());
    }

    #[test]
    fn test_verifier_min_video_bitrate_advisory() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::MinVideoBitrate(8_000_000));
        let report = v.check_file(&base_info());
        assert!(!report.is_ok());
        assert!(report.is_deliverable()); // advisory only
    }

    #[test]
    fn test_report_blocking_vs_total() {
        let mut v = OutputVerifier::new();
        v.add_constraint(OutputConstraint::MaxVideoBitrate(4_000_000)); // blocking
        v.add_constraint(OutputConstraint::MinVideoBitrate(8_000_000)); // advisory
        let report = v.check_file(&base_info());
        assert_eq!(report.violation_count(), 2);
        assert_eq!(report.blocking_violations().len(), 1);
    }
}
