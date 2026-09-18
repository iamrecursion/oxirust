//! Continuity error detection for shot sequences.

/// Type of continuity error.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContinuityErrorType {
    /// Jump cut between similar shots.
    JumpCut,
    /// 180-degree axis rule violation.
    AxisViolation,
    /// Action continuity mismatch.
    ActionMatch,
    /// Costume or prop inconsistency.
    CostumeProp,
    /// Lighting change between shots.
    LightingChange,
    /// Background change between shots.
    BackgroundChange,
}

impl ContinuityErrorType {
    /// Human-readable name.
    #[allow(dead_code)]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::JumpCut => "Jump Cut",
            Self::AxisViolation => "Axis Violation (180° Rule)",
            Self::ActionMatch => "Action Match",
            Self::CostumeProp => "Costume/Prop Mismatch",
            Self::LightingChange => "Lighting Change",
            Self::BackgroundChange => "Background Change",
        }
    }
}

/// A continuity error found between two shots.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuityError {
    /// First shot in the pair.
    pub shot_a: u64,
    /// Second shot in the pair.
    pub shot_b: u64,
    /// Type of continuity error.
    pub error_type: ContinuityErrorType,
    /// Human-readable description.
    pub description: String,
    /// Confidence score (0.0..=1.0).
    pub confidence: f32,
}

impl ContinuityError {
    /// Create a new continuity error.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(
        shot_a: u64,
        shot_b: u64,
        error_type: ContinuityErrorType,
        description: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            shot_a,
            shot_b,
            error_type,
            description: description.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Coverage type used in continuity analysis.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoverageType {
    /// Wide establishing shot.
    Wide,
    /// Medium shot.
    Medium,
    /// Close-up shot.
    Close,
    /// Extreme close-up.
    Extreme,
    /// Aerial shot.
    Aerial,
    /// Insert shot.
    Insert,
    /// Cutaway.
    CutAway,
}

/// Data about a shot used for continuity analysis.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ShotData {
    /// Shot identifier.
    pub id: u64,
    /// Camera angle in degrees (0° = front, 90° = right profile).
    pub camera_angle: f32,
    /// Coverage type of this shot.
    pub coverage: CoverageType,
    /// Shot duration in frames.
    pub duration_frames: u64,
}

impl ShotData {
    /// Create a new shot data entry.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(id: u64, camera_angle: f32, coverage: CoverageType, duration_frames: u64) -> Self {
        Self {
            id,
            camera_angle,
            coverage,
            duration_frames,
        }
    }
}

/// Detects jump cuts in a shot sequence.
#[allow(dead_code)]
pub struct JumpCutDetector;

impl JumpCutDetector {
    /// Detect jump cuts: adjacent shots with same angle and coverage.
    #[allow(dead_code)]
    #[must_use]
    pub fn detect(shots: &[ShotData]) -> Vec<ContinuityError> {
        let mut errors = Vec::new();

        for window in shots.windows(2) {
            let a = &window[0];
            let b = &window[1];

            // Same coverage type + similar angle (within 15°) = jump cut
            if a.coverage == b.coverage {
                let raw_diff = (b.camera_angle - a.camera_angle).abs() % 360.0;
                let angle_diff = if raw_diff > 180.0 {
                    360.0 - raw_diff
                } else {
                    raw_diff
                };

                if angle_diff < 15.0 {
                    errors.push(ContinuityError::new(
                        a.id,
                        b.id,
                        ContinuityErrorType::JumpCut,
                        format!(
                            "Jump cut: same coverage ({}) and similar angle (diff {:.1}°)",
                            a.coverage.name(),
                            angle_diff
                        ),
                        0.8,
                    ));
                }
            }
        }

        errors
    }
}

/// Checks the 180-degree axis rule.
#[allow(dead_code)]
pub struct AxisViolation;

impl AxisViolation {
    /// Check if two camera angles violate the 180-degree rule.
    ///
    /// A violation occurs when both cameras cross to opposite sides
    /// (angle difference > 180°, i.e., a reversal across the axis).
    #[allow(dead_code)]
    #[must_use]
    pub fn check(shot_a_angle: f32, shot_b_angle: f32) -> bool {
        let raw_diff = (shot_b_angle - shot_a_angle).abs() % 360.0;
        let diff = if raw_diff > 180.0 {
            360.0 - raw_diff
        } else {
            raw_diff
        };
        // Violation if the angular diff is close to 180°
        diff > 160.0
    }
}

/// A continuity report summarizing errors in a shot sequence.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ContinuityReport {
    /// All continuity errors found.
    pub errors: Vec<ContinuityError>,
    /// Ratio of shots with errors to total shots.
    pub error_rate: f32,
    /// Most common error type (if any).
    pub most_common_error: Option<ContinuityErrorType>,
}

impl ContinuityReport {
    /// Build a continuity report from a list of errors and total shot count.
    #[allow(dead_code)]
    #[must_use]
    pub fn build(errors: Vec<ContinuityError>, total_shots: usize) -> Self {
        let error_rate = if total_shots == 0 {
            0.0
        } else {
            errors.len() as f32 / total_shots as f32
        };

        let most_common_error = Self::find_most_common(&errors);

        Self {
            errors,
            error_rate,
            most_common_error,
        }
    }

    fn find_most_common(errors: &[ContinuityError]) -> Option<ContinuityErrorType> {
        if errors.is_empty() {
            return None;
        }

        let mut counts = std::collections::HashMap::new();
        for err in errors {
            *counts.entry(err.error_type).or_insert(0_u32) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(et, _)| et)
    }

    /// Number of errors in the report.
    #[allow(dead_code)]
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

impl CoverageType {
    /// Human-readable name.
    #[allow(dead_code)]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wide => "Wide",
            Self::Medium => "Medium",
            Self::Close => "Close",
            Self::Extreme => "Extreme",
            Self::Aerial => "Aerial",
            Self::Insert => "Insert",
            Self::CutAway => "CutAway",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shot(id: u64, angle: f32, coverage: CoverageType, duration: u64) -> ShotData {
        ShotData::new(id, angle, coverage, duration)
    }

    #[test]
    fn test_continuity_error_confidence_clamped() {
        let err = ContinuityError::new(1, 2, ContinuityErrorType::JumpCut, "test", 1.5);
        assert_eq!(err.confidence, 1.0);
    }

    #[test]
    fn test_continuity_error_type_name() {
        assert_eq!(ContinuityErrorType::JumpCut.name(), "Jump Cut");
        assert_eq!(
            ContinuityErrorType::AxisViolation.name(),
            "Axis Violation (180° Rule)"
        );
    }

    #[test]
    fn test_jump_cut_detector_empty() {
        let errors = JumpCutDetector::detect(&[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_jump_cut_detector_single_shot() {
        let shots = vec![make_shot(1, 0.0, CoverageType::Medium, 50)];
        let errors = JumpCutDetector::detect(&shots);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_jump_cut_detected() {
        let shots = vec![
            make_shot(1, 0.0, CoverageType::Medium, 50),
            make_shot(2, 5.0, CoverageType::Medium, 50), // same coverage, similar angle
        ];
        let errors = JumpCutDetector::detect(&shots);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].error_type, ContinuityErrorType::JumpCut);
    }

    #[test]
    fn test_jump_cut_not_detected_different_coverage() {
        let shots = vec![
            make_shot(1, 0.0, CoverageType::Wide, 50),
            make_shot(2, 5.0, CoverageType::Medium, 50), // different coverage
        ];
        let errors = JumpCutDetector::detect(&shots);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_jump_cut_not_detected_large_angle_diff() {
        let shots = vec![
            make_shot(1, 0.0, CoverageType::Medium, 50),
            make_shot(2, 90.0, CoverageType::Medium, 50), // large angle change
        ];
        let errors = JumpCutDetector::detect(&shots);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_axis_violation_true() {
        // Near 180° difference
        assert!(AxisViolation::check(0.0, 180.0));
        assert!(AxisViolation::check(0.0, 170.0));
    }

    #[test]
    fn test_axis_violation_false() {
        // Same side
        assert!(!AxisViolation::check(0.0, 45.0));
        assert!(!AxisViolation::check(0.0, 90.0));
    }

    #[test]
    fn test_continuity_report_empty() {
        let report = ContinuityReport::build(vec![], 10);
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.error_rate, 0.0);
        assert!(report.most_common_error.is_none());
    }

    #[test]
    fn test_continuity_report_with_errors() {
        let errors = vec![
            ContinuityError::new(1, 2, ContinuityErrorType::JumpCut, "Jump cut", 0.8),
            ContinuityError::new(2, 3, ContinuityErrorType::JumpCut, "Jump cut", 0.7),
            ContinuityError::new(3, 4, ContinuityErrorType::AxisViolation, "Axis", 0.9),
        ];
        let report = ContinuityReport::build(errors, 10);
        assert_eq!(report.error_count(), 3);
        assert!((report.error_rate - 0.3).abs() < 1e-6);
        assert_eq!(report.most_common_error, Some(ContinuityErrorType::JumpCut));
    }

    #[test]
    fn test_continuity_report_zero_shots() {
        let report = ContinuityReport::build(vec![], 0);
        assert_eq!(report.error_rate, 0.0);
    }

    #[test]
    fn test_shot_data_creation() {
        let shot = ShotData::new(42, 90.0, CoverageType::Close, 100);
        assert_eq!(shot.id, 42);
        assert_eq!(shot.camera_angle, 90.0);
        assert_eq!(shot.coverage, CoverageType::Close);
        assert_eq!(shot.duration_frames, 100);
    }
}
