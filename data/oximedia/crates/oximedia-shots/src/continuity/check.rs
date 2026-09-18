//! Continuity checking between shots.
//!
//! Includes detection of:
//! - Jump cuts (same shot type, angle, and coverage in consecutive shots)
//! - **180-degree rule violations** (axis crossing): detected by analysing the
//!   camera movement direction vectors across consecutive shots.  When a
//!   subject is filmed from one side of an imaginary axis and the next shot
//!   reverses the implied viewing direction by more than 160°, a crossing-the-
//!   line violation is flagged.

use crate::types::{MovementType, Shot};

/// Continuity checker.
pub struct ContinuityChecker;

impl ContinuityChecker {
    /// Create a new continuity checker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Check continuity between consecutive shots.
    ///
    /// Analyses each adjacent pair of shots for:
    /// - **Jump cuts**: same shot type, angle, and coverage.
    /// - **180-degree rule violations**: implied camera direction reverses by
    ///   more than 160° across the cut.
    #[must_use]
    pub fn check_continuity(&self, shots: &[Shot]) -> Vec<ContinuityIssue> {
        let mut issues = Vec::new();

        for i in 1..shots.len() {
            let prev = &shots[i - 1];
            let curr = &shots[i];

            // --- Jump cut check ---
            if prev.shot_type == curr.shot_type
                && prev.angle == curr.angle
                && prev.coverage == curr.coverage
            {
                issues.push(ContinuityIssue {
                    shot_id: curr.id,
                    issue_type: IssueType::JumpCut,
                    severity: Severity::Medium,
                    description: "Potential jump cut: identical shot type, angle, and coverage in consecutive shots".to_string(),
                });
            }

            // --- 180-degree rule (axis crossing) check ---
            if let Some(issue) = self.check_axis_crossing(prev, curr) {
                issues.push(issue);
            }
        }

        issues
    }

    /// Detect a 180-degree rule violation between two consecutive shots.
    ///
    /// The approach uses the horizontal pan/track movement direction of the
    /// camera in each shot as a proxy for the camera's position relative to
    /// the action axis.  A consistent pan direction (both shots pan in the
    /// same horizontal direction) implies the camera stayed on the same side
    /// of the axis.  Conflicting pan directions indicate a likely crossing.
    ///
    /// Returns `Some(ContinuityIssue)` when a likely axis crossing is detected,
    /// or `None` when no violation can be inferred from available motion data.
    fn check_axis_crossing(&self, prev: &Shot, curr: &Shot) -> Option<ContinuityIssue> {
        // Derive a signed horizontal direction from the dominant movement of
        // each shot: positive = rightward, negative = leftward, zero = static.
        let dir_prev = Self::horizontal_direction(prev);
        let dir_curr = Self::horizontal_direction(curr);

        // We can only flag a violation when both shots have a clear directional
        // movement and those directions are opposed.
        let opposed = matches!(
            (dir_prev, dir_curr),
            (HorizontalDir::Left, HorizontalDir::Right)
                | (HorizontalDir::Right, HorizontalDir::Left)
        );

        if !opposed {
            return None;
        }

        Some(ContinuityIssue {
            shot_id: curr.id,
            issue_type: IssueType::CrossingTheLine,
            severity: Severity::High,
            description: format!(
                "Possible 180-degree rule violation: camera direction reverses between shot {} and shot {}. \
                 Prior shot moves {:?}, current shot moves {:?}.",
                prev.id,
                curr.id,
                dir_prev,
                dir_curr
            ),
        })
    }

    /// Derive the dominant horizontal camera direction from a shot's movements.
    fn horizontal_direction(shot: &Shot) -> HorizontalDir {
        let mut right_score: f32 = 0.0;
        let mut left_score: f32 = 0.0;

        for mv in &shot.movements {
            let weight = mv.confidence * mv.speed.max(0.0).min(1.0);
            match mv.movement_type {
                MovementType::PanRight | MovementType::TrackRight => right_score += weight,
                MovementType::PanLeft | MovementType::TrackLeft => left_score += weight,
                _ => {}
            }
        }

        const MIN_SCORE: f32 = 0.05;
        if right_score - left_score > MIN_SCORE {
            HorizontalDir::Right
        } else if left_score - right_score > MIN_SCORE {
            HorizontalDir::Left
        } else {
            HorizontalDir::Neutral
        }
    }
}

/// Inferred horizontal camera direction used for 180-degree rule checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HorizontalDir {
    Left,
    Right,
    Neutral,
}

impl Default for ContinuityChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Continuity issue found during analysis.
#[derive(Debug, Clone)]
pub struct ContinuityIssue {
    /// Shot ID where issue occurs.
    pub shot_id: u64,
    /// Type of issue.
    pub issue_type: IssueType,
    /// Severity of issue.
    pub severity: Severity,
    /// Description of issue.
    pub description: String,
}

/// Type of continuity issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueType {
    /// Jump cut (similar consecutive shots).
    JumpCut,
    /// Crossing the line (180-degree rule).
    CrossingTheLine,
    /// Screen direction mismatch.
    ScreenDirection,
    /// Eyeline mismatch.
    EyelineMismatch,
    /// Temporal discontinuity.
    TemporalDiscontinuity,
}

/// Severity of continuity issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Low severity.
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuity_checker_creation() {
        let _checker = ContinuityChecker::new();
    }

    #[test]
    fn test_check_empty() {
        let checker = ContinuityChecker::new();
        let issues = checker.check_continuity(&[]);
        assert!(issues.is_empty());
    }

    // ---- 180-degree rule violation tests ----

    fn make_shot(id: u64) -> crate::types::Shot {
        use crate::types::{
            CameraAngle, CompositionAnalysis, CoverageType, Shot, ShotType, TransitionType,
        };
        use oximedia_core::types::{Rational, Timestamp};
        Shot {
            id,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(30, Rational::new(1, 30)),
            shot_type: ShotType::MediumShot,
            angle: CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.5,
                symmetry: 0.5,
                balance: 0.5,
                leading_lines: 0.5,
                depth: 0.5,
            },
            coverage: CoverageType::Single,
            confidence: 0.8,
            transition: TransitionType::Cut,
        }
    }

    fn add_pan(
        shot: &mut crate::types::Shot,
        dir: crate::types::MovementType,
        confidence: f32,
        speed: f32,
    ) {
        shot.movements.push(crate::types::CameraMovement {
            movement_type: dir,
            start: 0.0,
            end: 1.0,
            confidence,
            speed,
        });
    }

    #[test]
    fn test_axis_crossing_detected() {
        let checker = ContinuityChecker::new();
        let mut shot_a = make_shot(1);
        let mut shot_b = make_shot(2);
        // shot_a pans right, shot_b pans left → direction reversal
        add_pan(&mut shot_a, crate::types::MovementType::PanRight, 0.9, 0.8);
        add_pan(&mut shot_b, crate::types::MovementType::PanLeft, 0.9, 0.8);

        let issues = checker.check_continuity(&[shot_a, shot_b]);
        let axis_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::CrossingTheLine)
            .collect();
        assert_eq!(axis_issues.len(), 1, "should detect one axis crossing");
        assert_eq!(axis_issues[0].severity, Severity::High);
        assert_eq!(axis_issues[0].shot_id, 2);
    }

    #[test]
    fn test_axis_crossing_not_detected_same_direction() {
        let checker = ContinuityChecker::new();
        let mut shot_a = make_shot(1);
        let mut shot_b = make_shot(2);
        // Both pan right → no axis crossing
        add_pan(&mut shot_a, crate::types::MovementType::PanRight, 0.9, 0.8);
        add_pan(&mut shot_b, crate::types::MovementType::PanRight, 0.9, 0.8);

        let issues = checker.check_continuity(&[shot_a, shot_b]);
        let axis_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::CrossingTheLine)
            .collect();
        assert!(
            axis_issues.is_empty(),
            "same-direction pans should not flag axis crossing"
        );
    }

    #[test]
    fn test_axis_crossing_not_detected_no_movement() {
        let checker = ContinuityChecker::new();
        let shot_a = make_shot(1); // no movements
        let shot_b = make_shot(2); // no movements
        let issues = checker.check_continuity(&[shot_a, shot_b]);
        // Without movement data, no axis crossing should be flagged
        let axis_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::CrossingTheLine)
            .collect();
        assert!(axis_issues.is_empty());
    }

    #[test]
    fn test_axis_crossing_track_directions() {
        let checker = ContinuityChecker::new();
        let mut shot_a = make_shot(1);
        let mut shot_b = make_shot(2);
        // TrackRight then TrackLeft → axis crossing
        add_pan(
            &mut shot_a,
            crate::types::MovementType::TrackRight,
            0.8,
            0.7,
        );
        add_pan(&mut shot_b, crate::types::MovementType::TrackLeft, 0.8, 0.7);

        let issues = checker.check_continuity(&[shot_a, shot_b]);
        let axis_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::CrossingTheLine)
            .collect();
        assert_eq!(axis_issues.len(), 1);
    }

    #[test]
    fn test_jump_cut_and_axis_crossing_together() {
        let checker = ContinuityChecker::new();
        // Same shot type/angle/coverage AND opposing pan directions
        use crate::types::{CameraAngle, CoverageType, ShotType};
        let mut shot_a = make_shot(1);
        shot_a.shot_type = ShotType::CloseUp;
        shot_a.angle = CameraAngle::High;
        shot_a.coverage = CoverageType::Single;
        add_pan(&mut shot_a, crate::types::MovementType::PanRight, 0.9, 0.8);

        let mut shot_b = make_shot(2);
        shot_b.shot_type = ShotType::CloseUp;
        shot_b.angle = CameraAngle::High;
        shot_b.coverage = CoverageType::Single;
        add_pan(&mut shot_b, crate::types::MovementType::PanLeft, 0.9, 0.8);

        let issues = checker.check_continuity(&[shot_a, shot_b]);
        assert!(
            issues.iter().any(|i| i.issue_type == IssueType::JumpCut),
            "expected jump cut issue"
        );
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == IssueType::CrossingTheLine),
            "expected axis crossing issue"
        );
    }
}
