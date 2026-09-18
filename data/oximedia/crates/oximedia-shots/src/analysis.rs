//! Advanced shot analysis and metrics.

use crate::types::{Scene, Shot, ShotType};

/// Advanced shot analyzer providing detailed metrics.
pub struct AdvancedAnalyzer;

impl AdvancedAnalyzer {
    /// Create a new advanced analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculate shot complexity score.
    #[must_use]
    pub fn calculate_complexity(&self, shot: &Shot) -> f32 {
        let mut complexity = 0.0;

        // Movement complexity
        complexity += shot.movements.len() as f32 * 0.2;

        // Composition complexity
        complexity += (1.0 - shot.composition.symmetry) * 0.3;
        complexity += shot.composition.leading_lines * 0.2;
        complexity += shot.composition.depth * 0.2;

        // Shot type complexity (wider shots are more complex)
        complexity += match shot.shot_type {
            ShotType::ExtremeCloseUp => 0.1,
            ShotType::CloseUp => 0.2,
            ShotType::MediumCloseUp => 0.3,
            ShotType::MediumShot => 0.4,
            ShotType::MediumLongShot => 0.5,
            ShotType::LongShot => 0.7,
            ShotType::ExtremeLongShot => 1.0,
            ShotType::Unknown => 0.0,
        };

        complexity.min(1.0)
    }

    /// Calculate visual interest score.
    #[must_use]
    pub fn calculate_visual_interest(&self, shot: &Shot) -> f32 {
        let mut interest = 0.0;

        // Composition contributes to interest
        interest += shot.composition.overall_score() * 0.4;

        // Movement adds interest
        if !shot.movements.is_empty() {
            interest += 0.3;
        }

        // Unusual angles add interest
        interest += match shot.angle {
            crate::types::CameraAngle::High | crate::types::CameraAngle::Low => 0.2,
            crate::types::CameraAngle::BirdsEye | crate::types::CameraAngle::Dutch => 0.3,
            crate::types::CameraAngle::EyeLevel => 0.0,
            crate::types::CameraAngle::Unknown => 0.0,
        };

        interest.min(1.0)
    }

    /// Analyze shot variety in a sequence.
    #[must_use]
    pub fn analyze_variety(&self, shots: &[Shot]) -> VarietyMetrics {
        if shots.is_empty() {
            return VarietyMetrics::default();
        }

        // Count unique shot types
        let mut shot_types = std::collections::HashSet::new();
        let mut angles = std::collections::HashSet::new();
        let mut coverages = std::collections::HashSet::new();

        for shot in shots {
            shot_types.insert(shot.shot_type);
            angles.insert(shot.angle);
            coverages.insert(shot.coverage);
        }

        let shot_type_variety = shot_types.len() as f32 / 8.0; // 8 possible types
        let angle_variety = angles.len() as f32 / 6.0; // 6 possible angles
        let coverage_variety = coverages.len() as f32 / 9.0; // 9 possible coverages

        // Calculate transition variety
        let mut transition_types = std::collections::HashSet::new();
        for shot in shots {
            transition_types.insert(shot.transition);
        }
        let transition_variety = transition_types.len() as f32 / 11.0; // 11 possible transitions

        // Calculate movement variety
        let mut movement_types = std::collections::HashSet::new();
        for shot in shots {
            for movement in &shot.movements {
                movement_types.insert(movement.movement_type);
            }
        }
        let movement_variety = if movement_types.is_empty() {
            0.0
        } else {
            movement_types.len() as f32 / 12.0 // 12 possible movements
        };

        VarietyMetrics {
            shot_type_variety,
            angle_variety,
            coverage_variety,
            transition_variety,
            movement_variety,
            overall_variety: (shot_type_variety
                + angle_variety
                + coverage_variety
                + transition_variety
                + movement_variety)
                / 5.0,
        }
    }

    /// Detect shot patterns and repetition.
    #[must_use]
    pub fn detect_patterns(&self, shots: &[Shot]) -> Vec<PatternMatch> {
        let mut patterns = Vec::new();

        // Detect repeating shot type sequences
        let window_size = 3;
        for i in 0..shots.len().saturating_sub(window_size * 2) {
            let window1: Vec<ShotType> = shots[i..i + window_size]
                .iter()
                .map(|s| s.shot_type)
                .collect();

            for j in (i + window_size)..shots.len().saturating_sub(window_size) {
                let window2: Vec<ShotType> = shots[j..j + window_size]
                    .iter()
                    .map(|s| s.shot_type)
                    .collect();

                if window1 == window2 {
                    patterns.push(PatternMatch {
                        pattern_type: PatternType::ShotTypeSequence,
                        start_shot: i,
                        end_shot: i + window_size,
                        repeat_shot: j,
                        confidence: 1.0,
                    });
                }
            }
        }

        patterns
    }

    /// Calculate tension score based on editing.
    #[must_use]
    pub fn calculate_tension(&self, shots: &[Shot]) -> Vec<f32> {
        let mut tension_scores = Vec::new();

        for i in 0..shots.len() {
            let mut tension: f32 = 0.0;

            // Shorter shots increase tension
            let duration = shots[i].duration_seconds();
            if duration < 1.0 {
                tension += 0.5;
            } else if duration < 2.0 {
                tension += 0.3;
            }

            // Close-ups increase tension
            tension += match shots[i].shot_type {
                ShotType::ExtremeCloseUp => 0.4,
                ShotType::CloseUp => 0.3,
                ShotType::MediumCloseUp => 0.2,
                _ => 0.0,
            };

            // Camera movement increases tension
            if !shots[i].movements.is_empty() {
                tension += 0.2;
            }

            // Unusual angles increase tension
            tension += match shots[i].angle {
                crate::types::CameraAngle::High | crate::types::CameraAngle::Low => 0.1,
                crate::types::CameraAngle::Dutch => 0.2,
                _ => 0.0,
            };

            tension_scores.push(tension.min(1.0));
        }

        tension_scores
    }

    /// Identify climactic moments based on editing patterns.
    #[must_use]
    pub fn identify_climaxes(&self, shots: &[Shot]) -> Vec<ClimacticMoment> {
        let mut climaxes = Vec::new();
        let tension_scores = self.calculate_tension(shots);

        let window_size = 5;
        for i in window_size..shots.len().saturating_sub(window_size) {
            let window_tension: f32 = tension_scores
                [i.saturating_sub(window_size)..=i + window_size]
                .iter()
                .sum::<f32>()
                / (window_size * 2 + 1) as f32;

            if window_tension > 0.6 && tension_scores[i] > 0.7 {
                climaxes.push(ClimacticMoment {
                    shot_index: i,
                    tension_score: tension_scores[i],
                    timestamp: shots[i].start.to_seconds(),
                });
            }
        }

        climaxes
    }

    /// Analyze scene pacing.
    #[must_use]
    pub fn analyze_scene_pacing(&self, scenes: &[Scene], shots: &[Shot]) -> Vec<ScenePacing> {
        let mut pacing = Vec::new();

        for scene in scenes {
            let scene_shots: Vec<&Shot> = shots
                .iter()
                .filter(|s| scene.shots.contains(&s.id))
                .collect();

            if scene_shots.is_empty() {
                continue;
            }

            let durations: Vec<f64> = scene_shots.iter().map(|s| s.duration_seconds()).collect();
            let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;

            let tempo = if avg_duration > 0.0 {
                scene_shots.len() as f64 / avg_duration
            } else {
                0.0
            };

            // Calculate variance
            let variance = durations
                .iter()
                .map(|d| (d - avg_duration) * (d - avg_duration))
                .sum::<f64>()
                / durations.len() as f64;

            pacing.push(ScenePacing {
                scene_id: scene.id,
                shot_count: scene_shots.len(),
                average_shot_duration: avg_duration,
                tempo,
                variance,
                pacing_type: if tempo > 1.5 {
                    PacingType::Fast
                } else if tempo > 0.5 {
                    PacingType::Medium
                } else {
                    PacingType::Slow
                },
            });
        }

        pacing
    }
}

impl Default for AdvancedAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Variety metrics for shot analysis.
#[derive(Debug, Clone, Copy)]
pub struct VarietyMetrics {
    /// Shot type variety (0.0 to 1.0).
    pub shot_type_variety: f32,
    /// Angle variety (0.0 to 1.0).
    pub angle_variety: f32,
    /// Coverage variety (0.0 to 1.0).
    pub coverage_variety: f32,
    /// Transition variety (0.0 to 1.0).
    pub transition_variety: f32,
    /// Movement variety (0.0 to 1.0).
    pub movement_variety: f32,
    /// Overall variety (0.0 to 1.0).
    pub overall_variety: f32,
}

impl Default for VarietyMetrics {
    fn default() -> Self {
        Self {
            shot_type_variety: 0.0,
            angle_variety: 0.0,
            coverage_variety: 0.0,
            transition_variety: 0.0,
            movement_variety: 0.0,
            overall_variety: 0.0,
        }
    }
}

/// Pattern match result.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Type of pattern.
    pub pattern_type: PatternType,
    /// Start shot index.
    pub start_shot: usize,
    /// End shot index.
    pub end_shot: usize,
    /// Repeat shot index.
    pub repeat_shot: usize,
    /// Confidence score.
    pub confidence: f32,
}

/// Type of detected pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    /// Shot type sequence.
    ShotTypeSequence,
    /// Coverage pattern.
    CoveragePattern,
    /// Movement pattern.
    MovementPattern,
}

/// Climactic moment in the edit.
#[derive(Debug, Clone)]
pub struct ClimacticMoment {
    /// Shot index.
    pub shot_index: usize,
    /// Tension score.
    pub tension_score: f32,
    /// Timestamp.
    pub timestamp: f64,
}

/// Scene pacing analysis.
#[derive(Debug, Clone)]
pub struct ScenePacing {
    /// Scene ID.
    pub scene_id: u64,
    /// Number of shots.
    pub shot_count: usize,
    /// Average shot duration.
    pub average_shot_duration: f64,
    /// Tempo (shots per second).
    pub tempo: f64,
    /// Variance in shot durations.
    pub variance: f64,
    /// Pacing type.
    pub pacing_type: PacingType,
}

/// Type of pacing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingType {
    /// Slow pacing.
    Slow,
    /// Medium pacing.
    Medium,
    /// Fast pacing.
    Fast,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};

    #[test]
    fn test_advanced_analyzer_creation() {
        let _analyzer = AdvancedAnalyzer::new();
    }

    #[test]
    fn test_calculate_complexity() {
        let analyzer = AdvancedAnalyzer::new();
        let shot = Shot {
            id: 1,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(60, Rational::new(1, 30)),
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
            coverage: CoverageType::Master,
            confidence: 0.8,
            transition: TransitionType::Cut,
        };

        let complexity = analyzer.calculate_complexity(&shot);
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }

    #[test]
    fn test_analyze_variety_empty() {
        let analyzer = AdvancedAnalyzer::new();
        let metrics = analyzer.analyze_variety(&[]);
        assert!((metrics.overall_variety - 0.0).abs() < f32::EPSILON);
    }
}
