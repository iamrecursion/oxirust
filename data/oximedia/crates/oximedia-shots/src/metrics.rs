//! Shot metrics and quality assessment.

use crate::types::Shot;

/// Shot quality metrics calculator.
pub struct QualityMetrics;

impl QualityMetrics {
    /// Create a new quality metrics calculator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculate overall quality score for a shot.
    #[must_use]
    pub fn calculate_quality(&self, shot: &Shot) -> QualityScore {
        let composition_score = shot.composition.overall_score();
        let technical_score = self.calculate_technical_score(shot);
        let artistic_score = self.calculate_artistic_score(shot);

        let overall = (composition_score + technical_score + artistic_score) / 3.0;

        QualityScore {
            composition: composition_score,
            technical: technical_score,
            artistic: artistic_score,
            overall,
        }
    }

    /// Calculate technical quality score.
    fn calculate_technical_score(&self, shot: &Shot) -> f32 {
        let mut score = shot.confidence;

        // Penalize unknown classifications
        if shot.shot_type == crate::types::ShotType::Unknown {
            score *= 0.7;
        }

        if shot.angle == crate::types::CameraAngle::Unknown {
            score *= 0.8;
        }

        // Reward stable shots (no excessive handheld)
        let has_handheld = shot
            .movements
            .iter()
            .any(|m| m.movement_type == crate::types::MovementType::Handheld);

        if has_handheld {
            score *= 0.9;
        }

        score
    }

    /// Calculate artistic quality score.
    fn calculate_artistic_score(&self, shot: &Shot) -> f32 {
        let mut score: f32 = 0.5;

        // Reward intentional movements
        if !shot.movements.is_empty() {
            let intentional = shot.movements.iter().any(|m| {
                matches!(
                    m.movement_type,
                    crate::types::MovementType::PanLeft
                        | crate::types::MovementType::PanRight
                        | crate::types::MovementType::TiltUp
                        | crate::types::MovementType::TiltDown
                        | crate::types::MovementType::ZoomIn
                        | crate::types::MovementType::ZoomOut
                        | crate::types::MovementType::DollyIn
                        | crate::types::MovementType::DollyOut
                )
            });

            if intentional {
                score += 0.3;
            }
        }

        // Reward interesting angles
        score += match shot.angle {
            crate::types::CameraAngle::High | crate::types::CameraAngle::Low => 0.1,
            crate::types::CameraAngle::BirdsEye | crate::types::CameraAngle::Dutch => 0.2,
            _ => 0.0,
        };

        score.min(1.0)
    }

    /// Calculate consistency score across multiple shots.
    #[must_use]
    pub fn calculate_consistency(&self, shots: &[Shot]) -> ConsistencyMetrics {
        if shots.len() < 2 {
            return ConsistencyMetrics::default();
        }

        // Check for consistent shot types in scene
        let shot_types: Vec<_> = shots.iter().map(|s| s.shot_type).collect();
        let unique_types: std::collections::HashSet<_> = shot_types.iter().collect();
        let style_consistency = 1.0 - (unique_types.len() as f32 / shot_types.len() as f32);

        // Placeholder for color/exposure consistency (would need actual pixel data)
        let color_consistency = 0.8_f32;
        let exposure_consistency = 0.75_f32;

        let overall = (color_consistency + exposure_consistency + style_consistency) / 3.0;

        ConsistencyMetrics {
            color: color_consistency,
            exposure: exposure_consistency,
            style: style_consistency,
            overall,
        }
    }

    /// Detect potential quality issues.
    #[must_use]
    pub fn detect_issues(&self, shots: &[Shot]) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        for (i, shot) in shots.iter().enumerate() {
            // Check for very short shots
            if shot.duration_seconds() < 0.5 {
                issues.push(QualityIssue {
                    shot_index: i,
                    issue_type: IssueType::TooShort,
                    severity: IssueSeverity::Medium,
                    description: format!(
                        "Shot {} is very short ({:.2}s)",
                        shot.id,
                        shot.duration_seconds()
                    ),
                });
            }

            // Check for very long shots
            if shot.duration_seconds() > 60.0 {
                issues.push(QualityIssue {
                    shot_index: i,
                    issue_type: IssueType::TooLong,
                    severity: IssueSeverity::Low,
                    description: format!(
                        "Shot {} is very long ({:.2}s)",
                        shot.id,
                        shot.duration_seconds()
                    ),
                });
            }

            // Check for low confidence classifications
            if shot.confidence < 0.5 {
                issues.push(QualityIssue {
                    shot_index: i,
                    issue_type: IssueType::LowConfidence,
                    severity: IssueSeverity::Medium,
                    description: format!(
                        "Shot {} has low classification confidence ({:.2})",
                        shot.id, shot.confidence
                    ),
                });
            }

            // Check for poor composition
            if shot.composition.overall_score() < 0.3 {
                issues.push(QualityIssue {
                    shot_index: i,
                    issue_type: IssueType::PoorComposition,
                    severity: IssueSeverity::Low,
                    description: format!(
                        "Shot {} has poor composition (score: {:.2})",
                        shot.id,
                        shot.composition.overall_score()
                    ),
                });
            }

            // Check for excessive handheld shake
            let handheld_count = shot
                .movements
                .iter()
                .filter(|m| m.movement_type == crate::types::MovementType::Handheld)
                .count();

            if handheld_count > 3 {
                issues.push(QualityIssue {
                    shot_index: i,
                    issue_type: IssueType::ExcessiveShake,
                    severity: IssueSeverity::Medium,
                    description: format!("Shot {} has excessive camera shake", shot.id),
                });
            }
        }

        issues
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Quality score for a shot.
#[derive(Debug, Clone, Copy)]
pub struct QualityScore {
    /// Composition quality (0.0 to 1.0).
    pub composition: f32,
    /// Technical quality (0.0 to 1.0).
    pub technical: f32,
    /// Artistic quality (0.0 to 1.0).
    pub artistic: f32,
    /// Overall quality (0.0 to 1.0).
    pub overall: f32,
}

/// Consistency metrics.
#[derive(Debug, Clone, Copy)]
pub struct ConsistencyMetrics {
    /// Color consistency (0.0 to 1.0).
    pub color: f32,
    /// Exposure consistency (0.0 to 1.0).
    pub exposure: f32,
    /// Style consistency (0.0 to 1.0).
    pub style: f32,
    /// Overall consistency (0.0 to 1.0).
    pub overall: f32,
}

impl Default for ConsistencyMetrics {
    fn default() -> Self {
        Self {
            color: 0.0,
            exposure: 0.0,
            style: 0.0,
            overall: 0.0,
        }
    }
}

/// Quality issue detected in shots.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Shot index.
    pub shot_index: usize,
    /// Type of issue.
    pub issue_type: IssueType,
    /// Severity.
    pub severity: IssueSeverity,
    /// Description.
    pub description: String,
}

/// Type of quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueType {
    /// Shot is too short.
    TooShort,
    /// Shot is too long.
    TooLong,
    /// Low classification confidence.
    LowConfidence,
    /// Poor composition.
    PoorComposition,
    /// Excessive camera shake.
    ExcessiveShake,
    /// Inconsistent exposure.
    InconsistentExposure,
    /// Inconsistent color grading.
    InconsistentColor,
}

/// Severity of quality issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Low severity.
    Low,
    /// Medium severity.
    Medium,
    /// High severity.
    High,
}

/// Performance metrics for shot detection.
pub struct PerformanceMetrics {
    /// Total processing time in seconds.
    pub processing_time: f64,
    /// Frames processed per second.
    pub fps: f64,
    /// Total frames processed.
    pub frames_processed: usize,
    /// Total shots detected.
    pub shots_detected: usize,
}

impl PerformanceMetrics {
    /// Create new performance metrics.
    #[must_use]
    pub const fn new(processing_time: f64, frames_processed: usize, shots_detected: usize) -> Self {
        let fps = if processing_time > 0.0 {
            frames_processed as f64 / processing_time
        } else {
            0.0
        };

        Self {
            processing_time,
            fps,
            frames_processed,
            shots_detected,
        }
    }

    /// Calculate average time per shot.
    #[must_use]
    pub fn time_per_shot(&self) -> f64 {
        if self.shots_detected > 0 {
            self.processing_time / self.shots_detected as f64
        } else {
            0.0
        }
    }

    /// Generate performance report.
    #[must_use]
    pub fn generate_report(&self) -> String {
        format!(
            "Performance Report:\n\
             - Processing time: {:.2}s\n\
             - Frames processed: {}\n\
             - Processing speed: {:.2} fps\n\
             - Shots detected: {}\n\
             - Time per shot: {:.2}s\n",
            self.processing_time,
            self.frames_processed,
            self.fps,
            self.shots_detected,
            self.time_per_shot()
        )
    }
}

/// Statistical analyzer for shot data.
pub struct StatisticalAnalyzer;

impl StatisticalAnalyzer {
    /// Create a new statistical analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Calculate percentiles for shot durations.
    #[must_use]
    pub fn calculate_percentiles(&self, shots: &[Shot]) -> Percentiles {
        if shots.is_empty() {
            return Percentiles::default();
        }

        let mut durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = durations.len();

        Percentiles {
            p25: durations[len / 4],
            p50: durations[len / 2],
            p75: durations[3 * len / 4],
            p90: durations[(9 * len) / 10],
            p95: durations[(95 * len) / 100],
            p99: durations[(99 * len) / 100],
        }
    }

    /// Calculate standard deviation of shot durations.
    #[must_use]
    pub fn calculate_std_dev(&self, shots: &[Shot]) -> f64 {
        if shots.is_empty() {
            return 0.0;
        }

        let durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        let mean = durations.iter().sum::<f64>() / durations.len() as f64;

        let variance = durations
            .iter()
            .map(|d| (d - mean) * (d - mean))
            .sum::<f64>()
            / durations.len() as f64;

        variance.sqrt()
    }

    /// Calculate coefficient of variation.
    #[must_use]
    pub fn calculate_cv(&self, shots: &[Shot]) -> f64 {
        if shots.is_empty() {
            return 0.0;
        }

        let durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        let mean = durations.iter().sum::<f64>() / durations.len() as f64;

        if mean == 0.0 {
            return 0.0;
        }

        let std_dev = self.calculate_std_dev(shots);
        std_dev / mean
    }

    /// Calculate correlation between shot duration and complexity.
    #[must_use]
    pub fn calculate_duration_complexity_correlation(&self, shots: &[Shot]) -> f64 {
        if shots.len() < 2 {
            return 0.0;
        }

        let durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        let complexities: Vec<f64> = shots
            .iter()
            .map(|s| crate::analysis::AdvancedAnalyzer::new().calculate_complexity(s) as f64)
            .collect();

        self.pearson_correlation(&durations, &complexities)
    }

    /// Calculate Pearson correlation coefficient.
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;

            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Duration percentiles.
#[derive(Debug, Clone, Copy)]
pub struct Percentiles {
    /// 25th percentile.
    pub p25: f64,
    /// 50th percentile (median).
    pub p50: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 90th percentile.
    pub p90: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl Default for Percentiles {
    fn default() -> Self {
        Self {
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, ShotType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};

    fn create_test_shot(id: u64, duration: i64) -> Shot {
        Shot {
            id,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(duration, Rational::new(1, 30)),
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
        }
    }

    #[test]
    fn test_quality_metrics() {
        let metrics = QualityMetrics::new();
        let shot = create_test_shot(1, 60);
        let quality = metrics.calculate_quality(&shot);
        assert!(quality.overall >= 0.0 && quality.overall <= 1.0);
    }

    #[test]
    fn test_detect_issues() {
        let metrics = QualityMetrics::new();
        let short_shot = create_test_shot(1, 10); // Very short shot
        let issues = metrics.detect_issues(&[short_shot]);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_performance_metrics() {
        let perf = PerformanceMetrics::new(10.0, 300, 20);
        assert!((perf.fps - 30.0).abs() < f64::EPSILON);
        assert!((perf.time_per_shot() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_statistical_analyzer() {
        let analyzer = StatisticalAnalyzer::new();
        let shots = vec![
            create_test_shot(1, 30),
            create_test_shot(2, 60),
            create_test_shot(3, 90),
        ];

        let percentiles = analyzer.calculate_percentiles(&shots);
        assert!(percentiles.p50 > 0.0);

        let std_dev = analyzer.calculate_std_dev(&shots);
        assert!(std_dev > 0.0);
    }
}
