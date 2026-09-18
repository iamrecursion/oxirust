//! Shot coverage mapping - cinematography coverage types and angles.

/// Cinematographic coverage type for shot planning.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoverageType {
    /// Wide shot establishing the scene.
    Wide,
    /// Medium shot.
    Medium,
    /// Close-up shot.
    Close,
    /// Extreme close-up.
    Extreme,
    /// Aerial shot.
    Aerial,
    /// Insert (detail) shot.
    Insert,
    /// Cutaway shot.
    CutAway,
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
            Self::Extreme => "Extreme Close-up",
            Self::Aerial => "Aerial",
            Self::Insert => "Insert",
            Self::CutAway => "Cut Away",
        }
    }
}

/// Camera angle for shot coverage.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShotAngle {
    /// Eye-level angle.
    Eye,
    /// High angle.
    High,
    /// Low angle.
    Low,
    /// Dutch (tilted) angle.
    Dutch,
    /// Bird's eye view.
    BirdsEye,
    /// Worm's eye view.
    WormEye,
}

impl ShotAngle {
    /// Human-readable name.
    #[allow(dead_code)]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Eye => "Eye Level",
            Self::High => "High Angle",
            Self::Low => "Low Angle",
            Self::Dutch => "Dutch Angle",
            Self::BirdsEye => "Bird's Eye",
            Self::WormEye => "Worm's Eye",
        }
    }
}

/// A map recording coverage type and angle for each shot.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct CoverageMap {
    /// `(shot_id, coverage_type, shot_angle)` entries.
    pub shots: Vec<(u64, CoverageType, ShotAngle)>,
}

impl CoverageMap {
    /// Create a new empty coverage map.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self { shots: Vec::new() }
    }

    /// Add a shot entry.
    #[allow(dead_code)]
    pub fn add(&mut self, shot_id: u64, coverage: CoverageType, angle: ShotAngle) {
        self.shots.push((shot_id, coverage, angle));
    }

    /// Get the number of shots in the map.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.shots.len()
    }

    /// Returns true if the map is empty.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shots.is_empty()
    }
}

impl Default for CoverageMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Coverage analysis report for a sequence of shots.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct CoverageReport {
    /// Primary (most common) coverage type.
    pub primary_coverage: CoverageType,
    /// Percentage of wide shots (0.0..=1.0).
    pub wide_pct: f32,
    /// Percentage of medium shots.
    pub medium_pct: f32,
    /// Percentage of close shots.
    pub close_pct: f32,
    /// Variety score (0.0..=1.0).
    pub variety_score: f32,
}

/// Analyzes coverage patterns across a sequence of shots.
#[allow(dead_code)]
pub struct CoverageAnalyzer;

impl CoverageAnalyzer {
    /// Create a new analyzer.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyze a slice of `(coverage_type, percentage)` pairs.
    ///
    /// `duration_pct` should sum to approximately 1.0 but this is not enforced.
    #[allow(dead_code)]
    #[must_use]
    pub fn analyze(duration_pct: &[(CoverageType, f32)]) -> CoverageReport {
        if duration_pct.is_empty() {
            return CoverageReport {
                primary_coverage: CoverageType::Medium,
                wide_pct: 0.0,
                medium_pct: 0.0,
                close_pct: 0.0,
                variety_score: 0.0,
            };
        }

        // Find primary coverage (highest percentage)
        let primary_coverage = duration_pct
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(ct, _)| *ct)
            .unwrap_or(CoverageType::Medium);

        let wide_pct: f32 = duration_pct
            .iter()
            .filter(|(ct, _)| *ct == CoverageType::Wide)
            .map(|(_, p)| *p)
            .sum();
        let medium_pct: f32 = duration_pct
            .iter()
            .filter(|(ct, _)| *ct == CoverageType::Medium)
            .map(|(_, p)| *p)
            .sum();
        let close_pct: f32 = duration_pct
            .iter()
            .filter(|(ct, _)| matches!(ct, CoverageType::Close | CoverageType::Extreme))
            .map(|(_, p)| *p)
            .sum();

        // Unique coverage types
        let unique_types: std::collections::HashSet<_> =
            duration_pct.iter().map(|(ct, _)| *ct).collect();
        let type_count = unique_types.len();
        let total_types = 7_usize; // total variants
        let variety_score = type_count as f32 / total_types as f32;

        CoverageReport {
            primary_coverage,
            wide_pct,
            medium_pct,
            close_pct,
            variety_score,
        }
    }
}

impl Default for CoverageAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Severity of an eyeline issue.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Minor issue (small angle discrepancy).
    Minor,
    /// Moderate issue.
    Moderate,
    /// Severe issue (clear 180-rule violation).
    Severe,
}

/// An eyeline continuity issue between two shots.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct EyelineIssue {
    /// Shot IDs involved in the issue.
    pub between_shots: (u64, u64),
    /// Absolute angle difference in degrees.
    pub angle_diff: f32,
    /// Severity of the issue.
    pub severity: IssueSeverity,
}

/// Checks eyeline continuity between shots.
#[allow(dead_code)]
pub struct EyelineChecker;

impl EyelineChecker {
    /// Check look-direction continuity across a sequence of shots.
    ///
    /// `shots` is a slice of `(shot_id, look_direction_degrees)`.
    ///
    /// A violation occurs when consecutive shots have a look direction change > 90°.
    #[allow(dead_code)]
    #[must_use]
    pub fn check_continuity(shots: &[(u64, f32)]) -> Vec<EyelineIssue> {
        let mut issues = Vec::new();

        for window in shots.windows(2) {
            let (id_a, dir_a) = window[0];
            let (id_b, dir_b) = window[1];

            // Normalized angular difference: smallest angle between two directions
            let raw_diff = (dir_b - dir_a).abs() % 360.0;
            let angle_diff = if raw_diff > 180.0 {
                360.0 - raw_diff
            } else {
                raw_diff
            };

            let severity = if angle_diff > 170.0 {
                Some(IssueSeverity::Severe)
            } else if angle_diff > 120.0 {
                Some(IssueSeverity::Moderate)
            } else if angle_diff > 90.0 {
                Some(IssueSeverity::Minor)
            } else {
                None
            };

            if let Some(severity) = severity {
                issues.push(EyelineIssue {
                    between_shots: (id_a, id_b),
                    angle_diff,
                    severity,
                });
            }
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_type_name() {
        assert_eq!(CoverageType::Wide.name(), "Wide");
        assert_eq!(CoverageType::Close.name(), "Close");
        assert_eq!(CoverageType::CutAway.name(), "Cut Away");
    }

    #[test]
    fn test_shot_angle_name() {
        assert_eq!(ShotAngle::Eye.name(), "Eye Level");
        assert_eq!(ShotAngle::BirdsEye.name(), "Bird's Eye");
    }

    #[test]
    fn test_coverage_map_add() {
        let mut map = CoverageMap::new();
        map.add(1, CoverageType::Wide, ShotAngle::Eye);
        map.add(2, CoverageType::Close, ShotAngle::Low);
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_coverage_map_empty() {
        let map = CoverageMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_coverage_analyzer_empty() {
        let report = CoverageAnalyzer::analyze(&[]);
        assert_eq!(report.wide_pct, 0.0);
        assert_eq!(report.variety_score, 0.0);
    }

    #[test]
    fn test_coverage_analyzer_single_type() {
        let data = vec![(CoverageType::Wide, 1.0_f32)];
        let report = CoverageAnalyzer::analyze(&data);
        assert_eq!(report.primary_coverage, CoverageType::Wide);
        assert!((report.wide_pct - 1.0).abs() < 1e-6);
        assert!((report.medium_pct).abs() < 1e-6);
    }

    #[test]
    fn test_coverage_analyzer_mixed() {
        let data = vec![
            (CoverageType::Wide, 0.5_f32),
            (CoverageType::Medium, 0.3),
            (CoverageType::Close, 0.2),
        ];
        let report = CoverageAnalyzer::analyze(&data);
        assert_eq!(report.primary_coverage, CoverageType::Wide);
        assert!((report.wide_pct - 0.5).abs() < 1e-6);
        assert!((report.medium_pct - 0.3).abs() < 1e-6);
        // 3 unique types out of 7
        assert!((report.variety_score - 3.0 / 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_eyeline_no_issues() {
        let shots = vec![(1, 0.0_f32), (2, 45.0), (3, 80.0)];
        let issues = EyelineChecker::check_continuity(&shots);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_eyeline_severe_violation() {
        // 180° look direction reversal
        let shots = vec![(1, 0.0_f32), (2, 180.0)];
        let issues = EyelineChecker::check_continuity(&shots);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Severe);
    }

    #[test]
    fn test_eyeline_moderate_violation() {
        let shots = vec![(1, 0.0_f32), (2, 150.0)];
        let issues = EyelineChecker::check_continuity(&shots);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Moderate);
    }

    #[test]
    fn test_eyeline_minor_violation() {
        let shots = vec![(1, 0.0_f32), (2, 100.0)];
        let issues = EyelineChecker::check_continuity(&shots);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, IssueSeverity::Minor);
    }

    #[test]
    fn test_eyeline_empty() {
        let issues = EyelineChecker::check_continuity(&[]);
        assert!(issues.is_empty());
    }
}
