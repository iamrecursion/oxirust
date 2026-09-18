//! Automatic camera selection for multi-camera production.

pub mod cache;
pub mod rules;
pub mod score;
pub mod select;

/// Automatic switcher
pub use select::AutoSwitcher;

/// Switching rules
pub use rules::{RuleEngine, SwitchingRule};

/// Angle scoring
pub use score::{AngleScorer, ScoringCriteria};

/// Angle score cache
pub use cache::AngleScoreCache;

/// Selection criteria for automatic switching
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Enable face detection
    pub face_detection: bool,
    /// Enable composition quality analysis
    pub composition_quality: bool,
    /// Enable audio activity detection
    pub audio_activity: bool,
    /// Enable motion detection
    pub motion_detection: bool,
    /// Enable speaker detection
    pub speaker_detection: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self {
            face_detection: true,
            composition_quality: true,
            audio_activity: true,
            motion_detection: true,
            speaker_detection: true,
            min_confidence: 0.7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_criteria() {
        let criteria = SelectionCriteria::default();
        assert!(criteria.face_detection);
        assert!(criteria.audio_activity);
        assert_eq!(criteria.min_confidence, 0.7);
    }
}
