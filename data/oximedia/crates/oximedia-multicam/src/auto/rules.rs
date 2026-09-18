//! Switching rules engine for automatic camera selection.

use super::score::AngleScore;
use crate::{AngleId, FrameNumber};

/// Switching rule
#[derive(Debug, Clone)]
pub enum SwitchingRule {
    /// Speaker detection - switch to active speaker
    SpeakerDetection {
        /// Sensitivity (0.0 to 1.0)
        sensitivity: f32,
    },

    /// Action following - follow movement and action
    ActionFollowing {
        /// Smoothness (0.0 to 1.0)
        smoothness: f32,
    },

    /// Shot variety - maintain variety in shots
    ShotVariety {
        /// Minimum duration per angle (ms)
        min_duration_ms: u32,
    },

    /// Composition bias - prefer better composed shots
    CompositionBias {
        /// Bias strength (0.0 to 1.0)
        strength: f32,
    },

    /// Face detection - prefer angles with visible faces
    FaceDetection {
        /// Minimum face size (pixels)
        min_face_size: u32,
    },

    /// Audio level - follow audio activity
    AudioLevel {
        /// Threshold (dB)
        threshold_db: f32,
    },

    /// Prefer wide shots
    PreferWide {
        /// Bias strength
        strength: f32,
    },

    /// Prefer close-ups
    PreferCloseUp {
        /// Bias strength
        strength: f32,
    },

    /// Motion detection - follow motion
    MotionDetection {
        /// Sensitivity
        sensitivity: f32,
    },

    /// Stay on current angle (anti-flip-flop)
    StickyCurrent {
        /// Stickiness factor (0.0 to 1.0)
        stickiness: f32,
    },
}

/// Rules engine
#[derive(Debug)]
pub struct RuleEngine {
    /// Active rules
    rules: Vec<SwitchingRule>,
}

impl RuleEngine {
    /// Create a new rule engine
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: SwitchingRule) {
        self.rules.push(rule);
    }

    /// Remove all rules
    pub fn clear_rules(&mut self) {
        self.rules.clear();
    }

    /// Get rules
    #[must_use]
    pub fn rules(&self) -> &[SwitchingRule] {
        &self.rules
    }

    /// Apply all rules to angle scores
    #[must_use]
    pub fn apply_rules(
        &self,
        scores: &[AngleScore],
        current_angle: AngleId,
        _current_frame: FrameNumber,
    ) -> Vec<AngleScore> {
        let mut modified = scores.to_vec();

        for rule in &self.rules {
            modified = self.apply_rule(rule, &modified, current_angle);
        }

        modified
    }

    /// Apply a single rule
    fn apply_rule(
        &self,
        rule: &SwitchingRule,
        scores: &[AngleScore],
        current_angle: AngleId,
    ) -> Vec<AngleScore> {
        let mut result = scores.to_vec();

        match rule {
            SwitchingRule::SpeakerDetection { sensitivity } => {
                self.apply_speaker_detection(&mut result, *sensitivity);
            }
            SwitchingRule::ActionFollowing { smoothness } => {
                self.apply_action_following(&mut result, *smoothness);
            }
            SwitchingRule::ShotVariety { min_duration_ms: _ } => {
                // Shot variety is handled at a higher level
            }
            SwitchingRule::CompositionBias { strength } => {
                self.apply_composition_bias(&mut result, *strength);
            }
            SwitchingRule::FaceDetection { min_face_size: _ } => {
                self.apply_face_detection(&mut result);
            }
            SwitchingRule::AudioLevel { threshold_db: _ } => {
                self.apply_audio_level(&mut result);
            }
            SwitchingRule::PreferWide { strength } => {
                self.apply_prefer_wide(&mut result, *strength);
            }
            SwitchingRule::PreferCloseUp { strength } => {
                self.apply_prefer_closeup(&mut result, *strength);
            }
            SwitchingRule::MotionDetection { sensitivity } => {
                self.apply_motion_detection(&mut result, *sensitivity);
            }
            SwitchingRule::StickyCurrent { stickiness } => {
                self.apply_sticky_current(&mut result, current_angle, *stickiness);
            }
        }

        result
    }

    /// Apply speaker detection rule
    fn apply_speaker_detection(&self, scores: &mut [AngleScore], sensitivity: f32) {
        for score in scores {
            score.total_score += score.audio_score * sensitivity;
        }
    }

    /// Apply action following rule
    fn apply_action_following(&self, scores: &mut [AngleScore], smoothness: f32) {
        for score in scores {
            score.total_score += score.motion_score * (1.0 - smoothness);
        }
    }

    /// Apply composition bias rule
    fn apply_composition_bias(&self, scores: &mut [AngleScore], strength: f32) {
        for score in scores {
            score.total_score += score.composition_score * strength;
        }
    }

    /// Apply face detection rule
    fn apply_face_detection(&self, scores: &mut [AngleScore]) {
        for score in scores {
            score.total_score += score.face_score * 0.5;
        }
    }

    /// Apply audio level rule
    fn apply_audio_level(&self, scores: &mut [AngleScore]) {
        for score in scores {
            score.total_score += score.audio_score * 0.3;
        }
    }

    /// Apply prefer wide shots rule
    fn apply_prefer_wide(&self, scores: &mut [AngleScore], strength: f32) {
        // Assume wider shots have lower face scores
        for score in scores {
            if score.face_score < 0.3 {
                score.total_score += strength;
            }
        }
    }

    /// Apply prefer close-up shots rule
    fn apply_prefer_closeup(&self, scores: &mut [AngleScore], strength: f32) {
        // Assume close-ups have higher face scores
        for score in scores {
            if score.face_score > 0.7 {
                score.total_score += strength;
            }
        }
    }

    /// Apply motion detection rule
    fn apply_motion_detection(&self, scores: &mut [AngleScore], sensitivity: f32) {
        for score in scores {
            score.total_score += score.motion_score * sensitivity;
        }
    }

    /// Apply sticky current angle rule
    fn apply_sticky_current(
        &self,
        scores: &mut [AngleScore],
        current_angle: AngleId,
        stickiness: f32,
    ) {
        if current_angle < scores.len() {
            scores[current_angle].total_score += stickiness;
        }
    }

    /// Create default rule set for talk show
    #[must_use]
    pub fn talk_show_rules() -> Self {
        let mut engine = Self::new();
        engine.add_rule(SwitchingRule::SpeakerDetection { sensitivity: 0.8 });
        engine.add_rule(SwitchingRule::FaceDetection { min_face_size: 100 });
        engine.add_rule(SwitchingRule::StickyCurrent { stickiness: 0.3 });
        engine.add_rule(SwitchingRule::ShotVariety {
            min_duration_ms: 2000,
        });
        engine
    }

    /// Create default rule set for sports
    #[must_use]
    pub fn sports_rules() -> Self {
        let mut engine = Self::new();
        engine.add_rule(SwitchingRule::ActionFollowing { smoothness: 0.3 });
        engine.add_rule(SwitchingRule::MotionDetection { sensitivity: 0.9 });
        engine.add_rule(SwitchingRule::PreferWide { strength: 0.5 });
        engine
    }

    /// Create default rule set for concert
    #[must_use]
    pub fn concert_rules() -> Self {
        let mut engine = Self::new();
        engine.add_rule(SwitchingRule::AudioLevel {
            threshold_db: -20.0,
        });
        engine.add_rule(SwitchingRule::ActionFollowing { smoothness: 0.5 });
        engine.add_rule(SwitchingRule::ShotVariety {
            min_duration_ms: 3000,
        });
        engine
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_engine_creation() {
        let engine = RuleEngine::new();
        assert_eq!(engine.rules().len(), 0);
    }

    #[test]
    fn test_add_rule() {
        let mut engine = RuleEngine::new();
        engine.add_rule(SwitchingRule::SpeakerDetection { sensitivity: 0.8 });
        assert_eq!(engine.rules().len(), 1);
    }

    #[test]
    fn test_clear_rules() {
        let mut engine = RuleEngine::new();
        engine.add_rule(SwitchingRule::SpeakerDetection { sensitivity: 0.8 });
        engine.clear_rules();
        assert_eq!(engine.rules().len(), 0);
    }

    #[test]
    fn test_talk_show_rules() {
        let engine = RuleEngine::talk_show_rules();
        assert!(!engine.rules().is_empty());
    }

    #[test]
    fn test_sports_rules() {
        let engine = RuleEngine::sports_rules();
        assert!(!engine.rules().is_empty());
    }

    #[test]
    fn test_concert_rules() {
        let engine = RuleEngine::concert_rules();
        assert!(!engine.rules().is_empty());
    }
}
