//! Audio description generation and management.
//!
//! Audio description provides narration of visual content for blind and visually
//! impaired users. This module supports multiple AD types and mixing strategies.

pub mod ad_scene;
pub mod generator;
pub mod mix;
pub mod scene_analysis;
pub mod script;
pub mod template;
pub mod timing;

pub use generator::{AudioDescriptionConfig, AudioDescriptionGenerator};
pub use mix::{AudioDescriptionMixer, MixConfig, MixStrategy};
pub use scene_analysis::{
    gaps_from_silence, SceneAudioDescriber, SceneDescriptionConfig, SceneTrigger,
    DEFAULT_AD_WORDS_PER_MINUTE,
};
pub use script::{AudioDescriptionEntry, AudioDescriptionScript};
pub use template::{
    ActionDescriptor, DescriptionStyle, SceneContext, SceneTemplateEngine, SettingDescriptor,
    SubjectDescriptor,
};
pub use timing::{Gap, TimingAnalyzer, TimingConstraints};

use serde::{Deserialize, Serialize};

/// Type of audio description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioDescriptionType {
    /// Standard audio description inserted in dialogue gaps.
    Standard,
    /// Extended audio description with video pause for longer descriptions.
    Extended,
    /// Open audio description always present in the main audio.
    Open,
    /// Closed audio description available as separate track.
    Closed,
}

impl AudioDescriptionType {
    /// Check if this type allows pausing the video.
    #[must_use]
    pub const fn allows_pause(&self) -> bool {
        matches!(self, Self::Extended)
    }

    /// Check if this type is mixed into main audio.
    #[must_use]
    pub const fn is_mixed(&self) -> bool {
        matches!(self, Self::Open)
    }

    /// Check if this type is a separate track.
    #[must_use]
    pub const fn is_separate_track(&self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// Audio description quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioDescriptionQuality {
    /// Basic quality suitable for testing.
    Basic,
    /// Standard broadcast quality.
    Standard,
    /// High quality for premium content.
    High,
    /// Professional quality with human review.
    Professional,
}

impl AudioDescriptionQuality {
    /// Get minimum description duration in milliseconds.
    #[must_use]
    pub const fn min_duration_ms(&self) -> i64 {
        match self {
            Self::Basic => 500,
            Self::Standard => 1000,
            Self::High => 1500,
            Self::Professional => 2000,
        }
    }

    /// Get minimum gap before next dialogue in milliseconds.
    #[must_use]
    pub const fn min_gap_after_ms(&self) -> i64 {
        match self {
            Self::Basic => 100,
            Self::Standard => 200,
            Self::High => 300,
            Self::Professional => 500,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ad_type_properties() {
        assert!(AudioDescriptionType::Extended.allows_pause());
        assert!(!AudioDescriptionType::Standard.allows_pause());
        assert!(AudioDescriptionType::Open.is_mixed());
        assert!(AudioDescriptionType::Closed.is_separate_track());
    }

    #[test]
    fn test_quality_constraints() {
        let basic = AudioDescriptionQuality::Basic;
        let professional = AudioDescriptionQuality::Professional;

        assert!(basic.min_duration_ms() < professional.min_duration_ms());
        assert!(basic.min_gap_after_ms() < professional.min_gap_after_ms());
    }
}
