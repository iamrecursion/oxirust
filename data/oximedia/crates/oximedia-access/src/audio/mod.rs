//! Audio enhancements for accessibility.

pub mod clarity;
pub mod noise;
pub mod normalize;

pub use clarity::AudioClarityEnhancer;
pub use noise::NoiseReducer;
pub use normalize::LoudnessNormalizer;

use serde::{Deserialize, Serialize};

/// Audio enhancement configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEnhancementConfig {
    /// Noise reduction level (0.0 to 1.0).
    pub noise_reduction: f32,
    /// Clarity enhancement level (0.0 to 1.0).
    pub clarity_enhancement: f32,
    /// Target loudness in LUFS.
    pub target_loudness: f32,
}

impl Default for AudioEnhancementConfig {
    fn default() -> Self {
        Self {
            noise_reduction: 0.5,
            clarity_enhancement: 0.5,
            target_loudness: -23.0, // EBU R128 standard
        }
    }
}
