//! Playback speed control with pitch preservation.

pub mod adaptation;
pub mod control;
pub mod pitch;

pub use control::SpeedController;
pub use pitch::PitchPreserver;

use serde::{Deserialize, Serialize};

/// Speed control configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedConfig {
    /// Playback speed multiplier (0.5 to 2.0).
    pub speed: f32,
    /// Preserve pitch when changing speed.
    pub preserve_pitch: bool,
    /// Quality level.
    pub quality: SpeedQuality,
}

/// Quality level for speed adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeedQuality {
    /// Fast processing, lower quality.
    Fast,
    /// Standard quality.
    Standard,
    /// High quality, slower processing.
    High,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        Self {
            speed: 1.0,
            preserve_pitch: true,
            quality: SpeedQuality::Standard,
        }
    }
}
