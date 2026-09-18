//! Audio/video crossfade transitions.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Type of crossfade curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CrossfadeType {
    /// Linear crossfade.
    Linear,

    /// Equal power crossfade (for audio).
    EqualPower,

    /// Exponential crossfade.
    Exponential,

    /// S-curve crossfade.
    SCurve,

    /// Custom curve with power value.
    Custom {
        /// Power value for the curve.
        power: f32,
    },
}

/// Crossfade configuration for audio and video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crossfade {
    /// Duration of the crossfade.
    pub duration: Duration,

    /// Video crossfade type.
    pub video_type: CrossfadeType,

    /// Audio crossfade type.
    pub audio_type: CrossfadeType,

    /// Whether to crossfade video.
    pub video_enabled: bool,

    /// Whether to crossfade audio.
    pub audio_enabled: bool,
}

impl Crossfade {
    /// Creates a new crossfade with default settings.
    #[must_use]
    pub const fn new(duration: Duration) -> Self {
        Self {
            duration,
            video_type: CrossfadeType::Linear,
            audio_type: CrossfadeType::EqualPower,
            video_enabled: true,
            audio_enabled: true,
        }
    }

    /// Sets the video crossfade type.
    #[must_use]
    pub const fn with_video_type(mut self, video_type: CrossfadeType) -> Self {
        self.video_type = video_type;
        self
    }

    /// Sets the audio crossfade type.
    #[must_use]
    pub const fn with_audio_type(mut self, audio_type: CrossfadeType) -> Self {
        self.audio_type = audio_type;
        self
    }

    /// Disables video crossfade.
    #[must_use]
    pub const fn without_video(mut self) -> Self {
        self.video_enabled = false;
        self
    }

    /// Disables audio crossfade.
    #[must_use]
    pub const fn without_audio(mut self) -> Self {
        self.audio_enabled = false;
        self
    }

    /// Calculates the fade value at a specific time using the video curve.
    ///
    /// # Arguments
    ///
    /// * `time` - Current time within the crossfade (0 to duration)
    ///
    /// # Returns
    ///
    /// Value between 0.0 and 1.0 representing the fade level
    #[must_use]
    pub fn video_fade_value(&self, time: Duration) -> f32 {
        self.calculate_fade_value(time, self.video_type)
    }

    /// Calculates the fade value at a specific time using the audio curve.
    ///
    /// # Arguments
    ///
    /// * `time` - Current time within the crossfade (0 to duration)
    ///
    /// # Returns
    ///
    /// Value between 0.0 and 1.0 representing the fade level
    #[must_use]
    pub fn audio_fade_value(&self, time: Duration) -> f32 {
        self.calculate_fade_value(time, self.audio_type)
    }

    fn calculate_fade_value(&self, time: Duration, fade_type: CrossfadeType) -> f32 {
        if self.duration.is_zero() {
            return 1.0;
        }

        let t = time.as_secs_f32() / self.duration.as_secs_f32();
        let t = t.clamp(0.0, 1.0);

        match fade_type {
            CrossfadeType::Linear => t,
            CrossfadeType::EqualPower => {
                // Equal power crossfade for audio
                (t * std::f32::consts::FRAC_PI_2).sin()
            }
            CrossfadeType::Exponential => t * t,
            CrossfadeType::SCurve => {
                // Smoothstep S-curve
                t * t * (3.0 - 2.0 * t)
            }
            CrossfadeType::Custom { power } => t.powf(power),
        }
    }
}

impl Default for Crossfade {
    fn default() -> Self {
        Self::new(Duration::from_secs(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crossfade() {
        let crossfade = Crossfade::new(Duration::from_secs(2))
            .with_video_type(CrossfadeType::Linear)
            .with_audio_type(CrossfadeType::EqualPower);

        assert_eq!(crossfade.duration, Duration::from_secs(2));
        assert!(crossfade.video_enabled);
        assert!(crossfade.audio_enabled);
    }

    #[test]
    fn test_fade_values() {
        let crossfade = Crossfade::new(Duration::from_secs(1));

        let value_start = crossfade.video_fade_value(Duration::ZERO);
        assert!((value_start - 0.0).abs() < f32::EPSILON);

        let value_end = crossfade.video_fade_value(Duration::from_secs(1));
        assert!((value_end - 1.0).abs() < f32::EPSILON);

        let value_mid = crossfade.video_fade_value(Duration::from_millis(500));
        assert!(value_mid > 0.4 && value_mid < 0.6);
    }

    #[test]
    fn test_equal_power_crossfade() {
        let crossfade =
            Crossfade::new(Duration::from_secs(1)).with_audio_type(CrossfadeType::EqualPower);

        let value = crossfade.audio_fade_value(Duration::from_millis(500));
        // Equal power at 50% should be around 0.707 (sqrt(0.5))
        assert!(value > 0.6 && value < 0.8);
    }
}
