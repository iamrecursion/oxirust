//! Audio mixing and processing for timeline.

use serde::{Deserialize, Serialize};

use crate::types::Position;

/// Audio fade type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FadeType {
    /// Linear fade.
    Linear,
    /// Logarithmic fade (more natural for audio).
    Logarithmic,
    /// Exponential fade.
    Exponential,
    /// S-curve fade.
    SCurve,
}

/// Audio fade configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioFade {
    /// Type of fade.
    pub fade_type: FadeType,
    /// Start position.
    pub start: Position,
    /// End position.
    pub end: Position,
    /// Start gain (0.0-1.0).
    pub start_gain: f32,
    /// End gain (0.0-1.0).
    pub end_gain: f32,
}

impl AudioFade {
    /// Creates a new audio fade.
    #[must_use]
    pub fn new(fade_type: FadeType, start: Position, end: Position) -> Self {
        Self {
            fade_type,
            start,
            end,
            start_gain: 1.0,
            end_gain: 0.0,
        }
    }

    /// Creates a fade-in.
    #[must_use]
    pub fn fade_in(start: Position, end: Position) -> Self {
        Self {
            fade_type: FadeType::Logarithmic,
            start,
            end,
            start_gain: 0.0,
            end_gain: 1.0,
        }
    }

    /// Creates a fade-out.
    #[must_use]
    pub fn fade_out(start: Position, end: Position) -> Self {
        Self {
            fade_type: FadeType::Logarithmic,
            start,
            end,
            start_gain: 1.0,
            end_gain: 0.0,
        }
    }

    /// Calculates gain at a given position.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn gain_at(&self, position: Position) -> f32 {
        if position < self.start {
            return self.start_gain;
        }
        if position >= self.end {
            return self.end_gain;
        }

        let range = (self.end.value() - self.start.value()) as f32;
        let offset = (position.value() - self.start.value()) as f32;
        let t = offset / range;

        let curve_t = match self.fade_type {
            FadeType::Linear => t,
            FadeType::Logarithmic => {
                // Logarithmic curve
                if t < 0.01 {
                    0.0
                } else {
                    (t.ln() + 4.605) / 4.605 // ln(100) ≈ 4.605
                }
            }
            FadeType::Exponential => t * t,
            FadeType::SCurve => {
                // Smooth S-curve
                t * t * (3.0 - 2.0 * t)
            }
        };

        self.start_gain + (self.end_gain - self.start_gain) * curve_t
    }
}

/// Audio pan position.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioPan {
    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub position: f32,
}

impl AudioPan {
    /// Creates a new pan position.
    ///
    /// # Errors
    ///
    /// Returns error if position is not in range -1.0 to 1.0.
    pub fn new(position: f32) -> crate::error::TimelineResult<Self> {
        if !(-1.0..=1.0).contains(&position) {
            return Err(crate::error::TimelineError::Other(format!(
                "Invalid pan position: {position} (must be -1.0 to 1.0)"
            )));
        }
        Ok(Self { position })
    }

    /// Creates a center pan.
    #[must_use]
    pub fn center() -> Self {
        Self { position: 0.0 }
    }

    /// Creates a hard left pan.
    #[must_use]
    pub fn left() -> Self {
        Self { position: -1.0 }
    }

    /// Creates a hard right pan.
    #[must_use]
    pub fn right() -> Self {
        Self { position: 1.0 }
    }

    /// Calculates left/right gain values.
    #[must_use]
    pub fn to_stereo_gains(self) -> (f32, f32) {
        // Constant power panning
        let angle = (self.position + 1.0) * std::f32::consts::FRAC_PI_4;
        let left_gain = angle.cos();
        let right_gain = angle.sin();
        (left_gain, right_gain)
    }
}

impl Default for AudioPan {
    fn default() -> Self {
        Self::center()
    }
}

/// Audio gain in decibels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioGain {
    /// Gain in decibels.
    pub db: f32,
}

impl AudioGain {
    /// Creates a new gain value.
    #[must_use]
    pub const fn new(db: f32) -> Self {
        Self { db }
    }

    /// Creates unity gain (0 dB).
    #[must_use]
    pub const fn unity() -> Self {
        Self { db: 0.0 }
    }

    /// Creates silence (-infinity dB).
    #[must_use]
    pub const fn silence() -> Self {
        Self { db: -100.0 }
    }

    /// Converts to linear gain factor.
    #[must_use]
    pub fn to_linear(self) -> f32 {
        10.0_f32.powf(self.db / 20.0)
    }

    /// Creates from linear gain factor.
    #[must_use]
    pub fn from_linear(linear: f32) -> Self {
        Self {
            db: 20.0 * linear.log10(),
        }
    }
}

impl Default for AudioGain {
    fn default() -> Self {
        Self::unity()
    }
}

/// Audio mixer configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioMixer {
    /// Master gain.
    pub master_gain: AudioGain,
    /// Master pan.
    pub master_pan: AudioPan,
    /// Master mute.
    pub master_mute: bool,
}

impl AudioMixer {
    /// Creates a new audio mixer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            master_gain: AudioGain::unity(),
            master_pan: AudioPan::center(),
            master_mute: false,
        }
    }

    /// Sets master gain.
    pub fn set_master_gain(&mut self, gain: AudioGain) {
        self.master_gain = gain;
    }

    /// Sets master pan.
    pub fn set_master_pan(&mut self, pan: AudioPan) {
        self.master_pan = pan;
    }

    /// Mutes the master.
    pub fn mute(&mut self) {
        self.master_mute = true;
    }

    /// Unmutes the master.
    pub fn unmute(&mut self) {
        self.master_mute = false;
    }
}

impl Default for AudioMixer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_fade_in() {
        let fade = AudioFade::fade_in(Position::new(0), Position::new(100));
        assert!((fade.gain_at(Position::new(0)) - 0.0).abs() < f32::EPSILON);
        assert!((fade.gain_at(Position::new(100)) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_fade_out() {
        let fade = AudioFade::fade_out(Position::new(0), Position::new(100));
        assert!((fade.gain_at(Position::new(0)) - 1.0).abs() < f32::EPSILON);
        assert!((fade.gain_at(Position::new(100)) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_fade_linear() {
        let fade = AudioFade::new(FadeType::Linear, Position::new(0), Position::new(100));
        let mid_gain = fade.gain_at(Position::new(50));
        assert!((mid_gain - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_audio_pan_center() {
        let pan = AudioPan::center();
        let (left, right) = pan.to_stereo_gains();
        assert!((left - right).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_pan_left() {
        let pan = AudioPan::left();
        let (left, right) = pan.to_stereo_gains();
        assert!(left > right);
    }

    #[test]
    fn test_audio_pan_right() {
        let pan = AudioPan::right();
        let (left, right) = pan.to_stereo_gains();
        assert!(right > left);
    }

    #[test]
    fn test_audio_pan_invalid() {
        assert!(AudioPan::new(1.5).is_err());
        assert!(AudioPan::new(-1.5).is_err());
    }

    #[test]
    fn test_audio_gain_unity() {
        let gain = AudioGain::unity();
        assert!((gain.to_linear() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_audio_gain_conversions() {
        let gain = AudioGain::new(6.0);
        let linear = gain.to_linear();
        let back = AudioGain::from_linear(linear);
        assert!((back.db - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_mixer_creation() {
        let mixer = AudioMixer::new();
        assert!((mixer.master_gain.db - 0.0).abs() < f32::EPSILON);
        assert!(!mixer.master_mute);
    }

    #[test]
    fn test_audio_mixer_mute() {
        let mut mixer = AudioMixer::new();
        assert!(!mixer.master_mute);
        mixer.mute();
        assert!(mixer.master_mute);
        mixer.unmute();
        assert!(!mixer.master_mute);
    }
}
