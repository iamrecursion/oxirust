//! Transition effects between clips.

use serde::{Deserialize, Serialize};

use crate::types::Duration;

/// Type of transition effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionType {
    /// Cross-dissolve/fade.
    Dissolve,
    /// Dip to black.
    DipToBlack,
    /// Dip to white.
    DipToWhite,
    /// Dip to custom color.
    DipToColor,
    /// Wipe transition.
    Wipe,
    /// Push transition.
    Push,
    /// Slide transition.
    Slide,
    /// Fade to audio.
    AudioCrossfade,
}

impl TransitionType {
    /// Checks if this is a video transition.
    #[must_use]
    pub const fn is_video(self) -> bool {
        !matches!(self, Self::AudioCrossfade)
    }

    /// Checks if this is an audio transition.
    #[must_use]
    pub const fn is_audio(self) -> bool {
        matches!(self, Self::AudioCrossfade)
    }
}

/// Alignment of transition relative to cut point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionAlignment {
    /// Transition centered on cut point.
    Center,
    /// Transition starts at cut point.
    StartAtCut,
    /// Transition ends at cut point.
    EndAtCut,
}

/// Direction for wipe transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WipeDirection {
    /// Left to right.
    LeftToRight,
    /// Right to left.
    RightToLeft,
    /// Top to bottom.
    TopToBottom,
    /// Bottom to top.
    BottomToTop,
}

/// A transition between two clips.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    /// Type of transition.
    pub transition_type: TransitionType,
    /// Duration of the transition.
    pub duration: Duration,
    /// Alignment relative to cut point.
    pub alignment: TransitionAlignment,
    /// Custom color for dip-to-color (RGBA 0.0-1.0).
    pub color: Option<[f32; 4]>,
    /// Direction for wipe/push/slide transitions.
    pub direction: Option<WipeDirection>,
    /// Softness/feather for wipe (0.0-1.0).
    pub softness: f32,
    /// Whether transition is enabled.
    pub enabled: bool,
}

impl Transition {
    /// Creates a new dissolve transition.
    #[must_use]
    pub fn dissolve(duration: Duration) -> Self {
        Self {
            transition_type: TransitionType::Dissolve,
            duration,
            alignment: TransitionAlignment::Center,
            color: None,
            direction: None,
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new dip-to-black transition.
    #[must_use]
    pub fn dip_to_black(duration: Duration) -> Self {
        Self {
            transition_type: TransitionType::DipToBlack,
            duration,
            alignment: TransitionAlignment::Center,
            color: Some([0.0, 0.0, 0.0, 1.0]),
            direction: None,
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new dip-to-white transition.
    #[must_use]
    pub fn dip_to_white(duration: Duration) -> Self {
        Self {
            transition_type: TransitionType::DipToWhite,
            duration,
            alignment: TransitionAlignment::Center,
            color: Some([1.0, 1.0, 1.0, 1.0]),
            direction: None,
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new dip-to-color transition.
    #[must_use]
    pub fn dip_to_color(duration: Duration, color: [f32; 4]) -> Self {
        Self {
            transition_type: TransitionType::DipToColor,
            duration,
            alignment: TransitionAlignment::Center,
            color: Some(color),
            direction: None,
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new wipe transition.
    #[must_use]
    pub fn wipe(duration: Duration, direction: WipeDirection, softness: f32) -> Self {
        Self {
            transition_type: TransitionType::Wipe,
            duration,
            alignment: TransitionAlignment::Center,
            color: None,
            direction: Some(direction),
            softness,
            enabled: true,
        }
    }

    /// Creates a new push transition.
    #[must_use]
    pub fn push(duration: Duration, direction: WipeDirection) -> Self {
        Self {
            transition_type: TransitionType::Push,
            duration,
            alignment: TransitionAlignment::Center,
            color: None,
            direction: Some(direction),
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new slide transition.
    #[must_use]
    pub fn slide(duration: Duration, direction: WipeDirection) -> Self {
        Self {
            transition_type: TransitionType::Slide,
            duration,
            alignment: TransitionAlignment::Center,
            color: None,
            direction: Some(direction),
            softness: 0.0,
            enabled: true,
        }
    }

    /// Creates a new audio crossfade.
    #[must_use]
    pub fn audio_crossfade(duration: Duration) -> Self {
        Self {
            transition_type: TransitionType::AudioCrossfade,
            duration,
            alignment: TransitionAlignment::Center,
            color: None,
            direction: None,
            softness: 0.0,
            enabled: true,
        }
    }

    /// Sets the transition alignment.
    #[must_use]
    pub fn with_alignment(mut self, alignment: TransitionAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Sets the transition duration.
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Enables or disables the transition.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_type_is_video() {
        assert!(TransitionType::Dissolve.is_video());
        assert!(TransitionType::Wipe.is_video());
        assert!(!TransitionType::AudioCrossfade.is_video());
    }

    #[test]
    fn test_transition_type_is_audio() {
        assert!(TransitionType::AudioCrossfade.is_audio());
        assert!(!TransitionType::Dissolve.is_audio());
    }

    #[test]
    fn test_dissolve_transition() {
        let trans = Transition::dissolve(Duration::new(24));
        assert_eq!(trans.transition_type, TransitionType::Dissolve);
        assert_eq!(trans.duration.value(), 24);
        assert_eq!(trans.alignment, TransitionAlignment::Center);
        assert!(trans.enabled);
    }

    #[test]
    fn test_dip_to_black() {
        let trans = Transition::dip_to_black(Duration::new(12));
        assert_eq!(trans.transition_type, TransitionType::DipToBlack);
        assert_eq!(trans.color, Some([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_dip_to_white() {
        let trans = Transition::dip_to_white(Duration::new(12));
        assert_eq!(trans.transition_type, TransitionType::DipToWhite);
        assert_eq!(trans.color, Some([1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_dip_to_color() {
        let color = [1.0, 0.0, 0.0, 1.0];
        let trans = Transition::dip_to_color(Duration::new(12), color);
        assert_eq!(trans.transition_type, TransitionType::DipToColor);
        assert_eq!(trans.color, Some(color));
    }

    #[test]
    fn test_wipe_transition() {
        let trans = Transition::wipe(Duration::new(24), WipeDirection::LeftToRight, 0.5);
        assert_eq!(trans.transition_type, TransitionType::Wipe);
        assert_eq!(trans.direction, Some(WipeDirection::LeftToRight));
        assert!((trans.softness - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_push_transition() {
        let trans = Transition::push(Duration::new(24), WipeDirection::TopToBottom);
        assert_eq!(trans.transition_type, TransitionType::Push);
        assert_eq!(trans.direction, Some(WipeDirection::TopToBottom));
    }

    #[test]
    fn test_slide_transition() {
        let trans = Transition::slide(Duration::new(24), WipeDirection::RightToLeft);
        assert_eq!(trans.transition_type, TransitionType::Slide);
        assert_eq!(trans.direction, Some(WipeDirection::RightToLeft));
    }

    #[test]
    fn test_audio_crossfade() {
        let trans = Transition::audio_crossfade(Duration::new(48));
        assert_eq!(trans.transition_type, TransitionType::AudioCrossfade);
        assert_eq!(trans.duration.value(), 48);
    }

    #[test]
    fn test_with_alignment() {
        let trans =
            Transition::dissolve(Duration::new(24)).with_alignment(TransitionAlignment::StartAtCut);
        assert_eq!(trans.alignment, TransitionAlignment::StartAtCut);
    }

    #[test]
    fn test_with_duration() {
        let trans = Transition::dissolve(Duration::new(24)).with_duration(Duration::new(48));
        assert_eq!(trans.duration.value(), 48);
    }

    #[test]
    fn test_set_enabled() {
        let mut trans = Transition::dissolve(Duration::new(24));
        assert!(trans.enabled);
        trans.set_enabled(false);
        assert!(!trans.enabled);
    }
}
