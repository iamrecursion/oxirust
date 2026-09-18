//! Transition effects between clips.
//!
//! Transitions provide smooth blending between adjacent clips on the timeline.

use oximedia_core::Rational;

use crate::clip::ClipId;
use crate::error::{EditError, EditResult};

/// A transition between two clips.
#[derive(Clone, Debug)]
pub struct Transition {
    /// Unique transition identifier.
    pub id: u64,
    /// Transition type.
    pub transition_type: TransitionType,
    /// Track index.
    pub track: usize,
    /// Timeline position where transition starts.
    pub start: i64,
    /// Transition duration.
    pub duration: i64,
    /// Timebase.
    pub timebase: Rational,
    /// Clip before transition.
    pub clip_a: ClipId,
    /// Clip after transition.
    pub clip_b: ClipId,
    /// Transition-specific parameters.
    pub parameters: TransitionParameters,
}

impl Transition {
    /// Create a new transition.
    #[must_use]
    pub fn new(
        id: u64,
        transition_type: TransitionType,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Self {
        Self {
            id,
            transition_type,
            track,
            start,
            duration,
            timebase: Rational::new(1, 1000),
            clip_a,
            clip_b,
            parameters: TransitionParameters::default(),
        }
    }

    /// Get the end position of the transition.
    #[must_use]
    pub fn end(&self) -> i64 {
        self.start + self.duration
    }

    /// Check if this transition is active at a given time.
    #[must_use]
    pub fn is_active_at(&self, time: i64) -> bool {
        time >= self.start && time < self.end()
    }

    /// Calculate transition progress (0.0 to 1.0) at a given time.
    #[must_use]
    pub fn progress_at(&self, time: i64) -> f64 {
        if time <= self.start {
            return 0.0;
        }
        if time >= self.end() {
            return 1.0;
        }
        if self.duration == 0 {
            return 1.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let progress = (time - self.start) as f64 / self.duration as f64;

        // Apply easing based on parameters
        self.parameters.apply_easing(progress)
    }

    /// Validate transition parameters.
    pub fn validate(&self) -> EditResult<()> {
        if self.duration <= 0 {
            return Err(EditError::InvalidTransition(
                "Duration must be positive".to_string(),
            ));
        }

        if self.clip_a == self.clip_b {
            return Err(EditError::InvalidTransition(
                "Cannot transition between same clip".to_string(),
            ));
        }

        Ok(())
    }
}

/// Type of transition effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransitionType {
    /// Video cross-dissolve.
    Dissolve,
    /// Audio cross-fade.
    CrossFade,
    /// Wipe transition (left to right).
    WipeLeft,
    /// Wipe transition (right to left).
    WipeRight,
    /// Wipe transition (top to bottom).
    WipeDown,
    /// Wipe transition (bottom to top).
    WipeUp,
    /// Slide transition.
    Slide,
    /// Push transition.
    Push,
    /// Zoom in transition.
    ZoomIn,
    /// Zoom out transition.
    ZoomOut,
    /// Fade through black.
    FadeThrough,
    /// Fade through white.
    FadeThroughWhite,
    /// Dip to color.
    DipToColor,
    /// Custom transition.
    Custom(String),
}

impl TransitionType {
    /// Check if this is a video transition.
    #[must_use]
    pub fn is_video(&self) -> bool {
        !matches!(self, Self::CrossFade)
    }

    /// Check if this is an audio transition.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::CrossFade)
    }
}

/// Parameters for transition effects.
#[derive(Clone, Debug)]
pub struct TransitionParameters {
    /// Easing function.
    pub easing: EasingFunction,
    /// Reverse the transition direction.
    pub reverse: bool,
    /// Transition color (for fade-through effects).
    pub color: Option<[f32; 4]>,
    /// Softness/feathering amount (0.0-1.0).
    pub softness: f32,
    /// Angle for directional transitions (in degrees).
    pub angle: f32,
}

impl Default for TransitionParameters {
    fn default() -> Self {
        Self {
            easing: EasingFunction::Linear,
            reverse: false,
            color: None,
            softness: 0.0,
            angle: 0.0,
        }
    }
}

impl TransitionParameters {
    /// Apply easing function to transition progress.
    #[must_use]
    pub fn apply_easing(&self, t: f64) -> f64 {
        let t = if self.reverse { 1.0 - t } else { t };
        self.easing.apply(t)
    }
}

/// Easing function for transition timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EasingFunction {
    /// Linear (no easing).
    Linear,
    /// Ease in (slow start).
    EaseIn,
    /// Ease out (slow end).
    EaseOut,
    /// Ease in and out (slow start and end).
    EaseInOut,
    /// Exponential ease in.
    ExpoIn,
    /// Exponential ease out.
    ExpoOut,
    /// Cubic ease in.
    CubicIn,
    /// Cubic ease out.
    CubicOut,
    /// Sine ease in.
    SineIn,
    /// Sine ease out.
    SineOut,
}

impl EasingFunction {
    /// Apply the easing function to a value (0.0 to 1.0).
    #[must_use]
    #[allow(clippy::excessive_precision)]
    pub fn apply(&self, t: f64) -> f64 {
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::ExpoIn => {
                if t == 0.0 {
                    0.0
                } else {
                    2.0_f64.powf(10.0 * (t - 1.0))
                }
            }
            Self::ExpoOut => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - 2.0_f64.powf(-10.0 * t)
                }
            }
            Self::CubicIn => t * t * t,
            Self::CubicOut => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            Self::SineIn => 1.0 - (t * std::f64::consts::FRAC_PI_2).cos(),
            Self::SineOut => (t * std::f64::consts::FRAC_PI_2).sin(),
        }
    }
}

/// Transition builder for creating transitions with validation.
#[derive(Debug)]
pub struct TransitionBuilder {
    transition_type: TransitionType,
    track: usize,
    start: i64,
    duration: i64,
    clip_a: ClipId,
    clip_b: ClipId,
    parameters: TransitionParameters,
}

impl TransitionBuilder {
    /// Create a new transition builder.
    #[must_use]
    pub fn new(
        transition_type: TransitionType,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Self {
        Self {
            transition_type,
            track,
            start,
            duration,
            clip_a,
            clip_b,
            parameters: TransitionParameters::default(),
        }
    }

    /// Set easing function.
    #[must_use]
    pub fn easing(mut self, easing: EasingFunction) -> Self {
        self.parameters.easing = easing;
        self
    }

    /// Set reverse flag.
    #[must_use]
    pub fn reverse(mut self, reverse: bool) -> Self {
        self.parameters.reverse = reverse;
        self
    }

    /// Set transition color.
    #[must_use]
    pub fn color(mut self, color: [f32; 4]) -> Self {
        self.parameters.color = Some(color);
        self
    }

    /// Set softness amount.
    #[must_use]
    pub fn softness(mut self, softness: f32) -> Self {
        self.parameters.softness = softness.clamp(0.0, 1.0);
        self
    }

    /// Set angle.
    #[must_use]
    pub fn angle(mut self, angle: f32) -> Self {
        self.parameters.angle = angle;
        self
    }

    /// Build the transition.
    pub fn build(self, id: u64) -> EditResult<Transition> {
        let transition = Transition {
            id,
            transition_type: self.transition_type,
            track: self.track,
            start: self.start,
            duration: self.duration,
            timebase: Rational::new(1, 1000),
            clip_a: self.clip_a,
            clip_b: self.clip_b,
            parameters: self.parameters,
        };

        transition.validate()?;
        Ok(transition)
    }
}

/// Preset transitions for common use cases.
pub struct TransitionPresets;

impl TransitionPresets {
    /// Create a standard cross-dissolve transition.
    #[must_use]
    pub fn dissolve(
        id: u64,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Transition {
        Transition::new(
            id,
            TransitionType::Dissolve,
            track,
            start,
            duration,
            clip_a,
            clip_b,
        )
    }

    /// Create an audio cross-fade transition.
    #[must_use]
    pub fn crossfade(
        id: u64,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Transition {
        Transition::new(
            id,
            TransitionType::CrossFade,
            track,
            start,
            duration,
            clip_a,
            clip_b,
        )
    }

    /// Create a fade through black transition.
    #[must_use]
    pub fn fade_through_black(
        id: u64,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Transition {
        let mut transition = Transition::new(
            id,
            TransitionType::FadeThrough,
            track,
            start,
            duration,
            clip_a,
            clip_b,
        );
        transition.parameters.color = Some([0.0, 0.0, 0.0, 1.0]);
        transition
    }

    /// Create a smooth dissolve with ease in/out.
    #[must_use]
    pub fn smooth_dissolve(
        id: u64,
        track: usize,
        start: i64,
        duration: i64,
        clip_a: ClipId,
        clip_b: ClipId,
    ) -> Transition {
        let mut transition = Transition::new(
            id,
            TransitionType::Dissolve,
            track,
            start,
            duration,
            clip_a,
            clip_b,
        );
        transition.parameters.easing = EasingFunction::EaseInOut;
        transition
    }
}

/// Transition manager for a timeline.
#[derive(Debug, Default)]
pub struct TransitionManager {
    /// All transitions in the timeline.
    transitions: Vec<Transition>,
    /// Next transition ID.
    next_id: u64,
}

impl TransitionManager {
    /// Create a new transition manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a transition.
    pub fn add(&mut self, mut transition: Transition) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        transition.id = id;
        self.transitions.push(transition);
        id
    }

    /// Remove a transition by ID.
    pub fn remove(&mut self, id: u64) -> Option<Transition> {
        if let Some(pos) = self.transitions.iter().position(|t| t.id == id) {
            Some(self.transitions.remove(pos))
        } else {
            None
        }
    }

    /// Get a transition by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Transition> {
        self.transitions.iter().find(|t| t.id == id)
    }

    /// Get mutable transition by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Transition> {
        self.transitions.iter_mut().find(|t| t.id == id)
    }

    /// Get all transitions on a track.
    #[must_use]
    pub fn get_track_transitions(&self, track: usize) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.track == track)
            .collect()
    }

    /// Get active transitions at a specific time on a track.
    #[must_use]
    pub fn get_active_at(&self, track: usize, time: i64) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.track == track && t.is_active_at(time))
            .collect()
    }

    /// Clear all transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    /// Get total number of transitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if there are no transitions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}
