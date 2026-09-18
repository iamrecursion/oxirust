//! Transition effects between graphics states

use crate::animation::Easing;
use crate::primitives::Point;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Transition type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    /// Cut (instant)
    Cut,
    /// Dissolve/fade
    Dissolve,
    /// Slide from edge
    Slide {
        /// Direction
        direction: SlideDirection,
    },
    /// Wipe from edge
    Wipe {
        /// Direction
        direction: WipeDirection,
    },
    /// Scale/zoom
    Scale {
        /// Start scale
        from: f32,
        /// End scale
        to: f32,
    },
    /// Rotate
    Rotate {
        /// Start angle (radians)
        from: f32,
        /// End angle (radians)
        to: f32,
    },
    /// Push (slide out old, slide in new)
    Push {
        /// Direction
        direction: SlideDirection,
    },
    /// 3D cube rotation
    Cube {
        /// Axis
        axis: CubeAxis,
    },
    /// 3D flip
    Flip {
        /// Axis
        axis: FlipAxis,
    },
}

/// Slide direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlideDirection {
    /// From left
    Left,
    /// From right
    Right,
    /// From top
    Top,
    /// From bottom
    Bottom,
}

/// Wipe direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WipeDirection {
    /// Left to right
    LeftToRight,
    /// Right to left
    RightToLeft,
    /// Top to bottom
    TopToBottom,
    /// Bottom to top
    BottomToTop,
}

/// Cube rotation axis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CubeAxis {
    /// Horizontal
    Horizontal,
    /// Vertical
    Vertical,
}

/// Flip axis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlipAxis {
    /// Horizontal flip
    Horizontal,
    /// Vertical flip
    Vertical,
}

/// Transition state
#[derive(Debug, Clone)]
pub struct Transition {
    /// Transition type
    pub transition_type: TransitionType,
    /// Duration
    pub duration: Duration,
    /// Easing function
    pub easing: Easing,
    /// Current time
    pub current_time: Duration,
    /// Is playing
    pub playing: bool,
}

impl Transition {
    /// Create a new transition
    #[must_use]
    pub fn new(transition_type: TransitionType, duration: Duration, easing: Easing) -> Self {
        Self {
            transition_type,
            duration,
            easing,
            current_time: Duration::ZERO,
            playing: false,
        }
    }

    /// Create a cut transition (instant)
    #[must_use]
    pub fn cut() -> Self {
        Self::new(TransitionType::Cut, Duration::ZERO, Easing::Linear)
    }

    /// Create a dissolve transition
    #[must_use]
    pub fn dissolve(duration: Duration) -> Self {
        Self::new(TransitionType::Dissolve, duration, Easing::Linear)
    }

    /// Create a slide transition
    #[must_use]
    pub fn slide(direction: SlideDirection, duration: Duration) -> Self {
        Self::new(
            TransitionType::Slide { direction },
            duration,
            Easing::EaseInOut,
        )
    }

    /// Create a wipe transition
    #[must_use]
    pub fn wipe(direction: WipeDirection, duration: Duration) -> Self {
        Self::new(TransitionType::Wipe { direction }, duration, Easing::Linear)
    }

    /// Start transition
    pub fn start(&mut self) {
        self.playing = true;
        self.current_time = Duration::ZERO;
    }

    /// Update transition
    pub fn update(&mut self, delta: Duration) -> bool {
        if !self.playing {
            return false;
        }

        self.current_time += delta;

        if self.current_time >= self.duration {
            self.current_time = self.duration;
            self.playing = false;
            return true;
        }

        false
    }

    /// Get progress (0.0 to 1.0)
    #[must_use]
    pub fn progress(&self) -> f32 {
        if self.duration.is_zero() {
            return 1.0;
        }
        let t = self.current_time.as_secs_f32() / self.duration.as_secs_f32();
        self.easing.apply(t.clamp(0.0, 1.0))
    }

    /// Is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.playing && self.current_time >= self.duration
    }

    /// Reset
    pub fn reset(&mut self) {
        self.current_time = Duration::ZERO;
        self.playing = false;
    }
}

/// Transition parameters for rendering
#[derive(Debug, Clone)]
pub struct TransitionParams {
    /// Progress (0.0 to 1.0)
    pub progress: f32,
    /// Opacity for old state
    pub old_opacity: f32,
    /// Opacity for new state
    pub new_opacity: f32,
    /// Transform for old state
    pub old_transform: TransitionTransform,
    /// Transform for new state
    pub new_transform: TransitionTransform,
}

/// Transform for transition
#[derive(Debug, Clone)]
pub struct TransitionTransform {
    /// Position offset
    pub position: Point,
    /// Scale
    pub scale: Point,
    /// Rotation (radians)
    pub rotation: f32,
}

impl Default for TransitionTransform {
    fn default() -> Self {
        Self {
            position: Point::new(0.0, 0.0),
            scale: Point::new(1.0, 1.0),
            rotation: 0.0,
        }
    }
}

impl Transition {
    /// Get rendering parameters
    #[must_use]
    pub fn get_params(&self, viewport_size: (f32, f32)) -> TransitionParams {
        let progress = self.progress();

        let (old_opacity, new_opacity, old_transform, new_transform) = match &self.transition_type {
            TransitionType::Cut => (
                if progress >= 1.0 { 0.0 } else { 1.0 },
                if progress >= 1.0 { 1.0 } else { 0.0 },
                TransitionTransform::default(),
                TransitionTransform::default(),
            ),

            TransitionType::Dissolve => (
                1.0 - progress,
                progress,
                TransitionTransform::default(),
                TransitionTransform::default(),
            ),

            TransitionType::Slide { direction } => {
                let (dx, dy) = match direction {
                    SlideDirection::Left => (-viewport_size.0, 0.0),
                    SlideDirection::Right => (viewport_size.0, 0.0),
                    SlideDirection::Top => (0.0, -viewport_size.1),
                    SlideDirection::Bottom => (0.0, viewport_size.1),
                };

                (
                    1.0,
                    1.0,
                    TransitionTransform::default(),
                    TransitionTransform {
                        position: Point::new(dx * (1.0 - progress), dy * (1.0 - progress)),
                        ..Default::default()
                    },
                )
            }

            TransitionType::Wipe { .. } => {
                // Wipe is handled differently in rendering
                (
                    1.0,
                    1.0,
                    TransitionTransform::default(),
                    TransitionTransform::default(),
                )
            }

            TransitionType::Scale { from, to } => {
                let scale = from + (to - from) * progress;
                (
                    1.0 - progress,
                    progress,
                    TransitionTransform::default(),
                    TransitionTransform {
                        scale: Point::new(scale, scale),
                        ..Default::default()
                    },
                )
            }

            TransitionType::Rotate { from, to } => {
                let rotation = from + (to - from) * progress;
                (
                    1.0 - progress,
                    progress,
                    TransitionTransform::default(),
                    TransitionTransform {
                        rotation,
                        ..Default::default()
                    },
                )
            }

            TransitionType::Push { direction } => {
                let (dx, dy) = match direction {
                    SlideDirection::Left => (-viewport_size.0, 0.0),
                    SlideDirection::Right => (viewport_size.0, 0.0),
                    SlideDirection::Top => (0.0, -viewport_size.1),
                    SlideDirection::Bottom => (0.0, viewport_size.1),
                };

                (
                    1.0,
                    1.0,
                    TransitionTransform {
                        position: Point::new(dx * progress, dy * progress),
                        ..Default::default()
                    },
                    TransitionTransform {
                        position: Point::new(dx * (progress - 1.0), dy * (progress - 1.0)),
                        ..Default::default()
                    },
                )
            }

            TransitionType::Cube { .. } | TransitionType::Flip { .. } => {
                // 3D transitions need special handling
                (
                    1.0 - progress,
                    progress,
                    TransitionTransform::default(),
                    TransitionTransform::default(),
                )
            }
        };

        TransitionParams {
            progress,
            old_opacity,
            new_opacity,
            old_transform,
            new_transform,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_cut() {
        let transition = Transition::cut();
        assert!(matches!(transition.transition_type, TransitionType::Cut));
        assert_eq!(transition.duration, Duration::ZERO);
    }

    #[test]
    fn test_transition_dissolve() {
        let transition = Transition::dissolve(Duration::from_secs(1));
        assert!(matches!(
            transition.transition_type,
            TransitionType::Dissolve
        ));
        assert_eq!(transition.duration, Duration::from_secs(1));
    }

    #[test]
    fn test_transition_slide() {
        let transition = Transition::slide(SlideDirection::Left, Duration::from_secs(1));
        assert!(matches!(
            transition.transition_type,
            TransitionType::Slide { .. }
        ));
    }

    #[test]
    fn test_transition_update() {
        let mut transition = Transition::dissolve(Duration::from_secs(1));
        transition.start();

        assert!(transition.playing);
        assert_eq!(transition.progress(), 0.0);

        let complete = transition.update(Duration::from_millis(500));
        assert!(!complete);
        assert!((transition.progress() - 0.5).abs() < 0.01);

        let complete = transition.update(Duration::from_millis(500));
        assert!(complete);
        assert_eq!(transition.progress(), 1.0);
    }

    #[test]
    fn test_transition_progress() {
        let mut transition = Transition::dissolve(Duration::from_secs(1));
        transition.current_time = Duration::from_millis(250);
        assert!((transition.progress() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_transition_reset() {
        let mut transition = Transition::dissolve(Duration::from_secs(1));
        transition.start();
        transition.update(Duration::from_millis(500));

        transition.reset();
        assert_eq!(transition.current_time, Duration::ZERO);
        assert!(!transition.playing);
    }

    #[test]
    fn test_transition_params_dissolve() {
        let mut transition = Transition::dissolve(Duration::from_secs(1));
        transition.current_time = Duration::from_millis(500);

        let params = transition.get_params((1920.0, 1080.0));
        assert!((params.progress - 0.5).abs() < 0.01);
        assert!((params.old_opacity - 0.5).abs() < 0.01);
        assert!((params.new_opacity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_transition_params_slide() {
        let mut transition = Transition::slide(SlideDirection::Left, Duration::from_secs(1));
        transition.current_time = Duration::from_millis(500);

        let params = transition.get_params((1920.0, 1080.0));
        assert!(params.new_transform.position.x < 0.0);
    }

    #[test]
    fn test_slide_direction() {
        assert_eq!(SlideDirection::Left, SlideDirection::Left);
        assert_ne!(SlideDirection::Left, SlideDirection::Right);
    }

    #[test]
    fn test_wipe_direction() {
        assert_eq!(WipeDirection::LeftToRight, WipeDirection::LeftToRight);
        assert_ne!(WipeDirection::LeftToRight, WipeDirection::RightToLeft);
    }
}
