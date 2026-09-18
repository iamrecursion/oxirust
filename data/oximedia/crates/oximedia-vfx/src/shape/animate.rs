//! Shape animation.

use crate::ParameterTrack;
use serde::{Deserialize, Serialize};

use super::draw::Shape;

/// Shape animation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeAnimationType {
    /// Position animation.
    Position,
    /// Scale animation.
    Scale,
    /// Rotation animation.
    Rotation,
    /// Opacity animation.
    Opacity,
}

/// Shape animation.
pub struct ShapeAnimation {
    shape: Shape,
    animation_type: ShapeAnimationType,
    track: ParameterTrack,
}

impl ShapeAnimation {
    /// Create a new shape animation.
    #[must_use]
    pub fn new(shape: Shape, animation_type: ShapeAnimationType) -> Self {
        Self {
            shape,
            animation_type,
            track: ParameterTrack::new(),
        }
    }

    /// Add keyframe.
    pub fn add_keyframe(&mut self, time: f64, value: f32, easing: crate::EasingFunction) {
        self.track.add_keyframe(time, value, easing);
    }

    /// Get animated shape at given time.
    #[must_use]
    pub fn get_shape(&self, time: f64) -> Shape {
        let mut shape = self.shape.clone();

        if let Some(value) = self.track.evaluate(time) {
            match self.animation_type {
                ShapeAnimationType::Position => {
                    shape.bounds.x += value;
                }
                ShapeAnimationType::Scale => {
                    let scale = value.max(0.01);
                    shape.bounds.width *= scale;
                    shape.bounds.height *= scale;
                }
                ShapeAnimationType::Rotation => {
                    // Rotation would require more complex transformation
                }
                ShapeAnimationType::Opacity => {
                    let alpha = (value * 255.0).clamp(0.0, 255.0) as u8;
                    shape.fill_color.a = alpha;
                }
            }
        }

        shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EasingFunction, Rect};

    #[test]
    fn test_shape_animation() {
        let shape = Shape::rectangle(Rect::new(0.0, 0.0, 100.0, 100.0));
        let mut animation = ShapeAnimation::new(shape, ShapeAnimationType::Scale);

        animation.add_keyframe(0.0, 1.0, EasingFunction::Linear);
        animation.add_keyframe(1.0, 2.0, EasingFunction::Linear);

        let animated = animation.get_shape(0.5);
        assert!(animated.bounds.width > 100.0);
    }
}
