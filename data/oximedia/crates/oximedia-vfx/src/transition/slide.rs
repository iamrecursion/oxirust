//! Slide transition effect.

use crate::{EasingFunction, EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// Slide direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlideDirection {
    /// Slide in from left.
    FromLeft,
    /// Slide in from right.
    FromRight,
    /// Slide in from top.
    FromTop,
    /// Slide in from bottom.
    FromBottom,
}

/// Slide transition.
///
/// Slides the new frame in over the old frame with easing.
pub struct Slide {
    direction: SlideDirection,
    easing: EasingFunction,
}

impl Slide {
    /// Create a new slide transition.
    #[must_use]
    pub const fn new(direction: SlideDirection) -> Self {
        Self {
            direction,
            easing: EasingFunction::EaseInOut,
        }
    }

    /// Set easing function.
    #[must_use]
    pub const fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }
}

impl TransitionEffect for Slide {
    fn name(&self) -> &'static str {
        "Slide"
    }

    fn description(&self) -> &'static str {
        "Slide new frame over old frame with easing"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let eased_progress = self.easing.apply(params.progress);
        let width = output.width as i32;
        let height = output.height as i32;

        for y in 0..output.height {
            for x in 0..output.width {
                let (to_x, to_y) = match self.direction {
                    SlideDirection::FromLeft => {
                        let offset = ((1.0 - eased_progress) * width as f32) as i32;
                        (x as i32 - width + offset, y as i32)
                    }
                    SlideDirection::FromRight => {
                        let offset = ((1.0 - eased_progress) * width as f32) as i32;
                        (x as i32 + width - offset, y as i32)
                    }
                    SlideDirection::FromTop => {
                        let offset = ((1.0 - eased_progress) * height as f32) as i32;
                        (x as i32, y as i32 - height + offset)
                    }
                    SlideDirection::FromBottom => {
                        let offset = ((1.0 - eased_progress) * height as f32) as i32;
                        (x as i32, y as i32 + height - offset)
                    }
                };

                let pixel = if to_x >= 0 && to_x < width && to_y >= 0 && to_y < height {
                    to.get_pixel(to_x as u32, to_y as u32)
                        .unwrap_or([0, 0, 0, 0])
                } else {
                    from.get_pixel(x, y).unwrap_or([0, 0, 0, 0])
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slide_directions() {
        let directions = [
            SlideDirection::FromLeft,
            SlideDirection::FromRight,
            SlideDirection::FromTop,
            SlideDirection::FromBottom,
        ];

        for direction in directions {
            let mut slide = Slide::new(direction);
            let from = Frame::new(100, 100).expect("should succeed in test");
            let to = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");

            let params = EffectParams::new().with_progress(0.5);
            slide
                .apply(&from, &to, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_slide_easing() {
        let slide = Slide::new(SlideDirection::FromLeft).with_easing(EasingFunction::EaseIn);
        assert_eq!(slide.easing, EasingFunction::EaseIn);
    }
}
