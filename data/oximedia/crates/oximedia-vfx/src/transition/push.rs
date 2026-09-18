//! Push transition effect.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// Push direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PushDirection {
    /// Push left.
    Left,
    /// Push right.
    Right,
    /// Push up.
    Up,
    /// Push down.
    Down,
}

/// Push transition.
///
/// Pushes one frame off screen while bringing the next frame in.
pub struct Push {
    direction: PushDirection,
}

impl Push {
    /// Create a new push transition.
    #[must_use]
    pub const fn new(direction: PushDirection) -> Self {
        Self { direction }
    }
}

impl TransitionEffect for Push {
    fn name(&self) -> &'static str {
        "Push"
    }

    fn description(&self) -> &'static str {
        "Push one frame out while bringing another in"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let width = output.width as i32;
        let height = output.height as i32;

        for y in 0..output.height {
            for x in 0..output.width {
                let (from_x, from_y, to_x, to_y) = match self.direction {
                    PushDirection::Left => {
                        let offset = (params.progress * width as f32) as i32;
                        (
                            x as i32 + offset,
                            y as i32,
                            x as i32 + offset - width,
                            y as i32,
                        )
                    }
                    PushDirection::Right => {
                        let offset = (params.progress * width as f32) as i32;
                        (
                            x as i32 - offset,
                            y as i32,
                            x as i32 - offset + width,
                            y as i32,
                        )
                    }
                    PushDirection::Up => {
                        let offset = (params.progress * height as f32) as i32;
                        (
                            x as i32,
                            y as i32 + offset,
                            x as i32,
                            y as i32 + offset - height,
                        )
                    }
                    PushDirection::Down => {
                        let offset = (params.progress * height as f32) as i32;
                        (
                            x as i32,
                            y as i32 - offset,
                            x as i32,
                            y as i32 - offset + height,
                        )
                    }
                };

                let pixel = if from_x >= 0 && from_x < width && from_y >= 0 && from_y < height {
                    from.get_pixel(from_x as u32, from_y as u32)
                        .unwrap_or([0, 0, 0, 0])
                } else if to_x >= 0 && to_x < width && to_y >= 0 && to_y < height {
                    to.get_pixel(to_x as u32, to_y as u32)
                        .unwrap_or([0, 0, 0, 0])
                } else {
                    [0, 0, 0, 0]
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
    fn test_push_directions() {
        let directions = [
            PushDirection::Left,
            PushDirection::Right,
            PushDirection::Up,
            PushDirection::Down,
        ];

        for direction in directions {
            let mut push = Push::new(direction);
            let from = Frame::new(100, 100).expect("should succeed in test");
            let to = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");

            let params = EffectParams::new().with_progress(0.5);
            push.apply(&from, &to, &mut output, &params)
                .expect("should succeed in test");
        }
    }
}
