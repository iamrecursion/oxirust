//! 3D transition effects.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// 3D transition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeDMode {
    /// Cube rotation.
    Cube,
    /// Page flip.
    Flip,
    /// Page curl.
    Curl,
    /// Cylinder rotation.
    Cylinder,
    /// Sphere morph.
    Sphere,
}

/// 3D transition effect.
///
/// Simulates 3D transformations between frames.
pub struct ThreeDTransition {
    mode: ThreeDMode,
    axis: Axis,
}

/// Rotation axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    /// X axis (horizontal).
    X,
    /// Y axis (vertical).
    Y,
    /// Z axis (depth).
    Z,
}

impl ThreeDTransition {
    /// Create a new 3D transition.
    #[must_use]
    pub const fn new(mode: ThreeDMode) -> Self {
        Self {
            mode,
            axis: Axis::Y,
        }
    }

    /// Set rotation axis.
    #[must_use]
    pub const fn with_axis(mut self, axis: Axis) -> Self {
        self.axis = axis;
        self
    }

    fn apply_cube(
        &self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        progress: f32,
    ) -> VfxResult<()> {
        let angle = progress * std::f32::consts::FRAC_PI_2;
        let width = output.width as f32;
        let height = output.height as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let nx = x as f32 / width - 0.5;
                let ny = y as f32 / height - 0.5;

                let (sx, visible_face) = match self.axis {
                    Axis::Y => {
                        let rotated_x = nx * angle.cos() - 0.5 * angle.sin();
                        let depth = nx * angle.sin() + 0.5 * angle.cos();
                        ((rotated_x + 0.5) * width, depth > 0.0)
                    }
                    Axis::X => {
                        let rotated_y = ny * angle.cos() - 0.5 * angle.sin();
                        let depth = ny * angle.sin() + 0.5 * angle.cos();
                        ((rotated_y + 0.5) * height, depth > 0.0)
                    }
                    Axis::Z => (x as f32, true),
                };

                let pixel = if visible_face {
                    if sx >= 0.0 && sx < from.width as f32 {
                        from.get_pixel(sx as u32, y).unwrap_or([0, 0, 0, 0])
                    } else {
                        [0, 0, 0, 0]
                    }
                } else if sx >= 0.0 && sx < to.width as f32 {
                    to.get_pixel(sx as u32, y).unwrap_or([0, 0, 0, 0])
                } else {
                    [0, 0, 0, 0]
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }

    fn apply_flip(
        &self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        progress: f32,
    ) -> VfxResult<()> {
        let _angle = progress * std::f32::consts::PI;
        let width = output.width as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let nx = x as f32 / width;
                let flip_progress = (nx + progress).rem_euclid(1.0);
                let local_angle = flip_progress * std::f32::consts::PI;

                let showing_front = local_angle.cos() > 0.0;
                let scale = local_angle.cos().abs();

                let sx = width * 0.5 + (x as f32 - width * 0.5) / scale.max(0.01);

                let pixel = if sx >= 0.0 && sx < width {
                    if showing_front {
                        from.get_pixel(sx as u32, y).unwrap_or([0, 0, 0, 0])
                    } else {
                        to.get_pixel(sx as u32, y).unwrap_or([0, 0, 0, 0])
                    }
                } else {
                    [0, 0, 0, 0]
                };

                // Apply shading
                let shade = scale * 0.5 + 0.5;
                let shaded = [
                    (f32::from(pixel[0]) * shade) as u8,
                    (f32::from(pixel[1]) * shade) as u8,
                    (f32::from(pixel[2]) * shade) as u8,
                    pixel[3],
                ];

                output.set_pixel(x, y, shaded);
            }
        }

        Ok(())
    }

    fn apply_curl(
        &self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        progress: f32,
    ) -> VfxResult<()> {
        let curl_amount = progress;
        let width = output.width as f32;
        let height = output.height as f32;

        for y in 0..output.height {
            for x in 0..output.width {
                let nx = x as f32 / width;
                let ny = y as f32 / height;

                let curl_start = 1.0 - curl_amount;
                let pixel = if nx > curl_start {
                    let curl_progress = (nx - curl_start) / curl_amount.max(0.001);
                    let curl_angle = curl_progress * std::f32::consts::PI;

                    let showing_back = curl_angle > std::f32::consts::FRAC_PI_2;
                    let cy = ny + curl_angle.sin() * 0.2;

                    if (0.0..1.0).contains(&cy) {
                        if showing_back {
                            to.get_pixel((nx * width) as u32, (cy * height) as u32)
                                .unwrap_or([0, 0, 0, 0])
                        } else {
                            from.get_pixel(x, y).unwrap_or([0, 0, 0, 0])
                        }
                    } else {
                        [0, 0, 0, 0]
                    }
                } else {
                    from.get_pixel(x, y).unwrap_or([0, 0, 0, 0])
                };

                output.set_pixel(x, y, pixel);
            }
        }

        Ok(())
    }
}

impl TransitionEffect for ThreeDTransition {
    fn name(&self) -> &'static str {
        "3D Transition"
    }

    fn description(&self) -> &'static str {
        "3D transformation transitions (cube, flip, curl)"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        match self.mode {
            ThreeDMode::Cube => self.apply_cube(from, to, output, params.progress),
            ThreeDMode::Flip => self.apply_flip(from, to, output, params.progress),
            ThreeDMode::Curl => self.apply_curl(from, to, output, params.progress),
            ThreeDMode::Cylinder => self.apply_cube(from, to, output, params.progress), // Similar to cube
            ThreeDMode::Sphere => self.apply_flip(from, to, output, params.progress), // Similar to flip
        }
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3d_modes() {
        let modes = [ThreeDMode::Cube, ThreeDMode::Flip, ThreeDMode::Curl];

        for mode in modes {
            let mut transition = ThreeDTransition::new(mode);
            let from = Frame::new(100, 100).expect("should succeed in test");
            let to = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");

            let params = EffectParams::new().with_progress(0.5);
            transition
                .apply(&from, &to, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_3d_axis() {
        let transition = ThreeDTransition::new(ThreeDMode::Cube).with_axis(Axis::X);
        assert_eq!(transition.axis, Axis::X);
    }
}
