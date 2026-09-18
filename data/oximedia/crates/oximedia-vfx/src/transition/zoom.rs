//! Zoom transition effect.

use crate::{EffectParams, Frame, TransitionEffect, VfxResult};
use serde::{Deserialize, Serialize};

/// Zoom mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZoomMode {
    /// Zoom in (scale up).
    In,
    /// Zoom out (scale down).
    Out,
    /// Cross-zoom (zoom out from, zoom in to).
    Cross,
}

/// Zoom transition.
///
/// Scales frames during transition with various zoom modes.
pub struct Zoom {
    mode: ZoomMode,
    blur_amount: f32,
}

impl Zoom {
    /// Create a new zoom transition.
    #[must_use]
    pub const fn new(mode: ZoomMode) -> Self {
        Self {
            mode,
            blur_amount: 0.0,
        }
    }

    /// Set motion blur amount (0.0 - 1.0).
    #[must_use]
    pub fn with_blur(mut self, blur: f32) -> Self {
        self.blur_amount = blur.clamp(0.0, 1.0);
        self
    }

    fn sample_bilinear(frame: &Frame, x: f32, y: f32) -> [u8; 4] {
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(frame.width - 1);
        let y1 = (y0 + 1).min(frame.height - 1);

        let fx = x.fract();
        let fy = y.fract();

        let p00 = frame.get_pixel(x0, y0).unwrap_or([0, 0, 0, 0]);
        let p10 = frame.get_pixel(x1, y0).unwrap_or([0, 0, 0, 0]);
        let p01 = frame.get_pixel(x0, y1).unwrap_or([0, 0, 0, 0]);
        let p11 = frame.get_pixel(x1, y1).unwrap_or([0, 0, 0, 0]);

        let mut result = [0u8; 4];
        for i in 0..4 {
            let v0 = f32::from(p00[i]) * (1.0 - fx) + f32::from(p10[i]) * fx;
            let v1 = f32::from(p01[i]) * (1.0 - fx) + f32::from(p11[i]) * fx;
            result[i] = (v0 * (1.0 - fy) + v1 * fy) as u8;
        }

        result
    }

    fn blend_pixel(from: [u8; 4], to: [u8; 4], t: f32) -> [u8; 4] {
        let t = t.clamp(0.0, 1.0);
        let inv_t = 1.0 - t;

        [
            (f32::from(from[0]) * inv_t + f32::from(to[0]) * t) as u8,
            (f32::from(from[1]) * inv_t + f32::from(to[1]) * t) as u8,
            (f32::from(from[2]) * inv_t + f32::from(to[2]) * t) as u8,
            (f32::from(from[3]) * inv_t + f32::from(to[3]) * t) as u8,
        ]
    }
}

impl TransitionEffect for Zoom {
    fn name(&self) -> &'static str {
        "Zoom"
    }

    fn description(&self) -> &'static str {
        "Zoom transition with various modes"
    }

    fn apply(
        &mut self,
        from: &Frame,
        to: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        let cx = output.width as f32 * 0.5;
        let cy = output.height as f32 * 0.5;

        for y in 0..output.height {
            for x in 0..output.width {
                let fx = x as f32;
                let fy = y as f32;

                let pixel = match self.mode {
                    ZoomMode::In => {
                        // Zoom in on "to" frame
                        let scale = 1.0 + params.progress * 0.5;
                        let sx = cx + (fx - cx) / scale;
                        let sy = cy + (fy - cy) / scale;

                        if sx >= 0.0 && sx < to.width as f32 && sy >= 0.0 && sy < to.height as f32 {
                            Self::sample_bilinear(to, sx, sy)
                        } else {
                            from.get_pixel(x, y).unwrap_or([0, 0, 0, 0])
                        }
                    }
                    ZoomMode::Out => {
                        // Zoom out from "from" frame
                        let scale = 1.0 + params.progress * 0.5;
                        let sx = cx + (fx - cx) * scale;
                        let sy = cy + (fy - cy) * scale;

                        if sx >= 0.0
                            && sx < from.width as f32
                            && sy >= 0.0
                            && sy < from.height as f32
                        {
                            Self::sample_bilinear(from, sx, sy)
                        } else {
                            to.get_pixel(x, y).unwrap_or([0, 0, 0, 0])
                        }
                    }
                    ZoomMode::Cross => {
                        // Cross zoom
                        let from_scale = 1.0 + params.progress;
                        let to_scale = 1.0 + (1.0 - params.progress);

                        let from_sx = cx + (fx - cx) * from_scale;
                        let from_sy = cy + (fy - cy) * from_scale;
                        let to_sx = cx + (fx - cx) / to_scale;
                        let to_sy = cy + (fy - cy) / to_scale;

                        let from_pixel = if from_sx >= 0.0
                            && from_sx < from.width as f32
                            && from_sy >= 0.0
                            && from_sy < from.height as f32
                        {
                            Self::sample_bilinear(from, from_sx, from_sy)
                        } else {
                            [0, 0, 0, 0]
                        };

                        let to_pixel = if to_sx >= 0.0
                            && to_sx < to.width as f32
                            && to_sy >= 0.0
                            && to_sy < to.height as f32
                        {
                            Self::sample_bilinear(to, to_sx, to_sy)
                        } else {
                            [0, 0, 0, 0]
                        };

                        Self::blend_pixel(from_pixel, to_pixel, params.progress)
                    }
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
    fn test_zoom_modes() {
        let modes = [ZoomMode::In, ZoomMode::Out, ZoomMode::Cross];

        for mode in modes {
            let mut zoom = Zoom::new(mode);
            let from = Frame::new(100, 100).expect("should succeed in test");
            let to = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");

            let params = EffectParams::new().with_progress(0.5);
            zoom.apply(&from, &to, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_zoom_blur() {
        let zoom = Zoom::new(ZoomMode::In).with_blur(0.5);
        assert_eq!(zoom.blur_amount, 0.5);
    }
}
