//! Lens distortion correction and application.

use crate::{EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Lens distortion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LensModel {
    /// Brown-Conrady model.
    BrownConrady,
    /// Division model.
    Division,
    /// Polynomial model.
    Polynomial,
}

/// Lens distortion effect.
pub struct LensDistortion {
    model: LensModel,
    k1: f32,
    k2: f32,
    k3: f32,
    correct: bool,
}

impl LensDistortion {
    /// Create a new lens distortion effect.
    #[must_use]
    pub const fn new(model: LensModel) -> Self {
        Self {
            model,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            correct: false,
        }
    }

    /// Set distortion coefficients.
    #[must_use]
    pub const fn with_coefficients(mut self, k1: f32, k2: f32, k3: f32) -> Self {
        self.k1 = k1;
        self.k2 = k2;
        self.k3 = k3;
        self
    }

    /// Enable correction mode (vs apply mode).
    #[must_use]
    pub const fn with_correct(mut self, correct: bool) -> Self {
        self.correct = correct;
        self
    }

    fn distort_point(&self, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        let r2 = dx * dx + dy * dy;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let factor = match self.model {
            LensModel::BrownConrady => 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6,
            LensModel::Division => 1.0 / (1.0 + self.k1 * r2 + self.k2 * r4),
            LensModel::Polynomial => 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6,
        };

        let factor = if self.correct { 1.0 / factor } else { factor };

        (cx + dx * factor, cy + dy * factor)
    }
}

impl VideoEffect for LensDistortion {
    fn name(&self) -> &'static str {
        "Lens Distortion"
    }

    fn description(&self) -> &'static str {
        "Apply or correct lens distortion"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let cx = output.width as f32 / 2.0;
        let cy = output.height as f32 / 2.0;

        for y in 0..output.height {
            for x in 0..output.width {
                let (src_x, src_y) = self.distort_point(x as f32, y as f32, cx, cy);

                let pixel = if src_x >= 0.0
                    && src_x < input.width as f32
                    && src_y >= 0.0
                    && src_y < input.height as f32
                {
                    input
                        .get_pixel(src_x as u32, src_y as u32)
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
    fn test_lens_distortion() {
        let mut lens =
            LensDistortion::new(LensModel::BrownConrady).with_coefficients(0.1, 0.01, 0.001);

        let input = Frame::new(100, 100).expect("should succeed in test");
        let mut output = Frame::new(100, 100).expect("should succeed in test");
        let params = EffectParams::new();
        lens.apply(&input, &mut output, &params)
            .expect("should succeed in test");
    }
}
