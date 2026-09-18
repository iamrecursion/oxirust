//! Gradient generator.

use crate::{Color, EffectParams, Frame, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

/// Gradient type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientType {
    /// Linear gradient.
    Linear,
    /// Radial gradient.
    Radial,
    /// Angular gradient.
    Angular,
    /// Reflected gradient.
    Reflected,
    /// Diamond gradient.
    Diamond,
}

/// Gradient generator.
pub struct Gradient {
    gradient_type: GradientType,
    start_color: Color,
    end_color: Color,
    angle: f32,
    center_x: f32,
    center_y: f32,
}

impl Gradient {
    /// Create a new gradient generator.
    #[must_use]
    pub const fn new(gradient_type: GradientType) -> Self {
        Self {
            gradient_type,
            start_color: Color::black(),
            end_color: Color::white(),
            angle: 0.0,
            center_x: 0.5,
            center_y: 0.5,
        }
    }

    /// Set start color.
    #[must_use]
    pub const fn with_start_color(mut self, color: Color) -> Self {
        self.start_color = color;
        self
    }

    /// Set end color.
    #[must_use]
    pub const fn with_end_color(mut self, color: Color) -> Self {
        self.end_color = color;
        self
    }

    /// Set gradient angle in degrees (for linear gradient).
    #[must_use]
    pub fn with_angle(mut self, angle: f32) -> Self {
        self.angle = angle;
        self
    }

    /// Set center point (0.0 - 1.0).
    #[must_use]
    pub fn with_center(mut self, x: f32, y: f32) -> Self {
        self.center_x = x.clamp(0.0, 1.0);
        self.center_y = y.clamp(0.0, 1.0);
        self
    }
}

impl VideoEffect for Gradient {
    fn name(&self) -> &'static str {
        "Gradient"
    }

    fn description(&self) -> &'static str {
        "Generate linear, radial, and angular gradients"
    }

    fn apply(
        &mut self,
        _input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        let width = output.width as f32;
        let height = output.height as f32;
        let cx = self.center_x * width;
        let cy = self.center_y * height;

        for y in 0..output.height {
            for x in 0..output.width {
                let fx = x as f32;
                let fy = y as f32;

                let t = match self.gradient_type {
                    GradientType::Linear => {
                        let angle_rad = self.angle.to_radians();
                        let dx = fx - cx;
                        let dy = fy - cy;
                        let proj = dx * angle_rad.cos() + dy * angle_rad.sin();
                        let max_dist = (width.abs() + height.abs()) / 2.0;
                        ((proj + max_dist / 2.0) / max_dist).clamp(0.0, 1.0)
                    }
                    GradientType::Radial => {
                        let dx = fx - cx;
                        let dy = fy - cy;
                        let dist = (dx * dx + dy * dy).sqrt();
                        let max_dist = (width * width + height * height).sqrt() / 2.0;
                        (dist / max_dist).clamp(0.0, 1.0)
                    }
                    GradientType::Angular => {
                        let dx = fx - cx;
                        let dy = fy - cy;
                        let angle = dy.atan2(dx);
                        ((angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI))
                            .clamp(0.0, 1.0)
                    }
                    GradientType::Reflected => {
                        let dx = (fx - cx).abs();
                        let dy = (fy - cy).abs();
                        let dist = (dx * dx + dy * dy).sqrt();
                        let max_dist = (width * width + height * height).sqrt() / 2.0;
                        let t = (dist / max_dist).clamp(0.0, 1.0);
                        if t < 0.5 {
                            t * 2.0
                        } else {
                            2.0 - t * 2.0
                        }
                    }
                    GradientType::Diamond => {
                        let dx = (fx - cx).abs();
                        let dy = (fy - cy).abs();
                        let max_dist = width.max(height) / 2.0;
                        ((dx + dy) / max_dist / 2.0).clamp(0.0, 1.0)
                    }
                };

                let color = self.start_color.lerp(self.end_color, t);
                output.set_pixel(x, y, color.to_rgba());
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
    fn test_gradient_types() {
        let types = [
            GradientType::Linear,
            GradientType::Radial,
            GradientType::Angular,
        ];

        for gradient_type in types {
            let mut gradient = Gradient::new(gradient_type);
            let input = Frame::new(100, 100).expect("should succeed in test");
            let mut output = Frame::new(100, 100).expect("should succeed in test");
            let params = EffectParams::new();
            gradient
                .apply(&input, &mut output, &params)
                .expect("should succeed in test");
        }
    }

    #[test]
    fn test_gradient_colors() {
        let gradient = Gradient::new(GradientType::Linear)
            .with_start_color(Color::rgb(255, 0, 0))
            .with_end_color(Color::rgb(0, 0, 255));

        assert_eq!(gradient.start_color, Color::rgb(255, 0, 0));
        assert_eq!(gradient.end_color, Color::rgb(0, 0, 255));
    }
}
