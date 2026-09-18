//! Visual effects (blur, glow, shadow, etc.)

use crate::color::Color;
use crate::primitives::Point;
use serde::{Deserialize, Serialize};

/// Drop shadow effect
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DropShadow {
    /// Offset X
    pub offset_x: f32,
    /// Offset Y
    pub offset_y: f32,
    /// Blur radius
    pub blur_radius: f32,
    /// Shadow color
    pub color: Color,
}

impl DropShadow {
    /// Create a new drop shadow
    #[must_use]
    pub fn new(offset_x: f32, offset_y: f32, blur_radius: f32, color: Color) -> Self {
        Self {
            offset_x,
            offset_y,
            blur_radius,
            color,
        }
    }
}

/// Glow effect
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Glow {
    /// Radius
    pub radius: f32,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Glow color
    pub color: Color,
}

impl Glow {
    /// Create a new glow
    #[must_use]
    pub fn new(radius: f32, intensity: f32, color: Color) -> Self {
        Self {
            radius,
            intensity: intensity.clamp(0.0, 1.0),
            color,
        }
    }
}

/// Blur effect
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Blur {
    /// Blur radius
    pub radius: f32,
    /// Blur type
    pub blur_type: BlurType,
}

impl Blur {
    /// Create a new blur
    #[must_use]
    pub fn new(radius: f32, blur_type: BlurType) -> Self {
        Self { radius, blur_type }
    }

    /// Create Gaussian blur
    #[must_use]
    pub fn gaussian(radius: f32) -> Self {
        Self::new(radius, BlurType::Gaussian)
    }

    /// Create motion blur
    #[must_use]
    pub fn motion(radius: f32, angle: f32) -> Self {
        Self::new(radius, BlurType::Motion { angle })
    }
}

/// Blur type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlurType {
    /// Gaussian blur
    Gaussian,
    /// Box blur
    Box,
    /// Motion blur
    Motion {
        /// Angle in radians
        angle: f32,
    },
}

/// Color adjustment
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ColorAdjustment {
    /// Brightness (-1.0 to 1.0)
    pub brightness: f32,
    /// Contrast (-1.0 to 1.0)
    pub contrast: f32,
    /// Saturation (-1.0 to 1.0)
    pub saturation: f32,
    /// Hue shift (degrees)
    pub hue_shift: f32,
}

impl ColorAdjustment {
    /// Create identity adjustment (no change)
    #[must_use]
    pub fn identity() -> Self {
        Self {
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            hue_shift: 0.0,
        }
    }

    /// Apply to color
    #[must_use]
    pub fn apply(&self, color: Color) -> Color {
        let mut rgb = color.to_float();

        // Brightness
        if self.brightness != 0.0 {
            rgb[0] = (rgb[0] + self.brightness).clamp(0.0, 1.0);
            rgb[1] = (rgb[1] + self.brightness).clamp(0.0, 1.0);
            rgb[2] = (rgb[2] + self.brightness).clamp(0.0, 1.0);
        }

        // Contrast
        if self.contrast != 0.0 {
            let factor = (1.0 + self.contrast).max(0.0);
            rgb[0] = ((rgb[0] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
            rgb[1] = ((rgb[1] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
            rgb[2] = ((rgb[2] - 0.5) * factor + 0.5).clamp(0.0, 1.0);
        }

        // Saturation
        if self.saturation != 0.0 {
            let gray = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            let factor = 1.0 + self.saturation;
            rgb[0] = (gray + (rgb[0] - gray) * factor).clamp(0.0, 1.0);
            rgb[1] = (gray + (rgb[1] - gray) * factor).clamp(0.0, 1.0);
            rgb[2] = (gray + (rgb[2] - gray) * factor).clamp(0.0, 1.0);
        }

        Color::from_float(rgb)
    }
}

impl Default for ColorAdjustment {
    fn default() -> Self {
        Self::identity()
    }
}

/// Transformation matrix for advanced effects
#[derive(Debug, Clone, Copy)]
pub struct Matrix3x3 {
    /// Matrix values (row-major)
    pub m: [[f32; 3]; 3],
}

impl Matrix3x3 {
    /// Create identity matrix
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create translation matrix
    #[must_use]
    pub fn translation(x: f32, y: f32) -> Self {
        Self {
            m: [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        }
    }

    /// Create scale matrix
    #[must_use]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            m: [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Create rotation matrix
    #[must_use]
    pub fn rotation(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Self {
            m: [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiply with another matrix
    #[must_use]
    pub fn multiply(&self, other: &Matrix3x3) -> Self {
        let mut result = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                result.m[i][j] = 0.0;
                for k in 0..3 {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        result
    }

    /// Transform point
    #[must_use]
    pub fn transform_point(&self, p: Point) -> Point {
        let x = self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2];
        let y = self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2];
        Point::new(x, y)
    }
}

/// Effect stack (multiple effects combined)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EffectStack {
    /// Drop shadow
    pub drop_shadow: Option<DropShadow>,
    /// Glow
    pub glow: Option<Glow>,
    /// Blur
    pub blur: Option<Blur>,
    /// Color adjustment
    pub color_adjustment: Option<ColorAdjustment>,
}

impl EffectStack {
    /// Create a new empty effect stack
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add drop shadow
    #[must_use]
    pub fn with_drop_shadow(mut self, shadow: DropShadow) -> Self {
        self.drop_shadow = Some(shadow);
        self
    }

    /// Add glow
    #[must_use]
    pub fn with_glow(mut self, glow: Glow) -> Self {
        self.glow = Some(glow);
        self
    }

    /// Add blur
    #[must_use]
    pub fn with_blur(mut self, blur: Blur) -> Self {
        self.blur = Some(blur);
        self
    }

    /// Add color adjustment
    #[must_use]
    pub fn with_color_adjustment(mut self, adjustment: ColorAdjustment) -> Self {
        self.color_adjustment = Some(adjustment);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_shadow() {
        let shadow = DropShadow::new(5.0, 5.0, 10.0, Color::new(0, 0, 0, 128));
        assert_eq!(shadow.offset_x, 5.0);
        assert_eq!(shadow.offset_y, 5.0);
        assert_eq!(shadow.blur_radius, 10.0);
    }

    #[test]
    fn test_glow() {
        let glow = Glow::new(10.0, 0.8, Color::WHITE);
        assert_eq!(glow.radius, 10.0);
        assert_eq!(glow.intensity, 0.8);
    }

    #[test]
    fn test_blur() {
        let blur = Blur::gaussian(5.0);
        assert_eq!(blur.radius, 5.0);
        assert!(matches!(blur.blur_type, BlurType::Gaussian));

        let motion = Blur::motion(10.0, 0.0);
        assert!(matches!(motion.blur_type, BlurType::Motion { .. }));
    }

    #[test]
    fn test_color_adjustment() {
        let adj = ColorAdjustment::identity();
        assert_eq!(adj.brightness, 0.0);
        assert_eq!(adj.contrast, 0.0);
        assert_eq!(adj.saturation, 0.0);

        let color = Color::rgb(128, 128, 128);
        let adjusted = adj.apply(color);
        assert_eq!(adjusted, color);
    }

    #[test]
    fn test_color_adjustment_brightness() {
        let mut adj = ColorAdjustment::identity();
        adj.brightness = 0.2;

        let color = Color::rgb(100, 100, 100);
        let adjusted = adj.apply(color);
        assert!(adjusted.r > color.r);
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix3x3::identity();
        let p = Point::new(10.0, 20.0);
        let transformed = m.transform_point(p);
        assert_eq!(transformed, p);
    }

    #[test]
    fn test_matrix_translation() {
        let m = Matrix3x3::translation(5.0, 10.0);
        let p = Point::new(10.0, 20.0);
        let transformed = m.transform_point(p);
        assert_eq!(transformed, Point::new(15.0, 30.0));
    }

    #[test]
    fn test_matrix_scale() {
        let m = Matrix3x3::scale(2.0, 3.0);
        let p = Point::new(10.0, 20.0);
        let transformed = m.transform_point(p);
        assert_eq!(transformed, Point::new(20.0, 60.0));
    }

    #[test]
    fn test_matrix_rotation() {
        let m = Matrix3x3::rotation(std::f32::consts::PI / 2.0);
        let p = Point::new(1.0, 0.0);
        let transformed = m.transform_point(p);
        assert!((transformed.x - 0.0).abs() < 0.0001);
        assert!((transformed.y - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_matrix_multiply() {
        let m1 = Matrix3x3::translation(5.0, 10.0);
        let m2 = Matrix3x3::scale(2.0, 2.0);
        let m = m1.multiply(&m2);

        let p = Point::new(10.0, 20.0);
        let transformed = m.transform_point(p);
        assert_eq!(transformed, Point::new(25.0, 50.0));
    }

    #[test]
    fn test_effect_stack() {
        let stack = EffectStack::new()
            .with_drop_shadow(DropShadow::new(2.0, 2.0, 4.0, Color::BLACK))
            .with_glow(Glow::new(5.0, 0.5, Color::WHITE))
            .with_blur(Blur::gaussian(3.0));

        assert!(stack.drop_shadow.is_some());
        assert!(stack.glow.is_some());
        assert!(stack.blur.is_some());
    }
}
