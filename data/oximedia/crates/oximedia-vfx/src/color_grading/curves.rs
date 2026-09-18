//! Color curves for precise color grading.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// A point on a color curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Input value (0.0 - 1.0).
    pub input: f32,
    /// Output value (0.0 - 1.0).
    pub output: f32,
}

impl CurvePoint {
    /// Create a new curve point.
    #[must_use]
    pub const fn new(input: f32, output: f32) -> Self {
        Self { input, output }
    }
}

/// A color adjustment curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorCurve {
    points: Vec<CurvePoint>,
}

impl ColorCurve {
    /// Create a new linear curve.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)],
        }
    }

    /// Add a control point.
    pub fn add_point(&mut self, input: f32, output: f32) {
        let point = CurvePoint::new(input.clamp(0.0, 1.0), output.clamp(0.0, 1.0));

        match self.points.binary_search_by(|p| {
            p.input
                .partial_cmp(&input)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => self.points[idx] = point,
            Err(idx) => self.points.insert(idx, point),
        }
    }

    /// Evaluate curve at input value.
    #[must_use]
    pub fn evaluate(&self, input: f32) -> f32 {
        let input = input.clamp(0.0, 1.0);

        if self.points.is_empty() {
            return input;
        }

        if self.points.len() == 1 {
            return self.points[0].output;
        }

        // Find surrounding points
        let idx = match self.points.binary_search_by(|p| {
            p.input
                .partial_cmp(&input)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(idx) => return self.points[idx].output,
            Err(idx) => idx,
        };

        if idx == 0 {
            return self.points[0].output;
        }

        if idx >= self.points.len() {
            return self.points[self.points.len() - 1].output;
        }

        // Linear interpolation
        let p1 = &self.points[idx - 1];
        let p2 = &self.points[idx];

        let t = (input - p1.input) / (p2.input - p1.input);
        p1.output + (p2.output - p1.output) * t
    }

    /// Generate lookup table.
    #[must_use]
    pub fn generate_lut(&self, size: usize) -> Vec<f32> {
        let mut lut = Vec::with_capacity(size);
        for i in 0..size {
            let input = i as f32 / (size - 1) as f32;
            lut.push(self.evaluate(input));
        }
        lut
    }
}

impl Default for ColorCurve {
    fn default() -> Self {
        Self::new()
    }
}

/// RGB + Luma curves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelCurves {
    /// Master/Luma curve.
    pub luma: ColorCurve,
    /// Red channel curve.
    pub red: ColorCurve,
    /// Green channel curve.
    pub green: ColorCurve,
    /// Blue channel curve.
    pub blue: ColorCurve,
}

impl ChannelCurves {
    /// Create new linear curves.
    #[must_use]
    pub fn new() -> Self {
        Self {
            luma: ColorCurve::new(),
            red: ColorCurve::new(),
            green: ColorCurve::new(),
            blue: ColorCurve::new(),
        }
    }

    /// Apply curves to frame.
    pub fn apply(&self, input: &Frame, output: &mut Frame) -> VfxResult<()> {
        // Generate LUTs for fast lookup
        let luma_lut = self.luma.generate_lut(256);
        let red_lut = self.red.generate_lut(256);
        let green_lut = self.green.generate_lut(256);
        let blue_lut = self.blue.generate_lut(256);

        for y in 0..input.height {
            for x in 0..input.width {
                if let Some(pixel) = input.get_pixel(x, y) {
                    let r = f32::from(pixel[0]) / 255.0;
                    let g = f32::from(pixel[1]) / 255.0;
                    let b = f32::from(pixel[2]) / 255.0;

                    // Apply luma curve
                    let luma = r * 0.299 + g * 0.587 + b * 0.114;
                    let luma_adjusted = luma_lut[(luma * 255.0) as usize];
                    let luma_factor = if luma > 0.0 {
                        luma_adjusted / luma
                    } else {
                        1.0
                    };

                    // Apply per-channel curves
                    let r_adjusted = red_lut[(r * 255.0) as usize] * luma_factor;
                    let g_adjusted = green_lut[(g * 255.0) as usize] * luma_factor;
                    let b_adjusted = blue_lut[(b * 255.0) as usize] * luma_factor;

                    let result = [
                        (r_adjusted * 255.0).clamp(0.0, 255.0) as u8,
                        (g_adjusted * 255.0).clamp(0.0, 255.0) as u8,
                        (b_adjusted * 255.0).clamp(0.0, 255.0) as u8,
                        pixel[3],
                    ];

                    output.set_pixel(x, y, result);
                }
            }
        }

        Ok(())
    }
}

impl Default for ChannelCurves {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curve_point() {
        let point = CurvePoint::new(0.5, 0.7);
        assert_eq!(point.input, 0.5);
        assert_eq!(point.output, 0.7);
    }

    #[test]
    fn test_color_curve_linear() {
        let curve = ColorCurve::new();
        assert_eq!(curve.evaluate(0.5), 0.5);
        assert_eq!(curve.evaluate(0.0), 0.0);
        assert_eq!(curve.evaluate(1.0), 1.0);
    }

    #[test]
    fn test_color_curve_custom() {
        let mut curve = ColorCurve::new();
        curve.add_point(0.5, 0.7);
        let value = curve.evaluate(0.5);
        assert!((value - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_curve_lut() {
        let curve = ColorCurve::new();
        let lut = curve.generate_lut(256);
        assert_eq!(lut.len(), 256);
        assert_eq!(lut[0], 0.0);
        assert!((lut[255] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_channel_curves() -> VfxResult<()> {
        let curves = ChannelCurves::new();
        let input = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;
        curves.apply(&input, &mut output)?;
        Ok(())
    }
}
