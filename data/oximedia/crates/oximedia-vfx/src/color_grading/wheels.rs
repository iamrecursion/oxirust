//! Color wheels for shadows, midtones, and highlights.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// Type of color wheel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WheelType {
    /// Lift (shadows).
    Lift,
    /// Gamma (midtones).
    Gamma,
    /// Gain (highlights).
    Gain,
    /// Offset (all tones).
    Offset,
}

/// A color wheel for tonal adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorWheel {
    /// Hue shift (-180 to 180 degrees).
    pub hue: f32,
    /// Saturation adjustment (-1.0 to 1.0).
    pub saturation: f32,
    /// Luminance adjustment (-1.0 to 1.0).
    pub luminance: f32,
}

impl ColorWheel {
    /// Create a new neutral color wheel.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hue: 0.0,
            saturation: 0.0,
            luminance: 0.0,
        }
    }

    /// Set hue shift.
    #[must_use]
    pub const fn with_hue(mut self, hue: f32) -> Self {
        self.hue = hue;
        self
    }

    /// Set saturation.
    #[must_use]
    pub const fn with_saturation(mut self, saturation: f32) -> Self {
        self.saturation = saturation;
        self
    }

    /// Set luminance.
    #[must_use]
    pub const fn with_luminance(mut self, luminance: f32) -> Self {
        self.luminance = luminance;
        self
    }
}

impl Default for ColorWheel {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete set of color wheels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorWheels {
    /// Lift wheel (shadows).
    pub lift: ColorWheel,
    /// Gamma wheel (midtones).
    pub gamma: ColorWheel,
    /// Gain wheel (highlights).
    pub gain: ColorWheel,
    /// Offset wheel (all tones).
    pub offset: ColorWheel,
}

impl ColorWheels {
    /// Create new neutral wheels.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lift: ColorWheel::new(),
            gamma: ColorWheel::new(),
            gain: ColorWheel::new(),
            offset: ColorWheel::new(),
        }
    }

    /// Apply color wheels to frame.
    pub fn apply(&self, input: &Frame, output: &mut Frame) -> VfxResult<()> {
        for y in 0..input.height {
            for x in 0..input.width {
                if let Some(pixel) = input.get_pixel(x, y) {
                    let r = f32::from(pixel[0]) / 255.0;
                    let g = f32::from(pixel[1]) / 255.0;
                    let b = f32::from(pixel[2]) / 255.0;

                    // Convert to HSL
                    let (h, s, l) = rgb_to_hsl(r, g, b);

                    // Determine tonal weight (shadows, midtones, highlights)
                    let shadow_weight = (1.0 - l).powf(2.0);
                    let highlight_weight = l.powf(2.0);
                    let midtone_weight = 1.0 - shadow_weight - highlight_weight;

                    // Apply lift, gamma, gain
                    let mut adjusted_h = h;
                    let mut adjusted_s = s;
                    let mut adjusted_l = l;

                    // Lift (shadows)
                    adjusted_h += self.lift.hue * shadow_weight / 360.0;
                    adjusted_s += self.lift.saturation * shadow_weight;
                    adjusted_l += self.lift.luminance * shadow_weight;

                    // Gamma (midtones)
                    adjusted_h += self.gamma.hue * midtone_weight / 360.0;
                    adjusted_s += self.gamma.saturation * midtone_weight;
                    adjusted_l += self.gamma.luminance * midtone_weight;

                    // Gain (highlights)
                    adjusted_h += self.gain.hue * highlight_weight / 360.0;
                    adjusted_s += self.gain.saturation * highlight_weight;
                    adjusted_l += self.gain.luminance * highlight_weight;

                    // Offset (all)
                    adjusted_h += self.offset.hue / 360.0;
                    adjusted_s += self.offset.saturation;
                    adjusted_l += self.offset.luminance;

                    // Clamp and wrap hue
                    adjusted_h = adjusted_h.rem_euclid(1.0);
                    adjusted_s = adjusted_s.clamp(0.0, 1.0);
                    adjusted_l = adjusted_l.clamp(0.0, 1.0);

                    // Convert back to RGB
                    let (r_out, g_out, b_out) = hsl_to_rgb(adjusted_h, adjusted_s, adjusted_l);

                    let result = [
                        (r_out * 255.0).clamp(0.0, 255.0) as u8,
                        (g_out * 255.0).clamp(0.0, 255.0) as u8,
                        (b_out * 255.0).clamp(0.0, 255.0) as u8,
                        pixel[3],
                    ];

                    output.set_pixel(x, y, result);
                }
            }
        }

        Ok(())
    }
}

impl Default for ColorWheels {
    fn default() -> Self {
        Self::new()
    }
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let l = (max + min) / 2.0;

    if delta == 0.0 {
        return (0.0, 0.0, l);
    }

    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };

    let h = if max == r {
        ((g - b) / delta + if g < b { 6.0 } else { 0.0 }) / 6.0
    } else if max == g {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;

    let hue_to_rgb = |p: f32, q: f32, t: f32| {
        let mut t = t;
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 1.0 / 2.0 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };

    (
        hue_to_rgb(p, q, h + 1.0 / 3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0 / 3.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_wheel() {
        let wheel = ColorWheel::new()
            .with_hue(10.0)
            .with_saturation(0.1)
            .with_luminance(0.05);
        assert_eq!(wheel.hue, 10.0);
        assert_eq!(wheel.saturation, 0.1);
    }

    #[test]
    fn test_color_wheels() {
        let wheels = ColorWheels::new();
        assert_eq!(wheels.lift.hue, 0.0);
        assert_eq!(wheels.gamma.hue, 0.0);
    }

    #[test]
    fn test_rgb_hsl_conversion() {
        let (h, s, l) = rgb_to_hsl(1.0, 0.0, 0.0);
        let (r, g, b) = hsl_to_rgb(h, s, l);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g < 0.01);
        assert!(b < 0.01);
    }

    #[test]
    fn test_color_wheels_apply() -> VfxResult<()> {
        let wheels = ColorWheels::new();
        let input = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;
        wheels.apply(&input, &mut output)?;
        Ok(())
    }
}
