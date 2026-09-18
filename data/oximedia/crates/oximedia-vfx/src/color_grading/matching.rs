//! Color matching between shots.

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// Parameters for color matching.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorMatchParams {
    /// Match strength (0.0 - 1.0).
    pub strength: f32,
    /// Preserve skin tones.
    pub preserve_skin_tones: bool,
    /// Skin tone hue range (in degrees).
    pub skin_hue_range: f32,
}

impl Default for ColorMatchParams {
    fn default() -> Self {
        Self {
            strength: 1.0,
            preserve_skin_tones: true,
            skin_hue_range: 30.0,
        }
    }
}

/// Color statistics for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorStats {
    /// Mean red.
    pub mean_r: f32,
    /// Mean green.
    pub mean_g: f32,
    /// Mean blue.
    pub mean_b: f32,
    /// Standard deviation red.
    pub std_r: f32,
    /// Standard deviation green.
    pub std_g: f32,
    /// Standard deviation blue.
    pub std_b: f32,
}

/// Tool for matching colors between frames.
#[derive(Debug, Clone)]
pub struct ColorMatcher {
    params: ColorMatchParams,
}

impl ColorMatcher {
    /// Create a new color matcher.
    #[must_use]
    pub const fn new(params: ColorMatchParams) -> Self {
        Self { params }
    }

    /// Calculate color statistics for a frame.
    pub fn calculate_stats(&self, frame: &Frame) -> VfxResult<ColorStats> {
        let mut sum_r = 0.0;
        let mut sum_g = 0.0;
        let mut sum_b = 0.0;
        let mut count = 0;

        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    sum_r += f32::from(pixel[0]);
                    sum_g += f32::from(pixel[1]);
                    sum_b += f32::from(pixel[2]);
                    count += 1;
                }
            }
        }

        let mean_r = sum_r / count as f32;
        let mean_g = sum_g / count as f32;
        let mean_b = sum_b / count as f32;

        let mut var_r = 0.0;
        let mut var_g = 0.0;
        let mut var_b = 0.0;

        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    let dr = f32::from(pixel[0]) - mean_r;
                    let dg = f32::from(pixel[1]) - mean_g;
                    let db = f32::from(pixel[2]) - mean_b;
                    var_r += dr * dr;
                    var_g += dg * dg;
                    var_b += db * db;
                }
            }
        }

        Ok(ColorStats {
            mean_r,
            mean_g,
            mean_b,
            std_r: (var_r / count as f32).sqrt(),
            std_g: (var_g / count as f32).sqrt(),
            std_b: (var_b / count as f32).sqrt(),
        })
    }

    /// Match colors from source to target statistics.
    pub fn match_colors(
        &self,
        input: &Frame,
        output: &mut Frame,
        source_stats: &ColorStats,
        target_stats: &ColorStats,
    ) -> VfxResult<()> {
        for y in 0..input.height {
            for x in 0..input.width {
                if let Some(pixel) = input.get_pixel(x, y) {
                    let r = f32::from(pixel[0]);
                    let g = f32::from(pixel[1]);
                    let b = f32::from(pixel[2]);

                    // Normalize to source stats
                    let r_norm = (r - source_stats.mean_r) / source_stats.std_r;
                    let g_norm = (g - source_stats.mean_g) / source_stats.std_g;
                    let b_norm = (b - source_stats.mean_b) / source_stats.std_b;

                    // Apply target stats
                    let r_matched = r_norm * target_stats.std_r + target_stats.mean_r;
                    let g_matched = g_norm * target_stats.std_g + target_stats.mean_g;
                    let b_matched = b_norm * target_stats.std_b + target_stats.mean_b;

                    // Check for skin tones
                    let is_skin = if self.params.preserve_skin_tones {
                        self.is_skin_tone(pixel)
                    } else {
                        false
                    };

                    // Apply strength and skin tone preservation
                    let strength = if is_skin { 0.0 } else { self.params.strength };

                    let r_final = r + (r_matched - r) * strength;
                    let g_final = g + (g_matched - g) * strength;
                    let b_final = b + (b_matched - b) * strength;

                    let result = [
                        r_final.clamp(0.0, 255.0) as u8,
                        g_final.clamp(0.0, 255.0) as u8,
                        b_final.clamp(0.0, 255.0) as u8,
                        pixel[3],
                    ];

                    output.set_pixel(x, y, result);
                }
            }
        }

        Ok(())
    }

    fn is_skin_tone(&self, pixel: [u8; 4]) -> bool {
        let r = f32::from(pixel[0]) / 255.0;
        let g = f32::from(pixel[1]) / 255.0;
        let b = f32::from(pixel[2]) / 255.0;

        // Simple skin tone detection in HSV space
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        if delta == 0.0 {
            return false;
        }

        let h = if max == r {
            ((g - b) / delta).rem_euclid(6.0) * 60.0
        } else if max == g {
            ((b - r) / delta + 2.0) * 60.0
        } else {
            ((r - g) / delta + 4.0) * 60.0
        };

        let s = delta / max;

        // Skin tone heuristic: hue 0-50 degrees, moderate saturation
        h < self.params.skin_hue_range && s > 0.2 && s < 0.6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_match_params() {
        let params = ColorMatchParams::default();
        assert_eq!(params.strength, 1.0);
        assert!(params.preserve_skin_tones);
    }

    #[test]
    fn test_color_matcher() -> VfxResult<()> {
        let matcher = ColorMatcher::new(ColorMatchParams::default());
        let frame = Frame::new(10, 10)?;
        let stats = matcher.calculate_stats(&frame)?;
        assert_eq!(stats.mean_r, 0.0);
        Ok(())
    }

    #[test]
    fn test_color_matching() -> VfxResult<()> {
        let matcher = ColorMatcher::new(ColorMatchParams::default());
        let input = Frame::new(10, 10)?;
        let mut output = Frame::new(10, 10)?;
        let source_stats = matcher.calculate_stats(&input)?;
        let target_stats = source_stats;
        matcher.match_colors(&input, &mut output, &source_stats, &target_stats)?;
        Ok(())
    }
}
