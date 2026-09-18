//! Attention prediction for video.

use crate::common::Point;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Attention map with focus points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMap {
    /// Attention values (0.0-1.0).
    pub data: Vec<f32>,
    /// Width.
    pub width: usize,
    /// Height.
    pub height: usize,
    /// Focus points (high attention regions).
    pub focus_points: Vec<Point>,
}

/// Attention predictor.
pub struct AttentionPredictor;

impl AttentionPredictor {
    /// Create a new attention predictor.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Predict attention map.
    ///
    /// # Errors
    ///
    /// Returns error if prediction fails.
    pub fn predict(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<AttentionMap> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        // Compute attention based on edges, contrast, and center bias
        let mut attention = vec![0.0f32; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = (y * width + x) * 3;
                let mut score = 0.0;

                // Edge strength
                for c in 0..3 {
                    let center = rgb_data[idx + c] as f32;
                    let left = rgb_data[idx - 3 + c] as f32;
                    let right = rgb_data[idx + 3 + c] as f32;
                    score += ((center - left).abs() + (center - right).abs()) / 2.0;
                }

                // Center bias
                let dx = (x as f32 - width as f32 / 2.0).abs() / (width as f32 / 2.0);
                let dy = (y as f32 - height as f32 / 2.0).abs() / (height as f32 / 2.0);
                let center_bias = 1.0 - ((dx * dx + dy * dy) / 2.0).sqrt();

                attention[y * width + x] = (score / 255.0 / 3.0) * center_bias;
            }
        }

        // Normalize
        let max_attn = attention.iter().copied().fold(f32::MIN, f32::max);
        if max_attn > 0.0 {
            for a in &mut attention {
                *a /= max_attn;
            }
        }

        // Find focus points (local maxima)
        let focus_points = self.find_focus_points(&attention, width, height);

        Ok(AttentionMap {
            data: attention,
            width,
            height,
            focus_points,
        })
    }

    fn find_focus_points(&self, attention: &[f32], width: usize, height: usize) -> Vec<Point> {
        let mut points = Vec::new();
        let window = 20;

        for y in (window..height - window).step_by(window) {
            for x in (window..width - window).step_by(window) {
                let idx = y * width + x;
                let value = attention[idx];

                if value > 0.7 {
                    // Check if local maximum
                    let mut is_max = true;
                    for dy in -(window as i32)..=window as i32 {
                        for dx in -(window as i32)..=window as i32 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                let nidx = ny as usize * width + nx as usize;
                                if attention[nidx] > value {
                                    is_max = false;
                                    break;
                                }
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }

                    if is_max {
                        points.push(Point::new(x as f32, y as f32));
                    }
                }
            }
        }

        points
    }
}

impl Default for AttentionPredictor {
    fn default() -> Self {
        Self::new()
    }
}
