//! Visual balance analysis.

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Balance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceMetrics {
    /// Horizontal balance (-1.0 to 1.0, 0 is centered).
    pub horizontal_balance: f32,
    /// Vertical balance (-1.0 to 1.0, 0 is centered).
    pub vertical_balance: f32,
    /// Color balance (0.0-1.0).
    pub color_balance: f32,
    /// Weight distribution score (0.0-1.0).
    pub weight_distribution: f32,
}

/// Balance analyzer.
pub struct BalanceAnalyzer;

impl BalanceAnalyzer {
    /// Create a new balance analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyze balance of an image.
    ///
    /// # Errors
    ///
    /// Returns error if analysis fails.
    pub fn analyze(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<BalanceMetrics> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(
                "RGB data size mismatch".to_string(),
            ));
        }

        let horizontal_balance = self.compute_horizontal_balance(rgb_data, width, height);
        let vertical_balance = self.compute_vertical_balance(rgb_data, width, height);
        let color_balance = self.compute_color_balance(rgb_data);
        let weight_distribution = self.compute_weight_distribution(rgb_data, width, height);

        Ok(BalanceMetrics {
            horizontal_balance,
            vertical_balance,
            color_balance,
            weight_distribution,
        })
    }

    fn compute_horizontal_balance(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let mut left_weight = 0.0;
        let mut right_weight = 0.0;

        for y in 0..height {
            for x in 0..width / 2 {
                let idx = (y * width + x) * 3;
                left_weight +=
                    (rgb_data[idx] as f32 + rgb_data[idx + 1] as f32 + rgb_data[idx + 2] as f32)
                        / 3.0;
            }
            for x in width / 2..width {
                let idx = (y * width + x) * 3;
                right_weight +=
                    (rgb_data[idx] as f32 + rgb_data[idx + 1] as f32 + rgb_data[idx + 2] as f32)
                        / 3.0;
            }
        }

        let total = left_weight + right_weight;
        if total > 0.0 {
            ((right_weight - left_weight) / total).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn compute_vertical_balance(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let mut top_weight = 0.0;
        let mut bottom_weight = 0.0;

        for y in 0..height / 2 {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                top_weight +=
                    (rgb_data[idx] as f32 + rgb_data[idx + 1] as f32 + rgb_data[idx + 2] as f32)
                        / 3.0;
            }
        }

        for y in height / 2..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                bottom_weight +=
                    (rgb_data[idx] as f32 + rgb_data[idx + 1] as f32 + rgb_data[idx + 2] as f32)
                        / 3.0;
            }
        }

        let total = top_weight + bottom_weight;
        if total > 0.0 {
            ((bottom_weight - top_weight) / total).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn compute_color_balance(&self, rgb_data: &[u8]) -> f32 {
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;

        for i in (0..rgb_data.len()).step_by(3) {
            r_sum += u64::from(rgb_data[i]);
            g_sum += u64::from(rgb_data[i + 1]);
            b_sum += u64::from(rgb_data[i + 2]);
        }

        let pixel_count = rgb_data.len() / 3;
        let r_avg = r_sum as f32 / pixel_count as f32;
        let g_avg = g_sum as f32 / pixel_count as f32;
        let b_avg = b_sum as f32 / pixel_count as f32;

        let max_avg = r_avg.max(g_avg).max(b_avg);
        let min_avg = r_avg.min(g_avg).min(b_avg);

        if max_avg > 0.0 {
            (1.0 - (max_avg - min_avg) / max_avg).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    fn compute_weight_distribution(&self, rgb_data: &[u8], width: usize, height: usize) -> f32 {
        let quadrants = 4;
        let quad_width = width / 2;
        let quad_height = height / 2;
        let mut weights = vec![0.0; quadrants];

        for q in 0..quadrants {
            let qx = (q % 2) * quad_width;
            let qy = (q / 2) * quad_height;

            for y in qy..(qy + quad_height).min(height) {
                for x in qx..(qx + quad_width).min(width) {
                    let idx = (y * width + x) * 3;
                    weights[q] += (rgb_data[idx] as f32
                        + rgb_data[idx + 1] as f32
                        + rgb_data[idx + 2] as f32)
                        / 3.0;
                }
            }
        }

        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            let mean = total / quadrants as f32;
            let variance: f32 =
                weights.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / quadrants as f32;
            (1.0 - (variance.sqrt() / mean).min(1.0)).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

impl Default for BalanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
