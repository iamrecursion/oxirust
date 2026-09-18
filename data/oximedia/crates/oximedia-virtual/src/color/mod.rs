//! Color pipeline management
//!
//! Manages end-to-end color pipeline for virtual production including
//! camera-to-LED matching and LUT application.

pub mod lut;
pub mod match_color;
pub mod pipeline;

use serde::{Deserialize, Serialize};

/// Color transform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorTransform {
    /// 3x3 matrix
    pub matrix: [[f32; 3]; 3],
    /// Offset vector
    pub offset: [f32; 3],
}

impl ColorTransform {
    /// Create identity transform
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            offset: [0.0, 0.0, 0.0],
        }
    }

    /// Apply transform to RGB color
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let r = rgb[0] * self.matrix[0][0]
            + rgb[1] * self.matrix[0][1]
            + rgb[2] * self.matrix[0][2]
            + self.offset[0];
        let g = rgb[0] * self.matrix[1][0]
            + rgb[1] * self.matrix[1][1]
            + rgb[2] * self.matrix[1][2]
            + self.offset[1];
        let b = rgb[0] * self.matrix[2][0]
            + rgb[1] * self.matrix[2][1]
            + rgb[2] * self.matrix[2][2]
            + self.offset[2];

        [
            r.max(0.0).min(1.0),
            g.max(0.0).min(1.0),
            b.max(0.0).min(1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_transform_identity() {
        let transform = ColorTransform::identity();
        let input = [0.5, 0.7, 0.3];
        let output = transform.apply(input);

        assert!((output[0] - input[0]).abs() < 1e-6);
        assert!((output[1] - input[1]).abs() < 1e-6);
        assert!((output[2] - input[2]).abs() < 1e-6);
    }
}
