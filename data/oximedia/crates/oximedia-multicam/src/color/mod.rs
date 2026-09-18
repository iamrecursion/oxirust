//! Color matching for multi-camera production.

pub mod balance;
pub mod match_color;

pub use balance::WhiteBalanceMatching;
pub use match_color::ColorMatcher;

use crate::AngleId;

/// Color statistics for an angle
#[derive(Debug, Clone, Copy)]
pub struct ColorStats {
    /// Angle identifier
    pub angle: AngleId,
    /// Mean RGB values
    pub mean_rgb: [f32; 3],
    /// Standard deviation RGB
    pub std_rgb: [f32; 3],
    /// Color temperature (Kelvin)
    pub temperature: f32,
    /// Tint adjustment
    pub tint: f32,
}

impl ColorStats {
    /// Create new color statistics
    #[must_use]
    pub fn new(angle: AngleId) -> Self {
        Self {
            angle,
            mean_rgb: [0.5, 0.5, 0.5],
            std_rgb: [0.1, 0.1, 0.1],
            temperature: 6500.0,
            tint: 0.0,
        }
    }

    /// Calculate color distance to another
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..3 {
            let diff = self.mean_rgb[i] - other.mean_rgb[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Color correction matrix
#[derive(Debug, Clone, Copy)]
pub struct ColorMatrix {
    /// 3x3 color transformation matrix
    pub matrix: [[f32; 3]; 3],
}

impl ColorMatrix {
    /// Create identity matrix
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Apply matrix to RGB color
    #[must_use]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let mut result = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i] += self.matrix[i][j] * rgb[j];
            }
        }
        result
    }

    /// Multiply two matrices
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        Self { matrix: result }
    }
}

impl Default for ColorMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_stats_creation() {
        let stats = ColorStats::new(0);
        assert_eq!(stats.angle, 0);
        assert_eq!(stats.mean_rgb, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_color_distance() {
        let stats1 = ColorStats::new(0);
        let mut stats2 = ColorStats::new(1);
        stats2.mean_rgb = [0.6, 0.6, 0.6];

        let distance = stats1.distance_to(&stats2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_identity_matrix() {
        let matrix = ColorMatrix::identity();
        let color = [1.0, 0.5, 0.25];
        let result = matrix.apply(color);
        assert_eq!(result, color);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = ColorMatrix::identity();
        let m2 = ColorMatrix::identity();
        let result = m1.multiply(&m2);

        // Identity * Identity = Identity
        assert_eq!(result.matrix[0][0], 1.0);
        assert_eq!(result.matrix[1][1], 1.0);
        assert_eq!(result.matrix[2][2], 1.0);
    }
}
