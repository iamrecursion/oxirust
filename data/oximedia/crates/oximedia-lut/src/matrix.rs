//! Color matrix operations.
//!
//! This module provides 3x3 and 3x4 matrix operations for color transformations,
//! including matrix multiplication, inversion, and concatenation.

use crate::{Matrix3x4, Rgb, Xyz};

pub use crate::Matrix3x3;

/// Apply a 3x3 matrix to an RGB color.
#[must_use]
pub fn apply_matrix3x3(matrix: &Matrix3x3, rgb: &Rgb) -> Rgb {
    [
        matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
        matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
        matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
    ]
}

/// Apply a 3x4 matrix (3x3 + offset) to an RGB color.
#[must_use]
pub fn apply_matrix3x4(matrix: &Matrix3x4, rgb: &Rgb) -> Rgb {
    [
        matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2] + matrix[0][3],
        matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2] + matrix[1][3],
        matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2] + matrix[2][3],
    ]
}

/// Multiply two 3x3 matrices.
#[must_use]
pub fn multiply_matrix3x3(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

/// Compute the determinant of a 3x3 matrix.
#[must_use]
pub fn determinant(m: &Matrix3x3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Invert a 3x3 matrix.
///
/// Returns `None` if the matrix is singular (determinant is zero).
#[must_use]
pub fn invert_matrix3x3(m: &Matrix3x3) -> Option<Matrix3x3> {
    let det = determinant(m);

    if det.abs() < 1e-10 {
        return None;
    }

    let inv_det = 1.0 / det;

    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

/// Transpose a 3x3 matrix.
#[must_use]
pub fn transpose(m: &Matrix3x3) -> Matrix3x3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Identity matrix.
pub const IDENTITY: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// RGB to XYZ conversion matrix for sRGB/Rec.709.
pub const RGB_TO_XYZ_REC709: Matrix3x3 = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

/// XYZ to RGB conversion matrix for sRGB/Rec.709.
pub const XYZ_TO_RGB_REC709: Matrix3x3 = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

/// RGB to XYZ conversion matrix for Rec.2020.
pub const RGB_TO_XYZ_REC2020: Matrix3x3 = [
    [0.636_958_0, 0.144_616_9, 0.168_881_0],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.000_000_0, 0.028_072_7, 1.060_985_1],
];

/// XYZ to RGB conversion matrix for Rec.2020.
pub const XYZ_TO_RGB_REC2020: Matrix3x3 = [
    [1.716_651_2, -0.355_670_8, -0.253_366_3],
    [-0.666_684_4, 1.616_481_2, 0.015_768_5],
    [0.017_639_9, -0.042_770_6, 0.942_103_1],
];

/// RGB to XYZ conversion matrix for DCI-P3 (D65 white point).
pub const RGB_TO_XYZ_DCIP3: Matrix3x3 = [
    [0.486_570_9, 0.265_667_7, 0.198_217_3],
    [0.228_974_6, 0.691_738_5, 0.079_286_9],
    [0.000_000_0, 0.045_113_4, 1.043_944_4],
];

/// XYZ to RGB conversion matrix for DCI-P3 (D65 white point).
pub const XYZ_TO_RGB_DCIP3: Matrix3x3 = [
    [2.493_496_9, -0.931_383_6, -0.402_710_8],
    [-0.829_489_0, 1.762_664_1, 0.023_624_7],
    [0.035_845_8, -0.076_172_4, 0.956_884_5],
];

/// RGB to XYZ conversion matrix for Adobe RGB.
pub const RGB_TO_XYZ_ADOBE: Matrix3x3 = [
    [0.576_730_9, 0.185_554_0, 0.188_185_2],
    [0.297_376_9, 0.627_349_1, 0.075_274_1],
    [0.027_034_3, 0.070_687_2, 0.991_108_5],
];

/// XYZ to RGB conversion matrix for Adobe RGB.
pub const XYZ_TO_RGB_ADOBE: Matrix3x3 = [
    [2.041_369_0, -0.564_946_4, -0.344_694_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.013_447_4, -0.118_389_7, 1.015_409_6],
];

/// Convert RGB to XYZ using a matrix.
#[must_use]
pub fn rgb_to_xyz(rgb: &Rgb, matrix: &Matrix3x3) -> Xyz {
    apply_matrix3x3(matrix, rgb)
}

/// Convert XYZ to RGB using a matrix.
#[must_use]
pub fn xyz_to_rgb(xyz: &Xyz, matrix: &Matrix3x3) -> Rgb {
    apply_matrix3x3(matrix, xyz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_matrix3x3() {
        let matrix = IDENTITY;
        let rgb = [0.5, 0.3, 0.7];
        let result = apply_matrix3x3(&matrix, &rgb);
        assert!((result[0] - rgb[0]).abs() < 1e-10);
        assert!((result[1] - rgb[1]).abs() < 1e-10);
        assert!((result[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_multiply_matrix3x3() {
        let result = multiply_matrix3x3(&IDENTITY, &IDENTITY);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_determinant() {
        let det = determinant(&IDENTITY);
        assert!((det - 1.0).abs() < 1e-10);

        // Singular matrix
        let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        let det = determinant(&singular);
        assert!(det.abs() < 1e-10);
    }

    #[test]
    fn test_invert_matrix3x3() {
        let inv = invert_matrix3x3(&IDENTITY).expect("should succeed in test");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }

        // Test that M * M^-1 = I
        let m = RGB_TO_XYZ_REC709;
        let m_inv = invert_matrix3x3(&m).expect("should succeed in test");
        let result = multiply_matrix3x3(&m, &m_inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_transpose() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mt = transpose(&m);
        assert_eq!(mt[0][1], 4.0);
        assert_eq!(mt[0][2], 7.0);
        assert_eq!(mt[1][0], 2.0);
        assert_eq!(mt[1][2], 8.0);
        assert_eq!(mt[2][0], 3.0);
        assert_eq!(mt[2][1], 6.0);
    }

    #[test]
    fn test_rgb_xyz_round_trip() {
        let rgb = [0.5, 0.3, 0.7];
        let xyz = rgb_to_xyz(&rgb, &RGB_TO_XYZ_REC709);
        let rgb2 = xyz_to_rgb(&xyz, &XYZ_TO_RGB_REC709);
        assert!((rgb[0] - rgb2[0]).abs() < 1e-6);
        assert!((rgb[1] - rgb2[1]).abs() < 1e-6);
        assert!((rgb[2] - rgb2[2]).abs() < 1e-6);
    }
}
