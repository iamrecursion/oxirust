//! Color matrix coefficients for `YCbCr`/RGB conversions.
//!
//! Implements the standard color matrix definitions from ITU-R BT.601, BT.709,
//! and BT.2020 for converting between RGB and `YCbCr` (Y'CbCr) representations.

use std::fmt;

/// A 3x3 color matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorMatrix3x3 {
    /// Row-major 3x3 matrix elements.
    pub m: [[f64; 3]; 3],
}

impl ColorMatrix3x3 {
    /// Creates a new color matrix from row-major elements.
    #[must_use]
    pub fn new(m: [[f64; 3]; 3]) -> Self {
        Self { m }
    }

    /// Returns the identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiplies this matrix by a 3-element column vector.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mul_vec(&self, v: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * v[0] + self.m[0][1] * v[1] + self.m[0][2] * v[2],
            self.m[1][0] * v[0] + self.m[1][1] * v[1] + self.m[1][2] * v[2],
            self.m[2][0] * v[0] + self.m[2][1] * v[1] + self.m[2][2] * v[2],
        ]
    }

    /// Multiplies this matrix by another matrix (self * other).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mul_mat(&self, other: &Self) -> Self {
        let mut result = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j];
            }
        }
        Self { m: result }
    }

    /// Returns the transpose of this matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    /// Computes the determinant of this matrix.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn determinant(&self) -> f64 {
        let m = &self.m;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Computes the inverse of this matrix, or `None` if singular.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.m;
        Some(Self {
            m: [
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
            ],
        })
    }
}

impl fmt::Display for ColorMatrix3x3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.m {
            writeln!(f, "[{:>10.6} {:>10.6} {:>10.6}]", row[0], row[1], row[2])?;
        }
        Ok(())
    }
}

/// Standard color matrix coefficients specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixCoefficients {
    /// ITU-R BT.601 (SDTV) — Kr=0.299, Kb=0.114
    Bt601,
    /// ITU-R BT.709 (HDTV) — Kr=0.2126, Kb=0.0722
    Bt709,
    /// ITU-R BT.2020 (UHDTV) — Kr=0.2627, Kb=0.0593
    Bt2020,
    /// SMPTE 240M — Kr=0.212, Kb=0.087
    Smpte240M,
    /// Identity (RGB = `YCbCr`, no conversion)
    Identity,
}

impl MatrixCoefficients {
    /// Returns the (Kr, Kb) luma coefficients for this standard.
    #[must_use]
    pub fn kr_kb(&self) -> (f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020 => (0.2627, 0.0593),
            Self::Smpte240M => (0.212, 0.087),
            Self::Identity => (0.0, 0.0),
        }
    }

    /// Builds the RGB-to-`YCbCr` matrix (full range) for this standard.
    ///
    /// Y  = Kr*R + Kg*G + Kb*B
    /// Cb = (B - Y) / (2*(1-Kb))
    /// Cr = (R - Y) / (2*(1-Kr))
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rgb_to_ycbcr_matrix(&self) -> ColorMatrix3x3 {
        if *self == Self::Identity {
            return ColorMatrix3x3::identity();
        }
        let (kr, kb) = self.kr_kb();
        let kg = 1.0 - kr - kb;
        ColorMatrix3x3::new([
            [kr, kg, kb],
            [-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5],
            [0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))],
        ])
    }

    /// Builds the `YCbCr`-to-RGB matrix (full range) for this standard.
    ///
    /// This is the inverse of `rgb_to_ycbcr_matrix`.
    #[must_use]
    pub fn ycbcr_to_rgb_matrix(&self) -> ColorMatrix3x3 {
        if *self == Self::Identity {
            return ColorMatrix3x3::identity();
        }
        self.rgb_to_ycbcr_matrix()
            .inverse()
            .unwrap_or_else(ColorMatrix3x3::identity)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Bt601 => "BT.601",
            Self::Bt709 => "BT.709",
            Self::Bt2020 => "BT.2020",
            Self::Smpte240M => "SMPTE 240M",
            Self::Identity => "Identity",
        }
    }
}

/// Converts an RGB pixel to `YCbCr` using the specified matrix coefficients.
///
/// # Arguments
/// * `rgb` - `[R, G, B]` in 0..1 range
/// * `coeffs` - Which standard to use
///
/// # Returns
/// `[Y, Cb, Cr]` values
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rgb_to_ycbcr(rgb: [f64; 3], coeffs: MatrixCoefficients) -> [f64; 3] {
    coeffs.rgb_to_ycbcr_matrix().mul_vec(rgb)
}

/// Converts a `YCbCr` pixel to RGB using the specified matrix coefficients.
///
/// # Arguments
/// * `ycbcr` - `[Y, Cb, Cr]` values
/// * `coeffs` - Which standard to use
///
/// # Returns
/// `[R, G, B]` in 0..1 range
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ycbcr_to_rgb(ycbcr: [f64; 3], coeffs: MatrixCoefficients) -> [f64; 3] {
    coeffs.ycbcr_to_rgb_matrix().mul_vec(ycbcr)
}

/// Applies studio-range (narrow range) quantization offsets.
///
/// For 8-bit: Y is [16..235], Cb/Cr is [16..240] centered at 128.
///
/// # Arguments
/// * `ycbcr` - Full-range `[Y, Cb, Cr]`
/// * `bit_depth` - Bit depth (8, 10, 12)
///
/// # Returns
/// Narrow-range integer values as `[Y, Cb, Cr]` in f64.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn full_to_narrow_range(ycbcr: [f64; 3], bit_depth: u32) -> [f64; 3] {
    let scale = (1u64 << bit_depth) as f64;
    let y_offset = 16.0 * scale / 256.0;
    let c_offset = 128.0 * scale / 256.0;
    let y_range = 219.0 * scale / 256.0;
    let c_range = 224.0 * scale / 256.0;
    [
        ycbcr[0] * y_range + y_offset,
        ycbcr[1] * c_range + c_offset,
        ycbcr[2] * c_range + c_offset,
    ]
}

/// Converts narrow-range integer values back to full-range `YCbCr`.
///
/// # Arguments
/// * `narrow` - Narrow-range `[Y, Cb, Cr]` values
/// * `bit_depth` - Bit depth (8, 10, 12)
///
/// # Returns
/// Full-range `[Y, Cb, Cr]`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn narrow_to_full_range(narrow: [f64; 3], bit_depth: u32) -> [f64; 3] {
    let scale = (1u64 << bit_depth) as f64;
    let y_offset = 16.0 * scale / 256.0;
    let c_offset = 128.0 * scale / 256.0;
    let y_range = 219.0 * scale / 256.0;
    let c_range = 224.0 * scale / 256.0;
    [
        (narrow[0] - y_offset) / y_range,
        (narrow[1] - c_offset) / c_range,
        (narrow[2] - c_offset) / c_range,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id = ColorMatrix3x3::identity();
        let v = [1.0, 2.0, 3.0];
        let result = id.mul_vec(v);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
        assert!((result[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_determinant() {
        let id = ColorMatrix3x3::identity();
        assert!((id.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = ColorMatrix3x3::new([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]);
        let inv = m.inverse().expect("should be invertible");
        assert!((inv.m[0][0] - 0.5).abs() < 1e-12);
        assert!((inv.m[1][1] - 1.0 / 3.0).abs() < 1e-12);
        assert!((inv.m[2][2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_singular_matrix_no_inverse() {
        let m = ColorMatrix3x3::new([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_transpose() {
        let m = ColorMatrix3x3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let t = m.transpose();
        assert!((t.m[0][1] - 4.0).abs() < 1e-12);
        assert!((t.m[1][0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_bt709_luma_coefficients() {
        let (kr, kb) = MatrixCoefficients::Bt709.kr_kb();
        let kg = 1.0 - kr - kb;
        assert!((kr + kg + kb - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bt601_luma_coefficients() {
        let (kr, kb) = MatrixCoefficients::Bt601.kr_kb();
        assert!((kr - 0.299).abs() < 1e-12);
        assert!((kb - 0.114).abs() < 1e-12);
    }

    #[test]
    fn test_rgb_to_ycbcr_roundtrip_bt709() {
        let rgb = [0.5, 0.3, 0.7];
        let ycbcr = rgb_to_ycbcr(rgb, MatrixCoefficients::Bt709);
        let back = ycbcr_to_rgb(ycbcr, MatrixCoefficients::Bt709);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-10,
                "channel {i} mismatch: {} vs {}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_roundtrip_bt2020() {
        let rgb = [0.8, 0.1, 0.4];
        let ycbcr = rgb_to_ycbcr(rgb, MatrixCoefficients::Bt2020);
        let back = ycbcr_to_rgb(ycbcr, MatrixCoefficients::Bt2020);
        for i in 0..3 {
            assert!(
                (back[i] - rgb[i]).abs() < 1e-10,
                "channel {i} mismatch: {} vs {}",
                back[i],
                rgb[i]
            );
        }
    }

    #[test]
    fn test_white_luma_is_one() {
        // White (1,1,1) should have Y=1 for any standard
        for coeffs in &[
            MatrixCoefficients::Bt601,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Bt2020,
        ] {
            let ycbcr = rgb_to_ycbcr([1.0, 1.0, 1.0], *coeffs);
            assert!(
                (ycbcr[0] - 1.0).abs() < 1e-10,
                "Y for white should be 1.0 for {:?}",
                coeffs
            );
        }
    }

    #[test]
    fn test_narrow_range_roundtrip() {
        let full = [0.5, 0.1, -0.2];
        let narrow = full_to_narrow_range(full, 8);
        let back = narrow_to_full_range(narrow, 8);
        for i in 0..3 {
            assert!(
                (back[i] - full[i]).abs() < 1e-10,
                "channel {i} roundtrip failed"
            );
        }
    }

    #[test]
    fn test_narrow_range_10bit() {
        let full = [0.0, 0.0, 0.0];
        let narrow = full_to_narrow_range(full, 10);
        // 10-bit Y=0 should map to 64 (16 * 1024/256)
        assert!((narrow[0] - 64.0).abs() < 1e-6);
        // 10-bit Cb/Cr=0 should map to 512 (128 * 1024/256)
        assert!((narrow[1] - 512.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_coefficients() {
        let m = MatrixCoefficients::Identity.rgb_to_ycbcr_matrix();
        let id = ColorMatrix3x3::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m.m[i][j] - id.m[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_label() {
        assert_eq!(MatrixCoefficients::Bt709.label(), "BT.709");
        assert_eq!(MatrixCoefficients::Bt2020.label(), "BT.2020");
    }
}
