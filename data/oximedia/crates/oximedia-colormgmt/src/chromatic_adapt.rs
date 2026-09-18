//! Chromatic adaptation transforms for color management.
//!
//! Provides Bradford, von Kries, and XYZ scaling adaptation methods
//! for converting colors between different illuminants/white points.

#![allow(dead_code)]

/// A 3x3 matrix stored in row-major order.
pub type Matrix3x3 = [[f64; 3]; 3];

/// Bradford chromatic adaptation matrix (M_A).
pub const BRADFORD_MA: Matrix3x3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Inverse Bradford matrix (M_A^{-1}).
pub const BRADFORD_MA_INV: Matrix3x3 = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

/// von Kries chromatic adaptation matrix.
pub const VON_KRIES_MA: Matrix3x3 = [
    [0.40024, 0.70760, -0.08081],
    [-0.22630, 1.16532, 0.04570],
    [0.00000, 0.00000, 0.91822],
];

/// Inverse von Kries matrix.
pub const VON_KRIES_MA_INV: Matrix3x3 = [
    [1.8599364, -1.1293816, 0.2198974],
    [0.3611914, 0.6388125, -0.0000064],
    [0.0000000, 0.0000000, 1.0890636],
];

/// XYZ scaling (identity) adaptation matrix.
pub const XYZ_SCALING_MA: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Standard CIE illuminant white points in XYZ.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum Illuminant {
    /// D50 illuminant (printing/photography standard).
    D50,
    /// D65 illuminant (sRGB, Rec.709 standard).
    D65,
    /// D55 illuminant.
    D55,
    /// Illuminant A (incandescent).
    A,
    /// Illuminant E (equal energy).
    E,
    /// Custom white point.
    Custom(f64, f64, f64),
}

impl Illuminant {
    /// Returns the XYZ white point for this illuminant.
    #[must_use]
    pub fn xyz(&self) -> [f64; 3] {
        match self {
            Self::D50 => [0.96422, 1.00000, 0.82521],
            Self::D65 => [0.95047, 1.00000, 1.08883],
            Self::D55 => [0.95682, 1.00000, 0.92149],
            Self::A => [1.09850, 1.00000, 0.35585],
            Self::E => [1.00000, 1.00000, 1.00000],
            Self::Custom(x, y, z) => [*x, *y, *z],
        }
    }
}

/// Chromatic adaptation method.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum AdaptationMethod {
    /// Bradford transform (most accurate for photographic use).
    Bradford,
    /// Von Kries diagonal adaptation.
    VonKries,
    /// XYZ scaling (simple diagonal scaling).
    XyzScaling,
}

impl AdaptationMethod {
    fn cone_response_matrix(&self) -> Matrix3x3 {
        match self {
            Self::Bradford => BRADFORD_MA,
            Self::VonKries => VON_KRIES_MA,
            Self::XyzScaling => XYZ_SCALING_MA,
        }
    }

    fn cone_response_matrix_inv(&self) -> Matrix3x3 {
        match self {
            Self::Bradford => BRADFORD_MA_INV,
            Self::VonKries => VON_KRIES_MA_INV,
            Self::XyzScaling => XYZ_SCALING_MA,
        }
    }
}

/// Multiply a 3x3 matrix by a 3-element column vector.
#[must_use]
pub fn mat3_mul_vec3(m: &Matrix3x3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiply two 3x3 matrices: result = a * b.
#[must_use]
pub fn mat3_mul_mat3(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

/// Build a chromatic adaptation matrix from source to destination white point.
///
/// Returns a 3x3 XYZ-to-XYZ matrix that transforms colors adapted to
/// `src_illuminant` into colors adapted to `dst_illuminant`.
///
/// # Arguments
///
/// * `src` - Source illuminant white point.
/// * `dst` - Destination illuminant white point.
/// * `method` - Chromatic adaptation method.
#[must_use]
pub fn build_adaptation_matrix(
    src: Illuminant,
    dst: Illuminant,
    method: AdaptationMethod,
) -> Matrix3x3 {
    let ma = method.cone_response_matrix();
    let ma_inv = method.cone_response_matrix_inv();

    let src_xyz = src.xyz();
    let dst_xyz = dst.xyz();

    // Map white points into cone response space
    let src_lms = mat3_mul_vec3(&ma, src_xyz);
    let dst_lms = mat3_mul_vec3(&ma, dst_xyz);

    // Build diagonal scaling matrix
    let scale: Matrix3x3 = [
        [dst_lms[0] / src_lms[0], 0.0, 0.0],
        [0.0, dst_lms[1] / src_lms[1], 0.0],
        [0.0, 0.0, dst_lms[2] / src_lms[2]],
    ];

    // M_adapt = M_A^{-1} * diag * M_A
    let tmp = mat3_mul_mat3(&scale, &ma);
    mat3_mul_mat3(&ma_inv, &tmp)
}

/// Apply chromatic adaptation to an XYZ color.
///
/// # Arguments
///
/// * `xyz` - Input color in XYZ.
/// * `src` - Source illuminant.
/// * `dst` - Destination illuminant.
/// * `method` - Adaptation method.
#[must_use]
pub fn adapt_xyz(
    xyz: [f64; 3],
    src: Illuminant,
    dst: Illuminant,
    method: AdaptationMethod,
) -> [f64; 3] {
    let m = build_adaptation_matrix(src, dst, method);
    mat3_mul_vec3(&m, xyz)
}

/// Convert XYZ under one illuminant to XYZ under another using Bradford.
///
/// This is the most common convenience function used in ICC workflows.
#[must_use]
pub fn bradford_adapt(xyz: [f64; 3], src: Illuminant, dst: Illuminant) -> [f64; 3] {
    adapt_xyz(xyz, src, dst, AdaptationMethod::Bradford)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn vec_approx_eq(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        approx_eq(a[0], b[0], tol) && approx_eq(a[1], b[1], tol) && approx_eq(a[2], b[2], tol)
    }

    #[test]
    fn test_d65_illuminant_xyz() {
        let wp = Illuminant::D65.xyz();
        assert!(approx_eq(wp[1], 1.0, 1e-10));
    }

    #[test]
    fn test_d50_illuminant_xyz() {
        let wp = Illuminant::D50.xyz();
        assert!(approx_eq(wp[1], 1.0, 1e-10));
    }

    #[test]
    fn test_custom_illuminant() {
        let il = Illuminant::Custom(0.9, 1.0, 0.8);
        let wp = il.xyz();
        assert!(approx_eq(wp[0], 0.9, 1e-10));
        assert!(approx_eq(wp[2], 0.8, 1e-10));
    }

    #[test]
    fn test_mat3_mul_vec3_identity() {
        let id = XYZ_SCALING_MA;
        let v = [0.5, 0.3, 0.7];
        let out = mat3_mul_vec3(&id, v);
        assert!(vec_approx_eq(out, v, 1e-10));
    }

    #[test]
    fn test_mat3_mul_mat3_identity() {
        let id = XYZ_SCALING_MA;
        let result = mat3_mul_mat3(&id, &id);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(result[i][j], expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_xyz_scaling_same_illuminant() {
        let xyz = [0.5, 0.4, 0.3];
        let adapted = adapt_xyz(
            xyz,
            Illuminant::D65,
            Illuminant::D65,
            AdaptationMethod::XyzScaling,
        );
        assert!(vec_approx_eq(adapted, xyz, 1e-6));
    }

    #[test]
    fn test_bradford_same_illuminant() {
        // D65 -> D65 should be identity
        let xyz = [0.5, 0.4, 0.3];
        let adapted = bradford_adapt(xyz, Illuminant::D65, Illuminant::D65);
        assert!(vec_approx_eq(adapted, xyz, 1e-6));
    }

    #[test]
    fn test_von_kries_same_illuminant() {
        let xyz = [0.5, 0.4, 0.3];
        let adapted = adapt_xyz(
            xyz,
            Illuminant::D50,
            Illuminant::D50,
            AdaptationMethod::VonKries,
        );
        assert!(vec_approx_eq(adapted, xyz, 1e-6));
    }

    #[test]
    fn test_build_adaptation_matrix_identity_diagonal() {
        // When src == dst, matrix should be close to identity
        let m =
            build_adaptation_matrix(Illuminant::D65, Illuminant::D65, AdaptationMethod::Bradford);
        for i in 0..3 {
            assert!(approx_eq(m[i][i], 1.0, 1e-5));
        }
    }

    #[test]
    fn test_bradford_d65_to_d50_preserves_neutral() {
        // A neutral gray (on white point locus) should map to a neutral gray
        let d65_wp = Illuminant::D65.xyz();
        let adapted = bradford_adapt(d65_wp, Illuminant::D65, Illuminant::D50);
        let d50_wp = Illuminant::D50.xyz();
        // Y channel should be preserved (luminance)
        assert!(approx_eq(adapted[1], d50_wp[1], 1e-5));
    }

    #[test]
    fn test_adaptation_method_cone_matrices_distinct() {
        let b = AdaptationMethod::Bradford.cone_response_matrix();
        let v = AdaptationMethod::VonKries.cone_response_matrix();
        // Bradford and VonKries differ
        assert!(!approx_eq(b[0][0], v[0][0], 1e-4));
    }

    #[test]
    fn test_xyz_scaling_matrix_is_identity() {
        let m = AdaptationMethod::XyzScaling.cone_response_matrix();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(m[i][j], expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_equal_energy_illuminant() {
        let wp = Illuminant::E.xyz();
        assert!(approx_eq(wp[0], 1.0, 1e-10));
        assert!(approx_eq(wp[1], 1.0, 1e-10));
        assert!(approx_eq(wp[2], 1.0, 1e-10));
    }

    #[test]
    fn test_illuminant_a_xyz_values() {
        let wp = Illuminant::A.xyz();
        // Illuminant A is warm/reddish, X > Z
        assert!(wp[0] > wp[2]);
    }
}
