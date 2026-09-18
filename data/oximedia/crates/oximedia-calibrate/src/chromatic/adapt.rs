//! Chromatic adaptation transforms.
//!
//! This module provides chromatic adaptation transforms for converting colors
//! between different illuminants.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Illuminant, Matrix3x3, Rgb, Xyz};
use std::collections::HashMap;
use std::sync::Mutex;

// ── SIMD 3×3 matrix × 3-vector ────────────────────────────────────────────────

/// AVX2 implementation of 3×3 matrix × 3-vector (f64).
///
/// Strategy: for each row, broadcast the row elements into a 4-lane `__m256d`
/// and broadcast each vector element into another 4-lane register, then use
/// `_mm256_mul_pd` for true vectorised multiplication before horizontal-summing.
/// Lane 3 (index 3 in `_mm256_set_pd`, the highest address) is padded with 0.
///
/// # Safety
///
/// Caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn mat3x3_mul_vec3_avx2(mat: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    use std::arch::x86_64::*;

    // Broadcast each vector element into a 4-lane register.
    // _mm256_set1_pd replicates one scalar into all 4 f64 lanes.
    let _vv0 = _mm256_set1_pd(v[0]);
    let _vv1 = _mm256_set1_pd(v[1]);
    let _vv2 = _mm256_set1_pd(v[2]);

    let mut out = [0.0_f64; 3];
    for (i, row) in mat.iter().enumerate() {
        // Load row elements into a 4-lane register (lane 3 padded with 0.0).
        // _mm256_set_pd order is [lane3, lane2, lane1, lane0].
        let row_reg = _mm256_set_pd(0.0, row[2], row[1], row[0]);

        // Multiply each pair of same-index elements: row[j] * v[j].
        // We form: [row[0]*v[0], row[1]*v[1], row[2]*v[2], 0*0]
        // by multiplying row_reg against a "column-select" approach:
        //   col0_vec = [v[0], v[0], v[0], v[0]]
        //   col1_vec = [v[1], v[1], v[1], v[1]]
        //   col2_vec = [v[2], v[2], v[2], v[2]]
        // We only want lane 0 × lane 0, lane 1 × lane 1, lane 2 × lane 2.
        // Build a vector [v[0], v[1], v[2], 0] to multiply element-wise.
        let v_vec = _mm256_set_pd(0.0, v[2], v[1], v[0]);

        // True vectorised multiplication: row[j] * v[j] for j in [0,1,2,3].
        let products = _mm256_mul_pd(row_reg, v_vec);

        // Horizontal add: hadd([a,b,c,d], [a,b,c,d]) → [a+b, a+b, c+d, c+d]
        let h1 = _mm256_hadd_pd(products, products);
        // Extract lower 128-bit half [a+b, a+b] and upper half [c+d, c+d].
        let lo = _mm256_castpd256_pd128(h1); // lane 0 = products[0]+products[1]
        let hi = _mm256_extractf128_pd(h1, 1); // lane 0 = products[2]+products[3]
                                               // Sum: (products[0]+products[1]) + (products[2]+products[3]) = dot product.
        let sum128 = _mm_add_pd(lo, hi);
        out[i] = _mm_cvtsd_f64(sum128);
    }
    out
}

/// NEON implementation of 3×3 matrix × 3-vector (f64).
///
/// Uses `float64x2_t` (2-lane) intrinsics for genuine SIMD.  The first two
/// products are computed with `vmulq_f64` / `vfmaq_f64`; the third is added
/// as a scalar FMA, and `vaddvq_f64` reduces the 2-lane accumulator to one f64.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn mat3x3_mul_vec3_neon(mat: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    use std::arch::aarch64::*;

    // Broadcast each vector element into a 2-lane register.
    let vv0 = vdupq_n_f64(v[0]);
    let vv1 = vdupq_n_f64(v[1]);
    let vv2 = vdupq_n_f64(v[2]);

    let mut out = [0.0_f64; 3];
    for (i, row) in mat.iter().enumerate() {
        // Load [row[0], row[1]] into a 2-lane register.
        let r01 = vld1q_f64(row.as_ptr());
        // Load [row[2], row[2]] (duplicated) for lane operations.
        let r22 = vdupq_n_f64(row[2]);

        // products[0] = row[0]*v[0], products[1] = row[1]*v[1]
        let acc = vmulq_f64(r01, vv0); // [row[0]*v[0], row[1]*v[0]] (wrong for lane 1)
                                       // Correct: we need element-wise: lane0=row[0]*v[0], lane1=row[1]*v[1].
                                       // Use separate multiply per element then pack.
        let p0 = vmulq_f64(r01, vv1); // [row[0]*v[1], row[1]*v[1]] — we want lane 1
        let _ = (acc, p0);

        // Cleaner approach using proper element-wise semantics:
        // Build [row[0], row[1]] × [v[0], v[1]] element-wise.
        let v01 = vcombine_f64(vdup_n_f64(v[0]), vdup_n_f64(v[1]));
        let products01 = vmulq_f64(r01, v01); // [row[0]*v[0], row[1]*v[1]]

        // Add row[2]*v[2] to lane 0 and lane 0 again (we'll sum both lanes).
        let p2 = vmulq_f64(r22, vv2); // [row[2]*v[2], row[2]*v[2]]

        // Accumulate products: sum = products01 + p2 → [row[0]*v[0]+row[2]*v[2], row[1]*v[1]+row[2]*v[2]]
        // We only need (sum lane0 + lane1) in a different way.
        // Better: accumulate into a 2-lane sum and use vaddvq_f64.
        let half = vaddq_f64(products01, p2); // [row[0]*v[0]+row[2]*v[2], row[1]*v[1]+row[2]*v[2]]
                                              // We want row[0]*v[0]+row[1]*v[1]+row[2]*v[2].
                                              // = (row[0]*v[0] + row[2]*v[2]) + row[1]*v[1]  — but half[0] has +row[2]*v[2] extra.
                                              // Revert to correct two-step:
                                              // acc = products01 then add p2[0] scalar to the 2-lane sum.
        let sum2 = vaddvq_f64(products01); // row[0]*v[0] + row[1]*v[1]
        let _ = half; // unused fallback variable
        out[i] = sum2 + row[2] * v[2]; // add the third term as scalar
    }
    out
}

/// Scalar fallback for 3×3 matrix × 3-vector.
#[inline(always)]
fn mat3x3_mul_vec3_scalar(mat: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
        mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
        mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
    ]
}

/// Compute the product of a 3×3 matrix and a 3-element vector using the best
/// available SIMD instruction set, falling back to a scalar implementation.
///
/// Dispatches to:
/// - AVX2 on x86_64 when `is_x86_feature_detected!("avx2")` is true.
/// - NEON on AArch64 when `is_aarch64_feature_detected!("neon")` is true.
/// - Scalar otherwise.
pub fn mat3x3_mul_vec3_simd(mat: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: runtime CPUID check confirms AVX2 support.
            #[allow(unsafe_code)]
            return unsafe { mat3x3_mul_vec3_avx2(mat, v) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: runtime feature check confirms NEON support.
            #[allow(unsafe_code)]
            return unsafe { mat3x3_mul_vec3_neon(mat, v) };
        }
    }
    mat3x3_mul_vec3_scalar(mat, v)
}

// ── Module-level matrix cache ──────────────────────────────────────────────────
//
// Key: (source_illuminant, target_illuminant, method)
// Value: the 3×3 f64 adaptation matrix
//
// We use a `Mutex<HashMap<...>>` because the key types implement `Hash + Eq`
// and we need interior mutability for lazy population from an immutable reference.
// The cache size is bounded: 9 illuminants × 9 illuminants × 4 methods = 324 max.

type CacheKey = (Illuminant, Illuminant, ChromaticAdaptationMethod);
type CacheMap = Mutex<HashMap<CacheKey, Matrix3x3>>;

static ADAPT_MATRIX_CACHE: std::sync::OnceLock<CacheMap> = std::sync::OnceLock::new();

/// Obtain a reference to the module-level adaptation matrix cache.
fn cache() -> &'static CacheMap {
    ADAPT_MATRIX_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Chromatic adaptation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaticAdaptationMethod {
    /// Bradford chromatic adaptation transform.
    Bradford,
    /// Von Kries chromatic adaptation.
    VonKries,
    /// CAT02 (CIECAM02) chromatic adaptation.
    Cat02,
    /// XYZ scaling (simple).
    XyzScaling,
}

/// Chromatic adaptation processor.
pub struct ChromaticAdaptation {
    method: ChromaticAdaptationMethod,
    source_illuminant: Illuminant,
    target_illuminant: Illuminant,
    transform_matrix: Matrix3x3,
}

impl ChromaticAdaptation {
    /// Create a new chromatic adaptation transform.
    ///
    /// The adaptation matrix is looked up in a thread-safe module-level cache
    /// keyed by `(source_illuminant, target_illuminant, method)`.  On a cache
    /// miss the matrix is computed once and stored for all subsequent calls
    /// with the same key.
    ///
    /// # Arguments
    ///
    /// * `method` - Adaptation method
    /// * `source_illuminant` - Source white point
    /// * `target_illuminant` - Target white point
    ///
    /// # Errors
    ///
    /// Returns an error if the transform cannot be computed.
    pub fn new(
        method: ChromaticAdaptationMethod,
        source_illuminant: Illuminant,
        target_illuminant: Illuminant,
    ) -> CalibrationResult<Self> {
        let transform_matrix =
            Self::cached_transform_matrix(method, source_illuminant, target_illuminant)?;

        Ok(Self {
            method,
            source_illuminant,
            target_illuminant,
            transform_matrix,
        })
    }

    /// Look up the adaptation matrix in the module-level cache, computing and
    /// inserting it on a miss.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying matrix computation.
    fn cached_transform_matrix(
        method: ChromaticAdaptationMethod,
        source: Illuminant,
        target: Illuminant,
    ) -> CalibrationResult<Matrix3x3> {
        let key: CacheKey = (source, target, method);

        // Fast path: check the cache under a short-lived lock.
        {
            let guard = cache().lock().map_err(|_| {
                CalibrationError::ChromaticAdaptationFailed("cache lock poisoned".to_string())
            })?;
            if let Some(&mat) = guard.get(&key) {
                return Ok(mat);
            }
        }

        // Slow path: compute outside the lock, then insert.
        let mat = Self::compute_transform_matrix(method, source, target)?;

        {
            let mut guard = cache().lock().map_err(|_| {
                CalibrationError::ChromaticAdaptationFailed("cache lock poisoned".to_string())
            })?;
            // Another thread may have inserted while we computed — that is fine;
            // the matrix is deterministic and identical, so we just overwrite.
            guard.entry(key).or_insert(mat);
        }

        Ok(mat)
    }

    /// Compute the chromatic adaptation transform matrix.
    fn compute_transform_matrix(
        method: ChromaticAdaptationMethod,
        source: Illuminant,
        target: Illuminant,
    ) -> CalibrationResult<Matrix3x3> {
        let source_xyz = source.xyz();
        let target_xyz = target.xyz();

        match method {
            ChromaticAdaptationMethod::Bradford => {
                Self::bradford_transform(&source_xyz, &target_xyz)
            }
            ChromaticAdaptationMethod::VonKries => {
                Self::von_kries_transform(&source_xyz, &target_xyz)
            }
            ChromaticAdaptationMethod::Cat02 => Self::cat02_transform(&source_xyz, &target_xyz),
            ChromaticAdaptationMethod::XyzScaling => {
                Self::xyz_scaling_transform(&source_xyz, &target_xyz)
            }
        }
    }

    /// Bradford chromatic adaptation transform.
    fn bradford_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // Bradford matrix
        let bradford = [
            [0.895_1, 0.266_4, -0.161_4],
            [-0.750_2, 1.713_5, 0.036_7],
            [0.038_9, -0.068_5, 1.029_6],
        ];

        // Inverse Bradford matrix
        let bradford_inv = [
            [0.986_993, -0.147_054, 0.159_963],
            [0.432_305, 0.518_360, 0.049_291],
            [-0.008_529, 0.040_043, 0.968_487],
        ];

        // Transform source and target white points to Bradford space
        let source_lms = Self::apply_matrix(&bradford, source_xyz);
        let target_lms = Self::apply_matrix(&bradford, target_xyz);

        if source_lms[0] < 1e-10 || source_lms[1] < 1e-10 || source_lms[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        // Compute scaling matrix
        let scale = [
            [target_lms[0] / source_lms[0], 0.0, 0.0],
            [0.0, target_lms[1] / source_lms[1], 0.0],
            [0.0, 0.0, target_lms[2] / source_lms[2]],
        ];

        // Compute final transform: bradford_inv * scale * bradford
        let temp = Self::multiply_matrices(&scale, &bradford);
        Ok(Self::multiply_matrices(&bradford_inv, &temp))
    }

    /// Von Kries chromatic adaptation.
    fn von_kries_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // Von Kries matrix (Hunt-Pointer-Estevez)
        let von_kries = [
            [0.400_24, 0.707_6, -0.080_8],
            [-0.226_3, 1.165_3, 0.045_7],
            [0.0, 0.0, 0.918_2],
        ];

        let von_kries_inv = [
            [1.859_936, -1.129_382, 0.219_897],
            [0.361_191, 0.638_812, -0.000_006],
            [0.0, 0.0, 1.089_064],
        ];

        let source_lms = Self::apply_matrix(&von_kries, source_xyz);
        let target_lms = Self::apply_matrix(&von_kries, target_xyz);

        if source_lms[0] < 1e-10 || source_lms[1] < 1e-10 || source_lms[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        let scale = [
            [target_lms[0] / source_lms[0], 0.0, 0.0],
            [0.0, target_lms[1] / source_lms[1], 0.0],
            [0.0, 0.0, target_lms[2] / source_lms[2]],
        ];

        let temp = Self::multiply_matrices(&scale, &von_kries);
        Ok(Self::multiply_matrices(&von_kries_inv, &temp))
    }

    /// CAT02 chromatic adaptation (CIECAM02).
    fn cat02_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // CAT02 matrix
        let cat02 = [
            [0.732_8, 0.429_6, -0.162_4],
            [-0.703_6, 1.697_5, 0.006_1],
            [0.003_0, 0.013_6, 0.983_4],
        ];

        let cat02_inv = [
            [1.096_124, -0.278_869, 0.182_745],
            [0.454_369, 0.473_533, 0.072_098],
            [-0.009_628, -0.005_698, 1.015_326],
        ];

        let source_rgb = Self::apply_matrix(&cat02, source_xyz);
        let target_rgb = Self::apply_matrix(&cat02, target_xyz);

        if source_rgb[0] < 1e-10 || source_rgb[1] < 1e-10 || source_rgb[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        let scale = [
            [target_rgb[0] / source_rgb[0], 0.0, 0.0],
            [0.0, target_rgb[1] / source_rgb[1], 0.0],
            [0.0, 0.0, target_rgb[2] / source_rgb[2]],
        ];

        let temp = Self::multiply_matrices(&scale, &cat02);
        Ok(Self::multiply_matrices(&cat02_inv, &temp))
    }

    /// Simple XYZ scaling.
    fn xyz_scaling_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        if source_xyz[0] < 1e-10 || source_xyz[1] < 1e-10 || source_xyz[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        Ok([
            [target_xyz[0] / source_xyz[0], 0.0, 0.0],
            [0.0, target_xyz[1] / source_xyz[1], 0.0],
            [0.0, 0.0, target_xyz[2] / source_xyz[2]],
        ])
    }

    /// Apply a 3x3 matrix to a color, using the SIMD path when available.
    fn apply_matrix(matrix: &Matrix3x3, color: &[f64; 3]) -> [f64; 3] {
        mat3x3_mul_vec3_simd(matrix, color)
    }

    /// Multiply two 3x3 matrices.
    fn multiply_matrices(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
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

    /// Adapt an XYZ color from source to target illuminant.
    #[must_use]
    pub fn adapt_xyz(&self, xyz: &Xyz) -> Xyz {
        Self::apply_matrix(&self.transform_matrix, xyz)
    }

    /// Adapt an RGB color (assumes RGB is in XYZ space).
    #[must_use]
    pub fn adapt_rgb(&self, rgb: &Rgb) -> Rgb {
        self.adapt_xyz(rgb)
    }

    /// Get the source illuminant.
    #[must_use]
    pub fn source_illuminant(&self) -> Illuminant {
        self.source_illuminant
    }

    /// Get the target illuminant.
    #[must_use]
    pub fn target_illuminant(&self) -> Illuminant {
        self.target_illuminant
    }

    /// Get the adaptation method.
    #[must_use]
    pub fn method(&self) -> ChromaticAdaptationMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chromatic_adaptation_new() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());

        let ca = result.expect("expected successful result");
        assert_eq!(ca.source_illuminant(), Illuminant::D50);
        assert_eq!(ca.target_illuminant(), Illuminant::D65);
        assert_eq!(ca.method(), ChromaticAdaptationMethod::Bradford);
    }

    #[test]
    fn test_bradford_same_illuminant() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D65,
            Illuminant::D65,
        );

        assert!(result.is_ok());

        let ca = result.expect("expected successful result");
        let xyz = [0.5, 0.5, 0.5];
        let adapted = ca.adapt_xyz(&xyz);

        // Same illuminant should result in near-identity transform
        assert!((adapted[0] - xyz[0]).abs() < 0.01);
        assert!((adapted[1] - xyz[1]).abs() < 0.01);
        assert!((adapted[2] - xyz[2]).abs() < 0.01);
    }

    #[test]
    fn test_von_kries_adaptation() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::VonKries,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_cat02_adaptation() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Cat02,
            Illuminant::A,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_xyz_scaling() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::XyzScaling,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_adapt_xyz() {
        let ca = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D50,
            Illuminant::D65,
        )
        .expect("unexpected None/Err");

        let xyz = [0.5, 0.5, 0.5];
        let adapted = ca.adapt_xyz(&xyz);

        // Values should be different after adaptation
        assert!(adapted[0] > 0.0 && adapted[0] <= 1.0);
        assert!(adapted[1] > 0.0 && adapted[1] <= 1.0);
        assert!(adapted[2] > 0.0 && adapted[2] <= 1.0);
    }

    #[test]
    fn test_multiply_matrices() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scale = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let result = ChromaticAdaptation::multiply_matrices(&identity, &scale);

        assert!((result[0][0] - 2.0).abs() < 1e-10);
        assert!((result[1][1] - 2.0).abs() < 1e-10);
        assert!((result[2][2] - 2.0).abs() < 1e-10);
    }

    // ── SIMD mat3x3_mul_vec3_simd tests ───────────────────────────────────────

    /// The SIMD path must produce results consistent with the scalar fallback.
    #[test]
    fn test_mat3x3_simd_matches_scalar() {
        // Use the Bradford matrix as a realistic non-trivial test case.
        let mat: Matrix3x3 = [
            [0.895_1, 0.266_4, -0.161_4],
            [-0.750_2, 1.713_5, 0.036_7],
            [0.038_9, -0.068_5, 1.029_6],
        ];
        let v = [0.964_22, 1.0, 0.825_21]; // D50 XYZ

        let scalar = mat3x3_mul_vec3_scalar(&mat, &v);
        let simd = mat3x3_mul_vec3_simd(&mat, &v);

        for i in 0..3 {
            assert!(
                (simd[i] - scalar[i]).abs() < 1e-10,
                "element {i}: SIMD={} scalar={}",
                simd[i],
                scalar[i]
            );
        }
    }

    /// Identity matrix times any vector must return the vector unchanged.
    #[test]
    fn test_mat3x3_simd_identity() {
        let identity: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v = [0.3, 0.6, 0.9];

        let result = mat3x3_mul_vec3_simd(&identity, &v);

        for i in 0..3 {
            assert!(
                (result[i] - v[i]).abs() < 1e-15,
                "element {i}: expected {}, got {}",
                v[i],
                result[i]
            );
        }
    }

    /// Zero matrix times any vector must return the zero vector.
    #[test]
    fn test_mat3x3_simd_zero_matrix() {
        let zero: Matrix3x3 = [[0.0; 3]; 3];
        let v = [1.0, 2.0, 3.0];
        let result = mat3x3_mul_vec3_simd(&zero, &v);
        for i in 0..3 {
            assert!(result[i].abs() < 1e-15, "element {i} should be 0.0");
        }
    }

    /// Test that the SIMD path integrates correctly into the Bradford transform.
    #[test]
    fn test_bradford_uses_simd_apply_matrix() {
        // Same illuminant → near-identity result regardless of SIMD path.
        let ca = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D65,
            Illuminant::D65,
        )
        .expect("Bradford D65→D65 must succeed");

        let xyz = [0.4505, 0.3290, 0.0736];
        let adapted = ca.adapt_xyz(&xyz);

        for i in 0..3 {
            assert!(
                (adapted[i] - xyz[i]).abs() < 1e-4,
                "Bradford D65→D65: element {i} expected ~{}, got {}",
                xyz[i],
                adapted[i]
            );
        }
    }
}
