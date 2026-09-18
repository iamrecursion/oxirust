//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

use super::functions_5::{simd_ln_f32, simd_ln_f64, simd_sin_f32};

/// SIMD-accelerated sine for f64 arrays
#[allow(dead_code)]
pub fn simd_sin_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const FRAC_1_PI: f64 = std::f64::consts::FRAC_1_PI;
    const PI: f64 = std::f64::consts::PI;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let pi = _mm256_set1_pd(PI);
                let two_pi = _mm256_set1_pd(2.0 * PI);
                let pi_over_2 = _mm256_set1_pd(PI / 2.0);
                let zero = _mm256_setzero_pd();
                let one = _mm256_set1_pd(1.0);
                let neg_one = _mm256_set1_pd(-1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c3 = _mm256_set1_pd(-1.0 / 6.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c7 = _mm256_set1_pd(-1.0 / 5040.0);
                let c9 = _mm256_set1_pd(1.0 / 362880.0);
                let c11 = _mm256_set1_pd(-1.0 / 39916800.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let mut x = _mm256_loadu_pd(slice.as_ptr());
                    let k = _mm256_round_pd(
                        _mm256_mul_pd(x, _mm256_set1_pd(FRAC_1_PI / 2.0)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));
                    let mask_gt_pi_2 = _mm256_cmp_pd(x, pi_over_2, _CMP_GT_OQ);
                    let mask_lt_neg_pi_2 =
                        _mm256_cmp_pd(x, _mm256_sub_pd(zero, pi_over_2), _CMP_LT_OQ);
                    let x_reduced_high = _mm256_sub_pd(pi, x);
                    x = _mm256_blendv_pd(x, x_reduced_high, mask_gt_pi_2);
                    let x_reduced_low = _mm256_sub_pd(_mm256_sub_pd(zero, pi), x);
                    x = _mm256_blendv_pd(x, x_reduced_low, mask_lt_neg_pi_2);
                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);
                    let x5 = _mm256_mul_pd(x3, x2);
                    let x7 = _mm256_mul_pd(x5, x2);
                    let x9 = _mm256_mul_pd(x7, x2);
                    let x11 = _mm256_mul_pd(x9, x2);
                    let mut sin_x = _mm256_mul_pd(c1, x);
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c3, x3));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c5, x5));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c7, x7));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c9, x9));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c11, x11));
                    sin_x = _mm256_max_pd(sin_x, neg_one);
                    sin_x = _mm256_min_pd(sin_x, one);
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].sin());
                }
                return Array1::from_vec(result);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;
                let pi = vdupq_n_f64(PI);
                let two_pi = vdupq_n_f64(2.0 * PI);
                let pi_over_2 = vdupq_n_f64(PI / 2.0);
                let neg_pi_over_2 = vdupq_n_f64(-PI / 2.0);
                let zero = vdupq_n_f64(0.0);
                let one = vdupq_n_f64(1.0);
                let neg_one = vdupq_n_f64(-1.0);
                let c1 = vdupq_n_f64(1.0);
                let c3 = vdupq_n_f64(-1.0 / 6.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c7 = vdupq_n_f64(-1.0 / 5040.0);
                let c9 = vdupq_n_f64(1.0 / 362880.0);
                let c11 = vdupq_n_f64(-1.0 / 39916800.0);
                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let mut x = vld1q_f64(slice.as_ptr());
                    let k = vrndnq_f64(vmulq_f64(x, vdupq_n_f64(FRAC_1_PI / 2.0)));
                    x = vsubq_f64(x, vmulq_f64(k, two_pi));
                    let mask_gt_pi_2 = vcgtq_f64(x, pi_over_2);
                    let mask_lt_neg_pi_2 = vcltq_f64(x, neg_pi_over_2);
                    let x_reduced_high = vsubq_f64(pi, x);
                    x = vbslq_f64(mask_gt_pi_2, x_reduced_high, x);
                    let x_reduced_low = vsubq_f64(vsubq_f64(zero, pi), x);
                    x = vbslq_f64(mask_lt_neg_pi_2, x_reduced_low, x);
                    let x2 = vmulq_f64(x, x);
                    let x3 = vmulq_f64(x2, x);
                    let x5 = vmulq_f64(x3, x2);
                    let x7 = vmulq_f64(x5, x2);
                    let x9 = vmulq_f64(x7, x2);
                    let x11 = vmulq_f64(x9, x2);
                    let mut sin_x = vmulq_f64(c1, x);
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c3, x3));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c5, x5));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c7, x7));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c9, x9));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c11, x11));
                    sin_x = vmaxq_f64(sin_x, neg_one);
                    sin_x = vminq_f64(sin_x, one);
                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 2;
                }
                for j in i..len {
                    result.push(input[j].sin());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        result.push(x.sin());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated cosine for f32 arrays
///
/// Computes cos(x) using range reduction and polynomial approximation.
/// Uses the identity: cos(x) = sin(x + π/2)
///
/// # Arguments
/// * `input` - Input array (radians)
///
/// # Returns
/// * Array of cos(x) values in range [-1, 1]
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_cos_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, std::f32::consts::PI];
/// let result = simd_cos_f32(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-6); // cos(0) = 1
/// ```
#[allow(dead_code)]
pub fn simd_cos_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let shifted: Array1<f32> = input.mapv(|x| x + std::f32::consts::FRAC_PI_2);
    simd_sin_f32(&shifted.view())
}
/// SIMD-accelerated cosine for f64 arrays
#[allow(dead_code)]
pub fn simd_cos_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let shifted: Array1<f64> = input.mapv(|x| x + std::f64::consts::FRAC_PI_2);
    simd_sin_f64(&shifted.view())
}
/// SIMD-accelerated log base 2 for f32 arrays
///
/// Computes log2(x) = ln(x) / ln(2)
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of log2(x) values
#[allow(dead_code)]
pub fn simd_log2_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let ln_result = simd_ln_f32(input);
    let inv_ln_2 = 1.0 / std::f32::consts::LN_2;
    ln_result.mapv(|x| x * inv_ln_2)
}
/// SIMD-accelerated log base 2 for f64 arrays
#[allow(dead_code)]
pub fn simd_log2_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let ln_result = simd_ln_f64(input);
    let inv_ln_2 = 1.0 / std::f64::consts::LN_2;
    ln_result.mapv(|x| x * inv_ln_2)
}
/// SIMD-accelerated log base 10 for f32 arrays
///
/// Computes log10(x) = ln(x) / ln(10)
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of log10(x) values
#[allow(dead_code)]
pub fn simd_log10_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let ln_result = simd_ln_f32(input);
    let inv_ln_10 = 1.0 / std::f32::consts::LN_10;
    ln_result.mapv(|x| x * inv_ln_10)
}
/// SIMD-accelerated log base 10 for f64 arrays
#[allow(dead_code)]
pub fn simd_log10_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let ln_result = simd_ln_f64(input);
    let inv_ln_10 = 1.0 / std::f64::consts::LN_10;
    ln_result.mapv(|x| x * inv_ln_10)
}
