//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

use super::functions_4::{simd_softplus_f32, simd_softplus_f64, simd_tanh_f32, simd_tanh_f64};

/// SIMD-accelerated Mish activation for f32 arrays
///
/// Computes Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// This combines our SIMD softplus and tanh implementations.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of Mish values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_mish_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_mish_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // Mish(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_mish_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let softplus = simd_softplus_f32(input);
    let tanh_softplus = simd_tanh_f32(&softplus.view());
    let mut result = Vec::with_capacity(len);
    for (i, &x) in input.iter().enumerate() {
        result.push(x * tanh_softplus[i]);
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated Mish for f64 arrays
#[allow(dead_code)]
pub fn simd_mish_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let softplus = simd_softplus_f64(input);
    let tanh_softplus = simd_tanh_f64(&softplus.view());
    let mut result = Vec::with_capacity(len);
    for (i, &x) in input.iter().enumerate() {
        result.push(x * tanh_softplus[i]);
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated natural logarithm for f32 arrays
///
/// Computes ln(x) using range reduction and polynomial approximation.
/// ln(x) = ln(2^k * m) = k * ln(2) + ln(m) where 1 <= m < 2
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of ln(x) values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_ln_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f32, std::f32::consts::E, 10.0];
/// let result = simd_ln_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // ln(1) = 0
/// assert!((result[1] - 1.0).abs() < 1e-5); // ln(e) = 1
/// ```
#[allow(dead_code)]
pub fn simd_ln_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let one = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(-0.5);
                let c3 = _mm256_set1_ps(1.0 / 3.0);
                let c4 = _mm256_set1_ps(-0.25);
                let c5 = _mm256_set1_ps(0.2);
                let c6 = _mm256_set1_ps(-1.0 / 6.0);
                let mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
                let exponent_bias = _mm256_set1_epi32(127);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());
                    let x_bits = _mm256_castps_si256(x);
                    let exponent = _mm256_sub_epi32(_mm256_srli_epi32(x_bits, 23), exponent_bias);
                    let k = _mm256_cvtepi32_ps(exponent);
                    let mantissa_bits = _mm256_or_si256(
                        _mm256_and_si256(x_bits, mantissa_mask),
                        _mm256_slli_epi32(exponent_bias, 23),
                    );
                    let m = _mm256_castsi256_ps(mantissa_bits);
                    let y = _mm256_sub_ps(m, one);
                    let y2 = _mm256_mul_ps(y, y);
                    let y3 = _mm256_mul_ps(y2, y);
                    let y4 = _mm256_mul_ps(y2, y2);
                    let y5 = _mm256_mul_ps(y4, y);
                    let y6 = _mm256_mul_ps(y4, y2);
                    let mut ln_m = _mm256_mul_ps(c1, y);
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c2, y2));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c3, y3));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c4, y4));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c5, y5));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c6, y6));
                    let ln_x = _mm256_add_ps(_mm256_mul_ps(k, ln_2), ln_m);
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    result.push(input[j].ln());
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
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let one = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(-0.5);
                let c3 = vdupq_n_f32(1.0 / 3.0);
                let c4 = vdupq_n_f32(-0.25);
                let c5 = vdupq_n_f32(0.2);
                let c6 = vdupq_n_f32(-1.0 / 6.0);
                let mantissa_mask = vdupq_n_s32(0x007FFFFF);
                let exponent_bias = vdupq_n_s32(127);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());
                    let x_bits = vreinterpretq_s32_f32(x);
                    let exponent = vsubq_s32(vshrq_n_s32(x_bits, 23), exponent_bias);
                    let k = vcvtq_f32_s32(exponent);
                    let mantissa_bits = vorrq_s32(
                        vandq_s32(x_bits, mantissa_mask),
                        vshlq_n_s32(exponent_bias, 23),
                    );
                    let m = vreinterpretq_f32_s32(mantissa_bits);
                    let y = vsubq_f32(m, one);
                    let y2 = vmulq_f32(y, y);
                    let y3 = vmulq_f32(y2, y);
                    let y4 = vmulq_f32(y2, y2);
                    let y5 = vmulq_f32(y4, y);
                    let y6 = vmulq_f32(y4, y2);
                    let mut ln_m = vmulq_f32(c1, y);
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c2, y2));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c3, y3));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c4, y4));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c5, y5));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c6, y6));
                    let ln_x = vaddq_f32(vmulq_f32(k, ln_2), ln_m);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].ln());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        result.push(x.ln());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated natural logarithm for f64 arrays
#[allow(dead_code)]
pub fn simd_ln_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let one = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(-0.5);
                let c3 = _mm256_set1_pd(1.0 / 3.0);
                let c4 = _mm256_set1_pd(-0.25);
                let c5 = _mm256_set1_pd(0.2);
                let c6 = _mm256_set1_pd(-1.0 / 6.0);
                let c7 = _mm256_set1_pd(1.0 / 7.0);
                let c8 = _mm256_set1_pd(-0.125);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());
                    let mut x_arr = [0.0f64; 4];
                    _mm256_storeu_pd(x_arr.as_mut_ptr(), x);
                    let mut k_arr = [0.0f64; 4];
                    let mut m_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let bits = x_arr[j].to_bits();
                        let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
                        k_arr[j] = exp as f64;
                        let mantissa_bits = (bits & 0x000FFFFFFFFFFFFF) | (1023u64 << 52);
                        m_arr[j] = f64::from_bits(mantissa_bits);
                    }
                    let k = _mm256_loadu_pd(k_arr.as_ptr());
                    let m = _mm256_loadu_pd(m_arr.as_ptr());
                    let y = _mm256_sub_pd(m, one);
                    let y2 = _mm256_mul_pd(y, y);
                    let y3 = _mm256_mul_pd(y2, y);
                    let y4 = _mm256_mul_pd(y2, y2);
                    let y5 = _mm256_mul_pd(y4, y);
                    let y6 = _mm256_mul_pd(y4, y2);
                    let y7 = _mm256_mul_pd(y6, y);
                    let y8 = _mm256_mul_pd(y4, y4);
                    let mut ln_m = _mm256_mul_pd(c1, y);
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c2, y2));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c3, y3));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c4, y4));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c5, y5));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c6, y6));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c7, y7));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c8, y8));
                    let ln_x = _mm256_add_pd(_mm256_mul_pd(k, ln_2), ln_m);
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].ln());
                }
                return Array1::from_vec(result);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            for &x in input.iter() {
                result.push(x.ln());
            }
            return Array1::from_vec(result);
        }
    }
    for &x in input.iter() {
        result.push(x.ln());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated sine for f32 arrays
///
/// Computes sin(x) using range reduction and polynomial approximation.
/// Range reduction: reduce x to [-π/2, π/2] using periodicity.
/// Then use Taylor series: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
///
/// # Arguments
/// * `input` - Input array (radians)
///
/// # Returns
/// * Array of sin(x) values in range [-1, 1]
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_sin_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
/// let result = simd_sin_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // sin(0) = 0
/// assert!((result[1] - 1.0).abs() < 1e-5); // sin(π/2) = 1
/// ```
#[allow(dead_code)]
pub fn simd_sin_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
    const PI: f32 = std::f32::consts::PI;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let pi = _mm256_set1_ps(PI);
                let two_pi = _mm256_set1_ps(2.0 * PI);
                let pi_over_2 = _mm256_set1_ps(PI / 2.0);
                let one = _mm256_set1_ps(1.0);
                let neg_one = _mm256_set1_ps(-1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c3 = _mm256_set1_ps(-1.0 / 6.0);
                let c5 = _mm256_set1_ps(1.0 / 120.0);
                let c7 = _mm256_set1_ps(-1.0 / 5040.0);
                let c9 = _mm256_set1_ps(1.0 / 362880.0);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());
                    let k = _mm256_round_ps(
                        _mm256_mul_ps(x, _mm256_set1_ps(FRAC_1_PI / 2.0)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    x = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));
                    let mask_gt_pi_2 = _mm256_cmp_ps(x, pi_over_2, _CMP_GT_OQ);
                    let mask_lt_neg_pi_2 =
                        _mm256_cmp_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), pi_over_2), _CMP_LT_OQ);
                    let x_reduced_high = _mm256_sub_ps(pi, x);
                    x = _mm256_blendv_ps(x, x_reduced_high, mask_gt_pi_2);
                    let x_reduced_low = _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), pi), x);
                    x = _mm256_blendv_ps(x, x_reduced_low, mask_lt_neg_pi_2);
                    let x2 = _mm256_mul_ps(x, x);
                    let x3 = _mm256_mul_ps(x2, x);
                    let x5 = _mm256_mul_ps(x3, x2);
                    let x7 = _mm256_mul_ps(x5, x2);
                    let x9 = _mm256_mul_ps(x7, x2);
                    let mut sin_x = _mm256_mul_ps(c1, x);
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c3, x3));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c5, x5));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c7, x7));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c9, x9));
                    sin_x = _mm256_max_ps(sin_x, neg_one);
                    sin_x = _mm256_min_ps(sin_x, one);
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 8;
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
                let pi = vdupq_n_f32(PI);
                let two_pi = vdupq_n_f32(2.0 * PI);
                let pi_over_2 = vdupq_n_f32(PI / 2.0);
                let neg_pi_over_2 = vdupq_n_f32(-PI / 2.0);
                let zero = vdupq_n_f32(0.0);
                let one = vdupq_n_f32(1.0);
                let neg_one = vdupq_n_f32(-1.0);
                let c1 = vdupq_n_f32(1.0);
                let c3 = vdupq_n_f32(-1.0 / 6.0);
                let c5 = vdupq_n_f32(1.0 / 120.0);
                let c7 = vdupq_n_f32(-1.0 / 5040.0);
                let c9 = vdupq_n_f32(1.0 / 362880.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());
                    let k = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(FRAC_1_PI / 2.0)));
                    x = vsubq_f32(x, vmulq_f32(k, two_pi));
                    let mask_gt_pi_2 = vcgtq_f32(x, pi_over_2);
                    let mask_lt_neg_pi_2 = vcltq_f32(x, neg_pi_over_2);
                    let x_reduced_high = vsubq_f32(pi, x);
                    x = vbslq_f32(mask_gt_pi_2, x_reduced_high, x);
                    let x_reduced_low = vsubq_f32(vsubq_f32(zero, pi), x);
                    x = vbslq_f32(mask_lt_neg_pi_2, x_reduced_low, x);
                    let x2 = vmulq_f32(x, x);
                    let x3 = vmulq_f32(x2, x);
                    let x5 = vmulq_f32(x3, x2);
                    let x7 = vmulq_f32(x5, x2);
                    let x9 = vmulq_f32(x7, x2);
                    let mut sin_x = vmulq_f32(c1, x);
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c3, x3));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c5, x5));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c7, x7));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c9, x9));
                    sin_x = vmaxq_f32(sin_x, neg_one);
                    sin_x = vminq_f32(sin_x, one);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sin_x);
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
    for &x in input.iter() {
        result.push(x.sin());
    }
    Array1::from_vec(result)
}
