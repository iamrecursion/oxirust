//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

use super::functions::{simd_exp_f32, simd_exp_f64};

/// SIMD-accelerated Softplus activation for f32 arrays
///
/// Computes softplus(x) = ln(1 + exp(x)) with numerical stability.
/// For large x, softplus(x) ≈ x to avoid overflow.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of softplus values (always positive)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_softplus_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_softplus_f32(&x.view());
/// assert!((result[0] - 0.6931472).abs() < 1e-5); // softplus(0) = ln(2)
/// ```
#[allow(dead_code)]
pub fn simd_softplus_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const THRESHOLD: f32 = 20.0;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let one = _mm256_set1_ps(1.0);
                let threshold = _mm256_set1_ps(THRESHOLD);
                let neg_threshold = _mm256_set1_ps(-THRESHOLD);
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_exp = _mm256_set1_ps(88.0);
                let min_exp = _mm256_set1_ps(-88.0);
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());
                    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, max_exp), min_exp);
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(x_clamped, _mm256_mul_ps(k_float, ln_2));
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);
                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);
                    let exp_x = _mm256_mul_ps(exp_r, scale);
                    let one_plus_exp = _mm256_add_ps(one, exp_x);
                    let mut temp_softplus = [0.0f32; 8];
                    let mut temp_x = [0.0f32; 8];
                    let mut temp_one_plus_exp = [0.0f32; 8];
                    _mm256_storeu_ps(temp_x.as_mut_ptr(), x);
                    _mm256_storeu_ps(temp_one_plus_exp.as_mut_ptr(), one_plus_exp);
                    for j in 0..8 {
                        let xj = temp_x[j];
                        if xj > THRESHOLD {
                            temp_softplus[j] = xj;
                        } else if xj < -THRESHOLD {
                            temp_softplus[j] = temp_one_plus_exp[j] - 1.0;
                        } else {
                            temp_softplus[j] = temp_one_plus_exp[j].ln();
                        }
                    }
                    result.extend_from_slice(&temp_softplus);
                    i += 8;
                }
                for j in i..len {
                    let x = input[j];
                    if x > THRESHOLD {
                        result.push(x);
                    } else if x < -THRESHOLD {
                        result.push(x.exp());
                    } else {
                        result.push((1.0 + x.exp()).ln());
                    }
                }
                return Array1::from_vec(result);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            let exp_vals = simd_exp_f32(input);
            for (i, &x) in input.iter().enumerate() {
                if x > THRESHOLD {
                    result.push(x);
                } else if x < -THRESHOLD {
                    result.push(exp_vals[i]);
                } else {
                    result.push((1.0f32 + exp_vals[i]).ln());
                }
            }
            return Array1::from_vec(result);
        }
    }
    for &x in input.iter() {
        if x > THRESHOLD {
            result.push(x);
        } else if x < -THRESHOLD {
            result.push(x.exp());
        } else {
            result.push((1.0f32 + x.exp()).ln());
        }
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated Softplus for f64 arrays
#[allow(dead_code)]
pub fn simd_softplus_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const THRESHOLD: f64 = 40.0;
    let exp_vals = simd_exp_f64(input);
    for (i, &x) in input.iter().enumerate() {
        if x > THRESHOLD {
            result.push(x);
        } else if x < -THRESHOLD {
            result.push(exp_vals[i]);
        } else {
            result.push((1.0f64 + exp_vals[i]).ln());
        }
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated tanh for f32 arrays
///
/// Computes tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) with numerical stability.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of tanh values in range (-1, 1)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_tanh_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_tanh_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // tanh(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_tanh_f32(input: &ArrayView1<f32>) -> Array1<f32> {
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
                let one = _mm256_set1_ps(1.0);
                let two = _mm256_set1_ps(2.0);
                let zero = _mm256_setzero_ps();
                let neg_one = _mm256_set1_ps(-1.0);
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_val = _mm256_set1_ps(44.0);
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());
                    let mask_positive = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
                    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
                    let two_abs_x = _mm256_mul_ps(two, abs_x);
                    let two_abs_x_clamped = _mm256_min_ps(two_abs_x, max_val);
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(two_abs_x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(two_abs_x_clamped, _mm256_mul_ps(k_float, ln_2));
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);
                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);
                    let exp_2abs_x = _mm256_mul_ps(exp_r, scale);
                    let exp_plus_one = _mm256_add_ps(exp_2abs_x, one);
                    let abs_tanh = _mm256_sub_ps(one, _mm256_div_ps(two, exp_plus_one));
                    let tanh_result =
                        _mm256_blendv_ps(_mm256_mul_ps(neg_one, abs_tanh), abs_tanh, mask_positive);
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    result.push(input[j].tanh());
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
                let one = vdupq_n_f32(1.0);
                let two = vdupq_n_f32(2.0);
                let zero = vdupq_n_f32(0.0);
                let neg_one = vdupq_n_f32(-1.0);
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let max_val = vdupq_n_f32(44.0);
                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());
                    let mask_positive = vcgeq_f32(x, zero);
                    let abs_x = vabsq_f32(x);
                    let two_abs_x = vmulq_f32(two, abs_x);
                    let two_abs_x_clamped = vminq_f32(two_abs_x, max_val);
                    let k_float = vrndnq_f32(vmulq_f32(two_abs_x_clamped, log2_e));
                    let r = vsubq_f32(two_abs_x_clamped, vmulq_f32(k_float, ln_2));
                    let r2 = vmulq_f32(r, r);
                    let r3 = vmulq_f32(r2, r);
                    let r4 = vmulq_f32(r2, r2);
                    let r5 = vmulq_f32(r4, r);
                    let r6 = vmulq_f32(r4, r2);
                    let mut exp_r = c0;
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c1, r));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c5, r5));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c6, r6));
                    let k_int = vcvtq_s32_f32(k_float);
                    let bias = vdupq_n_s32(127);
                    let exp_bits = vshlq_n_s32(vaddq_s32(k_int, bias), 23);
                    let scale = vreinterpretq_f32_s32(exp_bits);
                    let exp_2abs_x = vmulq_f32(exp_r, scale);
                    let exp_plus_one = vaddq_f32(exp_2abs_x, one);
                    let abs_tanh = vsubq_f32(one, vdivq_f32(two, exp_plus_one));
                    let neg_abs_tanh = vmulq_f32(neg_one, abs_tanh);
                    let tanh_result = vbslq_f32(mask_positive, abs_tanh, neg_abs_tanh);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].tanh());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        result.push(x.tanh());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated tanh for f64 arrays
#[allow(dead_code)]
pub fn simd_tanh_f64(input: &ArrayView1<f64>) -> Array1<f64> {
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
                let one = _mm256_set1_pd(1.0);
                let two = _mm256_set1_pd(2.0);
                let zero = _mm256_setzero_pd();
                let neg_one = _mm256_set1_pd(-1.0);
                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let max_val = _mm256_set1_pd(354.5);
                let c0 = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(0.5);
                let c3 = _mm256_set1_pd(1.0 / 6.0);
                let c4 = _mm256_set1_pd(1.0 / 24.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c6 = _mm256_set1_pd(1.0 / 720.0);
                let c7 = _mm256_set1_pd(1.0 / 5040.0);
                let c8 = _mm256_set1_pd(1.0 / 40320.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());
                    let mask_positive = _mm256_cmp_pd(x, zero, _CMP_GE_OQ);
                    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
                    let two_abs_x = _mm256_mul_pd(two, abs_x);
                    let two_abs_x_clamped = _mm256_min_pd(two_abs_x, max_val);
                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(two_abs_x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(two_abs_x_clamped, _mm256_mul_pd(k_float, ln_2));
                    let r2 = _mm256_mul_pd(r, r);
                    let r3 = _mm256_mul_pd(r2, r);
                    let r4 = _mm256_mul_pd(r2, r2);
                    let r5 = _mm256_mul_pd(r4, r);
                    let r6 = _mm256_mul_pd(r4, r2);
                    let r7 = _mm256_mul_pd(r6, r);
                    let r8 = _mm256_mul_pd(r4, r4);
                    let mut exp_r = c0;
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c1, r));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c2, r2));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c3, r3));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c4, r4));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c5, r5));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c6, r6));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c7, r7));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c8, r8));
                    let k_long = _mm256_cvtpd_epi32(k_float);
                    let mut k_arr = [0i32; 4];
                    _mm_storeu_si128(k_arr.as_mut_ptr() as *mut __m128i, k_long);
                    let mut scale_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let exp_bits = ((k_arr[j] as i64 + 1023) << 52) as u64;
                        scale_arr[j] = f64::from_bits(exp_bits);
                    }
                    let scale = _mm256_loadu_pd(scale_arr.as_ptr());
                    let exp_2abs_x = _mm256_mul_pd(exp_r, scale);
                    let exp_plus_one = _mm256_add_pd(exp_2abs_x, one);
                    let abs_tanh = _mm256_sub_pd(one, _mm256_div_pd(two, exp_plus_one));
                    let tanh_result =
                        _mm256_blendv_pd(_mm256_mul_pd(neg_one, abs_tanh), abs_tanh, mask_positive);
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].tanh());
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
                let one = vdupq_n_f64(1.0);
                let two = vdupq_n_f64(2.0);
                let zero = vdupq_n_f64(0.0);
                let neg_one = vdupq_n_f64(-1.0);
                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);
                let max_val = vdupq_n_f64(354.5);
                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c6 = vdupq_n_f64(1.0 / 720.0);
                let c7 = vdupq_n_f64(1.0 / 5040.0);
                let c8 = vdupq_n_f64(1.0 / 40320.0);
                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let x = vld1q_f64(slice.as_ptr());
                    let mask_positive = vcgeq_f64(x, zero);
                    let abs_x = vabsq_f64(x);
                    let two_abs_x = vmulq_f64(two, abs_x);
                    let two_abs_x_clamped = vminq_f64(two_abs_x, max_val);
                    let k_float = vrndnq_f64(vmulq_f64(two_abs_x_clamped, log2_e));
                    let r = vsubq_f64(two_abs_x_clamped, vmulq_f64(k_float, ln_2));
                    let r2 = vmulq_f64(r, r);
                    let r3 = vmulq_f64(r2, r);
                    let r4 = vmulq_f64(r2, r2);
                    let r5 = vmulq_f64(r4, r);
                    let r6 = vmulq_f64(r4, r2);
                    let r7 = vmulq_f64(r6, r);
                    let r8 = vmulq_f64(r4, r4);
                    let mut exp_r = c0;
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c1, r));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c2, r2));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c3, r3));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c4, r4));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c5, r5));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c6, r6));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c7, r7));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c8, r8));
                    let k_int = vcvtq_s64_f64(k_float);
                    let bias = vdupq_n_s64(1023);
                    let exp_bits = vshlq_n_s64(vaddq_s64(k_int, bias), 52);
                    let scale = vreinterpretq_f64_s64(exp_bits);
                    let exp_2abs_x = vmulq_f64(exp_r, scale);
                    let exp_plus_one = vaddq_f64(exp_2abs_x, one);
                    let abs_tanh = vsubq_f64(one, vdivq_f64(two, exp_plus_one));
                    let neg_abs_tanh = vmulq_f64(neg_one, abs_tanh);
                    let tanh_result = vbslq_f64(mask_positive, abs_tanh, neg_abs_tanh);
                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 2;
                }
                for j in i..len {
                    result.push(input[j].tanh());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        result.push(x.tanh());
    }
    Array1::from_vec(result)
}
