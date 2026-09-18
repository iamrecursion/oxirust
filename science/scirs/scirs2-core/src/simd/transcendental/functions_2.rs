//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated sigmoid (logistic) function for f32 arrays
///
/// Computes σ(x) = 1 / (1 + exp(-x)) with numerical stability.
/// Uses SIMD exp approximation for significant speedup.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of sigmoid values in range (0, 1)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_sigmoid_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 2.0, -2.0];
/// let result = simd_sigmoid_f32(&x.view());
/// assert!((result[0] - 0.5).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn simd_sigmoid_f32(input: &ArrayView1<f32>) -> Array1<f32> {
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
                let zero = _mm256_setzero_ps();
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);
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
                    let neg_x = _mm256_sub_ps(zero, x);
                    let exp_arg = _mm256_blendv_ps(x, neg_x, mask_positive);
                    let exp_arg_clamped = _mm256_max_ps(_mm256_min_ps(exp_arg, max_val), min_val);
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(exp_arg_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(exp_arg_clamped, _mm256_mul_ps(k_float, ln_2));
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
                    let exp_val = _mm256_mul_ps(exp_r, scale);
                    let one_plus_exp = _mm256_add_ps(one, exp_val);
                    let sig_positive = _mm256_div_ps(one, one_plus_exp);
                    let sig_negative = _mm256_div_ps(exp_val, one_plus_exp);
                    let sigmoid_result =
                        _mm256_blendv_ps(sig_negative, sig_positive, mask_positive);
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
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
                let zero = vdupq_n_f32(0.0);
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);
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
                    let neg_x = vsubq_f32(zero, x);
                    let exp_arg = vbslq_f32(mask_positive, neg_x, x);
                    let exp_arg_clamped = vmaxq_f32(vminq_f32(exp_arg, max_val), min_val);
                    let k_float = vrndnq_f32(vmulq_f32(exp_arg_clamped, log2_e));
                    let r = vsubq_f32(exp_arg_clamped, vmulq_f32(k_float, ln_2));
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
                    let exp_val = vmulq_f32(exp_r, scale);
                    let one_plus_exp = vaddq_f32(one, exp_val);
                    let sig_positive = vdivq_f32(one, one_plus_exp);
                    let sig_negative = vdivq_f32(exp_val, one_plus_exp);
                    let sigmoid_result = vbslq_f32(mask_positive, sig_positive, sig_negative);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        if x >= 0.0 {
            result.push(1.0 / (1.0 + (-x).exp()));
        } else {
            let exp_x = x.exp();
            result.push(exp_x / (1.0 + exp_x));
        }
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated sigmoid for f64 arrays
#[allow(dead_code)]
pub fn simd_sigmoid_f64(input: &ArrayView1<f64>) -> Array1<f64> {
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
                let zero = _mm256_setzero_pd();
                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let max_val = _mm256_set1_pd(709.0);
                let min_val = _mm256_set1_pd(-709.0);
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
                    let neg_x = _mm256_sub_pd(zero, x);
                    let exp_arg = _mm256_blendv_pd(x, neg_x, mask_positive);
                    let exp_arg_clamped = _mm256_max_pd(_mm256_min_pd(exp_arg, max_val), min_val);
                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(exp_arg_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(exp_arg_clamped, _mm256_mul_pd(k_float, ln_2));
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
                    let exp_val = _mm256_mul_pd(exp_r, scale);
                    let one_plus_exp = _mm256_add_pd(one, exp_val);
                    let sig_positive = _mm256_div_pd(one, one_plus_exp);
                    let sig_negative = _mm256_div_pd(exp_val, one_plus_exp);
                    let sigmoid_result =
                        _mm256_blendv_pd(sig_negative, sig_positive, mask_positive);
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
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
                let zero = vdupq_n_f64(0.0);
                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);
                let max_val = vdupq_n_f64(709.0);
                let min_val = vdupq_n_f64(-709.0);
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
                    let neg_x = vsubq_f64(zero, x);
                    let exp_arg = vbslq_f64(mask_positive, neg_x, x);
                    let exp_arg_clamped = vmaxq_f64(vminq_f64(exp_arg, max_val), min_val);
                    let k_float = vrndnq_f64(vmulq_f64(exp_arg_clamped, log2_e));
                    let r = vsubq_f64(exp_arg_clamped, vmulq_f64(k_float, ln_2));
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
                    let exp_val = vmulq_f64(exp_r, scale);
                    let one_plus_exp = vaddq_f64(one, exp_val);
                    let sig_positive = vdivq_f64(one, one_plus_exp);
                    let sig_negative = vdivq_f64(exp_val, one_plus_exp);
                    let sigmoid_result = vbslq_f64(mask_positive, sig_positive, sig_negative);
                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 2;
                }
                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        if x >= 0.0 {
            result.push(1.0 / (1.0 + (-x).exp()));
        } else {
            let exp_x = x.exp();
            result.push(exp_x / (1.0 + exp_x));
        }
    }
    Array1::from_vec(result)
}
