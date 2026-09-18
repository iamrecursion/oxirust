//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

use super::functions_2::simd_sigmoid_f64;

/// SIMD-accelerated GELU (Gaussian Error Linear Unit) for f32 arrays
///
/// Uses the fast tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// This is the approximation used in GPT-2 and BERT models.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of GELU values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_gelu_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_gelu_f32(&x.view());
/// assert!((result[0]).abs() < 1e-6); // GELU(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_gelu_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    const COEFF: f32 = 0.044715;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let half = _mm256_set1_ps(0.5);
                let one = _mm256_set1_ps(1.0);
                let sqrt_2_pi = _mm256_set1_ps(SQRT_2_OVER_PI);
                let coeff = _mm256_set1_ps(COEFF);
                let c27 = _mm256_set1_ps(27.0);
                let c9 = _mm256_set1_ps(9.0);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());
                    let x2 = _mm256_mul_ps(x, x);
                    let x3 = _mm256_mul_ps(x2, x);
                    let inner =
                        _mm256_mul_ps(sqrt_2_pi, _mm256_add_ps(x, _mm256_mul_ps(coeff, x3)));
                    let inner2 = _mm256_mul_ps(inner, inner);
                    let num = _mm256_mul_ps(inner, _mm256_add_ps(c27, inner2));
                    let den = _mm256_add_ps(c27, _mm256_mul_ps(c9, inner2));
                    let tanh_inner = _mm256_div_ps(num, den);
                    let neg_one = _mm256_set1_ps(-1.0);
                    let tanh_clamped = _mm256_max_ps(neg_one, _mm256_min_ps(one, tanh_inner));
                    let gelu =
                        _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_clamped)));
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
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
                let half = vdupq_n_f32(0.5);
                let one = vdupq_n_f32(1.0);
                let neg_one = vdupq_n_f32(-1.0);
                let sqrt_2_pi = vdupq_n_f32(SQRT_2_OVER_PI);
                let coeff = vdupq_n_f32(COEFF);
                let c27 = vdupq_n_f32(27.0);
                let c9 = vdupq_n_f32(9.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());
                    let x2 = vmulq_f32(x, x);
                    let x3 = vmulq_f32(x2, x);
                    let inner = vmulq_f32(sqrt_2_pi, vaddq_f32(x, vmulq_f32(coeff, x3)));
                    let inner2 = vmulq_f32(inner, inner);
                    let num = vmulq_f32(inner, vaddq_f32(c27, inner2));
                    let den = vaddq_f32(c27, vmulq_f32(c9, inner2));
                    let tanh_inner = vdivq_f32(num, den);
                    let tanh_clamped = vmaxq_f32(neg_one, vminq_f32(one, tanh_inner));
                    let gelu = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_clamped)));
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        let tanh_val = inner.tanh();
        result.push(0.5 * x * (1.0 + tanh_val));
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated GELU for f64 arrays
#[allow(dead_code)]
pub fn simd_gelu_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let mut result = Vec::with_capacity(len);
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const COEFF: f64 = 0.044715;
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let half = _mm256_set1_pd(0.5);
                let one = _mm256_set1_pd(1.0);
                let neg_one = _mm256_set1_pd(-1.0);
                let sqrt_2_pi = _mm256_set1_pd(SQRT_2_OVER_PI);
                let coeff = _mm256_set1_pd(COEFF);
                let c27 = _mm256_set1_pd(27.0);
                let c9 = _mm256_set1_pd(9.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());
                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);
                    let inner =
                        _mm256_mul_pd(sqrt_2_pi, _mm256_add_pd(x, _mm256_mul_pd(coeff, x3)));
                    let inner2 = _mm256_mul_pd(inner, inner);
                    let num = _mm256_mul_pd(inner, _mm256_add_pd(c27, inner2));
                    let den = _mm256_add_pd(c27, _mm256_mul_pd(c9, inner2));
                    let tanh_inner = _mm256_div_pd(num, den);
                    let tanh_clamped = _mm256_max_pd(neg_one, _mm256_min_pd(one, tanh_inner));
                    let gelu =
                        _mm256_mul_pd(half, _mm256_mul_pd(x, _mm256_add_pd(one, tanh_clamped)));
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
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
                let half = vdupq_n_f64(0.5);
                let one = vdupq_n_f64(1.0);
                let neg_one = vdupq_n_f64(-1.0);
                let sqrt_2_pi = vdupq_n_f64(SQRT_2_OVER_PI);
                let coeff = vdupq_n_f64(COEFF);
                let c27 = vdupq_n_f64(27.0);
                let c9 = vdupq_n_f64(9.0);
                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let x = vld1q_f64(slice.as_ptr());
                    let x2 = vmulq_f64(x, x);
                    let x3 = vmulq_f64(x2, x);
                    let inner = vmulq_f64(sqrt_2_pi, vaddq_f64(x, vmulq_f64(coeff, x3)));
                    let inner2 = vmulq_f64(inner, inner);
                    let num = vmulq_f64(inner, vaddq_f64(c27, inner2));
                    let den = vaddq_f64(c27, vmulq_f64(c9, inner2));
                    let tanh_inner = vdivq_f64(num, den);
                    let tanh_clamped = vmaxq_f64(neg_one, vminq_f64(one, tanh_inner));
                    let gelu = vmulq_f64(half, vmulq_f64(x, vaddq_f64(one, tanh_clamped)));
                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 2;
                }
                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        let tanh_val = inner.tanh();
        result.push(0.5 * x * (1.0 + tanh_val));
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated Swish (SiLU) activation for f32 arrays
///
/// Computes swish(x) = x * sigmoid(x)
/// This is also known as SiLU (Sigmoid Linear Unit).
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of swish values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_swish_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_swish_f32(&x.view());
/// assert!((result[0]).abs() < 1e-6); // swish(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_swish_f32(input: &ArrayView1<f32>) -> Array1<f32> {
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
                    let sigmoid = _mm256_blendv_ps(sig_negative, sig_positive, mask_positive);
                    let swish = _mm256_mul_ps(x, sigmoid);
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), swish);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    let x = input[j];
                    let sigmoid = if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let exp_x = x.exp();
                        exp_x / (1.0 + exp_x)
                    };
                    result.push(x * sigmoid);
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
                    let sigmoid = vbslq_f32(mask_positive, sig_positive, sig_negative);
                    let swish = vmulq_f32(x, sigmoid);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), swish);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    let x = input[j];
                    let sigmoid = if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let exp_x = x.exp();
                        exp_x / (1.0 + exp_x)
                    };
                    result.push(x * sigmoid);
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &x in input.iter() {
        let sigmoid = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        };
        result.push(x * sigmoid);
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated Swish for f64 arrays
#[allow(dead_code)]
pub fn simd_swish_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }
    let sigmoid = simd_sigmoid_f64(input);
    let mut result = Vec::with_capacity(len);
    for (i, &x) in input.iter().enumerate() {
        result.push(x * sigmoid[i]);
    }
    Array1::from_vec(result)
}
