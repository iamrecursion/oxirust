//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated exp for f32 arrays using polynomial approximation
///
/// Uses range reduction and a degree-6 polynomial approximation.
/// Achieves ~10^-7 relative error for most inputs.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of exp(x) values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_exp_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0, 2.0];
/// let result = simd_exp_f32(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
/// ```
#[allow(dead_code)]
pub fn simd_exp_f32(input: &ArrayView1<f32>) -> Array1<f32> {
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
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let half = _mm256_set1_ps(0.5);
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());
                    x = _mm256_max_ps(x, min_val);
                    x = _mm256_min_ps(x, max_val);
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(x, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(x, _mm256_mul_ps(k_float, ln_2));
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
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    result.push(input[j].exp());
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
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());
                    x = vmaxq_f32(x, min_val);
                    x = vminq_f32(x, max_val);
                    let k_float = vrndnq_f32(vmulq_f32(x, log2_e));
                    let r = vsubq_f32(x, vmulq_f32(k_float, ln_2));
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
                    let exp_x = vmulq_f32(exp_r, scale);
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].exp());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated exp for f64 arrays using polynomial approximation
///
/// Uses range reduction and a higher-degree polynomial for double precision.
/// Achieves ~10^-15 relative error for most inputs.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of exp(x) values
#[allow(dead_code)]
pub fn simd_exp_f64(input: &ArrayView1<f64>) -> Array1<f64> {
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
                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let c0 = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(0.5);
                let c3 = _mm256_set1_pd(1.0 / 6.0);
                let c4 = _mm256_set1_pd(1.0 / 24.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c6 = _mm256_set1_pd(1.0 / 720.0);
                let c7 = _mm256_set1_pd(1.0 / 5040.0);
                let c8 = _mm256_set1_pd(1.0 / 40320.0);
                let c9 = _mm256_set1_pd(1.0 / 362880.0);
                let c10 = _mm256_set1_pd(1.0 / 3628800.0);
                let max_val = _mm256_set1_pd(709.0);
                let min_val = _mm256_set1_pd(-709.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let mut x = _mm256_loadu_pd(slice.as_ptr());
                    x = _mm256_max_pd(x, min_val);
                    x = _mm256_min_pd(x, max_val);
                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(x, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(x, _mm256_mul_pd(k_float, ln_2));
                    let r2 = _mm256_mul_pd(r, r);
                    let r4 = _mm256_mul_pd(r2, r2);
                    let r6 = _mm256_mul_pd(r4, r2);
                    let r8 = _mm256_mul_pd(r4, r4);
                    let r10 = _mm256_mul_pd(r8, r2);
                    let mut even = c0;
                    even = _mm256_add_pd(even, _mm256_mul_pd(c2, r2));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c4, r4));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c6, r6));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c8, r8));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c10, r10));
                    let r3 = _mm256_mul_pd(r2, r);
                    let r5 = _mm256_mul_pd(r4, r);
                    let r7 = _mm256_mul_pd(r6, r);
                    let r9 = _mm256_mul_pd(r8, r);
                    let mut odd = _mm256_mul_pd(c1, r);
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c3, r3));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c5, r5));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c7, r7));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c9, r9));
                    let exp_r = _mm256_add_pd(even, odd);
                    let k_long = _mm256_cvtpd_epi32(k_float);
                    let mut k_arr = [0i32; 4];
                    _mm_storeu_si128(k_arr.as_mut_ptr() as *mut __m128i, k_long);
                    let mut scale_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let exp_bits = ((k_arr[j] as i64 + 1023) << 52) as u64;
                        scale_arr[j] = f64::from_bits(exp_bits);
                    }
                    let scale = _mm256_loadu_pd(scale_arr.as_ptr());
                    let exp_x = _mm256_mul_pd(exp_r, scale);
                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].exp());
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
                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);
                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c6 = vdupq_n_f64(1.0 / 720.0);
                let c7 = vdupq_n_f64(1.0 / 5040.0);
                let c8 = vdupq_n_f64(1.0 / 40320.0);
                let max_val = vdupq_n_f64(709.0);
                let min_val = vdupq_n_f64(-709.0);
                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let mut x = vld1q_f64(slice.as_ptr());
                    x = vmaxq_f64(x, min_val);
                    x = vminq_f64(x, max_val);
                    let k_float = vrndnq_f64(vmulq_f64(x, log2_e));
                    let r = vsubq_f64(x, vmulq_f64(k_float, ln_2));
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
                    let exp_x = vmulq_f64(exp_r, scale);
                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 2;
                }
                for j in i..len {
                    result.push(input[j].exp());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}
/// SIMD-accelerated fast exp for f32 (lower precision but faster)
///
/// Uses a simpler polynomial approximation for maximum speed.
/// Suitable for neural network inference where ~10^-4 accuracy is acceptable.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of approximate exp(x) values
#[allow(dead_code)]
pub fn simd_exp_fast_f32(input: &ArrayView1<f32>) -> Array1<f32> {
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
                let scale = _mm256_set1_ps(12102203.0);
                let offset = _mm256_set1_ps(1065353216.0 - 486411.0);
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);
                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());
                    x = _mm256_max_ps(x, min_val);
                    x = _mm256_min_ps(x, max_val);
                    let approx = _mm256_add_ps(_mm256_mul_ps(x, scale), offset);
                    let exp_x = _mm256_castsi256_ps(_mm256_cvtps_epi32(approx));
                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }
                for j in i..len {
                    result.push(input[j].exp());
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
                let scale = vdupq_n_f32(12102203.0);
                let offset = vdupq_n_f32(1065353216.0 - 486411.0);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);
                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());
                    x = vmaxq_f32(x, min_val);
                    x = vminq_f32(x, max_val);
                    let approx = vaddq_f32(vmulq_f32(x, scale), offset);
                    let exp_x = vreinterpretq_f32_s32(vcvtq_s32_f32(approx));
                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }
                for j in i..len {
                    result.push(input[j].exp());
                }
                return Array1::from_vec(result);
            }
        }
    }
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}
