//! Basic NEON arithmetic operations
//!
//! Optimized implementations of fundamental operations using ARM NEON intrinsics.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized vector addition for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_add_f32_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    // Process 4 elements at a time using NEON
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vr = vaddq_f32(va, vb);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// NEON-optimized vector addition for f32 (safe wrapper)
pub fn neon_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_add_f32_impl(a, b, out) }
        } else {
            fallback_add_f32(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_add_f32(a, b, out)
    }
}

/// NEON-optimized vector addition for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_add_f64_impl(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    // Process 2 elements at a time using NEON
    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vb = vld1q_f64(b.as_ptr().add(i));
        let vr = vaddq_f64(va, vb);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    // Handle remaining elements
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// NEON-optimized vector addition for f64 (safe wrapper)
pub fn neon_add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_add_f64_impl(a, b, out) }
        } else {
            fallback_add_f64(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_add_f64(a, b, out)
    }
}

/// NEON-optimized vector multiplication for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_mul_f32_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vr = vmulq_f32(va, vb);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

pub fn neon_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_mul_f32_impl(a, b, out) }
        } else {
            fallback_mul_f32(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_mul_f32(a, b, out)
    }
}

/// NEON-optimized vector multiplication for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_mul_f64_impl(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vb = vld1q_f64(b.as_ptr().add(i));
        let vr = vmulq_f64(va, vb);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    while i < len {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

pub fn neon_mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_mul_f64_impl(a, b, out) }
        } else {
            fallback_mul_f64(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_mul_f64(a, b, out)
    }
}

/// NEON-optimized vector subtraction for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sub_f32_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vr = vsubq_f32(va, vb);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

pub fn neon_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_sub_f32_impl(a, b, out) }
        } else {
            fallback_sub_f32(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_sub_f32(a, b, out)
    }
}

/// NEON-optimized vector subtraction for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sub_f64_impl(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vb = vld1q_f64(b.as_ptr().add(i));
        let vr = vsubq_f64(va, vb);
        vst1q_f64(out.as_mut_ptr().add(i), vr);
        i += 2;
    }

    while i < len {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

pub fn neon_sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_sub_f64_impl(a, b, out) }
        } else {
            fallback_sub_f64(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_sub_f64(a, b, out)
    }
}

/// NEON-optimized vector division for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a`, `b`, and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_div_f32_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let vr = vdivq_f32(va, vb);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = a[i] / b[i];
        i += 1;
    }
}

pub fn neon_div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_div_f32_impl(a, b, out) }
        } else {
            fallback_div_f32(a, b, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_div_f32(a, b, out)
    }
}

/// NEON-optimized dot product for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a` and `b` slices must be valid for reading.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_f32_impl(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 elements at a time
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, va, vb); // Fused multiply-add
        i += 4;
    }

    // Horizontal reduction
    let mut result = vaddvq_f32(sum);

    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

pub fn neon_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_dot_f32_impl(a, b) }
        } else {
            fallback_dot_f32(a, b)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_dot_f32(a, b)
    }
}

/// NEON-optimized dot product for f64
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `a` and `b` slices must be valid for reading.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_dot_f64_impl(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut sum = vdupq_n_f64(0.0);

    // Process 2 elements at a time
    while i + 2 <= len {
        let va = vld1q_f64(a.as_ptr().add(i));
        let vb = vld1q_f64(b.as_ptr().add(i));
        sum = vfmaq_f64(sum, va, vb);
        i += 2;
    }

    // Horizontal reduction
    let mut result = vaddvq_f64(sum);

    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

pub fn neon_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_dot_f64_impl(a, b) }
        } else {
            fallback_dot_f64(a, b)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_dot_f64(a, b)
    }
}

// Scalar fallback implementations
fn fallback_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] + b[i];
    }
}

fn fallback_add_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] + b[i];
    }
}

fn fallback_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] * b[i];
    }
}

fn fallback_mul_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] * b[i];
    }
}

fn fallback_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] - b[i];
    }
}

fn fallback_sub_f64(a: &[f64], b: &[f64], out: &mut [f64]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] - b[i];
    }
}

fn fallback_div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] / b[i];
    }
}

fn fallback_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

fn fallback_dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_add_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 5];

        neon_add_f32(&a, &b, &mut out);

        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_neon_mul_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut out = vec![0.0; 4];

        neon_mul_f32(&a, &b, &mut out);

        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_neon_dot_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = neon_dot_f32(&a, &b);

        assert!((result - 10.0).abs() < 1e-6);
    }
}
