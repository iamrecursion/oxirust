//! NEON-optimized activation functions for neural networks
//!
//! Mobile-optimized implementations of common activation functions using ARM NEON.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON-optimized ReLU activation for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `x` and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32_impl(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    let mut i = 0;
    let zero = vdupq_n_f32(0.0);

    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vr = vmaxq_f32(vx, zero); // max(x, 0)
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = x[i].max(0.0);
        i += 1;
    }
}

pub fn neon_relu_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_relu_f32_impl(x, out) }
        } else {
            fallback_relu_f32(x, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_relu_f32(x, out)
    }
}

/// NEON-optimized Leaky ReLU activation for f32
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `x` and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_leaky_relu_f32_impl(x: &[f32], alpha: f32, out: &mut [f32]) {
    let len = x.len().min(out.len());
    let mut i = 0;
    let zero = vdupq_n_f32(0.0);
    let valpha = vdupq_n_f32(alpha);

    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        // if x > 0 then x else alpha * x
        let pos_mask = vcgtq_f32(vx, zero);
        let neg_part = vmulq_f32(vx, valpha);
        let vr = vbslq_f32(pos_mask, vx, neg_part);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = if x[i] > 0.0 { x[i] } else { alpha * x[i] };
        i += 1;
    }
}

pub fn neon_leaky_relu_f32(x: &[f32], alpha: f32, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_leaky_relu_f32_impl(x, alpha, out) }
        } else {
            fallback_leaky_relu_f32(x, alpha, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_leaky_relu_f32(x, alpha, out)
    }
}

/// NEON-optimized sigmoid activation for f32
///
/// Uses Pade approximation: sigmoid(x) ≈ 0.5 + x/(2*(1+|x|))
/// This is more battery-efficient than exp-based sigmoid on mobile
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `x` and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32_impl(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    let mut i = 0;
    let vhalf = vdupq_n_f32(0.5);
    let vtwo = vdupq_n_f32(2.0);
    let vone = vdupq_n_f32(1.0);

    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        // Pade approximation: 0.5 + x / (2 * (1 + |x|))
        let vabs = vabsq_f32(vx);
        let denom = vmulq_f32(vtwo, vaddq_f32(vone, vabs));
        let vr = vaddq_f32(vhalf, vdivq_f32(vx, denom));
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        out[i] = 0.5 + x[i] / (2.0 * (1.0 + x[i].abs()));
        i += 1;
    }
}

pub fn neon_sigmoid_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_sigmoid_f32_impl(x, out) }
        } else {
            fallback_sigmoid_f32(x, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_sigmoid_f32(x, out)
    }
}

/// NEON-optimized tanh activation for f32
///
/// Uses rational approximation for battery efficiency
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `x` and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_tanh_f32_impl(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    let mut i = 0;

    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        // Rational approximation: x * (27 + x^2) / (27 + 9*x^2)
        let vx2 = vmulq_f32(vx, vx);
        let v27 = vdupq_n_f32(27.0);
        let v9 = vdupq_n_f32(9.0);

        let num = vmulq_f32(vx, vaddq_f32(v27, vx2));
        let denom = vaddq_f32(v27, vmulq_f32(v9, vx2));
        let vr = vdivq_f32(num, denom);

        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        let x2 = x[i] * x[i];
        out[i] = x[i] * (27.0 + x2) / (27.0 + 9.0 * x2);
        i += 1;
    }
}

pub fn neon_tanh_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_tanh_f32_impl(x, out) }
        } else {
            fallback_tanh_f32(x, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_tanh_f32(x, out)
    }
}

/// NEON-optimized GELU activation for f32
///
/// Uses approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON instructions (aarch64).
/// The `x` and `out` slices must be valid for reading/writing respectively.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32_impl(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    let mut i = 0;

    let vhalf = vdupq_n_f32(0.5);
    let vsqrt_2_pi = vdupq_n_f32(0.7978845608); // sqrt(2/π)
    let vc = vdupq_n_f32(0.044715);

    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vx3 = vmulq_f32(vmulq_f32(vx, vx), vx);

        // inner = sqrt(2/π) * (x + 0.044715 * x^3)
        let inner = vmulq_f32(vsqrt_2_pi, vaddq_f32(vx, vmulq_f32(vc, vx3)));

        // tanh approximation
        let inner2 = vmulq_f32(inner, inner);
        let v27 = vdupq_n_f32(27.0);
        let v9 = vdupq_n_f32(9.0);
        let tanh_num = vmulq_f32(inner, vaddq_f32(v27, inner2));
        let tanh_denom = vaddq_f32(v27, vmulq_f32(v9, inner2));
        let tanh_val = vdivq_f32(tanh_num, tanh_denom);

        // 0.5 * x * (1 + tanh_val)
        let vone = vdupq_n_f32(1.0);
        let vr = vmulq_f32(vmulq_f32(vhalf, vx), vaddq_f32(vone, tanh_val));

        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    while i < len {
        let x3 = x[i] * x[i] * x[i];
        let inner = 0.7978845608 * (x[i] + 0.044715 * x3);
        let inner2 = inner * inner;
        let tanh_val = inner * (27.0 + inner2) / (27.0 + 9.0 * inner2);
        out[i] = 0.5 * x[i] * (1.0 + tanh_val);
        i += 1;
    }
}

pub fn neon_gelu_f32(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_gelu_f32_impl(x, out) }
        } else {
            fallback_gelu_f32(x, out)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fallback_gelu_f32(x, out)
    }
}

// Fallback implementations
fn fallback_relu_f32(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    for i in 0..len {
        out[i] = x[i].max(0.0);
    }
}

fn fallback_leaky_relu_f32(x: &[f32], alpha: f32, out: &mut [f32]) {
    let len = x.len().min(out.len());
    for i in 0..len {
        out[i] = if x[i] > 0.0 { x[i] } else { alpha * x[i] };
    }
}

fn fallback_sigmoid_f32(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    for i in 0..len {
        out[i] = 0.5 + x[i] / (2.0 * (1.0 + x[i].abs()));
    }
}

fn fallback_tanh_f32(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    for i in 0..len {
        let x2 = x[i] * x[i];
        out[i] = x[i] * (27.0 + x2) / (27.0 + 9.0 * x2);
    }
}

fn fallback_gelu_f32(x: &[f32], out: &mut [f32]) {
    let len = x.len().min(out.len());
    for i in 0..len {
        let x3 = x[i] * x[i] * x[i];
        let inner = 0.7978845608 * (x[i] + 0.044715 * x3);
        let inner2 = inner * inner;
        let tanh_val = inner * (27.0 + inner2) / (27.0 + 9.0 * inner2);
        out[i] = 0.5 * x[i] * (1.0 + tanh_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_relu() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0; 5];

        neon_relu_f32(&x, &mut out);

        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_neon_leaky_relu() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0; 5];
        let alpha = 0.1;

        neon_leaky_relu_f32(&x, alpha, &mut out);

        assert_eq!(out, vec![-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_neon_sigmoid() {
        let x = vec![0.0, 1.0, -1.0];
        let mut out = vec![0.0; 3];

        neon_sigmoid_f32(&x, &mut out);

        // sigmoid(0) ≈ 0.5
        assert!((out[0] - 0.5).abs() < 0.01);
        // sigmoid(1) > 0.5
        assert!(out[1] > 0.5);
        // sigmoid(-1) < 0.5
        assert!(out[2] < 0.5);
    }
}
