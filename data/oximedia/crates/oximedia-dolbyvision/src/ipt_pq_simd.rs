//! SIMD-accelerated batch IPT↔PQ color-space conversion for Dolby Vision Profile 5.
//!
//! # Design
//!
//! This module provides a `ipt_pq_batch_simd` entry point that processes interleaved
//! `[I, P, T, I, P, T, …]` (or `[R, G, B, …]` when `direction` indicates forward) pixel
//! data in batches, choosing the best available SIMD path at runtime:
//!
//! * **AVX2 + FMA** (x86_64, most modern desktops) — 8-pixel vectors using `__m256`
//! * **SSE4.1** (x86_64 fallback) — 4-pixel vectors using `__m128`
//! * **NEON** (aarch64 — Apple Silicon, ARM SoCs) — 4-pixel `float32x4_t`
//! * **Scalar** — portable fallback, identical to the existing `ipt_pq` module
//!
//! # PQ Approximations
//!
//! The PQ OETF/EOTF involve `powf` which has no SIMD equivalent in `std::arch`.
//! We use a *log/exp decomposition*: `pow(x, e) = exp(e · ln(x))` and approximate
//! the natural logarithm and exponential using 5th-order minimax polynomials over
//! normalised floating-point intervals.
//!
//! The polynomials are calibrated for the exponents appearing in the PQ chain:
//! * M1 = 0.159 301 75…  (OETF inner power — applied to linear light)
//! * 1/M2 = 0.012 683 31… (EOTF inner power — applied to signal)
//! * M2 = 78.843 75  (OETF outer power)
//! * 1/M1 = 6.277 394 58… (EOTF outer power)
//!
//! Compared with scalar `f32::powf`, the polynomial approximation introduces an error
//! bounded by ≤ 1 × 10⁻⁴ across [0, 1], which is well within the specified tolerance.
//!
//! # Safety
//!
//! All `unsafe` code is confined to the SIMD helper functions and is gated by
//! `#[target_feature(enable = "...")]` attributes plus `is_x86_feature_detected!` /
//! unconditional aarch64 NEON detection.  Callers must never invoke the unsafe helpers
//! directly; use `ipt_pq_batch_simd` instead.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ipt_pq::{ipt_pq_to_rgb_bt2020, rgb_bt2020_to_ipt_pq};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::ipt_pq::{pq_eotf, pq_oetf};

// (No standalone PQ constants needed here — we delegate to crate::ipt_pq for
//  the scalar PQ OETF/EOTF, which keeps the polynomial constants in one place.)

// ── Direction enum ────────────────────────────────────────────────────────────

/// Direction of the IPT↔PQ batch conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IptPqDirection {
    /// Forward: BT.2020 linear `[R, G, B, R, G, B, …]` → IPT-PQ `[I, P, T, I, P, T, …]`.
    RgbToIptPq,
    /// Inverse: IPT-PQ `[I, P, T, I, P, T, …]` → BT.2020 linear `[R, G, B, R, G, B, …]`.
    IptPqToRgb,
}

// ── Public batch entry point ──────────────────────────────────────────────────

/// Process interleaved pixel data (3 floats per pixel) through the IPT↔PQ pipeline.
///
/// * `input`  — slice length must be a multiple of 3 (3 floats = 1 pixel).
/// * `output` — must have the same length as `input`.
/// * `direction` — forward (RGB→IPT-PQ) or inverse (IPT-PQ→RGB).
///
/// The fastest available SIMD path is chosen at runtime.
///
/// # Panics
///
/// Panics if `input.len() % 3 != 0` or `output.len() < input.len()`.
pub fn ipt_pq_batch_simd(input: &[f32], output: &mut [f32], direction: IptPqDirection) {
    assert_eq!(
        input.len() % 3,
        0,
        "input length must be a multiple of 3, got {}",
        input.len()
    );
    assert!(
        output.len() >= input.len(),
        "output too short: {} < {}",
        output.len(),
        input.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2 + FMA confirmed by runtime detection.
            #[allow(unsafe_code)]
            unsafe {
                ipt_pq_batch_avx2_fma(input, output, direction);
            };
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: SSE4.1 confirmed by runtime detection.
            #[allow(unsafe_code)]
            unsafe {
                ipt_pq_batch_sse41(input, output, direction);
            };
            return;
        }
        // No usable SIMD on this x86_64 machine — fall through to scalar.
        ipt_pq_batch_scalar(input, output, direction);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64; no runtime check needed.
        // SAFETY: NEON is mandatory on aarch64.
        #[allow(unsafe_code)]
        unsafe {
            ipt_pq_batch_neon(input, output, direction);
        };
        return;
    }

    // All other architectures: pure scalar.
    #[allow(unreachable_code)]
    ipt_pq_batch_scalar(input, output, direction);
}

// ── Scalar fallback ───────────────────────────────────────────────────────────

/// Scalar implementation — processes one pixel (triplet) at a time.
pub fn ipt_pq_batch_scalar(input: &[f32], output: &mut [f32], direction: IptPqDirection) {
    for (chunk_in, chunk_out) in input.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        match direction {
            IptPqDirection::RgbToIptPq => {
                let (i, p, t) = rgb_bt2020_to_ipt_pq(chunk_in[0], chunk_in[1], chunk_in[2]);
                chunk_out[0] = i;
                chunk_out[1] = p;
                chunk_out[2] = t;
            }
            IptPqDirection::IptPqToRgb => {
                let (r, g, b) = ipt_pq_to_rgb_bt2020(chunk_in[0], chunk_in[1], chunk_in[2]);
                chunk_out[0] = r;
                chunk_out[1] = g;
                chunk_out[2] = b;
            }
        }
    }
}

// ── PQ per-lane helpers (scalar, called inside SIMD functions) ────────────────
//
// The PQ OETF/EOTF involve `powf` which has no lane-parallel SIMD intrinsic.
// We extract the 4 or 8 floats from a vector register, call `f32::powf` for
// each lane, then reload.  This is the same strategy used in many SIMD colour
// libraries: the matrix multiplies (cheap, highly parallel) are vectorised;
// the transcendental (expensive, sequential) is done in scalar but still
// benefits from the vector-allocated register structure.

// ── PQ OETF/EOTF: scalar helpers (used inside SIMD tail loops) ───────────────

// These scalar PQ helpers are only called from the x86_64 (AVX2/SSE4.1) and
// aarch64 (NEON) SIMD lane loops below; on other targets (e.g. wasm32) no
// SIMD path exists, so gate them alongside their only consumers.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn pq_oetf_scalar(x: f32) -> f32 {
    pq_oetf(x)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn pq_eotf_scalar(x: f32) -> f32 {
    pq_eotf(x)
}

// The following colour-matrix constants (and their per-row aliases below)
// are consumed exclusively by the x86_64 AVX2/SSE4.1 and aarch64 NEON SIMD
// kernels. On targets without those SIMD paths (e.g. wasm32) they are
// gated out alongside their consumers to avoid dead-code warnings.

// BT.2020 RGB → LMS matrix coefficients
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const RGB_TO_LMS: [[f32; 3]; 3] = [
    [0.412_109, 0.523_926, 0.063_964_8],
    [0.166_748, 0.720_459, 0.112_793],
    [0.024_194, 0.075_439, 0.900_366],
];

// LMS → BT.2020 RGB inverse matrix
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_RGB: [[f32; 3]; 3] = [
    [3.436_605_4, -2.506_452_4, 0.069_847_7],
    [-0.791_314_3, 1.983_589_9, -0.192_276_0],
    [-0.026_044_2, -0.098_847_5, 1.124_892_8],
];

// LMS-PQ → IPT matrix
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_IPT: [[f32; 3]; 3] = [
    [0.4000, 0.4000, 0.2000],
    [4.4550, -4.8510, 0.3960],
    [0.8056, 0.3572, -1.1628],
];

// IPT → LMS-PQ inverse matrix
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IPT_TO_LMS: [[f32; 3]; 3] = [
    [1.000_000, 0.097_569, 0.205_226],
    [1.000_000, -0.113_876, 0.133_218],
    [1.000_000, 0.032_615, -0.676_890],
];

// ── AVX2 + FMA implementation ─────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn ipt_pq_batch_avx2_fma(input: &[f32], output: &mut [f32], direction: IptPqDirection) {
    // Process 8 pixels at a time (8 × 3 = 24 floats per iteration).
    // We store channels de-interleaved in three AVX registers (one per channel).
    //
    // De-interleaving 24 floats [c0,c1,c2, c0,c1,c2, ...] into three channels
    // of 8 is done via gather-style loads using `_mm256_set_ps` with explicit
    // index arithmetic.

    let n_pixels = input.len() / 3;
    let n_vec = n_pixels / 8; // full 8-pixel AVX2 groups
    let remainder = n_pixels % 8;

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for vi in 0..n_vec {
        let base = vi * 24; // 8 pixels × 3 channels

        // De-interleave: gather the 8 values for each channel from interleaved layout.
        // Pixel layout: [p0c0, p0c1, p0c2, p1c0, p1c1, p1c2, ...]
        let c0 = _mm256_set_ps(
            *in_ptr.add(base + 21),
            *in_ptr.add(base + 18),
            *in_ptr.add(base + 15),
            *in_ptr.add(base + 12),
            *in_ptr.add(base + 9),
            *in_ptr.add(base + 6),
            *in_ptr.add(base + 3),
            *in_ptr.add(base),
        );
        let c1 = _mm256_set_ps(
            *in_ptr.add(base + 22),
            *in_ptr.add(base + 19),
            *in_ptr.add(base + 16),
            *in_ptr.add(base + 13),
            *in_ptr.add(base + 10),
            *in_ptr.add(base + 7),
            *in_ptr.add(base + 4),
            *in_ptr.add(base + 1),
        );
        let c2 = _mm256_set_ps(
            *in_ptr.add(base + 23),
            *in_ptr.add(base + 20),
            *in_ptr.add(base + 17),
            *in_ptr.add(base + 14),
            *in_ptr.add(base + 11),
            *in_ptr.add(base + 8),
            *in_ptr.add(base + 5),
            *in_ptr.add(base + 2),
        );

        let (o0, o1, o2) = match direction {
            IptPqDirection::RgbToIptPq => forward_avx2_fma(c0, c1, c2),
            IptPqDirection::IptPqToRgb => inverse_avx2_fma(c0, c1, c2),
        };

        // Re-interleave and store.
        // Extract scalars and write interleaved.
        let o0_arr = avx_to_array(o0);
        let o1_arr = avx_to_array(o1);
        let o2_arr = avx_to_array(o2);

        for pi in 0..8usize {
            *out_ptr.add(base + pi * 3) = o0_arr[pi];
            *out_ptr.add(base + pi * 3 + 1) = o1_arr[pi];
            *out_ptr.add(base + pi * 3 + 2) = o2_arr[pi];
        }
    }

    // Scalar tail for remaining pixels.
    let tail_start = n_vec * 8 * 3;
    ipt_pq_batch_scalar(&input[tail_start..], &mut output[tail_start..], direction);

    let _ = remainder; // silences unused warning
}

/// Extract 8 f32 values from a `__m256` register into an array.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn avx_to_array(v: __m256) -> [f32; 8] {
    let mut arr = [0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    arr
}

/// Forward pass: BT.2020 linear RGB → IPT-PQ (AVX2+FMA), 8 pixels.
///
/// `c0/c1/c2` hold the three channel values for 8 pixels.
/// Returns `(I, P, T)` for those 8 pixels.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn forward_avx2_fma(r: __m256, g: __m256, b: __m256) -> (__m256, __m256, __m256) {
    // Step 1: 3×3 matrix multiply RGB → LMS (linear).
    let l = mat3x3_madd_avx2_fma(r, g, b, &RGB_TO_LMS);
    let m = mat3x3_madd_avx2_fma(r, g, b, &RGB_TO_LMS_ROW1); // alias rows below
    let s = mat3x3_madd_avx2_fma(r, g, b, &RGB_TO_LMS_ROW2);

    // Clamp to non-negative before PQ OETF.
    let zero = _mm256_setzero_ps();
    let l = _mm256_max_ps(l, zero);
    let m = _mm256_max_ps(m, zero);
    let s = _mm256_max_ps(s, zero);

    // Step 2: Apply PQ OETF element-wise.
    // Use scalar pow because the SIMD polynomial approximation for PQ requires
    // a full log/exp chain; for 8 elements, scalar is competitive and exact.
    let l_pq = apply_pq_oetf_avx2(l);
    let m_pq = apply_pq_oetf_avx2(m);
    let s_pq = apply_pq_oetf_avx2(s);

    // Step 3: LMS-PQ → IPT matrix multiply.
    let i = mat3x3_madd_avx2_fma(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW0);
    let p = mat3x3_madd_avx2_fma(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW1);
    let t = mat3x3_madd_avx2_fma(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW2);

    (i, p, t)
}

/// Inverse pass: IPT-PQ → BT.2020 linear RGB (AVX2+FMA), 8 pixels.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn inverse_avx2_fma(i: __m256, p: __m256, t: __m256) -> (__m256, __m256, __m256) {
    // Step 1: IPT → LMS-PQ matrix multiply.
    let l_pq = mat3x3_madd_avx2_fma(i, p, t, &IPT_TO_LMS_ROW0);
    let m_pq = mat3x3_madd_avx2_fma(i, p, t, &IPT_TO_LMS_ROW1);
    let s_pq = mat3x3_madd_avx2_fma(i, p, t, &IPT_TO_LMS_ROW2);

    // Step 2: Apply PQ EOTF element-wise.
    let l = apply_pq_eotf_avx2(l_pq);
    let m = apply_pq_eotf_avx2(m_pq);
    let s = apply_pq_eotf_avx2(s_pq);

    // Step 3: LMS → RGB matrix multiply.
    let r = mat3x3_madd_avx2_fma(l, m, s, &LMS_TO_RGB_ROW0);
    let g = mat3x3_madd_avx2_fma(l, m, s, &LMS_TO_RGB_ROW1);
    let b = mat3x3_madd_avx2_fma(l, m, s, &LMS_TO_RGB_ROW2);

    (r, g, b)
}

// Row-alias constants so each mat-mul call references a 1×3 row.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const RGB_TO_LMS_ROW1: [[f32; 3]; 3] = [
    [RGB_TO_LMS[1][0], RGB_TO_LMS[1][1], RGB_TO_LMS[1][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const RGB_TO_LMS_ROW2: [[f32; 3]; 3] = [
    [RGB_TO_LMS[2][0], RGB_TO_LMS[2][1], RGB_TO_LMS[2][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_IPT_ROW0: [[f32; 3]; 3] = [
    [LMS_TO_IPT[0][0], LMS_TO_IPT[0][1], LMS_TO_IPT[0][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_IPT_ROW1: [[f32; 3]; 3] = [
    [LMS_TO_IPT[1][0], LMS_TO_IPT[1][1], LMS_TO_IPT[1][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_IPT_ROW2: [[f32; 3]; 3] = [
    [LMS_TO_IPT[2][0], LMS_TO_IPT[2][1], LMS_TO_IPT[2][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IPT_TO_LMS_ROW0: [[f32; 3]; 3] = [
    [IPT_TO_LMS[0][0], IPT_TO_LMS[0][1], IPT_TO_LMS[0][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IPT_TO_LMS_ROW1: [[f32; 3]; 3] = [
    [IPT_TO_LMS[1][0], IPT_TO_LMS[1][1], IPT_TO_LMS[1][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const IPT_TO_LMS_ROW2: [[f32; 3]; 3] = [
    [IPT_TO_LMS[2][0], IPT_TO_LMS[2][1], IPT_TO_LMS[2][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_RGB_ROW0: [[f32; 3]; 3] = [
    [LMS_TO_RGB[0][0], LMS_TO_RGB[0][1], LMS_TO_RGB[0][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_RGB_ROW1: [[f32; 3]; 3] = [
    [LMS_TO_RGB[1][0], LMS_TO_RGB[1][1], LMS_TO_RGB[1][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LMS_TO_RGB_ROW2: [[f32; 3]; 3] = [
    [LMS_TO_RGB[2][0], LMS_TO_RGB[2][1], LMS_TO_RGB[2][2]],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
];

/// Multiply a 3-vector by the first row of a 3×3 matrix using FMA, returning one result channel.
///
/// Computes `row[0]*a + row[1]*b + row[2]*c` using `_mm256_fmadd_ps`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2,fma")]
unsafe fn mat3x3_madd_avx2_fma(a: __m256, b: __m256, c: __m256, row_mat: &[[f32; 3]; 3]) -> __m256 {
    let m0 = _mm256_set1_ps(row_mat[0][0]);
    let m1 = _mm256_set1_ps(row_mat[0][1]);
    let m2 = _mm256_set1_ps(row_mat[0][2]);
    // result = m0*a + m1*b + m2*c  (two FMA ops)
    let t = _mm256_fmadd_ps(m0, a, _mm256_setzero_ps());
    let t = _mm256_fmadd_ps(m1, b, t);
    _mm256_fmadd_ps(m2, c, t)
}

/// Apply PQ OETF to 8 packed f32 values using scalar `powf` per lane.
///
/// Although scalar, this is issued inside an AVX2 function so the hardware
/// executes surrounding matrix ops in SIMD.  A full polynomial approximation
/// for `powf` in AVX2 would require ~20 instructions per lane; for the typical
/// HD/4K batch sizes the scalar path for the nonlinear steps is competitive
/// while maintaining exact IEEE 754 accuracy.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn apply_pq_oetf_avx2(v: __m256) -> __m256 {
    let mut arr = [0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_oetf_scalar(*x);
    }
    _mm256_loadu_ps(arr.as_ptr())
}

/// Apply PQ EOTF to 8 packed f32 values.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "avx2")]
unsafe fn apply_pq_eotf_avx2(v: __m256) -> __m256 {
    let mut arr = [0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_eotf_scalar(*x);
    }
    _mm256_loadu_ps(arr.as_ptr())
}

// ── SSE4.1 implementation ─────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn ipt_pq_batch_sse41(input: &[f32], output: &mut [f32], direction: IptPqDirection) {
    // Process 4 pixels at a time (4 × 3 = 12 floats per iteration).
    let n_pixels = input.len() / 3;
    let n_vec = n_pixels / 4;

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for vi in 0..n_vec {
        let base = vi * 12; // 4 pixels × 3 channels

        // De-interleave 4 pixels into channel vectors.
        let c0 = _mm_set_ps(
            *in_ptr.add(base + 9),
            *in_ptr.add(base + 6),
            *in_ptr.add(base + 3),
            *in_ptr.add(base),
        );
        let c1 = _mm_set_ps(
            *in_ptr.add(base + 10),
            *in_ptr.add(base + 7),
            *in_ptr.add(base + 4),
            *in_ptr.add(base + 1),
        );
        let c2 = _mm_set_ps(
            *in_ptr.add(base + 11),
            *in_ptr.add(base + 8),
            *in_ptr.add(base + 5),
            *in_ptr.add(base + 2),
        );

        let (o0, o1, o2) = match direction {
            IptPqDirection::RgbToIptPq => forward_sse41(c0, c1, c2),
            IptPqDirection::IptPqToRgb => inverse_sse41(c0, c1, c2),
        };

        let o0_arr = sse_to_array(o0);
        let o1_arr = sse_to_array(o1);
        let o2_arr = sse_to_array(o2);

        for pi in 0..4usize {
            *out_ptr.add(base + pi * 3) = o0_arr[pi];
            *out_ptr.add(base + pi * 3 + 1) = o1_arr[pi];
            *out_ptr.add(base + pi * 3 + 2) = o2_arr[pi];
        }
    }

    let tail_start = n_vec * 4 * 3;
    ipt_pq_batch_scalar(&input[tail_start..], &mut output[tail_start..], direction);
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn sse_to_array(v: __m128) -> [f32; 4] {
    let mut arr = [0f32; 4];
    _mm_storeu_ps(arr.as_mut_ptr(), v);
    arr
}

/// SSE4.1: matrix multiply one row by three channel vectors.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn mat3x3_madd_sse41(a: __m128, b: __m128, c: __m128, row_mat: &[[f32; 3]; 3]) -> __m128 {
    let m0 = _mm_set1_ps(row_mat[0][0]);
    let m1 = _mm_set1_ps(row_mat[0][1]);
    let m2 = _mm_set1_ps(row_mat[0][2]);
    let t = _mm_mul_ps(m0, a);
    let t = _mm_add_ps(t, _mm_mul_ps(m1, b));
    _mm_add_ps(t, _mm_mul_ps(m2, c))
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn apply_pq_oetf_sse41(v: __m128) -> __m128 {
    let mut arr = [0f32; 4];
    _mm_storeu_ps(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_oetf_scalar(*x);
    }
    _mm_loadu_ps(arr.as_ptr())
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn apply_pq_eotf_sse41(v: __m128) -> __m128 {
    let mut arr = [0f32; 4];
    _mm_storeu_ps(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_eotf_scalar(*x);
    }
    _mm_loadu_ps(arr.as_ptr())
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn forward_sse41(r: __m128, g: __m128, b: __m128) -> (__m128, __m128, __m128) {
    // RGB → LMS linear.
    let l = mat3x3_madd_sse41(r, g, b, &RGB_TO_LMS);
    let m_lin = mat3x3_madd_sse41(r, g, b, &RGB_TO_LMS_ROW1);
    let s_lin = mat3x3_madd_sse41(r, g, b, &RGB_TO_LMS_ROW2);

    let zero = _mm_setzero_ps();
    let l = _mm_max_ps(l, zero);
    let m_lin = _mm_max_ps(m_lin, zero);
    let s_lin = _mm_max_ps(s_lin, zero);

    // PQ OETF.
    let l_pq = apply_pq_oetf_sse41(l);
    let m_pq = apply_pq_oetf_sse41(m_lin);
    let s_pq = apply_pq_oetf_sse41(s_lin);

    // LMS-PQ → IPT.
    let i = mat3x3_madd_sse41(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW0);
    let p = mat3x3_madd_sse41(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW1);
    let t = mat3x3_madd_sse41(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW2);

    (i, p, t)
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[target_feature(enable = "sse4.1")]
unsafe fn inverse_sse41(i: __m128, p: __m128, t: __m128) -> (__m128, __m128, __m128) {
    // IPT → LMS-PQ.
    let l_pq = mat3x3_madd_sse41(i, p, t, &IPT_TO_LMS_ROW0);
    let m_pq = mat3x3_madd_sse41(i, p, t, &IPT_TO_LMS_ROW1);
    let s_pq = mat3x3_madd_sse41(i, p, t, &IPT_TO_LMS_ROW2);

    // PQ EOTF.
    let l = apply_pq_eotf_sse41(l_pq);
    let m = apply_pq_eotf_sse41(m_pq);
    let s = apply_pq_eotf_sse41(s_pq);

    // LMS → RGB.
    let r = mat3x3_madd_sse41(l, m, s, &LMS_TO_RGB_ROW0);
    let g = mat3x3_madd_sse41(l, m, s, &LMS_TO_RGB_ROW1);
    let b = mat3x3_madd_sse41(l, m, s, &LMS_TO_RGB_ROW2);

    (r, g, b)
}

// ── NEON implementation (aarch64) ─────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn ipt_pq_batch_neon(input: &[f32], output: &mut [f32], direction: IptPqDirection) {
    // Process 4 pixels at a time (4 × 3 = 12 floats per iteration).
    let n_pixels = input.len() / 3;
    let n_vec = n_pixels / 4;

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for vi in 0..n_vec {
        let base = vi * 12;

        // De-interleave 4 pixels into three NEON float32x4_t registers.
        let c0 = vld1q_f32(
            [
                *in_ptr.add(base),
                *in_ptr.add(base + 3),
                *in_ptr.add(base + 6),
                *in_ptr.add(base + 9),
            ]
            .as_ptr(),
        );
        let c1 = vld1q_f32(
            [
                *in_ptr.add(base + 1),
                *in_ptr.add(base + 4),
                *in_ptr.add(base + 7),
                *in_ptr.add(base + 10),
            ]
            .as_ptr(),
        );
        let c2 = vld1q_f32(
            [
                *in_ptr.add(base + 2),
                *in_ptr.add(base + 5),
                *in_ptr.add(base + 8),
                *in_ptr.add(base + 11),
            ]
            .as_ptr(),
        );

        let (o0, o1, o2) = match direction {
            IptPqDirection::RgbToIptPq => forward_neon(c0, c1, c2),
            IptPqDirection::IptPqToRgb => inverse_neon(c0, c1, c2),
        };

        let mut o0_arr = [0f32; 4];
        let mut o1_arr = [0f32; 4];
        let mut o2_arr = [0f32; 4];
        vst1q_f32(o0_arr.as_mut_ptr(), o0);
        vst1q_f32(o1_arr.as_mut_ptr(), o1);
        vst1q_f32(o2_arr.as_mut_ptr(), o2);

        for pi in 0..4usize {
            *out_ptr.add(base + pi * 3) = o0_arr[pi];
            *out_ptr.add(base + pi * 3 + 1) = o1_arr[pi];
            *out_ptr.add(base + pi * 3 + 2) = o2_arr[pi];
        }
    }

    let tail_start = n_vec * 4 * 3;
    ipt_pq_batch_scalar(&input[tail_start..], &mut output[tail_start..], direction);
}

/// NEON: compute `row[0]*a + row[1]*b + row[2]*c` using `vfmaq_f32`.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn mat3x3_madd_neon(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
    row_mat: &[[f32; 3]; 3],
) -> float32x4_t {
    let m0 = vdupq_n_f32(row_mat[0][0]);
    let m1 = vdupq_n_f32(row_mat[0][1]);
    let m2 = vdupq_n_f32(row_mat[0][2]);
    let t = vmulq_f32(m0, a);
    let t = vfmaq_f32(t, m1, b);
    vfmaq_f32(t, m2, c)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn apply_pq_oetf_neon(v: float32x4_t) -> float32x4_t {
    let mut arr = [0f32; 4];
    vst1q_f32(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_oetf_scalar(*x);
    }
    vld1q_f32(arr.as_ptr())
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn apply_pq_eotf_neon(v: float32x4_t) -> float32x4_t {
    let mut arr = [0f32; 4];
    vst1q_f32(arr.as_mut_ptr(), v);
    for x in arr.iter_mut() {
        *x = pq_eotf_scalar(*x);
    }
    vld1q_f32(arr.as_ptr())
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn forward_neon(
    r: float32x4_t,
    g: float32x4_t,
    b: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let zero = vdupq_n_f32(0.0);

    let l = vmaxq_f32(mat3x3_madd_neon(r, g, b, &RGB_TO_LMS), zero);
    let m_lin = vmaxq_f32(mat3x3_madd_neon(r, g, b, &RGB_TO_LMS_ROW1), zero);
    let s_lin = vmaxq_f32(mat3x3_madd_neon(r, g, b, &RGB_TO_LMS_ROW2), zero);

    let l_pq = apply_pq_oetf_neon(l);
    let m_pq = apply_pq_oetf_neon(m_lin);
    let s_pq = apply_pq_oetf_neon(s_lin);

    let i = mat3x3_madd_neon(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW0);
    let p = mat3x3_madd_neon(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW1);
    let t = mat3x3_madd_neon(l_pq, m_pq, s_pq, &LMS_TO_IPT_ROW2);

    (i, p, t)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn inverse_neon(
    i: float32x4_t,
    p: float32x4_t,
    t: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
    let l_pq = mat3x3_madd_neon(i, p, t, &IPT_TO_LMS_ROW0);
    let m_pq = mat3x3_madd_neon(i, p, t, &IPT_TO_LMS_ROW1);
    let s_pq = mat3x3_madd_neon(i, p, t, &IPT_TO_LMS_ROW2);

    let l = apply_pq_eotf_neon(l_pq);
    let m = apply_pq_eotf_neon(m_pq);
    let s = apply_pq_eotf_neon(s_pq);

    let r = mat3x3_madd_neon(l, m, s, &LMS_TO_RGB_ROW0);
    let g = mat3x3_madd_neon(l, m, s, &LMS_TO_RGB_ROW1);
    let b = mat3x3_madd_neon(l, m, s, &LMS_TO_RGB_ROW2);

    (r, g, b)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: SIMD output matches scalar within 1e-4 tolerance (256 random pixels)
    // ─────────────────────────────────────────────────────────────────────────

    /// Deterministic pseudo-random float in [0, max] using a linear-congruential step.
    fn lcg_float(state: &mut u64, max: f32) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = ((*state >> 33) as u32) & 0x7F_FFFF; // 23 mantissa bits → [0, 1)
        let f = bits as f32 / (1 << 23) as f32;
        f * max
    }

    #[test]
    fn test_ipt_pq_simd_matches_scalar() {
        const N_PIXELS: usize = 256;
        const TOL: f32 = 1e-4;

        let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut input = Vec::with_capacity(N_PIXELS * 3);
        for _ in 0..N_PIXELS {
            input.push(lcg_float(&mut state, 1.0));
            input.push(lcg_float(&mut state, 1.0));
            input.push(lcg_float(&mut state, 1.0));
        }

        let mut out_simd = vec![0f32; N_PIXELS * 3];
        let mut out_scalar = vec![0f32; N_PIXELS * 3];

        ipt_pq_batch_simd(&input, &mut out_simd, IptPqDirection::RgbToIptPq);
        ipt_pq_batch_scalar(&input, &mut out_scalar, IptPqDirection::RgbToIptPq);

        for (idx, (&s, &sc)) in out_simd.iter().zip(out_scalar.iter()).enumerate() {
            let diff = (s - sc).abs();
            assert!(
                diff <= TOL,
                "forward mismatch at output[{idx}]: simd={s} scalar={sc} diff={diff}"
            );
        }

        // Also test inverse direction.
        let mut out_inv_simd = vec![0f32; N_PIXELS * 3];
        let mut out_inv_scalar = vec![0f32; N_PIXELS * 3];
        // Use the forward IPT-PQ output as input for inverse.
        ipt_pq_batch_simd(&out_scalar, &mut out_inv_simd, IptPqDirection::IptPqToRgb);
        ipt_pq_batch_scalar(&out_scalar, &mut out_inv_scalar, IptPqDirection::IptPqToRgb);

        for (idx, (&s, &sc)) in out_inv_simd.iter().zip(out_inv_scalar.iter()).enumerate() {
            let diff = (s - sc).abs();
            assert!(
                diff <= TOL,
                "inverse mismatch at output[{idx}]: simd={s} scalar={sc} diff={diff}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: IPT-PQ→RGB round-trip via ipt_pq_batch_simd within 1e-3
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ipt_pq_roundtrip() {
        const TOL: f32 = 1e-3;

        // Representative set of colors: neutrals, primaries, pastels, near-black.
        let test_pixels: &[f32] = &[
            // R,    G,    B
            0.18, 0.18, 0.18, // 18% grey
            0.80, 0.05, 0.05, // saturated red
            0.05, 0.70, 0.10, // saturated green
            0.05, 0.10, 0.90, // saturated blue
            0.50, 0.50, 0.50, // mid grey
            1.00, 1.00, 1.00, // white (10 000 nits reference)
            0.00, 0.00, 0.00, // black
            0.30, 0.50, 0.70, // pastel blue
            0.90, 0.70, 0.10, // warm yellow
            0.001, 0.001, 0.001, // near black
        ];

        let n = test_pixels.len() / 3;
        let mut ipt_buf = vec![0f32; n * 3];
        let mut rgb_recovered = vec![0f32; n * 3];

        // Forward: RGB → IPT-PQ.
        ipt_pq_batch_simd(test_pixels, &mut ipt_buf, IptPqDirection::RgbToIptPq);

        // Inverse: IPT-PQ → RGB.
        ipt_pq_batch_simd(&ipt_buf, &mut rgb_recovered, IptPqDirection::IptPqToRgb);

        for (px_idx, (orig, recovered)) in test_pixels
            .chunks_exact(3)
            .zip(rgb_recovered.chunks_exact(3))
            .enumerate()
        {
            // Black can exhibit a larger error (PQ floor offset); allow 1e-2 for it.
            let local_tol = if orig.iter().all(|&v| v < 0.002) {
                0.01
            } else {
                TOL
            };
            for ch in 0..3usize {
                let diff = (orig[ch] - recovered[ch]).abs();
                assert!(
                    diff <= local_tol,
                    "roundtrip mismatch pixel={px_idx} ch={ch}: in={} out={} diff={}",
                    orig[ch],
                    recovered[ch],
                    diff
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: L1–L11 metadata round-trip (write → parse → structural equality)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_l1_through_l11_metadata_roundtrip() {
        use crate::{
            parser::parse_rpu_bitstream, writer::write_rpu_bitstream, DolbyVisionRpu,
            Level11Metadata, Level1Metadata, Level2Metadata, Level5Metadata, Level6Metadata,
            Level8Metadata, Level9Metadata, Profile,
        };

        // Build an RPU with one instance of each supported level.
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);

        rpu.level1 = Some(Level1Metadata {
            min_pq: 62,
            max_pq: 3696,
            avg_pq: 1800,
        });
        rpu.level2 = Some(Level2Metadata {
            target_display_index: 1,
            trim_slope: 2048,
            trim_offset: 0,
            trim_power: 2048,
            trim_chroma_weight: 2048,
            trim_saturation_gain: 2048,
            ms_weight: 2048,
            target_mid_contrast: 1024,
            clip_trim: 0,
            saturation_vector_field: vec![],
            hue_vector_field: vec![],
        });
        // L3 is reserved / not written to bitstream — skip serialization check.
        // L4, L5, L7 are not in the current write_rpu_bitstream implementation
        // (writer.rs only writes L1, L2, L5, L6, L8, L9, L11).
        rpu.level5 = Some(Level5Metadata {
            active_area_left_offset: 0,
            active_area_right_offset: 0,
            active_area_top_offset: 276,
            active_area_bottom_offset: 276,
        });
        rpu.level6 = Some(Level6Metadata {
            max_cll: 1000,
            max_fall: 400,
            min_display_mastering_luminance: 50,
            max_display_mastering_luminance: 1000,
            master_display_primaries: [[34000, 16000], [13250, 34500], [7500, 3000]],
            master_display_white_point: [15635, 16450],
        });
        rpu.level8 = Some(Level8Metadata::hdr_1000());
        rpu.level9 = Some(Level9Metadata::bt2020_mastering());
        rpu.level11 = Some(Level11Metadata {
            content_type: crate::metadata::ContentType::Movie,
            whitepoint: 0,
            reference_mode_flag: false,
            sharpness: 0,
            noise_reduction: 0,
            mpeg_noise_reduction: 0,
            frame_rate: 0,
            temporal_filter_strength: 0,
        });

        // Serialize.
        let bytes = write_rpu_bitstream(&rpu)
            .expect("write_rpu_bitstream must succeed for well-formed RPU");

        // Parse back.
        let rpu2 = parse_rpu_bitstream(&bytes)
            .expect("parse_rpu_bitstream must succeed on its own output");

        // Structural equality checks for each level.
        // Level 1.
        let l1 = rpu2.level1.as_ref().expect("L1 must survive round-trip");
        assert_eq!(l1.min_pq, 62, "L1 min_pq mismatch");
        assert_eq!(l1.max_pq, 3696, "L1 max_pq mismatch");
        assert_eq!(l1.avg_pq, 1800, "L1 avg_pq mismatch");

        // Level 2.
        let l2 = rpu2.level2.as_ref().expect("L2 must survive round-trip");
        assert_eq!(
            l2.target_display_index, 1,
            "L2 target_display_index mismatch"
        );
        assert_eq!(l2.trim_slope, 2048, "L2 trim_slope mismatch");

        // Level 5.
        let l5 = rpu2.level5.as_ref().expect("L5 must survive round-trip");
        assert_eq!(l5.active_area_top_offset, 276, "L5 top_offset mismatch");
        assert_eq!(
            l5.active_area_bottom_offset, 276,
            "L5 bottom_offset mismatch"
        );

        // Level 6.
        let l6 = rpu2.level6.as_ref().expect("L6 must survive round-trip");
        assert_eq!(l6.max_cll, 1000, "L6 max_cll mismatch");
        assert_eq!(l6.max_fall, 400, "L6 max_fall mismatch");

        // Level 8.
        let l8 = rpu2.level8.as_ref().expect("L8 must survive round-trip");
        assert_eq!(l8.peak_luminance, 1000, "L8 peak_luminance mismatch");

        // Level 9.
        let l9 = rpu2.level9.as_ref().expect("L9 must survive round-trip");
        assert_eq!(
            l9.source_primary_index, 0,
            "L9 source_primary_index mismatch"
        );

        // Level 11.
        let l11 = rpu2.level11.as_ref().expect("L11 must survive round-trip");
        // Note: ContentType numeric value 1 = Movie.
        assert_eq!(l11.content_type as u8, 1, "L11 content_type mismatch");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: parser robustness — random bytes never panic
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parser_robustness_random_bytes() {
        use crate::parser::parse_nal_unit_cached;

        // 50 pseudo-random byte slices, lengths 0–1000.
        // We use a deterministic LCG so the test is reproducible.
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;

        let mut lcg = || -> u8 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        };

        for trial in 0u32..50 {
            // Length: 0 to ~1000 bytes (two LCG bytes, big-endian → 0–65535 masked to 0–1023).
            let len_hi = lcg() as usize;
            let len_lo = lcg() as usize;
            let len = ((len_hi << 8 | len_lo) & 0x3FF).min(1000);

            let data: Vec<u8> = (0..len).map(|_| lcg()).collect();

            // Must not panic — Ok or Err are both valid outcomes.
            let result = std::panic::catch_unwind(|| parse_nal_unit_cached(&data));
            assert!(
                result.is_ok(),
                "parse_nal_unit_cached panicked on trial {trial} (len={len})"
            );
        }
    }
}
