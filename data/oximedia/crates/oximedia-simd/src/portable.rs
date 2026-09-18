//! Portable SIMD-friendly operations using scalar code structured for
//! auto-vectorization by the compiler.
//!
//! All functions in this module use explicit 4-wide unrolled loops or
//! fixed-stride accumulation patterns that modern compilers (LLVM/rustc)
//! are able to auto-vectorize to SSE/AVX/NEON without target-specific
//! intrinsics.  The resulting code is therefore portable across all
//! architectures supported by `oximedia-simd`.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

// ── Dot product ───────────────────────────────────────────────────────────────

/// Compute the dot product of two `f32` slices using a 4-wide unrolled
/// accumulator pattern that enables auto-vectorization.
///
/// The shorter of the two slices determines how many elements are consumed.
/// Returns `0.0` when either slice is empty.
///
/// # Example
/// ```rust
/// use oximedia_simd::portable::portable_dot_f32;
/// let a = [1.0f32, 2.0, 3.0, 4.0];
/// let b = [4.0f32, 3.0, 2.0, 1.0];
/// let result = portable_dot_f32(&a, &b);
/// assert!((result - 20.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn portable_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let simd_len = len & !3; // round down to multiple of 4

    // Four independent accumulators allow the compiler to emit 4-wide SIMD.
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    let mut i = 0usize;
    while i < simd_len {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        i += 4;
    }

    let mut total = acc0 + acc1 + acc2 + acc3;

    // Scalar tail for elements that did not fill a full group of 4.
    for j in simd_len..len {
        total += a[j] * b[j];
    }

    total
}

// ── 4×4 matrix multiply ───────────────────────────────────────────────────────

/// Multiply two column-major 4×4 `f32` matrices and return the result.
///
/// Both `a` and `b` are stored in **row-major** order:
/// element `(row, col)` is at index `row * 4 + col`.
///
/// # Example
/// ```rust
/// use oximedia_simd::portable::portable_matmul_4x4;
/// let identity = [
///     1.0f32, 0.0, 0.0, 0.0,
///     0.0,    1.0, 0.0, 0.0,
///     0.0,    0.0, 1.0, 0.0,
///     0.0,    0.0, 0.0, 1.0,
/// ];
/// let result = portable_matmul_4x4(&identity, &identity);
/// for (a, b) in result.iter().zip(identity.iter()) {
///     assert!((a - b).abs() < 1e-6);
/// }
/// ```
#[must_use]
pub fn portable_matmul_4x4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];

    // Unrolled 4×4×4 multiply: each output row uses 4 dot products.
    for row in 0..4usize {
        let ar = row * 4;
        for col in 0..4usize {
            // out[row][col] = dot(a[row][*], b[*][col])
            out[ar + col] = a[ar] * b[col]
                + a[ar + 1] * b[4 + col]
                + a[ar + 2] * b[8 + col]
                + a[ar + 3] * b[12 + col];
        }
    }

    out
}

// ── RGBA → Luma (BT.601) ──────────────────────────────────────────────────────

/// Convert packed RGBA bytes to a luma (Y) plane using BT.601 coefficients.
///
/// The output contains one byte per input pixel: `Y = (66·R + 129·G + 25·B + 128) / 256 + 16`.
/// This is the BT.601 limited-range formula expressed in Q8 fixed-point arithmetic.
///
/// Alpha channel bytes in the input are ignored.
///
/// # Arguments
/// * `rgba` – Packed RGBA input (`width × height × 4` bytes).
///
/// # Returns
/// A `Vec<u8>` of length `rgba.len() / 4` containing the luma values.
///
/// # Example
/// ```rust
/// use oximedia_simd::portable::portable_rgba_to_luma;
/// // Pure red pixel in RGBA
/// let rgba = [255u8, 0, 0, 255];
/// let luma = portable_rgba_to_luma(&rgba);
/// assert_eq!(luma.len(), 1);
/// // Y = (66*255 + 129*0 + 25*0 + 128) / 256 + 16 = (16830 + 128) / 256 + 16 = 82
/// assert_eq!(luma[0], 82);
/// ```
#[must_use]
pub fn portable_rgba_to_luma(rgba: &[u8]) -> Vec<u8> {
    let pixels = rgba.len() / 4;
    let mut out = vec![0u8; pixels];

    // BT.601 limited-range Q8 coefficients:
    //   Y = (66·R + 129·G + 25·B + 128) >> 8 + 16
    // All intermediate values fit comfortably in u32.
    let simd_len = pixels & !3; // round down to multiple of 4

    let mut i = 0usize;
    while i < simd_len {
        // Process 4 pixels per iteration (4-wide unroll for auto-vectorization).
        let p0 = i * 4;
        let p1 = p0 + 4;
        let p2 = p0 + 8;
        let p3 = p0 + 12;

        out[i] = luma_bt601(rgba[p0], rgba[p0 + 1], rgba[p0 + 2]);
        out[i + 1] = luma_bt601(rgba[p1], rgba[p1 + 1], rgba[p1 + 2]);
        out[i + 2] = luma_bt601(rgba[p2], rgba[p2 + 1], rgba[p2 + 2]);
        out[i + 3] = luma_bt601(rgba[p3], rgba[p3 + 1], rgba[p3 + 2]);

        i += 4;
    }

    // Scalar tail.
    for j in simd_len..pixels {
        let base = j * 4;
        out[j] = luma_bt601(rgba[base], rgba[base + 1], rgba[base + 2]);
    }

    out
}

/// BT.601 limited-range luma computation in Q8 fixed-point (inline for the
/// auto-vectorization hint — same computation every call site).
#[inline(always)]
fn luma_bt601(r: u8, g: u8, b: u8) -> u8 {
    let r = r as u32;
    let g = g as u32;
    let b = b as u32;
    let y = (66 * r + 129 * g + 25 * b + 128) >> 8;
    // Add 16 and clamp to [16, 235] for limited-range signal.
    (y + 16).min(235) as u8
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── portable_dot_f32 ─────────────────────────────────────────────────────

    #[test]
    fn test_dot_f32_empty() {
        assert!(approx_eq(portable_dot_f32(&[], &[]), 0.0, 1e-9));
    }

    #[test]
    fn test_dot_f32_single_element() {
        assert!(approx_eq(portable_dot_f32(&[3.0], &[4.0]), 12.0, 1e-5));
    }

    #[test]
    fn test_dot_f32_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!(approx_eq(portable_dot_f32(&a, &b), 32.0, 1e-4));
    }

    #[test]
    fn test_dot_f32_exactly_4_elements() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [4.0f32, 3.0, 2.0, 1.0];
        // 4 + 6 + 6 + 4 = 20
        assert!(approx_eq(portable_dot_f32(&a, &b), 20.0, 1e-4));
    }

    #[test]
    fn test_dot_f32_more_than_4_elements() {
        // 5 elements to exercise tail handling
        let a = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        assert!(approx_eq(portable_dot_f32(&a, &b), 10.0, 1e-4));
    }

    #[test]
    fn test_dot_f32_orthogonal_vectors() {
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0];
        assert!(approx_eq(portable_dot_f32(&a, &b), 0.0, 1e-9));
    }

    #[test]
    fn test_dot_f32_mismatched_lengths_uses_shorter() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 1.0, 1.0]; // shorter
                                    // Only first 3 used: 1+2+3 = 6
        assert!(approx_eq(portable_dot_f32(&a, &b), 6.0, 1e-4));
    }

    #[test]
    fn test_dot_f32_large_input_consistency() {
        // 33 elements (non-multiple of 4) — compare with naive sum
        let a: Vec<f32> = (0..33).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..33).map(|i| (33 - i) as f32).collect();
        let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = portable_dot_f32(&a, &b);
        assert!(approx_eq(result, naive, 0.5)); // small floating-point reorder allowed
    }

    #[test]
    fn test_dot_f32_negative_values() {
        let a = [-1.0f32, -2.0, 3.0, 4.0];
        let b = [1.0f32, 1.0, -1.0, -1.0];
        // -1 + -2 + -3 + -4 = -10
        assert!(approx_eq(portable_dot_f32(&a, &b), -10.0, 1e-4));
    }

    // ── portable_matmul_4x4 ──────────────────────────────────────────────────

    #[test]
    fn test_matmul_identity_times_identity() {
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let result = portable_matmul_4x4(&identity, &identity);
        for (a, b) in result.iter().zip(identity.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_matmul_identity_times_matrix() {
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let m: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let result = portable_matmul_4x4(&identity, &m);
        for (a, b) in result.iter().zip(m.iter()) {
            assert!(approx_eq(*a, *b, 1e-5));
        }
    }

    #[test]
    fn test_matmul_matrix_times_identity() {
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let m: [f32; 16] = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ];
        let result = portable_matmul_4x4(&m, &identity);
        for (a, b) in result.iter().zip(m.iter()) {
            assert!(approx_eq(*a, *b, 1e-5));
        }
    }

    #[test]
    fn test_matmul_known_result() {
        // Rotation-like matrix multiply with known result
        let a: [f32; 16] = [
            1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        // A*B: row0 = [1*1+2*3, 1*0+2*1, 0, 0] = [7, 2, 0, 0]
        let result = portable_matmul_4x4(&a, &b);
        assert!(approx_eq(result[0], 7.0, 1e-5));
        assert!(approx_eq(result[1], 2.0, 1e-5));
        assert!(approx_eq(result[2], 0.0, 1e-5));
        assert!(approx_eq(result[3], 0.0, 1e-5));
    }

    #[test]
    fn test_matmul_zero_matrix() {
        let zero = [0.0f32; 16];
        let m: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let result = portable_matmul_4x4(&zero, &m);
        for &v in &result {
            assert!(approx_eq(v, 0.0, 1e-9));
        }
    }

    // ── portable_rgba_to_luma ────────────────────────────────────────────────

    #[test]
    fn test_luma_empty_input() {
        let out = portable_rgba_to_luma(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_luma_black_pixel() {
        // R=0, G=0, B=0 → Y = (0 + 0 + 0 + 128) >> 8 + 16 = 0 + 16 = 16
        let rgba = [0u8, 0, 0, 255];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma.len(), 1);
        assert_eq!(luma[0], 16);
    }

    #[test]
    fn test_luma_white_pixel() {
        // R=255, G=255, B=255 → Y = (66*255 + 129*255 + 25*255 + 128) >> 8 + 16
        // = (16830 + 32895 + 6375 + 128) >> 8 + 16 = 56228 >> 8 + 16 = 219 + 16 = 235
        let rgba = [255u8, 255, 255, 255];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma.len(), 1);
        assert_eq!(luma[0], 235);
    }

    #[test]
    fn test_luma_pure_red() {
        // R=255, G=0, B=0 → Y = (66*255 + 0 + 0 + 128) >> 8 + 16 = 16958 >> 8 + 16 = 66 + 16 = 82
        let rgba = [255u8, 0, 0, 255];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma[0], 82);
    }

    #[test]
    fn test_luma_pure_green() {
        // R=0, G=255, B=0 → Y = (0 + 129*255 + 0 + 128) >> 8 + 16
        //   = 33023 >> 8 + 16 = 128 + 16 = 144  (33023 = 128*256 + 255, truncates to 128)
        let rgba = [0u8, 255, 0, 255];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma[0], 144);
    }

    #[test]
    fn test_luma_pure_blue() {
        // R=0, G=0, B=255 → Y = (0 + 0 + 25*255 + 128) >> 8 + 16 = 6503 >> 8 + 16 = 25 + 16 = 41
        let rgba = [0u8, 0, 255, 255];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma[0], 41);
    }

    #[test]
    fn test_luma_four_pixels_4wide_path() {
        // Exactly 4 pixels — exercises the 4-wide unrolled path with no tail.
        let rgba: Vec<u8> = (0..4)
            .flat_map(|i| [i * 60, i * 40, i * 20, 255u8])
            .collect();
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma.len(), 4);
        // Just verify monotonicity (brighter pixels → higher Y).
        for w in luma.windows(2) {
            assert!(w[1] >= w[0], "luma should be non-decreasing: {:?}", luma);
        }
    }

    #[test]
    fn test_luma_five_pixels_with_tail() {
        // 5 pixels: 4 in the SIMD path + 1 in the scalar tail.
        let rgba = vec![
            128u8, 128, 128, 255, 0, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255,
        ];
        let luma = portable_rgba_to_luma(&rgba);
        assert_eq!(luma.len(), 5);
        assert!(luma.iter().all(|&v| v >= 16 && v <= 235));
    }

    #[test]
    fn test_luma_alpha_ignored() {
        // Same RGB, different alpha — luma must be identical.
        let a = [100u8, 150, 200, 0];
        let b = [100u8, 150, 200, 255];
        let la = portable_rgba_to_luma(&a);
        let lb = portable_rgba_to_luma(&b);
        assert_eq!(la, lb);
    }

    #[test]
    fn test_luma_limited_range_bounds() {
        // All outputs must be in [16, 235] for limited-range BT.601.
        let test_pixels: &[(u8, u8, u8)] = &[
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (255, 128, 0),
        ];
        for &(r, g, b) in test_pixels {
            let rgba = [r, g, b, 255];
            let luma = portable_rgba_to_luma(&rgba);
            let y = luma[0];
            assert!(
                y >= 16 && y <= 235,
                "Y={y} out of limited range for R={r} G={g} B={b}"
            );
        }
    }
}
