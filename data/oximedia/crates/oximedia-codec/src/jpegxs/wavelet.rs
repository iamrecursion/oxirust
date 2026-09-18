// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! LeGall 5/3 wavelet for JPEG XS.
//!
//! JPEG XS uses the same reversible LeGall 5/3 integer lifting transform as
//! JPEG 2000 lossless mode (ISO 15444-1 §F.3.2), but this implementation is
//! self-contained so the `jpegxs` feature does not depend on `jpeg2000`.
//!
//! # Lifting steps (inverse 5/3)
//!
//! Given the low-pass subband L (even-indexed) and high-pass subband H
//! (odd-indexed) from a single decomposition level:
//!
//! 1. **Inverse update** (undo the update step):
//!    `L[n] -= floor((H[n-1] + H[n] + 2) / 4)`
//!    (edge: `H[-1] = H[0]`)
//!
//! 2. **Inverse predict** (undo the predict step):
//!    `H[n] += floor((L[n] + L[n+1]) / 2)`
//!    (edge: `L[n_low] = L[n_low - 1]`)
//!
//! 3. **Interleave**: `x[2n] = L[n]`, `x[2n+1] = H[n]`.

use super::JxsError;

/// 5/3 inverse 1D wavelet transform (LeGall lifting, integer arithmetic).
///
/// Reconstructs a signal from its low-pass (`low`) and high-pass (`high`)
/// subbands.  Output length = `low.len() + high.len()`.
///
/// This is the JPEG XS (and JPEG 2000 lossless) reconstruction path.
pub fn inverse_53_1d(low: &[i32], high: &[i32]) -> Vec<i32> {
    let n_low = low.len();
    let n_high = high.len();
    let n = n_low + n_high;

    if n_low == 0 {
        return Vec::new();
    }

    let mut s = low.to_vec();
    let mut d = high.to_vec();

    // Step 1: Inverse update — undo L[n] += floor((H[n-1] + H[n] + 2) / 4)
    // Forward update used H (= predicted H, not yet changed).
    // Inverse: L_orig = L_updated - floor((H[n-1] + H[n] + 2) / 4)
    // Edge: H[-1] = H[0]
    for n_idx in 0..n_low {
        let d_prev = if n_idx == 0 {
            if n_high > 0 {
                d[0]
            } else {
                0
            }
        } else {
            d[n_idx - 1]
        };
        let d_curr = if n_idx < n_high {
            d[n_idx]
        } else if n_high > 0 {
            d[n_high - 1]
        } else {
            0
        };
        s[n_idx] -= (d_prev + d_curr + 2) >> 2;
    }

    // Step 2: Inverse predict — undo H[n] -= floor((L[n] + L[n+1]) / 2)
    // Forward predict: H_pred = H_orig - floor((L_orig[n] + L_orig[n+1]) / 2)
    // Inverse: H_orig = H_pred + floor((L_orig[n] + L_orig[n+1]) / 2)
    // Edge: L[n_low] = L[n_low - 1]
    for n_idx in 0..n_high {
        let s_curr = s[n_idx];
        let s_next = if n_idx + 1 < n_low {
            s[n_idx + 1]
        } else {
            s[n_low - 1]
        };
        d[n_idx] += (s_curr + s_next) >> 1;
    }

    // Interleave: output[2n] = s[n], output[2n+1] = d[n]
    let mut out = vec![0i32; n];
    for i in 0..n_low {
        out[2 * i] = s[i];
    }
    for i in 0..n_high {
        out[2 * i + 1] = d[i];
    }
    out
}

/// 5/3 forward 1D wavelet transform (for test round-trips only).
///
/// Splits `signal` into `(low, high)` subbands using the forward lifting steps:
/// 1. Predict: `H[n] -= floor((L[n] + L[n+1]) / 2)`
/// 2. Update:  `L[n] += floor((H[n-1] + H[n] + 2) / 4)`
pub fn forward_53_1d(signal: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![signal[0]], Vec::new());
    }

    let n_low = (n + 1) / 2;
    let n_high = n / 2;

    let mut low: Vec<i32> = signal.iter().step_by(2).copied().collect();
    let mut high: Vec<i32> = signal.iter().skip(1).step_by(2).copied().collect();

    // Forward predict: H[n] -= floor((L[n] + L[n+1]) / 2)
    for i in 0..n_high {
        let l_curr = low[i];
        let l_next = if i + 1 < n_low {
            low[i + 1]
        } else {
            low[n_low - 1]
        };
        high[i] -= (l_curr + l_next) >> 1;
    }

    // Forward update: L[n] += floor((H[n-1] + H[n] + 2) / 4)
    for i in 0..n_low {
        let h_prev = if i == 0 {
            if n_high > 0 {
                high[0]
            } else {
                0
            }
        } else {
            high[i - 1]
        };
        let h_curr = if i < n_high {
            high[i]
        } else if n_high > 0 {
            high[n_high - 1]
        } else {
            0
        };
        low[i] += (h_prev + h_curr + 2) >> 2;
    }

    (low, high)
}

/// 5/3 inverse 2D wavelet transform for one decomposition level.
///
/// Reconstructs an image of `width × height` pixels from the four subbands
/// produced by a single level of 2D decomposition:
///
/// ```text
/// | LL | HL |   (low rows,  low cols | high cols)
/// | LH | HH |   (high rows, low cols | high cols)
/// ```
///
/// - `ll` has dimensions `ceil(height/2) × ceil(width/2)` (row-major)
/// - `hl` has dimensions `ceil(height/2) × floor(width/2)`
/// - `lh` has dimensions `floor(height/2) × ceil(width/2)`
/// - `hh` has dimensions `floor(height/2) × floor(width/2)`
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if any subband is too small.
pub fn inverse_53_2d(
    ll: &[i32],
    hl: &[i32],
    lh: &[i32],
    hh: &[i32],
    width: usize,
    height: usize,
) -> Result<Vec<i32>, JxsError> {
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let n_low_w = (width + 1) / 2; // ceil(width/2)   — low-pass columns
    let n_high_w = width / 2; // floor(width/2)  — high-pass columns
    let n_low_h = (height + 1) / 2; // ceil(height/2)  — low-pass rows
    let n_high_h = height / 2; // floor(height/2) — high-pass rows

    // Sanity-check subband sizes.
    let exp_ll = n_low_h * n_low_w;
    let exp_hl = n_low_h * n_high_w;
    let exp_lh = n_high_h * n_low_w;
    let exp_hh = n_high_h * n_high_w;

    if ll.len() < exp_ll {
        return Err(JxsError::InvalidHeader(format!(
            "LL subband too small: expected {exp_ll}, got {}",
            ll.len()
        )));
    }
    if hl.len() < exp_hl {
        return Err(JxsError::InvalidHeader(format!(
            "HL subband too small: expected {exp_hl}, got {}",
            hl.len()
        )));
    }
    if lh.len() < exp_lh {
        return Err(JxsError::InvalidHeader(format!(
            "LH subband too small: expected {exp_lh}, got {}",
            lh.len()
        )));
    }
    if hh.len() < exp_hh {
        return Err(JxsError::InvalidHeader(format!(
            "HH subband too small: expected {exp_hh}, got {}",
            hh.len()
        )));
    }

    // Step 1: Horizontal inverse 1D on each row of the low-row and high-row blocks.
    // Low-row block (from LL + HL): n_low_h rows of full width.
    let mut low_rows = vec![0i32; n_low_h * width];
    for row in 0..n_low_h {
        let l_row = &ll[row * n_low_w..(row + 1) * n_low_w];
        let h_row = if n_high_w > 0 {
            &hl[row * n_high_w..(row + 1) * n_high_w]
        } else {
            &[]
        };
        let reconstructed = inverse_53_1d(l_row, h_row);
        let copy_len = reconstructed.len().min(width);
        low_rows[row * width..row * width + copy_len].copy_from_slice(&reconstructed[..copy_len]);
    }

    // High-row block (from LH + HH): n_high_h rows of full width.
    let mut high_rows = vec![0i32; n_high_h * width];
    for row in 0..n_high_h {
        let l_row = &lh[row * n_low_w..(row + 1) * n_low_w];
        let h_row = if n_high_w > 0 {
            &hh[row * n_high_w..(row + 1) * n_high_w]
        } else {
            &[]
        };
        let reconstructed = inverse_53_1d(l_row, h_row);
        let copy_len = reconstructed.len().min(width);
        high_rows[row * width..row * width + copy_len].copy_from_slice(&reconstructed[..copy_len]);
    }

    // Step 2: Vertical inverse 1D on each column of the merged row buffers.
    let mut output = vec![0i32; height * width];
    for col in 0..width {
        let low_col: Vec<i32> = (0..n_low_h).map(|r| low_rows[r * width + col]).collect();
        let high_col: Vec<i32> = (0..n_high_h).map(|r| high_rows[r * width + col]).collect();
        let col_out = inverse_53_1d(&low_col, &high_col);
        let copy_len = col_out.len().min(height);
        for (row, &val) in col_out[..copy_len].iter().enumerate() {
            output[row * width + col] = val;
        }
    }

    Ok(output)
}

/// 5/3 forward 1D wavelet transform (LeGall lifting, integer arithmetic).
///
/// This is a thin alias of [`forward_53_1d`] under the task's naming convention.
/// It is the exact inverse of [`inverse_53_1d`]: for any integer `signal`,
/// `inverse_53_1d(forward_wavelet_1d(signal)) == signal`.
pub fn forward_wavelet_1d(signal: &[i32]) -> (Vec<i32>, Vec<i32>) {
    forward_53_1d(signal)
}

/// 5/3 forward 2D wavelet transform for one decomposition level.
///
/// Produces the four subbands `(ll, hl, lh, hh)` from a `width × height`
/// row-major image. This is the **exact inverse** of [`inverse_53_2d`]: it
/// applies the lifting steps in the reverse order, so
/// `inverse_53_2d(forward_wavelet_2d(image, w, h), w, h) == image` byte-for-byte
/// on integer input.
///
/// Subband dimensions match [`inverse_53_2d`]:
/// - `ll`: `ceil(height/2) × ceil(width/2)`
/// - `hl`: `ceil(height/2) × floor(width/2)`
/// - `lh`: `floor(height/2) × ceil(width/2)`
/// - `hh`: `floor(height/2) × floor(width/2)`
///
/// # Errors
/// Returns `JxsError::InvalidHeader` if `image.len() < width * height`.
pub fn forward_wavelet_2d(
    image: &[i32],
    width: usize,
    height: usize,
) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>), JxsError> {
    if width == 0 || height == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    if image.len() < width * height {
        return Err(JxsError::InvalidHeader(format!(
            "forward_wavelet_2d: image too small, expected {}, got {}",
            width * height,
            image.len()
        )));
    }

    let n_low_w = (width + 1) / 2; // ceil(width/2)
    let n_high_w = width / 2; // floor(width/2)
    let n_low_h = (height + 1) / 2; // ceil(height/2)
    let n_high_h = height / 2; // floor(height/2)

    // ── Step 1: vertical forward 1D on each column ──────────────────────────
    // Splits each column into low (n_low_h) and high (n_high_h) rows. This is
    // the reverse of the inverse transform's final (vertical) step.
    let mut low_rows = vec![0i32; n_low_h * width];
    let mut high_rows = vec![0i32; n_high_h * width];
    for col in 0..width {
        let col_vals: Vec<i32> = (0..height).map(|r| image[r * width + col]).collect();
        let (low, high) = forward_53_1d(&col_vals);
        for (r, &v) in low.iter().enumerate().take(n_low_h) {
            low_rows[r * width + col] = v;
        }
        for (r, &v) in high.iter().enumerate().take(n_high_h) {
            high_rows[r * width + col] = v;
        }
    }

    // ── Step 2: horizontal forward 1D on each row ───────────────────────────
    // Low-row block → (LL | HL); high-row block → (LH | HH). This is the
    // reverse of the inverse transform's first (horizontal) step.
    let mut ll = vec![0i32; n_low_h * n_low_w];
    let mut hl = vec![0i32; n_low_h * n_high_w];
    for row in 0..n_low_h {
        let r = &low_rows[row * width..(row + 1) * width];
        let (low, high) = forward_53_1d(r);
        for (c, &v) in low.iter().enumerate().take(n_low_w) {
            ll[row * n_low_w + c] = v;
        }
        for (c, &v) in high.iter().enumerate().take(n_high_w) {
            hl[row * n_high_w + c] = v;
        }
    }

    let mut lh = vec![0i32; n_high_h * n_low_w];
    let mut hh = vec![0i32; n_high_h * n_high_w];
    for row in 0..n_high_h {
        let r = &high_rows[row * width..(row + 1) * width];
        let (low, high) = forward_53_1d(r);
        for (c, &v) in low.iter().enumerate().take(n_low_w) {
            lh[row * n_low_w + c] = v;
        }
        for (c, &v) in high.iter().enumerate().take(n_high_w) {
            hh[row * n_high_w + c] = v;
        }
    }

    Ok((ll, hl, lh, hh))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d_even_length() {
        let orig = vec![10i32, 20, 30, 40, 50, 60, 70, 80];
        let (low, high) = forward_53_1d(&orig);
        let recovered = inverse_53_1d(&low, &high);
        assert_eq!(
            recovered, orig,
            "1D round-trip failed for even-length signal"
        );
    }

    #[test]
    fn roundtrip_1d_odd_length() {
        let orig = vec![3i32, 1, 4, 1, 5, 9, 2];
        let (low, high) = forward_53_1d(&orig);
        let recovered = inverse_53_1d(&low, &high);
        assert_eq!(
            recovered, orig,
            "1D round-trip failed for odd-length signal"
        );
    }

    #[test]
    fn roundtrip_1d_single_element() {
        let orig = vec![42i32];
        let (low, high) = forward_53_1d(&orig);
        assert_eq!(high.len(), 0);
        let recovered = inverse_53_1d(&low, &high);
        assert_eq!(recovered, orig);
    }

    #[test]
    fn roundtrip_1d_constant_signal_has_zero_high() {
        let orig = vec![128i32; 8];
        let (low, high) = forward_53_1d(&orig);
        for &h in &high {
            assert_eq!(h, 0, "detail coefficients must be zero for constant signal");
        }
        let recovered = inverse_53_1d(&low, &high);
        assert_eq!(recovered, orig);
    }

    #[test]
    fn roundtrip_53_1d_constant() {
        // Simple roundtrip for constant-valued signal.
        let orig: Vec<i32> = vec![64; 16];
        let (low, high) = forward_53_1d(&orig);
        let rec = inverse_53_1d(&low, &high);
        assert_eq!(rec, orig);
    }

    #[test]
    fn roundtrip_53_2d_constant() {
        // A constant 4×4 image with all detail subbands zero should reconstruct perfectly.
        let n = 4usize;
        let constant = 64i32;

        let n_low_w = (n + 1) / 2;
        let n_high_w = n / 2;
        let n_low_h = (n + 1) / 2;
        let n_high_h = n / 2;

        // Build the true LL from the forward transform of a constant image.
        let image = vec![constant; n * n];
        let mut ll_rows = vec![0i32; n_low_h * n_low_w];
        let mut hl_rows = vec![0i32; n_low_h * n_high_w];

        // Horizontal forward transform on each row.
        let mut row_low = vec![vec![0i32; n_low_w]; n];
        let mut row_high = vec![vec![0i32; n_high_w]; n];
        for row in 0..n {
            let (l, h) = forward_53_1d(&image[row * n..(row + 1) * n]);
            row_low[row] = l;
            row_high[row] = h;
        }

        // Vertical forward transform on low columns.
        for col in 0..n_low_w {
            let col_vals: Vec<i32> = (0..n).map(|r| row_low[r][col]).collect();
            let (l, _h) = forward_53_1d(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll_rows[r * n_low_w + col] = v;
            }
        }
        // Vertical forward transform on high columns.
        for col in 0..n_high_w {
            let col_vals: Vec<i32> = (0..n).map(|r| row_high[r][col]).collect();
            let (l, _h) = forward_53_1d(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                hl_rows[r * n_high_w + col] = v;
            }
        }

        // All-zero LH and HH for constant image.
        let lh = vec![0i32; n_high_h * n_low_w];
        let hh = vec![0i32; n_high_h * n_high_w];

        let result = inverse_53_2d(&ll_rows, &hl_rows, &lh, &hh, n, n).unwrap();
        assert_eq!(result.len(), n * n);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, constant, "sample {i}: expected {constant}, got {v}");
        }
    }

    #[test]
    fn inverse_53_2d_subband_too_small_returns_error() {
        let ll = vec![0i32; 1]; // too small for 4×4
        let hl = vec![0i32; 4];
        let lh = vec![0i32; 4];
        let hh = vec![0i32; 4];
        let result = inverse_53_2d(&ll, &hl, &lh, &hh, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn inverse_53_2d_zero_dimensions_returns_empty() {
        let result = inverse_53_2d(&[], &[], &[], &[], 0, 0).unwrap();
        assert!(result.is_empty());
    }

    /// Forward then inverse 2D must reproduce the input exactly (integer lossless).
    fn assert_fwd_inv_identity(image: &[i32], w: usize, h: usize) {
        let (ll, hl, lh, hh) = forward_wavelet_2d(image, w, h).unwrap();
        let rec = inverse_53_2d(&ll, &hl, &lh, &hh, w, h).unwrap();
        assert_eq!(rec, image, "forward→inverse 2D not identity for {w}x{h}");
    }

    #[test]
    fn forward_inverse_2d_constant() {
        assert_fwd_inv_identity(&vec![64i32; 16], 4, 4);
    }

    #[test]
    fn forward_inverse_2d_gradient_even() {
        let (w, h) = (8usize, 8usize);
        let img: Vec<i32> = (0..w * h).map(|i| (i % w + i / w) as i32).collect();
        assert_fwd_inv_identity(&img, w, h);
    }

    #[test]
    fn forward_inverse_2d_gradient_odd_dims() {
        let (w, h) = (7usize, 5usize);
        let img: Vec<i32> = (0..w * h).map(|i| (3 * (i % w) + (i / w)) as i32).collect();
        assert_fwd_inv_identity(&img, w, h);
    }

    #[test]
    fn forward_inverse_2d_pseudo_random() {
        let (w, h) = (16usize, 9usize);
        // Deterministic pseudo-random integers in [0, 255].
        let mut state: u32 = 0x1234_5678;
        let img: Vec<i32> = (0..w * h)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 24) & 0xFF) as i32
            })
            .collect();
        assert_fwd_inv_identity(&img, w, h);
    }

    #[test]
    fn forward_wavelet_1d_matches_forward_53_1d() {
        let sig = vec![10i32, 20, 30, 40, 55, 12, 7, 99];
        assert_eq!(forward_wavelet_1d(&sig), forward_53_1d(&sig));
    }

    #[test]
    fn forward_wavelet_2d_too_small_errors() {
        let img = vec![1i32; 3];
        assert!(forward_wavelet_2d(&img, 4, 4).is_err());
    }
}
