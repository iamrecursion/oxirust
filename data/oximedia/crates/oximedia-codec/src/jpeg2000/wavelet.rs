//! 5-3 reversible (lossless) and 9-7 irreversible (lossy) 1D and 2D inverse
//! discrete wavelet transforms.
//!
//! Implements:
//! - Integer lifting steps from ISO/IEC 15444-1 §F.3.2 (inverse 5-3 WT, lossless)
//! - CDF 9/7 inverse lifting from ISO/IEC 15444-1 §F.3.1 (irreversible, lossy)
//!
//! The forward transforms are also provided for test round-trip validation.
//!
//! ## Lifting steps (inverse 5-3)
//!
//! Given L (low / even-indexed) and H (high / odd-indexed) subbands of length
//! n_l and n_h respectively:
//!
//! 1. **Inverse predict**: `H[n] += floor((L[n] + L[n+1]) / 2)`
//!    (edge: `L[-1] = L[0]`, `L[n_l] = L[n_l - 1]`)
//! 2. **Inverse update**: `L[n] -= floor((H[n-1] + H[n] + 2) / 4)`
//!    (edge: `H[-1] = H[0]`)
//!
//! ## CDF 9/7 inverse lifting (irreversible, floating-point)
//!
//! Uses the Daubechies/Cohen/Feauveau 9/7 filter with lifting factorisation from
//! ISO/IEC 15444-1 §F.3.1.  The filter constants are (alpha, beta, gamma, delta, K).
//! The inverse applies four lifting steps in reverse order with symmetric boundary
//! extension, then rescales and interleaves the low/high samples.

use super::{Jp2Error, Jp2Result};

// ── Wavelet kind enum ─────────────────────────────────────────────────────────

/// Selects the wavelet filter family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletKind {
    /// 5-3 integer lifting (reversible, lossless).
    Reversible53,
    /// CDF 9/7 floating-point lifting (irreversible, lossy).
    Irreversible97,
}

// ── CDF 9/7 lifting constants ─────────────────────────────────────────────────

/// CDF 9/7 lifting step α (predict 1).
const CDF97_ALPHA: f64 = -1.586_134_342_059_924;
/// CDF 9/7 lifting step β (update 1).
const CDF97_BETA: f64 = -0.052_980_118_572_961;
/// CDF 9/7 lifting step γ (predict 2).
const CDF97_GAMMA: f64 = 0.882_911_075_530_934;
/// CDF 9/7 lifting step δ (update 2).
const CDF97_DELTA: f64 = 0.443_506_852_043_971;
/// CDF 9/7 normalisation factor for low-pass samples.
const CDF97_K: f64 = 1.230_174_104_914_001;
/// CDF 9/7 normalisation factor for high-pass samples (= 1/K).
const CDF97_K_INV: f64 = 0.812_893_066_115_961;

/// Tree of subbands for one decomposition level.
/// Each level stores HL, LH, HH subbands; the LL band from the top level
/// is the coarsest approximation.
#[derive(Debug, Clone)]
pub struct SubbandLevel {
    /// Horizontal low / vertical high (HL) subband — rows low-pass, cols high-pass.
    pub hl: Vec<i32>,
    /// Horizontal high / vertical low (LH) subband — rows high-pass, cols low-pass.
    pub lh: Vec<i32>,
    /// Horizontal high / vertical high (HH) subband — all high-pass.
    pub hh: Vec<i32>,
    /// Width of each subband.
    pub width: usize,
    /// Height of each subband.
    pub height: usize,
}

/// Complete subband tree for all decomposition levels.
#[derive(Debug, Clone)]
pub struct SubbandTree {
    /// Coarsest LL (DC) subband.
    pub ll: Vec<i32>,
    /// Width of the LL subband.
    pub ll_width: usize,
    /// Height of the LL subband.
    pub ll_height: usize,
    /// Detail subbands, from coarsest (index 0) to finest (index n-1).
    pub levels: Vec<SubbandLevel>,
}

// ── 1D transforms ─────────────────────────────────────────────────────────────

/// 5-3 forward 1D transform (used for test round-trips only).
///
/// Splits `signal` into low (even) and high (odd) subbands in-place using
/// the forward lifting:
/// 1. `H[n] -= floor((L[n] + L[n+1]) / 2)`  (predict)
/// 2. `L[n] += floor((H[n-1] + H[n] + 2) / 4)` (update)
///
/// Returns `(low, high)`.
#[must_use]
pub fn forward_53(signal: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![signal[0]], Vec::new());
    }

    let n_l = (n + 1) / 2;
    let n_h = n / 2;

    let mut low: Vec<i32> = signal.iter().step_by(2).copied().collect();
    let mut high: Vec<i32> = signal.iter().skip(1).step_by(2).copied().collect();

    // Predict step: H[n] -= floor((L[n] + L[n+1]) / 2)
    for n in 0..n_h {
        let l_n = low[n];
        let l_n1 = if n + 1 < n_l {
            low[n + 1]
        } else {
            low[n_l - 1]
        };
        high[n] -= (l_n + l_n1) >> 1;
    }

    // Update step: L[n] += floor((H[n-1] + H[n] + 2) / 4)
    for n in 0..n_l {
        let h_prev = if n > 0 {
            high[n - 1]
        } else {
            if n_h > 0 {
                high[0]
            } else {
                0
            }
        };
        let h_n = if n < n_h {
            high[n]
        } else {
            if n_h > 0 {
                high[n_h - 1]
            } else {
                0
            }
        };
        low[n] += (h_prev + h_n + 2) >> 2;
    }

    (low, high)
}

/// 5-3 inverse 1D wavelet transform (ISO 15444-1 §F.3.2).
///
/// Reconstructs a signal from its low-pass (`low`) and high-pass (`high`)
/// subband coefficients using the 5-3 integer lifting steps.
///
/// ## Step order (inverse = reverse of forward)
///
/// The forward 5-3 lifting is:
/// 1. Predict: `H[n] -= floor((L[n] + L[n+1]) / 2)`
/// 2. Update:  `L[n] += floor((H[n-1] + H[n] + 2) / 4)` (uses predicted H)
///
/// The inverse must reverse the order and invert each step:
/// 1. Inverse update:  `L_orig[n] = L[n] - floor((H[n-1] + H[n] + 2) / 4)` (H = predicted H, unchanged)
/// 2. Inverse predict: `H_orig[n] = H[n] + floor((L_orig[n] + L_orig[n+1]) / 2)` (uses recovered L)
#[must_use]
pub fn inverse_wavelet_1d(low: &[i32], high: &[i32]) -> Vec<i32> {
    let n_l = low.len();
    let n_h = high.len();
    if n_l == 0 {
        return Vec::new();
    }

    let mut s = low.to_vec(); // low (even samples, stored predicted)
    let mut d = high.to_vec(); // high (odd samples, stored predicted by forward predict)

    // Step 1: Inverse update — recover L_orig from L (the updated low).
    // Uses d (= the predicted H, which has not been changed yet).
    // Forward update: L_updated = L_orig + floor((H_pred[n-1] + H_pred[n] + 2) / 4)
    // Inverse:        L_orig = L_updated - floor((H_pred[n-1] + H_pred[n] + 2) / 4)
    // Edge: H_pred[-1] = H_pred[0]
    for n in 0..n_l {
        let d_prev = if n > 0 {
            d[n - 1]
        } else if n_h > 0 {
            d[0]
        } else {
            0
        };
        let d_n = if n < n_h {
            d[n]
        } else if n_h > 0 {
            d[n_h - 1]
        } else {
            0
        };
        s[n] -= (d_prev + d_n + 2) >> 2;
    }

    // Step 2: Inverse predict — recover H_orig from H_pred.
    // Forward predict: H_pred = H_orig - floor((L_orig[n] + L_orig[n+1]) / 2)
    // Inverse:         H_orig = H_pred + floor((L_orig[n] + L_orig[n+1]) / 2)
    // Edge: L_orig[n_l] = L_orig[n_l - 1]
    for n in 0..n_h {
        let s_n = s[n];
        let s_n1 = if n + 1 < n_l { s[n + 1] } else { s[n_l - 1] };
        d[n] += (s_n + s_n1) >> 1;
    }

    // Interleave: x[2n] = s[n], x[2n+1] = d[n]
    let total = n_l + n_h;
    let mut result = vec![0i32; total];
    for n in 0..n_l {
        result[2 * n] = s[n];
    }
    for n in 0..n_h {
        result[2 * n + 1] = d[n];
    }
    result
}

/// 5-3 inverse 2D wavelet transform for a single decomposition level.
///
/// Given the four subbands from a single decomposition level (LL, HL, LH, HH),
/// reconstructs the output plane of dimensions `width × height`.
///
/// Subband layout (LL is the previous level's output):
///
/// ```text
/// | LL | HL |
/// | LH | HH |
/// ```
///
/// - `ll`  is `ceil(height/2) × ceil(width/2)` samples (row-major)
/// - `hl`  is `ceil(height/2) × floor(width/2)` samples
/// - `lh`  is `floor(height/2) × ceil(width/2)` samples
/// - `hh`  is `floor(height/2) × floor(width/2)` samples
pub fn inverse_wavelet_2d(
    ll: &[i32],
    hl: &[i32],
    lh: &[i32],
    hh: &[i32],
    width: usize,
    height: usize,
) -> Jp2Result<Vec<i32>> {
    let n_l_h = (width + 1) / 2; // ceil(width/2)   — low cols
    let n_h_h = width / 2; // floor(width/2)  — high cols
    let n_l_v = (height + 1) / 2; // ceil(height/2)  — low rows
    let n_h_v = height / 2; // floor(height/2) — high rows

    // Sanity check subband sizes.
    let expected_ll = n_l_v * n_l_h;
    let expected_hl = n_l_v * n_h_h;
    let expected_lh = n_h_v * n_l_h;
    let expected_hh = n_h_v * n_h_h;

    if ll.len() < expected_ll {
        return Err(Jp2Error::InternalError(format!(
            "LL subband too small: expected {expected_ll}, got {}",
            ll.len()
        )));
    }
    if hl.len() < expected_hl {
        return Err(Jp2Error::InternalError(format!(
            "HL subband too small: expected {expected_hl}, got {}",
            hl.len()
        )));
    }
    if lh.len() < expected_lh {
        return Err(Jp2Error::InternalError(format!(
            "LH subband too small: expected {expected_lh}, got {}",
            lh.len()
        )));
    }
    if hh.len() < expected_hh {
        return Err(Jp2Error::InternalError(format!(
            "HH subband too small: expected {expected_hh}, got {}",
            hh.len()
        )));
    }

    // Step 1: Apply horizontal inverse wavelet to each row of the low-row and high-row bands.
    // This gives us a width×n_l_v low-row intermediate and a width×n_h_v high-row intermediate.

    let mut low_rows: Vec<i32> = vec![0; n_l_v * width]; // low rows, full width
    for row in 0..n_l_v {
        let l = &ll[row * n_l_h..(row + 1) * n_l_h];
        let h = &hl[row * n_h_h..(row + 1) * n_h_h];
        let reconstructed = inverse_wavelet_1d(l, h);
        let out_len = reconstructed.len().min(width);
        low_rows[row * width..row * width + out_len].copy_from_slice(&reconstructed[..out_len]);
    }

    let mut high_rows: Vec<i32> = vec![0; n_h_v * width]; // high rows, full width
    for row in 0..n_h_v {
        let l = &lh[row * n_l_h..(row + 1) * n_l_h];
        let h = &hh[row * n_h_h..(row + 1) * n_h_h];
        let reconstructed = inverse_wavelet_1d(l, h);
        let out_len = reconstructed.len().min(width);
        high_rows[row * width..row * width + out_len].copy_from_slice(&reconstructed[..out_len]);
    }

    // Step 2: Apply vertical inverse wavelet to each column.
    let mut output = vec![0i32; height * width];
    for col in 0..width {
        let low_col: Vec<i32> = (0..n_l_v).map(|r| low_rows[r * width + col]).collect();
        let high_col: Vec<i32> = (0..n_h_v).map(|r| high_rows[r * width + col]).collect();
        let col_out = inverse_wavelet_1d(&low_col, &high_col);
        let out_len = col_out.len().min(height);
        for (row, &val) in col_out[..out_len].iter().enumerate() {
            output[row * width + col] = val;
        }
    }

    Ok(output)
}

// ── CDF 9/7 inverse wavelet ────────────────────────────────────────────────────

/// Symmetric boundary extension: return the value at index `i` in a slice of
/// length `n`, mirroring at both ends (whole-sample symmetric extension).
///
/// For `n = 6`: valid indices 0..5, mirror: -1→0, -2→1, n→n-1, n+1→n-2, …
#[inline]
fn sym_ext(buf: &[f64], i: i64) -> f64 {
    let n = buf.len() as i64;
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return buf[0];
    }
    // Reflect: period = 2*(n-1)
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j > n - 1 {
        j = period - j;
    }
    buf[j as usize]
}

/// CDF 9/7 inverse 1D wavelet transform.
///
/// Takes the low-pass subband `low` and high-pass subband `high` (each of
/// length `n_l` and `n_h` respectively, where `n_l = ceil(n/2)`,
/// `n_h = floor(n/2)` for output length `n = n_l + n_h`), and reconstructs
/// the full-resolution signal using the four-step inverse lifting scheme from
/// ISO/IEC 15444-1 §F.3.1.
///
/// All arithmetic uses `f64` for the intermediate precision required by the
/// 9/7 irreversible filter.
#[must_use]
pub fn inverse_wavelet_1d_97(low: &[f64], high: &[f64]) -> Vec<f64> {
    let n_l = low.len();
    let n_h = high.len();
    if n_l == 0 {
        return Vec::new();
    }

    // De-scale: s[n] *= K, d[n] *= 1/K
    let mut s: Vec<f64> = low.iter().map(|&v| v * CDF97_K).collect();
    let mut d: Vec<f64> = high.iter().map(|&v| v * CDF97_K_INV).collect();

    // Inverse update-δ: s[n] -= δ * (d[n-1] + d[n])
    for n in 0..n_l {
        let d_prev = sym_ext(&d, n as i64 - 1);
        let d_n = sym_ext(&d, n as i64);
        s[n] -= CDF97_DELTA * (d_prev + d_n);
    }

    // Inverse predict-γ: d[n] -= γ * (s[n] + s[n+1])
    for n in 0..n_h {
        let s_n = sym_ext(&s, n as i64);
        let s_n1 = sym_ext(&s, n as i64 + 1);
        d[n] -= CDF97_GAMMA * (s_n + s_n1);
    }

    // Inverse update-β: s[n] -= β * (d[n-1] + d[n])
    for n in 0..n_l {
        let d_prev = sym_ext(&d, n as i64 - 1);
        let d_n = sym_ext(&d, n as i64);
        s[n] -= CDF97_BETA * (d_prev + d_n);
    }

    // Inverse predict-α: d[n] -= α * (s[n] + s[n+1])
    for n in 0..n_h {
        let s_n = sym_ext(&s, n as i64);
        let s_n1 = sym_ext(&s, n as i64 + 1);
        d[n] -= CDF97_ALPHA * (s_n + s_n1);
    }

    // Interleave: x[2n] = s[n], x[2n+1] = d[n]
    let total = n_l + n_h;
    let mut result = vec![0.0f64; total];
    for n in 0..n_l {
        result[2 * n] = s[n];
    }
    for n in 0..n_h {
        result[2 * n + 1] = d[n];
    }
    result
}

/// CDF 9/7 inverse 2D wavelet transform for a single decomposition level.
///
/// Same row-then-column structure as [`inverse_wavelet_2d`] but using
/// [`inverse_wavelet_1d_97`] with `f64` arithmetic throughout.
///
/// The output is returned as `Vec<f64>`; callers are responsible for
/// clamping to the target bit depth before converting to integer samples.
pub fn inverse_wavelet_2d_97(
    ll: &[f64],
    hl: &[f64],
    lh: &[f64],
    hh: &[f64],
    width: usize,
    height: usize,
) -> Jp2Result<Vec<f64>> {
    let n_l_h = (width + 1) / 2; // ceil(width/2)   — low cols
    let n_h_h = width / 2; // floor(width/2)  — high cols
    let n_l_v = (height + 1) / 2; // ceil(height/2)  — low rows
    let n_h_v = height / 2; // floor(height/2) — high rows

    let expected_ll = n_l_v * n_l_h;
    let expected_hl = n_l_v * n_h_h;
    let expected_lh = n_h_v * n_l_h;
    let expected_hh = n_h_v * n_h_h;

    if ll.len() < expected_ll {
        return Err(Jp2Error::InternalError(format!(
            "LL subband too small for 9-7: expected {expected_ll}, got {}",
            ll.len()
        )));
    }
    if hl.len() < expected_hl {
        return Err(Jp2Error::InternalError(format!(
            "HL subband too small for 9-7: expected {expected_hl}, got {}",
            hl.len()
        )));
    }
    if lh.len() < expected_lh {
        return Err(Jp2Error::InternalError(format!(
            "LH subband too small for 9-7: expected {expected_lh}, got {}",
            lh.len()
        )));
    }
    if hh.len() < expected_hh {
        return Err(Jp2Error::InternalError(format!(
            "HH subband too small for 9-7: expected {expected_hh}, got {}",
            hh.len()
        )));
    }

    // Step 1: Horizontal inverse wavelet on each row of the low-row and high-row bands.
    let mut low_rows: Vec<f64> = vec![0.0; n_l_v * width];
    for row in 0..n_l_v {
        let l = &ll[row * n_l_h..(row + 1) * n_l_h];
        let h = &hl[row * n_h_h..(row + 1) * n_h_h];
        let reconstructed = inverse_wavelet_1d_97(l, h);
        let out_len = reconstructed.len().min(width);
        low_rows[row * width..row * width + out_len].copy_from_slice(&reconstructed[..out_len]);
    }

    let mut high_rows: Vec<f64> = vec![0.0; n_h_v * width];
    for row in 0..n_h_v {
        let l = &lh[row * n_l_h..(row + 1) * n_l_h];
        let h = &hh[row * n_h_h..(row + 1) * n_h_h];
        let reconstructed = inverse_wavelet_1d_97(l, h);
        let out_len = reconstructed.len().min(width);
        high_rows[row * width..row * width + out_len].copy_from_slice(&reconstructed[..out_len]);
    }

    // Step 2: Vertical inverse wavelet on each column.
    let mut output = vec![0.0f64; height * width];
    for col in 0..width {
        let low_col: Vec<f64> = (0..n_l_v).map(|r| low_rows[r * width + col]).collect();
        let high_col: Vec<f64> = (0..n_h_v).map(|r| high_rows[r * width + col]).collect();
        let col_out = inverse_wavelet_1d_97(&low_col, &high_col);
        let out_len = col_out.len().min(height);
        for (row, &val) in col_out[..out_len].iter().enumerate() {
            output[row * width + col] = val;
        }
    }

    Ok(output)
}

/// Tree of subbands for one decomposition level (9/7 irreversible, f64 samples).
#[derive(Debug, Clone)]
pub struct SubbandLevel97 {
    /// HL subband (rows low-pass, cols high-pass).
    pub hl: Vec<f64>,
    /// LH subband (rows high-pass, cols low-pass).
    pub lh: Vec<f64>,
    /// HH subband (all high-pass).
    pub hh: Vec<f64>,
    /// Width of each subband.
    pub width: usize,
    /// Height of each subband.
    pub height: usize,
}

/// Complete subband tree for all decomposition levels (9/7 irreversible, f64).
#[derive(Debug, Clone)]
pub struct SubbandTree97 {
    /// Coarsest LL (DC) subband, in f64.
    pub ll: Vec<f64>,
    /// Width of the LL subband.
    pub ll_width: usize,
    /// Height of the LL subband.
    pub ll_height: usize,
    /// Detail subbands, from coarsest (index 0) to finest (index n-1).
    pub levels: Vec<SubbandLevel97>,
}

/// Reconstruct the full image from a multi-level [`SubbandTree97`] using the
/// CDF 9/7 inverse wavelet.
///
/// Returns `Vec<f64>` samples (not yet clipped to bit depth).
pub fn reconstruct_levels_97(
    subbands: &SubbandTree97,
    num_levels: usize,
    width: usize,
    height: usize,
) -> Jp2Result<Vec<f64>> {
    if subbands.levels.len() < num_levels {
        return Err(Jp2Error::InternalError(format!(
            "SubbandTree97 has {} levels, need {num_levels}",
            subbands.levels.len()
        )));
    }

    let mut current_ll = subbands.ll.clone();
    let mut current_w = subbands.ll_width;
    let mut current_h = subbands.ll_height;

    for level_idx in 0..num_levels {
        let level = &subbands.levels[level_idx];
        let out_w = level.width * 2;
        let out_h = level.height * 2;
        let target_w = if level_idx == num_levels - 1 {
            width
        } else {
            out_w
        };
        let target_h = if level_idx == num_levels - 1 {
            height
        } else {
            out_h
        };

        current_ll = inverse_wavelet_2d_97(
            &current_ll,
            &level.hl,
            &level.lh,
            &level.hh,
            target_w,
            target_h,
        )?;
        current_w = target_w;
        current_h = target_h;
    }

    let _ = current_w;
    let _ = current_h;
    Ok(current_ll)
}

// ── Forward 9/7 ────────────────────────────────────────────────────────────────

/// CDF 9/7 forward 1D transform — the exact inverse of [`inverse_wavelet_1d_97`].
///
/// Applies the four lifting steps in forward order (α-predict, β-update,
/// γ-predict, δ-update) with whole-sample symmetric boundary extension, then
/// rescales the low-pass / high-pass samples by `1/K` and `1/K_inv = K`
/// respectively to match the inverse de-scale at
/// [`inverse_wavelet_1d_97`].
///
/// Returns `(low, high)` of lengths `ceil(n/2)` and `floor(n/2)` for input
/// length `n = signal.len()`.
#[must_use]
pub fn forward_wavelet_1d_97(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![signal[0]], Vec::new());
    }

    let n_l = (n + 1) / 2;
    let n_h = n / 2;

    let mut s: Vec<f64> = signal.iter().step_by(2).copied().collect();
    let mut d: Vec<f64> = signal.iter().skip(1).step_by(2).copied().collect();

    // Step 1 — predict-α: d[n] += α * (s[n] + s[n+1])
    for n in 0..n_h {
        let s_n = sym_ext(&s, n as i64);
        let s_n1 = sym_ext(&s, n as i64 + 1);
        d[n] += CDF97_ALPHA * (s_n + s_n1);
    }

    // Step 2 — update-β: s[n] += β * (d[n-1] + d[n])
    for n in 0..n_l {
        let d_prev = sym_ext(&d, n as i64 - 1);
        let d_n = sym_ext(&d, n as i64);
        s[n] += CDF97_BETA * (d_prev + d_n);
    }

    // Step 3 — predict-γ: d[n] += γ * (s[n] + s[n+1])
    for n in 0..n_h {
        let s_n = sym_ext(&s, n as i64);
        let s_n1 = sym_ext(&s, n as i64 + 1);
        d[n] += CDF97_GAMMA * (s_n + s_n1);
    }

    // Step 4 — update-δ: s[n] += δ * (d[n-1] + d[n])
    for n in 0..n_l {
        let d_prev = sym_ext(&d, n as i64 - 1);
        let d_n = sym_ext(&d, n as i64);
        s[n] += CDF97_DELTA * (d_prev + d_n);
    }

    // Scale
    let s_scaled: Vec<f64> = s.iter().map(|&v| v / CDF97_K).collect();
    let d_scaled: Vec<f64> = d.iter().map(|&v| v / CDF97_K_INV).collect();

    (s_scaled, d_scaled)
}

/// CDF 9/7 forward 2D wavelet transform for a single decomposition level.
///
/// This is the exact inverse of [`inverse_wavelet_2d_97`]: it applies the
/// vertical forward 1D transform on each column (splitting into low/high rows)
/// and then the horizontal forward 1D transform on each row of those two bands,
/// yielding the four subbands.
///
/// Returns `(ll, hl, lh, hh)` with the standard JPEG 2000 sizes for an `input`
/// plane of `width × height` `f64` samples (row-major):
///
/// - `ll`: `ceil(height/2) × ceil(width/2)`
/// - `hl`: `ceil(height/2) × floor(width/2)`
/// - `lh`: `floor(height/2) × ceil(width/2)`
/// - `hh`: `floor(height/2) × floor(width/2)`
pub fn forward_wavelet_2d_97(
    input: &[f64],
    width: usize,
    height: usize,
) -> Jp2Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    if input.len() < width * height {
        return Err(Jp2Error::InternalError(format!(
            "forward_wavelet_2d_97 input too small: expected {}, got {}",
            width * height,
            input.len()
        )));
    }
    let n_l_h = width.div_ceil(2); // ceil(width/2)
    let n_h_h = width / 2; // floor(width/2)
    let n_l_v = height.div_ceil(2); // ceil(height/2)
    let n_h_v = height / 2; // floor(height/2)

    // Step 1: vertical forward on each column → low_rows / high_rows (full width).
    let mut low_rows = vec![0.0f64; n_l_v * width];
    let mut high_rows = vec![0.0f64; n_h_v * width];
    let mut col_vals = vec![0.0f64; height];
    for col in 0..width {
        for (row, slot) in col_vals.iter_mut().enumerate() {
            *slot = input[row * width + col];
        }
        let (low, high) = forward_wavelet_1d_97(&col_vals);
        for (r, &v) in low.iter().enumerate() {
            low_rows[r * width + col] = v;
        }
        for (r, &v) in high.iter().enumerate() {
            high_rows[r * width + col] = v;
        }
    }

    // Step 2: horizontal forward on each low-row → LL (low cols) and HL (high cols).
    let mut ll = vec![0.0f64; n_l_v * n_l_h];
    let mut hl = vec![0.0f64; n_l_v * n_h_h];
    for row in 0..n_l_v {
        let (low, high) = forward_wavelet_1d_97(&low_rows[row * width..(row + 1) * width]);
        let l_len = low.len().min(n_l_h);
        ll[row * n_l_h..row * n_l_h + l_len].copy_from_slice(&low[..l_len]);
        let h_len = high.len().min(n_h_h);
        if n_h_h > 0 {
            hl[row * n_h_h..row * n_h_h + h_len].copy_from_slice(&high[..h_len]);
        }
    }

    // Step 3: horizontal forward on each high-row → LH (low cols) and HH (high cols).
    let mut lh = vec![0.0f64; n_h_v * n_l_h];
    let mut hh = vec![0.0f64; n_h_v * n_h_h];
    for row in 0..n_h_v {
        let (low, high) = forward_wavelet_1d_97(&high_rows[row * width..(row + 1) * width]);
        let l_len = low.len().min(n_l_h);
        lh[row * n_l_h..row * n_l_h + l_len].copy_from_slice(&low[..l_len]);
        let h_len = high.len().min(n_h_h);
        if n_h_h > 0 {
            hh[row * n_h_h..row * n_h_h + h_len].copy_from_slice(&high[..h_len]);
        }
    }

    Ok((ll, hl, lh, hh))
}

/// Decompose an image into a multi-level [`SubbandTree97`] using the CDF 9/7
/// forward wavelet — the exact inverse of [`reconstruct_levels_97`].
///
/// `num_levels` decomposition levels are applied, repeatedly transforming the
/// running LL band in `f64` precision. The returned tree stores detail subbands
/// from coarsest (index 0) to finest (index `num_levels - 1`), matching the
/// layout the decoder consumes via [`reconstruct_levels_97`]. A `num_levels`
/// of 0 returns the image unchanged as the LL band (no transform).
pub fn decompose_levels_97(
    image: &[f64],
    width: usize,
    height: usize,
    num_levels: usize,
) -> Jp2Result<SubbandTree97> {
    if image.len() < width * height {
        return Err(Jp2Error::InternalError(format!(
            "decompose_levels_97 image too small: expected {}, got {}",
            width * height,
            image.len()
        )));
    }

    let mut current = image[..width * height].to_vec();
    let mut cur_w = width;
    let mut cur_h = height;

    // Build detail levels finest-first, then reverse.
    let mut levels_fine_first: Vec<SubbandLevel97> = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        let (ll, hl, lh, hh) = forward_wavelet_2d_97(&current, cur_w, cur_h)?;
        let detail_w = cur_w / 2;
        let detail_h = cur_h / 2;
        levels_fine_first.push(SubbandLevel97 {
            hl,
            lh,
            hh,
            width: detail_w,
            height: detail_h,
        });
        current = ll;
        cur_w = cur_w.div_ceil(2);
        cur_h = cur_h.div_ceil(2);
    }

    levels_fine_first.reverse();

    Ok(SubbandTree97 {
        ll: current,
        ll_width: cur_w,
        ll_height: cur_h,
        levels: levels_fine_first,
    })
}

/// 5-3 forward 2D wavelet transform for a single decomposition level.
///
/// This is the exact inverse of [`inverse_wavelet_2d`]: it applies the vertical
/// forward transform on each column (splitting into low/high rows) and then the
/// horizontal forward transform on each row of those two bands, yielding the
/// four subbands.
///
/// Returns `(ll, hl, lh, hh)` with the standard JPEG 2000 sizes for an
/// `input` plane of `width × height` samples (row-major):
///
/// - `ll`: `ceil(height/2) × ceil(width/2)`
/// - `hl`: `ceil(height/2) × floor(width/2)`
/// - `lh`: `floor(height/2) × ceil(width/2)`
/// - `hh`: `floor(height/2) × floor(width/2)`
pub fn forward_wavelet_2d(
    input: &[i32],
    width: usize,
    height: usize,
) -> Jp2Result<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>)> {
    if input.len() < width * height {
        return Err(Jp2Error::InternalError(format!(
            "forward_wavelet_2d input too small: expected {}, got {}",
            width * height,
            input.len()
        )));
    }
    let n_l_h = width.div_ceil(2); // ceil(width/2)  — low cols
    let n_h_h = width / 2; // floor(width/2) — high cols
    let n_l_v = height.div_ceil(2); // ceil(height/2) — low rows
    let n_h_v = height / 2; // floor(height/2)— high rows

    // Step 1: vertical forward on each column → low_rows / high_rows (full width).
    let mut low_rows = vec![0i32; n_l_v * width];
    let mut high_rows = vec![0i32; n_h_v * width];
    let mut col_vals = vec![0i32; height];
    for col in 0..width {
        for (row, slot) in col_vals.iter_mut().enumerate() {
            *slot = input[row * width + col];
        }
        let (low, high) = forward_53(&col_vals);
        for (r, &v) in low.iter().enumerate() {
            low_rows[r * width + col] = v;
        }
        for (r, &v) in high.iter().enumerate() {
            high_rows[r * width + col] = v;
        }
    }

    // Step 2: horizontal forward on each low-row → LL (low cols) and HL (high cols).
    let mut ll = vec![0i32; n_l_v * n_l_h];
    let mut hl = vec![0i32; n_l_v * n_h_h];
    for row in 0..n_l_v {
        let (low, high) = forward_53(&low_rows[row * width..(row + 1) * width]);
        let l_len = low.len().min(n_l_h);
        ll[row * n_l_h..row * n_l_h + l_len].copy_from_slice(&low[..l_len]);
        let h_len = high.len().min(n_h_h);
        if n_h_h > 0 {
            hl[row * n_h_h..row * n_h_h + h_len].copy_from_slice(&high[..h_len]);
        }
    }

    // Step 3: horizontal forward on each high-row → LH (low cols) and HH (high cols).
    let mut lh = vec![0i32; n_h_v * n_l_h];
    let mut hh = vec![0i32; n_h_v * n_h_h];
    for row in 0..n_h_v {
        let (low, high) = forward_53(&high_rows[row * width..(row + 1) * width]);
        let l_len = low.len().min(n_l_h);
        lh[row * n_l_h..row * n_l_h + l_len].copy_from_slice(&low[..l_len]);
        let h_len = high.len().min(n_h_h);
        if n_h_h > 0 {
            hh[row * n_h_h..row * n_h_h + h_len].copy_from_slice(&high[..h_len]);
        }
    }

    Ok((ll, hl, lh, hh))
}

/// Decompose an image into a multi-level [`SubbandTree`] using the 5-3 forward
/// wavelet — the exact inverse of [`reconstruct_levels`].
///
/// `num_levels` decomposition levels are applied, repeatedly transforming the
/// running LL band. The returned tree stores detail subbands from coarsest
/// (index 0) to finest (index `num_levels - 1`), matching the layout the
/// decoder expects. A `num_levels` of 0 returns the image unchanged as the LL
/// band (no transform).
pub fn decompose_levels(
    image: &[i32],
    width: usize,
    height: usize,
    num_levels: usize,
) -> Jp2Result<SubbandTree> {
    if image.len() < width * height {
        return Err(Jp2Error::InternalError(format!(
            "decompose_levels image too small: expected {}, got {}",
            width * height,
            image.len()
        )));
    }

    let mut current = image[..width * height].to_vec();
    let mut cur_w = width;
    let mut cur_h = height;

    // Build detail levels finest-first, then reverse.
    let mut levels_fine_first: Vec<SubbandLevel> = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        let (ll, hl, lh, hh) = forward_wavelet_2d(&current, cur_w, cur_h)?;
        let detail_w = cur_w / 2;
        let detail_h = cur_h / 2;
        levels_fine_first.push(SubbandLevel {
            hl,
            lh,
            hh,
            width: detail_w,
            height: detail_h,
        });
        current = ll;
        cur_w = cur_w.div_ceil(2);
        cur_h = cur_h.div_ceil(2);
    }

    levels_fine_first.reverse();

    Ok(SubbandTree {
        ll: current,
        ll_width: cur_w,
        ll_height: cur_h,
        levels: levels_fine_first,
    })
}

/// Reconstruct the full image from a multi-level [`SubbandTree`].
///
/// Iterates from coarsest to finest decomposition level, applying the inverse
/// 2D wavelet at each level. The final output is a flat row-major array of
/// `width × height` samples.
pub fn reconstruct_levels(
    subbands: &SubbandTree,
    num_levels: usize,
    width: usize,
    height: usize,
) -> Jp2Result<Vec<i32>> {
    if subbands.levels.len() < num_levels {
        return Err(Jp2Error::InternalError(format!(
            "SubbandTree has {} levels, need {num_levels}",
            subbands.levels.len()
        )));
    }

    // Start from the coarsest LL approximation.
    let mut current_ll = subbands.ll.clone();
    let mut current_w = subbands.ll_width;
    let mut current_h = subbands.ll_height;

    // Iterate from coarsest level (index 0) to finest (index num_levels-1).
    for level_idx in 0..num_levels {
        let level = &subbands.levels[level_idx];
        let out_w = level.width * 2;
        let out_h = level.height * 2;
        // Clamp to actual requested image dimensions at the finest level.
        let target_w = if level_idx == num_levels - 1 {
            width
        } else {
            out_w
        };
        let target_h = if level_idx == num_levels - 1 {
            height
        } else {
            out_h
        };

        current_ll = inverse_wavelet_2d(
            &current_ll,
            &level.hl,
            &level.lh,
            &level.hh,
            target_w,
            target_h,
        )?;
        current_w = target_w;
        current_h = target_h;
    }

    let _ = current_w;
    let _ = current_h;
    Ok(current_ll)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_1d_even_length() {
        let original = vec![10i32, 20, 30, 40, 50, 60, 70, 80];
        let (low, high) = forward_53(&original);
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(
            recovered, original,
            "1D round-trip failed for even-length signal"
        );
    }

    #[test]
    fn roundtrip_1d_odd_length() {
        let original = vec![3i32, 1, 4, 1, 5, 9, 2];
        let (low, high) = forward_53(&original);
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(
            recovered, original,
            "1D round-trip failed for odd-length signal"
        );
    }

    #[test]
    fn roundtrip_1d_single_element() {
        let original = vec![42i32];
        let (low, high) = forward_53(&original);
        assert_eq!(high.len(), 0);
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(recovered, original);
    }

    #[test]
    fn roundtrip_1d_two_elements() {
        let original = vec![100i32, 200];
        let (low, high) = forward_53(&original);
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(recovered, original);
    }

    #[test]
    fn constant_signal_1d() {
        // A constant signal should have all-zero high-pass (detail) coefficients.
        let original = vec![128i32; 8];
        let (low, high) = forward_53(&original);
        for &h in &high {
            assert_eq!(
                h, 0,
                "Detail coefficients should be zero for constant signal"
            );
        }
        let recovered = inverse_wavelet_1d(&low, &high);
        assert_eq!(recovered, original);
    }

    #[test]
    fn roundtrip_2d_constant_image() {
        // A constant 4×4 image: all subbands except LL should be zero.
        let width = 4;
        let height = 4;
        let constant_val = 128i32;

        // For a constant image after 1 level of decomposition:
        // LL = constant (scaled), HL = LH = HH = 0
        // We directly test inverse_wavelet_2d.
        let n_l_h = (width + 1) / 2; // 2
        let n_l_v = (height + 1) / 2; // 2
        let n_h_h = width / 2; // 2
        let n_h_v = height / 2; // 2

        let ll = vec![constant_val; n_l_v * n_l_h];
        let hl = vec![0i32; n_l_v * n_h_h];
        let lh = vec![0i32; n_h_v * n_l_h];
        let hh = vec![0i32; n_h_v * n_h_h];

        // Apply forward transform to constant 4×4 image to find LL value.
        // For constant c: after 5-3 forward, LL should also be c (predict step changes H, not L).
        // Actually let's derive LL from the forward transform.
        let image: Vec<i32> = vec![constant_val; width * height];
        // Apply 2D forward (manual, for testing).
        // Horizontal forward on each row:
        let mut row_transformed = vec![0i32; width * height];
        for row in 0..height {
            let (l, h) = forward_53(&image[row * width..(row + 1) * width]);
            for (i, &v) in l.iter().enumerate() {
                row_transformed[row * width + i] = v;
            }
            for (i, &v) in h.iter().enumerate() {
                row_transformed[row * width + n_l_h + i] = v;
            }
        }
        // Vertical forward on each column (for low-col region only):
        let mut ll_forward = vec![0i32; n_l_v * n_l_h];
        for col in 0..n_l_h {
            let col_vals: Vec<i32> = (0..height)
                .map(|r| row_transformed[r * width + col])
                .collect();
            let (l, _h) = forward_53(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll_forward[r * n_l_h + col] = v;
            }
        }

        let output = inverse_wavelet_2d(&ll_forward, &hl, &lh, &hh, width, height)
            .expect("inverse_wavelet_2d");
        for (i, &v) in output.iter().enumerate() {
            assert_eq!(
                v, constant_val,
                "Sample {i} should be {constant_val}, got {v}"
            );
        }
    }

    #[test]
    fn reconstruct_levels_with_zero_details() {
        // Single-level decomposition of a constant 4×4 image.
        let width = 4;
        let height = 4;
        let val = 64i32;

        let n_l_h = (width + 1) / 2;
        let n_l_v = (height + 1) / 2;
        let n_h_h = width / 2;
        let n_h_v = height / 2;

        // Use forward 2D to get the real LL for constant signal.
        // For constant = 64, after 5-3 forward 1 level: LL is also 64, details are 0.
        let ll = vec![val; n_l_v * n_l_h];
        let hl = vec![0i32; n_l_v * n_h_h];
        let lh = vec![0i32; n_h_v * n_l_h];
        let hh = vec![0i32; n_h_v * n_h_h];

        let tree = SubbandTree {
            ll: ll.clone(),
            ll_width: n_l_h,
            ll_height: n_l_v,
            levels: vec![SubbandLevel {
                hl,
                lh,
                hh,
                width: n_l_h,
                height: n_l_v,
            }],
        };
        let output = reconstruct_levels(&tree, 1, width, height).expect("reconstruct");
        // All output should equal val (64) since all detail subbands were zero.
        // The inverse of a forward transform on constant input returns constant.
        assert_eq!(output.len(), width * height);
        for &v in &output {
            assert_eq!(v, val, "Expected all samples to be {val}");
        }
    }

    // ── CDF 9/7 tests ─────────────────────────────────────────────────────────

    #[test]
    fn cdf97_inverse_roundtrip_8_samples() {
        // Forward 9-7 transform, then inverse should recover original within 1e-4.
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (low, high) = forward_wavelet_1d_97(&signal);
        let recovered = inverse_wavelet_1d_97(&low, &high);
        assert_eq!(recovered.len(), signal.len(), "Recovered length mismatch");
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Sample {i}: original {orig}, recovered {rec}, diff {}",
                (orig - rec).abs()
            );
        }
    }

    #[test]
    fn cdf97_inverse_roundtrip_constant_signal() {
        // A constant signal: forward then inverse should recover constant.
        let signal = vec![128.0f64; 8];
        let (low, high) = forward_wavelet_1d_97(&signal);
        let recovered = inverse_wavelet_1d_97(&low, &high);
        assert_eq!(recovered.len(), signal.len());
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Constant signal sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn inverse_wavelet_97_constant_signal() {
        // Constant signal through 9-7 IDWT should remain constant.
        let n = 8usize;
        let n_l_h = (n + 1) / 2;
        let n_h_h = n / 2;
        let n_l_v = (n + 1) / 2;
        let n_h_v = n / 2;

        // For a constant image, forward 9-7 produces: LL = constant, HL=LH=HH = 0.
        // We first compute the true LL via forward_97 so the inverse recovers correctly.
        // For a unit-constant signal c, forward 9-7 produces LL = c / K² (due to scaling).
        // We avoid the forward path here: just feed LL=128.0 and zero detail subbands
        // and verify the inverse is close to 128.0 (within the 9-7 filter tolerance).
        //
        // A proper constant roundtrip: generate LL, HL, LH, HH via forward transform.
        let image = vec![128.0f64; n * n];

        // Horizontal forward
        let mut ll_h = vec![0.0f64; n_l_h * n];
        let mut hl_h = vec![0.0f64; n_h_h * n];
        for row in 0..n {
            let (l, h) = forward_wavelet_1d_97(&image[row * n..(row + 1) * n]);
            for (c, &v) in l.iter().enumerate() {
                ll_h[row * n_l_h + c] = v;
            }
            for (c, &v) in h.iter().enumerate() {
                hl_h[row * n_h_h + c] = v;
            }
        }

        // Vertical forward on low-col region
        let mut ll = vec![0.0f64; n_l_v * n_l_h];
        let mut lh = vec![0.0f64; n_h_v * n_l_h];
        for col in 0..n_l_h {
            let col_vals: Vec<f64> = (0..n).map(|r| ll_h[r * n_l_h + col]).collect();
            let (l, h) = forward_wavelet_1d_97(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll[r * n_l_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                lh[r * n_l_h + col] = v;
            }
        }

        // Vertical forward on high-col region
        let mut hl = vec![0.0f64; n_l_v * n_h_h];
        let mut hh = vec![0.0f64; n_h_v * n_h_h];
        for col in 0..n_h_h {
            let col_vals: Vec<f64> = (0..n).map(|r| hl_h[r * n_h_h + col]).collect();
            let (l, h) = forward_wavelet_1d_97(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                hl[r * n_h_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                hh[r * n_h_h + col] = v;
            }
        }

        let result = inverse_wavelet_2d_97(&ll, &hl, &lh, &hh, n, n).unwrap();
        assert_eq!(result.len(), n * n);
        for (i, v) in result.iter().enumerate() {
            assert!(
                (v - 128.0).abs() < 1e-4,
                "Sample {i}: expected ~128.0, got {v}"
            );
        }
    }

    #[test]
    fn reconstruct_levels_97_constant_signal() {
        let n = 8usize;
        let n_l_h = (n + 1) / 2;
        let n_h_h = n / 2;
        let n_l_v = (n + 1) / 2;
        let n_h_v = n / 2;

        let image = vec![64.0f64; n * n];

        // Build subbands via forward 9-7 2D
        let mut ll_h = vec![0.0f64; n_l_h * n];
        let mut hl_h = vec![0.0f64; n_h_h * n];
        for row in 0..n {
            let (l, h) = forward_wavelet_1d_97(&image[row * n..(row + 1) * n]);
            for (c, &v) in l.iter().enumerate() {
                ll_h[row * n_l_h + c] = v;
            }
            for (c, &v) in h.iter().enumerate() {
                hl_h[row * n_h_h + c] = v;
            }
        }

        let mut ll = vec![0.0f64; n_l_v * n_l_h];
        let mut lh = vec![0.0f64; n_h_v * n_l_h];
        for col in 0..n_l_h {
            let col_vals: Vec<f64> = (0..n).map(|r| ll_h[r * n_l_h + col]).collect();
            let (l, h) = forward_wavelet_1d_97(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                ll[r * n_l_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                lh[r * n_l_h + col] = v;
            }
        }

        let mut hl = vec![0.0f64; n_l_v * n_h_h];
        let mut hh = vec![0.0f64; n_h_v * n_h_h];
        for col in 0..n_h_h {
            let col_vals: Vec<f64> = (0..n).map(|r| hl_h[r * n_h_h + col]).collect();
            let (l, h) = forward_wavelet_1d_97(&col_vals);
            for (r, &v) in l.iter().enumerate() {
                hl[r * n_h_h + col] = v;
            }
            for (r, &v) in h.iter().enumerate() {
                hh[r * n_h_h + col] = v;
            }
        }

        let tree = SubbandTree97 {
            ll,
            ll_width: n_l_h,
            ll_height: n_l_v,
            levels: vec![SubbandLevel97 {
                hl,
                lh,
                hh,
                width: n_l_h,
                height: n_l_v,
            }],
        };

        let output = reconstruct_levels_97(&tree, 1, n, n).expect("reconstruct_97");
        assert_eq!(output.len(), n * n);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 64.0).abs() < 1e-4,
                "Sample {i}: expected ~64.0, got {v}"
            );
        }
    }

    // ── CDF 9/7 forward + decompose tests (Wave 10 Slice 2) ──────────────────

    #[test]
    fn cdf97_forward_inverse_identity_8() {
        // forward_wavelet_1d_97 then inverse_wavelet_1d_97 must be the identity
        // within 1e-6 for an 8-sample input.
        let signal: Vec<f64> = (0..8).map(|i| (i as f64) * 11.0 + 3.0).collect();
        let (low, high) = forward_wavelet_1d_97(&signal);
        let recovered = inverse_wavelet_1d_97(&low, &high);
        assert_eq!(recovered.len(), signal.len());
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn cdf97_forward_inverse_identity_16() {
        let signal: Vec<f64> = (0..16).map(|i| (i as f64) * 2.5 - 7.0).collect();
        let (low, high) = forward_wavelet_1d_97(&signal);
        let recovered = inverse_wavelet_1d_97(&low, &high);
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn cdf97_forward_inverse_identity_32() {
        let signal: Vec<f64> = (0..32).map(|i| ((i * 7) % 31) as f64).collect();
        let (low, high) = forward_wavelet_1d_97(&signal);
        let recovered = inverse_wavelet_1d_97(&low, &high);
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-6,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn decompose_reconstruct_97_identity_1_level() {
        let w = 16usize;
        let h = 16usize;
        let image: Vec<f64> = (0..w * h).map(|i| (i as f64) % 53.0).collect();
        let tree = decompose_levels_97(&image, w, h, 1).expect("decompose_97");
        let recon = reconstruct_levels_97(&tree, 1, w, h).expect("reconstruct_97");
        assert_eq!(recon.len(), w * h);
        for (i, (&orig, &rec)) in image.iter().zip(recon.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-4,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn decompose_reconstruct_97_identity_2_levels() {
        let w = 16usize;
        let h = 16usize;
        let image: Vec<f64> = (0..w * h)
            .map(|i| (((i % w) + (i / w)) as f64) * 5.0)
            .collect();
        let tree = decompose_levels_97(&image, w, h, 2).expect("decompose_97");
        let recon = reconstruct_levels_97(&tree, 2, w, h).expect("reconstruct_97");
        assert_eq!(recon.len(), w * h);
        for (i, (&orig, &rec)) in image.iter().zip(recon.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-3,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }

    #[test]
    fn decompose_reconstruct_97_identity_3_levels() {
        let w = 32usize;
        let h = 32usize;
        let image: Vec<f64> = (0..w * h).map(|i| ((i * 13) % 200) as f64).collect();
        let tree = decompose_levels_97(&image, w, h, 3).expect("decompose_97");
        let recon = reconstruct_levels_97(&tree, 3, w, h).expect("reconstruct_97");
        assert_eq!(recon.len(), w * h);
        for (i, (&orig, &rec)) in image.iter().zip(recon.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-2,
                "Sample {i}: orig {orig}, rec {rec}"
            );
        }
    }
}
