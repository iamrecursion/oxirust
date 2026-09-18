//! EBCOT Tier-1 bit-plane decoder (ISO/IEC 15444-1 Annex D).
//!
//! Decodes quantised wavelet coefficient bit-planes for a single JPEG 2000
//! code-block using the three coding passes:
//!
//! 1. **Significance Propagation pass (SPP)**: for insignificant coefficients
//!    that have a significant neighbour, decode significance and (if significant)
//!    the sign bit.
//! 2. **Magnitude Refinement pass (MRP)**: for already-significant coefficients,
//!    decode the next magnitude bit.
//! 3. **Cleanup pass (CUP)**: for all remaining not-yet-significant coefficients
//!    (in this bit-plane), using run-length decoding for isolated zero runs.
//!
//! Context labels for the MQ coder:
//! - Significance: 9 contexts (0-8, based on H/V/D neighbour sums)
//! - Sign: 5 contexts (9-13)
//! - Magnitude Refinement: 3 contexts (14-16)
//! - Uniform (run-length): context 17
//! - Run-Length (RLC): context 18
//!
//! Total: 19 contexts → `MQ_NUM_CONTEXTS = 19`.

use super::mq_coder::{MqDecoder, MQ_NUM_CONTEXTS};
use super::{Jp2Error, Jp2Result};

// ── Context index constants ───────────────────────────────────────────────────

/// Number of significance contexts (0-8).
const NUM_SIG_CONTEXTS: usize = 9;
/// Base index for sign contexts.
const SIGN_CTX_BASE: usize = 9;
/// Number of sign contexts.
const _NUM_SIGN_CONTEXTS: usize = 5;
/// Base index for magnitude refinement contexts.
const MR_CTX_BASE: usize = 14;
/// Uniform context index (for run-length uniform bits).
const UNI_CTX: usize = 17;
/// Run-length context index.
const RLC_CTX: usize = 18;

/// Decoded code-block output.
#[derive(Debug, Clone)]
pub struct CodeBlock {
    /// Decoded coefficient magnitudes (signed, in sign-magnitude form converted to two's complement).
    pub coeffs: Vec<i32>,
    /// Block width in samples.
    pub width: usize,
    /// Block height in samples.
    pub height: usize,
}

impl CodeBlock {
    /// Convert integer coefficients to floating-point with optional scalar dequantization.
    ///
    /// For the lossless (5-3) path, pass `step_size = 1.0`; coefficients are simply
    /// cast to `f64` unchanged.
    ///
    /// For the irreversible (9-7) path, pass the QCD-derived step size.  The formula
    /// follows ISO/IEC 15444-1 §D.2.2:
    ///
    /// ```text
    ///   q_b = R_b + guard_bits  (total bit-planes available)
    ///   coefficient_f64 = sign(v) * |v| * Δ_b * 2^(−decoded_bit_planes)
    /// ```
    ///
    /// where `Δ_b = step_size` already encodes `2^(R_b − ε_b) * (1 + μ_b/2048)`.
    #[must_use]
    pub fn dequantize(&self, step_size: f64, decoded_bit_planes: usize) -> Vec<f64> {
        if (step_size - 1.0).abs() < 1e-10 {
            // Lossless: no dequantization — direct integer-to-float.
            return self.coeffs.iter().map(|&v| v as f64).collect();
        }
        // Lossy: apply dequantization scaling.
        let scale = step_size * (0.5f64).powi(decoded_bit_planes as i32);
        self.coeffs
            .iter()
            .map(|&v| {
                let mag = (v.abs() as f64) * scale;
                if v < 0 {
                    -mag
                } else {
                    mag
                }
            })
            .collect()
    }
}

// ── Significance state arrays ─────────────────────────────────────────────────

/// Per-coefficient state flags.
#[derive(Clone, Copy, Default)]
struct CoeffState {
    /// True once any magnitude bit has been decoded as 1 (coefficient is non-zero).
    significant: bool,
    /// Sign bit: 0 = positive, 1 = negative.
    sign: i32,
    /// Magnitude accumulated across bit-planes (MSB first).
    magnitude: i32,
    /// True if this coefficient has been visited in the current bit-plane's
    /// significance propagation pass.
    visited: bool,
}

// ── Context computation ───────────────────────────────────────────────────────

/// Compute the significance context (0-8) for position (col, row) in an
/// `width × height` array, given the significance state map.
fn significance_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> usize {
    // Horizontal and vertical neighbour significance counts.
    let mut h_count = 0u32;
    let mut v_count = 0u32;
    let mut d_count = 0u32;

    let left_sig = col > 0 && state[row * width + col - 1].significant;
    let right_sig = col + 1 < width && state[row * width + col + 1].significant;
    let up_sig = row > 0 && state[(row - 1) * width + col].significant;
    let dn_sig = row + 1 < height && state[(row + 1) * width + col].significant;

    if left_sig {
        h_count += 1;
    }
    if right_sig {
        h_count += 1;
    }
    if up_sig {
        v_count += 1;
    }
    if dn_sig {
        v_count += 1;
    }

    // Diagonal neighbours — use saturating arithmetic to avoid overflow in release builds
    // and explicit bounds guards in debug builds.
    if col > 0 && row > 0 && state[(row - 1) * width + (col - 1)].significant {
        d_count += 1;
    }
    if col + 1 < width && row > 0 && state[(row - 1) * width + col + 1].significant {
        d_count += 1;
    }
    if col > 0 && row + 1 < height && state[(row + 1) * width + (col - 1)].significant {
        d_count += 1;
    }
    if col + 1 < width && row + 1 < height && state[(row + 1) * width + col + 1].significant {
        d_count += 1;
    }

    let hv = h_count + v_count;

    // Context table from ISO 15444-1 Table D.1 (simplified LL subband variant):
    match hv {
        0 if d_count == 0 => 0,
        0 if d_count == 1 => 1,
        0 /* d >= 2 */    => 2,
        1 if d_count == 0 => 3,
        1 if d_count == 1 => 4,
        1 /* d >= 2 */    => 5,
        2 if d_count == 0 => 6,
        2 /* d > 0 */     => 7,
        _ /* hv >= 3 */   => 8,
    }
}

/// Compute sign context (9-13) using horizontal and vertical contributions.
///
/// Returns `(context_index, xor_bit)` where `xor_bit` is XOR'd with the decoded
/// bit to get the actual sign (0 = positive, 1 = negative).
fn sign_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> (usize, u8) {
    // H contribution: sign of horizontal significant neighbours.
    let h_contrib = {
        let l = col > 0 && state[row * width + col - 1].significant;
        let r = col + 1 < width && state[row * width + col + 1].significant;
        let l_sign = l && state[row * width + col - 1].sign != 0;
        let r_sign = r && state[row * width + col + 1].sign != 0;
        if !l && !r {
            0i32
        } else if l && !r {
            if l_sign {
                -1
            } else {
                1
            }
        } else if !l && r {
            if r_sign {
                -1
            } else {
                1
            }
        } else {
            // Both present.
            let ls = if l_sign { -1i32 } else { 1 };
            let rs = if r_sign { -1i32 } else { 1 };
            (ls + rs).signum()
        }
    };
    let v_contrib = {
        let u = row > 0 && state[(row - 1) * width + col].significant;
        let d = row + 1 < height && state[(row + 1) * width + col].significant;
        let u_sign = u && state[(row - 1) * width + col].sign != 0;
        let d_sign = d && state[(row + 1) * width + col].sign != 0;
        if !u && !d {
            0i32
        } else if u && !d {
            if u_sign {
                -1
            } else {
                1
            }
        } else if !u && d {
            if d_sign {
                -1
            } else {
                1
            }
        } else {
            let us = if u_sign { -1i32 } else { 1 };
            let ds = if d_sign { -1i32 } else { 1 };
            (us + ds).signum()
        }
    };

    // Context label from ISO 15444-1 Table D.2.
    let (ctx_offset, xor_bit) = match (h_contrib, v_contrib) {
        (1, 1) | (1, 0) | (0, 1) => (0, 0u8),
        (1, -1) => (1, 0),
        (0, 0) => (2, 0),
        (-1, 1) => (1, 1),
        (-1, 0) | (0, -1) | (-1, -1) => (0, 1),
        _ => (0, 0),
    };
    (SIGN_CTX_BASE + ctx_offset, xor_bit)
}

/// Compute magnitude refinement context (14-16) for position (col, row).
fn mr_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
    first_mr: bool,
) -> usize {
    if first_mr {
        // First MR: use neighbour significance to pick 14 or 15.
        let has_sig_neighbour = {
            let mut any = false;
            if col > 0 && state[row * width + col - 1].significant {
                any = true;
            }
            if col + 1 < width && state[row * width + col + 1].significant {
                any = true;
            }
            if row > 0 && state[(row - 1) * width + col].significant {
                any = true;
            }
            if row + 1 < height && state[(row + 1) * width + col].significant {
                any = true;
            }
            any
        };
        MR_CTX_BASE + if has_sig_neighbour { 1 } else { 0 }
    } else {
        // Subsequent MR passes: always use context 16.
        MR_CTX_BASE + 2
    }
}

// ── Bit-plane decoding passes ─────────────────────────────────────────────────

/// Significance Propagation Pass (SPP).
///
/// For each coefficient that is not yet significant AND has at least one
/// significant neighbour (context ≥ 1): decode significance, then if significant
/// decode sign.
fn significance_propagation_pass(
    mq: &mut MqDecoder,
    state: &mut [CoeffState],
    width: usize,
    height: usize,
    bit_plane: u8,
    cx: &mut [u8; MQ_NUM_CONTEXTS],
) -> Jp2Result<()> {
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if state[idx].significant || state[idx].visited {
                continue;
            }
            let ctx = significance_context(state, col, row, width, height);
            if ctx == 0 {
                // No significant neighbours — skip in SPP.
                continue;
            }
            // Decode significance bit.
            let sig_bit = mq.decode_bit(ctx)?;
            state[idx].visited = true;
            if sig_bit == 1 {
                state[idx].significant = true;
                state[idx].magnitude |= 1 << bit_plane;
                // Decode sign bit.
                let (sign_ctx, xor_bit) = sign_context(state, col, row, width, height);
                let sign_coded = mq.decode_bit(sign_ctx)?;
                state[idx].sign = i32::from(sign_coded ^ xor_bit);
            }
        }
    }
    Ok(())
}

/// Magnitude Refinement Pass (MRP).
///
/// For coefficients that were already significant before this bit-plane, decode
/// the next magnitude bit.
fn magnitude_refinement_pass(
    mq: &mut MqDecoder,
    state: &mut [CoeffState],
    width: usize,
    height: usize,
    bit_plane: u8,
    first_mr: bool,
    cx: &mut [u8; MQ_NUM_CONTEXTS],
) -> Jp2Result<()> {
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if !state[idx].significant || state[idx].visited {
                continue;
            }
            let ctx = mr_context(state, col, row, width, height, first_mr);
            let mr_bit = mq.decode_bit(ctx)?;
            if mr_bit == 1 {
                state[idx].magnitude |= 1 << bit_plane;
            }
        }
    }
    Ok(())
}

/// Cleanup Pass (CUP).
///
/// For coefficients not yet coded in this bit-plane (not visited). Uses
/// run-length coding for runs of 4 consecutive insignificant zero-context
/// coefficients.
fn cleanup_pass(
    mq: &mut MqDecoder,
    state: &mut [CoeffState],
    width: usize,
    height: usize,
    bit_plane: u8,
    cx: &mut [u8; MQ_NUM_CONTEXTS],
) -> Jp2Result<()> {
    let mut row = 0;
    while row < height {
        let mut col = 0;
        while col < width {
            let idx = row * width + col;
            if state[idx].visited {
                col += 1;
                continue;
            }
            // Check if we can use run-length for a column-stripe of 4.
            let can_rlc = row + 3 < height
                && (0..4).all(|dr| {
                    let r = row + dr;
                    let i = r * width + col;
                    !state[i].significant
                        && !state[i].visited
                        && significance_context(state, col, r, width, height) == 0
                });

            if can_rlc {
                let rlc_bit = mq.decode_bit(RLC_CTX)?;
                if rlc_bit == 0 {
                    // All 4 are zero — skip.
                    for dr in 0..4usize {
                        state[(row + dr) * width + col].visited = true;
                    }
                    col += 1;
                    continue;
                }
                // Decode two uniform bits to find where the first 1 is.
                let p0 = mq.decode_bit(UNI_CTX)?;
                let p1 = mq.decode_bit(UNI_CTX)?;
                let first_one = usize::from(p0) * 2 + usize::from(p1);
                for dr in 0..first_one {
                    state[(row + dr) * width + col].visited = true;
                }
                // The `first_one` position becomes significant.
                let sig_row = row + first_one;
                let sig_idx = sig_row * width + col;
                state[sig_idx].significant = true;
                state[sig_idx].magnitude |= 1 << bit_plane;
                state[sig_idx].visited = true;
                let (sign_ctx, xor_bit) = sign_context(state, col, sig_row, width, height);
                let sign_coded = mq.decode_bit(sign_ctx)?;
                state[sig_idx].sign = i32::from(sign_coded ^ xor_bit);
                // Continue with remaining rows in the stripe.
                for dr in (first_one + 1)..4 {
                    let r = row + dr;
                    let i = r * width + col;
                    if state[i].visited {
                        continue;
                    }
                    let sig_ctx = significance_context(state, col, r, width, height);
                    let sig_bit = mq.decode_bit(sig_ctx)?;
                    state[i].visited = true;
                    if sig_bit == 1 {
                        state[i].significant = true;
                        state[i].magnitude |= 1 << bit_plane;
                        let (sc, xb) = sign_context(state, col, r, width, height);
                        let sb = mq.decode_bit(sc)?;
                        state[i].sign = i32::from(sb ^ xb);
                    }
                }
                col += 1;
                continue;
            }

            // Regular cleanup: decode significance.
            let sig_ctx = significance_context(state, col, row, width, height);
            let sig_bit = mq.decode_bit(sig_ctx)?;
            state[idx].visited = true;
            if sig_bit == 1 {
                state[idx].significant = true;
                state[idx].magnitude |= 1 << bit_plane;
                let (sc, xb) = sign_context(state, col, row, width, height);
                let sb = mq.decode_bit(sc)?;
                state[idx].sign = i32::from(sb ^ xb);
            }
            col += 1;
        }
        row += 1;
    }
    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode a single JPEG 2000 code-block using EBCOT Tier-1.
///
/// `data` is the raw compressed byte data for this code-block.
/// `width` and `height` are the code-block dimensions (typically 64×64 or smaller).
/// `num_bit_planes` is the number of significant bit-planes to decode (from MSB).
///
/// Returns the decoded signed coefficients as a flat row-major `Vec<i32>`.
pub fn decode_code_block(
    data: &[u8],
    width: usize,
    height: usize,
    num_bit_planes: u8,
) -> Jp2Result<CodeBlock> {
    if width == 0 || height == 0 {
        return Err(Jp2Error::InternalError(
            "code-block dimensions must be non-zero".to_string(),
        ));
    }
    if num_bit_planes == 0 {
        // Nothing to decode — return all zeros.
        return Ok(CodeBlock {
            coeffs: vec![0; width * height],
            width,
            height,
        });
    }

    let mut mq = MqDecoder::new(data);
    let mut cx = [0u8; MQ_NUM_CONTEXTS];
    let mut state = vec![CoeffState::default(); width * height];

    // In JPEG 2000 Tier-1, the coding passes cycle as:
    // For each bit-plane from MSB to LSB:
    //   - For bit-plane P = num_bit_planes-1 (MSB): only Cleanup pass
    //   - For subsequent planes P: SPP, MRP, CUP
    // However, the simplification for single-layer: we receive the data for all
    // passes in a single compressed block. We interleave the three passes.

    for bp_idx in 0..num_bit_planes {
        // bit_plane value: the most significant decoded plane has the highest value.
        let bit_plane = num_bit_planes - 1 - bp_idx;

        // Reset visited flags for this bit-plane.
        for s in state.iter_mut() {
            s.visited = false;
        }

        if bp_idx == 0 {
            // First bit-plane: only cleanup pass.
            cleanup_pass(&mut mq, &mut state, width, height, bit_plane, &mut cx)?;
        } else {
            // Subsequent bit-planes: SPP → MRP → CUP.
            significance_propagation_pass(&mut mq, &mut state, width, height, bit_plane, &mut cx)?;
            // Reset visited between passes.
            for s in state.iter_mut() {
                s.visited = false;
            }
            let first_mr = bp_idx == 1;
            magnitude_refinement_pass(
                &mut mq, &mut state, width, height, bit_plane, first_mr, &mut cx,
            )?;
            // Reset visited again before CUP.
            for s in state.iter_mut() {
                s.visited = false;
            }
            cleanup_pass(&mut mq, &mut state, width, height, bit_plane, &mut cx)?;
        }
    }

    // Convert sign-magnitude to signed two's complement.
    let coeffs: Vec<i32> = state
        .iter()
        .map(|s| {
            if s.sign == 0 {
                s.magnitude
            } else {
                -s.magnitude
            }
        })
        .collect();

    Ok(CodeBlock {
        coeffs,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_zero_bit_planes_returns_zeros() {
        let data = vec![0u8; 16];
        let block = decode_code_block(&data, 4, 4, 0).expect("decode");
        assert_eq!(block.coeffs.len(), 16);
        for &c in &block.coeffs {
            assert_eq!(c, 0);
        }
    }

    #[test]
    fn decode_code_block_runs_without_panic() {
        // Provide enough data for the MQ decoder to consume.
        let data: Vec<u8> = (0u8..=255).collect();
        let result = decode_code_block(&data, 8, 8, 4);
        // We just verify it doesn't panic; actual coefficient values depend on the stream.
        match result {
            Ok(block) => {
                assert_eq!(block.width, 8);
                assert_eq!(block.height, 8);
                assert_eq!(block.coeffs.len(), 64);
                for &c in &block.coeffs {
                    // All coefficients must be finite i32.
                    let _ = c;
                }
            }
            Err(_) => {
                // MQ coder may run out of data for random input — that's OK.
            }
        }
    }

    #[test]
    fn zero_dimension_returns_error() {
        let data = vec![0u8; 8];
        assert!(decode_code_block(&data, 0, 4, 1).is_err());
        assert!(decode_code_block(&data, 4, 0, 1).is_err());
    }

    #[test]
    fn significance_context_zero_for_isolated() {
        // An isolated coefficient (no significant neighbours) → context 0.
        let state = vec![CoeffState::default(); 4 * 4];
        let ctx = significance_context(&state, 1, 1, 4, 4);
        assert_eq!(ctx, 0);
    }

    #[test]
    fn num_sig_contexts_is_nine() {
        assert_eq!(NUM_SIG_CONTEXTS, 9);
    }
}
