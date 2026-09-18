//! EBCOT Tier-1 bit-plane *encoder* (ISO/IEC 15444-1 Annex D) — the exact
//! forward counterpart of [`super::tier1`].
//!
//! For each code-block the encoder walks the same bit-plane scan and the same
//! three coding passes as the decoder
//! ([`super::tier1::decode_code_block`]) — significance propagation (SPP),
//! magnitude refinement (MRP) and cleanup (CUP) — but instead of *reading* each
//! binary decision from the MQ decoder it *computes* the decision from the known
//! coefficient and *writes* it to the [`MqEncoder`]. Because the per-coefficient
//! state machine (significant / visited / sign / magnitude) and the context
//! formation are identical to the decoder, decoding the produced byte stream
//! reconstructs exactly the original coefficients.
//!
//! Context labels (identical to the decoder):
//! - Significance: 9 contexts (0-8)
//! - Sign: 5 contexts (9-13)
//! - Magnitude refinement: 3 contexts (14-16)
//! - Uniform (run-length): context 17
//! - Run-length (RLC): context 18

use super::mq_encoder::MqEncoder;
use super::{Jp2Error, Jp2Result};

// ── Context index constants (mirror tier1.rs) ─────────────────────────────────

/// Base index for sign contexts.
const SIGN_CTX_BASE: usize = 9;
/// Base index for magnitude refinement contexts.
const MR_CTX_BASE: usize = 14;
/// Uniform context index (for run-length uniform bits).
const UNI_CTX: usize = 17;
/// Run-length context index.
const RLC_CTX: usize = 18;

/// Per-coefficient encoder state (mirrors the decoder's `CoeffState`).
#[derive(Clone, Copy, Default)]
struct CoeffState {
    /// True once the most-significant 1 bit has been coded.
    significant: bool,
    /// Sign bit: 0 = positive, 1 = negative.
    sign: i32,
    /// True if visited in the current bit-plane pass.
    visited: bool,
}

// ── Context computation (verbatim from tier1.rs) ──────────────────────────────

/// Significance context (0-8) for position (col, row) — identical to the decoder.
fn significance_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> usize {
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

    match hv {
        0 if d_count == 0 => 0,
        0 if d_count == 1 => 1,
        0 => 2,
        1 if d_count == 0 => 3,
        1 if d_count == 1 => 4,
        1 => 5,
        2 if d_count == 0 => 6,
        2 => 7,
        _ => 8,
    }
}

/// Sign context (9-13) and XOR bit — identical to the decoder.
fn sign_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> (usize, u8) {
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

/// Magnitude refinement context (14-16) — identical to the decoder.
fn mr_context(
    state: &[CoeffState],
    col: usize,
    row: usize,
    width: usize,
    height: usize,
    first_mr: bool,
) -> usize {
    if first_mr {
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
        MR_CTX_BASE + 2
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Magnitude bit of coefficient `idx` at bit-plane `bp`.
#[inline]
fn mag_bit(mag: &[i32], idx: usize, bit_plane: u8) -> u8 {
    ((mag[idx] >> bit_plane) & 1) as u8
}

// ── Coding passes (forward) ───────────────────────────────────────────────────

/// Significance Propagation Pass (forward) — mirrors the decoder's SPP.
fn significance_propagation_pass(
    mq: &mut MqEncoder,
    state: &mut [CoeffState],
    mag: &[i32],
    sign: &[i32],
    width: usize,
    height: usize,
    bit_plane: u8,
) {
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if state[idx].significant || state[idx].visited {
                continue;
            }
            let ctx = significance_context(state, col, row, width, height);
            if ctx == 0 {
                continue;
            }
            let sig_bit = mag_bit(mag, idx, bit_plane);
            mq.encode_decision(ctx, sig_bit);
            state[idx].visited = true;
            if sig_bit == 1 {
                state[idx].significant = true;
                let (sign_ctx, xor_bit) = sign_context(state, col, row, width, height);
                let sign_val = (sign[idx] != 0) as u8;
                mq.encode_decision(sign_ctx, sign_val ^ xor_bit);
                state[idx].sign = i32::from(sign_val);
            }
        }
    }
}

/// Magnitude Refinement Pass (forward) — mirrors the decoder's MRP.
fn magnitude_refinement_pass(
    mq: &mut MqEncoder,
    state: &mut [CoeffState],
    mag: &[i32],
    width: usize,
    height: usize,
    bit_plane: u8,
    first_mr: bool,
) {
    for row in 0..height {
        for col in 0..width {
            let idx = row * width + col;
            if !state[idx].significant || state[idx].visited {
                continue;
            }
            let ctx = mr_context(state, col, row, width, height, first_mr);
            let mr_bit = mag_bit(mag, idx, bit_plane);
            mq.encode_decision(ctx, mr_bit);
        }
    }
}

/// Cleanup Pass (forward) — mirrors the decoder's CUP, including the
/// four-sample run-length coding of zero-context column stripes.
fn cleanup_pass(
    mq: &mut MqEncoder,
    state: &mut [CoeffState],
    mag: &[i32],
    sign: &[i32],
    width: usize,
    height: usize,
    bit_plane: u8,
) {
    let mut row = 0;
    while row < height {
        let mut col = 0;
        while col < width {
            let idx = row * width + col;
            if state[idx].visited {
                col += 1;
                continue;
            }
            let can_rlc = row + 3 < height
                && (0..4).all(|dr| {
                    let r = row + dr;
                    let i = r * width + col;
                    !state[i].significant
                        && !state[i].visited
                        && significance_context(state, col, r, width, height) == 0
                });

            if can_rlc {
                // Find the first row in the stripe whose magnitude bit is 1.
                let mut first_one: Option<usize> = None;
                for dr in 0..4usize {
                    if mag_bit(mag, (row + dr) * width + col, bit_plane) == 1 {
                        first_one = Some(dr);
                        break;
                    }
                }
                match first_one {
                    None => {
                        // RLC bit = 0: all four samples stay insignificant.
                        mq.encode_decision(RLC_CTX, 0);
                        for dr in 0..4usize {
                            state[(row + dr) * width + col].visited = true;
                        }
                        col += 1;
                        continue;
                    }
                    Some(fo) => {
                        // RLC bit = 1, then two UNIFORM bits give the position.
                        mq.encode_decision(RLC_CTX, 1);
                        let p0 = ((fo >> 1) & 1) as u8;
                        let p1 = (fo & 1) as u8;
                        mq.encode_decision(UNI_CTX, p0);
                        mq.encode_decision(UNI_CTX, p1);
                        for dr in 0..fo {
                            state[(row + dr) * width + col].visited = true;
                        }
                        // The `fo` position becomes significant.
                        let sig_row = row + fo;
                        let sig_idx = sig_row * width + col;
                        state[sig_idx].significant = true;
                        state[sig_idx].visited = true;
                        let (sign_ctx, xor_bit) = sign_context(state, col, sig_row, width, height);
                        let sign_val = (sign[sig_idx] != 0) as u8;
                        mq.encode_decision(sign_ctx, sign_val ^ xor_bit);
                        state[sig_idx].sign = i32::from(sign_val);
                        // Remaining rows in the stripe coded normally.
                        for dr in (fo + 1)..4 {
                            let r = row + dr;
                            let i = r * width + col;
                            if state[i].visited {
                                continue;
                            }
                            let sctx = significance_context(state, col, r, width, height);
                            let sbit = mag_bit(mag, i, bit_plane);
                            mq.encode_decision(sctx, sbit);
                            state[i].visited = true;
                            if sbit == 1 {
                                state[i].significant = true;
                                let (sc, xb) = sign_context(state, col, r, width, height);
                                let sv = (sign[i] != 0) as u8;
                                mq.encode_decision(sc, sv ^ xb);
                                state[i].sign = i32::from(sv);
                            }
                        }
                        col += 1;
                        continue;
                    }
                }
            }

            // Regular cleanup: code significance for this single sample.
            let sctx = significance_context(state, col, row, width, height);
            let sbit = mag_bit(mag, idx, bit_plane);
            mq.encode_decision(sctx, sbit);
            state[idx].visited = true;
            if sbit == 1 {
                state[idx].significant = true;
                let (sc, xb) = sign_context(state, col, row, width, height);
                let sv = (sign[idx] != 0) as u8;
                mq.encode_decision(sc, sv ^ xb);
                state[idx].sign = i32::from(sv);
            }
            col += 1;
        }
        row += 1;
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Encode a single code-block's coefficients with EBCOT Tier-1.
///
/// `coeffs` are the signed quantised wavelet coefficients in row-major order;
/// `width` × `height` is the code-block size; `num_bit_planes` is the number of
/// magnitude bit-planes to code (from the MSB down) — it must equal the value
/// the decoder uses (the component bit depth) so the two halves agree.
///
/// Returns the MQ-compressed byte stream for this code-block.
pub fn encode_code_block(
    coeffs: &[i32],
    width: usize,
    height: usize,
    num_bit_planes: u8,
) -> Jp2Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(Jp2Error::InternalError(
            "code-block dimensions must be non-zero".to_string(),
        ));
    }
    if coeffs.len() < width * height {
        return Err(Jp2Error::InternalError(format!(
            "code-block coeffs too small: expected {}, got {}",
            width * height,
            coeffs.len()
        )));
    }
    if num_bit_planes == 0 {
        // Nothing to code (the decoder returns all-zeros for 0 bit-planes).
        return Ok(Vec::new());
    }

    let mag: Vec<i32> = coeffs[..width * height].iter().map(|&v| v.abs()).collect();
    let sign: Vec<i32> = coeffs[..width * height]
        .iter()
        .map(|&v| i32::from(v < 0))
        .collect();

    let mut mq = MqEncoder::new();
    let mut state = vec![CoeffState::default(); width * height];

    for bp_idx in 0..num_bit_planes {
        let bit_plane = num_bit_planes - 1 - bp_idx;

        for s in state.iter_mut() {
            s.visited = false;
        }

        if bp_idx == 0 {
            cleanup_pass(&mut mq, &mut state, &mag, &sign, width, height, bit_plane);
        } else {
            significance_propagation_pass(
                &mut mq, &mut state, &mag, &sign, width, height, bit_plane,
            );
            for s in state.iter_mut() {
                s.visited = false;
            }
            let first_mr = bp_idx == 1;
            magnitude_refinement_pass(
                &mut mq, &mut state, &mag, width, height, bit_plane, first_mr,
            );
            for s in state.iter_mut() {
                s.visited = false;
            }
            cleanup_pass(&mut mq, &mut state, &mag, &sign, width, height, bit_plane);
        }
    }

    Ok(mq.flush())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::tier1::decode_code_block;

    /// Tiny deterministic LCG for test data.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
    }

    fn roundtrip_block(coeffs: &[i32], w: usize, h: usize, nbp: u8) {
        let bytes = encode_code_block(coeffs, w, h, nbp).expect("encode");
        let block = decode_code_block(&bytes, w, h, nbp).expect("decode");
        assert_eq!(block.coeffs.len(), coeffs.len());
        for (i, (&a, &b)) in coeffs.iter().zip(block.coeffs.iter()).enumerate() {
            assert_eq!(a, b, "coeff {i} mismatch: enc {a} dec {b}");
        }
    }

    #[test]
    fn roundtrip_all_zero() {
        roundtrip_block(&vec![0i32; 16], 4, 4, 8);
    }

    #[test]
    fn roundtrip_single_positive() {
        let mut c = vec![0i32; 16];
        c[5] = 7;
        roundtrip_block(&c, 4, 4, 8);
    }

    #[test]
    fn roundtrip_single_negative() {
        let mut c = vec![0i32; 16];
        c[10] = -13;
        roundtrip_block(&c, 4, 4, 8);
    }

    #[test]
    fn roundtrip_small_values() {
        let c: Vec<i32> = (0..64).map(|i| (i as i32 % 7) - 3).collect();
        roundtrip_block(&c, 8, 8, 8);
    }

    #[test]
    fn roundtrip_rlc_stripes() {
        // Mostly zero with a few sparse coefficients to exercise the RLC path.
        let mut c = vec![0i32; 16 * 16];
        c[16 * 7 + 3] = 5;
        c[16 * 12 + 9] = -2;
        c[16 * 2 + 14] = 1;
        roundtrip_block(&c, 16, 16, 8);
    }

    #[test]
    fn roundtrip_random_dense() {
        let mut rng = Lcg::new(0xabcd_1234_5678_9999);
        let c: Vec<i32> = (0..16 * 16)
            .map(|_| {
                let v = (rng.next_u32() % 256) as i32 - 128;
                v
            })
            .collect();
        roundtrip_block(&c, 16, 16, 8);
    }

    #[test]
    fn roundtrip_random_sparse() {
        let mut rng = Lcg::new(0x1111_2222_3333_4444);
        let c: Vec<i32> = (0..16 * 16)
            .map(|_| {
                if rng.next_u32() % 8 == 0 {
                    (rng.next_u32() % 64) as i32 - 32
                } else {
                    0
                }
            })
            .collect();
        roundtrip_block(&c, 16, 16, 8);
    }

    #[test]
    fn roundtrip_non_square() {
        let c: Vec<i32> = (0..(13 * 7)).map(|i| ((i * 5) % 11) as i32 - 5).collect();
        roundtrip_block(&c, 13, 7, 8);
    }

    #[test]
    fn roundtrip_16bit_values() {
        let mut rng = Lcg::new(0x9999_8888_7777_6666);
        let c: Vec<i32> = (0..8 * 8)
            .map(|_| (rng.next_u32() % 65536) as i32 - 32768)
            .collect();
        roundtrip_block(&c, 8, 8, 16);
    }
}
