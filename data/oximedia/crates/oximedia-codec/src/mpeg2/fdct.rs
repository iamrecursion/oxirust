//! 8×8 forward DCT for MPEG-2 block encoding (ISO/IEC 13818-2 Annex A).
//!
//! This is the separable two-dimensional **forward** DCT-II — the exact
//! mathematical inverse of the [`super::idct`] kernel — so that
//! `fdct_8x8 → dequantize_intra → idct_8x8` reconstructs a block within the
//! IEEE-1180 / quantisation tolerance.
//!
//! ## Scale matching
//!
//! The decoder's [`super::idct::idct_8x8`] maps a DC-only coefficient
//! `F(0,0) = D` to the flat spatial value `D / 8`. The forward transform here
//! is normalised so that a flat spatial block of value `V` produces
//! `F(0,0) ≈ 8·V`, i.e. it lands the coefficients in the same fixed-point scale
//! that [`super::dequant::dequantize_intra`] reconstructs (`F[0] = 8·QF[0]` for
//! `intra_dc_precision == 0`, `F_ac = QF·W·q_scale/16`). Forward quantisation
//! ([`super::quantize_fwd`]) then divides by that same scale, so the pipeline is
//! consistent end to end.
//!
//! The Q15 cosine constants below are an independent copy of the ones used by
//! the inverse transform (the decoder keeps them module-private), so the two
//! transforms share identical fixed-point cosines and round-trip cleanly.

/// Q15 cosine constants. `COS_Q15[k] = round(cos(k·π/16) · 32768)`. Identical to
/// the decoder's inverse-DCT table so the forward/inverse pair matches exactly.
const COS_Q15: [i64; 9] = [
    32768, // cos(0π/16) = 1.0
    32138, // cos(1π/16)
    30274, // cos(2π/16)
    27246, // cos(3π/16)
    23170, // cos(4π/16) = 1/√2
    18205, // cos(5π/16)
    12540, // cos(6π/16)
    6393,  // cos(7π/16)
    0,     // cos(8π/16) = 0
];

/// `cos(angle·π/16)` in Q15 with 32-step periodicity (mirrors the decoder).
fn cos_q15_periodic(angle: usize) -> i64 {
    let angle = angle % 32;
    match angle {
        0..=8 => COS_Q15[angle],
        9..=15 => -COS_Q15[16 - angle],
        16..=23 => -COS_Q15[angle - 16],
        _ => COS_Q15[32 - angle],
    }
}

/// Total right-shift applied after both forward passes.
///
/// Derivation: for a flat spatial block of value `V`, the unshifted two-pass
/// accumulator at `(v_freq, h_freq) = (0, 0)` is
/// `64·V·COS_Q15[4]²` (eight samples per pass, DC weighted by `COS_Q15[4]`
/// each pass). To land on `F(0,0) = 8·V` we need
/// `64·V·COS_Q15[4]² / 2^S = 8·V`, i.e. `2^S = 8·COS_Q15[4]² = 8·23170² ≈ 2³²`.
const FDCT_SHIFT: u32 = 32;

/// 1-D 8-point forward DCT in Q15 (un-normalised; the column pass applies the
/// final [`FDCT_SHIFT`]). The DC term (`k == 0`) carries the `C(0) = 1/√2`
/// weight via `COS_Q15[4]`, matching the inverse transform's DC handling.
fn fdct_1d(input: &[i64; 8]) -> [i64; 8] {
    let mut out = [0i64; 8];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut acc: i64 = 0;
        for (n, &x) in input.iter().enumerate() {
            let cos_val = if k == 0 {
                COS_Q15[4]
            } else {
                cos_q15_periodic((2 * n + 1) * k)
            };
            acc += x * cos_val;
        }
        *slot = acc;
    }
    out
}

/// Two-dimensional 8×8 forward DCT.
///
/// Input: 64 spatial samples in **raster order** (`row * 8 + col`), already
/// signed (the caller subtracts the level offset before calling — for MPEG-2
/// intra the samples are passed directly, since the DC predictor lives in the
/// quantised domain and the reconstruction applies no extra level shift).
///
/// Output: 64 DCT coefficients in raster order at `[v_freq * 8 + h_freq]`,
/// matching the layout the inverse DCT consumes.
#[must_use]
pub fn fdct_8x8(block: &[i32; 64]) -> [i32; 64] {
    // Pass 1: row transform → horizontal-frequency components, unshifted.
    let mut intermediate = [0i64; 64];
    for row in 0..8 {
        let row_in: [i64; 8] = std::array::from_fn(|c| i64::from(block[row * 8 + c]));
        let row_out = fdct_1d(&row_in);
        for (h_freq, &v) in row_out.iter().enumerate() {
            intermediate[row * 8 + h_freq] = v;
        }
    }

    // Pass 2: column transform → vertical-frequency components, then normalise.
    let round: i64 = 1 << (FDCT_SHIFT - 1);
    let mut output = [0i32; 64];
    for h_freq in 0..8usize {
        for v_freq in 0..8usize {
            let mut acc: i64 = 0;
            for r in 0..8usize {
                let cos_val = if v_freq == 0 {
                    COS_Q15[4]
                } else {
                    cos_q15_periodic((2 * r + 1) * v_freq)
                };
                acc += intermediate[r * 8 + h_freq] * cos_val;
            }
            output[v_freq * 8 + h_freq] = ((acc + round) >> FDCT_SHIFT) as i32;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpeg2::idct::idct_8x8;

    #[test]
    fn dc_only_forward_gives_eight_times_mean() {
        // Flat spatial block of value V → DC coefficient ≈ 8·V.
        for v in [0i32, 16, 64, 128, 200, 255] {
            let block = [v; 64];
            let freq = fdct_8x8(&block);
            assert!(
                (freq[0] - 8 * v).abs() <= 1,
                "DC for V={v}: got {}, expected ~{}",
                freq[0],
                8 * v
            );
            // All AC must be ~0 for a flat block.
            for (i, &c) in freq.iter().enumerate().skip(1) {
                assert!(c.abs() <= 1, "AC[{i}] for flat block should be ~0, got {c}");
            }
        }
    }

    #[test]
    fn zero_block_is_zero() {
        let freq = fdct_8x8(&[0i32; 64]);
        assert!(freq.iter().all(|&v| v == 0));
    }

    #[test]
    fn fdct_idct_round_trip_flat() {
        let block = [100i32; 64];
        let freq = fdct_8x8(&block);
        let spatial = idct_8x8(&freq);
        for (i, &v) in spatial.iter().enumerate() {
            assert!(
                (v - 100).abs() <= 2,
                "flat round-trip [{i}]: expected 100, got {v}"
            );
        }
    }

    #[test]
    fn fdct_idct_round_trip_ramp() {
        // Integer fixed-point round-trip noise allowed up to ±16 LSB.
        let block: [i32; 64] = std::array::from_fn(|i| (i as i32) - 32);
        let freq = fdct_8x8(&block);
        let spatial = idct_8x8(&freq);
        for (i, &v) in spatial.iter().enumerate() {
            assert!(
                (v - block[i]).abs() <= 16,
                "ramp round-trip [{i}]: expected {}, got {v}",
                block[i]
            );
        }
    }

    #[test]
    fn dc_energy_dominates_for_smooth_block() {
        let block: [i32; 64] = std::array::from_fn(|i| {
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            r * 6 + c * 3 + 40
        });
        let freq = fdct_8x8(&block);
        assert!(
            freq[0].abs() >= freq[63].abs(),
            "DC {} should dominate AC63 {}",
            freq[0],
            freq[63]
        );
    }

    #[test]
    fn negative_flat_block() {
        // The DCT operates on signed input; verify a constant negative block.
        let block = [-30i32; 64];
        let freq = fdct_8x8(&block);
        assert!((freq[0] - (8 * -30)).abs() <= 1, "neg DC {}", freq[0]);
    }
}
