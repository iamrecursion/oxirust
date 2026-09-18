//! ProRes entropy encoding (RDD 36 §6.5.5–6.5.6).
//!
//! Mirror of [`super::entropy`]'s decoder. Uses the same adaptive
//! Golomb-Rice codes:
//!
//! For unsigned value `v` with parameter `k`:
//!
//! ```text
//!  quotient  q = v >> k
//!  remainder r = v & ((1 << k) - 1)
//!
//!  bitstream = (q ones) (one zero) (r in k bits)
//! ```
//!
//! DC coefficients use differential coding: each block's DC is stored as
//! the signed delta from the previous block's DC. AC coefficients are
//! run/level pairs with an implicit +1 bias on the level magnitude.

use super::bitwriter::BitWriter;
use super::entropy::{next_k_ac_level, next_k_ac_run, next_k_dc};

/// Encode one unsigned Golomb-Rice codeword with parameter `k`.
///
/// Writes `(value >> k)` ones, then a zero terminator, then the
/// `k`-bit remainder.
pub fn encode_unsigned_codeword(writer: &mut BitWriter, value: u32, k: u32) {
    let quotient = value >> k;
    let remainder = value & ((1u32 << k) - 1);
    // Write `quotient` one-bits.
    for _ in 0..quotient {
        writer.write_bit(true);
    }
    // Write the zero terminator.
    writer.write_bit(false);
    // Write the `k`-bit remainder MSB-first.
    if k > 0 {
        writer.write_bits(remainder, k as u8);
    }
}

/// Encode one signed ProRes codeword.
///
/// Format: unsigned magnitude codeword, then 1 sign bit (1 = negative)
/// if the magnitude is non-zero.
pub fn encode_signed_codeword(writer: &mut BitWriter, value: i32, k: u32) {
    let magnitude = value.unsigned_abs();
    encode_unsigned_codeword(writer, magnitude, k);
    if magnitude != 0 {
        writer.write_bit(value < 0);
    }
}

/// Encode one 8×8 block of quantized coefficients in ProRes scan order.
///
/// `quantized` must already be in zigzag-scan order (index 0 = DC, 1..63
/// = AC coefficients in progressive zigzag scan order). `prev_dc` is the
/// running DC predictor for differential coding.
///
/// Returns the new DC predictor (the absolute DC value of this block).
pub fn encode_block(writer: &mut BitWriter, quantized: &[i16; 64], prev_dc: i16) -> i16 {
    // ─── DC coefficient ──────────────────────────────────────────────
    let dc = quantized[0];
    let dc_delta = i32::from(dc) - i32::from(prev_dc);
    let k_dc = next_k_dc(prev_dc.unsigned_abs() as u32);
    encode_signed_codeword(writer, dc_delta, k_dc);

    // ─── AC coefficients (positions 1..64 in scan order) ────────────
    // Encoder mirrors the decoder exactly:
    // - Write run (count of consecutive zeros).
    // - Then write level magnitude - 1 (implicit +1 bias) + sign bit.
    // - Adapt k values identically to the decoder.
    // - Terminate with a trailing run = remaining positions if last
    //   non-zero coefficient is reached early.
    let mut k_run = 3u32;
    let mut k_level = 1u32;
    let mut pos = 1usize;

    // Find the last non-zero AC coefficient position.
    let last_nonzero = {
        let mut last = 0usize;
        for i in 1..64 {
            if quantized[i] != 0 {
                last = i;
            }
        }
        last
    };

    while pos < 64 {
        // Count the run of zeros from pos.
        let mut run = 0u32;
        while (pos + run as usize) < 64 && quantized[pos + run as usize] == 0 {
            run += 1;
        }

        if pos + run as usize >= 64 || pos > last_nonzero {
            // No more non-zero coefficients; encode trailing EOB run.
            let remaining = (64 - pos) as u32;
            encode_unsigned_codeword(writer, remaining, k_run);
            break;
        }

        // Encode the run.
        encode_unsigned_codeword(writer, run, k_run);
        pos += run as usize;

        // Encode the level (with +1 bias removed, so encode magnitude-1).
        let level = quantized[pos];
        let magnitude = level.unsigned_abs() as u32;
        // magnitude >= 1 (we only get here on non-zero coefficients).
        let mag_minus1 = magnitude - 1;
        encode_unsigned_codeword(writer, mag_minus1, k_level);
        // Sign bit.
        writer.write_bit(level < 0);

        // Adapt.
        k_run = next_k_ac_run(run);
        k_level = next_k_ac_level(magnitude);
        pos += 1;
    }

    dc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::bitreader::BitReader;
    use crate::prores::entropy::{decode_block, decode_signed_codeword, decode_unsigned_codeword};

    #[test]
    fn encode_decode_unsigned_roundtrip() {
        // For ProRes, values are always sent with a k large enough that the
        // unary quotient never exceeds 31 (decoder limit). We test realistic
        // k/value combinations.
        let cases: &[(u32, u32)] = &[
            (0, 0),
            (0, 1),
            (0, 3),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 5),
            (2, 0),
            (2, 3),
            (2, 7),
            (2, 15),
            (3, 0),
            (3, 7),
            (3, 15),
            (3, 63),
            (4, 0),
            (4, 15),
            (4, 31),
            (4, 127),
            (5, 0),
            (5, 31),
            (5, 255),
            (7, 127),
            (7, 1023),
        ];
        for &(k, val) in cases {
            let mut w = BitWriter::new();
            encode_unsigned_codeword(&mut w, val, k);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = decode_unsigned_codeword(&mut r, k).expect("decode");
            assert_eq!(decoded, val, "k={k}, val={val}");
        }
    }

    #[test]
    fn encode_decode_signed_roundtrip() {
        let cases: &[(u32, i32)] = &[
            (0, 0),
            (0, 1),
            (0, -1),
            (0, 3),
            (0, -3),
            (3, 0),
            (3, 7),
            (3, -7),
            (5, -100),
            (5, 100),
            (7, 500),
            (7, -500),
            (7, 1023),
            (7, -1023),
        ];
        for &(k, val) in cases {
            let mut w = BitWriter::new();
            encode_signed_codeword(&mut w, val, k);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let decoded = decode_signed_codeword(&mut r, k).expect("decode");
            assert_eq!(decoded, val, "k={k}, val={val}");
        }
    }

    #[test]
    fn encode_block_decode_block_dc_only() {
        // DC value must be small enough that with k=0 (used for prev_dc=0),
        // the quotient doesn't exceed the decoder's 31-count limit.
        // DC delta = 12 - 0 = 12, k=0 → quotient=12, ok.
        let mut quantized = [0i16; 64];
        quantized[0] = 12;
        let mut w = BitWriter::new();
        encode_block(&mut w, &quantized, 0);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let (decoded_coeffs, new_dc) = decode_block(&mut r, 0).expect("decode");
        assert_eq!(new_dc, 12, "DC should be 12");
        assert_eq!(decoded_coeffs[0], 12);
        for i in 1..64 {
            assert_eq!(decoded_coeffs[i], 0, "AC[{i}] should be 0");
        }
    }

    #[test]
    fn encode_block_decode_block_ac_roundtrip() {
        // Block with several non-zero AC coefficients.
        let mut quantized = [0i16; 64];
        quantized[0] = 100; // DC
        quantized[1] = 5; // AC at scan pos 1
        quantized[5] = -3; // AC at scan pos 5 (run of 3 zeros)
        quantized[20] = 7; // AC further out
        quantized[63] = -1; // Last AC

        let mut w = BitWriter::new();
        let new_dc = encode_block(&mut w, &quantized, 50);
        assert_eq!(new_dc, 100, "encode_block returns absolute DC");

        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let (decoded_coeffs, decoded_dc) = decode_block(&mut r, 50).expect("decode");
        assert_eq!(decoded_dc, 100);
        assert_eq!(decoded_coeffs[0], 100);
        assert_eq!(decoded_coeffs[1], 5);
        assert_eq!(decoded_coeffs[5], -3);
        assert_eq!(decoded_coeffs[20], 7);
        assert_eq!(decoded_coeffs[63], -1);
        // Zeros in between should remain zero.
        assert_eq!(decoded_coeffs[2], 0);
        assert_eq!(decoded_coeffs[4], 0);
    }

    #[test]
    fn encode_block_all_zeros() {
        let quantized = [0i16; 64];
        let mut w = BitWriter::new();
        encode_block(&mut w, &quantized, 0);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        let (decoded_coeffs, decoded_dc) = decode_block(&mut r, 0).expect("decode");
        assert_eq!(decoded_dc, 0);
        assert!(decoded_coeffs.iter().all(|&v| v == 0));
    }
}
