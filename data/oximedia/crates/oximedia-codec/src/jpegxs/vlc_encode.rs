// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS forward VLC (encoder side).
//!
//! This module is the **exact inverse** of the decode VLC machinery in
//! [`super::vlc`] and [`super::entropy`]. It emits the run, magnitude and sign
//! codewords that [`super::entropy::decode_subband`] reads back:
//!
//! - **Run code** (number of zero coefficients before the next non-zero):
//!   `run` ones followed by a `0` for `run` in `0..=7`. For `run >= 8` the
//!   8-bit escape `11111111` is emitted, followed by an escape continuation
//!   encoding `run - 8`.
//! - **Magnitude code** (1-indexed level = `|coeff|`): `level - 1` ones
//!   followed by a `0` for `level` in `1..=8`. For `level >= 9` the 8-bit
//!   escape `11111111` is emitted, followed by an escape continuation encoding
//!   `level - 9`.
//! - **Sign bit**: `0` for positive, `1` for negative.
//!
//! ## Escape continuation
//!
//! A non-negative extra value `e` is coded as a 6-bit `nbits` field giving the
//! number of significant bits of `e` (0 when `e == 0`), followed by exactly
//! `nbits` bits of `e`, most-significant bit first. This is self-delimiting and
//! exactly reversible by `read_escape_value`, which the decoder calls.

use super::bitreader::BitReader;
use super::bitwriter::BitWriter;
use super::{JxsError, JxsResult};

/// Largest directly-coded zero run (unary). Runs `>= RUN_ESCAPE_BASE` use the escape.
pub(crate) const RUN_ESCAPE_BASE: u32 = 8;
/// Largest directly-coded magnitude level (unary). Levels `>= MAG_ESCAPE_BASE`
/// use the escape.
pub(crate) const MAG_ESCAPE_BASE: i32 = 9;
/// Number of bits used to encode the significant-bit count in an escape
/// continuation. `u32` values need at most 32 significant bits, which fits in 6.
const ESCAPE_NBITS_FIELD: u8 = 6;

/// Write an escape continuation value `e` (the extra magnitude/run beyond the
/// escape base). Reversible by `read_escape_value`.
pub(crate) fn write_escape_value(writer: &mut BitWriter, e: u32) {
    let nbits: u8 = if e == 0 {
        0
    } else {
        (32 - e.leading_zeros()) as u8
    };
    writer.write_bits_u32(u32::from(nbits), ESCAPE_NBITS_FIELD);
    if nbits > 0 {
        writer.write_bits_u32(e, nbits);
    }
}

/// Read an escape continuation value written by `write_escape_value`.
///
/// # Errors
/// Returns `JxsError::TruncatedStream` if the stream ends prematurely, or
/// `JxsError::VlcError` if the encoded bit count exceeds 32.
pub(crate) fn read_escape_value(reader: &mut BitReader<'_>) -> JxsResult<u32> {
    let nbits = reader.read_bits_u32(ESCAPE_NBITS_FIELD)? as u8;
    if nbits == 0 {
        return Ok(0);
    }
    if nbits > 32 {
        return Err(JxsError::VlcError(format!(
            "escape continuation nbits={nbits} exceeds 32"
        )));
    }
    reader.read_bits_u32(nbits)
}

/// Emit the run codeword for a zero-run of length `run`.
pub(crate) fn write_run(writer: &mut BitWriter, run: u32) {
    if run < RUN_ESCAPE_BASE {
        // Unary: `run` ones then a terminating zero.
        for _ in 0..run {
            writer.write_bit(1);
        }
        writer.write_bit(0);
    } else {
        // Escape prefix `11111111`, then continuation for (run - RUN_ESCAPE_BASE).
        writer.write_bits_u32(0xFF, 8);
        write_escape_value(writer, run - RUN_ESCAPE_BASE);
    }
}

/// Emit the magnitude codeword for a 1-indexed level (`level >= 1`).
fn write_magnitude(writer: &mut BitWriter, level: i32) {
    if level < MAG_ESCAPE_BASE {
        // Unary: `level - 1` ones then a terminating zero.
        for _ in 0..(level - 1) {
            writer.write_bit(1);
        }
        writer.write_bit(0);
    } else {
        // Escape prefix `11111111`, then continuation for (level - MAG_ESCAPE_BASE).
        writer.write_bits_u32(0xFF, 8);
        write_escape_value(writer, (level - MAG_ESCAPE_BASE) as u32);
    }
}

/// Emit one non-zero coefficient: magnitude codeword followed by the sign bit.
///
/// `value` must be non-zero. The 1-indexed level is `|value|`; the sign bit is
/// `1` when `value < 0`, else `0`.
fn write_nonzero_coeff(writer: &mut BitWriter, value: i32) {
    let level = value.unsigned_abs() as i32;
    write_magnitude(writer, level);
    writer.write_bit(if value < 0 { 1 } else { 0 });
}

/// Encode one subband's coefficients into `writer`, exactly mirroring
/// [`super::entropy::decode_subband`].
///
/// The coefficient stream is modelled as a sequence of `(zero_run, value)`
/// pairs: skip `zero_run` zero coefficients, then emit one non-zero `value`.
/// A final trailing zero-run (if any) is emitted as a single run code so the
/// decoder advances to the end of the subband and terminates its loop. When the
/// last coefficient is non-zero and sits at the final position, no trailing run
/// is emitted (the decoder's `while idx < num_coeffs` loop exits naturally).
pub fn encode_subband(writer: &mut BitWriter, coeffs: &[i32]) {
    let num = coeffs.len();
    let mut zero_run: u32 = 0;
    let mut last_nonzero: Option<usize> = None;

    for (idx, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            zero_run += 1;
        } else {
            write_run(writer, zero_run);
            write_nonzero_coeff(writer, c);
            zero_run = 0;
            last_nonzero = Some(idx);
        }
    }

    // Trailing zeros after the last non-zero coefficient.
    let trailing = match last_nonzero {
        Some(pos) => (num - (pos + 1)) as u32,
        None => num as u32, // all-zero subband
    };
    if trailing > 0 {
        write_run(writer, trailing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::entropy::decode_subband;
    use crate::jpegxs::vlc::{default_magnitude_table, default_run_table};

    /// Round-trip one subband: encode coeffs, decode back, assert equality.
    fn roundtrip(coeffs: &[i32], width: usize, height: usize) {
        assert_eq!(coeffs.len(), width * height);
        let mut w = BitWriter::new();
        encode_subband(&mut w, coeffs);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let run_t = default_run_table();
        let mag_t = default_magnitude_table();
        let decoded = decode_subband(&mut r, &run_t, &mag_t, width, height).expect("decode");
        assert_eq!(decoded.coeffs, coeffs, "subband round-trip mismatch");
    }

    #[test]
    fn escape_value_roundtrip_small() {
        for e in 0u32..=300 {
            let mut w = BitWriter::new();
            write_escape_value(&mut w, e);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_escape_value(&mut r).unwrap(), e, "escape e={e}");
        }
    }

    #[test]
    fn escape_value_roundtrip_powers_of_two() {
        for k in 0u32..32 {
            let e = 1u32 << k;
            let mut w = BitWriter::new();
            write_escape_value(&mut w, e);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            assert_eq!(read_escape_value(&mut r).unwrap(), e, "escape e=2^{k}");
        }
    }

    #[test]
    fn run_then_value_roundtrips() {
        // 3 zeros, then +5, then 2 zeros, then -2.
        roundtrip(&[0, 0, 0, 5, 0, 0, -2], 7, 1);
    }

    #[test]
    fn all_levels_1_to_8_roundtrip() {
        // Each non-zero separated by one zero so runs stay small.
        let mut coeffs = Vec::new();
        for level in 1..=8 {
            coeffs.push(level);
            coeffs.push(0);
        }
        let n = coeffs.len();
        roundtrip(&coeffs, n, 1);
    }

    #[test]
    fn levels_above_escape_roundtrip() {
        // Magnitudes 9..40 force the magnitude escape path.
        let coeffs: Vec<i32> = (9..40).collect();
        let n = coeffs.len();
        roundtrip(&coeffs, n, 1);
    }

    #[test]
    fn negative_values_roundtrip() {
        roundtrip(&[-1, -8, -9, -100, -255], 5, 1);
    }

    #[test]
    fn long_zero_run_uses_escape() {
        // 50 zeros then a 7 — run escape needed (50 >= 8).
        let mut coeffs = vec![0i32; 50];
        coeffs.push(7);
        let n = coeffs.len();
        roundtrip(&coeffs, n, 1);
    }

    #[test]
    fn all_zero_subband_roundtrips() {
        roundtrip(&vec![0i32; 64], 8, 8);
    }

    #[test]
    fn constant_nonzero_subband_roundtrips() {
        // Mimics an LL band of a constant image: all equal large values.
        roundtrip(&vec![128i32; 16], 4, 4);
    }

    #[test]
    fn last_coeff_nonzero_at_end_roundtrips() {
        // No trailing run should be emitted; decoder loop exits naturally.
        roundtrip(&[0, 0, 3, 0, 5], 5, 1);
    }

    #[test]
    fn mixed_large_values_2d_roundtrip() {
        let coeffs: Vec<i32> = vec![255, 0, -200, 0, 0, 0, 0, 0, 0, 17, 0, -9, 8, -8, 1, -1];
        roundtrip(&coeffs, 4, 4);
    }
}
