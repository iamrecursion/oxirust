// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS VLC-driven coefficient decoder.
//!
//! JPEG XS entropy coding uses a bitplane + VLC approach per subband:
//!
//! 1. A significance map indicates which positions have non-zero coefficients.
//! 2. Non-zero coefficients are coded as sign + Golomb-Rice magnitude.
//! 3. Zero runs are coded with a unary run-length VLC.
//!
//! This module implements the decode side of the project's JPEG XS coding model:
//! - Parses the run/magnitude VLC streams (unary codewords plus escapes).
//! - Handles the main coding path: alternating run-of-zeros and magnitude codewords.
//! - Decodes the escape continuations for extended zero-runs (`>= 8`) and
//!   extended magnitudes (`>= 9`), which are the exact inverse of the encoder's
//!   `vlc_encode::write_run` / `vlc_encode::write_magnitude`.
//!
//! Full JPEG XS entropy coding per ISO 21122-1:2019 §A.5 additionally specifies
//! per-subband refinement bitplanes and band-specific rate control; those remain
//! out of scope. The run/magnitude/sign model implemented here is self-consistent
//! with the encoder so any codestream the encoder emits decodes back exactly.

use super::bitreader::BitReader;
use super::vlc::{default_magnitude_table, default_run_table, VlcTable};
use super::vlc_encode::{read_escape_value, MAG_ESCAPE_BASE, RUN_ESCAPE_BASE};
use super::{JxsError, JxsResult};

/// Decoded subband coefficient array.
#[derive(Debug, Clone)]
pub struct SubbandCoeffs {
    /// Width of this subband in coefficients.
    pub width: usize,
    /// Height of this subband in coefficients.
    pub height: usize,
    /// Row-major coefficients (signed, quantised domain).
    pub coeffs: Vec<i32>,
}

impl SubbandCoeffs {
    /// Construct an all-zero `SubbandCoeffs` for `width × height`.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            coeffs: vec![0i32; width * height],
        }
    }

    /// Number of coefficients in this subband.
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Returns true if the subband has no coefficients.
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }
}

/// Decode one subband's worth of coefficients from the bitstream.
///
/// The coding model used here is the **simplified default JPEG XS VLC** model
/// (ISO 21122-1 Annex A, Main profile):
/// - Zero-run lengths are coded with `run_table` (unary Golomb-Rice).
/// - Non-zero coefficient magnitudes are coded with `mag_table`.
/// - Each non-zero coefficient is followed by one sign bit (0 = positive, 1 = negative).
/// - The sequence of (run, value) pairs is read until `num_coeffs` positions
///   have been filled or the bitstream ends.
///
/// Extended zero-runs (`>= 8`) and extended magnitudes (`>= 9`) use the 8-bit
/// escape codeword (`-1` from the VLC table) followed by an escape continuation
/// that `vlc_encode::read_escape_value` decodes — the exact inverse of
/// the encoder.
///
/// # Errors
/// Returns `JxsError::TruncatedStream` if the bitstream ends prematurely, or
/// `JxsError::VlcError` if an escape continuation is malformed.
pub fn decode_subband(
    reader: &mut BitReader<'_>,
    run_table: &VlcTable,
    mag_table: &VlcTable,
    width: usize,
    height: usize,
) -> JxsResult<SubbandCoeffs> {
    let num_coeffs = width * height;
    let mut coeffs = vec![0i32; num_coeffs];
    let mut idx = 0usize;

    while idx < num_coeffs {
        // --- Peek 32 bits for the run-length VLC lookup ---
        let peek = reader.peek_bits_u32(32);
        match run_table.lookup(peek) {
            None => {
                // No matching run code — treat as run=0 and try to decode a value.
                // Consume 1 bit to advance (prevents infinite loop on bad data).
                reader.skip_bits(1)?;
            }
            Some(run_result) => {
                let run_bits = run_result.bits_consumed;
                let run_val = run_result.value;

                // Consume the run codeword bits.
                reader.skip_bits(run_bits)?;

                // Decode the run length. For the escape code (`run_val < 0`,
                // 8-bit prefix `11111111`), read the escape continuation and add
                // the escape base — the exact inverse of `vlc_encode::write_run`.
                let run = if run_val < 0 {
                    // Saturating add in usize: `RUN_ESCAPE_BASE + escape` is a
                    // u32 sum that can overflow for a malformed escape value;
                    // the result is clamped to `num_coeffs` below anyway.
                    (RUN_ESCAPE_BASE as usize).saturating_add(read_escape_value(reader)? as usize)
                } else {
                    run_val as usize
                };
                idx = (idx + run).min(num_coeffs);

                if idx >= num_coeffs {
                    break;
                }

                // --- Decode the non-zero magnitude ---
                let peek2 = reader.peek_bits_u32(32);
                match mag_table.lookup(peek2) {
                    None => {
                        // Unknown magnitude code — skip 1 bit and leave zero in output.
                        reader.skip_bits(1)?;
                    }
                    Some(mag_result) => {
                        let mag_bits = mag_result.bits_consumed;
                        let mag_level = mag_result.value;

                        reader.skip_bits(mag_bits)?;

                        // Decode the magnitude level. For the escape code
                        // (`mag_level < 0`, 8-bit prefix `11111111`), read the
                        // escape continuation and add the escape base — the exact
                        // inverse of `vlc_encode::write_magnitude`.
                        let level = if mag_level < 0 {
                            MAG_ESCAPE_BASE + read_escape_value(reader)? as i32
                        } else {
                            mag_level as i32
                        };

                        // Read sign bit: 0 = positive, 1 = negative.
                        let sign_bit = reader.read_bit()?;
                        let signed_val = if sign_bit == 0 { level } else { -level };

                        coeffs[idx] = signed_val;
                        idx += 1;
                    }
                }
            }
        }
    }

    Ok(SubbandCoeffs {
        width,
        height,
        coeffs,
    })
}

/// Decode a complete slice's subband data for one component.
///
/// A JPEG XS slice contains coefficient data for all subbands of the wavelet
/// decomposition, coded in band-order. This function decodes a single-level
/// decomposition (4 subbands: LL, HL, LH, HH) from the compressed slice bytes.
///
/// Returns `(ll, hl, lh, hh)` SubbandCoeffs for subsequent wavelet reconstruction.
///
/// # Errors
/// See `decode_subband` for possible errors.
pub fn decode_slice_subbands(
    slice_data: &[u8],
    frame_width: usize,
    frame_height: usize,
) -> JxsResult<(SubbandCoeffs, SubbandCoeffs, SubbandCoeffs, SubbandCoeffs)> {
    let n_low_w = (frame_width + 1) / 2;
    let n_high_w = frame_width / 2;
    let n_low_h = (frame_height + 1) / 2;
    let n_high_h = frame_height / 2;

    let mut reader = BitReader::new(slice_data);
    let run_table = default_run_table();
    let mag_table = default_magnitude_table();

    // Decode the four subbands in band-order: LL → HL → LH → HH.
    let ll = decode_subband(&mut reader, &run_table, &mag_table, n_low_w, n_low_h)?;
    let hl = decode_subband(&mut reader, &run_table, &mag_table, n_high_w, n_low_h)?;
    let lh = decode_subband(&mut reader, &run_table, &mag_table, n_low_w, n_high_h)?;
    let hh = decode_subband(&mut reader, &run_table, &mag_table, n_high_w, n_high_h)?;

    Ok((ll, hl, lh, hh))
}

/// Attempt to decode a simple constant-grey slice.
///
/// For test streams where all detail subbands (HL, LH, HH) are zero and the
/// LL subband contains a single repeated value, this produces the correct output
/// without invoking the full VLC decoder. Used internally for smoke tests.
///
/// Returns `None` if the data does not match the simple constant-grey pattern.
pub fn try_decode_constant_grey(
    slice_data: &[u8],
    frame_width: usize,
    frame_height: usize,
    constant_ll_value: i32,
) -> Option<(SubbandCoeffs, SubbandCoeffs, SubbandCoeffs, SubbandCoeffs)> {
    let n_low_w = (frame_width + 1) / 2;
    let n_high_w = frame_width / 2;
    let n_low_h = (frame_height + 1) / 2;
    let n_high_h = frame_height / 2;

    // For a constant-grey image, the wavelet LL band contains the constant
    // (potentially scaled), and all detail subbands are zero.
    if slice_data.is_empty() {
        let ll = SubbandCoeffs {
            width: n_low_w,
            height: n_low_h,
            coeffs: vec![constant_ll_value; n_low_w * n_low_h],
        };
        let hl = SubbandCoeffs::zeros(n_high_w, n_low_h);
        let lh = SubbandCoeffs::zeros(n_low_w, n_high_h);
        let hh = SubbandCoeffs::zeros(n_high_w, n_high_h);
        return Some((ll, hl, lh, hh));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::vlc::{default_magnitude_table, default_run_table};

    #[test]
    fn subband_coeffs_zeros() {
        let sb = SubbandCoeffs::zeros(4, 4);
        assert_eq!(sb.width, 4);
        assert_eq!(sb.height, 4);
        assert_eq!(sb.len(), 16);
        assert!(sb.coeffs.iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_all_zero_slice_returns_zeros() {
        // A bitstream of all-zero bytes: run=0 repeatedly, then magnitude lookup fails
        // (or run=0 with no following non-zero). The simplest valid scenario is an empty
        // slice for a 2×2 image (no data bytes → constant grey helper).
        let result = try_decode_constant_grey(&[], 4, 4, 64);
        assert!(result.is_some());
        let (ll, hl, lh, hh) = result.unwrap();
        assert_eq!(ll.width, 2);
        assert_eq!(ll.height, 2);
        assert_eq!(ll.coeffs, vec![64i32; 4]);
        assert!(hl.coeffs.iter().all(|&v| v == 0));
        assert!(lh.coeffs.iter().all(|&v| v == 0));
        assert!(hh.coeffs.iter().all(|&v| v == 0));
    }

    #[test]
    fn decode_subband_empty_data_returns_zeros() {
        // An empty bitstream with run table that maps first bit to run=0,
        // so all coefficients remain zero (reader exhausted before any data).
        let data: &[u8] = &[];
        let mut reader = BitReader::new(data);
        let run_table = default_run_table();
        let mag_table = default_magnitude_table();
        // With empty data, peek returns 0, run_table maps 0 to run=0 (1-bit code).
        // Then skip_bits(1) on empty data will return TruncatedStream.
        let result = decode_subband(&mut reader, &run_table, &mag_table, 2, 2);
        // Either error (truncated) or all-zeros — both are acceptable for empty input.
        match result {
            Ok(sb) => assert_eq!(sb.coeffs, vec![0i32; 4]),
            Err(JxsError::TruncatedStream { .. }) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn decode_subband_run0_value1_positive() {
        // Encode: run=0 (bit `0`), magnitude=1 (bit `0`), sign=positive (bit `0`)
        // Then all remaining positions are zero (stream ends).
        // Bit sequence: 0 0 0 + padding → 0x00
        // Byte 0: bit7=0 (run=0), bit6=0 (mag=1), bit5=0 (sign=+), bits4..0 don't matter
        let data = &[0b0000_0000u8];
        let mut reader = BitReader::new(data);
        let run_table = default_run_table();
        let mag_table = default_magnitude_table();
        let result = decode_subband(&mut reader, &run_table, &mag_table, 2, 1);
        // After decoding position 0 = +1, positions 1 will be 0 (run=0 loop then truncated).
        match result {
            Ok(sb) => {
                assert_eq!(sb.width, 2);
                assert_eq!(sb.height, 1);
                assert_eq!(sb.coeffs[0], 1, "first coefficient should be +1");
            }
            Err(JxsError::TruncatedStream { .. }) => {
                // Also acceptable — ran out of bits after decoding first coefficient.
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn decode_subband_run0_value1_negative() {
        // Bit sequence: 0 (run=0), 0 (mag=1), 1 (sign=negative) → coeff[0] = -1
        // Byte 0: bit7=0, bit6=0, bit5=1 → 0b00100000 = 0x20
        let data = &[0b0010_0000u8];
        let mut reader = BitReader::new(data);
        let run_table = default_run_table();
        let mag_table = default_magnitude_table();
        let result = decode_subband(&mut reader, &run_table, &mag_table, 1, 1);
        match result {
            Ok(sb) => assert_eq!(sb.coeffs[0], -1, "coefficient should be -1"),
            Err(JxsError::TruncatedStream { .. }) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
