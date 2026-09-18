//! Forward variable-length coding for MPEG-2 intra blocks (ISO/IEC 13818-2
//! Annex B), the encoder-side inverse of [`super::vlc_tables`] /
//! [`super::entropy`].
//!
//! Three things are emitted:
//!
//! - **DC size + differential** (Tables B-12 luma / B-13 chroma): the size
//!   category of the DC differential is looked up and its codeword written,
//!   followed by `size` differential bits in the offset-binary form the decoder
//!   reads (`raw` for positive, `raw + 2^size - 1` for negative).
//! - **AC run/level** (Tables B-14 default / B-15 alternate): for each
//!   `(run, level)` pair a table codeword (plus a separate sign bit) is written;
//!   when the pair is not representable by a *uniquely decodable* table entry the
//!   MPEG-2 **escape** sequence is used (`0000 01` + 6-bit run + 12-bit signed
//!   level).
//! - **End-of-block** (the table's EOB code).
//!
//! ## Decoder-faithful table inversion
//!
//! The decode tables in [`super::vlc_tables`] contain a few duplicate
//! codewords (e.g. in B-14 the 13-bit `0000000011110` is listed for both
//! `(12, 2)` and `(16, 1)`; the decoder's [`super::vlc_tables::match_vlc`]
//! returns the first array match). To stay bit-exact with the decoder, this
//! module never trusts a raw table code: each candidate is **verified** by
//! decoding it back through `match_vlc`, and a pair that has no verifiable code
//! is escaped instead. That guarantees every emitted symbol round-trips.

use super::bitwriter::BitWriter;
use super::vlc_tables::{match_vlc, AcSymbol, AcTablePtr, DcTablePtr, EOB_RUN, ESCAPE_RUN};
use super::Mpeg2Error;
use super::Mpeg2Result;

/// Compute the DC `size` category for a differential value: the number of bits
/// needed to represent `|diff|` (0 for `diff == 0`).
#[must_use]
pub fn dc_size_category(diff: i32) -> u8 {
    if diff == 0 {
        0
    } else {
        let mag = diff.unsigned_abs();
        (32 - mag.leading_zeros()) as u8
    }
}

/// Emit the offset-binary differential bits for `diff` given its `size`.
///
/// For `diff > 0` the raw value is `diff` (leading bit 1); for `diff < 0` it is
/// `diff + (2^size - 1)` (leading bit 0) — exactly what
/// [`super::entropy::decode_dc`] inverts.
fn write_dc_differential_bits(writer: &mut BitWriter, diff: i32, size: u8) {
    if size == 0 {
        return;
    }
    let raw = if diff > 0 {
        diff as u32
    } else {
        (diff + ((1i32 << size) - 1)) as u32
    };
    writer.write_bits(raw, size);
}

/// Look up the DC-size codeword for `size` in `table` (B-12 or B-13).
fn dc_size_codeword(table: DcTablePtr, size: u8) -> Mpeg2Result<(u16, u8)> {
    for &(code, len, sz) in table {
        if sz == size {
            return Ok((code, len));
        }
    }
    Err(Mpeg2Error::InvalidData(format!(
        "no DC size codeword for size category {size}"
    )))
}

/// Encode one DC differential: its size codeword (from `table`) followed by the
/// differential bits.
///
/// # Errors
///
/// Returns [`Mpeg2Error::InvalidData`] if `diff` is too large for the size
/// table (cannot happen for the legal `[-2047, 2047]` quantised DC range).
pub fn encode_dc(writer: &mut BitWriter, table: DcTablePtr, diff: i32) -> Mpeg2Result<()> {
    let size = dc_size_category(diff);
    let (code, len) = dc_size_codeword(table, size)?;
    writer.write_bits(u32::from(code), len);
    write_dc_differential_bits(writer, diff, size);
    Ok(())
}

/// Build an MSB-aligned 32-bit peek word from a right-justified `code` of `len`
/// bits, so it can be validated through the decoder's [`match_vlc`].
fn msb_aligned(code: u32, len: u8) -> u32 {
    if len == 0 {
        return 0;
    }
    code << (32 - u32::from(len))
}

/// Verify that `(code, len)` decodes to exactly `(run, level)` via the decoder
/// matcher — i.e. it is the unique shortest match the decoder will pick.
fn code_decodes_to(table: AcTablePtr, code: u32, len: u8, run: u8, level: u16) -> bool {
    let peek = msb_aligned(code, len);
    match match_vlc(table, peek) {
        Ok(AcSymbol::RunLevel {
            run: r,
            level: l,
            bits,
        }) => r == run && l == level && bits == len,
        _ => false,
    }
}

/// Find a *verifiable* table codeword for `(run, level)` in `table`, preferring
/// the shortest length. Returns `None` if no entry decodes back correctly (the
/// caller then escapes).
fn find_ac_codeword(table: AcTablePtr, run: u8, level: u16) -> Option<(u32, u8)> {
    let mut best: Option<(u32, u8)> = None;
    for &(code, len, r, l) in table {
        if r == run && l == level && code_decodes_to(table, code, len, run, level) {
            match best {
                Some((_, blen)) if blen <= len => {}
                _ => best = Some((code, len)),
            }
        }
    }
    best
}

/// Locate the escape and EOB codewords in `table` (their lengths differ between
/// B-14 and B-15).
fn special_codes(table: AcTablePtr) -> Mpeg2Result<((u32, u8), (u32, u8))> {
    let mut escape: Option<(u32, u8)> = None;
    let mut eob: Option<(u32, u8)> = None;
    for &(code, len, run, _) in table {
        match run {
            ESCAPE_RUN => escape = Some((code, len)),
            EOB_RUN => eob = Some((code, len)),
            _ => {}
        }
    }
    match (escape, eob) {
        (Some(e), Some(b)) => Ok((e, b)),
        _ => Err(Mpeg2Error::InvalidData(
            "AC table missing escape or EOB codeword".into(),
        )),
    }
}

/// Emit the MPEG-2 escape sequence for `(run, signed_level)`: the escape prefix
/// followed by a 6-bit run and a 12-bit two's-complement level.
fn write_escape(writer: &mut BitWriter, escape: (u32, u8), run: u8, signed_level: i32) {
    let (code, len) = escape;
    writer.write_bits(code, len);
    writer.write_bits(u32::from(run), 6);
    let level12 = (signed_level & 0xFFF) as u32;
    writer.write_bits(level12, 12);
}

/// Encode one AC `(run, signed_level)` pair into `writer` using `table`.
///
/// A verifiable table codeword (plus a separate sign bit) is used when present;
/// otherwise the escape sequence is emitted. `signed_level` must be non-zero and
/// within `[-2047, 2047]` (the forbidden `-2048` and `0` are rejected).
///
/// # Errors
///
/// Returns [`Mpeg2Error::InvalidData`] for an out-of-range run/level.
pub fn encode_ac_run_level(
    writer: &mut BitWriter,
    table: AcTablePtr,
    run: u8,
    signed_level: i32,
) -> Mpeg2Result<()> {
    if signed_level == 0 || !(-2047..=2047).contains(&signed_level) {
        return Err(Mpeg2Error::InvalidData(format!(
            "AC level {signed_level} out of legal non-zero range [-2047, 2047]"
        )));
    }
    if run > 63 {
        return Err(Mpeg2Error::InvalidData(format!("AC run {run} exceeds 63")));
    }
    // Table levels are stored as small magnitudes (`u16`). The validated level
    // is within ±2047 so the conversion always succeeds; attempt a verifiable
    // table code, otherwise fall through to the escape sequence.
    if let Ok(magnitude) = u16::try_from(signed_level.unsigned_abs()) {
        if let Some((code, len)) = find_ac_codeword(table, run, magnitude) {
            writer.write_bits(code, len);
            // Sign bit: 0 = positive, 1 = negative (matches decode_ac).
            writer.write_bit(signed_level < 0);
            return Ok(());
        }
    }
    let (escape, _eob) = special_codes(table)?;
    write_escape(writer, escape, run, signed_level);
    Ok(())
}

/// Emit the end-of-block codeword for `table`.
///
/// # Errors
///
/// Returns [`Mpeg2Error::InvalidData`] if the table lacks an EOB code.
pub fn encode_eob(writer: &mut BitWriter, table: AcTablePtr) -> Mpeg2Result<()> {
    let (_escape, eob) = special_codes(table)?;
    let (code, len) = eob;
    writer.write_bits(code, len);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpeg2::bitreader::BitReader;
    use crate::mpeg2::entropy::{decode_ac, decode_dc, BlockComponent, DcPredictors};
    use crate::mpeg2::vlc_tables::{AC_TABLE_B14, AC_TABLE_B15, DC_SIZE_CHROMA, DC_SIZE_LUMA};

    #[test]
    fn dc_size_category_values() {
        assert_eq!(dc_size_category(0), 0);
        assert_eq!(dc_size_category(1), 1);
        assert_eq!(dc_size_category(-1), 1);
        assert_eq!(dc_size_category(2), 2);
        assert_eq!(dc_size_category(3), 2);
        assert_eq!(dc_size_category(-3), 2);
        assert_eq!(dc_size_category(4), 3);
        assert_eq!(dc_size_category(255), 8);
        assert_eq!(dc_size_category(-256), 9);
    }

    /// Round-trip a DC differential through encode_dc → decode_dc.
    fn dc_round_trip(table_is_luma: bool, diff: i32) {
        let component = if table_is_luma {
            BlockComponent::Luma
        } else {
            BlockComponent::Cb
        };
        let table = if table_is_luma {
            DC_SIZE_LUMA
        } else {
            DC_SIZE_CHROMA
        };
        let mut w = BitWriter::new();
        encode_dc(&mut w, table, diff).expect("encode dc");
        // Pad so the reader has a full final byte to work with.
        w.write_bits(0, 16);
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let mut preds = DcPredictors { y: 0, cb: 0, cr: 0 };
        let decoded = decode_dc(&mut r, &mut preds, component).expect("decode dc");
        assert_eq!(decoded, diff, "DC diff {diff} (luma={table_is_luma})");
    }

    #[test]
    fn dc_differentials_round_trip_luma() {
        for d in [-255, -16, -3, -1, 0, 1, 2, 3, 100, 255, 1024, -1024] {
            dc_round_trip(true, d);
        }
    }

    #[test]
    fn dc_differentials_round_trip_chroma() {
        for d in [-255, -3, -1, 0, 1, 3, 100, 255, 2047, -2047] {
            dc_round_trip(false, d);
        }
    }

    /// Round-trip one AC run/level (then EOB) through the AC encoder/decoder.
    fn ac_round_trip(table: AcTablePtr, intra_vlc_format: bool, run: u8, level: i32) {
        let mut w = BitWriter::new();
        encode_ac_run_level(&mut w, table, run, level).expect("encode ac");
        encode_eob(&mut w, table).expect("encode eob");
        w.write_bits(0, 24);
        let bytes = w.into_bytes();

        let mut r = BitReader::new(&bytes);
        let mut block = [0i32; 64];
        decode_ac(&mut r, &mut block, intra_vlc_format, false).expect("decode ac");

        // Find the single placed coefficient: progressive scan position for
        // scan index (run + 1).
        let scan_index = run as usize + 1;
        let raster = crate::mpeg2::zigzag::SCAN_PROGRESSIVE[scan_index];
        assert_eq!(
            block[raster], level,
            "AC (run={run}, level={level}) at raster {raster}"
        );
        // Everything else zero.
        for (i, &v) in block.iter().enumerate() {
            if i != raster {
                assert_eq!(v, 0, "stray coeff at {i}");
            }
        }
    }

    #[test]
    fn ac_b14_table_entries_round_trip() {
        // Every (run, level) in B-14 (both signs) must round-trip.
        for &(_, _, run, level) in AC_TABLE_B14 {
            if run == EOB_RUN || run == ESCAPE_RUN {
                continue;
            }
            ac_round_trip(AC_TABLE_B14, false, run, i32::from(level));
            ac_round_trip(AC_TABLE_B14, false, run, -i32::from(level));
        }
    }

    #[test]
    fn ac_b15_table_entries_round_trip() {
        for &(_, _, run, level) in AC_TABLE_B15 {
            if run == EOB_RUN || run == ESCAPE_RUN {
                continue;
            }
            ac_round_trip(AC_TABLE_B15, true, run, i32::from(level));
            ac_round_trip(AC_TABLE_B15, true, run, -i32::from(level));
        }
    }

    #[test]
    fn ac_escape_used_for_large_level() {
        // A level well beyond any table entry must escape and round-trip.
        ac_round_trip(AC_TABLE_B14, false, 0, 300);
        ac_round_trip(AC_TABLE_B14, false, 5, -777);
        ac_round_trip(AC_TABLE_B15, true, 2, 1000);
    }

    #[test]
    fn ac_escape_used_for_large_run() {
        // run 40 with level 1 is not in the table → escape.
        ac_round_trip(AC_TABLE_B14, false, 40, 1);
        ac_round_trip(AC_TABLE_B14, false, 50, -2);
    }

    #[test]
    fn ac_rejects_zero_and_forbidden_level() {
        let mut w = BitWriter::new();
        assert!(encode_ac_run_level(&mut w, AC_TABLE_B14, 0, 0).is_err());
        assert!(encode_ac_run_level(&mut w, AC_TABLE_B14, 0, -2048).is_err());
        assert!(encode_ac_run_level(&mut w, AC_TABLE_B14, 0, 2048).is_err());
    }

    #[test]
    fn colliding_b14_run16_level1_round_trips_via_escape() {
        // (16, 1) in B-14 only has a colliding 13-bit code; the encoder must
        // escape it. Verify it still decodes correctly.
        ac_round_trip(AC_TABLE_B14, false, 16, 1);
        ac_round_trip(AC_TABLE_B14, false, 16, -1);
    }
}
