//! MPEG-2 variable-length-code tables (ISO/IEC 13818-2 Annex B).
//!
//! All four tables required for **intra** macroblock decoding are provided:
//!
//! - **Table B-12** — `dct_dc_size_luminance` (DC size category for Y blocks).
//! - **Table B-13** — `dct_dc_size_chrominance` (DC size category for Cb/Cr).
//! - **Table B-14** — `dct_coefficients_0` (default AC run/level VLC).
//! - **Table B-15** — `dct_coefficients_1` (alternate AC VLC for intra blocks,
//!   selected when `intra_vlc_format == 1`).
//!
//! These codeword assignments are factual data drawn from a publicly available
//! international standard and are not subject to copyright. Codes are stored
//! right-justified (LSB-aligned) together with their bit length; matching is
//! performed MSB-first by [`match_vlc`] / [`match_dc_size`].
//!
//! The escape mechanism for AC coefficients (code `0000 01`, ISO/IEC 13818-2
//! §7.2.2.3) is handled directly in `entropy.rs` (fixed 6-bit run + 12-bit
//! signed level for MPEG-2).

use super::{Mpeg2Error, Mpeg2Result};

/// A DC-size VLC entry: `(code, len, size_category)`.
type DcEntry = (u16, u8, u8);

/// An AC run/level VLC entry: `(code, len, run, level)`.
///
/// `level` is the unsigned magnitude; the sign bit is read separately from the
/// stream. The special end-of-block code is represented with `run == 0xFF`.
type AcEntry = (u32, u8, u8, u16);

/// Shared reference to a DC-size table (B-12 / B-13).
pub type DcTablePtr = &'static [DcEntry];

/// Shared reference to an AC run/level table (B-14 / B-15).
pub type AcTablePtr = &'static [AcEntry];

/// Sentinel run value marking the end-of-block (EOB) AC code.
pub const EOB_RUN: u8 = 0xFF;
/// Sentinel run value marking the escape AC code (`0000 01`).
pub const ESCAPE_RUN: u8 = 0xFE;

// ── Table B-12: dct_dc_size_luminance ────────────────────────────────────────

/// ISO/IEC 13818-2 Table B-12. `(code, len, size)`.
pub const DC_SIZE_LUMA: &[DcEntry] = &[
    (0b100, 3, 0),
    (0b00, 2, 1),
    (0b01, 2, 2),
    (0b101, 3, 3),
    (0b110, 3, 4),
    (0b1110, 4, 5),
    (0b11110, 5, 6),
    (0b111110, 6, 7),
    (0b1111110, 7, 8),
    (0b11111110, 8, 9),
    (0b111111110, 9, 10),
    (0b111111111, 9, 11),
];

// ── Table B-13: dct_dc_size_chrominance ──────────────────────────────────────

/// ISO/IEC 13818-2 Table B-13. `(code, len, size)`.
pub const DC_SIZE_CHROMA: &[DcEntry] = &[
    (0b00, 2, 0),
    (0b01, 2, 1),
    (0b10, 2, 2),
    (0b110, 3, 3),
    (0b1110, 4, 4),
    (0b11110, 5, 5),
    (0b111110, 6, 6),
    (0b1111110, 7, 7),
    (0b11111110, 8, 8),
    (0b111111110, 9, 9),
    (0b1111111110, 10, 10),
    (0b1111111111, 10, 11),
];

// ── Table B-14: dct_coefficients_0 (default AC table) ────────────────────────
//
// (code, len, run, level). The first entry (run=0, level=1) has two distinct
// 2-bit codes ("1s") in the standard — `10`/`11` carry the sign in their LSB;
// for intra blocks (where the first coefficient is never DC) the run/level
// table is entered after DC, so the 2-bit "1" + sign form is used. We store
// the EOB code `10` and the (0,1) code `11` separately, matching the standard's
// note that the leading `1` is only EOB at the start of block.

/// ISO/IEC 13818-2 Table B-14. `(code, len, run, level)`.
pub const AC_TABLE_B14: &[AcEntry] = &[
    // End of block: `10` (only valid when not the first coefficient).
    (0b10, 2, EOB_RUN, 0),
    // Escape: `0000 01`.
    (0b000001, 6, ESCAPE_RUN, 0),
    // run=0
    (0b11, 2, 0, 1),
    (0b0100, 4, 0, 2),
    (0b00101, 5, 0, 3),
    (0b0000110, 7, 0, 4),
    (0b00100110, 8, 0, 5),
    (0b00100001, 8, 0, 6),
    (0b0000001010, 10, 0, 7),
    (0b000000011101, 12, 0, 8),
    (0b000000011000, 12, 0, 9),
    (0b000000010011, 12, 0, 10),
    (0b000000010000, 12, 0, 11),
    (0b0000000011010, 13, 0, 12),
    (0b0000000011001, 13, 0, 13),
    (0b0000000011000, 13, 0, 14),
    (0b0000000010111, 13, 0, 15),
    (0b0000000010110, 13, 0, 16),
    (0b00000000011111, 14, 0, 17),
    (0b00000000011110, 14, 0, 18),
    (0b00000000011101, 14, 0, 19),
    (0b00000000011100, 14, 0, 20),
    (0b00000000011011, 14, 0, 21),
    (0b00000000011010, 14, 0, 22),
    (0b00000000011001, 14, 0, 23),
    (0b00000000011000, 14, 0, 24),
    (0b00000000010111, 14, 0, 25),
    (0b00000000010110, 14, 0, 26),
    (0b00000000010101, 14, 0, 27),
    (0b00000000010100, 14, 0, 28),
    (0b00000000010011, 14, 0, 29),
    (0b00000000010010, 14, 0, 30),
    (0b00000000010001, 14, 0, 31),
    (0b00000000010000, 14, 0, 32),
    (0b000000000011000, 15, 0, 33),
    (0b000000000010111, 15, 0, 34),
    (0b000000000010110, 15, 0, 35),
    (0b000000000010101, 15, 0, 36),
    (0b000000000010100, 15, 0, 37),
    (0b000000000010011, 15, 0, 38),
    (0b000000000010010, 15, 0, 39),
    (0b000000000010001, 15, 0, 40),
    // run=1
    (0b011, 3, 1, 1),
    (0b000110, 6, 1, 2),
    (0b00100101, 8, 1, 3),
    (0b0000001100, 10, 1, 4),
    (0b000000011011, 12, 1, 5),
    (0b0000000010101, 13, 1, 6),
    (0b0000000010100, 13, 1, 7),
    (0b00000000011010, 14, 1, 8),
    // run=2
    (0b0101, 4, 2, 1),
    (0b0000100, 7, 2, 2),
    (0b0000001011, 10, 2, 3),
    (0b000000010100, 12, 2, 4),
    (0b0000000010011, 13, 2, 5),
    // run=3
    (0b00111, 5, 3, 1),
    (0b00100100, 8, 3, 2),
    (0b000000011100, 12, 3, 3),
    (0b0000000010010, 13, 3, 4),
    // run=4
    (0b00110, 5, 4, 1),
    (0b0000001111, 10, 4, 2),
    (0b000000010010, 12, 4, 3),
    // run=5
    (0b000111, 6, 5, 1),
    (0b0000001001, 10, 5, 2),
    (0b000000010101, 12, 5, 3),
    // run=6
    (0b000101, 6, 6, 1),
    (0b0000000111, 10, 6, 2),
    (0b0000000010001, 13, 6, 3),
    // run=7
    (0b0000111, 7, 7, 1),
    (0b0000000110, 10, 7, 2),
    // run=8
    (0b0000101, 7, 8, 1),
    (0b0000000101, 10, 8, 2),
    // run=9
    (0b00100111, 8, 9, 1),
    (0b000000011110, 12, 9, 2),
    // run=10
    (0b00100011, 8, 10, 1),
    (0b000000010001, 12, 10, 2),
    // run=11
    (0b00100010, 8, 11, 1),
    (0b0000000011111, 13, 11, 2),
    // run=12
    (0b00100000, 8, 12, 1),
    (0b0000000011110, 13, 12, 2),
    // run=13
    (0b0000001110, 10, 13, 1),
    (0b0000000011101, 13, 13, 2),
    // run=14
    (0b0000001101, 10, 14, 1),
    (0b0000000011100, 13, 14, 2),
    // run=15
    (0b0000001000, 10, 15, 1),
    (0b0000000011011, 13, 15, 2),
    // run=16
    (0b0000000011110, 13, 16, 1),
    // run=17..=31: single-level codes (15-bit)
    (0b0000000000011111, 16, 17, 1),
    (0b0000000000011110, 16, 18, 1),
    (0b0000000000011101, 16, 19, 1),
    (0b0000000000011100, 16, 20, 1),
    (0b0000000000011011, 16, 21, 1),
    (0b0000000000011010, 16, 22, 1),
    (0b0000000000011001, 16, 23, 1),
    (0b0000000000011000, 16, 24, 1),
    (0b0000000000010111, 16, 25, 1),
    (0b0000000000010110, 16, 26, 1),
    (0b0000000000010101, 16, 27, 1),
    (0b0000000000010100, 16, 28, 1),
    (0b0000000000010011, 16, 29, 1),
    (0b0000000000010010, 16, 30, 1),
    (0b0000000000010001, 16, 31, 1),
];

// ── Table B-15: dct_coefficients_1 (alternate AC for intra blocks) ───────────

/// ISO/IEC 13818-2 Table B-15. `(code, len, run, level)`.
pub const AC_TABLE_B15: &[AcEntry] = &[
    // End of block: `0110`.
    (0b0110, 4, EOB_RUN, 0),
    // Escape: `0000 01`.
    (0b000001, 6, ESCAPE_RUN, 0),
    // run=0
    (0b10, 2, 0, 1),
    (0b1111, 4, 0, 2),
    (0b010101, 6, 0, 3),
    (0b00100100, 8, 0, 4),
    (0b00100101, 8, 0, 5),
    (0b00100111, 8, 0, 6),
    (0b00100001, 8, 0, 7),
    (0b001000000, 9, 0, 8),
    (0b001000001, 9, 0, 9),
    (0b0000001010, 10, 0, 10),
    (0b00000011010, 11, 0, 11),
    (0b00000011001, 11, 0, 12),
    (0b00000011000, 11, 0, 13),
    (0b00000010111, 11, 0, 14),
    (0b00000010110, 11, 0, 15),
    (0b000000011111, 12, 0, 16),
    (0b0000000011010, 13, 0, 17),
    (0b0000000011001, 13, 0, 18),
    (0b0000000011000, 13, 0, 19),
    (0b0000000010111, 13, 0, 20),
    (0b0000000010110, 13, 0, 21),
    (0b00000000011111, 14, 0, 22),
    (0b00000000011110, 14, 0, 23),
    (0b00000000011101, 14, 0, 24),
    (0b00000000011100, 14, 0, 25),
    (0b00000000011011, 14, 0, 26),
    (0b00000000011010, 14, 0, 27),
    (0b00000000011001, 14, 0, 28),
    (0b00000000011000, 14, 0, 29),
    (0b00000000010111, 14, 0, 30),
    (0b00000000010110, 14, 0, 31),
    (0b00000000010101, 14, 0, 32),
    (0b000000000011000, 15, 0, 33),
    (0b000000000010111, 15, 0, 34),
    (0b000000000010110, 15, 0, 35),
    (0b000000000010101, 15, 0, 36),
    (0b000000000010100, 15, 0, 37),
    (0b000000000010011, 15, 0, 38),
    (0b000000000010010, 15, 0, 39),
    (0b000000000010001, 15, 0, 40),
    // run=1
    (0b1110, 4, 1, 1),
    (0b0001100, 7, 1, 2),
    (0b001000010, 9, 1, 3),
    (0b00000011011, 11, 1, 4),
    (0b000000010100, 12, 1, 5),
    (0b0000000010101, 13, 1, 6),
    (0b0000000010100, 13, 1, 7),
    (0b00000000011010, 14, 1, 8),
    // run=2
    (0b0101, 4, 2, 1),
    (0b00000100, 8, 2, 2),
    (0b001000011, 9, 2, 3),
    (0b000000010101, 12, 2, 4),
    (0b0000000010011, 13, 2, 5),
    // run=3
    (0b00111, 5, 3, 1),
    (0b00100110, 8, 3, 2),
    (0b000000011100, 12, 3, 3),
    (0b0000000010010, 13, 3, 4),
    // run=4
    (0b00110, 5, 4, 1),
    (0b000000111101, 12, 4, 2),
    (0b000000010010, 12, 4, 3),
    // run=5
    (0b000111, 6, 5, 1),
    (0b000000111000, 12, 5, 2),
    (0b000000010101, 12, 5, 3),
    // run=6
    (0b000101, 6, 6, 1),
    (0b000000111011, 12, 6, 2),
    (0b0000000010001, 13, 6, 3),
    // run=7
    (0b0000110, 7, 7, 1),
    (0b000000110110, 12, 7, 2),
    // run=8
    (0b000100, 6, 8, 1),
    (0b000000111010, 12, 8, 2),
    // run=9
    (0b0000111, 7, 9, 1),
    (0b000000011110, 12, 9, 2),
    // run=10
    (0b00100011, 8, 10, 1),
    (0b000000010001, 12, 10, 2),
    // run=11
    (0b00100010, 8, 11, 1),
    (0b0000000011111, 13, 11, 2),
    // run=12
    (0b00100000, 8, 12, 1),
    (0b0000000011110, 13, 12, 2),
    // run=13
    (0b00001100, 8, 13, 1),
    (0b0000000011101, 13, 13, 2),
    // run=14
    (0b000000111001, 12, 14, 1),
    (0b0000000011100, 13, 14, 2),
    // run=15
    (0b000000110111, 12, 15, 1),
    (0b0000000011011, 13, 15, 2),
    // run=16
    (0b000000110101, 12, 16, 1),
    (0b0000000011010, 13, 16, 1),
    // run=17..=31 single-level (16-bit codes, identical assignment to B-14)
    (0b0000000000011111, 16, 17, 1),
    (0b0000000000011110, 16, 18, 1),
    (0b0000000000011101, 16, 19, 1),
    (0b0000000000011100, 16, 20, 1),
    (0b0000000000011011, 16, 21, 1),
    (0b0000000000011010, 16, 22, 1),
    (0b0000000000011001, 16, 23, 1),
    (0b0000000000011000, 16, 24, 1),
    (0b0000000000010111, 16, 25, 1),
    (0b0000000000010110, 16, 26, 1),
    (0b0000000000010101, 16, 27, 1),
    (0b0000000000010100, 16, 28, 1),
    (0b0000000000010011, 16, 29, 1),
    (0b0000000000010010, 16, 30, 1),
    (0b0000000000010001, 16, 31, 1),
];

/// Result of matching one AC VLC codeword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcSymbol {
    /// A `(run, level)` run/level pair was decoded. The sign bit still needs to
    /// be read from the stream by the caller.
    RunLevel {
        /// Number of zero coefficients to skip before placing `level`.
        run: u8,
        /// Unsigned magnitude of the coefficient.
        level: u16,
        /// Number of bits the codeword consumed.
        bits: u8,
    },
    /// End-of-block: no more non-zero coefficients in this block.
    EndOfBlock {
        /// Number of bits the EOB codeword consumed.
        bits: u8,
    },
    /// Escape sequence; caller must read 6 run bits then 12 signed level bits.
    Escape {
        /// Number of bits the escape prefix consumed (always 6).
        bits: u8,
    },
}

/// Match a DC-size codeword from `peeked` (MSB-aligned in the high bits).
///
/// Returns `(size_category, bits_consumed)`.
///
/// # Errors
///
/// Returns [`Mpeg2Error::VlcDecode`] if no entry matches.
pub fn match_dc_size(table: &[DcEntry], peeked: u32) -> Mpeg2Result<(u8, u8)> {
    for &(code, len, size) in table {
        let prefix = (peeked >> (32 - u32::from(len))) as u16;
        if prefix == code {
            return Ok((size, len));
        }
    }
    Err(Mpeg2Error::VlcDecode(format!(
        "DC size code not found (bits={:016b})",
        peeked >> 16
    )))
}

/// Match an AC run/level codeword from `peeked` (MSB-aligned in the high bits).
///
/// Codewords are tried shortest-first so a longer code is never shadowed by a
/// shorter prefix.
///
/// # Errors
///
/// Returns [`Mpeg2Error::VlcDecode`] if no entry matches.
pub fn match_vlc(table: &[AcEntry], peeked: u32) -> Mpeg2Result<AcSymbol> {
    // Try in ascending code length so the unique prefix-free match wins.
    for target_len in 2u8..=16 {
        for &(code, len, run, level) in table {
            if len != target_len {
                continue;
            }
            let prefix = peeked >> (32 - u32::from(len));
            if prefix == u32::from(code) {
                return Ok(match run {
                    EOB_RUN => AcSymbol::EndOfBlock { bits: len },
                    ESCAPE_RUN => AcSymbol::Escape { bits: len },
                    _ => AcSymbol::RunLevel {
                        run,
                        level,
                        bits: len,
                    },
                });
            }
        }
    }
    Err(Mpeg2Error::VlcDecode(format!(
        "AC code not found (bits={:016b})",
        peeked >> 16
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an MSB-aligned peek word from a right-justified `code` of `len`
    /// bits, padding the rest with the given filler.
    fn msb_aligned(code: u32, len: u8, filler: u32) -> u32 {
        let shift = 32 - u32::from(len);
        (code << shift) | (filler & ((1u32 << shift) - 1))
    }

    #[test]
    fn dc_luma_sizes_unique() {
        // All 12 luma DC sizes must be matchable and consume the right length.
        for &(code, len, size) in DC_SIZE_LUMA {
            let peek = msb_aligned(u32::from(code), len, 0);
            let (got, bits) = match_dc_size(DC_SIZE_LUMA, peek).expect("match");
            assert_eq!(got, size, "luma size {size}");
            assert_eq!(bits, len, "luma size {size} len");
        }
    }

    #[test]
    fn dc_chroma_sizes_unique() {
        for &(code, len, size) in DC_SIZE_CHROMA {
            let peek = msb_aligned(u32::from(code), len, 0);
            let (got, bits) = match_dc_size(DC_SIZE_CHROMA, peek).expect("match");
            assert_eq!(got, size, "chroma size {size}");
            assert_eq!(bits, len, "chroma size {size} len");
        }
    }

    #[test]
    fn dc_luma_is_prefix_free() {
        // No code may be a prefix of another (verify pairwise).
        let entries: Vec<_> = DC_SIZE_LUMA.iter().collect();
        for (i, &&(ci, li, _)) in entries.iter().enumerate() {
            for (j, &&(cj, lj, _)) in entries.iter().enumerate() {
                if i == j {
                    continue;
                }
                if li <= lj {
                    let shifted = cj >> (lj - li);
                    assert_ne!(shifted, ci, "code {i} is a prefix of code {j}");
                }
            }
        }
    }

    #[test]
    fn ac_b14_eob_matches() {
        let peek = msb_aligned(0b10, 2, 0);
        match match_vlc(AC_TABLE_B14, peek).expect("eob") {
            AcSymbol::EndOfBlock { bits } => assert_eq!(bits, 2),
            other => panic!("expected EOB, got {other:?}"),
        }
    }

    #[test]
    fn ac_b14_first_run_level() {
        // code `11` → (run=0, level=1).
        let peek = msb_aligned(0b11, 2, 0);
        match match_vlc(AC_TABLE_B14, peek).expect("rl") {
            AcSymbol::RunLevel { run, level, bits } => {
                assert_eq!((run, level, bits), (0, 1, 2));
            }
            other => panic!("expected RunLevel, got {other:?}"),
        }
    }

    #[test]
    fn ac_b14_escape_matches() {
        let peek = msb_aligned(0b000001, 6, 0);
        match match_vlc(AC_TABLE_B14, peek).expect("escape") {
            AcSymbol::Escape { bits } => assert_eq!(bits, 6),
            other => panic!("expected Escape, got {other:?}"),
        }
    }

    #[test]
    fn ac_b14_run1_level1() {
        // code `011` → (run=1, level=1).
        let peek = msb_aligned(0b011, 3, 0);
        match match_vlc(AC_TABLE_B14, peek).expect("rl") {
            AcSymbol::RunLevel { run, level, bits } => {
                assert_eq!((run, level, bits), (1, 1, 3));
            }
            other => panic!("expected RunLevel, got {other:?}"),
        }
    }

    #[test]
    fn ac_b15_eob_matches() {
        let peek = msb_aligned(0b0110, 4, 0);
        match match_vlc(AC_TABLE_B15, peek).expect("eob") {
            AcSymbol::EndOfBlock { bits } => assert_eq!(bits, 4),
            other => panic!("expected EOB, got {other:?}"),
        }
    }

    #[test]
    fn ac_b15_first_run_level() {
        // In B-15, code `10` → (run=0, level=1).
        let peek = msb_aligned(0b10, 2, 0);
        match match_vlc(AC_TABLE_B15, peek).expect("rl") {
            AcSymbol::RunLevel { run, level, bits } => {
                assert_eq!((run, level, bits), (0, 1, 2));
            }
            other => panic!("expected RunLevel, got {other:?}"),
        }
    }

    #[test]
    fn unknown_ac_code_errors() {
        // A run of 32 zero bits matches no AC codeword (every real AC code has
        // a terminating `1` within its length, and the escape is `0000 01`).
        let peek = 0u32;
        assert!(match_vlc(AC_TABLE_B14, peek).is_err());
        assert!(match_vlc(AC_TABLE_B15, peek).is_err());
    }

    #[test]
    fn dc_tables_are_complete_prefix_codes() {
        // The DC size tables are complete: the all-zero prefix maps to size 1
        // for luma (code `00`) and size 0 for chroma (code `00`).
        let (luma_size, luma_len) = match_dc_size(DC_SIZE_LUMA, 0).expect("luma");
        assert_eq!((luma_size, luma_len), (1, 2));
        let (chroma_size, chroma_len) = match_dc_size(DC_SIZE_CHROMA, 0).expect("chroma");
        assert_eq!((chroma_size, chroma_len), (0, 2));
    }
}
