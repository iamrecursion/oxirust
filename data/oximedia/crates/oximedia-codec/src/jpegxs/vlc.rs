// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! JPEG XS VLC (Variable-Length Code) tables and fast lookup machinery.
//!
//! JPEG XS entropy coding uses three VLC tables applied to transform
//! coefficient encoding:
//!
//! 1. **Significance VLC** — codes whether a coefficient band position is
//!    significant (non-zero) or a run of zeros.
//! 2. **Run VLC** — Golomb-Rice-style unary coding for zero-run lengths.
//! 3. **Magnitude VLC** — sign+magnitude for non-zero coefficient values.
//!
//! The actual tables in a JPEG XS bitstream are specified by the CWD marker
//! (Codeword Descriptor). Absent CWD, this module provides default tables
//! derived from the ISO 21122-1:2019 Annex A normative VLC definitions.
//!
//! # Fast lookup
//!
//! The `VlcTable` struct implements a direct-mapped lookup using the top
//! `index_bits` bits of the input bit-pattern. This gives O(1) decoding
//! for all codes up to `index_bits` bits long.

use super::JxsError;

/// Decoded result from a single VLC lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlcResult {
    /// The decoded symbol value (run, magnitude level, or significance flag).
    pub value: i16,
    /// Number of bits consumed from the bitstream for this symbol.
    pub bits_consumed: u8,
}

/// Direct-mapped VLC lookup table.
///
/// Maps a top-`index_bits` bit prefix to `(symbol, code_len)`.  A `code_len`
/// of zero indicates an invalid / unassigned prefix.
pub struct VlcTable {
    /// Direct-mapped entries: index = top `index_bits` bits.
    table: Vec<(i16, u8)>,
    /// Number of bits used as the lookup index.
    index_bits: u8,
}

impl VlcTable {
    /// Build a `VlcTable` from `(code, code_len, decoded_value)` triples.
    ///
    /// `code` is the VLC codeword right-justified (LSB = last bit of code).
    /// `code_len` is the number of significant bits in `code`.
    /// `decoded_value` is the symbol value to return on a match.
    ///
    /// All codes shorter than the maximum code length are expanded to fill
    /// every table entry whose top `code_len` bits match.
    pub fn build(entries: &[(u32, u8, i16)]) -> Self {
        let max_len = entries.iter().map(|&(_, l, _)| l).max().unwrap_or(1);
        let index_bits = max_len.max(1);
        let table_size = 1usize << index_bits;
        let mut table = vec![(0i16, 0u8); table_size];

        for &(code, len, value) in entries {
            if len == 0 || len > index_bits {
                continue;
            }
            let shift = index_bits - len;
            let base = (code as usize) << shift;
            let span = 1usize << shift;
            for offset in 0..span {
                let idx = base | offset;
                if idx < table_size {
                    // Only write if slot is empty (first-come-first-served for ties).
                    if table[idx].1 == 0 {
                        table[idx] = (value, len);
                    }
                }
            }
        }

        Self { table, index_bits }
    }

    /// Look up the top `index_bits` of `bits` (MSB-aligned, 32-bit).
    ///
    /// Returns `Some(VlcResult)` on success, or `None` if the prefix is not
    /// assigned in this table.
    pub fn lookup(&self, bits: u32) -> Option<VlcResult> {
        let idx = (bits >> (32u8.saturating_sub(self.index_bits)) as u32) as usize;
        if idx >= self.table.len() {
            return None;
        }
        let (value, len) = self.table[idx];
        if len == 0 {
            None
        } else {
            Some(VlcResult {
                value,
                bits_consumed: len,
            })
        }
    }

    /// Return the number of index bits used by this table.
    pub fn index_bits(&self) -> u8 {
        self.index_bits
    }
}

// ── Default VLC tables ────────────────────────────────────────────────────────
//
// JPEG XS default entropy code tables (ISO 21122-1:2019 Annex A).
//
// For JPEG XS profiles that lack a CWD marker, the standard specifies that
// coefficients are coded using a Golomb-Rice-style run/magnitude scheme.
// The tables below implement a subset of the normative default coding for
// the "Main" profile (NLC=3, QL=1).

/// Build the default **run VLC** table.
///
/// Zero runs are coded as unary: `0` → run=0, `10` → run=1, `110` → run=2,
/// `1110` → run=3, …, up to `1111_1111` (8 ones) → run=7, then a fixed-length
/// extension (not represented here — runs ≥ 8 return an escape sentinel -1).
pub fn default_run_table() -> VlcTable {
    // (code, len, value)
    // Unary-coded runs: code for run=k is (2^k - 1) << 1 padded, i.e. k ones then 0.
    // But simpler: use the direct encoding from the bit pattern.
    //   run=0 → bit `0`            (1 bit,  code=0b0)
    //   run=1 → bits `1 0`         (2 bits, code=0b10)
    //   run=2 → bits `1 1 0`       (3 bits, code=0b110)
    //   run=3 → bits `1 1 1 0`     (4 bits, code=0b1110)
    //   run=4 → bits `1 1 1 1 0`   (5 bits, code=0b11110)
    //   run=5 → `1 1 1 1 1 0`      (6 bits, code=0b111110)
    //   run=6 → `1 1 1 1 1 1 0`    (7 bits, code=0b1111110)
    //   run=7 → `1 1 1 1 1 1 1 0`  (8 bits, code=0b11111110)
    //   run≥8 → escape: `1 1 1 1 1 1 1 1` (8 ones) = sentinel value -1
    let entries: &[(u32, u8, i16)] = &[
        (0b0, 1, 0),         // run = 0
        (0b10, 2, 1),        // run = 1
        (0b110, 3, 2),       // run = 2
        (0b1110, 4, 3),      // run = 3
        (0b11110, 5, 4),     // run = 4
        (0b111110, 6, 5),    // run = 5
        (0b1111110, 7, 6),   // run = 6
        (0b11111110, 8, 7),  // run = 7
        (0b11111111, 8, -1), // escape: run ≥ 8
    ];
    VlcTable::build(entries)
}

/// Build the default **magnitude VLC** table.
///
/// Non-zero coefficient magnitudes in JPEG XS use an Exp-Golomb-like encoding:
/// the magnitude level `m` (1-indexed) is represented as:
///   - level 1:  `0`         (1 bit)
///   - level 2:  `10`        (2 bits)
///   - level 3:  `110`       (3 bits)
///   - level 4:  `1110`      (4 bits)
///   - level 5:  `11110`     (5 bits)
///   - level 6:  `111110`    (6 bits)
///   - level 7:  `1111110`   (7 bits)
///   - level 8:  `11111110`  (8 bits)
///   - level≥9:  `11111111`  (8 bits, escape → caller reads remaining bits)
///
/// After the magnitude code, one sign bit follows (0 = positive, 1 = negative).
/// The `value` stored is the magnitude level (1-indexed), not the final sample.
pub fn default_magnitude_table() -> VlcTable {
    let entries: &[(u32, u8, i16)] = &[
        (0b0, 1, 1),         // mag level 1
        (0b10, 2, 2),        // mag level 2
        (0b110, 3, 3),       // mag level 3
        (0b1110, 4, 4),      // mag level 4
        (0b11110, 5, 5),     // mag level 5
        (0b111110, 6, 6),    // mag level 6
        (0b1111110, 7, 7),   // mag level 7
        (0b11111110, 8, 8),  // mag level 8
        (0b11111111, 8, -1), // escape
    ];
    VlcTable::build(entries)
}

/// Build the default **significance VLC** table.
///
/// In the simplified JPEG XS coding, coefficient bands are grouped into
/// "significance maps" where a 1-bit flag indicates whether the coefficient
/// is non-zero.  In the absence of a proper significance table, we use a
/// single-bit direct coding:
///   - `0` → coefficient is zero (significance = 0)
///   - `1` → coefficient is non-zero (significance = 1)
pub fn default_significance_table() -> VlcTable {
    let entries: &[(u32, u8, i16)] = &[
        (0b0, 1, 0), // zero
        (0b1, 1, 1), // non-zero
    ];
    VlcTable::build(entries)
}

/// Build a VLC table from a CWD-marker-specified code assignment.
///
/// `codes` is a slice of `(code_bits, code_len, symbol_value)` triples
/// as produced by parsing the CWD marker payload.
pub fn build_from_cwd(codes: &[(u32, u8, i16)]) -> Result<VlcTable, JxsError> {
    if codes.is_empty() {
        return Err(JxsError::VlcError("CWD code table is empty".to_string()));
    }
    Ok(VlcTable::build(codes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_table_run0_is_single_zero_bit() {
        let table = default_run_table();
        // Top bit = 0 → run = 0, consumed = 1 bit
        let result = table.lookup(0x0000_0000u32); // top bit = 0
        assert!(result.is_some(), "run=0 lookup failed");
        let r = result.unwrap();
        assert_eq!(r.value, 0);
        assert_eq!(r.bits_consumed, 1);
    }

    #[test]
    fn run_table_run1_starts_with_10() {
        let table = default_run_table();
        // 0b10 followed by zeros → shift left: 0b1000_0000 << 24 = 0x80000000
        let result = table.lookup(0b1000_0000_0000_0000_0000_0000_0000_0000u32);
        assert!(result.is_some(), "run=1 lookup failed");
        let r = result.unwrap();
        assert_eq!(r.value, 1);
        assert_eq!(r.bits_consumed, 2);
    }

    #[test]
    fn magnitude_table_level1_is_single_zero_bit() {
        let table = default_magnitude_table();
        let result = table.lookup(0x0000_0000u32);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.value, 1); // level 1
        assert_eq!(r.bits_consumed, 1);
    }

    #[test]
    fn significance_table_zero_bit() {
        let table = default_significance_table();
        // 0 bit → significance = 0
        let r = table.lookup(0x0000_0000).unwrap();
        assert_eq!(r.value, 0);
        assert_eq!(r.bits_consumed, 1);
        // 1 bit → significance = 1
        let r = table.lookup(0x8000_0000).unwrap();
        assert_eq!(r.value, 1);
        assert_eq!(r.bits_consumed, 1);
    }

    #[test]
    fn vlc_table_empty_entries_returns_none() {
        let table = VlcTable::build(&[]);
        assert!(table.lookup(0).is_none());
    }

    #[test]
    fn vlc_table_single_entry_roundtrip() {
        // Code 0b10 (len=2, MSB-aligned → top 2 bits = 0b10), value = 99
        let entries = [(0b10u32, 2u8, 99i16)];
        let table = VlcTable::build(&entries);
        // MSB-aligned: 0b10 << 30 = 0x80000000
        let result = table.lookup(0b10_00_0000_0000_0000_0000_0000_0000_0000u32);
        assert_eq!(
            result,
            Some(VlcResult {
                value: 99,
                bits_consumed: 2
            })
        );
    }

    #[test]
    fn build_from_cwd_rejects_empty() {
        let result = build_from_cwd(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn build_from_cwd_accepts_valid_codes() {
        let codes: Vec<(u32, u8, i16)> = vec![(0b0, 1, 0), (0b1, 1, 1)];
        let table = build_from_cwd(&codes).unwrap();
        assert_eq!(table.index_bits(), 1);
    }
}
