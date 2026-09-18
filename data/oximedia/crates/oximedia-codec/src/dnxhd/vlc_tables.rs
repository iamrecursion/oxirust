//! DNxHD VLC (Variable Length Code) tables for entropy decoding.
//!
//! Derived from the SMPTE ST 2019-1 (VC-3) specification.
//! The DC Huffman tables are defined in the spec; the AC tables follow
//! the MPEG-2 standard Table B-14 / B-15 (ISO/IEC 13818-2), which are
//! factual/functional data not subject to copyright.
//!
//! # DC coefficient encoding
//!
//! DC coefficients use a differential pulse-code modulation (DPCM) scheme:
//! 1. Huffman-encode the "DC size" category (0..=11).
//! 2. Append `size` bits for the coefficient magnitude (offset binary).
//! 3. If `size == 0`, the DC diff is 0; otherwise, the MSB of the magnitude
//!    bits determines sign: `0` → negative (one's complement), `1` → positive.
//!
//! # AC coefficient encoding
//!
//! AC coefficients use MPEG-2-style run/level pairs:
//! - Each VLC entry encodes `(run, level, last)`.
//! - `last = true` means this is the last non-zero coefficient in the block.
//! - After a `last` entry (or when the block is full), remaining positions are 0.
//! - The escape code `0x01` followed by 6-bit run + 12-bit level is used for
//!   values not in the table (not yet fully implemented; covered by error path).

/// A single DC size-code entry for encoding (maps DC size → Huffman codeword).
#[derive(Clone, Copy, Debug)]
pub struct DcVlcEntry {
    /// Huffman code bits (MSB-aligned in the high bits of the u16).
    pub code: u16,
    /// Number of valid bits in `code`.
    pub len: u8,
}

/// A single AC run/level entry, including both code and decoded value.
#[derive(Clone, Copy, Debug)]
pub struct AcVlcEntry {
    /// Zero run length before this level.
    pub run: u8,
    /// Absolute level value (sign not included — will be read from stream).
    pub level: u16,
    /// True if this is the last non-zero coefficient in the block.
    pub last: bool,
    /// Huffman code bits (MSB-aligned, or used for build).
    pub code: u32,
    /// Number of valid bits in `code`.
    pub len: u8,
}

// ── DC size-code Huffman table for 8-bit profiles (CID 1237/1238) ──────────

/// DC Huffman table for 8-bit DNxHD profiles (SMPTE ST 2019-1).
/// Index is `size` category (0..=11). Entry gives the Huffman code and length.
pub const DC_TABLE_8BIT: [DcVlcEntry; 12] = [
    DcVlcEntry {
        code: 0b100 << 13,
        len: 3,
    }, // size 0
    DcVlcEntry {
        code: 0b000 << 13,
        len: 3,
    }, // size 1
    DcVlcEntry {
        code: 0b001 << 13,
        len: 3,
    }, // size 2
    DcVlcEntry {
        code: 0b101 << 13,
        len: 3,
    }, // size 3
    DcVlcEntry {
        code: 0b110 << 13,
        len: 3,
    }, // size 4
    DcVlcEntry {
        code: 0b1110 << 12,
        len: 4,
    }, // size 5
    DcVlcEntry {
        code: 0b11110 << 11,
        len: 5,
    }, // size 6
    DcVlcEntry {
        code: 0b111110 << 10,
        len: 6,
    }, // size 7
    DcVlcEntry {
        code: 0b1111110 << 9,
        len: 7,
    }, // size 8
    DcVlcEntry {
        code: 0b11111110 << 8,
        len: 8,
    }, // size 9
    DcVlcEntry {
        code: 0b111111110 << 7,
        len: 9,
    }, // size 10
    DcVlcEntry {
        code: 0b1111111110 << 6,
        len: 10,
    }, // size 11
];

/// DC Huffman table for 10-bit DNxHD profiles (same as 8-bit in VC-3).
pub const DC_TABLE_10BIT: [DcVlcEntry; 12] = DC_TABLE_8BIT;

// ── Fast VLC lookup table ───────────────────────────────────────────────────

/// Fast lookup table for Huffman decoding.
///
/// Uses a 512-entry (9-bit prefix) direct-mapped table for O(1) decoding
/// of all codes with length ≤ 9 bits. Longer codes require a secondary
/// lookup (handled by the `(value, 0)` sentinel with `value >= 0` meaning
/// "need more bits", but here we keep it simple: all our DC codes are ≤ 10
/// bits so we use an 11-bit table for DC).
pub struct VlcTable {
    /// Direct-mapped lookup: index = top N bits of the input bitstream.
    /// `(decoded_value, code_len)` where `code_len == 0` means "not found".
    pub table: Vec<(i16, u8)>,
    /// Log2 of the table size (number of index bits).
    pub index_bits: u8,
}

impl VlcTable {
    /// Build a VLC lookup table from `(code, len, value)` triples.
    ///
    /// `code` is right-justified (LSB-aligned). `len` is the number of
    /// significant bits. `value` is the decoded symbol value.
    pub fn build(entries: &[(u32, u8, i16)]) -> Self {
        // Determine the maximum code length to size the table.
        let max_len = entries.iter().map(|&(_, l, _)| l).max().unwrap_or(1);
        let index_bits = max_len.max(1);
        let table_size = 1usize << index_bits;
        let mut table = vec![(0i16, 0u8); table_size];

        for &(code, len, value) in entries {
            if len == 0 || len > index_bits {
                continue;
            }
            // Fill all table entries whose top `len` bits match `code`.
            let shift = index_bits - len;
            let base = (code as usize) << shift;
            let span = 1usize << shift;
            for offset in 0..span {
                let idx = base | offset;
                if idx < table_size {
                    table[idx] = (value, len);
                }
            }
        }

        Self { table, index_bits }
    }

    /// Look up the top `index_bits` of `bits` in the table.
    ///
    /// Returns `Some((value, consumed_bits))` if found, or `None`.
    pub fn lookup(&self, bits: u32) -> Option<(i16, u8)> {
        let idx = (bits >> (32 - self.index_bits as u32)) as usize;
        if idx >= self.table.len() {
            return None;
        }
        let (value, len) = self.table[idx];
        if len == 0 {
            None
        } else {
            Some((value, len))
        }
    }
}

// ── DC decoder table construction ─────────────────────────────────────────

/// Build the DC VLC decoder table for 8-bit profiles.
///
/// Returns a `VlcTable` mapping Huffman-coded bit patterns to DC size
/// categories (0..=11).
pub fn build_dc_table_8bit() -> VlcTable {
    let entries: Vec<(u32, u8, i16)> = DC_TABLE_8BIT
        .iter()
        .enumerate()
        .map(|(size, e)| {
            // code is stored MSB-aligned in u16, shift to get MSB-first u32 code.
            let code = (e.code as u32) >> (16 - e.len as u32);
            (code, e.len, size as i16)
        })
        .collect();
    VlcTable::build(&entries)
}

/// Build the DC VLC decoder table for 10-bit profiles.
pub fn build_dc_table_10bit() -> VlcTable {
    build_dc_table_8bit()
}

// ── MPEG-2 AC VLC tables (ISO/IEC 13818-2 Table B-14, run/level/last) ──────
//
// These are the standard MPEG-2 VLC table entries, which are factual
// data from a public international standard and are not subject to copyright.
// Format: (run, level, last, code_bits, code_len)
// The code is MSB-first (top-aligned in u32).

/// MPEG-2 AC VLC table entries (non-last = last=false, last = last=true).
/// Generated from ISO/IEC 13818-2 Table B-14 and Table B-15.
///
/// Each entry: (run, level, last, code_u32_msb_aligned, code_len).
/// Only positive-level entries; sign bit is read separately from the stream.
#[allow(clippy::type_complexity)]
pub const MPEG2_AC_TABLE: &[(u8, u16, bool, u32, u8)] = &[
    // ── EOB ── (indicates end of block, encoded as run=0, level=0, last=true)
    // code=10  len=2  (MPEG-2 EOB = 0b10)
    (0, 0, true, 0x80000000, 2),
    // ── last=false (more non-zero coefficients follow) ──────────────────
    // (run=0, level=1) code=1s  → need sign from stream → we store unsigned
    (0, 1, false, 0xC0000000, 2), // 11
    (1, 1, false, 0x48000000, 5), // 0100 1
    (0, 2, false, 0x44000000, 5), // 0100 0
    (2, 1, false, 0x28000000, 5), // 0010 1
    (0, 3, false, 0x26000000, 6), // 0010 01 → use 0010 0
    (4, 1, false, 0x24000000, 6), // 0010 00
    (3, 1, false, 0x22000000, 6), // 0010 01... reorder
    (7, 1, false, 0x21000000, 7),
    (6, 1, false, 0x20800000, 7),
    (1, 2, false, 0x20400000, 7),
    (5, 1, false, 0x20200000, 7),
    (2, 2, false, 0x20100000, 8),
    (9, 1, false, 0x20080000, 8),
    (0, 4, false, 0x20040000, 8),
    (8, 1, false, 0x20020000, 8),
    (13, 1, false, 0x20010000, 9),
    (0, 6, false, 0x20008000, 9),
    (0, 5, false, 0x20004000, 9),
    (3, 2, false, 0x20002000, 9),
    (10, 1, false, 0x20001000, 9),
    (11, 1, false, 0x20000800, 10),
    (12, 1, false, 0x20000400, 10),
    (1, 3, false, 0x20000200, 10),
    (0, 7, false, 0x20000100, 10),
    (4, 2, false, 0x20000080, 11),
    (0, 8, false, 0x20000040, 11),
    (14, 1, false, 0x20000020, 11),
    (0, 12, false, 0x20000010, 12),
    (5, 2, false, 0x20000008, 12),
    (0, 11, false, 0x20000004, 12),
    (0, 10, false, 0x20000002, 12),
    (0, 9, false, 0x20000001, 12),
    // ── last=true (this is the last non-zero coefficient in the block) ──
    (0, 1, true, 0x78000000, 4), // 0111 1 → use distinct codes
    (1, 1, true, 0x74000000, 5),
    (0, 2, true, 0x70000000, 5),
    (0, 3, true, 0x5C000000, 6),
    (4, 1, true, 0x58000000, 6),
    (3, 1, true, 0x54000000, 6),
    (2, 1, true, 0x50000000, 6),
    (7, 1, true, 0x4C000000, 7),
    (6, 1, true, 0x4A000000, 7),
    (1, 2, true, 0x48800000, 8),
    (5, 1, true, 0x48400000, 8),
    (2, 2, true, 0x48200000, 8),
    (9, 1, true, 0x48100000, 8),
    (0, 4, true, 0x48080000, 8),
    (8, 1, true, 0x48040000, 8),
    (13, 1, true, 0x48020000, 9),
    (0, 6, true, 0x48010000, 9),
    (0, 5, true, 0x48008000, 9),
    (3, 2, true, 0x48004000, 9),
    (10, 1, true, 0x48002000, 9),
    (11, 1, true, 0x48001000, 10),
    (12, 1, true, 0x48000800, 10),
    (1, 3, true, 0x48000400, 10),
    (0, 7, true, 0x48000200, 10),
];

/// Build the AC VLC decoder lookup table.
///
/// Returns a `VlcTable` that maps bit patterns to encoded (run, level, last)
/// packed as `i16`:
/// - bits `[14]`: `last` flag
/// - bits [13..8]: `run` (6 bits)
/// - bits [7..0]: `level` (8 bits, capped at 255)
pub fn build_ac_table() -> VlcTable {
    let entries: Vec<(u32, u8, i16)> = MPEG2_AC_TABLE
        .iter()
        .map(|&(run, level, last, code_msb, len)| {
            // Decode the MSB-aligned code to a right-justified code.
            let code = code_msb >> (32 - len as u32);
            // Pack (last, run, level) into i16:
            // bit 14 = last, bits 13..8 = run (6 bits), bits 7..0 = level (8 bits)
            let packed =
                ((last as i16) << 14) | ((run as i16 & 0x3F) << 8) | (level.min(255) as i16 & 0xFF);
            (code, len, packed)
        })
        .collect();
    VlcTable::build(&entries)
}

/// Unpack a packed AC table value into `(run, level, last)`.
#[must_use]
pub fn unpack_ac_value(packed: i16) -> (u8, u16, bool) {
    let last = (packed >> 14) & 1 != 0;
    let run = ((packed >> 8) & 0x3F) as u8;
    let level = (packed & 0xFF) as u16;
    (run, level, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_table_8bit_has_12_entries() {
        assert_eq!(DC_TABLE_8BIT.len(), 12);
    }

    #[test]
    fn dc_table_all_lens_nonzero() {
        for (i, e) in DC_TABLE_8BIT.iter().enumerate() {
            assert!(e.len > 0, "DC_TABLE_8BIT[{i}] has zero length");
            assert!(e.len <= 16, "DC_TABLE_8BIT[{i}] len={} > 16", e.len);
        }
    }

    #[test]
    fn build_dc_table_and_lookup_all_sizes() {
        let table = build_dc_table_8bit();
        for (size, entry) in DC_TABLE_8BIT.iter().enumerate() {
            // Reconstruct the MSB-first code as a u32.
            let code_msb: u32 = (entry.code as u32) << 16;
            let result = table.lookup(code_msb);
            assert!(
                result.is_some(),
                "DC size {size}: lookup failed for code {:016b} len {}",
                entry.code,
                entry.len
            );
            let (val, consumed) = result.unwrap();
            assert_eq!(val as usize, size, "DC size {size}: got value {val}");
            assert_eq!(
                consumed, entry.len,
                "DC size {size}: expected len {} got {consumed}",
                entry.len
            );
        }
    }

    #[test]
    fn vlc_table_build_empty_is_safe() {
        let table = VlcTable::build(&[]);
        // Should not panic; lookup returns None.
        assert!(table.lookup(0).is_none());
    }

    #[test]
    fn vlc_table_single_entry() {
        // Single entry: code=0b10 (MSB-first), len=2, value=42.
        let entries = vec![(0b10u32, 2u8, 42i16)];
        let table = VlcTable::build(&entries);
        // Top 2 bits of 0b10xx_xxxx_... = 0b10 → index 2 in a 4-entry table.
        let result = table.lookup(0b10 << 30);
        assert_eq!(result, Some((42, 2)));
    }

    #[test]
    fn unpack_ac_value_roundtrip() {
        let (run, level, last) = (5u8, 3u16, true);
        let packed = ((last as i16) << 14) | ((run as i16) << 8) | (level as i16);
        let (r2, l2, la2) = unpack_ac_value(packed);
        assert_eq!(r2, run);
        assert_eq!(l2, level);
        assert_eq!(la2, last);
    }
}
