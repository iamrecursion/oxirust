//! MSB-first bit reader for MPEG-2 video bitstreams (ISO/IEC 13818-2).
//!
//! MPEG-2 packs bits most-significant-bit-first. This reader is pattern-copied
//! from the DNxHD/ProRes bit reader but uses the local [`Mpeg2Error`] so the
//! `mpeg2` module compiles with only `--features mpeg2`.
//!
//! It additionally provides byte-aligned start-code scanning: MPEG-2 separates
//! syntax structures with 32-bit start codes of the form `00 00 01 <code>`
//! (ISO/IEC 13818-2 §6.2.1), always byte-aligned in the elementary stream.

use super::{Mpeg2Error, Mpeg2Result};

/// 24-bit start-code prefix `0x000001` that precedes every MPEG-2 start code.
pub const START_CODE_PREFIX: u32 = 0x0000_01;

/// `picture_start_code` value (`0x00`).
pub const PICTURE_START_CODE: u8 = 0x00;
/// First `slice_start_code` value. Slice codes span `0x01..=0xAF`.
pub const SLICE_START_CODE_MIN: u8 = 0x01;
/// Last `slice_start_code` value.
pub const SLICE_START_CODE_MAX: u8 = 0xAF;
/// `user_data_start_code` value (`0xB2`).
pub const USER_DATA_START_CODE: u8 = 0xB2;
/// `sequence_header_code` value (`0xB3`).
pub const SEQUENCE_HEADER_CODE: u8 = 0xB3;
/// `extension_start_code` value (`0xB5`).
pub const EXTENSION_START_CODE: u8 = 0xB5;
/// `sequence_end_code` value (`0xB7`).
pub const SEQUENCE_END_CODE: u8 = 0xB7;
/// `group_start_code` (GOP header) value (`0xB8`).
pub const GROUP_START_CODE: u8 = 0xB8;

/// MSB-first bit reader over an immutable byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to consume.
    byte_pos: usize,
    /// Bit offset within the current byte (0 == MSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` wrapping `data`, positioned at the start.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Create a `BitReader` starting at byte offset `byte_pos`.
    #[must_use]
    pub fn new_at(data: &'a [u8], byte_pos: usize) -> Self {
        Self {
            data,
            byte_pos: byte_pos.min(data.len()),
            bit_pos: 0,
        }
    }

    /// Number of bits remaining to be read.
    #[must_use]
    pub fn remaining_bits(&self) -> usize {
        let bytes_left = self.data.len().saturating_sub(self.byte_pos);
        bytes_left * 8 - self.bit_pos as usize
    }

    /// Current byte position (whole bytes consumed, rounding down).
    #[must_use]
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// `true` if the reader is currently byte-aligned.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos == 0
    }

    /// Advance to the next byte boundary, discarding any partial-byte bits.
    pub fn align_to_byte(&mut self) {
        if self.bit_pos != 0 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
    }

    /// Read a single bit as a `bool` (`true` == 1).
    pub fn read_bit(&mut self) -> Mpeg2Result<bool> {
        if self.byte_pos >= self.data.len() {
            return Err(Mpeg2Error::UnexpectedEof { need: 1, have: 0 });
        }
        let byte = self.data[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit != 0)
    }

    /// Read `n` bits (0..=32) as a big-endian `u32`.
    pub fn read_bits(&mut self, n: u8) -> Mpeg2Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(Mpeg2Error::InvalidData(format!(
                "cannot read {n} bits in a single call (max 32)"
            )));
        }
        if self.remaining_bits() < n as usize {
            return Err(Mpeg2Error::UnexpectedEof {
                need: n as usize,
                have: self.remaining_bits(),
            });
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            let bit = u32::from(self.read_bit()?);
            value = (value << 1) | bit;
        }
        Ok(value)
    }

    /// Read `n` bits and return them as an `i32` (unsigned magnitude; caller
    /// applies any sign interpretation).
    pub fn read_bits_i32(&mut self, n: u8) -> Mpeg2Result<i32> {
        Ok(self.read_bits(n)? as i32)
    }

    /// Peek up to 32 bits without consuming them, MSB-aligned into the high bits
    /// of the returned `u32`. Fewer than 32 bits remaining are zero-padded on
    /// the low end. Used by the VLC matchers.
    #[must_use]
    pub fn peek_bits_msb_aligned(&self) -> u32 {
        let mut value: u32 = 0;
        let mut byte_pos = self.byte_pos;
        let mut bit_pos = self.bit_pos;
        for _ in 0..32 {
            let bit = if byte_pos < self.data.len() {
                (self.data[byte_pos] >> (7 - bit_pos)) & 1
            } else {
                0
            };
            value = (value << 1) | u32::from(bit);
            bit_pos += 1;
            if bit_pos == 8 {
                bit_pos = 0;
                byte_pos += 1;
            }
        }
        value
    }

    /// Consume exactly `n` bits, discarding the value. Used after a VLC match.
    pub fn skip_bits(&mut self, n: u8) -> Mpeg2Result<()> {
        let _ = self.read_bits(n.min(32))?;
        Ok(())
    }
}

/// A start code located in a byte slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartCode {
    /// The start-code value byte (the byte following the `00 00 01` prefix).
    pub code: u8,
    /// Byte offset of the `0x00 0x00 0x01` prefix.
    pub prefix_offset: usize,
    /// Byte offset of the first payload byte (just after the code byte).
    pub payload_offset: usize,
}

/// Find the first start code `00 00 01 <code>` at or after `from` (byte index).
///
/// Returns `None` if no complete start code is present in the remaining bytes.
#[must_use]
pub fn find_start_code(data: &[u8], from: usize) -> Option<StartCode> {
    if data.len() < 4 {
        return None;
    }
    let mut i = from;
    while i + 3 < data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
            return Some(StartCode {
                code: data[i + 3],
                prefix_offset: i,
                payload_offset: i + 4,
            });
        }
        i += 1;
    }
    None
}

/// Find the first start code whose value equals `target`, at or after `from`.
#[must_use]
pub fn find_specific_start_code(data: &[u8], from: usize, target: u8) -> Option<StartCode> {
    let mut search_from = from;
    while let Some(sc) = find_start_code(data, search_from) {
        if sc.code == target {
            return Some(sc);
        }
        search_from = sc.prefix_offset + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_single_bits_msb_first() {
        // 0xB4 = 0b1011_0100
        let buf = [0xB4u8];
        let mut r = BitReader::new(&buf);
        let expected = [true, false, true, true, false, true, false, false];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(r.read_bit().unwrap(), exp, "bit {i}");
        }
        assert_eq!(r.remaining_bits(), 0);
    }

    #[test]
    fn read_bits_across_boundary() {
        let buf = [0xABu8, 0xCD];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(12).unwrap(), 0xABC);
        assert_eq!(r.remaining_bits(), 4);
        assert_eq!(r.read_bits(4).unwrap(), 0xD);
    }

    #[test]
    fn read_zero_bits_is_zero() {
        let buf = [0xFFu8];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(0).unwrap(), 0);
        assert_eq!(r.remaining_bits(), 8);
    }

    #[test]
    fn out_of_input_errors() {
        let buf = [0xFFu8];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits(8).unwrap();
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn peek_does_not_consume() {
        let buf = [0xC0u8, 0x00];
        let r = BitReader::new(&buf);
        let peeked = r.peek_bits_msb_aligned();
        // Top two bits set → 0b11 in the MSBs.
        assert_eq!(peeked >> 30, 0b11);
        assert_eq!(r.remaining_bits(), 16);
    }

    #[test]
    fn align_to_byte_skips_partial() {
        let buf = [0xFFu8, 0xAA];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits(3).unwrap();
        r.align_to_byte();
        assert!(r.is_byte_aligned());
        assert_eq!(r.byte_pos(), 1);
        assert_eq!(r.read_bits(8).unwrap(), 0xAA);
    }

    #[test]
    fn find_start_code_basic() {
        // prefix then code 0xB3 (sequence header).
        let buf = [0xFF, 0x00, 0x00, 0x01, 0xB3, 0x12, 0x34];
        let sc = find_start_code(&buf, 0).expect("start code");
        assert_eq!(sc.code, 0xB3);
        assert_eq!(sc.prefix_offset, 1);
        assert_eq!(sc.payload_offset, 5);
    }

    #[test]
    fn find_specific_start_code_skips_others() {
        // First a sequence header (B3), then a picture start code (00).
        let buf = [0x00, 0x00, 0x01, 0xB3, 0xAA, 0x00, 0x00, 0x01, 0x00, 0xBB];
        let sc = find_specific_start_code(&buf, 0, PICTURE_START_CODE).expect("picture sc");
        assert_eq!(sc.code, PICTURE_START_CODE);
        assert_eq!(sc.payload_offset, 9);
    }

    #[test]
    fn find_start_code_none_when_absent() {
        let buf = [0xDE, 0xAD, 0xBE, 0xEF];
        assert!(find_start_code(&buf, 0).is_none());
    }
}
