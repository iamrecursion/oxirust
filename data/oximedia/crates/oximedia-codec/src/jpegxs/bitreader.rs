// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! MSB-first bit reader for JPEG XS entropy-coded payloads.
//!
//! JPEG XS uses MSB-first bit packing without byte-stuffing (unlike JPEG 2000).
//! Bits are packed from the most-significant bit of each byte downward.

use super::JxsError;

/// MSB-first bit reader over an immutable byte slice.
///
/// Bits are read starting from the most-significant bit of `data[0]`,
/// then descending to the least-significant, then moving to `data[1]`, etc.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the current byte being consumed.
    byte_pos: usize,
    /// Bit position within the current byte: 7 = MSB (first to be read), 0 = LSB (last).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` wrapping `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 7,
        }
    }

    /// Read one bit. Returns `0` or `1`.
    ///
    /// # Errors
    /// Returns `JxsError::TruncatedStream` if the stream is exhausted.
    pub fn read_bit(&mut self) -> Result<u8, JxsError> {
        if self.byte_pos >= self.data.len() {
            return Err(JxsError::TruncatedStream {
                need: self.byte_pos + 1,
                have: self.data.len(),
            });
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        if self.bit_pos == 0 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        } else {
            self.bit_pos -= 1;
        }
        Ok(bit)
    }

    /// Read `n` bits (0..=32) as a big-endian `u32`.
    ///
    /// # Errors
    /// Returns `JxsError::TruncatedStream` if the stream is exhausted before `n` bits are read.
    pub fn read_bits_u32(&mut self, n: u8) -> Result<u32, JxsError> {
        let mut val: u32 = 0;
        for _ in 0..n {
            val = (val << 1) | u32::from(self.read_bit()?);
        }
        Ok(val)
    }

    /// Read 8 bits as a `u8`.
    ///
    /// # Errors
    /// Returns `JxsError::TruncatedStream` if fewer than 8 bits remain.
    pub fn read_u8(&mut self) -> Result<u8, JxsError> {
        self.read_bits_u32(8).map(|v| v as u8)
    }

    /// Read 16 bits as a big-endian `u16`.
    ///
    /// # Errors
    /// Returns `JxsError::TruncatedStream` if fewer than 16 bits remain.
    pub fn read_u16_be(&mut self) -> Result<u16, JxsError> {
        self.read_bits_u32(16).map(|v| v as u16)
    }

    /// Advance to the next byte boundary, discarding any remaining bits in the current byte.
    ///
    /// If already byte-aligned (just read the last bit of a byte), this is a no-op.
    pub fn byte_align(&mut self) {
        if self.bit_pos != 7 {
            self.byte_pos += 1;
            self.bit_pos = 7;
        }
    }

    /// Return the current byte position (number of fully consumed bytes).
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Return the number of whole bytes remaining after the current bit position.
    pub fn remaining_bytes(&self) -> usize {
        self.data.len().saturating_sub(self.byte_pos)
    }

    /// Peek at the next `n` bits (0..=32) as a `u32` without advancing the position.
    ///
    /// If fewer than `n` bits remain, pads with zeros on the right.
    pub fn peek_bits_u32(&self, n: u8) -> u32 {
        // Build a temporary clone of the reader state for peeking.
        let mut tmp_byte = self.byte_pos;
        let mut tmp_bit = self.bit_pos;
        let mut val: u32 = 0;
        for _ in 0..n {
            let bit = if tmp_byte < self.data.len() {
                (self.data[tmp_byte] >> tmp_bit) & 1
            } else {
                0
            };
            val = (val << 1) | u32::from(bit);
            if tmp_bit == 0 {
                tmp_byte += 1;
                tmp_bit = 7;
            } else {
                tmp_bit -= 1;
            }
        }
        val
    }

    /// Skip `n` bits without reading them.
    ///
    /// # Errors
    /// Returns `JxsError::TruncatedStream` if fewer than `n` bits remain.
    pub fn skip_bits(&mut self, n: u8) -> Result<(), JxsError> {
        for _ in 0..n {
            self.read_bit()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bit_msb_first() {
        let data = &[0b1010_0011u8];
        let mut r = BitReader::new(data);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 1);
    }

    #[test]
    fn read_bits_u32_full_byte() {
        let data = &[0xABu8];
        let mut r = BitReader::new(data);
        assert_eq!(r.read_bits_u32(8).unwrap(), 0xAB);
    }

    #[test]
    fn read_bits_u32_crosses_byte_boundary() {
        let data = &[0xABu8, 0xCDu8];
        let mut r = BitReader::new(data);
        // Read 12 bits: top 8 = 0xAB, next 4 = 0xC → 0xABC
        assert_eq!(r.read_bits_u32(12).unwrap(), 0xABC);
    }

    #[test]
    fn read_u16_be() {
        let data = &[0x12u8, 0x34u8];
        let mut r = BitReader::new(data);
        assert_eq!(r.read_u16_be().unwrap(), 0x1234);
    }

    #[test]
    fn truncated_stream_error() {
        let data = &[0xFFu8];
        let mut r = BitReader::new(data);
        let _ = r.read_bits_u32(8).unwrap();
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn byte_align_advances_past_partial_byte() {
        let data = &[0xFFu8, 0xAAu8];
        let mut r = BitReader::new(data);
        // Read 3 bits from byte 0
        let _ = r.read_bits_u32(3).unwrap();
        // Now byte_align: should skip to byte 1
        r.byte_align();
        assert_eq!(r.byte_pos(), 1);
        // Next read should come from byte 1 (0xAA)
        assert_eq!(r.read_u8().unwrap(), 0xAA);
    }

    #[test]
    fn byte_align_on_boundary_is_noop() {
        let data = &[0xFFu8, 0xAAu8];
        let mut r = BitReader::new(data);
        // Read all 8 bits of byte 0 — now perfectly aligned
        let _ = r.read_bits_u32(8).unwrap();
        assert_eq!(r.byte_pos(), 1);
        r.byte_align();
        // Still at byte 1
        assert_eq!(r.byte_pos(), 1);
    }

    #[test]
    fn peek_does_not_advance() {
        let data = &[0b1100_0000u8];
        let mut r = BitReader::new(data);
        let peeked = r.peek_bits_u32(4);
        assert_eq!(peeked, 0b1100);
        // Position should be unchanged
        assert_eq!(r.byte_pos(), 0);
        // Read the same bits
        let read = r.read_bits_u32(4).unwrap();
        assert_eq!(read, 0b1100);
    }

    #[test]
    fn skip_bits_advances_position() {
        let data = &[0b1111_0000u8];
        let mut r = BitReader::new(data);
        r.skip_bits(4).unwrap();
        // Next 4 bits should be 0b0000
        assert_eq!(r.read_bits_u32(4).unwrap(), 0b0000);
    }
}
