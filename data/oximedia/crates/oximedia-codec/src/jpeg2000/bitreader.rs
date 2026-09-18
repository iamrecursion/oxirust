//! MSB-first bit reader for JPEG 2000 codestream bytes.
//!
//! Handles the JPEG 2000 byte-stuffing rule: after a 0xFF byte, the next
//! byte's most-significant bit is a stuffed zero and is dropped. This
//! prevents any 0xFF 0xXX pattern (where XX > 0x8F) from being confused
//! with a marker.
//!
//! Bit ordering: MSB-first within each byte (bit 7 read first).

use super::{Jp2Error, Jp2Result};

/// MSB-first bit reader with JPEG 2000 byte-stuffing support.
pub struct J2kBitReader<'a> {
    buf: &'a [u8],
    /// Current byte-level position.
    pos: usize,
    /// Bit position within `cur_byte`: 7 = next bit is MSB, -1 = byte exhausted.
    bit_pos: i8,
    /// Current byte being consumed.
    cur_byte: u8,
}

impl<'a> J2kBitReader<'a> {
    /// Create a new bit reader over the given byte slice.
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            bit_pos: -1,
            cur_byte: 0,
        }
    }

    /// Fetch the next raw byte, applying JPEG 2000 byte-stuffing.
    ///
    /// If the previous byte was 0xFF and the next byte is <= 0x8F,
    /// a stuffed zero bit is prepended to the next byte (bit 7 = 0,
    /// actual bits 6..0 from the byte). If the next byte is > 0x8F
    /// that is a marker — no stuffing occurs and we treat the byte
    /// as raw continuation data.
    fn load_next_byte(&mut self) -> Jp2Result<()> {
        if self.pos >= self.buf.len() {
            return Err(Jp2Error::BitReaderOutOfInput);
        }
        let prev_was_ff = self.pos > 0 && self.buf[self.pos - 1] == 0xFF;
        let byte = self.buf[self.pos];
        self.pos += 1;

        if prev_was_ff && byte <= 0x8F {
            // Stuffed byte: bit 7 is a stuffed 0; bits 6..0 come from byte[6..0].
            // Effective value is `byte & 0x7F` with 7 valid bits.
            self.cur_byte = byte & 0x7F;
            self.bit_pos = 6; // only 7 bits valid (6 down to 0)
        } else {
            self.cur_byte = byte;
            self.bit_pos = 7; // 8 bits valid (7 down to 0)
        }
        Ok(())
    }

    /// Read a single bit (0 or 1).
    pub fn read_bit(&mut self) -> Jp2Result<u8> {
        if self.bit_pos < 0 {
            self.load_next_byte()?;
        }
        let bit = (self.cur_byte >> self.bit_pos) & 1;
        self.bit_pos -= 1;
        Ok(bit)
    }

    /// Read `n` bits (0..=32) as a big-endian `u32`.
    pub fn read_bits(&mut self, n: u8) -> Jp2Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            let bit = self.read_bit()?;
            value = (value << 1) | u32::from(bit);
        }
        Ok(value)
    }

    /// Return the number of bytes remaining (approximate — does not account
    /// for partially consumed bytes).
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Align the reader to the next byte boundary (discard remaining bits in
    /// the current byte). No-op if already at a byte boundary.
    pub fn align_to_byte(&mut self) {
        self.bit_pos = -1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bits_basic() {
        // 0b1010_1010 = 0xAA
        let buf = [0xAAu8, 0x55];
        let mut r = J2kBitReader::new(&buf);
        assert_eq!(r.read_bit().expect("bit"), 1);
        assert_eq!(r.read_bit().expect("bit"), 0);
        assert_eq!(r.read_bits(6).expect("bits"), 0b10_1010);
        // Second byte 0x55 = 0b0101_0101
        assert_eq!(r.read_bits(8).expect("bits"), 0x55);
    }

    #[test]
    fn read_bits_zero_returns_zero() {
        let buf = [0xFFu8];
        let mut r = J2kBitReader::new(&buf);
        assert_eq!(r.read_bits(0).expect("zero"), 0);
    }

    #[test]
    fn byte_stuffing_drops_top_bit_after_ff() {
        // 0xFF followed by 0x50 (<=0x8F): stuffed → bit 7 dropped, effective = 0x50 & 0x7F = 0x50
        // 7 bits: 101_0000
        let buf = [0xFFu8, 0x50];
        let mut r = J2kBitReader::new(&buf);
        // Consume the 0xFF byte (8 bits = 0xFF)
        let first = r.read_bits(8).expect("first byte");
        assert_eq!(first, 0xFF);
        // Now the stuffed byte: 7 valid bits from 0x50 & 0x7F = 0x50 = 0101_0000
        let stuffed = r.read_bits(7).expect("stuffed byte");
        assert_eq!(stuffed, 0x50u8 as u32);
    }

    #[test]
    fn out_of_input_error() {
        let buf: [u8; 0] = [];
        let mut r = J2kBitReader::new(&buf);
        assert!(r.read_bit().is_err());
    }

    #[test]
    fn remaining_decreases() {
        let buf = [0x00u8, 0x00, 0x00];
        let mut r = J2kBitReader::new(&buf);
        assert_eq!(r.remaining(), 3);
        let _ = r.read_bits(8).expect("ok");
        assert_eq!(r.remaining(), 2);
    }

    #[test]
    fn align_to_byte_discards_partial() {
        let buf = [0b1111_0000u8, 0xAB];
        let mut r = J2kBitReader::new(&buf);
        let _ = r.read_bits(4).expect("ok"); // consume top 4 bits
        r.align_to_byte();
        // Next read should be from 0xAB
        let v = r.read_bits(8).expect("ok");
        assert_eq!(v, 0xAB);
    }
}
