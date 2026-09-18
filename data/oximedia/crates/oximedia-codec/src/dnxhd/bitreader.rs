//! MSB-first bit reader for DNxHD entropy-coded payloads.
//!
//! DNxHD (VC-3 / SMPTE ST 2019-1) uses MSB-first bit packing, matching
//! the MPEG-2 and ProRes convention. This reader is adapted from the ProRes
//! bit reader but uses `DecodeError` from the parent module.

use super::DecodeError;

/// MSB-first bit reader over an immutable byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` wrapping `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Number of bits remaining.
    pub fn remaining_bits(&self) -> usize {
        let bytes_left = self.data.len().saturating_sub(self.byte_pos);
        bytes_left * 8 - self.bit_pos as usize
    }

    /// Current byte position (number of whole bytes consumed, rounding down).
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// Read a single bit.
    pub fn read_bit(&mut self) -> Result<bool, DecodeError> {
        if self.byte_pos >= self.data.len() {
            return Err(DecodeError::BufferTooSmall { need: 1, have: 0 });
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
    pub fn read_bits_u32(&mut self, n: u8) -> Result<u32, DecodeError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(DecodeError::InvalidData(format!(
                "cannot read {n} bits in a single call (max 32)"
            )));
        }
        if self.remaining_bits() < n as usize {
            return Err(DecodeError::BufferTooSmall {
                need: n as usize,
                have: self.remaining_bits(),
            });
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            let bit = self.read_bit()? as u32;
            value = (value << 1) | bit;
        }
        Ok(value)
    }

    /// Read one whole byte.
    pub fn read_byte(&mut self) -> Result<u8, DecodeError> {
        self.read_bits_u32(8).map(|v| v as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_single_bits_msb_first() {
        // 0xB4 = 0b10110100
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
        // 12 bits = 0xABC
        assert_eq!(r.read_bits_u32(12).unwrap(), 0xABC);
        assert_eq!(r.remaining_bits(), 4);
        assert_eq!(r.read_bits_u32(4).unwrap(), 0xD);
    }

    #[test]
    fn read_zero_bits_is_zero() {
        let buf = [0xFFu8];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits_u32(0).unwrap(), 0);
        assert_eq!(r.remaining_bits(), 8);
    }

    #[test]
    fn read_byte_works() {
        let buf = [0x5Au8];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_byte().unwrap(), 0x5A);
        assert_eq!(r.remaining_bits(), 0);
    }

    #[test]
    fn out_of_input_errors() {
        let buf = [0xFFu8];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits_u32(8).unwrap();
        assert!(r.read_bit().is_err());
    }
}
