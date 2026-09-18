//! Bitstream packing and reading for Vorbis encoding/decoding.
//!
//! Vorbis uses LSB-first bit packing for its bitstream format.
//! Both [`BitPacker`] (encoder side) and [`BitReader`] (decoder side)
//! follow the same LSB-first ordering defined in the Vorbis I spec §A.2.

#![forbid(unsafe_code)]

use crate::AudioError;

/// Bitstream packer for Vorbis encoding.
#[derive(Debug, Clone)]
pub struct BitPacker {
    /// Accumulated bytes.
    bytes: Vec<u8>,
    /// Current byte being packed.
    current_byte: u8,
    /// Number of bits used in current byte.
    bit_position: u8,
}

impl BitPacker {
    /// Create a new bit packer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current_byte: 0,
            bit_position: 0,
        }
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << self.bit_position;
        }
        self.bit_position += 1;

        if self.bit_position == 8 {
            self.flush_byte();
        }
    }

    /// Write multiple bits (LSB first).
    pub fn write_bits(&mut self, value: u32, bits: u8) {
        for i in 0..bits {
            let bit = (value >> i) & 1;
            self.write_bit(bit != 0);
        }
    }

    /// Write a byte directly.
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(u32::from(byte), 8);
    }

    /// Write multiple bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_byte(byte);
        }
    }

    /// Write a signed integer.
    #[allow(clippy::cast_sign_loss)]
    pub fn write_signed(&mut self, value: i32, bits: u8) {
        let unsigned = value as u32;
        self.write_bits(unsigned, bits);
    }

    /// Flush current byte to buffer.
    fn flush_byte(&mut self) {
        self.bytes.push(self.current_byte);
        self.current_byte = 0;
        self.bit_position = 0;
    }

    /// Get the current bit position in the stream.
    #[must_use]
    pub fn position(&self) -> usize {
        self.bytes.len() * 8 + self.bit_position as usize
    }

    /// Finish packing and return the byte buffer.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_position > 0 {
            self.flush_byte();
        }
        self.bytes
    }

    /// Get a reference to the current bytes (without consuming).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the current size in bytes (including partial byte).
    #[must_use]
    pub fn size(&self) -> usize {
        let mut size = self.bytes.len();
        if self.bit_position > 0 {
            size += 1;
        }
        size
    }
}

impl Default for BitPacker {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// BitReader — LSB-first decoder side (Vorbis spec §A.2)
// ─────────────────────────────────────────────────────────────────

/// Bitstream reader for Vorbis decoding (LSB-first, per Vorbis spec).
///
/// Bits are extracted least-significant-bit first within each byte,
/// mirroring the ordering that [`BitPacker`] writes.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    /// Source bytes.
    data: &'a [u8],
    /// Current byte index.
    byte_pos: usize,
    /// Next bit index within the current byte (0 = LSB).
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader over `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Return the total number of bits already consumed.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }

    /// Return `true` when every bit has been consumed.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.byte_pos >= self.data.len()
    }

    /// Read a single bit (LSB-first).
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Eof`] when no more bits are available.
    pub fn read_bit(&mut self) -> Result<bool, AudioError> {
        if self.byte_pos >= self.data.len() {
            return Err(AudioError::Eof);
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Ok(bit != 0)
    }

    /// Read up to 32 bits LSB-first and return them as a `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Eof`] when the stream is exhausted before `n` bits
    /// are read, or [`AudioError::InvalidData`] if `n > 32`.
    pub fn read_bits(&mut self, n: u8) -> Result<u32, AudioError> {
        if n > 32 {
            return Err(AudioError::InvalidData(
                "Cannot read more than 32 bits at once".into(),
            ));
        }
        let mut result = 0u32;
        for i in 0..n {
            let bit = self.read_bit()? as u32;
            result |= bit << i; // LSB first
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_bit() {
        let mut packer = BitPacker::new();
        packer.write_bit(true);
        packer.write_bit(false);
        packer.write_bit(true);
        packer.write_bit(true);
        packer.write_bit(false);
        packer.write_bit(false);
        packer.write_bit(false);
        packer.write_bit(false);

        let bytes = packer.finish();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b0000_1101); // LSB first
    }

    #[test]
    fn test_write_bits() {
        let mut packer = BitPacker::new();
        packer.write_bits(0b1101, 4);
        packer.write_bits(0b0010, 4);

        let bytes = packer.finish();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b0010_1101);
    }

    #[test]
    fn test_write_byte() {
        let mut packer = BitPacker::new();
        packer.write_byte(0xAB);
        packer.write_byte(0xCD);

        let bytes = packer.finish();
        assert_eq!(bytes, vec![0xAB, 0xCD]);
    }

    #[test]
    fn test_write_bytes() {
        let mut packer = BitPacker::new();
        packer.write_bytes(&[0x01, 0x02, 0x03]);

        let bytes = packer.finish();
        assert_eq!(bytes, vec![0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_write_across_bytes() {
        let mut packer = BitPacker::new();
        packer.write_bits(0xFF, 6); // 6 bits of 1s: 0b111111
        packer.write_bits(0xFF, 6); // 6 more bits of 1s: 0b111111

        // Total: 12 bits of 1s = 0b111111111111
        // Byte 0 (bits 0-7): 0b11111111 = 0xFF = 255
        // Byte 1 (bits 8-11): 0b00001111 = 0x0F = 15
        let bytes = packer.finish();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0xFF); // All 8 bits set
        assert_eq!(bytes[1], 0x0F); // 4 bits set (bits 0-3 since LSB)
    }

    #[test]
    fn test_position() {
        let mut packer = BitPacker::new();
        assert_eq!(packer.position(), 0);

        packer.write_bits(0, 3);
        assert_eq!(packer.position(), 3);

        packer.write_bits(0, 5);
        assert_eq!(packer.position(), 8);

        packer.write_bits(0, 4);
        assert_eq!(packer.position(), 12);
    }

    #[test]
    fn test_size() {
        let mut packer = BitPacker::new();
        assert_eq!(packer.size(), 0);

        packer.write_bits(0, 3);
        assert_eq!(packer.size(), 1); // Partial byte

        packer.write_bits(0, 5);
        assert_eq!(packer.size(), 1); // Complete byte

        packer.write_bits(0, 4);
        assert_eq!(packer.size(), 2); // One complete, one partial
    }

    #[test]
    fn test_signed() {
        let mut packer = BitPacker::new();
        packer.write_signed(-1, 8);

        let bytes = packer.finish();
        assert_eq!(bytes[0], 0xFF);
    }

    // ─── BitReader round-trip tests ────────────────────────────────

    #[test]
    fn test_bit_reader_single_bit() {
        // byte 0x01 = 0b0000_0001: LSB is 1, rest 0
        let data = [0x01u8];
        let mut reader = BitReader::new(&data);
        assert!(reader.read_bit().expect("read bit 0"));
        assert!(!reader.read_bit().expect("read bit 1"));
    }

    #[test]
    fn test_bit_reader_exhausted() {
        let data = [0x00u8];
        let mut reader = BitReader::new(&data);
        // consume all 8 bits
        for _ in 0..8 {
            let _ = reader.read_bit();
        }
        assert!(reader.is_exhausted());
        assert!(reader.read_bit().is_err());
    }

    #[test]
    fn test_bit_reader_read_bits() {
        // 0b1101 = 13 packed LSB-first in low nibble of 0x0D
        let data = [0x0Du8];
        let mut reader = BitReader::new(&data);
        let val = reader.read_bits(4).expect("read 4 bits");
        assert_eq!(val, 0b1101); // LSB-first: bit0=1, bit1=0, bit2=1, bit3=1
    }

    #[test]
    fn test_packer_reader_round_trip() {
        // Encode a sequence of known values then decode them back.
        let mut packer = BitPacker::new();
        packer.write_bits(0b1011, 4); // 4-bit value
        packer.write_bits(0b00110, 5); // 5-bit value
        let bytes = packer.finish();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(4).expect("4 bits"), 0b1011);
        assert_eq!(reader.read_bits(5).expect("5 bits"), 0b00110);
    }

    #[test]
    fn test_bit_reader_position() {
        let data = [0xFFu8, 0xFFu8];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.position(), 0);
        let _ = reader.read_bits(3);
        assert_eq!(reader.position(), 3);
        let _ = reader.read_bits(5);
        assert_eq!(reader.position(), 8);
        let _ = reader.read_bits(4);
        assert_eq!(reader.position(), 12);
    }
}
