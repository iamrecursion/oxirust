// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! MSB-first bit writer for JPEG XS entropy-coded payloads.
//!
//! This is the exact inverse of [`super::bitreader::BitReader`]: bits are
//! packed starting from the most-significant bit of each output byte downward,
//! with **no byte-stuffing** (JPEG XS, unlike JPEG 2000, does not stuff bytes).
//!
//! A buffer written with [`BitWriter`] and then flushed reads back identically
//! through `BitReader`, which is asserted by the round-trip unit tests.

/// MSB-first bit writer accumulating into an owned `Vec<u8>`.
///
/// Bits are written starting from bit 7 (MSB) of the current byte, descending
/// to bit 0 (LSB), then advancing to the next byte. Partial final bytes are
/// zero-padded on flush, matching `BitReader`'s discard-of-trailing-bits
/// behaviour at the end of a stream.
#[derive(Debug, Default)]
pub struct BitWriter {
    /// Completed bytes plus the in-progress byte once flushed.
    bytes: Vec<u8>,
    /// The byte currently being assembled (bits packed MSB-first).
    current: u8,
    /// Number of bits already placed into `current` (0..=7).
    nbits: u8,
}

impl BitWriter {
    /// Create a new, empty `BitWriter`.
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            nbits: 0,
        }
    }

    /// Create a `BitWriter` with reserved capacity for `cap` bytes.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            current: 0,
            nbits: 0,
        }
    }

    /// Write a single bit (`0` or `1`). Any non-zero `bit` is treated as `1`.
    pub fn write_bit(&mut self, bit: u8) {
        // Place the bit at position (7 - nbits) so the first bit becomes the MSB.
        let shift = 7 - self.nbits;
        if bit & 1 != 0 {
            self.current |= 1u8 << shift;
        }
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.nbits = 0;
        }
    }

    /// Write the low `n` bits of `val` (0..=32), most-significant bit first.
    ///
    /// Bits above bit `n-1` are ignored. For `n == 0` this is a no-op.
    pub fn write_bits_u32(&mut self, val: u32, n: u8) {
        let count = n.min(32);
        for i in (0..count).rev() {
            let bit = ((val >> i) & 1) as u8;
            self.write_bit(bit);
        }
    }

    /// Write a full byte (8 bits), most-significant bit first.
    pub fn write_u8(&mut self, val: u8) {
        self.write_bits_u32(u32::from(val), 8);
    }

    /// Write a big-endian `u16` (16 bits), most-significant bit first.
    pub fn write_u16_be(&mut self, val: u16) {
        self.write_bits_u32(u32::from(val), 16);
    }

    /// Number of complete bytes emitted so far (excludes the in-progress byte).
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// `true` if no bits have been written yet.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty() && self.nbits == 0
    }

    /// Return `true` if the writer is currently on a byte boundary
    /// (i.e. the next bit would start a fresh byte).
    pub fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Pad the in-progress byte with zero bits up to the next byte boundary.
    ///
    /// If already aligned, this is a no-op. This mirrors `BitReader::byte_align`,
    /// which discards the same trailing bits on the read side.
    pub fn byte_align(&mut self) {
        if self.nbits != 0 {
            self.bytes.push(self.current);
            self.current = 0;
            self.nbits = 0;
        }
    }

    /// Flush any partial final byte (zero-padded) and return the byte buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.byte_align();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegxs::bitreader::BitReader;

    #[test]
    fn write_bit_msb_first() {
        let mut w = BitWriter::new();
        for &b in &[1u8, 0, 1, 0, 0, 0, 1, 1] {
            w.write_bit(b);
        }
        let out = w.finish();
        assert_eq!(out, vec![0b1010_0011]);
    }

    #[test]
    fn write_bits_u32_full_byte() {
        let mut w = BitWriter::new();
        w.write_bits_u32(0xAB, 8);
        assert_eq!(w.finish(), vec![0xAB]);
    }

    #[test]
    fn write_bits_crosses_byte_boundary() {
        let mut w = BitWriter::new();
        // 12 bits: 0xABC → bytes 0xAB, 0xC0 (last 4 bits zero-padded)
        w.write_bits_u32(0xABC, 12);
        assert_eq!(w.finish(), vec![0xAB, 0xC0]);
    }

    #[test]
    fn write_u16_be_roundtrip_with_reader() {
        let mut w = BitWriter::new();
        w.write_u16_be(0x1234);
        let out = w.finish();
        let mut r = BitReader::new(&out);
        assert_eq!(r.read_u16_be().unwrap(), 0x1234);
    }

    #[test]
    fn roundtrip_mixed_widths_via_reader() {
        // Write a mixture of widths, then read them back with BitReader.
        let mut w = BitWriter::new();
        w.write_bit(1);
        w.write_bits_u32(0b101, 3);
        w.write_u8(0xF0);
        w.write_bits_u32(0x3FF, 10);
        w.write_bit(0);
        let out = w.finish();

        let mut r = BitReader::new(&out);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bits_u32(3).unwrap(), 0b101);
        assert_eq!(r.read_u8().unwrap(), 0xF0);
        assert_eq!(r.read_bits_u32(10).unwrap(), 0x3FF);
        assert_eq!(r.read_bit().unwrap(), 0);
    }

    #[test]
    fn byte_align_pads_with_zeros() {
        let mut w = BitWriter::new();
        w.write_bits_u32(0b111, 3);
        w.byte_align();
        // 3 ones then 5 zero pad → 0b1110_0000
        assert_eq!(w.byte_len(), 1);
        w.write_u8(0xAA);
        assert_eq!(w.finish(), vec![0b1110_0000, 0xAA]);
    }

    #[test]
    fn empty_writer_finishes_empty() {
        let w = BitWriter::new();
        assert!(w.is_empty());
        assert!(w.finish().is_empty());
    }

    #[test]
    fn write_bits_u32_zero_width_is_noop() {
        let mut w = BitWriter::new();
        w.write_bits_u32(0xFFFF_FFFF, 0);
        assert!(w.is_empty());
    }

    #[test]
    fn long_run_of_ones_roundtrips() {
        // 8 ones in a row forms 0xFF — verify it reads back exactly (no stuffing).
        let mut w = BitWriter::new();
        for _ in 0..8 {
            w.write_bit(1);
        }
        let out = w.finish();
        assert_eq!(out, vec![0xFF]);
        let mut r = BitReader::new(&out);
        for _ in 0..8 {
            assert_eq!(r.read_bit().unwrap(), 1);
        }
    }

    #[test]
    fn alignment_flag_tracks_state() {
        let mut w = BitWriter::new();
        assert!(w.is_byte_aligned());
        w.write_bit(1);
        assert!(!w.is_byte_aligned());
        w.write_bits_u32(0, 7);
        assert!(w.is_byte_aligned());
    }
}
