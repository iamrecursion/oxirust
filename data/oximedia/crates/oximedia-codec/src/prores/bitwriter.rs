//! MSB-first bit writer for ProRes entropy-coded slice payloads.
//!
//! The inverse of [`super::bitreader::BitReader`]. Bits are packed into bytes
//! MSB-first: the first written bit occupies bit 7 of byte 0, then bit 6, …
//! bit 0, then bit 7 of byte 1, etc. This matches the ProRes / H.26x bitstream
//! convention defined in SMPTE RDD 36.

/// MSB-first bit writer that accumulates bits into a byte vector.
pub struct BitWriter {
    buf: Vec<u8>,
    cur_byte: u8,
    bits_in_cur: u8,
    total_bits: usize,
}

impl BitWriter {
    /// Create a new, empty `BitWriter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            cur_byte: 0,
            bits_in_cur: 0,
            total_bits: 0,
        }
    }

    /// Write a single bit (false = 0, true = 1).
    pub fn write_bit(&mut self, bit: bool) {
        let b = u8::from(bit);
        // Shift to the MSB position currently being filled.
        self.cur_byte |= b << (7 - self.bits_in_cur);
        self.bits_in_cur += 1;
        self.total_bits += 1;
        if self.bits_in_cur == 8 {
            self.buf.push(self.cur_byte);
            self.cur_byte = 0;
            self.bits_in_cur = 0;
        }
    }

    /// Write the least-significant `count` bits of `value`, MSB-first.
    ///
    /// `count` must be in `[0, 32]`. When `count == 0` this is a no-op.
    pub fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            let bit = ((value >> i) & 1) != 0;
            self.write_bit(bit);
        }
    }

    /// Pad the current partial byte with zero bits and push it to the buffer.
    ///
    /// If the writer is already byte-aligned this is a no-op.
    pub fn flush(&mut self) {
        if self.bits_in_cur != 0 {
            self.buf.push(self.cur_byte);
            self.cur_byte = 0;
            self.bits_in_cur = 0;
        }
    }

    /// Flush and return the underlying byte vector.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.buf
    }

    /// Number of complete bytes already committed to the buffer (not counting
    /// any partial byte still being filled).
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True when no bits have been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.bits_in_cur == 0
    }

    /// Total number of bits written so far (including any partial byte).
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.total_bits
    }

    /// Number of bytes that will be produced after flushing (rounds up to the
    /// nearest byte boundary).
    #[must_use]
    pub fn byte_count(&self) -> usize {
        (self.total_bits + 7) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prores::bitreader::BitReader;

    #[test]
    fn write_then_read_single_bits_msb_first() {
        // Pattern 1011 0100 (= 0xB4)
        let mut w = BitWriter::new();
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(true);
        w.write_bit(false);
        w.write_bit(false);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0xB4]);
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 0);
    }

    #[test]
    fn write_bits_packs_correctly_across_byte_boundary() {
        // Write 12 bits = 0xABC = 1010_1011_1100.
        // Should produce bytes 0xAB 0xC0 (last nibble zero-padded).
        let mut w = BitWriter::new();
        w.write_bits(0xABC, 12);
        let bytes = w.into_bytes();
        assert_eq!(bytes[0], 0xAB);
        assert_eq!(bytes[1], 0xC0);
    }

    #[test]
    fn write_zero_bits_is_noop() {
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 0);
        assert!(w.is_empty());
    }

    #[test]
    fn flush_pads_with_zeros() {
        let mut w = BitWriter::new();
        // Write 5 ones → should be 1111_1000 = 0xF8 after flush.
        w.write_bits(0x1F, 5);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0xF8]);
    }

    #[test]
    fn byte_count_rounds_up() {
        let mut w = BitWriter::new();
        w.write_bits(0, 1);
        assert_eq!(w.byte_count(), 1);
        w.write_bits(0, 7);
        assert_eq!(w.byte_count(), 1);
        w.write_bits(0, 1);
        assert_eq!(w.byte_count(), 2);
    }

    #[test]
    fn round_trip_32_bits() {
        let val = 0xDEAD_BEEFu32;
        let mut w = BitWriter::new();
        w.write_bits(val, 32);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(32).unwrap(), val);
    }
}
