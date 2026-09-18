//! MSB-first bit writer for MPEG-2 video bitstreams (ISO/IEC 13818-2).
//!
//! Exact inverse of [`super::bitreader::BitReader`]: bits are packed
//! most-significant-bit-first within each byte (the first bit written lands in
//! bit 7 of the first byte), so a [`BitReader`](super::bitreader::BitReader)
//! reading the produced bytes recovers them in the same order.
//!
//! MPEG-2 video elementary streams use **no byte-stuffing**; start-code
//! emulation is avoided by the bit layout itself (the writer never inserts a
//! `00 00 01` triple inside coded data because every coded block ends with a
//! terminating `1` somewhere within the coefficient/EOB codes — and the
//! encoder byte-aligns with zero stuffing-bits only at slice boundaries, where
//! a real start code legitimately follows).
//!
//! A 32-bit start code of the form `00 00 01 <code>` (ISO/IEC 13818-2 §6.2.1)
//! is always emitted byte-aligned via [`BitWriter::write_start_code`].

/// MSB-first bit writer accumulating into a growing byte buffer.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    /// Completed whole bytes plus, while `bit_count > 0`, a partial byte in
    /// `partial` that has not yet been pushed.
    bytes: Vec<u8>,
    /// Accumulator for the byte currently being filled (MSB side first).
    partial: u8,
    /// Number of bits already placed into `partial` (0..=7).
    bit_count: u8,
}

impl BitWriter {
    /// Create a new, empty `BitWriter`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `BitWriter` with capacity for `cap` bytes pre-reserved.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
            partial: 0,
            bit_count: 0,
        }
    }

    /// Number of whole bits written so far (including any partial byte).
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.bit_count as usize
    }

    /// `true` if the writer is currently on a byte boundary.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_count == 0
    }

    /// Append a single bit (`true` == 1) MSB-first.
    pub fn write_bit(&mut self, bit: bool) {
        self.partial = (self.partial << 1) | u8::from(bit);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.bytes.push(self.partial);
            self.partial = 0;
            self.bit_count = 0;
        }
    }

    /// Append the low `n` bits of `value` (0..=32), MSB-first.
    ///
    /// Bit `n-1` of `value` is written first (it becomes the most significant of
    /// the emitted field), matching [`crate::mpeg2::bitreader::BitReader::read_bits`].
    pub fn write_bits(&mut self, value: u32, n: u8) {
        let n = n.min(32);
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    /// Pad to the next byte boundary with zero stuffing-bits (ISO/IEC 13818-2
    /// `next_start_code()` stuffing pattern uses `0` bits after alignment is
    /// already a byte). If already aligned this is a no-op.
    pub fn align_to_byte_with_zeros(&mut self) {
        while self.bit_count != 0 {
            self.write_bit(false);
        }
    }

    /// Append a whole, already byte-aligned byte. Requires byte alignment;
    /// callers ensure this by aligning first.
    pub fn write_byte_aligned(&mut self, byte: u8) {
        debug_assert!(
            self.is_byte_aligned(),
            "write_byte_aligned requires alignment"
        );
        self.bytes.push(byte);
    }

    /// Emit a 32-bit start code `00 00 01 <code>` on a byte boundary, aligning
    /// first with zero stuffing-bits if necessary.
    pub fn write_start_code(&mut self, code: u8) {
        self.align_to_byte_with_zeros();
        self.bytes.push(0x00);
        self.bytes.push(0x00);
        self.bytes.push(0x01);
        self.bytes.push(code);
    }

    /// Consume the writer, returning the byte buffer. Any partial final byte is
    /// flushed with trailing zero bits so no bits are lost.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.align_to_byte_with_zeros();
        self.bytes
    }

    /// Borrow the completed bytes without consuming; only valid on a byte
    /// boundary (otherwise the partial byte is not yet visible).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpeg2::bitreader::BitReader;

    #[test]
    fn single_bits_round_trip_msb_first() {
        let mut w = BitWriter::new();
        let pattern = [true, false, true, true, false, true, false, false];
        for &b in &pattern {
            w.write_bit(b);
        }
        let bytes = w.into_bytes();
        // 0b1011_0100 = 0xB4.
        assert_eq!(bytes, vec![0xB4]);

        let mut r = BitReader::new(&bytes);
        for (i, &exp) in pattern.iter().enumerate() {
            assert_eq!(r.read_bit().expect("bit"), exp, "bit {i}");
        }
    }

    #[test]
    fn write_bits_matches_read_bits() {
        let mut w = BitWriter::new();
        w.write_bits(0xABC, 12);
        w.write_bits(0xD, 4);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0xAB, 0xCD]);

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(12).expect("12"), 0xABC);
        assert_eq!(r.read_bits(4).expect("4"), 0xD);
    }

    #[test]
    fn align_pads_with_zeros() {
        let mut w = BitWriter::new();
        w.write_bits(0b111, 3);
        w.align_to_byte_with_zeros();
        assert!(w.is_byte_aligned());
        let bytes = w.into_bytes();
        // 0b1110_0000 = 0xE0.
        assert_eq!(bytes, vec![0xE0]);
    }

    #[test]
    fn start_code_is_byte_aligned() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3); // unaligned content
        w.write_start_code(0xB3);
        let bytes = w.into_bytes();
        // 3 content bits padded to 0xA0, then 00 00 01 B3.
        assert_eq!(bytes, vec![0xA0, 0x00, 0x00, 0x01, 0xB3]);
    }

    #[test]
    fn into_bytes_flushes_partial() {
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        let bytes = w.into_bytes();
        assert_eq!(bytes, vec![0x80]);
    }

    #[test]
    fn bit_len_tracks_partial() {
        let mut w = BitWriter::new();
        w.write_bits(0, 10);
        assert_eq!(w.bit_len(), 10);
        assert!(!w.is_byte_aligned());
    }

    #[test]
    fn write_byte_aligned_appends() {
        let mut w = BitWriter::new();
        w.write_byte_aligned(0x12);
        w.write_byte_aligned(0x34);
        assert_eq!(w.as_bytes(), &[0x12, 0x34]);
    }

    #[test]
    fn large_field_round_trips() {
        let mut w = BitWriter::new();
        w.write_bits(0xDEAD_BEEF, 32);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(32).expect("32"), 0xDEAD_BEEF);
    }
}
