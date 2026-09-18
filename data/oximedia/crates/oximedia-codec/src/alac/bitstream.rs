//! MSB-first bit reader and writer for ALAC frames.
//!
//! ALAC packs bits most-significant-bit-first into a byte stream with **no**
//! byte-stuffing (unlike JPEG). The reader and writer here read/write up to 32
//! bits per call and track the exact bit position so the encoder and decoder
//! agree to the bit.

use super::{AlacError, AlacResult};

/// MSB-first bit reader over a borrowed byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Create a reader over `data`, starting at bit 0.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    /// Total number of bits available in the stream.
    #[must_use]
    pub fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Number of bits already consumed.
    #[must_use]
    pub fn position(&self) -> usize {
        self.bit_pos
    }

    /// Number of bits remaining.
    #[must_use]
    pub fn bits_left(&self) -> usize {
        self.total_bits().saturating_sub(self.bit_pos)
    }

    /// Read a single bit as a `bool`.
    pub fn read_bit(&mut self) -> AlacResult<bool> {
        if self.bit_pos >= self.total_bits() {
            return Err(AlacError::Truncated("read_bit past end of frame".into()));
        }
        let byte = self.data[self.bit_pos >> 3];
        let shift = 7 - (self.bit_pos & 7);
        self.bit_pos += 1;
        Ok((byte >> shift) & 1 != 0)
    }

    /// Read `count` bits (0..=32) as an unsigned value, MSB-first.
    pub fn read_bits(&mut self, count: u32) -> AlacResult<u32> {
        debug_assert!(count <= 32);
        if count == 0 {
            return Ok(0);
        }
        if self.bit_pos + count as usize > self.total_bits() {
            return Err(AlacError::Truncated(format!(
                "read_bits({count}) past end of frame"
            )));
        }
        let mut value: u32 = 0;
        let mut remaining = count;
        while remaining > 0 {
            let byte_index = self.bit_pos >> 3;
            let bit_in_byte = self.bit_pos & 7;
            let avail = 8 - bit_in_byte;
            let take = remaining.min(avail as u32);
            let byte = u32::from(self.data[byte_index]);
            // Extract `take` bits starting at `bit_in_byte` (from the MSB side).
            let shift = avail as u32 - take;
            let mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            let chunk = (byte >> shift) & mask;
            value = (value << take) | chunk;
            self.bit_pos += take as usize;
            remaining -= take;
        }
        Ok(value)
    }

    /// Read `count` bits and sign-extend from bit `count-1`.
    pub fn read_signed(&mut self, count: u32) -> AlacResult<i32> {
        let raw = self.read_bits(count)?;
        Ok(sign_extend(raw, count))
    }

    /// Count consecutive 1 bits (used by the modified-Rice unary prefix), then
    /// consume the terminating 0 bit. Returns the run length.
    pub fn read_unary(&mut self) -> AlacResult<u32> {
        let mut count = 0u32;
        loop {
            let bit = self.read_bit()?;
            if !bit {
                break;
            }
            count += 1;
            if count > (1u32 << 20) {
                return Err(AlacError::InvalidBitstream(
                    "unary run length exceeds sanity bound".into(),
                ));
            }
        }
        Ok(count)
    }

    /// Peek up to 24 bits without consuming them; missing bits past the end are
    /// returned as zeros. Used by the adaptive-Golomb decoder which reads a
    /// long unary prefix.
    pub fn peek_24(&self) -> u32 {
        let mut value = 0u32;
        for i in 0..24 {
            let pos = self.bit_pos + i;
            let bit = if pos < self.total_bits() {
                let byte = self.data[pos >> 3];
                let shift = 7 - (pos & 7);
                u32::from((byte >> shift) & 1)
            } else {
                0
            };
            value = (value << 1) | bit;
        }
        value
    }

    /// Advance the cursor by `count` bits (used after [`BitReader::peek_24`]).
    pub fn skip(&mut self, count: usize) -> AlacResult<()> {
        if self.bit_pos + count > self.total_bits() {
            return Err(AlacError::Truncated("skip past end of frame".into()));
        }
        self.bit_pos += count;
        Ok(())
    }
}

/// MSB-first bit writer producing a byte vector.
#[derive(Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Accumulator for the partially-filled current byte.
    cur: u8,
    /// Number of bits filled in `cur` (0..8).
    fill: u8,
}

impl BitWriter {
    /// Create an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of bits written so far.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.fill as usize
    }

    /// Write a single bit.
    pub fn write_bit(&mut self, bit: bool) {
        self.cur = (self.cur << 1) | u8::from(bit);
        self.fill += 1;
        if self.fill == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.fill = 0;
        }
    }

    /// Write the low `count` bits (0..=32) of `value`, MSB-first.
    pub fn write_bits(&mut self, value: u32, count: u32) {
        debug_assert!(count <= 32);
        let mut remaining = count;
        while remaining > 0 {
            let bit_index = remaining - 1;
            let bit = (value >> bit_index) & 1 != 0;
            self.write_bit(bit);
            remaining -= 1;
        }
    }

    /// Write a `count`-bit two's-complement representation of `value`.
    pub fn write_signed(&mut self, value: i32, count: u32) {
        let mask = if count >= 32 {
            u32::MAX
        } else {
            (1u32 << count) - 1
        };
        self.write_bits((value as u32) & mask, count);
    }

    /// Write `n` one bits followed by a single zero bit (unary code).
    pub fn write_unary(&mut self, n: u32) {
        for _ in 0..n {
            self.write_bit(true);
        }
        self.write_bit(false);
    }

    /// Finish writing and return the byte buffer, zero-padding the final byte.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.fill > 0 {
            let padded = self.cur << (8 - self.fill);
            self.bytes.push(padded);
            self.cur = 0;
            self.fill = 0;
        }
        self.bytes
    }

    /// Finish and return both the zero-padded bytes and the exact bit length.
    #[must_use]
    pub fn finish_with_len(self) -> (Vec<u8>, usize) {
        let bit_len = self.bit_len();
        (self.finish(), bit_len)
    }

    /// Append the first `bit_len` bits of `bytes` (MSB-first) onto this writer.
    ///
    /// Used to splice a separately-built, possibly non-byte-aligned bit segment
    /// into a larger stream without losing alignment.
    pub fn append_bits(&mut self, bytes: &[u8], bit_len: usize) {
        for i in 0..bit_len {
            let byte = bytes[i >> 3];
            let bit = (byte >> (7 - (i & 7))) & 1 != 0;
            self.write_bit(bit);
        }
    }
}

/// Sign-extend the low `bits` of `value` to a full `i32`.
#[inline]
#[must_use]
pub fn sign_extend(value: u32, bits: u32) -> i32 {
    if bits == 0 || bits >= 32 {
        return value as i32;
    }
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_bits_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0xABCD, 16);
        w.write_bits(0, 5);
        w.write_bits(0x7FFF_FFFF, 31);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(16).unwrap(), 0xABCD);
        assert_eq!(r.read_bits(5).unwrap(), 0);
        assert_eq!(r.read_bits(31).unwrap(), 0x7FFF_FFFF);
    }

    #[test]
    fn test_signed_roundtrip() {
        let mut w = BitWriter::new();
        for &v in &[-1i32, -100, 0, 1, 100, -2048, 2047] {
            w.write_signed(v, 12);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &v in &[-1i32, -100, 0, 1, 100, -2048, 2047] {
            assert_eq!(r.read_signed(12).unwrap(), v);
        }
    }

    #[test]
    fn test_unary_roundtrip() {
        let mut w = BitWriter::new();
        for n in [0u32, 1, 5, 17, 33] {
            w.write_unary(n);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for n in [0u32, 1, 5, 17, 33] {
            assert_eq!(r.read_unary().unwrap(), n);
        }
    }

    #[test]
    fn test_read_past_end_errors() {
        let bytes = [0xFFu8];
        let mut r = BitReader::new(&bytes);
        assert!(r.read_bits(8).is_ok());
        assert!(r.read_bit().is_err());
        assert!(r.read_bits(1).is_err());
    }

    #[test]
    fn test_sign_extend() {
        assert_eq!(sign_extend(0b1111, 4), -1);
        assert_eq!(sign_extend(0b0111, 4), 7);
        assert_eq!(sign_extend(0b1000, 4), -8);
        assert_eq!(sign_extend(0, 16), 0);
    }

    #[test]
    fn test_peek_24() {
        let bytes = [0xAB, 0xCD, 0xEF];
        let r = BitReader::new(&bytes);
        assert_eq!(r.peek_24(), 0xABCDEF);
    }
}
