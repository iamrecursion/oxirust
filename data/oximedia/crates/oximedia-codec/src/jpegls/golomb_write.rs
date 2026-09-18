//! Golomb-Rice bit writer for the JPEG-LS encoder (ISO 14495-1 §6.3, Annex A.3).
//!
//! This module is the forward (encode) counterpart of [`super::golomb`]'s
//! [`BitReader`](super::golomb::BitReader). It emits scan data MSB-first with
//! JPEG byte-stuffing: whenever a `0xFF` data byte is written, a `0x00` byte is
//! appended so that the reader can distinguish literal `0xFF` data from the
//! start of a marker.
//!
//! [`encode_golomb_unsigned_limited`] is the exact inverse of the decoder's
//! [`decode_golomb_unsigned_limited`](super::golomb::decode_golomb_unsigned_limited):
//! a value encoded here decodes back to the identical value, including the
//! LIMIT overflow escape used for residuals too large for the normal unary code.

/// MSB-first bit writer for JPEG-LS scan data with JPEG byte-stuffing.
///
/// Bits are accumulated into an 8-bit buffer most-significant-bit first. When a
/// full byte is emitted and that byte equals `0xFF`, a stuffed `0x00` byte is
/// written immediately after it, exactly mirroring the un-stuffing performed by
/// [`super::golomb::BitReader`].
pub struct BitWriter {
    data: Vec<u8>,
    bit_buf: u8,
    bits_in_buf: u8,
}

impl BitWriter {
    /// Create a new, empty `BitWriter`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Create a new `BitWriter` with pre-reserved output capacity (in bytes).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Number of fully-emitted bytes so far (excludes the partial bit buffer).
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Emit one bit (LSB of `bit` is used).
    ///
    /// When a complete byte accumulates it is pushed to the output, and a
    /// stuffed `0x00` is appended after any `0xFF` data byte.
    #[inline]
    pub fn write_bit(&mut self, bit: u8) {
        self.bit_buf = (self.bit_buf << 1) | (bit & 1);
        self.bits_in_buf += 1;
        if self.bits_in_buf == 8 {
            let byte = self.bit_buf;
            self.data.push(byte);
            if byte == 0xFF {
                // JPEG byte-stuffing: a literal 0xFF is followed by 0x00.
                self.data.push(0x00);
            }
            self.bit_buf = 0;
            self.bits_in_buf = 0;
        }
    }

    /// Emit the low `n` bits of `value`, most-significant first.
    ///
    /// `n` must be in `0..=32`; writing zero bits is a no-op.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    /// Flush any buffered bits (zero-padding the final byte) and return the
    /// accumulated byte buffer.
    ///
    /// The padding bits are zero, which is what the decoder's reader expects
    /// (trailing zero bits after the last coded symbol are simply never read).
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_buf > 0 {
            let byte = self.bit_buf << (8 - self.bits_in_buf);
            self.data.push(byte);
            if byte == 0xFF {
                self.data.push(0x00);
            }
            self.bit_buf = 0;
            self.bits_in_buf = 0;
        }
        self.data
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Golomb-Rice encode one unsigned residual `value` with order `k`.
///
/// This is the exact inverse of
/// [`decode_golomb_unsigned_limited`](super::golomb::decode_golomb_unsigned_limited).
///
/// Let `overflow_threshold = limit - k - 1` and `unary = value >> k`:
///
/// - **Normal** (`unary < overflow_threshold`): emit `unary` zero bits, a `1`
///   terminator, then the `k`-bit suffix `value & ((1 << k) - 1)`.
/// - **Overflow** (`unary >= overflow_threshold`): emit `overflow_threshold`
///   zero bits, a `1` terminator, then `(value + 1)` encoded in `qbpp` bits.
///
/// The decoder recognises the overflow escape precisely when exactly
/// `overflow_threshold` zero bits precede the terminator, so the `>=` test here
/// guarantees a unique, round-trip-safe encoding.
pub fn encode_golomb_unsigned_limited(
    writer: &mut BitWriter,
    value: i32,
    k: i32,
    limit: i32,
    qbpp: u8,
) {
    let overflow_threshold = limit - k - 1;
    let unary = value >> k;
    if unary >= overflow_threshold {
        // Overflow escape: threshold zeros, a 1, then (value + 1) in qbpp bits.
        for _ in 0..overflow_threshold {
            writer.write_bit(0);
        }
        writer.write_bit(1);
        writer.write_bits((value + 1) as u32, qbpp);
    } else {
        // Normal: `unary` zeros, a 1, then the k-bit suffix.
        for _ in 0..unary {
            writer.write_bit(0);
        }
        writer.write_bit(1);
        let suffix = if k > 0 { value & ((1 << k) - 1) } else { 0 };
        writer.write_bits(suffix as u32, k as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpegls::golomb::{
        compute_limit, compute_qbpp, decode_golomb_unsigned_limited, BitReader,
    };

    #[test]
    fn byte_stuffing_inserts_zero_after_ff() {
        // Writing eight 1-bits produces a 0xFF data byte, which must be stuffed.
        let mut w = BitWriter::new();
        w.write_bits(0xFF, 8);
        let out = w.finish();
        assert_eq!(
            out,
            vec![0xFF, 0x00],
            "0xFF data byte must be followed by 0x00"
        );
    }

    #[test]
    fn flush_pads_final_byte_with_zeros() {
        // Three bits 1,0,1 → buffer 0b101, flushed → 0b1010_0000 = 0xA0.
        let mut w = BitWriter::new();
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        let out = w.finish();
        assert_eq!(out, vec![0xA0]);
    }

    #[test]
    fn writer_reader_bit_roundtrip() {
        let mut w = BitWriter::new();
        let pattern = [1u8, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1];
        for &b in &pattern {
            w.write_bit(b);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for (i, &expected) in pattern.iter().enumerate() {
            let got = r.read_bit().expect("bit should be readable");
            assert_eq!(got as u8, expected, "bit {i} mismatch");
        }
    }

    #[test]
    fn golomb_normal_roundtrip_small_values() {
        let max_val = 255i32;
        let limit = compute_limit(max_val);
        let qbpp = compute_qbpp(max_val);
        for k in 0..=8i32 {
            for value in 0..=64i32 {
                let mut w = BitWriter::new();
                encode_golomb_unsigned_limited(&mut w, value, k, limit, qbpp);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                let decoded = decode_golomb_unsigned_limited(&mut r, k, limit, qbpp)
                    .expect("golomb value should decode");
                assert_eq!(decoded, value, "k={k} value={value} round-trip failed");
            }
        }
    }

    #[test]
    fn golomb_overflow_roundtrip_large_values() {
        // Force the overflow escape with large residuals across several k values.
        let max_val = 255i32;
        let limit = compute_limit(max_val);
        let qbpp = compute_qbpp(max_val);
        for k in 0..=4i32 {
            for value in [200i32, 255, 300, 400, 510] {
                let mut w = BitWriter::new();
                encode_golomb_unsigned_limited(&mut w, value, k, limit, qbpp);
                let bytes = w.finish();
                let mut r = BitReader::new(&bytes);
                let decoded = decode_golomb_unsigned_limited(&mut r, k, limit, qbpp)
                    .expect("overflow golomb value should decode");
                assert_eq!(
                    decoded, value,
                    "overflow k={k} value={value} round-trip failed"
                );
            }
        }
    }

    #[test]
    fn golomb_roundtrip_through_stuffed_ff() {
        // A sequence whose coded bits cross 0xFF byte boundaries, exercising the
        // interaction of byte-stuffing with multi-symbol decode.
        let max_val = 255i32;
        let limit = compute_limit(max_val);
        let qbpp = compute_qbpp(max_val);
        let values = [510i32, 0, 1, 255, 7, 300, 2, 4, 510, 510];
        let k = 0;
        let mut w = BitWriter::new();
        for &v in &values {
            encode_golomb_unsigned_limited(&mut w, v, k, limit, qbpp);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for (i, &v) in values.iter().enumerate() {
            let decoded = decode_golomb_unsigned_limited(&mut r, k, limit, qbpp)
                .unwrap_or_else(|| panic!("value {i} should decode"));
            assert_eq!(decoded, v, "value {i} mismatch through stuffed stream");
        }
    }
}
