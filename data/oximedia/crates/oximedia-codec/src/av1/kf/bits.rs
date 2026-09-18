//! Spec-exact bitstream reader for AV1 header parsing.
//!
//! Implements the parsing descriptors of AV1 spec section 4.10 exactly:
//! `f(n)`, `uvlc()`, `le(n)`, `leb128()`, `su(n)`, `ns(n)`, plus
//! `trailing_bits`/`byte_alignment` helpers. Bit order is MSB-first within
//! each byte (spec 8.1 `read_bit`).

use crate::error::{CodecError, CodecResult};

/// MSB-first bit reader over a byte slice with exact position tracking.
pub struct BitRdr<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitRdr<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current bit position (spec `get_position()`).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Number of whole bytes consumed (position must be byte aligned).
    pub fn byte_offset(&self) -> usize {
        self.pos >> 3
    }

    fn read_bit(&mut self) -> CodecResult<u32> {
        let byte = self.pos >> 3;
        if byte >= self.data.len() {
            return Err(CodecError::InvalidBitstream(
                "AV1: bitstream over-read in header".into(),
            ));
        }
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(u32::from((self.data[byte] >> shift) & 1))
    }

    /// `f(n)` — reads n bits MSB first (spec 8.1). n <= 32.
    pub fn f(&mut self, n: u32) -> CodecResult<u32> {
        debug_assert!(n <= 32);
        let mut x: u64 = 0;
        for _ in 0..n {
            x = 2 * x + u64::from(self.read_bit()?);
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok(x as u32)
    }

    /// `f(1)` as bool.
    pub fn flag(&mut self) -> CodecResult<bool> {
        Ok(self.read_bit()? != 0)
    }

    /// `uvlc()` (spec 4.10.3).
    pub fn uvlc(&mut self) -> CodecResult<u64> {
        let mut leading_zeros = 0u32;
        loop {
            let done = self.read_bit()?;
            if done != 0 {
                break;
            }
            leading_zeros += 1;
        }
        if leading_zeros >= 32 {
            return Ok((1u64 << 32) - 1);
        }
        let value = u64::from(self.f(leading_zeros)?);
        Ok(value + (1u64 << leading_zeros) - 1)
    }

    /// `le(n)` — unsigned little-endian n-byte number (spec 4.10.4).
    pub fn le(&mut self, n: u32) -> CodecResult<u64> {
        debug_assert!(self.pos % 8 == 0, "le() requires byte alignment");
        let mut t: u64 = 0;
        for i in 0..n {
            let byte = u64::from(self.f(8)?);
            t |= byte << (8 * i);
        }
        Ok(t)
    }

    /// `leb128()` (spec 4.10.5). At most 8 bytes; value must fit in 32 bits.
    pub fn leb128(&mut self) -> CodecResult<u64> {
        let mut value: u64 = 0;
        for i in 0..8u32 {
            let leb128_byte = u64::from(self.f(8)?);
            value |= (leb128_byte & 0x7f) << (i * 7);
            if (leb128_byte & 0x80) == 0 {
                break;
            }
            if i == 7 {
                return Err(CodecError::InvalidBitstream(
                    "AV1: leb128 uses more than 8 bytes".into(),
                ));
            }
        }
        if value > (1u64 << 32) - 1 {
            return Err(CodecError::InvalidBitstream(
                "AV1: leb128 value exceeds 32 bits".into(),
            ));
        }
        Ok(value)
    }

    /// `su(n)` — signed integer from n bits (spec 4.10.6).
    pub fn su(&mut self, n: u32) -> CodecResult<i32> {
        let value = i64::from(self.f(n)?);
        let sign_mask = 1i64 << (n - 1);
        let v = if value & sign_mask != 0 {
            value - 2 * sign_mask
        } else {
            value
        };
        #[allow(clippy::cast_possible_truncation)]
        Ok(v as i32)
    }

    /// `ns(n)` — non-symmetric unsigned with max n values (spec 4.10.7).
    pub fn ns(&mut self, n: u32) -> CodecResult<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// `byte_alignment()` (spec 5.3.5): skip zero bits until byte aligned.
    pub fn byte_alignment(&mut self) -> CodecResult<()> {
        while self.pos & 7 != 0 {
            let _zero_bit = self.read_bit()?;
        }
        Ok(())
    }
}

/// `FloorLog2(x)` for x >= 1 (spec 4.7).
pub fn floor_log2(x: u32) -> u32 {
    debug_assert!(x >= 1);
    x.ilog2()
}

/// `tile_log2(blkSize, target)` (spec 5.9.16): smallest k such that
/// `blkSize << k >= target`.
pub fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (u64::from(blk_size) << k) < u64::from(target) {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_reads_msb_first() {
        let mut r = BitRdr::new(&[0b1010_1100, 0xFF]);
        assert_eq!(r.f(1).expect("bit"), 1);
        assert_eq!(r.f(3).expect("bits"), 0b010);
        assert_eq!(r.f(4).expect("bits"), 0b1100);
        assert_eq!(r.position(), 8);
    }

    #[test]
    fn su_sign_extends() {
        // su(1+6): value 7 bits; 0b1111111 = -1
        let mut r = BitRdr::new(&[0b1111_1110]);
        assert_eq!(r.su(7).expect("su"), -1);
    }

    #[test]
    fn ns_matches_spec_example() {
        // spec 4.10.7 example for n = 5: 0->00, 1->01, 2->10, 3->110, 4->111
        let cases: [(&[u8], u32); 5] = [
            (&[0b0000_0000], 0),
            (&[0b0100_0000], 1),
            (&[0b1000_0000], 2),
            (&[0b1100_0000], 3),
            (&[0b1110_0000], 4),
        ];
        for (bytes, want) in cases {
            let mut r = BitRdr::new(bytes);
            assert_eq!(r.ns(5).expect("ns"), want);
        }
    }

    #[test]
    fn leb128_two_bytes() {
        let mut r = BitRdr::new(&[0x80, 0x01]);
        assert_eq!(r.leb128().expect("leb"), 128);
    }
}
