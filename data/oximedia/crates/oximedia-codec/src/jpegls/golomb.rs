//! Golomb-Rice bit reader for JPEG-LS (ISO 14495-1 §6.3).
//!
//! JPEG-LS scan data is MSB-first with JPEG byte-stuffing: a `0xFF` byte
//! followed by `0x00` represents a literal `0xFF` in the bitstream. A `0xFF`
//! followed by any non-zero byte is a marker that signals the end of the scan.

/// MSB-first bit reader for JPEG-LS scan data with byte-stuffing support.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bits_in_buf: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over a scan-data byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Returns the current byte position (after all fully-consumed bytes).
    pub fn byte_pos(&self) -> usize {
        self.pos
    }

    /// Fill `bit_buf` with up to 32 bits from the underlying byte stream.
    fn refill(&mut self) {
        while self.bits_in_buf <= 24 && self.pos < self.data.len() {
            let byte = self.data[self.pos];
            self.pos += 1;

            // JPEG byte-stuffing: 0xFF followed by 0x00 → literal 0xFF byte.
            // 0xFF followed by any other value is a marker — the scan has ended.
            if byte == 0xFF {
                if self.pos < self.data.len() {
                    let next = self.data[self.pos];
                    if next == 0x00 {
                        // Stuffed zero: consume it, treat 0xFF as data.
                        self.pos += 1;
                    } else {
                        // Non-zero following byte is a marker: do not include this
                        // 0xFF in the bitstream, and do not advance past the marker.
                        self.pos -= 1; // un-consume the 0xFF
                        break;
                    }
                } else {
                    // 0xFF at end of data (no following byte) — stop here.
                    self.pos -= 1;
                    break;
                }
            }

            self.bit_buf = (self.bit_buf << 8) | u32::from(byte);
            self.bits_in_buf += 8;
        }
    }

    /// Read a single bit, returning `Some(0)` or `Some(1)`, or `None` on EOS.
    pub fn read_bit(&mut self) -> Option<u32> {
        if self.bits_in_buf == 0 {
            self.refill();
        }
        if self.bits_in_buf == 0 {
            return None;
        }
        self.bits_in_buf -= 1;
        Some((self.bit_buf >> self.bits_in_buf) & 1)
    }

    /// Read `n` bits (MSB first), returning the value as a `u32`.
    ///
    /// Returns `None` if there are fewer than `n` bits remaining.
    pub fn read_bits(&mut self, n: u8) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if self.bits_in_buf < n {
            self.refill();
        }
        if self.bits_in_buf < n {
            return None;
        }
        self.bits_in_buf -= n;
        Some((self.bit_buf >> self.bits_in_buf) & ((1u32 << n) - 1))
    }
}

/// Compute the JPEG-LS LIMIT parameter for a given `max_val`.
///
/// ISO 14495-1 §6.3: LIMIT = min(2*(J+8), 32+4*J) where J = floor(log2(MaxVal+1)).
/// J is the number of bits per sample (e.g., 8 for 8-bit images).
#[inline]
pub fn compute_limit(max_val: i32) -> i32 {
    // floor(log2(n)) = 31 - leading_zeros(n) for n > 0.
    let j = if max_val > 0 {
        31 - (max_val + 1).leading_zeros() as i32
    } else {
        0
    };
    let opt1 = 2 * (j + 8);
    let opt2 = 32 + j * 4;
    opt1.min(opt2)
}

/// Compute qbpp: bits needed to represent mapped error values.
///
/// qbpp = ceil(log2(MaxVal+2)) per ISO 14495-1 §6.3. For 8-bit images: qbpp = 9.
#[inline]
pub fn compute_qbpp(max_val: i32) -> u8 {
    // ceil(log2(n)) = 32 - (n-1).leading_zeros() for n > 1.
    let n = max_val + 2;
    if n <= 1 {
        return 1;
    }
    let bits = 32u32 - (n as u32 - 1).leading_zeros();
    (bits as u8).max(1)
}

/// Decode one Golomb-Rice unsigned residual with order `k`.
///
/// Encoding: unary prefix (run of 0-bits terminated by a 1-bit), followed by
/// `k` suffix bits. The unsigned value is `(unary_count << k) | suffix`.
///
/// When the unary count reaches `limit - k - 1`, the overflow encoding is used:
/// the remaining value is read directly in `qbpp` bits.
pub fn decode_golomb_unsigned(reader: &mut BitReader<'_>, k: i32) -> Option<i32> {
    // Use a generous limit here; callers use decode_golomb_unsigned_limited for proper bounds.
    decode_golomb_unsigned_limited(reader, k, 64, 16)
}

/// Decode Golomb-Rice with explicit LIMIT and qbpp parameters.
///
/// Both normal and overflow encodings per ISO 14495-1 §A.3/C.3:
///
/// - **Normal**: `[q zeros] [1] [k bits]` → decoded = `(q << k) | suffix`
/// - **Overflow** (q == limit-k-1): `[limit-k-1 zeros] [1] [qbpp bits]` → decoded = `raw - 1`
///
/// The overflow fires when exactly `limit - k - 1` zero bits precede the `1` terminator.
pub fn decode_golomb_unsigned_limited(
    reader: &mut BitReader<'_>,
    k: i32,
    limit: i32,
    qbpp: u8,
) -> Option<i32> {
    let overflow_threshold = limit - k - 1;
    let mut unary = 0i32;
    loop {
        let bit = reader.read_bit()?;
        if bit == 1 {
            // Terminator found — check for overflow AFTER consuming the 1.
            if unary == overflow_threshold {
                // Overflow: read qbpp bits as (MErrval + 1); return MErrval.
                let raw = reader.read_bits(qbpp)? as i32;
                return Some(raw - 1);
            }
            break;
        }
        unary += 1;
        // Guard: unary should never exceed limit (malformed stream).
        if unary > overflow_threshold {
            return None;
        }
    }
    let suffix = reader.read_bits(k as u8)? as i32;
    Some((unary << k) | suffix)
}

/// Map an unsigned Golomb residual to a signed error for the lossless case
/// (NEAR = 0).
///
/// ISO 14495-1 §A.3 specifies: even → positive half, odd → negative half.
/// For NEAR = 0 this simplifies to:
/// - `e_mapped` even  → `e_mapped / 2`
/// - `e_mapped` odd   → `-(e_mapped + 1) / 2`
#[inline]
pub fn unmap_error_lossless(e_mapped: i32) -> i32 {
    if e_mapped % 2 == 0 {
        e_mapped / 2
    } else {
        -(e_mapped + 1) / 2
    }
}

/// Map a signed error to an unsigned Golomb residual for the lossless encoder.
///
/// Inverse of [`unmap_error_lossless`]:
/// - `err >= 0` → `2 * err`
/// - `err < 0`  → `-2 * err - 1`
#[inline]
pub fn map_error_lossless(err: i32) -> i32 {
    if err >= 0 {
        2 * err
    } else {
        -2 * err - 1
    }
}

/// Map a signed quantised error to an unsigned Golomb residual for the near-lossless
/// case (NEAR > 0), per ISO 14495-1 §A.4.
///
/// In the near-lossless regime the error range in the sign-normalised domain is
/// `[-(RANGE/2), RANGE/2]` where `RANGE = (max_val + 2*near + 1)`.  The mapping
/// wraps negative values into the upper half of `[0, RANGE)`:
///
/// - `err >= 0`  →  `err`
/// - `err < 0`   →  `err + RANGE`
///
/// For NEAR = 0 this is equivalent to the lossless map applied to the non-negative
/// half only (the negative branch is never reached when quantisation step is 1).
#[inline]
pub fn map_error_near(err: i32, near: i32, max_val: i32) -> i32 {
    let range = max_val + 2 * near + 1;
    if err < 0 {
        err + range
    } else {
        err
    }
}

/// Unmap an unsigned Golomb residual to a signed quantised error for the
/// near-lossless case (NEAR > 0), per ISO 14495-1 §A.4.
///
/// Inverse of [`map_error_near`]: values above `RANGE/2` are mapped to the
/// corresponding negative signed error.
///
/// - `e_mapped <= RANGE/2`  →  `e_mapped`
/// - `e_mapped > RANGE/2`   →  `e_mapped - RANGE`
#[inline]
pub fn unmap_error_near(e_mapped: i32, near: i32, max_val: i32) -> i32 {
    let range = max_val + 2 * near + 1;
    let half_range = range / 2;
    if e_mapped > half_range {
        e_mapped - range
    } else {
        e_mapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golomb_k0_unary_three() {
        // k=0: pure unary; "0001" → value 3
        let data = [0b0001_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(decode_golomb_unsigned(&mut r, 0), Some(3));
    }

    #[test]
    fn golomb_k2_value_one() {
        // k=2: unary=0 (one '1' bit immediately), suffix = 0b01 = 1 → value = (0<<2)|1 = 1
        // Bit layout: 1 01 ..... → byte = 0b101_00000
        let data = [0b1010_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(decode_golomb_unsigned(&mut r, 2), Some(1));
    }

    #[test]
    fn golomb_k1_value_five() {
        // k=1: value 5 → unary = 5>>1 = 2, suffix = 5&1 = 1
        // Bits: 0 0 1 [suffix=1] = 0011
        let data = [0b0011_0000u8];
        let mut r = BitReader::new(&data);
        assert_eq!(decode_golomb_unsigned(&mut r, 1), Some(5));
    }

    #[test]
    fn unmap_roundtrip_lossless() {
        for err in -100..=100i32 {
            let mapped = map_error_lossless(err);
            assert_eq!(
                unmap_error_lossless(mapped),
                err,
                "roundtrip failed for err={err}"
            );
        }
    }

    #[test]
    fn byte_stuffing_skips_zero() {
        // 0xFF 0x00 should produce a literal 0xFF byte in the bitstream.
        // 8 bits of 0xFF followed by stuffed 0x00, then 0b10000000 (another bit).
        let data = [0xFF, 0x00, 0x80];
        let mut r = BitReader::new(&data);
        // Read the first 8 bits (all ones from 0xFF — the stuffed-zero is skipped)
        let val = r.read_bits(8).expect("should read 8 bits");
        assert_eq!(val, 0xFF);
        // The 0x00 stuffing byte is consumed; next bit comes from 0x80
        let bit = r.read_bit().expect("should read bit from 0x80");
        assert_eq!(bit, 1);
    }

    #[test]
    fn marker_bytes_stop_reading() {
        // 0xAA 0xFF 0xD9 (EOI marker) — only the 0xAA byte should be readable.
        // The 0xFF is the start of a marker and should NOT be included as data bits.
        let data = [0xAA, 0xFF, 0xD9];
        let mut r = BitReader::new(&data);
        let val = r.read_bits(8).expect("should read 0xAA");
        assert_eq!(val, 0xAA);
        // Now at 0xFF 0xD9 — should return None (no more data bits)
        let after = r.read_bit();
        assert!(
            after.is_none(),
            "should be None after marker, got {after:?}"
        );
    }
}
