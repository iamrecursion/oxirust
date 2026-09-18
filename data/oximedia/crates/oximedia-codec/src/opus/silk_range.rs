//! Normative range decoder for Opus (RFC 6716 §4.1).
//!
//! This is the entropy decoder used by the SILK and hybrid Opus decoders. It
//! implements the range coder exactly as specified in RFC 6716 §4.1 (and the
//! BSD-licensed reference `entdec.c`), including the inverse-CDF (`icdf`)
//! symbol decode, single-bit decode with explicit probability, the raw-bit
//! reader (which consumes bytes from the *end* of the buffer), the uniformly
//! distributed integer decoder, and `ec_tell`.
//!
//! The decoder reads "range coder" bytes from the front of the buffer and "raw"
//! bytes from the back; the two streams meet in the middle. This dual-ended
//! layout is fundamental to Opus framing: it is what allows the SILK and CELT
//! layers of a hybrid packet to share one bitstream (RFC 6716 §3.1, §4.5).

use crate::{CodecError, CodecResult};

/// Number of bits kept in `rng`/`val`.
const EC_CODE_BITS: u32 = 32;
/// Number of bits consumed/produced per renormalisation step.
const EC_SYM_BITS: u32 = 8;
/// Mask of the low `EC_SYM_BITS` bits.
const EC_SYM_MASK: u32 = (1u32 << EC_SYM_BITS) - 1;
/// Highest bit of the code window.
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
/// Threshold below which renormalisation pulls in a new byte.
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
/// Extra precision bits carried beyond a whole symbol (`(32-2)%8+1 = 7`).
const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;
/// Largest `ft` decoded directly before `ec_dec_uint` splits into raw bits.
const EC_UINT_BITS: u32 = 8;

/// Normative Opus range decoder (RFC 6716 §4.1 `ec_dec`).
#[derive(Debug)]
pub struct SilkRangeDecoder<'a> {
    /// Input byte buffer (range bytes from front, raw bits from back).
    buf: &'a [u8],
    /// Index of the next range-coder byte to read from the front.
    front: usize,
    /// Index (exclusive) of the next raw byte to read from the back.
    back: usize,
    /// Raw-bit window buffered from the back of the buffer.
    end_window: u32,
    /// Number of valid bits in `end_window`.
    end_bits: u32,
    /// Number of range-coder bits consumed so far (`nbits_total`).
    nbits_total: i32,
    /// Current value within the coded range.
    val: u32,
    /// Size of the current range.
    rng: u32,
    /// Last byte fragment carried into renormalisation (`rem`).
    rem: i32,
    /// Scale factor saved between `ec_decode` and `ec_dec_update` (`ext`).
    ext: u32,
}

impl<'a> SilkRangeDecoder<'a> {
    /// Creates a new range decoder over `buf`, performing `ec_dec_init`.
    pub fn new(buf: &'a [u8]) -> CodecResult<Self> {
        if buf.is_empty() {
            return Err(CodecError::InvalidData(
                "Opus range decoder requires non-empty input".to_string(),
            ));
        }
        let mut dec = Self {
            buf,
            front: 0,
            back: buf.len(),
            end_window: 0,
            end_bits: 0,
            nbits_total: (EC_CODE_BITS - EC_SYM_BITS) as i32 + 1,
            val: 0,
            rng: 1u32 << EC_CODE_EXTRA,
            rem: 0,
            ext: 0,
        };
        dec.rem = i32::from(dec.read_byte());
        dec.val = dec
            .rng
            .wrapping_sub(1)
            .wrapping_sub((dec.rem as u32) >> (EC_SYM_BITS - EC_CODE_EXTRA));
        dec.normalize();
        Ok(dec)
    }

    /// Reads the next range-coder byte from the front of the buffer.
    fn read_byte(&mut self) -> u8 {
        if self.front < self.back {
            let b = self.buf[self.front];
            self.front += 1;
            b
        } else {
            0
        }
    }

    /// Reads the next raw byte from the back of the buffer.
    fn read_byte_from_end(&mut self) -> u8 {
        if self.back > self.front {
            self.back -= 1;
            self.buf[self.back]
        } else {
            0
        }
    }

    /// Renormalises the decoder (`ec_dec_normalize`).
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;
            let sym = self.rem;
            self.rem = i32::from(self.read_byte());
            let sym = ((sym << EC_SYM_BITS) | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val =
                ((self.val << EC_SYM_BITS) + (EC_SYM_MASK & !(sym as u32))) & (EC_CODE_TOP - 1);
        }
    }

    /// Returns `fs`, used to locate a symbol within total frequency `ft`
    /// (`ec_decode`).
    fn ec_decode(&mut self, ft: u32) -> u32 {
        self.ext = self.rng / ft;
        let s = self.val / self.ext;
        ft - (s + 1).min(ft)
    }

    /// Updates decoder state after the symbol with bounds `[fl, fh)` of total
    /// frequency `ft` has been identified (`ec_dec_update`).
    fn ec_dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext.wrapping_mul(ft - fh);
        self.val -= s;
        self.rng = if fl > 0 {
            self.ext.wrapping_mul(fh - fl)
        } else {
            self.rng - s
        };
        self.normalize();
    }

    /// Decodes one symbol from an inverse-CDF table (`ec_dec_icdf`).
    ///
    /// `icdf` holds `(1 << ftb) * (1 - CDF)` entries with `icdf[len-1] == 0`.
    /// Returns the decoded symbol index.
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> CodecResult<usize> {
        if icdf.is_empty() {
            return Err(CodecError::InvalidData("empty ICDF table".to_string()));
        }
        let scale = self.rng >> ftb;
        let mut s = self.rng;
        let mut t;
        let mut ret: i32 = -1;
        loop {
            t = s;
            ret += 1;
            let idx = ret as usize;
            if idx >= icdf.len() {
                // Reached the terminal 0 entry; clamp to the last symbol.
                ret = (icdf.len() - 1) as i32;
                s = 0;
                break;
            }
            s = scale * u32::from(icdf[idx]);
            if self.val >= s {
                break;
            }
        }
        self.val -= s;
        self.rng = t - s;
        self.normalize();
        Ok(ret as usize)
    }

    /// Decodes a single bit whose probability of being true is `1 / 2^logp`
    /// (`ec_dec_bit_logp`).
    pub fn decode_bit_logp(&mut self, logp: u32) -> CodecResult<bool> {
        let r = self.rng;
        let d = self.val;
        let s = r >> logp;
        let ret = d < s;
        if ret {
            self.rng = s;
        } else {
            self.val = d - s;
            self.rng = r - s;
        }
        self.normalize();
        Ok(ret)
    }

    /// Decodes a symbol from a cumulative-frequency table.
    ///
    /// `cdf[0]` must be 0 and `cdf[len-1]` is the total `ft`. Returns the index
    /// `k` with `cdf[k] <= fs < cdf[k+1]`.
    pub fn decode_cdf(&mut self, cdf: &[u16]) -> CodecResult<usize> {
        if cdf.len() < 2 {
            return Err(CodecError::InvalidData("CDF too short".to_string()));
        }
        let ft = u32::from(cdf[cdf.len() - 1]);
        if ft == 0 {
            return Err(CodecError::InvalidData("CDF total is zero".to_string()));
        }
        let fs = self.ec_decode(ft);
        let mut k = 0usize;
        while k + 1 < cdf.len() && u32::from(cdf[k + 1]) <= fs {
            k += 1;
        }
        let fl = u32::from(cdf[k]);
        let fh = u32::from(cdf[k + 1]);
        self.ec_dec_update(fl, fh, ft);
        Ok(k)
    }

    /// Decodes a uniformly distributed integer in `[0, ft)` (`ec_dec_uint`).
    pub fn decode_uint(&mut self, ft: u32) -> CodecResult<u32> {
        if ft <= 1 {
            return Ok(0);
        }
        let ft_minus_1 = ft - 1;
        let nbits = 32 - ft_minus_1.leading_zeros();
        if nbits > EC_UINT_BITS {
            let extra = nbits - EC_UINT_BITS;
            let top = (ft_minus_1 >> extra) + 1;
            let high = self.decode_uniform_symbol(top)?;
            let low = self.decode_raw_bits(extra)?;
            let t = (high << extra) | low;
            Ok(t.min(ft_minus_1))
        } else {
            let t = self.decode_uniform_symbol(ft)?;
            Ok(t.min(ft_minus_1))
        }
    }

    /// Decodes a uniform symbol in `[0, ft)` directly from the range coder.
    fn decode_uniform_symbol(&mut self, ft: u32) -> CodecResult<u32> {
        if ft == 0 {
            return Ok(0);
        }
        let fs = self.ec_decode(ft);
        let k = fs.min(ft - 1);
        self.ec_dec_update(k, k + 1, ft);
        Ok(k)
    }

    /// Reads `bits` raw bits from the back of the buffer (`ec_dec_bits`).
    pub fn decode_raw_bits(&mut self, bits: u32) -> CodecResult<u32> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(CodecError::InvalidData(
                "cannot decode more than 32 raw bits".to_string(),
            ));
        }
        while self.end_bits < bits {
            let byte = self.read_byte_from_end();
            self.end_window |= u32::from(byte) << self.end_bits;
            self.end_bits += EC_SYM_BITS;
        }
        let value = if bits == 32 {
            self.end_window
        } else {
            self.end_window & ((1u32 << bits) - 1)
        };
        self.end_window >>= bits;
        self.end_bits -= bits;
        Ok(value)
    }

    /// Returns the number of bits consumed so far, rounded up (`ec_tell`).
    pub fn tell(&self) -> i32 {
        self.nbits_total - (log2_floor(self.rng) as i32)
    }

    /// Number of raw-bit bytes consumed from the back.
    pub fn raw_bytes_consumed(&self) -> usize {
        self.buf.len() - self.back
    }

    /// Number of range-coder bytes consumed from the front.
    pub fn front_bytes_consumed(&self) -> usize {
        self.front
    }

    /// Total bytes in the underlying buffer.
    pub fn total_bytes(&self) -> usize {
        self.buf.len()
    }
}

/// Returns `floor(log2(x))` for `x > 0` (0 for `x == 0`, matching `EC_ILOG`).
fn log2_floor(x: u32) -> u32 {
    x.checked_ilog2().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_decoder_init() {
        let data = [0x80u8, 0x00, 0x00, 0x00];
        let dec = SilkRangeDecoder::new(&data);
        assert!(dec.is_ok());
    }

    #[test]
    fn test_range_decoder_empty() {
        let data: [u8; 0] = [];
        assert!(SilkRangeDecoder::new(&data).is_err());
    }

    #[test]
    fn test_decode_icdf_terminates() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a];
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        // 2-symbol distribution {1/2, 1/2}: icdf = [128, 0].
        let icdf = [128u8, 0];
        let sym = dec.decode_icdf(&icdf, 8).expect("icdf");
        assert!(sym < 2);
    }

    #[test]
    fn test_decode_bit_logp_finite() {
        let data = [0xFFu8, 0xFF, 0xFF, 0xFF];
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        for _ in 0..16 {
            let _ = dec.decode_bit_logp(1).expect("bit");
        }
    }

    #[test]
    fn test_decode_uint_in_range() {
        let data = [0xA5u8, 0x5A, 0xC3, 0x3C, 0x0F, 0xF0];
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        for ft in [2u32, 5, 17, 256, 1024, 65536] {
            let v = dec.decode_uint(ft).expect("uint");
            assert!(v < ft, "value {v} out of range for ft {ft}");
        }
    }

    #[test]
    fn test_decode_raw_bits() {
        // A 16-byte buffer leaves room so the front (range coder) and back
        // (raw bits) streams do not collide.
        let mut data = [0u8; 16];
        data[15] = 0xAB;
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        // Raw bits come from the back of the buffer, LSB-first per byte.
        let v = dec.decode_raw_bits(4).expect("raw");
        assert_eq!(v, 0x0B);
        let v2 = dec.decode_raw_bits(4).expect("raw");
        assert_eq!(v2, 0x0A);
    }

    #[test]
    fn test_tell_monotonic() {
        let data = [0x33u8; 16];
        let mut dec = SilkRangeDecoder::new(&data).expect("init");
        let mut last = dec.tell();
        for _ in 0..20 {
            let _ = dec.decode_bit_logp(2).expect("bit");
            let now = dec.tell();
            assert!(now >= last, "ec_tell must be monotonic");
            last = now;
        }
    }
}
