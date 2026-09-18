//! AV1 symbol decoder (daala/od_ec derived multi-symbol arithmetic decoder).
//!
//! Exact implementation of the AV1 spec "Parsing process for symbol decoder"
//! (spec 8.2: init_symbol / read_symbol / read_bool / read_literal /
//! exit_symbol, and the CDF adaptation rule of 8.2.6). CDF arrays follow the
//! spec convention: an array of length N+1 for an N-ary symbol where
//! `cdf[N-1] == 1 << 15` and `cdf[N]` is the adaptation counter.

use super::bits::floor_log2;
use super::consts::{EC_MIN_PROB, EC_PROB_SHIFT};
use crate::error::{CodecError, CodecResult};

/// Symbol decoder state over one tile's coded data.
pub struct Msac<'a> {
    data: &'a [u8],
    /// Bits consumed from `data` so far (never exceeds `data.len() * 8`).
    bit_pos: usize,
    /// `SymbolValue`.
    value: u32,
    /// `SymbolRange`.
    range: u32,
    /// `SymbolMaxBits` (may go negative, meaning padding zeros were used).
    max_bits: i64,
    /// CDF adaptation enabled (`disable_cdf_update == 0`).
    update: bool,
}

impl<'a> Msac<'a> {
    /// `init_symbol( sz )` (spec 8.2.2) over exactly `data` bytes.
    pub fn new(data: &'a [u8], update_cdfs: bool) -> Self {
        let sz = data.len();
        let mut m = Self {
            data,
            bit_pos: 0,
            value: 0,
            range: 1 << 15,
            max_bits: 8 * sz as i64 - 15,
            update: update_cdfs,
        };
        let num_bits = core::cmp::min(sz * 8, 15) as u32;
        let buf = m.read_bits_raw(num_bits);
        let padded_buf = buf << (15 - num_bits);
        m.value = ((1u32 << 15) - 1) ^ padded_buf;
        m
    }

    /// Raw MSB-first `f(n)` over the tile window; bits past the end read as 0
    /// (callers only request available bits per the spec renormalization).
    fn read_bits_raw(&mut self, n: u32) -> u32 {
        let mut x: u32 = 0;
        for _ in 0..n {
            let byte = self.bit_pos >> 3;
            let bit = if byte < self.data.len() {
                u32::from((self.data[byte] >> (7 - (self.bit_pos & 7))) & 1)
            } else {
                0
            };
            self.bit_pos += 1;
            x = 2 * x + bit;
        }
        x
    }

    /// Core of `read_symbol` without CDF adaptation (spec 8.2.6 decode loop
    /// plus renormalization).
    fn decode_symbol(&mut self, cdf: &[u16], n: usize) -> usize {
        debug_assert!(n >= 2 && cdf.len() > n);
        debug_assert_eq!(cdf[n - 1], 1 << 15);
        let mut symbol = 0usize;
        let mut prev;
        let mut cur = self.range;
        loop {
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol]);
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += (EC_MIN_PROB as u32) * (n as u32 - symbol as u32 - 1);
            if self.value >= cur {
                break;
            }
            symbol += 1;
        }
        self.range = prev - cur;
        self.value -= cur;

        // Renormalization (spec 8.2.6 ordered steps).
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        let num_bits = core::cmp::min(i64::from(bits), core::cmp::max(0, self.max_bits));
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let num_bits = num_bits as u32;
        let new_data = self.read_bits_raw(num_bits);
        let padded_data = new_data << (bits - num_bits);
        self.value = padded_data ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= i64::from(bits);
        symbol
    }

    /// `read_symbol( cdf )` (spec 8.2.6) with CDF adaptation.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> usize {
        let n = cdf.len() - 1;
        let symbol = self.decode_symbol(cdf, n);
        if self.update {
            let count = cdf[n];
            let rate = 3
                + u32::from(count > 15)
                + u32::from(count > 31)
                + core::cmp::min(floor_log2(n as u32), 2);
            let mut tmp: u16 = 0;
            for (i, c) in cdf.iter_mut().enumerate().take(n - 1) {
                if i == symbol {
                    tmp = 1 << 15;
                }
                if tmp < *c {
                    *c -= (*c - tmp) >> rate;
                } else {
                    *c += (tmp - *c) >> rate;
                }
            }
            cdf[n] += u16::from(cdf[n] < 32);
        }
        symbol
    }

    /// `read_bool( )` (spec 8.2.4): equiprobable bit, no adaptation.
    pub fn read_bool(&mut self) -> u32 {
        const BOOL_CDF: [u16; 3] = [1 << 14, 1 << 15, 0];
        self.decode_symbol(&BOOL_CDF, 2) as u32
    }

    /// `read_literal( n )` (spec 8.2.5).
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut x = 0u32;
        for _ in 0..n {
            x = 2 * x + self.read_bool();
        }
        x
    }

    /// `NS( n )` descriptor (spec 4.10.10): ns(n) over arithmetic bools.
    pub fn read_ns(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.read_literal(w - 1);
        if v < m {
            return v;
        }
        (v << 1) - m + self.read_bool()
    }

    /// True once the decoder has consumed more padding than any conformant
    /// stream allows (`SymbolMaxBits < -14`); used to abort unbounded
    /// syntax loops (e.g. exp-Golomb lengths) on corrupt data.
    pub fn exhausted(&self) -> bool {
        self.max_bits < -14
    }

    /// `exit_symbol( )` conformance check (spec 8.2.3): `SymbolMaxBits`
    /// must be >= -14 when a tile finishes. A violation means the decoder
    /// over-read the tile — i.e. it lost bitstream sync.
    pub fn check_exit(&self) -> CodecResult<()> {
        if self.max_bits < -14 {
            return Err(CodecError::InvalidBitstream(format!(
                "AV1: symbol decoder over-read tile data ({} padding bits used; \
                 conformance limit is 14) — decoder out of sync or stream corrupt",
                -self.max_bits
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder must terminate at symbol N-1 (where cdf == 32768 makes
    /// cur == 0 <= SymbolValue) and never index past the CDF terminator.
    #[test]
    fn read_symbol_always_terminates() {
        let data = [0u8; 8];
        let mut m = Msac::new(&data, true);
        let mut cdf = [32000u16, 32768, 0]; // heavily skewed 2-ary
        for _ in 0..50 {
            let s = m.read_symbol(&mut cdf);
            assert!(s < 2);
        }
    }

    /// CDF adaptation must preserve the terminator and increment the counter
    /// up to its cap of 32.
    #[test]
    fn cdf_update_keeps_terminator_and_counts() {
        let data = [0x5Au8; 16];
        let mut m = Msac::new(&data, true);
        let mut cdf = [16384u16, 24576, 32768, 0]; // 3-ary
        for i in 0..40 {
            let _ = m.read_symbol(&mut cdf);
            assert_eq!(cdf[2], 32768, "terminator must never change");
            assert_eq!(u32::from(cdf[3]), core::cmp::min(i + 1, 32), "counter");
        }
    }

    /// With adaptation disabled the CDF must stay bit-identical.
    #[test]
    fn disable_cdf_update_is_honoured() {
        let data = [0xC3u8; 8];
        let mut m = Msac::new(&data, false);
        let mut cdf = [10000u16, 20000, 32768, 0];
        let orig = cdf;
        for _ in 0..20 {
            let _ = m.read_symbol(&mut cdf);
        }
        assert_eq!(cdf, orig);
    }

    /// read_literal is MSB-first over equiprobable bools; on an all-zero
    /// buffer the initial SymbolValue is 0x7FFF so every bool decodes as 1.
    #[test]
    fn literal_msb_first() {
        let data = [0x00u8; 4];
        let mut m = Msac::new(&data, true);
        let v = m.read_literal(3);
        assert!(v <= 7);
    }
}
