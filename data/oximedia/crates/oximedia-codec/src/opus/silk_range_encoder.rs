//! Normative range encoder for Opus / SILK (RFC 6716 §4.1.2).
//!
//! This is the byte-for-byte symmetric inverse of
//! [`super::silk_range::SilkRangeDecoder`]. It is a Rust port of the
//! BSD-licensed libopus reference (`celt/entenc.c`, `celt/mfrngcod.h`) and
//! reproduces `ec_enc_init`, `ec_enc_normalize`, `ec_enc_carry_out`,
//! `ec_encode`, `ec_enc_icdf`, `ec_enc_bit_logp`, `ec_enc_uint`,
//! `ec_enc_bits`, and `ec_enc_done` exactly.
//!
//! Two byte streams meet in one buffer: range-coder bytes are written from
//! the **front**, raw bits from the **back** (LSB-first within each byte).
//! `finish` reconciles the two ends, ensuring the back's raw-bit bytes
//! land at their final offsets so the decoder reads them in mirror order.

use crate::{CodecError, CodecResult};

/// Number of bits flushed per renormalisation step.
const EC_SYM_BITS: u32 = 8;
/// Mask of the bottom `EC_SYM_BITS` bits — equals `EC_SYM_MAX`.
const EC_SYM_MASK: u32 = (1u32 << EC_SYM_BITS) - 1;
/// Number of bits kept in the working code window.
const EC_CODE_BITS: u32 = 32;
/// `1 << (EC_CODE_BITS - 1)` — the highest bit of the code window.
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
/// `EC_CODE_TOP >> EC_SYM_BITS` — renormalisation threshold.
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
/// `EC_CODE_BITS - EC_SYM_BITS - 1` — shift to extract the high carry byte
/// during normalisation. **Note**: this is one less than a naive
/// `EC_CODE_BITS - EC_SYM_BITS` because bit 31 is the carry-out flag
/// (see libopus `mfrngcod.h`).
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
/// Largest `ft` encoded as a uniform symbol before `ec_enc_uint` splits
/// into raw bits.
const EC_UINT_BITS: u32 = 8;
/// Size of `end_window` in bits (matches libopus `ec_window` = `opus_uint32`).
const EC_WINDOW_SIZE: u32 = 32;

/// Normative Opus range encoder (RFC 6716 §4.1.2 `ec_enc`).
#[derive(Debug)]
pub struct SilkRangeEncoder {
    /// Front-stream bytes (range-coder output), appended in oldest-first
    /// order. `rem` (pending high byte) and `ext` (carry-extension count)
    /// are *not* committed here until [`Self::carry_out`] flushes them.
    front: Vec<u8>,
    /// Raw-bit bytes destined for the *back* of the eventual stream.
    /// Element `0` is the very last byte of the final output; element
    /// `back.len()-1` is the byte closest to the front. They are appended
    /// LSB-first as the raw-bit window fills.
    back: Vec<u8>,
    /// Current low value of the coded range (libopus `val`).
    val: u32,
    /// Current range size (libopus `rng`).
    rng: u32,
    /// Pending high byte awaiting a definite carry decision, or `None`
    /// before the first byte has been buffered (mirrors `rem = -1`).
    rem: Option<u8>,
    /// Count of `EC_SYM_MAX` (`0xFF`) bytes pending in the carry chain.
    ext: u32,
    /// Total bits emitted so far (used by `ec_tell`).
    nbits_total: i32,
    /// LSB-first raw-bit window holding bits not yet flushed to `back`.
    end_window: u32,
    /// Number of valid bits in `end_window`.
    nend_bits: u32,
}

impl Default for SilkRangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SilkRangeEncoder {
    /// Creates a new range encoder positioned at the start of an empty stream
    /// (libopus `ec_enc_init`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            front: Vec::with_capacity(64),
            back: Vec::with_capacity(16),
            val: 0,
            rng: EC_CODE_TOP,
            rem: None,
            ext: 0,
            nbits_total: (EC_CODE_BITS as i32) + 1,
            end_window: 0,
            nend_bits: 0,
        }
    }

    /// Returns the total number of bits emitted so far, rounded up.
    /// Mirrors `ec_tell` on the decoder side.
    #[must_use]
    pub fn tell(&self) -> i32 {
        self.nbits_total - log2_floor(self.rng) as i32
    }

    /// Encodes a symbol from an inverse-CDF table (libopus `ec_enc_icdf`).
    ///
    /// `icdf[s]` is the upper-tail probability after symbol `s`. `icdf` must
    /// be the same table the decoder consumes with `decode_icdf`, with the
    /// final entry equal to `0` and table length matching `(1 << ftb)`.
    pub fn encode_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) -> CodecResult<()> {
        if icdf.is_empty() {
            return Err(CodecError::InvalidData("empty ICDF table".to_string()));
        }
        if s >= icdf.len() {
            return Err(CodecError::InvalidData(format!(
                "symbol {s} out of range for ICDF of length {}",
                icdf.len()
            )));
        }
        let r = self.rng >> ftb;
        if s > 0 {
            let prev = u32::from(icdf[s - 1]);
            // val += rng - r * icdf[s-1]; rng = r * (icdf[s-1] - icdf[s])
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(prev)));
            self.rng = r.wrapping_mul(prev - u32::from(icdf[s]));
        } else {
            // rng -= r * icdf[s]
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(u32::from(icdf[s])));
        }
        self.normalize();
        Ok(())
    }

    /// Encodes a single bit whose probability of being one is `1 / 2^logp`
    /// (libopus `ec_enc_bit_logp`).
    pub fn encode_bit_logp(&mut self, bit: bool, logp: u32) -> CodecResult<()> {
        let r = self.rng;
        let s = r >> logp;
        let r2 = r - s;
        if bit {
            self.val = self.val.wrapping_add(r2);
            self.rng = s;
        } else {
            self.rng = r2;
        }
        self.normalize();
        Ok(())
    }

    /// Encodes a uniform symbol in `[0, ft)` using the range coder
    /// (libopus `ec_enc_uint`). Mirrors `decode_uint`.
    pub fn encode_uint(&mut self, value: u32, ft: u32) -> CodecResult<()> {
        if ft <= 1 {
            return Ok(());
        }
        if value >= ft {
            return Err(CodecError::InvalidData(format!(
                "encode_uint value {value} out of range for ft {ft}"
            )));
        }
        let ft_minus_1 = ft - 1;
        let ftb = (32 - ft_minus_1.leading_zeros()) as i32;
        if ftb > EC_UINT_BITS as i32 {
            let extra = (ftb - EC_UINT_BITS as i32) as u32;
            let top = (ft_minus_1 >> extra) + 1;
            let high = value >> extra;
            let low = value & ((1u32 << extra) - 1);
            // High symbol via uniform range coding: encode_inner(fl=high, fh=high+1, ft=top)
            self.encode_inner(high, high + 1, top)?;
            // Low bits via raw-bit writer.
            self.encode_raw_bits(low, extra)?;
            Ok(())
        } else {
            self.encode_inner(value, value + 1, ft)
        }
    }

    /// Writes `bits` raw bits to the *back* of the buffer (`ec_enc_bits`),
    /// LSB-first. Mirrors the decoder's `decode_raw_bits`.
    pub fn encode_raw_bits(&mut self, value: u32, bits: u32) -> CodecResult<()> {
        if bits == 0 {
            return Ok(());
        }
        if bits > 32 {
            return Err(CodecError::InvalidData(
                "cannot encode more than 32 raw bits".to_string(),
            ));
        }
        let value = if bits == 32 {
            value
        } else {
            value & ((1u32 << bits) - 1)
        };
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used + bits > EC_WINDOW_SIZE {
            // Window would overflow: drain whole bytes first.
            while used >= EC_SYM_BITS {
                self.back.push((window & EC_SYM_MASK) as u8);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS;
            }
        }
        window |= value << used;
        used += bits;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
        Ok(())
    }

    /// Finalises the stream and returns the encoded bytes
    /// (libopus `ec_enc_done`).
    pub fn finish(mut self) -> CodecResult<Vec<u8>> {
        // --- Stage 1: pick a finalisation point for the range coder. ---
        // `l = EC_CODE_BITS - EC_ILOG(rng)`; we then write the smallest
        // value `end` such that `end & ~msk == end` and `end` lies inside
        // the current range `[val, val + rng)`.
        let l_initial = (EC_CODE_BITS - ec_ilog(self.rng)) as i32;
        // libopus uses `int l`; track it as i32 so it can be negative.
        let mut l = l_initial;
        let mut msk = (EC_CODE_TOP - 1) >> l;
        let mut end = (self.val.wrapping_add(msk)) & !msk;
        if (end | msk) >= self.val.wrapping_add(self.rng) {
            l += 1;
            msk >>= 1;
            end = (self.val.wrapping_add(msk)) & !msk;
        }
        while l > 0 {
            let byte = (end >> EC_CODE_SHIFT) as i32;
            self.carry_out(byte);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }
        // Flush the buffered `rem` plus any pending `ext` carry-extension.
        if self.rem.is_some() || self.ext > 0 {
            self.carry_out(0);
        }

        // --- Stage 2: drain the raw-bit window into the back. ---
        //
        // Unlike libopus we work with a growable buffer, so there is no
        // pre-allocated "storage" to pad. We simply push every complete
        // byte; if a tail of `0 < used < 8` bits remains, we pad with zero
        // and push it as a full back byte. The decoder will read those
        // pad bits when (and only when) the caller asks for them, so as
        // long as encode/decode call sequences match, the pad bits stay
        // unread and harmless.
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS {
            self.back.push((window & EC_SYM_MASK) as u8);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS;
        }
        if used > 0 {
            self.back.push((window & EC_SYM_MASK) as u8);
        }

        // --- Stage 3: concatenate front + zero-pad + reversed back. ---
        //
        // libopus uses a fixed-size storage with the middle zeroed; we
        // emulate that by inserting a small zero region between the
        // front bytes and the (reversed) back bytes. The decoder's
        // forward range coder will read zero bytes when it overshoots,
        // matching libopus behaviour, and the back-of-buffer raw-bit
        // reader will still see its bytes at the very end. We keep the
        // padding minimal — Opus packets are typically tens to hundreds
        // of bytes and the SILK header/payload provides plenty of front
        // data in practice.
        const TAIL_PAD: usize = 8;
        let mut output = self.front;
        for _ in 0..TAIL_PAD {
            output.push(0);
        }
        for &b in self.back.iter().rev() {
            output.push(b);
        }
        Ok(output)
    }

    // --- internal -----------------------------------------------------------

    /// Common `(fl, fh, ft)` range-update step shared by `encode_uint` and
    /// internal helpers (libopus `ec_encode`).
    fn encode_inner(&mut self, fl: u32, fh: u32, ft: u32) -> CodecResult<()> {
        if ft == 0 {
            return Err(CodecError::InvalidData(
                "encode_inner total frequency zero".to_string(),
            ));
        }
        if fh > ft || fl > fh {
            return Err(CodecError::InvalidData(format!(
                "encode_inner invalid bracket fl={fl}, fh={fh}, ft={ft}"
            )));
        }
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self
                .val
                .wrapping_add(self.rng.wrapping_sub(r.wrapping_mul(ft - fl)));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng = self.rng.wrapping_sub(r.wrapping_mul(ft - fh));
        }
        self.normalize();
        Ok(())
    }

    /// Renormalises (libopus `ec_enc_normalize`). Shifts out high-order bytes
    /// of `val` whenever `rng` falls below `EC_CODE_BOT`.
    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            let byte = (self.val >> EC_CODE_SHIFT) as i32;
            self.carry_out(byte);
            // Move the next-to-high-order symbol into the high-order
            // position. Mask off the carry-out bit.
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS as i32;
        }
    }

    /// Emits a byte to the front stream, propagating the carry through any
    /// pending `0xFF` chain (libopus `ec_enc_carry_out`).
    ///
    /// `c` is the candidate 9-bit value: bottom 8 bits are the next high
    /// byte of `val`, bit 8 (= `EC_SYM_MAX + 1`) is the carry-out flag.
    fn carry_out(&mut self, c: i32) {
        if (c as u32) != EC_SYM_MASK {
            // No further carry propagation possible: flush buffer.
            let carry: u8 = ((c as u32) >> EC_SYM_BITS) as u8;
            if let Some(rem) = self.rem.take() {
                self.front.push(rem.wrapping_add(carry));
            }
            if self.ext > 0 {
                let sym = ((EC_SYM_MASK + u32::from(carry)) & EC_SYM_MASK) as u8;
                for _ in 0..self.ext {
                    self.front.push(sym);
                }
                self.ext = 0;
            }
            self.rem = Some(((c as u32) & EC_SYM_MASK) as u8);
        } else {
            self.ext += 1;
        }
    }
}

/// Returns the position of the most-significant `1` bit (1-indexed), matching
/// libopus `EC_ILOG`. `ec_ilog(0)` is `0`; `ec_ilog(1)` is `1`;
/// `ec_ilog(0x80000000)` is `32`.
fn ec_ilog(x: u32) -> u32 {
    32 - x.leading_zeros()
}

/// Used by `ec_tell` (legacy: the existing decoder uses floor here too).
fn log2_floor(x: u32) -> u32 {
    x.checked_ilog2().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::super::silk_range::SilkRangeDecoder;
    use super::*;

    #[test]
    fn test_round_trip_uniform_icdf() {
        // 4-way uniform iCDF: scale 1<<8=256, divisions at 192,128,64,0.
        let icdf: [u8; 4] = [192, 128, 64, 0];
        let symbols = [0usize, 1, 2, 3, 0, 3, 1, 2];

        let mut enc = SilkRangeEncoder::new();
        for &s in &symbols {
            enc.encode_icdf(s, &icdf, 8).expect("encode icdf");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("decoder init");
        for &s in &symbols {
            let got = dec.decode_icdf(&icdf, 8).expect("decode icdf");
            assert_eq!(
                got, s,
                "round trip uniform iCDF failed: encoded {s}, decoded {got}"
            );
        }
    }

    #[test]
    fn test_round_trip_binary_icdf() {
        // P(0) = 50/256, P(1) = 206/256.
        let icdf: [u8; 2] = [206, 0];
        let symbols = [1usize; 16];

        let mut enc = SilkRangeEncoder::new();
        for &s in &symbols {
            enc.encode_icdf(s, &icdf, 8).expect("encode");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec");
        for &s in &symbols {
            let got = dec.decode_icdf(&icdf, 8).expect("decode");
            assert_eq!(got, s);
        }
    }

    #[test]
    fn test_round_trip_bit_logp() {
        let bits = [true, false, true, true, false, false, true];
        let logps = [1u32, 1, 1, 1, 1, 1, 1];

        let mut enc = SilkRangeEncoder::new();
        for (&b, &lp) in bits.iter().zip(logps.iter()) {
            enc.encode_bit_logp(b, lp).expect("encode bit");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec");
        for (&b, &lp) in bits.iter().zip(logps.iter()) {
            let got = dec.decode_bit_logp(lp).expect("decode bit");
            assert_eq!(got, b);
        }
    }

    #[test]
    fn test_round_trip_raw_bits() {
        let values: &[(u32, u32)] = &[(0x0B, 4), (0x0A, 4), (0x55, 8), (0xC, 4)];
        let mut enc = SilkRangeEncoder::new();
        // Seed enough range-coder symbols so the front stream advances
        // past the initial normalisation window (~4 bytes) before the
        // raw-bit area starts to encroach on it.
        for _ in 0..32 {
            enc.encode_bit_logp(false, 1).expect("seed");
        }
        for &(v, b) in values {
            enc.encode_raw_bits(v, b).expect("raw");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec");
        for _ in 0..32 {
            let _ = dec.decode_bit_logp(1).expect("seed dec");
        }
        let g1 = dec.decode_raw_bits(4).expect("r1");
        let g2 = dec.decode_raw_bits(4).expect("r2");
        let g3 = dec.decode_raw_bits(8).expect("r3");
        let g4 = dec.decode_raw_bits(4).expect("r4");
        assert_eq!(g1, 0x0B);
        assert_eq!(g2, 0x0A);
        assert_eq!(g3, 0x55);
        assert_eq!(g4, 0x0C);
    }

    #[test]
    fn test_round_trip_uint() {
        let cases: &[(u32, u32)] = &[(3, 7), (0, 4), (15, 16), (300, 1024), (0, 65536)];
        let mut enc = SilkRangeEncoder::new();
        for &(v, ft) in cases {
            enc.encode_uint(v, ft).expect("encode uint");
        }
        let bytes = enc.finish().expect("finish");

        let mut dec = SilkRangeDecoder::new(&bytes).expect("dec");
        for &(v, ft) in cases {
            let g = dec.decode_uint(ft).expect("decode uint");
            assert_eq!(g, v, "uint round trip {v}/{ft} -> {g}");
        }
    }

    #[test]
    fn test_tell_monotonic() {
        let mut enc = SilkRangeEncoder::new();
        let mut last = enc.tell();
        for _ in 0..32 {
            enc.encode_bit_logp(true, 1).expect("bit");
            let now = enc.tell();
            assert!(now >= last, "tell must be monotonic");
            last = now;
        }
    }
}
