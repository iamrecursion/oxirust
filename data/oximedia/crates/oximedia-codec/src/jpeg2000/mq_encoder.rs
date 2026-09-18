//! MQ arithmetic encoder (ISO/IEC 15444-1 Annex C).
//!
//! This is the exact forward (encoding) counterpart of the [`MqDecoder`] in
//! [`super::mq_coder`]. It uses the **same** 47-entry probability transition
//! table (`super::mq_coder::MQ_TABLE`) — there is exactly one table shared
//! between the two halves of the codec, so the Qe / NMPS / NLPS / SWITCH data
//! can never diverge.
//!
//! ## Algorithm
//!
//! The encoder implements the standard ENCODE / CODEMPS / CODELPS / RENORME /
//! BYTEOUT / FLUSH procedures of ISO/IEC 15444-1 Annex C (Figures C.4 – C.10),
//! the canonical pair of the DECODE procedure implemented by the decoder. Each
//! `encode_decision(cx, d)` narrows the current probability interval `[C, C+A)`
//! and the renormalisation/BYTEOUT machinery serialises the settled high bits of
//! `C` to bytes, propagating carries and inserting a stuffed zero bit after every
//! `0xFF` output byte (the rule the decoder's BYTEIN reverses).
//!
//! [`MqDecoder`]: super::mq_coder::MqDecoder

use super::mq_coder::{MqState, MQ_NUM_CONTEXTS, MQ_TABLE};

/// MQ arithmetic encoder state.
///
/// Uses the array-and-pointer byte model of the standard software encoder: the
/// output buffer holds a leading sentinel byte at index 0 and `bp` points at the
/// most-recently written byte (so a carry can still reach it before the next
/// byte is produced). The leading sentinel is stripped by [`MqEncoder::flush`].
pub struct MqEncoder {
    /// Interval register (mirrors the decoder's `A`).
    a: u32,
    /// Code register: running lower bound of the codeword.
    c: u32,
    /// Count of free bit positions before the next byte must be emitted.
    ct: u32,
    /// Output byte buffer (`buf[0]` is the leading sentinel).
    buf: Vec<u8>,
    /// Index of the most-recently written byte in `buf`.
    bp: usize,
    /// MPS (most probable symbol) for each context: 0 or 1.
    cx_mps: [u8; MQ_NUM_CONTEXTS],
    /// State index for each context (0..46).
    cx_state: [u8; MQ_NUM_CONTEXTS],
}

impl Default for MqEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MqEncoder {
    /// Create a new MQ encoder (INITENC, ISO/IEC 15444-1 Annex C Figure C.10).
    ///
    /// All contexts reset to state 0 / MPS 0, matching the decoder's initial
    /// per-context state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            a: 0x8000,
            c: 0,
            ct: 12,
            // buf[0] is the sentinel; `bp` starts at 0 pointing at it.
            buf: vec![0u8],
            bp: 0,
            cx_mps: [0u8; MQ_NUM_CONTEXTS],
            cx_state: [0u8; MQ_NUM_CONTEXTS],
        }
    }

    /// Number of bytes written after the sentinel (diagnostics only).
    #[must_use]
    pub fn settled_len(&self) -> usize {
        self.bp
    }

    /// Encode one binary decision `d` (0 or 1) under context `cx`.
    ///
    /// Returns `false` if `cx` is out of range; callers in this crate always
    /// pass valid indices so this never fires in practice.
    pub fn encode_decision(&mut self, cx: usize, d: u8) -> bool {
        if cx >= MQ_NUM_CONTEXTS {
            return false;
        }
        let entry: MqState = MQ_TABLE[usize::from(self.cx_state[cx])];
        let qe = u32::from(entry.qe);
        self.a = self.a.wrapping_sub(qe);

        if d == self.cx_mps[cx] {
            self.code_mps(cx, qe, &entry);
        } else {
            self.code_lps(cx, qe, &entry);
        }
        true
    }

    /// CODEMPS (ISO/IEC 15444-1 Annex C, Figure C.8).
    fn code_mps(&mut self, cx: usize, qe: u32, entry: &MqState) {
        if self.a & 0x8000 != 0 {
            // Interval still normalised: code MPS in the upper sub-interval,
            // no renormalisation, no state change.
            self.c = self.c.wrapping_add(qe);
            return;
        }
        if self.a >= qe {
            // Normal case: MPS in the upper sub-interval.
            self.c = self.c.wrapping_add(qe);
        } else {
            // Conditional exchange: MPS in the lower sub-interval.
            self.a = qe;
        }
        self.cx_state[cx] = entry.nmps;
        self.renorm_e();
    }

    /// CODELPS (ISO/IEC 15444-1 Annex C, Figure C.9).
    fn code_lps(&mut self, cx: usize, qe: u32, entry: &MqState) {
        if self.a >= qe {
            // Normal case: LPS in the lower sub-interval.
            self.a = qe;
        } else {
            // Conditional exchange: LPS in the upper sub-interval.
            self.c = self.c.wrapping_add(qe);
        }
        if entry.sw == 1 {
            self.cx_mps[cx] = 1 - self.cx_mps[cx];
        }
        self.cx_state[cx] = entry.nlps;
        self.renorm_e();
    }

    /// RENORME (ISO/IEC 15444-1 Annex C, Figure C.7).
    fn renorm_e(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    /// Append a fresh byte to the buffer and advance `bp` to it.
    #[inline]
    fn advance(&mut self, byte: u8) {
        self.buf.push(byte);
        self.bp += 1;
    }

    /// BYTEOUT (ISO/IEC 15444-1 Annex C, Figure C.6) — array-and-pointer form.
    ///
    /// Emits one settled byte, propagating a carry (bit 27) into the previously
    /// written byte, and inserting a stuffed zero bit after a `0xFF` byte (only
    /// 7 data bits follow a `0xFF`). Transcribed from the standard software
    /// encoder so it is the exact pair of the decoder's BYTEIN.
    fn byte_out(&mut self) {
        if self.buf[self.bp] == 0xFF {
            let byte = ((self.c >> 20) & 0xFF) as u8;
            self.advance(byte);
            self.c &= 0xF_FFFF;
            self.ct = 7;
        } else if self.c & 0x800_0000 != 0 {
            // Carry out of bit 27 propagates into the current byte.
            self.buf[self.bp] = self.buf[self.bp].wrapping_add(1);
            if self.buf[self.bp] == 0xFF {
                self.c &= 0x7FF_FFFF;
                let byte = ((self.c >> 20) & 0xFF) as u8;
                self.advance(byte);
                self.c &= 0xF_FFFF;
                self.ct = 7;
            } else {
                let byte = ((self.c >> 19) & 0xFF) as u8;
                self.advance(byte);
                self.c &= 0x7_FFFF;
                self.ct = 8;
            }
        } else {
            let byte = ((self.c >> 19) & 0xFF) as u8;
            self.advance(byte);
            self.c &= 0x7_FFFF;
            self.ct = 8;
        }
    }

    /// Flush the encoder and return the complete compressed byte stream
    /// (FLUSH / SETBITS, ISO/IEC 15444-1 Annex C, Figures C.11 / C.12).
    #[must_use]
    pub fn flush(mut self) -> Vec<u8> {
        // SETBITS: maximise the trailing bits so the truncated codeword still
        // lies inside the final interval.
        let tempc = self.c.wrapping_add(self.a);
        self.c |= 0xFFFF;
        if self.c >= tempc {
            self.c = self.c.wrapping_sub(0x8000);
        }

        // Two BYTEOUTs settle the remaining significant bits.
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= self.ct;
        self.byte_out();

        // The encoded codeword is buf[1..=bp] (index 0 is the sentinel). The
        // standard advances bp once more if the last byte is not 0xFF; that
        // trailing byte is the redundant terminator, so we simply return the
        // written bytes after the sentinel.
        let mut out = self.buf;
        out.remove(0);
        // `bp` no longer indexes `out` after the sentinel removal; the written
        // bytes are exactly the first `self.bp` elements.
        out.truncate(self.bp);
        if out.is_empty() {
            out.push(0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg2000::mq_coder::MqDecoder;

    /// Tiny deterministic LCG so the tests need no external RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
    }

    fn roundtrip(decisions: &[(usize, u8)]) {
        let mut enc = MqEncoder::new();
        for &(cx, d) in decisions {
            assert!(enc.encode_decision(cx, d), "encode failed for cx={cx}");
        }
        let bytes = enc.flush();

        let mut dec = MqDecoder::new(&bytes);
        for (i, &(cx, expected)) in decisions.iter().enumerate() {
            let got = dec.decode_bit(cx).expect("decode");
            assert_eq!(got, expected, "decision {i} (cx={cx}) mismatch");
        }
    }

    #[test]
    fn roundtrip_all_zero_single_ctx() {
        let decisions: Vec<(usize, u8)> = (0..100).map(|_| (0usize, 0u8)).collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_all_one_single_ctx() {
        let decisions: Vec<(usize, u8)> = (0..100).map(|_| (0usize, 1u8)).collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_alternating_single_ctx() {
        let decisions: Vec<(usize, u8)> = (0..200).map(|i| (0usize, (i % 2) as u8)).collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_random_single_ctx() {
        let mut rng = Lcg::new(0x1234_5678_9abc_def0);
        let decisions: Vec<(usize, u8)> = (0..1000)
            .map(|_| (0usize, (rng.next_u32() & 1) as u8))
            .collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_random_multi_ctx() {
        let mut rng = Lcg::new(0xdead_beef_cafe_babe);
        let decisions: Vec<(usize, u8)> = (0..2000)
            .map(|_| {
                let cx = (rng.next_u32() as usize) % MQ_NUM_CONTEXTS;
                let d = (rng.next_u32() & 1) as u8;
                (cx, d)
            })
            .collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_skewed_mostly_zero() {
        let mut rng = Lcg::new(0x0f0f_0f0f_0f0f_0f0f);
        let decisions: Vec<(usize, u8)> = (0..3000)
            .map(|_| {
                let d = if rng.next_u32() % 16 == 0 { 1u8 } else { 0u8 };
                (3usize, d)
            })
            .collect();
        roundtrip(&decisions);
    }

    #[test]
    fn roundtrip_short_streams() {
        for len in 1usize..40 {
            let decisions: Vec<(usize, u8)> =
                (0..len).map(|i| (0usize, (i % 3 == 0) as u8)).collect();
            roundtrip(&decisions);
        }
    }

    #[test]
    fn stress_many_seeds() {
        for seed in 0u64..1500 {
            let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
            let len = 1 + (rng.next_u32() as usize % 6000);
            let decisions: Vec<(usize, u8)> = (0..len)
                .map(|_| {
                    let cx = (rng.next_u32() as usize) % MQ_NUM_CONTEXTS;
                    let d = (rng.next_u32() & 1) as u8;
                    (cx, d)
                })
                .collect();

            let mut enc = MqEncoder::new();
            for &(cx, d) in &decisions {
                enc.encode_decision(cx, d);
            }
            let bytes = enc.flush();
            let nbytes = bytes.len();
            let mut dec = MqDecoder::new(&bytes);
            for (i, &(cx, expected)) in decisions.iter().enumerate() {
                let got = dec.decode_bit(cx).expect("decode");
                assert_eq!(
                    got, expected,
                    "seed={seed} len={len} divergence at {i} cx={cx} nbytes={nbytes}"
                );
            }
        }
    }
}
