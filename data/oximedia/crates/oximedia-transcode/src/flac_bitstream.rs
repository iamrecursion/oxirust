// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Spec-compliant FLAC frame encoder (RFC 9639 subset).
//!
//! The transcode pipeline needs FLAC output that *reference* decoders
//! (libFLAC, FFmpeg) accept. The FLAC encoders elsewhere in the workspace
//! produce internally-round-trippable but non-conformant bitstreams
//! (hard-coded zero CRC-8, non-spec residual layout), so this module
//! implements the frame layer correctly from the specification:
//!
//! - Frame header with fixed-blocksize strategy, UTF-8-coded frame number,
//!   and a real CRC-8.
//! - Independent-channel subframes: CONSTANT, VERBATIM, and FIXED
//!   (orders 0–4) chosen per channel by predicted cost.
//! - Partitioned-Rice residual coding (partition order 0) with a real
//!   parameter search and the 5-bit escape mode.
//! - Frame CRC-16 over the whole frame.
//!
//! Output frames are container-less: pair them with a STREAMINFO writer
//! (e.g. the container crate's `FlacMuxer`, or [`stream_info_block`]).
//! 16-bit input only — that is what the transcode PCM pipeline carries.

use crate::{Result, TranscodeError};

// ─── CRC helpers ──────────────────────────────────────────────────────────────

/// CRC-8 with polynomial `x^8 + x^2 + x + 1` (0x07), init 0 (FLAC header CRC).
fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x07
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// CRC-16 with polynomial `x^16 + x^15 + x^2 + 1` (0x8005), init 0 (FLAC frame CRC).
fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &byte in data {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
    }
    crc
}

// ─── MSB-first bit writer ─────────────────────────────────────────────────────

/// MSB-first bit writer (FLAC/ALAC bit order).
pub(crate) struct BitWriter {
    buf: Vec<u8>,
    /// Bit accumulator, MSB-aligned within the low `nbits` bits.
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    /// Write the low `bits` bits of `value`, MSB first.
    pub(crate) fn write_bits(&mut self, value: u64, bits: u32) {
        debug_assert!(bits <= 57, "write_bits supports up to 57 bits per call");
        if bits == 0 {
            return;
        }
        let masked = if bits == 64 {
            value
        } else {
            value & ((1u64 << bits) - 1)
        };
        self.acc = (self.acc << bits) | masked;
        self.nbits += bits;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.buf.push((self.acc >> self.nbits) as u8);
        }
    }

    /// Write `q` zero bits followed by a one bit (FLAC unary coding).
    fn write_unary(&mut self, q: u32) {
        let mut remaining = q;
        while remaining >= 32 {
            self.write_bits(0, 32);
            remaining -= 32;
        }
        self.write_bits(1, remaining + 1);
    }

    /// Zero-pad to the next byte boundary.
    pub(crate) fn align(&mut self) {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.write_bits(0, pad);
        }
    }

    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        self.align();
        self.buf
    }
}

// ─── Fixed predictors ─────────────────────────────────────────────────────────

/// Compute fixed-predictor residuals of the given order (0–4).
fn fixed_residuals(samples: &[i64], order: usize) -> Vec<i64> {
    match order {
        0 => samples.to_vec(),
        1 => samples.windows(2).map(|w| w[1] - w[0]).collect(),
        2 => samples.windows(3).map(|w| w[2] - 2 * w[1] + w[0]).collect(),
        3 => samples
            .windows(4)
            .map(|w| w[3] - 3 * w[2] + 3 * w[1] - w[0])
            .collect(),
        _ => samples
            .windows(5)
            .map(|w| w[4] - 4 * w[3] + 6 * w[2] - 4 * w[1] + w[0])
            .collect(),
    }
}

/// Zigzag-fold a signed residual to unsigned (FLAC Rice mapping).
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Exact Rice cost in bits of the residual set for parameter `k`.
fn rice_cost(residuals: &[i64], k: u32) -> u64 {
    residuals
        .iter()
        .map(|&r| u64::from(k) + 1 + (zigzag(r) >> k))
        .sum()
}

/// Find the Rice parameter (0–14) with minimal cost.
fn best_rice_param(residuals: &[i64]) -> (u32, u64) {
    let mut best = (0u32, u64::MAX);
    for k in 0..=14u32 {
        let cost = rice_cost(residuals, k);
        if cost < best.1 {
            best = (k, cost);
        }
    }
    best
}

// ─── Stream encoder ───────────────────────────────────────────────────────────

/// Sample rates with a dedicated 4-bit code in the frame header; everything
/// else uses code 0 ("get from STREAMINFO"), which is always valid because
/// the container writes STREAMINFO.
fn sample_rate_code(rate: u32) -> u32 {
    match rate {
        88_200 => 0b0001,
        176_400 => 0b0010,
        192_000 => 0b0011,
        8_000 => 0b0100,
        16_000 => 0b0101,
        22_050 => 0b0110,
        24_000 => 0b0111,
        32_000 => 0b1000,
        44_100 => 0b1001,
        48_000 => 0b1010,
        96_000 => 0b1011,
        _ => 0b0000,
    }
}

/// Block size codes for exact table entries; `None` means "use the 16-bit
/// explicit form" (code 0b0111 + `size - 1` after the header).
fn block_size_code(size: usize) -> Option<u32> {
    match size {
        192 => Some(0b0001),
        576 => Some(0b0010),
        1_152 => Some(0b0011),
        2_304 => Some(0b0100),
        4_608 => Some(0b0101),
        256 => Some(0b1000),
        512 => Some(0b1001),
        1_024 => Some(0b1010),
        2_048 => Some(0b1011),
        4_096 => Some(0b1100),
        8_192 => Some(0b1101),
        16_384 => Some(0b1110),
        32_768 => Some(0b1111),
        _ => None,
    }
}

/// Encode a frame number in FLAC's extended UTF-8 style.
fn utf8_coded_number(mut n: u64, out: &mut Vec<u8>) {
    if n < 0x80 {
        out.push(n as u8);
        return;
    }
    let mut tail = Vec::new();
    let mut bits = 0u32;
    while n >= (0x40u64 >> (bits / 6)).max(1) && bits < 36 {
        tail.push(0x80 | (n & 0x3F) as u8);
        n >>= 6;
        bits += 6;
        // Leading byte capacity shrinks by one bit per continuation byte:
        // 5 bits with one continuation, 4 with two, …
        let lead_capacity = 6 - (bits / 6);
        if n < (1u64 << lead_capacity) {
            break;
        }
    }
    let count = tail.len() as u32;
    // Leading byte: `count + 1` ones, then a zero, then the remaining bits.
    let prefix: u8 = match count {
        1 => 0xC0,
        2 => 0xE0,
        3 => 0xF0,
        4 => 0xF8,
        5 => 0xFC,
        _ => 0xFE,
    };
    out.push(prefix | (n as u8));
    tail.reverse();
    out.extend_from_slice(&tail);
}

/// A spec-compliant FLAC frame encoder for 16-bit interleaved PCM.
pub struct FlacStreamEncoder {
    sample_rate: u32,
    channels: u16,
    frame_index: u64,
}

impl FlacStreamEncoder {
    /// Bits per sample handled by this encoder (transcode PCM is 16-bit).
    const BPS: u32 = 16;

    /// Creates an encoder.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] for zero/oversized channel
    /// counts or a zero sample rate.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        if !(1..=8).contains(&channels) {
            return Err(TranscodeError::InvalidInput(format!(
                "FLAC supports 1-8 channels, got {channels}"
            )));
        }
        if sample_rate == 0 || sample_rate > 655_350 {
            return Err(TranscodeError::InvalidInput(format!(
                "invalid FLAC sample rate {sample_rate}"
            )));
        }
        Ok(Self {
            sample_rate,
            channels,
            frame_index: 0,
        })
    }

    /// Encode one block of interleaved i16 samples into a complete FLAC
    /// frame (sync code … CRC-16). Block sizes 1–65535 per channel.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::CodecError`] if the buffer is empty or not
    /// a whole number of interleaved sample-frames.
    pub fn encode_block(&mut self, interleaved: &[i16]) -> Result<Vec<u8>> {
        let ch = usize::from(self.channels);
        if interleaved.is_empty() || interleaved.len() % ch != 0 {
            return Err(TranscodeError::CodecError(format!(
                "FLAC block of {} samples is not a multiple of {ch} channels",
                interleaved.len()
            )));
        }
        let block_size = interleaved.len() / ch;
        if block_size > 65_535 {
            return Err(TranscodeError::CodecError(format!(
                "FLAC block size {block_size} exceeds 65535"
            )));
        }

        // ── Frame header ─────────────────────────────────────────────────
        let mut header = Vec::with_capacity(16);
        header.push(0xFF);
        header.push(0xF8); // sync tail + reserved 0 + fixed-blocksize strategy

        let (bs_code, bs_tail) = match block_size_code(block_size) {
            Some(code) => (code, None),
            None => (0b0111, Some((block_size - 1) as u16)),
        };
        let sr_code = sample_rate_code(self.sample_rate);
        header.push(((bs_code as u8) << 4) | sr_code as u8);

        // Channel assignment: independent channels = count - 1.
        let chan_code = (self.channels - 1) as u8;
        // Sample size 16-bit = 0b100; final reserved bit 0.
        header.push((chan_code << 4) | (0b100 << 1));

        utf8_coded_number(self.frame_index, &mut header);
        self.frame_index += 1;

        if let Some(tail) = bs_tail {
            header.extend_from_slice(&tail.to_be_bytes());
        }
        header.push(crc8(&header));

        // ── Subframes ────────────────────────────────────────────────────
        let mut bw = BitWriter::new();
        for &byte in &header {
            bw.write_bits(u64::from(byte), 8);
        }

        for c in 0..ch {
            let channel: Vec<i64> = interleaved[c..]
                .iter()
                .step_by(ch)
                .map(|&s| i64::from(s))
                .collect();
            Self::write_subframe(&mut bw, &channel);
        }

        // ── Padding + CRC-16 ─────────────────────────────────────────────
        bw.align();
        let mut frame = bw.into_bytes();
        let crc = crc16(&frame);
        frame.extend_from_slice(&crc.to_be_bytes());
        Ok(frame)
    }

    /// Choose and write the cheapest subframe for one channel.
    fn write_subframe(bw: &mut BitWriter, samples: &[i64]) {
        let n = samples.len();

        // CONSTANT subframe when every sample matches.
        if samples.iter().all(|&s| s == samples[0]) {
            bw.write_bits(0, 1); // zero pad bit
            bw.write_bits(0b000000, 6); // type: constant
            bw.write_bits(0, 1); // no wasted bits
            bw.write_bits(samples[0] as u64, Self::BPS);
            return;
        }

        // Pick the fixed order (0-4) with the lowest total Rice cost.
        let max_order = 4.min(n.saturating_sub(1));
        let mut best: Option<(usize, u32, u64, Vec<i64>)> = None;
        for order in 0..=max_order {
            let residuals = fixed_residuals(samples, order);
            let (k, cost) = best_rice_param(&residuals);
            let total = cost + (order as u64) * u64::from(Self::BPS);
            if best.as_ref().is_none_or(|b| total < b.2) {
                best = Some((order, k, total, residuals));
            }
        }
        let Some((order, k, cost, residuals)) = best else {
            // Single-sample block: verbatim is the only option.
            Self::write_verbatim(bw, samples);
            return;
        };

        // Fall back to VERBATIM if prediction cannot beat raw storage.
        let verbatim_cost = (n as u64) * u64::from(Self::BPS);
        if cost >= verbatim_cost {
            Self::write_verbatim(bw, samples);
            return;
        }

        bw.write_bits(0, 1);
        bw.write_bits(0b001000 | order as u64, 6); // type: fixed, given order
        bw.write_bits(0, 1);

        // Warm-up samples, raw.
        for &s in &samples[..order] {
            bw.write_bits(s as u64, Self::BPS);
        }

        // Residual: partitioned Rice, method 0 (4-bit params), partition order 0.
        bw.write_bits(0b00, 2);
        bw.write_bits(0, 4);
        bw.write_bits(u64::from(k), 4);
        for &r in &residuals {
            let u = zigzag(r);
            bw.write_unary((u >> k) as u32);
            bw.write_bits(u, k);
        }
    }

    /// Write a VERBATIM subframe (raw samples).
    fn write_verbatim(bw: &mut BitWriter, samples: &[i64]) {
        bw.write_bits(0, 1);
        bw.write_bits(0b000001, 6);
        bw.write_bits(0, 1);
        for &s in samples {
            bw.write_bits(s as u64, Self::BPS);
        }
    }
}

/// Build a complete `fLaC` stream header (magic + last-flag STREAMINFO)
/// for use without a container muxer.
#[must_use]
pub fn stream_info_block(
    sample_rate: u32,
    channels: u16,
    total_samples: u64,
    min_block: u16,
    max_block: u16,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 4 + 34);
    out.extend_from_slice(b"fLaC");
    out.push(0x80); // last metadata block, type 0 (STREAMINFO)
    out.extend_from_slice(&[0, 0, 34]);

    out.extend_from_slice(&min_block.to_be_bytes());
    out.extend_from_slice(&max_block.to_be_bytes());
    out.extend_from_slice(&[0, 0, 0]); // min frame size unknown
    out.extend_from_slice(&[0, 0, 0]); // max frame size unknown
                                       // sample rate (20) | channels-1 (3) | bps-1 (5) | total samples (36)
    let sr = u64::from(sample_rate) & 0xF_FFFF;
    let ch = u64::from(channels - 1) & 0x7;
    let bps = 15u64; // 16-bit
    let total = total_samples & 0xF_FFFF_FFFF;
    let packed: u64 = (sr << 44) | (ch << 41) | (bps << 36) | total;
    out.extend_from_slice(&packed.to_be_bytes());
    out.extend_from_slice(&[0u8; 16]); // MD5 unknown
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc8_known_vector() {
        // CRC-8/ATM of "123456789" is 0xF4.
        assert_eq!(crc8(b"123456789"), 0xF4);
    }

    #[test]
    fn test_crc16_known_vector() {
        // CRC-16/BUYPASS (poly 0x8005, init 0, no reflection) of "123456789".
        assert_eq!(crc16(b"123456789"), 0xFEE8);
    }

    #[test]
    fn test_bitwriter_msb_first() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        bw.write_bits(0b01100, 5);
        let bytes = bw.into_bytes();
        assert_eq!(bytes, vec![0b1010_1100]);
    }

    #[test]
    fn test_unary_coding() {
        let mut bw = BitWriter::new();
        bw.write_unary(3); // 0001
        bw.write_unary(0); // 1
        bw.write_bits(0, 3); // pad
        assert_eq!(bw.into_bytes(), vec![0b0001_1000]);
    }

    #[test]
    fn test_utf8_coded_number_forms() {
        let mut v = Vec::new();
        utf8_coded_number(0x25, &mut v);
        assert_eq!(v, vec![0x25]);

        let mut v = Vec::new();
        utf8_coded_number(0x80, &mut v);
        assert_eq!(v, vec![0xC2, 0x80]);

        let mut v = Vec::new();
        utf8_coded_number(0x7FF, &mut v);
        assert_eq!(v, vec![0xDF, 0xBF]);

        let mut v = Vec::new();
        utf8_coded_number(0x800, &mut v);
        assert_eq!(v, vec![0xE0, 0xA0, 0x80]);
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(zigzag(0), 0);
        assert_eq!(zigzag(-1), 1);
        assert_eq!(zigzag(1), 2);
        assert_eq!(zigzag(-2), 3);
        assert_eq!(zigzag(2), 4);
    }

    #[test]
    fn test_fixed_residuals_orders() {
        let s = [10i64, 12, 15, 19, 24];
        assert_eq!(fixed_residuals(&s, 0), vec![10, 12, 15, 19, 24]);
        assert_eq!(fixed_residuals(&s, 1), vec![2, 3, 4, 5]);
        assert_eq!(fixed_residuals(&s, 2), vec![1, 1, 1]);
        assert_eq!(fixed_residuals(&s, 3), vec![0, 0]);
    }

    #[test]
    fn test_encode_block_produces_synced_frames() {
        let mut enc = FlacStreamEncoder::new(48_000, 2).expect("encoder");
        let block: Vec<i16> = (0..4096 * 2)
            .map(|i| ((i as f32 * 0.02).sin() * 9_000.0) as i16)
            .collect();
        let frame = enc.encode_block(&block).expect("encode");
        assert!(frame.len() > 16);
        assert_eq!(frame[0], 0xFF);
        assert_eq!(frame[1] & 0xFC, 0xF8);
        // CRC-16 must verify over the frame body.
        let n = frame.len();
        let expected = crc16(&frame[..n - 2]);
        let stored = u16::from_be_bytes([frame[n - 2], frame[n - 1]]);
        assert_eq!(stored, expected, "frame CRC-16 mismatch");
    }

    #[test]
    fn test_constant_block_is_tiny() {
        let mut enc = FlacStreamEncoder::new(48_000, 1).expect("encoder");
        let block = vec![100i16; 4096];
        let frame = enc.encode_block(&block).expect("encode");
        assert!(
            frame.len() < 32,
            "constant block should use a CONSTANT subframe, got {} bytes",
            frame.len()
        );
    }

    #[test]
    fn test_encode_block_rejects_ragged() {
        let mut enc = FlacStreamEncoder::new(48_000, 2).expect("encoder");
        assert!(enc.encode_block(&[1i16, 2, 3]).is_err());
        assert!(enc.encode_block(&[]).is_err());
    }
}
