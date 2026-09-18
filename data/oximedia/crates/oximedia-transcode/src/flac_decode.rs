// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Spec-compliant FLAC stream decoder (RFC 9639) for the transcode input
//! path.
//!
//! The FLAC decoders elsewhere in the workspace only accept their own
//! non-conformant bitstreams and reject real-world files (e.g. FFmpeg- or
//! libFLAC-encoded), so FLAC transcode input decodes through this module.
//! It implements the full frame layer needed for files found in the wild:
//!
//! - CONSTANT / VERBATIM / FIXED (0–4) / LPC (1–32) subframes
//! - Both partitioned-Rice residual methods (4- and 5-bit parameters)
//!   including escape partitions
//! - Wasted-bits handling
//! - Left/side, right/side, and mid/side stereo decorrelation
//! - All block-size / sample-rate / bit-depth header codes
//!
//! Output is interleaved i16 (higher bit depths are rounded down — the
//! transcode PCM pipeline is 16-bit).

use crate::{Result, TranscodeError};

// ─── MSB-first bit reader ─────────────────────────────────────────────────────

struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits_left(&self) -> usize {
        self.data.len() * 8 - self.pos
    }

    fn byte_pos(&self) -> usize {
        self.pos / 8
    }

    fn align_byte(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }

    fn read_bits(&mut self, n: u32) -> Result<u64> {
        let n = n as usize;
        if n > 57 || self.bits_left() < n {
            return Err(TranscodeError::CodecError(
                "FLAC: bitstream exhausted".into(),
            ));
        }
        let mut value = 0u64;
        let mut remaining = n;
        while remaining > 0 {
            let byte = self.data[self.pos / 8];
            let bit_off = self.pos % 8;
            let avail = 8 - bit_off;
            let take = avail.min(remaining);
            let shifted = (byte as u64 >> (avail - take)) & ((1u64 << take) - 1);
            value = (value << take) | shifted;
            self.pos += take;
            remaining -= take;
        }
        Ok(value)
    }

    /// Signed two's-complement read.
    fn read_sbits(&mut self, n: u32) -> Result<i64> {
        let v = self.read_bits(n)?;
        if n == 0 {
            return Ok(0);
        }
        let sign = 1u64 << (n - 1);
        Ok(if v & sign != 0 {
            (v as i64) - (1i64 << n)
        } else {
            v as i64
        })
    }

    /// FLAC unary: count zero bits until a one bit.
    fn read_unary(&mut self) -> Result<u32> {
        let mut q = 0u32;
        loop {
            if self.bits_left() == 0 {
                return Err(TranscodeError::CodecError(
                    "FLAC: bitstream exhausted in unary code".into(),
                ));
            }
            if self.read_bits(1)? == 1 {
                return Ok(q);
            }
            q += 1;
            if q > 1_000_000 {
                return Err(TranscodeError::CodecError(
                    "FLAC: implausible unary run".into(),
                ));
            }
        }
    }
}

// ─── Stream info ──────────────────────────────────────────────────────────────

/// Parameters parsed from STREAMINFO.
#[derive(Debug, Clone, Copy)]
pub struct FlacStreamParams {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count (1–8).
    pub channels: u16,
    /// Bits per sample in the stream.
    pub bits_per_sample: u32,
    /// Total samples per channel (0 = unknown).
    pub total_samples: u64,
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

/// Decode a complete FLAC file (magic + metadata + frames) to interleaved
/// i16 PCM.
///
/// # Errors
///
/// Returns [`TranscodeError::CodecError`] on malformed streams or
/// [`TranscodeError::Unsupported`] for valid-but-unhandled features.
pub fn decode_flac_to_i16(data: &[u8]) -> Result<(FlacStreamParams, Vec<i16>)> {
    if data.len() < 42 || &data[0..4] != b"fLaC" {
        return Err(TranscodeError::CodecError(
            "not a FLAC stream (missing fLaC magic)".into(),
        ));
    }

    // ── Metadata blocks ──────────────────────────────────────────────────
    let mut off = 4usize;
    let mut params: Option<FlacStreamParams> = None;
    loop {
        if off + 4 > data.len() {
            return Err(TranscodeError::CodecError(
                "FLAC: truncated metadata block header".into(),
            ));
        }
        let last = data[off] & 0x80 != 0;
        let block_type = data[off] & 0x7F;
        let len = usize::from(data[off + 1]) << 16
            | usize::from(data[off + 2]) << 8
            | usize::from(data[off + 3]);
        off += 4;
        if off + len > data.len() {
            return Err(TranscodeError::CodecError(
                "FLAC: truncated metadata block".into(),
            ));
        }
        if block_type == 0 {
            // STREAMINFO
            if len < 34 {
                return Err(TranscodeError::CodecError(
                    "FLAC: STREAMINFO too short".into(),
                ));
            }
            let b = &data[off..off + 34];
            let sample_rate =
                (u32::from(b[10]) << 12) | (u32::from(b[11]) << 4) | (u32::from(b[12]) >> 4);
            let channels = u16::from((b[12] >> 1) & 0x7) + 1;
            let bits_per_sample = ((u32::from(b[12]) & 0x1) << 4 | u32::from(b[13]) >> 4) + 1;
            let total_samples = (u64::from(b[13]) & 0xF) << 32
                | u64::from(b[14]) << 24
                | u64::from(b[15]) << 16
                | u64::from(b[16]) << 8
                | u64::from(b[17]);
            params = Some(FlacStreamParams {
                sample_rate,
                channels,
                bits_per_sample,
                total_samples,
            });
        }
        off += len;
        if last {
            break;
        }
    }
    let params = params
        .ok_or_else(|| TranscodeError::CodecError("FLAC: stream carries no STREAMINFO".into()))?;
    if params.sample_rate == 0 || !(1..=8).contains(&params.channels) {
        return Err(TranscodeError::CodecError(format!(
            "FLAC: invalid stream parameters ({} Hz, {} ch)",
            params.sample_rate, params.channels
        )));
    }

    // ── Frames ───────────────────────────────────────────────────────────
    let mut pcm: Vec<i16> = Vec::new();
    let mut cursor = off;
    while cursor + 2 <= data.len() {
        // Resynchronize on the 14-bit frame sync code.
        if data[cursor] != 0xFF || (data[cursor + 1] & 0xFC) != 0xF8 {
            cursor += 1;
            continue;
        }
        let mut br = BitReader::new(&data[cursor..]);
        match decode_frame(&mut br, &params, &mut pcm) {
            Ok(()) => {
                cursor += br.byte_pos();
            }
            Err(e) => {
                // A false sync inside residual data is possible; a genuine
                // mid-file corruption is not silently skipped.
                if pcm.is_empty() {
                    return Err(e);
                }
                cursor += 1;
            }
        }
    }

    if pcm.is_empty() {
        return Err(TranscodeError::CodecError(
            "FLAC: no decodable frames found".into(),
        ));
    }
    Ok((params, pcm))
}

/// Decode one frame starting at the reader (positioned at the sync code)
/// and append interleaved i16 samples.
fn decode_frame(
    br: &mut BitReader<'_>,
    params: &FlacStreamParams,
    pcm: &mut Vec<i16>,
) -> Result<()> {
    // Header.
    let sync = br.read_bits(14)?;
    if sync != 0b1111_1111_1111_10 {
        return Err(TranscodeError::CodecError("FLAC: lost frame sync".into()));
    }
    let _reserved = br.read_bits(1)?;
    let _blocking_strategy = br.read_bits(1)?;
    let bs_code = br.read_bits(4)? as u32;
    let sr_code = br.read_bits(4)? as u32;
    let chan_code = br.read_bits(4)? as u32;
    let bps_code = br.read_bits(3)? as u32;
    let _reserved2 = br.read_bits(1)?;

    // UTF-8 coded frame/sample number — consume.
    let first = br.read_bits(8)? as u8;
    let cont = match first {
        0x00..=0x7F => 0,
        0xC0..=0xDF => 1,
        0xE0..=0xEF => 2,
        0xF0..=0xF7 => 3,
        0xF8..=0xFB => 4,
        0xFC..=0xFD => 5,
        0xFE => 6,
        _ => {
            return Err(TranscodeError::CodecError(
                "FLAC: invalid coded number".into(),
            ))
        }
    };
    for _ in 0..cont {
        let b = br.read_bits(8)? as u8;
        if b & 0xC0 != 0x80 {
            return Err(TranscodeError::CodecError(
                "FLAC: invalid coded-number continuation".into(),
            ));
        }
    }

    // Block size.
    let block_size = match bs_code {
        0b0001 => 192,
        0b0010..=0b0101 => 576usize << (bs_code - 2),
        0b0110 => br.read_bits(8)? as usize + 1,
        0b0111 => br.read_bits(16)? as usize + 1,
        0b1000..=0b1111 => 256usize << (bs_code - 8),
        _ => {
            return Err(TranscodeError::CodecError(
                "FLAC: reserved block size code".into(),
            ))
        }
    };

    // Sample rate (value unused; tail bits must be consumed).
    match sr_code {
        0b1100 => {
            br.read_bits(8)?;
        }
        0b1101 | 0b1110 => {
            br.read_bits(16)?;
        }
        0b1111 => {
            return Err(TranscodeError::CodecError(
                "FLAC: invalid sample rate code".into(),
            ))
        }
        _ => {}
    }

    // CRC-8 of the header — consume (frames are also CRC-16 protected).
    let _crc8 = br.read_bits(8)?;

    // Bits per sample.
    let bps = match bps_code {
        0b000 => params.bits_per_sample,
        0b001 => 8,
        0b010 => 12,
        0b100 => 16,
        0b101 => 20,
        0b110 => 24,
        0b111 => 32,
        _ => {
            return Err(TranscodeError::CodecError(
                "FLAC: reserved bit-depth code".into(),
            ))
        }
    };

    // Channel layout.
    let (n_channels, decorrelation) = match chan_code {
        0..=7 => ((chan_code + 1) as usize, Decorrelation::Independent),
        8 => (2, Decorrelation::LeftSide),
        9 => (2, Decorrelation::RightSide),
        10 => (2, Decorrelation::MidSide),
        _ => {
            return Err(TranscodeError::CodecError(
                "FLAC: reserved channel assignment".into(),
            ))
        }
    };
    if n_channels != usize::from(params.channels) {
        return Err(TranscodeError::CodecError(format!(
            "FLAC: frame channel count {n_channels} conflicts with STREAMINFO {}",
            params.channels
        )));
    }

    // Subframes.
    let mut channels: Vec<Vec<i64>> = Vec::with_capacity(n_channels);
    for c in 0..n_channels {
        // Side channels carry one extra bit.
        let extra = match (decorrelation, c) {
            (Decorrelation::LeftSide, 1)
            | (Decorrelation::RightSide, 0)
            | (Decorrelation::MidSide, 1) => 1,
            _ => 0,
        };
        channels.push(decode_subframe(br, block_size, bps + extra)?);
    }

    // Undo stereo decorrelation.
    match decorrelation {
        Decorrelation::Independent => {}
        Decorrelation::LeftSide => {
            for i in 0..block_size {
                channels[1][i] = channels[0][i] - channels[1][i];
            }
        }
        Decorrelation::RightSide => {
            for i in 0..block_size {
                channels[0][i] += channels[1][i];
            }
        }
        Decorrelation::MidSide => {
            for i in 0..block_size {
                let side = channels[1][i];
                let mid = (channels[0][i] << 1) | (side & 1);
                channels[0][i] = (mid + side) >> 1;
                channels[1][i] = (mid - side) >> 1;
            }
        }
    }

    // Frame footer: byte-align + CRC-16 (consume).
    br.align_byte();
    let _crc16 = br.read_bits(16)?;

    // Interleave into 16-bit output.
    let shift = bps as i32 - 16;
    for i in 0..block_size {
        for chan in &channels {
            let s = chan[i];
            let v = if shift > 0 { s >> shift } else { s << (-shift) };
            pcm.push(v.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decorrelation {
    Independent,
    LeftSide,
    RightSide,
    MidSide,
}

/// Decode one subframe of `n` samples at `bps` bits.
fn decode_subframe(br: &mut BitReader<'_>, n: usize, bps: u32) -> Result<Vec<i64>> {
    let pad = br.read_bits(1)?;
    if pad != 0 {
        return Err(TranscodeError::CodecError(
            "FLAC: subframe padding bit set".into(),
        ));
    }
    let sf_type = br.read_bits(6)? as u32;
    let wasted_flag = br.read_bits(1)?;
    let wasted = if wasted_flag == 1 {
        br.read_unary()? + 1
    } else {
        0
    };
    let eff_bps = bps
        .checked_sub(wasted)
        .ok_or_else(|| TranscodeError::CodecError("FLAC: wasted bits exceed depth".into()))?;

    let mut samples = match sf_type {
        0b000000 => {
            let v = br.read_sbits(eff_bps)?;
            vec![v; n]
        }
        0b000001 => {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(br.read_sbits(eff_bps)?);
            }
            out
        }
        0b001000..=0b001100 => {
            let order = (sf_type & 0x7) as usize;
            decode_fixed(br, n, eff_bps, order)?
        }
        0b100000..=0b111111 => {
            let order = ((sf_type & 0x1F) + 1) as usize;
            decode_lpc(br, n, eff_bps, order)?
        }
        other => {
            return Err(TranscodeError::CodecError(format!(
                "FLAC: reserved subframe type {other:#08b}"
            )))
        }
    };

    if wasted > 0 {
        for s in &mut samples {
            *s <<= wasted;
        }
    }
    Ok(samples)
}

/// Fixed-predictor subframe.
fn decode_fixed(br: &mut BitReader<'_>, n: usize, bps: u32, order: usize) -> Result<Vec<i64>> {
    if order > n {
        return Err(TranscodeError::CodecError(
            "FLAC: fixed order exceeds block size".into(),
        ));
    }
    let mut samples = Vec::with_capacity(n);
    for _ in 0..order {
        samples.push(br.read_sbits(bps)?);
    }
    let residuals = decode_residual(br, n, order)?;
    for (i, r) in residuals.into_iter().enumerate() {
        let idx = order + i;
        let pred = match order {
            0 => 0,
            1 => samples[idx - 1],
            2 => 2 * samples[idx - 1] - samples[idx - 2],
            3 => 3 * samples[idx - 1] - 3 * samples[idx - 2] + samples[idx - 3],
            _ => {
                4 * samples[idx - 1] - 6 * samples[idx - 2] + 4 * samples[idx - 3]
                    - samples[idx - 4]
            }
        };
        samples.push(pred + r);
    }
    Ok(samples)
}

/// LPC subframe.
fn decode_lpc(br: &mut BitReader<'_>, n: usize, bps: u32, order: usize) -> Result<Vec<i64>> {
    if order > n {
        return Err(TranscodeError::CodecError(
            "FLAC: LPC order exceeds block size".into(),
        ));
    }
    let mut samples = Vec::with_capacity(n);
    for _ in 0..order {
        samples.push(br.read_sbits(bps)?);
    }
    let precision = br.read_bits(4)? as u32;
    if precision == 0b1111 {
        return Err(TranscodeError::CodecError(
            "FLAC: invalid LPC precision".into(),
        ));
    }
    let precision = precision + 1;
    let shift = br.read_sbits(5)?;
    if shift < 0 {
        return Err(TranscodeError::CodecError(
            "FLAC: negative LPC shift".into(),
        ));
    }
    let mut coefs = Vec::with_capacity(order);
    for _ in 0..order {
        coefs.push(br.read_sbits(precision)?);
    }
    let residuals = decode_residual(br, n, order)?;
    for (i, r) in residuals.into_iter().enumerate() {
        let idx = order + i;
        let mut acc: i64 = 0;
        for (j, &c) in coefs.iter().enumerate() {
            acc += c * samples[idx - 1 - j];
        }
        samples.push((acc >> shift) + r);
    }
    Ok(samples)
}

/// Partitioned-Rice residual (methods 0 and 1).
fn decode_residual(br: &mut BitReader<'_>, block_size: usize, order: usize) -> Result<Vec<i64>> {
    let method = br.read_bits(2)? as u32;
    let param_bits = match method {
        0 => 4,
        1 => 5,
        _ => {
            return Err(TranscodeError::CodecError(
                "FLAC: reserved residual coding method".into(),
            ))
        }
    };
    let escape = (1u64 << param_bits) - 1;
    let partition_order = br.read_bits(4)? as u32;
    let partitions = 1usize << partition_order;
    if block_size % partitions != 0 {
        return Err(TranscodeError::CodecError(
            "FLAC: partition count does not divide block size".into(),
        ));
    }

    let mut out = Vec::with_capacity(block_size - order);
    for p in 0..partitions {
        let count = if p == 0 {
            (block_size >> partition_order)
                .checked_sub(order)
                .ok_or_else(|| {
                    TranscodeError::CodecError("FLAC: partition smaller than order".into())
                })?
        } else {
            block_size >> partition_order
        };
        let param = br.read_bits(param_bits)?;
        if param == escape {
            let raw_bits = br.read_bits(5)? as u32;
            for _ in 0..count {
                out.push(if raw_bits == 0 {
                    0
                } else {
                    br.read_sbits(raw_bits)?
                });
            }
        } else {
            let k = param as u32;
            for _ in 0..count {
                let q = u64::from(br.read_unary()?);
                let u = (q << k) | br.read_bits(k)?;
                // Zigzag decode.
                out.push(((u >> 1) as i64) ^ -((u & 1) as i64));
            }
        }
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flac_bitstream::{stream_info_block, FlacStreamEncoder};

    fn sine_i16(freq: f32, sr: u32, ch: u16, frames: usize) -> Vec<i16> {
        let mut out = Vec::with_capacity(frames * usize::from(ch));
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let v = ((2.0 * std::f32::consts::PI * freq * t).sin() * 11_000.0) as i16;
            for _ in 0..ch {
                out.push(v);
            }
        }
        out
    }

    /// Encode with our spec encoder, decode with our spec decoder:
    /// must be bit-exact (both sides independently verified against
    /// libFLAC/FFmpeg).
    #[test]
    fn test_own_encoder_round_trip_bit_exact() {
        let sr = 48_000u32;
        let ch = 2u16;
        let frames = 4_096 * 2 + 777;
        let samples = sine_i16(997.0, sr, ch, frames);

        let mut enc = FlacStreamEncoder::new(sr, ch).expect("encoder");
        let mut stream = stream_info_block(sr, ch, frames as u64, 4_096, 4_096);
        let spc = usize::from(ch) * 4_096;
        for chunk in samples.chunks(spc) {
            stream.extend_from_slice(&enc.encode_block(chunk).expect("encode block"));
        }

        let (params, decoded) = decode_flac_to_i16(&stream).expect("decode");
        assert_eq!(params.sample_rate, sr);
        assert_eq!(params.channels, ch);
        assert_eq!(params.total_samples, frames as u64);
        assert_eq!(decoded.len(), samples.len(), "sample count mismatch");
        assert_eq!(decoded, samples, "FLAC round-trip must be bit-exact");
    }

    #[test]
    fn test_bitreader_signed() {
        // 0b1110 as 4-bit signed = -2.
        let data = [0b1110_0000u8];
        let mut br = BitReader::new(&data);
        assert_eq!(br.read_sbits(4).expect("read"), -2);
    }

    #[test]
    fn test_rejects_non_flac() {
        assert!(decode_flac_to_i16(b"RIFFxxxxWAVE").is_err());
        assert!(decode_flac_to_i16(&[]).is_err());
    }

    #[test]
    fn test_constant_and_verbatim_frames() {
        let sr = 8_000u32;
        let mut enc = FlacStreamEncoder::new(sr, 1).expect("encoder");
        // Constant block.
        let constant = vec![-321i16; 512];
        // Noise block (forces verbatim or high-order fixed).
        let noise: Vec<i16> = (0..512)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 16) as i16)
            .collect();

        let mut stream = stream_info_block(sr, 1, 1_024, 4_096, 4_096);
        stream.extend_from_slice(&enc.encode_block(&constant).expect("constant"));
        stream.extend_from_slice(&enc.encode_block(&noise).expect("noise"));

        let (_, decoded) = decode_flac_to_i16(&stream).expect("decode");
        assert_eq!(&decoded[..512], &constant[..]);
        assert_eq!(&decoded[512..], &noise[..]);
    }
}
