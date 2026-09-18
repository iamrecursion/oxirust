//! PCM (Pulse Code Modulation) codec — trivial encode/decode for raw audio.
//!
//! PCM is the canonical raw audio format used as a reference baseline and for
//! uncompressed audio interchange. This module implements:
//!
//! - **`PcmEncoder`** — converts `AudioFrame` samples to raw bytes in a
//!   configurable byte order and sample format.
//! - **`PcmDecoder`** — parses raw PCM bytes back into an `AudioFrame`.
//!
//! Supported sample formats: `U8`, `I16` (LE/BE), `I24` (LE/BE, stored as
//! 3-byte samples), `I32` (LE/BE), `F32` (IEEE 754, LE/BE), `F64` (LE/BE).
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::pcm::{PcmConfig, PcmEncoder, PcmDecoder, PcmFormat, ByteOrder};
//! use oximedia_codec::audio::{AudioFrame, SampleFormat};
//!
//! let config = PcmConfig {
//!     format: PcmFormat::I16,
//!     byte_order: ByteOrder::Little,
//!     sample_rate: 44100,
//!     channels: 2,
//! };
//!
//! let encoder = PcmEncoder::new(config.clone());
//! // Build a stereo frame with 128 zero samples (raw f32 bytes, little-endian)
//! let raw_f32: Vec<f32> = vec![0.0f32; 256];
//! let raw_bytes: Vec<u8> = raw_f32.iter().flat_map(|s| s.to_le_bytes()).collect();
//! let frame = AudioFrame::new(raw_bytes, 128, 44100, 2, SampleFormat::F32);
//! let bytes = encoder.encode_frame(&frame).expect("encode");
//!
//! let decoder = PcmDecoder::new(config);
//! let decoded = decoder.decode_bytes(&bytes).expect("decode");
//! assert_eq!(decoded.sample_count, 128);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(dead_code)]

use crate::audio::{AudioFrame, SampleFormat};
use crate::error::{CodecError, CodecResult};

// =============================================================================
// PCM Format Enum
// =============================================================================

/// PCM sample encoding format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcmFormat {
    /// 8-bit unsigned (0..255, centre = 128).
    U8,
    /// 16-bit signed integer.
    I16,
    /// 24-bit signed integer (3 bytes per sample, sign-extended on decode).
    I24,
    /// 32-bit signed integer.
    I32,
    /// 32-bit IEEE-754 float.
    F32,
    /// 64-bit IEEE-754 float.
    F64,
}

impl PcmFormat {
    /// Bytes per sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I24 => 3,
            Self::I32 => 4,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

// =============================================================================
// Byte order
// =============================================================================

/// Byte ordering for multi-byte samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian (LSB first, e.g. WAV default).
    Little,
    /// Big-endian (MSB first, e.g. AIFF default).
    Big,
}

// =============================================================================
// PcmConfig
// =============================================================================

/// Configuration for PCM encoder and decoder.
#[derive(Clone, Debug)]
pub struct PcmConfig {
    /// PCM sample format.
    pub format: PcmFormat,
    /// Byte order for multi-byte formats.
    pub byte_order: ByteOrder,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u8,
}

impl Default for PcmConfig {
    fn default() -> Self {
        Self {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 2,
        }
    }
}

// =============================================================================
// Encoding helpers (f32 normalised → raw bytes)
// =============================================================================

/// Clamp a float to `[-1.0, 1.0]` and convert to `i16`.
#[inline]
fn f32_to_i16(s: f32) -> i16 {
    let v = (s.clamp(-1.0, 1.0) * 32767.0) as i32;
    v.clamp(-32768, 32767) as i16
}

/// Clamp a float to `[-1.0, 1.0]` and convert to `i24` (return i32 in ±8_388_607).
#[inline]
fn f32_to_i24(s: f32) -> i32 {
    let v = (s.clamp(-1.0, 1.0) * 8_388_607.0) as i64;
    v.clamp(-8_388_608, 8_388_607) as i32
}

/// Clamp a float to `[-1.0, 1.0]` and convert to `i32`.
#[inline]
fn f32_to_i32(s: f32) -> i32 {
    let v = (s.clamp(-1.0, 1.0) * 2_147_483_647.0_f64 as f32) as i64;
    v.clamp(-2_147_483_648, 2_147_483_647) as i32
}

/// Convert `f32` to `u8` (maps `-1..1` → `0..255`, centre = 128).
#[inline]
fn f32_to_u8(s: f32) -> u8 {
    let v = ((s.clamp(-1.0, 1.0) + 1.0) * 127.5) as i32;
    v.clamp(0, 255) as u8
}

// =============================================================================
// Decoding helpers (raw bytes → f32 normalised)
// =============================================================================

#[inline]
fn i16_to_f32(v: i16) -> f32 {
    v as f32 / 32768.0
}

#[inline]
fn i24_to_f32(v: i32) -> f32 {
    v as f32 / 8_388_608.0
}

#[inline]
fn i32_to_f32(v: i32) -> f32 {
    v as f32 / 2_147_483_648.0_f64 as f32
}

#[inline]
fn u8_to_f32(v: u8) -> f32 {
    (v as f32 - 128.0) / 128.0
}

// =============================================================================
// Internal helpers for frame sample extraction
// =============================================================================

/// Extract normalised f32 samples from an `AudioFrame`'s raw byte buffer.
///
/// The frame's `format` field determines how the bytes are interpreted.
/// Returns an error if the byte buffer length is not aligned for the format.
fn frame_to_f32_samples(frame: &AudioFrame) -> CodecResult<Vec<f32>> {
    match frame.format {
        SampleFormat::F32 => {
            if frame.samples.len() % 4 != 0 {
                return Err(CodecError::InvalidData(
                    "F32 frame: sample byte count is not a multiple of 4".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(frame.samples.len() / 4);
            for chunk in frame.samples.chunks_exact(4) {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                out.push(f32::from_le_bytes(arr));
            }
            Ok(out)
        }
        SampleFormat::I16 => {
            if frame.samples.len() % 2 != 0 {
                return Err(CodecError::InvalidData(
                    "I16 frame: sample byte count is not a multiple of 2".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(frame.samples.len() / 2);
            for chunk in frame.samples.chunks_exact(2) {
                let arr = [chunk[0], chunk[1]];
                let v = i16::from_le_bytes(arr);
                out.push(i16_to_f32(v));
            }
            Ok(out)
        }
        SampleFormat::I32 => {
            if frame.samples.len() % 4 != 0 {
                return Err(CodecError::InvalidData(
                    "I32 frame: sample byte count is not a multiple of 4".to_string(),
                ));
            }
            let mut out = Vec::with_capacity(frame.samples.len() / 4);
            for chunk in frame.samples.chunks_exact(4) {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let v = i32::from_le_bytes(arr);
                out.push(i32_to_f32(v));
            }
            Ok(out)
        }
        SampleFormat::U8 => {
            let mut out = Vec::with_capacity(frame.samples.len());
            for &b in &frame.samples {
                out.push(u8_to_f32(b));
            }
            Ok(out)
        }
    }
}

/// Convert a slice of normalised f32 samples into raw F32-LE bytes.
fn f32_samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

// =============================================================================
// PcmEncoder
// =============================================================================

/// Encodes `AudioFrame` samples to raw PCM bytes.
///
/// The output is interleaved PCM data with no header — suitable for embedding
/// inside WAV/AIFF containers or piping to raw audio sinks.
#[derive(Debug, Clone)]
pub struct PcmEncoder {
    config: PcmConfig,
}

impl PcmEncoder {
    /// Create a new encoder with the given configuration.
    #[must_use]
    pub fn new(config: PcmConfig) -> Self {
        Self { config }
    }

    /// Encode one `AudioFrame` to raw bytes.
    ///
    /// The frame's `format` field controls how `samples` bytes are interpreted
    /// before re-encoding into the configured `PcmFormat`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if the frame channel count or
    /// sample rate does not match the encoder configuration.
    pub fn encode_frame(&self, frame: &AudioFrame) -> CodecResult<Vec<u8>> {
        let ch = frame.channels as u8;
        if ch != self.config.channels {
            return Err(CodecError::InvalidParameter(format!(
                "PCM encoder: expected {} channels, got {}",
                self.config.channels, ch
            )));
        }
        if frame.sample_rate != self.config.sample_rate {
            return Err(CodecError::InvalidParameter(format!(
                "PCM encoder: expected sample_rate={}, got {}",
                self.config.sample_rate, frame.sample_rate
            )));
        }

        // Decode the frame's internal byte representation to normalised f32.
        let f32_samples = frame_to_f32_samples(frame)?;

        let bps = self.config.format.bytes_per_sample();
        let mut out = Vec::with_capacity(f32_samples.len() * bps);
        let le = self.config.byte_order == ByteOrder::Little;

        for s in f32_samples {
            match self.config.format {
                PcmFormat::U8 => out.push(f32_to_u8(s)),
                PcmFormat::I16 => {
                    let v = f32_to_i16(s);
                    if le {
                        out.extend_from_slice(&v.to_le_bytes());
                    } else {
                        out.extend_from_slice(&v.to_be_bytes());
                    }
                }
                PcmFormat::I24 => {
                    let v = f32_to_i24(s);
                    if le {
                        out.push((v & 0xFF) as u8);
                        out.push(((v >> 8) & 0xFF) as u8);
                        out.push(((v >> 16) & 0xFF) as u8);
                    } else {
                        out.push(((v >> 16) & 0xFF) as u8);
                        out.push(((v >> 8) & 0xFF) as u8);
                        out.push((v & 0xFF) as u8);
                    }
                }
                PcmFormat::I32 => {
                    let v = f32_to_i32(s);
                    if le {
                        out.extend_from_slice(&v.to_le_bytes());
                    } else {
                        out.extend_from_slice(&v.to_be_bytes());
                    }
                }
                PcmFormat::F32 => {
                    if le {
                        out.extend_from_slice(&s.to_le_bytes());
                    } else {
                        out.extend_from_slice(&s.to_be_bytes());
                    }
                }
                PcmFormat::F64 => {
                    let d = f64::from(s);
                    if le {
                        out.extend_from_slice(&d.to_le_bytes());
                    } else {
                        out.extend_from_slice(&d.to_be_bytes());
                    }
                }
            }
        }

        Ok(out)
    }

    /// Encode a slice of interleaved `f32` samples directly.
    ///
    /// # Errors
    ///
    /// Returns error if `samples.len()` is not a multiple of `channels`.
    pub fn encode_raw(&self, samples: &[f32]) -> CodecResult<Vec<u8>> {
        if self.config.channels > 0 && samples.len() % self.config.channels as usize != 0 {
            return Err(CodecError::InvalidParameter(
                "PCM encode_raw: sample count not multiple of channels".to_string(),
            ));
        }
        let sample_count = if self.config.channels > 0 {
            samples.len() / self.config.channels as usize
        } else {
            samples.len()
        };
        // Store f32 samples as raw LE bytes in the frame.
        let raw_bytes = f32_samples_to_bytes(samples);
        let frame = AudioFrame::new(
            raw_bytes,
            sample_count,
            self.config.sample_rate,
            self.config.channels as usize,
            SampleFormat::F32,
        );
        self.encode_frame(&frame)
    }

    /// Return a reference to the encoder configuration.
    #[must_use]
    pub fn config(&self) -> &PcmConfig {
        &self.config
    }
}

// =============================================================================
// PcmDecoder
// =============================================================================

/// Decodes raw PCM bytes to an `AudioFrame`.
///
/// No header parsing is performed — the caller must know the format ahead of time.
#[derive(Debug, Clone)]
pub struct PcmDecoder {
    config: PcmConfig,
}

impl PcmDecoder {
    /// Create a new decoder with the given configuration.
    #[must_use]
    pub fn new(config: PcmConfig) -> Self {
        Self { config }
    }

    /// Decode raw PCM bytes into an `AudioFrame`.
    ///
    /// Samples are decoded to normalised f32 and stored as F32-LE bytes in
    /// the returned `AudioFrame`.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if `bytes.len()` is not an
    /// exact multiple of bytes-per-sample.
    pub fn decode_bytes(&self, bytes: &[u8]) -> CodecResult<AudioFrame> {
        let bps = self.config.format.bytes_per_sample();
        if bytes.len() % bps != 0 {
            return Err(CodecError::InvalidBitstream(format!(
                "PCM decode: byte count {} is not a multiple of bytes-per-sample {}",
                bytes.len(),
                bps
            )));
        }

        let n_samples = bytes.len() / bps;
        let mut f32_samples: Vec<f32> = Vec::with_capacity(n_samples);
        let le = self.config.byte_order == ByteOrder::Little;

        let mut i = 0;
        while i < bytes.len() {
            let s = match self.config.format {
                PcmFormat::U8 => {
                    let v = bytes[i];
                    i += 1;
                    u8_to_f32(v)
                }
                PcmFormat::I16 => {
                    let arr = [bytes[i], bytes[i + 1]];
                    let v = if le {
                        i16::from_le_bytes(arr)
                    } else {
                        i16::from_be_bytes(arr)
                    };
                    i += 2;
                    i16_to_f32(v)
                }
                PcmFormat::I24 => {
                    let raw = if le {
                        (bytes[i] as i32)
                            | ((bytes[i + 1] as i32) << 8)
                            | ((bytes[i + 2] as i32) << 16)
                    } else {
                        ((bytes[i] as i32) << 16)
                            | ((bytes[i + 1] as i32) << 8)
                            | (bytes[i + 2] as i32)
                    };
                    // Sign-extend from 24 bits
                    let v = if raw & 0x80_0000 != 0 {
                        raw | !0xFF_FFFF_i32
                    } else {
                        raw
                    };
                    i += 3;
                    i24_to_f32(v)
                }
                PcmFormat::I32 => {
                    let arr = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
                    let v = if le {
                        i32::from_le_bytes(arr)
                    } else {
                        i32::from_be_bytes(arr)
                    };
                    i += 4;
                    i32_to_f32(v)
                }
                PcmFormat::F32 => {
                    let arr = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];
                    let v = if le {
                        f32::from_le_bytes(arr)
                    } else {
                        f32::from_be_bytes(arr)
                    };
                    i += 4;
                    v
                }
                PcmFormat::F64 => {
                    let arr = [
                        bytes[i],
                        bytes[i + 1],
                        bytes[i + 2],
                        bytes[i + 3],
                        bytes[i + 4],
                        bytes[i + 5],
                        bytes[i + 6],
                        bytes[i + 7],
                    ];
                    let v = if le {
                        f64::from_le_bytes(arr)
                    } else {
                        f64::from_be_bytes(arr)
                    };
                    i += 8;
                    v as f32
                }
            };
            f32_samples.push(s);
        }

        // Store decoded f32 samples as raw LE bytes.
        let raw_bytes = f32_samples_to_bytes(&f32_samples);
        let channels = self.config.channels as usize;
        let sample_count = f32_samples
            .len()
            .checked_div(channels)
            .unwrap_or(f32_samples.len());

        Ok(AudioFrame::new(
            raw_bytes,
            sample_count,
            self.config.sample_rate,
            channels,
            SampleFormat::F32,
        ))
    }

    /// Return a reference to the decoder configuration.
    #[must_use]
    pub fn config(&self) -> &PcmConfig {
        &self.config
    }

    /// Compute the number of frames (per-channel sample groups) in a byte slice.
    #[must_use]
    pub fn frame_count(&self, bytes: &[u8]) -> usize {
        let bps = self.config.format.bytes_per_sample();
        let total_samples = bytes.len() / bps;
        let ch = self.config.channels as usize;
        total_samples.checked_div(ch).unwrap_or(0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `AudioFrame` containing `f32` samples encoded as raw LE bytes.
    fn make_frame(samples: Vec<f32>, sample_rate: u32, channels: u8) -> AudioFrame {
        let sample_count = if channels > 0 {
            samples.len() / channels as usize
        } else {
            samples.len()
        };
        let raw_bytes = f32_samples_to_bytes(&samples);
        AudioFrame::new(
            raw_bytes,
            sample_count,
            sample_rate,
            channels as usize,
            SampleFormat::F32,
        )
    }

    /// Extract decoded f32 samples from a frame (stored as F32-LE bytes).
    fn frame_samples(frame: &AudioFrame) -> Vec<f32> {
        frame_to_f32_samples(frame).expect("frame_to_f32_samples")
    }

    // ---------- U8 -----------------------------------------------------------

    #[test]
    fn test_u8_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::U8,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 1,
        };
        let input = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 44100, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len());
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        assert_eq!(got.len(), input.len());
        // U8 has 1/256 resolution — check rough roundtrip.
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!((orig - g).abs() < 0.02, "U8 roundtrip: orig={orig} got={g}");
        }
    }

    // ---------- I16 LE -------------------------------------------------------

    #[test]
    fn test_i16_le_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 2,
        };
        let input: Vec<f32> = (0..256).map(|i| i as f32 / 128.0 - 1.0).collect();
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 48000, 2);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 2);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        assert_eq!(got.len(), input.len());
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!(
                (orig - g).abs() < 0.0001,
                "I16 roundtrip: orig={orig} got={g}"
            );
        }
    }

    // ---------- I16 BE -------------------------------------------------------

    #[test]
    fn test_i16_be_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Big,
            sample_rate: 44100,
            channels: 1,
        };
        let input = vec![0.0f32, 0.25, -0.25, 0.99, -0.99];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 44100, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 2);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!((orig - g).abs() < 0.0001, "I16 BE roundtrip");
        }
    }

    // ---------- I24 ----------------------------------------------------------

    #[test]
    fn test_i24_le_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I24,
            byte_order: ByteOrder::Little,
            sample_rate: 96000,
            channels: 2,
        };
        let input: Vec<f32> = vec![0.0, 0.5, -0.5, 0.999, -0.999];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 96000, 2);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 3);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!(
                (orig - g).abs() < 0.000001,
                "I24 LE roundtrip orig={orig} got={g}"
            );
        }
    }

    #[test]
    fn test_i24_be_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I24,
            byte_order: ByteOrder::Big,
            sample_rate: 96000,
            channels: 1,
        };
        let input: Vec<f32> = vec![0.0, -0.1, 0.3, -0.7, 0.9];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 96000, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!(
                (orig - g).abs() < 0.000001,
                "I24 BE roundtrip orig={orig} got={g}"
            );
        }
    }

    // ---------- I32 ----------------------------------------------------------

    #[test]
    fn test_i32_le_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I32,
            byte_order: ByteOrder::Little,
            sample_rate: 192000,
            channels: 1,
        };
        let input = vec![0.0f32, 0.5, -0.5, 0.9999, -0.9999];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 192000, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 4);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!(
                (orig - g).abs() < 0.0001,
                "I32 LE roundtrip orig={orig} got={g}"
            );
        }
    }

    // ---------- F32 ----------------------------------------------------------

    #[test]
    fn test_f32_le_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::F32,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 2,
        };
        let input: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 48000, 2);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 4);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert_eq!(orig, g, "F32 LE should be lossless");
        }
    }

    #[test]
    fn test_f32_be_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::F32,
            byte_order: ByteOrder::Big,
            sample_rate: 44100,
            channels: 1,
        };
        let input = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 44100, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert_eq!(orig, g, "F32 BE should be lossless");
        }
    }

    // ---------- F64 ----------------------------------------------------------

    #[test]
    fn test_f64_le_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::F64,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 1,
        };
        let input = vec![0.0f32, 0.123, -0.456, 0.789, -0.999];
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let frame = make_frame(input.clone(), 48000, 1);
        let bytes = enc.encode_frame(&frame).expect("encode");
        assert_eq!(bytes.len(), input.len() * 8);
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        for (&orig, &g) in input.iter().zip(got.iter()) {
            assert!(
                (orig - g).abs() < 1e-6,
                "F64 LE roundtrip orig={orig} got={g}"
            );
        }
    }

    // ---------- Error cases --------------------------------------------------

    #[test]
    fn test_mismatched_channels_error() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let enc = PcmEncoder::new(cfg);
        // Mono frame with stereo encoder
        let frame = make_frame(vec![0.0f32; 128], 44100, 1);
        assert!(enc.encode_frame(&frame).is_err());
    }

    #[test]
    fn test_mismatched_sample_rate_error() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 1,
        };
        let enc = PcmEncoder::new(cfg);
        let frame = make_frame(vec![0.0f32; 64], 48000, 1); // wrong rate
        assert!(enc.encode_frame(&frame).is_err());
    }

    #[test]
    fn test_decode_bad_alignment_error() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 1,
        };
        let dec = PcmDecoder::new(cfg);
        let bytes = vec![0u8; 3]; // not multiple of 2
        assert!(dec.decode_bytes(&bytes).is_err());
    }

    #[test]
    fn test_encode_raw_roundtrip() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let raw: Vec<f32> = (0..64).map(|i| (i as f32 / 32.0) - 1.0).collect();
        let bytes = enc.encode_raw(&raw).expect("encode_raw");
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        assert_eq!(got.len(), raw.len());
    }

    #[test]
    fn test_encode_raw_bad_alignment_error() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let enc = PcmEncoder::new(cfg);
        // 3 samples is not divisible by 2 channels
        let raw = vec![0.0f32; 3];
        assert!(enc.encode_raw(&raw).is_err());
    }

    #[test]
    fn test_frame_count() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let dec = PcmDecoder::new(cfg);
        // 256 bytes / 2 bps = 128 samples / 2 ch = 64 frames
        assert_eq!(dec.frame_count(&vec![0u8; 256]), 64);
    }

    #[test]
    fn test_silence_encode_decode() {
        let cfg = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        let silence = make_frame(vec![0.0f32; 512], 44100, 2);
        let bytes = enc.encode_frame(&silence).expect("encode");
        // All bytes should be zero for silence
        assert!(bytes.iter().all(|&b| b == 0));
        let decoded = dec.decode_bytes(&bytes).expect("decode");
        let got = frame_samples(&decoded);
        assert!(got.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_config_accessor() {
        let cfg = PcmConfig::default();
        let enc = PcmEncoder::new(cfg.clone());
        let dec = PcmDecoder::new(cfg);
        assert_eq!(enc.config().channels, 2);
        assert_eq!(dec.config().sample_rate, 48000);
    }
}
