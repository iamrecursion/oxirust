//! Audio sample format descriptions and conversion utilities.
//!
//! This module defines the [`SampleFormat`] enum that categorises common PCM
//! sample representations, the [`AudioFormat`] struct that fully describes an
//! audio stream, and [`FormatConverter`] which converts buffers between
//! different bit-depths and normalisation conventions.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::audio_format::{SampleFormat, AudioFormat, FormatConverter};
//!
//! let src = AudioFormat::new(SampleFormat::F32Le, 48_000, 2);
//! let dst = AudioFormat::new(SampleFormat::S16Le, 48_000, 2);
//! let converter = FormatConverter::new(src, dst);
//!
//! let input: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0];
//! let output: Vec<i16> = converter.f32_to_s16(&input);
//! assert_eq!(output.len(), 4);
//! ```

#![allow(dead_code)]

/// PCM sample format.
///
/// Covers the most common integer and floating-point representations used in
/// professional audio production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit PCM (0–255, midpoint at 128).
    U8,
    /// Signed 16-bit little-endian PCM.
    S16Le,
    /// Signed 16-bit big-endian PCM.
    S16Be,
    /// Signed 24-bit little-endian PCM (packed in 3 bytes).
    S24Le,
    /// Signed 32-bit little-endian PCM.
    S32Le,
    /// 32-bit IEEE 754 floating-point little-endian.
    F32Le,
    /// 64-bit IEEE 754 floating-point little-endian.
    F64Le,
}

impl SampleFormat {
    /// Return the number of bytes per sample.
    #[must_use]
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16Le | Self::S16Be => 2,
            Self::S24Le => 3,
            Self::S32Le | Self::F32Le => 4,
            Self::F64Le => 8,
        }
    }

    /// Return the bit depth of the format.
    #[must_use]
    pub fn bit_depth(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::S16Le | Self::S16Be => 16,
            Self::S24Le => 24,
            Self::S32Le => 32,
            Self::F32Le => 32,
            Self::F64Le => 64,
        }
    }

    /// Return `true` if this format uses floating-point samples.
    #[must_use]
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32Le | Self::F64Le)
    }

    /// Return `true` if this format stores samples as signed integers.
    #[must_use]
    pub fn is_signed_int(self) -> bool {
        matches!(self, Self::S16Le | Self::S16Be | Self::S24Le | Self::S32Le)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::U8 => "U8",
            Self::S16Le => "S16LE",
            Self::S16Be => "S16BE",
            Self::S24Le => "S24LE",
            Self::S32Le => "S32LE",
            Self::F32Le => "F32LE",
            Self::F64Le => "F64LE",
        }
    }
}

/// Complete description of an audio stream's format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFormat {
    /// PCM sample format.
    pub sample_format: SampleFormat,
    /// Sample rate in Hz (e.g. 44_100, 48_000, 96_000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo, etc.).
    pub channels: u16,
}

impl AudioFormat {
    /// Create a new audio format descriptor.
    #[must_use]
    pub fn new(sample_format: SampleFormat, sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_format,
            sample_rate,
            channels,
        }
    }

    /// Return the byte rate (bytes per second) for this format.
    #[must_use]
    pub fn byte_rate(&self) -> u64 {
        #[allow(clippy::cast_precision_loss)]
        {
            (self.sample_rate as u64)
                * (self.channels as u64)
                * (self.sample_format.bytes_per_sample() as u64)
        }
    }

    /// Return the frame size in bytes (one sample per channel).
    #[must_use]
    pub fn frame_size_bytes(&self) -> usize {
        self.sample_format.bytes_per_sample() * (self.channels as usize)
    }

    /// Return `true` if the two formats are compatible for direct mixing
    /// (same sample rate, channels, and sample format).
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::new(SampleFormat::F32Le, 48_000, 2)
    }
}

/// Converts audio buffers between sample formats.
///
/// Handles the common case of converting `f32` (normalised ±1.0) to or from
/// signed integer formats.
pub struct FormatConverter {
    /// Source format.
    pub src: AudioFormat,
    /// Destination format.
    pub dst: AudioFormat,
}

impl FormatConverter {
    /// Create a new format converter.
    #[must_use]
    pub fn new(src: AudioFormat, dst: AudioFormat) -> Self {
        Self { src, dst }
    }

    /// Convert a slice of `f32` (±1.0) samples to `i16`.
    ///
    /// Values are clamped before conversion to prevent overflow.
    #[must_use]
    pub fn f32_to_s16(&self, input: &[f32]) -> Vec<i16> {
        input
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                #[allow(clippy::cast_possible_truncation)]
                let out = (clamped * 32767.0) as i16;
                out
            })
            .collect()
    }

    /// Convert a slice of `i16` samples to `f32` (±1.0).
    #[must_use]
    pub fn s16_to_f32(&self, input: &[i16]) -> Vec<f32> {
        input
            .iter()
            .map(|&s| {
                #[allow(clippy::cast_precision_loss)]
                let out = (s as f32) / 32768.0;
                out
            })
            .collect()
    }

    /// Convert a slice of `f32` (±1.0) samples to `i32`.
    #[must_use]
    pub fn f32_to_s32(&self, input: &[f32]) -> Vec<i32> {
        input
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                #[allow(clippy::cast_possible_truncation)]
                let out = (clamped * 2_147_483_647.0) as i32;
                out
            })
            .collect()
    }

    /// Convert a slice of `i32` samples to `f32` (±1.0).
    #[must_use]
    pub fn s32_to_f32(&self, input: &[i32]) -> Vec<f32> {
        input
            .iter()
            .map(|&s| {
                #[allow(clippy::cast_precision_loss)]
                let out = (s as f32) / 2_147_483_648.0;
                out
            })
            .collect()
    }

    /// Normalise a `f32` buffer so that the peak equals `target_level`.
    ///
    /// If the buffer is all zeros the function is a no-op.
    pub fn normalize_f32(buffer: &mut [f32], target_level: f32) {
        let peak = buffer.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
        if peak > 1e-9 {
            let gain = target_level / peak;
            for s in buffer.iter_mut() {
                *s *= gain;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bytes_u8() {
        assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
    }

    #[test]
    fn test_sample_format_bytes_s16() {
        assert_eq!(SampleFormat::S16Le.bytes_per_sample(), 2);
    }

    #[test]
    fn test_sample_format_bytes_s24() {
        assert_eq!(SampleFormat::S24Le.bytes_per_sample(), 3);
    }

    #[test]
    fn test_sample_format_bytes_f32() {
        assert_eq!(SampleFormat::F32Le.bytes_per_sample(), 4);
    }

    #[test]
    fn test_sample_format_bytes_f64() {
        assert_eq!(SampleFormat::F64Le.bytes_per_sample(), 8);
    }

    #[test]
    fn test_sample_format_is_float() {
        assert!(SampleFormat::F32Le.is_float());
        assert!(SampleFormat::F64Le.is_float());
        assert!(!SampleFormat::S16Le.is_float());
    }

    #[test]
    fn test_sample_format_is_signed_int() {
        assert!(SampleFormat::S16Le.is_signed_int());
        assert!(SampleFormat::S32Le.is_signed_int());
        assert!(!SampleFormat::F32Le.is_signed_int());
        assert!(!SampleFormat::U8.is_signed_int());
    }

    #[test]
    fn test_sample_format_bit_depth() {
        assert_eq!(SampleFormat::S16Le.bit_depth(), 16);
        assert_eq!(SampleFormat::S24Le.bit_depth(), 24);
        assert_eq!(SampleFormat::F32Le.bit_depth(), 32);
    }

    #[test]
    fn test_sample_format_label() {
        assert_eq!(SampleFormat::S16Le.label(), "S16LE");
        assert_eq!(SampleFormat::F32Le.label(), "F32LE");
    }

    #[test]
    fn test_audio_format_byte_rate() {
        let fmt = AudioFormat::new(SampleFormat::S16Le, 48_000, 2);
        assert_eq!(fmt.byte_rate(), 48_000 * 2 * 2);
    }

    #[test]
    fn test_audio_format_frame_size() {
        let fmt = AudioFormat::new(SampleFormat::F32Le, 48_000, 2);
        assert_eq!(fmt.frame_size_bytes(), 8);
    }

    #[test]
    fn test_audio_format_compatibility() {
        let a = AudioFormat::new(SampleFormat::F32Le, 48_000, 2);
        let b = AudioFormat::new(SampleFormat::F32Le, 48_000, 2);
        let c = AudioFormat::new(SampleFormat::S16Le, 48_000, 2);
        assert!(a.is_compatible_with(&b));
        assert!(!a.is_compatible_with(&c));
    }

    #[test]
    fn test_f32_to_s16_zero() {
        let conv = FormatConverter::new(AudioFormat::default(), AudioFormat::default());
        let out = conv.f32_to_s16(&[0.0]);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn test_f32_to_s16_positive_full_scale() {
        let conv = FormatConverter::new(AudioFormat::default(), AudioFormat::default());
        let out = conv.f32_to_s16(&[1.0]);
        assert_eq!(out[0], 32767);
    }

    #[test]
    fn test_f32_to_s16_negative_full_scale() {
        let conv = FormatConverter::new(AudioFormat::default(), AudioFormat::default());
        let out = conv.f32_to_s16(&[-1.0]);
        assert_eq!(out[0], -32767);
    }

    #[test]
    fn test_s16_to_f32_zero() {
        let conv = FormatConverter::new(AudioFormat::default(), AudioFormat::default());
        let out = conv.s16_to_f32(&[0]);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_s16_to_f32_range() {
        let conv = FormatConverter::new(AudioFormat::default(), AudioFormat::default());
        let out = conv.s16_to_f32(&[32767, -32768]);
        assert!(out[0] > 0.99 && out[0] <= 1.0);
        assert!(out[1] >= -1.0 && out[1] < -0.99);
    }

    #[test]
    fn test_normalize_f32() {
        let mut buf = vec![0.1_f32, 0.2, 0.05, -0.15];
        FormatConverter::normalize_f32(&mut buf, 1.0);
        let peak = buf.iter().map(|&s| s.abs()).fold(0.0_f32, f32::max);
        assert!((peak - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_f32_silence_noop() {
        let mut buf = vec![0.0_f32; 8];
        FormatConverter::normalize_f32(&mut buf, 1.0);
        assert!(buf.iter().all(|&v| v == 0.0));
    }
}
