//! Audio frame types and sample format definitions.
//!
//! This module provides common types for audio codec implementations,
//! including frame representation and sample format handling.

use crate::{CodecError, CodecResult};

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 32-bit floating point samples (normalized -1.0 to 1.0)
    F32,
    /// 16-bit signed integer samples
    I16,
    /// 32-bit signed integer samples
    I32,
    /// 8-bit unsigned integer samples
    U8,
}

impl SampleFormat {
    /// Returns the size in bytes of a single sample in this format.
    #[must_use]
    pub const fn sample_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::U8 => 1,
        }
    }

    /// Returns whether this format uses floating point representation.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F32)
    }

    /// Returns whether this format uses signed integer representation.
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        !matches!(self, Self::U8)
    }
}

/// Audio channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Single channel (mono)
    Mono,
    /// Two channels (stereo: left, right)
    Stereo,
    /// 5.1 surround sound
    Surround51,
    /// 7.1 surround sound
    Surround71,
    /// Custom channel count
    Custom(u8),
}

impl ChannelLayout {
    /// Returns the number of channels in this layout.
    #[must_use]
    pub const fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(n) => *n as usize,
        }
    }
}

/// An audio frame containing decoded samples.
///
/// Audio frames store PCM samples in interleaved format for multi-channel audio.
/// For example, stereo audio is stored as [L0, R0, L1, R1, L2, R2, ...].
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Raw sample data (interleaved if multi-channel)
    pub samples: Vec<u8>,
    /// Number of samples per channel
    pub sample_count: usize,
    /// Sample rate in Hz (e.g., 48000 for 48kHz)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo, etc.)
    pub channels: usize,
    /// Sample format
    pub format: SampleFormat,
    /// Presentation timestamp in sample units
    pub pts: Option<i64>,
    /// Duration in sample units
    pub duration: Option<u64>,
}

impl AudioFrame {
    /// Creates a new audio frame.
    ///
    /// # Arguments
    ///
    /// * `samples` - Raw sample data (interleaved if multi-channel)
    /// * `sample_count` - Number of samples per channel
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `format` - Sample format
    pub fn new(
        samples: Vec<u8>,
        sample_count: usize,
        sample_rate: u32,
        channels: usize,
        format: SampleFormat,
    ) -> Self {
        Self {
            samples,
            sample_count,
            sample_rate,
            channels,
            format,
            pts: None,
            duration: None,
        }
    }

    /// Creates a new audio frame with timing information.
    pub fn with_timing(
        samples: Vec<u8>,
        sample_count: usize,
        sample_rate: u32,
        channels: usize,
        format: SampleFormat,
        pts: i64,
        duration: u64,
    ) -> Self {
        Self {
            samples,
            sample_count,
            sample_rate,
            channels,
            format,
            pts: Some(pts),
            duration: Some(duration),
        }
    }

    /// Returns the total number of samples (all channels combined).
    #[must_use]
    pub const fn total_samples(&self) -> usize {
        self.sample_count * self.channels
    }

    /// Returns the size in bytes of the audio data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.total_samples() * self.format.sample_size()
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        f64::from(self.sample_count as u32) / f64::from(self.sample_rate)
    }

    /// Converts samples to f32 slice if format is F32.
    ///
    /// Note: This method is not available as it requires unsafe code.
    /// Use `samples` field directly with proper byte-to-f32 conversion.
    #[allow(dead_code)]
    fn as_f32_internal(&self) -> CodecResult<Vec<f32>> {
        if self.format != SampleFormat::F32 {
            return Err(CodecError::InvalidData(
                "Sample format is not F32".to_string(),
            ));
        }
        if self.samples.len() % 4 != 0 {
            return Err(CodecError::InvalidData(
                "Sample data length is not a multiple of 4".to_string(),
            ));
        }
        let mut result = Vec::with_capacity(self.samples.len() / 4);
        for chunk in self.samples.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            result.push(f32::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Converts samples to i16 slice if format is I16.
    ///
    /// Note: This method is not available as it requires unsafe code.
    /// Use `samples` field directly with proper byte-to-i16 conversion.
    #[allow(dead_code)]
    fn as_i16_internal(&self) -> CodecResult<Vec<i16>> {
        if self.format != SampleFormat::I16 {
            return Err(CodecError::InvalidData(
                "Sample format is not I16".to_string(),
            ));
        }
        if self.samples.len() % 2 != 0 {
            return Err(CodecError::InvalidData(
                "Sample data length is not a multiple of 2".to_string(),
            ));
        }
        let mut result = Vec::with_capacity(self.samples.len() / 2);
        for chunk in self.samples.chunks_exact(2) {
            let bytes: [u8; 2] = [chunk[0], chunk[1]];
            result.push(i16::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Converts bytes to f32 and returns a new Vec.
    pub fn to_f32(&self) -> CodecResult<Vec<f32>> {
        match self.format {
            SampleFormat::F32 => {
                if self.samples.len() % 4 != 0 {
                    return Err(CodecError::InvalidData(
                        "Sample data length is not a multiple of 4".to_string(),
                    ));
                }
                let mut result = Vec::with_capacity(self.samples.len() / 4);
                for chunk in self.samples.chunks_exact(4) {
                    let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            SampleFormat::I16 => {
                if self.samples.len() % 2 != 0 {
                    return Err(CodecError::InvalidData(
                        "Sample data length is not a multiple of 2".to_string(),
                    ));
                }
                let mut result = Vec::with_capacity(self.samples.len() / 2);
                for chunk in self.samples.chunks_exact(2) {
                    let bytes: [u8; 2] = [chunk[0], chunk[1]];
                    let i16_val = i16::from_le_bytes(bytes);
                    result.push(f32::from(i16_val) / 32768.0);
                }
                Ok(result)
            }
            _ => Err(CodecError::InvalidData(
                "Unsupported format conversion".to_string(),
            )),
        }
    }

    /// Converts bytes to i16 and returns a new Vec.
    pub fn to_i16(&self) -> CodecResult<Vec<i16>> {
        match self.format {
            SampleFormat::I16 => {
                if self.samples.len() % 2 != 0 {
                    return Err(CodecError::InvalidData(
                        "Sample data length is not a multiple of 2".to_string(),
                    ));
                }
                let mut result = Vec::with_capacity(self.samples.len() / 2);
                for chunk in self.samples.chunks_exact(2) {
                    let bytes: [u8; 2] = [chunk[0], chunk[1]];
                    result.push(i16::from_le_bytes(bytes));
                }
                Ok(result)
            }
            SampleFormat::F32 => {
                if self.samples.len() % 4 != 0 {
                    return Err(CodecError::InvalidData(
                        "Sample data length is not a multiple of 4".to_string(),
                    ));
                }
                let mut result = Vec::with_capacity(self.samples.len() / 4);
                for chunk in self.samples.chunks_exact(4) {
                    let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let f32_val = f32::from_le_bytes(bytes);
                    result.push((f32_val.clamp(-1.0, 1.0) * 32767.0) as i16);
                }
                Ok(result)
            }
            _ => Err(CodecError::InvalidData(
                "Unsupported format conversion".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_size() {
        assert_eq!(SampleFormat::F32.sample_size(), 4);
        assert_eq!(SampleFormat::I16.sample_size(), 2);
        assert_eq!(SampleFormat::I32.sample_size(), 4);
        assert_eq!(SampleFormat::U8.sample_size(), 1);
    }

    #[test]
    fn test_channel_layout_count() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayout::Custom(4).channel_count(), 4);
    }

    #[test]
    fn test_audio_frame_creation() {
        let samples = vec![0u8; 1920 * 2 * 4]; // 1920 samples, stereo, f32
        let frame = AudioFrame::new(samples, 1920, 48000, 2, SampleFormat::F32);
        assert_eq!(frame.sample_count, 1920);
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.channels, 2);
        assert_eq!(frame.total_samples(), 3840);
        assert_eq!(frame.byte_size(), 15360);
    }

    #[test]
    fn test_audio_frame_duration() {
        let samples = vec![0u8; 480 * 2 * 4];
        let frame = AudioFrame::new(samples, 480, 48000, 2, SampleFormat::F32);
        assert!((frame.duration_seconds() - 0.01).abs() < 0.0001);
    }
}
