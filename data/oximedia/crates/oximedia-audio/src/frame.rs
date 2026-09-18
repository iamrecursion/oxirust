//! Audio frame types.

use bytes::Bytes;
use oximedia_core::{Rational, SampleFormat, Timestamp};

/// Decoded audio frame.
#[derive(Clone, Debug)]
pub struct AudioFrame {
    /// Sample format.
    pub format: SampleFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel layout.
    pub channels: ChannelLayout,
    /// Sample data.
    pub samples: AudioBuffer,
    /// Presentation timestamp.
    pub timestamp: Timestamp,
}

impl AudioFrame {
    /// Create a new audio frame.
    #[must_use]
    pub fn new(format: SampleFormat, sample_rate: u32, channels: ChannelLayout) -> Self {
        Self {
            format,
            sample_rate,
            channels,
            samples: AudioBuffer::Interleaved(Bytes::new()),
            timestamp: Timestamp::new(0, Rational::new(1, i64::from(sample_rate))),
        }
    }

    /// Get number of samples per channel.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        let bytes = match &self.samples {
            AudioBuffer::Interleaved(data) => data.len(),
            AudioBuffer::Planar(planes) => planes.first().map_or(0, Bytes::len),
        };
        let bytes_per_sample = self.format.bytes_per_sample();
        let channel_count = self.channels.count();
        if bytes_per_sample == 0 || channel_count == 0 {
            0
        } else if self.samples.is_planar() {
            bytes / bytes_per_sample
        } else {
            bytes / (bytes_per_sample * channel_count)
        }
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let result = self.sample_count() as f64 / f64::from(self.sample_rate);
            result
        }
    }

    /// Check if frame is silent.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        match &self.samples {
            AudioBuffer::Interleaved(data) => data.iter().all(|&b| b == 0 || b == 128),
            AudioBuffer::Planar(planes) => {
                planes.iter().all(|p| p.iter().all(|&b| b == 0 || b == 128))
            }
        }
    }
}

/// Audio sample buffer.
#[derive(Clone, Debug)]
pub enum AudioBuffer {
    /// Interleaved samples (LRLRLR...).
    Interleaved(Bytes),
    /// Planar samples (LLL...RRR...).
    Planar(Vec<Bytes>),
}

impl AudioBuffer {
    /// Check if buffer is planar.
    #[must_use]
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::Planar(_))
    }

    /// Get total size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Interleaved(data) => data.len(),
            Self::Planar(planes) => planes.iter().map(Bytes::len).sum(),
        }
    }
}

/// Channel layout.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum ChannelLayout {
    /// Mono (1 channel).
    #[default]
    Mono,
    /// Stereo (2 channels: L, R).
    Stereo,
    /// 2.1 surround (3 channels: L, R, LFE).
    Surround21,
    /// 5.1 surround (6 channels: L, R, C, LFE, Ls, Rs).
    Surround51,
    /// 7.1 surround (8 channels: L, R, C, LFE, Ls, Rs, Lb, Rb).
    Surround71,
    /// Custom channel layout.
    Custom(Vec<Channel>),
}

impl ChannelLayout {
    /// Get channel count.
    #[must_use]
    pub fn count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround21 => 3,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Custom(channels) => channels.len(),
        }
    }

    /// Create layout from channel count.
    #[must_use]
    pub fn from_count(count: usize) -> Self {
        match count {
            1 => Self::Mono,
            2 => Self::Stereo,
            3 => Self::Surround21,
            6 => Self::Surround51,
            8 => Self::Surround71,
            _ => Self::Custom(vec![Channel::Unknown; count]),
        }
    }
}

/// Individual channel identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// Unknown/unspecified.
    Unknown,
    /// Front left.
    FrontLeft,
    /// Front right.
    FrontRight,
    /// Front center.
    FrontCenter,
    /// Low frequency effects.
    Lfe,
    /// Side/surround left.
    SideLeft,
    /// Side/surround right.
    SideRight,
    /// Back/rear left.
    BackLeft,
    /// Back/rear right.
    BackRight,
    /// Back center.
    BackCenter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout() {
        assert_eq!(ChannelLayout::Mono.count(), 1);
        assert_eq!(ChannelLayout::Stereo.count(), 2);
        assert_eq!(ChannelLayout::Surround51.count(), 6);
    }

    #[test]
    fn test_from_count() {
        assert_eq!(ChannelLayout::from_count(1), ChannelLayout::Mono);
        assert_eq!(ChannelLayout::from_count(2), ChannelLayout::Stereo);
    }

    #[test]
    fn test_audio_buffer() {
        let buffer = AudioBuffer::Interleaved(Bytes::from_static(&[0, 1, 2, 3]));
        assert!(!buffer.is_planar());
        assert_eq!(buffer.size(), 4);
    }
}
