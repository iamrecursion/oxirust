//! MP3 frame header parsing and validation.
//!
//! This module handles parsing of MPEG-1/2 Layer I/II/III frame headers,
//! including bitrate, sample rate, channel mode, and frame size calculation.

use crate::{AudioError, AudioResult};

/// MPEG version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpegVersion {
    /// MPEG-1.
    Mpeg1,
    /// MPEG-2 LSF (Low Sampling Frequency).
    Mpeg2,
    /// MPEG-2.5 (unofficial extension).
    Mpeg25,
}

/// MPEG layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Layer I.
    I,
    /// Layer II.
    II,
    /// Layer III (MP3).
    III,
}

/// Channel mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMode {
    /// Stereo.
    Stereo,
    /// Joint stereo (intensity stereo and/or MS stereo).
    JointStereo(JointStereoMode),
    /// Dual channel (two independent mono channels).
    DualChannel,
    /// Single channel (mono).
    Mono,
}

/// Joint stereo mode extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointStereoMode {
    /// Intensity stereo is on/off.
    pub intensity: bool,
    /// MS stereo is on/off.
    pub ms_stereo: bool,
    /// Bound for intensity stereo (Layer I/II).
    pub bound: u8,
}

/// Emphasis mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emphasis {
    /// No emphasis.
    None,
    /// 50/15 microseconds emphasis.
    Ms5015,
    /// CCIT J.17.
    CcitJ17,
}

/// MP3 frame header.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    /// MPEG version.
    pub version: MpegVersion,
    /// Layer.
    pub layer: Layer,
    /// Protection (CRC) present.
    pub protection: bool,
    /// Bitrate in bits per second.
    pub bitrate: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Padding bit.
    pub padding: bool,
    /// Private bit.
    pub private: bool,
    /// Channel mode.
    pub mode: ChannelMode,
    /// Copyright bit.
    pub copyright: bool,
    /// Original bit.
    pub original: bool,
    /// Emphasis.
    pub emphasis: Emphasis,
    /// Frame size in bytes.
    pub frame_size: usize,
    /// Number of samples per channel.
    pub samples: usize,
}

impl FrameHeader {
    /// Parse frame header from 4 bytes.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(data: &[u8]) -> AudioResult<Self> {
        if data.len() < 4 {
            return Err(AudioError::InvalidData("Need 4 bytes for header".into()));
        }

        // Check sync word (11 bits, all 1s)
        if data[0] != 0xFF || (data[1] & 0xE0) != 0xE0 {
            return Err(AudioError::InvalidData("Invalid sync word".into()));
        }

        // Parse version (2 bits)
        let version = match (data[1] >> 3) & 0x03 {
            0 => MpegVersion::Mpeg25,
            2 => MpegVersion::Mpeg2,
            3 => MpegVersion::Mpeg1,
            _ => return Err(AudioError::InvalidData("Reserved MPEG version".into())),
        };

        // Parse layer (2 bits)
        let layer = match (data[1] >> 1) & 0x03 {
            1 => Layer::III,
            2 => Layer::II,
            3 => Layer::I,
            _ => return Err(AudioError::InvalidData("Reserved layer".into())),
        };

        // Protection bit (0 = protected by CRC)
        let protection = (data[1] & 0x01) == 0;

        // Parse bitrate (4 bits)
        let bitrate_index = (data[2] >> 4) & 0x0F;
        let bitrate = Self::get_bitrate(version, layer, bitrate_index)?;

        // Parse sample rate (2 bits)
        let samplerate_index = (data[2] >> 2) & 0x03;
        let sample_rate = Self::get_sample_rate(version, samplerate_index)?;

        // Padding bit
        let padding = (data[2] & 0x02) != 0;

        // Private bit
        let private = (data[2] & 0x01) != 0;

        // Parse channel mode (2 bits)
        let mode_value = (data[3] >> 6) & 0x03;
        let mode_ext = (data[3] >> 4) & 0x03;

        let mode = match mode_value {
            0 => ChannelMode::Stereo,
            1 => {
                let (intensity, ms_stereo, bound) = match layer {
                    Layer::III => {
                        // For Layer III, mode_ext bits indicate intensity/MS stereo
                        let intensity = (mode_ext & 0x01) != 0;
                        let ms_stereo = (mode_ext & 0x02) != 0;
                        (intensity, ms_stereo, 0)
                    }
                    _ => {
                        // For Layer I/II, mode_ext indicates bound
                        let bound = (mode_ext + 1) * 4;
                        (false, false, bound)
                    }
                };
                ChannelMode::JointStereo(JointStereoMode {
                    intensity,
                    ms_stereo,
                    bound,
                })
            }
            2 => ChannelMode::DualChannel,
            3 => ChannelMode::Mono,
            _ => unreachable!(),
        };

        // Copyright and original bits
        let copyright = (data[3] & 0x08) != 0;
        let original = (data[3] & 0x04) != 0;

        // Parse emphasis (2 bits)
        let emphasis = match data[3] & 0x03 {
            0 => Emphasis::None,
            1 => Emphasis::Ms5015,
            3 => Emphasis::CcitJ17,
            _ => return Err(AudioError::InvalidData("Reserved emphasis".into())),
        };

        // Calculate frame size
        let frame_size = Self::calculate_frame_size(version, layer, bitrate, sample_rate, padding)?;

        // Get samples per frame
        let samples = Self::get_samples_per_frame(version, layer);

        Ok(Self {
            version,
            layer,
            protection,
            bitrate,
            sample_rate,
            padding,
            private,
            mode,
            copyright,
            original,
            emphasis,
            frame_size,
            samples,
        })
    }

    /// Get bitrate from version, layer, and index.
    fn get_bitrate(version: MpegVersion, layer: Layer, index: u8) -> AudioResult<u32> {
        if index == 0x00 || index == 0x0F {
            return Err(AudioError::InvalidData("Invalid bitrate index".into()));
        }

        // Bitrate tables (kbps)
        let bitrate = match (version, layer) {
            // MPEG-1, Layer I
            (MpegVersion::Mpeg1, Layer::I) => [
                0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
            ][index as usize],
            // MPEG-1, Layer II
            (MpegVersion::Mpeg1, Layer::II) => [
                0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
            ][index as usize],
            // MPEG-1, Layer III
            (MpegVersion::Mpeg1, Layer::III) => [
                0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
            ][index as usize],
            // MPEG-2/2.5, Layer I
            (MpegVersion::Mpeg2 | MpegVersion::Mpeg25, Layer::I) => [
                0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
            ][index as usize],
            // MPEG-2/2.5, Layer II & III
            (MpegVersion::Mpeg2 | MpegVersion::Mpeg25, Layer::II | Layer::III) => {
                [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160][index as usize]
            }
        };

        Ok(bitrate * 1000)
    }

    /// Get sample rate from version and index.
    fn get_sample_rate(version: MpegVersion, index: u8) -> AudioResult<u32> {
        if index == 0x03 {
            return Err(AudioError::InvalidData("Reserved sample rate".into()));
        }

        let sample_rate = match version {
            MpegVersion::Mpeg1 => [44100, 48000, 32000][index as usize],
            MpegVersion::Mpeg2 => [22050, 24000, 16000][index as usize],
            MpegVersion::Mpeg25 => [11025, 12000, 8000][index as usize],
        };

        Ok(sample_rate)
    }

    /// Calculate frame size in bytes.
    fn calculate_frame_size(
        _version: MpegVersion,
        layer: Layer,
        bitrate: u32,
        sample_rate: u32,
        padding: bool,
    ) -> AudioResult<usize> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidData("Zero sample rate".into()));
        }

        let padding_size = if padding {
            match layer {
                Layer::I => 4,
                Layer::II | Layer::III => 1,
            }
        } else {
            0
        };

        let frame_size = match layer {
            Layer::I => {
                // Layer I: (12 * bitrate / sample_rate + padding) * 4
                ((12 * bitrate / sample_rate) + padding_size) as usize
            }
            Layer::II | Layer::III => {
                // Layer II/III: 144 * bitrate / sample_rate + padding
                ((144 * bitrate / sample_rate) + padding_size) as usize
            }
        };

        Ok(frame_size)
    }

    /// Get number of samples per frame.
    const fn get_samples_per_frame(version: MpegVersion, layer: Layer) -> usize {
        match layer {
            Layer::I => 384,
            Layer::II => 1152,
            Layer::III => match version {
                MpegVersion::Mpeg1 => 1152,
                MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
            },
        }
    }

    /// Get number of channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        match self.mode {
            ChannelMode::Mono => 1,
            _ => 2,
        }
    }

    /// Check if frame is mono.
    #[must_use]
    pub const fn is_mono(&self) -> bool {
        matches!(self.mode, ChannelMode::Mono)
    }

    /// Check if frame uses joint stereo.
    #[must_use]
    pub const fn is_joint_stereo(&self) -> bool {
        matches!(self.mode, ChannelMode::JointStereo(_))
    }

    /// Get frame duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration(&self) -> f64 {
        self.samples as f64 / f64::from(self.sample_rate)
    }
}

/// Find next frame sync in data.
///
/// Returns the offset of the next frame sync, or `None` if not found.
#[must_use]
pub fn find_sync(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0 {
            return Some(i);
        }
    }
    None
}

/// Validate that frame header is consistent with previous header.
///
/// Used for VBR detection and stream validation.
#[must_use]
pub fn is_compatible(h1: &FrameHeader, h2: &FrameHeader) -> bool {
    h1.version == h2.version
        && h1.layer == h2.layer
        && h1.sample_rate == h2.sample_rate
        && h1.mode == h2.mode
}
