//! FLAC frame parsing.
//!
//! A FLAC frame contains one block of audio samples. Each frame has:
//! - Frame header (sync code, blocking strategy, sample rate, channels, etc.)
//! - One subframe per channel
//! - Frame footer (CRC-16)
//!
//! # Frame Header Structure
//!
//! - Sync code (14 bits: 0x3FFE)
//! - Reserved (1 bit)
//! - Blocking strategy (1 bit)
//! - Block size (4 bits)
//! - Sample rate (4 bits)
//! - Channel assignment (4 bits)
//! - Sample size (3 bits)
//! - Reserved (1 bit)
//! - Frame/sample number (8-56 bits, UTF-8 coded)
//! - Optional block size (8/16 bits)
//! - Optional sample rate (8/16 bits)
//! - CRC-8

#![forbid(unsafe_code)]

use crate::AudioError;

/// FLAC sync code (14 bits).
pub const SYNC_CODE: u16 = 0x3FFE;

/// Blocking strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockingStrategy {
    /// Fixed block size (frame number in header).
    #[default]
    Fixed,
    /// Variable block size (sample number in header).
    Variable,
}

impl BlockingStrategy {
    /// Create from bit value.
    #[must_use]
    pub fn from_bit(bit: bool) -> Self {
        if bit {
            BlockingStrategy::Variable
        } else {
            BlockingStrategy::Fixed
        }
    }
}

/// Channel assignment in FLAC frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAssignment {
    /// Independent channels (1-8).
    Independent(u8),
    /// Left/side stereo.
    LeftSide,
    /// Right/side stereo.
    RightSide,
    /// Mid/side stereo.
    MidSide,
}

impl Default for ChannelAssignment {
    fn default() -> Self {
        ChannelAssignment::Independent(1)
    }
}

impl ChannelAssignment {
    /// Create from raw channel assignment value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0..=7 => Some(ChannelAssignment::Independent(value + 1)),
            8 => Some(ChannelAssignment::LeftSide),
            9 => Some(ChannelAssignment::RightSide),
            10 => Some(ChannelAssignment::MidSide),
            _ => None,
        }
    }

    /// Get number of channels.
    #[must_use]
    pub fn channels(self) -> u8 {
        match self {
            ChannelAssignment::Independent(n) => n,
            ChannelAssignment::LeftSide
            | ChannelAssignment::RightSide
            | ChannelAssignment::MidSide => 2,
        }
    }

    /// Check if this is a stereo decorrelation mode.
    #[must_use]
    pub fn is_stereo_decorrelated(self) -> bool {
        matches!(
            self,
            ChannelAssignment::LeftSide | ChannelAssignment::RightSide | ChannelAssignment::MidSide
        )
    }

    /// Get the side channel index for stereo decorrelation.
    #[must_use]
    pub fn side_channel(self) -> Option<usize> {
        match self {
            ChannelAssignment::LeftSide | ChannelAssignment::MidSide => Some(1),
            ChannelAssignment::RightSide => Some(0),
            ChannelAssignment::Independent(_) => None,
        }
    }
}

/// Sample size in bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SampleSize {
    /// Get from STREAMINFO.
    #[default]
    FromStreamInfo,
    /// 8 bits per sample.
    Bits8,
    /// 12 bits per sample.
    Bits12,
    /// 16 bits per sample.
    Bits16,
    /// 20 bits per sample.
    Bits20,
    /// 24 bits per sample.
    Bits24,
    /// 32 bits per sample.
    Bits32,
}

impl SampleSize {
    /// Create from raw value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(SampleSize::FromStreamInfo),
            1 => Some(SampleSize::Bits8),
            2 => Some(SampleSize::Bits12),
            // 3 is reserved
            4 => Some(SampleSize::Bits16),
            5 => Some(SampleSize::Bits20),
            6 => Some(SampleSize::Bits24),
            7 => Some(SampleSize::Bits32),
            _ => None,
        }
    }

    /// Get bits per sample.
    #[must_use]
    pub fn bits(self) -> Option<u8> {
        match self {
            SampleSize::FromStreamInfo => None,
            SampleSize::Bits8 => Some(8),
            SampleSize::Bits12 => Some(12),
            SampleSize::Bits16 => Some(16),
            SampleSize::Bits20 => Some(20),
            SampleSize::Bits24 => Some(24),
            SampleSize::Bits32 => Some(32),
        }
    }
}

/// Block size specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum BlockSize {
    /// Reserved (invalid).
    #[default]
    Reserved,
    /// 192 samples.
    Samples192,
    /// 576 * 2^n samples (n = 0..5).
    Samples576Mult(u8),
    /// 256 * 2^n samples (n = 0..7).
    Samples256Mult(u8),
    /// 8-bit value from end of header.
    GetFromEnd8Bit,
    /// 16-bit value from end of header.
    GetFromEnd16Bit,
}

impl BlockSize {
    /// Create from raw value.
    #[must_use]
    pub fn from_value(value: u8) -> Self {
        match value {
            1 => BlockSize::Samples192,
            2..=5 => BlockSize::Samples576Mult(value - 2),
            6 => BlockSize::GetFromEnd8Bit,
            7 => BlockSize::GetFromEnd16Bit,
            8..=15 => BlockSize::Samples256Mult(value - 8),
            // 0 and any other values are reserved
            _ => BlockSize::Reserved,
        }
    }

    /// Get fixed block size if known.
    #[must_use]
    pub fn fixed_size(self) -> Option<u32> {
        match self {
            BlockSize::Samples192 => Some(192),
            BlockSize::Samples576Mult(n) => Some(576 * (1 << n)),
            BlockSize::Samples256Mult(n) => Some(256 * (1 << n)),
            BlockSize::Reserved | BlockSize::GetFromEnd8Bit | BlockSize::GetFromEnd16Bit => None,
        }
    }

    /// Check if block size needs to be read from header end.
    #[must_use]
    pub fn needs_extra_bytes(self) -> bool {
        matches!(self, BlockSize::GetFromEnd8Bit | BlockSize::GetFromEnd16Bit)
    }
}

/// Sample rate specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
pub enum SampleRateSpec {
    /// Get from STREAMINFO.
    #[default]
    FromStreamInfo,
    /// 88.2 kHz.
    Rate88200,
    /// 176.4 kHz.
    Rate176400,
    /// 192 kHz.
    Rate192000,
    /// 8 kHz.
    Rate8000,
    /// 16 kHz.
    Rate16000,
    /// 22.05 kHz.
    Rate22050,
    /// 24 kHz.
    Rate24000,
    /// 32 kHz.
    Rate32000,
    /// 44.1 kHz.
    Rate44100,
    /// 48 kHz.
    Rate48000,
    /// 96 kHz.
    Rate96000,
    /// 8-bit value in kHz from end of header.
    GetFromEnd8BitKHz,
    /// 16-bit value in Hz from end of header.
    GetFromEnd16BitHz,
    /// 16-bit value in 10Hz from end of header.
    GetFromEnd16BitTensHz,
    /// Invalid.
    Invalid,
}

impl SampleRateSpec {
    /// Create from raw value.
    #[must_use]
    pub fn from_value(value: u8) -> Self {
        match value {
            0 => SampleRateSpec::FromStreamInfo,
            1 => SampleRateSpec::Rate88200,
            2 => SampleRateSpec::Rate176400,
            3 => SampleRateSpec::Rate192000,
            4 => SampleRateSpec::Rate8000,
            5 => SampleRateSpec::Rate16000,
            6 => SampleRateSpec::Rate22050,
            7 => SampleRateSpec::Rate24000,
            8 => SampleRateSpec::Rate32000,
            9 => SampleRateSpec::Rate44100,
            10 => SampleRateSpec::Rate48000,
            11 => SampleRateSpec::Rate96000,
            12 => SampleRateSpec::GetFromEnd8BitKHz,
            13 => SampleRateSpec::GetFromEnd16BitHz,
            14 => SampleRateSpec::GetFromEnd16BitTensHz,
            // 15 and other values are invalid
            _ => SampleRateSpec::Invalid,
        }
    }

    /// Get fixed sample rate if known.
    #[must_use]
    pub fn fixed_rate(self) -> Option<u32> {
        match self {
            SampleRateSpec::Rate88200 => Some(88200),
            SampleRateSpec::Rate176400 => Some(176_400),
            SampleRateSpec::Rate192000 => Some(192_000),
            SampleRateSpec::Rate8000 => Some(8000),
            SampleRateSpec::Rate16000 => Some(16000),
            SampleRateSpec::Rate22050 => Some(22050),
            SampleRateSpec::Rate24000 => Some(24000),
            SampleRateSpec::Rate32000 => Some(32000),
            SampleRateSpec::Rate44100 => Some(44100),
            SampleRateSpec::Rate48000 => Some(48000),
            SampleRateSpec::Rate96000 => Some(96000),
            _ => None,
        }
    }
}

/// FLAC frame header.
#[derive(Debug, Clone, Default)]
pub struct FrameHeader {
    /// Blocking strategy.
    pub blocking_strategy: BlockingStrategy,
    /// Block size in samples.
    pub block_size: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel assignment.
    pub channel_assignment: ChannelAssignment,
    /// Sample size in bits.
    pub sample_size: SampleSize,
    /// Bits per sample (resolved from `sample_size` or STREAMINFO).
    pub bits_per_sample: u8,
    /// Frame number (if fixed blocking).
    pub frame_number: Option<u32>,
    /// Sample number (if variable blocking).
    pub sample_number: Option<u64>,
    /// CRC-8 of header.
    pub crc8: u8,
}

impl FrameHeader {
    /// Maximum block size in FLAC.
    pub const MAX_BLOCK_SIZE: u32 = 65535;

    /// Minimum block size in FLAC.
    pub const MIN_BLOCK_SIZE: u32 = 16;

    /// Parse frame header from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn parse(data: &[u8], streaminfo_bps: u8) -> Result<(Self, usize), AudioError> {
        if data.len() < 4 {
            return Err(AudioError::InvalidData("Frame header too short".into()));
        }

        // Check sync code
        let sync = u16::from_be_bytes([data[0], data[1]]) >> 2;
        if sync != SYNC_CODE {
            return Err(AudioError::InvalidData("Invalid sync code".into()));
        }

        // Reserved bit must be 0
        if (data[1] & 0x02) != 0 {
            return Err(AudioError::InvalidData("Reserved bit set".into()));
        }

        let blocking_strategy = BlockingStrategy::from_bit((data[1] & 0x01) != 0);
        let block_size_spec = BlockSize::from_value((data[2] >> 4) & 0x0F);
        let sample_rate_spec = SampleRateSpec::from_value(data[2] & 0x0F);
        let channel_assignment = ChannelAssignment::from_value((data[3] >> 4) & 0x0F)
            .ok_or_else(|| AudioError::InvalidData("Invalid channel assignment".into()))?;
        let sample_size = SampleSize::from_value((data[3] >> 1) & 0x07)
            .ok_or_else(|| AudioError::InvalidData("Invalid sample size".into()))?;

        // Reserved bit must be 0
        if (data[3] & 0x01) != 0 {
            return Err(AudioError::InvalidData("Reserved bit set".into()));
        }

        let mut offset = 4;

        // Parse UTF-8 coded frame/sample number
        let (frame_number, sample_number) = if blocking_strategy == BlockingStrategy::Fixed {
            let (num, bytes) = Self::parse_utf8_u32(&data[offset..])?;
            offset += bytes;
            (Some(num), None)
        } else {
            let (num, bytes) = Self::parse_utf8_u64(&data[offset..])?;
            offset += bytes;
            (None, Some(num))
        };

        // Parse optional block size
        let block_size = if let Some(size) = block_size_spec.fixed_size() {
            size
        } else {
            match block_size_spec {
                BlockSize::GetFromEnd8Bit => {
                    if offset >= data.len() {
                        return Err(AudioError::InvalidData("Missing block size byte".into()));
                    }
                    let size = u32::from(data[offset]) + 1;
                    offset += 1;
                    size
                }
                BlockSize::GetFromEnd16Bit => {
                    if offset + 1 >= data.len() {
                        return Err(AudioError::InvalidData("Missing block size bytes".into()));
                    }
                    let size = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]])) + 1;
                    offset += 2;
                    size
                }
                _ => {
                    return Err(AudioError::InvalidData("Invalid block size".into()));
                }
            }
        };

        // Parse optional sample rate
        let sample_rate = if let Some(rate) = sample_rate_spec.fixed_rate() {
            rate
        } else {
            match sample_rate_spec {
                SampleRateSpec::GetFromEnd8BitKHz => {
                    if offset >= data.len() {
                        return Err(AudioError::InvalidData("Missing sample rate byte".into()));
                    }
                    let rate = u32::from(data[offset]) * 1000;
                    offset += 1;
                    rate
                }
                SampleRateSpec::GetFromEnd16BitHz => {
                    if offset + 1 >= data.len() {
                        return Err(AudioError::InvalidData("Missing sample rate bytes".into()));
                    }
                    let rate = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]]));
                    offset += 2;
                    rate
                }
                SampleRateSpec::GetFromEnd16BitTensHz => {
                    if offset + 1 >= data.len() {
                        return Err(AudioError::InvalidData("Missing sample rate bytes".into()));
                    }
                    let rate = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]])) * 10;
                    offset += 2;
                    rate
                }
                SampleRateSpec::Invalid => {
                    return Err(AudioError::InvalidData("Invalid sample rate".into()));
                }
                // FromStreamInfo and others: will be filled from STREAMINFO
                _ => 0,
            }
        };

        // CRC-8
        if offset >= data.len() {
            return Err(AudioError::InvalidData("Missing CRC-8".into()));
        }
        let crc8 = data[offset];
        offset += 1;

        let bits_per_sample = sample_size.bits().unwrap_or(streaminfo_bps);

        Ok((
            Self {
                blocking_strategy,
                block_size,
                sample_rate,
                channel_assignment,
                sample_size,
                bits_per_sample,
                frame_number,
                sample_number,
                crc8,
            },
            offset,
        ))
    }

    /// Parse UTF-8 coded u32 (for frame number).
    fn parse_utf8_u32(data: &[u8]) -> Result<(u32, usize), AudioError> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("Empty UTF-8 data".into()));
        }

        let first = data[0];
        let (value, bytes) = if first & 0x80 == 0 {
            (u32::from(first), 1)
        } else if first & 0xE0 == 0xC0 {
            if data.len() < 2 {
                return Err(AudioError::InvalidData("Truncated UTF-8".into()));
            }
            let v = ((u32::from(first) & 0x1F) << 6) | (u32::from(data[1]) & 0x3F);
            (v, 2)
        } else if first & 0xF0 == 0xE0 {
            if data.len() < 3 {
                return Err(AudioError::InvalidData("Truncated UTF-8".into()));
            }
            let v = ((u32::from(first) & 0x0F) << 12)
                | ((u32::from(data[1]) & 0x3F) << 6)
                | (u32::from(data[2]) & 0x3F);
            (v, 3)
        } else if first & 0xF8 == 0xF0 {
            if data.len() < 4 {
                return Err(AudioError::InvalidData("Truncated UTF-8".into()));
            }
            let v = ((u32::from(first) & 0x07) << 18)
                | ((u32::from(data[1]) & 0x3F) << 12)
                | ((u32::from(data[2]) & 0x3F) << 6)
                | (u32::from(data[3]) & 0x3F);
            (v, 4)
        } else {
            return Err(AudioError::InvalidData("Invalid UTF-8 lead byte".into()));
        };

        Ok((value, bytes))
    }

    /// Parse UTF-8 coded u64 (for sample number).
    fn parse_utf8_u64(data: &[u8]) -> Result<(u64, usize), AudioError> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("Empty UTF-8 data".into()));
        }

        let first = data[0];
        let leading_ones = first.leading_ones() as usize;

        let bytes = if leading_ones == 0 {
            1
        } else {
            leading_ones.min(7)
        };

        if data.len() < bytes {
            return Err(AudioError::InvalidData("Truncated UTF-8".into()));
        }

        let mut value = u64::from(first & (0xFF >> (leading_ones + 1)));
        for byte in data.iter().take(bytes).skip(1) {
            value = (value << 6) | u64::from(byte & 0x3F);
        }

        Ok((value, bytes))
    }

    /// Get number of channels.
    #[must_use]
    pub fn channels(&self) -> u8 {
        self.channel_assignment.channels()
    }
}

/// FLAC frame containing header and subframes.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct FlacFrame {
    /// Frame header.
    pub header: FrameHeader,
    /// Decoded samples per channel.
    pub samples: Vec<Vec<i32>>,
    /// CRC-16 of entire frame.
    pub crc16: u16,
}

impl FlacFrame {
    /// Create a new empty frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create frame with header.
    #[must_use]
    pub fn with_header(header: FrameHeader) -> Self {
        let channels = header.channels() as usize;
        Self {
            header,
            samples: vec![Vec::new(); channels],
            crc16: 0,
        }
    }

    /// Get number of samples per channel.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.header.block_size as usize
    }

    /// Get number of channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.header.channels() as usize
    }

    /// Apply stereo decorrelation if needed.
    pub fn apply_decorrelation(&mut self) {
        if self.samples.len() != 2 {
            return;
        }

        let block_size = self.sample_count();
        if self.samples[0].len() != block_size || self.samples[1].len() != block_size {
            return;
        }

        match self.header.channel_assignment {
            ChannelAssignment::LeftSide => {
                // Left is left, side = left - right, so right = left - side
                for i in 0..block_size {
                    let left = self.samples[0][i];
                    let side = self.samples[1][i];
                    self.samples[1][i] = left - side;
                }
            }
            ChannelAssignment::RightSide => {
                // Side is stored in channel 0, right in channel 1
                // side = left - right, so left = side + right
                for i in 0..block_size {
                    let side = self.samples[0][i];
                    let right = self.samples[1][i];
                    self.samples[0][i] = side + right;
                }
            }
            ChannelAssignment::MidSide => {
                // mid = (left + right) / 2, side = left - right
                // left = mid + side/2, right = mid - side/2
                for i in 0..block_size {
                    let mid = self.samples[0][i];
                    let side = self.samples[1][i];
                    // Use integer math to avoid rounding issues
                    let left = mid + (side >> 1) + (side & 1);
                    let right = mid - (side >> 1);
                    self.samples[0][i] = left;
                    self.samples[1][i] = right;
                }
            }
            ChannelAssignment::Independent(_) => {
                // No decorrelation needed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocking_strategy() {
        assert_eq!(BlockingStrategy::from_bit(false), BlockingStrategy::Fixed);
        assert_eq!(BlockingStrategy::from_bit(true), BlockingStrategy::Variable);
    }

    #[test]
    fn test_channel_assignment() {
        assert_eq!(
            ChannelAssignment::from_value(0),
            Some(ChannelAssignment::Independent(1))
        );
        assert_eq!(
            ChannelAssignment::from_value(1),
            Some(ChannelAssignment::Independent(2))
        );
        assert_eq!(
            ChannelAssignment::from_value(7),
            Some(ChannelAssignment::Independent(8))
        );
        assert_eq!(
            ChannelAssignment::from_value(8),
            Some(ChannelAssignment::LeftSide)
        );
        assert_eq!(
            ChannelAssignment::from_value(9),
            Some(ChannelAssignment::RightSide)
        );
        assert_eq!(
            ChannelAssignment::from_value(10),
            Some(ChannelAssignment::MidSide)
        );
        assert_eq!(ChannelAssignment::from_value(11), None);
    }

    #[test]
    fn test_channel_assignment_channels() {
        assert_eq!(ChannelAssignment::Independent(1).channels(), 1);
        assert_eq!(ChannelAssignment::Independent(6).channels(), 6);
        assert_eq!(ChannelAssignment::LeftSide.channels(), 2);
        assert_eq!(ChannelAssignment::MidSide.channels(), 2);
    }

    #[test]
    fn test_sample_size() {
        assert_eq!(SampleSize::from_value(0), Some(SampleSize::FromStreamInfo));
        assert_eq!(SampleSize::from_value(1), Some(SampleSize::Bits8));
        assert_eq!(SampleSize::from_value(4), Some(SampleSize::Bits16));
        assert_eq!(SampleSize::from_value(6), Some(SampleSize::Bits24));
        assert_eq!(SampleSize::from_value(3), None); // Reserved
    }

    #[test]
    fn test_sample_size_bits() {
        assert_eq!(SampleSize::Bits8.bits(), Some(8));
        assert_eq!(SampleSize::Bits16.bits(), Some(16));
        assert_eq!(SampleSize::Bits24.bits(), Some(24));
        assert_eq!(SampleSize::FromStreamInfo.bits(), None);
    }

    #[test]
    fn test_block_size() {
        assert_eq!(BlockSize::from_value(1).fixed_size(), Some(192));
        assert_eq!(BlockSize::from_value(2).fixed_size(), Some(576));
        assert_eq!(BlockSize::from_value(8).fixed_size(), Some(256));
        assert!(BlockSize::from_value(6).needs_extra_bytes());
        assert!(BlockSize::from_value(7).needs_extra_bytes());
    }

    #[test]
    fn test_sample_rate_spec() {
        assert_eq!(SampleRateSpec::from_value(9).fixed_rate(), Some(44100));
        assert_eq!(SampleRateSpec::from_value(10).fixed_rate(), Some(48000));
        assert_eq!(SampleRateSpec::from_value(11).fixed_rate(), Some(96000));
        assert_eq!(SampleRateSpec::from_value(0).fixed_rate(), None);
    }

    #[test]
    fn test_flac_frame() {
        let header = FrameHeader {
            block_size: 4096,
            channel_assignment: ChannelAssignment::Independent(2),
            bits_per_sample: 16,
            ..Default::default()
        };
        let frame = FlacFrame::with_header(header);
        assert_eq!(frame.sample_count(), 4096);
        assert_eq!(frame.channel_count(), 2);
    }

    #[test]
    fn test_stereo_decorrelation_side_channel() {
        assert_eq!(ChannelAssignment::LeftSide.side_channel(), Some(1));
        assert_eq!(ChannelAssignment::RightSide.side_channel(), Some(0));
        assert_eq!(ChannelAssignment::MidSide.side_channel(), Some(1));
        assert_eq!(ChannelAssignment::Independent(2).side_channel(), None);
    }
}
