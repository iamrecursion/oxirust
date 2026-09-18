//! Codec traits for video encoding and decoding.

use crate::{CodecResult, VideoFrame};
use oximedia_core::{CodecId, PixelFormat, Rational};

/// Video decoder trait.
///
/// Implements a push-pull decoding model:
/// 1. Send compressed packets with [`send_packet`](VideoDecoder::send_packet)
/// 2. Receive decoded frames with [`receive_frame`](VideoDecoder::receive_frame)
///
/// # Example
///
/// ```ignore
/// while let Some(packet) = demuxer.read_packet()? {
///     decoder.send_packet(&packet)?;
///     while let Some(frame) = decoder.receive_frame()? {
///         process_frame(frame);
///     }
/// }
/// decoder.flush()?;
/// while let Some(frame) = decoder.receive_frame()? {
///     process_frame(frame);
/// }
/// ```
pub trait VideoDecoder: Send {
    /// Get codec identifier.
    fn codec(&self) -> CodecId;

    /// Send a compressed packet to the decoder.
    ///
    /// # Errors
    ///
    /// Returns error if the packet is invalid or decoder is in error state.
    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()>;

    /// Receive a decoded frame.
    ///
    /// Returns `Ok(None)` if more data is needed.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>>;

    /// Flush the decoder.
    ///
    /// Call this after all packets have been sent to retrieve remaining frames.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    fn flush(&mut self) -> CodecResult<()>;

    /// Reset the decoder state.
    fn reset(&mut self);

    /// Get decoder output format.
    fn output_format(&self) -> Option<PixelFormat>;

    /// Get decoded frame dimensions.
    fn dimensions(&self) -> Option<(u32, u32)>;
}

/// Video encoder trait.
///
/// Implements a push-pull encoding model:
/// 1. Send raw frames with [`send_frame`](VideoEncoder::send_frame)
/// 2. Receive compressed packets with [`receive_packet`](VideoEncoder::receive_packet)
pub trait VideoEncoder: Send {
    /// Get codec identifier.
    fn codec(&self) -> CodecId;

    /// Send a raw frame to the encoder.
    ///
    /// # Errors
    ///
    /// Returns error if the frame format is invalid.
    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()>;

    /// Receive an encoded packet.
    ///
    /// Returns `Ok(None)` if more frames are needed.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>>;

    /// Flush the encoder.
    ///
    /// Call this after all frames have been sent to retrieve remaining packets.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    fn flush(&mut self) -> CodecResult<()>;

    /// Get encoder configuration.
    fn config(&self) -> &EncoderConfig;
}

/// Encoded packet output from encoder.
#[derive(Clone, Debug)]
pub struct EncodedPacket {
    /// Compressed data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Is keyframe.
    pub keyframe: bool,
    /// Duration in timebase units.
    pub duration: Option<i64>,
}

/// Decoder configuration.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    /// Codec to decode.
    pub codec: CodecId,
    /// Extra data (codec-specific).
    pub extradata: Option<Vec<u8>>,
    /// Number of decoder threads (0 = auto).
    pub threads: usize,
    /// Enable low-latency mode.
    pub low_latency: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::Av1,
            extradata: None,
            threads: 0,
            low_latency: false,
        }
    }
}

/// Encoder configuration.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Target codec.
    pub codec: CodecId,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Input pixel format.
    pub pixel_format: PixelFormat,
    /// Frame rate.
    pub framerate: Rational,
    /// Bitrate mode.
    pub bitrate: BitrateMode,
    /// Encoder preset.
    pub preset: EncoderPreset,
    /// Profile (codec-specific).
    pub profile: Option<String>,
    /// Keyframe interval.
    pub keyint: u32,
    /// Number of encoder threads (0 = auto).
    pub threads: usize,
    /// Timebase for output packets.
    pub timebase: Rational,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::Av1,
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Yuv420p,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Crf(28.0),
            preset: EncoderPreset::Medium,
            profile: None,
            keyint: 250,
            threads: 0,
            timebase: Rational::new(1, 1000),
        }
    }
}

impl EncoderConfig {
    /// Create AV1 encoder config.
    #[must_use]
    pub fn av1(width: u32, height: u32) -> Self {
        Self {
            codec: CodecId::Av1,
            width,
            height,
            ..Default::default()
        }
    }

    /// Create VP9 encoder config.
    #[must_use]
    pub fn vp9(width: u32, height: u32) -> Self {
        Self {
            codec: CodecId::Vp9,
            width,
            height,
            ..Default::default()
        }
    }

    /// Create VP8 encoder config.
    #[must_use]
    pub fn vp8(width: u32, height: u32) -> Self {
        Self {
            codec: CodecId::Vp8,
            width,
            height,
            ..Default::default()
        }
    }

    /// Set CRF quality.
    #[must_use]
    pub fn with_crf(mut self, crf: f32) -> Self {
        self.bitrate = BitrateMode::Crf(crf);
        self
    }

    /// Set target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = BitrateMode::Cbr(bitrate);
        self
    }

    /// Set encoder preset.
    #[must_use]
    pub fn with_preset(mut self, preset: EncoderPreset) -> Self {
        self.preset = preset;
        self
    }
}

/// Bitrate control mode.
#[derive(Clone, Debug)]
pub enum BitrateMode {
    /// Constant bitrate (bits/sec).
    Cbr(u64),
    /// Variable bitrate with target and maximum.
    Vbr {
        /// Target bitrate.
        target: u64,
        /// Maximum bitrate.
        max: u64,
    },
    /// Constant quality (CRF value, lower = better).
    Crf(f32),
    /// Lossless encoding.
    Lossless,
}

/// Encoder preset (speed vs quality tradeoff).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EncoderPreset {
    /// Fastest encoding, lowest quality.
    Ultrafast,
    /// Very fast encoding.
    Superfast,
    /// Fast encoding.
    Veryfast,
    /// Faster than medium.
    Faster,
    /// Fast encoding.
    Fast,
    /// Balanced speed and quality.
    #[default]
    Medium,
    /// Slower, better quality.
    Slow,
    /// Even slower, better quality.
    Slower,
    /// Very slow, high quality.
    Veryslow,
    /// Maximum quality, slowest.
    Placebo,
}

impl EncoderPreset {
    /// Get speed value (0-10, 0 = slowest).
    #[must_use]
    pub fn speed(&self) -> u8 {
        match self {
            Self::Ultrafast => 10,
            Self::Superfast => 9,
            Self::Veryfast => 8,
            Self::Faster => 7,
            Self::Fast => 6,
            Self::Medium => 5,
            Self::Slow => 4,
            Self::Slower => 3,
            Self::Veryslow => 2,
            Self::Placebo => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_builder() {
        let config = EncoderConfig::av1(1920, 1080)
            .with_crf(24.0)
            .with_preset(EncoderPreset::Slow);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.preset, EncoderPreset::Slow);
        assert!(
            matches!(config.bitrate, BitrateMode::Crf(crf) if (crf - 24.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn test_preset_speed() {
        assert_eq!(EncoderPreset::Ultrafast.speed(), 10);
        assert_eq!(EncoderPreset::Medium.speed(), 5);
        assert_eq!(EncoderPreset::Placebo.speed(), 0);
    }
}
