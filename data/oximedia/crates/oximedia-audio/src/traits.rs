//! Audio codec traits.

use crate::{AudioFrame, AudioResult, ChannelLayout};
use oximedia_core::{CodecId, SampleFormat};

/// Audio decoder trait.
pub trait AudioDecoder: Send {
    /// Get codec identifier.
    fn codec(&self) -> CodecId;

    /// Send compressed packet.
    ///
    /// # Errors
    ///
    /// Returns error if packet is invalid.
    fn send_packet(&mut self, data: &[u8], pts: i64) -> AudioResult<()>;

    /// Receive decoded frame.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>>;

    /// Flush decoder.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    fn flush(&mut self) -> AudioResult<()>;

    /// Reset decoder state.
    fn reset(&mut self);

    /// Get output sample format.
    fn output_format(&self) -> Option<SampleFormat>;

    /// Get sample rate.
    fn sample_rate(&self) -> Option<u32>;

    /// Get channel layout.
    fn channel_layout(&self) -> Option<ChannelLayout>;
}

/// Audio encoder trait.
pub trait AudioEncoder: Send {
    /// Get codec identifier.
    fn codec(&self) -> CodecId;

    /// Send audio frame.
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid.
    fn send_frame(&mut self, frame: &AudioFrame) -> AudioResult<()>;

    /// Receive encoded packet.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    fn receive_packet(&mut self) -> AudioResult<Option<EncodedAudioPacket>>;

    /// Flush encoder.
    ///
    /// # Errors
    ///
    /// Returns error if flush fails.
    fn flush(&mut self) -> AudioResult<()>;

    /// Get encoder configuration.
    fn config(&self) -> &AudioEncoderConfig;
}

/// Encoded audio packet.
#[derive(Clone, Debug)]
pub struct EncodedAudioPacket {
    /// Compressed data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Duration in samples.
    pub duration: u32,
}

/// Audio decoder configuration.
#[derive(Clone, Debug)]
pub struct AudioDecoderConfig {
    /// Codec.
    pub codec: CodecId,
    /// Sample rate.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Extra data (codec headers).
    pub extradata: Option<Vec<u8>>,
}

impl Default for AudioDecoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::Opus,
            sample_rate: 48000,
            channels: 2,
            extradata: None,
        }
    }
}

/// Audio encoder configuration.
#[derive(Clone, Debug)]
pub struct AudioEncoderConfig {
    /// Codec.
    pub codec: CodecId,
    /// Sample rate.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Bitrate (bits/sec).
    pub bitrate: u32,
    /// Frame size in samples.
    pub frame_size: u32,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            codec: CodecId::Opus,
            sample_rate: 48000,
            channels: 2,
            bitrate: 128_000,
            frame_size: 960, // 20ms at 48kHz
        }
    }
}
