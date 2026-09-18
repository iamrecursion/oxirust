//! Vorbis codec implementation.
//!
//! Vorbis is an open-source, royalty-free lossy audio codec.
//! It uses MDCT transforms and psychoacoustic modeling.
//!
//! # Modules
//!
//! - [`header`] - Header parsing
//! - [`codebook`] - Codebook structures
//! - [`floor`] - Floor types
//! - [`encoder`] - Vorbis encoder
//! - [`decoder`] - Vorbis decoder (full state machine + audio decode)
//! - [`bitpack`] - Bitstream packing
//! - [`mdct`] - MDCT transform
//! - [`psycho`] - Psychoacoustic model
//! - [`residue`] - Residue encoding / decoding

#![forbid(unsafe_code)]

pub mod bitpack;
pub mod codebook;
pub mod decoder;
pub mod encoder;
pub mod floor;
pub mod header;
pub mod mdct;
pub mod psycho;
pub mod residue;

use crate::{AudioDecoder, AudioDecoderConfig, AudioError, AudioFrame, AudioResult, ChannelLayout};
use oximedia_core::{CodecId, SampleFormat};

// Re-export submodule types
pub use codebook::{Codebook, CodebookEntry, HuffmanTree};
pub use decoder::VorbisDecoderInner;
pub use encoder::{QualityMode, VorbisEncoder};
pub use floor::{Floor, FloorType0, FloorType1};
pub use header::{CommentHeader, IdentificationHeader, SetupHeader, VorbisHeader};

/// Vorbis decoder.
///
/// Delegates to [`VorbisDecoderInner`] for all state-machine and
/// audio-decode logic.
pub struct VorbisDecoder {
    inner: VorbisDecoderInner,
}

impl VorbisDecoder {
    /// Create new Vorbis decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Vorbis {
            return Err(AudioError::InvalidParameter("Expected Vorbis codec".into()));
        }
        let inner = VorbisDecoderInner::new(config)?;
        Ok(Self { inner })
    }

    /// Parse Vorbis identification header.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse_identification_header(data: &[u8]) -> AudioResult<IdentificationHeader> {
        IdentificationHeader::parse(data)
    }

    /// Parse Vorbis comment header.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse_comment_header(data: &[u8]) -> AudioResult<CommentHeader> {
        CommentHeader::parse(data)
    }

    /// Check if decoder is ready for audio packets.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }
}

impl AudioDecoder for VorbisDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> AudioResult<()> {
        self.inner.send_packet(data, pts)
    }

    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>> {
        self.inner.receive_frame()
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.inner.flush()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn output_format(&self) -> Option<SampleFormat> {
        Some(SampleFormat::F32)
    }

    fn sample_rate(&self) -> Option<u32> {
        self.inner.sample_rate()
    }

    fn channel_layout(&self) -> Option<ChannelLayout> {
        self.inner.channel_layout()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_decoder() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let decoder = VorbisDecoder::new(&config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Vorbis);
    }

    #[test]
    fn test_vorbis_decoder_wrong_codec() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        assert!(VorbisDecoder::new(&config).is_err());
    }

    #[test]
    fn test_vorbis_decoder_reset() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let mut decoder = VorbisDecoder::new(&config).expect("should succeed");
        decoder.reset();
        assert!(!decoder.is_ready());
    }

    #[test]
    fn test_vorbis_decoder_not_ready() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let decoder = VorbisDecoder::new(&config).expect("should succeed");
        assert!(!decoder.is_ready());
    }
}
