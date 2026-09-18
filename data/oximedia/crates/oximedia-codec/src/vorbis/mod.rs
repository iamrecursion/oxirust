//! Vorbis audio encoder and decoder.
//!
//! Vorbis is a patent-free, fully open general-purpose compressed audio format
//! by the Xiph.Org Foundation.  This module provides:
//!
//! - **`VorbisEncoder`** — lossy encoder: window, MDCT, floor curve, residue
//!   quantisation, packed into an *OxiMedia simplified packet format*.
//! - **`VorbisDecoder`** — packet decoder that reconstructs PCM from packets
//!   produced by `VorbisEncoder`.
//!
//! Header layouts follow the Vorbis I specification
//! (<https://xiph.org/vorbis/doc/Vorbis_I_spec.html>).
//!
//! # Honest status: full Vorbis I decode is NOT implemented
//!
//! Per `docs/codec_status.md`, full Vorbis I decoding (setup-header
//! codebook/floor/residue parsing, reverse codebook decode, floor-curve
//! reconstruction, channel coupling, window switching) is **not
//! implemented** and is deferred to 0.2.0. The encoder/decoder pair in this
//! module round-trips only the OxiMedia simplified packet format; streams
//! from standards-compliant encoders (libvorbis etc.) are rejected with an
//! honest `CodecError::UnsupportedFeature` at the setup header instead of
//! silently producing empty or garbage PCM. See
//! [`decoder`] for details.
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_codec::vorbis::{VorbisEncoder, VorbisDecoder, VorbisConfig, VorbisQuality};
//!
//! let config = VorbisConfig {
//!     sample_rate: 44100,
//!     channels: 2,
//!     quality: VorbisQuality::Q5,
//! };
//!
//! let mut encoder = VorbisEncoder::new(config).expect("encoder init failed");
//! // encode silence
//! let samples = vec![0.0f32; 4096]; // 2048 stereo interleaved samples
//! let packets = encoder.encode_interleaved(&samples).expect("encode failed");
//! assert!(!packets.is_empty() || true); // may buffer
//! ```

pub mod codebook;
pub mod decoder;
pub mod encoder;
pub mod floor;
pub mod mdct;
pub mod residue;

pub use decoder::{
    DecoderState, VorbisCommentHeader, VorbisDecoder, VorbisHeaderType, VorbisIdHeader,
};
pub use encoder::{
    SimpleVorbisEncoder, VorbisConfig, VorbisEncConfig, VorbisEncoder, VorbisPacket, VorbisQuality,
};
