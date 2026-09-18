//! Ogg container muxer.
//!
//! Implements the Ogg bitstream format as specified in
//! [RFC 3533](https://www.rfc-editor.org/rfc/rfc3533).
//!
//! # Supported Codecs
//!
//! - **Opus**: High-quality, low-latency audio codec
//! - **Vorbis**: General-purpose audio codec
//! - **FLAC**: Lossless audio codec
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::mux::{OggMuxer, Muxer, MuxerConfig};
//!
//! let config = MuxerConfig::new();
//! let mut muxer = OggMuxer::new(sink, config);
//!
//! muxer.add_stream(audio_info)?;
//! muxer.write_header().await?;
//!
//! for packet in packets {
//!     muxer.write_packet(&packet).await?;
//! }
//!
//! muxer.write_trailer().await?;
//! ```

mod stream;
mod writer;

pub use stream::OggStreamWriter;
pub use writer::OggMuxer;
