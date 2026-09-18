//! FLAC native container muxer.
//!
//! This module provides a muxer for creating native FLAC files.
//! FLAC (Free Lossless Audio Codec) uses its own container format
//! with metadata blocks followed by audio frames.
//!
//! # Metadata Blocks
//!
//! - STREAMINFO (required): Contains stream parameters
//! - PADDING: Optional padding for future metadata
//! - APPLICATION: Application-specific data
//! - SEEKTABLE: Seek points for random access
//! - `VORBIS_COMMENT`: Metadata tags (title, artist, etc.)
//! - CUESHEET: CD-style cue information
//! - PICTURE: Embedded artwork
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::mux::{FlacMuxer, Muxer, MuxerConfig};
//!
//! let config = MuxerConfig::new()
//!     .with_title("My Song")
//!     .with_muxing_app("OxiMedia");
//!
//! let mut muxer = FlacMuxer::new(sink, config);
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

mod writer;

pub use writer::FlacMuxer;
