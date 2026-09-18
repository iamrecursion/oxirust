//! Matroska/`WebM` muxer.
//!
//! This module provides a muxer for creating Matroska (.mkv) and `WebM` (.webm)
//! container files. Both use the EBML (Extensible Binary Meta Language)
//! format for structure.
//!
//! # `WebM` vs Matroska
//!
//! `WebM` is a subset of Matroska with restrictions:
//! - Video: VP8, VP9, or AV1 only
//! - Audio: Vorbis or Opus only
//! - No subtitles (except `WebVTT` in some implementations)
//!
//! The muxer automatically determines whether to output Matroska or `WebM`
//! based on the codecs used.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::mux::{MatroskaMuxer, Muxer, MuxerConfig};
//!
//! let config = MuxerConfig::new()
//!     .with_title("My Video");
//!
//! let mut muxer = MatroskaMuxer::new(sink, config);
//! muxer.add_stream(video_info)?;
//! muxer.add_stream(audio_info)?;
//!
//! muxer.write_header().await?;
//!
//! for packet in packets {
//!     muxer.write_packet(&packet).await?;
//! }
//!
//! muxer.write_trailer().await?;
//! ```

mod cluster;
mod cues;
mod seek_head;
mod writer;

pub use cluster::ClusterWriter;
pub use cues::CueWriter;
pub use writer::MatroskaMuxer;
