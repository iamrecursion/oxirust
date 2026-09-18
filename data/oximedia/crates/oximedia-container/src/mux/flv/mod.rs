//! Adobe FLV (Flash Video) container muxer.
//!
//! Provides [`FlvMuxer`], which writes FLV byte streams conforming to
//! *Adobe Video File Format Specification Version 10* (2008).
//!
//! Only patent-free (Green List) codecs are accepted:
//! - **Audio**: MP3 (codec 2), PCM (codec 0)
//! - **Video**: Sorenson H.263 (codec 2) — patents expired 2019
//!
//! AAC and AVC/H.264 are explicitly rejected.

mod amf0;
mod writer;

pub use writer::FlvMuxer;
