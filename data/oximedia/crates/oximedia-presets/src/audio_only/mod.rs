//! Audio-only encoding presets for podcast and music distribution.
//!
//! Provides FLAC (lossless) and Opus (efficient lossy) presets targeting:
//! - Podcast hosting platforms (RSS-compatible MP3/OGG)
//! - Music distribution services (FLAC lossless masters)
//! - Podcast archival (FLAC preservation copies)
//! - Music streaming (Opus high-quality delivery)

pub mod flac;
pub mod opus_podcast;
