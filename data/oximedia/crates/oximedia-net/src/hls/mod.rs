//! HLS (HTTP Live Streaming) protocol implementation.
//!
//! This module provides types and utilities for working with Apple's HLS protocol,
//! including playlist parsing, segment fetching, and adaptive bitrate control.
//!
//! # Key Types
//!
//! - [`MasterPlaylist`] - Multi-bitrate playlist with variant streams
//! - [`MediaPlaylist`] - Single bitrate playlist with segments
//! - [`Segment`] - Individual media segment
//! - [`SegmentFetcher`] - Async segment downloader
//! - [`AbrController`] - Adaptive bitrate controller trait
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::hls::{MasterPlaylist, MediaPlaylist, SegmentFetcher};
//!
//! async fn play_stream(url: &str) -> NetResult<()> {
//!     let master = MasterPlaylist::parse(playlist_data)?;
//!     let best_variant = master.best_variant_for_bandwidth(5_000_000);
//!     // ...
//!     Ok(())
//! }
//! ```

mod abr;
mod client;
pub mod ll_hls;
mod playlist;
mod segment;

// Re-export legacy ABR types for backward compatibility
pub use abr::{
    AbrController, AbrDecision, BufferBasedAbr, QualityLevel, ThroughputBasedAbr,
    ThroughputEstimator,
};

// Re-export client types
pub use client::{
    BufferedSegment, ClientState, ClientStats, HlsClient, HlsClientBuilder, HlsClientConfig,
};

pub use playlist::{
    MasterPlaylist, MediaPlaylist, MediaType, PlaylistTag, PlaylistType, Segment, StreamInf,
    VariantStream,
};
pub use segment::{ByteRange, FetchConfig, FetchResult, SegmentCache, SegmentFetcher};
