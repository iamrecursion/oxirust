//! HLS (HTTP Live Streaming) packaging.
//!
//! This module provides comprehensive HLS packaging support including:
//!
//! - Master playlist generation (multi-variant)
//! - Media playlist generation (per-variant)
//! - TS and fMP4 segment formats
//! - Variant stream management
//! - Encryption support (AES-128, SAMPLE-AES)
//! - Live and VOD packaging
//!
//! # Example
//!
//! ```ignore
//! use oximedia_packager::hls::{HlsPackager, HlsPackagerBuilder};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create HLS packager
//! let mut packager = HlsPackagerBuilder::new()
//!     .with_segment_duration(Duration::from_secs(6))
//!     .with_output_directory("output/hls".into())
//!     .build()?;
//!
//! // Package video to HLS
//! packager.package("input.mkv").await?;
//! # Ok(())
//! # }
//! ```

pub mod packager;
pub mod playlist;
pub mod variant;

pub use packager::{HlsPackager, HlsPackagerBuilder};
pub use playlist::{
    MasterPlaylistBuilder, MediaPlaylistBuilder, MediaType, PlaylistType, VariantStream,
};
pub use variant::{VariantConfig, VariantManager, VariantSet};
