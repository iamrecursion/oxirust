//! DASH (Dynamic Adaptive Streaming over HTTP) packaging.
//!
//! This module provides comprehensive DASH packaging support including:
//!
//! - MPD manifest generation
//! - Multi-representation packaging
//! - CMAF (Common Media Application Format) support
//! - Segment templates
//! - Audio and subtitle adaptation sets
//! - Live and VOD packaging
//!
//! # Example
//!
//! ```ignore
//! use oximedia_packager::dash::{DashPackager, DashPackagerBuilder};
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create DASH packager
//! let mut packager = DashPackagerBuilder::new()
//!     .with_segment_duration(Duration::from_secs(4))
//!     .with_output_directory("output/dash".into())
//!     .build()?;
//!
//! // Package video to DASH
//! packager.package("input.mkv").await?;
//! # Ok(())
//! # }
//! ```

pub mod cmaf;
pub mod mpd;
pub mod packager;

pub use cmaf::{CmafHeader, CmafTrack, TrackType};
pub use mpd::{
    AdaptationSet, DashProfile, MpdBuilder, MpdType, Period, Representation, SegmentTemplate,
};
pub use packager::{DashPackager, DashPackagerBuilder};
