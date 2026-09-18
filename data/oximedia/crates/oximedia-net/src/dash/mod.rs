//! DASH (Dynamic Adaptive Streaming over HTTP) protocol implementation.
//!
//! This module provides types and utilities for working with MPEG-DASH streaming,
//! including MPD (Media Presentation Description) parsing, segment handling, and
//! adaptive bitrate streaming.
//!
//! # Key Types
//!
//! - [`Mpd`] - Media Presentation Description (manifest)
//! - [`Period`] - Time period within presentation
//! - [`AdaptationSet`] - Group of interchangeable representations
//! - [`Representation`] - Single encoded version of content
//! - [`SegmentTemplate`] - Template for generating segment URLs
//! - [`DashSegment`] - Individual media segment
//! - [`DashClient`] - Main client for DASH streaming with ABR support
//!
//! # Example
//!
//! ```ignore
//! use oximedia_net::dash::{Mpd, DashClient, DashClientConfig};
//! use oximedia_net::abr::{AbrConfig, HybridAbrController};
//!
//! async fn stream_dash(mpd_xml: &str) -> NetResult<()> {
//!     // Parse MPD
//!     let mpd = Mpd::parse(mpd_xml)?;
//!
//!     // Create client with config
//!     let config = DashClientConfig::new()
//!         .with_base_url("https://cdn.example.com/")
//!         .with_abr(true);
//!     let mut client = DashClient::new(mpd, config);
//!
//!     // Set up ABR controller
//!     let abr_config = AbrConfig::default();
//!     let abr = HybridAbrController::new(abr_config);
//!     client.set_abr_controller(Box::new(abr));
//!
//!     // Create streaming session
//!     let mut session = client.create_session(0, 0, Some(1_500_000))?;
//!
//!     // Fetch initialization segment
//!     let init = client.fetch_initialization_segment(&session).await?;
//!     println!("Init segment: {} bytes", init.bytes_downloaded);
//!
//!     // Fetch media segments with adaptive quality
//!     loop {
//!         // Perform ABR decision
//!         client.perform_abr_decision(&mut session).await?;
//!
//!         // Fetch next segment
//!         let segment = client.fetch_next_segment(&mut session).await?;
//!         println!("Segment {}: {} bytes, {} Mbps",
//!             segment.segment_info.number,
//!             segment.bytes_downloaded,
//!             segment.throughput_bps() / 1_000_000.0);
//!
//!         // Process segment data...
//!         // Update buffer...
//!     }
//! }
//! ```

mod client;
pub mod live;
pub mod ll_dash;
mod mpd;
mod segment;

pub use client::{
    DashClient, DashClientConfig, FetchResult, RepresentationSelection, StreamSession,
};
pub use mpd::{
    AdaptationSet, ContentComponent, ContentProtection, Descriptor, Mpd, MpdType, Period,
    ProgramInformation, Representation, SegmentBase, SegmentList, SegmentTemplate, SegmentTimeline,
    SegmentTimelineEntry, UrlType,
};
pub use segment::{DashSegment, SegmentGenerator, SegmentInfo};

// Re-export unified ABR types for convenience
pub use crate::abr::{
    AbrConfig, AbrDecision, AbrMode, AdaptiveBitrateController, HybridAbrController, QualityLevel,
    SimpleThroughputAbr,
};
