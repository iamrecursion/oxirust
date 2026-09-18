//! Fragmented MP4 (fMP4) packaging helpers for `oximedia-stream`.
//!
//! Re-exports the segment-index builder from `oximedia-container` so that
//! downstream packagers can emit DASH-compatible `sidx` boxes without taking
//! a direct dependency on the container crate's internal MP4 writer paths.
//!
//! This thin re-export layer exists deliberately: it gives the streaming
//! crate a stable surface for fMP4 helpers that may grow over time, and it
//! decouples consumers from container-internal module reshuffling.
//!
//! # Example
//!
//! ```
//! use oximedia_stream::fmp4::build_sidx;
//!
//! // sidx for a single subsegment with one SAP entry
//! let sidx = build_sidx(
//!     /* reference_id           */ 1,
//!     /* timescale              */ 90_000,
//!     /* earliest_pts           */ 0,
//!     /* referenced_size        */ 4096,
//!     /* subsegment_duration    */ 90_000,
//!     /* is_sap                 */ true,
//! );
//!
//! assert_eq!(&sidx[4..8], b"sidx");
//! ```

pub use oximedia_container::segment_index::build_sidx;
