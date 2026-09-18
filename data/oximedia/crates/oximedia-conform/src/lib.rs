//! `OxiMedia` Conform - Professional Media Conforming System.
//!
//! This crate provides comprehensive tools for conforming media timelines from
//! EDL, XML, and AAF formats to actual media files. It supports various matching
//! strategies, quality control, and export to multiple formats.
//!
//! # Features
//!
//! - **EDL Import**: CMX 3600, CMX 3400 format support
//! - **XML Import**: Final Cut Pro, Premiere, `DaVinci` Resolve
//! - **AAF Import**: Avid Media Composer timelines
//! - **Multiple Matching Strategies**: Filename, timecode, content hash, duration
//! - **Media Database**: SQLite-based catalog with search
//! - **Quality Control**: Validation and verification
//! - **Timeline Reconstruction**: Multi-track video/audio
//! - **Export Formats**: MP4, Matroska, EDL, XML, AAF, frame sequences
//!
//! # Example
//!
//! ```no_run
//! use oximedia_conform::{ConformSession, ConformConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut session = ConformSession::new(
//!     "My Conform".to_string(),
//!     PathBuf::from("timeline.edl"),
//!     vec![PathBuf::from("/media/sources")],
//!     PathBuf::from("/output/conformed"),
//!     ConformConfig::default(),
//! )?;
//!
//! // Run the complete conform workflow
//! let report = session.run().await?;
//!
//! println!("Conformed {}/{} clips",
//!          report.stats.matched_count,
//!          report.stats.total_clips);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod afd_wss;
pub mod analysis;
pub mod audio_conform;
pub mod batch;
pub mod broadcast_safe;
pub mod cache;
pub mod clip_resolver;
pub mod config;
pub mod conform_diff;
pub mod conform_enhancements;
pub mod conform_log;
pub mod conform_manifest;
pub mod conform_snapshot;
pub mod constants;
pub mod database;
pub mod deliverable_check;
pub mod delivery_map;
pub mod encode_param_check;
pub mod error;
pub mod exporters;
pub mod format_conform;
pub mod frame_rate_convert;
pub mod importers;
pub mod loudness_conform;
pub mod matching;
pub mod media;
pub mod media_relink;
pub mod partial_conform;
pub mod prelude;
pub mod progress;
pub mod proxy_conform;
pub mod pulldown;
pub mod qc;
pub mod reconstruction;
pub mod reel_management;
pub mod reporting;
pub mod session;
pub mod source_verify;
pub mod spec_profile;
pub mod spec_validator;
pub mod test_card;
pub mod timecode_conform;
pub mod timeline;
pub mod types;
pub mod utils;
pub mod versioning;

// Re-export commonly used types
pub use batch::{BatchJob, BatchProcessor, BatchResult, BatchStatistics};
pub use config::ConformConfig;
pub use error::{ConformError, ConformResult};
pub use exporters::report::{AmbiguousMatch, MatchReport, MatchStatistics};
pub use importers::aaf::{AafImporter, AafMob, AafParser, AafProject, AafSlot};
pub use importers::fcpxml::{FcpxmlClip, FcpxmlParser, FcpxmlSequence};
pub use media::{MediaCatalog, MediaScanner, ScanProgress};
pub use progress::{ProgressInfo, ProgressStage, ProgressTracker};
pub use session::{ConformSession, SessionStatus};
pub use timeline::{Timeline, TimelineClip, Track, TrackKind};
pub use types::{
    ClipMatch, ClipReference, ConformClip, ConformProject, ConformSequence, FrameRate, MatchMethod,
    MediaFile, OutputFormat, Timecode, TrackType,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get the library version.
#[must_use]
pub const fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
