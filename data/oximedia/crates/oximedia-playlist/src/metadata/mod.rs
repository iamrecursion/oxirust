//! Metadata tracking and as-run log generation.

pub mod asrun;
pub mod track;

pub use asrun::{AsRunEntry, AsRunLog};
pub use track::{MetadataTracker, PlaybackEvent};
