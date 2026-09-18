//! DASH (Dynamic Adaptive Streaming over HTTP) packaging module.

mod manifest;
mod packager;
mod segment;

pub use manifest::{AdaptationSet, MpdGenerator, Period, Representation};
pub use packager::{DashConfig, DashPackager};
pub use segment::{DashSegmentWriter, InitSegment, MediaSegment};
