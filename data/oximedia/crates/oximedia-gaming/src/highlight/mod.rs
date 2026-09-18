//! Highlight detection and bookmarking.

pub mod detect;
pub mod detector;
pub mod marker;

pub use detect::{
    DetectionConfig, HighlightDetector as FrameHighlightDetector,
    HighlightEvent as FrameHighlightEvent,
};
pub use detector::{
    AudioEventDetector, ChatActivityDetector, HighlightDetector, HighlightEvent, HighlightTimeline,
    HighlightType,
};
pub use marker::{HighlightMarker, Marker};
