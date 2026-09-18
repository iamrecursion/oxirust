//! Event detection in video.

pub mod detector;
pub mod sports;

pub use detector::{EventDetector, VideoEvent};
pub use sports::{SportsEvent, SportsEventDetector};
