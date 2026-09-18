//! Professional timecode support.
//!
//! Provides timecode track handling for broadcast workflows.

#![forbid(unsafe_code)]

pub mod track;

pub use track::{Timecode, TimecodeFormat, TimecodeTrack};
