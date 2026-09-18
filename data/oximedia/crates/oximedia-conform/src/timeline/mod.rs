//! Timeline structures for reconstructed sequences.

pub mod clip;
pub mod track;
pub mod transition;

pub use clip::TimelineClip;
pub use track::{Timeline, Track, TrackKind};
pub use transition::{Transition, TransitionType};
