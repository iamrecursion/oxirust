//! Transition management between playlist items.

pub mod crossfade;
pub mod manager;

pub use crossfade::{Crossfade, CrossfadeType};
pub use manager::{TransitionManager, TransitionType};
