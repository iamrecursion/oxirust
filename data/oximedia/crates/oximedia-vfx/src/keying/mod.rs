//! Keying and compositing effects.
//!
//! This module provides green/blue screen keying with advanced algorithms,
//! spill suppression, and edge refinement.

pub mod advanced;
pub mod edge;
pub mod spill;

pub use advanced::{AdvancedKey, KeyColor};
pub use edge::{EdgeRefine, EdgeRefinementMethod};
pub use spill::{SpillMethod, SpillSuppress};
