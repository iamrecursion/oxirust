//! Frame-accurate marker system.

pub mod export;
pub mod manager;
pub mod range;
pub mod types;

pub use manager::MarkerManager;
pub use types::{Marker, MarkerId, MarkerType};
