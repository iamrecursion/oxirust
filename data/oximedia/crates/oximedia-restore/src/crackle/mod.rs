//! Crackle detection and removal.

pub mod detector;
pub mod remover;

pub use detector::{Crackle, CrackleDetector};
pub use remover::CrackleRemover;
