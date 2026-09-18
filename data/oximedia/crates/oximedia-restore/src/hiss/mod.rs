//! Hiss detection and removal.

pub mod detector;
pub mod remover;

pub use detector::{HissDetector, HissProfile};
pub use remover::{HissRemover, HissRemoverConfig};
