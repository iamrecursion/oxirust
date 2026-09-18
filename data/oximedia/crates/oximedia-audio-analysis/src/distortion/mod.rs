//! Distortion analysis module.

pub mod clipping;
pub mod detect;
pub mod thd;

pub use clipping::{detect_clipping, ClippingResult};
pub use detect::{DistortionDetector, DistortionResult};
pub use thd::total_harmonic_distortion;
