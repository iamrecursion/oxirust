//! Hum detection and removal.

pub mod detector;
pub mod remover;

pub use detector::{detect_hum_autocorrelation, HumDetector, HumDetectorConfig, HumFrequencies};
pub use remover::{CombFilter, HumRemover, NotchFilter};
