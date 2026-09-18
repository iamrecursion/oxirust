//! Wow and flutter detection and correction.

pub mod corrector;
pub mod detector;

pub use corrector::WowFlutterCorrector;
pub use detector::{WowFlutterDetector, WowFlutterProfile};
