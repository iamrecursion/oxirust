//! Rolling shutter correction.
//!
//! Detects and corrects rolling shutter artifacts in video.

pub mod correct;
pub mod detect;

pub use correct::RollingShutterCorrector;
pub use detect::RollingShutterDetector;
