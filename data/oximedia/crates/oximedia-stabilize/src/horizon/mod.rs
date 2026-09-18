//! Horizon detection and leveling.
//!
//! Automatically detects the horizon line and applies corrections to level it.

pub mod detect;
pub mod level;

pub use detect::HorizonDetector;
pub use level::HorizonLeveler;
