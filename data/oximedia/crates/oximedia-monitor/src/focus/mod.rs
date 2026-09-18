//! Focus peaking and assist.

pub mod peaking;

pub use peaking::FocusPeaking;

// Re-export from scopes module
pub use crate::scopes::focus::{FocusAssist, FocusMetrics, PeakingColor};
