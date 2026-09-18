//! Tempo detection and BPM estimation.

pub mod detect;
pub mod estimate;
pub mod utils;

pub use detect::TempoDetector;
pub use estimate::BpmEstimator;
pub use utils::bounded_acf_with_early_exit;
