//! Shot classification algorithms.

pub mod angle;
pub mod composition;
pub mod shottype;

pub use angle::AngleClassifier;
pub use composition::CompositionAnalyzer;
pub use shottype::ShotTypeClassifier;
