//! Dynamic range analysis module.

pub mod crest;
pub mod range;
pub mod rms;

pub use crest::crest_factor;
pub use range::{DynamicsAnalyzer, DynamicsResult};
pub use rms::{rms_level, rms_over_time};
