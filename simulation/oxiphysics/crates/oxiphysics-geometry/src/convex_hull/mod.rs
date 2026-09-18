//! Auto-generated module structure

pub mod convexhull_traits;
pub mod functions;
pub mod functions_2;
#[cfg(test)]
mod hull_regression_tests;
pub mod incrementalconvexhull_traits;
pub(crate) mod quickhull_core;
mod solid;
mod topology;
pub mod types;

// Re-export all types
pub use functions::*;
pub use topology::{HullTopology, TopologyParts};
pub use types::*;
