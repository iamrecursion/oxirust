//! Auto-generated module structure

pub mod stabilitytolerance_traits;
pub mod advancednumericalstabilityconfig_traits;
pub mod stabilitymetrics_traits;
pub mod stabilitytrendanalysis_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use stabilitytolerance_traits::*;
pub use advancednumericalstabilityconfig_traits::*;
pub use stabilitymetrics_traits::*;
pub use stabilitytrendanalysis_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
#[path = "../numerical_stability_enhancements_tests.rs"]
mod tests;
