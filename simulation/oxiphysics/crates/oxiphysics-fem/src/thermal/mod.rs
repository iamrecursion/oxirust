//! Auto-generated module structure

pub mod functions;
pub mod functions_2;
mod linear_solve;
pub mod types;

#[cfg(test)]
mod solver_regression_tests;
#[cfg(test)]
mod step_consistency_tests;

// Re-export all types
pub use functions::*;
pub use functions_2::*;
pub use types::*;
