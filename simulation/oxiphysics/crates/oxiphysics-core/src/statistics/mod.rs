//! Auto-generated module structure

pub(crate) mod functions;
pub(crate) mod functions_2;
mod poisson;
pub mod types;
pub mod welfordonline_traits;

// Re-export all types
pub use functions::*;
pub(crate) use poisson::poisson_ptrs;
pub use types::*;

#[cfg(test)]
mod regression_tests;
