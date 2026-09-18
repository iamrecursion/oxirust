//! Auto-generated module structure

pub mod density_diffusion;
pub mod functions;
mod gather;
pub mod types;
pub mod wcsphparams_traits;

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests;
#[cfg(test)]
mod regression_tests;

// Re-export all types
pub use density_diffusion::*;
pub use functions::*;
pub use types::*;
