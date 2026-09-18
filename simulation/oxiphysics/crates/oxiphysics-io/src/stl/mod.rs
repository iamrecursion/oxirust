//! Auto-generated module structure

pub mod functions;
pub mod functions_2;
pub mod hole_filling;
#[cfg(test)]
mod regression_tests;
pub mod types;
#[cfg(test)]
mod weld_tests;

// Re-export all types
pub use functions::*;
pub use functions_2::*;
pub use hole_filling::repair_close_holes;
pub use types::*;
