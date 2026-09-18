//! Auto-generated module structure

pub mod ale_fsi;
pub mod functions;
pub mod pod_rom;
pub mod thermo_mechanical;
pub mod types;

#[cfg(test)]
mod regression_tests;

// Re-export all types
pub use ale_fsi::*;
pub use functions::*;
pub use pod_rom::*;
pub use thermo_mechanical::*;
pub use types::*;
