//! Auto-generated module structure

pub mod ecosystemtestconfig_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use ecosystemtestconfig_traits::*;
pub use functions::*;
pub use types::*;

#[cfg(test)]
#[path = "../ecosystem_integration_tests.rs"]
mod tests;
