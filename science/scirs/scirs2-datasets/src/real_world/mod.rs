//! Auto-generated module structure

pub mod functions;
pub mod realworldconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use realworldconfig_traits::*;
pub use types::*;

#[cfg(test)]
#[path = "../real_world_tests.rs"]
mod tests;
