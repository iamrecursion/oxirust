//! Auto-generated module structure

pub mod adaptivestreamconfig_traits;
pub mod functions;
pub mod learningstatistics_traits;
pub mod neuraladaptivesystem_traits;
pub mod optimizationconfig_traits;
pub mod quantuminspiredoptimizer_traits;
pub mod types;

// Re-export all types
pub use adaptivestreamconfig_traits::*;
pub use functions::*;
pub use learningstatistics_traits::*;
pub use neuraladaptivesystem_traits::*;
pub use optimizationconfig_traits::*;
pub use quantuminspiredoptimizer_traits::*;
pub use types::*;

#[cfg(test)]
#[path = "../adaptive_streaming_engine_tests.rs"]
mod tests;
