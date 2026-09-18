//! XPBD constraints and strain-limiting utilities.

pub mod functions;
pub mod strain_limit;
pub mod types;

#[cfg(test)]
mod bending_tests;
#[cfg(test)]
mod strain_limiter_tests;

// Re-export all types
pub use functions::*;
pub use strain_limit::{StrainLimitConfig, XpbdStrainLimiter, apply_strain_limiting};
pub use types::*;
