//! Auto-generated module structure

/// Trait implementations for blowmolding_traits.
mod blowmolding_traits;
/// Functions and utilities.
mod functions;
/// Closest-point queries and inside/outside classification against a mold.
mod mold_surface;
/// Regression tests for the mold surface and for `BlowMolding::mold_collision`.
#[cfg(test)]
mod mold_tests;
/// Regression tests for the pressure load and inflation of `InflatableShell`.
#[cfg(test)]
mod shell_tests;
/// Type definitions.
mod types;
/// Regression tests for the mesh winding and the volume constraint.
#[cfg(test)]
mod volume_tests;

// Re-export all types
pub use functions::*;
pub use types::*;
