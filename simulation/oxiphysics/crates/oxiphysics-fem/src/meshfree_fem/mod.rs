//! Auto-generated module structure

/// Essential boundary conditions for MLS approximations.
mod boundary;
#[cfg(test)]
mod boundary_tests;
/// Bridging-domain coupling of meshfree and finite element regions.
mod bridging;
#[cfg(test)]
mod bridging_tests;
/// Delaunay triangulation for background integration cells.
mod delaunay;
#[cfg(test)]
mod delaunay_tests;
/// Sparse assembly and iterative solution of the EFG system.
mod efg_sparse;
#[cfg(test)]
mod efg_sparse_tests;
/// Functions and utilities.
mod functions;
/// Type definitions.
mod types;

// Re-export all types
pub use boundary::*;
pub use bridging::*;
pub use delaunay::*;
pub use efg_sparse::*;
pub use functions::*;
pub use types::*;
