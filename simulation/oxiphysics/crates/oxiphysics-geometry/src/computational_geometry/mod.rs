//! Auto-generated module structure

pub mod delaunay_2d;
pub mod delaunay_3d;
pub mod functions;
pub mod functions_2;
pub mod types;

#[cfg(test)]
mod circle_predicate_tests;
#[cfg(test)]
mod slab_locator_tests;
#[cfg(test)]
mod visibility_tests;

// Re-export all types
pub use delaunay_3d::{Tet, Tetrahedralization, alpha_shape, delaunay_3d};
pub use functions::*;
pub use types::*;
