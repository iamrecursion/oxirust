//! Core types and structures for fluid dynamics simulations.
//!
//! This module contains the fundamental data structures and types used throughout
//! the fluid dynamics module, including state representations, boundary conditions,
//! and solver parameters.

pub mod boundary_conditions;
pub mod parameters;
pub mod state;

// Re-export all public types for convenient access
pub use boundary_conditions::FluidBoundaryCondition;
pub use parameters::NavierStokesParams;
pub use state::{FluidState, FluidState3D};
