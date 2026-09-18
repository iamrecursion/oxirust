//! Compressible flow solvers and related functionality
//!
//! This module provides comprehensive tools for simulating compressible fluid flows,
//! including Euler and Navier-Stokes equations with SIMD optimizations.
//!
//! # Module Structure
//!
//! - [`solver`]: Main compressible flow solver implementation
//! - [`state`]: Compressible flow state representation and manipulation
//! - [`flux_computation`]: Flux calculations and Riemann solver implementations
//!
//! # Features
//!
//! - High-performance SIMD-optimized solvers
//! - Adaptive time stepping with CFL condition
//! - Conservative flux calculations
//! - Various boundary condition implementations
//! - Fourth-order Runge-Kutta time integration

pub mod flux_computation;
pub mod solver;
pub mod state;

// Re-export main types and functions
pub use flux_computation::CompressibleFluxes;
pub use solver::CompressibleFlowSolver;
pub use state::CompressibleState;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressible_module_exports() {
        // Test that main types can be imported and instantiated
        let solver = CompressibleFlowSolver::new(10, 10, 10, 0.1, 0.1, 0.1);
        let state = solver.initialize_state();

        assert_eq!(state.density.dim(), (10, 10, 10));
        assert_eq!(state.momentum.len(), 3);
        assert_eq!(state.energy.dim(), (10, 10, 10));
    }

    #[test]
    fn test_adaptive_timestep() {
        let solver = CompressibleFlowSolver::new(5, 5, 5, 0.1, 0.1, 0.1);
        let state = solver.initialize_state();

        let dt = solver.calculate_adaptive_timestep(&state);
        assert!(dt > 0.0);
        assert!(dt.is_finite());
    }
}
