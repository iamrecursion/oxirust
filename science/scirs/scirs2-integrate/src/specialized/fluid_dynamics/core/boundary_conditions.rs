//! Boundary condition types for fluid dynamics simulations.

/// Boundary condition types for fluid dynamics simulations
#[derive(Debug, Clone, Copy)]
pub enum FluidBoundaryCondition {
    /// No-slip (velocity = 0 at boundary)
    /// Used for solid wall boundaries where fluid adheres to the surface
    NoSlip,

    /// Free-slip (normal velocity = 0, tangential stress = 0)
    /// Used for slip walls or symmetry boundaries
    FreeSlip,

    /// Periodic boundary
    /// Used when the flow field repeats across the boundary
    Periodic,

    /// Inflow with specified velocity
    /// Takes (u, v) velocity components for 2D or (u, v) for the first two components in 3D
    Inflow(f64, f64),

    /// Outflow (zero gradient)
    /// Used for outlets where flow naturally exits the domain
    Outflow,
}

impl FluidBoundaryCondition {
    /// Check if the boundary condition is a wall type (NoSlip or FreeSlip)
    pub fn is_wall(&self) -> bool {
        matches!(
            self,
            FluidBoundaryCondition::NoSlip | FluidBoundaryCondition::FreeSlip
        )
    }

    /// Check if the boundary condition requires velocity specification
    pub fn requires_velocity(&self) -> bool {
        matches!(self, FluidBoundaryCondition::Inflow(_, _))
    }

    /// Get the inflow velocity if this is an inflow boundary condition
    pub fn get_inflow_velocity(&self) -> Option<(f64, f64)> {
        match self {
            FluidBoundaryCondition::Inflow(u, v) => Some((*u, *v)),
            _ => None,
        }
    }
}

impl Default for FluidBoundaryCondition {
    /// Default boundary condition is no-slip
    fn default() -> Self {
        FluidBoundaryCondition::NoSlip
    }
}
