//! Parameter structures for fluid dynamics solvers.

/// Navier-Stokes solver parameters
#[derive(Debug, Clone)]
pub struct NavierStokesParams {
    /// Kinematic viscosity
    pub nu: f64,
    /// Density (for incompressible flow, usually 1.0)
    pub rho: f64,
    /// Time step
    pub dt: f64,
    /// Maximum iterations for pressure solver
    pub max_pressure_iter: usize,
    /// Tolerance for pressure solver
    pub pressure_tol: f64,
    /// Use semi-Lagrangian advection
    pub semi_lagrangian: bool,
}

impl Default for NavierStokesParams {
    fn default() -> Self {
        Self {
            nu: 0.01,
            rho: 1.0,
            dt: 0.01,
            max_pressure_iter: 100,
            pressure_tol: 1e-6,
            semi_lagrangian: false,
        }
    }
}

impl NavierStokesParams {
    /// Create new parameters with specified Reynolds number
    /// Reynolds number = U * L / nu, where U is characteristic velocity, L is characteristic length
    pub fn with_reynolds_number(
        reynolds: f64,
        characteristic_velocity: f64,
        characteristic_length: f64,
    ) -> Self {
        let nu = characteristic_velocity * characteristic_length / reynolds;
        Self {
            nu,
            ..Default::default()
        }
    }

    /// Create new parameters with specified viscosity
    pub fn with_viscosity(nu: f64) -> Self {
        Self {
            nu,
            ..Default::default()
        }
    }

    /// Create new parameters with specified time step
    pub fn with_time_step(dt: f64) -> Self {
        Self {
            dt,
            ..Default::default()
        }
    }

    /// Calculate the Reynolds number given characteristic velocity and length
    pub fn reynolds_number(&self, characteristic_velocity: f64, characteristiclength: f64) -> f64 {
        characteristic_velocity * characteristiclength / self.nu
    }

    /// Calculate CFL number for stability analysis
    /// CFL = u * dt / dx + v * dt / dy (+ w * dt / dz for 3D)
    pub fn cfl_number_2d(&self, u_max: f64, vmax: f64, dx: f64, dy: f64) -> f64 {
        u_max * self.dt / dx + vmax * self.dt / dy
    }

    /// Calculate CFL number for 3D case
    pub fn cfl_number_3d(
        &self,
        u_max: f64,
        v_max: f64,
        w_max: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> f64 {
        u_max * self.dt / dx + v_max * self.dt / dy + w_max * self.dt / dz
    }

    /// Calculate viscous CFL number
    /// Viscous CFL = nu * dt * (1/dx^2 + 1/dy^2 + 1/dz^2)
    pub fn viscous_cfl_2d(&self, dx: f64, dy: f64) -> f64 {
        self.nu * self.dt * (1.0 / (dx * dx) + 1.0 / (dy * dy))
    }

    /// Calculate viscous CFL number for 3D case
    pub fn viscous_cfl_3d(&self, dx: f64, dy: f64, dz: f64) -> f64 {
        self.nu * self.dt * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    }

    /// Validate parameters for numerical stability
    pub fn validate(&self) -> Result<(), String> {
        if self.nu <= 0.0 {
            return Err("Kinematic viscosity must be positive".to_string());
        }
        if self.rho <= 0.0 {
            return Err("Density must be positive".to_string());
        }
        if self.dt <= 0.0 {
            return Err("Time step must be positive".to_string());
        }
        if self.max_pressure_iter == 0 {
            return Err("Maximum pressure iterations must be greater than zero".to_string());
        }
        if self.pressure_tol <= 0.0 {
            return Err("Pressure tolerance must be positive".to_string());
        }
        Ok(())
    }
}
