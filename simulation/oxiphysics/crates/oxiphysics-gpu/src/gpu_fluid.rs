// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated fluid simulation using a MAC (Marker-and-Cell) grid.
//!
//! This module provides a CPU mock of GPU Navier-Stokes incompressible fluid
//! simulation.  The MAC grid stores pressure at cell centres and velocity
//! components on cell faces, following the Harlow-Welch staggered layout.
//!
//! One time step of [`FluidSimGpu`] is semi-Lagrangian advection, implicit
//! viscous diffusion with the kinematic viscosity `ν = μ/ρ`, a Jacobi
//! solve of the pressure Poisson equation `∇²p = (ρ/Δt) ∇·u` and the
//! projection `u ← u − (Δt/ρ) ∇p`, which together leave the velocity
//! divergence-free to the accuracy of the pressure solve.

use crate::error::{Error, Result};

// ── MAC cell / grid ──────────────────────────────────────────────────────────

/// Check the time step and density shared by the pressure solve and the
/// projection.
fn check_dt_rho(dt: f64, rho: f64) -> Result<()> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(Error::general(format!(
            "pressure projection needs a finite positive time step, got dt = {dt}"
        )));
    }
    if !(rho.is_finite() && rho > 0.0) {
        return Err(Error::general(format!(
            "pressure projection needs a finite positive density, got rho = {rho}"
        )));
    }
    Ok(())
}

/// A single cell in a Marker-and-Cell (MAC) grid.
///
/// Pressure is stored at the cell centre; `u_x` is the velocity on the east
/// face and `u_y` is the velocity on the north face.
#[derive(Debug, Clone, Copy)]
pub struct MacCell {
    /// Pressure at cell centre (Pa).
    pub pressure: f64,
    /// Velocity on the east face (x-direction, m/s).
    pub u_x: f64,
    /// Velocity on the north face (y-direction, m/s).
    pub u_y: f64,
    /// `true` if the cell contains fluid (not a solid obstacle).
    pub is_fluid: bool,
}

impl Default for MacCell {
    fn default() -> Self {
        Self {
            pressure: 0.0,
            u_x: 0.0,
            u_y: 0.0,
            is_fluid: true,
        }
    }
}

/// A 2-D Marker-and-Cell (MAC) grid for incompressible fluid simulation.
///
/// The grid is stored in row-major order: `cells[j * nx + i]` is column `i`,
/// row `j`.  Cell `(i, j)` is centred at `((i + ½) dx, (j + ½) dx)`; its `u_x`
/// sits on the east face at `((i + 1) dx, (j + ½) dx)` and its `u_y` on the
/// north face at `((i + ½) dx, (j + 1) dx)`.  The west face of column 0 and
/// the south face of row 0 lie on the domain boundary, are not stored and
/// count as zero velocity.
#[derive(Debug, Clone)]
pub struct MacGrid {
    /// Number of cells in the x direction.
    pub nx: usize,
    /// Number of cells in the y direction.
    pub ny: usize,
    /// Cell width / height (uniform, square cells, m).
    pub dx: f64,
    /// Flat cell storage (`ny * nx` entries).
    pub cells: Vec<MacCell>,
}

impl MacGrid {
    /// Create a new MAC grid filled with default (fluid) cells.
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            nx,
            ny,
            dx,
            cells: vec![MacCell::default(); nx * ny],
        }
    }

    /// Return a shared reference to the cell at column `i`, row `j`.
    #[inline]
    pub fn cell(&self, i: usize, j: usize) -> &MacCell {
        &self.cells[j * self.nx + i]
    }

    /// Return a mutable reference to the cell at column `i`, row `j`.
    #[inline]
    pub fn cell_mut(&mut self, i: usize, j: usize) -> &mut MacCell {
        &mut self.cells[j * self.nx + i]
    }

    /// Discrete divergence of velocity at cell `(i, j)`.
    ///
    /// Uses the MAC staggered layout: east-face minus west-face for `u_x`,
    /// north-face minus south-face for `u_y`.  The west face of column 0 and
    /// the south face of row 0 are not stored and are treated as zero
    /// (no-flux).
    pub fn divergence(&self, i: usize, j: usize) -> f64 {
        let ux_e = self.cells[j * self.nx + i].u_x;
        let ux_w = if i > 0 {
            self.cells[j * self.nx + (i - 1)].u_x
        } else {
            0.0
        };
        let uy_n = self.cells[j * self.nx + i].u_y;
        let uy_s = if j > 0 {
            self.cells[(j - 1) * self.nx + i].u_y
        } else {
            0.0
        };
        (ux_e - ux_w + uy_n - uy_s) / self.dx
    }

    /// Jacobi iteration pressure solve for the Poisson equation `∇²p = ρ/Δt · ∇·u`.
    ///
    /// `n_iter` sweeps are performed over the interior fluid cells (every cell
    /// except the outermost ring; solid cells are skipped).  The outermost
    /// ring keeps its stored pressure -- zero unless set otherwise -- and, like
    /// a solid cell, acts as a fixed (Dirichlet) pressure for its neighbours.
    ///
    /// Followed by [`project`](Self::project) with the same `dt` and `rho`
    /// (`u ← u − Δt/ρ ∇p`), the divergence left in an interior cell is `Δt/ρ`
    /// times the residual of the Poisson equation there, so it vanishes as the
    /// Jacobi iteration converges, independently of `dt` and `rho`.
    ///
    /// # Errors
    ///
    /// Returns an error when `dt` or `rho` is not a finite positive number; the
    /// pressure field is left unchanged in that case.
    pub fn pressure_solve_jacobi(&mut self, n_iter: usize, dt: f64, rho: f64) -> Result<()> {
        check_dt_rho(dt, rho)?;
        let nx = self.nx;
        let ny = self.ny;
        let dx2 = self.dx * self.dx;
        let scale = rho / dt;
        // The velocities do not change during the sweeps, so the right-hand
        // side is evaluated once.
        let mut rhs = vec![0.0_f64; nx * ny];
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                rhs[j * nx + i] = scale * self.divergence(i, j);
            }
        }
        // Jacobi: every sweep reads the previous iterate `p` and writes `p_next`.
        let mut p: Vec<f64> = self.cells.iter().map(|c| c.pressure).collect();
        let mut p_next = p.clone();
        for _ in 0..n_iter {
            for j in 1..ny.saturating_sub(1) {
                for i in 1..nx.saturating_sub(1) {
                    let idx = j * nx + i;
                    if !self.cells[idx].is_fluid {
                        continue;
                    }
                    p_next[idx] = (p[idx + 1] + p[idx - 1] + p[idx + nx] + p[idx - nx]
                        - dx2 * rhs[idx])
                        / 4.0;
                }
            }
            std::mem::swap(&mut p, &mut p_next);
        }
        for (cell, &value) in self.cells.iter_mut().zip(p.iter()) {
            cell.pressure = value;
        }
        Ok(())
    }

    /// Semi-Lagrangian advection of velocity components.
    ///
    /// Each face velocity is traced back by `dt` seconds with the velocity
    /// interpolated at the face and the value at the departure point is
    /// interpolated bilinearly from the faces of the same component (see
    /// [`interpolate_velocity`]).
    pub fn advect_velocity(&mut self, dt: f64) {
        let nx = self.nx;
        let ny = self.ny;
        let dx = self.dx;
        let old = self.cells.clone();

        for j in 0..ny {
            for i in 0..nx {
                if !old[j * nx + i].is_fluid {
                    continue;
                }
                // Advect u_x (east face)
                let x = (i as f64 + 1.0) * dx;
                let y = (j as f64 + 0.5) * dx;
                let [vx, vy] = interpolate_velocity_cells(&old, nx, ny, dx, x, y);
                let xp = x - dt * vx;
                let yp = y - dt * vy;
                let [new_ux, _] = interpolate_velocity_cells(&old, nx, ny, dx, xp, yp);

                // Advect u_y (north face)
                let x2 = (i as f64 + 0.5) * dx;
                let y2 = (j as f64 + 1.0) * dx;
                let [vx2, vy2] = interpolate_velocity_cells(&old, nx, ny, dx, x2, y2);
                let xp2 = x2 - dt * vx2;
                let yp2 = y2 - dt * vy2;
                let [_, new_uy] = interpolate_velocity_cells(&old, nx, ny, dx, xp2, yp2);

                let idx = j * nx + i;
                self.cells[idx].u_x = new_ux;
                self.cells[idx].u_y = new_uy;
            }
        }
    }

    /// Implicit viscous diffusion of both face-velocity components.
    ///
    /// Solves `(1 − ν Δt ∇²) u_new = u` with `n_iter` Jacobi sweeps started
    /// from `u`, i.e. `u_new = (u + α Σ u_neighbours) / (1 + 4 α)` with
    /// `α = ν Δt / dx²`.  The scheme is unconditionally stable for any `ν`
    /// and `Δt`; an incomplete solve (large `α`, few sweeps) stays bounded by
    /// the initial velocities and just diffuses less.
    ///
    /// Only the faces stored with fluid cells are updated; the faces of solid
    /// cells keep their values and act as fixed neighbours.  Faces outside the
    /// grid count as zero velocity, as in [`divergence`](Self::divergence).
    ///
    /// `nu` is the kinematic viscosity `μ/ρ` (m²/s).  Nothing changes when
    /// `nu == 0` or `n_iter == 0`.
    ///
    /// # Errors
    ///
    /// Returns an error when `nu` is negative or not finite, or `dt` is not a
    /// finite positive number; the velocities are left unchanged in that case.
    pub fn diffuse_velocity(&mut self, nu: f64, dt: f64, n_iter: usize) -> Result<()> {
        if !(nu.is_finite() && nu >= 0.0) {
            return Err(Error::general(format!(
                "viscous diffusion needs a finite non-negative viscosity, got nu = {nu}"
            )));
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::general(format!(
                "viscous diffusion needs a finite positive time step, got dt = {dt}"
            )));
        }
        if nu == 0.0 || n_iter == 0 {
            return Ok(());
        }
        let nx = self.nx;
        let ny = self.ny;
        let alpha = nu * dt / (self.dx * self.dx);
        let inv_diag = 1.0 / (1.0 + 4.0 * alpha);

        let source_x: Vec<f64> = self.cells.iter().map(|c| c.u_x).collect();
        let source_y: Vec<f64> = self.cells.iter().map(|c| c.u_y).collect();
        let mut ux = source_x.clone();
        let mut uy = source_y.clone();
        let mut ux_next = ux.clone();
        let mut uy_next = uy.clone();

        // Sum of the four face neighbours of `(i, j)` in `field`; faces outside
        // the grid are zero.
        let neighbour_sum = |field: &[f64], i: usize, j: usize| -> f64 {
            let east = if i + 1 < nx {
                field[j * nx + i + 1]
            } else {
                0.0
            };
            let west = if i > 0 { field[j * nx + i - 1] } else { 0.0 };
            let north = if j + 1 < ny {
                field[(j + 1) * nx + i]
            } else {
                0.0
            };
            let south = if j > 0 { field[(j - 1) * nx + i] } else { 0.0 };
            east + west + north + south
        };

        for _ in 0..n_iter {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;
                    if !self.cells[idx].is_fluid {
                        continue;
                    }
                    ux_next[idx] = (source_x[idx] + alpha * neighbour_sum(&ux, i, j)) * inv_diag;
                    uy_next[idx] = (source_y[idx] + alpha * neighbour_sum(&uy, i, j)) * inv_diag;
                }
            }
            std::mem::swap(&mut ux, &mut ux_next);
            std::mem::swap(&mut uy, &mut uy_next);
        }
        for ((cell, &vx), &vy) in self.cells.iter_mut().zip(ux.iter()).zip(uy.iter()) {
            cell.u_x = vx;
            cell.u_y = vy;
        }
        Ok(())
    }

    /// Pressure-projection step: subtract the pressure gradient from velocity,
    /// `u ← u − (Δt/ρ) ∇p`.
    ///
    /// `dt` is the time-step and `rho` the fluid density (kg/m³), the same
    /// values given to [`pressure_solve_jacobi`](Self::pressure_solve_jacobi).
    /// The pressure outside the grid is taken as zero, and the faces of solid
    /// cells are not changed.
    ///
    /// # Errors
    ///
    /// Returns an error when `dt` or `rho` is not a finite positive number; the
    /// velocities are left unchanged in that case.
    pub fn project(&mut self, dt: f64, rho: f64) -> Result<()> {
        check_dt_rho(dt, rho)?;
        let nx = self.nx;
        let ny = self.ny;
        let factor = dt / (rho * self.dx);
        let p: Vec<f64> = self.cells.iter().map(|c| c.pressure).collect();
        for j in 0..ny {
            for i in 0..nx {
                if !self.cells[j * nx + i].is_fluid {
                    continue;
                }
                // east face correction
                let p_e = if i + 1 < nx { p[j * nx + i + 1] } else { 0.0 };
                let p_c = p[j * nx + i];
                self.cells[j * nx + i].u_x -= factor * (p_e - p_c);
                // north face correction
                let p_n = if j + 1 < ny { p[(j + 1) * nx + i] } else { 0.0 };
                self.cells[j * nx + i].u_y -= factor * (p_n - p_c);
            }
        }
        Ok(())
    }
}

// ── Helper: bilinear velocity interpolation ──────────────────────────────────

/// Bilinear interpolation on an integer lattice.
///
/// `(fa, fb)` are continuous lattice coordinates, clamped to the node ranges
/// `a_range` and `b_range` (both inclusive); `node(a, b)` is the value stored
/// at lattice node `(a, b)`.
fn bilinear_lattice(
    fa: f64,
    fb: f64,
    a_range: (isize, isize),
    b_range: (isize, isize),
    node: impl Fn(isize, isize) -> f64,
) -> f64 {
    let fa = fa.clamp(a_range.0 as f64, a_range.1 as f64);
    let fb = fb.clamp(b_range.0 as f64, b_range.1 as f64);
    let a_floor = fa.floor();
    let b_floor = fb.floor();
    let ta = fa - a_floor;
    let tb = fb - b_floor;
    let a0 = a_floor as isize;
    let b0 = b_floor as isize;
    let a1 = (a0 + 1).min(a_range.1);
    let b1 = (b0 + 1).min(b_range.1);
    (1.0 - ta) * (1.0 - tb) * node(a0, b0)
        + ta * (1.0 - tb) * node(a1, b0)
        + (1.0 - ta) * tb * node(a0, b1)
        + ta * tb * node(a1, b1)
}

/// Interpolate the staggered velocity at world position `(x, y)` from a flat
/// cell slice.
///
/// Each component is sampled from its own face lattice: `u_x(i, j)` lives at
/// `((i + 1) dx, (j + ½) dx)` and `u_y(i, j)` at `((i + ½) dx, (j + 1) dx)`.
/// The unstored west and south boundary faces carry zero velocity (no-flux
/// walls at `x = 0` and `y = 0`); points outside the domain are clamped to its
/// boundary.  An empty grid yields zero velocity.
fn interpolate_velocity_cells(
    cells: &[MacCell],
    nx: usize,
    ny: usize,
    dx: f64,
    x: f64,
    y: f64,
) -> [f64; 2] {
    if nx == 0 || ny == 0 {
        return [0.0, 0.0];
    }
    let last_i = nx as isize - 1;
    let last_j = ny as isize - 1;
    // Lattice nodes below index 0 are the unstored boundary faces: zero.
    let u_x_at = |a: isize, b: isize| -> f64 {
        if a < 0 || b < 0 {
            return 0.0;
        }
        cells
            .get(b as usize * nx + a as usize)
            .map_or(0.0, |c| c.u_x)
    };
    let u_y_at = |a: isize, b: isize| -> f64 {
        if a < 0 || b < 0 {
            return 0.0;
        }
        cells
            .get(b as usize * nx + a as usize)
            .map_or(0.0, |c| c.u_y)
    };
    let ux = bilinear_lattice(
        x / dx - 1.0,
        y / dx - 0.5,
        (-1, last_i),
        (0, last_j),
        u_x_at,
    );
    let uy = bilinear_lattice(
        x / dx - 0.5,
        y / dx - 1.0,
        (0, last_i),
        (-1, last_j),
        u_y_at,
    );
    [ux, uy]
}

// ── FluidSimGpu ──────────────────────────────────────────────────────────────

/// High-level GPU-accelerated fluid simulation driver.
///
/// Wraps a `MacGrid` and orchestrates advection, viscous diffusion and
/// pressure projection each time-step.
pub struct FluidSimGpu {
    /// The MAC grid holding all fluid state.
    pub grid: MacGrid,
    /// Dynamic viscosity (Pa·s).
    pub viscosity: f64,
    /// Fluid density (kg/m³).
    pub density: f64,
    /// Jacobi sweeps of the pressure solve per time-step (default 20).
    pub pressure_iterations: usize,
    /// Jacobi sweeps of the implicit viscous solve per time-step (default 20).
    pub diffusion_iterations: usize,
}

impl FluidSimGpu {
    /// Number of Jacobi sweeps per solve a new simulation starts with.
    pub const DEFAULT_ITERATIONS: usize = 20;

    /// Create a new fluid simulation on a `nx × ny` grid.
    ///
    /// `viscosity` is the dynamic viscosity in Pa·s and `density` the fluid
    /// density in kg/m³.
    pub fn new(nx: usize, ny: usize, dx: f64, viscosity: f64, density: f64) -> Self {
        Self {
            grid: MacGrid::new(nx, ny, dx),
            viscosity,
            density,
            pressure_iterations: Self::DEFAULT_ITERATIONS,
            diffusion_iterations: Self::DEFAULT_ITERATIONS,
        }
    }

    /// Advance the simulation by time-step `dt` seconds.
    ///
    /// Pipeline: semi-Lagrangian advection → implicit viscous diffusion with
    /// the kinematic viscosity `ν = viscosity / density`
    /// ([`diffusion_iterations`](Self::diffusion_iterations) Jacobi sweeps) →
    /// pressure solve of `∇²p = (ρ/Δt) ∇·u`
    /// ([`pressure_iterations`](Self::pressure_iterations) Jacobi sweeps) →
    /// projection `u ← u − (Δt/ρ) ∇p`.
    ///
    /// # Errors
    ///
    /// Returns an error when `dt` or the density is not a finite positive
    /// number, or the viscosity is negative or not finite; the grid is left
    /// unchanged in that case.
    pub fn step(&mut self, dt: f64) -> Result<()> {
        check_dt_rho(dt, self.density)?;
        if !(self.viscosity.is_finite() && self.viscosity >= 0.0) {
            return Err(Error::general(format!(
                "fluid step needs a finite non-negative viscosity, got viscosity = {}",
                self.viscosity
            )));
        }
        self.grid.advect_velocity(dt);
        self.grid
            .diffuse_velocity(self.viscosity / self.density, dt, self.diffusion_iterations)?;
        self.grid
            .pressure_solve_jacobi(self.pressure_iterations, dt, self.density)?;
        self.grid.project(dt, self.density)
    }
}

// ── VorticityConfinement ─────────────────────────────────────────────────────

/// Vorticity confinement adds back dissipated rotation energy.
///
/// A positive `strength` amplifies existing vorticity, keeping swirling
/// structures alive longer.
pub struct VorticityConfinement {
    /// Confinement coefficient (dimensionless, typically 0.01 – 0.5).
    pub strength: f64,
}

impl VorticityConfinement {
    /// Create a new vorticity confinement operator.
    pub fn new(strength: f64) -> Self {
        Self { strength }
    }

    /// Compute the scalar vorticity (∂u_y/∂x − ∂u_x/∂y) at every interior cell
    /// centre.
    ///
    /// The face velocities are first averaged to the cell centres
    /// (`ū_x = ½ (u_x(i−1, j) + u_x(i, j))`, `ū_y = ½ (u_y(i, j−1) + u_y(i, j))`)
    /// and both derivatives are central differences of those, so they refer
    /// to the same point.  The outermost ring of cells has zero vorticity.
    ///
    /// Returns a flat `Vec<f64>` of length `nx * ny`.
    pub fn compute_vorticity(&self, grid: &MacGrid) -> Vec<f64> {
        let nx = grid.nx;
        let ny = grid.ny;
        let dx = grid.dx;
        let mut omega = vec![0.0f64; nx * ny];
        // Cell-centred u_x needs column i >= 1, cell-centred u_y needs row j >= 1.
        let centred_u_x = |i: usize, j: usize| {
            0.5 * (grid.cells[j * nx + i - 1].u_x + grid.cells[j * nx + i].u_x)
        };
        let centred_u_y = |i: usize, j: usize| {
            0.5 * (grid.cells[(j - 1) * nx + i].u_y + grid.cells[j * nx + i].u_y)
        };
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                let duy_dx = (centred_u_y(i + 1, j) - centred_u_y(i - 1, j)) / (2.0 * dx);
                let dux_dy = (centred_u_x(i, j + 1) - centred_u_x(i, j - 1)) / (2.0 * dx);
                omega[j * nx + i] = duy_dx - dux_dy;
            }
        }
        omega
    }

    /// Apply the vorticity confinement force to `grid`.
    ///
    /// Computes `f = strength * dx * η × (∇|ω| / |∇|ω||)` and adds it to the
    /// face velocities.
    pub fn apply_confinement_force(&self, grid: &mut MacGrid, dt: f64) {
        let omega = self.compute_vorticity(grid);
        let nx = grid.nx;
        let ny = grid.ny;
        let dx = grid.dx;
        let omega_abs: Vec<f64> = omega.iter().map(|v| v.abs()).collect();
        for j in 1..ny.saturating_sub(1) {
            for i in 1..nx.saturating_sub(1) {
                let grad_x = (omega_abs[j * nx + i + 1] - omega_abs[j * nx + i - 1]) / (2.0 * dx);
                let grad_y =
                    (omega_abs[(j + 1) * nx + i] - omega_abs[(j - 1) * nx + i]) / (2.0 * dx);
                let len = (grad_x * grad_x + grad_y * grad_y).sqrt();
                if len < 1e-12 {
                    continue;
                }
                let nx_hat = grad_x / len;
                let ny_hat = grad_y / len;
                let w = omega[j * nx + i];
                // force = strength * dx * w * (N × ω)
                let fx = self.strength * dx * w * ny_hat;
                let fy = -self.strength * dx * w * nx_hat;
                grid.cells[j * nx + i].u_x += dt * fx;
                grid.cells[j * nx + i].u_y += dt * fy;
            }
        }
    }
}

// ── FluidObstacle ────────────────────────────────────────────────────────────

/// A rigid obstacle embedded in the fluid grid.
///
/// Cells listed in `cells` are marked as solid, and no-slip boundary
/// conditions are enforced on their faces.
pub struct FluidObstacle {
    /// List of `(i, j)` cell coordinates that form the obstacle.
    pub cells: Vec<(usize, usize)>,
}

impl FluidObstacle {
    /// Create a new obstacle from a list of cell coordinates.
    pub fn new(cells: Vec<(usize, usize)>) -> Self {
        Self { cells }
    }

    /// Mark all obstacle cells as solid in `grid`.
    pub fn set_solid(&self, grid: &mut MacGrid) {
        for &(i, j) in &self.cells {
            if i < grid.nx && j < grid.ny {
                grid.cell_mut(i, j).is_fluid = false;
            }
        }
    }

    /// Enforce no-slip (zero velocity) on solid cell faces.
    pub fn apply_no_slip(&self, grid: &mut MacGrid) {
        for &(i, j) in &self.cells {
            if i < grid.nx && j < grid.ny {
                let c = grid.cell_mut(i, j);
                c.u_x = 0.0;
                c.u_y = 0.0;
                c.pressure = 0.0;
            }
        }
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// CFL-stable time-step for the MAC grid.
///
/// Returns `safety * dx / u_max` where `u_max` is the maximum face speed.
/// Falls back to `safety * dx` when the grid is quiescent.
pub fn cfl_timestep(grid: &MacGrid, safety: f64) -> f64 {
    let mut u_max: f64 = 0.0;
    for c in &grid.cells {
        u_max = u_max.max(c.u_x.abs()).max(c.u_y.abs());
    }
    if u_max < 1e-15 {
        return safety * grid.dx;
    }
    safety * grid.dx / u_max
}

/// Compute the Reynolds number `Re = u_max * L / nu`.
///
/// * `u_max` – characteristic velocity (m/s)
/// * `L` – characteristic length (m)
/// * `nu` – kinematic viscosity (m²/s)
pub fn reynolds_number_grid(u_max: f64, l: f64, nu: f64) -> f64 {
    if nu.abs() < 1e-30 {
        return f64::INFINITY;
    }
    u_max * l / nu
}

/// Bilinearly interpolate velocity at world coordinates `(x, y)`.
///
/// Coordinates are in metres; the grid's `dx` sets the cell size.
pub fn interpolate_velocity(grid: &MacGrid, x: f64, y: f64) -> [f64; 2] {
    interpolate_velocity_cells(&grid.cells, grid.nx, grid.ny, grid.dx, x, y)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(nx: usize, ny: usize, ux: f64, uy: f64) -> MacGrid {
        let mut g = MacGrid::new(nx, ny, 1.0);
        for c in g.cells.iter_mut() {
            c.u_x = ux;
            c.u_y = uy;
        }
        g
    }

    // ── MacCell default ──────────────────────────────────────────────────────

    #[test]
    fn mac_cell_default_is_fluid() {
        let c = MacCell::default();
        assert!(c.is_fluid);
        assert_eq!(c.pressure, 0.0);
        assert_eq!(c.u_x, 0.0);
        assert_eq!(c.u_y, 0.0);
    }

    #[test]
    fn mac_cell_copy() {
        let c = MacCell {
            pressure: 1.0,
            u_x: 2.0,
            u_y: 3.0,
            is_fluid: false,
        };
        let d = c;
        assert_eq!(d.pressure, 1.0);
        assert!(!d.is_fluid);
    }

    // ── MacGrid construction ─────────────────────────────────────────────────

    #[test]
    fn mac_grid_new_dimensions() {
        let g = MacGrid::new(4, 5, 0.1);
        assert_eq!(g.nx, 4);
        assert_eq!(g.ny, 5);
        assert_eq!(g.cells.len(), 20);
    }

    #[test]
    fn mac_grid_cell_accessor() {
        let mut g = MacGrid::new(3, 3, 1.0);
        g.cell_mut(1, 2).u_x = 42.0;
        assert_eq!(g.cell(1, 2).u_x, 42.0);
    }

    // ── Divergence ───────────────────────────────────────────────────────────

    #[test]
    fn divergence_uniform_flow_is_zero() {
        // In a uniform field all face velocities are equal → div = 0 at interior.
        let g = uniform_grid(5, 5, 1.0, 0.0);
        // Interior cell
        let div = g.divergence(2, 2);
        assert!(div.abs() < 1e-12, "div = {div}");
    }

    #[test]
    fn divergence_source_cell() {
        // Set u_x at east face larger than west → positive divergence.
        let mut g = MacGrid::new(4, 4, 1.0);
        g.cell_mut(1, 1).u_x = 2.0; // east face of cell (1,1)
        g.cell_mut(0, 1).u_x = 1.0; // east face of cell (0,1) = west face of (1,1)
        let div = g.divergence(1, 1);
        assert!(div > 0.0, "expected positive div, got {div}");
    }

    #[test]
    fn divergence_boundary_left_edge() {
        let mut g = MacGrid::new(4, 4, 1.0);
        g.cell_mut(0, 0).u_x = 1.0;
        // west face is boundary (zero) → div = (1 - 0 + 0 - 0)/dx
        let div = g.divergence(0, 0);
        assert!((div - 1.0).abs() < 1e-12, "div = {div}");
    }

    #[test]
    fn divergence_all_zeros() {
        let g = MacGrid::new(4, 4, 1.0);
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(g.divergence(i, j), 0.0);
            }
        }
    }

    // ── Test helpers ─────────────────────────────────────────────────────────

    /// Deterministic pseudo-random value in `[-1, 1)`.
    fn next_unit(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0
    }

    /// An `n × n` grid with random face velocities in `[-1, 1)`.
    fn random_grid(n: usize, dx: f64, seed: u64) -> MacGrid {
        let mut g = MacGrid::new(n, n, dx);
        let mut state = seed;
        for c in g.cells.iter_mut() {
            c.u_x = next_unit(&mut state);
            c.u_y = next_unit(&mut state);
        }
        g
    }

    /// Largest `|∇·u|` over the interior cells (all but the outermost ring),
    /// the cells the pressure solve acts on.
    fn max_interior_divergence(g: &MacGrid) -> f64 {
        let mut worst = 0.0_f64;
        for j in 1..g.ny - 1 {
            for i in 1..g.nx - 1 {
                worst = worst.max(g.divergence(i, j).abs());
            }
        }
        worst
    }

    /// Largest `|∇²p − (ρ/Δt) ∇·u|` over the interior fluid cells, evaluated
    /// with an independent 5-point stencil.
    fn poisson_residual(g: &MacGrid, dt: f64, rho: f64) -> f64 {
        let mut worst = 0.0_f64;
        for j in 1..g.ny - 1 {
            for i in 1..g.nx - 1 {
                if !g.cell(i, j).is_fluid {
                    continue;
                }
                let laplacian = (g.cell(i + 1, j).pressure
                    + g.cell(i - 1, j).pressure
                    + g.cell(i, j + 1).pressure
                    + g.cell(i, j - 1).pressure
                    - 4.0 * g.cell(i, j).pressure)
                    / (g.dx * g.dx);
                worst = worst.max((laplacian - rho / dt * g.divergence(i, j)).abs());
            }
        }
        worst
    }

    /// Solve and project `g`; returns `(max|div| before, max|div| after)` over
    /// the interior cells.
    fn solve_and_project(g: &mut MacGrid, sweeps: usize, dt: f64, rho: f64) -> (f64, f64) {
        let before = max_interior_divergence(g);
        g.pressure_solve_jacobi(sweeps, dt, rho)
            .expect("valid solver parameters");
        g.project(dt, rho).expect("valid solver parameters");
        (before, max_interior_divergence(g))
    }

    // ── Pressure Jacobi ──────────────────────────────────────────────────────

    #[test]
    fn pressure_jacobi_reduces_magnitude_with_source() {
        let mut g = MacGrid::new(6, 6, 1.0);
        // Seed divergence at centre
        g.cell_mut(3, 3).u_x = 1.0;
        g.pressure_solve_jacobi(50, 0.1, 1.0)
            .expect("valid solver parameters");
        // The east face of (3, 3) is the west face of (4, 3): cell (3, 3) is a
        // source (div = +1) and (4, 3) a sink (div = -1), so the pressure is
        // low at the source and high at the sink.
        assert!(g.cell(3, 3).pressure < -1e-6, "{}", g.cell(3, 3).pressure);
        assert!(g.cell(4, 3).pressure > 1e-6, "{}", g.cell(4, 3).pressure);
        // The outermost ring keeps its zero pressure.
        for j in 0..6 {
            for i in 0..6 {
                if i == 0 || j == 0 || i == 5 || j == 5 {
                    assert_eq!(g.cell(i, j).pressure, 0.0);
                }
            }
        }
    }

    #[test]
    fn pressure_jacobi_zero_field_stays_zero() {
        let mut g = MacGrid::new(5, 5, 1.0);
        g.pressure_solve_jacobi(10, 0.01, 1.0)
            .expect("valid solver parameters");
        for c in &g.cells {
            assert_eq!(c.pressure, 0.0);
        }
    }

    #[test]
    fn pressure_jacobi_convergence_monotone() {
        // More iterations must lower the Poisson residual.
        let (dt, rho) = (0.1, 1.0);
        let mut g0 = MacGrid::new(6, 6, 1.0);
        g0.cell_mut(3, 3).u_x = 2.0;
        let mut g1 = g0.clone();
        let mut g2 = g0.clone();
        g1.pressure_solve_jacobi(10, dt, rho)
            .expect("valid solver parameters");
        g2.pressure_solve_jacobi(100, dt, rho)
            .expect("valid solver parameters");
        // With p = 0 the residual is the right-hand side: (rho/dt) * |div| = 20.
        let initial = poisson_residual(&g0, dt, rho);
        assert!((initial - 20.0).abs() < 1e-12, "initial residual {initial}");
        let after_10 = poisson_residual(&g1, dt, rho);
        let after_100 = poisson_residual(&g2, dt, rho);
        assert!(after_10 < initial, "10 sweeps: {after_10} vs {initial}");
        assert!(after_100 <= after_10, "{after_100} vs {after_10}");
        assert!(
            after_100 < 1e-3 * after_10,
            "100 sweeps barely improved on 10: {after_100} vs {after_10}"
        );
    }

    /// The Poisson right-hand side is `ρ/Δt · ∇·u`: the pressure is
    /// proportional to `ρ/Δt`.
    #[test]
    fn pressure_jacobi_scales_with_rho_over_dt() {
        let mut reference = random_grid(8, 0.1, 21);
        let mut scaled = reference.clone();
        // rho/dt is 1 / 0.1 = 10 for the reference and 2 / 0.05 = 40 for the
        // scaled grid: a factor of 4.
        reference
            .pressure_solve_jacobi(30, 0.1, 1.0)
            .expect("valid solver parameters");
        scaled
            .pressure_solve_jacobi(30, 0.05, 2.0)
            .expect("valid solver parameters");
        let mut largest = 0.0_f64;
        for (a, b) in reference.cells.iter().zip(scaled.cells.iter()) {
            largest = largest.max(a.pressure.abs());
            assert!(
                (b.pressure - 4.0 * a.pressure).abs() <= 1e-12 * (1.0 + a.pressure.abs()),
                "{} vs 4 * {}",
                b.pressure,
                a.pressure
            );
        }
        assert!(largest > 1e-2, "the pressure field must be non-trivial");
    }

    #[test]
    fn pressure_jacobi_rejects_invalid_parameters() {
        let mut g = random_grid(6, 0.1, 5);
        let before = g.clone();
        for &(dt, rho) in &[
            (0.0, 1.0),
            (-0.01, 1.0),
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (0.01, 0.0),
            (0.01, -1.0),
            (0.01, f64::NAN),
        ] {
            assert!(g.pressure_solve_jacobi(10, dt, rho).is_err(), "{dt} {rho}");
            assert!(g.project(dt, rho).is_err(), "{dt} {rho}");
        }
        for (a, b) in g.cells.iter().zip(before.cells.iter()) {
            assert_eq!((a.pressure, a.u_x, a.u_y), (b.pressure, b.u_x, b.u_y));
        }
    }

    // ── Pressure projection ───────────────────────────────────────────────────

    #[test]
    fn project_corrects_velocity() {
        let mut g = MacGrid::new(4, 4, 1.0);
        // Set uniform pressure gradient
        for j in 0..4 {
            for i in 0..4 {
                g.cell_mut(i, j).pressure = i as f64;
            }
        }
        let ux_before = g.cell(1, 1).u_x;
        g.project(0.1, 1.0).expect("valid solver parameters");
        let ux_after = g.cell(1, 1).u_x;
        // dp/dx = 1.0, so the correction is -dt / (rho dx) * dp = -0.1; the
        // pressure does not vary in y, so u_y is untouched.
        assert!((ux_after - ux_before + 0.1).abs() < 1e-12, "{ux_after}");
        assert_eq!(g.cell(1, 1).u_y, 0.0);
    }

    /// `u ← u − (Δt/ρ) ∇p`, face by face, with the pressure outside the grid
    /// taken as zero.
    #[test]
    fn project_scales_with_dt_over_rho() {
        let (dt, rho, dx) = (0.2, 4.0, 0.5);
        let mut g = MacGrid::new(4, 4, dx);
        // p = 2 i + 3 j: constant gradients (2, 3) / dx.
        for j in 0..4 {
            for i in 0..4 {
                g.cell_mut(i, j).pressure = 2.0 * i as f64 + 3.0 * j as f64;
            }
        }
        g.project(dt, rho).expect("valid solver parameters");
        let factor = dt / (rho * dx);
        // Interior faces: u_x -= factor * 2, u_y -= factor * 3.
        assert!((g.cell(1, 1).u_x + factor * 2.0).abs() < 1e-14);
        assert!((g.cell(1, 1).u_y + factor * 3.0).abs() < 1e-14);
        // The east column sees zero pressure beyond the grid: p_e = 0, p_c = 6 + 3 j.
        assert!((g.cell(3, 1).u_x - factor * (6.0 + 3.0)).abs() < 1e-14);
    }

    #[test]
    fn project_solid_cell_unchanged() {
        let mut g = MacGrid::new(4, 4, 1.0);
        g.cell_mut(2, 2).is_fluid = false;
        g.cell_mut(2, 2).u_x = 5.0;
        for j in 0..4 {
            for i in 0..4 {
                g.cell_mut(i, j).pressure = 1.0;
            }
        }
        g.project(0.1, 1.0).expect("valid solver parameters");
        // Solid cell should be untouched
        assert_eq!(g.cell(2, 2).u_x, 5.0);
        // A fluid cell next to it with uniform pressure sees no gradient.
        assert!(g.cell(1, 1).u_x.abs() < 1e-14);
    }

    /// Before the fix the Poisson right-hand side lacked `ρ/Δt` while the
    /// update carried `Δt/ρ`, so with `Δt = 0.01` only 1 % of the divergence
    /// was removed.  Now it vanishes to the accuracy of the Jacobi solve:
    /// 200 sweeps shrink the slowest mode of the 8x8 interior by about
    /// `cos(pi/9)^200 ~ 3e-6`.
    #[test]
    fn projection_removes_divergence_of_random_field() {
        let mut g = random_grid(10, 0.1, 0x00C0_FFEE);
        let (before, after) = solve_and_project(&mut g, 200, 0.01, 1.0);
        assert!(before > 1.0, "the random field must be strongly divergent");
        assert!(after < 1e-3, "max|div| fell from {before} to only {after}");
        assert!(after < 1e-2 * before);
    }

    /// `dt` and `rho` cancel between the Poisson right-hand side and the
    /// gradient update, so the projected velocity does not depend on them.
    #[test]
    fn projection_is_independent_of_dt_and_rho() {
        let mut reference = random_grid(8, 0.2, 7);
        solve_and_project(&mut reference, 150, 0.01, 1.0);
        for &(dt, rho) in &[(0.5, 1.0), (0.001, 997.0), (2.0, 0.25)] {
            let mut g = random_grid(8, 0.2, 7);
            let (before, after) = solve_and_project(&mut g, 150, dt, rho);
            assert!(before / after > 100.0, "dt={dt} rho={rho}");
            for (a, b) in g.cells.iter().zip(reference.cells.iter()) {
                assert!(
                    (a.u_x - b.u_x).abs() < 1e-8 && (a.u_y - b.u_y).abs() < 1e-8,
                    "velocities differ for dt={dt} rho={rho}"
                );
            }
        }
    }

    /// After the projection the divergence left in each interior cell is
    /// `Δt/ρ` times the Poisson residual there.
    #[test]
    fn projection_leaves_dt_over_rho_times_poisson_residual() {
        let (dt, rho) = (0.02, 3.0);
        let mut g = random_grid(8, 0.25, 99);
        g.pressure_solve_jacobi(12, dt, rho)
            .expect("valid solver parameters");
        let residual = poisson_residual(&g, dt, rho);
        g.project(dt, rho).expect("valid solver parameters");
        let left = max_interior_divergence(&g);
        assert!(residual > 0.0, "twelve sweeps cannot solve the system");
        assert!(
            (left - dt / rho * residual).abs() <= 1e-9 * left,
            "left {left} vs dt/rho * residual {}",
            dt / rho * residual
        );
    }

    // ── CFL timestep ─────────────────────────────────────────────────────────

    #[test]
    fn cfl_timestep_uniform_flow() {
        let g = uniform_grid(4, 4, 2.0, 0.0);
        let dt = cfl_timestep(&g, 0.5);
        // safety * dx / u_max = 0.5 * 1.0 / 2.0 = 0.25
        assert!((dt - 0.25).abs() < 1e-12, "dt = {dt}");
    }

    #[test]
    fn cfl_timestep_quiescent() {
        let g = MacGrid::new(4, 4, 0.1);
        let dt = cfl_timestep(&g, 0.5);
        // Fallback: safety * dx = 0.05
        assert!((dt - 0.05).abs() < 1e-12, "dt = {dt}");
    }

    #[test]
    fn cfl_timestep_uses_max_component() {
        let mut g = MacGrid::new(4, 4, 1.0);
        g.cell_mut(1, 1).u_x = 1.0;
        g.cell_mut(2, 2).u_y = 5.0;
        let dt = cfl_timestep(&g, 1.0);
        assert!((dt - 1.0 / 5.0).abs() < 1e-12, "dt = {dt}");
    }

    #[test]
    fn cfl_timestep_safety_factor() {
        let g = uniform_grid(4, 4, 4.0, 0.0);
        let dt_full = cfl_timestep(&g, 1.0);
        let dt_half = cfl_timestep(&g, 0.5);
        assert!((dt_full - 2.0 * dt_half).abs() < 1e-12);
    }

    // ── Reynolds number ───────────────────────────────────────────────────────

    #[test]
    fn reynolds_number_basic() {
        let re = reynolds_number_grid(1.0, 1.0, 1e-6);
        assert!((re - 1.0e6).abs() < 1.0, "re = {re}");
    }

    #[test]
    fn reynolds_number_zero_viscosity() {
        let re = reynolds_number_grid(1.0, 1.0, 0.0);
        assert!(re.is_infinite());
    }

    #[test]
    fn reynolds_number_proportional_to_velocity() {
        let re1 = reynolds_number_grid(1.0, 1.0, 1e-3);
        let re2 = reynolds_number_grid(2.0, 1.0, 1e-3);
        assert!((re2 - 2.0 * re1).abs() < 1e-6);
    }

    // ── Velocity interpolation ───────────────────────────────────────────────

    #[test]
    fn interpolate_velocity_at_cell_centre() {
        let mut g = MacGrid::new(4, 4, 1.0);
        for c in g.cells.iter_mut() {
            c.u_x = 3.0;
            c.u_y = 1.5;
        }
        let v = interpolate_velocity(&g, 1.5, 1.5);
        assert!((v[0] - 3.0).abs() < 1e-12, "v[0] = {}", v[0]);
        assert!((v[1] - 1.5).abs() < 1e-12, "v[1] = {}", v[1]);
    }

    /// Points outside the domain clamp to its boundary: beyond the west/south
    /// walls the unstored boundary faces carry zero velocity, beyond the
    /// east/north walls the outermost stored faces apply.
    #[test]
    fn interpolate_velocity_clamps_outside() {
        let g = uniform_grid(4, 4, 1.0, 2.0);
        assert_eq!(interpolate_velocity(&g, -10.0, -10.0), [0.0, 0.0]);
        assert_eq!(interpolate_velocity(&g, 100.0, 100.0), [1.0, 2.0]);
        // Far west but mid-height: u_x is the zero wall face, u_y is the
        // clamped edge column.
        assert_eq!(interpolate_velocity(&g, -10.0, 2.0), [0.0, 2.0]);
    }

    #[test]
    fn interpolate_velocity_zero_field() {
        let g = MacGrid::new(4, 4, 1.0);
        let v = interpolate_velocity(&g, 1.0, 1.0);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
    }

    #[test]
    fn interpolate_velocity_corner() {
        let g = uniform_grid(4, 4, 1.0, 2.0);
        // The corner (0, 0) is on both the west and the south wall.
        assert_eq!(interpolate_velocity(&g, 0.0, 0.0), [0.0, 0.0]);
        // The opposite corner (nx dx, ny dx) lies on the stored east/north faces.
        assert_eq!(interpolate_velocity(&g, 4.0, 4.0), [1.0, 2.0]);
    }

    /// Face velocities are linear in the lattice index, so bilinear
    /// interpolation must reproduce the linear function exactly at the true
    /// face positions: `u_x(i, j)` at `((i + 1) dx, (j + ½) dx)` and `u_y(i, j)`
    /// at `((i + ½) dx, (j + 1) dx)`.  Treating the faces as cell-centred
    /// values shifts both fields by half a cell.
    #[test]
    fn interpolate_velocity_samples_the_staggered_faces() {
        let dx = 0.25;
        let mut g = MacGrid::new(6, 5, dx);
        for j in 0..5 {
            for i in 0..6 {
                let c = g.cell_mut(i, j);
                c.u_x = 1.0 + 2.0 * i as f64 + 3.0 * j as f64;
                c.u_y = -4.0 + 5.0 * i as f64 - 7.0 * j as f64;
            }
        }
        // Exactly on the faces.
        for j in 0..5 {
            for i in 0..6 {
                let ux = interpolate_velocity(&g, (i as f64 + 1.0) * dx, (j as f64 + 0.5) * dx);
                assert!((ux[0] - g.cell(i, j).u_x).abs() < 1e-12, "u_x({i}, {j})");
                let uy = interpolate_velocity(&g, (i as f64 + 0.5) * dx, (j as f64 + 1.0) * dx);
                assert!((uy[1] - g.cell(i, j).u_y).abs() < 1e-12, "u_y({i}, {j})");
            }
        }
        // Between the faces, in the region covered by the u_x lattice.
        for &(x, y) in &[(0.5, 0.4), (0.9, 0.7), (1.2, 0.3), (0.7, 0.9)] {
            let v = interpolate_velocity(&g, x, y);
            let want_x = 1.0 + 2.0 * (x / dx - 1.0) + 3.0 * (y / dx - 0.5);
            let want_y = -4.0 + 5.0 * (x / dx - 0.5) - 7.0 * (y / dx - 1.0);
            assert!((v[0] - want_x).abs() < 1e-12, "u_x at ({x}, {y})");
            assert!((v[1] - want_y).abs() < 1e-12, "u_y at ({x}, {y})");
        }
    }

    /// The west boundary face carries zero velocity: `u_x` decays linearly to
    /// zero over the first cell, `u_y` over the south wall.
    #[test]
    fn interpolate_velocity_boundary_faces_are_zero() {
        let g = uniform_grid(4, 4, 1.0, 2.0);
        // x = 0 wall: u_x = 0 for any height; x = 0.5 is halfway to the first face.
        assert_eq!(interpolate_velocity(&g, 0.0, 2.0)[0], 0.0);
        assert!((interpolate_velocity(&g, 0.5, 2.0)[0] - 0.5).abs() < 1e-12);
        assert!((interpolate_velocity(&g, 1.0, 2.0)[0] - 1.0).abs() < 1e-12);
        // y = 0 wall: u_y = 0 for any x.
        assert_eq!(interpolate_velocity(&g, 2.0, 0.0)[1], 0.0);
        assert!((interpolate_velocity(&g, 2.0, 0.5)[1] - 1.0).abs() < 1e-12);
        assert!((interpolate_velocity(&g, 2.0, 1.0)[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn interpolate_velocity_empty_grid_is_zero() {
        let g = MacGrid::new(0, 3, 1.0);
        assert_eq!(interpolate_velocity(&g, 0.5, 0.5), [0.0, 0.0]);
        let g = MacGrid::new(3, 0, 1.0);
        assert_eq!(interpolate_velocity(&g, 0.5, 0.5), [0.0, 0.0]);
    }

    // ── VorticityConfinement ─────────────────────────────────────────────────

    #[test]
    fn vorticity_zero_flow() {
        let g = MacGrid::new(5, 5, 1.0);
        let vc = VorticityConfinement::new(0.1);
        let omega = vc.compute_vorticity(&g);
        assert!(omega.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn vorticity_shear_flow() {
        let mut g = MacGrid::new(5, 5, 1.0);
        // u_x increases with j: ∂u_x/∂y = 1, so ω = ∂u_y/∂x − ∂u_x/∂y = -1.
        for j in 0..5 {
            for i in 0..5 {
                g.cell_mut(i, j).u_x = j as f64;
            }
        }
        let vc = VorticityConfinement::new(0.1);
        let omega = vc.compute_vorticity(&g);
        for j in 0..5 {
            for i in 0..5 {
                let interior = (1..4).contains(&i) && (1..4).contains(&j);
                let want = if interior { -1.0 } else { 0.0 };
                assert!(
                    (omega[j * 5 + i] - want).abs() < 1e-12,
                    "omega({i}, {j}) = {}",
                    omega[j * 5 + i]
                );
            }
        }
    }

    /// Solid-body rotation `u = -Ω (y - yc)`, `v = Ω (x - xc)` has the uniform
    /// vorticity `2 Ω`, with every face velocity taken at its true position.
    #[test]
    fn vorticity_solid_body_rotation() {
        let (nx, ny, dx, rate) = (9, 7, 0.2, 1.5);
        let mut g = MacGrid::new(nx, ny, dx);
        let (xc, yc) = (0.5 * nx as f64 * dx, 0.5 * ny as f64 * dx);
        for j in 0..ny {
            for i in 0..nx {
                let c = g.cell_mut(i, j);
                // u_x at ((i + 1) dx, (j + ½) dx), u_y at ((i + ½) dx, (j + 1) dx).
                c.u_x = -rate * ((j as f64 + 0.5) * dx - yc);
                c.u_y = rate * ((i as f64 + 0.5) * dx - xc);
            }
        }
        let omega = VorticityConfinement::new(0.1).compute_vorticity(&g);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                assert!(
                    (omega[j * nx + i] - 2.0 * rate).abs() < 1e-12,
                    "omega({i}, {j}) = {}",
                    omega[j * nx + i]
                );
            }
        }
    }

    /// For `u_x = y²` (zero `u_y`) the vorticity at a cell centre is
    /// `-2 y_c` exactly: the two derivative terms must refer to the cell centre,
    /// not to different cell corners.
    #[test]
    fn vorticity_is_evaluated_at_the_cell_centre() {
        let (nx, ny, dx) = (6, 6, 0.5);
        let mut g = MacGrid::new(nx, ny, dx);
        for j in 0..ny {
            for i in 0..nx {
                let y_face = (j as f64 + 0.5) * dx;
                g.cell_mut(i, j).u_x = y_face * y_face;
            }
        }
        let omega = VorticityConfinement::new(0.1).compute_vorticity(&g);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let y_centre = (j as f64 + 0.5) * dx;
                assert!(
                    (omega[j * nx + i] + 2.0 * y_centre).abs() < 1e-12,
                    "omega({i}, {j}) = {}",
                    omega[j * nx + i]
                );
            }
        }
    }

    /// Counter-clockwise Lamb-Oseen vortex around the grid centre,
    /// `u_θ(r) = (1 - exp(-r²/w²)) / r`, whose vorticity `2 exp(-r²/w²) / w²`
    /// is positive and falls monotonically away from the core.  Face
    /// velocities are sampled at their true positions.
    fn lamb_oseen_vortex(n: usize, dx: f64) -> MacGrid {
        let mut g = MacGrid::new(n, n, dx);
        let centre = 0.5 * n as f64 * dx;
        let width = 0.3;
        // u_θ / r = (1 - exp(-r²/w²)) / r²; `exp_m1` keeps it accurate near the core.
        let swirl_over_r = |x: f64, y: f64| {
            let r2 = (x - centre).powi(2) + (y - centre).powi(2);
            -(-r2 / (width * width)).exp_m1() / r2
        };
        for j in 0..n {
            for i in 0..n {
                let (xu, yu) = ((i as f64 + 1.0) * dx, (j as f64 + 0.5) * dx);
                let (xv, yv) = ((i as f64 + 0.5) * dx, (j as f64 + 1.0) * dx);
                let c = g.cell_mut(i, j);
                c.u_x = -(yu - centre) * swirl_over_r(xu, yu);
                c.u_y = (xv - centre) * swirl_over_r(xv, yv);
            }
        }
        g
    }

    /// Angular momentum `Σ (x - c) u_y - (y - c) u_x` of the face velocities
    /// about the grid centre `c`.
    fn angular_momentum(g: &MacGrid) -> f64 {
        let centre = 0.5 * g.nx as f64 * g.dx;
        let mut total = 0.0;
        for j in 0..g.ny {
            for i in 0..g.nx {
                // u_x(i, j) sits at height (j + ½) dx, u_y(i, j) at abscissa (i + ½) dx.
                let y_of_u_x = (j as f64 + 0.5) * g.dx;
                let x_of_u_y = (i as f64 + 0.5) * g.dx;
                total += -(y_of_u_x - centre) * g.cell(i, j).u_x
                    + (x_of_u_y - centre) * g.cell(i, j).u_y;
            }
        }
        total
    }

    /// The force `ε dx (N × ω)` acts along `N × ω`; with `N` pointing at the
    /// vortex core it runs around the core in the sense of the rotation, so it
    /// adds angular momentum.
    #[test]
    fn vorticity_confinement_reinforces_the_rotation() {
        let mut g = lamb_oseen_vortex(15, 0.1);
        let before = angular_momentum(&g);
        assert!(before > 0.0, "the vortex must be counter-clockwise");
        VorticityConfinement::new(0.5).apply_confinement_force(&mut g, 0.01);
        let after = angular_momentum(&g);
        assert!(
            after > before * (1.0 + 1e-4),
            "angular momentum {before} -> {after}"
        );
    }

    /// Cell by cell the update is `Δu_x = Δt ε dx ω N_y`,
    /// `Δu_y = -Δt ε dx ω N_x`, recomputed here from the public vorticity.
    #[test]
    fn vorticity_confinement_applies_the_documented_force() {
        let (strength, dt) = (0.3, 0.02);
        let before = lamb_oseen_vortex(11, 0.1);
        let mut after = before.clone();
        let vc = VorticityConfinement::new(strength);
        vc.apply_confinement_force(&mut after, dt);
        let omega = vc.compute_vorticity(&before);
        let (nx, dx) = (before.nx, before.dx);
        let mut changed = 0;
        for j in 1..before.ny - 1 {
            for i in 1..nx - 1 {
                let abs_at = |ii: usize, jj: usize| omega[jj * nx + ii].abs();
                let gx = (abs_at(i + 1, j) - abs_at(i - 1, j)) / (2.0 * dx);
                let gy = (abs_at(i, j + 1) - abs_at(i, j - 1)) / (2.0 * dx);
                let len = (gx * gx + gy * gy).sqrt();
                let (want_dx, want_dy) = if len < 1e-12 {
                    (0.0, 0.0)
                } else {
                    let w = omega[j * nx + i];
                    (
                        strength * dx * w * gy / len * dt,
                        -strength * dx * w * gx / len * dt,
                    )
                };
                let got_dx = after.cell(i, j).u_x - before.cell(i, j).u_x;
                let got_dy = after.cell(i, j).u_y - before.cell(i, j).u_y;
                assert!((got_dx - want_dx).abs() < 1e-12, "du_x({i}, {j})");
                assert!((got_dy - want_dy).abs() < 1e-12, "du_y({i}, {j})");
                if want_dx.abs() > 1e-9 || want_dy.abs() > 1e-9 {
                    changed += 1;
                }
            }
        }
        assert!(changed > 20, "only {changed} cells received a force");
    }

    #[test]
    fn vorticity_confinement_zero_strength_or_dt_changes_nothing() {
        let reference = lamb_oseen_vortex(9, 0.1);
        let mut g = reference.clone();
        VorticityConfinement::new(0.0).apply_confinement_force(&mut g, 0.01);
        VorticityConfinement::new(0.4).apply_confinement_force(&mut g, 0.0);
        for (a, b) in g.cells.iter().zip(reference.cells.iter()) {
            assert_eq!((a.u_x, a.u_y), (b.u_x, b.u_y));
        }
    }

    #[test]
    fn vorticity_confinement_leaves_irrotational_flow_alone() {
        // Uniform flow has no vorticity, hence no confinement force.
        let reference = uniform_grid(6, 6, 0.1, -0.1);
        let mut g = reference.clone();
        VorticityConfinement::new(0.2).apply_confinement_force(&mut g, 0.01);
        for (a, b) in g.cells.iter().zip(reference.cells.iter()) {
            assert_eq!((a.u_x, a.u_y), (b.u_x, b.u_y));
        }
    }

    #[test]
    fn vorticity_len_equals_grid_size() {
        let g = MacGrid::new(6, 7, 0.5);
        let vc = VorticityConfinement::new(0.1);
        let omega = vc.compute_vorticity(&g);
        assert_eq!(omega.len(), 6 * 7);
    }

    // ── FluidObstacle ────────────────────────────────────────────────────────

    #[test]
    fn obstacle_set_solid() {
        let mut g = MacGrid::new(5, 5, 1.0);
        let obs = FluidObstacle::new(vec![(2, 2), (3, 2)]);
        obs.set_solid(&mut g);
        assert!(!g.cell(2, 2).is_fluid);
        assert!(!g.cell(3, 2).is_fluid);
        assert!(g.cell(1, 1).is_fluid);
    }

    #[test]
    fn obstacle_apply_no_slip() {
        let mut g = MacGrid::new(5, 5, 1.0);
        g.cell_mut(2, 2).u_x = 10.0;
        g.cell_mut(2, 2).u_y = 5.0;
        let obs = FluidObstacle::new(vec![(2, 2)]);
        obs.apply_no_slip(&mut g);
        assert_eq!(g.cell(2, 2).u_x, 0.0);
        assert_eq!(g.cell(2, 2).u_y, 0.0);
    }

    #[test]
    fn obstacle_out_of_bounds_no_panic() {
        let mut g = uniform_grid(4, 4, 1.0, 2.0);
        let before = g.clone();
        let obs = FluidObstacle::new(vec![(100, 100), (4, 0), (0, 4)]);
        obs.set_solid(&mut g); // must not panic
        obs.apply_no_slip(&mut g);
        // Cells outside the grid are ignored: nothing changed.
        for (a, b) in g.cells.iter().zip(before.cells.iter()) {
            assert_eq!(
                (a.is_fluid, a.u_x, a.u_y, a.pressure),
                (b.is_fluid, b.u_x, b.u_y, b.pressure)
            );
        }
    }

    #[test]
    fn obstacle_multiple_cells() {
        let mut g = MacGrid::new(10, 10, 1.0);
        let cells: Vec<_> = (2..5).flat_map(|i| (2..5).map(move |j| (i, j))).collect();
        let obs = FluidObstacle::new(cells);
        obs.set_solid(&mut g);
        assert!(!g.cell(3, 3).is_fluid);
        assert!(g.cell(0, 0).is_fluid);
        assert_eq!(g.cells.iter().filter(|c| !c.is_fluid).count(), 9);
    }

    // ── Viscous diffusion ────────────────────────────────────────────────────

    /// Σ of the four face neighbours of `(i, j)` in the `n × n` field `f`,
    /// with zero outside the grid.
    fn neighbour_total(f: &[f64], n: usize, i: usize, j: usize) -> f64 {
        let at = |ii: isize, jj: isize| {
            if ii < 0 || jj < 0 || ii >= n as isize || jj >= n as isize {
                0.0
            } else {
                f[jj as usize * n + ii as usize]
            }
        };
        let (ii, jj) = (i as isize, j as isize);
        at(ii + 1, jj) + at(ii - 1, jj) + at(ii, jj + 1) + at(ii, jj - 1)
    }

    fn kinetic_energy(g: &MacGrid) -> f64 {
        g.cells
            .iter()
            .map(|c| 0.5 * (c.u_x * c.u_x + c.u_y * c.u_y))
            .sum()
    }

    /// Converged Jacobi sweeps solve `(1 + 4 α) u' − α Σ u'_neighbours = u`.
    #[test]
    fn diffuse_velocity_solves_the_implicit_system() {
        let (nu, dt, dx, n) = (0.2, 0.5, 0.5, 10);
        let alpha = nu * dt / (dx * dx);
        let source = random_grid(n, dx, 17);
        let mut g = source.clone();
        g.diffuse_velocity(nu, dt, 300)
            .expect("valid diffusion parameters");
        let components: [(fn(&MacCell) -> f64, &str); 2] = [(|c| c.u_x, "u_x"), (|c| c.u_y, "u_y")];
        for (pick, name) in components {
            let old: Vec<f64> = source.cells.iter().map(pick).collect();
            let new: Vec<f64> = g.cells.iter().map(pick).collect();
            let mut moved = 0.0_f64;
            for j in 0..n {
                for i in 0..n {
                    let residual = (1.0 + 4.0 * alpha) * new[j * n + i]
                        - alpha * neighbour_total(&new, n, i, j)
                        - old[j * n + i];
                    assert!(
                        residual.abs() < 1e-12,
                        "{name}({i}, {j}) residual {residual}"
                    );
                    moved = moved.max((new[j * n + i] - old[j * n + i]).abs());
                }
            }
            assert!(moved > 0.1, "{name} barely diffused: {moved}");
        }
    }

    /// The sine mode `sin(π (i+1)/(n+1)) sin(π (j+1)/(n+1))` vanishes at the
    /// zero ghost faces, so one implicit step multiplies it by exactly
    /// `1 / (1 + 4 α (1 − cos(π/(n+1))))`.
    #[test]
    fn diffuse_velocity_damps_an_eigenmode_by_the_implicit_factor() {
        let (nu, dt, dx, n) = (0.05, 0.4, 0.2, 9);
        let alpha = nu * dt / (dx * dx);
        let theta = std::f64::consts::PI / (n as f64 + 1.0);
        let mut g = MacGrid::new(n, n, dx);
        for j in 0..n {
            for i in 0..n {
                g.cell_mut(i, j).u_x =
                    ((i as f64 + 1.0) * theta).sin() * ((j as f64 + 1.0) * theta).sin();
            }
        }
        let initial = g.clone();
        g.diffuse_velocity(nu, dt, 400)
            .expect("valid diffusion parameters");
        let factor = 1.0 / (1.0 + 4.0 * alpha * (1.0 - theta.cos()));
        assert!(factor < 0.9, "the test mode must actually decay");
        for (a, b) in g.cells.iter().zip(initial.cells.iter()) {
            assert!((a.u_x - factor * b.u_x).abs() < 1e-12);
            assert_eq!(a.u_y, 0.0);
        }
    }

    #[test]
    fn diffuse_velocity_zero_viscosity_or_sweeps_is_identity() {
        let source = random_grid(7, 0.1, 3);
        let mut g = source.clone();
        g.diffuse_velocity(0.0, 0.01, 50)
            .expect("valid diffusion parameters");
        g.diffuse_velocity(1.0, 0.01, 0)
            .expect("valid diffusion parameters");
        for (a, b) in g.cells.iter().zip(source.cells.iter()) {
            assert_eq!((a.u_x, a.u_y), (b.u_x, b.u_y));
        }
    }

    /// The implicit scheme is unconditionally stable: a huge `α` with a few
    /// sweeps stays bounded by the initial velocities and loses energy.
    #[test]
    fn diffuse_velocity_is_bounded_for_huge_diffusion_numbers() {
        let source = random_grid(8, 0.01, 11);
        let initial_peak = source
            .cells
            .iter()
            .fold(0.0_f64, |m, c| m.max(c.u_x.abs()).max(c.u_y.abs()));
        let mut g = source.clone();
        // alpha = nu dt / dx^2 = 1e4.
        g.diffuse_velocity(1.0, 1.0, 3)
            .expect("valid diffusion parameters");
        for c in &g.cells {
            assert!(c.u_x.is_finite() && c.u_x.abs() <= initial_peak);
            assert!(c.u_y.is_finite() && c.u_y.abs() <= initial_peak);
        }
        assert!(kinetic_energy(&g) < kinetic_energy(&source));
    }

    #[test]
    fn diffuse_velocity_skips_solid_cells() {
        let mut g = MacGrid::new(7, 7, 1.0);
        g.cell_mut(3, 3).is_fluid = false;
        g.cell_mut(3, 3).u_x = 5.0;
        g.cell_mut(3, 3).u_y = -5.0;
        g.diffuse_velocity(0.3, 1.0, 40)
            .expect("valid diffusion parameters");
        // The solid faces are fixed values ...
        assert_eq!((g.cell(3, 3).u_x, g.cell(3, 3).u_y), (5.0, -5.0));
        // ... which the fluid neighbours diffuse towards.
        assert!(g.cell(2, 3).u_x > 0.1, "{}", g.cell(2, 3).u_x);
        assert!(g.cell(4, 3).u_x > 0.1, "{}", g.cell(4, 3).u_x);
        assert!(g.cell(3, 2).u_y < -0.1, "{}", g.cell(3, 2).u_y);
    }

    #[test]
    fn diffuse_velocity_rejects_invalid_parameters() {
        let mut g = random_grid(5, 0.1, 9);
        let before = g.clone();
        for &(nu, dt) in &[
            (-1e-3, 0.01),
            (f64::NAN, 0.01),
            (f64::INFINITY, 0.01),
            (1e-3, 0.0),
            (1e-3, -0.01),
            (1e-3, f64::NAN),
        ] {
            assert!(g.diffuse_velocity(nu, dt, 10).is_err(), "{nu} {dt}");
        }
        for (a, b) in g.cells.iter().zip(before.cells.iter()) {
            assert_eq!((a.u_x, a.u_y), (b.u_x, b.u_y));
        }
    }

    // ── FluidSimGpu ──────────────────────────────────────────────────────────

    /// With enough sweeps one step leaves a divergence-free velocity, however
    /// divergent the starting field was; before the fix `step` removed only
    /// `dt` (here 1 %) of the divergence.
    #[test]
    fn fluid_sim_gpu_step_projects_the_velocity() {
        let mut sim = FluidSimGpu::new(12, 12, 0.1, 1e-3, 1000.0);
        sim.grid = random_grid(12, 0.1, 31);
        sim.pressure_iterations = 400;
        sim.step(0.01).expect("valid step parameters");
        assert!(
            sim.grid
                .cells
                .iter()
                .all(|c| c.u_x.is_finite() && c.u_y.is_finite())
        );
        let left = max_interior_divergence(&sim.grid);
        assert!(left < 1e-3, "max|div| after one step: {left}");
        // The step really did something to the field.
        let kinetic = kinetic_energy(&sim.grid);
        assert!(kinetic > 1.0, "kinetic energy {kinetic}");
    }

    /// `step` is the documented pipeline with `ν = viscosity / density`.
    #[test]
    fn fluid_sim_gpu_step_matches_the_documented_pipeline() {
        let (mu, rho, dt) = (0.5, 2.0, 0.05);
        let mut sim = FluidSimGpu::new(8, 8, 0.2, mu, rho);
        sim.grid = random_grid(8, 0.2, 77);
        sim.pressure_iterations = 7;
        sim.diffusion_iterations = 5;
        let mut by_hand = sim.grid.clone();
        sim.step(dt).expect("valid step parameters");
        by_hand.advect_velocity(dt);
        by_hand
            .diffuse_velocity(mu / rho, dt, 5)
            .expect("valid diffusion parameters");
        by_hand
            .pressure_solve_jacobi(7, dt, rho)
            .expect("valid solver parameters");
        by_hand.project(dt, rho).expect("valid solver parameters");
        for (a, b) in sim.grid.cells.iter().zip(by_hand.cells.iter()) {
            assert_eq!((a.u_x, a.u_y, a.pressure), (b.u_x, b.u_y, b.pressure));
        }
    }

    /// Viscosity and density enter through `ν = μ/ρ` only (the density cancels
    /// in the projection): `(2, 2)` behaves like `(1, 1)`, `(1, 2)` does not.
    #[test]
    fn fluid_sim_gpu_uses_viscosity_and_density() {
        let run = |mu: f64, rho: f64| {
            let mut sim = FluidSimGpu::new(8, 8, 0.1, mu, rho);
            sim.grid = random_grid(8, 0.1, 5);
            sim.step(0.01).expect("valid step parameters");
            sim.grid
        };
        let base = run(1.0, 1.0);
        let scaled = run(2.0, 2.0);
        let thinner = run(1.0, 2.0);
        let mut same = 0.0_f64;
        let mut different = 0.0_f64;
        for ((a, b), c) in base
            .cells
            .iter()
            .zip(scaled.cells.iter())
            .zip(thinner.cells.iter())
        {
            same = same.max((a.u_x - b.u_x).abs()).max((a.u_y - b.u_y).abs());
            different = different
                .max((a.u_x - c.u_x).abs())
                .max((a.u_y - c.u_y).abs());
        }
        assert!(same < 1e-10, "equal nu must give equal fields: {same}");
        assert!(
            different > 1e-2,
            "halving nu must change the field: {different}"
        );
    }

    #[test]
    fn fluid_sim_gpu_step_rejects_invalid_parameters() {
        let mut sim = FluidSimGpu::new(6, 6, 0.1, 1e-3, 1000.0);
        sim.grid = random_grid(6, 0.1, 13);
        let before = sim.grid.clone();
        for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            assert!(sim.step(dt).is_err(), "dt = {dt}");
        }
        sim.density = 0.0;
        assert!(sim.step(0.01).is_err());
        sim.density = 1000.0;
        sim.viscosity = -1.0;
        assert!(sim.step(0.01).is_err());
        sim.viscosity = f64::NAN;
        assert!(sim.step(0.01).is_err());
        for (a, b) in sim.grid.cells.iter().zip(before.cells.iter()) {
            assert_eq!((a.u_x, a.u_y, a.pressure), (b.u_x, b.u_y, b.pressure));
        }
    }

    #[test]
    fn fluid_sim_gpu_quiescent_stable() {
        let mut sim = FluidSimGpu::new(8, 8, 0.1, 1e-3, 1000.0);
        for _ in 0..5 {
            sim.step(0.001).expect("valid step parameters");
        }
        // All velocities should remain zero
        for c in &sim.grid.cells {
            assert!(c.u_x.abs() < 1e-10, "u_x = {}", c.u_x);
            assert!(c.u_y.abs() < 1e-10, "u_y = {}", c.u_y);
        }
    }

    #[test]
    fn fluid_sim_gpu_fields_accessible() {
        let sim = FluidSimGpu::new(4, 4, 0.05, 0.001, 998.0);
        assert_eq!(sim.grid.nx, 4);
        assert!((sim.viscosity - 0.001).abs() < 1e-12);
        assert!((sim.density - 998.0).abs() < 1e-12);
        assert_eq!(sim.pressure_iterations, FluidSimGpu::DEFAULT_ITERATIONS);
        assert_eq!(sim.diffusion_iterations, FluidSimGpu::DEFAULT_ITERATIONS);
        assert_eq!(FluidSimGpu::DEFAULT_ITERATIONS, 20);
    }

    /// More pressure sweeps per step remove more of the divergence.
    #[test]
    fn fluid_sim_gpu_pressure_iterations_field_controls_accuracy() {
        let divergence_after = |sweeps: usize| {
            let mut sim = FluidSimGpu::new(10, 10, 0.1, 0.0, 1.0);
            sim.grid = random_grid(10, 0.1, 2);
            sim.pressure_iterations = sweeps;
            sim.step(0.01).expect("valid step parameters");
            max_interior_divergence(&sim.grid)
        };
        let coarse = divergence_after(5);
        let fine = divergence_after(300);
        assert!(
            fine < 1e-2 * coarse,
            "5 sweeps: {coarse}, 300 sweeps: {fine}"
        );
    }

    // ── Advection ────────────────────────────────────────────────────────────

    #[test]
    fn advect_zero_dt_no_change() {
        let mut g = uniform_grid(5, 5, 1.0, 0.5);
        let before: Vec<_> = g.cells.iter().map(|c| [c.u_x, c.u_y]).collect();
        g.advect_velocity(0.0);
        let after: Vec<_> = g.cells.iter().map(|c| [c.u_x, c.u_y]).collect();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b[0] - a[0]).abs() < 1e-10);
            assert!((b[1] - a[1]).abs() < 1e-10);
        }
    }

    #[test]
    fn advect_does_not_explode() {
        let mut g = uniform_grid(6, 6, 0.3, 0.2);
        for _ in 0..10 {
            g.advect_velocity(0.01);
        }
        // Semi-Lagrangian values are convex combinations of old face values
        // (and of the zero wall faces), so they never exceed the initial peak.
        for c in &g.cells {
            assert!(c.u_x.is_finite() && c.u_x.abs() <= 0.3 + 1e-12);
            assert!(c.u_y.is_finite() && c.u_y.abs() <= 0.2 + 1e-12);
        }
    }

    /// With `dt = dx` and a uniform unit `u_x` flow the `u_y` field moves
    /// exactly one face east: the faces really are traced back from their own
    /// positions.
    #[test]
    fn advect_transports_u_y_with_a_uniform_u_x_flow() {
        let dx = 1.0;
        let mut g = MacGrid::new(8, 8, dx);
        for c in g.cells.iter_mut() {
            c.u_x = 1.0;
        }
        g.cell_mut(3, 3).u_y = 1.0;
        g.advect_velocity(dx);
        for j in 0..8 {
            for i in 0..8 {
                let want = if (i, j) == (4, 3) { 1.0 } else { 0.0 };
                assert!(
                    (g.cell(i, j).u_y - want).abs() < 1e-12,
                    "u_y({i}, {j}) = {}",
                    g.cell(i, j).u_y
                );
            }
        }
    }

    #[test]
    fn advect_transports_u_x_with_a_uniform_u_y_flow() {
        let dx = 1.0;
        let mut g = MacGrid::new(8, 8, dx);
        for c in g.cells.iter_mut() {
            c.u_y = 1.0;
        }
        g.cell_mut(3, 3).u_x = 1.0;
        g.advect_velocity(dx);
        for j in 0..8 {
            for i in 0..8 {
                let want = if (i, j) == (3, 4) { 1.0 } else { 0.0 };
                assert!(
                    (g.cell(i, j).u_x - want).abs() < 1e-12,
                    "u_x({i}, {j}) = {}",
                    g.cell(i, j).u_x
                );
            }
        }
    }

    /// Away from the west and south walls a uniform flow is a fixed point of
    /// the advection; at the walls the zero boundary faces slow it down.
    #[test]
    fn advect_leaves_the_interior_of_a_uniform_flow_unchanged() {
        let mut g = uniform_grid(6, 6, 0.3, 0.2);
        g.advect_velocity(0.5);
        for j in 0..6 {
            for i in 0..6 {
                let c = g.cell(i, j);
                // First column: traced back 0.15 of a cell towards the zero wall face.
                let want_x = if i == 0 { 0.3 * 0.85 } else { 0.3 };
                let want_y = if j == 0 { 0.2 * 0.9 } else { 0.2 };
                assert!((c.u_x - want_x).abs() < 1e-12, "u_x({i}, {j}) = {}", c.u_x);
                assert!((c.u_y - want_y).abs() < 1e-12, "u_y({i}, {j}) = {}", c.u_y);
            }
        }
    }
}
