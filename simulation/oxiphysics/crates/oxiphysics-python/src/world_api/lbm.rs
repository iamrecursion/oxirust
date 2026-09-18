// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! LBM (Lattice Boltzmann Method) Simulation.

use crate::lbm_api::equilibrium;
use crate::lbm_api::forcing::{forced_velocity, guo_source, stored_velocity};
use pyo3::prelude::*;

#[cfg(test)]
mod tests;

// ===========================================================================
// LBM (Lattice Boltzmann Method) Simulation
// ===========================================================================

/// Configuration for a 2-D D2Q9 Lattice-Boltzmann simulation.
///
/// All quantities are in lattice units: the cell size and the time step are 1.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLbmConfig {
    /// Grid width (number of cells in X direction).
    #[pyo3(get, set)]
    pub width: usize,
    /// Grid height (number of cells in Y direction).
    #[pyo3(get, set)]
    pub height: usize,
    /// Kinematic viscosity (nu). Controls the relaxation rate.
    #[pyo3(get, set)]
    pub viscosity: f64,
    /// External body-force acceleration `[fx, fy]` applied to the fluid.
    #[pyo3(get, set)]
    pub body_force: [f64; 2],
    /// Close the domain with no-slip walls on all four sides.
    ///
    /// The walls are half-way bounce-back walls half a cell outside the
    /// outermost cells (at `x = -1/2`, `x = width - 1/2`, `y = -1/2` and
    /// `y = height - 1/2`), so the fluid fills all `width x height` cells.
    /// Without walls the domain is periodic in both directions.
    #[pyo3(get, set)]
    pub walls: bool,
    /// Velocity `[ux, uy]` of the top wall (the lid at `y = height - 1/2`).
    ///
    /// The lid is a moving no-slip wall, so it only exists together with
    /// `walls`; on a periodic domain (`walls == false`) the value has no
    /// effect. Keep `uy` at zero in a closed box: a velocity normal to the
    /// wall makes it a source or sink of mass.
    #[pyo3(get, set)]
    pub lid_velocity: [f64; 2],
}

#[pymethods]
impl PyLbmConfig {
    /// Create a new LBM configuration: a periodic domain without walls, lid
    /// or body force.
    #[new]
    pub fn new(width: usize, height: usize, viscosity: f64) -> Self {
        Self {
            width,
            height,
            viscosity: viscosity.max(1e-6),
            body_force: [0.0; 2],
            walls: false,
            lid_velocity: [0.0; 2],
        }
    }

    /// Create the classic lid-driven cavity: a closed 64×64 box whose top
    /// wall moves with `U = 0.1` along x.
    ///
    /// The viscosity `nu = 0.064` makes the Reynolds number
    /// `Re = U H / nu = 0.1 * 64 / 0.064 = 100`. The flow settles into one
    /// primary vortex (clockwise for a lid moving towards +x), so along the
    /// vertical centre line `ux` is positive below the lid and negative in the
    /// lower half of the box.
    #[staticmethod]
    pub fn lid_driven_cavity() -> Self {
        let mut config = Self::new(64, 64, 0.064);
        config.walls = true;
        config.lid_velocity = [0.1, 0.0];
        config
    }

    /// Compute the relaxation parameter omega from viscosity.
    ///
    /// `omega = 1 / (3 * nu + 0.5)`
    pub fn omega(&self) -> f64 {
        1.0 / (3.0 * self.viscosity + 0.5)
    }
}

/// D2Q9 Lattice-Boltzmann grid simulation.
///
/// Stores distribution functions for all 9 velocity directions at every cell.
/// Uses the BGK collision operator with a single relaxation time and applies
/// a body force with the Guo scheme, so a body acceleration `a` adds exactly
/// `rho * a` of momentum per step.
///
/// Boundaries:
/// - without `walls` the domain is periodic in x and y;
/// - with `walls` the box is closed by no-slip half-way bounce-back walls and
///   the top wall moves with the configured lid velocity;
/// - cells flagged with [`PyLbmGrid::set_solid`] are no-slip obstacles inside
///   the domain; a solid cell holds no fluid, reports zero velocity and zero
///   density, and its faces are bounce-back walls as well.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PyLbmGrid {
    /// Number of cells in X.
    width: usize,
    /// Number of cells in Y.
    height: usize,
    /// Relaxation rate omega.
    omega: f64,
    /// External body acceleration `[ax, ay]`.
    body_force: [f64; 2],
    /// Whether the domain is closed by no-slip walls (periodic otherwise).
    walls: bool,
    /// Velocity of the top wall (only used with `walls`).
    lid_velocity: [f64; 2],
    /// Solid-cell mask, indexed `[y * width + x]`.
    solid: Vec<bool>,
    /// Distribution functions, indexed `[cell_idx * 9 + q]`.
    f: Vec<f64>,
    /// Post-collision / swap buffer.
    f_tmp: Vec<f64>,
    /// Accumulated simulation steps.
    step_count: u64,
}

// D2Q9 weights and discrete velocities
const D2Q9_W: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
// ex, ey for each of the 9 directions
const D2Q9_EX: [f64; 9] = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0];
const D2Q9_EY: [f64; 9] = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];
/// Index of the direction opposite to each direction.
const D2Q9_OPP: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

/// Where the population that streams into a cell along one link comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Upstream {
    /// From the fluid cell with this index.
    Fluid(usize),
    /// The link is cut by a static no-slip wall half-way along it.
    Wall,
    /// The link is cut by the moving top wall (the lid).
    Lid,
}

#[pymethods]
impl PyLbmGrid {
    /// Create a new LBM grid from configuration. All cells are initialised to
    /// equilibrium with unit density and zero velocity.
    ///
    /// Under a body force the populations of a cell hold the momentum
    /// `rho (u - a / 2)` (see [`PyLbmGrid::velocity_at`]), so a fresh grid
    /// still reads zero velocity.
    #[new]
    pub fn new(config: &PyLbmConfig) -> Self {
        let n = config.width * config.height;
        let [ux, uy] = stored_velocity([0.0; 2], config.body_force);
        let rest = equilibrium(1.0, ux, uy);
        let f: Vec<f64> = rest.iter().copied().cycle().take(n * 9).collect();
        Self {
            width: config.width,
            height: config.height,
            omega: config.omega(),
            body_force: config.body_force,
            walls: config.walls,
            lid_velocity: config.lid_velocity,
            solid: vec![false; n],
            f_tmp: f.clone(),
            f,
            step_count: 0,
        }
    }

    /// Return the grid width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Return the grid height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Return the number of completed steps.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Advance the simulation by one LBM time step.
    ///
    /// 1. Collision: BGK relaxation towards the equilibrium of the physical
    ///    velocity `u = sum(e f) / rho + a / 2`, plus the Guo source term of
    ///    the body acceleration `a`, which adds exactly `rho * a` of momentum
    ///    per step. Solid cells are skipped.
    /// 2. Streaming: every population moves one cell along its direction.
    ///    A population that would come from outside a walled domain or from a
    ///    solid cell is instead the reflected population of the opposite
    ///    direction (half-way bounce-back: the wall sits midway along the
    ///    link), with the extra term `6 w_q rho (e_q . u_lid)` when the link
    ///    is cut by the moving lid. Without walls the domain wraps around
    ///    (periodic).
    pub fn step(&mut self) {
        self.collide();
        self.stream();
        self.step_count += 1;
    }

    /// Get the macroscopic velocity `[ux, uy]` at cell `(x, y)`.
    ///
    /// This is the physical velocity of the Guo forcing scheme,
    /// `u = sum(e f) / rho + a / 2` for the body acceleration `a`.
    /// Returns `[0.0, 0.0]` if coordinates are out of bounds or the cell is
    /// solid.
    pub fn velocity_at(&self, x: usize, y: usize) -> [f64; 2] {
        if x >= self.width || y >= self.height {
            return [0.0; 2];
        }
        let cell = y * self.width + x;
        if self.solid[cell] {
            return [0.0; 2];
        }
        let pop = &self.f[cell * 9..cell * 9 + 9];
        let rho: f64 = pop.iter().sum();
        let mut momentum = [0.0f64; 2];
        for (q, &fq) in pop.iter().enumerate() {
            momentum[0] += D2Q9_EX[q] * fq;
            momentum[1] += D2Q9_EY[q] * fq;
        }
        forced_velocity(momentum, rho, self.body_force)
    }

    /// Get the macroscopic density at cell `(x, y)`.
    ///
    /// Returns `0.0` if coordinates are out of bounds or the cell is solid.
    pub fn density_at(&self, x: usize, y: usize) -> f64 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let cell = y * self.width + x;
        if self.solid[cell] {
            return 0.0;
        }
        self.f[cell * 9..cell * 9 + 9].iter().sum()
    }

    /// Mark cell `(x, y)` as a no-slip solid cell (`solid == true`) or as
    /// fluid (`solid == false`).
    ///
    /// Returns `false` and changes nothing when the cell lies outside the grid.
    /// A cell that changes its kind is reset to the rest state (unit density,
    /// zero velocity): the fluid it held is discarded, so set the obstacles up
    /// before stepping. Solid cells are how walls are built on a periodic
    /// domain: solid cells in the first and last row make a channel that is
    /// periodic along x, with the walls half a cell inside those rows.
    pub fn set_solid(&mut self, x: usize, y: usize, solid: bool) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let cell = y * self.width + x;
        if self.solid[cell] != solid {
            self.solid[cell] = solid;
            let [ux, uy] = stored_velocity([0.0; 2], self.body_force);
            let rest = equilibrium(1.0, ux, uy);
            self.f[cell * 9..cell * 9 + 9].copy_from_slice(&rest);
            self.f_tmp[cell * 9..cell * 9 + 9].copy_from_slice(&rest);
        }
        true
    }

    /// Whether cell `(x, y)` is a solid cell (`false` outside the grid).
    pub fn is_solid(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && self.solid[y * self.width + x]
    }

    /// Return the entire velocity field as a flat `Vec<f64>` of `[ux, uy]` pairs,
    /// row-major (`y` outer, `x` inner). Solid cells read `[0, 0]`.
    pub fn velocity_field(&self) -> Vec<f64> {
        let n = self.width * self.height;
        let mut out = vec![0.0f64; n * 2];
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = y * self.width + x;
                let v = self.velocity_at(x, y);
                out[cell * 2] = v[0];
                out[cell * 2 + 1] = v[1];
            }
        }
        out
    }

    /// Return the entire density field as a flat `Vec<f64>`, row-major
    /// (`y` outer, `x` inner). Solid cells read `0`.
    pub fn density_field(&self) -> Vec<f64> {
        let n = self.width * self.height;
        let mut out = vec![0.0f64; n];
        for y in 0..self.height {
            for x in 0..self.width {
                out[y * self.width + x] = self.density_at(x, y);
            }
        }
        out
    }
}

impl PyLbmGrid {
    /// BGK collision with Guo forcing of every fluid cell: `f` → `f_tmp`.
    fn collide(&mut self) {
        let omega = self.omega;
        let accel = self.body_force;
        let forced = accel != [0.0; 2];
        for (cell, (pop, post)) in self
            .f
            .chunks_exact(9)
            .zip(self.f_tmp.chunks_exact_mut(9))
            .enumerate()
        {
            if self.solid[cell] {
                continue;
            }
            let rho: f64 = pop.iter().sum();
            let mut momentum = [0.0f64; 2];
            for (q, &fq) in pop.iter().enumerate() {
                momentum[0] += D2Q9_EX[q] * fq;
                momentum[1] += D2Q9_EY[q] * fq;
            }
            let u = forced_velocity(momentum, rho, accel);
            let force = [rho * accel[0], rho * accel[1]];
            let feq = equilibrium(rho, u[0], u[1]);
            for q in 0..9 {
                let source = if forced {
                    guo_source(omega, D2Q9_W[q], [D2Q9_EX[q], D2Q9_EY[q]], u, force)
                } else {
                    0.0
                };
                post[q] = pop[q] * (1.0 - omega) + feq[q] * omega + source;
            }
        }
    }

    /// Where the population of direction `q` that arrives in cell `(x, y)`
    /// comes from.
    fn upstream(&self, x: usize, y: usize, q: usize) -> Upstream {
        let width = self.width as isize;
        let height = self.height as isize;
        let sx = x as isize - D2Q9_EX[q] as isize;
        let sy = y as isize - D2Q9_EY[q] as isize;
        let (sx, sy) = if self.walls {
            // The lid takes the corner links as well: the cell below it then
            // receives the `+-U` corrections of both diagonals, which cancel,
            // so the lid injects momentum but no mass.
            if sy >= height {
                return Upstream::Lid;
            }
            if sx < 0 || sx >= width || sy < 0 {
                return Upstream::Wall;
            }
            (sx, sy)
        } else {
            (sx.rem_euclid(width), sy.rem_euclid(height))
        };
        let source = (sy * width + sx) as usize;
        if self.solid[source] {
            Upstream::Wall
        } else {
            Upstream::Fluid(source)
        }
    }

    /// Streaming with bounce-back: the post-collision populations in `f_tmp`
    /// move into `f`.
    fn stream(&mut self) {
        let lid = self.lid_velocity;
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = y * self.width + x;
                if self.solid[cell] {
                    continue;
                }
                let dst = cell * 9;
                for q in 0..9 {
                    let value = match self.upstream(x, y, q) {
                        Upstream::Fluid(source) => self.f_tmp[source * 9 + q],
                        Upstream::Wall => self.f_tmp[dst + D2Q9_OPP[q]],
                        Upstream::Lid => {
                            let rho: f64 = self.f_tmp[dst..dst + 9].iter().sum();
                            let e_dot_u = D2Q9_EX[q] * lid[0] + D2Q9_EY[q] * lid[1];
                            self.f_tmp[dst + D2Q9_OPP[q]] + 6.0 * D2Q9_W[q] * rho * e_dot_u
                        }
                    };
                    self.f[dst + q] = value;
                }
            }
        }
    }
}
