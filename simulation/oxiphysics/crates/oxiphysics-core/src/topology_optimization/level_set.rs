// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Level-set topology optimization with the Hamilton-Jacobi equation.
//!
//! The structure is the region `φ > 0` of a level-set function sampled at the
//! nodes of the finite-element grid: node `(row, col)` has the index
//! `row * (nx + 1) + col`, exactly as in [`super::fe_q4`]. The unit grid
//! spacing is the element width. The boundary `φ = 0` moves with the normal
//! speed `V` by the Hamilton-Jacobi equation `∂φ/∂t + V |∇φ| = 0`, which
//! carries the front along `∇φ`, that is *into* the structure for `V > 0`
//! (the structure shrinks) and out of it for `V < 0`.
//!
//! [`LevelSetTopOpt::hj_update`] advances the equation with the first-order
//! Godunov upwind scheme of Osher and Sethian under a CFL step limit, and
//! [`LevelSetTopOpt::reinitialize`] restores the signed-distance property
//! `|∇φ| = 1` by the Sussman-Smereka-Osher pseudo-time equation.
//! [`LevelSetTopOpt::run`] closes the loop with the Q4 finite-element
//! analysis of [`super::fe_q4`].

use super::fe_q4::{LinearSolver, Q4Model};
use super::{GridSize, MaterialParams, invalid_input, volume_fraction};
use crate::error::PhysicsResult;
use std::f64::consts::PI;

/// Courant number of the explicit upwind Hamilton-Jacobi update. The scheme
/// moves information by at most `CFL` grid spacings per step; `1/2` leaves a
/// margin for the two space dimensions (the limit is `1/√2`).
const HJ_CFL: f64 = 0.5;

/// Pseudo-time step of the reinitialization, in grid spacings. The sign
/// function is bounded by one, so the front speed is at most one spacing per
/// unit pseudo-time and `0.3` keeps the 2-D Godunov scheme monotone.
const REINIT_TIME_STEP: f64 = 0.3;

/// Lowest density of the smoothed Heaviside, the stiffness of void relative
/// to solid in the ersatz material.
const VOID_DENSITY: f64 = 1e-3;

/// Half-width of the smoothed Heaviside in grid spacings.
const HEAVISIDE_HALF_WIDTH: f64 = 1.0;

/// Level-set topology optimization using the Hamilton-Jacobi equation.
///
/// The structural boundary is implicitly represented as the zero contour of
/// the level-set function φ, positive inside the structure. The
/// Hamilton-Jacobi PDE governs its evolution: `∂φ/∂t + V_n |∇φ| = 0`.
#[derive(Debug, Clone)]
pub struct LevelSetTopOpt {
    /// Grid size.
    pub grid: GridSize,
    /// Level-set function values (one per grid node, length = (nx+1)*(ny+1)).
    pub phi: Vec<f64>,
    /// Volume fraction target.
    pub volume_fraction: f64,
    /// Largest time step of the Hamilton-Jacobi update. Steps are shortened
    /// further wherever the CFL condition demands it, see
    /// [`LevelSetTopOpt::hj_update`].
    pub dt: f64,
    /// Material parameters.
    pub material: MaterialParams,
    /// Compliance history.
    pub compliance_history: Vec<f64>,
    /// Number of iterations of [`LevelSetTopOpt::run`] between two
    /// reinitializations of φ; `0` never reinitializes.
    pub reinit_interval: usize,
    /// Pseudo-time steps of each reinitialization in
    /// [`LevelSetTopOpt::run`].
    pub reinit_steps: usize,
    /// Linear solver of the finite-element analysis in
    /// [`LevelSetTopOpt::run`].
    pub linear_solver: LinearSolver,
}

/// Outcome of [`LevelSetTopOpt::run`].
#[derive(Debug, Clone, PartialEq)]
pub struct LevelSetResult {
    /// Element densities of the final level-set function, one per element in
    /// row-major order (see [`LevelSetTopOpt::element_density`]).
    pub density: Vec<f64>,
    /// Compliance `fᵀu` of the final design.
    pub compliance: f64,
    /// Compliance at the start of every iteration; the first entry belongs to
    /// the initial level-set function.
    pub compliance_history: Vec<f64>,
    /// Mean element density after every iteration.
    pub volume_history: Vec<f64>,
    /// Mean element density of the final design.
    pub volume_fraction: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Nodal displacements of the final design under the loads.
    pub displacement: Vec<f64>,
}

impl LevelSetTopOpt {
    /// Creates a new `LevelSetTopOpt` with a circular initial inclusion.
    pub fn new(grid: GridSize, volume_fraction: f64, dt: f64, material: MaterialParams) -> Self {
        let nn = (grid.nx + 1) * (grid.ny + 1);
        let cx = grid.nx as f64 * 0.5;
        let cy = grid.ny as f64 * 0.5;
        let r = (grid.nx.min(grid.ny) as f64) * 0.4;
        let phi: Vec<f64> = (0..nn)
            .map(|idx| {
                let row = idx / (grid.nx + 1);
                let col = idx % (grid.nx + 1);
                r - ((col as f64 - cx).powi(2) + (row as f64 - cy).powi(2)).sqrt()
            })
            .collect();
        Self {
            grid,
            phi,
            volume_fraction,
            dt,
            material,
            compliance_history: Vec::new(),
            reinit_interval: 5,
            reinit_steps: 10,
            linear_solver: LinearSolver::Auto,
        }
    }

    /// Sets how often, and with how many pseudo-time steps, [`LevelSetTopOpt::run`]
    /// reinitializes φ. An `interval` of `0` disables reinitialization.
    pub fn with_reinitialization(mut self, interval: usize, steps: usize) -> Self {
        self.reinit_interval = interval;
        self.reinit_steps = steps;
        self
    }

    /// Sets the linear solver of the finite-element analysis.
    pub fn with_linear_solver(mut self, solver: LinearSolver) -> Self {
        self.linear_solver = solver;
        self
    }

    /// Extracts element density from level-set (Heaviside of nodal averages).
    pub fn element_density(&self) -> Vec<f64> {
        element_density_of(&self.phi, self.grid)
    }

    /// Smooth Heaviside function. Returns values in \[1e-3, 1.0\].
    ///
    /// `H(x) = 1e-3 + (1 - 1e-3) (1 + x/ε + sin(π x/ε)/π) / 2` on
    /// `[-ε, ε]`, continuous with the constant `1e-3` below and `1` above.
    fn heaviside(x: f64, eps: f64) -> f64 {
        if x > eps {
            1.0
        } else if x < -eps {
            VOID_DENSITY
        } else {
            let smooth = 0.5 * (1.0 + x / eps + (x * PI / eps).sin() / PI);
            (VOID_DENSITY + (1.0 - VOID_DENSITY) * smooth).clamp(VOID_DENSITY, 1.0)
        }
    }

    /// Dirac delta approximation (derivative of Heaviside).
    fn dirac(x: f64, eps: f64) -> f64 {
        if x.abs() > eps {
            0.0
        } else {
            (1.0 - VOID_DENSITY) * (1.0 + (x * PI / eps).cos()) / (2.0 * eps)
        }
    }

    /// Computes mock compliance given a velocity field.
    pub fn mock_compliance(&self, velocity: &[f64]) -> f64 {
        velocity.iter().map(|v| v.powi(2)).sum::<f64>() / velocity.len() as f64
    }

    /// Performs one Hamilton-Jacobi update step and returns the time step used.
    ///
    /// Advances `∂φ/∂t + V_n |∇φ| = 0` with the first-order Godunov upwind
    /// scheme of Osher and Sethian on the unit-spaced node grid:
    ///
    /// `φ_new = φ - dt (max(V,0) ∇⁺ + min(V,0) ∇⁻)` with
    /// `∇⁺ = sqrt(max(D⁻x,0)² + min(D⁺x,0)² + max(D⁻y,0)² + min(D⁺y,0)²)` and
    /// `∇⁻` the same with `max` and `min` exchanged, where `D⁻` and `D⁺` are
    /// the backward and forward differences. They are one-sided at the
    /// boundary of the grid, where the missing difference is taken equal to
    /// the existing one (linear extrapolation of φ). `velocity` holds `V_n`
    /// per node; missing entries count as zero.
    ///
    /// The step is the field's `dt`, shortened to `1/2 / max|V_n|` whenever
    /// the CFL condition would otherwise be violated; the step actually used
    /// is returned. If `phi` does not have one value per node nothing is
    /// changed and `0` is returned.
    pub fn hj_update(&mut self, velocity: &[f64]) -> f64 {
        let nodes = (self.grid.nx + 1) * (self.grid.ny + 1);
        if self.phi.len() != nodes {
            return 0.0;
        }
        // Entries beyond the last node play no role, not even for the CFL limit.
        let velocity = &velocity[..velocity.len().min(nodes)];
        let dt = cfl_limited_dt(self.dt, velocity);
        self.phi = hj_step(&self.phi, velocity, self.grid, dt);
        dt
    }

    /// Re-initializes φ as a signed distance function.
    ///
    /// Runs `n_iter` explicit steps of the pseudo-time equation of Sussman,
    /// Smereka and Osher, `φ_τ + S(φ₀) (|∇φ| - 1) = 0` with the smoothed sign
    /// `S = φ₀ / sqrt(φ₀² + h²)` of the initial field `φ₀` and the grid
    /// spacing `h = 1`. `|∇φ|` is the upwind Godunov gradient, taken from the
    /// side the zero contour feeds, so information travels outward from the
    /// front by 0.3 spacings per step. Nodes next to the front barely move;
    /// the steps needed to restore `|∇φ| = 1` up to a distance `d` from the
    /// front are about `d / 0.3`.
    pub fn reinitialize(&mut self, n_iter: usize) {
        let nodes = (self.grid.nx + 1) * (self.grid.ny + 1);
        if self.phi.len() != nodes {
            return;
        }
        let sign: Vec<f64> = self.phi.iter().map(|&p| p / (p * p + 1.0).sqrt()).collect();
        for _ in 0..n_iter {
            self.phi = reinitialization_step(&self.phi, &sign, self.grid, REINIT_TIME_STEP);
        }
    }

    /// Runs mock level-set optimization.
    ///
    /// The "compliance" and the front speed are artificial, no structure is
    /// analysed; use [`LevelSetTopOpt::run`] for real designs.
    pub fn run_mock(&mut self, n_iter: usize) {
        self.compliance_history.clear();
        for _i in 0..n_iter {
            let density = self.element_density();
            let compliance: f64 = density
                .iter()
                .map(|&r| self.material.e_solid * r)
                .sum::<f64>()
                / density.len() as f64;
            self.compliance_history.push(compliance);

            // Mock velocity: negative sensitivity
            let nn = (self.grid.nx + 1) * (self.grid.ny + 1);
            let velocity: Vec<f64> = (0..nn)
                .map(|idx| {
                    let phi_val = self.phi[idx];
                    -Self::dirac(phi_val, 1.0)
                })
                .collect();
            self.hj_update(&velocity);
        }
    }

    /// Runs the level-set compliance minimisation with finite-element analysis.
    ///
    /// Every iteration the element densities `ρ_e` of the current level set
    /// give the ersatz moduli `E_e = E_void + ρ_e (E_solid - E_void)`; the
    /// plane-stress Q4 model of [`super::fe_q4`] is solved for the nodal force
    /// vector `loads` (length `2 (nx + 1) (ny + 1)`, see [`Q4Model::dof`]) with
    /// the degrees of freedom `fixed_dofs` held at zero. The shape derivative
    /// of the compliance moves the boundary outwards where the strain-energy
    /// density `w_e = E_e u_eᵀ K_e u_e` exceeds a Lagrange multiplier `λ`, so
    /// the speed of the zero contour into the structure (see the module
    /// documentation) is `V = λ - w` at the nodes, with `w` the average of the
    /// adjacent elements and normalised by its maximum. `λ` is found by
    /// bisection such that the updated level set meets the volume fraction and
    /// the level set is advanced by [`LevelSetTopOpt::hj_update`]. Every
    /// [`LevelSetTopOpt::reinit_interval`] iterations the level set is
    /// reinitialized before the analysis; the update of that iteration then
    /// restores the volume fraction, so the returned design is the result of a
    /// Hamilton-Jacobi step and meets the target up to the step limit.
    ///
    /// On success the level-set function is left in `self.phi` and the
    /// compliance history in `self.compliance_history`; on error both are
    /// restored. The method cannot create holes: new holes come only from the
    /// initial level set, so start from a hole pattern when the topology
    /// should change.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInput`](crate::error::PhysicsError::InvalidInput)
    /// for inconsistent sizes or unusable parameters (volume fraction outside
    /// `(0, 1)`, non-positive step, `E_solid <= E_void`) and for loads that do
    /// no work; everything [`Q4Model::analyse`] and
    /// [`Q4Analysis::solve`](super::fe_q4::Q4Analysis::solve) report.
    pub fn run(
        &mut self,
        loads: &[f64],
        fixed_dofs: &[usize],
        n_iter: usize,
    ) -> PhysicsResult<LevelSetResult> {
        self.validate_run_setup()?;
        let saved_phi = self.phi.clone();
        let outcome = self.run_iterations(loads, fixed_dofs, n_iter);
        match outcome {
            Ok((result, history)) => {
                self.compliance_history = history;
                Ok(result)
            }
            Err(error) => {
                self.phi = saved_phi;
                Err(error)
            }
        }
    }

    /// The optimisation loop of [`LevelSetTopOpt::run`]; returns the result
    /// and the compliance history, and leaves a partially updated φ on error.
    fn run_iterations(
        &mut self,
        loads: &[f64],
        fixed_dofs: &[usize],
        n_iter: usize,
    ) -> PhysicsResult<(LevelSetResult, Vec<f64>)> {
        let model = Q4Model::new(self.grid, self.material.nu)?;
        let mut compliance_history = Vec::with_capacity(n_iter);
        let mut volume_history = Vec::with_capacity(n_iter);
        let mut displacement: Option<Vec<f64>> = None;

        for iteration in 0..n_iter {
            // Reinitialization shifts the contour by a fraction of an element,
            // which changes the volume; doing it before the analysis lets the
            // multiplier search below restore the volume fraction.
            if self.reinit_interval > 0
                && iteration.is_multiple_of(self.reinit_interval)
                && iteration > 0
            {
                self.reinitialize(self.reinit_steps);
            }
            let modulus = self.ersatz_modulus();
            let analysis = model.analyse(&modulus, fixed_dofs, self.linear_solver)?;
            let solution = analysis.solve_from(loads, displacement.as_deref())?;
            if !solution.compliance.is_finite() || solution.compliance <= 0.0 {
                return Err(invalid_input(
                    "loads",
                    "the loads do no work on the supported structure (zero compliance)",
                ));
            }
            compliance_history.push(solution.compliance);

            let energy_density: Vec<f64> = modulus
                .iter()
                .zip(solution.element_energy.iter())
                .map(|(young, energy)| young * energy)
                .collect();
            let mut nodal_energy = element_field_to_nodes(&energy_density, self.grid);
            let scale = nodal_energy.iter().copied().fold(0.0f64, f64::max);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(invalid_input(
                    "loads",
                    "the strain-energy density vanishes everywhere, nothing to optimise",
                ));
            }
            for energy in &mut nodal_energy {
                *energy /= scale;
            }
            let multiplier = self.volume_multiplier(&nodal_energy);
            let velocity: Vec<f64> = nodal_energy.iter().map(|w| multiplier - w).collect();
            self.hj_update(&velocity);
            displacement = Some(solution.displacement);
            volume_history.push(volume_fraction(&self.element_density()));
        }

        let density = self.element_density();
        let modulus = self.ersatz_modulus();
        let analysis = model.analyse(&modulus, fixed_dofs, self.linear_solver)?;
        let final_solution = analysis.solve_from(loads, displacement.as_deref())?;
        let result = LevelSetResult {
            volume_fraction: volume_fraction(&density),
            density,
            compliance: final_solution.compliance,
            compliance_history: compliance_history.clone(),
            volume_history,
            iterations: n_iter,
            displacement: final_solution.displacement,
        };
        Ok((result, compliance_history))
    }

    /// Element moduli `E_void + ρ_e (E_solid - E_void)` of the current φ.
    fn ersatz_modulus(&self) -> Vec<f64> {
        self.element_density()
            .iter()
            .map(|&rho| self.material.e_void + rho * (self.material.e_solid - self.material.e_void))
            .collect()
    }

    /// The Lagrange multiplier `λ` for which one Hamilton-Jacobi step with the
    /// speed `λ - w` leaves the volume fraction at the target.
    ///
    /// `scaled_energy` is the strain-energy density at the nodes, normalised
    /// to `[0, 1]`. A negative `λ` makes the whole contour move outwards and a
    /// `λ` above one makes it move inwards, so the bracket `[-2, 3]` spans from
    /// nearly uniform growth to nearly uniform shrinkage (the step is limited
    /// by the CFL condition either way). The updated volume decreases
    /// monotonically with `λ` because the upwind update is monotone in the
    /// speed; if even the extreme multipliers cannot reach the target the
    /// nearer one is returned.
    fn volume_multiplier(&self, scaled_energy: &[f64]) -> f64 {
        let target = self.volume_fraction;
        let volume_after = |multiplier: f64| -> f64 {
            let velocity: Vec<f64> = scaled_energy.iter().map(|w| multiplier - w).collect();
            let dt = cfl_limited_dt(self.dt, &velocity);
            let phi = hj_step(&self.phi, &velocity, self.grid, dt);
            volume_fraction(&element_density_of(&phi, self.grid))
        };
        let (mut low, mut high) = (-2.0f64, 3.0f64);
        if volume_after(low) <= target {
            return low;
        }
        if volume_after(high) >= target {
            return high;
        }
        for _ in 0..60 {
            let middle = 0.5 * (low + high);
            if volume_after(middle) > target {
                low = middle;
            } else {
                high = middle;
            }
        }
        0.5 * (low + high)
    }

    /// Checks the parameters [`LevelSetTopOpt::run`] depends on.
    fn validate_run_setup(&self) -> PhysicsResult<()> {
        let nodes = (self.grid.nx + 1) * (self.grid.ny + 1);
        if self.phi.len() != nodes || self.phi.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input(
                "phi",
                format!("expected {nodes} finite values, one per grid node"),
            ));
        }
        if !(self.volume_fraction > 0.0 && self.volume_fraction < 1.0) {
            return Err(invalid_input(
                "volume_fraction",
                format!("must lie in (0, 1), got {}", self.volume_fraction),
            ));
        }
        if !(self.dt > 0.0 && self.dt.is_finite()) {
            return Err(invalid_input(
                "dt",
                format!("must be positive and finite, got {}", self.dt),
            ));
        }
        let material = self.material;
        if !(material.e_void >= 0.0
            && material.e_solid.is_finite()
            && material.e_solid > material.e_void)
        {
            return Err(invalid_input(
                "material",
                format!(
                    "needs 0 <= e_void < e_solid, got e_void {} and e_solid {}",
                    material.e_void, material.e_solid
                ),
            ));
        }
        Ok(())
    }
}

/// Element densities of the level set `phi` on `grid`: the smoothed
/// Heaviside of the average of the four corner values of each element.
fn element_density_of(phi: &[f64], grid: GridSize) -> Vec<f64> {
    let nx = grid.nx;
    let stride = nx + 1;
    (0..grid.n_elem())
        .map(|e| {
            let row = e / nx;
            let col = e % nx;
            // Four corner nodes of this element
            let n0 = row * stride + col;
            let n1 = n0 + 1;
            let n2 = n0 + stride;
            let n3 = n2 + 1;
            let avg = 0.25 * (phi[n0] + phi[n1] + phi[n2] + phi[n3]);
            LevelSetTopOpt::heaviside(avg, HEAVISIDE_HALF_WIDTH)
        })
        .collect()
}

/// Average of an element field over the elements adjacent to every node.
fn element_field_to_nodes(field: &[f64], grid: GridSize) -> Vec<f64> {
    let nx = grid.nx;
    let stride = nx + 1;
    let nodes = stride * (grid.ny + 1);
    let mut sum = vec![0.0f64; nodes];
    let mut count = vec![0u32; nodes];
    for (e, &value) in field.iter().enumerate() {
        let n0 = (e / nx) * stride + e % nx;
        for node in [n0, n0 + 1, n0 + stride, n0 + stride + 1] {
            sum[node] += value;
            count[node] += 1;
        }
    }
    sum.iter()
        .zip(count.iter())
        .map(|(&total, &adjacent)| total / f64::from(adjacent.max(1)))
        .collect()
}

/// The time step of the explicit upwind scheme for the speeds `velocity`: the
/// requested `dt`, or the CFL limit `HJ_CFL / max|V|` if that is shorter.
fn cfl_limited_dt(dt: f64, velocity: &[f64]) -> f64 {
    let max_speed = velocity.iter().fold(0.0f64, |peak, v| peak.max(v.abs()));
    if max_speed * dt > HJ_CFL {
        HJ_CFL / max_speed
    } else {
        dt
    }
}

/// One-sided differences `[D⁻x, D⁺x, D⁻y, D⁺y]` of `phi` at node
/// `(row, col)` for unit grid spacing. A difference that would leave the grid
/// is replaced by the opposite one (linear extrapolation); a grid line with a
/// single node has zero differences.
fn one_sided_differences(phi: &[f64], grid: GridSize, row: usize, col: usize) -> [f64; 4] {
    let (nx, ny) = (grid.nx, grid.ny);
    let stride = nx + 1;
    let idx = row * stride + col;
    let backward = |has_previous: bool, previous: usize| {
        if has_previous {
            Some(phi[idx] - phi[previous])
        } else {
            None
        }
    };
    let forward = |has_next: bool, next: usize| {
        if has_next {
            Some(phi[next] - phi[idx])
        } else {
            None
        }
    };
    let backward_x = backward(col > 0, idx.wrapping_sub(1));
    let forward_x = forward(col < nx, idx + 1);
    let backward_y = backward(row > 0, idx.wrapping_sub(stride));
    let forward_y = forward(row < ny, idx + stride);
    [
        backward_x.or(forward_x).unwrap_or(0.0),
        forward_x.or(backward_x).unwrap_or(0.0),
        backward_y.or(forward_y).unwrap_or(0.0),
        forward_y.or(backward_y).unwrap_or(0.0),
    ]
}

/// One explicit Godunov upwind step of `φ_t + V |∇φ| = 0`.
fn hj_step(phi: &[f64], velocity: &[f64], grid: GridSize, dt: f64) -> Vec<f64> {
    let stride = grid.nx + 1;
    let mut next = phi.to_vec();
    for row in 0..=grid.ny {
        for col in 0..=grid.nx {
            let idx = row * stride + col;
            let speed = velocity.get(idx).copied().unwrap_or(0.0);
            if speed == 0.0 {
                continue;
            }
            let [dxm, dxp, dym, dyp] = one_sided_differences(phi, grid, row, col);
            let grad_plus = (dxm.max(0.0).powi(2)
                + dxp.min(0.0).powi(2)
                + dym.max(0.0).powi(2)
                + dyp.min(0.0).powi(2))
            .sqrt();
            let grad_minus = (dxp.max(0.0).powi(2)
                + dxm.min(0.0).powi(2)
                + dyp.max(0.0).powi(2)
                + dym.min(0.0).powi(2))
            .sqrt();
            next[idx] = phi[idx] - dt * (speed.max(0.0) * grad_plus + speed.min(0.0) * grad_minus);
        }
    }
    next
}

/// One explicit step of the reinitialization equation
/// `φ_τ = -S (|∇φ|_Godunov - 1)` with the smoothed sign `sign`.
fn reinitialization_step(phi: &[f64], sign: &[f64], grid: GridSize, dtau: f64) -> Vec<f64> {
    let stride = grid.nx + 1;
    let mut next = phi.to_vec();
    for row in 0..=grid.ny {
        for col in 0..=grid.nx {
            let idx = row * stride + col;
            let [dxm, dxp, dym, dyp] = one_sided_differences(phi, grid, row, col);
            let s = sign[idx];
            let gradient = if s > 0.0 {
                (dxm.max(0.0).powi(2).max(dxp.min(0.0).powi(2))
                    + dym.max(0.0).powi(2).max(dyp.min(0.0).powi(2)))
                .sqrt()
            } else {
                (dxm.min(0.0).powi(2).max(dxp.max(0.0).powi(2))
                    + dym.min(0.0).powi(2).max(dyp.max(0.0).powi(2)))
                .sqrt()
            };
            next[idx] = phi[idx] - dtau * s * (gradient - 1.0);
        }
    }
    next
}

#[cfg(test)]
mod tests;
