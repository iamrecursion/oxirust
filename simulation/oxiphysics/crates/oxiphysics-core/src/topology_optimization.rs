// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Topology optimization methods including SIMP, level-set, and manufacturing filters.
//!
//! This module provides solid-topology optimization solvers for structural
//! compliance minimization under volume constraints, with density filtering,
//! sensitivity filtering, and Heaviside projection.
//!
//! The structural response comes from the plane-stress Q4 finite-element
//! solver of the [`fe_q4`] submodule. [`TopOptSolver::run`] (SIMP with
//! optimality criteria), [`LevelSetTopOpt::run`] and
//! [`MultiLoadTopOpt::run_fe`] analyse the design with it in every iteration,
//! while [`MultiLoadTopOpt::run`] takes the response from a caller-supplied
//! closure for models the solver does not cover. [`TopOptSolver::run_mock`]
//! and [`LevelSetTopOpt::run_mock`] drive the optimisers with an artificial
//! response that involves no finite-element analysis; they exist for testing
//! only.

use crate::error::{PhysicsError, PhysicsResult};

pub mod fe_q4;
mod level_set;
mod manufacturing;

pub use level_set::{LevelSetResult, LevelSetTopOpt};
pub use manufacturing::ManufacturingFilters;

use fe_q4::{LinearSolver, Q4Model};

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// 2D grid size descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSize {
    /// Number of elements in the x direction.
    pub nx: usize,
    /// Number of elements in the y direction.
    pub ny: usize,
}

impl GridSize {
    /// Creates a new `GridSize`.
    pub fn new(nx: usize, ny: usize) -> Self {
        Self { nx, ny }
    }

    /// Returns total number of elements.
    pub fn n_elem(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns (row, col) for a flat element index.
    pub fn row_col(&self, idx: usize) -> (usize, usize) {
        (idx / self.nx, idx % self.nx)
    }
}

/// Material parameters for structural problems.
#[derive(Debug, Clone, Copy)]
pub struct MaterialParams {
    /// Young's modulus of the solid material.
    pub e_solid: f64,
    /// Young's modulus of the void material (usually small positive).
    pub e_void: f64,
    /// Poisson's ratio.
    pub nu: f64,
}

impl MaterialParams {
    /// Creates new material parameters.
    pub fn new(e_solid: f64, e_void: f64, nu: f64) -> Self {
        Self {
            e_solid,
            e_void,
            nu,
        }
    }
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            e_solid: 1.0,
            e_void: 1e-9,
            nu: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// SIMP — Solid Isotropic Material with Penalization
// ---------------------------------------------------------------------------

/// Solid Isotropic Material with Penalization (SIMP) model.
///
/// SIMP interpolates material stiffness as `E(ρ) = E_void + ρ^p * (E_solid - E_void)`.
/// The penalization parameter `p` discourages intermediate densities.
#[derive(Debug, Clone)]
pub struct SiMP {
    /// Penalization exponent.
    pub penalty: f64,
    /// Material parameters.
    pub material: MaterialParams,
    /// Volume fraction target.
    pub volume_fraction: f64,
    /// Filter radius for density filtering.
    pub filter_radius: f64,
    /// Grid size.
    pub grid: GridSize,
    /// Current density field (length = grid.n_elem()).
    pub density: Vec<f64>,
}

impl SiMP {
    /// Creates a new SIMP model with uniform initial density equal to volume_fraction.
    pub fn new(
        grid: GridSize,
        volume_fraction: f64,
        penalty: f64,
        filter_radius: f64,
        material: MaterialParams,
    ) -> Self {
        let n = grid.n_elem();
        let density = vec![volume_fraction; n];
        Self {
            penalty,
            material,
            volume_fraction,
            filter_radius,
            grid,
            density,
        }
    }

    /// Computes the interpolated Young's modulus for a given density.
    ///
    /// `E(ρ) = E_void + ρ^p * (E_solid - E_void)`
    pub fn interpolated_stiffness(&self, rho: f64) -> f64 {
        let rho_clamped = rho.clamp(0.0, 1.0);
        self.material.e_void
            + rho_clamped.powf(self.penalty) * (self.material.e_solid - self.material.e_void)
    }

    /// Computes the derivative of interpolated stiffness with respect to density.
    ///
    /// `dE/dρ = p * ρ^(p-1) * (E_solid - E_void)`
    pub fn stiffness_sensitivity(&self, rho: f64) -> f64 {
        let rho_clamped = rho.clamp(1e-12, 1.0);
        self.penalty
            * rho_clamped.powf(self.penalty - 1.0)
            * (self.material.e_solid - self.material.e_void)
    }

    /// Returns vector of interpolated stiffnesses for current density field.
    pub fn stiffness_field(&self) -> Vec<f64> {
        self.density
            .iter()
            .map(|&r| self.interpolated_stiffness(r))
            .collect()
    }

    /// Returns sensitivities `dE/dρ` for current density field.
    pub fn sensitivity_field(&self) -> Vec<f64> {
        self.density
            .iter()
            .map(|&r| self.stiffness_sensitivity(r))
            .collect()
    }

    /// Computes element-wise compliance sensitivity given element displacements.
    ///
    /// `dc/dρ_e = -p * ρ_e^(p-1) * (E_solid - E_void) * u_e^T * K_e_unit * u_e`
    pub fn compliance_sensitivity(&self, element_strain_energy: &[f64]) -> Vec<f64> {
        self.compliance_sensitivity_at(&self.density, element_strain_energy)
    }

    /// Computes element-wise compliance sensitivity at an arbitrary density field.
    ///
    /// Same formula as [`SiMP::compliance_sensitivity`], evaluated at `density`
    /// instead of the stored field: `element_strain_energy[e]` is
    /// `u_e^T K_e_unit u_e` for the displacements obtained with `density`.
    pub fn compliance_sensitivity_at(
        &self,
        density: &[f64],
        element_strain_energy: &[f64],
    ) -> Vec<f64> {
        density
            .iter()
            .zip(element_strain_energy.iter())
            .map(|(&rho, &se)| {
                let rho_c = rho.clamp(1e-12, 1.0);
                -self.penalty
                    * rho_c.powf(self.penalty - 1.0)
                    * (self.material.e_solid - self.material.e_void)
                    * se
            })
            .collect()
    }

    /// Computes current volume fraction.
    pub fn current_volume(&self) -> f64 {
        let sum: f64 = self.density.iter().sum();
        sum / self.density.len() as f64
    }

    /// Returns whether the density field is valid (all values in \[0,1\]).
    pub fn is_valid(&self) -> bool {
        self.density.iter().all(|&r| (0.0..=1.0).contains(&r))
    }
}

// ---------------------------------------------------------------------------
// FilteringMethods
// ---------------------------------------------------------------------------

/// Lower bound `γ` on the element density in the denominator of the
/// sensitivity filter (Sigmund's `max(γ, ρ_e)`; `top88.m` uses `1e-3`).
const SENSITIVITY_FILTER_GAMMA: f64 = 1e-3;

/// Provides density filtering and sensitivity filtering for topology optimization.
///
/// Filtering suppresses checkerboard patterns and mesh dependence in SIMP.
///
/// The filter weights are the linear "hat" weights `H_ef = max(0, R - d_ef)`
/// of the element centres at distance `d_ef` (measured in element widths)
/// within the radius `R`. They are precomputed once, in a compressed sparse
/// row layout, in `O(n R²)` time and memory for a grid of `n` elements.
#[derive(Debug, Clone)]
pub struct FilteringMethods {
    /// Filter radius.
    pub radius: f64,
    /// Grid size.
    pub grid: GridSize,
    /// Start of the neighbour list of every element in `neighbor_entries`
    /// (length `n_elem + 1`).
    neighbor_offsets: Vec<usize>,
    /// `(element, weight)` pairs of all neighbour lists, each list in
    /// ascending element order.
    neighbor_entries: Vec<(usize, f64)>,
}

impl FilteringMethods {
    /// Creates a new `FilteringMethods` and precomputes neighbor lists.
    pub fn new(radius: f64, grid: GridSize) -> Self {
        let (neighbor_offsets, neighbor_entries) = Self::compute_neighbors(radius, grid);
        Self {
            radius,
            grid,
            neighbor_offsets,
            neighbor_entries,
        }
    }

    /// Builds the neighbour lists of every element of `grid` for `radius`.
    ///
    /// Only the `(2⌈R⌉ - 1)²` window of elements that can lie closer than the
    /// radius is visited around each element, so the cost is `O(n R²)`
    /// instead of the `O(n²)` of a scan over all element pairs. Element `j`
    /// is a neighbour of `i` when its centre distance `d` is strictly below
    /// the radius, with weight `radius - d`; the lists are in ascending `j`.
    fn compute_neighbors(radius: f64, grid: GridSize) -> (Vec<usize>, Vec<(usize, f64)>) {
        let (nx, ny) = (grid.nx, grid.ny);
        let n = grid.n_elem();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut entries: Vec<(usize, f64)> = Vec::new();
        offsets.push(0);
        // A neighbour has |dr|, |dc| < radius, i.e. at most ceil(radius) - 1
        // cells away. The clamp keeps an unbounded radius from overflowing
        // and a NaN radius saturates to zero (and then matches nothing).
        let extent = nx.max(ny);
        let reach = (radius.ceil() - 1.0).clamp(0.0, extent as f64) as usize;
        for row in 0..ny {
            let rows = row.saturating_sub(reach)..=(row + reach).min(ny.saturating_sub(1));
            for col in 0..nx {
                let cols = col.saturating_sub(reach)..=(col + reach).min(nx.saturating_sub(1));
                for other_row in rows.clone() {
                    let dr = other_row.abs_diff(row);
                    for other_col in cols.clone() {
                        let dc = other_col.abs_diff(col);
                        let dist = ((dr * dr + dc * dc) as f64).sqrt();
                        if dist < radius {
                            entries.push((other_row * nx + other_col, radius - dist));
                        }
                    }
                }
                offsets.push(entries.len());
            }
        }
        (offsets, entries)
    }

    /// Returns the `(element, weight)` neighbours of `element` — itself and
    /// every element whose centre lies within the filter radius — in
    /// ascending element order. Out-of-range elements have no neighbours.
    pub fn neighbors(&self, element: usize) -> &[(usize, f64)] {
        let bounds = element.checked_add(1).and_then(|next| {
            Some((
                *self.neighbor_offsets.get(element)?,
                *self.neighbor_offsets.get(next)?,
            ))
        });
        match bounds {
            Some((start, end)) => self.neighbor_entries.get(start..end).unwrap_or(&[]),
            None => &[],
        }
    }

    /// Applies the sensitivity filter of Sigmund to a sensitivity field.
    ///
    /// `dc_filtered_e = Σ_f H_ef ρ_f dc_f / (max(γ, ρ_e) Σ_f H_ef)` with
    /// `γ = 1e-3` (the `ft = 1` filter of `top88.m`). The `ρ_f` weighting
    /// makes the filter the sensitivity of the density-filtered problem; it
    /// amplifies the sensitivity of low-density elements by `1 / ρ_e`
    /// instead of averaging it away.
    ///
    /// An element without neighbours (a radius of at most zero, or NaN) keeps
    /// its raw sensitivity, and if `density` or `sensitivity` does not have
    /// one entry per grid element the sensitivities are returned unchanged.
    pub fn sensitivity_filter(&self, density: &[f64], sensitivity: &[f64]) -> Vec<f64> {
        let n = self.grid.n_elem();
        if density.len() != n || sensitivity.len() != n {
            return sensitivity.to_vec();
        }
        (0..n)
            .map(|e| {
                let mut weighted = 0.0f64;
                let mut weight_sum = 0.0f64;
                for &(f, h) in self.neighbors(e) {
                    weighted += h * density[f] * sensitivity[f];
                    weight_sum += h;
                }
                if weight_sum > 0.0 {
                    weighted / (density[e].max(SENSITIVITY_FILTER_GAMMA) * weight_sum)
                } else {
                    sensitivity[e]
                }
            })
            .collect()
    }

    /// Applies density filter (linear weighted average).
    ///
    /// `ρ_filtered_e = Σ H_f * ρ_f / Σ H_f`
    pub fn density_filter(&self, density: &[f64]) -> Vec<f64> {
        let n = self.grid.n_elem();
        let mut filtered = vec![0.0; n];
        for e in 0..n {
            let mut num = 0.0f64;
            let mut denom = 0.0f64;
            for &(f, h) in self.neighbors(e) {
                num += h * density[f];
                denom += h;
            }
            filtered[e] = if denom > 0.0 { num / denom } else { density[e] };
        }
        filtered
    }

    /// Applies Heaviside projection filter.
    ///
    /// `ρ̄_e = (tanh(β*η) + tanh(β*(ρ_e - η))) / (tanh(β*η) + tanh(β*(1-η)))`
    pub fn heaviside_projection(&self, density: &[f64], beta: f64, eta: f64) -> Vec<f64> {
        let denom = (beta * eta).tanh() + (beta * (1.0 - eta)).tanh();
        density
            .iter()
            .map(|&rho| ((beta * eta).tanh() + (beta * (rho - eta)).tanh()) / denom.max(1e-12))
            .collect()
    }

    /// Computes derivative of Heaviside projection with respect to filtered density.
    pub fn heaviside_sensitivity(&self, density: &[f64], beta: f64, eta: f64) -> Vec<f64> {
        let denom = (beta * eta).tanh() + (beta * (1.0 - eta)).tanh();
        density
            .iter()
            .map(|&rho| beta * (1.0 - (beta * (rho - eta)).tanh().powi(2)) / denom.max(1e-12))
            .collect()
    }

    /// Propagates sensitivity through density filter (chain rule).
    pub fn filter_sensitivity_chain(&self, sensitivity: &[f64], density: &[f64]) -> Vec<f64> {
        // Adjoint of density filter
        let n = self.grid.n_elem();
        let mut out = vec![0.0; n];
        for (e, &element_sensitivity) in sensitivity.iter().enumerate().take(n) {
            let nw = self.neighbors(e);
            let denom: f64 = nw.iter().map(|&(_, h)| h).sum::<f64>().max(1e-12);
            for &(f, h) in nw {
                out[f] += element_sensitivity * h / denom;
            }
        }
        let _ = density; // density may be used for weighted variants
        out
    }
}

// ---------------------------------------------------------------------------
// TopOptSolver — optimality criteria
// ---------------------------------------------------------------------------

/// Topology optimization solver using the Optimality Criteria (OC) method.
///
/// The OC update rule adjusts densities to satisfy KKT conditions for
/// compliance minimization subject to a volume constraint.
/// [`TopOptSolver::run`] drives it with the plane-stress finite-element
/// analysis of [`fe_q4`].
#[derive(Debug, Clone)]
pub struct TopOptSolver {
    /// SIMP model.
    pub simp: SiMP,
    /// Filter.
    pub filter: FilteringMethods,
    /// Move limit for density update.
    pub move_limit: f64,
    /// Damping exponent for OC update.
    pub damping: f64,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Current compliance history.
    pub compliance_history: Vec<f64>,
    /// Linear solver of the finite-element analysis in [`TopOptSolver::run`].
    pub linear_solver: LinearSolver,
}

/// Outcome of [`TopOptSolver::run`].
#[derive(Debug, Clone, PartialEq)]
pub struct TopOptResult {
    /// Optimised density field, one entry per element in row-major order.
    pub density: Vec<f64>,
    /// Compliance `fᵀu` of [`TopOptResult::density`].
    pub compliance: f64,
    /// Compliance at the start of every iteration, that is of the design the
    /// iteration then updated; the first entry belongs to the initial design.
    pub compliance_history: Vec<f64>,
    /// Number of optimality-criteria iterations performed.
    pub iterations: usize,
    /// Whether the largest density change of the last iteration dropped
    /// below the solver tolerance (as opposed to stopping at `max_iter`).
    pub converged: bool,
    /// Largest density change of the last iteration; `0` if none ran.
    pub max_density_change: f64,
    /// Mean density of [`TopOptResult::density`].
    pub volume_fraction: f64,
    /// Nodal displacements of the optimised design under the loads.
    pub displacement: Vec<f64>,
}

impl TopOptSolver {
    /// Creates a new `TopOptSolver`.
    pub fn new(simp: SiMP, filter: FilteringMethods) -> Self {
        Self {
            simp,
            filter,
            move_limit: 0.2,
            damping: 0.5,
            tolerance: 1e-4,
            max_iter: 200,
            compliance_history: Vec::new(),
            linear_solver: LinearSolver::Auto,
        }
    }

    /// Sets the linear solver of the finite-element analysis.
    pub fn with_linear_solver(mut self, solver: LinearSolver) -> Self {
        self.linear_solver = solver;
        self
    }

    /// Sets the move limit.
    pub fn with_move_limit(mut self, m: f64) -> Self {
        self.move_limit = m;
        self
    }

    /// Sets the damping parameter.
    pub fn with_damping(mut self, d: f64) -> Self {
        self.damping = d;
        self
    }

    /// Sets convergence tolerance.
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Sets maximum iterations.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Performs one OC density update step.
    ///
    /// Returns the new density vector and the Lagrange multiplier used. The
    /// multiplier is found by bisection on a logarithmic scale, so the update
    /// meets `volume_target` whatever the magnitude of the sensitivities.
    pub fn oc_update(
        &self,
        density: &[f64],
        sensitivity: &[f64],
        volume_target: f64,
    ) -> (Vec<f64>, f64) {
        oc_density_update(
            density,
            sensitivity,
            volume_target,
            self.move_limit,
            self.damping,
        )
    }

    /// Computes compliance given stiffness field and element strain energies.
    ///
    /// Compliance = Σ E(ρ_e) * se_e (sum of strain energy contributions).
    pub fn compute_compliance(&self, strain_energy: &[f64]) -> f64 {
        strain_energy.iter().sum()
    }

    /// Runs the topology optimization loop with a mock FEA (diagonal stiffness).
    ///
    /// For testing purposes the strain energy of element `e` is approximated
    /// as `E(ρ_e) * load^2 / n` where `load` is a uniform load parameter.
    /// No structure is analysed; use [`TopOptSolver::run`] for real designs.
    pub fn run_mock(&mut self, load: f64) -> Vec<f64> {
        let n = self.simp.grid.n_elem();
        let vf = self.simp.volume_fraction;
        let mut density = self.simp.density.clone();
        self.compliance_history.clear();

        for _iter in 0..self.max_iter {
            // Mock FEA: strain energy proportional to stiffness
            let se: Vec<f64> = density
                .iter()
                .map(|&rho| self.simp.interpolated_stiffness(rho) * load * load / n as f64)
                .collect();
            let compliance = self.compute_compliance(&se);
            self.compliance_history.push(compliance);

            // Sensitivity
            let raw_sens = self.simp.compliance_sensitivity(&se);
            // Filter sensitivity
            let filtered_sens = self.filter.sensitivity_filter(&density, &raw_sens);
            // OC update
            let (new_density, _lm) = self.oc_update(&density, &filtered_sens, vf);

            // Check convergence
            let change: f64 = density
                .iter()
                .zip(new_density.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            density = new_density;
            if change < self.tolerance {
                break;
            }
        }

        self.simp.density = density.clone();
        density
    }

    /// Runs the SIMP compliance minimisation with finite-element analysis.
    ///
    /// Every iteration assembles the plane-stress Q4 stiffness matrix of
    /// [`fe_q4`] from the SIMP moduli `E(ρ_e)`, solves `K u = f` for the nodal
    /// force vector `loads` (length `2 (nx + 1) (ny + 1)`, see
    /// [`Q4Model::dof`] for the numbering) with the degrees of freedom
    /// `fixed_dofs` held at zero, and computes the compliance sensitivity
    /// `dc_e = -p ρ_e^(p-1) (E_solid - E_void) u_eᵀ K_e u_e`. The sensitivities
    /// pass through the Sigmund sensitivity filter and an optimality-criteria
    /// update that meets the volume fraction of the SIMP model. This is the
    /// algorithm of the `top88.m` code with its `ft = 1` filter.
    ///
    /// The loop stops after `max_iter` iterations or as soon as the largest
    /// density change drops below `tolerance`. On success the final density
    /// is stored in `self.simp.density` and the per-iteration compliance in
    /// `self.compliance_history`; on error both are left unchanged.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInput`] for inconsistent sizes or unusable
    /// parameters (volume fraction outside `(0, 1]`, penalty below one,
    /// non-positive move limit or damping, `E_solid <= E_void`) and for loads
    /// that do no work on the supported structure; everything
    /// [`Q4Model::analyse`] and [`fe_q4::Q4Analysis::solve`] report, such as an
    /// unsupported structure or a failed linear solve.
    ///
    /// # Examples
    ///
    /// A cantilever clamped along its left edge and loaded downwards at the
    /// middle of the free end:
    ///
    /// ```
    /// use oxiphysics_core::topology_optimization::fe_q4::Q4Model;
    /// use oxiphysics_core::topology_optimization::{
    ///     FilteringMethods, GridSize, MaterialParams, SiMP, TopOptSolver,
    /// };
    ///
    /// let grid = GridSize::new(16, 8);
    /// let simp = SiMP::new(grid, 0.4, 3.0, 1.5, MaterialParams::default());
    /// let mut solver =
    ///     TopOptSolver::new(simp, FilteringMethods::new(1.5, grid)).with_max_iter(30);
    ///
    /// let model = Q4Model::new(grid, 0.3)?;
    /// let mut loads = vec![0.0; model.n_dofs()];
    /// loads[model.dof(model.node_index(4, 16), 1)] = -1.0;
    /// let clamped: Vec<usize> = (0..=8)
    ///     .flat_map(|row| {
    ///         let node = model.node_index(row, 0);
    ///         [model.dof(node, 0), model.dof(node, 1)]
    ///     })
    ///     .collect();
    ///
    /// let result = solver.run(&loads, &clamped)?;
    /// assert!((result.volume_fraction - 0.4).abs() < 1e-9);
    /// assert!(result.compliance < result.compliance_history[0]);
    /// # Ok::<(), oxiphysics_core::PhysicsError>(())
    /// ```
    pub fn run(&mut self, loads: &[f64], fixed_dofs: &[usize]) -> PhysicsResult<TopOptResult> {
        self.validate_run_setup()?;
        let model = Q4Model::new(self.simp.grid, self.simp.material.nu)?;
        let volume_target = self.simp.volume_fraction;
        let mut density = self.simp.density.clone();
        let mut history = Vec::new();
        let mut displacement: Option<Vec<f64>> = None;
        let mut converged = false;
        let mut max_change = 0.0f64;

        for _ in 0..self.max_iter {
            let analysis = self.analyse_design(&model, &density, fixed_dofs)?;
            let solution = analysis.solve_from(loads, displacement.as_deref())?;
            if !solution.compliance.is_finite() || solution.compliance <= 0.0 {
                return Err(invalid_input(
                    "loads",
                    "the loads do no work on the supported structure (zero compliance)",
                ));
            }
            history.push(solution.compliance);
            let raw_sensitivity = self
                .simp
                .compliance_sensitivity_at(&density, &solution.element_energy);
            let sensitivity = self.filter.sensitivity_filter(&density, &raw_sensitivity);
            let (next_density, _multiplier) = self.oc_update(&density, &sensitivity, volume_target);
            max_change = density
                .iter()
                .zip(next_density.iter())
                .map(|(before, after)| (before - after).abs())
                .fold(0.0f64, f64::max);
            density = next_density;
            displacement = Some(solution.displacement);
            if max_change < self.tolerance {
                converged = true;
                break;
            }
        }

        let analysis = self.analyse_design(&model, &density, fixed_dofs)?;
        let final_solution = analysis.solve_from(loads, displacement.as_deref())?;
        let volume = volume_fraction(&density);
        let iterations = history.len();
        self.simp.density = density.clone();
        self.compliance_history = history.clone();
        Ok(TopOptResult {
            density,
            compliance: final_solution.compliance,
            compliance_history: history,
            iterations,
            converged,
            max_density_change: max_change,
            volume_fraction: volume,
            displacement: final_solution.displacement,
        })
    }

    /// Assembles the finite-element system of the design `density`.
    fn analyse_design<'m>(
        &self,
        model: &'m Q4Model,
        density: &[f64],
        fixed_dofs: &[usize],
    ) -> PhysicsResult<fe_q4::Q4Analysis<'m>> {
        let modulus: Vec<f64> = density
            .iter()
            .map(|&rho| self.simp.interpolated_stiffness(rho))
            .collect();
        model.analyse(&modulus, fixed_dofs, self.linear_solver)
    }

    /// Checks the parameters [`TopOptSolver::run`] depends on.
    fn validate_run_setup(&self) -> PhysicsResult<()> {
        let n = self.simp.grid.n_elem();
        if self.simp.density.len() != n || self.simp.density.iter().any(|rho| !rho.is_finite()) {
            return Err(invalid_input(
                "simp.density",
                format!("expected {n} finite entries, one per element"),
            ));
        }
        if self.filter.grid != self.simp.grid {
            return Err(invalid_input(
                "filter",
                format!(
                    "filter grid {}x{} differs from the SIMP grid {}x{}",
                    self.filter.grid.nx, self.filter.grid.ny, self.simp.grid.nx, self.simp.grid.ny
                ),
            ));
        }
        let volume_target = self.simp.volume_fraction;
        if !(volume_target > 0.0 && volume_target <= 1.0) {
            return Err(invalid_input(
                "simp.volume_fraction",
                format!("must lie in (0, 1], got {volume_target}"),
            ));
        }
        let penalty = self.simp.penalty;
        if !(penalty >= 1.0 && penalty.is_finite()) {
            return Err(invalid_input(
                "simp.penalty",
                format!("must be finite and at least 1, got {penalty}"),
            ));
        }
        let material = self.simp.material;
        if !(material.e_void >= 0.0
            && material.e_solid.is_finite()
            && material.e_solid > material.e_void)
        {
            return Err(invalid_input(
                "simp.material",
                format!(
                    "needs 0 <= e_void < e_solid, got e_void {} and e_solid {}",
                    material.e_void, material.e_solid
                ),
            ));
        }
        if !(self.move_limit > 0.0 && self.move_limit.is_finite()) {
            return Err(invalid_input(
                "move_limit",
                format!("must be positive and finite, got {}", self.move_limit),
            ));
        }
        if !(self.damping > 0.0 && self.damping.is_finite()) {
            return Err(invalid_input(
                "damping",
                format!("must be positive and finite, got {}", self.damping),
            ));
        }
        if !(self.tolerance >= 0.0 && self.tolerance.is_finite()) {
            return Err(invalid_input(
                "tolerance",
                format!("must be non-negative and finite, got {}", self.tolerance),
            ));
        }
        Ok(())
    }

    /// Checks whether the volume constraint is satisfied within tolerance.
    pub fn volume_constraint_satisfied(&self, density: &[f64], tol: f64) -> bool {
        let vol: f64 = density.iter().sum::<f64>() / density.len() as f64;
        (vol - self.simp.volume_fraction).abs() < tol
    }
}

// ---------------------------------------------------------------------------
// MultiLoadTopOpt
// ---------------------------------------------------------------------------

/// Load case for multi-load topology optimization.
#[derive(Debug, Clone)]
pub struct LoadCase {
    /// Description of the load, for example a nodal force vector.
    ///
    /// [`MultiLoadTopOpt::run`] and [`MultiLoadTopOpt::evaluate`] never
    /// interpret it: it is handed unchanged to the finite-element closure they
    /// are given. [`MultiLoadTopOpt::run_fe`] and [`Q4Model::simp_oracle`] read
    /// it as the nodal force vector of the Q4 model, of length
    /// `2 (nx + 1) (ny + 1)`.
    pub loads: Vec<f64>,
    /// Weight for this load case in the weighted compliance objective.
    pub weight: f64,
}

impl LoadCase {
    /// Creates a new load case.
    pub fn new(loads: Vec<f64>, weight: f64) -> Self {
        Self { loads, weight }
    }
}

/// Multi-load case topology optimization.
///
/// Minimizes the weighted sum of the compliances of several load cases,
/// `c(ρ) = Σ_k w_k c_k(ρ)`, subject to a volume constraint, with SIMP material
/// interpolation, sensitivity filtering and an optimality-criteria update.
///
/// The structural response is supplied as a closure `fea(density, load_case)`
/// that solves the equilibrium problem for the given density field and load
/// case and returns `(compliance, element_strain_energy)`. The latter holds
/// `u_e^T K_e_unit u_e` for every element, the strain energy of the element
/// with unit Young's modulus, from which the sensitivity
/// `dc_k/dρ_e = -p ρ_e^(p-1) (E_solid - E_void) u_e^T K_e_unit u_e`
/// follows (see [`SiMP::compliance_sensitivity_at`]). [`MultiLoadTopOpt::run`]
/// accepts any such closure, so other structural models can drive it;
/// [`MultiLoadTopOpt::run_fe`] uses the plane-stress Q4 solver of [`fe_q4`]
/// through [`Q4Model::simp_oracle`].
#[derive(Debug, Clone)]
pub struct MultiLoadTopOpt {
    /// SIMP model.
    pub simp: SiMP,
    /// Filter.
    pub filter: FilteringMethods,
    /// Load cases.
    pub load_cases: Vec<LoadCase>,
    /// Move limit.
    pub move_limit: f64,
    /// Damping for OC update.
    pub damping: f64,
    /// Compliance history per iteration.
    pub compliance_history: Vec<f64>,
}

/// Shorthand for an [`PhysicsError::InvalidInput`].
fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> PhysicsError {
    PhysicsError::InvalidInput {
        field: field.into(),
        reason: reason.into(),
    }
}

impl MultiLoadTopOpt {
    /// Creates a new `MultiLoadTopOpt`.
    pub fn new(simp: SiMP, filter: FilteringMethods, load_cases: Vec<LoadCase>) -> Self {
        Self {
            simp,
            filter,
            load_cases,
            move_limit: 0.2,
            damping: 0.5,
            compliance_history: Vec::new(),
        }
    }

    /// Load-proxy value `Σ_k w_k Σ_e E(ρ_e) f_e² / n`; **not** a compliance.
    ///
    /// The proxy multiplies the squared load by the stiffness, whereas the
    /// compliance `f^T K(ρ)^-1 f` falls as the structure gets stiffer, so
    /// minimising the proxy removes stiffness. It solves no finite-element
    /// problem and is kept only for API compatibility.
    #[deprecated(
        since = "0.1.3",
        note = "not a compliance: grows with stiffness and solves no FE problem; use `MultiLoadTopOpt::evaluate`"
    )]
    pub fn weighted_compliance(&self, density: &[f64]) -> f64 {
        let n = density.len();
        self.load_cases
            .iter()
            .map(|lc| {
                let c: f64 = density
                    .iter()
                    .zip(lc.loads.iter())
                    .map(|(&rho, &f)| self.simp.interpolated_stiffness(rho) * f * f / n as f64)
                    .sum();
                lc.weight * c
            })
            .sum()
    }

    /// Compliance-sensitivity formula evaluated with the proxy "strain energy"
    /// `E(ρ_e) f_e² / n`.
    ///
    /// That quantity is not the strain energy of any structure, and the result
    /// is neither the gradient of [`MultiLoadTopOpt::weighted_compliance`] nor
    /// of a compliance. Kept only for API compatibility.
    #[deprecated(
        since = "0.1.3",
        note = "sensitivity of a load proxy, not of the compliance; use `MultiLoadTopOpt::evaluate`"
    )]
    pub fn aggregated_sensitivity(&self, density: &[f64]) -> Vec<f64> {
        let n = density.len();
        let mut agg = vec![0.0f64; n];
        for lc in &self.load_cases {
            for (e, (&rho, &f)) in density.iter().zip(lc.loads.iter()).enumerate() {
                let rho_c = rho.clamp(1e-12, 1.0);
                let se = self.simp.interpolated_stiffness(rho) * f * f / n as f64;
                let dsens = -self.simp.penalty
                    * rho_c.powf(self.simp.penalty - 1.0)
                    * (self.simp.material.e_solid - self.simp.material.e_void)
                    * se;
                agg[e] += lc.weight * dsens;
            }
        }
        agg
    }

    /// Evaluates the weighted compliance and its sensitivity at `density`.
    ///
    /// `fea(density, load_case)` must return the compliance of the load case
    /// and the per-element strain energies `u_e^T K_e_unit u_e` (see the type
    /// documentation). Returns `(Σ_k w_k c_k, Σ_k w_k dc_k/dρ)`; the
    /// sensitivity is unfiltered.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInput`] if `density` does not have one entry per
    /// grid element, there is no load case, a weight is negative or not
    /// finite, or the closure returns a compliance or strain energies that
    /// are not finite or have the wrong length. Errors returned by the
    /// closure are propagated unchanged.
    pub fn evaluate<F>(&self, density: &[f64], mut fea: F) -> PhysicsResult<(f64, Vec<f64>)>
    where
        F: FnMut(&[f64], &LoadCase) -> PhysicsResult<(f64, Vec<f64>)>,
    {
        let n = self.simp.grid.n_elem();
        if density.len() != n {
            return Err(invalid_input(
                "density",
                format!(
                    "expected one entry per element ({n}), got {}",
                    density.len()
                ),
            ));
        }
        if self.load_cases.is_empty() {
            return Err(invalid_input(
                "load_cases",
                "at least one load case is required",
            ));
        }
        let mut compliance = 0.0;
        let mut sensitivity = vec![0.0; n];
        for (k, load_case) in self.load_cases.iter().enumerate() {
            if !load_case.weight.is_finite() || load_case.weight < 0.0 {
                return Err(invalid_input(
                    format!("load_cases[{k}].weight"),
                    format!("must be finite and non-negative, got {}", load_case.weight),
                ));
            }
            let (case_compliance, strain_energy) = fea(density, load_case)?;
            if strain_energy.len() != n {
                return Err(invalid_input(
                    "strain_energy",
                    format!(
                        "load case {k}: expected one entry per element ({n}), got {}",
                        strain_energy.len()
                    ),
                ));
            }
            if !case_compliance.is_finite() || strain_energy.iter().any(|v| !v.is_finite()) {
                return Err(invalid_input(
                    "fea",
                    format!("load case {k}: compliance and strain energies must be finite"),
                ));
            }
            compliance += load_case.weight * case_compliance;
            let case_sensitivity = self.simp.compliance_sensitivity_at(density, &strain_energy);
            for (total, d) in sensitivity.iter_mut().zip(case_sensitivity.iter()) {
                *total += load_case.weight * d;
            }
        }
        Ok((compliance, sensitivity))
    }

    /// Runs the multi-load topology optimization.
    ///
    /// Every iteration evaluates all load cases with `fea` (see
    /// [`MultiLoadTopOpt::evaluate`]), filters the aggregated sensitivity and
    /// applies an optimality-criteria update that meets the volume fraction of
    /// the SIMP model. The weighted compliance of each iteration is appended
    /// to [`MultiLoadTopOpt::compliance_history`]. The loop stops after
    /// `max_iter` iterations or once the largest density change drops below
    /// `tol`; the final density is stored in `self.simp.density` and returned.
    ///
    /// # Errors
    ///
    /// Everything [`MultiLoadTopOpt::evaluate`] reports, plus
    /// [`PhysicsError::InvalidInput`] if the stored density or the filter do
    /// not match the grid. On error `self.simp.density` is left unchanged.
    pub fn run<F>(&mut self, max_iter: usize, tol: f64, mut fea: F) -> PhysicsResult<Vec<f64>>
    where
        F: FnMut(&[f64], &LoadCase) -> PhysicsResult<(f64, Vec<f64>)>,
    {
        let n = self.simp.grid.n_elem();
        if self.simp.density.len() != n {
            return Err(invalid_input(
                "simp.density",
                format!(
                    "expected one entry per element ({n}), got {}",
                    self.simp.density.len()
                ),
            ));
        }
        if self.filter.grid.n_elem() != n {
            return Err(invalid_input(
                "filter",
                format!(
                    "filter grid has {} elements but the SIMP grid has {n}",
                    self.filter.grid.n_elem()
                ),
            ));
        }
        let vf = self.simp.volume_fraction;
        let mut density = self.simp.density.clone();
        self.compliance_history.clear();

        for _iter in 0..max_iter {
            let (compliance, raw_sens) = self.evaluate(&density, &mut fea)?;
            self.compliance_history.push(compliance);

            let filtered_sens = self.filter.sensitivity_filter(&density, &raw_sens);
            let (new_density, _multiplier) =
                oc_density_update(&density, &filtered_sens, vf, self.move_limit, self.damping);

            let change: f64 = density
                .iter()
                .zip(new_density.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            density = new_density;
            if change < tol {
                break;
            }
        }

        self.simp.density = density.clone();
        Ok(density)
    }

    /// Runs the multi-load optimization with the built-in plane-stress Q4
    /// finite-element solver of [`fe_q4`].
    ///
    /// `LoadCase::loads` of every load case is the nodal force vector, of
    /// length `2 (nx + 1) (ny + 1)` in the degree-of-freedom numbering of
    /// [`Q4Model::dof`], and `fixed_dofs` lists the supports (zero
    /// displacement). All load cases of one iteration share a single
    /// assembly and factorisation of the stiffness matrix. Apart from that
    /// this is [`MultiLoadTopOpt::run`] with the closure of
    /// [`Q4Model::simp_oracle`]; see there for the iteration and the errors.
    pub fn run_fe(
        &mut self,
        max_iter: usize,
        tol: f64,
        fixed_dofs: &[usize],
    ) -> PhysicsResult<Vec<f64>> {
        let model = Q4Model::new(self.simp.grid, self.simp.material.nu)?;
        let simp = self.simp.clone();
        let fea = model.simp_oracle(&simp, fixed_dofs, LinearSolver::Auto);
        self.run(max_iter, tol, fea)
    }
}

// ---------------------------------------------------------------------------
// Optimality-criteria update
// ---------------------------------------------------------------------------

/// One optimality-criteria (OC) density update under a volume constraint.
///
/// For a trial multiplier `λ > 0` every element becomes
/// `ρ_e (max(-dc_e, 0) / λ)^η`, limited to `[ρ_e - m, ρ_e + m] ∩ [0, 1]` for
/// the move limit `m` and the damping exponent `η`. The mean density falls
/// monotonically as `λ` grows, so the multiplier that meets `volume_target` is
/// found by bisection. The search runs on `ln λ` between bounds derived from
/// the largest sensitivity, which makes it independent of the unit system: a
/// fixed linear bracket such as `[0, 1e9]` cannot resolve multipliers of the
/// order of `1e-6` and below.
///
/// Returns the updated densities and the multiplier. If no element has a
/// negative sensitivity (nothing asks for material) every density drops to its
/// lower move bound and the multiplier is `0`.
fn oc_density_update(
    density: &[f64],
    sensitivity: &[f64],
    volume_target: f64,
    move_limit: f64,
    damping: f64,
) -> (Vec<f64>, f64) {
    let n = density.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let move_limit = move_limit.max(0.0);
    let gain: Vec<f64> = (0..n)
        .map(|e| (-sensitivity.get(e).copied().unwrap_or(0.0)).max(0.0))
        .collect();
    let bounds: Vec<(f64, f64)> = density
        .iter()
        .map(|&rho| ((rho - move_limit).max(0.0), (rho + move_limit).min(1.0)))
        .collect();
    let densities_for = |lambda: f64| -> Vec<f64> {
        density
            .iter()
            .zip(gain.iter())
            .zip(bounds.iter())
            .map(|((&rho, &g), &(lower, upper))| {
                (rho * (g / lambda).powf(damping)).max(lower).min(upper)
            })
            .collect()
    };

    let gain_max = gain.iter().copied().fold(0.0_f64, f64::max);
    if !gain_max.is_finite() || gain_max <= 0.0 {
        return (bounds.iter().map(|&(lower, _)| lower).collect(), 0.0);
    }
    let ln_gain_max = gain_max.ln();
    let mut ln_lo = ln_gain_max - 60.0 * std::f64::consts::LN_10;
    let mut ln_hi = ln_gain_max + 30.0 * std::f64::consts::LN_10;
    for _ in 0..200 {
        let mid = 0.5 * (ln_lo + ln_hi);
        if mid <= ln_lo || mid >= ln_hi {
            break;
        }
        let volume = densities_for(mid.exp()).iter().sum::<f64>() / n as f64;
        if volume > volume_target {
            ln_lo = mid;
        } else {
            ln_hi = mid;
        }
    }
    let lambda = (0.5 * (ln_lo + ln_hi)).exp();
    (densities_for(lambda), lambda)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Computes the volume fraction of a density field.
pub fn volume_fraction(density: &[f64]) -> f64 {
    if density.is_empty() {
        return 0.0;
    }
    density.iter().sum::<f64>() / density.len() as f64
}

/// Normalizes sensitivities to \[-1, 0\] range for stability.
pub fn normalize_sensitivity(sensitivity: &mut [f64]) {
    let min_s = sensitivity.iter().copied().fold(f64::INFINITY, f64::min);
    let max_s = sensitivity
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = (max_s - min_s).max(1e-12);
    for s in sensitivity.iter_mut() {
        *s = (*s - min_s) / range - 1.0;
    }
}

/// Computes binary (0/1) design from density field with threshold.
pub fn binarize(density: &[f64], threshold: f64) -> Vec<bool> {
    density.iter().map(|&r| r >= threshold).collect()
}

/// Counts number of solid elements in a binary design.
pub fn count_solid(binary: &[bool]) -> usize {
    binary.iter().filter(|&&b| b).count()
}

/// Checks if a topology design is connected (solid phase) using BFS.
pub fn is_connected(density: &[f64], grid: GridSize, threshold: f64) -> bool {
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let solid: Vec<bool> = density.iter().map(|&r| r >= threshold).collect();
    let start = solid.iter().position(|&s| s);
    let Some(start) = start else { return true };

    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited[start] = true;

    while let Some(e) = queue.pop_front() {
        let (row, col) = (e / nx, e % nx);
        let neighbors = [
            if row > 0 {
                Some((row - 1) * nx + col)
            } else {
                None
            },
            if row + 1 < ny {
                Some((row + 1) * nx + col)
            } else {
                None
            },
            if col > 0 {
                Some(row * nx + col - 1)
            } else {
                None
            },
            if col + 1 < nx {
                Some(row * nx + col + 1)
            } else {
                None
            },
        ];
        for nbr in neighbors.into_iter().flatten() {
            if solid[nbr] && !visited[nbr] {
                visited[nbr] = true;
                queue.push_back(nbr);
            }
        }
    }

    // All solid elements should be visited
    solid.iter().zip(visited.iter()).all(|(&s, &v)| !s || v)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn small_grid() -> GridSize {
        GridSize::new(4, 4)
    }

    fn default_material() -> MaterialParams {
        MaterialParams::default()
    }

    // --- GridSize ---

    #[test]
    fn test_grid_size_n_elem() {
        let g = GridSize::new(5, 3);
        assert_eq!(g.n_elem(), 15);
    }

    #[test]
    fn test_grid_size_row_col() {
        let g = GridSize::new(5, 3);
        assert_eq!(g.row_col(7), (1, 2));
    }

    // --- SiMP ---

    #[test]
    fn test_simp_initial_density() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        assert!((simp.current_volume() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_simp_stiffness_solid() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        let e = simp.interpolated_stiffness(1.0);
        assert!((e - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simp_stiffness_void() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        let e = simp.interpolated_stiffness(0.0);
        assert!((e - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn test_simp_stiffness_interpolation() {
        let simp = SiMP::new(small_grid(), 0.5, 2.0, 2.0, default_material());
        // For penalty=2, rho=0.5: E = 1e-9 + 0.25 * (1 - 1e-9) ≈ 0.25
        let e = simp.interpolated_stiffness(0.5);
        assert!((e - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_simp_sensitivity_positive() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        let s = simp.stiffness_sensitivity(0.5);
        assert!(s > 0.0);
    }

    #[test]
    fn test_simp_compliance_sensitivity_negative() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        let se = vec![1.0; 16];
        let cs = simp.compliance_sensitivity(&se);
        assert!(cs.iter().all(|&s| s < 0.0));
    }

    #[test]
    fn test_simp_is_valid() {
        let simp = SiMP::new(small_grid(), 0.5, 3.0, 2.0, default_material());
        assert!(simp.is_valid());
    }

    #[test]
    fn test_simp_stiffness_field_length() {
        let grid = small_grid();
        let simp = SiMP::new(grid, 0.5, 3.0, 2.0, default_material());
        assert_eq!(simp.stiffness_field().len(), grid.n_elem());
    }

    // --- FilteringMethods ---

    #[test]
    fn test_density_filter_uniform() {
        let grid = GridSize::new(3, 3);
        let filter = FilteringMethods::new(1.5, grid);
        let density = vec![0.5; 9];
        let filtered = filter.density_filter(&density);
        assert!(filtered.iter().all(|&r| (r - 0.5).abs() < 1e-10));
    }

    #[test]
    fn test_density_filter_smooths() {
        let grid = GridSize::new(5, 5);
        let filter = FilteringMethods::new(2.0, grid);
        let mut density = vec![0.0; 25];
        density[12] = 1.0; // center spike
        let filtered = filter.density_filter(&density);
        // Surrounding elements should have nonzero filtered density
        assert!(filtered[11] > 0.0 || filtered[13] > 0.0);
    }

    #[test]
    fn test_sensitivity_filter_output_length() {
        let grid = GridSize::new(3, 3);
        let filter = FilteringMethods::new(1.5, grid);
        let density = vec![0.5; 9];
        let sens = vec![-1.0; 9];
        let out = filter.sensitivity_filter(&density, &sens);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn test_heaviside_projection_solid() {
        let grid = GridSize::new(2, 2);
        let filter = FilteringMethods::new(1.5, grid);
        let density = vec![1.0; 4];
        let proj = filter.heaviside_projection(&density, 10.0, 0.5);
        assert!(proj.iter().all(|&r| r > 0.9));
    }

    #[test]
    fn test_heaviside_projection_void() {
        let grid = GridSize::new(2, 2);
        let filter = FilteringMethods::new(1.5, grid);
        let density = vec![0.0; 4];
        let proj = filter.heaviside_projection(&density, 10.0, 0.5);
        assert!(proj.iter().all(|&r| r < 0.1));
    }

    #[test]
    fn test_heaviside_sensitivity_positive() {
        let grid = GridSize::new(2, 2);
        let filter = FilteringMethods::new(1.5, grid);
        let density = vec![0.5; 4];
        let sens = filter.heaviside_sensitivity(&density, 5.0, 0.5);
        assert!(sens.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_filter_sensitivity_chain() {
        let grid = GridSize::new(3, 3);
        let filter = FilteringMethods::new(1.5, grid);
        let sens = vec![1.0; 9];
        let density = vec![0.5; 9];
        let out = filter.filter_sensitivity_chain(&sens, &density);
        assert_eq!(out.len(), 9);
    }

    // --- TopOptSolver ---

    #[test]
    fn test_topopt_oc_update_volume() {
        let grid = GridSize::new(4, 4);
        let simp = SiMP::new(grid, 0.4, 3.0, 1.5, MaterialParams::default());
        let filter = FilteringMethods::new(1.5, grid);
        let solver = TopOptSolver::new(simp, filter);
        let density = vec![0.5; 16];
        let sensitivity = vec![-1.0; 16];
        let (new_d, _lm) = solver.oc_update(&density, &sensitivity, 0.4);
        let vol: f64 = new_d.iter().sum::<f64>() / 16.0;
        assert!((vol - 0.4).abs() < 0.05);
    }

    #[test]
    fn test_topopt_run_mock_convergence() {
        let grid = GridSize::new(4, 4);
        let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
        let filter = FilteringMethods::new(1.5, grid);
        let mut solver = TopOptSolver::new(simp, filter).with_max_iter(10);
        let result = solver.run_mock(1.0);
        assert_eq!(result.len(), 16);
        assert!(!solver.compliance_history.is_empty());
    }

    #[test]
    fn test_topopt_compliance_positive() {
        let grid = GridSize::new(4, 4);
        let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
        let filter = FilteringMethods::new(1.5, grid);
        let solver = TopOptSolver::new(simp, filter);
        let se = vec![1.0; 16];
        let c = solver.compute_compliance(&se);
        assert!(c > 0.0);
    }

    #[test]
    fn test_topopt_volume_constraint() {
        let grid = GridSize::new(4, 4);
        let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
        let filter = FilteringMethods::new(1.5, grid);
        let solver = TopOptSolver::new(simp, filter);
        let density = vec![0.5; 16];
        assert!(solver.volume_constraint_satisfied(&density, 0.01));
    }

    #[test]
    fn test_topopt_with_damping() {
        let grid = GridSize::new(4, 4);
        let simp = SiMP::new(grid, 0.5, 3.0, 1.5, MaterialParams::default());
        let filter = FilteringMethods::new(1.5, grid);
        let solver = TopOptSolver::new(simp, filter).with_damping(0.3);
        assert!((solver.damping - 0.3).abs() < 1e-10);
    }

    // --- LevelSetTopOpt ---

    #[test]
    fn test_levelset_init_phi_length() {
        let grid = GridSize::new(4, 4);
        let ls = LevelSetTopOpt::new(grid, 0.5, 0.1, MaterialParams::default());
        assert_eq!(ls.phi.len(), 25); // (4+1)*(4+1)
    }

    #[test]
    fn test_levelset_element_density_length() {
        let grid = GridSize::new(4, 4);
        let ls = LevelSetTopOpt::new(grid, 0.5, 0.1, MaterialParams::default());
        assert_eq!(ls.element_density().len(), 16);
    }

    #[test]
    fn test_levelset_element_density_range() {
        let grid = GridSize::new(4, 4);
        let ls = LevelSetTopOpt::new(grid, 0.5, 0.1, MaterialParams::default());
        let d = ls.element_density();
        assert!(d.iter().all(|&r| (0.0..=1.0).contains(&r)));
    }

    #[test]
    fn test_levelset_hj_update_changes_phi() {
        let grid = GridSize::new(4, 4);
        let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.1, MaterialParams::default());
        let phi_before = ls.phi.clone();
        let velocity = vec![1.0; 25];
        ls.hj_update(&velocity);
        let changed = phi_before
            .iter()
            .zip(ls.phi.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn test_levelset_run_mock() {
        let grid = GridSize::new(4, 4);
        let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.01, MaterialParams::default());
        ls.run_mock(5);
        assert_eq!(ls.compliance_history.len(), 5);
    }

    #[test]
    fn test_levelset_reinitialize() {
        let grid = GridSize::new(4, 4);
        let mut ls = LevelSetTopOpt::new(grid, 0.5, 0.1, MaterialParams::default());
        let phi_before = ls.phi.clone();
        ls.reinitialize(3);
        // phi should change
        let _ = phi_before;
    }

    // --- ManufacturingFilters ---

    #[test]
    fn test_mfg_min_length_scale_output_len() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let density = vec![0.5; 16];
        let out = mf.minimum_length_scale(&density, 10.0);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_mfg_min_length_scale_range() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let density = vec![0.5; 16];
        let out = mf.minimum_length_scale(&density, 10.0);
        assert!(out.iter().all(|&r| (0.0..=1.0).contains(&r)));
    }

    #[test]
    fn test_mfg_overhang_base_unchanged() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 3.0);
        let mut density = vec![0.0; 16];
        // Set bottom row solid
        for d in density[..4].iter_mut() {
            *d = 1.0;
        }
        let out = mf.overhang_filter(&density);
        // Bottom row (build direction 0) should be unchanged
        assert!((out[0] - density[0]).abs() < 1e-10);
    }

    #[test]
    fn test_mfg_casting_filter_len() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let density = vec![0.5; 16];
        let out = mf.casting_filter(&density);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_mfg_extrusion_filter_x() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let mut density = vec![0.0; 16];
        density[0] = 1.0; // only top-left
        let out = mf.extrusion_filter(&density, 0); // extrude in x
        // column 0 should all be 1.0
        for row in 0..4 {
            assert!((out[row * 4] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mfg_extrusion_filter_y() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let mut density = vec![0.0; 16];
        density[0] = 1.0; // only top-left corner
        let out = mf.extrusion_filter(&density, 1); // extrude in y
        // row 0 should all be 1.0
        for &o in out[..4].iter() {
            assert!((o - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mfg_milling_filter_no_enclosed() {
        let grid = GridSize::new(4, 4);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        let density = vec![0.0; 16]; // all void = all accessible
        let out = mf.milling_filter(&density, 0.5);
        assert!(out.iter().all(|&r| r < 0.5)); // all remain void
    }

    #[test]
    fn test_mfg_milling_filter_fills_enclosed() {
        let grid = GridSize::new(5, 5);
        let mf = ManufacturingFilters::new(grid, 2.0, PI / 4.0);
        // Ring of solid around center void
        let mut density = vec![1.0; 25];
        density[12] = 0.0; // center void
        let out = mf.milling_filter(&density, 0.5);
        // The center should be filled (enclosed)
        assert!(out[12] >= 0.5);
    }

    // --- Utility functions ---

    #[test]
    fn test_volume_fraction() {
        let d = vec![0.5; 10];
        assert!((volume_fraction(&d) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_volume_fraction_empty() {
        assert_eq!(volume_fraction(&[]), 0.0);
    }

    #[test]
    fn test_normalize_sensitivity() {
        let mut s = vec![-4.0, -2.0, -1.0];
        normalize_sensitivity(&mut s);
        let max_s = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_s = s.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(max_s <= 0.0);
        assert!(min_s >= -1.0);
    }

    #[test]
    fn test_binarize() {
        let d = vec![0.3, 0.7, 0.5, 0.1];
        let b = binarize(&d, 0.5);
        assert_eq!(b, vec![false, true, true, false]);
    }

    #[test]
    fn test_count_solid() {
        let b = vec![true, false, true, true];
        assert_eq!(count_solid(&b), 3);
    }

    #[test]
    fn test_is_connected_uniform() {
        let grid = GridSize::new(3, 3);
        let density = vec![1.0; 9];
        assert!(is_connected(&density, grid, 0.5));
    }

    #[test]
    fn test_is_connected_disconnected() {
        let grid = GridSize::new(3, 3);
        let mut density = vec![0.0; 9];
        density[0] = 1.0;
        density[8] = 1.0; // corners only — not connected in 4-connectivity
        assert!(!is_connected(&density, grid, 0.5));
    }

    #[test]
    fn test_is_connected_all_void() {
        let grid = GridSize::new(3, 3);
        let density = vec![0.0; 9];
        assert!(is_connected(&density, grid, 0.5)); // vacuously true
    }
}

#[cfg(test)]
mod filter_and_solver_tests;
#[cfg(test)]
mod multiload_casting_tests;
