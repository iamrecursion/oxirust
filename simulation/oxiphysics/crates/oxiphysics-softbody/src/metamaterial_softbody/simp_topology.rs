// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SIMP topology optimisation of 2-D metamaterial unit cells.
//!
//! * Bilinear Q4 plane-stress elements on a regular `nx × ny` grid of square
//!   elements (unit thickness), with the element stiffness matrix integrated
//!   by 2 × 2 Gauss quadrature (identical to the closed form used in
//!   Sigmund's "99-line topology optimization code", 2001).
//! * Modified SIMP interpolation `E(ρ) = E₀ (ε + ρᵖ (1 − ε))`.
//! * **Compliance** problems ([`MetamaterialOptimization::solve_compliance`]):
//!   global stiffness assembly in CSR form, Jacobi-PCG solve for one load
//!   case, adjoint sensitivities `∂c/∂ρₑ = −∂Eₑ/∂ρₑ · uₑᵀ k₀ uₑ`, Sigmund's
//!   sensitivity filter and the optimality-criteria (OC) update.
//! * **Target Poisson's ratio** problems
//!   ([`MetamaterialOptimization::poisson_objective`]): three-load-case
//!   periodic homogenisation (Andreassen & Andreasen, "How to determine
//!   composite material properties using numerical homogenization",
//!   Comput. Mater. Sci. 83, 2014) with objective `(ν_H − ν*)²`, minimised by
//!   a volume-preserving projected-gradient method with backtracking.

use crate::numerical_softbody::{CsrMatrix, PcgParams, PcgSolver};
use crate::{Error, Result};

/// Lower bound on design densities used by the update schemes.
const MIN_DENSITY: f64 = 1.0e-3_f64;
/// Relative PCG tolerance for the finite-element solves.
const PCG_TOLERANCE: f64 = 1.0e-10_f64;

/// Design variable for topology optimization of a metamaterial unit cell.
#[derive(Debug, Clone)]
pub struct DesignElement {
    /// Element density (0 = void, 1 = solid).
    pub density: f64,
    /// Element centre position \[m\].
    pub pos: [f64; 2],
    /// Element edge length \[m\].
    pub size: f64,
    /// Young's modulus of the solid material in this element \[Pa\].
    pub stiffness_contrib: f64,
}

impl DesignElement {
    /// Create a new design element.
    pub fn new(pos: [f64; 2], size: f64, e_material: f64) -> Self {
        Self {
            density: 1.0,
            pos,
            size,
            stiffness_contrib: e_material,
        }
    }

    /// Classic SIMP penalized stiffness `E₀ ρᵖ`.
    pub fn penalized_stiffness(&self, penalty: f64) -> f64 {
        self.stiffness_contrib * self.density.powf(penalty)
    }

    /// Modified SIMP modulus `E₀ (ε + ρᵖ (1 − ε))` with stiffness floor `ε`.
    pub fn simp_modulus(&self, penalty: f64, floor: f64) -> f64 {
        self.stiffness_contrib * (floor + self.density.powf(penalty) * (1.0 - floor))
    }

    /// Derivative of [`simp_modulus`](Self::simp_modulus) with respect to the
    /// density: `p ρᵖ⁻¹ (1 − ε) E₀`.
    pub fn simp_modulus_derivative(&self, penalty: f64, floor: f64) -> f64 {
        self.stiffness_contrib * penalty * self.density.powf(penalty - 1.0) * (1.0 - floor)
    }
}

/// Stiffness matrix of a square bilinear (Q4) plane-stress element with unit
/// Young's modulus and unit thickness.
///
/// The matrix is independent of the element size.  Local node order is
/// counter-clockwise from the lower-left corner, `(0,0), (a,0), (a,a), (0,a)`,
/// and the DOF order is `[u₀ₓ, u₀ᵧ, u₁ₓ, u₁ᵧ, …]`.
pub fn q4_plane_stress_stiffness(nu: f64) -> [[f64; 8]; 8] {
    let scale = 1.0 / (1.0 - nu * nu);
    let d = [
        [scale, scale * nu, 0.0],
        [scale * nu, scale, 0.0],
        [0.0, 0.0, scale * (1.0 - nu) * 0.5],
    ];
    let xi_n = [-1.0_f64, 1.0, 1.0, -1.0];
    let eta_n = [-1.0_f64, -1.0, 1.0, 1.0];
    let g = 1.0 / 3.0_f64.sqrt();
    let mut ke = [[0.0_f64; 8]; 8];
    // Natural coordinates of an element with side 2: Jacobian = identity.
    for &xi in &[-g, g] {
        for &eta in &[-g, g] {
            let mut b = [[0.0_f64; 8]; 3];
            for a in 0..4 {
                let dndx = 0.25 * xi_n[a] * (1.0 + eta_n[a] * eta);
                let dndy = 0.25 * eta_n[a] * (1.0 + xi_n[a] * xi);
                b[0][2 * a] = dndx;
                b[1][2 * a + 1] = dndy;
                b[2][2 * a] = dndy;
                b[2][2 * a + 1] = dndx;
            }
            // ke += Bᵀ D B
            let mut db = [[0.0_f64; 8]; 3];
            for (db_row, d_row) in db.iter_mut().zip(d.iter()) {
                for (c, db_rc) in db_row.iter_mut().enumerate() {
                    *db_rc = (0..3).map(|k| d_row[k] * b[k][c]).sum();
                }
            }
            for (r, ke_row) in ke.iter_mut().enumerate() {
                for (c, ke_rc) in ke_row.iter_mut().enumerate() {
                    *ke_rc += (0..3).map(|k| b[k][r] * db[k][c]).sum::<f64>();
                }
            }
        }
    }
    ke
}

/// `uᵀ K v` for an 8 × 8 element matrix.
fn element_energy(ke: &[[f64; 8]; 8], u: &[f64; 8], v: &[f64; 8]) -> f64 {
    ke.iter()
        .zip(u.iter())
        .map(|(row, ui)| ui * row.iter().zip(v.iter()).map(|(k, vj)| k * vj).sum::<f64>())
        .sum()
}

/// A single static load case on the nodal grid of a [`MetamaterialOptimization`].
///
/// Nodes are indexed `node = j · (nx + 1) + i` for `i ∈ 0..=nx`,
/// `j ∈ 0..=ny`; node `(i, j)` sits at `(i·a, j·a)`.  DOF `2·node` is the
/// x-displacement and `2·node + 1` the y-displacement.
#[derive(Debug, Clone, PartialEq)]
pub struct SimpLoadCase {
    /// Number of elements in x.
    pub nx: usize,
    /// Number of elements in y.
    pub ny: usize,
    /// Nodal force vector (length `2 (nx+1) (ny+1)`) \[N\].
    pub forces: Vec<f64>,
    /// Indices of fixed (zero-displacement) DOFs.
    pub fixed_dofs: Vec<usize>,
}

impl SimpLoadCase {
    /// Unloaded, unsupported load case for an `nx × ny` element grid.
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            nx,
            ny,
            forces: vec![0.0; 2 * (nx + 1) * (ny + 1)],
            fixed_dofs: Vec::new(),
        }
    }

    /// Classic cantilever: the left edge (`i = 0`) is clamped and a downward
    /// point load of magnitude `load` acts at the mid-height node of the
    /// right edge.
    pub fn cantilever(nx: usize, ny: usize, load: f64) -> Self {
        let mut lc = Self::new(nx, ny);
        for j in 0..=ny {
            let node = lc.node_index(0, j);
            lc.fixed_dofs.push(2 * node);
            lc.fixed_dofs.push(2 * node + 1);
        }
        let node = lc.node_index(nx, ny / 2);
        lc.forces[2 * node + 1] = -load;
        lc
    }

    /// Global node index of grid node `(i, j)`.
    pub fn node_index(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    fn checked_node(&self, i: usize, j: usize) -> Result<usize> {
        if i > self.nx || j > self.ny {
            return Err(Error::InvalidParameter(format!(
                "node ({i}, {j}) outside {}x{} element grid",
                self.nx, self.ny
            )));
        }
        Ok(self.node_index(i, j))
    }

    /// Clamp both DOFs of node `(i, j)`.
    pub fn fix_node(&mut self, i: usize, j: usize) -> Result<()> {
        let node = self.checked_node(i, j)?;
        self.fixed_dofs.push(2 * node);
        self.fixed_dofs.push(2 * node + 1);
        Ok(())
    }

    /// Add a nodal force `[fx, fy]` at node `(i, j)`.
    pub fn add_nodal_force(&mut self, i: usize, j: usize, force: [f64; 2]) -> Result<()> {
        let node = self.checked_node(i, j)?;
        self.forces[2 * node] += force[0];
        self.forces[2 * node + 1] += force[1];
        Ok(())
    }
}

/// Result of a compliance analysis.
#[derive(Debug, Clone)]
pub struct ComplianceAnalysis {
    /// Compliance `c = fᵀ u` \[J\].
    pub compliance: f64,
    /// Full nodal displacement vector \[m\].
    pub displacements: Vec<f64>,
    /// Sensitivities `∂c/∂ρₑ` (all ≤ 0).
    pub sensitivities: Vec<f64>,
}

/// Homogenised (effective) plane-stress properties of a periodic unit cell.
#[derive(Debug, Clone)]
pub struct HomogenizedProperties {
    /// Effective elasticity matrix in Voigt notation `[ε₁₁, ε₂₂, γ₁₂]` \[Pa\].
    pub c_h: [[f64; 3]; 3],
    /// Effective Poisson's ratio `ν_H = 2 C₁₂ / (C₁₁ + C₂₂)` (equal to both
    /// `ν₁₂` and `ν₂₁` for square-symmetric designs).
    pub poisson_ratio: f64,
    /// Per-element sensitivities `∂C_H/∂ρₑ`.
    pub element_sensitivities: Vec<[[f64; 3]; 3]>,
}

/// Value and gradient of the target-Poisson objective `(ν_H − ν*)²`.
#[derive(Debug, Clone)]
pub struct PoissonObjective {
    /// Objective value.
    pub objective: f64,
    /// Current homogenised Poisson's ratio.
    pub poisson_ratio: f64,
    /// Gradient with respect to the element densities.
    pub sensitivities: Vec<f64>,
}

/// Summary of one optimisation iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimpIterate {
    /// Objective value of the design analysed in this iteration.
    pub objective: f64,
    /// Maximum absolute density change of the update.
    pub change: f64,
}

/// Topology optimization for metamaterial unit cell properties.
///
/// Uses SIMP (Solid Isotropic Material with Penalization) on a finite-element
/// discretisation of the unit cell to optimise the material distribution
/// either for minimum compliance under a given [`SimpLoadCase`] or for a
/// target homogenised Poisson's ratio (`target_poisson`).
#[derive(Debug, Clone)]
pub struct MetamaterialOptimization {
    /// Design elements (unit cell discretization), row-major `iy · nx + ix`.
    pub elements: Vec<DesignElement>,
    /// SIMP penalty exponent.
    pub penalty: f64,
    /// Volume fraction constraint.
    pub volume_fraction: f64,
    /// Target Poisson's ratio of the homogenised unit cell.
    pub target_poisson: f64,
    /// Optimization iteration count.
    pub iteration: usize,
    /// Sensitivity filter radius \[m\].
    pub filter_radius: f64,
    /// Poisson's ratio of the solid base material.
    pub material_poisson: f64,
    /// Modified-SIMP stiffness floor `ε` (void modulus = `ε E₀`).
    pub stiffness_floor: f64,
    /// Current move limit of the target-Poisson projected-gradient update.
    pub poisson_move_limit: f64,
    nx: usize,
    ny: usize,
    element_size: f64,
}

impl MetamaterialOptimization {
    /// Create a new topology optimization problem.
    ///
    /// * `nx`, `ny`       — grid resolution (square elements of edge `cell_size / nx`)
    /// * `cell_size`      — unit cell width \[m\]
    /// * `e_material`     — solid material Young's modulus \[Pa\]
    /// * `vol_frac`       — target volume fraction (0 to 1)
    /// * `target_poisson` — target Poisson's ratio
    ///
    /// All densities start at `vol_frac`; the base material has ν = 0.3.
    pub fn new(
        nx: usize,
        ny: usize,
        cell_size: f64,
        e_material: f64,
        vol_frac: f64,
        target_poisson: f64,
    ) -> Self {
        let dx = cell_size / nx.max(1) as f64;
        let mut elements = Vec::with_capacity(nx * ny);
        for iy in 0..ny {
            for ix in 0..nx {
                let pos = [ix as f64 * dx + dx * 0.5, iy as f64 * dx + dx * 0.5];
                let mut e = DesignElement::new(pos, dx, e_material);
                e.density = vol_frac;
                elements.push(e);
            }
        }
        Self {
            elements,
            penalty: 3.0,
            volume_fraction: vol_frac,
            target_poisson,
            iteration: 0,
            filter_radius: 2.0 * dx,
            material_poisson: 0.3,
            stiffness_floor: 1.0e-9,
            poisson_move_limit: 0.1,
            nx,
            ny,
            element_size: dx,
        }
    }

    /// Number of elements in x.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Number of elements in y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Element edge length \[m\].
    pub fn element_size(&self) -> f64 {
        self.element_size
    }

    /// Compute current volume fraction.
    pub fn current_volume_fraction(&self) -> f64 {
        if self.elements.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.elements.iter().map(|e| e.density).sum();
        sum / self.elements.len() as f64
    }

    /// Compute effective stiffness of current design \[Pa\].
    pub fn effective_stiffness(&self) -> f64 {
        self.elements
            .iter()
            .map(|e| e.penalized_stiffness(self.penalty) * e.size * e.size)
            .sum::<f64>()
            / self.elements.len().max(1) as f64
    }

    fn validate(&self) -> Result<()> {
        if self.nx == 0 || self.ny == 0 {
            return Err(Error::InvalidParameter(
                "SIMP grid must have nx, ny >= 1".into(),
            ));
        }
        if self.elements.len() != self.nx * self.ny {
            return Err(Error::InvalidParameter(format!(
                "expected {} design elements, found {}",
                self.nx * self.ny,
                self.elements.len()
            )));
        }
        if !(self.penalty.is_finite() && self.penalty >= 1.0) {
            return Err(Error::InvalidParameter(format!(
                "SIMP penalty must be >= 1, got {}",
                self.penalty
            )));
        }
        if !(self.material_poisson > -1.0 && self.material_poisson < 0.5) {
            return Err(Error::InvalidParameter(format!(
                "material Poisson's ratio must lie in (-1, 0.5), got {}",
                self.material_poisson
            )));
        }
        if !(self.stiffness_floor > 0.0 && self.stiffness_floor < 1.0) {
            return Err(Error::InvalidParameter(format!(
                "stiffness floor must lie in (0, 1), got {}",
                self.stiffness_floor
            )));
        }
        if let Some(e) = self.elements.iter().find(|e| {
            let density_ok = e.density.is_finite() && (0.0..=1.0).contains(&e.density);
            let modulus_ok = e.stiffness_contrib.is_finite() && e.stiffness_contrib > 0.0;
            !(density_ok && modulus_ok)
        }) {
            return Err(Error::InvalidParameter(format!(
                "invalid design element: density {}, modulus {}",
                e.density, e.stiffness_contrib
            )));
        }
        Ok(())
    }

    /// Global DOFs of element `(ix, iy)` on the non-periodic node grid.
    fn element_dofs(&self, ix: usize, iy: usize) -> [usize; 8] {
        let row = self.nx + 1;
        let nodes = [
            iy * row + ix,
            iy * row + ix + 1,
            (iy + 1) * row + ix + 1,
            (iy + 1) * row + ix,
        ];
        let mut dofs = [0usize; 8];
        for (k, &n) in nodes.iter().enumerate() {
            dofs[2 * k] = 2 * n;
            dofs[2 * k + 1] = 2 * n + 1;
        }
        dofs
    }

    /// Global DOFs of element `(ix, iy)` on the periodic node grid
    /// (`nx × ny` unique nodes).
    fn periodic_element_dofs(&self, ix: usize, iy: usize) -> [usize; 8] {
        let wrap = |i: usize, j: usize| (j % self.ny) * self.nx + (i % self.nx);
        let nodes = [
            wrap(ix, iy),
            wrap(ix + 1, iy),
            wrap(ix + 1, iy + 1),
            wrap(ix, iy + 1),
        ];
        let mut dofs = [0usize; 8];
        for (k, &n) in nodes.iter().enumerate() {
            dofs[2 * k] = 2 * n;
            dofs[2 * k + 1] = 2 * n + 1;
        }
        dofs
    }

    /// Assemble the reduced stiffness matrix `Σ Eₑ k₀` over the DOFs that are
    /// mapped to `Some(row)` by `dof_map`.
    fn assemble_reduced(
        &self,
        ke: &[[f64; 8]; 8],
        edofs: &[[usize; 8]],
        dof_map: &[Option<usize>],
        n_reduced: usize,
    ) -> CsrMatrix {
        let mut rows = Vec::with_capacity(edofs.len() * 64);
        let mut cols = Vec::with_capacity(edofs.len() * 64);
        let mut vals = Vec::with_capacity(edofs.len() * 64);
        for (elem, dofs) in self.elements.iter().zip(edofs.iter()) {
            let e_mod = elem.simp_modulus(self.penalty, self.stiffness_floor);
            for (r, &gr) in dofs.iter().enumerate() {
                let Some(Some(rr)) = dof_map.get(gr) else {
                    continue;
                };
                for (c, &gc) in dofs.iter().enumerate() {
                    let Some(Some(cc)) = dof_map.get(gc) else {
                        continue;
                    };
                    rows.push(*rr);
                    cols.push(*cc);
                    vals.push(e_mod * ke[r][c]);
                }
            }
        }
        CsrMatrix::from_coo(n_reduced, n_reduced, &rows, &cols, &vals)
    }

    fn pcg_solve(k: &CsrMatrix, b: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();
        if k.nrows != n || k.ncols != n {
            return Err(Error::General(
                "stiffness matrix / load vector size mismatch".into(),
            ));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let solver = PcgSolver::new(PcgParams {
            max_iter: (50 * n).max(2000),
            tolerance: PCG_TOLERANCE,
        });
        let res = solver.solve(k, b, None);
        if !res.converged || res.x.iter().any(|v| !v.is_finite()) {
            return Err(Error::General(format!(
                "SIMP finite-element solve did not converge (residual {:.3e} after {} iterations)",
                res.residual_norm, res.iterations
            )));
        }
        Ok(res.x)
    }

    fn element_vector(full: &[f64], dofs: &[usize; 8]) -> [f64; 8] {
        let mut out = [0.0_f64; 8];
        for (o, &d) in out.iter_mut().zip(dofs.iter()) {
            *o = full.get(d).copied().unwrap_or(0.0);
        }
        out
    }

    /// Solve the static problem `K(ρ) u = f` for `load` and return the
    /// compliance `fᵀu` together with its exact sensitivities
    /// `∂c/∂ρₑ = −∂Eₑ/∂ρₑ · uₑᵀ k₀ uₑ`.
    pub fn solve_compliance(&self, load: &SimpLoadCase) -> Result<ComplianceAnalysis> {
        self.validate()?;
        let n_dof = 2 * (self.nx + 1) * (self.ny + 1);
        if load.nx != self.nx || load.ny != self.ny || load.forces.len() != n_dof {
            return Err(Error::InvalidParameter(format!(
                "load case grid {}x{} does not match design grid {}x{}",
                load.nx, load.ny, self.nx, self.ny
            )));
        }
        if load.fixed_dofs.is_empty() {
            return Err(Error::InvalidParameter(
                "load case has no fixed DOFs; the stiffness matrix is singular".into(),
            ));
        }
        let mut fixed = vec![false; n_dof];
        for &d in &load.fixed_dofs {
            match fixed.get_mut(d) {
                Some(f) => *f = true,
                None => {
                    return Err(Error::InvalidParameter(format!(
                        "fixed DOF {d} out of range ({n_dof} DOFs)"
                    )));
                }
            }
        }
        let mut dof_map = vec![None; n_dof];
        let mut n_free = 0usize;
        for (slot, &is_fixed) in dof_map.iter_mut().zip(fixed.iter()) {
            if !is_fixed {
                *slot = Some(n_free);
                n_free += 1;
            }
        }
        let ke = q4_plane_stress_stiffness(self.material_poisson);
        let edofs: Vec<[usize; 8]> = (0..self.ny)
            .flat_map(|iy| (0..self.nx).map(move |ix| (ix, iy)))
            .map(|(ix, iy)| self.element_dofs(ix, iy))
            .collect();
        let k = self.assemble_reduced(&ke, &edofs, &dof_map, n_free);
        let mut f_red = vec![0.0_f64; n_free];
        for (slot, &f) in dof_map.iter().zip(load.forces.iter()) {
            if let Some(r) = slot {
                f_red[*r] = f;
            }
        }
        let u_red = Self::pcg_solve(&k, &f_red)?;
        let mut u = vec![0.0_f64; n_dof];
        for (ui, slot) in u.iter_mut().zip(dof_map.iter()) {
            if let Some(r) = slot {
                *ui = u_red[*r];
            }
        }
        let mut compliance = 0.0_f64;
        let mut sensitivities = Vec::with_capacity(self.elements.len());
        for (elem, dofs) in self.elements.iter().zip(edofs.iter()) {
            let ue = Self::element_vector(&u, dofs);
            let energy = element_energy(&ke, &ue, &ue);
            compliance += elem.simp_modulus(self.penalty, self.stiffness_floor) * energy;
            sensitivities
                .push(-elem.simp_modulus_derivative(self.penalty, self.stiffness_floor) * energy);
        }
        Ok(ComplianceAnalysis {
            compliance,
            displacements: u,
            sensitivities,
        })
    }

    /// Compute sensitivity of compliance to density (`dC/dρ`) for `load`.
    pub fn compute_sensitivities(&self, load: &SimpLoadCase) -> Result<Vec<f64>> {
        Ok(self.solve_compliance(load)?.sensitivities)
    }

    /// Filter neighbour lists: for every element, the elements whose centres
    /// lie within `filter_radius`, with hat weights `r − d`.
    ///
    /// Uses the grid structure, so the cost is `O(n · k)` for `k` neighbours.
    pub fn filter_neighbors(&self) -> Vec<Vec<(usize, f64)>> {
        let r = self.filter_radius;
        let n = self.elements.len();
        if n != self.nx * self.ny || r.is_nan() || r <= 0.0 || self.element_size <= 0.0 {
            // Without a consistent grid, each element is its own neighbour.
            return (0..n).map(|i| vec![(i, r.max(1.0))]).collect();
        }
        let reach = (r / self.element_size).ceil() as usize;
        let mut lists = Vec::with_capacity(n);
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                let i = iy * self.nx + ix;
                let pi = self.elements[i].pos;
                let mut list = Vec::new();
                for jy in iy.saturating_sub(reach)..(iy + reach + 1).min(self.ny) {
                    for jx in ix.saturating_sub(reach)..(ix + reach + 1).min(self.nx) {
                        let j = jy * self.nx + jx;
                        let pj = self.elements[j].pos;
                        let d = ((pj[0] - pi[0]).powi(2) + (pj[1] - pi[1]).powi(2)).sqrt();
                        if d < r {
                            list.push((j, r - d));
                        }
                    }
                }
                lists.push(list);
            }
        }
        lists
    }

    /// Apply Sigmund's sensitivity filter
    /// `ĉₑ = Σⱼ Hₑⱼ ρⱼ cⱼ / (ρₑ Σⱼ Hₑⱼ)`.
    ///
    /// Returns an unfiltered copy when `raw` does not have one entry per
    /// element.
    pub fn filter_sensitivities(&self, raw: &[f64]) -> Vec<f64> {
        let neighbors = self.filter_neighbors();
        self.filter_sensitivities_with(raw, &neighbors)
    }

    /// Sensitivity filter with precomputed [`filter_neighbors`](Self::filter_neighbors).
    pub fn filter_sensitivities_with(
        &self,
        raw: &[f64],
        neighbors: &[Vec<(usize, f64)>],
    ) -> Vec<f64> {
        let n = self.elements.len();
        if raw.len() != n || neighbors.len() != n {
            return raw.to_vec();
        }
        let mut filtered = vec![0.0_f64; n];
        for (i, list) in neighbors.iter().enumerate() {
            let mut acc = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for &(j, h) in list {
                if let (Some(ej), Some(&rj)) = (self.elements.get(j), raw.get(j)) {
                    acc += h * ej.density * rj;
                    weight_sum += h;
                }
            }
            filtered[i] = if weight_sum > 1e-30 {
                acc / (weight_sum * self.elements[i].density.max(MIN_DENSITY))
            } else {
                raw[i]
            };
        }
        filtered
    }

    /// OC (Optimality Criteria) update step.
    ///
    /// Minimizes compliance subject to the volume fraction constraint
    /// (`sensitivities` must be compliance sensitivities, i.e. ≤ 0).  The
    /// Lagrange multiplier is bracketed adaptively and found by bisection,
    /// so the update is independent of the magnitude of the sensitivities.
    /// Returns the maximum change in density (convergence criterion).
    pub fn oc_update(&mut self, sensitivities: &[f64]) -> f64 {
        let n = self.elements.len();
        if sensitivities.len() != n || n == 0 {
            return 0.0;
        }
        let move_limit = 0.2;
        let densities: Vec<f64> = self.elements.iter().map(|e| e.density).collect();
        let target = self.volume_fraction;
        let update = |lambda: f64, out: &mut Vec<f64>| -> f64 {
            out.clear();
            let mut vol = 0.0_f64;
            for (&rho, &s) in densities.iter().zip(sensitivities.iter()) {
                let be = (-s / lambda).max(0.0);
                let lo = (rho - move_limit).max(MIN_DENSITY);
                let hi = (rho + move_limit).min(1.0);
                let r = (rho * be.sqrt()).clamp(lo, hi);
                out.push(r);
                vol += r;
            }
            vol / n as f64
        };
        let mut rho_new = Vec::with_capacity(n);
        let mut l1 = 0.0_f64;
        let mut l2 = 1.0_f64;
        // Grow the upper bracket until the volume constraint is satisfied.
        for _ in 0..2100 {
            if update(l2, &mut rho_new) <= target {
                break;
            }
            l1 = l2;
            l2 *= 2.0;
        }
        for _ in 0..200 {
            if (l2 - l1) <= 1e-12 * (l1 + l2) {
                break;
            }
            let lmid = 0.5 * (l1 + l2);
            if update(lmid, &mut rho_new) > target {
                l1 = lmid;
            } else {
                l2 = lmid;
            }
        }
        update(l2, &mut rho_new);
        let mut change = 0.0_f64;
        for (elem, &r) in self.elements.iter_mut().zip(rho_new.iter()) {
            change = change.max((r - elem.density).abs());
            elem.density = r;
        }
        self.iteration += 1;
        change
    }

    /// One minimum-compliance iteration: FE analysis, sensitivity filter
    /// and OC update.  Returns the compliance of the analysed design and
    /// the density change.
    pub fn step_compliance(&mut self, load: &SimpLoadCase) -> Result<SimpIterate> {
        let analysis = self.solve_compliance(load)?;
        let filtered = self.filter_sensitivities(&analysis.sensitivities);
        let change = self.oc_update(&filtered);
        Ok(SimpIterate {
            objective: analysis.compliance,
            change,
        })
    }

    /// Homogenise the current design as a periodic unit cell.
    ///
    /// Solves the three unit-strain load cases `ε = (1,0,0), (0,1,0),
    /// (0,0,1)` with periodic boundary conditions (one node fixed against
    /// rigid translation) and evaluates
    /// `C_H,ij = (1/|Y|) Σₑ Eₑ (u⁰ᵢ − χᵢ)ₑᵀ k₀ (u⁰ⱼ − χⱼ)ₑ` together with the
    /// self-adjoint sensitivities `∂C_H,ij/∂ρₑ`.
    pub fn homogenize(&self) -> Result<HomogenizedProperties> {
        self.validate()?;
        let n_nodes = self.nx * self.ny;
        let n_dof = 2 * n_nodes;
        // Fix node 0 (both DOFs) to remove the translational null space.
        let dof_map: Vec<Option<usize>> = (0..n_dof)
            .map(|d| if d < 2 { None } else { Some(d - 2) })
            .collect();
        let n_red = n_dof - 2;
        let ke = q4_plane_stress_stiffness(self.material_poisson);
        let edofs: Vec<[usize; 8]> = (0..self.ny)
            .flat_map(|iy| (0..self.nx).map(move |ix| (ix, iy)))
            .map(|(ix, iy)| self.periodic_element_dofs(ix, iy))
            .collect();
        let k = self.assemble_reduced(&ke, &edofs, &dof_map, n_red);

        // Element-local unit-strain displacement fields u⁰ = ε·x at the
        // local nodes (0,0), (a,0), (a,a), (0,a).
        let a = self.element_size;
        let local = [[0.0, 0.0], [a, 0.0], [a, a], [0.0, a]];
        let mut u0 = [[0.0_f64; 8]; 3];
        for (node, xy) in local.iter().enumerate() {
            u0[0][2 * node] = xy[0];
            u0[1][2 * node + 1] = xy[1];
            u0[2][2 * node] = 0.5 * xy[1];
            u0[2][2 * node + 1] = 0.5 * xy[0];
        }

        let mut chi: Vec<Vec<f64>> = Vec::with_capacity(3);
        for case in &u0 {
            let mut f = vec![0.0_f64; n_red];
            for (elem, dofs) in self.elements.iter().zip(edofs.iter()) {
                let e_mod = elem.simp_modulus(self.penalty, self.stiffness_floor);
                for (r, &gd) in dofs.iter().enumerate() {
                    if let Some(Some(rr)) = dof_map.get(gd) {
                        let fe: f64 = ke[r].iter().zip(case.iter()).map(|(k, u)| k * u).sum();
                        f[*rr] += e_mod * fe;
                    }
                }
            }
            let x_red = Self::pcg_solve(&k, &f)?;
            let mut full = vec![0.0_f64; n_dof];
            for (slot, map) in full.iter_mut().zip(dof_map.iter()) {
                if let Some(r) = map {
                    *slot = x_red[*r];
                }
            }
            chi.push(full);
        }

        let area = (self.nx * self.ny) as f64 * a * a;
        let mut c_h = [[0.0_f64; 3]; 3];
        let mut element_sensitivities = Vec::with_capacity(self.elements.len());
        for (elem, dofs) in self.elements.iter().zip(edofs.iter()) {
            let mut d = [[0.0_f64; 8]; 3];
            for (case, (d_case, u0_case)) in d.iter_mut().zip(u0.iter()).enumerate() {
                let chi_e = Self::element_vector(&chi[case], dofs);
                for ((dv, &u), &c) in d_case.iter_mut().zip(u0_case.iter()).zip(chi_e.iter()) {
                    *dv = u - c;
                }
            }
            let e_mod = elem.simp_modulus(self.penalty, self.stiffness_floor);
            let de_mod = elem.simp_modulus_derivative(self.penalty, self.stiffness_floor);
            let mut sens = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in i..3 {
                    let q = element_energy(&ke, &d[i], &d[j]) / area;
                    c_h[i][j] += e_mod * q;
                    sens[i][j] = de_mod * q;
                    sens[j][i] = sens[i][j];
                }
            }
            element_sensitivities.push(sens);
        }
        c_h[1][0] = c_h[0][1];
        c_h[2][0] = c_h[0][2];
        c_h[2][1] = c_h[1][2];
        let denom = c_h[0][0] + c_h[1][1];
        if denom.abs() < 1e-300 {
            return Err(Error::General("homogenised stiffness vanishes".into()));
        }
        Ok(HomogenizedProperties {
            c_h,
            poisson_ratio: 2.0 * c_h[0][1] / denom,
            element_sensitivities,
        })
    }

    /// Target-Poisson objective `(ν_H − target_poisson)²` and its gradient.
    pub fn poisson_objective(&self) -> Result<PoissonObjective> {
        let h = self.homogenize()?;
        let s = h.c_h[0][0] + h.c_h[1][1];
        let c12 = h.c_h[0][1];
        let nu = h.poisson_ratio;
        let residual = nu - self.target_poisson;
        let sensitivities = h
            .element_sensitivities
            .iter()
            .map(|dc| {
                let ds = dc[0][0] + dc[1][1];
                let dnu = 2.0 * (dc[0][1] * s - c12 * ds) / (s * s);
                2.0 * residual * dnu
            })
            .collect();
        Ok(PoissonObjective {
            objective: residual * residual,
            poisson_ratio: nu,
            sensitivities,
        })
    }

    /// Volume-preserving projected-gradient update.
    ///
    /// Takes a step of maximum size `move_limit` against `gradient` and
    /// projects onto `{ρ ∈ [ρ_min, 1]ⁿ, |ρ − ρ_old|∞ ≤ move_limit,
    /// mean(ρ) = volume_fraction}` (multiplier found by bisection).
    /// Returns the maximum density change.
    pub fn projected_gradient_update(&mut self, gradient: &[f64], move_limit: f64) -> Result<f64> {
        let n = self.elements.len();
        if gradient.len() != n || n == 0 {
            return Err(Error::InvalidParameter(format!(
                "gradient length {} does not match {} elements",
                gradient.len(),
                n
            )));
        }
        if !(move_limit.is_finite() && move_limit > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "move limit must be positive, got {move_limit}"
            )));
        }
        let g_max = gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        let alpha = if g_max > 0.0 { move_limit / g_max } else { 0.0 };
        let densities: Vec<f64> = self.elements.iter().map(|e| e.density).collect();
        let candidate = |lambda: f64, out: &mut Vec<f64>| -> f64 {
            out.clear();
            let mut vol = 0.0_f64;
            for (&rho, &g) in densities.iter().zip(gradient.iter()) {
                let lo = (rho - move_limit).max(MIN_DENSITY);
                let hi = (rho + move_limit).min(1.0);
                let r = (rho - alpha * g - lambda).clamp(lo.min(hi), hi);
                out.push(r);
                vol += r;
            }
            vol / n as f64
        };
        let mut rho_new = Vec::with_capacity(n);
        let (mut l1, mut l2) = (-3.0_f64, 3.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (l1 + l2);
            if candidate(mid, &mut rho_new) > self.volume_fraction {
                l1 = mid;
            } else {
                l2 = mid;
            }
            if l2 - l1 < 1e-14 {
                break;
            }
        }
        candidate(0.5 * (l1 + l2), &mut rho_new);
        let mut change = 0.0_f64;
        for (elem, &r) in self.elements.iter_mut().zip(rho_new.iter()) {
            change = change.max((r - elem.density).abs());
            elem.density = r;
        }
        Ok(change)
    }

    /// Seed the design with a soft circular inclusion of radius
    /// `min(nx, ny)·a / 3` at the cell centre (densities halved there, the
    /// removed material spread over the remaining elements so the volume
    /// fraction is unchanged), as in Andreassen et al. (2014).  A uniform design has an identically zero
    /// Poisson gradient by symmetry, so some seeding is required before
    /// target-Poisson optimisation.
    pub fn seed_central_inclusion(&mut self) {
        let a = self.element_size;
        let centre = [0.5 * self.nx as f64 * a, 0.5 * self.ny as f64 * a];
        let radius = self.nx.min(self.ny) as f64 * a / 3.0;
        let mut removed = 0.0_f64;
        let mut outside = Vec::new();
        for (idx, e) in self.elements.iter_mut().enumerate() {
            let d = ((e.pos[0] - centre[0]).powi(2) + (e.pos[1] - centre[1]).powi(2)).sqrt();
            if d < radius {
                let new_rho = (0.5 * e.density).max(MIN_DENSITY);
                removed += e.density - new_rho;
                e.density = new_rho;
            } else {
                outside.push(idx);
            }
        }
        // Redistribute the removed material outside the inclusion so that
        // the volume fraction is preserved (as far as the bound ρ ≤ 1 allows).
        let mut remaining = removed;
        for _ in 0..outside.len().max(1) {
            let free: Vec<usize> = outside
                .iter()
                .copied()
                .filter(|&i| self.elements[i].density < 1.0)
                .collect();
            if remaining <= 1e-15 || free.is_empty() {
                break;
            }
            let share = remaining / free.len() as f64;
            for i in free {
                let e = &mut self.elements[i];
                let add = share.min(1.0 - e.density);
                e.density += add;
                remaining -= add;
            }
        }
    }

    /// One target-Poisson iteration.
    ///
    /// If the design is exactly uniform it is first seeded with
    /// [`seed_central_inclusion`](Self::seed_central_inclusion).  A
    /// projected-gradient step with move limit `poisson_move_limit` is
    /// accepted only if it lowers `(ν_H − target)²`; otherwise the move
    /// limit is halved (up to eight times) and the design is left unchanged
    /// if no decrease is found.  Returns the objective of the resulting
    /// design and the density change.
    pub fn step_poisson(&mut self) -> Result<SimpIterate> {
        let (min_rho, max_rho) = self
            .elements
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), e| {
                (lo.min(e.density), hi.max(e.density))
            });
        if max_rho - min_rho < 1e-12 {
            self.seed_central_inclusion();
        }
        let current = self.poisson_objective()?;
        let saved: Vec<f64> = self.elements.iter().map(|e| e.density).collect();
        for _ in 0..8 {
            let change =
                self.projected_gradient_update(&current.sensitivities, self.poisson_move_limit)?;
            let trial = self.poisson_objective()?;
            if trial.objective < current.objective {
                self.poisson_move_limit = (self.poisson_move_limit * 1.2).min(0.2);
                self.iteration += 1;
                return Ok(SimpIterate {
                    objective: trial.objective,
                    change,
                });
            }
            for (e, &rho) in self.elements.iter_mut().zip(saved.iter()) {
                e.density = rho;
            }
            self.poisson_move_limit *= 0.5;
        }
        self.iteration += 1;
        Ok(SimpIterate {
            objective: current.objective,
            change: 0.0,
        })
    }
}
