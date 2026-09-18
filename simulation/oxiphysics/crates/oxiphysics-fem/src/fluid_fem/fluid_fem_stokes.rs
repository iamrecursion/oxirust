// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Classical Stokes and Navier-Stokes FEM solvers.
//!
//! Provides Taylor-Hood element assembly, lid-driven cavity setup,
//! and a 1-D linearized implicit Navier-Stokes FEM driver:
//!
//! - []: Stokes flow FEM solver (Taylor-Hood elements)
//! - []: P1 triangular stiffness matrix
//! - []: pressure-velocity coupling matrix
//! - []: benchmark cavity viscous assembly
//! - []: 1-D linearized implicit NS FEM driver
//! - []: RHS for projection method pressure Poisson

use crate::parallel_solver::{CsrMatrix, ParallelPcgSolver};
use crate::{Error, Result};

/// Relative residual tolerance of the conjugate-gradient solve in
/// [`NavierStokesFEM::step_euler`].
const NS_PCG_RELATIVE_TOLERANCE: f64 = 1e-10;

// ============================================================================
// Stokes FEM
// ============================================================================

/// Stokes flow FEM solver using Taylor-Hood elements.
///
/// Solves the incompressible Stokes equations:
/// −μ∇²u + ∇p = f,  ∇·u = 0
pub struct StokesFEM {
    /// Number of velocity nodes.
    pub n_vel_nodes: usize,
    /// Number of pressure nodes.
    pub n_press_nodes: usize,
    /// Dynamic viscosity (Pa·s).
    pub viscosity: f64,
}

impl StokesFEM {
    /// Create a new Stokes FEM solver.
    pub fn new(n_vel_nodes: usize, n_press_nodes: usize, viscosity: f64) -> Self {
        Self {
            n_vel_nodes,
            n_press_nodes,
            viscosity,
        }
    }

    /// Total degrees of freedom (2 velocity components per velocity node + 1 pressure per pressure node).
    pub fn total_dofs(&self) -> usize {
        2 * self.n_vel_nodes + self.n_press_nodes
    }
}

/// Compute the Taylor-Hood element stiffness matrix for a triangular element.
///
/// Given element coordinates `coords` (3 vertices) and viscosity `viscosity`,
/// returns a 6×6 matrix representing the viscous contribution from the
/// velocity DOFs (u₁,v₁, u₂,v₂, u₃,v₃).
pub fn stokes_element_matrix(coords: &[[f64; 2]; 3], viscosity: f64) -> [[f64; 6]; 6] {
    // Compute shape function derivatives (constant for linear triangle)
    let x1 = coords[0][0];
    let y1 = coords[0][1];
    let x2 = coords[1][0];
    let y2 = coords[1][1];
    let x3 = coords[2][0];
    let y3 = coords[2][1];
    let area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    let area = area2.abs() / 2.0;
    if area < 1e-300 {
        return [[0.0; 6]; 6];
    }
    // dN/dx and dN/dy for each of the 3 P1 shape functions
    let dndx = [(y2 - y3) / area2, (y3 - y1) / area2, (y1 - y2) / area2];
    let dndy = [(x3 - x2) / area2, (x1 - x3) / area2, (x2 - x1) / area2];
    let mut ke = [[0.0f64; 6]; 6];
    // Assemble using B^T * C * B * area for plane-stress incompressible
    // DOF ordering: (u1, v1, u2, v2, u3, v3) → rows/cols 0..5
    for a in 0..3 {
        for b in 0..3 {
            // ε_xx contribution: dNa/dx * dNb/dx
            let kuu = viscosity * (dndx[a] * dndx[b] + dndy[a] * dndy[b]) * area;
            let row_u = 2 * a;
            let col_u = 2 * b;
            ke[row_u][col_u] += kuu;
            ke[row_u + 1][col_u + 1] += kuu;
        }
    }
    ke
}

/// Pressure-velocity coupling matrix for a triangular Stokes element.
///
/// Returns a 6×3 matrix B such that the coupling block B * p couples
/// pressure DOFs (P1) to velocity DOFs (P2-like, treated as P1 here).
pub fn stokes_pressure_gradient(coords: &[[f64; 2]; 3]) -> [[f64; 3]; 6] {
    let x1 = coords[0][0];
    let y1 = coords[0][1];
    let x2 = coords[1][0];
    let y2 = coords[1][1];
    let x3 = coords[2][0];
    let y3 = coords[2][1];
    let area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    let area = area2.abs() / 2.0;
    if area < 1e-300 {
        return [[0.0; 3]; 6];
    }
    let dndx = [(y2 - y3) / area2, (y3 - y1) / area2, (y1 - y2) / area2];
    let dndy = [(x3 - x2) / area2, (x1 - x3) / area2, (x2 - x1) / area2];
    let mut bp = [[0.0f64; 3]; 6];
    // Row 2a   (u-component): -dNa/dx * Nb_area
    // Row 2a+1 (v-component): -dNa/dy * Nb_area
    for a in 0..3 {
        for (b, _) in (0..3usize).enumerate() {
            // Integrate N_b over triangle: area/3 for linear elements
            let nb_int = area / 3.0;
            bp[2 * a][b] = -dndx[a] * nb_int;
            bp[2 * a + 1][b] = -dndy[a] * nb_int;
        }
    }
    bp
}

/// Set up the Stokes viscous system for the lid-driven cavity benchmark.
///
/// Assembles the viscous stiffness matrix `∫ μ ∇u · ∇v dΩ` by splitting each
/// Q4 cell of the n×n uniform grid into two P1 triangles, calling
/// `stokes_element_matrix()` per triangle, and scattering the 6×6 element
/// contributions into the global dense `(2*n_nodes) × (2*n_nodes)` matrix.
///
/// Boundary conditions are applied by zeroing rows/columns for Dirichlet DOFs and
/// putting the prescribed values into the RHS via the elimination method:
/// - Bottom, left, right walls: u = v = 0.
/// - Top lid (y = 1):           u = 1, v = 0.
///
/// Returns `(matrix, rhs)` where `matrix` is `(2*n_nodes) × (2*n_nodes)` and
/// `rhs` is `2*n_nodes`. `n_nodes = (n+1)^2`.
pub fn lid_driven_cavity_setup(n: usize, viscosity: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_nodes = (n + 1) * (n + 1);
    let n_dofs = 2 * n_nodes;
    let h = 1.0 / n as f64;

    let mut mat = vec![vec![0.0f64; n_dofs]; n_dofs];
    let mut rhs = vec![0.0f64; n_dofs];

    // Assemble viscous stiffness: split each quad into 2 triangles
    // Nodes of quad (iy, ix) in row-major order:
    //   n00 = iy*(n+1)+ix,  n10 = iy*(n+1)+(ix+1)
    //   n01 = (iy+1)*(n+1)+ix,  n11 = (iy+1)*(n+1)+(ix+1)
    for iy in 0..n {
        for ix in 0..n {
            let n00 = iy * (n + 1) + ix;
            let n10 = iy * (n + 1) + (ix + 1);
            let n01 = (iy + 1) * (n + 1) + ix;
            let n11 = (iy + 1) * (n + 1) + (ix + 1);

            let x0 = ix as f64 * h;
            let y0 = iy as f64 * h;

            // Triangle 1: n00, n10, n11 (lower-right)
            let coords1 = [[x0, y0], [x0 + h, y0], [x0 + h, y0 + h]];
            let ke1 = stokes_element_matrix(&coords1, viscosity);
            let nodes1 = [n00, n10, n11];
            scatter_element_to_dense(&mut mat, &ke1, &nodes1);

            // Triangle 2: n00, n11, n01 (upper-left)
            let coords2 = [[x0, y0], [x0 + h, y0 + h], [x0, y0 + h]];
            let ke2 = stokes_element_matrix(&coords2, viscosity);
            let nodes2 = [n00, n11, n01];
            scatter_element_to_dense(&mut mat, &ke2, &nodes2);
        }
    }

    // Identify Dirichlet DOFs
    let mut dirichlet = vec![(0usize, 0.0f64); 0]; // (global_dof, prescribed_value)

    // Walls: bottom (iy=0), left (ix=0), right (ix=n)  →  u=v=0
    for ix in 0..=n {
        let node = ix; // iy=0
        dirichlet.push((2 * node, 0.0));
        dirichlet.push((2 * node + 1, 0.0));
    }
    for iy in 1..n {
        // left wall ix=0
        let node_l = iy * (n + 1);
        dirichlet.push((2 * node_l, 0.0));
        dirichlet.push((2 * node_l + 1, 0.0));
        // right wall ix=n
        let node_r = iy * (n + 1) + n;
        dirichlet.push((2 * node_r, 0.0));
        dirichlet.push((2 * node_r + 1, 0.0));
    }
    // Lid: top row (iy=n) → u=1, v=0
    for ix in 0..=n {
        let node = n * (n + 1) + ix;
        dirichlet.push((2 * node, 1.0)); // u = 1
        dirichlet.push((2 * node + 1, 0.0)); // v = 0
    }

    // Apply Dirichlet BCs by elimination
    for &(dof, val) in &dirichlet {
        // RHS correction: rhs[j] -= mat[j][dof] * val  for all free j
        for j in 0..n_dofs {
            rhs[j] -= mat[j][dof] * val;
        }
        // Zero row and column, set diagonal to 1, rhs to val
        for (j, _) in (0..n_dofs).enumerate() {
            mat[dof][j] = 0.0;
            mat[j][dof] = 0.0;
        }
        mat[dof][dof] = 1.0;
        rhs[dof] = val;
    }

    (mat, rhs)
}

/// Scatter a 6×6 P1 triangular element stiffness matrix into the global dense
/// matrix using the 3-node DOF mapping (u_i = 2*node, v_i = 2*node+1).
fn scatter_element_to_dense(mat: &mut [Vec<f64>], ke: &[[f64; 6]; 6], nodes: &[usize; 3]) {
    for a in 0..3 {
        let row_u = 2 * nodes[a];
        let row_v = 2 * nodes[a] + 1;
        for b in 0..3 {
            let col_u = 2 * nodes[b];
            let col_v = 2 * nodes[b] + 1;
            mat[row_u][col_u] += ke[2 * a][2 * b];
            mat[row_u][col_v] += ke[2 * a][2 * b + 1];
            mat[row_v][col_u] += ke[2 * a + 1][2 * b];
            mat[row_v][col_v] += ke[2 * a + 1][2 * b + 1];
        }
    }
}

/// Compute the stream function from a 2D velocity field on a regular grid.
///
/// Integrates ψ by: dψ/dy = u, dψ/dx = -v using trapezoidal rule.
/// Grid is `nx × ny` with spacing `dx`.
pub fn stream_function_from_velocity(
    ux: &[f64],
    _uy: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
) -> Vec<f64> {
    let n = nx * ny;
    let mut psi = vec![0.0f64; n];
    if n == 0 || dx <= 0.0 {
        return psi;
    }
    // Integrate along each column from bottom to top using ux: dψ/dy = ux
    for i in 0..nx {
        psi[i] = 0.0; // bottom boundary
        for j in 1..ny {
            let idx_prev = (j - 1) * nx + i;
            let idx_curr = j * nx + i;
            if idx_curr < n && idx_prev < ux.len() {
                psi[idx_curr] = psi[idx_prev] + dx * ux[idx_prev];
            }
        }
    }
    psi
}

/// Compute the 2D vorticity field ω = ∂v/∂x − ∂u/∂y on a regular grid.
///
/// Uses central differences on the interior; one-sided differences on boundaries.
pub fn vorticity_2d(ux: &[f64], uy: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> Vec<f64> {
    let n = nx * ny;
    let mut omega = vec![0.0f64; n];
    if n == 0 {
        return omega;
    }
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            // dv/dx
            let dv_dx = if i == 0 {
                let r = j * nx + (i + 1).min(nx - 1);
                (uy[r] - uy[idx]) / dx
            } else if i == nx - 1 {
                let l = j * nx + i - 1;
                (uy[idx] - uy[l]) / dx
            } else {
                let r = j * nx + i + 1;
                let l = j * nx + i - 1;
                (uy[r] - uy[l]) / (2.0 * dx)
            };
            // du/dy
            let du_dy = if j == 0 {
                let u = (j + 1).min(ny - 1) * nx + i;
                (ux[u] - ux[idx]) / dy
            } else if j == ny - 1 {
                let d = (j - 1) * nx + i;
                (ux[idx] - ux[d]) / dy
            } else {
                let u = (j + 1) * nx + i;
                let d = (j - 1) * nx + i;
                (ux[u] - ux[d]) / (2.0 * dy)
            };
            omega[idx] = dv_dx - du_dy;
        }
    }
    omega
}

/// Compute the Reynolds number Re = ρ v L / μ.
///
/// - `rho`: fluid density (kg/m³)
/// - `v`: characteristic velocity (m/s)
/// - `l`: characteristic length (m)
/// - `mu`: dynamic viscosity (Pa·s)
pub fn reynolds_number(rho: f64, v: f64, l: f64, mu: f64) -> f64 {
    if mu.abs() < 1e-300 {
        return 0.0;
    }
    rho * v * l / mu
}

/// Compute the Stokes number Stk = ρ_p d_p² v / (18 μ L).
///
/// - `rho_p`: particle density (kg/m³)
/// - `d_p`: particle diameter (m)
/// - `v`: characteristic flow velocity (m/s)
/// - `mu`: dynamic viscosity (Pa·s)
/// - `l`: characteristic length (m)
pub fn stokes_number(rho_p: f64, d_p: f64, v: f64, mu: f64, l: f64) -> f64 {
    if mu.abs() < 1e-300 || l.abs() < 1e-300 {
        return 0.0;
    }
    rho_p * d_p * d_p * v / (18.0 * mu * l)
}

/// 1-D linearized implicit Navier-Stokes FEM driver on a uniform line mesh.
///
/// Models the 1-D convection-diffusion momentum equation:
///   ρ ∂u/∂t + ρ u⁰ ∂u/∂x − μ ∂²u/∂x² = −∂p/∂x + f
///
/// Discretized with linear Galerkin elements on `n_nodes` evenly-spaced nodes
/// (spacing `dx = 1/(n_nodes-1)`) and a lumped mass matrix.
/// The implicit time-step solves:
///   (M/dt + K_visc) u^{n+1} = M/dt u^n − C(u^n) u^n − ∇p + f
/// using `ParallelPcgSolver` on the CSR tridiagonal system.  Every term of this
/// system is an element *integral* and therefore carries one factor `dx`:
/// the lumped mass is `ρ dx`, the viscous stiffness `∫ μ N'ᵢ N'ⱼ dx` has
/// diagonal `2μ/dx` and off-diagonal `−μ/dx`, and the loads are multiplied by
/// `dx`.  Divided by `dx` each row is the usual finite-difference stencil
/// `ρ (uⁿ⁺¹ − uⁿ)/dt = μ (uᵢ₋₁ − 2uᵢ + uᵢ₊₁)/dx² − C(uⁿ) + f − ∂p/∂x`.
///
/// Boundary conditions: Dirichlet u=0 at both ends (indices 0 and n_nodes−1).
pub struct NavierStokesFEM {
    /// Number of nodes on the 1-D mesh.
    pub n_nodes: usize,
    /// Reynolds number Re = ρ U L / μ (used for reference; set μ = 1/Re).
    pub re: f64,
    /// Current simulation time (s).
    pub time: f64,
    /// Nodal velocity DOF vector (length n_nodes, one scalar per node).
    pub velocity: Vec<f64>,
    /// Fluid density ρ (kg/m³).
    pub density: f64,
    /// Dynamic viscosity μ = 1/Re (Pa·s) for unit ρ, U, L.
    pub viscosity: f64,
    /// Uniform node spacing dx = 1/(n_nodes−1).
    pub dx: f64,
    /// Uniform pressure gradient driving force −∂p/∂x.
    pub pressure_grad: f64,
    /// Body force per unit volume (N/m³).
    pub body_force: f64,
}

impl NavierStokesFEM {
    /// Create a new 1-D Navier-Stokes FEM driver.
    ///
    /// `n_nodes` is the number of uniformly-spaced nodes;
    /// `re` is the Reynolds number (μ = 1/Re is derived from it).
    pub fn new(n_nodes: usize, re: f64) -> Self {
        let dx = if n_nodes > 1 {
            1.0 / (n_nodes - 1) as f64
        } else {
            1.0
        };
        let viscosity = 1.0 / re.max(1e-300);
        Self {
            n_nodes,
            re,
            time: 0.0,
            velocity: vec![0.0; n_nodes],
            density: 1.0,
            viscosity,
            dx,
            pressure_grad: 0.0,
            body_force: 0.0,
        }
    }

    /// Build the CSR tridiagonal system `(M/dt + K_visc)` and its right-hand side
    /// (which carries the explicit convection, pressure-gradient and body-force
    /// loads).  Requires `n_nodes >= 3` and `dt > 0`.
    ///
    /// All entries are element integrals, i.e. they carry one factor `dx`:
    ///
    /// - Lumped mass: m_i = ρ dx at the interior nodes.
    /// - Viscous stiffness: Galerkin `∫ μ N'ᵢ N'ⱼ dx` of linear elements →
    ///   tridiagonal with diagonal 2μ/dx and off-diagonal −μ/dx (the
    ///   finite-difference stencil μ(−1, 2, −1)/dx² times `dx`).
    /// - Convection: first-order upwind ρ u⁰ ∂u/∂x, multiplied by `dx`.
    /// - Loads: (−∂p/∂x + f) dx.
    ///
    /// Dirichlet rows (i=0 and i=n_nodes−1) are identity rows with zero
    /// right-hand side.  The Dirichlet value is zero, so the boundary columns
    /// are dropped from the interior rows; this keeps the whole matrix
    /// symmetric positive definite, as the conjugate-gradient solver requires.
    /// The velocity at the Dirichlet nodes is taken as zero everywhere, also in
    /// the explicit convection term.
    fn build_system_matrix(&self, dt: f64) -> (CsrMatrix, Vec<f64>) {
        let n = self.n_nodes;
        let dx = self.dx;
        let rho = self.density;
        let inv_dt = 1.0 / dt;
        // Nodal velocity with the homogeneous Dirichlet values imposed.
        let u = |j: usize| {
            if j == 0 || j + 1 == n {
                0.0
            } else {
                self.velocity[j]
            }
        };
        // Galerkin viscous stiffness ∫ μ N'N' dx of a linear element: μ/dx.
        let k_c = self.viscosity / dx;
        let m_i = rho * dx;

        // Tridiagonal CSR: interior rows have up to 3 non-zeros, boundary rows 1.
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::with_capacity(3 * n);
        let mut values = Vec::with_capacity(3 * n);
        let mut rhs = vec![0.0f64; n];
        row_offsets.push(0);

        for (i, rhs_i) in rhs.iter_mut().enumerate() {
            if i == 0 || i == n - 1 {
                // Dirichlet: identity row, u = 0 at the boundaries (rhs stays 0).
                col_indices.push(i);
                values.push(1.0);
            } else {
                let u_i = u(i);
                // First-order upwind for convection: ρ u⁰ ∂u/∂x
                let conv = if u_i >= 0.0 {
                    rho * u_i * (u_i - u(i - 1)) / dx
                } else {
                    rho * u_i * (u(i + 1) - u_i) / dx
                };
                // M/dt u^n - C(u^n) dx + (-∂p/∂x + f) dx
                *rhs_i =
                    m_i * inv_dt * u_i - conv * dx + (-self.pressure_grad + self.body_force) * dx;
                if i > 1 {
                    col_indices.push(i - 1);
                    values.push(-k_c);
                }
                col_indices.push(i);
                values.push(m_i * inv_dt + 2.0 * k_c);
                if i + 1 < n - 1 {
                    col_indices.push(i + 1);
                    values.push(-k_c);
                }
            }
            row_offsets.push(col_indices.len());
        }

        let mat = CsrMatrix {
            nrows: n,
            ncols: n,
            row_offsets,
            col_indices,
            values,
        };
        (mat, rhs)
    }

    /// Advance time by `dt` seconds using the linearized implicit Euler step.
    ///
    /// Solves `(M/dt + K_visc) u^{n+1} = rhs` via `ParallelPcgSolver`.  A mesh
    /// with fewer than three nodes has no interior unknowns (both nodes are
    /// Dirichlet): the step then only zeroes the boundary values and advances
    /// the time.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidInput`] if `dt` is not finite and positive, or if
    ///   `velocity.len()` differs from `n_nodes`.
    /// - [`Error::NotConverged`] if the conjugate-gradient solve does not reach
    ///   its tolerance, for instance because the velocity field contains
    ///   non-finite values.  The reported residual is relative to the norm of
    ///   the right-hand side.
    ///
    /// On error `velocity` and `time` are left unchanged.
    pub fn step_euler(&mut self, dt: f64) -> Result<()> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::InvalidInput(format!(
                "time step must be finite and positive, got {dt}"
            )));
        }
        let n = self.n_nodes;
        if self.velocity.len() != n {
            return Err(Error::InvalidInput(format!(
                "velocity has {} entries for {} nodes",
                self.velocity.len(),
                n
            )));
        }
        if n < 3 {
            self.velocity.iter_mut().for_each(|u| *u = 0.0);
            self.time += dt;
            return Ok(());
        }
        let (mat, rhs) = self.build_system_matrix(dt);
        // Warm start from the current velocity with the Dirichlet values imposed.
        let mut u_new = self.velocity.clone();
        u_new[0] = 0.0;
        u_new[n - 1] = 0.0;
        // Jacobi-preconditioned CG needs on the order of `n` iterations for a
        // diffusion-dominated tridiagonal system.  The default absolute residual
        // floor of 1e-14 would turn the relative criterion into a much looser
        // one for small velocities, so only guard against an exactly vanishing
        // residual.
        let solver = ParallelPcgSolver {
            max_iterations: (2 * n).max(500),
            tolerance: NS_PCG_RELATIVE_TOLERANCE,
            abs_tolerance: f64::MIN_POSITIVE,
        };
        let stats = solver.solve(&mat, &rhs, &mut u_new);
        if !stats.converged {
            let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
            return Err(Error::NotConverged {
                solver: "ParallelPcgSolver".to_string(),
                iterations: stats.iterations,
                residual: stats.residual_norm / rhs_norm.max(f64::MIN_POSITIVE),
            });
        }
        self.velocity = u_new;
        self.time += dt;
        Ok(())
    }

    /// Total kinetic energy: ½ ρ dx ∑ u_i² (trapezoidal-rule integration).
    pub fn kinetic_energy(&self) -> f64 {
        let rho = self.density;
        let dx = self.dx;
        let n = self.n_nodes;
        let mut ke = 0.0;
        for i in 0..n {
            let w = if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
            ke += w * rho * dx * self.velocity[i] * self.velocity[i];
        }
        0.5 * ke
    }

    /// Current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }
}

/// Right-hand side for the pressure Poisson equation in the projection method.
///
/// RHS_i = div_u_i / dt  (used in fractional-step methods).
pub fn pressure_poisson_rhs(div_u: &[f64], dt: f64) -> Vec<f64> {
    if dt.abs() < 1e-300 {
        return vec![0.0; div_u.len()];
    }
    div_u.iter().map(|&d| d / dt).collect()
}

// ============================================================================
// Extended tests for the new Stokes/NS functions
// ============================================================================

#[cfg(test)]
mod stokes_tests {
    use super::*;

    #[test]
    fn test_stokes_fem_new() {
        let s = StokesFEM::new(6, 3, 1e-3);
        assert_eq!(s.n_vel_nodes, 6);
        assert_eq!(s.n_press_nodes, 3);
        assert!((s.viscosity - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn test_stokes_fem_total_dofs() {
        let s = StokesFEM::new(6, 3, 1e-3);
        assert_eq!(s.total_dofs(), 15); // 2*6 + 3
    }

    #[test]
    fn test_stokes_element_matrix_symmetry() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ke = stokes_element_matrix(&coords, 1.0);
        for (i, row) in ke.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    (val - ke[j][i]).abs() < 1e-12,
                    "not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_stokes_element_matrix_zero_viscosity() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ke = stokes_element_matrix(&coords, 0.0);
        for row in &ke {
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn test_stokes_element_matrix_degenerate() {
        // Degenerate triangle (zero area)
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]];
        let ke = stokes_element_matrix(&coords, 1.0);
        for row in &ke {
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn test_stokes_element_matrix_viscosity_scales() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let ke1 = stokes_element_matrix(&coords, 1.0);
        let ke2 = stokes_element_matrix(&coords, 2.0);
        assert!((ke2[0][0] - 2.0 * ke1[0][0]).abs() < 1e-12);
    }

    #[test]
    fn test_stokes_pressure_gradient_dimensions() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let bp = stokes_pressure_gradient(&coords);
        assert_eq!(bp.len(), 6);
        assert_eq!(bp[0].len(), 3);
    }

    #[test]
    fn test_stokes_pressure_gradient_degenerate() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]];
        let bp = stokes_pressure_gradient(&coords);
        for row in &bp {
            for &v in row {
                assert_eq!(v, 0.0);
            }
        }
    }

    #[test]
    fn test_lid_driven_cavity_setup_sizes() {
        let (mat, rhs) = lid_driven_cavity_setup(4, 1e-3);
        let n_nodes = 5 * 5; // (n+1)^2
        assert_eq!(mat.len(), 2 * n_nodes);
        assert_eq!(rhs.len(), 2 * n_nodes);
    }

    #[test]
    fn test_lid_driven_cavity_rhs_lid() {
        let (_mat, rhs) = lid_driven_cavity_setup(4, 1e-3);
        // Top-row nodes should have rhs entry = 1 (u-component of lid)
        let n = 4;
        let n1 = n + 1;
        let any_lid_nonzero = (0..n1).any(|i| {
            let lid_node = n * n1 + i;
            if lid_node < n1 * n1 && 2 * lid_node < rhs.len() {
                rhs[2 * lid_node] > 0.5
            } else {
                false
            }
        });
        assert!(any_lid_nonzero);
    }

    #[test]
    fn test_stream_function_from_velocity_size() {
        let nx = 5;
        let ny = 4;
        let ux = vec![1.0; nx * ny];
        let uy = vec![0.0; nx * ny];
        let psi = stream_function_from_velocity(&ux, &uy, nx, ny, 0.1);
        assert_eq!(psi.len(), nx * ny);
    }

    #[test]
    fn test_stream_function_from_velocity_zero() {
        let nx = 3;
        let ny = 3;
        let ux = vec![0.0; nx * ny];
        let uy = vec![0.0; nx * ny];
        let psi = stream_function_from_velocity(&ux, &uy, nx, ny, 0.1);
        for v in &psi {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_stream_function_from_velocity_integrates() {
        // Uniform ux=1 → ψ increases with j
        let nx = 2;
        let ny = 3;
        let ux = vec![1.0; nx * ny];
        let uy = vec![0.0; nx * ny];
        let psi = stream_function_from_velocity(&ux, &uy, nx, ny, 0.5);
        // psi at j=1 > psi at j=0 for same column
        assert!(psi[nx] > psi[0]);
    }

    #[test]
    fn test_vorticity_2d_size() {
        let nx = 5;
        let ny = 4;
        let ux = vec![0.0; nx * ny];
        let uy = vec![0.0; nx * ny];
        let omega = vorticity_2d(&ux, &uy, nx, ny, 0.1, 0.1);
        assert_eq!(omega.len(), nx * ny);
    }

    #[test]
    fn test_vorticity_2d_zero_field() {
        let nx = 4;
        let ny = 4;
        let ux = vec![0.0; nx * ny];
        let uy = vec![0.0; nx * ny];
        let omega = vorticity_2d(&ux, &uy, nx, ny, 0.1, 0.1);
        for v in &omega {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn test_vorticity_2d_solid_rotation() {
        // u = -y, v = x → ω = ∂v/∂x - ∂u/∂y = 1 - (-1) = 2
        let nx = 5;
        let ny = 5;
        let dx = 0.25;
        let dy = 0.25;
        let mut ux = vec![0.0; nx * ny];
        let mut uy = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                ux[j * nx + i] = -y;
                uy[j * nx + i] = x;
            }
        }
        let omega = vorticity_2d(&ux, &uy, nx, ny, dx, dy);
        // Interior points should be close to 2.0
        let idx = 2 * nx + 2;
        assert!((omega[idx] - 2.0).abs() < 0.1, "omega={}", omega[idx]);
    }

    #[test]
    fn test_reynolds_number_formula() {
        let re = reynolds_number(1000.0, 1.0, 0.1, 0.001);
        assert!((re - 1e5).abs() < 1.0);
    }

    #[test]
    fn test_reynolds_number_zero_viscosity() {
        let re = reynolds_number(1000.0, 1.0, 0.1, 0.0);
        assert_eq!(re, 0.0);
    }

    #[test]
    fn test_reynolds_number_proportional() {
        let re1 = reynolds_number(1.0, 1.0, 1.0, 1.0);
        let re2 = reynolds_number(2.0, 1.0, 1.0, 1.0);
        assert!((re2 - 2.0 * re1).abs() < 1e-12);
    }

    #[test]
    fn test_stokes_number_formula() {
        let stk = stokes_number(1000.0, 1e-4, 1.0, 1e-3, 0.1);
        let expected = 1000.0 * 1e-8 * 1.0 / (18.0 * 1e-3 * 0.1);
        assert!((stk - expected).abs() < 1e-14);
    }

    #[test]
    fn test_stokes_number_zero_viscosity() {
        let stk = stokes_number(1000.0, 1e-4, 1.0, 0.0, 0.1);
        assert_eq!(stk, 0.0);
    }

    #[test]
    fn test_stokes_number_zero_length() {
        let stk = stokes_number(1000.0, 1e-4, 1.0, 1e-3, 0.0);
        assert_eq!(stk, 0.0);
    }

    #[test]
    fn test_ns_fem_new() {
        let ns = NavierStokesFEM::new(10, 100.0);
        assert_eq!(ns.n_nodes, 10);
        assert_eq!(ns.velocity.len(), 10);
        assert_eq!(ns.time(), 0.0);
    }

    #[test]
    fn test_ns_fem_step_advances_time() {
        let mut ns = NavierStokesFEM::new(4, 100.0);
        ns.step_euler(0.01).expect("implicit Euler step");
        assert!((ns.time() - 0.01).abs() < 1e-14);
    }

    #[test]
    fn test_ns_fem_step_multiple() {
        let mut ns = NavierStokesFEM::new(4, 100.0);
        ns.step_euler(0.01).expect("implicit Euler step");
        ns.step_euler(0.01).expect("implicit Euler step");
        assert!((ns.time() - 0.02).abs() < 1e-13);
    }

    #[test]
    fn test_ns_fem_kinetic_energy_zero() {
        let ns = NavierStokesFEM::new(4, 100.0);
        assert_eq!(ns.kinetic_energy(), 0.0);
    }

    #[test]
    fn test_ns_fem_kinetic_energy_nonzero() {
        let mut ns = NavierStokesFEM::new(4, 100.0);
        ns.velocity = vec![1.0; 4];
        assert!(ns.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_pressure_poisson_rhs_size() {
        let div_u = vec![0.1, 0.2, 0.3];
        let rhs = pressure_poisson_rhs(&div_u, 0.01);
        assert_eq!(rhs.len(), 3);
    }

    #[test]
    fn test_pressure_poisson_rhs_formula() {
        let div_u = vec![1.0, 2.0];
        let rhs = pressure_poisson_rhs(&div_u, 0.5);
        assert!((rhs[0] - 2.0).abs() < 1e-12);
        assert!((rhs[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_pressure_poisson_rhs_zero_dt() {
        let div_u = vec![1.0, 2.0];
        let rhs = pressure_poisson_rhs(&div_u, 0.0);
        for v in &rhs {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_vorticity_2d_different_dx_dy() {
        let nx = 3;
        let ny = 3;
        let ux = vec![0.0; nx * ny];
        let uy = vec![1.0; nx * ny]; // uniform v = 1 → dv/dx = 0
        let omega = vorticity_2d(&ux, &uy, nx, ny, 0.1, 0.2);
        // All vorticity components should be zero for uniform field
        for v in &omega {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_stokes_fem_viscosity_stored() {
        let s = StokesFEM::new(3, 3, 0.42);
        assert!((s.viscosity - 0.42).abs() < 1e-15);
    }

    // ── B2: lid_driven_cavity_setup viscous assembly tests ────────────────

    #[test]
    fn test_lid_cavity_matrix_nonzero() {
        // With proper assembly, the system matrix should have non-zero off-diagonal entries
        let (mat, _rhs) = lid_driven_cavity_setup(4, 1e-2);
        let n_nodes = 5 * 5;
        let n_dofs = 2 * n_nodes;
        assert_eq!(mat.len(), n_dofs);
        // At least some interior entries must be non-zero
        let nonzero_count = mat
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v.abs() > 1e-20)
            .count();
        assert!(
            nonzero_count > n_dofs, // must have off-diagonal terms
            "Expected off-diagonal non-zeros, got {nonzero_count}"
        );
    }

    #[test]
    fn test_lid_cavity_rhs_lid_nodes() {
        // Lid nodes (top row, u-DOF) must have rhs = 1 after BC application
        let n = 3;
        let (_mat, rhs) = lid_driven_cavity_setup(n, 1e-3);
        let n_nodes = (n + 1) * (n + 1);
        // Top row: iy=n → node = n*(n+1)+ix for ix in 0..=n
        for ix in 0..=n {
            let lid_node = n * (n + 1) + ix;
            assert!(
                lid_node < n_nodes,
                "lid_node {lid_node} out of range {n_nodes}"
            );
            let u_dof = 2 * lid_node;
            assert!(
                (rhs[u_dof] - 1.0).abs() < 1e-10,
                "Lid u-DOF {u_dof}: expected rhs=1, got {}",
                rhs[u_dof]
            );
        }
    }

    #[test]
    fn test_lid_cavity_boundary_rhs_zero_walls() {
        // Wall DOFs (non-lid boundary) must have rhs = 0
        let n = 3;
        let (_mat, rhs) = lid_driven_cavity_setup(n, 1e-3);
        // Bottom row: iy=0
        for ix in 0..=n {
            let node = ix;
            assert!(
                rhs[2 * node].abs() < 1e-10,
                "Bottom u-DOF {}: expected 0, got {}",
                2 * node,
                rhs[2 * node]
            );
        }
    }

    // ── B3: NavierStokesFEM implicit step tests ───────────────────────────

    #[test]
    fn test_ns_fem_velocity_len_equals_n_nodes() {
        let ns = NavierStokesFEM::new(8, 50.0);
        assert_eq!(
            ns.velocity.len(),
            ns.n_nodes,
            "velocity len must equal n_nodes"
        );
    }

    #[test]
    fn test_ns_fem_step_preserves_boundary_conditions() {
        // Dirichlet BCs: u[0] = u[n-1] = 0 after any step
        let mut ns = NavierStokesFEM::new(6, 10.0);
        // Give interior nodes some velocity
        for i in 1..5 {
            ns.velocity[i] = 1.0;
        }
        ns.step_euler(0.01).expect("implicit Euler step");
        assert!(
            ns.velocity[0].abs() < 1e-10,
            "u[0] should remain 0, got {}",
            ns.velocity[0]
        );
        assert!(
            ns.velocity[5].abs() < 1e-10,
            "u[n-1] should remain 0, got {}",
            ns.velocity[5]
        );
    }

    #[test]
    fn test_ns_fem_viscous_dissipation() {
        // Interior velocity should decay over time due to viscous dissipation
        let mut ns = NavierStokesFEM::new(5, 1.0); // low Re → high viscosity
        for i in 1..4 {
            ns.velocity[i] = 1.0;
        }
        let ke_before = ns.kinetic_energy();
        for _ in 0..20 {
            ns.step_euler(0.01).expect("implicit Euler step");
        }
        let ke_after = ns.kinetic_energy();
        assert!(
            ke_after < ke_before,
            "Kinetic energy should decrease due to viscosity: before={ke_before}, after={ke_after}"
        );
    }

    #[test]
    fn test_ns_fem_zero_initial_stays_zero() {
        // Zero initial velocity with no forcing → stays zero
        let mut ns = NavierStokesFEM::new(4, 100.0);
        ns.step_euler(0.01).expect("implicit Euler step");
        for (i, &v) in ns.velocity.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "velocity[{i}] should remain 0 without forcing, got {v}"
            );
        }
    }

    // ── F1183: Galerkin scaling of the 1-D viscous term, solver status ─────

    /// Projection of the nodal velocity onto the first sine mode of `[0, 1]`.
    fn sine_mode_amplitude(ns: &NavierStokesFEM) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &u) in ns.velocity.iter().enumerate() {
            let s = (std::f64::consts::PI * i as f64 * ns.dx).sin();
            num += u * s;
            den += s * s;
        }
        num / den
    }

    #[test]
    fn test_ns_fem_system_matrix_uses_galerkin_scaling() {
        // mu = 1/Re = 0.25, rho = 1, dx = 0.2, dt = 0.01.
        let ns = NavierStokesFEM::new(6, 4.0);
        let (mat, _rhs) = ns.build_system_matrix(0.01);
        let dense = |i: usize, j: usize| -> f64 {
            let start = mat.row_offsets[i];
            let end = mat.row_offsets[i + 1];
            mat.col_indices[start..end]
                .iter()
                .zip(mat.values[start..end].iter())
                .find(|(c, _)| **c == j)
                .map_or(0.0, |(_, v)| *v)
        };
        let (mu, dx, rho, dt) = (0.25, 0.2, 1.0, 0.01);
        for i in 2..=3 {
            // Diagonal: rho dx / dt + 2 mu / dx (not 2 mu / dx^2), off-diagonal: -mu / dx.
            assert!((dense(i, i) - (rho * dx / dt + 2.0 * mu / dx)).abs() < 1e-12);
            assert!((dense(i, i - 1) + mu / dx).abs() < 1e-12);
            assert!((dense(i, i + 1) + mu / dx).abs() < 1e-12);
        }
        // Dirichlet rows are identity rows and the matrix is symmetric.
        assert_eq!(dense(0, 0), 1.0);
        assert_eq!(dense(5, 5), 1.0);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (dense(i, j) - dense(j, i)).abs() < 1e-14,
                    "matrix not symmetric at ({i}, {j})"
                );
            }
        }
        assert_eq!(dense(1, 0), 0.0);
        assert_eq!(dense(4, 5), 0.0);
    }

    #[test]
    fn test_ns_fem_sine_mode_decays_at_the_analytic_viscous_rate() {
        // u0 = 1e-6 sin(pi x) without forcing: convection is negligible and the
        // mode decays like exp(-(mu/rho) pi^2 t).  With the old mu/dx^2 scaling
        // the effective viscosity was mu/dx = 100 mu.
        // (re, density, dt, steps, amplitude); the tiny amplitude checks that the
        // solver tolerance is relative to the right-hand side, not absolute.
        let cases = [
            (1.0, 1.0, 5.0e-4, 200, 1.0e-6),
            (2.0, 2.0, 1.0e-3, 400, 1.0e-6),
            (1.0, 1.0, 1.0e-3, 100, 1.0e-12),
        ];
        for &(re, density, dt, steps, amplitude) in &cases {
            let mut ns = NavierStokesFEM::new(101, re);
            ns.density = density;
            for i in 0..ns.n_nodes {
                ns.velocity[i] = amplitude * (std::f64::consts::PI * i as f64 * ns.dx).sin();
            }
            let a0 = sine_mode_amplitude(&ns);
            for _ in 0..steps {
                ns.step_euler(dt).expect("implicit Euler step");
            }
            let t = ns.time();
            assert!((t - dt * steps as f64).abs() < 1e-12);
            let rate = (ns.viscosity / ns.density) * std::f64::consts::PI.powi(2);
            let expected = (-rate * t).exp();
            let ratio = sine_mode_amplitude(&ns) / a0;
            assert!(
                (ratio / expected - 1.0).abs() < 0.01,
                "Re = {re}, rho = {density}: amplitude ratio {ratio}, analytic {expected}"
            );
        }
    }

    #[test]
    fn test_ns_fem_forced_steady_state_is_the_poiseuille_parabola() {
        // mu u'' = -(f - dp/dx) with u(0) = u(1) = 0 has u = F/(2 mu) x (1 - x).
        // Linear elements are nodally exact for a constant load in 1-D.
        let mut ns = NavierStokesFEM::new(41, 1.0);
        ns.body_force = 0.004;
        ns.pressure_grad = -0.006;
        let load = 0.01;
        for _ in 0..40 {
            ns.step_euler(0.5).expect("implicit Euler step");
        }
        let mu = ns.viscosity;
        let u_max = load / (8.0 * mu);
        for (i, &u) in ns.velocity.iter().enumerate() {
            let x = i as f64 * ns.dx;
            let exact = load / (2.0 * mu) * x * (1.0 - x);
            assert!(
                (u - exact).abs() < 0.01 * u_max,
                "u[{i}] = {u}, parabola {exact}"
            );
        }
    }

    #[test]
    fn test_ns_fem_step_reports_non_convergence_and_keeps_state() {
        let mut ns = NavierStokesFEM::new(8, 10.0);
        ns.velocity[3] = f64::NAN;
        let err = ns
            .step_euler(0.01)
            .expect_err("NaN velocity cannot be solved");
        assert!(
            matches!(&err, Error::NotConverged { solver, .. } if solver == "ParallelPcgSolver"),
            "unexpected error: {err}"
        );
        assert_eq!(ns.time(), 0.0, "time must not advance on failure");
        assert!(ns.velocity[3].is_nan(), "velocity must be left untouched");
    }

    #[test]
    fn test_ns_fem_rejects_invalid_time_steps_and_state() {
        let mut ns = NavierStokesFEM::new(6, 10.0);
        for dt in [0.0, -0.01, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(ns.step_euler(dt), Err(Error::InvalidInput(_))),
                "dt = {dt} must be rejected"
            );
        }
        assert_eq!(ns.time(), 0.0);
        ns.velocity.pop();
        assert!(matches!(ns.step_euler(0.01), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_ns_fem_mesh_without_interior_nodes_only_advances_time() {
        for n in 0..3 {
            let mut ns = NavierStokesFEM::new(n, 10.0);
            ns.velocity.iter_mut().for_each(|u| *u = 1.0);
            ns.step_euler(0.25).expect("no unknowns to solve");
            assert!((ns.time() - 0.25).abs() < 1e-15);
            assert!(ns.velocity.iter().all(|&u| u == 0.0));
        }
    }

    #[test]
    fn test_ns_fem_imposes_dirichlet_values_on_a_violating_initial_state() {
        // A velocity that violates the boundary condition is projected onto it.
        let mut ns = NavierStokesFEM::new(11, 1.0);
        ns.velocity = vec![1.0; 11];
        ns.step_euler(0.01).expect("implicit Euler step");
        assert_eq!(ns.velocity[0], 0.0);
        assert_eq!(ns.velocity[10], 0.0);
        assert!(ns.velocity[5] > 0.0 && ns.velocity[5] < 1.0);
    }

    #[test]
    fn test_ns_fem_step_ignores_values_stored_at_dirichlet_nodes() {
        // The explicit upwind convection of the first and last interior node must
        // see the imposed value u = 0, not whatever is stored at the boundary.
        let interior = [0.0, 0.8, -0.5, 0.3, 0.9, -0.2, 0.4, 0.0];
        let mut clean = NavierStokesFEM::new(8, 200.0);
        clean.velocity = interior.to_vec();
        let mut dirty = NavierStokesFEM::new(8, 200.0);
        dirty.velocity = interior.to_vec();
        dirty.velocity[0] = 5.0;
        dirty.velocity[7] = -3.0;
        clean.step_euler(0.01).expect("clean step");
        dirty.step_euler(0.01).expect("dirty step");
        for (a, b) in clean.velocity.iter().zip(dirty.velocity.iter()) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
    }
}
