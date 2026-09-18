// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Bilinear quadrilateral (Q4) plane-stress finite elements on a structured grid.
//!
//! This is the structural solver behind [`TopOptSolver::run`],
//! [`LevelSetTopOpt::run`] and [`MultiLoadTopOpt::run_fe`]: it turns a field of
//! element Young's moduli plus nodal loads and supports into displacements,
//! the compliance `fᵀu` and the element strain energies `u_eᵀ K_e u_e` that
//! the compliance sensitivities are built from.
//!
//! # Mesh and numbering
//!
//! The design domain is the `nx × ny` grid of unit-square elements of
//! [`GridSize`], with unit thickness. Row `0` is the bottom edge (`y = 0`) and
//! column `0` the left edge (`x = 0`). Node `(row, col)` has the index
//! `row * (nx + 1) + col` and the two degrees of freedom `2 * node` (`u_x`)
//! and `2 * node + 1` (`u_y`); [`Q4Model::node_index`] and [`Q4Model::dof`]
//! compute them. The nodes of an element are numbered counter-clockwise from
//! its lower-left corner, see [`Q4Model::element_dofs`].
//!
//! # Linear algebra
//!
//! The global stiffness matrix `K = Σ_e E_e K_unit` is assembled into a
//! compressed-sparse-row (CSR) matrix whose pattern is built once per model.
//! Prescribed degrees of freedom (always zero displacement) are removed
//! symmetrically by zeroing their rows and columns and placing a one on the
//! diagonal. The system is then solved either directly, by a banded Cholesky
//! factorisation after renumbering the nodes along the shorter side of the
//! grid, or iteratively by Jacobi-preconditioned conjugate gradients, see
//! [`LinearSolver`]. SIMP stiffness fields span many orders of magnitude, which
//! makes plain Jacobi-PCG slow, so the direct solver is the default whenever
//! its factor fits in memory.
//!
//! [`TopOptSolver::run`]: super::TopOptSolver::run
//! [`LevelSetTopOpt::run`]: super::LevelSetTopOpt::run
//! [`MultiLoadTopOpt::run_fe`]: super::MultiLoadTopOpt::run_fe

use super::{GridSize, LoadCase, SiMP, invalid_input};
use crate::error::{PhysicsError, PhysicsResult, check_poisson_ratio};

/// A Cholesky pivot below this fraction of the original diagonal entry means
/// the matrix is singular (a rigid-body mode or a floating void region).
const SINGULAR_PIVOT_RATIO: f64 = 1e-13;

/// Largest number of band entries for which [`LinearSolver::Auto`] still
/// factorises directly (200 MB of `f64`; a 300 x 100 grid needs 12.5 million).
const AUTO_DIRECT_MAX_ENTRIES: usize = 25_000_000;

/// Relative residual target of the conjugate-gradient solver chosen by
/// [`LinearSolver::Auto`].
const AUTO_PCG_TOLERANCE: f64 = 1e-10;

/// Number of rigid-body motions of a plane structure, the least number of
/// constrained degrees of freedom of a stable support.
const PLANE_RIGID_BODY_MODES: usize = 3;

/// Natural coordinates `(ξ, η)` of the element nodes: lower-left,
/// lower-right, upper-right, upper-left.
const NODE_NATURAL_COORDINATES: [(f64, f64); 4] =
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

/// Stiffness matrix of a plane-stress Q4 element with unit Young's modulus.
///
/// The element is the unit square with unit thickness, so the matrix of an
/// element of modulus `E` is `E` times the returned one. It is integrated
/// with the 2×2 Gauss rule, which is exact for the bilinear rectangle. Rows
/// and columns follow [`Q4Model::element_dofs`]: the `(u_x, u_y)` pairs of the
/// lower-left, lower-right, upper-right and upper-left node.
///
/// The matrix is symmetric, positive semi-definite and has exactly the three
/// rigid-body modes in its kernel.
pub fn q4_element_stiffness(poisson_ratio: f64) -> [[f64; 8]; 8] {
    let nu = poisson_ratio;
    let scale = 1.0 - nu * nu;
    let constitutive = [
        [1.0 / scale, nu / scale, 0.0],
        [nu / scale, 1.0 / scale, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu) / scale],
    ];
    let g = (1.0f64 / 3.0).sqrt();
    let mut ke = [[0.0f64; 8]; 8];
    for (xi, eta) in [(-g, -g), (g, -g), (g, g), (-g, g)] {
        // Strain-displacement matrix B. The element is the unit square, so
        // ∂/∂x = 2 ∂/∂ξ and ∂/∂y = 2 ∂/∂η, and det J = 1/4 (weights are 1).
        let mut b = [[0.0f64; 8]; 3];
        for (node, &(xa, ya)) in NODE_NATURAL_COORDINATES.iter().enumerate() {
            let dn_dx = 0.5 * xa * (1.0 + ya * eta);
            let dn_dy = 0.5 * ya * (1.0 + xa * xi);
            b[0][2 * node] = dn_dx;
            b[1][2 * node + 1] = dn_dy;
            b[2][2 * node] = dn_dy;
            b[2][2 * node + 1] = dn_dx;
        }
        let mut db = [[0.0f64; 8]; 3];
        for (db_row, d_row) in db.iter_mut().zip(constitutive.iter()) {
            for (col, entry) in db_row.iter_mut().enumerate() {
                *entry = d_row
                    .iter()
                    .zip(b.iter())
                    .map(|(&d, b_row)| d * b_row[col])
                    .sum();
            }
        }
        for (i, ke_row) in ke.iter_mut().enumerate() {
            for (j, entry) in ke_row.iter_mut().enumerate() {
                let integrand: f64 = b
                    .iter()
                    .zip(db.iter())
                    .map(|(bi, dbi)| bi[i] * dbi[j])
                    .sum();
                *entry += 0.25 * integrand;
            }
        }
    }
    ke
}

/// How the linear system `K u = f` is solved.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LinearSolver {
    /// Banded Cholesky factorisation while the factor stays below about
    /// 200 MB (grids up to roughly 300 x 100 elements),
    /// Jacobi-preconditioned conjugate gradients otherwise.
    #[default]
    Auto,
    /// Banded Cholesky factorisation. It is exact up to round-off and
    /// insensitive to the stiffness contrast between solid and void, but
    /// needs `O(n · b)` memory and `O(n · b²)` time for `n` unknowns of
    /// half-bandwidth `b ≈ 2 · min(nx, ny)`.
    Direct,
    /// Jacobi-preconditioned conjugate gradients on the assembled CSR
    /// matrix. Its cost per iteration is linear in the number of unknowns,
    /// but the iteration count grows with the stiffness contrast.
    Pcg {
        /// Relative residual `‖K u − f‖ / ‖f‖` at which the iteration stops.
        tolerance: f64,
        /// Iteration limit; reaching it is reported as
        /// [`PhysicsError::ConvergenceFailed`].
        max_iterations: usize,
    },
}

/// Plane-stress Q4 model of an `nx × ny` grid of unit-square elements.
///
/// See the [module documentation](self) for the mesh, the degree-of-freedom
/// numbering and the solvers. The model holds everything that does not depend
/// on the material field (element matrix, CSR pattern, node renumbering) so
/// that one model serves every iteration of an optimisation.
#[derive(Debug, Clone)]
pub struct Q4Model {
    grid: GridSize,
    poisson_ratio: f64,
    /// Unit-modulus element stiffness, see [`q4_element_stiffness`].
    unit_stiffness: [[f64; 8]; 8],
    /// Internal degree of freedom of every user-numbered degree of freedom;
    /// the internal order numbers the nodes along the shorter grid side to
    /// keep the matrix bandwidth small.
    to_internal: Vec<usize>,
    /// CSR row pointers of the (internally numbered) stiffness pattern.
    row_ptr: Vec<usize>,
    /// CSR column indices, ascending within every row.
    col_idx: Vec<usize>,
    /// Largest `i - j` over the stored entries with `j <= i`.
    half_bandwidth: usize,
}

impl Q4Model {
    /// Creates the model of `grid` for Poisson's ratio `poisson_ratio`.
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInput`] for a grid without elements and
    /// [`PhysicsError::MaterialError`] for a Poisson ratio outside `(-1, 0.5)`.
    pub fn new(grid: GridSize, poisson_ratio: f64) -> PhysicsResult<Self> {
        if grid.nx == 0 || grid.ny == 0 {
            return Err(invalid_input(
                "grid",
                format!(
                    "needs at least one element in each direction, got {} x {}",
                    grid.nx, grid.ny
                ),
            ));
        }
        check_poisson_ratio(poisson_ratio)?;
        let nx1 = grid.nx + 1;
        let ny1 = grid.ny + 1;
        let n_nodes = nx1 * ny1;
        // Number the nodes along the shorter side first: the bandwidth is
        // set by the node-index distance between neighbouring grid lines.
        let rows_fastest = grid.ny < grid.nx;
        let internal_node = |row: usize, col: usize| {
            if rows_fastest {
                col * ny1 + row
            } else {
                row * nx1 + col
            }
        };

        let mut to_internal = vec![0usize; 2 * n_nodes];
        let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); 2 * n_nodes];
        for row in 0..ny1 {
            for col in 0..nx1 {
                let user = row * nx1 + col;
                let node = internal_node(row, col);
                to_internal[2 * user] = 2 * node;
                to_internal[2 * user + 1] = 2 * node + 1;

                let mut columns = Vec::with_capacity(18);
                for other_row in row.saturating_sub(1)..=(row + 1).min(ny1 - 1) {
                    for other_col in col.saturating_sub(1)..=(col + 1).min(nx1 - 1) {
                        let other = internal_node(other_row, other_col);
                        columns.push(2 * other);
                        columns.push(2 * other + 1);
                    }
                }
                columns.sort_unstable();
                pattern[2 * node + 1] = columns.clone();
                pattern[2 * node] = columns;
            }
        }

        let mut row_ptr = Vec::with_capacity(2 * n_nodes + 1);
        let mut col_idx = Vec::new();
        let mut half_bandwidth = 0usize;
        row_ptr.push(0);
        for (row, columns) in pattern.iter().enumerate() {
            for &column in columns {
                if column <= row {
                    half_bandwidth = half_bandwidth.max(row - column);
                }
                col_idx.push(column);
            }
            row_ptr.push(col_idx.len());
        }

        Ok(Self {
            grid,
            poisson_ratio,
            unit_stiffness: q4_element_stiffness(poisson_ratio),
            to_internal,
            row_ptr,
            col_idx,
            half_bandwidth,
        })
    }

    /// The grid of the model.
    pub fn grid(&self) -> GridSize {
        self.grid
    }

    /// Poisson's ratio of the material.
    pub fn poisson_ratio(&self) -> f64 {
        self.poisson_ratio
    }

    /// Unit-modulus element stiffness matrix, see [`q4_element_stiffness`].
    pub fn unit_element_stiffness(&self) -> &[[f64; 8]; 8] {
        &self.unit_stiffness
    }

    /// Number of nodes, `(nx + 1) (ny + 1)`.
    pub fn n_nodes(&self) -> usize {
        (self.grid.nx + 1) * (self.grid.ny + 1)
    }

    /// Number of degrees of freedom, twice the number of nodes.
    pub fn n_dofs(&self) -> usize {
        2 * self.n_nodes()
    }

    /// Index of the node in grid row `row` (`0..=ny`, bottom to top) and
    /// column `col` (`0..=nx`, left to right).
    pub fn node_index(&self, row: usize, col: usize) -> usize {
        row * (self.grid.nx + 1) + col
    }

    /// Degree of freedom of `node` in direction `component`: `0` for `u_x`
    /// and `1` for `u_y`.
    pub fn dof(&self, node: usize, component: usize) -> usize {
        2 * node + component
    }

    /// The eight degrees of freedom of element `element` (row-major element
    /// index `row * nx + col`): the `(u_x, u_y)` pairs of its lower-left,
    /// lower-right, upper-right and upper-left node, the order of the rows
    /// and columns of [`q4_element_stiffness`].
    pub fn element_dofs(&self, element: usize) -> [usize; 8] {
        let nx = self.grid.nx;
        let lower_left = (element / nx) * (nx + 1) + element % nx;
        let nodes = [
            lower_left,
            lower_left + 1,
            lower_left + nx + 2,
            lower_left + nx + 1,
        ];
        let mut dofs = [0usize; 8];
        for (pair, &node) in dofs.chunks_exact_mut(2).zip(nodes.iter()) {
            pair[0] = 2 * node;
            pair[1] = 2 * node + 1;
        }
        dofs
    }

    /// Assembles and prepares the system for the element moduli `modulus`
    /// (one per element, row-major) and the prescribed zero-displacement
    /// degrees of freedom `fixed_dofs`; [`Q4Analysis::solve`] then solves it
    /// for any number of load vectors.
    ///
    /// # Errors
    ///
    /// * [`PhysicsError::InvalidInput`] if `modulus` has the wrong length,
    ///   holds a negative or non-finite value, or if `solver` is a conjugate
    ///   gradient solver with a non-positive tolerance or iteration limit;
    /// * [`PhysicsError::OutOfBounds`] for a prescribed degree of freedom
    ///   outside the model;
    /// * [`PhysicsError::FemBoundaryConditionError`] if fewer than three
    ///   distinct degrees of freedom are prescribed, since the three
    ///   rigid-body motions of a plane structure cannot be suppressed then;
    /// * [`PhysicsError::NumericalDivergence`] if the direct solver finds the
    ///   matrix singular (an unsupported structure, or void with zero
    ///   stiffness cutting off a region).
    pub fn analyse(
        &self,
        modulus: &[f64],
        fixed_dofs: &[usize],
        solver: LinearSolver,
    ) -> PhysicsResult<Q4Analysis<'_>> {
        let n_elem = self.grid.n_elem();
        if modulus.len() != n_elem {
            return Err(invalid_input(
                "modulus",
                format!(
                    "expected one entry per element ({n_elem}), got {}",
                    modulus.len()
                ),
            ));
        }
        if let Some((element, &value)) = modulus
            .iter()
            .enumerate()
            .find(|&(_, &value)| !value.is_finite() || value < 0.0)
        {
            return Err(invalid_input(
                format!("modulus[{element}]"),
                format!("must be finite and non-negative, got {value}"),
            ));
        }
        let n_dofs = self.n_dofs();
        let mut fixed = vec![false; n_dofs];
        for &dof in fixed_dofs {
            let internal = self
                .to_internal
                .get(dof)
                .copied()
                .ok_or(PhysicsError::OutOfBounds {
                    index: dof,
                    max: n_dofs,
                })?;
            fixed[internal] = true;
        }
        let n_fixed = fixed.iter().filter(|&&is_fixed| is_fixed).count();
        if n_fixed < PLANE_RIGID_BODY_MODES {
            return Err(PhysicsError::FemBoundaryConditionError {
                node_id: fixed_dofs.first().map_or(0, |&dof| dof / 2),
                message: format!(
                    "{n_fixed} distinct degrees of freedom are prescribed but a plane structure \
                     needs at least {PLANE_RIGID_BODY_MODES} to remove its rigid-body motions"
                ),
            });
        }

        let prepared = match solver {
            LinearSolver::Direct => PreparedSolver::Direct,
            LinearSolver::Pcg {
                tolerance,
                max_iterations,
            } => PreparedSolver::Iterative {
                tolerance,
                max_iterations,
            },
            LinearSolver::Auto => {
                if prefers_direct_factorisation(n_dofs, self.half_bandwidth) {
                    PreparedSolver::Direct
                } else {
                    PreparedSolver::Iterative {
                        tolerance: AUTO_PCG_TOLERANCE,
                        max_iterations: n_dofs.max(1000).saturating_mul(10),
                    }
                }
            }
        };
        if let PreparedSolver::Iterative {
            tolerance,
            max_iterations,
        } = prepared
            && (!(tolerance > 0.0 && tolerance.is_finite()) || max_iterations == 0)
        {
            return Err(invalid_input(
                "solver",
                format!(
                    "conjugate gradients need a positive finite tolerance and a positive \
                     iteration limit, got tolerance {tolerance} and limit {max_iterations}"
                ),
            ));
        }

        let values = self.assemble(modulus, &fixed)?;
        let operator = match prepared {
            PreparedSolver::Direct => Operator::Direct(BandCholesky::factor(self, &values)?),
            PreparedSolver::Iterative {
                tolerance,
                max_iterations,
            } => {
                let inverse_diagonal = self.inverse_diagonal(&values)?;
                Operator::Iterative {
                    inverse_diagonal,
                    tolerance,
                    max_iterations,
                }
            }
        };
        Ok(Q4Analysis {
            model: self,
            fixed,
            values,
            operator,
        })
    }

    /// Assembles and solves in one call: the displacements of the structure
    /// with element moduli `modulus` under the nodal force vector `loads`
    /// (length [`Q4Model::n_dofs`]). See [`Q4Model::analyse`] for the errors.
    pub fn solve(
        &self,
        modulus: &[f64],
        loads: &[f64],
        fixed_dofs: &[usize],
        solver: LinearSolver,
    ) -> PhysicsResult<Q4Solution> {
        self.analyse(modulus, fixed_dofs, solver)?.solve(loads)
    }

    /// Returns a finite-element closure for [`MultiLoadTopOpt::run`] and
    /// [`MultiLoadTopOpt::evaluate`].
    ///
    /// For a density field and a load case the closure assembles the stiffness
    /// matrix with the SIMP interpolation of `simp`, solves it for the nodal
    /// force vector `load_case.loads` and returns the compliance together
    /// with the unit-modulus element strain energies `u_eᵀ K_e u_e`. The
    /// factorisation is cached, so the load cases of one iteration, which
    /// share the density field, cost a single assembly and factorisation.
    ///
    /// [`MultiLoadTopOpt::run`]: super::MultiLoadTopOpt::run
    /// [`MultiLoadTopOpt::evaluate`]: super::MultiLoadTopOpt::evaluate
    pub fn simp_oracle<'a>(
        &'a self,
        simp: &'a SiMP,
        fixed_dofs: &'a [usize],
        solver: LinearSolver,
    ) -> impl FnMut(&[f64], &LoadCase) -> PhysicsResult<(f64, Vec<f64>)> + 'a {
        let mut cache: Option<(Vec<f64>, Q4Analysis<'a>)> = None;
        move |density, load_case| {
            let cached = cache.take().filter(|(last, _)| last.as_slice() == density);
            let (last_density, analysis) = match cached {
                Some(entry) => entry,
                None => {
                    let modulus: Vec<f64> = density
                        .iter()
                        .map(|&rho| simp.interpolated_stiffness(rho))
                        .collect();
                    (
                        density.to_vec(),
                        self.analyse(&modulus, fixed_dofs, solver)?,
                    )
                }
            };
            let solution = analysis.solve(&load_case.loads);
            cache = Some((last_density, analysis));
            let solution = solution?;
            Ok((solution.compliance, solution.element_energy))
        }
    }

    /// Assembles the CSR values of the global stiffness matrix.
    ///
    /// Rows and columns of prescribed degrees of freedom are left out and
    /// their diagonal entries set to one.
    fn assemble(&self, modulus: &[f64], fixed: &[bool]) -> PhysicsResult<Vec<f64>> {
        let mut values = vec![0.0f64; self.col_idx.len()];
        for (element, &young) in modulus.iter().enumerate() {
            let dofs = self.element_dofs(element).map(|dof| self.to_internal[dof]);
            for (a, &row) in dofs.iter().enumerate() {
                if fixed[row] {
                    continue;
                }
                let range = self.row_ptr[row]..self.row_ptr[row + 1];
                let columns = &self.col_idx[range.clone()];
                let row_values = &mut values[range];
                for (b, &column) in dofs.iter().enumerate() {
                    if fixed[column] {
                        continue;
                    }
                    let slot = columns.binary_search(&column).map_err(|_| {
                        PhysicsError::FemAssemblyError {
                            message: format!(
                                "element {element} couples degrees of freedom {row} and {column}, \
                                 which are missing from the stiffness pattern"
                            ),
                        }
                    })?;
                    row_values[slot] += young * self.unit_stiffness[a][b];
                }
            }
        }
        for (row, _) in fixed.iter().enumerate().filter(|&(_, &is_fixed)| is_fixed) {
            let slot = self.diagonal_slot(row)?;
            values[slot] = 1.0;
        }
        Ok(values)
    }

    /// Position of the diagonal entry of `row` in the CSR arrays.
    fn diagonal_slot(&self, row: usize) -> PhysicsResult<usize> {
        let start = self.row_ptr[row];
        let columns = &self.col_idx[start..self.row_ptr[row + 1]];
        columns
            .binary_search(&row)
            .map(|offset| start + offset)
            .map_err(|_| PhysicsError::FemAssemblyError {
                message: format!("the stiffness pattern has no diagonal entry for DOF {row}"),
            })
    }

    /// `1 / K_ii` for the Jacobi preconditioner.
    fn inverse_diagonal(&self, values: &[f64]) -> PhysicsResult<Vec<f64>> {
        (0..self.n_dofs())
            .map(|row| {
                let diagonal = values[self.diagonal_slot(row)?];
                if diagonal > 0.0 && diagonal.is_finite() {
                    Ok(1.0 / diagonal)
                } else {
                    Err(PhysicsError::NumericalDivergence {
                        message: format!(
                            "the stiffness matrix has a non-positive diagonal entry at DOF {row}: \
                             an element with zero stiffness leaves it unsupported"
                        ),
                    })
                }
            })
            .collect()
    }

    /// `y = K x` for the CSR matrix with the given values.
    fn multiply(&self, values: &[f64], x: &[f64]) -> Vec<f64> {
        self.row_ptr
            .windows(2)
            .map(|bounds| {
                let (start, end) = (bounds[0], bounds[1]);
                self.col_idx[start..end]
                    .iter()
                    .zip(values[start..end].iter())
                    .map(|(&column, &value)| value * x[column])
                    .sum()
            })
            .collect()
    }
}

/// Solver chosen in [`Q4Model::analyse`] before the matrix is assembled.
#[derive(Debug, Clone, Copy)]
enum PreparedSolver {
    Direct,
    Iterative {
        tolerance: f64,
        max_iterations: usize,
    },
}

/// What [`Q4Analysis::solve`] applies to a right-hand side.
#[derive(Debug, Clone)]
enum Operator {
    Direct(BandCholesky),
    Iterative {
        inverse_diagonal: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    },
}

/// An assembled and factorised (or preconditioned) structure that can be
/// solved for any number of load vectors; created by [`Q4Model::analyse`].
#[derive(Debug, Clone)]
pub struct Q4Analysis<'a> {
    model: &'a Q4Model,
    /// Prescribed degrees of freedom, in the internal numbering.
    fixed: Vec<bool>,
    /// CSR values of the stiffness matrix with the supports applied.
    values: Vec<f64>,
    operator: Operator,
}

/// Result of a finite-element solve, see [`Q4Analysis::solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct Q4Solution {
    /// Nodal displacements `(u_x, u_y)` per node, in the numbering of
    /// [`Q4Model::dof`] (zero at the prescribed degrees of freedom).
    pub displacement: Vec<f64>,
    /// `u_eᵀ K_e u_e` for every element: the strain energy twice over for an
    /// element of unit Young's modulus. Multiplied by the element modulus it
    /// gives the element's share of the compliance, and it is the quantity the
    /// SIMP compliance sensitivity `-p ρ^(p-1) (E_solid - E_void) u_eᵀ K_e u_e`
    /// is built from.
    pub element_energy: Vec<f64>,
    /// Compliance `fᵀu`, twice the strain energy of the structure.
    pub compliance: f64,
    /// Relative residual `‖K u − f‖ / ‖f‖` of the linear solve, over the
    /// free degrees of freedom.
    pub relative_residual: f64,
    /// Conjugate-gradient iterations performed; zero for the direct solver.
    pub iterations: usize,
}

impl Q4Analysis<'_> {
    /// Solves for the nodal force vector `loads` (length [`Q4Model::n_dofs`];
    /// entries on prescribed degrees of freedom are reactions and ignored).
    ///
    /// # Errors
    ///
    /// [`PhysicsError::InvalidInput`] if `loads` has the wrong length or holds
    /// a non-finite value; [`PhysicsError::ConvergenceFailed`] if conjugate
    /// gradients exhaust their iterations; [`PhysicsError::NumericalDivergence`]
    /// if the iteration breaks down on an indefinite matrix.
    pub fn solve(&self, loads: &[f64]) -> PhysicsResult<Q4Solution> {
        self.solve_from(loads, None)
    }

    /// Like [`Q4Analysis::solve`], starting the conjugate-gradient iteration
    /// from the displacements `initial_guess` (a previous solution, in the
    /// user numbering). The direct solver ignores it.
    pub fn solve_from(
        &self,
        loads: &[f64],
        initial_guess: Option<&[f64]>,
    ) -> PhysicsResult<Q4Solution> {
        let model = self.model;
        let n_dofs = model.n_dofs();
        if loads.len() != n_dofs {
            return Err(invalid_input(
                "loads",
                format!(
                    "expected one entry per degree of freedom ({n_dofs}), got {}",
                    loads.len()
                ),
            ));
        }
        if let Some(dof) = loads.iter().position(|value| !value.is_finite()) {
            return Err(invalid_input(
                format!("loads[{dof}]"),
                format!("must be finite, got {}", loads[dof]),
            ));
        }
        let mut rhs = vec![0.0f64; n_dofs];
        for (dof, &load) in loads.iter().enumerate() {
            let internal = model.to_internal[dof];
            if !self.fixed[internal] {
                rhs[internal] = load;
            }
        }
        let rhs_norm = norm(&rhs);

        let mut iterations = 0usize;
        let internal_solution = if rhs_norm == 0.0 {
            vec![0.0f64; n_dofs]
        } else {
            match &self.operator {
                Operator::Direct(factor) => {
                    let mut x = rhs.clone();
                    factor.solve_in_place(&mut x);
                    x
                }
                Operator::Iterative {
                    inverse_diagonal,
                    tolerance,
                    max_iterations,
                } => {
                    let mut start = vec![0.0f64; n_dofs];
                    if let Some(guess) = initial_guess.filter(|guess| guess.len() == n_dofs) {
                        for (dof, &value) in guess.iter().enumerate() {
                            let internal = model.to_internal[dof];
                            if !self.fixed[internal] && value.is_finite() {
                                start[internal] = value;
                            }
                        }
                    }
                    let (x, used) = self.conjugate_gradients(
                        &rhs,
                        start,
                        inverse_diagonal,
                        *tolerance,
                        *max_iterations,
                    )?;
                    iterations = used;
                    x
                }
            }
        };
        if internal_solution.iter().any(|value| !value.is_finite()) {
            return Err(PhysicsError::NumericalDivergence {
                message: "the displacement solution is not finite: the structure is \
                          (nearly) mechanism-like"
                    .to_string(),
            });
        }

        let residual = if rhs_norm == 0.0 {
            0.0
        } else {
            let product = model.multiply(&self.values, &internal_solution);
            let error: Vec<f64> = product
                .iter()
                .zip(rhs.iter())
                .map(|(k_u, f)| k_u - f)
                .collect();
            norm(&error) / rhs_norm
        };

        let mut displacement = vec![0.0f64; n_dofs];
        for (dof, value) in displacement.iter_mut().enumerate() {
            *value = internal_solution[model.to_internal[dof]];
        }
        let n_elem = model.grid.n_elem();
        let mut element_energy = Vec::with_capacity(n_elem);
        for element in 0..n_elem {
            let dofs = model.element_dofs(element);
            let u = dofs.map(|dof| displacement[dof]);
            let mut energy = 0.0;
            for (row, &u_row) in model.unit_stiffness.iter().zip(u.iter()) {
                let k_u: f64 = row.iter().zip(u.iter()).map(|(k, u_col)| k * u_col).sum();
                energy += u_row * k_u;
            }
            element_energy.push(energy);
        }
        // The displacement vanishes at the prescribed degrees of freedom, so
        // their reactions do not contribute to the work of the loads.
        let compliance = dot(loads, &displacement);
        Ok(Q4Solution {
            displacement,
            element_energy,
            compliance,
            relative_residual: residual,
            iterations,
        })
    }

    /// Jacobi-preconditioned conjugate gradients from `start`; returns the
    /// solution and the number of iterations.
    fn conjugate_gradients(
        &self,
        rhs: &[f64],
        start: Vec<f64>,
        inverse_diagonal: &[f64],
        tolerance: f64,
        max_iterations: usize,
    ) -> PhysicsResult<(Vec<f64>, usize)> {
        let model = self.model;
        let rhs_norm = norm(rhs);
        let mut x = start;
        let product = model.multiply(&self.values, &x);
        let mut r: Vec<f64> = rhs
            .iter()
            .zip(product.iter())
            .map(|(b, k_x)| b - k_x)
            .collect();
        let mut z: Vec<f64> = r
            .iter()
            .zip(inverse_diagonal.iter())
            .map(|(r_i, m_i)| r_i * m_i)
            .collect();
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut residual = norm(&r) / rhs_norm;
        let mut iterations = 0usize;
        while residual > tolerance {
            if iterations >= max_iterations {
                return Err(PhysicsError::ConvergenceFailed {
                    iterations,
                    tolerance,
                    residual,
                });
            }
            let k_p = model.multiply(&self.values, &p);
            let curvature = dot(&p, &k_p);
            if !(curvature > 0.0 && curvature.is_finite()) {
                return Err(PhysicsError::NumericalDivergence {
                    message: format!(
                        "conjugate gradients met a non-positive curvature {curvature} after \
                         {iterations} iterations: the stiffness matrix is not positive definite"
                    ),
                });
            }
            let alpha = rz / curvature;
            for ((x_i, r_i), (&p_i, &k_p_i)) in
                x.iter_mut().zip(r.iter_mut()).zip(p.iter().zip(k_p.iter()))
            {
                *x_i += alpha * p_i;
                *r_i -= alpha * k_p_i;
            }
            for ((z_i, &r_i), &m_i) in z.iter_mut().zip(r.iter()).zip(inverse_diagonal.iter()) {
                *z_i = r_i * m_i;
            }
            let rz_next = dot(&r, &z);
            let beta = rz_next / rz;
            for (p_i, &z_i) in p.iter_mut().zip(z.iter()) {
                *p_i = z_i + beta * *p_i;
            }
            rz = rz_next;
            residual = norm(&r) / rhs_norm;
            iterations += 1;
        }
        Ok((x, iterations))
    }
}

/// Lower Cholesky factor `L` (`A = L Lᵀ`) of a symmetric positive-definite
/// banded matrix.
///
/// Row `i` of the band occupies `data[i * (b + 1) .. (i + 1) * (b + 1)]` for
/// the half-bandwidth `b`; the entry `L[i][j]` with `i - b <= j <= i` sits at
/// offset `j + b - i`, so every row is stored in ascending column order with
/// the diagonal last.
#[derive(Debug, Clone)]
struct BandCholesky {
    size: usize,
    half_bandwidth: usize,
    data: Vec<f64>,
}

impl BandCholesky {
    /// Factorises the matrix given by the CSR `values` of `model`.
    fn factor(model: &Q4Model, values: &[f64]) -> PhysicsResult<Self> {
        let size = model.n_dofs();
        let bandwidth = model.half_bandwidth;
        let width = bandwidth + 1;
        let mut data = vec![0.0f64; size * width];
        for row in 0..size {
            let range = model.row_ptr[row]..model.row_ptr[row + 1];
            for (&column, &value) in model.col_idx[range.clone()]
                .iter()
                .zip(values[range].iter())
            {
                if column <= row {
                    data[row * width + column + bandwidth - row] = value;
                }
            }
        }

        for i in 0..size {
            let first = i.saturating_sub(bandwidth);
            let (done, rest) = data.split_at_mut(i * width);
            let row_i = &mut rest[..width];
            let original_diagonal = row_i[bandwidth];
            for j in first..=i {
                let terms = j - first;
                let start_i = first + bandwidth - i;
                if j < i {
                    let row_j = &done[j * width..(j + 1) * width];
                    let start_j = first + bandwidth - j;
                    let s = row_i[j + bandwidth - i]
                        - dot(
                            &row_i[start_i..start_i + terms],
                            &row_j[start_j..start_j + terms],
                        );
                    row_i[j + bandwidth - i] = s / row_j[bandwidth];
                } else {
                    let s = original_diagonal
                        - dot(
                            &row_i[start_i..start_i + terms],
                            &row_i[start_i..start_i + terms],
                        );
                    if !(s > SINGULAR_PIVOT_RATIO * original_diagonal && s.is_finite()) {
                        return Err(PhysicsError::NumericalDivergence {
                            message: format!(
                                "the stiffness matrix is singular at DOF {i} (pivot {s:e} for \
                                 diagonal {original_diagonal:e}): the supports leave a rigid-body \
                                 motion or zero-stiffness void cuts off part of the structure"
                            ),
                        });
                    }
                    row_i[bandwidth] = s.sqrt();
                }
            }
        }
        Ok(Self {
            size,
            half_bandwidth: bandwidth,
            data,
        })
    }

    /// Solves `A x = b` in place: `x` holds `b` on entry and `x` on exit.
    fn solve_in_place(&self, x: &mut [f64]) {
        let bandwidth = self.half_bandwidth;
        let width = bandwidth + 1;
        for i in 0..self.size {
            let first = i.saturating_sub(bandwidth);
            let row = &self.data[i * width..(i + 1) * width];
            let start = first + bandwidth - i;
            let s = x[i] - dot(&row[start..start + (i - first)], &x[first..i]);
            x[i] = s / row[bandwidth];
        }
        for i in (0..self.size).rev() {
            let first = i.saturating_sub(bandwidth);
            let row = &self.data[i * width..(i + 1) * width];
            x[i] /= row[bandwidth];
            let x_i = x[i];
            let start = first + bandwidth - i;
            for (x_k, &l_ik) in x[first..i]
                .iter_mut()
                .zip(row[start..start + (i - first)].iter())
            {
                *x_k -= l_ik * x_i;
            }
        }
    }
}

/// Whether [`LinearSolver::Auto`] factorises a system of `n_dofs` unknowns
/// with half-bandwidth `half_bandwidth` directly, that is whether its band
/// factor fits in [`AUTO_DIRECT_MAX_ENTRIES`] entries.
fn prefers_direct_factorisation(n_dofs: usize, half_bandwidth: usize) -> bool {
    n_dofs.saturating_mul(half_bandwidth + 1) <= AUTO_DIRECT_MAX_ENTRIES
}

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests;
