// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Finite Element Method (FEM) simulation API for Python interop.
//!
//! Python-facing 2-D plane-stress structural FEM built on the `oxiphysics-fem`
//! engine:
//! - Linear elastic 2-D triangular elements (CST - Constant Strain Triangle);
//!   the element kernels live here because the engine has no 2-D CST element
//! - Sparse global stiffness assembly into the engine's CSR matrix
//!   (`oxiphysics::fem::sparse::CsrMatrix`)
//! - Dirichlet boundary conditions applied by elimination, nodal load vectors
//! - Static solve with the engine's Jacobi-preconditioned conjugate gradient
//!   (`oxiphysics::fem::solvers::pcg_solve`, relative residual criterion)
//! - Modal analysis with a lumped mass matrix and the engine's subspace
//!   iteration (`oxiphysics::fem::eigen::subspace_iteration`)
//! - Displacement and von Mises stress output

use crate::error::{Error, Result as BridgeResult};
use oxiphysics::fem::eigen::subspace_iteration;
use oxiphysics::fem::solvers::pcg_solve;
use oxiphysics::fem::sparse::CsrMatrix;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Linear elastic isotropic material properties.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemMaterial {
    /// Young's modulus E (Pa).
    #[pyo3(get, set)]
    pub young_modulus: f64,
    /// Poisson's ratio ν (dimensionless, 0..0.5).
    #[pyo3(get, set)]
    pub poisson_ratio: f64,
    /// Mass density ρ (kg/m³).
    #[pyo3(get, set)]
    pub density: f64,
}

#[pymethods]
impl PyFemMaterial {
    /// Create a new material.
    #[new]
    pub fn new(young_modulus: f64, poisson_ratio: f64, density: f64) -> Self {
        Self {
            young_modulus,
            poisson_ratio,
            density,
        }
    }

    /// Steel-like material (E=200 GPa, ν=0.3, ρ=7850 kg/m³).
    #[staticmethod]
    pub fn steel() -> Self {
        Self::new(200e9, 0.3, 7850.0)
    }

    /// Aluminum-like material (E=70 GPa, ν=0.33, ρ=2700 kg/m³).
    #[staticmethod]
    pub fn aluminum() -> Self {
        Self::new(70e9, 0.33, 2700.0)
    }

    /// Rubber-like material (E=0.01 GPa, ν=0.49, ρ=1100 kg/m³).
    #[staticmethod]
    pub fn rubber() -> Self {
        Self::new(10e6, 0.49, 1100.0)
    }

    /// Concrete-like material (E=30 GPa, ν=0.2, ρ=2400 kg/m³).
    #[staticmethod]
    pub fn concrete() -> Self {
        Self::new(30e9, 0.2, 2400.0)
    }

    /// Compute the plane-stress constitutive matrix D (3×3 as flat [f64; 9]).
    ///
    /// Returns the D matrix that relates stress to strain: σ = D ε
    pub fn plane_stress_d(&self) -> Vec<f64> {
        let e = self.young_modulus;
        let nu = self.poisson_ratio;
        let c = e / (1.0 - nu * nu);
        vec![
            c,
            c * nu,
            0.0,
            c * nu,
            c,
            0.0,
            0.0,
            0.0,
            c * (1.0 - nu) * 0.5,
        ]
    }
}

impl Default for PyFemMaterial {
    fn default() -> Self {
        Self::steel()
    }
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A single node in the FEM mesh.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemNode {
    /// Node coordinates [x, y].
    pub position: [f64; 2],
}

#[pymethods]
impl PyFemNode {
    /// Create a new node at the given 2-D position.
    #[new]
    pub fn new(x: f64, y: f64) -> Self {
        Self { position: [x, y] }
    }

    /// Get the x coordinate.
    pub fn x(&self) -> f64 {
        self.position[0]
    }

    /// Get the y coordinate.
    pub fn y(&self) -> f64 {
        self.position[1]
    }

    /// Get position as [x, y].
    pub fn position(&self) -> Vec<f64> {
        self.position.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Element (CST triangular element)
// ---------------------------------------------------------------------------

/// A triangular (CST) element defined by three node indices.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemElement {
    /// Indices of the three corner nodes (counter-clockwise preferred).
    pub nodes: [usize; 3],
    /// Index into the material list.
    #[pyo3(get, set)]
    pub material_id: usize,
    /// Element thickness t (for plane-stress, metres).
    #[pyo3(get, set)]
    pub thickness: f64,
}

#[pymethods]
impl PyFemElement {
    /// Create a new triangular element.
    #[new]
    pub fn new(n0: usize, n1: usize, n2: usize, material_id: usize, thickness: f64) -> Self {
        Self {
            nodes: [n0, n1, n2],
            material_id,
            thickness,
        }
    }

    /// Get nodes as [n0, n1, n2].
    pub fn nodes(&self) -> Vec<usize> {
        self.nodes.to_vec()
    }
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

/// A Dirichlet (fixed DOF) boundary condition.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemDirichletBC {
    /// Node index.
    #[pyo3(get, set)]
    pub node: usize,
    /// DOF index within the node (0=x, 1=y).
    #[pyo3(get, set)]
    pub dof: usize,
    /// Prescribed displacement value.
    #[pyo3(get, set)]
    pub value: f64,
}

#[pymethods]
impl PyFemDirichletBC {
    /// Create a new Dirichlet BC.
    #[new]
    pub fn new(node: usize, dof: usize, value: f64) -> Self {
        Self { node, dof, value }
    }

    /// Fix the x-DOF of `node` to zero.
    #[staticmethod]
    pub fn fix_x(node: usize) -> Self {
        Self {
            node,
            dof: 0,
            value: 0.0,
        }
    }

    /// Fix the y-DOF of `node` to zero.
    #[staticmethod]
    pub fn fix_y(node: usize) -> Self {
        Self {
            node,
            dof: 1,
            value: 0.0,
        }
    }

    /// Fix both DOFs of `node` (fully pinned). Returns a list of two BCs.
    #[staticmethod]
    pub fn pin(node: usize) -> Vec<Self> {
        vec![Self::fix_x(node), Self::fix_y(node)]
    }
}

/// A Neumann (nodal force) boundary condition.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemNodalForce {
    /// Node index.
    #[pyo3(get, set)]
    pub node: usize,
    /// Force vector [fx, fy] in Newtons.
    pub force: [f64; 2],
}

#[pymethods]
impl PyFemNodalForce {
    /// Apply force `[fx, fy]` to `node`.
    #[new]
    pub fn new(node: usize, fx: f64, fy: f64) -> Self {
        Self {
            node,
            force: [fx, fy],
        }
    }

    /// Get force as [fx, fy].
    pub fn force(&self) -> Vec<f64> {
        self.force.to_vec()
    }
}

// ---------------------------------------------------------------------------
// FEM Mesh
// ---------------------------------------------------------------------------

/// A 2-D FEM mesh with nodes, elements, materials, and boundary conditions.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemMesh {
    /// All nodes in the mesh.
    pub nodes: Vec<PyFemNode>,
    /// All triangular elements in the mesh.
    pub elements: Vec<PyFemElement>,
    /// Material library.
    pub materials: Vec<PyFemMaterial>,
    /// Dirichlet (fixed) boundary conditions.
    pub dirichlet_bcs: Vec<PyFemDirichletBC>,
    /// Nodal force loads.
    pub nodal_forces: Vec<PyFemNodalForce>,
}

#[pymethods]
impl PyFemMesh {
    /// Create an empty mesh.
    #[new]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
            materials: vec![PyFemMaterial::default()],
            dirichlet_bcs: Vec::new(),
            nodal_forces: Vec::new(),
        }
    }

    /// Add a node at `(x, y)`. Returns node index.
    pub fn add_node(&mut self, x: f64, y: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PyFemNode::new(x, y));
        idx
    }

    /// Add a material. Returns material index.
    pub fn add_material(&mut self, mat: PyFemMaterial) -> usize {
        let idx = self.materials.len();
        self.materials.push(mat);
        idx
    }

    /// Add a CST element connecting nodes `n0, n1, n2`. Returns element index.
    pub fn add_element(&mut self, n0: usize, n1: usize, n2: usize) -> usize {
        self.add_element_with_material(n0, n1, n2, 0, 1.0)
    }

    /// Add a CST element with explicit material and thickness. Returns element index.
    pub fn add_element_with_material(
        &mut self,
        n0: usize,
        n1: usize,
        n2: usize,
        material_id: usize,
        thickness: f64,
    ) -> usize {
        let idx = self.elements.len();
        self.elements
            .push(PyFemElement::new(n0, n1, n2, material_id, thickness));
        idx
    }

    /// Add a Dirichlet boundary condition.
    pub fn add_dirichlet_bc(&mut self, bc: PyFemDirichletBC) {
        self.dirichlet_bcs.push(bc);
    }

    /// Pin both DOFs of `node` (fix x and y to zero).
    pub fn pin_node(&mut self, node: usize) {
        self.dirichlet_bcs.push(PyFemDirichletBC::fix_x(node));
        self.dirichlet_bcs.push(PyFemDirichletBC::fix_y(node));
    }

    /// Apply a nodal force at `node`.
    pub fn apply_nodal_force(&mut self, node: usize, fx: f64, fy: f64) {
        self.nodal_forces.push(PyFemNodalForce::new(node, fx, fy));
    }

    /// Number of degrees of freedom (2 per node).
    pub fn num_dofs(&self) -> usize {
        self.nodes.len() * 2
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of elements.
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Compute the signed area of triangular element `e`.
    pub fn element_area(&self, e: usize) -> f64 {
        let elem = &self.elements[e];
        let [n0, n1, n2] = elem.nodes;
        let [x0, y0] = self.nodes[n0].position;
        let [x1, y1] = self.nodes[n1].position;
        let [x2, y2] = self.nodes[n2].position;
        0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    }

    /// Build a simple structured rectangular mesh of triangles.
    ///
    /// Creates a `nx × ny` grid of quads, each quad split into 2 triangles.
    /// `domain` = `[ox, oy, lx, ly]` (origin + size); `grid` = `[nx, ny]`.
    /// Returns the number of nodes added.
    pub fn build_rectangle(
        &mut self,
        domain: Vec<f64>,
        grid: Vec<usize>,
        material_id: usize,
        thickness: f64,
    ) -> usize {
        let (ox, oy, lx, ly) = if domain.len() >= 4 {
            (domain[0], domain[1], domain[2], domain[3])
        } else {
            (0.0, 0.0, 1.0, 1.0)
        };
        let (nx, ny) = if grid.len() >= 2 {
            (grid[0], grid[1])
        } else {
            (1, 1)
        };
        self.build_rectangle_impl(ox, oy, lx, ly, nx, ny, material_id, thickness)
    }

    /// Return the bounding box of all nodes as `[xmin, ymin, xmax, ymax]`.
    pub fn bounding_box(&self) -> Option<Vec<f64>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut xmin = f64::INFINITY;
        let mut ymin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for n in &self.nodes {
            xmin = xmin.min(n.position[0]);
            ymin = ymin.min(n.position[1]);
            xmax = xmax.max(n.position[0]);
            ymax = ymax.max(n.position[1]);
        }
        Some(vec![xmin, ymin, xmax, ymax])
    }

    /// Return the total area of all elements (sum of |area_i|).
    pub fn total_area(&self) -> f64 {
        (0..self.num_elements())
            .map(|e| self.element_area(e).abs())
            .sum()
    }

    /// Return the centroid of element `e` as `[cx, cy]`.
    pub fn element_centroid(&self, e: usize) -> Option<Vec<f64>> {
        if e >= self.elements.len() {
            return None;
        }
        let [n0, n1, n2] = self.elements[e].nodes;
        let [x0, y0] = self.nodes[n0].position;
        let [x1, y1] = self.nodes[n1].position;
        let [x2, y2] = self.nodes[n2].position;
        Some(vec![(x0 + x1 + x2) / 3.0, (y0 + y1 + y2) / 3.0])
    }

    /// Find the node closest to position `(px, py)`.
    pub fn closest_node(&self, px: f64, py: f64) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let (idx, _) = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let dx = n.position[0] - px;
                let dy = n.position[1] - py;
                (i, dx * dx + dy * dy)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some(idx)
    }

    /// Return nodes on the left boundary (x <= xmin + tol).
    pub fn left_boundary_nodes(&self, tol: f64) -> Vec<usize> {
        if let Some(bb) = self.bounding_box() {
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.position[0] <= bb[0] + tol)
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Return nodes on the right boundary (x >= xmax - tol).
    pub fn right_boundary_nodes(&self, tol: f64) -> Vec<usize> {
        if let Some(bb) = self.bounding_box() {
            self.nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.position[0] >= bb[2] - tol)
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Pin all nodes on the left boundary.
    pub fn pin_left_boundary(&mut self, tol: f64) {
        let nodes = self.left_boundary_nodes(tol);
        for n in nodes {
            self.pin_node(n);
        }
    }
}

impl PyFemMesh {
    /// Internal helper for building a rectangular mesh with explicit coordinates.
    pub fn build_rectangle_impl(
        &mut self,
        ox: f64,
        oy: f64,
        lx: f64,
        ly: f64,
        nx: usize,
        ny: usize,
        material_id: usize,
        thickness: f64,
    ) -> usize {
        let dx = lx / nx as f64;
        let dy = ly / ny as f64;
        let node_start = self.nodes.len();

        // Add nodes
        for j in 0..=(ny) {
            for i in 0..=(nx) {
                self.add_node(ox + i as f64 * dx, oy + j as f64 * dy);
            }
        }

        // Add triangular elements (2 per quad)
        let row = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
                let n0 = node_start + j * row + i;
                let n1 = node_start + j * row + i + 1;
                let n2 = node_start + (j + 1) * row + i;
                let n3 = node_start + (j + 1) * row + i + 1;
                self.add_element_with_material(n0, n1, n3, material_id, thickness);
                self.add_element_with_material(n0, n3, n2, material_id, thickness);
            }
        }

        (ny + 1) * (nx + 1)
    }
}

impl Default for PyFemMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FEM Solver result
// ---------------------------------------------------------------------------

/// Result of a FEM linear static solve.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyFemSolveResult {
    /// Nodal displacement vector (flat: [ux0, uy0, ux1, uy1, ...]).
    #[pyo3(get)]
    pub displacements: Vec<f64>,
    /// Von Mises stress per element.
    #[pyo3(get)]
    pub von_mises_stress: Vec<f64>,
    /// Number of PCG solver iterations used.
    #[pyo3(get)]
    pub solver_iterations: usize,
    /// Final absolute residual norm ‖b − K u‖ of the system after Dirichlet
    /// elimination (convergence is judged on ‖r‖ / ‖b‖).
    #[pyo3(get)]
    pub residual_norm: f64,
    /// Whether PCG reached the relative tolerance within the iteration budget.
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyFemSolveResult {
    /// Displacement of node `i` as `[ux, uy]`.
    pub fn node_displacement(&self, i: usize) -> Option<Vec<f64>> {
        let base = i * 2;
        if base + 1 < self.displacements.len() {
            Some(vec![self.displacements[base], self.displacements[base + 1]])
        } else {
            None
        }
    }

    /// Maximum von Mises stress over all elements.
    pub fn max_von_mises(&self) -> f64 {
        self.von_mises_stress
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Mean von Mises stress.
    pub fn mean_von_mises(&self) -> f64 {
        if self.von_mises_stress.is_empty() {
            return 0.0;
        }
        self.von_mises_stress.iter().sum::<f64>() / self.von_mises_stress.len() as f64
    }

    /// Return the displacement field as a list of `[ux, uy]` pairs.
    pub fn displacement_pairs(&self) -> Vec<Vec<f64>> {
        self.displacements
            .chunks_exact(2)
            .map(|c| vec![c[0], c[1]])
            .collect()
    }

    /// Maximum absolute displacement component.
    pub fn max_displacement_magnitude(&self) -> f64 {
        self.displacement_pairs()
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
            .fold(0.0f64, f64::max)
    }

    /// Return all von Mises stresses as a sorted (ascending) copy.
    pub fn sorted_von_mises(&self) -> Vec<f64> {
        let mut v = self.von_mises_stress.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Return the `q`-th quantile of the von Mises stress distribution (q in `[0, 1]`).
    pub fn von_mises_quantile(&self, q: f64) -> f64 {
        if self.von_mises_stress.is_empty() {
            return 0.0;
        }
        let sorted = self.sorted_von_mises();
        let idx = ((q.clamp(0.0, 1.0)) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx]
    }
}

// ---------------------------------------------------------------------------
// FEM Solver
// ---------------------------------------------------------------------------

/// Linear static FEM solver for 2-D plane-stress problems.
///
/// Assembles the global stiffness matrix K from CST elements as sparse
/// triplets into the engine's [`CsrMatrix`] (`oxiphysics::fem::sparse`),
/// applies Dirichlet boundary conditions by elimination (prescribed values are
/// moved to the right-hand side and the constrained rows/columns are replaced
/// by identity rows), and solves K u = f with the engine's Jacobi-preconditioned
/// conjugate gradient [`pcg_solve`] (`oxiphysics::fem::solvers`), which stops
/// when the relative residual ‖r‖ / ‖b‖ drops below `tolerance`.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyFemSolver {
    /// Maximum PCG iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Relative PCG convergence tolerance on ‖r‖ / ‖b‖.
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl PyFemSolver {
    /// Create a solver with default settings (10 000 iterations, relative tolerance 1e-10).
    #[new]
    pub fn new() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-10,
        }
    }

    /// Create a solver with custom settings.
    #[staticmethod]
    pub fn with_settings(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Solve the linear static problem K u = f.
    ///
    /// Returns `None` if the mesh has no nodes or no elements.
    ///
    /// # Errors
    ///
    /// Raises an error when the solver settings are invalid (zero iteration
    /// budget, non-positive tolerance) or the mesh is inconsistent: element,
    /// load or boundary-condition indices out of range, a DOF index other than
    /// 0/1, degenerate elements, invalid material parameters, non-finite input,
    /// conflicting prescribed values on one DOF, or a load applied to a node
    /// that belongs to no element (and therefore has no stiffness).
    pub fn solve(&self, mesh: &PyFemMesh) -> BridgeResult<Option<PyFemSolveResult>> {
        let n_nodes = mesh.num_nodes();
        if n_nodes == 0 || mesh.elements.is_empty() {
            return Ok(None);
        }
        if self.max_iterations == 0 {
            return Err(Error::invalid_param(
                "max_iterations",
                "the PCG iteration budget must be positive",
            ));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(Error::invalid_param(
                "tolerance",
                format!(
                    "the relative PCG tolerance must be positive and finite, got {}",
                    self.tolerance
                ),
            ));
        }
        validate_mesh(mesh)?;
        let n_dofs = n_nodes * 2;

        let k_triplets = assemble_stiffness_triplets(mesh);
        let prescribed = prescribed_dofs(mesh)?;
        let node_used = nodes_used_by_elements(mesh);

        // --- Load vector f ---
        let mut rhs = vec![0.0f64; n_dofs];
        for nf in &mesh.nodal_forces {
            rhs[nf.node * 2] += nf.force[0];
            rhs[nf.node * 2 + 1] += nf.force[1];
        }

        // --- Diagonal of K (used to scale the identity rows of eliminated DOFs) ---
        let mut k_diag = vec![0.0f64; n_dofs];
        for &(i, j, v) in &k_triplets {
            if i == j {
                k_diag[i] += v;
            }
        }

        // --- Nodes without any element have no stiffness at all ---
        // A load on such a DOF has no solution; an unloaded, unconstrained one
        // is decoupled from the structure and is held at zero displacement.
        let mut detached = vec![false; n_dofs];
        for (node, &used) in node_used.iter().enumerate() {
            if used {
                continue;
            }
            for dof in [node * 2, node * 2 + 1] {
                if prescribed[dof].is_some() {
                    continue;
                }
                if rhs[dof] != 0.0 {
                    return Err(Error::invalid_param(
                        "nodal_forces",
                        format!(
                            "node {node} carries a load but belongs to no element, \
                             so it has no stiffness"
                        ),
                    ));
                }
                detached[dof] = true;
            }
        }

        // --- Dirichlet BCs by elimination ---
        // Rows of prescribed DOFs are dropped; columns of prescribed DOFs are
        // moved to the right-hand side: b_i -= K_id * g_d.
        let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(k_triplets.len() + n_dofs);
        for &(i, j, v) in &k_triplets {
            if prescribed[i].is_some() {
                continue;
            }
            match prescribed[j] {
                Some(g) => rhs[i] -= v * g,
                None => triplets.push((i, j, v)),
            }
        }
        // Identity rows (scaled by the original diagonal to keep the matrix
        // well conditioned): K_dd u_d = K_dd g_d.
        for dof in 0..n_dofs {
            if let Some(g) = prescribed[dof] {
                let scale = if k_diag[dof] > 0.0 { k_diag[dof] } else { 1.0 };
                triplets.push((dof, dof, scale));
                rhs[dof] = scale * g;
                k_diag[dof] = scale;
            } else if detached[dof] {
                triplets.push((dof, dof, 1.0));
                rhs[dof] = 0.0;
                k_diag[dof] = 1.0;
            }
        }
        let k_reduced = CsrMatrix::from_triplets(n_dofs, n_dofs, &triplets);

        // Jacobi preconditioner. Every remaining free DOF belongs to at least one
        // non-degenerate element with a positive-definite D matrix, so its
        // diagonal entry is strictly positive.
        let mut inv_diag = Vec::with_capacity(n_dofs);
        for (dof, &d) in k_diag.iter().enumerate() {
            if !(d.is_finite() && d > 0.0) {
                return Err(Error::general(format!(
                    "stiffness diagonal of DOF {dof} is not positive ({d}); \
                     the mesh is not a valid plane-stress model"
                )));
            }
            inv_diag.push(1.0 / d);
        }

        // --- Solve with the engine's PCG ---
        let (u, stats) = pcg_solve(
            &k_reduced,
            &rhs,
            &inv_diag,
            self.max_iterations,
            self.tolerance,
        );

        // Eliminated rows are decoupled identities, so their exact values are
        // known; overwrite the iterate to remove PCG round-off on them.
        let mut u = u;
        for (dof, value) in u.iter_mut().enumerate() {
            if let Some(g) = prescribed[dof] {
                *value = g;
            } else if detached[dof] {
                *value = 0.0;
            }
        }

        // --- Von Mises stress per element ---
        let von_mises = mesh
            .elements
            .iter()
            .map(|elem| {
                let mat = &mesh.materials[elem.material_id];
                let [sx, sy, txy] = element_stress(mesh, elem, mat, &u);
                (sx * sx - sx * sy + sy * sy + 3.0 * txy * txy).sqrt()
            })
            .collect();

        Ok(Some(PyFemSolveResult {
            displacements: u,
            von_mises_stress: von_mises,
            solver_iterations: stats.iterations,
            residual_norm: stats.residual,
            converged: stats.converged,
        }))
    }
}

impl Default for PyFemSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal: validation and assembly shared by the static and modal solvers
// ---------------------------------------------------------------------------

/// Check that every index in the mesh is in range and every parameter is valid.
fn validate_mesh(mesh: &PyFemMesh) -> BridgeResult<()> {
    let n_nodes = mesh.num_nodes();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if !(node.position[0].is_finite() && node.position[1].is_finite()) {
            return Err(Error::invalid_param(
                "nodes",
                format!("node {i} has a non-finite position"),
            ));
        }
    }
    for (m, mat) in mesh.materials.iter().enumerate() {
        if !(mat.young_modulus.is_finite() && mat.young_modulus > 0.0) {
            return Err(Error::invalid_param(
                "young_modulus",
                format!(
                    "material {m}: Young's modulus must be positive, got {}",
                    mat.young_modulus
                ),
            ));
        }
        if !(mat.poisson_ratio.is_finite() && mat.poisson_ratio > -1.0 && mat.poisson_ratio < 0.5) {
            return Err(Error::invalid_param(
                "poisson_ratio",
                format!(
                    "material {m}: Poisson's ratio must lie in (-1, 0.5), got {}",
                    mat.poisson_ratio
                ),
            ));
        }
    }
    for (e, elem) in mesh.elements.iter().enumerate() {
        if let Some(&bad) = elem.nodes.iter().find(|&&n| n >= n_nodes) {
            return Err(Error::invalid_param(
                "elements",
                format!("element {e} references node {bad}, but the mesh has {n_nodes} nodes"),
            ));
        }
        let [n0, n1, n2] = elem.nodes;
        if n0 == n1 || n1 == n2 || n0 == n2 {
            return Err(Error::invalid_param(
                "elements",
                format!("element {e} repeats a node: {:?}", elem.nodes),
            ));
        }
        if elem.material_id >= mesh.materials.len() {
            return Err(Error::invalid_param(
                "material_id",
                format!(
                    "element {e} uses material {}, but the mesh has {} materials",
                    elem.material_id,
                    mesh.materials.len()
                ),
            ));
        }
        if !(elem.thickness.is_finite() && elem.thickness > 0.0) {
            return Err(Error::invalid_param(
                "thickness",
                format!(
                    "element {e}: thickness must be positive, got {}",
                    elem.thickness
                ),
            ));
        }
        let geom = TriangleGeometry::of(mesh, elem);
        let p0 = mesh.nodes[n0].position;
        let p1 = mesh.nodes[n1].position;
        let p2 = mesh.nodes[n2].position;
        let edge_sq = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
        let longest_sq = edge_sq(p0, p1).max(edge_sq(p1, p2)).max(edge_sq(p0, p2));
        if geom.area.abs() <= 1e-12 * longest_sq {
            return Err(Error::invalid_param(
                "elements",
                format!(
                    "element {e} is degenerate (area {} for nodes {:?})",
                    geom.area, elem.nodes
                ),
            ));
        }
    }
    for (b, bc) in mesh.dirichlet_bcs.iter().enumerate() {
        if bc.node >= n_nodes {
            return Err(Error::invalid_param(
                "dirichlet_bcs",
                format!(
                    "boundary condition {b} references node {}, but the mesh has {n_nodes} nodes",
                    bc.node
                ),
            ));
        }
        if bc.dof > 1 {
            return Err(Error::invalid_param(
                "dof",
                format!(
                    "boundary condition {b}: DOF must be 0 (x) or 1 (y), got {}",
                    bc.dof
                ),
            ));
        }
        if !bc.value.is_finite() {
            return Err(Error::invalid_param(
                "value",
                format!("boundary condition {b} prescribes a non-finite value"),
            ));
        }
    }
    for (l, nf) in mesh.nodal_forces.iter().enumerate() {
        if nf.node >= n_nodes {
            return Err(Error::invalid_param(
                "nodal_forces",
                format!(
                    "load {l} references node {}, but the mesh has {n_nodes} nodes",
                    nf.node
                ),
            ));
        }
        if !(nf.force[0].is_finite() && nf.force[1].is_finite()) {
            return Err(Error::invalid_param(
                "nodal_forces",
                format!("load {l} is not finite"),
            ));
        }
    }
    Ok(())
}

/// Prescribed displacement per global DOF (`None` for free DOFs).
///
/// Repeating a boundary condition with the same value is allowed; prescribing
/// two different values on one DOF is an error. Requires a validated mesh.
fn prescribed_dofs(mesh: &PyFemMesh) -> BridgeResult<Vec<Option<f64>>> {
    let mut prescribed: Vec<Option<f64>> = vec![None; mesh.num_dofs()];
    for bc in &mesh.dirichlet_bcs {
        let dof = bc.node * 2 + bc.dof;
        match prescribed[dof] {
            Some(previous) if previous != bc.value => {
                return Err(Error::invalid_param(
                    "dirichlet_bcs",
                    format!(
                        "node {} DOF {} is prescribed twice with different values ({previous} and {})",
                        bc.node, bc.dof, bc.value
                    ),
                ));
            }
            _ => prescribed[dof] = Some(bc.value),
        }
    }
    Ok(prescribed)
}

/// Flags for nodes that belong to at least one element. Requires a validated mesh.
fn nodes_used_by_elements(mesh: &PyFemMesh) -> Vec<bool> {
    let mut used = vec![false; mesh.num_nodes()];
    for elem in &mesh.elements {
        for &n in &elem.nodes {
            used[n] = true;
        }
    }
    used
}

/// Global DOF indices `[2 n0, 2 n0 + 1, 2 n1, 2 n1 + 1, 2 n2, 2 n2 + 1]` of an element.
fn element_dofs(elem: &PyFemElement) -> [usize; 6] {
    let [n0, n1, n2] = elem.nodes;
    [n0 * 2, n0 * 2 + 1, n1 * 2, n1 * 2 + 1, n2 * 2, n2 * 2 + 1]
}

/// Assemble the global stiffness matrix as COO triplets (duplicates are summed
/// by [`CsrMatrix::from_triplets`]). Requires a validated mesh.
fn assemble_stiffness_triplets(mesh: &PyFemMesh) -> Vec<(usize, usize, f64)> {
    let mut triplets = Vec::with_capacity(mesh.elements.len() * 36);
    for elem in &mesh.elements {
        let mat = &mesh.materials[elem.material_id];
        let ke = element_stiffness(mesh, elem, mat);
        let dofs = element_dofs(elem);
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                triplets.push((gi, gj, ke[i * 6 + j]));
            }
        }
    }
    triplets
}

/// Signed area and strain-displacement matrix of a CST element.
struct TriangleGeometry {
    /// Signed area (positive for counter-clockwise node order).
    area: f64,
    /// B matrix (3×6, row-major): [ε_xx, ε_yy, γ_xy] = B u_e.
    b_mat: [f64; 18],
}

impl TriangleGeometry {
    /// Compute the geometry of `elem`. Node indices must be valid.
    fn of(mesh: &PyFemMesh, elem: &PyFemElement) -> Self {
        let [n0, n1, n2] = elem.nodes;
        let [x0, y0] = mesh.nodes[n0].position;
        let [x1, y1] = mesh.nodes[n1].position;
        let [x2, y2] = mesh.nodes[n2].position;
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        if area == 0.0 {
            return Self {
                area,
                b_mat: [0.0; 18],
            };
        }
        let inv_2a = 1.0 / (2.0 * area);
        let b1x = (y1 - y2) * inv_2a;
        let b2x = (y2 - y0) * inv_2a;
        let b3x = (y0 - y1) * inv_2a;
        let b1y = (x2 - x1) * inv_2a;
        let b2y = (x0 - x2) * inv_2a;
        let b3y = (x1 - x0) * inv_2a;
        Self {
            area,
            b_mat: [
                b1x, 0.0, b2x, 0.0, b3x, 0.0, 0.0, b1y, 0.0, b2y, 0.0, b3y, b1y, b1x, b2y, b2x,
                b3y, b3x,
            ],
        }
    }
}

/// CST element stiffness Ke = Bᵀ D B · |A| · t (6×6, row-major).
fn element_stiffness(mesh: &PyFemMesh, elem: &PyFemElement, mat: &PyFemMaterial) -> [f64; 36] {
    let geom = TriangleGeometry::of(mesh, elem);
    let b_mat = geom.b_mat;
    let d = mat_plane_stress_d(mat);
    let factor = geom.area.abs() * elem.thickness;

    // D * B (3×6)
    let mut db = [0.0f64; 18];
    for i in 0..3 {
        for j in 0..6 {
            db[i * 6 + j] = (0..3).map(|kk| d[i * 3 + kk] * b_mat[kk * 6 + j]).sum();
        }
    }
    // Bᵀ (D B) * factor (6×6)
    let mut ke = [0.0f64; 36];
    for i in 0..6 {
        for j in 0..6 {
            let sum: f64 = (0..3).map(|kk| b_mat[kk * 6 + i] * db[kk * 6 + j]).sum();
            ke[i * 6 + j] = sum * factor;
        }
    }
    ke
}

/// Element stress vector [σx, σy, τxy] = D B u_e.
fn element_stress(
    mesh: &PyFemMesh,
    elem: &PyFemElement,
    mat: &PyFemMaterial,
    u: &[f64],
) -> [f64; 3] {
    let geom = TriangleGeometry::of(mesh, elem);
    let dofs = element_dofs(elem);
    let ue: [f64; 6] = dofs.map(|dof| u.get(dof).copied().unwrap_or(0.0));

    let mut strain = [0.0f64; 3];
    for (i, s) in strain.iter_mut().enumerate() {
        *s = (0..6).map(|j| geom.b_mat[i * 6 + j] * ue[j]).sum();
    }
    let d = mat_plane_stress_d(mat);
    let mut stress = [0.0f64; 3];
    for (i, s) in stress.iter_mut().enumerate() {
        *s = (0..3).map(|j| d[i * 3 + j] * strain[j]).sum();
    }
    stress
}

// ---------------------------------------------------------------------------
// Internal: plane stress D matrix returning fixed-size array
// ---------------------------------------------------------------------------

fn mat_plane_stress_d(mat: &PyFemMaterial) -> [f64; 9] {
    let e = mat.young_modulus;
    let nu = mat.poisson_ratio;
    let c = e / (1.0 - nu * nu);
    [
        c,
        c * nu,
        0.0,
        c * nu,
        c,
        0.0,
        0.0,
        0.0,
        c * (1.0 - nu) * 0.5,
    ]
}

// ===========================================================================
// Modal analysis (free-vibration eigenvalue problem)
// ===========================================================================

/// Result of a modal analysis.
#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct ModalAnalysisResult {
    /// Natural angular frequencies in rad/s (ascending order).
    #[pyo3(get)]
    pub natural_frequencies: Vec<f64>,
    /// Mode shapes over all DOFs (flat `[ux0, uy0, ux1, uy1, ...]`),
    /// mass-normalised so that φᵀ M φ = 1; constrained DOFs are zero.
    pub mode_shapes: Vec<Vec<f64>>,
    /// Number of modes extracted.
    #[pyo3(get)]
    pub num_modes: usize,
}

#[pymethods]
impl ModalAnalysisResult {
    /// Return the `i`-th natural frequency in Hz.
    pub fn frequency_hz(&self, i: usize) -> Option<f64> {
        self.natural_frequencies
            .get(i)
            .map(|&f| f / (2.0 * std::f64::consts::PI))
    }

    /// Return the period (in seconds) of the `i`-th mode.
    pub fn period_s(&self, i: usize) -> Option<f64> {
        self.frequency_hz(i).filter(|&f| f > 1e-15).map(|f| 1.0 / f)
    }

    /// Return the mode shape for mode `i` as a flat `Vec<f64>`.
    pub fn mode_shape(&self, i: usize) -> Option<Vec<f64>> {
        self.mode_shapes.get(i).cloned()
    }
}

/// Modal analyzer for undamped free vibration, K φ = ω² M φ.
///
/// The stiffness K is assembled from the CST elements and the mass M is the
/// row-sum lumped mass (ρ · t · |A| / 3 per element node, in both directions).
/// Constrained DOFs (any Dirichlet BC; prescribed values are irrelevant for
/// free vibration) and DOFs of nodes that belong to no element are removed.
/// The reduced pencil is transformed to the standard symmetric problem
/// A = M^-1/2 K M^-1/2, whose lowest `num_modes` eigenpairs are computed by
/// the engine's block inverse (subspace) iteration
/// [`subspace_iteration`] (`oxiphysics::fem::eigen`). The reduced operator is
/// held densely, so memory and time grow as n_free² and n_free³.
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct ModalAnalyzer {
    /// Number of modes to extract (lowest frequencies first).
    #[pyo3(get, set)]
    pub num_modes: usize,
    /// Maximum subspace-iteration sweeps.
    #[pyo3(get, set)]
    pub max_iter: usize,
    /// Relative eigen-residual tolerance ‖A y − λ y‖ ≤ tol · |λ|.
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl ModalAnalyzer {
    /// Create an analyzer that extracts up to `num_modes` modes.
    #[new]
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            max_iter: 500,
            tolerance: 1e-8,
        }
    }

    /// Run modal analysis on the mesh.
    ///
    /// Returns `None` if the mesh has no nodes or no elements. When every DOF
    /// is constrained the result holds zero modes.
    ///
    /// # Errors
    ///
    /// Raises an error for an invalid mesh (see `FemSolver.solve`), a
    /// non-positive density on a material used by an element, invalid
    /// iteration settings, or when the eigensolver cannot produce modes.
    pub fn analyze(&self, mesh: &PyFemMesh) -> BridgeResult<Option<ModalAnalysisResult>> {
        if mesh.num_nodes() == 0 || mesh.elements.is_empty() {
            return Ok(None);
        }
        if self.max_iter == 0 {
            return Err(Error::invalid_param(
                "max_iter",
                "the subspace-iteration budget must be positive",
            ));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(Error::invalid_param(
                "tolerance",
                format!(
                    "the eigen-residual tolerance must be positive and finite, got {}",
                    self.tolerance
                ),
            ));
        }
        validate_mesh(mesh)?;
        for (e, elem) in mesh.elements.iter().enumerate() {
            let rho = mesh.materials[elem.material_id].density;
            if !(rho.is_finite() && rho > 0.0) {
                return Err(Error::invalid_param(
                    "density",
                    format!(
                        "element {e} uses material {} with density {rho}; \
                         modal analysis needs a positive density",
                        elem.material_id
                    ),
                ));
            }
        }
        let n_dofs = mesh.num_dofs();

        // --- Lumped mass: ρ t |A| / 3 per element node and direction ---
        let mut mass = vec![0.0f64; n_dofs];
        for elem in &mesh.elements {
            let rho = mesh.materials[elem.material_id].density;
            let geom = TriangleGeometry::of(mesh, elem);
            let nodal = rho * elem.thickness * geom.area.abs() / 3.0;
            for dof in element_dofs(elem) {
                mass[dof] += nodal;
            }
        }

        // --- Free DOFs: unconstrained and carrying mass ---
        let prescribed = prescribed_dofs(mesh)?;
        let mut reduced_index: Vec<Option<usize>> = vec![None; n_dofs];
        let mut free_dofs: Vec<usize> = Vec::new();
        for dof in 0..n_dofs {
            if prescribed[dof].is_none() && mass[dof] > 0.0 {
                reduced_index[dof] = Some(free_dofs.len());
                free_dofs.push(dof);
            }
        }
        let n_free = free_dofs.len();
        let modes = self.num_modes.min(n_free);
        if modes == 0 {
            return Ok(Some(ModalAnalysisResult {
                natural_frequencies: Vec::new(),
                mode_shapes: Vec::new(),
                num_modes: 0,
            }));
        }

        // --- A = M^-1/2 K M^-1/2 on the free DOFs ---
        let inv_sqrt_mass: Vec<f64> = free_dofs.iter().map(|&d| 1.0 / mass[d].sqrt()).collect();
        let mut a = vec![vec![0.0f64; n_free]; n_free];
        for (i, j, v) in assemble_stiffness_triplets(mesh) {
            if let (Some(ri), Some(rj)) = (reduced_index[i], reduced_index[j]) {
                a[ri][rj] += v * inv_sqrt_mass[ri] * inv_sqrt_mass[rj];
            }
        }

        let pairs = subspace_iteration(&a, n_free, modes, self.max_iter, self.tolerance);
        if pairs.is_empty() {
            return Err(Error::general(
                "modal analysis: the eigensolver returned no eigenpairs \
                 (non-finite stiffness or mass)",
            ));
        }

        let mut natural_frequencies = Vec::with_capacity(pairs.len());
        let mut mode_shapes = Vec::with_capacity(pairs.len());
        for (lambda, y) in pairs {
            natural_frequencies.push(lambda.max(0.0).sqrt());
            // φ = M^-1/2 y; with a unit y this gives φᵀ M φ = 1.
            let mut phi = vec![0.0f64; n_dofs];
            for (r, &dof) in free_dofs.iter().enumerate() {
                phi[dof] = y[r] * inv_sqrt_mass[r];
            }
            mode_shapes.push(phi);
        }

        Ok(Some(ModalAnalysisResult {
            num_modes: natural_frequencies.len(),
            natural_frequencies,
            mode_shapes,
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "fem_api_tests.rs"]
mod tests;

/// Register all `fem` classes into a Python sub-module.
///
/// Called from the top-level `#[pymodule]` in `lib.rs`.
pub fn register_fem_module(parent: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    let child = pyo3::types::PyModule::new(parent.py(), "fem")?;
    child.add_class::<PyFemMaterial>()?;
    child.add_class::<PyFemNode>()?;
    child.add_class::<PyFemElement>()?;
    child.add_class::<PyFemDirichletBC>()?;
    child.add_class::<PyFemNodalForce>()?;
    child.add_class::<PyFemMesh>()?;
    child.add_class::<PyFemSolveResult>()?;
    child.add_class::<PyFemSolver>()?;
    child.add_class::<ModalAnalysisResult>()?;
    child.add_class::<ModalAnalyzer>()?;
    parent.add_submodule(&child)?;
    Ok(())
}
