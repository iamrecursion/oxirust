//! Dirichlet-Neumann partitioned ALE fluid-structure interaction.
//!
//! [`AleFsiSolver`] couples a quasi-static (inertia-free) Stokes fluid on a moving
//! triangular mesh with a lumped mass-spring structure along the fluid-structure
//! interface.  One time step is a partitioned fixed-point iteration on the
//! interface displacement δ (the Dirichlet-Neumann scheme):
//!
//! 1. **Mesh motion.**  The interface displacement δ_k·n_k moves the mesh; the
//!    interior nodes follow from a harmonic (Laplace) extension on the reference
//!    mesh, so the mesh velocity is w = (x − xⁿ)/Δt.
//! 2. **Fluid (Dirichlet).**  The Stokes problem with viscosity μ is solved on the
//!    moved mesh with the interface velocity w = (δ − δⁿ)/Δt, plug inflow, no-slip
//!    walls and the pressure p₀ on the open boundary (module `stokes`).  The
//!    force on every interface node is the consistent momentum residual, i.e. the
//!    pressure and the viscous traction −p·n + μ(∇v + ∇vᵀ)·n integrated against
//!    the nodal basis functions.
//! 3. **Structure (Neumann).**  Every interface node is a damped oscillator
//!    m·a + c·v + k·u = F along its reference normal, integrated with the implicit
//!    Newmark method (module `structure`) under the fluid force of the current
//!    iterate; this yields the displacement δ̃ the structure takes for that load.
//! 4. **Relaxation.**  The interface displacement is updated with Aitken's dynamic
//!    relaxation, δ ← δ + ω(δ̃ − δ), until ‖δ̃ − δ‖ < tol·(1 + ‖δ‖) or
//!    [`AleFsiSolver::max_iter`] iterations were spent.
//!
//! The fluid is a Stokes fluid: there is no convective or local inertia, `rho_f`
//! only enters through the body force ρ_f·g, and the mesh velocity acts through
//! the interface boundary condition only.  The fluid and mesh-motion systems are
//! solved with dense LU factorisations, so the solver is meant for small and
//! medium meshes (a few hundred nodes).
//!
//! A scalar Aitken factor handles moderately strong coupling (the viscous drag of
//! the fluid acting like a structural damper) in a few tens of iterations.  A very
//! light structure in a very viscous fluid drives the fixed-point map far outside
//! its stability region; the iteration then stalls (`converged == false`) or
//! tangles the mesh (an error).  Reduce `dt` or increase the structural mass in
//! that case.

mod geometry;
mod mesh_motion;
mod stokes;
mod structure;
#[cfg(test)]
mod tests;

use super::types::{AitkenRelaxation, AleInterface, AleNode};
use crate::error::{Error, Result};
use geometry::{Triangle, element_corners};
use mesh_motion::HarmonicExtension;
use std::collections::BTreeMap;
use stokes::{BoundaryEdge, NodeKind, StokesProblem, solve_stokes};
use structure::{ModalState, NewmarkScheme, Oscillator};

/// An element must keep at least this fraction of its reference area (and its
/// orientation) when the mesh moves, otherwise the mesh counts as tangled.
const MIN_AREA_RATIO: f64 = 1e-3;

/// Convergence report of one coupled ALE-FSI time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleFsiReport {
    /// Number of Dirichlet-Neumann coupling iterations (fluid solves) performed.
    pub iterations: usize,
    /// Norm ‖δ̃ − δ‖₂ of the interface-displacement residual of the last
    /// iteration \[m\].
    pub interface_residual: f64,
    /// Aitken relaxation factor of the last iteration.
    pub omega: f64,
    /// Whether the residual dropped below the tolerance within the iteration limit.
    pub converged: bool,
}

/// Dirichlet-Neumann partitioned ALE-FSI solver (Stokes fluid, lumped
/// mass-spring structure on the interface); see the [module
/// documentation](self) for the scheme.
///
/// The fluid mesh is given by the nodes ([`AleNode`], reference position
/// `material_pos`) and the triangles [`Self::elements`].  The boundary roles are:
///
/// * **interface** — [`AleInterface::node_indices`], moving with the structure;
/// * **inlet** — [`Self::inlet_nodes`], prescribed plug velocity;
/// * **wall** — [`Self::wall_nodes`], no slip;
/// * **open boundary** — every other boundary edge that touches a node without
///   prescribed velocity; it carries the pressure [`Self::outlet_pressure`].
///   At least one such edge is required, because an incompressible fluid whose
///   boundary velocity is prescribed everywhere has no unique pressure.
///
/// A node listed several times takes the first role in the order interface,
/// inlet, wall.  [`Self::rectangular_channel`] builds a structured channel.
#[derive(Debug, Clone)]
pub struct AleFsiSolver {
    /// All ALE nodes of the fluid mesh (the interface nodes are a subset).
    pub nodes: Vec<AleNode>,
    /// FSI interface: node indices and the unit normals pointing from the
    /// structure into the fluid; positive deflection moves a node into the fluid.
    pub interface: AleInterface,
    /// Triangles (node indices) of the fluid mesh.
    pub elements: Vec<[usize; 3]>,
    /// Inlet nodes; they carry the plug velocity `(inlet_velocity, 0)` given to
    /// [`Self::step`] and [`Self::solve_fluid`].
    pub inlet_nodes: Vec<usize>,
    /// No-slip wall nodes.
    pub wall_nodes: Vec<usize>,
    /// Pressure p₀ imposed on the open boundary \[Pa\].
    pub outlet_pressure: f64,
    /// Gravitational acceleration g \[m/s²\]; the fluid carries the body force ρ_f·g.
    pub gravity: [f64; 2],
    /// Structural mass density \[kg/m³\]
    pub struct_rho: f64,
    /// Structural thickness \[m\]; the interface carries ρ_s·thickness mass per
    /// unit area.
    pub thickness: f64,
    /// Stiffness of the elastic foundation per unit interface area \[Pa/m\]: an
    /// interface node with tributary length `ds` has the spring constant
    /// `struct_stiffness·ds` (per unit depth), so a uniform pressure `p` deflects
    /// the structure statically by `p / struct_stiffness`.
    pub struct_stiffness: f64,
    /// Mass-proportional structural damping coefficient \[1/s\] (c = struct_damping·m).
    pub struct_damping: f64,
    /// Newmark parameter β (> 0); ¼ with γ = ½ is the undamped average-acceleration scheme.
    pub newmark_beta: f64,
    /// Newmark parameter γ (≥ ½); γ > ½ adds algorithmic damping.
    pub newmark_gamma: f64,
    /// Fluid dynamic viscosity \[Pa·s\]
    pub mu: f64,
    /// Fluid density \[kg/m³\]
    pub rho_f: f64,
    /// Initial Aitken relaxation factor of every coupling iteration (in (0, 1]).
    pub omega: f64,
    /// Convergence tolerance of the coupling iteration: ‖δ̃ − δ‖₂ < tol·(1 + ‖δ‖₂).
    pub tol: f64,
    /// Maximum number of coupling iterations per time step.
    pub max_iter: usize,
    /// Normal velocity of every interface node \[m/s\] (state of the structure).
    pub struct_velocity: Vec<f64>,
    /// Normal acceleration of every interface node \[m/s²\] (state of the structure).
    pub struct_acceleration: Vec<f64>,
    /// Force of the fluid on every interface node from the last fluid solve
    /// \[N per unit depth\]: pressure and viscous traction integrated against the
    /// nodal basis function.  An interface node that is also the end of a wall or
    /// inlet edge additionally carries the fluid force on the adjacent half of
    /// that edge; only the component along the interface normal loads the
    /// structure.
    pub interface_force: Vec<[f64; 2]>,
    /// Whether `struct_acceleration` holds the dynamic-equilibrium acceleration of
    /// the current time level (computed lazily before the first step).
    accel_initialised: bool,
}

/// Boundary roles and reference data derived from the solver configuration.
#[derive(Debug, Clone)]
struct Topology {
    /// Boundary role of every node.
    kinds: Vec<NodeKind>,
    /// Whether the node lies on the boundary of the fluid mesh.
    on_boundary: Vec<bool>,
    /// Open-boundary edges (boundary edges with a velocity-free node).
    open_edges: Vec<BoundaryEdge>,
    /// Tributary length of every interface node in the reference configuration.
    interface_length: Vec<f64>,
}

/// Everything that depends only on the (fixed) configuration of one step.
struct Context {
    topology: Topology,
    /// Reference node positions.
    reference: Vec<[f64; 2]>,
    /// Signed twice the area of every element in the reference mesh.
    reference_twice_area: Vec<f64>,
    /// Harmonic mesh-motion operator.
    extension: HarmonicExtension,
    /// Unit interface normals (structure → fluid).
    normals: Vec<[f64; 2]>,
    /// One oscillator per interface node.
    oscillators: Vec<Oscillator>,
    /// Longest reference edge, a length scale of the mesh.
    length_scale: f64,
}

/// Elements that own an edge, as (first node, second node, node opposite to the
/// edge) in the element's own node order.
type EdgeOwners = Vec<(usize, usize, usize)>;

/// Outcome of one coupling iteration.
struct CouplingIterate {
    /// Fluid state of the iteration.
    fluid: FluidState,
    /// Structural response (Newmark update) to the fluid force.
    response: Vec<ModalState>,
    /// Norm of the interface-displacement residual.
    residual_norm: f64,
    /// Whether the residual satisfies the tolerance.
    converged: bool,
    /// Iteration number (1-based).
    iterations: usize,
}

/// Result of one fluid solve on a moved mesh.
struct FluidState {
    velocity: Vec<[f64; 2]>,
    pressure: Vec<f64>,
    /// Force of the fluid on every interface node.
    force: Vec<[f64; 2]>,
    /// Normal component of `force` per interface node.
    normal_force: Vec<f64>,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn positive_finite(value: f64, name: &str) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "{name} must be positive and finite, got {value}"
        )))
    }
}

fn non_negative_finite(value: f64, name: &str) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "{name} must be non-negative and finite, got {value}"
        )))
    }
}

impl AleFsiSolver {
    /// Create a solver for the given nodes and interface.
    ///
    /// The fluid mesh, the inlet and wall nodes still have to be provided
    /// ([`Self::elements`], [`Self::inlet_nodes`], [`Self::wall_nodes`]), see
    /// [`Self::rectangular_channel`] for a ready-made mesh.  The defaults are a
    /// steel plate of 1 cm on a 10⁶ Pa/m foundation under water without gravity.
    pub fn new(nodes: Vec<AleNode>, interface: AleInterface) -> Self {
        let n_interface = interface.node_indices.len();
        Self {
            nodes,
            interface,
            elements: Vec::new(),
            inlet_nodes: Vec::new(),
            wall_nodes: Vec::new(),
            outlet_pressure: 0.0,
            gravity: [0.0, 0.0],
            struct_rho: 7850.0,
            thickness: 0.01,
            struct_stiffness: 1.0e6,
            struct_damping: 0.0,
            newmark_beta: 0.25,
            newmark_gamma: 0.5,
            mu: 1e-3,
            rho_f: 1000.0,
            omega: 0.5,
            tol: 1e-9,
            max_iter: 50,
            struct_velocity: vec![0.0; n_interface],
            struct_acceleration: vec![0.0; n_interface],
            interface_force: vec![[0.0; 2]; n_interface],
            accel_initialised: false,
        }
    }

    /// Index of grid node `(i, j)` in the mesh of [`Self::rectangular_channel`]
    /// with `nx` cells in x.
    pub fn grid_node(nx: usize, i: usize, j: usize) -> usize {
        j * (nx + 1) + i
    }

    /// Structured channel `[0, length] × [0, height]` with `nx × ny` cells, each
    /// split into two triangles.
    ///
    /// The bottom edge (y = 0) is the fluid-structure interface (normals `(0, 1)`,
    /// into the fluid), the left edge the inlet (plug flow; its two corner nodes
    /// belong to the interface and the lid), the top edge a no-slip lid, and the
    /// right edge the open outlet with [`Self::outlet_pressure`].
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for `nx == 0`, `ny == 0` or a non-positive size.
    pub fn rectangular_channel(nx: usize, ny: usize, length: f64, height: f64) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(Error::InvalidInput(format!(
                "the channel needs at least one cell in each direction, got {nx}×{ny}"
            )));
        }
        positive_finite(length, "channel length")?;
        positive_finite(height, "channel height")?;
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push(AleNode::new(
                    length * i as f64 / nx as f64,
                    height * j as f64 / ny as f64,
                ));
            }
        }
        let node = |i: usize, j: usize| Self::grid_node(nx, i, j);
        let interface = AleInterface {
            node_indices: (0..=nx).map(|i| node(i, 0)).collect(),
            normals: vec![[0.0, 1.0]; nx + 1],
        };
        let mut solver = Self::new(nodes, interface);
        for j in 0..ny {
            for i in 0..nx {
                solver
                    .elements
                    .push([node(i, j), node(i + 1, j), node(i + 1, j + 1)]);
                solver
                    .elements
                    .push([node(i, j), node(i + 1, j + 1), node(i, j + 1)]);
            }
        }
        solver.inlet_nodes = (1..ny).map(|j| node(0, j)).collect();
        solver.wall_nodes = (0..=nx).map(|i| node(i, ny)).collect();
        Ok(solver)
    }

    // -----------------------------------------------------------------------
    // Validation and derived data
    // -----------------------------------------------------------------------

    /// Check the whole configuration (mesh, boundary roles, material and
    /// numerical parameters) without changing anything.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] describing the first inconsistency, or the errors
    /// of the mesh-motion factorisation.
    pub fn validate(&self) -> Result<()> {
        self.prepare().map(|_| ())
    }

    fn check_parameters(&self) -> Result<()> {
        positive_finite(self.mu, "viscosity μ")?;
        non_negative_finite(self.rho_f, "fluid density")?;
        positive_finite(self.struct_rho, "structural density")?;
        positive_finite(self.thickness, "structural thickness")?;
        non_negative_finite(self.struct_stiffness, "structural stiffness")?;
        non_negative_finite(self.struct_damping, "structural damping")?;
        if !self.outlet_pressure.is_finite() || self.gravity.iter().any(|g| !g.is_finite()) {
            return Err(Error::InvalidInput(
                "outlet pressure and gravity must be finite".into(),
            ));
        }
        if !(self.newmark_beta.is_finite() && self.newmark_beta > 0.0) {
            return Err(Error::InvalidInput(format!(
                "Newmark β must be positive (an implicit scheme is required), got {}",
                self.newmark_beta
            )));
        }
        if !(self.newmark_gamma.is_finite() && self.newmark_gamma >= 0.5) {
            return Err(Error::InvalidInput(format!(
                "Newmark γ must be at least ½, got {}",
                self.newmark_gamma
            )));
        }
        if !(self.omega.is_finite() && self.omega > 0.0 && self.omega <= 1.0) {
            return Err(Error::InvalidInput(format!(
                "the initial relaxation factor must lie in (0, 1], got {}",
                self.omega
            )));
        }
        positive_finite(self.tol, "coupling tolerance")?;
        if self.max_iter == 0 {
            return Err(Error::InvalidInput(
                "at least one coupling iteration is required".into(),
            ));
        }
        Ok(())
    }

    /// Classify the nodes, find the boundary and the open edges and measure the
    /// interface.
    fn build_topology(&self) -> Result<Topology> {
        let n = self.nodes.len();
        if self.elements.is_empty() {
            return Err(Error::InvalidInput(
                "the fluid mesh has no elements; set `elements` (see `rectangular_channel`)".into(),
            ));
        }
        let mut node_used = vec![false; n];
        let mut edges: BTreeMap<(usize, usize), EdgeOwners> = BTreeMap::new();
        for (e, element) in self.elements.iter().enumerate() {
            if element.iter().any(|&node| node >= n) {
                return Err(Error::InvalidInput(format!(
                    "element {e} references a node outside the mesh of {n} nodes"
                )));
            }
            if element[0] == element[1] || element[1] == element[2] || element[0] == element[2] {
                return Err(Error::InvalidInput(format!(
                    "element {e} repeats a node: {element:?}"
                )));
            }
            for (u, v, opposite) in [
                (element[0], element[1], element[2]),
                (element[1], element[2], element[0]),
                (element[2], element[0], element[1]),
            ] {
                node_used[u] = true;
                node_used[v] = true;
                edges
                    .entry((u.min(v), u.max(v)))
                    .or_default()
                    .push((u, v, opposite));
            }
        }
        if let Some(orphan) = node_used.iter().position(|used| !used) {
            return Err(Error::InvalidInput(format!(
                "node {orphan} does not belong to any element"
            )));
        }
        let interface = &self.interface;
        if interface.node_indices.len() != interface.normals.len()
            || interface.node_indices.len() != self.struct_velocity.len()
            || interface.node_indices.len() != self.struct_acceleration.len()
        {
            return Err(Error::InvalidInput(format!(
                "the interface has {} nodes, {} normals, {} velocities and {} accelerations",
                interface.node_indices.len(),
                interface.normals.len(),
                self.struct_velocity.len(),
                self.struct_acceleration.len()
            )));
        }
        if interface.node_indices.is_empty() {
            return Err(Error::InvalidInput("the interface has no nodes".into()));
        }
        let mut kinds = vec![NodeKind::Free; n];
        let assign = |list: &[usize], kind: NodeKind, kinds: &mut [NodeKind], what: &str| {
            for &node in list {
                let slot = kinds.get_mut(node).ok_or_else(|| {
                    Error::InvalidInput(format!("{what} node {node} is outside the mesh"))
                })?;
                if *slot == NodeKind::Free {
                    *slot = kind;
                }
            }
            Ok::<(), Error>(())
        };
        let mut seen = vec![false; n];
        for &node in &interface.node_indices {
            if node >= n {
                return Err(Error::InvalidInput(format!(
                    "interface node {node} is outside the mesh"
                )));
            }
            if seen[node] {
                return Err(Error::InvalidInput(format!(
                    "interface node {node} is listed twice"
                )));
            }
            seen[node] = true;
        }
        assign(
            &interface.node_indices,
            NodeKind::Interface,
            &mut kinds,
            "interface",
        )?;
        assign(&self.inlet_nodes, NodeKind::Inlet, &mut kinds, "inlet")?;
        assign(&self.wall_nodes, NodeKind::Wall, &mut kinds, "wall")?;

        let mut on_boundary = vec![false; n];
        let mut open_edges = Vec::new();
        let mut interface_length = vec![0.0; n];
        for ((u, v), owners) in &edges {
            match owners.as_slice() {
                [owner] => {
                    on_boundary[*u] = true;
                    on_boundary[*v] = true;
                    if kinds[*u] == NodeKind::Free || kinds[*v] == NodeKind::Free {
                        open_edges.push(BoundaryEdge {
                            nodes: [owner.0, owner.1],
                            opposite: owner.2,
                        });
                    }
                    if kinds[*u] == NodeKind::Interface && kinds[*v] == NodeKind::Interface {
                        let a = self.nodes[*u].material_pos;
                        let b = self.nodes[*v].material_pos;
                        let half = 0.5 * (b[0] - a[0]).hypot(b[1] - a[1]);
                        interface_length[*u] += half;
                        interface_length[*v] += half;
                    }
                }
                [_, _] => {}
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "edge ({u}, {v}) belongs to {} elements; the mesh is not a manifold",
                        owners.len()
                    )));
                }
            }
        }
        if open_edges.is_empty() {
            return Err(Error::InvalidInput(
                "the fluid domain has no open boundary (every boundary node has a prescribed \
                 velocity); the pressure would be undetermined"
                    .into(),
            ));
        }
        for &node in &interface.node_indices {
            if !on_boundary[node] {
                return Err(Error::InvalidInput(format!(
                    "interface node {node} does not lie on the boundary of the fluid mesh"
                )));
            }
        }
        let interface_length: Vec<f64> = interface
            .node_indices
            .iter()
            .map(|&node| interface_length[node])
            .collect();
        if let Some(k) = interface_length
            .iter()
            .position(|l| !l.is_finite() || *l <= 0.0)
        {
            return Err(Error::InvalidInput(format!(
                "interface node {} has no adjacent interface edge",
                interface.node_indices[k]
            )));
        }
        Ok(Topology {
            kinds,
            on_boundary,
            open_edges,
            interface_length,
        })
    }

    /// Validate the configuration and derive the fixed data of a step.
    fn prepare(&self) -> Result<Context> {
        self.check_parameters()?;
        let topology = self.build_topology()?;
        let reference: Vec<[f64; 2]> = self.nodes.iter().map(|node| node.material_pos).collect();
        let mut reference_twice_area = Vec::with_capacity(self.elements.len());
        let mut length_scale = 0.0_f64;
        for (e, element) in self.elements.iter().enumerate() {
            let triangle = element_corners(&reference, element)
                .and_then(|corners| Triangle::new(&corners))
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "element {e} is degenerate or not finite in the reference mesh"
                    ))
                })?;
            reference_twice_area.push(triangle.twice_signed_area);
            length_scale = length_scale.max(triangle.longest_edge);
        }
        let mut normals = Vec::with_capacity(self.interface.normals.len());
        for (k, normal) in self.interface.normals.iter().enumerate() {
            let length = normal[0].hypot(normal[1]);
            if !length.is_finite() || length <= 0.0 {
                return Err(Error::InvalidInput(format!(
                    "the normal of interface node {} must be a finite, non-zero vector",
                    self.interface.node_indices[k]
                )));
            }
            normals.push([normal[0] / length, normal[1] / length]);
        }
        let extension = HarmonicExtension::new(&reference, &self.elements, &topology.on_boundary)?;
        let areal_mass = self.struct_rho * self.thickness;
        let oscillators = topology
            .interface_length
            .iter()
            .map(|&ds| Oscillator {
                mass: areal_mass * ds,
                damping: self.struct_damping * areal_mass * ds,
                stiffness: self.struct_stiffness * ds,
            })
            .collect();
        Ok(Context {
            topology,
            reference,
            reference_twice_area,
            extension,
            normals,
            oscillators,
            length_scale,
        })
    }

    // -----------------------------------------------------------------------
    // Interface state
    // -----------------------------------------------------------------------

    /// Normal deflection δ_k = u_k·n_k of every interface node (positive into the
    /// fluid), read from the node displacements.
    pub fn interface_deflection(&self) -> Vec<f64> {
        self.interface
            .node_indices
            .iter()
            .zip(self.interface.normals.iter())
            .map(|(&node, normal)| {
                let length = normal[0].hypot(normal[1]);
                match self.nodes.get(node) {
                    Some(n) if length > 0.0 => {
                        (n.displacement[0] * normal[0] + n.displacement[1] * normal[1]) / length
                    }
                    _ => 0.0,
                }
            })
            .collect()
    }

    /// Interface deflections for a step; the displacement of an interface node
    /// must be parallel to its normal because the structure moves along it.
    fn checked_deflection(&self, ctx: &Context) -> Result<Vec<f64>> {
        let mut delta = Vec::with_capacity(ctx.normals.len());
        for (k, (&node, normal)) in self
            .interface
            .node_indices
            .iter()
            .zip(ctx.normals.iter())
            .enumerate()
        {
            let d = self.nodes[node].displacement;
            let along = d[0] * normal[0] + d[1] * normal[1];
            let across = [d[0] - along * normal[0], d[1] - along * normal[1]];
            if !along.is_finite() || across[0].hypot(across[1]) > 1e-9 * ctx.length_scale {
                return Err(Error::InvalidInput(format!(
                    "the displacement {d:?} of interface node {k} (node {node}) is not along its \
                     normal {normal:?}"
                )));
            }
            delta.push(along);
        }
        Ok(delta)
    }

    /// Node positions for the interface deflections `delta`: reference mesh plus
    /// the harmonic extension of δ_k·n_k.  Fails if the mesh is tangled.
    fn positions_for(&self, ctx: &Context, delta: &[f64]) -> Result<Vec<[f64; 2]>> {
        let mut prescribed = vec![[0.0; 2]; ctx.reference.len()];
        for ((&node, normal), &d) in self
            .interface
            .node_indices
            .iter()
            .zip(ctx.normals.iter())
            .zip(delta.iter())
        {
            prescribed[node] = [d * normal[0], d * normal[1]];
        }
        let displacement = ctx.extension.extend(&prescribed);
        let positions: Vec<[f64; 2]> = ctx
            .reference
            .iter()
            .zip(displacement.iter())
            .map(|(x, d)| [x[0] + d[0], x[1] + d[1]])
            .collect();
        for (e, element) in self.elements.iter().enumerate() {
            let reference_area = ctx.reference_twice_area[e];
            let healthy = element_corners(&positions, element)
                .and_then(|corners| Triangle::new(&corners))
                .is_some_and(|t| {
                    t.twice_signed_area * reference_area > 0.0
                        && t.twice_signed_area.abs() >= MIN_AREA_RATIO * reference_area.abs()
                });
            if !healthy {
                return Err(Error::General(format!(
                    "the ALE mesh is tangled: element {e} lost its orientation or more than \
                     {:.1}% of its area; reduce the time step or the load",
                    100.0 * (1.0 - MIN_AREA_RATIO)
                )));
            }
        }
        Ok(positions)
    }

    // -----------------------------------------------------------------------
    // Fluid
    // -----------------------------------------------------------------------

    /// Solve the Stokes problem on the mesh moved by `delta` with the interface
    /// velocity `interface_velocity` (normal components).
    fn fluid_state(
        &self,
        ctx: &Context,
        delta: &[f64],
        interface_velocity: &[f64],
        inlet_velocity: f64,
    ) -> Result<(Vec<[f64; 2]>, FluidState)> {
        let positions = self.positions_for(ctx, delta)?;
        let mut prescribed = vec![[0.0; 2]; positions.len()];
        for (node, kind) in ctx.topology.kinds.iter().enumerate() {
            if *kind == NodeKind::Inlet {
                prescribed[node] = [inlet_velocity, 0.0];
            }
        }
        for ((&node, normal), &w) in self
            .interface
            .node_indices
            .iter()
            .zip(ctx.normals.iter())
            .zip(interface_velocity.iter())
        {
            prescribed[node] = [w * normal[0], w * normal[1]];
        }
        let solution = solve_stokes(&StokesProblem {
            positions: &positions,
            elements: &self.elements,
            kinds: &ctx.topology.kinds,
            prescribed: &prescribed,
            open_edges: &ctx.topology.open_edges,
            viscosity: self.mu,
            body_force: [self.rho_f * self.gravity[0], self.rho_f * self.gravity[1]],
            open_pressure: self.outlet_pressure,
        })?;
        let force: Vec<[f64; 2]> = self
            .interface
            .node_indices
            .iter()
            .map(|&node| solution.boundary_force[node])
            .collect();
        let normal_force = force
            .iter()
            .zip(ctx.normals.iter())
            .map(|(f, n)| f[0] * n[0] + f[1] * n[1])
            .collect();
        Ok((
            positions,
            FluidState {
                velocity: solution.velocity,
                pressure: solution.pressure,
                force,
                normal_force,
            },
        ))
    }

    /// Write the moved mesh and the fluid fields into the nodes.
    fn store_fluid(
        &mut self,
        positions: &[[f64; 2]],
        fluid: FluidState,
        mesh_velocity: Option<Vec<[f64; 2]>>,
    ) {
        for (index, node) in self.nodes.iter_mut().enumerate() {
            node.spatial_pos = positions[index];
            node.displacement = [
                positions[index][0] - node.material_pos[0],
                positions[index][1] - node.material_pos[1],
            ];
            node.fluid_velocity = fluid.velocity[index];
            node.pressure = fluid.pressure[index];
            if let Some(velocity) = &mesh_velocity {
                node.mesh_velocity = velocity[index];
            }
        }
        self.interface_force = fluid.force;
    }

    /// Solve the fluid problem on the current mesh without advancing time.
    ///
    /// The mesh is moved to the current interface deflection, the interface
    /// velocity is [`Self::struct_velocity`], and the nodal velocity, pressure and
    /// [`Self::interface_force`] are updated.  This is the "Dirichlet" half of a
    /// coupling iteration and is useful to evaluate the load of a prescribed
    /// motion.  `inlet_velocity` is the plug velocity of the inlet nodes.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for an inconsistent configuration or a displacement
    /// that is not along the interface normal, [`Error::General`] for a tangled
    /// mesh, [`Error::SingularSystem`] if the fluid system cannot be solved.  The
    /// solver is unchanged on error.
    pub fn solve_fluid(&mut self, inlet_velocity: f64) -> Result<()> {
        if !inlet_velocity.is_finite() {
            return Err(Error::InvalidInput(format!(
                "the inlet velocity must be finite, got {inlet_velocity}"
            )));
        }
        let ctx = self.prepare()?;
        let delta = self.checked_deflection(&ctx)?;
        let (positions, fluid) =
            self.fluid_state(&ctx, &delta, &self.struct_velocity, inlet_velocity)?;
        self.store_fluid(&positions, fluid, None);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Time stepping
    // -----------------------------------------------------------------------

    /// Advance the coupled problem by one time step `dt` with the Dirichlet-Neumann
    /// iteration described in the [module documentation](self); `inlet_velocity`
    /// is the plug velocity of the inlet nodes.
    ///
    /// The displacements of the interface nodes (along their normals), the
    /// [`Self::struct_velocity`] and the [`Self::struct_acceleration`] define the
    /// state at the start of the step.  Before the very first step the
    /// acceleration is replaced by the dynamic-equilibrium value m·a = F − c·v −
    /// k·u with the fluid force F of the initial state.  On return the nodes hold
    /// the new mesh (positions, displacement, mesh velocity) and the fluid
    /// velocity and pressure of the last coupling iteration.
    ///
    /// Not reaching the tolerance within [`Self::max_iter`] iterations is not an
    /// error: the last iterate is kept and [`AleFsiReport::converged`] is `false`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for an invalid `dt`, configuration or initial state,
    /// [`Error::General`] if the mesh gets tangled during the iteration,
    /// [`Error::SingularSystem`] if a fluid system cannot be solved.  The solver is
    /// unchanged on error.
    pub fn step(&mut self, dt: f64, inlet_velocity: f64) -> Result<AleFsiReport> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::InvalidInput(format!(
                "the time step must be finite and positive, got {dt}"
            )));
        }
        if !inlet_velocity.is_finite() {
            return Err(Error::InvalidInput(format!(
                "the inlet velocity must be finite, got {inlet_velocity}"
            )));
        }
        let ctx = self.prepare()?;
        let n_interface = self.interface.node_indices.len();
        let delta_n = self.checked_deflection(&ctx)?;
        let velocity_n = self.struct_velocity.clone();
        let mut acceleration_n = self.struct_acceleration.clone();
        if !self.accel_initialised {
            let (_, rest) = self.fluid_state(&ctx, &delta_n, &velocity_n, inlet_velocity)?;
            for k in 0..n_interface {
                acceleration_n[k] = ctx.oscillators[k].equilibrium_acceleration(
                    delta_n[k],
                    velocity_n[k],
                    rest.normal_force[k],
                );
            }
        }
        let scheme = NewmarkScheme {
            beta: self.newmark_beta,
            gamma: self.newmark_gamma,
            dt,
        };
        let state_n: Vec<ModalState> = (0..n_interface)
            .map(|k| ModalState {
                displacement: delta_n[k],
                velocity: velocity_n[k],
                acceleration: acceleration_n[k],
            })
            .collect();
        let old_positions = self.positions_for(&ctx, &delta_n)?;

        // Dirichlet-Neumann iteration on the interface displacement.
        let mut delta_iterate: Vec<f64> = (0..n_interface)
            .map(|k| delta_n[k] + dt * velocity_n[k])
            .collect();
        let mut aitken = AitkenRelaxation::new(n_interface, self.omega);
        let mut last: Option<CouplingIterate> = None;
        for iteration in 1..=self.max_iter {
            let interface_velocity: Vec<f64> = delta_iterate
                .iter()
                .zip(delta_n.iter())
                .map(|(d, d0)| (d - d0) / dt)
                .collect();
            let (_, fluid) =
                self.fluid_state(&ctx, &delta_iterate, &interface_velocity, inlet_velocity)?;
            let response: Vec<ModalState> = (0..n_interface)
                .map(|k| ctx.oscillators[k].advance(&state_n[k], fluid.normal_force[k], &scheme))
                .collect();
            let predicted: Vec<f64> = response.iter().map(|s| s.displacement).collect();
            let residual: Vec<f64> = predicted
                .iter()
                .zip(delta_iterate.iter())
                .map(|(p, d)| p - d)
                .collect();
            let residual_norm = norm(&residual);
            aitken.residual_prev = std::mem::replace(&mut aitken.residual_curr, residual);
            aitken.solution_prev = delta_iterate.clone();
            let converged = aitken.is_converged(self.tol);
            if iteration > 1 && !converged {
                aitken.update_omega();
            }
            last = Some(CouplingIterate {
                fluid,
                response,
                residual_norm,
                converged,
                iterations: iteration,
            });
            if converged {
                break;
            }
            delta_iterate = aitken.relax(&delta_iterate, &predicted);
        }
        let Some(CouplingIterate {
            fluid,
            response,
            residual_norm,
            converged,
            iterations,
        }) = last
        else {
            return Err(Error::InvalidInput(
                "at least one coupling iteration is required".into(),
            ));
        };

        // Commit the new time level.
        let delta_new: Vec<f64> = response.iter().map(|s| s.displacement).collect();
        let new_positions = self.positions_for(&ctx, &delta_new)?;
        let mesh_velocity: Vec<[f64; 2]> = new_positions
            .iter()
            .zip(old_positions.iter())
            .map(|(new, old)| [(new[0] - old[0]) / dt, (new[1] - old[1]) / dt])
            .collect();
        self.struct_velocity = response.iter().map(|s| s.velocity).collect();
        self.struct_acceleration = response.iter().map(|s| s.acceleration).collect();
        self.store_fluid(&new_positions, fluid, Some(mesh_velocity));
        self.accel_initialised = true;
        Ok(AleFsiReport {
            iterations,
            interface_residual: residual_norm,
            omega: aitken.omega,
            converged,
        })
    }

    /// Total force of the fluid on the structure \[N per unit depth\]: the sum of
    /// [`Self::interface_force`] from the last fluid solve (zero before the first
    /// solve).
    pub fn total_fluid_force(&self) -> [f64; 2] {
        self.interface_force
            .iter()
            .fold([0.0; 2], |acc, f| [acc[0] + f[0], acc[1] + f[1]])
    }

    /// Maximum structural deflection \[m\]: the largest displacement magnitude of
    /// the interface nodes.
    pub fn max_deflection(&self) -> f64 {
        self.interface
            .node_indices
            .iter()
            .filter_map(|&idx| self.nodes.get(idx))
            .map(|node| node.displacement[0].hypot(node.displacement[1]))
            .fold(0.0_f64, f64::max)
    }
}
