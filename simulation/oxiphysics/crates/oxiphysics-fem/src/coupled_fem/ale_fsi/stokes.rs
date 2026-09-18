//! Stabilised P1/P1 Stokes discretisation on a triangular mesh.
//!
//! Finds the velocity v (P1) and pressure p (P1) of the incompressible Stokes
//! problem
//!
//! −∇·σ = f,  σ = −p·I + μ(∇v + ∇vᵀ),  ∇·v = 0
//!
//! on the current (ALE-moved) mesh with
//!
//! * velocity Dirichlet conditions on the interface (structure velocity), inlet
//!   (plug flow) and wall (no-slip) nodes, and
//! * the traction condition σ·n = −p₀·n on every boundary edge that has a node
//!   without prescribed velocity (the open boundary, e.g. an outlet).
//!
//! Equal-order P1/P1 elements violate the inf-sup condition, so the continuity
//! equation is stabilised with the consistent pressure-stabilising
//! Petrov-Galerkin (PSPG) term of Hughes, Franca & Balestra,
//!
//! ∫ q ∇·v dV + Σ_e τ_e ∫_e ∇q·(∇p − f) dV = 0,  τ_e = h_e² / (12 μ),
//!
//! with `h_e` the longest edge of element `e`.  The stabilisation is consistent
//! (it vanishes for the exact solution), so hydrostatic pressure fields p = p₀ +
//! ρ g·(x₀ − x) are reproduced exactly.  For P1 velocities the strong momentum
//! residual reduces to ∇p − f.
//!
//! The reduced saddle-point system is solved with a dense LU factorisation (it
//! is not positive definite), which limits the method to small and medium
//! meshes.  It is solved in the viscosity-scaled unknowns (v, p/μ), whose matrix
//! does not depend on μ, so very small and very large viscosities factorise
//! equally well.  The force exerted by the fluid on every node with prescribed
//! velocity is the consistent momentum residual (the reaction of the Dirichlet
//! constraint), which integrates pressure and viscous tractions exactly against
//! the nodal basis functions.

use super::geometry::{Triangle, element_corners};
use crate::dense_lu::LuFactor;
use crate::error::{Error, Result};

/// Largest number of unknowns of the dense saddle-point solve.
const MAX_FREE_DOFS: usize = 4096;

/// Boundary role of a mesh node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeKind {
    /// No prescribed velocity (interior node or open boundary).
    Free,
    /// Fluid-structure interface: velocity equals the structure velocity.
    Interface,
    /// Inlet: prescribed plug velocity.
    Inlet,
    /// No-slip wall.
    Wall,
}

/// A boundary edge of the mesh together with the node of its element that does
/// not lie on the edge (it fixes the outward direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BoundaryEdge {
    pub(super) nodes: [usize; 2],
    pub(super) opposite: usize,
}

/// Input of one Stokes solve.
pub(super) struct StokesProblem<'a> {
    /// Current node positions.
    pub(super) positions: &'a [[f64; 2]],
    /// Triangle connectivity.
    pub(super) elements: &'a [[usize; 3]],
    /// Boundary role of every node.
    pub(super) kinds: &'a [NodeKind],
    /// Prescribed velocity of every node whose kind is not [`NodeKind::Free`].
    pub(super) prescribed: &'a [[f64; 2]],
    /// Open-boundary edges carrying the traction −p₀·n.
    pub(super) open_edges: &'a [BoundaryEdge],
    /// Dynamic viscosity μ \[Pa·s\].
    pub(super) viscosity: f64,
    /// Body force density f = ρ·g \[N/m³\].
    pub(super) body_force: [f64; 2],
    /// Pressure p₀ imposed on the open boundary \[Pa\].
    pub(super) open_pressure: f64,
}

/// Result of one Stokes solve.
#[derive(Debug, Clone)]
pub(super) struct StokesSolution {
    /// Nodal velocity.
    pub(super) velocity: Vec<[f64; 2]>,
    /// Nodal pressure.
    pub(super) pressure: Vec<f64>,
    /// Force exerted by the fluid on the boundary at every node with prescribed
    /// velocity \[N per unit depth\]; zero at free nodes.
    pub(super) boundary_force: Vec<[f64; 2]>,
}

/// Local degree-of-freedom layout of an element: (vₓ, v_y, p) per node.
type ElementMatrix = [[f64; 9]; 9];
type ElementVector = [f64; 9];

/// Stabilised Stokes element matrix and load vector.
///
/// Row/column `3a + i` is the velocity component `i` of local node `a`, row/column
/// `3a + 2` the pressure of node `a`.  Momentum rows read
/// K·v − B·p = f; continuity rows Bᵀ·v + S·p = g.
fn stokes_element(
    triangle: &Triangle,
    viscosity: f64,
    body_force: [f64; 2],
) -> (ElementMatrix, ElementVector) {
    let area = triangle.area;
    let grad = &triangle.grad;
    let tau = triangle.longest_edge * triangle.longest_edge / (12.0 * viscosity);
    let mut matrix = [[0.0; 9]; 9];
    let mut load = [0.0; 9];
    for (a, ga) in grad.iter().enumerate() {
        for (b, gb) in grad.iter().enumerate() {
            let grad_dot = ga[0] * gb[0] + ga[1] * gb[1];
            // Viscous block μ·A·[δ_ij ∇λ_a·∇λ_b + ∂_jλ_a ∂_iλ_b] of the stress form.
            for i in 0..2 {
                for j in 0..2 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    matrix[3 * a + i][3 * b + j] =
                        viscosity * area * (delta * grad_dot + ga[j] * gb[i]);
                }
                // −∫ p ∇·w: pressure column of momentum rows.
                matrix[3 * a + i][3 * b + 2] = -area / 3.0 * ga[i];
                // ∫ q ∇·v: velocity columns of continuity rows.
                matrix[3 * a + 2][3 * b + i] = area / 3.0 * gb[i];
            }
            // PSPG pressure block τ·A·∇λ_a·∇λ_b.
            matrix[3 * a + 2][3 * b + 2] = tau * area * grad_dot;
        }
        for i in 0..2 {
            load[3 * a + i] = area / 3.0 * body_force[i];
        }
        load[3 * a + 2] = tau * area * (ga[0] * body_force[0] + ga[1] * body_force[1]);
    }
    (matrix, load)
}

/// Unit outward normal (away from the opposite node) and length of an edge.
fn edge_normal(positions: &[[f64; 2]], edge: &BoundaryEdge) -> Option<([f64; 2], f64)> {
    let a = *positions.get(edge.nodes[0])?;
    let b = *positions.get(edge.nodes[1])?;
    let c = *positions.get(edge.opposite)?;
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length = dx.hypot(dy);
    if !length.is_finite() || length == 0.0 {
        return None;
    }
    let mut normal = [dy / length, -dx / length];
    if normal[0] * (c[0] - a[0]) + normal[1] * (c[1] - a[1]) > 0.0 {
        normal = [-normal[0], -normal[1]];
    }
    Some((normal, length))
}

/// Solve the stabilised Stokes problem.
///
/// # Errors
/// [`Error::InvalidInput`] for inconsistent array sizes, a non-positive or
/// non-finite viscosity, a degenerate element or edge, an element or edge that
/// references a missing node, no open boundary, or too many unknowns for the
/// dense solve; [`Error::SingularSystem`] if the saddle-point system cannot be
/// factorised or the solution is not finite.
pub(super) fn solve_stokes(problem: &StokesProblem<'_>) -> Result<StokesSolution> {
    let n = problem.positions.len();
    if problem.kinds.len() != n || problem.prescribed.len() != n {
        return Err(Error::InvalidInput(format!(
            "Stokes solve needs one boundary kind and prescribed velocity per node, got {} and {} \
             for {n} nodes",
            problem.kinds.len(),
            problem.prescribed.len()
        )));
    }
    if !(problem.viscosity.is_finite() && problem.viscosity > 0.0) {
        return Err(Error::InvalidInput(format!(
            "viscosity must be positive and finite, got {}",
            problem.viscosity
        )));
    }
    if problem.open_edges.is_empty() {
        return Err(Error::InvalidInput(
            "the fluid domain has no open boundary: an incompressible flow with prescribed \
             velocity on the whole boundary has no unique pressure"
                .into(),
        ));
    }
    // Numbering of the unknowns: both velocity components of free nodes and the
    // pressure of every node.
    let mut free: Vec<[Option<usize>; 3]> = vec![[None; 3]; n];
    let mut is_pressure: Vec<bool> = Vec::new();
    let mut n_free = 0usize;
    for (entry, &kind) in free.iter_mut().zip(problem.kinds.iter()) {
        let velocity_free = kind == NodeKind::Free;
        for (component, slot) in entry.iter_mut().enumerate() {
            if component == 2 || velocity_free {
                *slot = Some(n_free);
                is_pressure.push(component == 2);
                n_free += 1;
            }
        }
    }
    if n_free > MAX_FREE_DOFS {
        return Err(Error::InvalidInput(format!(
            "the dense Stokes solve supports at most {MAX_FREE_DOFS} unknowns, got {n_free}"
        )));
    }
    let prescribed_value = |node: usize, component: usize| -> f64 {
        problem
            .prescribed
            .get(node)
            .map_or(0.0, |velocity| velocity[component.min(1)])
    };

    let mut matrix = vec![vec![0.0; n_free]; n_free];
    let mut rhs = vec![0.0; n_free];
    let mut element_data: Vec<(ElementMatrix, ElementVector)> =
        Vec::with_capacity(problem.elements.len());
    for (e, element) in problem.elements.iter().enumerate() {
        let corners = element_corners(problem.positions, element).ok_or_else(|| {
            Error::InvalidInput(format!("element {e} references a node outside the mesh"))
        })?;
        let triangle = Triangle::new(&corners).ok_or_else(|| {
            Error::InvalidInput(format!("element {e} is degenerate in the current mesh"))
        })?;
        let (local_matrix, local_load) =
            stokes_element(&triangle, problem.viscosity, problem.body_force);
        for (la, &na) in element.iter().enumerate() {
            for lc in 0..3 {
                let Some(row) = free[na][lc] else { continue };
                rhs[row] += local_load[3 * la + lc];
                for (lb, &nb) in element.iter().enumerate() {
                    for ld in 0..3 {
                        let value = local_matrix[3 * la + lc][3 * lb + ld];
                        match free[nb][ld] {
                            Some(col) => matrix[row][col] += value,
                            None => rhs[row] -= value * prescribed_value(nb, ld),
                        }
                    }
                }
            }
        }
        element_data.push((local_matrix, local_load));
    }

    // Traction −p₀·n on the open boundary: nodal load −p₀·n·ℓ/2 on both end nodes.
    let mut neumann = vec![[0.0; 2]; n];
    for (k, edge) in problem.open_edges.iter().enumerate() {
        let (normal, length) = edge_normal(problem.positions, edge).ok_or_else(|| {
            Error::InvalidInput(format!(
                "open-boundary edge {k} is degenerate or out of range"
            ))
        })?;
        let load = [
            -problem.open_pressure * normal[0] * length / 2.0,
            -problem.open_pressure * normal[1] * length / 2.0,
        ];
        for &node in &edge.nodes {
            for (component, value) in load.iter().enumerate() {
                neumann[node][component] += value;
                if let Some(row) = free[node][component] {
                    rhs[row] += value;
                }
            }
        }
    }

    // Condition the saddle-point system independently of the viscosity.  With the
    // pressure p = μ·q and the momentum rows divided by μ the matrix no longer
    // depends on μ (the PSPG parameter τ is proportional to 1/μ); without this
    // scaling the pressure block dwarfs the viscous block for small μ and the
    // factorisation mistakes the small viscous pivots for a singular system.
    let viscosity = problem.viscosity;
    for ((row, rhs_entry), &pressure_row) in matrix.iter_mut().zip(rhs.iter_mut()).zip(&is_pressure)
    {
        if !pressure_row {
            row.iter_mut().for_each(|entry| *entry /= viscosity);
            *rhs_entry /= viscosity;
        }
        for (entry, &pressure_column) in row.iter_mut().zip(&is_pressure) {
            if pressure_column {
                *entry *= viscosity;
            }
        }
    }
    let lu = LuFactor::new(&matrix).ok_or(Error::SingularSystem)?;
    let mut solution = lu.solve(&rhs);
    if solution.iter().any(|x| !x.is_finite()) {
        return Err(Error::SingularSystem);
    }
    for (value, &pressure_unknown) in solution.iter_mut().zip(&is_pressure) {
        if pressure_unknown {
            *value *= viscosity;
        }
    }

    let mut velocity = vec![[0.0; 2]; n];
    let mut pressure = vec![0.0; n];
    for node in 0..n {
        for component in 0..2 {
            velocity[node][component] = match free[node][component] {
                Some(index) => solution[index],
                None => prescribed_value(node, component),
            };
        }
        if let Some(index) = free[node][2] {
            pressure[node] = solution[index];
        }
    }

    // Reaction of the velocity constraints: the momentum residual
    // K·v − B·p − f − N of the constrained rows.
    let mut residual = vec![[0.0; 2]; n];
    for (element, (local_matrix, local_load)) in problem.elements.iter().zip(element_data.iter()) {
        let mut local_solution = [0.0; 9];
        for (la, &node) in element.iter().enumerate() {
            local_solution[3 * la] = velocity[node][0];
            local_solution[3 * la + 1] = velocity[node][1];
            local_solution[3 * la + 2] = pressure[node];
        }
        for (la, &node) in element.iter().enumerate() {
            if problem.kinds[node] == NodeKind::Free {
                continue;
            }
            for (component, entry) in residual[node].iter_mut().enumerate() {
                let row = 3 * la + component;
                let product: f64 = local_matrix[row]
                    .iter()
                    .zip(local_solution.iter())
                    .map(|(m, u)| m * u)
                    .sum();
                *entry += product - local_load[row];
            }
        }
    }
    let boundary_force: Vec<[f64; 2]> = residual
        .iter()
        .zip(neumann.iter())
        .zip(problem.kinds.iter())
        .map(|((r, load), &kind)| {
            if kind == NodeKind::Free {
                [0.0, 0.0]
            } else {
                [load[0] - r[0], load[1] - r[1]]
            }
        })
        .collect();
    Ok(StokesSolution {
        velocity,
        pressure,
        boundary_force,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Elements owning an edge: (first node, second node, opposite node).
    type EdgeOwners = Vec<(usize, usize, usize)>;

    /// Structured `nx × ny` mesh on `[0, lx] × [0, ly]` (two triangles per cell).
    struct Grid {
        nx: usize,
        ny: usize,
        positions: Vec<[f64; 2]>,
        elements: Vec<[usize; 3]>,
    }

    impl Grid {
        fn new(nx: usize, ny: usize, lx: f64, ly: f64) -> Self {
            let idx = |i: usize, j: usize| j * (nx + 1) + i;
            let mut positions = Vec::new();
            for j in 0..=ny {
                for i in 0..=nx {
                    positions.push([lx * i as f64 / nx as f64, ly * j as f64 / ny as f64]);
                }
            }
            let mut elements = Vec::new();
            for j in 0..ny {
                for i in 0..nx {
                    elements.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                    elements.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
                }
            }
            Self {
                nx,
                ny,
                positions,
                elements,
            }
        }

        fn node(&self, i: usize, j: usize) -> usize {
            j * (self.nx + 1) + i
        }

        /// Boundary edges (with the opposite node) that touch a free node.
        fn open_edges(&self, kinds: &[NodeKind]) -> Vec<BoundaryEdge> {
            let mut edges: BTreeMap<(usize, usize), EdgeOwners> = BTreeMap::new();
            for e in &self.elements {
                for (u, v, o) in [(e[0], e[1], e[2]), (e[1], e[2], e[0]), (e[2], e[0], e[1])] {
                    edges
                        .entry((u.min(v), u.max(v)))
                        .or_default()
                        .push((u, v, o));
                }
            }
            edges
                .values()
                .filter(|owners| owners.len() == 1)
                .map(|owners| owners[0])
                .filter(|&(u, v, _)| kinds[u] == NodeKind::Free || kinds[v] == NodeKind::Free)
                .map(|(u, v, o)| BoundaryEdge {
                    nodes: [u, v],
                    opposite: o,
                })
                .collect()
        }
    }

    /// Container: walls on both sides (and the lower corners are interface
    /// nodes), interface on the bottom, open top.
    fn container_kinds(grid: &Grid) -> Vec<NodeKind> {
        let mut kinds = vec![NodeKind::Free; grid.positions.len()];
        for j in 1..=grid.ny {
            kinds[grid.node(0, j)] = NodeKind::Wall;
            kinds[grid.node(grid.nx, j)] = NodeKind::Wall;
        }
        for i in 0..=grid.nx {
            kinds[grid.node(i, 0)] = NodeKind::Interface;
        }
        kinds
    }

    fn solve(
        grid: &Grid,
        kinds: &[NodeKind],
        prescribed: &[[f64; 2]],
        viscosity: f64,
        body_force: [f64; 2],
        open_pressure: f64,
    ) -> StokesSolution {
        let open_edges = grid.open_edges(kinds);
        solve_stokes(&StokesProblem {
            positions: &grid.positions,
            elements: &grid.elements,
            kinds,
            prescribed,
            open_edges: &open_edges,
            viscosity,
            body_force,
            open_pressure,
        })
        .expect("well-posed Stokes problem")
    }

    #[test]
    fn hydrostatic_pressure_and_weight_reaction_are_exact() {
        let grid = Grid::new(6, 4, 1.5, 1.0);
        let kinds = container_kinds(&grid);
        let prescribed = vec![[0.0; 2]; grid.positions.len()];
        let (rho, g) = (1000.0, 9.81);
        let solution = solve(&grid, &kinds, &prescribed, 1e-3, [0.0, -rho * g], 0.0);
        for (p, x) in solution.pressure.iter().zip(grid.positions.iter()) {
            let exact = rho * g * (1.0 - x[1]);
            assert!((p - exact).abs() < 1e-9 * rho * g, "p = {p}, exact {exact}");
        }
        let v_max = solution
            .velocity
            .iter()
            .map(|v| v[0].abs().max(v[1].abs()))
            .fold(0.0_f64, f64::max);
        assert!(v_max < 1e-6, "the fluid must be at rest, |v| = {v_max}");
        // The bottom carries the weight of the fluid column: p·ℓ per node, pointing down.
        let ds = 1.5 / 6.0;
        for i in 0..=grid.nx {
            let weight = rho * g * 1.0 * ds * if i == 0 || i == grid.nx { 0.5 } else { 1.0 };
            let force = solution.boundary_force[grid.node(i, 0)];
            assert!(
                (force[1] + weight).abs() < 1e-8 * weight,
                "node {i}: F_y = {}, weight {weight}",
                force[1]
            );
        }
    }

    #[test]
    fn open_boundary_pressure_pushes_the_wall_away_from_the_fluid() {
        // No gravity, p₀ on the open top: the fluid is at rest with p = p₀ and presses
        // the (lower) interface with −p₀·ℓ in y, never towards the fluid.
        let grid = Grid::new(5, 3, 1.0, 1.0);
        let kinds = container_kinds(&grid);
        let prescribed = vec![[0.0; 2]; grid.positions.len()];
        let p0 = 250.0;
        let solution = solve(&grid, &kinds, &prescribed, 0.7, [0.0, 0.0], p0);
        for p in &solution.pressure {
            assert!((p - p0).abs() < 1e-9 * p0, "{p}");
        }
        let ds = 1.0 / 5.0;
        let total: f64 = (0..=grid.nx)
            .map(|i| solution.boundary_force[grid.node(i, 0)][1])
            .sum();
        assert!((total + p0 * 1.0).abs() < 1e-9 * p0, "total {total}");
        for i in 0..=grid.nx {
            let f = solution.boundary_force[grid.node(i, 0)];
            let expected = -p0 * ds * if i == 0 || i == grid.nx { 0.5 } else { 1.0 };
            assert!((f[1] - expected).abs() < 1e-9 * p0 * ds, "node {i}: {f:?}");
            if i == 0 || i == grid.nx {
                // A corner node also carries half of the pressure force on the adjacent
                // side-wall edge of length 1/ny; the wall is pushed outwards.
                let wall = 0.5 * p0 / grid.ny as f64;
                let sign = if i == 0 { -1.0 } else { 1.0 };
                assert!(
                    (f[0] - sign * wall).abs() < 1e-9 * p0 * ds,
                    "corner {i}: {f:?}"
                );
            } else {
                assert!(
                    f[0].abs() < 1e-9 * p0 * ds,
                    "no tangential force at rest: {f:?}"
                );
            }
        }
    }

    #[test]
    fn uniform_translation_is_reproduced_exactly() {
        // Every Dirichlet node moves with (U, 0) and the open edges carry p₀: the exact
        // solution is v = (U, 0) everywhere and p = p₀.
        let grid = Grid::new(4, 4, 1.0, 1.0);
        let mut kinds = vec![NodeKind::Free; grid.positions.len()];
        for j in 0..=grid.ny {
            kinds[grid.node(0, j)] = NodeKind::Inlet;
        }
        for i in 1..=grid.nx {
            kinds[grid.node(i, 0)] = NodeKind::Interface;
            kinds[grid.node(i, grid.ny)] = NodeKind::Wall;
        }
        let speed = 0.8;
        let prescribed: Vec<[f64; 2]> = kinds
            .iter()
            .map(|k| {
                if *k == NodeKind::Free {
                    [0.0; 2]
                } else {
                    [speed, 0.0]
                }
            })
            .collect();
        let solution = solve(&grid, &kinds, &prescribed, 0.01, [0.0, 0.0], 12.0);
        for (v, p) in solution.velocity.iter().zip(solution.pressure.iter()) {
            assert!((v[0] - speed).abs() < 1e-10 && v[1].abs() < 1e-10, "{v:?}");
            assert!((p - 12.0).abs() < 1e-8, "{p}");
        }
    }

    #[test]
    fn global_force_balance_holds_for_arbitrary_boundary_data() {
        // The momentum equations of all free rows are satisfied, so the reactions of
        // the constrained nodes must balance the body force and the open-boundary load.
        let grid = Grid::new(6, 4, 2.0, 1.0);
        let mut kinds = vec![NodeKind::Free; grid.positions.len()];
        for j in 0..=grid.ny {
            kinds[grid.node(0, j)] = NodeKind::Inlet;
        }
        for i in 1..=grid.nx {
            kinds[grid.node(i, 0)] = NodeKind::Interface;
            kinds[grid.node(i, grid.ny)] = NodeKind::Wall;
        }
        let prescribed: Vec<[f64; 2]> = grid
            .positions
            .iter()
            .zip(kinds.iter())
            .map(|(x, k)| match k {
                NodeKind::Inlet => [1.0 + x[1], 0.2],
                NodeKind::Interface => [0.0, 0.3 * (3.0 * x[0]).sin()],
                _ => [0.0, 0.0],
            })
            .collect();
        let body = [30.0, -120.0];
        let p0 = 40.0;
        let solution = solve(&grid, &kinds, &prescribed, 0.5, body, p0);
        let reaction: [f64; 2] = solution
            .boundary_force
            .iter()
            .fold([0.0; 2], |acc, f| [acc[0] + f[0], acc[1] + f[1]]);
        // Total body force on the fluid, and the open-boundary traction −p₀ ∮ n.
        let area = 2.0 * 1.0;
        let open_edges = grid.open_edges(&kinds);
        let mut open_load = [0.0; 2];
        for edge in &open_edges {
            let (normal, length) = edge_normal(&grid.positions, edge).expect("valid edge");
            open_load[0] += -p0 * normal[0] * length;
            open_load[1] += -p0 * normal[1] * length;
        }
        // Force on the boundary + body force + open load = 0 (fluid equilibrium,
        // with `boundary_force` the force of the fluid *on* the constraints).
        for d in 0..2 {
            let balance = -reaction[d] + body[d] * area + open_load[d];
            let scale = reaction[d].abs().max(body[d].abs() * area).max(1.0);
            assert!(
                balance.abs() < 1e-8 * scale,
                "direction {d}: imbalance {balance}"
            );
        }
    }

    #[test]
    fn pressure_scales_with_viscosity_and_velocity_does_not() {
        let grid = Grid::new(8, 3, 3.0, 1.0);
        let mut kinds = vec![NodeKind::Free; grid.positions.len()];
        for j in 1..grid.ny {
            kinds[grid.node(0, j)] = NodeKind::Inlet;
        }
        for i in 0..=grid.nx {
            kinds[grid.node(i, 0)] = NodeKind::Interface;
            kinds[grid.node(i, grid.ny)] = NodeKind::Wall;
        }
        let prescribed: Vec<[f64; 2]> = kinds
            .iter()
            .map(|k| {
                if *k == NodeKind::Inlet {
                    [1.0, 0.0]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let a = solve(&grid, &kinds, &prescribed, 1.0, [0.0; 2], 0.0);
        let b = solve(&grid, &kinds, &prescribed, 3.0, [0.0; 2], 0.0);
        let p_scale = a.pressure.iter().fold(0.0_f64, |m, p| m.max(p.abs()));
        for (pa, pb) in a.pressure.iter().zip(b.pressure.iter()) {
            assert!((pb - 3.0 * pa).abs() < 1e-9 * 3.0 * p_scale, "{pa} {pb}");
        }
        for (va, vb) in a.velocity.iter().zip(b.velocity.iter()) {
            assert!((va[0] - vb[0]).abs() < 1e-9 && (va[1] - vb[1]).abs() < 1e-9);
        }
        // Pressure drives the flow: positive near the inlet, zero at the open outlet.
        assert!(a.pressure[grid.node(1, 1)] > a.pressure[grid.node(grid.nx - 1, 1)]);
    }

    #[test]
    fn extreme_viscosities_keep_the_saddle_point_system_regular() {
        // Regression: for μ = 1e-9 the PSPG block (∝ 1/μ) dwarfed the viscous block and
        // the LU factorisation reported a singular system.  In the viscosity-scaled
        // unknowns the matrix does not depend on μ: the hydrostatic pressure and the
        // weight reaction come out exactly for every viscosity.
        let grid = Grid::new(5, 3, 1.0, 1.0);
        let kinds = container_kinds(&grid);
        let prescribed = vec![[0.0; 2]; grid.positions.len()];
        let (rho, g) = (1000.0, 9.81);
        let ds = 1.0 / 5.0;
        for mu in [1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6] {
            let solution = solve(&grid, &kinds, &prescribed, mu, [0.0, -rho * g], 0.0);
            for (p, x) in solution.pressure.iter().zip(grid.positions.iter()) {
                let exact = rho * g * (1.0 - x[1]);
                assert!(
                    (p - exact).abs() < 1e-7 * rho * g,
                    "μ = {mu:e}: p = {p}, exact {exact}"
                );
            }
            for i in 0..=grid.nx {
                let weight = rho * g * ds * if i == 0 || i == grid.nx { 0.5 } else { 1.0 };
                let force = solution.boundary_force[grid.node(i, 0)];
                assert!(
                    (force[1] + weight).abs() < 1e-7 * weight,
                    "μ = {mu:e}, node {i}: F_y = {}, weight {weight}",
                    force[1]
                );
            }
        }
    }

    #[test]
    fn invalid_problems_are_rejected() {
        let grid = Grid::new(3, 3, 1.0, 1.0);
        let kinds = container_kinds(&grid);
        let prescribed = vec![[0.0; 2]; grid.positions.len()];
        let open_edges = grid.open_edges(&kinds);
        let base = StokesProblem {
            positions: &grid.positions,
            elements: &grid.elements,
            kinds: &kinds,
            prescribed: &prescribed,
            open_edges: &open_edges,
            viscosity: 1.0,
            body_force: [0.0; 2],
            open_pressure: 0.0,
        };
        assert!(solve_stokes(&base).is_ok());
        assert!(
            solve_stokes(&StokesProblem {
                viscosity: 0.0,
                ..base
            })
            .is_err()
        );
        assert!(
            solve_stokes(&StokesProblem {
                viscosity: f64::NAN,
                ..base
            })
            .is_err()
        );
        // Without any open boundary the pressure level is undetermined.
        assert!(matches!(
            solve_stokes(&StokesProblem {
                open_edges: &[],
                ..base
            }),
            Err(Error::InvalidInput(_))
        ));
        assert!(
            solve_stokes(&StokesProblem {
                prescribed: &prescribed[1..],
                ..base
            })
            .is_err()
        );
        // A degenerate element is reported, not silently skipped.
        let mut flat = grid.positions.clone();
        flat[grid.elements[0][1]] = flat[grid.elements[0][0]];
        assert!(matches!(
            solve_stokes(&StokesProblem {
                positions: &flat,
                ..base
            }),
            Err(Error::InvalidInput(_))
        ));
        let bad_edges = [BoundaryEdge {
            nodes: [0, 999],
            opposite: 1,
        }];
        assert!(
            solve_stokes(&StokesProblem {
                open_edges: &bad_edges,
                ..base
            })
            .is_err()
        );
    }
}
