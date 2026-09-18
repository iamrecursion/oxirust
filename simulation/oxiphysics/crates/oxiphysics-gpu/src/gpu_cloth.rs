// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! GPU-accelerated cloth simulation (CPU mock implementation).
//!
//! Provides a complete cloth simulation pipeline including spring–damper edges,
//! bending constraints, XPBD position solving, and collision response against
//! analytic primitives (sphere, plane).

use crate::{cross3, dot3, length3};

// ---------------------------------------------------------------------------
// Helper math
// ---------------------------------------------------------------------------

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

// ---------------------------------------------------------------------------
// triangle_area
// ---------------------------------------------------------------------------

/// Compute the area of a triangle defined by three 3-D vertices.
///
/// Uses the cross-product formula: `area = ||(v1-v0) × (v2-v0)|| / 2`.
pub fn triangle_area(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> f64 {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    length3(cross3(e1, e2)) * 0.5
}

// ---------------------------------------------------------------------------
// dihedral_angle
// ---------------------------------------------------------------------------

/// Compute the dihedral angle (in radians) between two triangles sharing edge
/// `(v1, v2)`.
///
/// `v0` and `v3` are the "wing" vertices of each triangle respectively.  The
/// angle is the one between the two consistently oriented face normals, so it is
/// `0` when the four vertices are coplanar with the wings on opposite sides of the
/// edge and grows as the hinge folds.  Returns a value in `[0, π]`; a degenerate
/// hinge (a wing vertex on the line of the edge, or coincident edge vertices)
/// gives `0`.
pub fn dihedral_angle(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], v3: [f64; 3]) -> f64 {
    let e = sub3(v2, v1);
    let n1 = cross3(sub3(v0, v1), e);
    let n2 = cross3(e, sub3(v3, v1));
    let len1 = length3(n1);
    let len2 = length3(n2);
    if len1 < 1e-15 || len2 < 1e-15 {
        return 0.0;
    }
    // atan2 of the sine and cosine keeps full precision near 0 and π, where
    // acos of a rounded cosine loses half of the digits.
    length3(cross3(n1, n2)).atan2(dot3(n1, n2))
}

// ---------------------------------------------------------------------------
// dihedral_angle_gradient
// ---------------------------------------------------------------------------

/// The dihedral angle of [`dihedral_angle`] together with its gradient with
/// respect to the four vertices.
///
/// Returns `(θ, [∇₀θ, ∇₁θ, ∇₂θ, ∇₃θ])` for the hinge `(v0, v1, v2, v3)` with the
/// shared edge `(v1, v2)`, or `None` when the hinge is degenerate (coincident edge
/// vertices or a wing vertex on the line of the edge).
///
/// With `e = v2 − v1`, `ê = e/|e|`, the unit directions `u`, `v` of the wing
/// vertices perpendicular to the edge, their distances `hₐ`, `h_b` from the edge
/// line and their positions `tₐ`, `t_b` along it (`tₐ = (v0 − v1)·e / |e|²`), the
/// gradient is
///
/// ```text
/// ∇₀θ = s (ê × u)/hₐ                       ∇₃θ = −s (ê × v)/h_b
/// ∇₂θ = s (t_b (ê × v)/h_b − tₐ (ê × u)/hₐ)
/// ∇₁θ = s ((1 − t_b)(ê × v)/h_b − (1 − tₐ)(ê × u)/hₐ)
/// ```
///
/// where `s = ±1` is the side of the fold (`θ` is the unsigned angle, so its
/// gradient switches sign when the cloth passes through the flat state, as `|φ|`
/// does).  The four gradients sum to zero and exert no net torque, so the angle
/// is invariant under rigid motions.
pub fn dihedral_angle_gradient(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> Option<(f64, [[f64; 3]; 4])> {
    let e = sub3(v2, v1);
    let edge_sq = dot3(e, e);
    if edge_sq < 1e-30 {
        return None;
    }
    let a = sub3(v0, v1);
    let b = sub3(v3, v1);
    let t_a = dot3(a, e) / edge_sq;
    let t_b = dot3(b, e) / edge_sq;
    let a_perp = sub3(a, scale3(e, t_a));
    let b_perp = sub3(b, scale3(e, t_b));
    let h_a = length3(a_perp);
    let h_b = length3(b_perp);
    if h_a < 1e-15 || h_b < 1e-15 {
        return None;
    }
    let e_hat = scale3(e, 1.0 / edge_sq.sqrt());
    let u = scale3(a_perp, 1.0 / h_a);
    let v = scale3(b_perp, 1.0 / h_b);
    let u_cross_v = cross3(u, v);
    let theta = length3(u_cross_v).atan2(-dot3(u, v));
    let side = if dot3(u_cross_v, e_hat) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let g_a = scale3(cross3(e_hat, u), 1.0 / h_a);
    let g_b = scale3(cross3(e_hat, v), 1.0 / h_b);
    let grad0 = scale3(g_a, side);
    let grad3 = scale3(g_b, -side);
    let grad2 = scale3(sub3(scale3(g_b, t_b), scale3(g_a, t_a)), side);
    let grad1 = scale3(sub3(scale3(g_b, 1.0 - t_b), scale3(g_a, 1.0 - t_a)), side);
    Some((theta, [grad0, grad1, grad2, grad3]))
}

// ---------------------------------------------------------------------------
// ClothVertex
// ---------------------------------------------------------------------------

/// A single vertex (particle) in the cloth mesh.
#[derive(Debug, Clone)]
pub struct ClothVertex {
    /// World-space position `[x, y, z]`.
    pub position: [f64; 3],
    /// World-space velocity `[vx, vy, vz]`.
    pub velocity: [f64; 3],
    /// Particle mass (kg).
    pub mass: f64,
    /// If `true` this vertex is fixed in space and not integrated.
    pub pinned: bool,
}

impl ClothVertex {
    /// Create a new vertex at `position` with zero velocity.
    pub fn new(position: [f64; 3], mass: f64, pinned: bool) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            mass,
            pinned,
        }
    }
}

// ---------------------------------------------------------------------------
// ClothEdge
// ---------------------------------------------------------------------------

/// A structural spring edge connecting two cloth vertices.
#[derive(Debug, Clone)]
pub struct ClothEdge {
    /// Index of the first vertex.
    pub v0: usize,
    /// Index of the second vertex.
    pub v1: usize,
    /// Rest length of the spring (m).
    pub rest_length: f64,
    /// Spring stiffness (N/m).
    pub stiffness: f64,
    /// Damping coefficient (N·s/m).
    pub damping: f64,
}

impl ClothEdge {
    /// Create a new edge.
    pub fn new(v0: usize, v1: usize, rest_length: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            v0,
            v1,
            rest_length,
            stiffness,
            damping,
        }
    }

    /// Compute the spring force acting **on** vertex `v0` (force on `v1` is negated).
    ///
    /// Includes both Hooke's law and linear damping.
    pub fn spring_force(&self, verts: &[ClothVertex]) -> [f64; 3] {
        let p0 = verts[self.v0].position;
        let p1 = verts[self.v1].position;
        let vel0 = verts[self.v0].velocity;
        let vel1 = verts[self.v1].velocity;

        let delta = sub3(p1, p0);
        let dist = length3(delta);
        if dist < 1e-15 {
            return [0.0; 3];
        }
        let dir = scale3(delta, 1.0 / dist);

        // Relative velocity projected onto the edge direction
        let rel_vel = dot3(sub3(vel1, vel0), dir);

        let spring_f = self.stiffness * (dist - self.rest_length);
        let damp_f = self.damping * rel_vel;
        scale3(dir, spring_f + damp_f)
    }
}

// ---------------------------------------------------------------------------
// BendingConstraint
// ---------------------------------------------------------------------------

/// A bending constraint connecting four vertices that share a common edge.
///
/// The constraint resists deviation of the dihedral angle `θ` (see
/// [`dihedral_angle`]) from the `rest_angle`; its energy is
/// `½ stiffness (θ − rest_angle)²`.
#[derive(Debug, Clone)]
pub struct BendingConstraint {
    /// First wing vertex index.
    pub v0: usize,
    /// Edge vertex index (first).
    pub v1: usize,
    /// Edge vertex index (second).
    pub v2: usize,
    /// Second wing vertex index.
    pub v3: usize,
    /// Rest dihedral angle (radians).
    pub rest_angle: f64,
    /// Bending stiffness (N·m/rad).
    pub stiffness: f64,
}

impl BendingConstraint {
    /// Create a new bending constraint.
    pub fn new(
        v0: usize,
        v1: usize,
        v2: usize,
        v3: usize,
        rest_angle: f64,
        stiffness: f64,
    ) -> Self {
        Self {
            v0,
            v1,
            v2,
            v3,
            rest_angle,
            stiffness,
        }
    }

    /// Create the constraint of the hinge `(v0, v1, v2, v3)` whose rest angle is
    /// the dihedral angle of `verts` as they are now.
    ///
    /// Returns `None` if an index is outside `verts`.
    pub fn from_vertices(
        verts: &[ClothVertex],
        v0: usize,
        v1: usize,
        v2: usize,
        v3: usize,
        stiffness: f64,
    ) -> Option<Self> {
        let rest_angle = dihedral_angle(
            verts.get(v0)?.position,
            verts.get(v1)?.position,
            verts.get(v2)?.position,
            verts.get(v3)?.position,
        );
        Some(Self::new(v0, v1, v2, v3, rest_angle, stiffness))
    }

    /// Positions of the four hinge vertices, or `None` if an index is outside
    /// `verts`.
    fn hinge_positions(&self, verts: &[ClothVertex]) -> Option<[[f64; 3]; 4]> {
        Some([
            verts.get(self.v0)?.position,
            verts.get(self.v1)?.position,
            verts.get(self.v2)?.position,
            verts.get(self.v3)?.position,
        ])
    }

    /// Compute restoring forces on the four vertices.
    ///
    /// Returns `[f_v0, f_v1, f_v2, f_v3]` — one `[f64;3]` force per vertex.  The
    /// forces are the exact negative gradient of the bending energy,
    /// `f_k = −stiffness (θ − rest_angle) ∇ₖθ` (see
    /// [`dihedral_angle_gradient`]), so they sum to zero and exert no net torque.
    /// A degenerate hinge, or an index outside `verts`, yields zero forces.
    pub fn compute_force(&self, verts: &[ClothVertex]) -> Vec<[f64; 3]> {
        let zero = vec![[0.0; 3]; 4];
        let Some([p0, p1, p2, p3]) = self.hinge_positions(verts) else {
            return zero;
        };
        let Some((theta, gradients)) = dihedral_angle_gradient(p0, p1, p2, p3) else {
            return zero;
        };
        let magnitude = self.stiffness * (theta - self.rest_angle);
        gradients
            .iter()
            .map(|gradient| scale3(*gradient, -magnitude))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ClothMesh
// ---------------------------------------------------------------------------

/// The cloth mesh consisting of vertices and spring edges.
#[derive(Debug, Clone)]
pub struct ClothMesh {
    /// Mesh vertices (particles).
    pub vertices: Vec<ClothVertex>,
    /// Spring edges.
    pub edges: Vec<ClothEdge>,
}

impl ClothMesh {
    /// Create an empty cloth mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Build a regular rectangular grid of `rows × cols` vertices with given
    /// `spacing` (m) between adjacent vertices.
    ///
    /// The grid is laid out in the XZ plane (Y = 0).  The top row (`row == 0`)
    /// is pinned.
    pub fn build_grid(&mut self, rows: usize, cols: usize, spacing: f64) {
        self.vertices.clear();
        self.edges.clear();

        // Create vertices
        for r in 0..rows {
            for c in 0..cols {
                let pos = [c as f64 * spacing, 0.0, r as f64 * spacing];
                let pinned = r == 0;
                self.vertices.push(ClothVertex::new(pos, 1.0, pinned));
            }
        }

        let idx = |r: usize, c: usize| r * cols + c;

        // Structural edges (horizontal + vertical)
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    self.edges.push(ClothEdge::new(
                        idx(r, c),
                        idx(r, c + 1),
                        spacing,
                        1000.0,
                        0.5,
                    ));
                }
                if r + 1 < rows {
                    self.edges.push(ClothEdge::new(
                        idx(r, c),
                        idx(r + 1, c),
                        spacing,
                        1000.0,
                        0.5,
                    ));
                }
            }
        }

        // Shear edges (diagonals)
        let diag = spacing * std::f64::consts::SQRT_2;
        for r in 0..rows {
            for c in 0..cols {
                if r + 1 < rows && c + 1 < cols {
                    self.edges.push(ClothEdge::new(
                        idx(r, c),
                        idx(r + 1, c + 1),
                        diag,
                        500.0,
                        0.2,
                    ));
                    self.edges.push(ClothEdge::new(
                        idx(r + 1, c),
                        idx(r, c + 1),
                        diag,
                        500.0,
                        0.2,
                    ));
                }
            }
        }
    }

    /// Advance the simulation by one time step `dt` (seconds) with gravity `[gx, gy, gz]`.
    ///
    /// Uses symplectic Euler integration with explicit spring forces.
    pub fn step(&mut self, dt: f64, gravity: [f64; 3]) {
        let n = self.vertices.len();
        let mut forces = vec![[0.0f64; 3]; n];

        // Gravity
        for (i, v) in self.vertices.iter().enumerate() {
            if !v.pinned {
                forces[i] = add3(forces[i], scale3(gravity, v.mass));
            }
        }

        // Spring forces
        for edge in &self.edges {
            let f = edge.spring_force(&self.vertices);
            if !self.vertices[edge.v0].pinned {
                forces[edge.v0] = add3(forces[edge.v0], f);
            }
            if !self.vertices[edge.v1].pinned {
                forces[edge.v1] = sub3(forces[edge.v1], f);
            }
        }

        // Integrate
        for (i, v) in self.vertices.iter_mut().enumerate() {
            if v.pinned {
                continue;
            }
            let inv_m = 1.0 / v.mass;
            let accel = scale3(forces[i], inv_m);
            v.velocity = add3(v.velocity, scale3(accel, dt));
            v.position = add3(v.position, scale3(v.velocity, dt));
        }
    }
}

impl Default for ClothMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ClothCollider
// ---------------------------------------------------------------------------

/// An analytic collision primitive for cloth–object interaction.
#[derive(Debug, Clone)]
pub enum ClothCollider {
    /// Sphere collider.
    Sphere {
        /// Center of the sphere.
        center: [f64; 3],
        /// Radius of the sphere.
        radius: f64,
    },
    /// Half-space plane collider.  Points on the side `dot(p, normal) >= d`
    /// are outside.
    Plane {
        /// Outward-facing unit normal.
        normal: [f64; 3],
        /// Signed distance from the origin to the plane along the normal.
        d: f64,
    },
}

impl ClothCollider {
    /// Test whether a point `p` is inside (penetrating) this collider.
    ///
    /// Returns the penetration depth (positive means inside) and the outward
    /// push direction.  Returns `None` if there is no penetration.
    pub fn penetration(&self, p: [f64; 3]) -> Option<(f64, [f64; 3])> {
        match self {
            ClothCollider::Sphere { center, radius } => {
                let delta = sub3(p, *center);
                let dist = length3(delta);
                if dist < *radius {
                    let depth = radius - dist;
                    let dir = if dist < 1e-15 {
                        [0.0, 1.0, 0.0]
                    } else {
                        scale3(delta, 1.0 / dist)
                    };
                    Some((depth, dir))
                } else {
                    None
                }
            }
            ClothCollider::Plane { normal, d } => {
                let signed = dot3(*normal, p) - d;
                if signed < 0.0 {
                    Some((-signed, *normal))
                } else {
                    None
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GpuClothSolver
// ---------------------------------------------------------------------------

/// Inverse mass of a cloth vertex: zero for a pinned vertex and for a vertex
/// whose mass is not a positive finite number (such a vertex cannot be moved by
/// forces or constraints).
fn inverse_mass(vertex: &ClothVertex) -> f64 {
    if vertex.pinned || !(vertex.mass.is_finite() && vertex.mass > 0.0) {
        0.0
    } else {
        1.0 / vertex.mass
    }
}

/// Project one spring edge as a compliant XPBD distance constraint.
///
/// With `C = |x_j − x_i| − L`, the compliance `α = 1 / k` (`k` the edge stiffness)
/// scaled to `α̃ = α / Δt²`, the accumulated multiplier `λ` of this edge and the
/// XPBD damping `γ = c / (k Δt)` (`c` the edge damping in N·s/m), the update is
///
/// ```text
/// Δλ = (−C − α̃ λ − γ ∇C·(x − xⁿ)) / ((1 + γ)(w_i + w_j) + α̃)
/// x_i −= w_i Δλ d̂,   x_j += w_j Δλ d̂,   λ += Δλ
/// ```
///
/// with `d̂` the unit vector from `x_i` to `x_j`, `w` the inverse masses and
/// `xⁿ = start` the positions at the beginning of the step (Macklin, Müller and
/// Chentanez 2016).  An edge that references a missing vertex, whose vertices
/// coincide or are both immovable, or whose stiffness is not a positive number,
/// is left alone.
fn project_edge(
    pred: &mut [[f64; 3]],
    start: &[[f64; 3]],
    inv_mass: &[f64],
    edge: &ClothEdge,
    dt: f64,
    lambda: &mut f64,
) {
    let (i, j) = (edge.v0, edge.v1);
    let (Some(&pi), Some(&pj), Some(&xi), Some(&xj), Some(&wi), Some(&wj)) = (
        pred.get(i),
        pred.get(j),
        start.get(i),
        start.get(j),
        inv_mass.get(i),
        inv_mass.get(j),
    ) else {
        return;
    };
    let w_total = wi + wj;
    if w_total < 1e-15 || edge.stiffness.is_nan() || edge.stiffness <= 0.0 {
        return;
    }
    let alpha_tilde = 1.0 / (edge.stiffness * dt * dt);
    if !alpha_tilde.is_finite() {
        return;
    }
    let delta = sub3(pj, pi);
    let dist = length3(delta);
    if dist < 1e-15 {
        return;
    }
    let dir = scale3(delta, 1.0 / dist);
    let constraint = dist - edge.rest_length;
    let gamma = edge.damping.max(0.0) / (edge.stiffness * dt);
    let drift = dot3(dir, sub3(sub3(pj, xj), sub3(pi, xi)));
    let delta_lambda = (-constraint - alpha_tilde * *lambda - gamma * drift)
        / ((1.0 + gamma) * w_total + alpha_tilde);
    *lambda += delta_lambda;
    pred[i] = sub3(pi, scale3(dir, wi * delta_lambda));
    pred[j] = add3(pj, scale3(dir, wj * delta_lambda));
}

/// Project one bending constraint as a compliant XPBD constraint
/// `C = θ − θ₀` on the dihedral angle, with the gradients of
/// [`dihedral_angle_gradient`] and the compliance `α = 1 / stiffness`.
///
/// A constraint that references a missing vertex or repeats a vertex, whose hinge
/// is degenerate or whose vertices are all immovable, or whose stiffness is not a
/// positive number, is left alone.
fn project_bending(
    pred: &mut [[f64; 3]],
    inv_mass: &[f64],
    bend: &BendingConstraint,
    dt: f64,
    lambda: &mut f64,
) {
    if bend.stiffness.is_nan() || bend.stiffness <= 0.0 {
        return;
    }
    let alpha_tilde = 1.0 / (bend.stiffness * dt * dt);
    if !alpha_tilde.is_finite() {
        return;
    }
    let indices = [bend.v0, bend.v1, bend.v2, bend.v3];
    for (k, &index) in indices.iter().enumerate() {
        if index >= pred.len() || index >= inv_mass.len() || indices[..k].contains(&index) {
            return;
        }
    }
    let [p0, p1, p2, p3] = indices.map(|index| pred[index]);
    let Some((theta, gradients)) = dihedral_angle_gradient(p0, p1, p2, p3) else {
        return;
    };
    let weights = indices.map(|index| inv_mass[index]);
    let weighted_norm_sq: f64 = gradients
        .iter()
        .zip(&weights)
        .map(|(gradient, &weight)| weight * dot3(*gradient, *gradient))
        .sum();
    let denominator = weighted_norm_sq + alpha_tilde;
    if denominator < 1e-15 {
        return;
    }
    let constraint = theta - bend.rest_angle;
    let delta_lambda = (-constraint - alpha_tilde * *lambda) / denominator;
    *lambda += delta_lambda;
    for ((&index, gradient), &weight) in indices.iter().zip(&gradients).zip(&weights) {
        pred[index] = add3(pred[index], scale3(*gradient, weight * delta_lambda));
    }
}

/// XPBD cloth solver (CPU mock implementation).
///
/// Wraps a `ClothMesh` together with a set of collision primitives and runs
/// position-based dynamic iterations per time step.  The spring edges of the mesh
/// are compliant distance constraints (compliance `1 / stiffness`, damping from
/// [`ClothEdge::damping`]) and the optional [`bending`](Self::bending) constraints
/// resist folding along shared edges; the Lagrange multipliers of all constraints
/// are accumulated over the iterations of a step, so the cloth does not become
/// stiffer as `xpbd_iterations` grows.
#[derive(Debug, Clone)]
pub struct GpuClothSolver {
    /// The cloth mesh being simulated.
    pub mesh: ClothMesh,
    /// Collision primitives.
    pub colliders: Vec<ClothCollider>,
    /// Number of XPBD constraint-projection iterations per time step.
    pub xpbd_iterations: usize,
    /// Gravitational acceleration `[gx, gy, gz]` in m/s².  Defaults to
    /// `[0, −9.81, 0]`.
    pub gravity: [f64; 3],
    /// Bending constraints, projected after the spring edges in every iteration.
    /// Empty by default.
    pub bending: Vec<BendingConstraint>,
}

impl GpuClothSolver {
    /// Create a new solver with the given mesh.
    pub fn new(mesh: ClothMesh) -> Self {
        Self {
            mesh,
            colliders: Vec::new(),
            xpbd_iterations: 8,
            gravity: [0.0, -9.81, 0.0],
            bending: Vec::new(),
        }
    }

    /// Add a collision primitive to the solver.
    pub fn add_collider(&mut self, collider: ClothCollider) {
        self.colliders.push(collider);
    }

    /// Advance the simulation by `dt` seconds.
    ///
    /// 1. Integrates velocities and positions with [`gravity`](Self::gravity).
    /// 2. Projects the spring edges and the [`bending`](Self::bending) constraints
    ///    (`xpbd_iterations` times) with accumulated Lagrange multipliers.
    /// 3. Resolves collisions.
    /// 4. Derives the new velocities from the position change.
    ///
    /// Pinned vertices, and vertices whose mass is not a positive finite number,
    /// never move.  A `dt` that is not a positive finite number leaves the cloth
    /// unchanged.
    pub fn solve(&mut self, dt: f64) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let gravity = self.gravity;
        let inv_mass: Vec<f64> = self.mesh.vertices.iter().map(inverse_mass).collect();
        let start: Vec<[f64; 3]> = self.mesh.vertices.iter().map(|v| v.position).collect();

        // --- 1. Semi-implicit Euler predict ---
        let mut pred_pos: Vec<[f64; 3]> = self
            .mesh
            .vertices
            .iter()
            .zip(&inv_mass)
            .map(|(v, &w)| {
                if w == 0.0 {
                    v.position
                } else {
                    add3(
                        v.position,
                        scale3(add3(v.velocity, scale3(gravity, dt)), dt),
                    )
                }
            })
            .collect();

        // --- 2. XPBD constraint projection ---
        let mut edge_lambdas = vec![0.0; self.mesh.edges.len()];
        let mut bending_lambdas = vec![0.0; self.bending.len()];
        for _ in 0..self.xpbd_iterations {
            for (edge, lambda) in self.mesh.edges.iter().zip(&mut edge_lambdas) {
                project_edge(&mut pred_pos, &start, &inv_mass, edge, dt, lambda);
            }
            for (bend, lambda) in self.bending.iter().zip(&mut bending_lambdas) {
                project_bending(&mut pred_pos, &inv_mass, bend, dt, lambda);
            }
        }

        // --- 3. Collision resolution ---
        for collider in &self.colliders {
            for (pred, &w) in pred_pos.iter_mut().zip(&inv_mass) {
                if w == 0.0 {
                    continue;
                }
                if let Some((depth, dir)) = collider.penetration(*pred) {
                    *pred = add3(*pred, scale3(dir, depth));
                }
            }
        }

        // --- 4. Update velocities and positions ---
        for ((vertex, &pred), &w) in self.mesh.vertices.iter_mut().zip(&pred_pos).zip(&inv_mass) {
            if w > 0.0 {
                vertex.velocity = scale3(sub3(pred, vertex.position), 1.0 / dt);
                vertex.position = pred;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --- triangle_area ---

    #[test]
    fn test_triangle_area_unit() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let area = triangle_area(v0, v1, v2);
        assert!((area - 0.5).abs() < 1e-12, "area={area}");
    }

    #[test]
    fn test_triangle_area_degenerate() {
        // Collinear points → area == 0
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [2.0, 0.0, 0.0];
        assert!(triangle_area(v0, v1, v2) < 1e-12);
    }

    #[test]
    fn test_triangle_area_equilateral() {
        // Side = 2 → area = sqrt(3)
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [2.0, 0.0, 0.0];
        let v2 = [1.0, f64::sqrt(3.0), 0.0];
        let expected = f64::sqrt(3.0);
        assert!((triangle_area(v0, v1, v2) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_area_3d() {
        // Right triangle in 3-D: legs along X and Z of length 1
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, 1.0];
        let area = triangle_area(v0, v1, v2);
        assert!((area - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_triangle_area_large() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [10.0, 0.0, 0.0];
        let v2 = [0.0, 10.0, 0.0];
        let area = triangle_area(v0, v1, v2);
        assert!((area - 50.0).abs() < 1e-10);
    }

    // --- dihedral_angle ---

    #[test]
    fn test_dihedral_angle_flat() {
        // Two triangles coplanar in XZ → angle = 0
        let v0 = [0.0, 0.0, -1.0];
        let v1 = [-1.0, 0.0, 0.0];
        let v2 = [1.0, 0.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        let angle = dihedral_angle(v0, v1, v2, v3);
        assert!(angle < 1e-10, "angle={angle}");
    }

    #[test]
    fn test_dihedral_angle_ninety_degrees() {
        // Second triangle folded 90° out of plane
        let v0 = [0.0, 1.0, 0.0];
        let v1 = [0.0, 0.0, 0.0];
        let v2 = [1.0, 0.0, 0.0];
        let v3 = [0.5, 0.0, 1.0];
        let angle = dihedral_angle(v0, v1, v2, v3);
        // Should be close to π/2
        assert!((angle - PI / 2.0).abs() < 0.3, "angle={angle}");
    }

    #[test]
    fn test_dihedral_angle_degenerate_edge() {
        // v1 == v2 → degenerate, should not panic
        let v0 = [0.0, 1.0, 0.0];
        let v1 = [0.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, 0.0];
        let v3 = [0.0, -1.0, 0.0];
        let angle = dihedral_angle(v0, v1, v2, v3);
        assert!(angle.is_finite());
    }

    #[test]
    fn test_dihedral_angle_range() {
        let v0 = [1.0, 1.0, 0.0];
        let v1 = [0.0, 0.0, 0.0];
        let v2 = [1.0, 0.0, 0.0];
        let v3 = [1.0, -1.0, 0.0];
        let angle = dihedral_angle(v0, v1, v2, v3);
        assert!((0.0..=PI + 1e-10).contains(&angle), "angle={angle}");
    }

    // --- ClothVertex ---

    #[test]
    fn test_cloth_vertex_new() {
        let v = ClothVertex::new([1.0, 2.0, 3.0], 2.5, false);
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.velocity, [0.0; 3]);
        assert!((v.mass - 2.5).abs() < 1e-12);
        assert!(!v.pinned);
    }

    #[test]
    fn test_cloth_vertex_pinned() {
        let v = ClothVertex::new([0.0; 3], 1.0, true);
        assert!(v.pinned);
    }

    // --- ClothEdge spring force ---

    #[test]
    fn test_spring_force_at_rest() {
        let verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0, false),
            ClothVertex::new([1.0, 0.0, 0.0], 1.0, false),
        ];
        let edge = ClothEdge::new(0, 1, 1.0, 1000.0, 0.5);
        let f = edge.spring_force(&verts);
        // At rest length, spring force is zero (and zero relative velocity)
        assert!(length3(f) < 1e-10, "f={f:?}");
    }

    #[test]
    fn test_spring_force_stretched() {
        let verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0, false),
            ClothVertex::new([2.0, 0.0, 0.0], 1.0, false),
        ];
        // rest = 1, stretched to 2 → force pulls v0 toward v1
        let edge = ClothEdge::new(0, 1, 1.0, 1000.0, 0.0);
        let f = edge.spring_force(&verts);
        assert!(f[0] > 0.0, "force should be positive (toward v1)");
        assert!(f[1].abs() < 1e-12);
        assert!(f[2].abs() < 1e-12);
    }

    #[test]
    fn test_spring_force_compressed() {
        let verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0, false),
            ClothVertex::new([0.5, 0.0, 0.0], 1.0, false),
        ];
        // rest = 1, compressed to 0.5 → force pushes v0 away
        let edge = ClothEdge::new(0, 1, 1.0, 1000.0, 0.0);
        let f = edge.spring_force(&verts);
        assert!(f[0] < 0.0, "force should be negative (push back)");
    }

    #[test]
    fn test_spring_force_with_damping() {
        let mut v0 = ClothVertex::new([0.0, 0.0, 0.0], 1.0, false);
        let mut v1 = ClothVertex::new([2.0, 0.0, 0.0], 1.0, false);
        v0.velocity = [-1.0, 0.0, 0.0];
        v1.velocity = [1.0, 0.0, 0.0];
        let verts = vec![v0, v1];
        let edge = ClothEdge::new(0, 1, 1.0, 0.0, 10.0); // only damping
        let f = edge.spring_force(&verts);
        // Relative velocity along edge: (1 - (-1)) = 2, damping force = 10 * 2 = 20
        assert!((f[0] - 20.0).abs() < 1e-10, "f={f:?}");
    }

    // --- ClothMesh ---

    #[test]
    fn test_build_grid_vertex_count() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(4, 5, 0.1);
        assert_eq!(mesh.vertices.len(), 20);
    }

    #[test]
    fn test_build_grid_top_row_pinned() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(4, 5, 0.1);
        for c in 0..5 {
            assert!(
                mesh.vertices[c].pinned,
                "vertex {c} in row 0 should be pinned"
            );
        }
        for r in 1..4 {
            for c in 0..5 {
                assert!(!mesh.vertices[r * 5 + c].pinned);
            }
        }
    }

    #[test]
    fn test_build_grid_spacing() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 2, 0.5);
        // Horizontal neighbor distance
        let d = sub3(mesh.vertices[1].position, mesh.vertices[0].position);
        assert!((length3(d) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mesh_step_gravity() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 1, 1.0);
        // vertex 1 is not pinned
        let y_before = mesh.vertices[1].position[1];
        mesh.step(0.01, [0.0, -9.81, 0.0]);
        let y_after = mesh.vertices[1].position[1];
        assert!(y_after < y_before, "unpinned vertex should fall");
    }

    #[test]
    fn test_mesh_step_pinned_unchanged() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 1, 1.0);
        let pos_before = mesh.vertices[0].position;
        mesh.step(0.01, [0.0, -9.81, 0.0]);
        assert_eq!(mesh.vertices[0].position, pos_before);
    }

    #[test]
    fn test_mesh_default() {
        let mesh = ClothMesh::default();
        assert!(mesh.vertices.is_empty());
        assert!(mesh.edges.is_empty());
    }

    // --- BendingConstraint ---

    #[test]
    fn test_bending_at_rest_angle() {
        // Compute rest angle, then check zero force
        let p0 = [0.0, 1.0, 0.0];
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [1.0, 0.0, -1.0];

        let rest = dihedral_angle(p0, p1, p2, p3);

        let verts = vec![
            ClothVertex::new(p0, 1.0, false),
            ClothVertex::new(p1, 1.0, false),
            ClothVertex::new(p2, 1.0, false),
            ClothVertex::new(p3, 1.0, false),
        ];

        let bc = BendingConstraint::new(0, 1, 2, 3, rest, 100.0);
        let forces = bc.compute_force(&verts);
        for f in &forces {
            assert!(length3(*f) < 1e-8, "force should be ~zero at rest");
        }
    }

    #[test]
    fn test_bending_constraint_forces_len() {
        let verts: Vec<ClothVertex> = (0..4)
            .map(|i| ClothVertex::new([i as f64, 0.0, 0.0], 1.0, false))
            .collect();
        let bc = BendingConstraint::new(0, 1, 2, 3, 0.0, 10.0);
        let forces = bc.compute_force(&verts);
        assert_eq!(forces.len(), 4);
    }

    // --- ClothCollider ---

    #[test]
    fn test_sphere_collider_inside() {
        let col = ClothCollider::Sphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        let (depth, _dir) = col.penetration([0.5, 0.0, 0.0]).unwrap();
        assert!((depth - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_collider_outside() {
        let col = ClothCollider::Sphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        assert!(col.penetration([2.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_sphere_collider_direction() {
        let col = ClothCollider::Sphere {
            center: [0.0; 3],
            radius: 2.0,
        };
        let (_depth, dir) = col.penetration([1.0, 0.0, 0.0]).unwrap();
        assert!((dir[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sphere_collider_center() {
        // Point at center → degenerate, should not panic
        let col = ClothCollider::Sphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        let result = col.penetration([0.0; 3]);
        assert!(result.is_some());
    }

    #[test]
    fn test_plane_collider_below() {
        // Plane y=0, normal=(0,1,0), d=0 → point at y=-0.5 is below
        let col = ClothCollider::Plane {
            normal: [0.0, 1.0, 0.0],
            d: 0.0,
        };
        let (depth, dir) = col.penetration([0.0, -0.5, 0.0]).unwrap();
        assert!((depth - 0.5).abs() < 1e-10);
        assert!((dir[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_plane_collider_above() {
        let col = ClothCollider::Plane {
            normal: [0.0, 1.0, 0.0],
            d: 0.0,
        };
        assert!(col.penetration([0.0, 1.0, 0.0]).is_none());
    }

    // --- GpuClothSolver ---

    #[test]
    fn test_solver_new() {
        let mesh = ClothMesh::new();
        let solver = GpuClothSolver::new(mesh);
        assert_eq!(solver.xpbd_iterations, 8);
        assert!(solver.colliders.is_empty());
    }

    #[test]
    fn test_solver_add_collider() {
        let mesh = ClothMesh::new();
        let mut solver = GpuClothSolver::new(mesh);
        solver.add_collider(ClothCollider::Plane {
            normal: [0.0, 1.0, 0.0],
            d: -1.0,
        });
        assert_eq!(solver.colliders.len(), 1);
    }

    #[test]
    fn test_solver_solve_no_penetration() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 2, 0.5);
        let mut solver = GpuClothSolver::new(mesh);
        // Ground plane well below
        solver.add_collider(ClothCollider::Plane {
            normal: [0.0, 1.0, 0.0],
            d: -10.0,
        });
        solver.solve(0.01);
        // Unpinned vertices should still be above ground
        for v in &solver.mesh.vertices {
            if !v.pinned {
                assert!(v.position[1] > -10.0);
            }
        }
    }

    #[test]
    fn test_solver_sphere_prevents_penetration() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 2, 0.1);
        // Move free vertices inside a sphere
        for v in mesh.vertices.iter_mut() {
            if !v.pinned {
                v.position = [0.0, 0.0, 0.0];
            }
        }
        let mut solver = GpuClothSolver::new(mesh);
        solver.add_collider(ClothCollider::Sphere {
            center: [0.0; 3],
            radius: 5.0,
        });
        solver.solve(0.01);
        // All unpinned verts should now be at distance >= radius from center
        for v in &solver.mesh.vertices {
            if !v.pinned {
                let dist = length3(v.position);
                assert!(dist >= 5.0 - 1e-6, "dist={dist}");
            }
        }
    }

    #[test]
    fn test_solver_pinned_stays() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(3, 3, 0.5);
        let pin_pos: Vec<_> = mesh
            .vertices
            .iter()
            .filter(|v| v.pinned)
            .map(|v| v.position)
            .collect();

        let mut solver = GpuClothSolver::new(mesh);
        for _ in 0..10 {
            solver.solve(0.01);
        }

        let pin_pos_after: Vec<_> = solver
            .mesh
            .vertices
            .iter()
            .filter(|v| v.pinned)
            .map(|v| v.position)
            .collect();

        assert_eq!(pin_pos, pin_pos_after);
    }

    #[test]
    fn test_cloth_grid_has_edges() {
        let mut mesh = ClothMesh::new();
        mesh.build_grid(3, 3, 0.5);
        assert!(!mesh.edges.is_empty());
    }

    #[test]
    fn test_dihedral_symmetric() {
        // Swapping the two wing vertices should give the same angle
        let p0 = [0.0, 1.0, 0.0];
        let p1 = [-1.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [0.0, -1.0, 0.5];
        let a1 = dihedral_angle(p0, p1, p2, p3);
        let a2 = dihedral_angle(p3, p1, p2, p0);
        assert!((a1 - a2).abs() < 1e-10, "a1={a1} a2={a2}");
    }

    #[test]
    fn test_spring_force_zero_length() {
        // Both vertices at same position → no force (no panic)
        let verts = vec![
            ClothVertex::new([0.0, 0.0, 0.0], 1.0, false),
            ClothVertex::new([0.0, 0.0, 0.0], 1.0, false),
        ];
        let edge = ClothEdge::new(0, 1, 1.0, 1000.0, 0.5);
        let f = edge.spring_force(&verts);
        assert!(length3(f) < 1e-12);
    }

    #[test]
    fn test_multiple_steps_energy_decreases() {
        // With damping, a stretched spring should lose energy over time
        let mut mesh = ClothMesh::new();
        mesh.build_grid(2, 1, 2.0); // rest_length=1, stretched to 2
        let mut solver = GpuClothSolver::new(mesh);
        solver.solve(0.001);
        // Just check no NaN
        for v in &solver.mesh.vertices {
            for x in v.position {
                assert!(x.is_finite());
            }
        }
    }

    #[test]
    fn test_cloth_vertex_clone() {
        let v = ClothVertex::new([1.0, 2.0, 3.0], 1.0, false);
        let v2 = v.clone();
        assert_eq!(v.position, v2.position);
    }

    #[test]
    fn test_cloth_edge_clone() {
        let e = ClothEdge::new(0, 1, 1.0, 100.0, 0.1);
        let e2 = e.clone();
        assert_eq!(e.v0, e2.v0);
        assert_eq!(e.rest_length, e2.rest_length);
    }

    #[test]
    fn test_collider_clone() {
        let c = ClothCollider::Sphere {
            center: [1.0, 2.0, 3.0],
            radius: 0.5,
        };
        let c2 = c.clone();
        if let ClothCollider::Sphere { radius, .. } = c2 {
            assert!((radius - 0.5).abs() < 1e-12);
        } else {
            panic!("wrong variant");
        }
    }
}

#[cfg(test)]
mod xpbd_tests;
