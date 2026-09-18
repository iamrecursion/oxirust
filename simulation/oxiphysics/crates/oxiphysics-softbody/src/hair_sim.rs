// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hair and fiber simulation using a discrete elastic rod (DER) model.
//!
//! A strand is a chain of nodes joined by segments (Bergou et al., "Discrete
//! Elastic Rods", SIGGRAPH 2008). Three energies act on it:
//!
//! * **stretch** - `E_stretch = k_s / 2 * sum (|e_j| - l_j)^2 / l_j`,
//! * **bend** - `E_bend = E_b / 2 * sum |kappa_i - kappa0_i|^2 * lbar_i` with
//!   the chord curvature `kappa_i = (t_i - t_(i-1)) / lbar_i`,
//! * **twist** - `E_twist = E_t / 2 * sum m_i^2 / lbar_i`.
//!
//! Twist needs an orientation that the node positions alone do not carry, so
//! every segment owns a material angle `theta_j` measured from a *reference
//! director* `ref_d1[j]`. The reference directors are carried along the
//! strand by parallel transport, which makes the twist at node `i`
//! `m_i = theta_i - theta_(i-1) + alpha_i`, where `alpha_i` is the reference
//! twist (the holonomy of the transport across node `i`).
//!
//! All forces are analytic gradients of these energies and cost `O(n)` per
//! evaluation: [`stretch_forces`], [`bending_forces`], [`twisting_forces`]
//! and their sum [`discrete_elastic_rod`].

// ─────────────────────────────────────────────────────────────────────────────
// Vector helpers (no nalgebra: use [f64; 3])
// ─────────────────────────────────────────────────────────────────────────────

/// 3-component vector addition.
#[inline]
fn v3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// 3-component vector subtraction.
#[inline]
fn v3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scalar multiplication of a 3-vector.
#[inline]
fn v3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two 3-vectors.
#[inline]
fn v3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3-vector.
#[inline]
fn v3_norm(a: [f64; 3]) -> f64 {
    v3_dot(a, a).sqrt()
}

/// Normalize a 3-vector. Returns the zero vector if norm < eps.
#[inline]
fn v3_normalize(a: [f64; 3]) -> [f64; 3] {
    let n = v3_norm(a);
    if n < 1e-15 {
        [0.0, 0.0, 0.0]
    } else {
        v3_scale(a, 1.0 / n)
    }
}

/// Cross product of two 3-vectors.
#[inline]
fn v3_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame algebra (parallel transport of the reference directors)
// ─────────────────────────────────────────────────────────────────────────────

/// A unit vector perpendicular to the unit vector `t`.
///
/// The coordinate axis least aligned with `t` is projected onto the plane
/// orthogonal to `t`, which keeps the result well conditioned for any `t`.
fn perpendicular_unit(t: [f64; 3]) -> [f64; 3] {
    let axis = if t[0].abs() <= t[1].abs() && t[0].abs() <= t[2].abs() {
        [1.0, 0.0, 0.0]
    } else if t[1].abs() <= t[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    v3_normalize(v3_sub(axis, v3_scale(t, v3_dot(axis, t))))
}

/// Apply to `v` the minimal rotation that takes the unit vector `from` onto
/// the unit vector `to`.
///
/// The rotation axis is `from x to` and no spin about `to` is added, so a
/// frame that is orthogonal to `from` stays orthogonal to `to` and does not
/// twist: this is the parallel transport of a frame between two tangents.
/// With `c = from . to` and `k = from x to` the rotation is
/// `R v = c v + k x v + k (k . v) / (1 + c)`.
fn transport(v: [f64; 3], from: [f64; 3], to: [f64; 3]) -> [f64; 3] {
    let sum = v3_add(from, to);
    // 1 + cos(angle) evaluated as |from + to|^2 / 2, free of cancellation
    // when the two tangents are nearly antiparallel.
    let one_plus_cos = 0.5 * v3_dot(sum, sum);
    if one_plus_cos < 1e-20 {
        // `to` is antiparallel to `from`: the minimal rotation is a half turn
        // about any axis perpendicular to `from`.
        let axis = perpendicular_unit(from);
        return v3_sub(v3_scale(axis, 2.0 * v3_dot(axis, v)), v);
    }
    let cos = v3_dot(from, to);
    let k = v3_cross(from, to);
    let rotated = v3_add(v3_scale(v, cos), v3_cross(k, v));
    v3_add(rotated, v3_scale(k, v3_dot(k, v) / one_plus_cos))
}

/// Signed angle from `a` to `b` about the unit vector `axis` (right-handed).
///
/// `a` and `b` are expected to be orthogonal to `axis`.
#[inline]
fn signed_angle(a: [f64; 3], b: [f64; 3], axis: [f64; 3]) -> f64 {
    v3_dot(axis, v3_cross(a, b)).atan2(v3_dot(a, b))
}

/// Wrap an angle to `[-pi, pi]`.
#[inline]
fn wrap_to_pi(angle: f64) -> f64 {
    angle - std::f64::consts::TAU * (angle / std::f64::consts::TAU).round()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single node (particle) in a hair strand.
#[derive(Debug, Clone)]
pub struct HairNode {
    /// World-space position (m).
    pub position: [f64; 3],
    /// Velocity (m/s).
    pub velocity: [f64; 3],
    /// Accumulated external force (N).
    pub force: [f64; 3],
    /// Inverse mass (1/kg). 0.0 for static/pinned nodes.
    pub inv_mass: f64,
}

impl HairNode {
    /// Create a new dynamic hair node.
    pub fn new(position: [f64; 3], mass: f64) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        Self {
            position,
            velocity: [0.0; 3],
            force: [0.0; 3],
            inv_mass,
        }
    }

    /// Create a static (pinned) hair node.
    pub fn new_static(position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0; 3],
            force: [0.0; 3],
            inv_mass: 0.0,
        }
    }

    /// Returns true if this node is static (pinned).
    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
}

/// A single hair strand modelled as a chain of nodes (discrete elastic rod).
///
/// Adjacent nodes are connected by elastic segments. The strand stores rest
/// lengths, bend stiffness, and twist stiffness for each segment, plus the
/// material-frame state (`theta`, `ref_d1`, `ref_tangent`, `ref_twist`) that
/// gives the twist energy its meaning.
#[derive(Debug, Clone)]
pub struct HairStrand {
    /// Nodes along the strand (root first, tip last).
    pub nodes: Vec<HairNode>,
    /// Natural (rest) length of each segment between consecutive nodes (m).
    pub rest_lengths: Vec<f64>,
    /// Rest curvature (bending) at each internal node (rad/m). Zero → straight.
    pub rest_curvature: Vec<[f64; 3]>,
    /// Bending stiffness E_b (N·m²). Higher = stiffer bends.
    pub bend_stiffness: f64,
    /// Twist stiffness E_t (N·m²). Higher = stiffer torsion.
    pub twist_stiffness: f64,
    /// Stretch stiffness k_s (N/m).
    pub stretch_stiffness: f64,
    /// Aerodynamic drag coefficient per unit length (kg/(m·s)).
    pub drag_coeff: f64,
    /// Material twist angle θⱼ (rad) of every segment: the rotation about the
    /// segment tangent from the reference director [`ref_d1`](Self::ref_d1)
    /// to the first material director. One entry per segment.
    ///
    /// The angles of the first (root) and the last (tip) segment are
    /// boundary values owned by the caller: they are never changed by the
    /// solver, so imposing `theta[last] - theta[0] != 0` twists the strand.
    /// The interior angles are relaxed to the minimum of the twist energy by
    /// [`solve_twist_quasistatic`], which [`step_strand`] calls every step.
    pub theta: Vec<f64>,
    /// Reference director of every segment: a unit vector orthogonal to the
    /// segment tangent that is carried along the strand by parallel transport
    /// (a Bishop frame). It defines the zero of [`theta`](Self::theta).
    pub ref_d1: Vec<[f64; 3]>,
    /// Unit tangent each reference director was last synchronised with by
    /// [`sync_reference_frames`]. Energies and forces transport `ref_d1` from
    /// this tangent to the current one, so the reference twist is a function
    /// of the node positions even when they were moved without stepping.
    pub ref_tangent: Vec<[f64; 3]>,
    /// Accumulated reference twist αᵢ (rad) at every node as of the last
    /// [`sync_reference_frames`]. It is tracked continuously (not wrapped to
    /// `[-π, π]`), so winding a segment around its neighbour several times
    /// keeps winding up the twist. The two end nodes stay zero.
    pub ref_twist: Vec<f64>,
}

impl HairStrand {
    /// Create a straight hair strand from root to tip.
    ///
    /// # Arguments
    /// * `root` — root position (m)
    /// * `tip` — tip position (m)
    /// * `num_segments` — number of segments (nodes = num_segments + 1)
    /// * `mass_per_node` — mass of each non-root node (kg)
    /// * `bend_stiffness` — E_b (N·m²)
    /// * `twist_stiffness` — E_t (N·m²)
    /// * `stretch_stiffness` — k_s (N/m)
    ///
    /// The root node is pinned. The material frame starts untwisted: all
    /// material angles are zero and the reference directors share one
    /// orientation (see [`reset_reference_frames`](Self::reset_reference_frames)).
    pub fn new_straight(
        root: [f64; 3],
        tip: [f64; 3],
        num_segments: usize,
        mass_per_node: f64,
        bend_stiffness: f64,
        twist_stiffness: f64,
        stretch_stiffness: f64,
    ) -> Self {
        assert!(num_segments >= 1, "At least one segment required");
        let n_nodes = num_segments + 1;
        let mut nodes = Vec::with_capacity(n_nodes);
        let seg_length = {
            let d = v3_sub(tip, root);
            v3_norm(d) / num_segments as f64
        };
        for i in 0..n_nodes {
            let t = i as f64 / num_segments as f64;
            let pos = [
                root[0] + t * (tip[0] - root[0]),
                root[1] + t * (tip[1] - root[1]),
                root[2] + t * (tip[2] - root[2]),
            ];
            if i == 0 {
                nodes.push(HairNode::new_static(pos));
            } else {
                nodes.push(HairNode::new(pos, mass_per_node));
            }
        }
        let rest_lengths = vec![seg_length; num_segments];
        let rest_curvature = vec![[0.0; 3]; n_nodes];
        let mut strand = Self {
            nodes,
            rest_lengths,
            rest_curvature,
            bend_stiffness,
            twist_stiffness,
            stretch_stiffness,
            drag_coeff: 0.0,
            theta: Vec::new(),
            ref_d1: Vec::new(),
            ref_tangent: Vec::new(),
            ref_twist: Vec::new(),
        };
        strand.reset_reference_frames();
        strand
    }

    /// Number of nodes in the strand.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of segments (= num_nodes - 1).
    pub fn num_segments(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    /// Total rest length of the strand.
    pub fn total_rest_length(&self) -> f64 {
        self.rest_lengths.iter().sum()
    }

    /// Re-initialise the material-frame state from the current node positions.
    ///
    /// The reference directors become a Bishop frame: the first one is an
    /// arbitrary unit vector orthogonal to the first tangent and every other
    /// one is the parallel transport of its predecessor, so the reference
    /// twist is zero everywhere. All material angles are reset to zero.
    ///
    /// [`HairStrand::new_straight`] calls this. Call it again after changing
    /// the number of nodes or after placing the nodes by hand, when the
    /// motion that led to the new shape should not contribute any twist.
    pub fn reset_reference_frames(&mut self) {
        let n = self.nodes.len();
        let segments = n.saturating_sub(1);
        let mut tangents: Vec<[f64; 3]> = Vec::with_capacity(segments);
        let mut previous = [0.0, 1.0, 0.0];
        for j in 0..segments {
            let tangent = self.segment_tangent(j).unwrap_or(previous);
            tangents.push(tangent);
            previous = tangent;
        }
        let mut directors: Vec<[f64; 3]> = Vec::with_capacity(segments);
        for (j, &tangent) in tangents.iter().enumerate() {
            let carried = match j.checked_sub(1) {
                Some(prev) => transport(directors[prev], tangents[prev], tangent),
                None => perpendicular_unit(tangent),
            };
            directors.push(orthonormalize(carried, tangent));
        }
        self.ref_tangent = tangents;
        self.ref_d1 = directors;
        self.theta = vec![0.0; segments];
        self.ref_twist = vec![0.0; n];
    }

    /// Unit tangent of segment `j`, or `None` when its two nodes coincide.
    fn segment_tangent(&self, j: usize) -> Option<[f64; 3]> {
        let edge = v3_sub(self.nodes[j + 1].position, self.nodes[j].position);
        let length = v3_norm(edge);
        if length < 1e-15 {
            None
        } else {
            Some(v3_scale(edge, 1.0 / length))
        }
    }

    /// Whether the material-frame state matches the number of nodes.
    ///
    /// The twist energy and its forces are zero while this is `false`; call
    /// [`reset_reference_frames`](Self::reset_reference_frames) to repair it.
    fn frames_consistent(&self) -> bool {
        let segments = self.nodes.len().saturating_sub(1);
        self.theta.len() == segments
            && self.ref_d1.len() == segments
            && self.ref_tangent.len() == segments
            && self.ref_twist.len() == self.nodes.len()
    }

    /// Voronoi length `l̄ᵢ = ½ (lᵢ₋₁ + lᵢ)` of interior node `i`, computed from
    /// the rest lengths and clamped away from zero so it can be a divisor.
    fn voronoi_length(&self, i: usize) -> f64 {
        (0.5 * (self.rest_lengths[i - 1] + self.rest_lengths[i])).max(1e-15)
    }
}

/// Remove from `d` its component along the unit vector `t` and renormalise.
///
/// Falls back to an arbitrary unit vector orthogonal to `t` when `d` is
/// (numerically) parallel to `t`.
fn orthonormalize(d: [f64; 3], t: [f64; 3]) -> [f64; 3] {
    let projected = v3_normalize(v3_sub(d, v3_scale(t, v3_dot(d, t))));
    if v3_norm(projected) > 0.5 {
        projected
    } else {
        perpendicular_unit(t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Discrete elastic rod energies and forces
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the bending energy of a strand.
///
/// Uses the discrete curvature κ at each interior node:
///   κᵢ = (tᵢ - tᵢ₋₁) / (lᵢ + lᵢ₋₁) * 2
/// where tᵢ = normalized edge tangent.
///
/// E_bend = (E_b / 2) Σ |κᵢ - κ₀ᵢ|² × l̄ᵢ,   l̄ᵢ = ½ (lᵢ₋₁ + lᵢ)
///
/// The analytic gradient of this energy is [`bending_forces`].
///
/// Returns bending energy in Joules.
pub fn bending_energy(strand: &HairStrand) -> f64 {
    let nodes = &strand.nodes;
    let n = nodes.len();
    if n < 3 {
        return 0.0;
    }
    let mut energy = 0.0;
    for i in 1..n - 1 {
        let e0 = v3_sub(nodes[i].position, nodes[i - 1].position);
        let e1 = v3_sub(nodes[i + 1].position, nodes[i].position);
        let t0 = v3_normalize(e0);
        let t1 = v3_normalize(e1);
        let l_avg = strand.voronoi_length(i);
        // Discrete curvature vector κ = (t1 - t0) / l_avg
        let kappa = v3_scale(v3_sub(t1, t0), 1.0 / l_avg);
        // Curvature deviation from rest
        let dkappa = v3_sub(kappa, strand.rest_curvature[i]);
        let kappa_sq = v3_dot(dkappa, dkappa);
        energy += 0.5 * strand.bend_stiffness * kappa_sq * l_avg;
    }
    energy
}

/// Reference twist αᵢ (rad) at every node; the two end nodes are zero.
///
/// Segment `j` owns a reference director `d1ⱼ ⟂ tⱼ`. It is first carried from
/// the tangent it was synchronised with (`strand.ref_tangent[j]`) to the
/// current tangent by the minimal rotation (*time* transport, which adds no
/// spin). The reference twist at interior node `i` is the signed angle, about
/// `tᵢ`, between `d1ᵢ` and `d1ᵢ₋₁` carried across the node from `tᵢ₋₁` to `tᵢ`
/// by the minimal rotation (*space* transport). It is the holonomy of the
/// transport: it vanishes for any planar deformation and is non-zero when the
/// tangents sweep out a solid angle (Bergou et al. 2008).
///
/// The angle is returned on the branch closest to the value stored in
/// `strand.ref_twist`, so the twist stays continuous over many time steps.
/// Returns all zeros when the strand has fewer than three nodes or its
/// material-frame state does not match its node count.
pub fn reference_twists(strand: &HairStrand) -> Vec<f64> {
    let n = strand.nodes.len();
    let mut twists = vec![0.0_f64; n];
    if n < 3 || !strand.frames_consistent() {
        return twists;
    }
    // Current tangent and transported reference director of every segment.
    let frames: Vec<Option<([f64; 3], [f64; 3])>> = (0..n - 1)
        .map(|j| {
            strand
                .segment_tangent(j)
                .map(|t| (t, transport(strand.ref_d1[j], strand.ref_tangent[j], t)))
        })
        .collect();
    for i in 1..n - 1 {
        let stored = strand.ref_twist[i];
        twists[i] = match (frames[i - 1], frames[i]) {
            (Some((t_prev, d_prev)), Some((t_next, d_next))) => {
                let carried = transport(d_prev, t_prev, t_next);
                let principal = signed_angle(carried, d_next, t_next);
                stored + wrap_to_pi(principal - stored)
            }
            // A collapsed segment has no tangent: keep the last known twist.
            _ => stored,
        };
    }
    twists
}

/// Twist `mᵢ = θᵢ − θᵢ₋₁ + αᵢ` (rad) at every node; the end nodes are zero.
///
/// `θ` are the material angles `strand.theta` and `α` the reference twists
/// from [`reference_twists`].
pub fn node_twists(strand: &HairStrand) -> Vec<f64> {
    let n = strand.nodes.len();
    let mut twists = reference_twists(strand);
    if n < 3 || !strand.frames_consistent() {
        return twists;
    }
    // Interior node `i` sits between segments `i - 1` and `i`.
    for (twist, angles) in twists[1..n - 1].iter_mut().zip(strand.theta.windows(2)) {
        *twist += angles[1] - angles[0];
    }
    twists
}

/// Compute the twisting energy of a strand.
///
/// The twist at interior node `i` is `mᵢ = θᵢ − θᵢ₋₁ + αᵢ` (see
/// [`node_twists`]), where `θ` are the per-segment material angles and `α` the
/// reference twist of the parallel-transported reference frame. Then
///
/// E_twist = (E_t / 2) Σ mᵢ² / l̄ᵢ,   l̄ᵢ = ½ (lᵢ₋₁ + lᵢ)
///
/// A straight strand with equal material angles carries no twist energy, a
/// planar bend adds none either, and a difference of material angles between
/// neighbouring segments does. The gradient with respect to the node
/// positions is [`twisting_forces`].
///
/// Returns twisting energy in Joules (zero for strands with fewer than three
/// nodes).
pub fn twisting_energy(strand: &HairStrand) -> f64 {
    let n = strand.nodes.len();
    if n < 3 || !strand.frames_consistent() {
        return 0.0;
    }
    let twists = node_twists(strand);
    (1..n - 1)
        .map(|i| 0.5 * strand.twist_stiffness * twists[i] * twists[i] / strand.voronoi_length(i))
        .sum()
}

/// Synchronise the reference frames with the current node positions.
///
/// Every reference director is carried (minimal rotation) to the current
/// tangent of its segment, `strand.ref_tangent` is updated and the continuous
/// reference twist `strand.ref_twist` is stored. [`step_strand`] calls this
/// after moving the nodes. Call it after moving nodes by hand when the new
/// shape should be reached by the shortest rotation of every segment; call
/// [`HairStrand::reset_reference_frames`] instead to forget the motion.
///
/// The forces of [`twisting_forces`] are the exact gradient of
/// [`twisting_energy`] at a synchronised state.
pub fn sync_reference_frames(strand: &mut HairStrand) {
    let n = strand.nodes.len();
    if n < 2 || !strand.frames_consistent() {
        return;
    }
    // The twists must be evaluated against the *old* frames.
    let twists = reference_twists(strand);
    for j in 0..n - 1 {
        if let Some(tangent) = strand.segment_tangent(j) {
            let carried = transport(strand.ref_d1[j], strand.ref_tangent[j], tangent);
            strand.ref_d1[j] = orthonormalize(carried, tangent);
            strand.ref_tangent[j] = tangent;
        }
    }
    strand.ref_twist = twists;
}

/// Relax the interior material angles to the minimum of the twist energy.
///
/// The angles of the first and the last segment are boundary values and stay
/// untouched. For every interior segment `j` the torque balance
/// `∂E_twist/∂θⱼ = 0` reads `mⱼ / l̄ⱼ = mⱼ₊₁ / l̄ⱼ₊₁`: a symmetric positive
/// definite tridiagonal system, solved here with the Thomas algorithm in
/// `O(n)`. The twist then spreads over the strand in proportion to the
/// Voronoi lengths, so a uniform strand twists uniformly.
///
/// Strands with fewer than three segments have no interior angle and are
/// left unchanged.
pub fn solve_twist_quasistatic(strand: &mut HairStrand) {
    let n = strand.nodes.len();
    if n < 4 || !strand.frames_consistent() {
        return;
    }
    let segments = n - 1;
    let unknowns = segments - 2;
    let alpha = reference_twists(strand);
    let weight: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 || i == n - 1 {
                0.0
            } else {
                1.0 / strand.voronoi_length(i)
            }
        })
        .collect();
    // Row r is the balance of segment j = r + 1:
    //   -w_j θ_{j-1} + (w_j + w_{j+1}) θ_j - w_{j+1} θ_{j+1} = w_{j+1} α_{j+1} - w_j α_j
    let mut lower = vec![0.0_f64; unknowns];
    let mut diagonal = vec![0.0_f64; unknowns];
    let mut upper = vec![0.0_f64; unknowns];
    let mut rhs = vec![0.0_f64; unknowns];
    for r in 0..unknowns {
        let j = r + 1;
        lower[r] = -weight[j];
        diagonal[r] = weight[j] + weight[j + 1];
        upper[r] = -weight[j + 1];
        rhs[r] = weight[j + 1] * alpha[j + 1] - weight[j] * alpha[j];
    }
    // The prescribed boundary angles move to the right-hand side.
    rhs[0] += weight[1] * strand.theta[0];
    rhs[unknowns - 1] += weight[segments - 1] * strand.theta[segments - 1];
    // Thomas algorithm: forward elimination ...
    let mut upper_scaled = vec![0.0_f64; unknowns];
    let mut rhs_scaled = vec![0.0_f64; unknowns];
    upper_scaled[0] = upper[0] / diagonal[0];
    rhs_scaled[0] = rhs[0] / diagonal[0];
    for r in 1..unknowns {
        let pivot = diagonal[r] - lower[r] * upper_scaled[r - 1];
        upper_scaled[r] = upper[r] / pivot;
        rhs_scaled[r] = (rhs[r] - lower[r] * rhs_scaled[r - 1]) / pivot;
    }
    // ... and back substitution.
    let mut solution = rhs_scaled[unknowns - 1];
    strand.theta[unknowns] = solution;
    for r in (0..unknowns - 1).rev() {
        solution = rhs_scaled[r] - upper_scaled[r] * solution;
        strand.theta[r + 1] = solution;
    }
}

/// Compute stretch energy of a strand.
///
/// E_stretch = (k_s / 2) Σ (|eᵢ| - l₀ᵢ)² / l₀ᵢ
///
/// Returns stretch energy in Joules.
pub fn stretch_energy(strand: &HairStrand) -> f64 {
    let nodes = &strand.nodes;
    let n = nodes.len();
    if n < 2 {
        return 0.0;
    }
    let mut energy = 0.0;
    for i in 0..n - 1 {
        let e = v3_sub(nodes[i + 1].position, nodes[i].position);
        let length = v3_norm(e);
        let l0 = strand.rest_lengths[i];
        let strain = length - l0;
        energy += 0.5 * strand.stretch_stiffness * strain * strain / l0.max(1e-15);
    }
    energy
}

/// Apply root constraint: pin the first node to its rest position.
///
/// This modifies the root node's position and zeroes its velocity.
pub fn root_constraint(strand: &mut HairStrand) {
    if !strand.nodes.is_empty() {
        strand.nodes[0].velocity = [0.0; 3];
        strand.nodes[0].force = [0.0; 3];
        // Position is kept fixed by its inv_mass = 0 in integration
    }
}

/// Apply aerodynamic wind force to all nodes in a strand.
///
/// Uses a simple drag model: F_drag = -0.5 × ρ × C_d × A × (v_node - v_wind) × |v_rel|
/// Simplified per-node: F = drag_coeff × (v_wind - v_node) per segment length.
///
/// # Arguments
/// * `strand` — the hair strand to update
/// * `wind_velocity` — ambient wind velocity (m/s)
pub fn wind_force(strand: &mut HairStrand, wind_velocity: [f64; 3]) {
    let n = strand.nodes.len();
    if n < 2 {
        return;
    }
    let cd = strand.drag_coeff;
    for i in 0..n {
        if strand.nodes[i].is_static() {
            continue;
        }
        // Segment length associated with this node.
        let seg_len = if i == 0 {
            strand.rest_lengths[0]
        } else if i == n - 1 {
            *strand
                .rest_lengths
                .last()
                .expect("collection should not be empty")
        } else {
            0.5 * (strand.rest_lengths[i - 1] + strand.rest_lengths[i])
        };
        let v_rel = v3_sub(wind_velocity, strand.nodes[i].velocity);
        let drag = v3_scale(v_rel, cd * seg_len);
        strand.nodes[i].force = v3_add(strand.nodes[i].force, drag);
    }
}

/// Add `scale * v` to `forces[index]`.
#[inline]
fn add_scaled(forces: &mut [[f64; 3]], index: usize, v: [f64; 3], scale: f64) {
    forces[index] = v3_add(forces[index], v3_scale(v, scale));
}

/// Accumulate the stretch force of every segment.
///
/// `E = k_s / 2 (|e| - l0)² / l0` gives `∂E/∂x_{j+1} = k_s (|e| - l0) / l0 · ê`
/// and the opposite on node `j`.
fn accumulate_stretch_forces(strand: &HairStrand, forces: &mut [[f64; 3]]) {
    let nodes = &strand.nodes;
    for j in 0..nodes.len().saturating_sub(1) {
        let edge = v3_sub(nodes[j + 1].position, nodes[j].position);
        let length = v3_norm(edge);
        if length < 1e-15 {
            continue;
        }
        let l0 = strand.rest_lengths[j];
        let tension = strand.stretch_stiffness * (length - l0) / l0.max(1e-15);
        let direction = v3_scale(edge, 1.0 / length);
        add_scaled(forces, j, direction, tension);
        add_scaled(forces, j + 1, direction, -tension);
    }
}

/// Accumulate the bending force of every interior node.
///
/// With `r = κ - κ₀` and `κ = (t₁ - t₀) / l̄` the energy of node `i` is
/// `E_b l̄ |r|² / 2`, so `∂E/∂t₁ = E_b r` and `∂E/∂t₀ = -E_b r`. The tangent
/// of an edge `e` depends on its end nodes through `∂t/∂e = (I - t tᵀ) / |e|`,
/// which yields the three-node stencil below.
fn accumulate_bending_forces(strand: &HairStrand, forces: &mut [[f64; 3]]) {
    let nodes = &strand.nodes;
    for i in 1..nodes.len().saturating_sub(1) {
        let e0 = v3_sub(nodes[i].position, nodes[i - 1].position);
        let e1 = v3_sub(nodes[i + 1].position, nodes[i].position);
        let len0 = v3_norm(e0);
        let len1 = v3_norm(e1);
        if len0 < 1e-15 || len1 < 1e-15 {
            continue;
        }
        let t0 = v3_scale(e0, 1.0 / len0);
        let t1 = v3_scale(e1, 1.0 / len1);
        let l_avg = strand.voronoi_length(i);
        let kappa = v3_scale(v3_sub(t1, t0), 1.0 / l_avg);
        let r = v3_sub(kappa, strand.rest_curvature[i]);
        // dE/dt1 = E_b r; project out the component along each tangent.
        let g = v3_scale(r, strand.bend_stiffness);
        let g1 = v3_sub(g, v3_scale(t1, v3_dot(t1, g)));
        let g0 = v3_sub(g, v3_scale(t0, v3_dot(t0, g)));
        let grad_next = v3_scale(g1, 1.0 / len1);
        let grad_prev = v3_scale(g0, 1.0 / len0);
        add_scaled(forces, i - 1, grad_prev, -1.0);
        add_scaled(forces, i + 1, grad_next, -1.0);
        add_scaled(forces, i, v3_add(grad_next, grad_prev), 1.0);
    }
}

/// Accumulate the twist force of every interior node.
///
/// The energy depends on the positions only through the reference twist `αᵢ`
/// (the material angles and the reference directors are held fixed). Its
/// gradient is the holonomy gradient of Bergou et al. (2008):
///
/// ```text
/// (κb)ᵢ = 2 eᵢ₋₁ × eᵢ / (|eᵢ₋₁||eᵢ| + eᵢ₋₁·eᵢ)
/// ∂αᵢ/∂xᵢ₋₁ = -(κb)ᵢ / (2|eᵢ₋₁|),   ∂αᵢ/∂xᵢ₊₁ = (κb)ᵢ / (2|eᵢ|)
/// ∂αᵢ/∂xᵢ   = -(∂αᵢ/∂xᵢ₋₁ + ∂αᵢ/∂xᵢ₊₁)
/// ```
///
/// so the force on node `k` is `-Σᵢ (E_t mᵢ / l̄ᵢ) ∂αᵢ/∂xₖ`. The expression is
/// singular when a segment folds back onto its neighbour; such nodes are
/// skipped.
fn accumulate_twist_forces(strand: &HairStrand, forces: &mut [[f64; 3]]) {
    let nodes = &strand.nodes;
    let n = nodes.len();
    if n < 3 || !strand.frames_consistent() {
        return;
    }
    let twists = node_twists(strand);
    for i in 1..n - 1 {
        let e0 = v3_sub(nodes[i].position, nodes[i - 1].position);
        let e1 = v3_sub(nodes[i + 1].position, nodes[i].position);
        let len0 = v3_norm(e0);
        let len1 = v3_norm(e1);
        if len0 < 1e-15 || len1 < 1e-15 {
            continue;
        }
        let denominator = len0 * len1 + v3_dot(e0, e1);
        if denominator < 1e-12 * len0 * len1 {
            continue;
        }
        let curvature_binormal = v3_scale(v3_cross(e0, e1), 2.0 / denominator);
        let torque = strand.twist_stiffness * twists[i] / strand.voronoi_length(i);
        let grad_prev = v3_scale(curvature_binormal, -0.5 / len0);
        let grad_next = v3_scale(curvature_binormal, 0.5 / len1);
        add_scaled(forces, i - 1, grad_prev, -torque);
        add_scaled(forces, i + 1, grad_next, -torque);
        add_scaled(forces, i, v3_add(grad_prev, grad_next), torque);
    }
}

/// Stretch forces (N) on every node: `-∂E_stretch/∂x`, in `O(n)`.
///
/// The force on pinned nodes (the reaction load) is included.
pub fn stretch_forces(strand: &HairStrand) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0_f64; 3]; strand.nodes.len()];
    accumulate_stretch_forces(strand, &mut forces);
    forces
}

/// Bending forces (N) on every node: `-∂E_bend/∂x`, in `O(n)`.
///
/// The force on pinned nodes (the reaction load) is included.
pub fn bending_forces(strand: &HairStrand) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0_f64; 3]; strand.nodes.len()];
    accumulate_bending_forces(strand, &mut forces);
    forces
}

/// Twisting forces (N) on every node: `-∂E_twist/∂x`, in `O(n)`.
///
/// Material angles and reference directors are held fixed, so the result is
/// the exact gradient of [`twisting_energy`] when the reference frames are
/// synchronised with the node positions (see [`sync_reference_frames`]).
/// The force on pinned nodes (the reaction load) is included.
pub fn twisting_forces(strand: &HairStrand) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0_f64; 3]; strand.nodes.len()];
    accumulate_twist_forces(strand, &mut forces);
    forces
}

/// Compute the elastic forces of a strand: stretch + bending + twisting.
///
/// The forces are the negative gradient of
/// `stretch_energy + bending_energy + twisting_energy` with respect to the
/// node positions, evaluated analytically in a single `O(n)` pass without
/// copying the strand. Pinned (static) nodes receive no force.
///
/// Returns forces as `Vec<[f64; 3]>` aligned with node indices.
pub fn discrete_elastic_rod(strand: &HairStrand) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0_f64; 3]; strand.nodes.len()];
    accumulate_stretch_forces(strand, &mut forces);
    accumulate_bending_forces(strand, &mut forces);
    accumulate_twist_forces(strand, &mut forces);
    for (force, node) in forces.iter_mut().zip(strand.nodes.iter()) {
        if node.is_static() {
            *force = [0.0; 3];
        }
    }
    forces
}

// ─────────────────────────────────────────────────────────────────────────────
// Hair-hair collision (strand-strand contact)
// ─────────────────────────────────────────────────────────────────────────────

/// Detect and respond to contact between two hair strands.
///
/// For each pair of nodes from different strands within `contact_radius`,
/// applies a repulsive spring force.
///
/// # Arguments
/// * `strand_a` — first strand (modified in place)
/// * `strand_b` — second strand (modified in place)
/// * `contact_radius` — minimum separation distance (m)
/// * `repulsion_stiffness` — spring constant for repulsion (N/m)
pub fn hair_hair_collision(
    strand_a: &mut HairStrand,
    strand_b: &mut HairStrand,
    contact_radius: f64,
    repulsion_stiffness: f64,
) {
    let na = strand_a.nodes.len();
    let nb = strand_b.nodes.len();
    for i in 0..na {
        for j in 0..nb {
            let d = v3_sub(strand_b.nodes[j].position, strand_a.nodes[i].position);
            let dist = v3_norm(d);
            if dist < contact_radius && dist > 1e-15 {
                let penetration = contact_radius - dist;
                let dir = v3_scale(d, 1.0 / dist);
                let impulse = v3_scale(dir, repulsion_stiffness * penetration);
                if !strand_a.nodes[i].is_static() {
                    strand_a.nodes[i].force = v3_sub(strand_a.nodes[i].force, impulse);
                }
                if !strand_b.nodes[j].is_static() {
                    strand_b.nodes[j].force = v3_add(strand_b.nodes[j].force, impulse);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Style target (rest-shape styling)
// ─────────────────────────────────────────────────────────────────────────────

/// Set the rest curvature of a strand to match a target set of positions.
///
/// The rest curvature at each node is computed from the target positions and
/// stored in `strand.rest_curvature`. This allows the strand to restore a
/// styled shape.
///
/// # Arguments
/// * `strand` — the strand to style
/// * `target_positions` — desired node positions (same count as strand nodes)
pub fn style_target(strand: &mut HairStrand, target_positions: &[[f64; 3]]) {
    assert_eq!(
        target_positions.len(),
        strand.nodes.len(),
        "target_positions must match node count"
    );
    let n = strand.nodes.len();
    for i in 0..n {
        if i == 0 || i == n - 1 {
            strand.rest_curvature[i] = [0.0; 3];
            continue;
        }
        let e0 = v3_sub(target_positions[i], target_positions[i - 1]);
        let e1 = v3_sub(target_positions[i + 1], target_positions[i]);
        let t0 = v3_normalize(e0);
        let t1 = v3_normalize(e1);
        let l0 = v3_norm(e0).max(1e-15);
        let l1 = v3_norm(e1).max(1e-15);
        let l_avg = 0.5 * (l0 + l1);
        let kappa = v3_scale(v3_sub(t1, t0), 1.0 / l_avg);
        strand.rest_curvature[i] = kappa;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple explicit integration step
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a hair strand one time step with explicit Euler integration.
///
/// Applies elastic rod forces, then integrates velocities and positions.
/// Afterwards the reference frames are carried to the new tangents
/// ([`sync_reference_frames`]) and the interior material angles are relaxed
/// ([`solve_twist_quasistatic`]).
///
/// # Arguments
/// * `strand` — the strand to integrate
/// * `gravity` — gravitational acceleration (m/s²)
/// * `dt` — time step (s)
pub fn step_strand(strand: &mut HairStrand, gravity: [f64; 3], dt: f64) {
    // Compute elastic forces.
    let elastic_forces = discrete_elastic_rod(strand);
    // Velocity damping coefficient (per step): prevents explosion with stiff springs.
    let damping = 0.98_f64;
    for (node, el_force) in strand.nodes.iter_mut().zip(elastic_forces.iter()) {
        if node.is_static() {
            continue;
        }
        let m_inv = node.inv_mass;
        let mass = 1.0 / m_inv.max(1e-15);
        // Gravity force = mass × g.
        let grav_f = v3_scale(gravity, mass);
        let total_force = v3_add(v3_add(node.force, *el_force), grav_f);
        let accel = v3_scale(total_force, m_inv);
        let new_vel = v3_scale(v3_add(node.velocity, v3_scale(accel, dt)), damping);
        node.velocity = new_vel;
        node.position = v3_add(node.position, v3_scale(node.velocity, dt));
        node.force = [0.0; 3];
    }
    sync_reference_frames(strand);
    solve_twist_quasistatic(strand);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // Helper: create a straight vertical strand.
    fn make_strand(num_segs: usize) -> HairStrand {
        HairStrand::new_straight(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            num_segs,
            0.001,
            1e-3,
            1e-4,
            1000.0,
        )
    }

    // 1. Strand creation: correct node count.
    #[test]
    fn test_strand_node_count() {
        let s = make_strand(4);
        assert_eq!(s.num_nodes(), 5, "4 segments → 5 nodes");
    }

    // 2. Strand creation: correct segment count.
    #[test]
    fn test_strand_segment_count() {
        let s = make_strand(4);
        assert_eq!(s.num_segments(), 4, "should have 4 segments");
    }

    // 3. Root node is static.
    #[test]
    fn test_root_is_static() {
        let s = make_strand(4);
        assert!(s.nodes[0].is_static(), "root node should be static");
    }

    // 4. Non-root nodes are dynamic.
    #[test]
    fn test_non_root_dynamic() {
        let s = make_strand(4);
        for i in 1..s.num_nodes() {
            assert!(!s.nodes[i].is_static(), "node {i} should be dynamic");
        }
    }

    // 5. Total rest length matches root-to-tip distance.
    #[test]
    fn test_total_rest_length() {
        let s = make_strand(5);
        let len = s.total_rest_length();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "Total rest length should be 1.0, got {len}"
        );
    }

    // 6. Bending energy of a straight strand is zero.
    #[test]
    fn test_bending_energy_straight_is_zero() {
        let s = make_strand(5);
        let e = bending_energy(&s);
        assert!(
            e.abs() < 1e-12,
            "Straight strand bending energy should be 0, got {e}"
        );
    }

    // 7. Twisting energy of a straight strand is zero.
    #[test]
    fn test_twisting_energy_straight_is_zero() {
        let s = make_strand(5);
        let e = twisting_energy(&s);
        assert!(
            e.abs() < 1e-12,
            "Straight strand twist energy should be 0, got {e}"
        );
    }

    // 8. Stretch energy of an undeformed strand is zero.
    #[test]
    fn test_stretch_energy_zero_undeformed() {
        let s = make_strand(5);
        let e = stretch_energy(&s);
        assert!(
            e.abs() < 1e-12,
            "Undeformed stretch energy should be 0, got {e}"
        );
    }

    // 9. Bending energy increases when a node is displaced.
    #[test]
    fn test_bending_energy_increases_on_displacement() {
        let mut s = make_strand(4);
        let e0 = bending_energy(&s);
        s.nodes[2].position[0] += 0.1; // lateral displacement
        let e1 = bending_energy(&s);
        assert!(
            e1 > e0,
            "Bending energy should increase after displacement: e0={e0}, e1={e1}"
        );
    }

    // 10. Stretch energy increases when a node is displaced along axis.
    #[test]
    fn test_stretch_energy_increases_on_displacement() {
        let mut s = make_strand(4);
        let e0 = stretch_energy(&s);
        s.nodes[4].position[1] += 0.2; // extend tip
        let e1 = stretch_energy(&s);
        assert!(
            e1 > e0,
            "Stretch energy should increase after extension: e0={e0}, e1={e1}"
        );
    }

    // 11. Root constraint zeroes root node velocity.
    #[test]
    fn test_root_constraint_zeroes_velocity() {
        let mut s = make_strand(4);
        s.nodes[0].velocity = [1.0, 2.0, 3.0];
        root_constraint(&mut s);
        assert_eq!(
            s.nodes[0].velocity,
            [0.0, 0.0, 0.0],
            "Root velocity should be zeroed"
        );
    }

    // 12. Wind force with zero drag coefficient produces zero force.
    #[test]
    fn test_wind_force_zero_drag() {
        let mut s = make_strand(4);
        s.drag_coeff = 0.0;
        wind_force(&mut s, [10.0, 0.0, 0.0]);
        for node in &s.nodes {
            assert_eq!(node.force, [0.0; 3], "Zero drag → zero wind force");
        }
    }

    // 13. Wind force with positive drag applies force on dynamic nodes.
    #[test]
    fn test_wind_force_nonzero_drag() {
        let mut s = make_strand(4);
        s.drag_coeff = 0.01;
        wind_force(&mut s, [1.0, 0.0, 0.0]);
        // Node 1..4 should have non-zero force.
        let total_fx: f64 = s.nodes[1..].iter().map(|n| n.force[0]).sum();
        assert!(
            total_fx > EPS,
            "Wind should produce force in x-direction, got {total_fx}"
        );
    }

    // 14. Wind force is zero on static (root) node.
    #[test]
    fn test_wind_force_not_on_static_node() {
        let mut s = make_strand(4);
        s.drag_coeff = 0.1;
        wind_force(&mut s, [10.0, 0.0, 0.0]);
        assert_eq!(
            s.nodes[0].force, [0.0; 3],
            "Static node should not receive wind force"
        );
    }

    // 15. Discrete elastic rod returns forces of correct size.
    #[test]
    fn test_discrete_elastic_rod_force_size() {
        let s = make_strand(4);
        let forces = discrete_elastic_rod(&s);
        assert_eq!(
            forces.len(),
            s.num_nodes(),
            "Force count should equal node count"
        );
    }

    // 16. Static node gets zero elastic force.
    #[test]
    fn test_discrete_elastic_rod_static_zero_force() {
        let s = make_strand(4);
        let forces = discrete_elastic_rod(&s);
        let f_root = v3_norm(forces[0]);
        assert!(
            f_root < EPS,
            "Static root node should have zero force, got {f_root}"
        );
    }

    // 17. Hair-hair collision: no force when strands are far apart.
    #[test]
    fn test_hair_collision_no_force_when_far() {
        let mut s1 = make_strand(3);
        let mut s2 = HairStrand::new_straight(
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            3,
            0.001,
            1e-3,
            1e-4,
            1000.0,
        );
        hair_hair_collision(&mut s1, &mut s2, 0.1, 1000.0);
        for n in &s1.nodes {
            assert_eq!(n.force, [0.0; 3], "No collision → zero force on s1");
        }
    }

    // 18. Hair-hair collision: repulsion when strands overlap.
    #[test]
    fn test_hair_collision_repulsion() {
        let mut s1 = make_strand(1);
        let mut s2 = make_strand(1);
        // Place s2 very close to s1 at node 1.
        s2.nodes[1].position = [0.01, 1.0, 0.0];
        hair_hair_collision(&mut s1, &mut s2, 0.1, 1000.0);
        let f_x = s1.nodes[1].force[0];
        // They should be pushed apart (force < 0 in x for s1 since s2 is at +x).
        assert!(
            f_x < 0.0 || f_x.abs() > EPS,
            "Collision should produce repulsive force"
        );
    }

    // 19. Style target: rest curvature at endpoints is zero.
    #[test]
    fn test_style_target_endpoints_zero() {
        let mut s = make_strand(4);
        let targets: Vec<[f64; 3]> = s.nodes.iter().map(|n| n.position).collect();
        style_target(&mut s, &targets);
        assert_eq!(s.rest_curvature[0], [0.0; 3]);
        assert_eq!(s.rest_curvature[s.num_nodes() - 1], [0.0; 3]);
    }

    // 20. Style target: straight strand has zero rest curvature everywhere.
    #[test]
    fn test_style_target_straight_zero_curvature() {
        let mut s = make_strand(4);
        let targets: Vec<[f64; 3]> = s.nodes.iter().map(|n| n.position).collect();
        style_target(&mut s, &targets);
        for (i, kappa) in s.rest_curvature.iter().enumerate() {
            let mag = v3_norm(*kappa);
            assert!(
                mag < 1e-8,
                "Straight target should have zero curvature at node {i}, got {mag}"
            );
        }
    }

    // 21. Stepping a free strand under gravity lowers the tip position.
    // Use a very soft strand (tiny stiffness) to keep explicit Euler stable.
    #[test]
    fn test_step_strand_gravity_lowers_tip() {
        let mut s = HairStrand::new_straight(
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            4,
            0.001,
            1e-6, // very soft bend
            1e-7, // very soft twist
            1.0,  // very soft stretch
        );
        let gravity = [0.0, -9.81, 0.0];
        let initial_y = s.nodes[4].position[1];
        for _ in 0..10 {
            step_strand(&mut s, gravity, 1.0 / 60.0);
        }
        let final_y = s.nodes[4].position[1];
        assert!(
            final_y < initial_y,
            "Tip should fall under gravity: initial={initial_y}, final={final_y}"
        );
    }

    // 22. Root node does not move during step_strand.
    #[test]
    fn test_step_strand_root_fixed() {
        let mut s = make_strand(4);
        let initial_root = s.nodes[0].position;
        for _ in 0..100 {
            step_strand(&mut s, [0.0, -9.81, 0.0], 1.0 / 60.0);
        }
        assert_eq!(
            s.nodes[0].position, initial_root,
            "Root should remain fixed"
        );
    }

    // 23. Bending energy is symmetric in sign of displacement.
    #[test]
    fn test_bending_energy_symmetric() {
        let mut s_pos = make_strand(4);
        let mut s_neg = make_strand(4);
        s_pos.nodes[2].position[0] += 0.05;
        s_neg.nodes[2].position[0] -= 0.05;
        let e_pos = bending_energy(&s_pos);
        let e_neg = bending_energy(&s_neg);
        assert!(
            (e_pos - e_neg).abs() < 1e-10,
            "Bending energy should be symmetric: e_pos={e_pos}, e_neg={e_neg}"
        );
    }

    // 24. Strand with 1 segment has zero bending energy (not enough nodes).
    #[test]
    fn test_bending_energy_single_segment() {
        let s = make_strand(1);
        let e = bending_energy(&s);
        assert!(
            e.abs() < EPS,
            "Single segment strand has no bending energy, got {e}"
        );
    }

    // 25. A lateral (planar) displacement bends the strand but does not twist it:
    // the transported reference frame follows the bend, so the twist energy is
    // exactly the one of the straight strand.
    #[test]
    fn test_twisting_energy_unchanged_by_planar_displacement() {
        let mut s = make_strand(5);
        let e0 = twisting_energy(&s);
        s.nodes[3].position[0] += 0.2;
        let e1 = twisting_energy(&s);
        assert!(
            bending_energy(&s) > 0.0,
            "the displacement must actually bend the strand"
        );
        assert!(
            (e1 - e0).abs() < 1e-18,
            "planar bending must not change the twist energy: e0={e0} e1={e1}"
        );
    }

    // 26. HairNode default velocity and force are zero.
    #[test]
    fn test_hair_node_defaults() {
        let node = HairNode::new([1.0, 2.0, 3.0], 0.01);
        assert_eq!(node.velocity, [0.0; 3]);
        assert_eq!(node.force, [0.0; 3]);
    }

    // 27. Elastic forces on straight strand are close to zero.
    #[test]
    fn test_elastic_forces_straight_near_zero() {
        let s = make_strand(5);
        let forces = discrete_elastic_rod(&s);
        for (i, f) in forces.iter().enumerate() {
            if s.nodes[i].is_static() {
                continue;
            }
            let mag = v3_norm(*f);
            assert!(
                mag < 1e-3,
                "Elastic forces on undeformed strand should be ~0 at node {i}, got {mag}"
            );
        }
    }
}
