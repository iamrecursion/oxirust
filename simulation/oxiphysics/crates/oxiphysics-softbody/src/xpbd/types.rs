//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::strain_limit::{StrainLimitConfig, XpbdStrainLimiter};
use oxiphysics_core::math::Vec3;
use std::f64::consts::{PI, TAU};

/// A self-contained XPBD simulation body.
pub struct XpbdBody {
    /// Current particle positions.
    pub positions: Vec<Vec3>,
    /// Positions at the beginning of the current sub-step.
    pub prev_positions: Vec<Vec3>,
    /// Particle velocities.
    pub velocities: Vec<Vec3>,
    /// Inverse masses (0 = fixed/pinned particle).
    pub inv_masses: Vec<f64>,
    /// Distance constraints.
    pub distance_constraints: Vec<XpbdDistanceConstraint>,
    /// Volume constraints.
    pub volume_constraints: Vec<XpbdVolumeConstraint>,
    /// Bending constraints.
    pub bending_constraints: Vec<XpbdBendingConstraint>,
    /// Optional per-triangle strain limiter; `None` (off) by default.
    ///
    /// While it is `Some`, [`XpbdBody::step`] and [`XpbdBody::step_with_stats`]
    /// run it once per sub-step, after the constraint projections and before
    /// the velocities are derived from the position change.  Set it with
    /// [`XpbdBody::enable_strain_limiting`], or assign a limiter built with
    /// [`XpbdStrainLimiter::new`] to limit towards an explicit rest shape.
    pub strain_limiter: Option<XpbdStrainLimiter>,
}
impl XpbdBody {
    /// Create an empty body, pre-allocating capacity for `n_particles`.
    pub fn new(n_particles: usize) -> Self {
        Self {
            positions: Vec::with_capacity(n_particles),
            prev_positions: Vec::with_capacity(n_particles),
            velocities: Vec::with_capacity(n_particles),
            inv_masses: Vec::with_capacity(n_particles),
            distance_constraints: Vec::new(),
            volume_constraints: Vec::new(),
            bending_constraints: Vec::new(),
            strain_limiter: None,
        }
    }
    /// Add a particle; returns its index.
    ///
    /// `mass = 0` creates a pinned (static) particle.
    pub fn add_particle(&mut self, pos: Vec3, mass: f64) -> usize {
        let idx = self.positions.len();
        self.positions.push(pos);
        self.prev_positions.push(pos);
        self.velocities.push(Vec3::zeros());
        let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.inv_masses.push(inv_m);
        idx
    }
    /// Pin a particle (set its inverse mass to 0).
    pub fn pin_particle(&mut self, idx: usize) {
        self.inv_masses[idx] = 0.0;
    }
    /// Add a distance constraint; rest length = current distance.
    pub fn add_distance_constraint(&mut self, a: usize, b: usize, compliance: f64) {
        let rest_length = (self.positions[b] - self.positions[a]).norm();
        self.distance_constraints
            .push(XpbdDistanceConstraint::new(a, b, rest_length, compliance));
    }
    /// Add a volume (tetrahedral) constraint; rest volume = current volume.
    pub fn add_volume_constraint(&mut self, p: [usize; 4], compliance: f64) {
        let [i0, i1, i2, i3] = p;
        let a = self.positions[i1] - self.positions[i0];
        let b = self.positions[i2] - self.positions[i0];
        let c = self.positions[i3] - self.positions[i0];
        let rest_volume = a.dot(&b.cross(&c)) / 6.0;
        self.volume_constraints
            .push(XpbdVolumeConstraint::new(p, rest_volume, compliance));
    }
    /// Turn on per-triangle strain limiting for `triangles` (index triples into
    /// the particles), with the current particle positions as the rest shape,
    /// the way [`XpbdBody::add_distance_constraint`] takes the current distance
    /// as the rest length.  An earlier limiter is replaced.
    ///
    /// From now on [`XpbdBody::step`] and [`XpbdBody::step_with_stats`] clamp
    /// the singular values of every triangle's deformation gradient into
    /// `[config.min_stretch, config.max_stretch]` once per sub-step, after the
    /// constraint projections and before the velocity update.  Strain limiting
    /// is off until this is called.  To limit towards a rest shape other than
    /// the current positions, assign `strain_limiter` a limiter built with
    /// [`XpbdStrainLimiter::new`].
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidParameter`] if a triangle refers to a particle
    /// that does not exist; the body, and its earlier limiter, are left
    /// unchanged then.
    pub fn enable_strain_limiting(
        &mut self,
        triangles: Vec<[usize; 3]>,
        config: StrainLimitConfig,
    ) -> crate::Result<()> {
        self.strain_limiter = Some(XpbdStrainLimiter::new(triangles, &self.positions, config)?);
        Ok(())
    }
    /// Turn per-triangle strain limiting off; later steps are plain XPBD again.
    pub fn disable_strain_limiting(&mut self) {
        self.strain_limiter = None;
    }
    /// Advance the simulation by `dt` using `n_substeps` sub-steps and
    /// `n_iter` constraint-projection iterations per sub-step.
    ///
    /// When a strain limiter is enabled ([`XpbdBody::enable_strain_limiting`]),
    /// it runs once per sub-step, after the constraint iterations and before
    /// the velocities are derived from the position change.
    pub fn step(&mut self, dt: f64, gravity: Vec3, n_substeps: usize, n_iter: usize) {
        let h = dt / (n_substeps as f64);
        for _ in 0..n_substeps {
            self.prev_positions.clone_from(&self.positions);
            for i in 0..self.positions.len() {
                if self.inv_masses[i] == 0.0 {
                    continue;
                }
                self.velocities[i] += gravity * h;
                self.positions[i] += self.velocities[i] * h;
            }
            for c in &mut self.distance_constraints {
                c.reset_lambda();
            }
            for c in &mut self.volume_constraints {
                c.reset_lambda();
            }
            for c in &mut self.bending_constraints {
                c.reset_lambda();
            }
            for _ in 0..n_iter {
                for c in &mut self.distance_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                }
                for c in &mut self.volume_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                }
                for c in &mut self.bending_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                }
            }
            self.limit_strain();
            for i in 0..self.positions.len() {
                if self.inv_masses[i] == 0.0 {
                    continue;
                }
                self.velocities[i] = (self.positions[i] - self.prev_positions[i]) / h;
            }
        }
    }
    /// Run the strain-limiting pass if a limiter is enabled; returns the number
    /// of triangle projections it performed.
    fn limit_strain(&mut self) -> usize {
        match self.strain_limiter.as_mut() {
            Some(limiter) => limiter.apply(&mut self.positions, &self.inv_masses),
            None => 0,
        }
    }
    /// Total kinetic energy: ½ Σ m_i |v_i|².
    pub fn total_kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .zip(self.inv_masses.iter())
            .filter(|(_, w)| **w > 0.0)
            .map(|(v, w)| 0.5 * (1.0 / w) * v.norm_squared())
            .sum()
    }
}
impl XpbdBody {
    /// Step with statistics collection.
    ///
    /// Advances the body exactly like [`XpbdBody::step`], strain limiter
    /// included; its triangle projections are counted in
    /// [`XpbdSolverStats::total_projections`].
    pub fn step_with_stats(
        &mut self,
        dt: f64,
        gravity: Vec3,
        n_substeps: usize,
        n_iter: usize,
    ) -> XpbdSolverStats {
        let mut stats = XpbdSolverStats {
            substeps: n_substeps,
            iterations_per_substep: n_iter,
            ..Default::default()
        };
        let h = dt / (n_substeps as f64);
        for _ in 0..n_substeps {
            self.prev_positions.clone_from(&self.positions);
            for i in 0..self.positions.len() {
                if self.inv_masses[i] == 0.0 {
                    continue;
                }
                self.velocities[i] += gravity * h;
                self.positions[i] += self.velocities[i] * h;
            }
            for c in &mut self.distance_constraints {
                c.reset_lambda();
            }
            for c in &mut self.volume_constraints {
                c.reset_lambda();
            }
            for c in &mut self.bending_constraints {
                c.reset_lambda();
            }
            for _ in 0..n_iter {
                for c in &mut self.distance_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                    stats.total_projections += 1;
                }
                for c in &mut self.volume_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                    stats.total_projections += 1;
                }
                for c in &mut self.bending_constraints {
                    c.solve(&mut self.positions, &self.inv_masses, h);
                    stats.total_projections += 1;
                }
            }
            stats.total_projections += self.limit_strain();
            for i in 0..self.positions.len() {
                if self.inv_masses[i] == 0.0 {
                    continue;
                }
                self.velocities[i] = (self.positions[i] - self.prev_positions[i]) / h;
            }
        }
        let mut max_res = 0.0_f64;
        let mut sum_res = 0.0_f64;
        let mut count = 0usize;
        for c in &self.distance_constraints {
            let r = c.evaluate(&self.positions).abs();
            max_res = max_res.max(r);
            sum_res += r;
            count += 1;
        }
        stats.max_residual = max_res;
        stats.avg_residual = if count > 0 {
            sum_res / count as f64
        } else {
            0.0
        };
        stats.kinetic_energy = self.total_kinetic_energy();
        stats
    }
    /// Number of particles.
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }
    /// Number of all constraints (distance + volume + bending).
    pub fn constraint_count(&self) -> usize {
        self.distance_constraints.len()
            + self.volume_constraints.len()
            + self.bending_constraints.len()
    }
    /// Add a bending constraint between four particles. Rest angle is read from positions.
    pub fn add_bending_constraint(
        &mut self,
        p0: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        compliance: f64,
    ) {
        self.bending_constraints.push(XpbdBendingConstraint::new(
            p0,
            p1,
            p2,
            p3,
            &self.positions,
            compliance,
        ));
    }
    /// Centre of mass position, weighted by mass.
    pub fn centre_of_mass(&self) -> Vec3 {
        let mut total_mass = 0.0;
        let mut com = Vec3::zeros();
        for (pos, &w) in self.positions.iter().zip(self.inv_masses.iter()) {
            let m = if w > 0.0 { 1.0 / w } else { 0.0 };
            com += *pos * m;
            total_mass += m;
        }
        if total_mass > 1e-30 {
            com / total_mass
        } else {
            Vec3::zeros()
        }
    }
    /// Maximum velocity magnitude across all particles.
    pub fn max_velocity(&self) -> f64 {
        self.velocities
            .iter()
            .map(|v| v.norm())
            .fold(0.0_f64, f64::max)
    }
}
/// Shortest shared-edge length (m) that still defines the hinge axis of an
/// [`XpbdBendingConstraint`].
const BENDING_MIN_EDGE_LENGTH: f64 = 1e-12;
/// Smallest height of a wing particle above the hinge line, relative to the
/// shared-edge length, for which its triangle still has a normal.
const BENDING_MIN_WING_ASPECT: f64 = 1e-9;
/// Interior dihedral angle of a bending hinge together with its gradient.
struct HingeState {
    /// Interior dihedral angle in `[0, 2π)`; `π` for a flat hinge.
    angle: f64,
    /// Gradient of `angle` with respect to the positions of `[p0, p1, p2, p3]`
    /// (rad/m).
    gradients: [Vec3; 4],
}
/// Map an angle to `(-π, π]`, the shortest rotation that carries one angle
/// onto another.
///
/// Angles already in that range are returned unchanged (bit for bit), so small
/// angles keep their full precision.
fn wrap_angle(angle: f64) -> f64 {
    if angle > -PI && angle <= PI {
        return angle;
    }
    let wrapped = angle - TAU * ((angle + PI) / TAU).floor();
    if wrapped <= -PI { PI } else { wrapped }
}
/// XPBD dihedral-angle bending constraint for two triangles sharing an edge.
///
/// `p0`/`p1` form the shared edge; `p2`/`p3` are the opposing "wing" vertices
/// of the triangles `(p0, p1, p2)` and `(p0, p1, p3)`.
///
/// The constrained quantity is the *interior dihedral angle* `θ ∈ [0, 2π)` of
/// the hinge.  It is `π` for a flat hinge, smaller than `π` when the wings are
/// folded towards the side that the normal `(p1 − p0) × (p2 − p0)` of the first
/// triangle points to, and larger than `π` when they are folded to the other
/// side.  With `e = p1 − p0`, `ê = e / |e|` and the area vectors
/// `N1 = e × (p2 − p0)`, `N2 = (p3 − p0) × e` (equal for a flat hinge):
///
/// ```text
/// θ = π − atan2((N2 × N1) · ê, N2 · N1)
/// C = wrap(θ − θ_rest)                      wrap: angle → (−π, π]
/// ```
///
/// Because `θ` is signed it is smooth through the flat state, so the
/// constraint tells a fold to one side from a fold to the other side and always
/// pushes the hinge back towards its rest angle (the shortest way round).  The
/// gradient is the exact one of Bridson, Marino and Fedkiw (2003) on all four
/// particles:
///
/// ```text
/// ∂θ/∂p2 = −|e| N1 / |N1|²          ∂θ/∂p3 = −|e| N2 / |N2|²
/// ∂θ/∂p0 = −((p2 − p1) · ê) N1 / |N1|² − ((p3 − p1) · ê) N2 / |N2|²
/// ∂θ/∂p1 = −(∂θ/∂p0 + ∂θ/∂p2 + ∂θ/∂p3)
/// ```
///
/// The four gradients sum to zero and carry no net torque, so a projection
/// conserves linear and angular momentum; the shared-edge particles take part
/// in the correction, and their inverse masses enter the XPBD denominator
/// together with those of the wings.
pub struct XpbdBendingConstraint {
    /// First shared-edge vertex index.
    pub p0: usize,
    /// Second shared-edge vertex index.
    pub p1: usize,
    /// Wing vertex of triangle 1.
    pub p2: usize,
    /// Wing vertex of triangle 2.
    pub p3: usize,
    /// Equilibrium interior dihedral angle (radians, `π` for a flat hinge).
    ///
    /// See the type documentation for the orientation convention; any value is
    /// accepted because the constraint compares angles modulo `2π`.
    pub rest_angle: f64,
    /// Compliance α.
    pub compliance: f64,
    /// Lagrange multiplier accumulated during the current sub-step.
    pub lambda: f64,
}
impl XpbdBendingConstraint {
    /// Build a bending constraint; rest angle is read from `positions`.
    ///
    /// A degenerate hinge (see [`current_angle`](Self::current_angle)) has no
    /// dihedral angle; the constraint is then built flat, with a rest angle
    /// of `π`.
    pub fn new(
        p0: usize,
        p1: usize,
        p2: usize,
        p3: usize,
        positions: &[Vec3],
        compliance: f64,
    ) -> Self {
        let rest_angle = Self::hinge_state(
            &positions[p0],
            &positions[p1],
            &positions[p2],
            &positions[p3],
        )
        .map_or(PI, |state| state.angle);
        Self {
            p0,
            p1,
            p2,
            p3,
            rest_angle,
            compliance,
            lambda: 0.0,
        }
    }
    /// Interior dihedral angle of the hinge `(p0, p1 | p2, p3)` together with
    /// its gradient, or `None` when the hinge is degenerate: the shared edge is
    /// shorter than `1e-12`, a wing lies on the hinge line (closer than
    /// `1e-9` edge lengths, so its triangle has no normal) or a coordinate is
    /// not finite.
    fn hinge_state(p0: &Vec3, p1: &Vec3, p2: &Vec3, p3: &Vec3) -> Option<HingeState> {
        let edge = p1 - p0;
        let edge_len = edge.norm();
        if !edge_len.is_finite() || edge_len <= BENDING_MIN_EDGE_LENGTH {
            return None;
        }
        let edge_hat = edge / edge_len;
        let normal_1 = edge.cross(&(p2 - p0));
        let normal_2 = (p3 - p0).cross(&edge);
        let norm_1_sq = normal_1.norm_squared();
        let norm_2_sq = normal_2.norm_squared();
        // |N| = |e| · (height of the wing above the hinge line).
        let min_norm_sq = (BENDING_MIN_WING_ASPECT * edge_len * edge_len).powi(2);
        if !norm_1_sq.is_finite()
            || !norm_2_sq.is_finite()
            || norm_1_sq <= min_norm_sq
            || norm_2_sq <= min_norm_sq
        {
            return None;
        }
        let fold = normal_2
            .cross(&normal_1)
            .dot(&edge_hat)
            .atan2(normal_2.dot(&normal_1));
        let weighted_1 = normal_1 / norm_1_sq;
        let weighted_2 = normal_2 / norm_2_sq;
        let grad_2 = weighted_1 * (-edge_len);
        let grad_3 = weighted_2 * (-edge_len);
        let grad_0 =
            weighted_1 * (-(p2 - p1).dot(&edge_hat)) + weighted_2 * (-(p3 - p1).dot(&edge_hat));
        let grad_1 = -(grad_0 + grad_2 + grad_3);
        Some(HingeState {
            angle: PI - fold,
            gradients: [grad_0, grad_1, grad_2, grad_3],
        })
    }
    /// Current interior dihedral angle in `[0, 2π)` (`π` for a flat hinge).
    ///
    /// A degenerate hinge (shared edge shorter than `1e-12`, a wing on the
    /// hinge line, or a non-finite coordinate) has no angle; the rest angle is
    /// returned so that the constraint counts as satisfied.
    pub fn current_angle(&self, positions: &[Vec3]) -> f64 {
        Self::hinge_state(
            &positions[self.p0],
            &positions[self.p1],
            &positions[self.p2],
            &positions[self.p3],
        )
        .map_or(self.rest_angle, |state| state.angle)
    }
    /// Constraint value `C = wrap(θ − θ_rest)` in `(−π, π]`.
    pub fn evaluate(&self, positions: &[Vec3]) -> f64 {
        wrap_angle(self.current_angle(positions) - self.rest_angle)
    }
    /// Gradient of the interior dihedral angle with respect to the positions
    /// of `[p0, p1, p2, p3]` (rad/m).
    ///
    /// The four gradients sum to zero (translation invariance) and carry no
    /// net torque (rotation invariance).  `None` for a degenerate hinge, where
    /// the angle is not defined.
    pub fn angle_gradients(&self, positions: &[Vec3]) -> Option<[Vec3; 4]> {
        Self::hinge_state(
            &positions[self.p0],
            &positions[self.p1],
            &positions[self.p2],
            &positions[self.p3],
        )
        .map(|state| state.gradients)
    }
    /// XPBD solve for the bending constraint.
    ///
    /// ```text
    /// α̃  = α / dt²
    /// Δλ = (−C − α̃ λ) / (Σ_i w_i |∇θ_i|² + α̃)
    /// Δx_i = w_i ∇θ_i Δλ                       i = p0, p1, p2, p3
    /// ```
    ///
    /// with the exact angle gradient on all four particles, so the correction
    /// conserves linear and angular momentum and restores the hinge towards
    /// its rest angle whichever way it is folded.  A degenerate hinge or one
    /// whose particles are all pinned is left alone.
    pub fn solve(&mut self, positions: &mut [Vec3], inv_masses: &[f64], dt: f64) {
        let indices = [self.p0, self.p1, self.p2, self.p3];
        let Some(state) = Self::hinge_state(
            &positions[indices[0]],
            &positions[indices[1]],
            &positions[indices[2]],
            &positions[indices[3]],
        ) else {
            return;
        };
        let violation = wrap_angle(state.angle - self.rest_angle);
        let weights = indices.map(|index| inv_masses[index]);
        let weighted_gradient_norm: f64 = state
            .gradients
            .iter()
            .zip(weights)
            .map(|(gradient, weight)| weight * gradient.norm_squared())
            .sum();
        if !weighted_gradient_norm.is_finite() || weighted_gradient_norm <= 0.0 {
            return;
        }
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda =
            (-violation - alpha_tilde * self.lambda) / (weighted_gradient_norm + alpha_tilde);
        if !delta_lambda.is_finite() {
            return;
        }
        self.lambda += delta_lambda;
        for ((&index, weight), gradient) in indices.iter().zip(weights).zip(state.gradients) {
            positions[index] += gradient * (weight * delta_lambda);
        }
    }
    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
/// XPBD distance constraint between two particles.
///
/// Keeps the distance between particle `particle_a` and `particle_b` close to
/// `rest_length`.  `compliance` α (m²/N) controls softness; 0 = rigid.
pub struct XpbdDistanceConstraint {
    /// Index of particle A.
    pub particle_a: usize,
    /// Index of particle B.
    pub particle_b: usize,
    /// Rest length.
    pub rest_length: f64,
    /// Compliance α = 1/stiffness (m²/N). 0 = rigid.
    pub compliance: f64,
    /// Damping coefficient.
    pub damping: f64,
    /// Lagrange multiplier accumulated during the current sub-step.
    pub lambda: f64,
}
impl XpbdDistanceConstraint {
    /// Create a new distance constraint.
    pub fn new(a: usize, b: usize, rest_length: f64, compliance: f64) -> Self {
        Self {
            particle_a: a,
            particle_b: b,
            rest_length,
            compliance,
            damping: 0.0,
            lambda: 0.0,
        }
    }
    /// Compute constraint value C = |r_b − r_a| − rest_length.
    pub fn evaluate(&self, positions: &[Vec3]) -> f64 {
        let diff = positions[self.particle_b] - positions[self.particle_a];
        diff.norm() - self.rest_length
    }
    /// XPBD solve: compute Δλ and apply position corrections.
    ///
    /// α̃ = α / dt²
    /// Δλ = (−C − α̃ · λ) / (w_a + w_b + α̃)
    /// Δx_a = −w_a · Δλ · ∇C_a   where ∇C_a = (r_a − r_b) / |r_a − r_b|
    /// Δx_b = +w_b · Δλ · ∇C_b
    pub fn solve(&mut self, positions: &mut [Vec3], inv_masses: &[f64], dt: f64) {
        let w_a = inv_masses[self.particle_a];
        let w_b = inv_masses[self.particle_b];
        let w_sum = w_a + w_b;
        if w_sum < 1e-30 {
            return;
        }
        let diff = positions[self.particle_b] - positions[self.particle_a];
        let dist = diff.norm();
        if dist < 1e-12 {
            return;
        }
        let c = dist - self.rest_length;
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha_tilde * self.lambda) / (w_sum + alpha_tilde);
        self.lambda += delta_lambda;
        let n = diff / dist;
        positions[self.particle_a] -= n * (w_a * delta_lambda);
        positions[self.particle_b] += n * (w_b * delta_lambda);
    }
    /// Reset the Lagrange multiplier (call at the start of each sub-step).
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
/// XPBD volume-conservation constraint for a tetrahedron.
pub struct XpbdVolumeConstraint {
    /// Indices of the four corner particles.
    pub particles: [usize; 4],
    /// Rest volume (positive).
    pub rest_volume: f64,
    /// Compliance α (m⁵/N).
    pub compliance: f64,
    /// Lagrange multiplier accumulated during the current sub-step.
    pub lambda: f64,
}
impl XpbdVolumeConstraint {
    /// Create a new volume constraint.
    pub fn new(particles: [usize; 4], rest_volume: f64, compliance: f64) -> Self {
        Self {
            particles,
            rest_volume,
            compliance,
            lambda: 0.0,
        }
    }
    /// Signed tetrahedral volume via scalar triple product / 6.
    pub fn current_volume(&self, positions: &[Vec3]) -> f64 {
        let [i0, i1, i2, i3] = self.particles;
        let a = positions[i1] - positions[i0];
        let b = positions[i2] - positions[i0];
        let c = positions[i3] - positions[i0];
        a.dot(&b.cross(&c)) / 6.0
    }
    /// Constraint value C = V − V_rest.
    pub fn evaluate(&self, positions: &[Vec3]) -> f64 {
        self.current_volume(positions) - self.rest_volume
    }
    /// XPBD solve for the volume constraint.
    pub fn solve(&mut self, positions: &mut [Vec3], inv_masses: &[f64], dt: f64) {
        let [i0, i1, i2, i3] = self.particles;
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let p3 = positions[i3];
        let vol = {
            let a = p1 - p0;
            let b = p2 - p0;
            let c = p3 - p0;
            a.dot(&b.cross(&c)) / 6.0
        };
        let c = vol - self.rest_volume;
        if c.abs() < 1e-14 {
            return;
        }
        let grad0 = (p1 - p2).cross(&(p3 - p2)) / 6.0;
        let grad1 = (p2 - p0).cross(&(p3 - p0)) / 6.0;
        let grad2 = (p0 - p1).cross(&(p3 - p1)) / 6.0;
        let grad3 = (p1 - p0).cross(&(p2 - p0)) / 6.0;
        let grads = [grad0, grad1, grad2, grad3];
        let idxs = [i0, i1, i2, i3];
        let mut w_sum = 0.0;
        for k in 0..4 {
            w_sum += inv_masses[idxs[k]] * grads[k].norm_squared();
        }
        if w_sum < 1e-14 {
            return;
        }
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-c - alpha_tilde * self.lambda) / (w_sum + alpha_tilde);
        self.lambda += delta_lambda;
        for k in 0..4 {
            positions[idxs[k]] += grads[k] * (delta_lambda * inv_masses[idxs[k]]);
        }
    }
    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}
/// XPBD damping constraint that damps velocity between two particles.
///
/// Implements Rayleigh damping as a constraint: damps the relative velocity
/// along the constraint gradient.
pub struct XpbdDampingConstraint {
    /// Index of particle A.
    pub particle_a: usize,
    /// Index of particle B.
    pub particle_b: usize,
    /// Damping coefficient beta (s/m). Higher = more damping.
    pub damping_beta: f64,
}
impl XpbdDampingConstraint {
    /// Create a new damping constraint.
    pub fn new(a: usize, b: usize, damping_beta: f64) -> Self {
        Self {
            particle_a: a,
            particle_b: b,
            damping_beta,
        }
    }
    /// Apply velocity damping between two particles.
    ///
    /// Reduces relative velocity along the line connecting the particles.
    /// `prev_positions` are positions at the start of the substep.
    pub fn apply(
        &self,
        positions: &mut [Vec3],
        prev_positions: &[Vec3],
        inv_masses: &[f64],
        dt: f64,
    ) {
        let w_a = inv_masses[self.particle_a];
        let w_b = inv_masses[self.particle_b];
        let w_sum = w_a + w_b;
        if w_sum < 1e-30 {
            return;
        }
        let diff = positions[self.particle_b] - positions[self.particle_a];
        let dist = diff.norm();
        if dist < 1e-12 {
            return;
        }
        let n = diff / dist;
        let va = (positions[self.particle_a] - prev_positions[self.particle_a]) / dt;
        let vb = (positions[self.particle_b] - prev_positions[self.particle_b]) / dt;
        let rel_vel = vb - va;
        let v_normal = rel_vel.dot(&n);
        let beta_tilde = self.damping_beta * dt;
        let delta_lambda = -v_normal * beta_tilde / (w_sum + beta_tilde);
        positions[self.particle_a] -= n * (w_a * delta_lambda);
        positions[self.particle_b] += n * (w_b * delta_lambda);
    }
}
/// Statistics collected during XPBD solving.
#[derive(Debug, Clone, Default)]
pub struct XpbdSolverStats {
    /// Total number of constraint projections performed, strain-limiter
    /// triangle projections included.
    pub total_projections: usize,
    /// Maximum constraint residual after the final iteration.
    pub max_residual: f64,
    /// Average constraint residual after the final iteration.
    pub avg_residual: f64,
    /// Total kinetic energy after the step.
    pub kinetic_energy: f64,
    /// Number of substeps performed.
    pub substeps: usize,
    /// Number of iterations per substep.
    pub iterations_per_substep: usize,
}
/// XPBD shape-matching constraint.
///
/// Matches a cluster of particles to a rotated version of a rest
/// configuration.  Each step, the current cluster centre-of-mass and
/// best-fit rotation are computed, then each particle is pulled toward its
/// rotated rest position with a given `stiffness` in \[0, 1\].
///
/// This is a position-level constraint, not a standard XPBD Lagrange multiplier
/// constraint.  It applies a direct position correction α per iteration.
pub struct XpbdShapeMatchingConstraint {
    /// Particle indices in this cluster.
    pub particles: Vec<usize>,
    /// Rest positions relative to the cluster centre-of-mass (in rest frame).
    pub rest_offsets: Vec<Vec3>,
    /// Mass-weighted rest centre-of-mass.
    pub rest_com: Vec3,
    /// Stiffness (0 = no correction, 1 = full correction each iteration).
    pub stiffness: f64,
}
impl XpbdShapeMatchingConstraint {
    /// Build a shape-matching constraint from a list of particle indices,
    /// their current positions (used as rest config), and masses.
    pub fn new(
        particle_indices: Vec<usize>,
        positions: &[Vec3],
        masses: &[f64],
        stiffness: f64,
    ) -> Self {
        let n = particle_indices.len();
        assert!(
            !particle_indices.is_empty(),
            "At least one particle required"
        );
        let mut total_mass = 0.0_f64;
        let mut rest_com = Vec3::zeros();
        for &i in &particle_indices {
            let m = masses[i].max(0.0);
            rest_com += positions[i] * m;
            total_mass += m;
        }
        if total_mass > 1e-30 {
            rest_com /= total_mass;
        }
        let rest_offsets: Vec<Vec3> = (0..n)
            .map(|k| positions[particle_indices[k]] - rest_com)
            .collect();
        Self {
            particles: particle_indices,
            rest_offsets,
            rest_com,
            stiffness: stiffness.clamp(0.0, 1.0),
        }
    }
    /// Compute the best-fit rotation matrix (3×3, row-major) between
    /// current deformed offsets and rest offsets via the polar decomposition
    /// of the deformation gradient A = Σ m_i * q_i * r_i^T.
    ///
    /// `q_i` = current offset, `r_i` = rest offset.
    ///
    /// Uses iterative polar decomposition: R_{n+1} = 0.5 * (R_n + (R_n^{-T})),
    /// starting from the normalised columns of A.
    fn best_fit_rotation(q: &[Vec3], r: &[Vec3], masses: &[f64]) -> [[f64; 3]; 3] {
        let mut a = [[0.0_f64; 3]; 3];
        for k in 0..q.len().min(r.len()).min(masses.len()) {
            let m = masses[k];
            for row in 0..3 {
                for col in 0..3 {
                    a[row][col] += m * q[k][row] * r[k][col];
                }
            }
        }
        polar_decompose_3x3(a)
    }
    /// Project the shape-matching constraint.
    ///
    /// Computes goal positions (current COM + R * rest_offset_i) and pulls
    /// each particle toward its goal with weight `stiffness`.
    pub fn solve(&self, positions: &mut [Vec3], inv_masses: &[f64]) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        let mut total_w = 0.0_f64;
        let mut cur_com = Vec3::zeros();
        for &i in &self.particles {
            let w = inv_masses[i];
            if w > 0.0 {
                cur_com += positions[i] * (1.0 / w);
                total_w += 1.0 / w;
            }
        }
        if total_w < 1e-30 {
            return;
        }
        cur_com /= total_w;
        let q: Vec<Vec3> = self
            .particles
            .iter()
            .map(|&i| positions[i] - cur_com)
            .collect();
        let masses: Vec<f64> = self
            .particles
            .iter()
            .map(|&i| {
                if inv_masses[i] > 0.0 {
                    1.0 / inv_masses[i]
                } else {
                    0.0
                }
            })
            .collect();
        let rot = Self::best_fit_rotation(&q, &self.rest_offsets, &masses);
        for (k, &i) in self.particles.iter().enumerate() {
            if inv_masses[i] == 0.0 {
                continue;
            }
            let r = &self.rest_offsets[k];
            let goal = Vec3::new(
                cur_com.x + rot[0][0] * r.x + rot[0][1] * r.y + rot[0][2] * r.z,
                cur_com.y + rot[1][0] * r.x + rot[1][1] * r.y + rot[1][2] * r.z,
                cur_com.z + rot[2][0] * r.x + rot[2][1] * r.y + rot[2][2] * r.z,
            );
            positions[i] += (goal - positions[i]) * self.stiffness;
        }
    }
}
/// XPBD substep configuration for adaptive or fixed-substep simulation.
///
/// Provides utilities to compute the optimal number of substeps given a
/// desired constraint residual tolerance, or to decide whether a step needs
/// refinement based on the kinetic energy.
#[derive(Debug, Clone, Copy)]
pub struct SubstepConfig {
    /// Minimum number of substeps per frame.
    pub min_substeps: usize,
    /// Maximum number of substeps per frame.
    pub max_substeps: usize,
    /// Desired maximum constraint residual per substep.
    pub target_residual: f64,
    /// Number of constraint-projection iterations per substep.
    pub iterations: usize,
}
impl SubstepConfig {
    /// Create a default substep configuration.
    pub fn default_config() -> Self {
        Self {
            min_substeps: 1,
            max_substeps: 32,
            target_residual: 1e-4,
            iterations: 5,
        }
    }
    /// Compute a recommended number of substeps for a given `dt` and
    /// expected maximum velocity `v_max` (m/s).
    ///
    /// The heuristic ensures that no particle moves more than
    /// `min_radius` per substep.
    pub fn substeps_for_velocity(&self, dt: f64, v_max: f64, min_radius: f64) -> usize {
        if v_max < 1e-12 || min_radius < 1e-12 {
            return self.min_substeps;
        }
        let n = (v_max * dt / min_radius).ceil() as usize;
        n.clamp(self.min_substeps, self.max_substeps)
    }
    /// Determine if residual is acceptable.
    pub fn residual_ok(&self, residual: f64) -> bool {
        residual <= self.target_residual
    }
    /// Effective substep size for `n_substeps` within frame time `dt`.
    pub fn substep_dt(&self, dt: f64, n_substeps: usize) -> f64 {
        dt / n_substeps.max(1) as f64
    }
    /// Total number of constraint projections for a frame.
    pub fn total_projections(&self, n_substeps: usize, n_constraints: usize) -> usize {
        n_substeps * self.iterations * n_constraints
    }
}
/// Self-collision pair detected between two particles.
#[derive(Debug, Clone, Copy)]
pub struct SelfCollisionPair {
    /// Index of first particle.
    pub i: usize,
    /// Index of second particle.
    pub j: usize,
    /// Sum of radii (contact distance threshold).
    pub contact_dist: f64,
}
/// XPBD self-collision handler.
///
/// Detects and resolves self-collisions between particles of the same body
/// using a position-based approach: if two particles are within `contact_dist`
/// of each other, they are pushed apart symmetrically.
#[derive(Debug, Clone)]
pub struct XpbdSelfCollision {
    /// Per-particle radius (m).
    pub radii: Vec<f64>,
    /// Collision compliance (lower = stiffer contact).
    pub compliance: f64,
    /// Minimum distance between particles before collision is detected.
    ///
    /// Defaults to the sum of the two particle radii.
    pub thickness: f64,
}
impl XpbdSelfCollision {
    /// Create a self-collision handler with uniform particle radius.
    pub fn new_uniform(n_particles: usize, radius: f64, compliance: f64) -> Self {
        Self {
            radii: vec![radius; n_particles],
            compliance,
            thickness: 0.0,
        }
    }
    /// Create a self-collision handler with per-particle radii.
    pub fn new(radii: Vec<f64>, compliance: f64) -> Self {
        Self {
            radii,
            compliance,
            thickness: 0.0,
        }
    }
    /// Detect all self-collision pairs.
    ///
    /// O(n²) brute-force; suitable for small bodies.  Returns all pairs
    /// where the distance between particles < sum of their radii + `thickness`.
    pub fn detect_pairs(&self, positions: &[Vec3]) -> Vec<SelfCollisionPair> {
        let n = positions.len().min(self.radii.len());
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let cd = self.radii[i] + self.radii[j] + self.thickness;
                let dist = (positions[j] - positions[i]).norm();
                if dist < cd {
                    pairs.push(SelfCollisionPair {
                        i,
                        j,
                        contact_dist: cd,
                    });
                }
            }
        }
        pairs
    }
    /// Resolve a single self-collision pair using position-based projection.
    ///
    /// Moves both particles apart so their distance equals `contact_dist`.
    /// `dt` is the current substep time.
    pub fn resolve_pair(
        &self,
        pair: &SelfCollisionPair,
        positions: &mut [Vec3],
        inv_masses: &[f64],
        dt: f64,
    ) {
        let (i, j) = (pair.i, pair.j);
        let diff = positions[j] - positions[i];
        let dist = diff.norm();
        if dist < 1e-12 {
            let sep = pair.contact_dist;
            positions[j] += Vec3::new(sep * 0.5, 0.0, 0.0);
            positions[i] -= Vec3::new(sep * 0.5, 0.0, 0.0);
            return;
        }
        let c = dist - pair.contact_dist;
        if c >= 0.0 {
            return;
        }
        let wi = inv_masses[i];
        let wj = inv_masses[j];
        let w_sum = wi + wj;
        if w_sum < 1e-30 {
            return;
        }
        let n_hat = diff / dist;
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lambda = (-c) / (w_sum + alpha_tilde);
        positions[i] -= n_hat * (wi * delta_lambda);
        positions[j] += n_hat * (wj * delta_lambda);
    }
    /// Detect and resolve all self-collision pairs in one pass.
    ///
    /// Returns the number of collisions resolved.
    pub fn resolve(&self, positions: &mut [Vec3], inv_masses: &[f64], dt: f64) -> usize {
        let pairs = self.detect_pairs(positions);
        let count = pairs.len();
        for pair in &pairs {
            self.resolve_pair(pair, positions, inv_masses, dt);
        }
        count
    }
    /// Velocity-based collision response: subtract the relative velocity
    /// component that drives the particles together.
    ///
    /// Call after position correction to prevent jitter.
    pub fn velocity_response(
        &self,
        pair: &SelfCollisionPair,
        positions: &[Vec3],
        velocities: &mut [Vec3],
        inv_masses: &[f64],
        restitution: f64,
    ) {
        let (i, j) = (pair.i, pair.j);
        let diff = positions[j] - positions[i];
        let dist = diff.norm();
        if dist < 1e-12 {
            return;
        }
        let n_hat = diff / dist;
        let v_rel = velocities[j] - velocities[i];
        let vn = v_rel.dot(&n_hat);
        if vn > 0.0 {
            return;
        }
        let wi = inv_masses[i];
        let wj = inv_masses[j];
        let w_sum = wi + wj;
        if w_sum < 1e-30 {
            return;
        }
        let j_imp = -(1.0 + restitution) * vn / w_sum;
        velocities[i] -= n_hat * (wi * j_imp);
        velocities[j] += n_hat * (wj * j_imp);
    }
}
