// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Extended PBD/XPBD constraint types for soft-body simulation.
//!
//! Implements advanced constraint types including:
//! - Isometric bending (discrete shells, Bergou)
//! - CST strain (constant strain triangle, Green-Lagrange)
//! - Neo-Hookean continuum constraint (tet elements)
//! - Volume conservation (closed mesh)
//! - Cable (inextensible, clamped)
//! - Rigid body coupling
//! - Sewing (cloth seam)
//! - Pressure (inflatable balloon)
//! - Anchor (world-space pin with compliance)
//! - Constraint solver (Gauss-Seidel PBD)

use oxiphysics_core::math::{Real, Vec3};

use crate::constraint::SoftConstraint;
use crate::particle::SoftParticle;

// ─────────────────────────────────────────────────────────────────────────────
// Helper math (no nalgebra for 3-element arrays)
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two `Vec3` values.
#[inline]
fn dot(a: Vec3, b: Vec3) -> Real {
    a.dot(&b)
}

/// Cross product of two `Vec3` values.
#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(&b)
}

/// Outer product A = a ⊗ b as a flat \[9\] row-major array.
#[cfg(test)]
#[inline]
fn outer(a: Vec3, b: Vec3) -> [Real; 9] {
    [
        a.x * b.x,
        a.x * b.y,
        a.x * b.z,
        a.y * b.x,
        a.y * b.y,
        a.y * b.z,
        a.z * b.x,
        a.z * b.y,
        a.z * b.z,
    ]
}

/// Matrix-vector product: 3×3 row-major `m` times `v`.
#[inline]
fn mat3_mul_vec3(m: &[Real; 9], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// IsometricBendingConstraint (Bergou 2006 quadratic bending)
// ─────────────────────────────────────────────────────────────────────────────

/// Relative sine of the angle between two vectors below which they are treated
/// as (anti)parallel, i.e. the angle between them is degenerate.
const PARALLEL_SINE: Real = 1.0e-12;

/// Cotangent of the angle between `a` and `b`, or `None` when either vector is
/// zero or the two are (anti)parallel.
fn cotangent(a: Vec3, b: Vec3) -> Option<Real> {
    let sine_scaled = cross(a, b).norm();
    if sine_scaled > PARALLEL_SINE * a.norm() * b.norm() {
        Some(dot(a, b) / sine_scaled)
    } else {
        None
    }
}

/// Isometric bending constraint for discrete shells (Bergou et al. 2006).
///
/// Uses the quadratic bending energy `E = 1/2 sum_ij Q_ij x_i . x_j` over a
/// hinge (shared edge between two triangles).  Indices: p0, p1 are the shared
/// edge; p2, p3 are the opposite vertices in each triangle.
///
/// With `e0 = p1 - p0`, `e1 = p2 - p0`, `e2 = p3 - p0`, `e3 = p2 - p1`,
/// `e4 = p3 - p1`, the cotangents `c01 = cot(e0, e1)`, `c02 = cot(e0, e2)`,
/// `c03 = cot(-e0, e3)`, `c04 = cot(-e0, e4)` and the triangle areas `A0`, `A1`,
/// the matrix is `Q = 3 / (A0 + A1) K K^T` with
/// `K = [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04]`.  `K` sums to zero, so
/// `E` does not change when the four particles are translated, and `K . x`
/// vanishes for every planar hinge whose wings lie on opposite sides of the
/// shared edge, so `E` is zero for a flat sheet and grows with the fold.  `E`
/// is also invariant under rotations.
///
/// Because `Q` has rank one, `E = 3 / (2 (A0 + A1)) |y|^2` with the bending
/// vector `y = sum_i K_i x_i`, and the gradient is `dE/dx_i = 3 K_i y / (A0 + A1)`.
/// The constraint evaluates `y` from the edge vectors `x_i - x_0`, so neither the
/// energy nor the gradient depends on the absolute position of the hinge, and
/// both come from the same `y`: for a flat hinge, whose `y` is zero up to
/// round-off, the projection then moves the particles by a round-off-sized
/// amount instead of dividing one rounding error by another.
///
/// The cotangent weights only depend on the shapes of the two triangles, so the
/// constraint measures the bending away from the *flat* state of the hinge
/// that was passed to [`IsometricBendingConstraint::new`]; a hinge that is
/// folded at construction time is driven towards flat, not towards its initial
/// fold.  A hinge with a degenerate triangle has no bending resistance.
#[derive(Debug, Clone)]
pub struct IsometricBendingConstraint {
    /// Particle indices \[p0, p1, p2, p3\].
    pub indices: [usize; 4],
    /// Cotangent weights `K` of the hinge (zero for a degenerate hinge).
    weights: [Real; 4],
    /// Prefactor `3 / (A0 + A1)` of the quadratic form `Q = scale K K^T`.
    scale: Real,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl IsometricBendingConstraint {
    /// Build the constraint from particle positions.
    ///
    /// `indices` = \[shared_edge_0, shared_edge_1, wing_0, wing_1\].
    pub fn new(indices: [usize; 4], particles: &[SoftParticle], compliance: Real) -> Self {
        let [i0, i1, i2, i3] = indices;
        let (weights, scale) = Self::hinge_weights(
            particles[i0].position,
            particles[i1].position,
            particles[i2].position,
            particles[i3].position,
        )
        .unwrap_or(([0.0; 4], 0.0));
        Self {
            indices,
            weights,
            scale,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }

    /// Cotangent weights `K` and the prefactor `3 / (A0 + A1)` of the hinge
    /// `(p0, p1 | p2, p3)`, or `None` when one of its triangles is degenerate.
    ///
    /// `K` sums to zero and annihilates the positions of every planar hinge
    /// whose wings lie on opposite sides of the shared edge.
    fn hinge_weights(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Option<([Real; 4], Real)> {
        let e0 = p1 - p0; // shared edge
        let e1 = p2 - p0;
        let e2 = p3 - p0;
        let e3 = p2 - p1;
        let e4 = p3 - p1;
        let c01 = cotangent(e0, e1)?;
        let c02 = cotangent(e0, e2)?;
        let c03 = cotangent(-e0, e3)?;
        let c04 = cotangent(-e0, e4)?;
        let area_sum = 0.5 * (cross(e0, e1).norm() + cross(e0, e2).norm());
        let k = [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04];
        Some((k, 3.0 / area_sum))
    }

    /// The 4×4 Bergou matrix `Q = 3 / (A0 + A1) K K^T` (row-major) of the hinge
    /// the constraint was built from; all zeros for a degenerate hinge.
    pub fn q_matrix(&self) -> [Real; 16] {
        let mut q = [0.0_f64; 16];
        for (row, k_row) in self.weights.iter().enumerate() {
            for (col, k_col) in self.weights.iter().enumerate() {
                q[row * 4 + col] = self.scale * k_row * k_col;
            }
        }
        q
    }

    /// Bending vector `y = sum_i K_i x_i`, evaluated from the edge vectors
    /// `x_i - x_0` (the weights sum to zero, so this is the same vector without
    /// the round-off that the absolute positions would add).
    fn bending_vector(&self, particles: &[SoftParticle]) -> Vec3 {
        let [i0, i1, i2, i3] = self.indices;
        let origin = particles[i0].position;
        (particles[i1].position - origin) * self.weights[1]
            + (particles[i2].position - origin) * self.weights[2]
            + (particles[i3].position - origin) * self.weights[3]
    }

    /// Quadratic bending energy `E = 1/2 sum_{i,j} Q_ij x_i . x_j`
    /// (`= scale/2 |y|^2`), zero for a flat hinge.
    pub fn bending_energy(&self, particles: &[SoftParticle]) -> Real {
        0.5 * self.scale * self.bending_vector(particles).norm_squared()
    }
}

impl SoftConstraint for IsometricBendingConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let y = self.bending_vector(particles);
        let c = 0.5 * self.scale * y.norm_squared();
        // Gradient of E with respect to each particle: grad_i E = sum_j Q_ij x_j
        // = scale K_i y.
        let grads = self.weights.map(|k| y * (self.scale * k));
        let w = self.indices.map(|i| particles[i].inverse_mass);
        // Weighted squared gradient norm
        let grad_w: Real = grads
            .iter()
            .zip(w.iter())
            .map(|(g, w_k)| w_k * g.norm_squared())
            .sum();

        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = grad_w + alpha;
        if denom.abs() < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;

        for ((&i, grad), w_k) in self.indices.iter().zip(grads.iter()).zip(w.iter()) {
            particles[i].position += grad * (w_k * d_lambda);
        }
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSTStrainConstraint (Constant Strain Triangle, Green-Lagrange)
// ─────────────────────────────────────────────────────────────────────────────

/// Constant strain triangle (CST) strain constraint.
///
/// The deformation gradient `F = [e1 e2] Dm⁻¹` maps the rest triangle, expressed
/// in its own 2D tangent frame, onto the current (3D) triangle.  The constraint
/// `C = tr(F^T F) - 2` is the trace of the Green-Lagrange strain `1/2 (F^T F - I)`
/// (times two): it is zero for every rigid placement of the rest triangle, so a
/// triangle that is only translated or rotated is left untouched, and positive
/// when the triangle is stretched.  Because only the trace is constrained, a
/// stretch along one direction that is balanced by a compression along the
/// other one (principal stretches with `s1² + s2² = 2`) also satisfies it.
#[derive(Debug, Clone)]
pub struct CSTStrainConstraint {
    /// Triangle vertex indices \[i0, i1, i2\].
    pub indices: [usize; 3],
    /// Inverse of rest-state shape matrix (2×2, col-major: \[m00,m10,m01,m11\]).
    dm_inv: [Real; 4],
    /// Rest area \[m²\].
    pub rest_area: Real,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl CSTStrainConstraint {
    /// Build the CST constraint from rest positions.
    pub fn new(indices: [usize; 3], particles: &[SoftParticle], compliance: Real) -> Self {
        let [i0, i1, i2] = indices;
        let x0 = particles[i0].position;
        let x1 = particles[i1].position;
        let x2 = particles[i2].position;
        let (dm_inv, rest_area) = Self::build_dm_inv(x0, x1, x2);
        Self {
            indices,
            dm_inv,
            rest_area,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }

    /// Build the local frame inverse shape matrix and rest area.
    fn build_dm_inv(p0: Vec3, p1: Vec3, p2: Vec3) -> ([Real; 4], Real) {
        // Local 2D coordinates using edge vectors
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = cross(e1, e2);
        let area = n.norm() * 0.5;
        // u1, u2 = local 2D projections
        let u1 = e1.norm();
        let u2 = dot(e2, e1.normalize());
        let v2 = (e2 - e1.normalize() * u2).norm();
        // Dm = [[u1, u2], [0, v2]]
        let det = u1 * v2;
        let dm_inv = if det.abs() > 1e-20 {
            [1.0 / u1, 0.0, -u2 / (u1 * v2), 1.0 / v2]
        } else {
            [0.0; 4]
        };
        (dm_inv, area)
    }

    /// Columns `(f0, f1)` of the 3×2 deformation gradient `F = [e1 e2] Dm⁻¹`.
    ///
    /// `dm_inv` holds `Dm⁻¹` column-major as `[m00, m10, m01, m11]`, so column
    /// `j` of `F` is `e1 * Dm⁻¹[0][j] + e2 * Dm⁻¹[1][j]`.
    fn deformation_gradient(&self, particles: &[SoftParticle]) -> (Vec3, Vec3) {
        let [i0, i1, i2] = self.indices;
        let x0 = particles[i0].position;
        let e1 = particles[i1].position - x0;
        let e2 = particles[i2].position - x0;
        let [m00, m10, m01, m11] = self.dm_inv;
        (e1 * m00 + e2 * m10, e1 * m01 + e2 * m11)
    }

    /// Gradient of `I1 = |f0|² + |f1|²` with respect to the three vertex
    /// positions `[x0, x1, x2]`.
    ///
    /// `x1` and `x2` enter `F` through the edges `e1 = x1 - x0` and
    /// `e2 = x2 - x0`, so `dI1/dx1 = 2 (f0 m00 + f1 m01)`,
    /// `dI1/dx2 = 2 (f0 m10 + f1 m11)` and `dI1/dx0 = -(dI1/dx1 + dI1/dx2)`.
    fn strain_gradient(&self, particles: &[SoftParticle]) -> [Vec3; 3] {
        let (f0, f1) = self.deformation_gradient(particles);
        let [m00, m10, m01, m11] = self.dm_inv;
        let g1 = (f0 * m00 + f1 * m01) * 2.0;
        let g2 = (f0 * m10 + f1 * m11) * 2.0;
        [-(g1 + g2), g1, g2]
    }

    /// Strain invariants `(I1, J)` of the deformation gradient: `I1 = tr(F^T F)`
    /// (2 at rest) and `J = sqrt(det(F^T F))`, the ratio of the current to the
    /// rest area (1 at rest).
    pub fn strain_invariants(&self, particles: &[SoftParticle]) -> (Real, Real) {
        let (f0, f1) = self.deformation_gradient(particles);
        let i1 = dot(f0, f0) + dot(f1, f1);
        let det = dot(f0, f0) * dot(f1, f1) - dot(f0, f1).powi(2);
        (i1, det.sqrt().max(0.0))
    }
}

impl SoftConstraint for CSTStrainConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let (i1_val, _area_ratio) = self.strain_invariants(particles);
        // Constraint: C = I1 - 2.0 (identity = 2 in 2D)
        let c = i1_val - 2.0;
        if c.abs() < 1e-8 {
            return;
        }

        let [i0, i1, i2] = self.indices;
        let w = [
            particles[i0].inverse_mass,
            particles[i1].inverse_mass,
            particles[i2].inverse_mass,
        ];

        let [g0, g1, g2] = self.strain_gradient(particles);

        let grad_w = w[0] * g0.norm_squared() + w[1] * g1.norm_squared() + w[2] * g2.norm_squared();
        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = grad_w + alpha;
        if denom.abs() < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;

        particles[i0].position += g0 * (w[0] * d_lambda);
        particles[i1].position += g1 * (w[1] * d_lambda);
        particles[i2].position += g2 * (w[2] * d_lambda);
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NeoHookeanConstraint (Continuum tet element)
// ─────────────────────────────────────────────────────────────────────────────

/// Largest Poisson ratio accepted by [`NeoHookeanConstraint::new`]: the first
/// Lamé parameter diverges as the ratio approaches the incompressible limit 1/2.
const MAX_POISSON_RATIO: Real = 0.4999;

/// Smallest Poisson ratio accepted by [`NeoHookeanConstraint::new`]: the shear
/// modulus diverges as the ratio approaches -1.
const MIN_POISSON_RATIO: Real = -0.99;

/// Stable Neo-Hookean continuum constraint for tetrahedral elements
/// (Smith et al. 2018; Macklin & Müller 2021).
///
/// The elastic energy density is
/// `Psi(F) = mu/2 (tr(F^T F) - 3) + K/2 (det F - gamma)^2` with `F = Ds Dm⁻¹`,
/// `K = lambda + mu` and `gamma = 1 + mu / K`.  It is split into two XPBD
/// constraints:
///
/// - the deviatoric constraint `C_D = sqrt(tr(F^T F))` with
///   `dC_D/dF = F / C_D` and compliance `1 / (mu V0)`;
/// - the hydrostatic constraint `C_H = det F - gamma` with
///   `dC_H/dF = cof(F) = [f2 x f3, f3 x f1, f1 x f2]` and compliance
///   `1 / (K V0) + compliance`.
///
/// The gradient with respect to vertex `k + 1` is `sum_j dC/dF[:, j] Dm⁻¹[k][j]`
/// (the columns of `dC/dF · Dm⁻ᵀ`) and the gradient of vertex 0 is minus the sum
/// of the other three.  `F = I` is stress free: the two constraint forces cancel
/// there, so the rest state (and every rigid motion of it) is a fixed point of
/// [`SoftConstraint::project`], which solves both constraints together as a
/// coupled 2 × 2 system.  The model stays well defined for inverted elements and
/// reproduces Hooke's law with the Lamé parameters `lame_mu` and `lame_lambda`
/// for small strains.
///
/// `K = lambda + mu` rather than `lambda` is what makes the small-strain limit of
/// `Psi` the linear elasticity of the Lamé parameters; with `K = lambda` the
/// bulk modulus would be `lambda - mu / 3`, which is negative (an unstable rest
/// state) for Poisson ratios below 1/8, and `gamma` would be undefined for
/// `lambda = 0`.
///
/// The projection itself is carried out by
/// [`crate::constraint::NeoHookeanConstraint`], which globalises the XPBD
/// Gauss-Newton step with a step-length control so that stiff elements (long
/// sub-steps, large `mu V0 dt²` relative to the vertex masses) do not overshoot.
#[derive(Debug, Clone)]
pub struct NeoHookeanConstraint {
    /// Tetrahedron vertex indices \[i0, i1, i2, i3\].
    pub indices: [usize; 4],
    /// Rest volume \[m³\].
    pub rest_volume: Real,
    /// Lame's first parameter λ.
    pub lame_lambda: Real,
    /// Lame's second parameter (shear modulus) µ.
    pub lame_mu: Real,
    /// Extra XPBD compliance \[1/J\] of the hydrostatic (volume) constraint, in
    /// series with its physical compliance `1 / (K V0)`.  Zero gives the pure
    /// material response; a positive value softens the volume response.
    pub compliance: Real,
    /// The element that holds the rest shape and the Lagrange multipliers and
    /// carries out the projection.  The public parameters above are copied into
    /// it whenever it is used.
    element: crate::constraint::NeoHookeanConstraint,
}

impl NeoHookeanConstraint {
    /// Build the Neo-Hookean tet constraint from the rest positions of the four
    /// particles.
    ///
    /// `young_modulus` in Pa, `poisson_ratio` dimensionless.  The Poisson ratio
    /// is limited to `[-0.99, 0.4999]` so that the Lamé parameters stay finite.
    /// `compliance` is an extra compliance of the volume constraint (see
    /// [`NeoHookeanConstraint::compliance`]).
    pub fn new(
        indices: [usize; 4],
        particles: &[SoftParticle],
        young_modulus: Real,
        poisson_ratio: Real,
        compliance: Real,
    ) -> Self {
        let nu = poisson_ratio.clamp(MIN_POISSON_RATIO, MAX_POISSON_RATIO);
        let lame_lambda = young_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let lame_mu = young_modulus / (2.0 * (1.0 + nu));
        let element = crate::constraint::NeoHookeanConstraint::from_particles(
            indices,
            particles,
            lame_mu,
            lame_lambda,
            compliance,
        );
        Self {
            indices,
            rest_volume: element.rest_volume,
            lame_lambda,
            lame_mu,
            compliance,
            element,
        }
    }

    /// Reset Lagrange multipliers.
    pub fn reset_lambda(&mut self) {
        self.element.reset_lambda();
    }

    /// The element with the public parameters of this constraint applied.
    fn current_element(&self) -> crate::constraint::NeoHookeanConstraint {
        let mut element = self.element.clone();
        element.indices = self.indices;
        element.mu = self.lame_mu;
        element.lambda_lame = self.lame_lambda;
        element.compliance = self.compliance;
        element.rest_volume = self.rest_volume;
        element
    }

    /// Value `gamma = 1 + mu / K` (`K = lambda + mu`) that `det(F)` is driven
    /// towards by the hydrostatic constraint.
    pub fn hydrostatic_target(&self) -> Real {
        self.current_element().hydrostatic_target()
    }

    /// Hydrostatic constraint value `C_H = det(F) - gamma`.
    ///
    /// It is `1 - gamma = -mu / K` in the rest state, where the deviatoric
    /// constraint balances it; use [`NeoHookeanConstraint::volume_strain`] for the
    /// plain volume change.
    pub fn hydrostatic_constraint(&self, particles: &[SoftParticle]) -> Real {
        let element = self.current_element();
        let f = element.deformation_gradient(particles);
        crate::constraint::NeoHookeanConstraint::compute_j(f) - element.hydrostatic_target()
    }

    /// Deviatoric constraint value `C_D = sqrt(tr(F^T F))` (`sqrt(3)` at rest).
    pub fn deviatoric_constraint(&self, particles: &[SoftParticle]) -> Real {
        let f = self.current_element().deformation_gradient(particles);
        crate::constraint::NeoHookeanConstraint::compute_i1(f).sqrt()
    }

    /// Volume strain `det(F) - 1`: the current volume relative to the rest
    /// volume, minus one (zero in the rest state).
    pub fn volume_strain(&self, particles: &[SoftParticle]) -> Real {
        let f = self.current_element().deformation_gradient(particles);
        crate::constraint::NeoHookeanConstraint::compute_j(f) - 1.0
    }
}

impl SoftConstraint for NeoHookeanConstraint {
    /// One coupled XPBD update of the deviatoric and hydrostatic constraints
    /// (see the type documentation).  Does nothing for a non-positive `dt_sub` or
    /// a degenerate rest shape.
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        self.element = self.current_element();
        self.element.project(particles, dt_sub);
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VolumeConservationConstraint (closed mesh)
// ─────────────────────────────────────────────────────────────────────────────

/// Global volume conservation constraint for a closed triangulated mesh.
///
/// Computes the signed volume via divergence theorem and projects to rest volume.
#[derive(Debug, Clone)]
pub struct VolumeConservationConstraint {
    /// Triangle faces (indices into particle array).
    pub triangles: Vec<[usize; 3]>,
    /// Rest volume \[m³\].
    pub rest_volume: Real,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl VolumeConservationConstraint {
    /// Build the constraint from a triangle soup and rest volume.
    pub fn new(triangles: Vec<[usize; 3]>, rest_volume: Real, compliance: Real) -> Self {
        Self {
            triangles,
            rest_volume,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }

    /// Compute signed mesh volume via divergence theorem.
    pub fn compute_volume(triangles: &[[usize; 3]], particles: &[SoftParticle]) -> Real {
        let mut vol = 0.0_f64;
        for &[i0, i1, i2] in triangles {
            let p0 = particles[i0].position;
            let p1 = particles[i1].position;
            let p2 = particles[i2].position;
            vol += dot(p0, cross(p1, p2));
        }
        vol / 6.0
    }
}

impl SoftConstraint for VolumeConservationConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let vol = Self::compute_volume(&self.triangles, particles);
        let c = vol - self.rest_volume;
        if c.abs() < 1e-10 {
            return;
        }

        // Gradient of volume w.r.t. each particle position
        let n_particles = particles.len();
        let mut grads = vec![Vec3::zeros(); n_particles];
        for &[i0, i1, i2] in &self.triangles {
            let p0 = particles[i0].position;
            let p1 = particles[i1].position;
            let p2 = particles[i2].position;
            grads[i0] += cross(p1, p2) / 6.0;
            grads[i1] += cross(p2, p0) / 6.0;
            grads[i2] += cross(p0, p1) / 6.0;
        }

        let grad_w: Real = (0..n_particles)
            .map(|i| particles[i].inverse_mass * grads[i].norm_squared())
            .sum();

        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = grad_w + alpha;
        if denom.abs() < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;

        for i in 0..n_particles {
            let w = particles[i].inverse_mass;
            particles[i].position += grads[i] * (w * d_lambda);
        }
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CableConstraint (inextensible cable)
// ─────────────────────────────────────────────────────────────────────────────

/// Inextensible cable constraint: project to rest length, clamp extensions only.
///
/// Unlike the standard distance constraint this only corrects stretch (not
/// compression), modelling an inextensible rope that can go slack.
#[derive(Debug, Clone)]
pub struct CableConstraint {
    /// Index of particle A.
    pub i: usize,
    /// Index of particle B.
    pub j: usize,
    /// Rest (maximum) cable length \[m\].
    pub rest_length: Real,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl CableConstraint {
    /// Create a cable constraint.
    pub fn new(i: usize, j: usize, rest_length: Real, compliance: Real) -> Self {
        Self {
            i,
            j,
            rest_length,
            compliance,
            lambda: 0.0,
        }
    }

    /// Build from current particle separation.
    pub fn from_particles(
        i: usize,
        j: usize,
        particles: &[SoftParticle],
        compliance: Real,
    ) -> Self {
        let rest = (particles[i].position - particles[j].position).norm();
        Self::new(i, j, rest, compliance)
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

impl SoftConstraint for CableConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let xi = particles[self.i].position;
        let xj = particles[self.j].position;
        let diff = xi - xj;
        let dist = diff.norm();
        if dist < 1e-12 || dist <= self.rest_length {
            // Slack: no correction needed
            return;
        }
        let n = diff / dist;
        let c = dist - self.rest_length;
        let wi = particles[self.i].inverse_mass;
        let wj = particles[self.j].inverse_mass;
        let grad_w = wi + wj;
        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = grad_w + alpha;
        if denom < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;
        particles[self.i].position += n * (wi * d_lambda);
        particles[self.j].position -= n * (wj * d_lambda);
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RigidBodyConstraint
// ─────────────────────────────────────────────────────────────────────────────

/// Couples a PBD particle to a rigid body pose (position + orientation).
///
/// Attaches the particle to a local-frame offset in the rigid body,
/// projecting the particle position back to the body surface.
#[derive(Debug, Clone)]
pub struct RigidBodyConstraint {
    /// Particle index.
    pub particle_idx: usize,
    /// Rigid body world position.
    pub body_position: Vec3,
    /// Rigid body orientation as rotation matrix (3×3, row-major).
    pub body_rotation: [Real; 9],
    /// Local-frame attachment offset.
    pub local_offset: Vec3,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl RigidBodyConstraint {
    /// Create a rigid body coupling constraint.
    pub fn new(
        particle_idx: usize,
        body_position: Vec3,
        body_rotation: [Real; 9],
        local_offset: Vec3,
        compliance: Real,
    ) -> Self {
        Self {
            particle_idx,
            body_position,
            body_rotation,
            local_offset,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }

    /// Compute world-space attachment point.
    pub fn world_attachment(&self) -> Vec3 {
        self.body_position + mat3_mul_vec3(&self.body_rotation, self.local_offset)
    }
}

impl SoftConstraint for RigidBodyConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let target = self.world_attachment();
        let pos = particles[self.particle_idx].position;
        let diff = pos - target;
        let dist = diff.norm();
        if dist < 1e-12 {
            return;
        }
        let c = dist;
        let n = diff / dist;
        let w = particles[self.particle_idx].inverse_mass;
        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = w + alpha;
        if denom < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;
        particles[self.particle_idx].position += n * (w * d_lambda);
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SewingConstraint (cloth seam)
// ─────────────────────────────────────────────────────────────────────────────

/// Stitches two particle pairs together (cloth seam / sewing constraint).
///
/// Drives pairs of particles to coincide, enabling cloth seam simulation.
/// Applies a progressive stiffening (compliance decreases each iteration).
#[derive(Debug, Clone)]
pub struct SewingConstraint {
    /// Particle pairs (a, b) to stitch together.
    pub pairs: Vec<[usize; 2]>,
    /// XPBD compliance per pair.
    pub compliance: Real,
    /// Accumulated Lagrange multipliers per pair.
    lambdas: Vec<Real>,
}

impl SewingConstraint {
    /// Create a sewing constraint from a list of particle index pairs.
    pub fn new(pairs: Vec<[usize; 2]>, compliance: Real) -> Self {
        let n = pairs.len();
        Self {
            pairs,
            compliance,
            lambdas: vec![0.0; n],
        }
    }

    /// Reset all Lagrange multipliers.
    pub fn reset_lambda(&mut self) {
        for l in &mut self.lambdas {
            *l = 0.0;
        }
    }
}

impl SoftConstraint for SewingConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let alpha = self.compliance / (dt_sub * dt_sub);
        for (k, &[ia, ib]) in self.pairs.iter().enumerate() {
            let xa = particles[ia].position;
            let xb = particles[ib].position;
            let diff = xa - xb;
            let dist = diff.norm();
            if dist < 1e-12 {
                continue;
            }
            let n = diff / dist;
            let c = dist;
            let wa = particles[ia].inverse_mass;
            let wb = particles[ib].inverse_mass;
            let denom = wa + wb + alpha;
            if denom < 1e-20 {
                continue;
            }
            let d_lambda = (-c - alpha * self.lambdas[k]) / denom;
            self.lambdas[k] += d_lambda;
            particles[ia].position += n * (wa * d_lambda);
            particles[ib].position -= n * (wb * d_lambda);
        }
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PressureConstraint (inflatable balloon)
// ─────────────────────────────────────────────────────────────────────────────

/// Pressure constraint for an inflatable closed mesh.
///
/// Maintains internal pressure P_0 * V_0 = P * V by enforcing volume
/// proportional to a target pressure ratio.
#[derive(Debug, Clone)]
pub struct PressureConstraint {
    /// Triangle faces (indices into particle array).
    pub triangles: Vec<[usize; 3]>,
    /// Reference volume \[m³\].
    pub rest_volume: Real,
    /// Pressure ratio P/P0 (1.0 = no overpressure, >1 = inflation).
    pub pressure_ratio: Real,
    /// XPBD compliance.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl PressureConstraint {
    /// Create a pressure constraint.
    pub fn new(
        triangles: Vec<[usize; 3]>,
        rest_volume: Real,
        pressure_ratio: Real,
        compliance: Real,
    ) -> Self {
        Self {
            triangles,
            rest_volume,
            pressure_ratio,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

impl SoftConstraint for PressureConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let vol = VolumeConservationConstraint::compute_volume(&self.triangles, particles);
        let target = self.rest_volume * self.pressure_ratio;
        let c = vol - target;
        if c.abs() < 1e-10 {
            return;
        }

        let n_particles = particles.len();
        let mut grads = vec![Vec3::zeros(); n_particles];
        for &[i0, i1, i2] in &self.triangles {
            let p0 = particles[i0].position;
            let p1 = particles[i1].position;
            let p2 = particles[i2].position;
            grads[i0] += cross(p1, p2) / 6.0;
            grads[i1] += cross(p2, p0) / 6.0;
            grads[i2] += cross(p0, p1) / 6.0;
        }

        let grad_w: Real = (0..n_particles)
            .map(|i| particles[i].inverse_mass * grads[i].norm_squared())
            .sum();

        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = grad_w + alpha;
        if denom < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;

        for i in 0..n_particles {
            let w = particles[i].inverse_mass;
            particles[i].position += grads[i] * (w * d_lambda);
        }
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnchorConstraint (pin to world-space point with compliance)
// ─────────────────────────────────────────────────────────────────────────────

/// Anchors a particle to a world-space point with a given compliance.
///
/// Equivalent to a soft spring to the anchor point.  Zero compliance
/// produces a hard pin; large compliance allows elastic stretching.
#[derive(Debug, Clone)]
pub struct AnchorConstraint {
    /// Particle index.
    pub particle_idx: usize,
    /// World-space anchor position.
    pub anchor: Vec3,
    /// XPBD compliance \[m/N\].
    pub compliance: Real,
    /// Accumulated Lagrange multiplier.
    lambda: Real,
}

impl AnchorConstraint {
    /// Create an anchor constraint.
    pub fn new(particle_idx: usize, anchor: Vec3, compliance: Real) -> Self {
        Self {
            particle_idx,
            anchor,
            compliance,
            lambda: 0.0,
        }
    }

    /// Reset the Lagrange multiplier.
    pub fn reset_lambda(&mut self) {
        self.lambda = 0.0;
    }
}

impl SoftConstraint for AnchorConstraint {
    fn project(&mut self, particles: &mut [SoftParticle], dt_sub: Real) {
        let pos = particles[self.particle_idx].position;
        let diff = pos - self.anchor;
        let dist = diff.norm();
        if dist < 1e-12 {
            return;
        }
        let n = diff / dist;
        let c = dist;
        let w = particles[self.particle_idx].inverse_mass;
        let alpha = self.compliance / (dt_sub * dt_sub);
        let denom = w + alpha;
        if denom < 1e-20 {
            return;
        }
        let d_lambda = (-c - alpha * self.lambda) / denom;
        self.lambda += d_lambda;
        particles[self.particle_idx].position += n * (w * d_lambda);
    }

    fn reset_lambda(&mut self) {
        Self::reset_lambda(self);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintSolver (Gauss-Seidel PBD iteration)
// ─────────────────────────────────────────────────────────────────────────────

/// PBD/XPBD Gauss-Seidel constraint solver for mixed constraint types.
///
/// Iterates over a heterogeneous list of constraints in order, updating
/// particle positions each iteration.  Supports sub-stepping.
#[derive(Debug)]
pub struct ConstraintSolver {
    /// Number of Gauss-Seidel iterations per solve call.
    pub iterations: usize,
    /// Number of sub-steps per `solve` call.
    pub sub_steps: usize,
}

impl ConstraintSolver {
    /// Create a solver with given iteration count and sub-step count.
    pub fn new(iterations: usize, sub_steps: usize) -> Self {
        Self {
            iterations,
            sub_steps,
        }
    }

    /// Solve all constraints for a single time step `dt`.
    ///
    /// Integrates velocity → position, then projects constraints, then
    /// updates velocity from position change.
    pub fn solve(
        &self,
        particles: &mut [SoftParticle],
        constraints: &mut [Box<dyn SoftConstraint>],
        gravity: Vec3,
        dt: Real,
    ) {
        let sub_dt = dt / self.sub_steps as Real;
        for _ in 0..self.sub_steps {
            // Velocity integration
            for p in particles.iter_mut() {
                if p.inverse_mass > 0.0 {
                    p.velocity += gravity * sub_dt;
                    p.prev_position = p.position;
                    p.position += p.velocity * sub_dt;
                }
            }
            // XPBD: the Lagrange multipliers restart from zero every sub-step.
            for c in constraints.iter_mut() {
                c.reset_lambda();
            }
            // Constraint projection (Gauss-Seidel)
            for _iter in 0..self.iterations {
                for c in constraints.iter_mut() {
                    c.project(particles, sub_dt);
                }
            }
            // Velocity update from position correction
            for p in particles.iter_mut() {
                if p.inverse_mass > 0.0 {
                    p.velocity = (p.position - p.prev_position) / sub_dt;
                }
            }
        }
    }

    /// Run only the constraint projection phase (no integration).
    ///
    /// One call is the projection phase of a single sub-step of length
    /// `dt_sub`, so the constraints' Lagrange multipliers are reset first and
    /// then accumulated over the `iterations` Gauss-Seidel sweeps.
    pub fn project_only(
        &self,
        particles: &mut [SoftParticle],
        constraints: &mut [Box<dyn SoftConstraint>],
        dt_sub: Real,
    ) {
        for c in constraints.iter_mut() {
            c.reset_lambda();
        }
        for _iter in 0..self.iterations {
            for c in constraints.iter_mut() {
                c.project(particles, dt_sub);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use oxiphysics_core::math::Vec3;

    fn make_particle(x: f64, y: f64, z: f64) -> SoftParticle {
        SoftParticle::new(Vec3::new(x, y, z), 1.0)
    }

    fn make_static(x: f64, y: f64, z: f64) -> SoftParticle {
        SoftParticle::new_static(Vec3::new(x, y, z))
    }

    // ── IsometricBendingConstraint ───────────────────────────────────────────

    #[test]
    fn isometric_bending_energy_zero_at_rest() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.5, 1.0, 0.0),
            make_particle(0.5, -1.0, 0.0),
        ];
        let c = IsometricBendingConstraint::new([0, 1, 2, 3], &particles, 0.0);
        let e = c.bending_energy(&particles);
        assert!(e.abs() < 1e-12, "a flat hinge has no bending energy: {e}");
    }

    #[test]
    fn isometric_bending_project_does_not_increase_energy() {
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.5, 1.0, 0.0),
            make_particle(0.5, -1.0, 0.5), // perturbed
        ];
        let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &particles, 1e-4);
        let e_before = c.bending_energy(&particles);
        c.project(&mut particles, 1.0 / 60.0);
        let e_after = c.bending_energy(&particles);
        assert!(
            e_after <= e_before + 1e-8,
            "energy should not increase: before={e_before}, after={e_after}"
        );
    }

    #[test]
    fn isometric_bending_reset_lambda() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.5, 1.0, 0.0),
            make_particle(0.5, -1.0, 0.0),
        ];
        let mut c = IsometricBendingConstraint::new([0, 1, 2, 3], &particles, 1e-4);
        c.lambda = 5.0;
        c.reset_lambda();
        assert_eq!(c.lambda, 0.0);
    }

    // ── CSTStrainConstraint ──────────────────────────────────────────────────

    #[test]
    fn cst_strain_invariants_near_two_at_rest() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
        ];
        let c = CSTStrainConstraint::new([0, 1, 2], &particles, 0.0);
        let (i1, _) = c.strain_invariants(&particles);
        // At rest F = I in 2D, I1 = 2
        assert!((i1 - 2.0).abs() < 1e-8, "I1={i1} should be 2.0 at rest");
    }

    #[test]
    fn cst_strain_project_moves_particles() {
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(2.0, 0.0, 0.0), // stretched
            make_particle(0.0, 1.0, 0.0),
        ];
        let mut c = CSTStrainConstraint::new([0, 1, 2], &particles, 0.0);
        // Stretch p1
        particles[1].position = Vec3::new(3.0, 0.0, 0.0);
        let pos_before = particles[1].position;
        for _ in 0..20 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let moved = (particles[1].position - pos_before).norm();
        assert!(moved > 1e-8, "particle should have moved: {moved}");
    }

    #[test]
    fn cst_strain_reset_lambda() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
        ];
        let mut c = CSTStrainConstraint::new([0, 1, 2], &particles, 0.0);
        c.lambda = 3.0;
        c.reset_lambda();
        assert_eq!(c.lambda, 0.0);
    }

    // ── NeoHookeanConstraint ─────────────────────────────────────────────────

    #[test]
    fn neohookean_constraint_values_at_rest() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let c = NeoHookeanConstraint::new([0, 1, 2, 3], &particles, 1e6, 0.3, 0.0);
        // At F = I the volume is unchanged and the deviatoric constraint is
        // sqrt(3); the hydrostatic constraint det F - gamma equals -mu / K, which
        // the deviatoric force balances.
        let strain = c.volume_strain(&particles);
        assert!(strain.abs() < 1e-12, "volume strain at rest: {strain}");
        let dev = c.deviatoric_constraint(&particles);
        assert!((dev - 3.0_f64.sqrt()).abs() < 1e-12, "C_D at rest: {dev}");
        let k = c.lame_lambda + c.lame_mu;
        let h = c.hydrostatic_constraint(&particles);
        assert!(
            (h + c.lame_mu / k).abs() < 1e-12,
            "C_H at rest should be -mu/K = {}: {h}",
            -c.lame_mu / k
        );
        assert!((c.hydrostatic_target() - (1.0 + c.lame_mu / k)).abs() < 1e-12);
    }

    #[test]
    fn neohookean_project_reduces_volume_error() {
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let mut c = NeoHookeanConstraint::new([0, 1, 2, 3], &particles, 1e6, 0.3, 0.0);
        // Compress vertex 3
        particles[3].position = Vec3::new(0.0, 0.0, 0.3);
        let c_before = c.volume_strain(&particles).abs();
        for _ in 0..50 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let c_after = c.volume_strain(&particles).abs();
        assert!(
            c_after < c_before,
            "volume error should decrease: before={c_before}, after={c_after}"
        );
    }

    #[test]
    fn neohookean_rest_volume_positive() {
        let particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let c = NeoHookeanConstraint::new([0, 1, 2, 3], &particles, 1e6, 0.3, 0.0);
        assert!(c.rest_volume > 0.0, "rest_volume={}", c.rest_volume);
    }

    // ── VolumeConservationConstraint ─────────────────────────────────────────

    #[test]
    fn volume_conservation_compute_volume_unit_cube() {
        // Approximate a cube with 12 triangles (2 per face of unit cube)
        let particles: Vec<SoftParticle> = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .iter()
        .map(|&[x, y, z]| make_particle(x, y, z))
        .collect();
        let triangles = vec![
            [0, 1, 2],
            [0, 2, 3], // bottom
            [4, 6, 5],
            [4, 7, 6], // top
            [0, 5, 1],
            [0, 4, 5], // front
            [2, 7, 3],
            [2, 6, 7], // back
            [0, 3, 7],
            [0, 7, 4], // left
            [1, 5, 6],
            [1, 6, 2], // right
        ];
        let vol = VolumeConservationConstraint::compute_volume(&triangles, &particles);
        assert!((vol.abs() - 1.0).abs() < 0.01, "vol={vol}");
    }

    #[test]
    fn volume_conservation_project_corrects_volume() {
        // Tetrahedron as a closed mesh (4 faces)
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let rest_vol = VolumeConservationConstraint::compute_volume(&tris, &particles).abs();
        // Perturb
        particles[3].position = Vec3::new(0.0, 0.0, 2.0);
        let mut vc = VolumeConservationConstraint::new(tris.clone(), rest_vol, 0.0);
        for _ in 0..100 {
            vc.reset_lambda();
            vc.project(&mut particles, 1.0 / 60.0);
        }
        let vol_after = VolumeConservationConstraint::compute_volume(&tris, &particles).abs();
        assert!(
            (vol_after - rest_vol).abs() < 0.05 * rest_vol + 1e-8,
            "vol_after={vol_after}, rest_vol={rest_vol}"
        );
    }

    // ── CableConstraint ──────────────────────────────────────────────────────

    #[test]
    fn cable_no_correction_when_slack() {
        let mut particles = vec![make_particle(0.0, 0.0, 0.0), make_particle(0.5, 0.0, 0.0)];
        let mut c = CableConstraint::new(0, 1, 1.0, 0.0); // rest = 1.0, current = 0.5 (slack)
        c.project(&mut particles, 1.0 / 60.0);
        assert!(
            (particles[1].position.x - 0.5).abs() < 1e-10,
            "slack cable should not move particles"
        );
    }

    #[test]
    fn cable_corrects_when_extended() {
        let mut particles = vec![make_particle(0.0, 0.0, 0.0), make_particle(2.0, 0.0, 0.0)];
        let mut c = CableConstraint::new(0, 1, 1.0, 0.0); // rest = 1.0, current = 2.0 (taut)
        for _ in 0..50 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let dist = (particles[0].position - particles[1].position).norm();
        assert!(
            (dist - 1.0).abs() < 0.01,
            "cable should be pulled to rest length: dist={dist}"
        );
    }

    #[test]
    fn cable_from_particles() {
        let particles = vec![make_particle(0.0, 0.0, 0.0), make_particle(1.5, 0.0, 0.0)];
        let c = CableConstraint::from_particles(0, 1, &particles, 0.0);
        assert!(
            (c.rest_length - 1.5).abs() < 1e-10,
            "rest={}",
            c.rest_length
        );
    }

    // ── RigidBodyConstraint ──────────────────────────────────────────────────

    #[test]
    fn rigid_body_world_attachment_identity() {
        // Identity rotation
        let rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = RigidBodyConstraint::new(
            0,
            Vec3::new(1.0, 2.0, 3.0),
            rot,
            Vec3::new(0.1, 0.0, 0.0),
            0.0,
        );
        let wa = c.world_attachment();
        assert!((wa.x - 1.1).abs() < 1e-10, "wa.x={}", wa.x);
        assert!((wa.y - 2.0).abs() < 1e-10, "wa.y={}", wa.y);
    }

    #[test]
    fn rigid_body_project_moves_particle_to_target() {
        let rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut particles = vec![make_particle(5.0, 5.0, 5.0)];
        let mut c = RigidBodyConstraint::new(0, Vec3::zeros(), rot, Vec3::zeros(), 0.0);
        for _ in 0..100 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let dist = particles[0].position.norm();
        assert!(dist < 0.01, "particle should move to origin: dist={dist}");
    }

    // ── SewingConstraint ─────────────────────────────────────────────────────

    #[test]
    fn sewing_reduces_separation() {
        let mut particles = vec![make_particle(0.0, 0.0, 0.0), make_particle(2.0, 0.0, 0.0)];
        let mut c = SewingConstraint::new(vec![[0, 1]], 0.0);
        for _ in 0..100 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let dist = (particles[0].position - particles[1].position).norm();
        assert!(dist < 0.01, "seam particles should converge: dist={dist}");
    }

    #[test]
    fn sewing_multiple_pairs() {
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(2.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(3.0, 0.0, 0.0),
        ];
        let mut c = SewingConstraint::new(vec![[0, 1], [2, 3]], 0.0);
        for _ in 0..100 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let d0 = (particles[0].position - particles[1].position).norm();
        let d1 = (particles[2].position - particles[3].position).norm();
        assert!(d0 < 0.01, "pair 0 should converge: {d0}");
        assert!(d1 < 0.01, "pair 1 should converge: {d1}");
    }

    // ── PressureConstraint ───────────────────────────────────────────────────

    #[test]
    fn pressure_constraint_inflates_mesh() {
        // Tetrahedron as closed mesh
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let rest_vol = VolumeConservationConstraint::compute_volume(&tris, &particles).abs();
        // Pressure ratio > 1 → inflate
        let mut pc = PressureConstraint::new(tris.clone(), rest_vol, 1.5, 1e-6);
        for _ in 0..100 {
            pc.reset_lambda();
            pc.project(&mut particles, 1.0 / 60.0);
        }
        let vol_after = VolumeConservationConstraint::compute_volume(&tris, &particles).abs();
        assert!(
            vol_after > rest_vol * 1.1,
            "mesh should inflate: vol_after={vol_after}, rest={rest_vol}"
        );
    }

    #[test]
    fn pressure_constraint_no_change_at_ratio_one() {
        let mut particles = vec![
            make_particle(0.0, 0.0, 0.0),
            make_particle(1.0, 0.0, 0.0),
            make_particle(0.0, 1.0, 0.0),
            make_particle(0.0, 0.0, 1.0),
        ];
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let rest_vol = VolumeConservationConstraint::compute_volume(&tris, &particles).abs();
        let mut pc = PressureConstraint::new(tris.clone(), rest_vol, 1.0, 0.0);
        let pos_before: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        pc.project(&mut particles, 1.0 / 60.0);
        for (i, p) in particles.iter().enumerate() {
            let d = (p.position - pos_before[i]).norm();
            assert!(
                d < 1e-8,
                "positions should not change at ratio=1: d[{i}]={d}"
            );
        }
    }

    // ── AnchorConstraint ─────────────────────────────────────────────────────

    #[test]
    fn anchor_moves_particle_to_anchor_point() {
        let mut particles = vec![make_particle(5.0, 5.0, 5.0)];
        let anchor = Vec3::new(1.0, 2.0, 3.0);
        let mut c = AnchorConstraint::new(0, anchor, 0.0);
        for _ in 0..100 {
            c.reset_lambda();
            c.project(&mut particles, 1.0 / 60.0);
        }
        let dist = (particles[0].position - anchor).norm();
        assert!(dist < 0.01, "particle should reach anchor: dist={dist}");
    }

    #[test]
    fn anchor_static_particle_not_moved() {
        let mut particles = vec![make_static(5.0, 5.0, 5.0)];
        let anchor = Vec3::zeros();
        let mut c = AnchorConstraint::new(0, anchor, 0.0);
        c.project(&mut particles, 1.0 / 60.0);
        let dist = (particles[0].position - Vec3::new(5.0, 5.0, 5.0)).norm();
        assert!(dist < 1e-10, "static particle should not move: dist={dist}");
    }

    #[test]
    fn anchor_compliance_allows_offset() {
        let mut particles = vec![make_particle(2.0, 0.0, 0.0)];
        let anchor = Vec3::zeros();
        // Very high compliance: particle barely moves
        let mut c = AnchorConstraint::new(0, anchor, 1e6);
        c.project(&mut particles, 1.0 / 60.0);
        let dist = particles[0].position.norm();
        assert!(
            dist > 1.0,
            "high compliance should allow large offset: dist={dist}"
        );
    }

    // ── ConstraintSolver ─────────────────────────────────────────────────────

    #[test]
    fn constraint_solver_gravity_falls_particle() {
        let mut particles = vec![make_particle(0.0, 10.0, 0.0)];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let mut constraints: Vec<Box<dyn SoftConstraint>> = Vec::new();
        let solver = ConstraintSolver::new(5, 2);
        let dt = 1.0 / 60.0;
        // Run for 2 seconds
        for _ in 0..120 {
            solver.solve(&mut particles, &mut constraints, gravity, dt);
        }
        let y = particles[0].position.y;
        // After 2 s: y ≈ 10 - 0.5*9.81*4 ≈ -9.6
        assert!(
            y < 0.0,
            "particle should fall well below zero after 2 s, y={y}"
        );
    }

    #[test]
    fn constraint_solver_anchor_holds_particle() {
        let mut particles = vec![make_particle(0.0, 5.0, 0.0)];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let anchor = Vec3::new(0.0, 5.0, 0.0);
        let mut constraints: Vec<Box<dyn SoftConstraint>> =
            vec![Box::new(AnchorConstraint::new(0, anchor, 0.0))];
        let solver = ConstraintSolver::new(10, 4);
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            solver.solve(&mut particles, &mut constraints, gravity, dt);
        }
        let dist = (particles[0].position - anchor).norm();
        assert!(
            dist < 0.1,
            "anchored particle should stay near anchor: dist={dist}"
        );
    }

    #[test]
    fn constraint_solver_cable_limits_extension() {
        let mut particles = vec![make_static(0.0, 0.0, 0.0), make_particle(0.0, -0.5, 0.0)];
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let rest_len = 1.0;
        let mut constraints: Vec<Box<dyn SoftConstraint>> =
            vec![Box::new(CableConstraint::new(0, 1, rest_len, 0.0))];
        let solver = ConstraintSolver::new(10, 4);
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            solver.solve(&mut particles, &mut constraints, gravity, dt);
        }
        let dist = (particles[0].position - particles[1].position).norm();
        assert!(
            dist <= rest_len + 0.05,
            "cable should limit extension: dist={dist}"
        );
    }

    #[test]
    fn constraint_solver_project_only() {
        let mut particles = vec![make_particle(0.0, 0.0, 0.0), make_particle(3.0, 0.0, 0.0)];
        let anchor = Vec3::zeros();
        let mut constraints: Vec<Box<dyn SoftConstraint>> = vec![Box::new(AnchorConstraint::new(
            1,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
        ))];
        let solver = ConstraintSolver::new(50, 1);
        solver.project_only(&mut particles, &mut constraints, 1.0 / 60.0);
        let dist = (particles[1].position - Vec3::new(1.0, 0.0, 0.0)).norm();
        assert!(dist < 0.1, "project_only should move particle: dist={dist}");
        let _ = anchor;
    }

    // ── helper math ─────────────────────────────────────────────────────────

    #[test]
    fn mat3_mul_vec3_applies_the_row_major_matrix() {
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0_f64];
        let r = mat3_mul_vec3(&m, Vec3::new(1.0, -1.0, 2.0));
        assert!((r.x - 5.0).abs() < 1e-12, "r.x={}", r.x);
        assert!((r.y - 11.0).abs() < 1e-12, "r.y={}", r.y);
        assert!((r.z - 19.0).abs() < 1e-12, "r.z={}", r.z);
    }

    #[test]
    fn outer_product_is_correct() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let o = outer(a, b);
        assert!((o[0] - 4.0).abs() < 1e-10); // a.x * b.x
        assert!((o[4] - 10.0).abs() < 1e-10); // a.y * b.y
        assert!((o[8] - 18.0).abs() < 1e-10); // a.z * b.z
    }
}
