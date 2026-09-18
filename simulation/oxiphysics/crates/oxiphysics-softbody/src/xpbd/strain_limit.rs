// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SVD-based per-triangle strain limiting for cloth simulation.
//!
//! Every triangle carries the 3×2 deformation gradient `F = [f0 f1] · Dm⁻¹`.
//! `Dm` is the 2×2 shape matrix of the *rest* triangle in its own tangent
//! frame, and `f0 = p1 − p0`, `f1 = p2 − p0` are the *current* edges as 3-D
//! vectors, out-of-plane components included.  `F` maps rest tangent-plane
//! coordinates to world vectors, so a rigid motion of the triangle (any
//! rotation, any translation) leaves both singular values of `F` at exactly
//! `1` and the limiter does not touch it.
//!
//! `F = U Σ Vᵀ` is evaluated with an analytic SVD (`U` has two orthonormal
//! world-space columns, `V` is a 2×2 rotation), the singular values are
//! clamped into `[min_stretch, max_stretch]`, and the edge changes
//! `(F' − F) · Dm` are turned into vertex displacements that reach exactly the
//! clamped shape with the smallest mass-weighted movement.  A pinned vertex
//! never moves; a triangle without pinned vertices is corrected without any net
//! linear momentum.
//!
//! [`apply_strain_limiting`] works on plain arrays.  To use it in an XPBD
//! simulation, give the cloth topology to the body with
//! [`XpbdBody::enable_strain_limiting`](super::XpbdBody::enable_strain_limiting):
//! the resulting [`XpbdStrainLimiter`] runs the pass in every substep of
//! [`XpbdBody::step`](super::XpbdBody::step) and
//! [`XpbdBody::step_with_stats`](super::XpbdBody::step_with_stats).

use crate::{Error, Result};
use oxiphysics_core::math::Vec3;

/// Per-triangle strain-limiting parameters.
///
/// The singular values of a triangle's deformation gradient are clamped into
/// `[min_stretch, max_stretch]`.  A bound that is `NaN` does not limit, and if
/// `min_stretch > max_stretch` the upper bound wins.
#[derive(Debug, Clone)]
pub struct StrainLimitConfig {
    /// Maximum allowed singular value (stretch). Default `1.05`.
    pub max_stretch: f64,
    /// Minimum allowed singular value (compression). Default `0.95`.
    pub min_stretch: f64,
    /// Number of Gauss–Seidel sweeps per call. Default `1`.
    pub iterations: u32,
}

impl Default for StrainLimitConfig {
    fn default() -> Self {
        Self {
            max_stretch: 1.05,
            min_stretch: 0.95,
            iterations: 1,
        }
    }
}

// ─────────────────────────── constants ───────────────────────────────────────

/// Smallest singular value (dimensionless) that still defines a left singular
/// vector.
const SIGMA_EPSILON: f64 = 1e-12;

/// Ratio `σ₂ / σ₁` below which the second singular value counts as zero: the
/// direction of `F v₂` is then rounding noise.
const RANK_TOLERANCE: f64 = 1e-9;

/// Height of the rest triangle over its first edge, relative to the longer of
/// the two rest edges at the first vertex, below which the triangle counts as
/// degenerate.
const MIN_REST_ASPECT: f64 = 1e-9;

/// Required change of a singular value (dimensionless): a smaller change is the
/// rounding noise of an earlier projection, and the triangle already counts as
/// inside its limits.
const STRAIN_NOISE: f64 = 1e-14;

// ─────────────────────────── helpers ─────────────────────────────────────────

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Clamp a singular value into `[min, max]` without panicking on a bad range.
///
/// Unlike [`f64::clamp`] this accepts `min > max` (the upper bound wins) and
/// `NaN` bounds (they do not limit).
#[inline]
fn limit_stretch(sigma: f64, min: f64, max: f64) -> f64 {
    sigma.max(min).min(max)
}

// ─────────────────────────── rest triangle ───────────────────────────────────

/// Orthonormal tangent frame and shape matrix of a rest triangle.
///
/// The frame is `t1 = (q1 − q0) / |q1 − q0|` and the unit vector `t2` in the
/// triangle's plane that is perpendicular to `t1` and points towards `q2`.  The
/// shape matrix `Dm = [[d11, d12], [0, d22]]` holds the rest edges `q1 − q0`
/// and `q2 − q0` as columns in that frame.
struct RestTriangle {
    /// First frame axis (along the first rest edge).
    t1: [f64; 3],
    /// Second frame axis (in the rest plane, perpendicular to `t1`).
    t2: [f64; 3],
    /// Unit rest normal `t1 × t2`.
    normal: [f64; 3],
    /// `Dm[0][0]`: length of the first rest edge.
    d11: f64,
    /// `Dm[0][1]`: component of the second rest edge along `t1`.
    d12: f64,
    /// `Dm[1][1]`: height of `q2` above the first rest edge.
    d22: f64,
}

impl RestTriangle {
    /// Frame and shape matrix of the rest triangle `(q0, q1, q2)`, or `None`
    /// when it is degenerate (a zero-length edge, a vertex on the line through
    /// the other two, or a non-finite coordinate).
    fn new(q0: [f64; 3], q1: [f64; 3], q2: [f64; 3]) -> Option<Self> {
        let e0 = sub3(q1, q0);
        let e1 = sub3(q2, q0);
        let d11 = norm3(e0);
        let e1_len = norm3(e1);
        if !d11.is_finite() || !e1_len.is_finite() || d11 <= 0.0 {
            return None;
        }
        let t1 = scale3(e0, 1.0 / d11);
        let d12 = dot3(e1, t1);
        let e1_orth = sub3(e1, scale3(t1, d12));
        let d22 = norm3(e1_orth);
        if !d22.is_finite() || d22 <= MIN_REST_ASPECT * d11.max(e1_len) {
            return None;
        }
        let t2 = scale3(e1_orth, 1.0 / d22);
        Some(Self {
            t1,
            t2,
            normal: cross3(t1, t2),
            d11,
            d12,
            d22,
        })
    }

    /// Columns of the deformation gradient `F = [f0 f1] · Dm⁻¹` for the current
    /// edges `f0 = p1 − p0` and `f1 = p2 − p0`.
    ///
    /// `Dm⁻¹ = [[1/d11, −d12/(d11 d22)], [0, 1/d22]]`, so the first column of
    /// `F` is `f0 / d11` and the second is `(f1 − d12 · F₀) / d22`.
    fn deformation_gradient(&self, f0: [f64; 3], f1: [f64; 3]) -> [[f64; 3]; 2] {
        let col0 = scale3(f0, 1.0 / self.d11);
        let col1 = scale3(sub3(f1, scale3(col0, self.d12)), 1.0 / self.d22);
        [col0, col1]
    }

    /// Edge changes `ΔDs = ΔF · Dm`, returned as the pair `(Δf0, Δf1)`, for a
    /// change `ΔF` of the deformation gradient given by its columns.
    fn edge_changes(&self, delta_f: [[f64; 3]; 2]) -> ([f64; 3], [f64; 3]) {
        let delta_f0 = scale3(delta_f[0], self.d11);
        let delta_f1 = add3(scale3(delta_f[0], self.d12), scale3(delta_f[1], self.d22));
        (delta_f0, delta_f1)
    }

    /// A rest-frame vector `(x, y)` as a world-space vector.
    fn embed(&self, v: [f64; 2]) -> [f64; 3] {
        add3(scale3(self.t1, v[0]), scale3(self.t2, v[1]))
    }

    /// Unit vector `u2` perpendicular to the unit vector `u1` for a left
    /// singular vector that `F` does not determine (`σ₂ = 0`).
    ///
    /// It is chosen so that `u1 × u2` points along the rest normal: a collapsed
    /// triangle regains its rest orientation instead of flipping.  When `u1`
    /// itself lies along the normal every in-plane direction is equivalent and
    /// the rest-frame direction `v2` is used.
    fn missing_left_vector(&self, u1: [f64; 3], v2: [f64; 2]) -> [f64; 3] {
        let across = cross3(self.normal, u1);
        let across_len = norm3(across);
        if across_len > 1e-3 {
            return scale3(across, 1.0 / across_len);
        }
        let in_plane = self.embed(v2);
        let ortho = sub3(in_plane, scale3(u1, dot3(in_plane, u1)));
        scale3(ortho, 1.0 / norm3(ortho))
    }
}

// ─────────────────────────── 3×2 SVD ─────────────────────────────────────────

/// Singular value decomposition `F = U Σ Vᵀ` of a 3×2 matrix.
struct Svd {
    /// Singular values `σ₁ ≥ σ₂ ≥ 0`.
    sigma: [f64; 2],
    /// Right singular vectors `v₁`, `v₂` in rest-frame coordinates: the rows of
    /// the rotation `Vᵀ` (`det V = +1`).
    v: [[f64; 2]; 2],
    /// Left singular vectors `u₁`, `u₂`: orthonormal world-space vectors with
    /// `F v_k = σ_k u_k`.
    u: [[f64; 3]; 2],
}

impl Svd {
    /// Analytic SVD of the 3×2 matrix with columns `f[0]`, `f[1]`.
    ///
    /// `FᵀF = [[a, b], [b, d]]` is diagonalised by the rotation through
    /// `½ atan2(2b, a − d)`, which gives `V` with `v₁` the dominant direction;
    /// then `σ_k = |F v_k|` and `u_k = F v_k / σ_k`.  `F v₂` is orthogonalised
    /// against `u₁` to remove rounding drift.  A left singular vector that `F`
    /// does not determine (`σ ≈ 0`) comes from the rest orientation of
    /// `rest`, see [`RestTriangle::missing_left_vector`].
    fn new(f: [[f64; 3]; 2], rest: &RestTriangle) -> Self {
        let a = dot3(f[0], f[0]);
        let b = dot3(f[0], f[1]);
        let d = dot3(f[1], f[1]);
        let angle = 0.5 * (2.0 * b).atan2(a - d);
        let (sin, cos) = angle.sin_cos();
        let v = [[cos, sin], [-sin, cos]];
        let image = |w: [f64; 2]| add3(scale3(f[0], w[0]), scale3(f[1], w[1]));
        let fv1 = image(v[0]);
        let fv2 = image(v[1]);
        let sigma = [norm3(fv1), norm3(fv2)];
        let u1 = if sigma[0] > SIGMA_EPSILON {
            scale3(fv1, 1.0 / sigma[0])
        } else {
            rest.embed(v[0])
        };
        let ortho = sub3(fv2, scale3(u1, dot3(fv2, u1)));
        let ortho_len = norm3(ortho);
        let u2 = if ortho_len > SIGMA_EPSILON.max(RANK_TOLERANCE * sigma[0]) {
            scale3(ortho, 1.0 / ortho_len)
        } else {
            rest.missing_left_vector(u1, v[1])
        };
        Self {
            sigma,
            v,
            u: [u1, u2],
        }
    }

    /// Change `ΔF = Σ_k δ_k u_k v_kᵀ` of the deformation gradient (as columns)
    /// when the singular values change by `delta = (δ₁, δ₂)`.
    fn delta_gradient(&self, delta: [f64; 2]) -> [[f64; 3]; 2] {
        let mut delta_f = [[0.0; 3]; 2];
        for ((u, v), d) in self.u.iter().zip(&self.v).zip(delta) {
            for (column, v_component) in delta_f.iter_mut().zip(v) {
                *column = add3(*column, scale3(*u, d * v_component));
            }
        }
        delta_f
    }
}

// ─────────────────────────── vertex displacements ────────────────────────────

/// Vertex displacements `[Δp0, Δp1, Δp2]` that change the edges `p1 − p0` by `a`
/// and `p2 − p0` by `b` with the smallest mass-weighted movement
/// `Σ |Δp_i|² / w_i`; `inv_masses` are the inverse masses `w_i`, and a vertex
/// with `w_i ≤ 0` is pinned.
///
/// The multipliers `(μ1, μ2)` of the two edge constraints solve
/// `[[w0 + w1, w0], [w0, w0 + w2]] (μ1, μ2) = (a, b)`, and
/// `Δp0 = −w0 (μ1 + μ2)`, `Δp1 = w1 μ1`, `Δp2 = w2 μ2`.  Both edge changes are
/// met exactly whenever at least two vertices are free, and with all three free
/// the displacements conserve linear momentum (`Σ Δp_i / w_i = 0`); a pinned
/// vertex stands in for an infinite mass and takes no part.  With a single free
/// vertex the two constraints cannot hold together: a free `p1` or `p2` takes
/// the change of its own edge and a free `p0` the least-squares compromise, the
/// average of the two.
fn distribute_edge_changes(inv_masses: [f64; 3], a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 3] {
    let [w0, w1, w2] = inv_masses.map(|w| w.max(0.0));
    let det = w0 * w1 + w0 * w2 + w1 * w2;
    if det.is_finite() && det > 0.0 {
        let dp0 = scale3(add3(scale3(a, w2), scale3(b, w1)), -w0 / det);
        let dp1 = scale3(sub3(scale3(a, w0 + w2), scale3(b, w0)), w1 / det);
        let dp2 = scale3(sub3(scale3(b, w0 + w1), scale3(a, w0)), w2 / det);
        return [dp0, dp1, dp2];
    }
    let zero = [0.0; 3];
    match (w0 > 0.0, w1 > 0.0, w2 > 0.0) {
        (true, false, false) => [scale3(add3(a, b), -0.5), zero, zero],
        (false, true, false) => [zero, a, zero],
        (false, false, true) => [zero, zero, b],
        _ => [zero; 3],
    }
}

// ─────────────────────────── main API ────────────────────────────────────────

/// Apply SVD-based per-triangle strain limiting.
///
/// For each triangle the 3×2 deformation gradient `F = [f0 f1] · Dm⁻¹` is
/// formed from the rest shape matrix `Dm` (in the rest triangle's own tangent
/// frame) and the *current* 3-D edges, so the whole motion of the triangle is
/// seen, out-of-plane components included: a rigidly rotated or translated
/// triangle has singular values of exactly `1` and is left alone.  `F` is
/// decomposed with an analytic SVD and any singular value outside
/// `[min_stretch, max_stretch]` is clamped; the edge changes `(F' − F) · Dm`
/// are then applied to the vertices with the smallest mass-weighted movement
/// (inverse-mass-proportional corrections; a triangle without pinned vertices
/// is corrected without net linear momentum), which meets the clamped shape
/// exactly whenever at least two vertices of the triangle are free.  Pinned
/// vertices (`inv_mass ≤ 0`) never move.
///
/// Triangles are processed one after the other (Gauss–Seidel), so a vertex
/// shared with a later triangle may be corrected again; `config.iterations`
/// repeats the sweep.  Rest triangles that are degenerate (zero-length edge, or
/// a vertex on the line through the other two) and triangles with non-finite
/// positions are skipped.
///
/// # Arguments
///
/// * `positions`      – current 3-D positions, modified in-place.
/// * `rest_positions` – reference (rest) 3-D positions, not modified.
/// * `triangles`      – index triples `[i0, i1, i2]` into the position arrays.
/// * `inv_masses`     – per-particle inverse mass (`0` = pinned / static).
/// * `config`         – algorithm parameters.
///
/// # Panics
///
/// Panics if a triangle refers to an index outside `positions`,
/// `rest_positions` or `inv_masses`.
pub fn apply_strain_limiting(
    positions: &mut [[f64; 3]],
    rest_positions: &[[f64; 3]],
    triangles: &[[usize; 3]],
    inv_masses: &[f64],
    config: &StrainLimitConfig,
) {
    for _ in 0..config.iterations {
        for &[i0, i1, i2] in triangles {
            let Some(rest) =
                RestTriangle::new(rest_positions[i0], rest_positions[i1], rest_positions[i2])
            else {
                continue;
            };

            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];
            let f = rest.deformation_gradient(sub3(p1, p0), sub3(p2, p0));
            if !f.iter().flatten().all(|x| x.is_finite()) {
                continue;
            }

            let svd = Svd::new(f, &rest);
            if !svd.sigma.iter().all(|s| s.is_finite()) {
                continue;
            }
            let delta = svd
                .sigma
                .map(|s| limit_stretch(s, config.min_stretch, config.max_stretch) - s);
            if delta.iter().all(|d| d.abs() <= STRAIN_NOISE) {
                continue;
            }

            let (change_0, change_1) = rest.edge_changes(svd.delta_gradient(delta));
            let weights = [inv_masses[i0], inv_masses[i1], inv_masses[i2]];
            let [dp0, dp1, dp2] = distribute_edge_changes(weights, change_0, change_1);
            positions[i0] = add3(p0, dp0);
            positions[i1] = add3(p1, dp1);
            positions[i2] = add3(p2, dp2);
        }
    }
}

// ─────────────────────────── XPBD body adapter ───────────────────────────────

/// Opt-in per-triangle strain limiting for an [`XpbdBody`](super::XpbdBody): the
/// cloth topology and rest shape that [`apply_strain_limiting`] needs, kept with
/// the body so that the pass can run inside its substep loop.
///
/// [`XpbdBody::step`](super::XpbdBody::step) and
/// [`XpbdBody::step_with_stats`](super::XpbdBody::step_with_stats) run the pass
/// once per substep, after the constraint projections and before the velocities
/// are recomputed from the position change.  The correction is therefore part of
/// the substep's displacement and reaches the velocities the way a constraint
/// projection does.  A body without a limiter
/// ([`XpbdBody::strain_limiter`](super::XpbdBody::strain_limiter) is `None`, the
/// default) is stepped exactly as plain XPBD.
///
/// Particles that are added to the body after the limiter was built are not
/// covered by it.
#[derive(Debug, Clone)]
pub struct XpbdStrainLimiter {
    /// Triangles as index triples into the body's particle arrays.
    triangles: Vec<[usize; 3]>,
    /// Rest position of every particle, indexed like the body's particles.
    rest_positions: Vec<[f64; 3]>,
    /// Stretch limits and number of Gauss–Seidel sweeps per pass.
    config: StrainLimitConfig,
    /// One more than the largest particle index any triangle refers to: the
    /// number of leading particles a pass reads and writes.
    particle_span: usize,
    /// The leading body positions in the `[f64; 3]` layout that
    /// [`apply_strain_limiting`] works on, reused from pass to pass.
    scratch: Vec<[f64; 3]>,
}

impl XpbdStrainLimiter {
    /// Limiter for `triangles` whose rest shape is `rest_positions`.
    ///
    /// `rest_positions` holds one position per particle, indexed like the
    /// particles of the body.
    /// [`XpbdBody::enable_strain_limiting`](super::XpbdBody::enable_strain_limiting)
    /// passes the body's current positions; call this directly to limit towards
    /// another rest shape (a cloth that starts out draped but is flat at rest).
    ///
    /// Triangles whose rest shape is degenerate are left unlimited, as in
    /// [`apply_strain_limiting`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if a triangle refers to a particle that has no
    /// rest position.
    pub fn new(
        triangles: Vec<[usize; 3]>,
        rest_positions: &[Vec3],
        config: StrainLimitConfig,
    ) -> Result<Self> {
        let mut particle_span = 0;
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            for &particle in triangle {
                if particle >= rest_positions.len() {
                    return Err(Error::InvalidParameter(format!(
                        "strain-limiting triangle {triangle_index} refers to particle \
                         {particle}, but only {} rest positions were given",
                        rest_positions.len()
                    )));
                }
                particle_span = particle_span.max(particle + 1);
            }
        }
        Ok(Self {
            triangles,
            rest_positions: rest_positions.iter().map(|p| [p.x, p.y, p.z]).collect(),
            config,
            particle_span,
            scratch: Vec::with_capacity(particle_span),
        })
    }

    /// The triangles that are limited, as index triples into the particles of the
    /// body.
    pub fn triangles(&self) -> &[[usize; 3]] {
        &self.triangles
    }

    /// Rest position of every particle, indexed like the body's particles.
    pub fn rest_positions(&self) -> &[[f64; 3]] {
        &self.rest_positions
    }

    /// The stretch limits and the number of sweeps per pass.
    pub fn config(&self) -> &StrainLimitConfig {
        &self.config
    }

    /// Replace the stretch limits and the number of sweeps per pass; the
    /// triangles and the rest shape stay.
    pub fn set_config(&mut self, config: StrainLimitConfig) {
        self.config = config;
    }

    /// Run one limiting pass over `positions` and return the number of triangle
    /// projections it performed (`config().iterations` sweeps over all
    /// triangles).
    ///
    /// A particle with `inv_masses[i] <= 0` is pinned and never moves.  This
    /// never panics: when `positions` or `inv_masses` hold fewer particles than
    /// the triangles refer to, nothing is limited and `0` is returned.
    pub fn apply(&mut self, positions: &mut [Vec3], inv_masses: &[f64]) -> usize {
        if self.triangles.is_empty() || self.config.iterations == 0 {
            return 0;
        }
        if positions.len() < self.particle_span || inv_masses.len() < self.particle_span {
            return 0;
        }
        self.scratch.clear();
        self.scratch.extend(
            positions
                .iter()
                .take(self.particle_span)
                .map(|p| [p.x, p.y, p.z]),
        );
        apply_strain_limiting(
            &mut self.scratch,
            &self.rest_positions,
            &self.triangles,
            inv_masses,
            &self.config,
        );
        for (position, limited) in positions.iter_mut().zip(&self.scratch) {
            *position = Vec3::new(limited[0], limited[1], limited[2]);
        }
        (self.config.iterations as usize).saturating_mul(self.triangles.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUNDING: f64 = 1e-12;

    fn unit_rest() -> RestTriangle {
        RestTriangle::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            .expect("the unit right triangle is not degenerate")
    }

    fn skewed_rest() -> RestTriangle {
        RestTriangle::new([0.5, -0.2, 1.0], [1.7, 0.1, 0.6], [0.9, 1.2, 0.9])
            .expect("the skewed triangle is not degenerate")
    }

    /// Check `F = U Σ Vᵀ` with orthonormal `U`, a proper rotation `V` and
    /// ordered non-negative singular values.
    fn assert_valid_svd(f: [[f64; 3]; 2], svd: &Svd) {
        for (j, column) in f.iter().enumerate() {
            let rebuilt = svd
                .u
                .iter()
                .zip(&svd.v)
                .zip(&svd.sigma)
                .fold([0.0; 3], |sum, ((u, v), sigma)| {
                    add3(sum, scale3(*u, sigma * v[j]))
                });
            let error = norm3(sub3(rebuilt, *column));
            assert!(error < ROUNDING, "column {j} rebuilt with error {error}");
        }
        assert!((norm3(svd.u[0]) - 1.0).abs() < ROUNDING);
        assert!((norm3(svd.u[1]) - 1.0).abs() < ROUNDING);
        assert!(dot3(svd.u[0], svd.u[1]).abs() < ROUNDING);
        let [v1, v2] = svd.v;
        assert!((v1[0] * v1[0] + v1[1] * v1[1] - 1.0).abs() < ROUNDING);
        assert!((v1[0] * v2[1] - v1[1] * v2[0] - 1.0).abs() < ROUNDING);
        assert!(svd.sigma[0] >= svd.sigma[1] - ROUNDING && svd.sigma[1] >= 0.0);
    }

    #[test]
    fn svd_handles_equal_and_ordered_diagonal_gradients() {
        let rest = unit_rest();
        // Equal singular values with a vanishing off-diagonal term.
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let svd = Svd::new(identity, &rest);
        assert_valid_svd(identity, &svd);
        assert!((svd.sigma[0] - 1.0).abs() < ROUNDING && (svd.sigma[1] - 1.0).abs() < ROUNDING);
        // Dominant direction along either axis.
        for (f, expected) in [
            ([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [3.0, 2.0]),
            ([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], [3.0, 2.0]),
        ] {
            let svd = Svd::new(f, &rest);
            assert_valid_svd(f, &svd);
            assert!(
                (svd.sigma[0] - expected[0]).abs() < ROUNDING,
                "{:?}",
                svd.sigma
            );
            assert!(
                (svd.sigma[1] - expected[1]).abs() < ROUNDING,
                "{:?}",
                svd.sigma
            );
        }
    }

    #[test]
    fn svd_reconstructs_general_gradients() {
        let rest = unit_rest();
        for f in [
            [[0.3, 1.2, -0.7], [0.9, -0.4, 0.5]],
            [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]],
            [[1e-3, 0.0, 5e-4], [2e-3, 1e-3, -1e-3]],
            [[40.0, -25.0, 10.0], [38.0, -24.0, 9.5]],
        ] {
            assert_valid_svd(f, &Svd::new(f, &rest));
        }
    }

    /// Deterministic xorshift64* generator for the randomized tests.
    struct Rng(u64);

    impl Rng {
        /// Uniform in `[0, 1)`.
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
            (bits >> 11) as f64 / (1_u64 << 53) as f64
        }

        /// Uniform in `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }

        fn vector(&mut self, lo: f64, hi: f64) -> [f64; 3] {
            [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
        }
    }

    #[test]
    fn svd_is_valid_for_random_gradients_of_every_conditioning() {
        let rest = skewed_rest();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for case in 0..2000 {
            let f = match case % 5 {
                // Generic matrices.
                0 => [rng.vector(-2.0, 2.0), rng.vector(-2.0, 2.0)],
                // Rank one: the second column is a multiple of the first.
                1 => {
                    let first = rng.vector(-2.0, 2.0);
                    [first, scale3(first, rng.range(-3.0, 3.0))]
                }
                // Nearly isometric: orthonormal columns plus a tiny perturbation.
                2 => {
                    let u = scale3(rng.vector(-1.0, 1.0), 1.0);
                    let u = scale3(u, 1.0 / norm3(u));
                    let w = cross3(u, rng.vector(-1.0, 1.0));
                    let w = scale3(w, 1.0 / norm3(w));
                    let noise = scale3(rng.vector(-1.0, 1.0), 1e-9);
                    [add3(u, noise), w]
                }
                // Tiny and huge scales.
                3 => [
                    scale3(rng.vector(-1.0, 1.0), 1e-8),
                    scale3(rng.vector(-1.0, 1.0), 1e-8),
                ],
                _ => [
                    scale3(rng.vector(-1.0, 1.0), 1e6),
                    scale3(rng.vector(-1.0, 1.0), 1e6),
                ],
            };
            let scale = 1.0_f64.max(norm3(f[0])).max(norm3(f[1]));
            let svd = Svd::new(f, &rest);
            assert!(svd.sigma.iter().all(|s| s.is_finite()), "case {case}");
            for (j, column) in f.iter().enumerate() {
                let rebuilt = svd
                    .u
                    .iter()
                    .zip(&svd.v)
                    .zip(&svd.sigma)
                    .fold([0.0; 3], |sum, ((u, v), sigma)| {
                        add3(sum, scale3(*u, sigma * v[j]))
                    });
                let error = norm3(sub3(rebuilt, *column));
                assert!(
                    error < ROUNDING * scale,
                    "case {case}: column {j} rebuilt with error {error} for {f:?}"
                );
            }
            assert!((norm3(svd.u[0]) - 1.0).abs() < ROUNDING, "case {case}");
            assert!((norm3(svd.u[1]) - 1.0).abs() < ROUNDING, "case {case}");
            assert!(dot3(svd.u[0], svd.u[1]).abs() < ROUNDING, "case {case}");
            assert!(
                svd.sigma[0] >= svd.sigma[1] - ROUNDING * scale,
                "case {case}"
            );
            // The singular values satisfy sigma1^2 + sigma2^2 = |F|_F^2 and
            // sigma1 sigma2 = |F0 x F1|.
            let frobenius_sq = dot3(f[0], f[0]) + dot3(f[1], f[1]);
            let sum_sq = svd.sigma[0] * svd.sigma[0] + svd.sigma[1] * svd.sigma[1];
            assert!(
                (sum_sq - frobenius_sq).abs() <= 1e-12 * frobenius_sq.max(1e-300),
                "case {case}: {sum_sq} vs {frobenius_sq}"
            );
            let area = norm3(cross3(f[0], f[1]));
            assert!(
                (svd.sigma[0] * svd.sigma[1] - area).abs() <= 1e-12 * scale * scale,
                "case {case}: {} vs {area}",
                svd.sigma[0] * svd.sigma[1]
            );
        }
    }

    #[test]
    fn svd_of_rotated_tangent_frames_has_unit_singular_values() {
        // F = [R t1  R t2] for a rotation R is a rigid motion of the rest
        // triangle: both singular values are exactly one.
        let rest = skewed_rest();
        let (sin, cos) = 0.9_f64.sin_cos();
        let axis = scale3([1.0, 2.0, -0.5], 1.0 / norm3([1.0, 2.0, -0.5]));
        let rotate = |v: [f64; 3]| {
            // Rodrigues' rotation formula about `axis` by 0.9 rad.
            add3(
                add3(scale3(v, cos), scale3(cross3(axis, v), sin)),
                scale3(axis, dot3(axis, v) * (1.0 - cos)),
            )
        };
        let f = [rotate(rest.t1), rotate(rest.t2)];
        let svd = Svd::new(f, &rest);
        assert_valid_svd(f, &svd);
        assert!((svd.sigma[0] - 1.0).abs() < ROUNDING, "{:?}", svd.sigma);
        assert!((svd.sigma[1] - 1.0).abs() < ROUNDING, "{:?}", svd.sigma);
    }

    #[test]
    fn svd_of_a_collapsed_triangle_regains_the_rest_orientation() {
        let rest = unit_rest();
        let normal = rest.normal;
        // A point (F = 0) and a segment inside the rest plane (rank one).
        for f in [
            [[0.0; 3]; 2],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, -1.5, 0.0], [0.0, -0.5, 0.0]],
        ] {
            let svd = Svd::new(f, &rest);
            assert_valid_svd(f, &svd);
            assert!(
                svd.sigma[1] < 1e-9,
                "second singular value {}",
                svd.sigma[1]
            );
            let orientation = dot3(cross3(svd.u[0], svd.u[1]), normal);
            assert!(
                (orientation - 1.0).abs() < ROUNDING,
                "u1 x u2 must follow the rest normal, got {orientation} for {f:?}"
            );
        }
        // A segment along the rest normal: any in-plane direction will do.
        let f = [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]];
        let svd = Svd::new(f, &rest);
        assert_valid_svd(f, &svd);
        assert!(dot3(svd.u[1], normal).abs() < ROUNDING);
    }

    #[test]
    fn deformation_gradient_of_the_rest_edges_is_the_tangent_frame() {
        let rest = skewed_rest();
        let q0 = [0.5, -0.2, 1.0];
        let q1 = [1.7, 0.1, 0.6];
        let q2 = [0.9, 1.2, 0.9];
        let f = rest.deformation_gradient(sub3(q1, q0), sub3(q2, q0));
        assert!(norm3(sub3(f[0], rest.t1)) < ROUNDING);
        assert!(norm3(sub3(f[1], rest.t2)) < ROUNDING);
        assert!((norm3(rest.t1) - 1.0).abs() < ROUNDING);
        assert!((norm3(rest.t2) - 1.0).abs() < ROUNDING);
        assert!(dot3(rest.t1, rest.t2).abs() < ROUNDING);
        assert!((norm3(rest.normal) - 1.0).abs() < ROUNDING);
        // Dm reproduces the rest edges.
        let e0 = rest.embed([rest.d11, 0.0]);
        let e1 = rest.embed([rest.d12, rest.d22]);
        assert!(norm3(sub3(e0, sub3(q1, q0))) < ROUNDING);
        assert!(norm3(sub3(e1, sub3(q2, q0))) < ROUNDING);
    }

    #[test]
    fn edge_changes_are_the_inverse_of_the_deformation_gradient() {
        let rest = skewed_rest();
        let f0 = [1.1, -0.3, 0.4];
        let f1 = [0.2, 0.9, -0.6];
        let delta_f = [[0.05, -0.1, 0.2], [-0.3, 0.02, 0.07]];
        let (change_0, change_1) = rest.edge_changes(delta_f);
        let before = rest.deformation_gradient(f0, f1);
        let after = rest.deformation_gradient(add3(f0, change_0), add3(f1, change_1));
        for ((b, a), d) in before.iter().zip(&after).zip(&delta_f) {
            assert!(norm3(sub3(sub3(*a, *b), *d)) < ROUNDING);
        }
    }

    #[test]
    fn rest_triangle_rejects_degenerate_triangles_only() {
        let origin = [0.0; 3];
        // Collinear, coincident and sliver triangles.
        assert!(RestTriangle::new(origin, [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]).is_none());
        assert!(RestTriangle::new(origin, origin, [0.0, 1.0, 0.0]).is_none());
        assert!(RestTriangle::new(origin, [1.0, 0.0, 0.0], origin).is_none());
        assert!(RestTriangle::new(origin, [1.0, 0.0, 0.0], [0.5, 1e-10, 0.0]).is_none());
        assert!(RestTriangle::new(origin, [1.0, 0.0, 0.0], [0.5, f64::NAN, 0.0]).is_none());
        // The threshold is relative: micrometre-sized triangles are fine.
        let tiny = RestTriangle::new(origin, [1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]);
        assert!(tiny.is_some());
        let huge = RestTriangle::new(origin, [1e6, 0.0, 0.0], [0.0, 1e6, 0.0]);
        assert!(huge.is_some());
    }

    #[test]
    fn free_vertices_get_exactly_the_requested_edge_changes() {
        let a = [0.3, -0.2, 0.5];
        let b = [-0.4, 0.1, 0.25];
        for weights in [
            [1.0, 1.0, 1.0],
            [0.5, 2.0, 4.0],
            [3.0, 0.25, 1.0],
            [0.0, 2.0, 0.5],
            [1.5, 0.0, 0.5],
            [2.0, 3.0, 0.0],
        ] {
            let [dp0, dp1, dp2] = distribute_edge_changes(weights, a, b);
            assert!(norm3(sub3(sub3(dp1, dp0), a)) < ROUNDING, "{weights:?}");
            assert!(norm3(sub3(sub3(dp2, dp0), b)) < ROUNDING, "{weights:?}");
            // With every vertex free the corrections carry no net momentum.
            if weights.iter().all(|w| *w > 0.0) {
                let momentum = [dp0, dp1, dp2]
                    .iter()
                    .zip(weights)
                    .fold([0.0; 3], |sum, (dp, w)| add3(sum, scale3(*dp, 1.0 / w)));
                assert!(norm3(momentum) < ROUNDING, "{weights:?}: {momentum:?}");
            }
            // Pinned vertices stay put.
            for (dp, w) in [dp0, dp1, dp2].iter().zip(weights) {
                if w == 0.0 {
                    assert_eq!(*dp, [0.0; 3]);
                }
            }
        }
    }

    #[test]
    fn a_single_free_vertex_takes_what_it_can() {
        let a = [0.3, -0.2, 0.5];
        let b = [-0.4, 0.1, 0.25];
        let zero = [0.0; 3];
        // Only p1 can change the first edge, only p2 the second one.
        assert_eq!(
            distribute_edge_changes([0.0, 2.0, 0.0], a, b),
            [zero, a, zero]
        );
        assert_eq!(
            distribute_edge_changes([0.0, 0.0, 0.5], a, b),
            [zero, zero, b]
        );
        // p0 moves both edges the opposite way: the least-squares compromise.
        let [dp0, dp1, dp2] = distribute_edge_changes([4.0, 0.0, 0.0], a, b);
        assert!(norm3(sub3(dp0, scale3(add3(a, b), -0.5))) < ROUNDING);
        assert_eq!((dp1, dp2), (zero, zero));
        // Nothing is free: nothing moves.  Negative and NaN weights are pinned.
        assert_eq!(distribute_edge_changes([0.0; 3], a, b), [zero; 3]);
        assert_eq!(
            distribute_edge_changes([-1.0, f64::NAN, 0.0], a, b),
            [zero; 3]
        );
    }

    #[test]
    fn stretch_limits_tolerate_bad_ranges() {
        assert_eq!(limit_stretch(2.0, 0.95, 1.05), 1.05);
        assert_eq!(limit_stretch(0.5, 0.95, 1.05), 0.95);
        assert_eq!(limit_stretch(1.0, 0.95, 1.05), 1.0);
        // The upper bound wins on an inverted range instead of panicking.
        assert_eq!(limit_stretch(1.0, 1.2, 1.1), 1.1);
        // A NaN bound does not limit; an infinite one neither.
        assert_eq!(limit_stretch(2.0, f64::NAN, f64::NAN), 2.0);
        assert_eq!(limit_stretch(2.0, 0.0, f64::INFINITY), 2.0);
    }
}
