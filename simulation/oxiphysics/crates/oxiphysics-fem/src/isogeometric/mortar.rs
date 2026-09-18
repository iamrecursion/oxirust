// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Mortar coupling of two NURBS patches along a shared interface edge.
//!
//! The interface of each patch is one of its four boundary edges
//! ([`PatchEdge`]).  A patch whose knot vector is clamped in the direction
//! normal to the edge has an edge that is *exactly* the NURBS curve formed by
//! the corresponding row (or column) of control points and weights, so the
//! trace space of the patch on its interface is spanned by the rational basis
//! functions of that curve.
//!
//! [`MortarPatchCoupling`] integrates the mortar (cross-mass) matrix
//!
//! ```text
//! D[a][b] = alpha * ∫_Γs R^s_a(x) R^m_b(π(x)) dΓ
//! ```
//!
//! over the slave interface `Γs`, where `R^s_a` are the rational basis
//! functions of the slave edge curve, `R^m_b` those of the master edge curve
//! and `π` is the closest-point projection onto the master edge curve.  The
//! integral is evaluated knot span by knot span of the slave edge with Gauss
//! quadrature, and every span is additionally cut at the projections of the
//! master knots, so that the integrand is smooth on every integration segment
//! even if the two discretisations do not match.

use super::functions::{basis_derivatives, dot3, find_knot_span, gauss_legendre_1d, norm3, sub3};
use super::types::{NurbsBasis, NurbsCurve, NurbsSurface};
use crate::{Error, Result};

/// One of the four boundary edges of a tensor-product NURBS surface patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchEdge {
    /// The edge `u = u_min`, parametrised by `v`.
    UMin,
    /// The edge `u = u_max`, parametrised by `v`.
    UMax,
    /// The edge `v = v_min`, parametrised by `u`.
    VMin,
    /// The edge `v = v_max`, parametrised by `u`.
    VMax,
}

/// Relative tolerance for comparing knot values.
const KNOT_TOLERANCE: f64 = 1e-14;

fn knots_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= KNOT_TOLERANCE * a.abs().max(b.abs()).max(1.0)
}

/// Check that the control net, weights and bases of `surface` are consistent.
fn validate_surface(surface: &NurbsSurface) -> Result<()> {
    let n_u = surface.control_net.len();
    if n_u == 0 || surface.control_net[0].is_empty() {
        return Err(Error::InvalidInput("surface has no control points".into()));
    }
    let n_v = surface.control_net[0].len();
    if surface.control_net.iter().any(|row| row.len() != n_v) {
        return Err(Error::InvalidInput(
            "control net rows have different lengths".into(),
        ));
    }
    if surface.weights.len() != n_u || surface.weights.iter().any(|row| row.len() != n_v) {
        return Err(Error::InvalidInput(
            "weights do not match the control net".into(),
        ));
    }
    if n_u != surface.u_basis.num_basis || n_v != surface.v_basis.num_basis {
        return Err(Error::InvalidInput(format!(
            "control net is {n_u} x {n_v} but the bases have {} x {} functions",
            surface.u_basis.num_basis, surface.v_basis.num_basis
        )));
    }
    if surface
        .weights
        .iter()
        .flatten()
        .any(|w| !w.is_finite() || *w <= 0.0)
    {
        return Err(Error::InvalidInput(
            "NURBS weights must be finite and positive".into(),
        ));
    }
    if surface
        .control_net
        .iter()
        .flatten()
        .any(|p| p.iter().any(|c| !c.is_finite()))
    {
        return Err(Error::InvalidInput("control points must be finite".into()));
    }
    for basis in [&surface.u_basis, &surface.v_basis] {
        if basis.knot.iter().any(|k| !k.is_finite()) || basis.knot.windows(2).any(|w| w[1] < w[0]) {
            return Err(Error::InvalidInput(
                "knot vectors must be finite and non-decreasing".into(),
            ));
        }
    }
    Ok(())
}

impl PatchEdge {
    /// `true` when the edge runs along the `v` direction (`u = const`).
    fn runs_along_v(self) -> bool {
        matches!(self, PatchEdge::UMin | PatchEdge::UMax)
    }

    /// `true` for the edges at the maximum parameter of the fixed direction.
    fn at_maximum(self) -> bool {
        matches!(self, PatchEdge::UMax | PatchEdge::VMax)
    }

    /// The boundary curve of `surface` along this edge.
    ///
    /// The knot vector of the surface in the direction normal to the edge must
    /// be clamped (end knot of multiplicity `p + 1`) at that edge, so that the
    /// edge is interpolated by the control points of the boundary row; the
    /// returned curve then coincides with the surface on the edge.  The curve
    /// carries the weights of the boundary row and the basis of the direction
    /// along the edge.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if the control net, weights or knot vectors are
    /// inconsistent or non-finite, if the surface is not clamped at the edge,
    /// or if the basis along the edge has degree zero or an empty parameter
    /// domain.
    pub fn boundary_curve(self, surface: &NurbsSurface) -> Result<NurbsCurve> {
        validate_surface(surface)?;
        let (fixed, free) = if self.runs_along_v() {
            (&surface.u_basis, &surface.v_basis)
        } else {
            (&surface.v_basis, &surface.u_basis)
        };
        let p = fixed.degree;
        let knots = &fixed.knot;
        let clamped = if self.at_maximum() {
            let end = knots[knots.len() - 1];
            knots[knots.len() - 1 - p..]
                .iter()
                .all(|&k| knots_equal(k, end))
        } else {
            knots[..=p].iter().all(|&k| knots_equal(k, knots[0]))
        };
        if !clamped {
            return Err(Error::InvalidInput(format!(
                "the knot vector is not clamped at the {self:?} edge, so the edge is not \
                 interpolated by the boundary control points"
            )));
        }
        if free.degree == 0 {
            return Err(Error::InvalidInput(
                "the basis along the interface must have degree >= 1".into(),
            ));
        }
        let (t0, t1) = free.domain();
        if t1 <= t0 {
            return Err(Error::InvalidInput(
                "the interface has an empty parameter domain".into(),
            ));
        }
        let index = if self.at_maximum() {
            fixed.num_basis - 1
        } else {
            0
        };
        let (points, weights): (Vec<[f64; 3]>, Vec<f64>) = if self.runs_along_v() {
            surface.control_net[index]
                .iter()
                .copied()
                .zip(surface.weights[index].iter().copied())
                .unzip()
        } else {
            surface
                .control_net
                .iter()
                .zip(surface.weights.iter())
                .map(|(points_row, weights_row)| (points_row[index], weights_row[index]))
                .unzip()
        };
        Ok(NurbsCurve::new(
            points,
            weights,
            free.degree,
            free.knot.clone(),
        ))
    }
}

/// Result of a closest-point projection onto an interface curve.
#[derive(Debug, Clone, Copy)]
struct Projection {
    /// Parameter of the closest point of the curve.
    t: f64,
    /// The closest point itself.
    point: [f64; 3],
    /// `true` if the point lies beyond the end of the curve, i.e. the closest
    /// point is an end point and the orthogonal projection falls outside.
    outside: bool,
}

/// Interface edge of a patch prepared for integration and projection.
#[derive(Debug, Clone)]
struct InterfaceTrace {
    curve: NurbsCurve,
    basis: NurbsBasis,
    domain: (f64, f64),
    /// Distinct knot values of the domain, ascending, including both end points.
    breaks: Vec<f64>,
    /// Parameters used to seed the closest-point search.
    sample_params: Vec<f64>,
    /// The curve points at `sample_params`.
    sample_points: Vec<[f64; 3]>,
    /// Chord-length estimate of the curve length.
    length: f64,
}

impl InterfaceTrace {
    fn new(surface: &NurbsSurface, edge: PatchEdge) -> Result<Self> {
        let curve = edge.boundary_curve(surface)?;
        let degree = curve.basis.degree;
        let (t0, t1) = curve.basis.domain();
        let tol = 1e-12 * (t1 - t0);
        let mut breaks = vec![t0];
        for &k in &curve.basis.knot {
            if k > t0 + tol && k < t1 - tol && k > breaks[breaks.len() - 1] + tol {
                breaks.push(k);
            }
        }
        breaks.push(t1);
        let per_span = 2 * degree + 6;
        let mut sample_params = Vec::with_capacity((breaks.len() - 1) * per_span + 1);
        for span in breaks.windows(2) {
            for s in 0..per_span {
                sample_params.push(span[0] + (span[1] - span[0]) * s as f64 / per_span as f64);
            }
        }
        sample_params.push(t1);
        let sample_points: Vec<[f64; 3]> =
            sample_params.iter().map(|&t| curve.point_at(t)).collect();
        let length = sample_points
            .windows(2)
            .map(|w| norm3(sub3(w[1], w[0])))
            .sum();
        let basis = NurbsBasis::new(degree, curve.basis.knot.clone(), curve.weights.clone());
        Ok(Self {
            curve,
            basis,
            domain: (t0, t1),
            breaks,
            sample_params,
            sample_points,
            length,
        })
    }

    /// Number of basis functions (control points) along the interface.
    fn num_basis(&self) -> usize {
        self.basis.num_basis()
    }

    /// Curve point and its first and second derivatives with respect to the
    /// curve parameter (rational derivative formulas).
    fn derivatives(&self, t: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let p = self.curve.basis.degree;
        let knots = &self.curve.basis.knot;
        let n = self.curve.control_points.len() - 1;
        let span = find_knot_span(n, p, t, knots);
        let ders = basis_derivatives(span, p, t, knots, 2);
        // Derivatives of the numerator A(t) = Σ w_i P_i N_i and of the weight
        // function W(t) = Σ w_i N_i; rows beyond the polynomial degree vanish.
        let mut a = [[0.0_f64; 3]; 3];
        let mut w = [0.0_f64; 3];
        for (order, row) in ders.iter().enumerate().take(3) {
            for (j, &value) in row.iter().enumerate().take(p + 1) {
                let index = span - p + j;
                let weight = self.curve.weights[index];
                let point = self.curve.control_points[index];
                for (component, a_c) in a[order].iter_mut().enumerate() {
                    *a_c += weight * point[component] * value;
                }
                w[order] += weight * value;
            }
        }
        let mut c0 = [0.0_f64; 3];
        let mut c1 = [0.0_f64; 3];
        let mut c2 = [0.0_f64; 3];
        for k in 0..3 {
            c0[k] = a[0][k] / w[0];
            c1[k] = (a[1][k] - w[1] * c0[k]) / w[0];
            c2[k] = (a[2][k] - 2.0 * w[1] * c1[k] - w[2] * c0[k]) / w[0];
        }
        (c0, c1, c2)
    }

    /// Gradient `(g, g')` of the half squared distance `|C(t) - x|² / 2`.
    fn distance_gradient(&self, t: f64, x: [f64; 3]) -> (f64, f64) {
        let (c, d1, d2) = self.derivatives(t);
        let r = sub3(c, x);
        (dot3(r, d1), dot3(d1, d1) + dot3(r, d2))
    }

    /// Closest-point projection of `x` onto the curve.
    ///
    /// The search is seeded with the nearest of a set of curve samples and
    /// refined by a safeguarded Newton iteration (bisection whenever a Newton
    /// step leaves the bracket) on the stationarity condition
    /// `(C(t) - x)·C'(t) = 0`.  If the distance increases monotonically
    /// towards an end of the curve, that end is returned and `outside` records
    /// whether `x` lies beyond it.
    fn project(&self, x: [f64; 3]) -> Projection {
        let (t0, t1) = self.domain;
        let last = self.sample_params.len() - 1;
        let mut best = 0;
        let mut best_distance = f64::INFINITY;
        for (k, point) in self.sample_points.iter().enumerate() {
            let r = sub3(*point, x);
            let distance = dot3(r, r);
            if distance < best_distance {
                best_distance = distance;
                best = k;
            }
        }
        let mut lo = self.sample_params[best.saturating_sub(1)];
        let mut hi = self.sample_params[(best + 1).min(last)];
        let outside_tolerance = 1e-9 * self.length;
        let (g_lo, _) = self.distance_gradient(lo, x);
        let (g_hi, _) = self.distance_gradient(hi, x);

        let end = |t: f64, g: f64| -> Projection {
            let (_, d1, _) = self.derivatives(t);
            let speed = norm3(d1);
            let beyond = if speed > 0.0 { g.abs() / speed } else { 0.0 };
            Projection {
                t,
                point: self.curve.point_at(t),
                outside: beyond > outside_tolerance,
            }
        };
        if best == 0 && g_lo >= 0.0 {
            // Distance grows from the start of the curve: x is behind it.
            return end(t0, g_lo);
        }
        if best == last && g_hi <= 0.0 {
            return end(t1, g_hi);
        }
        let mut t = if g_lo >= 0.0 {
            lo
        } else if g_hi <= 0.0 {
            hi
        } else {
            // Sign change g(lo) < 0 < g(hi): safeguarded Newton inside the bracket.
            let mut t = self.sample_params[best];
            for _ in 0..100 {
                let (g, gp) = self.distance_gradient(t, x);
                if g == 0.0 {
                    break;
                }
                if g < 0.0 {
                    lo = t;
                } else {
                    hi = t;
                }
                let newton = if gp > 0.0 { t - g / gp } else { f64::NAN };
                let next = if newton > lo && newton < hi {
                    newton
                } else {
                    0.5 * (lo + hi)
                };
                let converged = (next - t).abs() <= 4.0 * f64::EPSILON * t.abs().max(1.0);
                t = next;
                if converged {
                    break;
                }
            }
            t
        };
        t = t.clamp(t0, t1);
        Projection {
            t,
            point: self.curve.point_at(t),
            outside: false,
        }
    }
}

/// Parameters of the slave interface at which the integration is cut: the
/// distinct slave knots and the projections of the master knots onto the slave.
fn integration_breakpoints(slave: &InterfaceTrace, master: &InterfaceTrace) -> Vec<f64> {
    let (u0, u1) = slave.domain;
    let tol = 1e-12 * (u1 - u0);
    let mut cuts = slave.breaks.clone();
    for &tau in &master.breaks {
        let projection = slave.project(master.curve.point_at(tau));
        if !projection.outside && projection.t > u0 + tol && projection.t < u1 - tol {
            cuts.push(projection.t);
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|later, earlier| (*later - *earlier).abs() <= tol);
    cuts
}

/// Patch coupling for multi-patch IGA using the mortar method.
///
/// Computes the mortar (cross-mass) coupling matrix between two adjacent NURBS
/// patches that share an interface, enabling displacement and traction
/// continuity.  The interface of each patch is a boundary edge
/// ([`PatchEdge`]): the *slave* edge is integrated over and the *master* edge
/// is reached by closest-point projection, so the two edges may be discretised
/// independently (different degrees, knot vectors, orientations).
///
/// A single parameter value cannot describe the usual configuration in which
/// the top edge of one patch meets the bottom edge of the next, hence each
/// patch has its own edge.  [`new`](Self::new) uses the edge `v = v_min` of
/// both patches; [`with_interface`](Self::with_interface) selects other edges.
/// Each patch must have a clamped knot vector in the direction normal to its
/// interface edge (see [`PatchEdge::boundary_curve`]).
#[derive(Debug, Clone)]
pub struct MortarPatchCoupling {
    /// The slave patch.
    pub slave_surface: NurbsSurface,
    /// The master patch.
    pub master_surface: NurbsSurface,
    /// Number of Gauss points per integration segment.
    pub n_gauss: usize,
    /// Penalty or Lagrange multiplier coefficient; scales the coupling matrix.
    pub alpha: f64,
    /// Interface edge of the slave patch.
    pub slave_edge: PatchEdge,
    /// Interface edge of the master patch.
    pub master_edge: PatchEdge,
}

impl MortarPatchCoupling {
    /// Create a new mortar patch coupling along the edge `v = v_min` of both
    /// patches.
    pub fn new(
        slave_surface: NurbsSurface,
        master_surface: NurbsSurface,
        n_gauss: usize,
        alpha: f64,
    ) -> Self {
        MortarPatchCoupling {
            slave_surface,
            master_surface,
            n_gauss,
            alpha,
            slave_edge: PatchEdge::VMin,
            master_edge: PatchEdge::VMin,
        }
    }

    /// Select the interface edges of the slave and master patch.
    pub fn with_interface(mut self, slave_edge: PatchEdge, master_edge: PatchEdge) -> Self {
        self.slave_edge = slave_edge;
        self.master_edge = master_edge;
        self
    }

    fn traces(&self) -> Result<(InterfaceTrace, InterfaceTrace)> {
        if self.n_gauss == 0 {
            return Err(Error::InvalidInput(
                "mortar integration needs at least one Gauss point".into(),
            ));
        }
        if !self.alpha.is_finite() {
            return Err(Error::InvalidInput(
                "the mortar coefficient alpha must be finite".into(),
            ));
        }
        Ok((
            InterfaceTrace::new(&self.slave_surface, self.slave_edge)?,
            InterfaceTrace::new(&self.master_surface, self.master_edge)?,
        ))
    }

    /// Number of slave and master interface DOFs `(n_slave, n_master)`: the
    /// numbers of control points along the two interface edges, i.e. the
    /// row and column counts of the coupling matrix.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if an interface edge cannot be extracted (see
    /// [`PatchEdge::boundary_curve`]).
    pub fn interface_dofs(&self) -> Result<(usize, usize)> {
        let slave = self.slave_edge.boundary_curve(&self.slave_surface)?;
        let master = self.master_edge.boundary_curve(&self.master_surface)?;
        Ok((slave.control_points.len(), master.control_points.len()))
    }

    /// Compute the L2 projection mortar coupling matrix.
    ///
    /// Returns the row-major matrix `D` (slave-master DOF coupling) of size
    /// `n_slave x n_master` (see [`interface_dofs`](Self::interface_dofs)) with
    ///
    /// `D[a * n_master + b] = alpha * ∫ R^s_a(x) R^m_b(π(x)) dΓ`,
    ///
    /// the integral running over the part of the slave interface that is
    /// covered by the master interface.  For two conforming copies of one
    /// patch `D` is `alpha` times the boundary mass matrix of the interface.
    /// Slave and master may be parametrised in opposite directions.
    ///
    /// Each slave knot span is integrated with `n_gauss` Gauss points, after it
    /// has been cut at the projections of the master knots.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `n_gauss` is zero, `alpha` is not finite, or
    /// an interface edge cannot be extracted (see [`PatchEdge::boundary_curve`]).
    pub fn try_coupling_matrix(&self) -> Result<Vec<f64>> {
        let (slave, master) = self.traces()?;
        let cuts = integration_breakpoints(&slave, &master);
        Ok(self.cross_mass(&slave, &master, &cuts))
    }

    /// Integrate the cross-mass matrix over the slave segments delimited by
    /// consecutive entries of `cuts` (ascending slave parameters).
    fn cross_mass(
        &self,
        slave: &InterfaceTrace,
        master: &InterfaceTrace,
        cuts: &[f64],
    ) -> Vec<f64> {
        let n_master = master.num_basis();
        let ps = slave.curve.basis.degree;
        let pm = master.curve.basis.degree;
        let (gauss_points, gauss_weights) = gauss_legendre_1d(self.n_gauss);
        let mut d_mat = vec![0.0_f64; slave.num_basis() * n_master];
        for segment in cuts.windows(2) {
            let (start, end) = (segment[0], segment[1]);
            if end <= start {
                continue;
            }
            let half = 0.5 * (end - start);
            let mid = 0.5 * (start + end);
            for (xi, weight) in gauss_points.iter().zip(gauss_weights.iter()) {
                let u = mid + half * xi;
                let (x, dx, _) = slave.derivatives(u);
                let projection = master.project(x);
                if projection.outside {
                    continue;
                }
                let w = self.alpha * weight * half * norm3(dx);
                let (s_span, r_slave, _) = slave.basis.evaluate_rational(u);
                let (m_span, r_master, _) = master.basis.evaluate_rational(projection.t);
                for (a, &ra) in r_slave.iter().enumerate() {
                    let row = s_span - ps + a;
                    for (b, &rb) in r_master.iter().enumerate() {
                        let col = m_span - pm + b;
                        d_mat[row * n_master + col] += w * ra * rb;
                    }
                }
            }
        }
        d_mat
    }

    /// Compute the L2 projection mortar coupling matrix.
    ///
    /// Infallible form of [`try_coupling_matrix`](Self::try_coupling_matrix):
    /// an invalid configuration yields an *empty* vector (a valid matrix always
    /// has at least one entry).
    pub fn coupling_matrix(&self) -> Vec<f64> {
        self.try_coupling_matrix().unwrap_or_default()
    }

    /// Compute the interface gap (displacement jump) at a set of sample
    /// parameters.
    ///
    /// The samples are uniform in the parameter of the slave interface edge;
    /// each entry is `(t, x_s - π(x_s))` where `t` in `[0, 1]` is the
    /// normalised slave parameter, `x_s` the slave point and `π(x_s)` its
    /// closest point on the master interface edge (the end point of the master
    /// edge if `x_s` projects beyond it).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if an interface edge cannot be extracted (see
    /// [`PatchEdge::boundary_curve`]).
    pub fn try_interface_gap(&self, n_samples: usize) -> Result<Vec<(f64, [f64; 3])>> {
        let slave = InterfaceTrace::new(&self.slave_surface, self.slave_edge)?;
        let master = InterfaceTrace::new(&self.master_surface, self.master_edge)?;
        let (u0, u1) = slave.domain;
        let mut gaps = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / (n_samples - 1).max(1) as f64;
            let xs = slave.curve.point_at(u0 + (u1 - u0) * t);
            let projection = master.project(xs);
            gaps.push((t, sub3(xs, projection.point)));
        }
        Ok(gaps)
    }

    /// Compute the interface gap (displacement jump) at a set of sample
    /// parameters.
    ///
    /// Infallible form of [`try_interface_gap`](Self::try_interface_gap): an
    /// invalid configuration yields an empty vector.
    pub fn interface_gap(&self, n_samples: usize) -> Vec<(f64, [f64; 3])> {
        self.try_interface_gap(n_samples).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isogeometric::functions::basis_functions;

    /// Clamped knot vector of one linear element.
    const UNIT: [f64; 4] = [0.0, 0.0, 1.0, 1.0];

    /// Greville abscissae of a clamped knot vector.
    fn greville(degree: usize, knots: &[f64]) -> Vec<f64> {
        let n = knots.len() - degree - 1;
        (0..n)
            .map(|i| knots[i + 1..=i + degree].iter().sum::<f64>() / degree as f64)
            .collect()
    }

    /// Straight strip of unit height.  Its edge `v = 0` is the segment
    /// `x in [x0, x0 + length]` at height `y0` (z = 0); its edge `v = 1` is the
    /// same segment one unit higher.  The control points sit at the Greville
    /// abscissae, so `x` is a linear function of the `u` parameter.
    fn strip(degree: usize, knots: &[f64], x0: f64, length: f64, y0: f64) -> NurbsSurface {
        let net: Vec<Vec<[f64; 3]>> = greville(degree, knots)
            .iter()
            .map(|g| {
                let x = x0 + length * g;
                vec![[x, y0, 0.0], [x, y0 + 1.0, 0.0]]
            })
            .collect();
        let weights = vec![vec![1.0, 1.0]; net.len()];
        NurbsSurface::new(net, weights, degree, 1, knots.to_vec(), UNIT.to_vec())
    }

    /// Like [`strip`] but with `u` running from `x0 + length` down to `x0`.
    fn reversed_strip(degree: usize, knots: &[f64], x0: f64, length: f64, y0: f64) -> NurbsSurface {
        let net: Vec<Vec<[f64; 3]>> = greville(degree, knots)
            .iter()
            .map(|g| {
                let x = x0 + length * (1.0 - g);
                vec![[x, y0, 0.0], [x, y0 + 1.0, 0.0]]
            })
            .collect();
        let weights = vec![vec![1.0, 1.0]; net.len()];
        NurbsSurface::new(net, weights, degree, 1, knots.to_vec(), UNIT.to_vec())
    }

    /// The patch with its `u` and `v` roles exchanged.
    fn transposed(surface: &NurbsSurface) -> NurbsSurface {
        let n_u = surface.control_net.len();
        let n_v = surface.control_net[0].len();
        let net = (0..n_v)
            .map(|j| (0..n_u).map(|i| surface.control_net[i][j]).collect())
            .collect();
        let weights = (0..n_v)
            .map(|j| (0..n_u).map(|i| surface.weights[i][j]).collect())
            .collect();
        NurbsSurface::new(
            net,
            weights,
            surface.v_basis.degree,
            surface.u_basis.degree,
            surface.v_basis.knot.clone(),
            surface.u_basis.knot.clone(),
        )
    }

    /// Index of the first non-zero basis function and the `degree + 1` values.
    fn basis_at(degree: usize, knots: &[f64], t: f64) -> (usize, Vec<f64>) {
        let n = knots.len() - degree - 1;
        let span = find_knot_span(n - 1, degree, t, knots);
        (span - degree, basis_functions(span, degree, t, knots))
    }

    /// `length * ∫_0^1 N^s_a(u) N^m_b(u) du`, row-major, by exact Gauss
    /// quadrature on the union of the knot spans.  This is the cross-mass
    /// matrix of two *straight* edges that are both parametrised linearly in
    /// the arc length, and does not involve any projection.
    fn cross_mass_reference(
        degree_s: usize,
        knots_s: &[f64],
        degree_m: usize,
        knots_m: &[f64],
        length: f64,
    ) -> Vec<f64> {
        let n_s = knots_s.len() - degree_s - 1;
        let n_m = knots_m.len() - degree_m - 1;
        let mut cuts: Vec<f64> = knots_s.iter().chain(knots_m.iter()).copied().collect();
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
        let (points, weights) = gauss_legendre_1d(8);
        let mut d = vec![0.0; n_s * n_m];
        for segment in cuts.windows(2) {
            let half = 0.5 * (segment[1] - segment[0]);
            let mid = 0.5 * (segment[1] + segment[0]);
            for (xi, w) in points.iter().zip(weights.iter()) {
                let t = mid + half * xi;
                let (i0, vs) = basis_at(degree_s, knots_s, t);
                let (j0, vm) = basis_at(degree_m, knots_m, t);
                for (a, va) in vs.iter().enumerate() {
                    for (b, vb) in vm.iter().enumerate() {
                        d[(i0 + a) * n_m + j0 + b] += length * half * w * va * vb;
                    }
                }
            }
        }
        d
    }

    fn assert_matrices_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: size mismatch");
        for (k, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "{what}: entry {k} is {a} but expected {e}"
            );
        }
    }

    fn quarter_circle_surface() -> NurbsSurface {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let net: Vec<Vec<[f64; 3]>> = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|p| vec![[p[0], p[1], 0.0], [p[0], p[1], 1.0]])
            .collect();
        let weights = vec![vec![1.0, 1.0], vec![w, w], vec![1.0, 1.0]];
        NurbsSurface::new(
            net,
            weights,
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            UNIT.to_vec(),
        )
    }

    #[test]
    fn identical_linear_patches_give_the_boundary_mass_matrix() {
        // Edge of length 2: the linear boundary mass matrix is L [[1/3, 1/6], [1/6, 1/3]].
        let s = strip(1, &UNIT, 0.0, 2.0, 0.0);
        let coupling = MortarPatchCoupling::new(s.clone(), s.clone(), 3, 1.0);
        assert_eq!(coupling.interface_dofs().expect("dofs"), (2, 2));
        let d = coupling.coupling_matrix();
        assert_matrices_close(
            &d,
            &[2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
            1e-13,
            "linear mass",
        );
        // alpha scales the matrix.
        let scaled = MortarPatchCoupling::new(s.clone(), s, 3, 2.5).coupling_matrix();
        let expected: Vec<f64> = d.iter().map(|v| 2.5 * v).collect();
        assert_matrices_close(&scaled, &expected, 1e-13, "alpha scaling");
    }

    #[test]
    fn stacked_patches_couple_slave_top_edge_to_master_bottom_edge() {
        let slave = strip(1, &UNIT, 0.0, 1.0, 0.0); // occupies y in [0, 1]
        let master = strip(1, &UNIT, 0.0, 1.0, 1.0); // occupies y in [1, 2]
        let stacked = MortarPatchCoupling::new(slave.clone(), master.clone(), 3, 1.0)
            .with_interface(PatchEdge::VMax, PatchEdge::VMin);
        assert_matrices_close(
            &stacked.coupling_matrix(),
            &[1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0],
            1e-13,
            "stacked mass",
        );
        for (_, gap) in stacked.interface_gap(5) {
            assert!(norm3(gap) < 1e-12, "the shared edge has a gap {gap:?}");
        }
        // The default (both patches at v_min) compares y = 0 with y = 1.
        let default = MortarPatchCoupling::new(slave, master, 3, 1.0);
        for (_, gap) in default.interface_gap(4) {
            assert!(norm3(sub3(gap, [0.0, -1.0, 0.0])) < 1e-12, "gap {gap:?}");
        }
    }

    #[test]
    fn opposite_parametrisation_of_the_master_reverses_its_columns() {
        let slave = strip(1, &UNIT, 0.0, 1.0, 0.0);
        let master = reversed_strip(1, &UNIT, 0.0, 1.0, 0.0);
        let d = MortarPatchCoupling::new(slave, master, 3, 1.0).coupling_matrix();
        // Master basis function 0 sits at x = 1, function 1 at x = 0.
        assert_matrices_close(
            &d,
            &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            1e-13,
            "reversed master",
        );
    }

    #[test]
    fn conforming_quadratic_patches_match_independent_quadrature() {
        let knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let length = 3.0;
        let s = strip(2, &knots, 1.0, length, 0.0);
        let d = MortarPatchCoupling::new(s.clone(), s, 3, 1.0).coupling_matrix();
        let expected = cross_mass_reference(2, &knots, 2, &knots, length);
        assert_eq!(d.len(), 16);
        assert_matrices_close(&d, &expected, 1e-12, "quadratic boundary mass matrix");
        // Symmetric, and the entries sum to the interface length.
        for a in 0..4 {
            for b in 0..4 {
                assert!((d[a * 4 + b] - d[b * 4 + a]).abs() < 1e-12);
            }
        }
        assert!((d.iter().sum::<f64>() - length).abs() < 1e-12);
    }

    #[test]
    fn non_conforming_meshes_give_the_exact_cross_mass_matrix() {
        // Slave: 2 linear elements; master: 3 quadratic elements of the same
        // straight edge.  Without cutting the slave spans at the master knots
        // the integrand would be non-smooth inside a slave span.
        let knots_s = [0.0, 0.0, 0.5, 1.0, 1.0];
        let knots_m = [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0];
        let length = 1.7;
        let slave = strip(1, &knots_s, 0.25, length, 0.0);
        let master = strip(2, &knots_m, 0.25, length, 0.0);
        let coupling = MortarPatchCoupling::new(slave, master, 4, 1.0);
        assert_eq!(coupling.interface_dofs().expect("dofs"), (3, 5));
        let d = coupling.coupling_matrix();
        let expected = cross_mass_reference(1, &knots_s, 2, &knots_m, length);
        assert_matrices_close(&d, &expected, 1e-12, "non-conforming cross mass");
    }

    #[test]
    fn linear_field_is_transferred_exactly_between_non_conforming_meshes() {
        // For a linear field x, the master interpolant equals x, hence
        // D X_m = M_ss X_s with the slave boundary mass matrix M_ss.
        let knots_s = [0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0];
        let knots_m = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0];
        let length = 2.0;
        let slave = strip(2, &knots_s, 0.0, length, 0.0);
        let master = strip(1, &knots_m, 0.0, length, 0.0);
        let d = MortarPatchCoupling::new(slave.clone(), master, 4, 1.0).coupling_matrix();
        let m_ss = MortarPatchCoupling::new(slave.clone(), slave, 4, 1.0).coupling_matrix();
        let x_s: Vec<f64> = greville(2, &knots_s).iter().map(|g| g * length).collect();
        let x_m: Vec<f64> = greville(1, &knots_m).iter().map(|g| g * length).collect();
        let n_s = x_s.len();
        let n_m = x_m.len();
        assert_eq!(d.len(), n_s * n_m);
        for a in 0..n_s {
            let transferred: f64 = (0..n_m).map(|b| d[a * n_m + b] * x_m[b]).sum();
            let reference: f64 = (0..n_s).map(|c| m_ss[a * n_s + c] * x_s[c]).sum();
            assert!(
                (transferred - reference).abs() < 1e-12,
                "row {a}: {transferred} vs {reference}"
            );
        }
        // Constants are transferred as well: row sums of D and M_ss agree.
        for a in 0..n_s {
            let d_row: f64 = (0..n_m).map(|b| d[a * n_m + b]).sum();
            let m_row: f64 = (0..n_s).map(|c| m_ss[a * n_s + c]).sum();
            assert!((d_row - m_row).abs() < 1e-12);
        }
    }

    #[test]
    fn partial_overlap_integrates_only_the_covered_part() {
        let knots = [0.0, 0.0, 0.5, 1.0, 1.0];
        let slave = strip(1, &knots, 0.0, 1.0, 0.0); // x in [0, 1]
        for &(x0, lo, hi) in &[(0.4, 0.4, 1.0), (-0.4, 0.0, 0.6)] {
            let master = strip(1, &knots, x0, 1.0, 0.0);
            let d = MortarPatchCoupling::new(slave.clone(), master, 4, 1.0).coupling_matrix();
            // Row sums are ∫ N^s_a dx over the overlap [lo, hi].
            let (points, weights) = gauss_legendre_1d(6);
            let mut cuts = [lo, 0.5, hi];
            cuts.sort_by(f64::total_cmp);
            let mut expected = [0.0_f64; 3];
            for segment in cuts.windows(2) {
                if segment[1] <= segment[0] {
                    continue;
                }
                let half = 0.5 * (segment[1] - segment[0]);
                let mid = 0.5 * (segment[1] + segment[0]);
                for (xi, w) in points.iter().zip(weights.iter()) {
                    let (i0, vals) = basis_at(1, &knots, mid + half * xi);
                    for (a, v) in vals.iter().enumerate() {
                        expected[i0 + a] += half * w * v;
                    }
                }
            }
            for (a, e) in expected.iter().enumerate() {
                let row: f64 = (0..3).map(|b| d[a * 3 + b]).sum();
                assert!(
                    (row - e).abs() < 1e-12,
                    "master offset {x0}: row {a} sums to {row}, overlap integral {e}"
                );
            }
            assert!((d.iter().sum::<f64>() - (hi - lo)).abs() < 1e-12);
        }
    }

    #[test]
    fn rational_quarter_circle_interface_is_integrated_exactly() {
        let circle = quarter_circle_surface();
        let coupling = MortarPatchCoupling::new(circle.clone(), circle, 12, 1.0);
        let d = coupling.coupling_matrix();
        assert_eq!(d.len(), 9);
        // Independent reference: composite Gauss quadrature of R_a R_b |C'|.
        let curve = coupling
            .slave_edge
            .boundary_curve(&coupling.slave_surface)
            .expect("edge");
        let basis = NurbsBasis::new(2, curve.basis.knot.clone(), curve.weights.clone());
        let (points, weights) = gauss_legendre_1d(6);
        let panels = 50;
        let mut expected = vec![0.0; 9];
        for panel in 0..panels {
            let a = panel as f64 / panels as f64;
            let b = (panel + 1) as f64 / panels as f64;
            for (xi, w) in points.iter().zip(weights.iter()) {
                let t = 0.5 * (a + b) + 0.5 * (b - a) * xi;
                let speed = norm3(curve.derivative_at(t));
                let (span, r, _) = basis.evaluate_rational(t);
                for (i, ri) in r.iter().enumerate() {
                    for (j, rj) in r.iter().enumerate() {
                        expected[(span - 2 + i) * 3 + span - 2 + j] +=
                            0.5 * (b - a) * w * speed * ri * rj;
                    }
                }
            }
        }
        assert_matrices_close(&d, &expected, 1e-9, "circle mass matrix");
        assert!(
            (d.iter().sum::<f64>() - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "the entries must sum to the arc length pi/2"
        );
    }

    #[test]
    fn u_edges_match_the_v_edges_of_the_transposed_patch() {
        let knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let slave = strip(2, &knots, 0.0, 1.0, 0.0);
        let master = strip(1, &[0.0, 0.0, 0.3, 1.0, 1.0], 0.0, 1.0, 1.0);
        let reference = MortarPatchCoupling::new(slave.clone(), master.clone(), 4, 1.0)
            .with_interface(PatchEdge::VMax, PatchEdge::VMin)
            .coupling_matrix();
        let transposed_coupling =
            MortarPatchCoupling::new(transposed(&slave), transposed(&master), 4, 1.0)
                .with_interface(PatchEdge::UMax, PatchEdge::UMin);
        assert_matrices_close(
            &transposed_coupling.coupling_matrix(),
            &reference,
            1e-13,
            "transposed patches",
        );
        for (_, gap) in transposed_coupling.interface_gap(7) {
            assert!(norm3(gap) < 1e-12);
        }
    }

    #[test]
    fn invalid_configurations_are_reported_not_ignored() {
        let good = strip(1, &UNIT, 0.0, 1.0, 0.0);
        // n_gauss = 0 and a non-finite coefficient.
        assert!(matches!(
            MortarPatchCoupling::new(good.clone(), good.clone(), 0, 1.0).try_coupling_matrix(),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            MortarPatchCoupling::new(good.clone(), good.clone(), 3, f64::NAN).try_coupling_matrix(),
            Err(Error::InvalidInput(_))
        ));
        // A patch that is not clamped in v has no interpolated v_min / v_max edge.
        let unclamped = NurbsSurface::new(
            vec![
                vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            1,
            1,
            UNIT.to_vec(),
            vec![0.0, 1.0, 2.0, 3.0],
        );
        let bad = MortarPatchCoupling::new(good.clone(), unclamped.clone(), 3, 1.0);
        assert!(matches!(
            bad.try_coupling_matrix(),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            bad.try_interface_gap(3),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(bad.interface_dofs(), Err(Error::InvalidInput(_))));
        assert!(bad.coupling_matrix().is_empty());
        assert!(bad.interface_gap(3).is_empty());
        // ... but its u_min edge is clamped and usable.
        assert!(PatchEdge::UMin.boundary_curve(&unclamped).is_ok());
        // Inconsistent control net.
        let mut ragged = good.clone();
        ragged.control_net[1].pop();
        assert!(matches!(
            PatchEdge::VMin.boundary_curve(&ragged),
            Err(Error::InvalidInput(_))
        ));
        let mut bad_weight = good.clone();
        bad_weight.weights[0][0] = 0.0;
        assert!(matches!(
            PatchEdge::VMin.boundary_curve(&bad_weight),
            Err(Error::InvalidInput(_))
        ));
        let mut too_few_rows = good;
        too_few_rows.control_net.pop();
        too_few_rows.weights.pop();
        assert!(matches!(
            PatchEdge::VMin.boundary_curve(&too_few_rows),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn gap_is_measured_to_the_closest_master_point_for_any_parametrisation() {
        // Different degree and knot vector, same straight edge, lifted by 0.25 in z.
        let slave = strip(1, &[0.0, 0.0, 0.5, 1.0, 1.0], 0.0, 1.0, 0.0);
        let mut master = strip(2, &[0.0, 0.0, 0.0, 0.2, 1.0, 1.0, 1.0], 0.0, 1.0, 0.0);
        for row in master.control_net.iter_mut() {
            for point in row.iter_mut() {
                point[2] += 0.25;
            }
        }
        let gaps = MortarPatchCoupling::new(slave, master, 3, 1.0).interface_gap(9);
        assert_eq!(gaps.len(), 9);
        for (k, (t, gap)) in gaps.iter().enumerate() {
            assert!((t - k as f64 / 8.0).abs() < 1e-15);
            assert!(
                norm3(sub3(*gap, [0.0, 0.0, -0.25])) < 1e-9,
                "gap at t = {t}: {gap:?}"
            );
        }
        // Edge cases of the sample count.
        let s = strip(1, &UNIT, 0.0, 1.0, 0.0);
        let coupling = MortarPatchCoupling::new(s.clone(), s, 3, 1.0);
        assert!(coupling.interface_gap(0).is_empty());
        let single = coupling.interface_gap(1);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].0, 0.0);
    }

    #[test]
    fn gap_beyond_the_master_end_is_the_distance_to_its_end_point() {
        let slave = strip(1, &UNIT, 0.0, 2.0, 0.0);
        let master = strip(1, &UNIT, 0.0, 1.0, 0.0);
        let gaps = MortarPatchCoupling::new(slave, master, 3, 1.0).interface_gap(5);
        let expected_x = [0.0, 0.0, 0.0, 0.5, 1.0];
        for ((_, gap), ex) in gaps.iter().zip(expected_x.iter()) {
            assert!(norm3(sub3(*gap, [*ex, 0.0, 0.0])) < 1e-12, "gap {gap:?}");
        }
    }

    #[test]
    fn boundary_curve_reproduces_the_surface_on_every_edge() {
        let surface = quarter_circle_surface();
        let (u0, u1) = surface.u_basis.domain();
        let (v0, v1) = surface.v_basis.domain();
        for k in 0..=10 {
            let s = k as f64 / 10.0;
            let u = u0 + (u1 - u0) * s;
            let v = v0 + (v1 - v0) * s;
            let cases = [
                (PatchEdge::VMin, surface.point_at(u, v0), u),
                (PatchEdge::VMax, surface.point_at(u, v1), u),
                (PatchEdge::UMin, surface.point_at(u0, v), v),
                (PatchEdge::UMax, surface.point_at(u1, v), v),
            ];
            for (edge, expected, t) in cases {
                let curve = edge.boundary_curve(&surface).expect("clamped patch");
                let point = curve.point_at(t);
                assert!(
                    norm3(sub3(point, expected)) < 1e-12,
                    "{edge:?} at {t}: {point:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn analytic_derivatives_agree_with_the_curve_derivative_and_finite_differences() {
        for surface in [
            quarter_circle_surface(),
            strip(
                3,
                &[0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0, 1.0],
                0.0,
                2.0,
                0.0,
            ),
        ] {
            let trace = InterfaceTrace::new(&surface, PatchEdge::VMin).expect("trace");
            for &t in &[0.05, 0.2, 0.45, 0.8, 0.95] {
                let (c, d1, d2) = trace.derivatives(t);
                assert!(norm3(sub3(c, trace.curve.point_at(t))) < 1e-13);
                assert!(norm3(sub3(d1, trace.curve.derivative_at(t))) < 1e-11);
                let h = 1e-6;
                let fd = [0, 1, 2].map(|k| {
                    (trace.curve.derivative_at(t + h)[k] - trace.curve.derivative_at(t - h)[k])
                        / (2.0 * h)
                });
                assert!(norm3(sub3(d2, fd)) < 1e-5, "t = {t}: {d2:?} vs {fd:?}");
            }
        }
    }

    #[test]
    fn projection_finds_the_closest_point_of_a_curved_interface() {
        let trace = InterfaceTrace::new(&quarter_circle_surface(), PatchEdge::VMin).expect("trace");
        for &radius in &[0.3, 1.0, 2.5] {
            for k in 1..20 {
                let phi = 0.5 * std::f64::consts::PI * k as f64 / 20.0;
                let x = [radius * phi.cos(), radius * phi.sin(), 0.4];
                let projection = trace.project(x);
                assert!(!projection.outside);
                let expected = [phi.cos(), phi.sin(), 0.0];
                assert!(
                    norm3(sub3(projection.point, expected)) < 1e-12,
                    "r = {radius}, phi = {phi}: {:?}",
                    projection.point
                );
            }
        }
        // Beyond either end of the arc the end point is returned and flagged.
        for &(phi, end) in &[(-0.3_f64, [1.0, 0.0, 0.0]), (1.9, [0.0, 1.0, 0.0])] {
            let projection = trace.project([phi.cos(), phi.sin(), 0.0]);
            assert!(projection.outside, "phi = {phi} is off the arc");
            assert!(norm3(sub3(projection.point, end)) < 1e-12);
        }
        // The end points themselves are on the arc.
        assert!(!trace.project([1.0, 0.0, 0.0]).outside);
        assert!(!trace.project([0.0, 1.0, 0.0]).outside);
    }

    #[test]
    fn projection_beats_dense_sampling_on_a_wiggly_cubic_edge() {
        let knots = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let heights = [0.0, 0.8, -0.6, 0.9, -0.7, 0.5, 0.0];
        let n = heights.len();
        let net: Vec<Vec<[f64; 3]>> = (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1) as f64;
                vec![[x, heights[i], 0.0], [x, heights[i], 1.0]]
            })
            .collect();
        let surface = NurbsSurface::new(
            net,
            vec![vec![1.0, 1.0]; n],
            3,
            1,
            knots.to_vec(),
            UNIT.to_vec(),
        );
        let trace = InterfaceTrace::new(&surface, PatchEdge::VMin).expect("trace");
        let dense: Vec<[f64; 3]> = (0..=20_000)
            .map(|k| trace.curve.point_at(k as f64 / 20_000.0))
            .collect();
        for ix in 0..12 {
            for iy in 0..8 {
                let x = [ix as f64 / 11.0, -1.0 + 2.0 * iy as f64 / 7.0, 0.3];
                let projection = trace.project(x);
                let found = norm3(sub3(projection.point, x));
                let sampled = dense
                    .iter()
                    .map(|p| norm3(sub3(*p, x)))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    found <= sampled + 1e-12,
                    "{x:?}: {found} vs sampled {sampled}"
                );
                assert!(
                    found >= sampled - 1e-6,
                    "{x:?}: {found} vs sampled {sampled}"
                );
                // At an interior solution the residual is orthogonal to the tangent.
                if !projection.outside && projection.t > 1e-9 && projection.t < 1.0 - 1e-9 {
                    let (_, d1, _) = trace.derivatives(projection.t);
                    let residual = sub3(projection.point, x);
                    assert!(
                        dot3(residual, d1).abs() < 1e-9 * (1.0 + norm3(d1) * found),
                        "{x:?}: not orthogonal at t = {}",
                        projection.t
                    );
                }
            }
        }
    }

    /// Extrude an edge curve one unit in z into a patch whose `v = v_min` edge
    /// is that curve.
    fn extruded(curve: &NurbsCurve) -> NurbsSurface {
        let net: Vec<Vec<[f64; 3]>> = curve
            .control_points
            .iter()
            .map(|p| vec![*p, [p[0], p[1], p[2] + 1.0]])
            .collect();
        let weights = curve.weights.iter().map(|&w| vec![w, w]).collect();
        NurbsSurface::new(
            net,
            weights,
            curve.basis.degree,
            1,
            curve.basis.knot.clone(),
            UNIT.to_vec(),
        )
    }

    /// `∫ R_i(t) |C'(t)| dt` of every rational basis function of `curve`, by
    /// composite Gauss quadrature on its knot spans.
    fn basis_arc_integrals(curve: &NurbsCurve) -> Vec<f64> {
        let basis = NurbsBasis::new(
            curve.basis.degree,
            curve.basis.knot.clone(),
            curve.weights.clone(),
        );
        let p = curve.basis.degree;
        let (points, weights) = gauss_legendre_1d(10);
        let mut integrals = vec![0.0; curve.control_points.len()];
        let (t0, t1) = curve.basis.domain();
        let mut cuts: Vec<f64> = curve
            .basis
            .knot
            .iter()
            .copied()
            .filter(|k| *k >= t0 && *k <= t1)
            .collect();
        cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
        for span in cuts.windows(2) {
            let half = 0.5 * (span[1] - span[0]);
            let mid = 0.5 * (span[1] + span[0]);
            for (xi, w) in points.iter().zip(weights.iter()) {
                let t = mid + half * xi;
                let speed = norm3(curve.derivative_at(t));
                let (index, values, _) = basis.evaluate_rational(t);
                for (j, value) in values.iter().enumerate() {
                    integrals[index - p + j] += half * w * speed * value;
                }
            }
        }
        integrals
    }

    #[test]
    fn curved_non_conforming_discretisations_have_matching_basis_integrals() {
        // The same quarter circle discretised as a single rational quadratic
        // span and, after knot insertion and degree elevation, as three
        // rational cubic spans.  Since both partitions of unity sum to one,
        //   sum_b D[a][b] = ∫ R^s_a ds   and   sum_a D[a][b] = ∫ R^m_b ds,
        // where the right-hand sides involve only one patch.  The second identity
        // fails if the slave spans are not cut at the projected master knots.
        let coarse_surface = quarter_circle_surface();
        let coarse_curve = PatchEdge::VMin
            .boundary_curve(&coarse_surface)
            .expect("edge");
        let refined =
            crate::isogeometric::KnotRefinement::h_refine_curve(&coarse_curve, &[0.3, 0.7]);
        let fine_curve = crate::isogeometric::DegreeElevation::elevate(&refined, 1);
        assert_eq!(fine_curve.basis.degree, 3);
        let fine_surface = extruded(&fine_curve);
        let coarse_integrals = basis_arc_integrals(&coarse_curve);
        let fine_integrals = basis_arc_integrals(&fine_curve);
        // Largest deviation of the column sums of `d` from the master integrals.
        let column_error = |d: &[f64], n_s: usize, n_m: usize, integrals: &[f64]| -> f64 {
            (0..n_m)
                .map(|b| {
                    let column: f64 = (0..n_s).map(|a| d[a * n_m + b]).sum();
                    (column - integrals[b]).abs()
                })
                .fold(0.0, f64::max)
        };
        // Without the cuts the single coarse slave span contains the kinks of the
        // fine master basis (at the projections of its knots 0.3 and 0.7).
        let uncut_coupling =
            MortarPatchCoupling::new(coarse_surface.clone(), fine_surface.clone(), 10, 1.0);
        let slave_trace =
            InterfaceTrace::new(&uncut_coupling.slave_surface, uncut_coupling.slave_edge)
                .expect("slave trace");
        let master_trace =
            InterfaceTrace::new(&uncut_coupling.master_surface, uncut_coupling.master_edge)
                .expect("master trace");
        let uncut = uncut_coupling.cross_mass(&slave_trace, &master_trace, &slave_trace.breaks);
        let uncut_error = column_error(
            &uncut,
            slave_trace.num_basis(),
            master_trace.num_basis(),
            &fine_integrals,
        );
        assert!(
            uncut_error > 1e-7,
            "integrating whole slave spans should not be accurate, error {uncut_error:e}"
        );
        for (slave, master, slave_integrals, master_integrals) in [
            (
                coarse_surface.clone(),
                fine_surface.clone(),
                coarse_integrals.clone(),
                fine_integrals.clone(),
            ),
            (
                fine_surface,
                coarse_surface,
                fine_integrals,
                coarse_integrals,
            ),
        ] {
            let coupling = MortarPatchCoupling::new(slave, master, 10, 1.0);
            let (n_s, n_m) = coupling.interface_dofs().expect("dofs");
            let d = coupling.coupling_matrix();
            assert_eq!(d.len(), n_s * n_m);
            for (a, expected) in slave_integrals.iter().enumerate() {
                let row: f64 = (0..n_m).map(|b| d[a * n_m + b]).sum();
                assert!(
                    (row - expected).abs() < 1e-9,
                    "row {a}: {row} vs ∫ R^s ds = {expected}"
                );
            }
            let error = column_error(&d, n_s, n_m, &master_integrals);
            assert!(
                error < 1e-9,
                "column sums deviate from ∫ R^m ds by {error:e}"
            );
            assert!((d.iter().sum::<f64>() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        }
    }
}
