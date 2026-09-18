//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{dot3, finite_diff_3d, solve_3x_systems};
use super::types::{BsplineBasis, KnotVector};
use crate::{Error, Result};

/// A rational tensor-product B-spline (NURBS) surface.
///
/// S(u,v) = Σ_i Σ_j w_{ij} N_{i,p}(u) N_{j,q}(v) P_{ij} / Σ_i Σ_j w_{ij} N_{i,p}(u) N_{j,q}(v)
#[derive(Debug, Clone)]
pub struct NurbsSurface {
    /// Basis in u direction.
    pub basis_u: BsplineBasis,
    /// Basis in v direction.
    pub basis_v: BsplineBasis,
    /// Control point grid \[i\]\[j\] = \[x, y, z\].
    pub control_points: Vec<Vec<[f64; 3]>>,
    /// Weight grid \[i\]\[j\].
    pub weights: Vec<Vec<f64>>,
}
impl NurbsSurface {
    /// Create a new NURBS surface.
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knot_u: KnotVector,
        knot_v: KnotVector,
        control_points: Vec<Vec<[f64; 3]>>,
        weights: Vec<Vec<f64>>,
    ) -> Self {
        let n_u = control_points.len();
        let n_v = if n_u > 0 { control_points[0].len() } else { 0 };
        Self {
            basis_u: BsplineBasis::new(degree_u, knot_u, n_u),
            basis_v: BsplineBasis::new(degree_v, knot_v, n_v),
            control_points,
            weights,
        }
    }
    /// Evaluate the NURBS surface at (u, v).
    pub fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, nu) = self.basis_u.eval_nonzero(u);
        let (span_v, nv) = self.basis_v.eval_nonzero(v);
        let p = self.basis_u.degree;
        let q = self.basis_v.degree;
        let n_u = self.control_points.len();
        let n_v = if n_u > 0 {
            self.control_points[0].len()
        } else {
            0
        };
        let mut num = [0.0_f64; 3];
        let mut denom = 0.0_f64;
        for (k, &nu_k) in nu[..=p].iter().enumerate() {
            let iu = span_u + k - p;
            if iu >= n_u {
                continue;
            }
            for (l, &nv_l) in nv[..=q].iter().enumerate() {
                let iv = span_v + l - q;
                if iv >= n_v {
                    continue;
                }
                let wn = self.weights[iu][iv] * nu_k * nv_l;
                denom += wn;
                let cp = self.control_points[iu][iv];
                for d in 0..3 {
                    num[d] += wn * cp[d];
                }
            }
        }
        if denom.abs() < 1e-30 {
            return self.control_points[0][0];
        }
        [num[0] / denom, num[1] / denom, num[2] / denom]
    }
    /// Compute the unit surface normal at (u, v) via numerical differentiation.
    pub fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let h = 1e-6;
        let su = {
            let p0 = self.eval(u - h, v);
            let p1 = self.eval(u + h, v);
            [
                (p1[0] - p0[0]) / (2.0 * h),
                (p1[1] - p0[1]) / (2.0 * h),
                (p1[2] - p0[2]) / (2.0 * h),
            ]
        };
        let sv = {
            let p0 = self.eval(u, v - h);
            let p1 = self.eval(u, v + h);
            [
                (p1[0] - p0[0]) / (2.0 * h),
                (p1[1] - p0[1]) / (2.0 * h),
                (p1[2] - p0[2]) / (2.0 * h),
            ]
        };
        let cross = [
            su[1] * sv[2] - su[2] * sv[1],
            su[2] * sv[0] - su[0] * sv[2],
            su[0] * sv[1] - su[1] * sv[0],
        ];
        let mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if mag < 1e-14 {
            [0.0, 0.0, 1.0]
        } else {
            [cross[0] / mag, cross[1] / mag, cross[2] / mag]
        }
    }
    /// Fit the surface to a rectangular grid of target points in the
    /// least-squares sense.
    ///
    /// `points[k][l]` is the target at grid row `k` (the `u` direction) and
    /// grid column `l` (the `v` direction).  The grid needs at least as many
    /// rows and columns as the control net has, otherwise the fit would be
    /// under-determined.
    ///
    /// The grid points are parameterised by averaged chord length: along each
    /// direction the chord-length parameters of every grid line are averaged
    /// (Piegl & Tiller, *The NURBS Book*, section 9.4.1) and mapped onto the
    /// parameter domain of the respective basis.  With the collocation
    /// matrices `N_u` (`points.len() × n_u`) and `N_v`
    /// (`points[0].len() × n_v`) the control net `P` minimising
    /// `‖N_u P N_vᵀ − D‖_F` follows from two separable passes of normal
    /// equations: `(N_uᵀ N_u) R = N_uᵀ D` for every grid column, then
    /// `(N_vᵀ N_v) Pᵀ = N_vᵀ Rᵀ` for every control row.
    ///
    /// The fitted surface is a polynomial B-spline surface, so every weight is
    /// reset to `1.0`.  The surface is left untouched when an error is returned.
    ///
    /// # Errors
    ///
    /// * [`Error::TooFewPoints`] for a surface without control points, an empty
    ///   grid, or a grid with fewer rows or columns than the control net.
    /// * [`Error::LengthMismatch`] for a ragged control net or ragged grid rows,
    ///   or a knot vector that does not match the control net.
    /// * [`Error::General`] for a non-finite grid point, or a singular system
    ///   (for example consecutive identical grid lines, which repeat a
    ///   parameter value, or a knot span without any grid parameter).
    pub fn fit_to_grid(&mut self, points: &[Vec<[f64; 3]>]) -> Result<()> {
        let n_u = self.control_points.len();
        let n_v = self.control_points.first().map_or(0, Vec::len);
        if n_u == 0 || n_v == 0 {
            return Err(Error::too_few_points(1, 0));
        }
        if let Some(row) = self.control_points.iter().find(|row| row.len() != n_v) {
            return Err(Error::length_mismatch(n_v, row.len()));
        }
        let rows = points.len();
        let cols = points.first().map_or(0, Vec::len);
        if rows < n_u {
            return Err(Error::too_few_points(n_u, rows));
        }
        if cols < n_v {
            return Err(Error::too_few_points(n_v, cols));
        }
        if let Some(row) = points.iter().find(|row| row.len() != cols) {
            return Err(Error::length_mismatch(cols, row.len()));
        }
        if let Some((k, l)) = points.iter().enumerate().find_map(|(k, row)| {
            row.iter()
                .position(|q| q.iter().any(|c| !c.is_finite()))
                .map(|l| (k, l))
        }) {
            return Err(Error::general(format!(
                "grid point ({k}, {l}) has a non-finite coordinate"
            )));
        }
        let (u0, u1) = parameter_domain(&self.basis_u, n_u)?;
        let (v0, v1) = parameter_domain(&self.basis_v, n_v)?;
        let u_params: Vec<f64> = averaged_chord_parameters(rows, cols, |k, l| points[k][l])
            .into_iter()
            .map(|t| u0 + (u1 - u0) * t)
            .collect();
        let v_params: Vec<f64> = averaged_chord_parameters(cols, rows, |l, k| points[k][l])
            .into_iter()
            .map(|t| v0 + (v1 - v0) * t)
            .collect();
        let nu_mat: Vec<Vec<f64>> = u_params.iter().map(|&t| self.basis_u.eval_all(t)).collect();
        let nv_mat: Vec<Vec<f64>> = v_params.iter().map(|&t| self.basis_v.eval_all(t)).collect();
        let nu_normal = normal_matrix(&nu_mat, n_u);
        let nv_normal = normal_matrix(&nv_mat, n_v);
        // Pass 1: for every grid column `l`, solve (N_uᵀ N_u) R_l = N_uᵀ D_l.
        let mut reduced: Vec<Vec<[f64; 3]>> = Vec::with_capacity(cols);
        for l in 0..cols {
            let mut rhs = vec![[0.0_f64; 3]; n_u];
            for (row_basis, grid_row) in nu_mat.iter().zip(points.iter()) {
                accumulate_weighted(&mut rhs, row_basis, grid_row[l]);
            }
            reduced.push(solve_3x_systems(&nu_normal, &rhs, n_u)?);
        }
        // Pass 2: for every control row `i`, solve (N_vᵀ N_v) P_i = N_vᵀ R_i.
        let mut net: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n_u);
        for i in 0..n_u {
            let mut rhs = vec![[0.0_f64; 3]; n_v];
            for (row_basis, column) in nv_mat.iter().zip(reduced.iter()) {
                accumulate_weighted(&mut rhs, row_basis, column[i]);
            }
            net.push(solve_3x_systems(&nv_normal, &rhs, n_v)?);
        }
        self.control_points = net;
        self.weights = vec![vec![1.0_f64; n_v]; n_u];
        Ok(())
    }
    /// Sample the surface at (nu × nv) parameter values.
    pub fn sample(&self, nu: usize, nv: usize) -> Vec<Vec<[f64; 3]>> {
        let (u0, u1) = self.basis_u.knot_vector.domain();
        let (v0, v1) = self.basis_v.knot_vector.domain();
        (0..nu)
            .map(|i| {
                let u = u0 + (u1 - u0) * i as f64 / (nu - 1).max(1) as f64;
                (0..nv)
                    .map(|j| {
                        let v = v0 + (v1 - v0) * j as f64 / (nv - 1).max(1) as f64;
                        self.eval(u, v)
                    })
                    .collect()
            })
            .collect()
    }
}
/// A tensor-product B-spline surface in 3D space.
///
/// S(u,v) = Σ_i Σ_j N_{i,p}(u) N_{j,q}(v) P_{ij}
#[derive(Debug, Clone)]
pub struct BsplineSurface {
    /// Basis in the u direction.
    pub basis_u: BsplineBasis,
    /// Basis in the v direction.
    pub basis_v: BsplineBasis,
    /// Control point grid \[i\]\[j\] = \[x, y, z\], row-major (u varies fastest).
    pub control_points: Vec<Vec<[f64; 3]>>,
}
impl BsplineSurface {
    /// Create a new B-spline surface.
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knot_u: KnotVector,
        knot_v: KnotVector,
        control_points: Vec<Vec<[f64; 3]>>,
    ) -> Self {
        let n_u = control_points.len();
        let n_v = control_points[0].len();
        Self {
            basis_u: BsplineBasis::new(degree_u, knot_u, n_u),
            basis_v: BsplineBasis::new(degree_v, knot_v, n_v),
            control_points,
        }
    }
    /// Create a bilinear patch (p=q=1) from four corner points.
    pub fn bilinear_patch(p00: [f64; 3], p10: [f64; 3], p01: [f64; 3], p11: [f64; 3]) -> Self {
        let kv = KnotVector::new(vec![0.0, 0.0, 1.0, 1.0]);
        let cps = vec![vec![p00, p01], vec![p10, p11]];
        Self {
            basis_u: BsplineBasis::new(1, kv.clone(), 2),
            basis_v: BsplineBasis::new(1, kv, 2),
            control_points: cps,
        }
    }
    /// Evaluate the surface at (u, v): returns \[x, y, z\].
    pub fn eval(&self, u: f64, v: f64) -> [f64; 3] {
        let (span_u, nu) = self.basis_u.eval_nonzero(u);
        let (span_v, nv) = self.basis_v.eval_nonzero(v);
        let p = self.basis_u.degree;
        let q = self.basis_v.degree;
        let n_u = self.control_points.len();
        let n_v = if n_u > 0 {
            self.control_points[0].len()
        } else {
            0
        };
        let mut point = [0.0_f64; 3];
        for (k, &nu_k) in nu[..=p].iter().enumerate() {
            let iu = span_u + k - p;
            if iu >= n_u {
                continue;
            }
            for (l, &nv_l) in nv[..=q].iter().enumerate() {
                let iv = span_v + l - q;
                if iv >= n_v {
                    continue;
                }
                let w = nu_k * nv_l;
                let cp = self.control_points[iu][iv];
                for d in 0..3 {
                    point[d] += w * cp[d];
                }
            }
        }
        point
    }
    /// Evaluate partial derivative ∂S/∂u at (u, v).
    ///
    /// A surface of degree 0 in `u` is piecewise constant in that direction, so
    /// the derivative is the zero vector.
    pub fn eval_du(&self, u: f64, v: f64) -> [f64; 3] {
        let p = self.basis_u.degree;
        let q = self.basis_v.degree;
        if p == 0 {
            return [0.0; 3];
        }
        let (span_u, ders_u) = self.basis_u.eval_nonzero_derivs(u, 1);
        let (span_v, nv) = self.basis_v.eval_nonzero(v);
        let n_u = self.control_points.len();
        let n_v = if n_u > 0 {
            self.control_points[0].len()
        } else {
            0
        };
        let mut result = [0.0_f64; 3];
        for (k, &du_k) in ders_u[1][..=p].iter().enumerate() {
            let iu = span_u + k - p;
            if iu >= n_u {
                continue;
            }
            for (l, &nv_l) in nv[..=q].iter().enumerate() {
                let iv = span_v + l - q;
                if iv >= n_v {
                    continue;
                }
                let w = du_k * nv_l;
                let cp = self.control_points[iu][iv];
                for d in 0..3 {
                    result[d] += w * cp[d];
                }
            }
        }
        result
    }
    /// Evaluate partial derivative ∂S/∂v at (u, v).
    ///
    /// A surface of degree 0 in `v` is piecewise constant in that direction, so
    /// the derivative is the zero vector.
    pub fn eval_dv(&self, u: f64, v: f64) -> [f64; 3] {
        let p = self.basis_u.degree;
        let q = self.basis_v.degree;
        if q == 0 {
            return [0.0; 3];
        }
        let (span_u, nu) = self.basis_u.eval_nonzero(u);
        let (span_v, ders_v) = self.basis_v.eval_nonzero_derivs(v, 1);
        let n_u = self.control_points.len();
        let n_v = if n_u > 0 {
            self.control_points[0].len()
        } else {
            0
        };
        let mut result = [0.0_f64; 3];
        for (k, &nu_k) in nu[..=p].iter().enumerate() {
            let iu = span_u + k - p;
            if iu >= n_u {
                continue;
            }
            for (l, &dv_l) in ders_v[1][..=q].iter().enumerate() {
                let iv = span_v + l - q;
                if iv >= n_v {
                    continue;
                }
                let w = nu_k * dv_l;
                let cp = self.control_points[iu][iv];
                for d in 0..3 {
                    result[d] += w * cp[d];
                }
            }
        }
        result
    }
    /// Compute the unit surface normal at (u, v).
    ///
    /// N = (∂S/∂u × ∂S/∂v) / |∂S/∂u × ∂S/∂v|
    pub fn normal(&self, u: f64, v: f64) -> [f64; 3] {
        let du = self.eval_du(u, v);
        let dv = self.eval_dv(u, v);
        let cross = [
            du[1] * dv[2] - du[2] * dv[1],
            du[2] * dv[0] - du[0] * dv[2],
            du[0] * dv[1] - du[1] * dv[0],
        ];
        let mag = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if mag < 1e-14 {
            [0.0, 0.0, 1.0]
        } else {
            [cross[0] / mag, cross[1] / mag, cross[2] / mag]
        }
    }
    /// Compute the first fundamental form coefficients E, F, G at (u, v).
    ///
    /// E = ∂S/∂u · ∂S/∂u, F = ∂S/∂u · ∂S/∂v, G = ∂S/∂v · ∂S/∂v
    pub fn first_fundamental_form(&self, u: f64, v: f64) -> (f64, f64, f64) {
        let su = self.eval_du(u, v);
        let sv = self.eval_dv(u, v);
        let e = su[0] * su[0] + su[1] * su[1] + su[2] * su[2];
        let f = su[0] * sv[0] + su[1] * sv[1] + su[2] * sv[2];
        let g = sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2];
        (e, f, g)
    }
    /// Compute Gaussian curvature K = (LN - M²)/(EG - F²) at (u, v).
    ///
    /// Uses finite differences for second derivatives.
    pub fn gaussian_curvature(&self, u: f64, v: f64) -> f64 {
        let h = 1e-5;
        let (e, f, g) = self.first_fundamental_form(u, v);
        let egf2 = e * g - f * f;
        if egf2.abs() < 1e-20 {
            return 0.0;
        }
        let suu = finite_diff_3d(|uu| self.eval_du(uu, v), u, h);
        let svv = finite_diff_3d(|vv| self.eval_dv(u, vv), v, h);
        let suv = finite_diff_3d(|uu| self.eval_dv(uu, v), u, h);
        let n = self.normal(u, v);
        let l = dot3(suu, n);
        let m_coef = dot3(suv, n);
        let nm = dot3(svv, n);
        (l * nm - m_coef * m_coef) / egf2
    }
    /// Compute mean curvature H = (EN - 2FM + GL)/(2(EG - F²)) at (u, v).
    pub fn mean_curvature(&self, u: f64, v: f64) -> f64 {
        let h = 1e-5;
        let (e, f, g) = self.first_fundamental_form(u, v);
        let egf2 = e * g - f * f;
        if egf2.abs() < 1e-20 {
            return 0.0;
        }
        let suu = finite_diff_3d(|uu| self.eval_du(uu, v), u, h);
        let svv = finite_diff_3d(|vv| self.eval_dv(u, vv), v, h);
        let suv = finite_diff_3d(|uu| self.eval_dv(uu, v), u, h);
        let n = self.normal(u, v);
        let l = dot3(suu, n);
        let m_coef = dot3(suv, n);
        let nm = dot3(svv, n);
        (e * nm - 2.0 * f * m_coef + g * l) / (2.0 * egf2)
    }
    /// Sample the surface at (nu × nv) parameter values.
    pub fn sample(&self, nu: usize, nv: usize) -> Vec<Vec<[f64; 3]>> {
        let (u0, u1) = self.basis_u.knot_vector.domain();
        let (v0, v1) = self.basis_v.knot_vector.domain();
        (0..nu)
            .map(|i| {
                let u = u0 + (u1 - u0) * i as f64 / (nu - 1).max(1) as f64;
                (0..nv)
                    .map(|j| {
                        let v = v0 + (v1 - v0) * j as f64 / (nv - 1).max(1) as f64;
                        self.eval(u, v)
                    })
                    .collect()
            })
            .collect()
    }
    /// Compute Gaussian and mean curvature at (u, v).
    ///
    /// Returns `(K_gauss, H_mean)`.
    pub fn compute_curvature(&self, u: f64, v: f64) -> (f64, f64) {
        (self.gaussian_curvature(u, v), self.mean_curvature(u, v))
    }
    /// Refine the surface by inserting new knots in the `u` and `v` directions.
    ///
    /// Every knot is inserted one at a time with Boehm's algorithm: in the
    /// running knot vector (which already contains the earlier insertions) the
    /// span `k` with `U_k ≤ x < U_{k+1}` is located and the control points
    /// `k-p+1 ..= k` are replaced by `Q_i = α_i P_i + (1 − α_i) P_{i-1}` with
    /// `α_i = (x − U_i) / (U_{i+p} − U_i)`.  Knot insertion is exact, so the
    /// refined surface describes the same geometry; it gains one control row
    /// (column) per inserted `u` (`v`) knot and its knot vectors are the sorted
    /// merge of the old knots and the new ones.  A knot may be given several
    /// times to raise its multiplicity.
    ///
    /// # Errors
    ///
    /// * [`Error::OutOfRange`] when a knot is not finite or lies outside the
    ///   parameter domain of its direction.
    /// * [`Error::Unsupported`] when a knot lies on the boundary of the domain
    ///   or its multiplicity would exceed the degree of its direction.
    /// * [`Error::TooFewPoints`], [`Error::LengthMismatch`] when the surface is
    ///   empty, ragged, or its knot vectors do not match the control net.
    pub fn refine_knots(&self, new_knots_u: &[f64], new_knots_v: &[f64]) -> Result<BsplineSurface> {
        let n_u = self.control_points.len();
        let n_v = self.control_points.first().map_or(0, Vec::len);
        if n_u == 0 || n_v == 0 {
            return Err(Error::too_few_points(1, 0));
        }
        if let Some(row) = self.control_points.iter().find(|row| row.len() != n_v) {
            return Err(Error::length_mismatch(n_v, row.len()));
        }
        let p = self.basis_u.degree;
        let q = self.basis_v.degree;
        let mut knots_u = self.basis_u.knot_vector.knots.clone();
        let mut knots_v = self.basis_v.knot_vector.knots.clone();
        let mut net = self.control_points.clone();
        for &knot in new_knots_u {
            let plan = KnotInsertion::new(&knots_u, p, net.len(), knot)?;
            net = plan.apply_to_rows(&net);
            knots_u = plan.knots;
        }
        for &knot in new_knots_v {
            let n_cur = net.first().map_or(0, Vec::len);
            let plan = KnotInsertion::new(&knots_v, q, n_cur, knot)?;
            for row in net.iter_mut() {
                *row = plan.apply(row);
            }
            knots_v = plan.knots;
        }
        Ok(BsplineSurface::new(
            p,
            q,
            KnotVector::try_new(knots_u)?,
            KnotVector::try_new(knots_v)?,
            net,
        ))
    }
}

/// Valid parameter domain `[U_p, U_{n}]` of a basis with `n_ctrl` functions.
///
/// # Errors
///
/// [`Error::LengthMismatch`] when the basis does not have `n_ctrl` functions or
/// its knot vector does not hold exactly `n_ctrl + degree + 1` knots.
fn parameter_domain(basis: &BsplineBasis, n_ctrl: usize) -> Result<(f64, f64)> {
    if basis.n_ctrl != n_ctrl {
        return Err(Error::length_mismatch(n_ctrl, basis.n_ctrl));
    }
    let expected = n_ctrl + basis.degree + 1;
    let knots = &basis.knot_vector.knots;
    if knots.len() != expected {
        return Err(Error::length_mismatch(expected, knots.len()));
    }
    Ok((knots[basis.degree], knots[n_ctrl]))
}

/// Averaged chord-length parameters in `[0, 1]` of `n` samples along each of
/// `lines` grid lines; `point(k, line)` is sample `k` of grid line `line`.
///
/// Lines of zero total length carry no parameter information and are skipped;
/// when no line has any extent the parameters are uniform.
fn averaged_chord_parameters<F>(n: usize, lines: usize, point: F) -> Vec<f64>
where
    F: Fn(usize, usize) -> [f64; 3],
{
    let mut sum = vec![0.0_f64; n];
    let mut used = 0_usize;
    for line in 0..lines {
        let mut cumulative = Vec::with_capacity(n);
        let mut total = 0.0_f64;
        cumulative.push(0.0_f64);
        for k in 1..n {
            let a = point(k - 1, line);
            let b = point(k, line);
            total += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            cumulative.push(total);
        }
        if total > 0.0 && total.is_finite() {
            for (acc, c) in sum.iter_mut().zip(cumulative.iter()) {
                *acc += c / total;
            }
            used += 1;
        }
    }
    if used == 0 {
        let last = n.saturating_sub(1).max(1) as f64;
        return (0..n).map(|k| k as f64 / last).collect();
    }
    sum.into_iter().map(|acc| acc / used as f64).collect()
}

/// `NᵀN` for a collocation matrix `n` with `cols` columns.
fn normal_matrix(n: &[Vec<f64>], cols: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0_f64; cols]; cols];
    for row in n {
        for (result_row, &ri) in result.iter_mut().zip(row.iter()) {
            if ri == 0.0 {
                continue;
            }
            for (entry, &rj) in result_row.iter_mut().zip(row.iter()) {
                *entry += ri * rj;
            }
        }
    }
    result
}

/// `rhs_i += basis_i · value` for every basis function `i` (one row of `Nᵀ D`).
fn accumulate_weighted(rhs: &mut [[f64; 3]], basis_row: &[f64], value: [f64; 3]) {
    for (acc, &w) in rhs.iter_mut().zip(basis_row.iter()) {
        if w == 0.0 {
            continue;
        }
        for (a, v) in acc.iter_mut().zip(value.iter()) {
            *a += w * v;
        }
    }
}

/// Boehm single-knot insertion into one parametric direction.
///
/// The span and the blend factors depend only on the knot vector, so one plan
/// serves every control curve of the net in that direction.
struct KnotInsertion {
    /// Knot vector after the insertion.
    knots: Vec<f64>,
    /// `alphas[i]` blends the new control point `Q_i = α_i P_i + (1 − α_i) P_{i-1}`
    /// (`n_ctrl + 1` entries; `α = 1` keeps `P_i`, `α = 0` keeps `P_{i-1}`).
    alphas: Vec<f64>,
}

impl KnotInsertion {
    /// Plan the insertion of `knot` into `knots` (degree `degree`, `n_ctrl`
    /// control points).
    fn new(knots: &[f64], degree: usize, n_ctrl: usize, knot: f64) -> Result<Self> {
        let p = degree;
        if n_ctrl <= p {
            return Err(Error::too_few_points(p.saturating_add(1), n_ctrl));
        }
        let expected = n_ctrl + p + 1;
        if knots.len() != expected {
            return Err(Error::length_mismatch(expected, knots.len()));
        }
        if knots.iter().any(|k| !k.is_finite()) || knots.windows(2).any(|w| w[1] < w[0]) {
            return Err(Error::general(
                "knot vector must be finite and non-decreasing",
            ));
        }
        let (lo, hi) = (knots[p], knots[n_ctrl]);
        if !knot.is_finite() || knot < lo || knot > hi {
            return Err(Error::out_of_range("knot", knot, lo, hi));
        }
        if knot == lo || knot == hi {
            return Err(Error::unsupported(
                "knot insertion",
                format!("knot {knot} lies on the boundary of the parameter domain [{lo}, {hi}]"),
            ));
        }
        let multiplicity = knots.iter().filter(|&&k| k == knot).count();
        if multiplicity + 1 > p.max(1) {
            return Err(Error::unsupported(
                "knot insertion",
                format!(
                    "knot {knot} already has multiplicity {multiplicity}; inserting it again \
                     would exceed the degree {p}"
                ),
            ));
        }
        // Span k with U_k <= knot < U_{k+1}; lo < knot < hi puts it in p..n_ctrl-1.
        let span = knots.partition_point(|&k| k <= knot) - 1;
        let mut alphas = Vec::with_capacity(n_ctrl + 1);
        for i in 0..=n_ctrl {
            let alpha = if i + p <= span {
                1.0
            } else if i > span {
                0.0
            } else {
                let denominator = knots[i + p] - knots[i];
                if denominator <= 0.0 {
                    return Err(Error::degenerate_geometry(format!(
                        "knot span [{}, {}] has zero length",
                        knots[i],
                        knots[i + p]
                    )));
                }
                (knot - knots[i]) / denominator
            };
            alphas.push(alpha);
        }
        let mut refined = Vec::with_capacity(knots.len() + 1);
        refined.extend_from_slice(&knots[..=span]);
        refined.push(knot);
        refined.extend_from_slice(&knots[span + 1..]);
        Ok(Self {
            knots: refined,
            alphas,
        })
    }

    /// Insert the knot into one control curve (`n_ctrl` points).
    fn apply(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.alphas
            .iter()
            .enumerate()
            .map(|(i, &alpha)| {
                if alpha >= 1.0 {
                    points[i]
                } else if alpha <= 0.0 {
                    points[i - 1]
                } else {
                    blend(points[i - 1], points[i], alpha)
                }
            })
            .collect()
    }

    /// Insert the knot along the first index of a control net: row `i` of the
    /// result is `α_i · rows[i] + (1 − α_i) · rows[i − 1]` component-wise.
    fn apply_to_rows(&self, rows: &[Vec<[f64; 3]>]) -> Vec<Vec<[f64; 3]>> {
        self.alphas
            .iter()
            .enumerate()
            .map(|(i, &alpha)| {
                if alpha >= 1.0 {
                    rows[i].clone()
                } else if alpha <= 0.0 {
                    rows[i - 1].clone()
                } else {
                    rows[i - 1]
                        .iter()
                        .zip(rows[i].iter())
                        .map(|(&previous, &current)| blend(previous, current, alpha))
                        .collect()
                }
            })
            .collect()
    }
}

/// `alpha · current + (1 − alpha) · previous`.
fn blend(previous: [f64; 3], current: [f64; 3], alpha: f64) -> [f64; 3] {
    [
        (1.0 - alpha) * previous[0] + alpha * current[0],
        (1.0 - alpha) * previous[1] + alpha * current[1],
        (1.0 - alpha) * previous[2] + alpha * current[2],
    ]
}
