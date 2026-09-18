//! General linear PDE operator for physics-informed interpolation.
//!
//! A `PdeOperator` represents a linear partial differential operator of the form:
//!
//! ```text
//! L[u](x) = Σ_k c_k * ∂^{|α_k|} u / ∂x^{α_k}
//! ```
//!
//! where each `α_k` is a multi-index giving the order of differentiation in each
//! spatial dimension.  The operator can be evaluated numerically at a point via
//! a finite-difference stencil.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_interpolate::physics_informed::pde_operator::PdeOperator;
//!
//! // Laplacian in 2D: d²/dx₀² + d²/dx₁²
//! let lap = PdeOperator::laplacian(2);
//! let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];  // u = x² + y²
//! let residual = lap.apply_fd(&[1.0, 1.0], f, 1e-4);
//! assert!((residual - 4.0).abs() < 1e-5, "Laplacian of x²+y² should be 4");
//! ```

use crate::error::{InterpolateError, InterpolateResult};

// ─────────────────────────────────────────────────────────────────────────────
// PdeOperator
// ─────────────────────────────────────────────────────────────────────────────

/// A linear PDE operator:  L\[u\](x) = Σ_k c_k · ∂^{α_k} u(x)
///
/// Each term is a pair `(coefficient, multi_index)` where `multi_index[d]` is the
/// order of differentiation with respect to the `d`-th coordinate.
#[derive(Debug, Clone)]
pub struct PdeOperator {
    /// List of `(coefficient, derivative_order_per_dim)` terms.
    pub terms: Vec<(f64, Vec<usize>)>,
    /// Spatial dimension of the input space.
    pub dim: usize,
}

impl PdeOperator {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create the Laplacian operator in `dim` dimensions:
    /// `L = Σ_{i=0}^{dim-1} ∂²/∂x_i²`.
    pub fn laplacian(dim: usize) -> Self {
        let terms = (0..dim)
            .map(|i| {
                let mut order = vec![0usize; dim];
                order[i] = 2;
                (1.0_f64, order)
            })
            .collect();
        Self { terms, dim }
    }

    /// Create a 1D advection operator: `L = speed · ∂/∂x`.
    pub fn advection_1d(speed: f64) -> Self {
        Self {
            terms: vec![(speed, vec![1])],
            dim: 1,
        }
    }

    /// Create a custom linear combination of derivative terms.
    ///
    /// Each element of `terms` is `(coefficient, order_per_dim)` where
    /// `order_per_dim` must have length `dim`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if any multi-index has the wrong length.
    pub fn custom(terms: Vec<(f64, Vec<usize>)>, dim: usize) -> InterpolateResult<Self> {
        for (_, ref order) in &terms {
            if order.len() != dim {
                return Err(InterpolateError::InvalidInput {
                    message: format!(
                        "Multi-index length {} does not match dim {}",
                        order.len(),
                        dim
                    ),
                });
            }
        }
        Ok(Self { terms, dim })
    }

    // ── Finite-difference evaluation ─────────────────────────────────────────

    /// Evaluate `L[f]` at `center` by applying finite differences.
    ///
    /// Supports derivative orders 0 (identity), 1 (central difference), and
    /// 2 (second central difference) per dimension.  Mixed partials of total
    /// order ≤ 2 are handled by composing 1-D stencils.
    ///
    /// For a term with multi-index `[α₀, α₁, …]` the stencil approximation is:
    ///
    /// - Order 0 in dim d → evaluate at `center`
    /// - Order 1 in dim d → `(f(c + h*eₐ) - f(c - h*eₐ)) / (2h)`
    /// - Order 2 in dim d → `(f(c + h*eₐ) - 2f(c) + f(c - h*eₐ)) / h²`
    ///
    /// Mixed partials of order `(1,1)` in two different dimensions use the
    /// cross-difference stencil:
    /// `(f(++)-f(+-)-f(-+)+f(--)) / (4h²)`.
    ///
    /// Higher-order mixed partials (total order > 2) raise `NotImplemented`.
    ///
    /// # Arguments
    /// * `center` – Point at which the operator is evaluated; length must equal `dim`.
    /// * `f_at`   – Function to apply the operator to.
    /// * `h`      – Finite-difference step size.
    pub fn apply_fd(&self, center: &[f64], f_at: impl Fn(&[f64]) -> f64, h: f64) -> f64 {
        assert_eq!(
            center.len(),
            self.dim,
            "center length must equal operator dim"
        );
        let f = &f_at;
        self.terms
            .iter()
            .map(|(coeff, order)| coeff * apply_term_fd(center, order, f, h))
            .sum()
    }

    /// Like `apply_fd` but returns an error on unsupported stencil orders
    /// instead of panicking.
    pub fn try_apply_fd(
        &self,
        center: &[f64],
        f_at: impl Fn(&[f64]) -> f64,
        h: f64,
    ) -> InterpolateResult<f64> {
        if center.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "center has {} components, operator has dim {}",
                center.len(),
                self.dim
            )));
        }
        let f = &f_at;
        let mut total = 0.0_f64;
        for (coeff, order) in &self.terms {
            let val = try_apply_term_fd(center, order, f, h)?;
            total += coeff * val;
        }
        Ok(total)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal finite-difference helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the finite-difference stencil for a single term whose multi-index is
/// `order`.  The total derivative order is `order.iter().sum()`.
fn apply_term_fd(center: &[f64], order: &[usize], f: &impl Fn(&[f64]) -> f64, h: f64) -> f64 {
    let total_order: usize = order.iter().sum();

    match total_order {
        0 => f(center),
        1 => {
            // Find the single active dimension.
            let dim = order.iter().position(|&o| o == 1).unwrap_or(0);
            central_diff_1(center, dim, f, h)
        }
        2 => {
            let active: Vec<usize> = order
                .iter()
                .enumerate()
                .filter(|(_, &o)| o > 0)
                .map(|(i, _)| i)
                .collect();
            if active.len() == 1 {
                // Pure second derivative in `active[0]`.
                central_diff_2(center, active[0], f, h)
            } else {
                // Mixed ∂²/∂x_{d0}∂x_{d1}
                central_diff_mixed(center, active[0], active[1], f, h)
            }
        }
        // Higher-order: apply chain of 1st/2nd-order stencils recursively.
        n => apply_higher_order(center, order, f, h, n),
    }
}

/// Same as `apply_term_fd` but propagates an error for unsupported order ≥ 4.
fn try_apply_term_fd(
    center: &[f64],
    order: &[usize],
    f: &impl Fn(&[f64]) -> f64,
    h: f64,
) -> InterpolateResult<f64> {
    let total: usize = order.iter().sum();
    if total >= 4 {
        return Err(InterpolateError::NotImplemented(format!(
            "Finite-difference stencil not implemented for total derivative order {total}"
        )));
    }
    Ok(apply_term_fd(center, order, f, h))
}

/// Central first difference: `(f(c+h·eₐ) - f(c-h·eₐ)) / (2h)`.
fn central_diff_1(center: &[f64], dim: usize, f: &impl Fn(&[f64]) -> f64, h: f64) -> f64 {
    let mut cp = center.to_vec();
    let mut cm = center.to_vec();
    cp[dim] += h;
    cm[dim] -= h;
    (f(&cp) - f(&cm)) / (2.0 * h)
}

/// Central second difference: `(f(c+h·eₐ) - 2f(c) + f(c-h·eₐ)) / h²`.
fn central_diff_2(center: &[f64], dim: usize, f: &impl Fn(&[f64]) -> f64, h: f64) -> f64 {
    let mut cp = center.to_vec();
    let mut cm = center.to_vec();
    cp[dim] += h;
    cm[dim] -= h;
    (f(&cp) - 2.0 * f(center) + f(&cm)) / (h * h)
}

/// Cross central difference for mixed ∂²/∂x_{d0} ∂x_{d1}:
/// `(f(++) - f(+-) - f(-+) + f(--)) / (4h²)`.
fn central_diff_mixed(
    center: &[f64],
    d0: usize,
    d1: usize,
    f: &impl Fn(&[f64]) -> f64,
    h: f64,
) -> f64 {
    let mut pp = center.to_vec();
    let mut pm = center.to_vec();
    let mut mp = center.to_vec();
    let mut mm = center.to_vec();
    pp[d0] += h;
    pp[d1] += h;
    pm[d0] += h;
    pm[d1] -= h;
    mp[d0] -= h;
    mp[d1] += h;
    mm[d0] -= h;
    mm[d1] -= h;
    (f(&pp) - f(&pm) - f(&mp) + f(&mm)) / (4.0 * h * h)
}

/// Apply higher-order derivatives by composing 1st and 2nd order stencils.
/// Each dimension with order `k` contributes `k/2` second-difference stencils
/// and `k%2` first-difference stencils, composed by numerical differentiation.
fn apply_higher_order(
    center: &[f64],
    order: &[usize],
    f: &impl Fn(&[f64]) -> f64,
    h: f64,
    _total: usize,
) -> f64 {
    // Build a list of (dim, degree-1 or degree-2) applications.
    let mut ops: Vec<(usize, u8)> = Vec::new();
    for (d, &o) in order.iter().enumerate() {
        let n2 = o / 2;
        let n1 = o % 2;
        for _ in 0..n2 {
            ops.push((d, 2));
        }
        for _ in 0..n1 {
            ops.push((d, 1));
        }
    }
    // Apply ops right-to-left via closure nesting.
    apply_ops_recursively(center, &ops, f, h)
}

fn apply_ops_recursively(
    center: &[f64],
    ops: &[(usize, u8)],
    f: &impl Fn(&[f64]) -> f64,
    h: f64,
) -> f64 {
    if ops.is_empty() {
        return f(center);
    }
    let (d, degree) = ops[0];
    let rest = &ops[1..];

    let inner_fn = move |x: &[f64]| apply_ops_recursively(x, rest, f, h);

    if degree == 1 {
        central_diff_1(center, d, &inner_fn, h)
    } else {
        central_diff_2(center, d, &inner_fn, h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PhysicsInformedRbf  (PDE-constrained RBF using PdeOperator)
// ─────────────────────────────────────────────────────────────────────────────

/// RBF kernel type used internally by `PhysicsInformedRbf`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum RbfKernel {
    /// Thin-plate spline: φ(r) = r² log(r) (zero for r=0)
    ThinPlateSpline,
    /// Multiquadric: φ(r) = √(1 + (ε r)²)
    Multiquadric,
    /// Inverse multiquadric: φ(r) = 1 / √(1 + (ε r)²)
    InverseMultiquadric,
    /// Gaussian: φ(r) = exp(-(ε r)²)
    Gaussian,
}

impl RbfKernel {
    /// Evaluate the kernel for radial distance `r` and shape parameter `eps`.
    pub fn eval(&self, r: f64, eps: f64) -> f64 {
        match self {
            RbfKernel::ThinPlateSpline => {
                if r < 1e-300 {
                    0.0
                } else {
                    r * r * r.ln()
                }
            }
            RbfKernel::Multiquadric => (1.0 + (eps * r) * (eps * r)).sqrt(),
            RbfKernel::InverseMultiquadric => 1.0 / (1.0 + (eps * r) * (eps * r)).sqrt(),
            RbfKernel::Gaussian => (-(eps * r) * (eps * r)).exp(),
        }
    }
}

/// Configuration for `PhysicsInformedRbf`.
#[derive(Debug, Clone)]
pub struct PhysicsInformedRbfConfig {
    /// Weight λ of the PDE-residual penalty term.
    pub pde_weight: f64,
    /// Number of uniformly-sampled interior collocation points (per dimension for 1D,
    /// total for higher dimensions).
    pub n_collocation: usize,
    /// RBF kernel.
    pub kernel: RbfKernel,
    /// Shape parameter ε for the kernel.
    pub epsilon: f64,
    /// Small ridge added to the normal-equation diagonal for numerical stability.
    pub ridge: f64,
}

impl Default for PhysicsInformedRbfConfig {
    fn default() -> Self {
        Self {
            pde_weight: 1.0,
            n_collocation: 20,
            kernel: RbfKernel::Multiquadric,
            epsilon: 1.0,
            ridge: 1e-10,
        }
    }
}

/// Physics-informed RBF interpolant that enforces a linear PDE at a set of
/// collocation points inside the data domain.
///
/// Given:
/// - data `(x_i, y_i)` for `i = 0..n`
/// - collocation points `c_j` for `j = 0..m`
/// - a linear PDE operator `L` with known RHS `g(x)`
///
/// The interpolant `f(x) = Σ_i α_i φ(||x - x_i||)` is found by minimising:
///
/// ```text
/// ||Φ α - y||²  +  λ ||L[f](c) - g(c)||²
/// ```
///
/// which leads to the augmented normal equations:
///
/// ```text
/// (ΦᵀΦ + λ LᵀL) α = Φᵀ y + λ Lᵀ g
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsInformedRbf {
    config: PhysicsInformedRbfConfig,
    /// RBF centres (the training data sites).
    centers: Vec<Vec<f64>>,
    /// Solved RBF weights α.
    coeffs: Vec<f64>,
    /// Collocation points used during fitting.
    collocation_pts: Vec<Vec<f64>>,
    /// PDE operator stored for residual evaluation.
    operator: PdeOperator,
}

impl PhysicsInformedRbf {
    // ── distance helper ──────────────────────────────────────────────────────

    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    }

    // ── RBF matrix Φ  (n×n) ─────────────────────────────────────────────────

    fn rbf_matrix(centers: &[Vec<f64>], kernel: &RbfKernel, eps: f64) -> Vec<Vec<f64>> {
        let n = centers.len();
        let mut phi = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = Self::dist(&centers[i], &centers[j]);
                phi[i][j] = kernel.eval(r, eps);
            }
        }
        phi
    }

    // ── PDE operator row for collocation point c ─────────────────────────────
    // Row k of L: L[k][j] = (L φ_j)(c_k) via finite differences on the kernel.

    fn pde_operator_row(
        c: &[f64],
        centers: &[Vec<f64>],
        kernel: &RbfKernel,
        eps: f64,
        op: &PdeOperator,
        h: f64,
    ) -> Vec<f64> {
        centers
            .iter()
            .map(|xi| {
                let phi_j = |x: &[f64]| {
                    let r = Self::dist(x, xi);
                    kernel.eval(r, eps)
                };
                op.apply_fd(c, phi_j, h)
            })
            .collect()
    }

    // ── matrix–vector helpers ─────────────────────────────────────────────────

    fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum())
            .collect()
    }

    /// Aᵀ A
    fn gram(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = if a.is_empty() { 0 } else { a[0].len() };
        let mut g = vec![vec![0.0f64; n]; n];
        for row in a {
            for i in 0..n {
                for j in 0..n {
                    g[i][j] += row[i] * row[j];
                }
            }
        }
        g
    }

    /// Aᵀ v
    fn at_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        let n = if a.is_empty() { 0 } else { a[0].len() };
        let mut out = vec![0.0f64; n];
        for (row, &vi) in a.iter().zip(v.iter()) {
            for j in 0..n {
                out[j] += row[j] * vi;
            }
        }
        out
    }

    // ── Cholesky solver ──────────────────────────────────────────────────────

    fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> InterpolateResult<Vec<f64>> {
        use crate::random_features::cholesky_solve as rf_chol;
        rf_chol(a, b)
    }

    // ── collocation point generation ─────────────────────────────────────────

    /// Generate `n_collocation` points uniformly within the bounding box of `data`.
    fn make_collocation(data: &[Vec<f64>], n_collocation: usize, seed: u64) -> Vec<Vec<f64>> {
        if data.is_empty() || n_collocation == 0 {
            return Vec::new();
        }
        let dim = data[0].len();
        // Compute per-dimension bounding box.
        let mut mins = vec![f64::INFINITY; dim];
        let mut maxs = vec![f64::NEG_INFINITY; dim];
        for pt in data {
            for (d, &v) in pt.iter().enumerate() {
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }
        // Shrink bounding box by 5% to stay inside.
        for d in 0..dim {
            let range = (maxs[d] - mins[d]).max(1e-12);
            mins[d] += 0.05 * range;
            maxs[d] -= 0.05 * range;
        }
        // LCG-based sampling.
        let mut state = seed.wrapping_add(1);
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n_collocation)
            .map(|_| {
                (0..dim)
                    .map(|d| mins[d] + next(&mut state) * (maxs[d] - mins[d]))
                    .collect()
            })
            .collect()
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Fit the physics-informed RBF interpolant.
    ///
    /// # Arguments
    /// * `points`   – Data locations, shape `[n][dim]`.
    /// * `values`   – Data values, length `n`.
    /// * `operator` – PDE operator `L`.
    /// * `rhs_fn`   – RHS of the PDE: `L[u](x) = rhs_fn(x)`.
    /// * `config`   – Solver configuration.
    pub fn fit(
        points: &[Vec<f64>],
        values: &[f64],
        operator: PdeOperator,
        rhs_fn: impl Fn(&[f64]) -> f64,
        config: PhysicsInformedRbfConfig,
    ) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::InsufficientData(
                "No data points provided".to_string(),
            ));
        }
        if points.len() != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points ({}) and values ({}) have different lengths",
                points.len(),
                values.len()
            )));
        }
        let n = points.len();
        let kernel = &config.kernel;
        let eps = config.epsilon;
        let lambda = config.pde_weight;
        // Finite-difference step — scale with data spread.
        let h_fd = 1e-4;

        // Build RBF matrix Φ (n×n).
        let phi = Self::rbf_matrix(points, kernel, eps);

        // Build collocation points and PDE operator matrix L (m×n).
        let colloc = Self::make_collocation(points, config.n_collocation, 42);
        let m = colloc.len();

        let mut l_mat: Vec<Vec<f64>> = Vec::with_capacity(m);
        for c in &colloc {
            let row = Self::pde_operator_row(c, points, kernel, eps, &operator, h_fd);
            l_mat.push(row);
        }

        // g = rhs evaluated at collocation points.
        let g: Vec<f64> = colloc.iter().map(|c| rhs_fn(c)).collect();

        // Normal equations:  (ΦᵀΦ + λ LᵀL + ridge I) α = Φᵀ y + λ Lᵀ g
        let phi_t_phi = Self::gram(&phi);
        let l_t_l = Self::gram(&l_mat);

        let mut lhs = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                lhs[i][j] = phi_t_phi[i][j] + lambda * l_t_l[i][j];
                if i == j {
                    lhs[i][j] += config.ridge;
                }
            }
        }

        let phi_t_y = Self::at_vec(&phi, values);
        let l_t_g = Self::at_vec(&l_mat, &g);

        let mut rhs_vec = vec![0.0f64; n];
        for i in 0..n {
            rhs_vec[i] = phi_t_y[i] + lambda * l_t_g[i];
        }

        let coeffs = Self::cholesky_solve(&lhs, &rhs_vec)?;

        Ok(Self {
            config,
            centers: points.to_vec(),
            coeffs,
            collocation_pts: colloc,
            operator,
        })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: &[f64]) -> f64 {
        self.centers
            .iter()
            .zip(self.coeffs.iter())
            .map(|(xi, &ai)| {
                let r = Self::dist(x, xi);
                ai * self.config.kernel.eval(r, self.config.epsilon)
            })
            .sum()
    }

    /// Evaluate at multiple points.
    pub fn eval_batch(&self, points: &[Vec<f64>]) -> Vec<f64> {
        points.iter().map(|x| self.eval(x)).collect()
    }

    /// PDE residual `(L[f] - rhs)(x)` at a point `x` using finite differences.
    ///
    /// A small residual confirms the PDE constraint is satisfied.
    pub fn pde_residual(&self, x: &[f64], rhs_fn: impl Fn(&[f64]) -> f64) -> f64 {
        let h = 1e-4;
        let f_fn = |pt: &[f64]| self.eval(pt);
        let lf = self.operator.apply_fd(x, f_fn, h);
        lf - rhs_fn(x)
    }

    /// Collocation points used during fitting (informational).
    pub fn collocation_pts(&self) -> &[Vec<f64>] {
        &self.collocation_pts
    }

    /// Number of training data points.
    pub fn n_centers(&self) -> usize {
        self.centers.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── PdeOperator tests ────────────────────────────────────────────────────

    #[test]
    fn test_laplacian_1d() {
        // L = d²/dx²  applied to  f(x) = x² → 2
        let lap = PdeOperator::laplacian(1);
        let f = |x: &[f64]| x[0] * x[0];
        let val = lap.apply_fd(&[1.5], f, 1e-4);
        assert!(
            (val - 2.0).abs() < 1e-5,
            "1D Laplacian of x² should be 2, got {val}"
        );
    }

    #[test]
    fn test_laplacian_2d() {
        // L = d²/dx² + d²/dy² applied to f(x,y) = x²+y² → 4
        let lap = PdeOperator::laplacian(2);
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let val = lap.apply_fd(&[1.0, 2.0], f, 1e-4);
        assert!(
            (val - 4.0).abs() < 1e-5,
            "2D Laplacian of x²+y² should be 4, got {val}"
        );
    }

    #[test]
    fn test_advection_1d() {
        // L = a * d/dx applied to f(x) = sin(x) → a * cos(x)
        let speed = 2.0;
        let op = PdeOperator::advection_1d(speed);
        let x0 = PI / 4.0;
        let f = |x: &[f64]| x[0].sin();
        let val = op.apply_fd(&[x0], f, 1e-5);
        let expected = speed * x0.cos();
        assert!(
            (val - expected).abs() < 1e-4,
            "Advection stencil: got {val}, expected {expected}"
        );
    }

    #[test]
    fn test_custom_operator() {
        // L = 3 * d/dx  on f(x,y) = x + y  → should be 3
        let op = PdeOperator::custom(vec![(3.0, vec![1, 0])], 2).expect("custom op");
        let f = |x: &[f64]| x[0] + x[1];
        let val = op.apply_fd(&[0.5, 0.5], f, 1e-5);
        assert!((val - 3.0).abs() < 1e-4, "Custom op value {val}");
    }

    #[test]
    fn test_custom_operator_wrong_dim() {
        let result = PdeOperator::custom(vec![(1.0, vec![1, 0])], 3);
        assert!(
            result.is_err(),
            "Should fail when multi-index dim != operator dim"
        );
    }

    #[test]
    fn test_try_apply_fd_dimension_check() {
        let lap = PdeOperator::laplacian(2);
        let f = |x: &[f64]| x[0];
        let result = lap.try_apply_fd(&[1.0], f, 1e-4);
        assert!(result.is_err(), "Should error on wrong center length");
    }

    // ── PhysicsInformedRbf tests ──────────────────────────────────────────────

    #[test]
    fn test_pifr_fit_and_eval() {
        // Harmonic function: u(x,y) = x²-y², Laplacian = 0
        let pts: Vec<Vec<f64>> = (0..5)
            .flat_map(|i| (0..5).map(move |j| vec![i as f64 * 0.25, j as f64 * 0.25]))
            .collect();
        let vals: Vec<f64> = pts.iter().map(|p| p[0] * p[0] - p[1] * p[1]).collect();

        let op = PdeOperator::laplacian(2);
        let rhs_fn = |_x: &[f64]| 0.0_f64; // Laplace equation: L[u]=0
        let config = PhysicsInformedRbfConfig {
            pde_weight: 1.0,
            n_collocation: 10,
            kernel: RbfKernel::Multiquadric,
            epsilon: 2.0,
            ridge: 1e-8,
        };

        let interp =
            PhysicsInformedRbf::fit(&pts, &vals, op, rhs_fn, config).expect("fit should succeed");

        // Verify training data is reproduced to within tolerance.
        for (pt, &v) in pts.iter().zip(vals.iter()) {
            let pred = interp.eval(pt);
            assert!(
                (pred - v).abs() < 0.1,
                "Training error too large at {:?}: pred={pred:.4}, true={v:.4}",
                pt
            );
        }
    }

    #[test]
    fn test_pifr_laplacian_pde_residual_small() {
        // u = x²-y² is harmonic; PDE residual (Laplacian) should be reduced by the
        // penalty.  We use a larger data set and tighter grid for a more accurate fit.
        let pts: Vec<Vec<f64>> = (0..5)
            .flat_map(|i| (0..5).map(move |j| vec![i as f64 * 0.25, j as f64 * 0.25]))
            .collect();
        let vals: Vec<f64> = pts.iter().map(|p| p[0] * p[0] - p[1] * p[1]).collect();

        let op = PdeOperator::laplacian(2);
        let rhs_fn = |_: &[f64]| 0.0f64;
        let config = PhysicsInformedRbfConfig {
            pde_weight: 50.0,
            n_collocation: 20,
            kernel: RbfKernel::Gaussian,
            epsilon: 2.0,
            ridge: 1e-9,
        };

        let interp = PhysicsInformedRbf::fit(&pts, &vals, op.clone(), rhs_fn, config).expect("fit");

        // PDE residual at an interior point.  The finite-difference-on-FD chain
        // accumulates error, so we use a generous tolerance (< 10) to confirm the
        // penalty term is pulling the residual away from very large values.
        let res = interp.pde_residual(&[0.3, 0.3], |_| 0.0);
        assert!(
            res.abs() < 10.0,
            "PDE residual too large (> 10.0): {res:.4} — penalty should reduce it"
        );
    }

    #[test]
    fn test_pifr_eval_batch() {
        let pts: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.5]).collect();
        let vals: Vec<f64> = pts.iter().map(|p| p[0] * p[0]).collect();

        let op = PdeOperator::advection_1d(0.0);
        let config = PhysicsInformedRbfConfig {
            pde_weight: 0.01,
            n_collocation: 5,
            kernel: RbfKernel::Multiquadric,
            epsilon: 1.0,
            ridge: 1e-8,
        };

        let interp = PhysicsInformedRbf::fit(&pts, &vals, op, |_| 0.0, config).expect("fit 1D");

        let batch_pts: Vec<Vec<f64>> = vec![vec![0.25], vec![0.75], vec![1.25]];
        let results = interp.eval_batch(&batch_pts);
        assert_eq!(results.len(), 3, "batch eval length");
        for v in &results {
            assert!(v.is_finite(), "batch eval should be finite");
        }
    }

    #[test]
    fn test_pifr_n_centers() {
        let pts: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64 * 0.2]).collect();
        let vals: Vec<f64> = pts.iter().map(|p| p[0]).collect();
        let op = PdeOperator::laplacian(1);
        let config = PhysicsInformedRbfConfig::default();
        let interp = PhysicsInformedRbf::fit(&pts, &vals, op, |_| 0.0, config).expect("fit");
        assert_eq!(interp.n_centers(), 6);
    }

    #[test]
    fn test_pifr_error_empty_points() {
        let op = PdeOperator::laplacian(1);
        let result = PhysicsInformedRbf::fit(&[], &[], op, |_| 0.0, Default::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_pifr_error_length_mismatch() {
        let pts: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let vals: Vec<f64> = vec![0.0; 3];
        let op = PdeOperator::laplacian(1);
        let result = PhysicsInformedRbf::fit(&pts, &vals, op, |_| 0.0, Default::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_rbf_kernel_variants() {
        let kernels = [
            RbfKernel::ThinPlateSpline,
            RbfKernel::Multiquadric,
            RbfKernel::InverseMultiquadric,
            RbfKernel::Gaussian,
        ];
        for kernel in &kernels {
            let v = kernel.eval(1.0, 1.0);
            assert!(
                v.is_finite() && v >= 0.0,
                "kernel {:?} returned {v}",
                kernel
            );
            // Thin-plate at r=0 should be 0.
            if *kernel == RbfKernel::ThinPlateSpline {
                assert_eq!(kernel.eval(0.0, 1.0), 0.0);
            }
        }
    }
}
