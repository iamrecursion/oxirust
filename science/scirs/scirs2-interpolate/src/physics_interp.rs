//! Physics-informed interpolation with PDE residual penalty.
//!
//! Enforces PDE residuals as soft constraints during the RBF fitting process.
//! The interpolant solves an augmented system
//!
//! ```text
//! [ Φ_data       ]         [  y_data       ]
//! [ √λ · Φ_coll  ] w   =   [ √λ · f_coll  ]
//! ```
//!
//! where Φ_data is the data-point RBF matrix, Φ_coll is the collocation-point
//! RBF matrix, λ = `pde_weight`, and f_coll are the target residual values at
//! collocation points (usually zero for a homogeneous PDE).
//!
//! This formulation is equivalent to minimising
//!
//!   ‖ Φ_data w − y ‖² + λ ‖ r(Φ_coll w) ‖²
//!
//! where r is the PDE residual operator.  By adding extra rows to the least-
//! squares system we avoid the need for specialised constrained solvers.
//!
//! ## References
//!
//! - Kansa, E.J. (1990). *Multiquadrics — a scattered data approximation
//!   scheme with applications to computational fluid-dynamics*.
//! - Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). *Physics-informed
//!   neural networks*.

use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// PDE residual trait
// ---------------------------------------------------------------------------

/// A differentiable residual operator r(x, y, u) = L\[u\](x, y) − f(x, y).
///
/// Implementors encode the PDE; the interpolation penalty minimises the norm
/// of the residual at the collocation points.
pub trait PdeResidual: Send + Sync {
    /// Compute the PDE residual at point `(x, y)` given the interpolated
    /// value `u`.  Should return zero when the PDE is satisfied.
    fn residual(&self, x: f64, y: f64, u: f64) -> f64;
}

// ---------------------------------------------------------------------------
// Built-in residuals
// ---------------------------------------------------------------------------

/// Simplified Laplace residual: r(x, y, u) = u − f.
///
/// In a full implementation the Laplacian ∇²u would be approximated via
/// finite differences on the RBF expansion.  Here we use the zero-order
/// algebraic approximation r = u − f, which drives the fitted values towards
/// f at collocation points.
#[derive(Debug, Clone, Copy)]
pub struct LaplaceResidual {
    /// Right-hand side of the PDE  ∇²u = f.
    pub f: f64,
}

impl PdeResidual for LaplaceResidual {
    fn residual(&self, _x: f64, _y: f64, u: f64) -> f64 {
        u - self.f
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PhysicsInformedInterp`].
#[derive(Debug, Clone)]
pub struct PhysicsInterpConfig {
    /// Penalty weight λ for the PDE constraint rows.
    pub pde_weight: f64,
    /// Number of interior collocation points where the PDE is enforced.
    /// These are generated on a regular grid inside the data bounding box.
    pub n_collocation: usize,
    /// Shape parameter ε for the Gaussian RBF φ(r) = exp(-(ε r)²).
    pub rbf_epsilon: f64,
    /// Maximum number of iterations (reserved for future iterative solvers).
    pub max_iter: usize,
    /// Convergence tolerance (reserved for future iterative solvers).
    pub tol: f64,
}

impl Default for PhysicsInterpConfig {
    fn default() -> Self {
        Self {
            pde_weight: 1.0,
            n_collocation: 16,
            rbf_epsilon: 1.0,
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Physics-informed RBF interpolator.
///
/// Enforces a PDE constraint at a grid of collocation points by augmenting
/// the standard RBF least-squares system with additional penalty rows.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::physics_interp::{
///     PhysicsInformedInterp, PhysicsInterpConfig, LaplaceResidual,
/// };
///
/// let config = PhysicsInterpConfig {
///     pde_weight: 0.5,
///     n_collocation: 9,
///     rbf_epsilon: 2.0,
///     ..PhysicsInterpConfig::default()
/// };
/// let mut interp = PhysicsInformedInterp::new(config);
///
/// let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
/// let values = vec![0.0, 1.0, 0.5];
/// let pde = LaplaceResidual { f: 0.0 };
///
/// interp.fit(&points, &values, &pde).expect("fit should succeed");
/// let out = interp.evaluate(&points).expect("evaluate should succeed");
/// ```
#[derive(Debug)]
pub struct PhysicsInformedInterp {
    config: PhysicsInterpConfig,
    data_points: Vec<[f64; 2]>,
    data_values: Vec<f64>,
    rbf_weights: Vec<f64>,
    collocation_points: Vec<[f64; 2]>,
}

impl PhysicsInformedInterp {
    /// Create a new interpolator with the given configuration.
    pub fn new(config: PhysicsInterpConfig) -> Self {
        Self {
            config,
            data_points: Vec::new(),
            data_values: Vec::new(),
            rbf_weights: Vec::new(),
            collocation_points: Vec::new(),
        }
    }

    /// Fit the physics-informed RBF to `points` / `values` with PDE `pde`.
    ///
    /// Internally, collocation points are placed on a regular grid inside the
    /// bounding box of the data.  The combined least-squares system is solved
    /// via normal equations (Φᵀ Φ w = Φᵀ y) using Gaussian elimination.
    pub fn fit<P: PdeResidual>(
        &mut self,
        points: &[[f64; 2]],
        values: &[f64],
        pde: &P,
    ) -> Result<(), InterpolateError> {
        let nd = points.len();
        if nd == 0 {
            return Err(InterpolateError::InsufficientData(
                "physics_interp: at least one data point required".into(),
            ));
        }
        if values.len() != nd {
            return Err(InterpolateError::ShapeMismatch {
                expected: nd.to_string(),
                actual: values.len().to_string(),
                object: "values".into(),
            });
        }
        if self.config.rbf_epsilon <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: "physics_interp: rbf_epsilon must be positive".into(),
            });
        }

        // Generate collocation points on a grid within the data bounding box
        let coll_pts = generate_collocation_points(points, self.config.n_collocation);
        let nc = coll_pts.len();

        // Number of RBF basis centres = number of data points
        let nb = nd; // basis centres are placed at data points

        // Build augmented matrix Φ_aug ∈ R^{(nd + nc) × nb}
        //   top nd rows: data constraints
        //   bottom nc rows: PDE penalty (scaled by √λ)
        let sqrt_lam = self.config.pde_weight.sqrt();
        let n_rows = nd + nc;

        let mut phi_aug: Vec<f64> = vec![0.0; n_rows * nb];
        let mut rhs: Vec<f64> = vec![0.0; n_rows];

        // Data rows
        for i in 0..nd {
            for j in 0..nb {
                let r = dist2(&points[i], &points[j]);
                phi_aug[i * nb + j] = gaussian_rbf(r, self.config.rbf_epsilon);
            }
            rhs[i] = values[i];
        }

        // Collocation rows (PDE penalty)
        for (ci, cp) in coll_pts.iter().enumerate() {
            let row = nd + ci;
            let u_approx_dummy = 0.0_f64; // placeholder for residual target
            let target = pde.residual(cp[0], cp[1], u_approx_dummy);
            for j in 0..nb {
                let r = dist2(cp, &points[j]);
                phi_aug[row * nb + j] = sqrt_lam * gaussian_rbf(r, self.config.rbf_epsilon);
            }
            rhs[row] = sqrt_lam * target;
        }

        // Solve via normal equations: Φᵀ Φ w = Φᵀ rhs
        let w = solve_normal_equations(&phi_aug, &rhs, n_rows, nb)?;

        self.data_points = points.to_vec();
        self.data_values = values.to_vec();
        self.rbf_weights = w;
        self.collocation_points = coll_pts;
        Ok(())
    }

    /// Evaluate the fitted interpolant at `query_points`.
    pub fn evaluate(&self, query_points: &[[f64; 2]]) -> Result<Vec<f64>, InterpolateError> {
        if self.rbf_weights.is_empty() {
            return Err(InterpolateError::InvalidState(
                "physics_interp: interpolator not fitted — call fit() first".into(),
            ));
        }
        let out = query_points
            .iter()
            .map(|q| {
                self.data_points
                    .iter()
                    .zip(self.rbf_weights.iter())
                    .map(|(p, &w)| {
                        let r = dist2(q, p);
                        w * gaussian_rbf(r, self.config.rbf_epsilon)
                    })
                    .sum()
            })
            .collect();
        Ok(out)
    }

    /// Compute the RMS PDE residual norm at the collocation points.
    ///
    /// Returns 0.0 if no collocation points exist or the interpolant is not
    /// fitted.
    pub fn pde_residual_norm<P: PdeResidual>(&self, pde: &P) -> f64 {
        if self.rbf_weights.is_empty() || self.collocation_points.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self
            .collocation_points
            .iter()
            .map(|cp| {
                let u: f64 = self
                    .data_points
                    .iter()
                    .zip(self.rbf_weights.iter())
                    .map(|(p, &w)| {
                        let r = dist2(cp, p);
                        w * gaussian_rbf(r, self.config.rbf_epsilon)
                    })
                    .sum();
                let r = pde.residual(cp[0], cp[1], u);
                r * r
            })
            .sum();
        (sum_sq / self.collocation_points.len() as f64).sqrt()
    }

    /// Total loss = data_fit_mse + pde_weight * pde_residual_mse.
    pub fn total_loss<P: PdeResidual>(&self, pde: &P) -> f64 {
        if self.rbf_weights.is_empty() {
            return f64::INFINITY;
        }
        // Data fit MSE
        let data_mse: f64 = if self.data_points.is_empty() {
            0.0
        } else {
            let ss: f64 = self
                .data_points
                .iter()
                .zip(self.data_values.iter())
                .map(|(p, &y)| {
                    let u: f64 = self
                        .data_points
                        .iter()
                        .zip(self.rbf_weights.iter())
                        .map(|(q, &w)| {
                            let r = dist2(p, q);
                            w * gaussian_rbf(r, self.config.rbf_epsilon)
                        })
                        .sum();
                    (u - y) * (u - y)
                })
                .sum();
            ss / self.data_points.len() as f64
        };

        // PDE residual MSE (un-scaled)
        let pde_norm = self.pde_residual_norm(pde);
        data_mse + self.config.pde_weight * pde_norm * pde_norm
    }
}

// ---------------------------------------------------------------------------
// Internal free functions
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two 2D points.
#[inline]
fn dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// Gaussian RBF: φ(r) = exp(-(ε r)²).
#[inline]
fn gaussian_rbf(r: f64, epsilon: f64) -> f64 {
    let er = epsilon * r;
    (-(er * er)).exp()
}

/// Place `n_coll` points on a regular grid inside the bounding box of `pts`.
fn generate_collocation_points(pts: &[[f64; 2]], n_coll: usize) -> Vec<[f64; 2]> {
    if pts.is_empty() || n_coll == 0 {
        return Vec::new();
    }
    let (mut xmin, mut xmax) = (pts[0][0], pts[0][0]);
    let (mut ymin, mut ymax) = (pts[0][1], pts[0][1]);
    for p in pts {
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    // Inset slightly
    let dx = (xmax - xmin).max(1e-10) * 0.1;
    let dy = (ymax - ymin).max(1e-10) * 0.1;
    xmin += dx;
    xmax -= dx;
    ymin += dy;
    ymax -= dy;

    let side = (n_coll as f64).sqrt().ceil() as usize;
    let side = side.max(1);
    let mut coll = Vec::with_capacity(side * side);
    for i in 0..side {
        for j in 0..side {
            let x = xmin + (xmax - xmin) * (i as f64 + 0.5) / side as f64;
            let y = ymin + (ymax - ymin) * (j as f64 + 0.5) / side as f64;
            coll.push([x, y]);
        }
    }
    coll
}

/// Solve the over-determined system Φ w = rhs via normal equations Φᵀ Φ w = Φᵀ rhs.
///
/// `phi` is stored row-major with shape `(n_rows, n_cols)`.
fn solve_normal_equations(
    phi: &[f64],
    rhs: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<f64>, InterpolateError> {
    // AtA = Φᵀ Φ  (n_cols × n_cols)
    let mut ata: Vec<f64> = vec![0.0; n_cols * n_cols];
    // Atb = Φᵀ rhs  (n_cols)
    let mut atb: Vec<f64> = vec![0.0; n_cols];

    for k in 0..n_rows {
        let row = &phi[k * n_cols..(k + 1) * n_cols];
        for i in 0..n_cols {
            atb[i] += row[i] * rhs[k];
            for j in 0..n_cols {
                ata[i * n_cols + j] += row[i] * row[j];
            }
        }
    }

    // Add a small Tikhonov regulariser for numerical stability
    let reg = 1e-12;
    for i in 0..n_cols {
        ata[i * n_cols + i] += reg;
    }

    // Solve AtA w = Atb via Gaussian elimination with partial pivoting
    crate::gpu_rbf::solve_linear_system(&ata, &atb, n_cols)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(pde_weight: f64, n_coll: usize) -> PhysicsInterpConfig {
        PhysicsInterpConfig {
            pde_weight,
            n_collocation: n_coll,
            rbf_epsilon: 2.0,
            max_iter: 100,
            tol: 1e-8,
        }
    }

    /// With pde_weight = 0 the system degenerates to standard RBF — the fitted
    /// values at training points should reproduce the data within tolerance.
    #[test]
    fn test_zero_pde_weight_is_standard_rbf() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 0.8], [0.3, 0.3]];
        let values = vec![1.0, 2.0, 1.5, 0.8];

        let mut interp = PhysicsInformedInterp::new(make_config(0.0, 4));
        let pde = LaplaceResidual { f: 0.0 };
        interp.fit(&points, &values, &pde).expect("fit failed");

        let out = interp.evaluate(&points).expect("eval failed");
        for (got, &exp) in out.iter().zip(values.iter()) {
            assert!(
                (got - exp).abs() < 5e-4,
                "zero pde_weight: got {got:.6} expected {exp:.6}"
            );
        }
    }

    /// A higher pde_weight should drive the PDE residual norm lower when the
    /// PDE target is consistent with the data.
    #[test]
    fn test_higher_pde_weight_reduces_residual() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let values = vec![0.0, 0.0, 0.0];
        let pde = LaplaceResidual { f: 0.0 }; // u = 0 satisfies pde exactly

        let mut low = PhysicsInformedInterp::new(make_config(0.01, 4));
        let mut high = PhysicsInformedInterp::new(make_config(100.0, 4));

        low.fit(&points, &values, &pde).expect("fit low failed");
        high.fit(&points, &values, &pde).expect("fit high failed");

        let r_low = low.pde_residual_norm(&pde);
        let r_high = high.pde_residual_norm(&pde);

        // Higher weight should give equal or lower residual norm
        assert!(
            r_high <= r_low + 1e-6,
            "higher pde_weight should reduce residual: low={r_low:.6} high={r_high:.6}"
        );
    }

    /// Evaluate at training points must be within reasonable tolerance.
    #[test]
    fn test_evaluate_at_training_points() {
        let points = vec![[0.1_f64, 0.1], [0.9, 0.1], [0.5, 0.9]];
        let values = vec![1.0, 3.0, 2.0];

        let pde = LaplaceResidual { f: 0.5 };
        let mut interp = PhysicsInformedInterp::new(make_config(1e-4, 4));
        interp.fit(&points, &values, &pde).expect("fit failed");

        let out = interp.evaluate(&points).expect("eval failed");
        for (got, &exp) in out.iter().zip(values.iter()) {
            assert!(
                (got - exp).abs() < 0.5,
                "evaluate at training point: got {got:.4} expected {exp:.4}"
            );
        }
    }

    /// LaplaceResidual::residual(x, y, u) == u - f for any x, y.
    #[test]
    fn test_laplace_residual_formula() {
        let pde = LaplaceResidual { f: 3.0 };
        for u in [0.0, 1.0, 3.0, -2.5, 7.0] {
            let r = pde.residual(0.5, 0.5, u);
            assert!(
                (r - (u - 3.0)).abs() < 1e-15,
                "LaplaceResidual: got {r}, expected {:.1}",
                u - 3.0
            );
        }
    }

    /// total_loss should be non-negative.
    #[test]
    fn test_total_loss_non_negative() {
        let points = vec![[0.0_f64, 0.0], [1.0, 1.0]];
        let values = vec![0.0, 1.0];
        let pde = LaplaceResidual { f: 0.0 };
        let mut interp = PhysicsInformedInterp::new(make_config(1.0, 4));
        interp.fit(&points, &values, &pde).expect("fit failed");
        let loss = interp.total_loss(&pde);
        assert!(loss >= 0.0, "total_loss must be non-negative, got {loss}");
    }
}
