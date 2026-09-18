//! Hessian matrix computation via finite differences.
//!
//! Computes `H[i,j] = ∂²f/∂x_i∂x_j` for scalar functions f: ℝ^n → ℝ.

use ndarray::{Array, ArrayD, IxDyn};

use super::jacobian::{JacobianConfig, JacobianError};

/// Statistics derived from an analysed Hessian matrix.
#[derive(Debug, Clone, Default)]
pub struct HessianStats {
    /// Whether the Hessian is symmetric within tolerance.
    pub is_symmetric: bool,
    /// Approximate positive-definiteness test (all diagonal entries > 0).
    pub is_pd_approx: bool,
    /// Maximum element-wise asymmetry `max|H[i,j] - H[j,i]|`.
    pub max_asymmetry: f64,
    /// Condition number approximation: max_diag / min_diag (diagonal dominance proxy).
    pub condition_approx: f64,
}

/// Computes Hessian matrices via finite differences.
pub struct HessianComputer {
    config: JacobianConfig,
}

impl HessianComputer {
    /// Create a new `HessianComputer` with the given configuration.
    pub fn new(config: JacobianConfig) -> Self {
        Self { config }
    }

    /// Compute `H[i,j] = ∂²f/∂x_i∂x_j` for scalar f: ℝ^n → ℝ.
    ///
    /// Uses the 4-point cross stencil for off-diagonal entries and the
    /// 3-point diagonal stencil for i == j:
    ///
    /// ```text
    /// H[i,j] (i≠j) = ( f(x+ei·ε+ej·ε) - f(x+ei·ε-ej·ε)
    ///                 - f(x-ei·ε+ej·ε) + f(x-ei·ε-ej·ε) ) / (4ε²)
    ///
    /// H[i,i]       = ( f(x+ei·ε) - 2f(x) + f(x-ei·ε) ) / ε²
    /// ```
    ///
    /// Returns an `[n, n]` matrix.
    pub fn compute<F>(&self, input: &ArrayD<f64>, f: F) -> Result<ArrayD<f64>, JacobianError>
    where
        F: Fn(&ArrayD<f64>) -> Result<f64, String>,
    {
        if self.config.epsilon <= 0.0 {
            return Err(JacobianError::InvalidEpsilon(self.config.epsilon));
        }
        if input.ndim() != 1 {
            return Err(JacobianError::NonFlatInput(input.shape().to_vec()));
        }
        if input.is_empty() {
            return Err(JacobianError::EmptyInput);
        }

        let n = input.len();
        let eps = self.config.epsilon;
        let eps2 = eps * eps;

        let f0 = f(input).map_err(JacobianError::EvalFailed)?;

        let mut hessian_flat = vec![0.0f64; n * n];

        for i in 0..n {
            // ── Diagonal: 3-point stencil ──────────────────────────────────
            let f_plus = {
                let mut x = input.clone();
                x[i] += eps;
                f(&x).map_err(JacobianError::EvalFailed)?
            };
            let f_minus = {
                let mut x = input.clone();
                x[i] -= eps;
                f(&x).map_err(JacobianError::EvalFailed)?
            };
            hessian_flat[i * n + i] = (f_plus - 2.0 * f0 + f_minus) / eps2;

            // ── Off-diagonal: 4-point cross stencil ───────────────────────
            for j in (i + 1)..n {
                let f_pp = {
                    let mut x = input.clone();
                    x[i] += eps;
                    x[j] += eps;
                    f(&x).map_err(JacobianError::EvalFailed)?
                };
                let f_pm = {
                    let mut x = input.clone();
                    x[i] += eps;
                    x[j] -= eps;
                    f(&x).map_err(JacobianError::EvalFailed)?
                };
                let f_mp = {
                    let mut x = input.clone();
                    x[i] -= eps;
                    x[j] += eps;
                    f(&x).map_err(JacobianError::EvalFailed)?
                };
                let f_mm = {
                    let mut x = input.clone();
                    x[i] -= eps;
                    x[j] -= eps;
                    f(&x).map_err(JacobianError::EvalFailed)?
                };
                let h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps2);
                hessian_flat[i * n + j] = h_ij;
                hessian_flat[j * n + i] = h_ij;
            }
        }

        let hessian = Array::from_shape_vec(IxDyn(&[n, n]), hessian_flat)
            .map_err(|e| JacobianError::EvalFailed(e.to_string()))?;

        Ok(hessian)
    }

    /// Compute statistics describing the Hessian's structure.
    pub fn analyze(hessian: &ArrayD<f64>, tol: f64) -> HessianStats {
        let is_symmetric = Self::check_symmetry(hessian, tol);
        let is_pd_approx = Self::check_positive_definite_approx(hessian);

        let n = hessian.shape()[0];
        let max_asymmetry = if n == 0 {
            0.0
        } else {
            let mut max_asym = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let asym = (hessian[[i, j]] - hessian[[j, i]]).abs();
                    if asym > max_asym {
                        max_asym = asym;
                    }
                }
            }
            max_asym
        };

        let condition_approx = if n == 0 {
            0.0
        } else {
            let diags: Vec<f64> = (0..n).map(|i| hessian[[i, i]].abs()).collect();
            let max_d = diags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_d = diags.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_d.abs() < 1e-15 {
                f64::INFINITY
            } else {
                max_d / min_d
            }
        };

        HessianStats {
            is_symmetric,
            is_pd_approx,
            max_asymmetry,
            condition_approx,
        }
    }

    /// Return true if `|H[i,j] - H[j,i]| <= tol` for all i, j.
    pub fn check_symmetry(hessian: &ArrayD<f64>, tol: f64) -> bool {
        if hessian.ndim() != 2 {
            return false;
        }
        let n = hessian.shape()[0];
        if hessian.shape()[1] != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                if (hessian[[i, j]] - hessian[[j, i]]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Approximate positive-definiteness test: all diagonal entries > 0.
    pub fn check_positive_definite_approx(hessian: &ArrayD<f64>) -> bool {
        if hessian.ndim() != 2 {
            return false;
        }
        let n = hessian.shape()[0];
        (0..n).all(|i| hessian[[i, i]] > 0.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn vec1d(data: &[f64]) -> ArrayD<f64> {
        Array::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).unwrap()
    }

    #[test]
    fn test_hessian_quadratic_bowl() {
        // f(x) = x0^2 + x1^2, H = 2*I
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0]);
        let h = comp
            .compute(&x, |v| Ok(v[0] * v[0] + v[1] * v[1]))
            .expect("hessian quadratic");
        assert!((h[[0, 0]] - 2.0).abs() < 1e-6, "H[0,0]={}", h[[0, 0]]);
        assert!((h[[1, 1]] - 2.0).abs() < 1e-6, "H[1,1]={}", h[[1, 1]]);
        assert!(h[[0, 1]].abs() < 1e-6, "H[0,1]={}", h[[0, 1]]);
        assert!(h[[1, 0]].abs() < 1e-6, "H[1,0]={}", h[[1, 0]]);
    }

    #[test]
    fn test_hessian_is_symmetric() {
        // Hessian of a smooth function should be (approximately) symmetric.
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0, 0.5]);
        let h = comp
            .compute(&x, |v| {
                Ok(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[0] * v[1])
            })
            .unwrap();
        assert!(
            HessianComputer::check_symmetry(&h, 1e-6),
            "Hessian should be symmetric"
        );
    }

    #[test]
    fn test_hessian_shape() {
        // H must be [n, n].
        for n in [2usize, 3, 5] {
            let comp = HessianComputer::new(JacobianConfig::default());
            let x = vec1d(&vec![1.0; n]);
            let h = comp
                .compute(&x, |v| Ok(v.iter().map(|a| a * a).sum()))
                .unwrap();
            assert_eq!(h.shape(), &[n, n], "shape for n={}", n);
        }
    }

    #[test]
    fn test_hessian_positive_definite_approx() {
        // f(x) = sum(x_i^2) is strictly convex, H = 2*I, all diag > 0.
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0, 3.0]);
        let h = comp
            .compute(&x, |v| Ok(v.iter().map(|a| a * a).sum()))
            .unwrap();
        assert!(HessianComputer::check_positive_definite_approx(&h));
    }

    #[test]
    fn test_hessian_diagonal_entries() {
        // f(x) = x0^3, H[0,0] = 6*x0 at x0=2 → 12
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[2.0]);
        let h = comp.compute(&x, |v| Ok(v[0].powi(3))).unwrap();
        assert!((h[[0, 0]] - 12.0).abs() < 1e-4, "H[0,0]={}", h[[0, 0]]);
    }

    #[test]
    fn test_hessian_cross_terms() {
        // f(x) = x0 * x1, H[0,0]=H[1,1]=0, H[0,1]=H[1,0]=1
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[2.0, 3.0]);
        let h = comp.compute(&x, |v| Ok(v[0] * v[1])).unwrap();
        assert!(h[[0, 0]].abs() < 1e-5, "H[0,0]={}", h[[0, 0]]);
        assert!(h[[1, 1]].abs() < 1e-5, "H[1,1]={}", h[[1, 1]]);
        assert!((h[[0, 1]] - 1.0).abs() < 1e-5, "H[0,1]={}", h[[0, 1]]);
        assert!((h[[1, 0]] - 1.0).abs() < 1e-5, "H[1,0]={}", h[[1, 0]]);
    }

    #[test]
    fn test_hessian_analyze() {
        let comp = HessianComputer::new(JacobianConfig::default());
        let x = vec1d(&[1.0, 2.0]);
        let h = comp.compute(&x, |v| Ok(v[0] * v[0] + v[1] * v[1])).unwrap();
        let stats = HessianComputer::analyze(&h, 1e-6);
        assert!(stats.is_symmetric);
        assert!(stats.is_pd_approx);
        assert!(stats.max_asymmetry < 1e-6);
        assert!((stats.condition_approx - 1.0).abs() < 1e-4);
    }
}
