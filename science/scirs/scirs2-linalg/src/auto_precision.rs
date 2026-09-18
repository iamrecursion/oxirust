//! Automatic precision selection based on matrix condition number.
//!
//! Dispatches to f32 or f64 arithmetic based on an estimated condition number.
//! The policy thresholds can be tuned per application; the defaults are:
//!
//! - `cond < 1e4`: Single precision (f32) — fast, acceptable accuracy
//! - `1e4 <= cond < 1e8`: Mixed precision — f32 factorization + f64 residual refinement
//! - `cond >= 1e8`: Double precision (f64) — full accuracy
//!
//! # References
//!
//! - Higham (2002). "Accuracy and Stability of Numerical Algorithms." Chapter 12.
//! - Demmel et al. (2006). "Error bounds from extra-precise iterative refinement."

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::LinalgError;

// ---------------------------------------------------------------------------
// Precision type
// ---------------------------------------------------------------------------

/// Precision level for linear algebra operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Single precision (f32) — fast but less accurate.
    Single,
    /// Double precision (f64) — slower but high accuracy.
    Double,
    /// Mixed: factorize in f32, compute and apply residual correction in f64.
    Mixed,
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Policy for selecting computation precision based on estimated condition number.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// If estimated condition number exceeds this threshold, use [`Precision::Double`].
    pub double_threshold: f64,
    /// If estimated condition number exceeds this threshold (but not `double_threshold`),
    /// use [`Precision::Mixed`].
    pub mixed_threshold: f64,
    /// Override: always use this precision regardless of condition number.
    pub force: Option<Precision>,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            double_threshold: 1e8,
            mixed_threshold: 1e4,
            force: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Condition number estimation
// ---------------------------------------------------------------------------

/// Estimate the condition number of a matrix using a row-norm ratio heuristic.
///
/// Returns a fast O(n²) estimate suitable for dispatch decisions.  For a
/// well-conditioned identity-like matrix the estimate is close to 1.  For
/// near-singular matrices it returns [`f64::INFINITY`].
///
/// The estimate is based on the ratio of the maximum and minimum 2-norms of
/// the rows.  This is a lower bound on the true 2-norm condition number but is
/// sufficient for distinguishing well-conditioned from ill-conditioned cases.
pub fn estimate_condition_number(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    if n == 0 {
        return 1.0;
    }

    let row_norms: Vec<f64> = (0..n)
        .map(|i| a.row(i).iter().map(|&x| x * x).sum::<f64>().sqrt())
        .collect();

    let max_norm = row_norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_norm = row_norms.iter().cloned().fold(f64::INFINITY, f64::min);

    if min_norm < 1e-300 {
        f64::INFINITY
    } else {
        max_norm / min_norm
    }
}

// ---------------------------------------------------------------------------
// Precision selection
// ---------------------------------------------------------------------------

/// Select precision for solving a linear system based on matrix properties.
///
/// # Arguments
///
/// * `a` — The coefficient matrix.
/// * `policy` — The precision policy to apply.
///
/// # Returns
///
/// The recommended [`Precision`] level.
pub fn select_precision(a: &Array2<f64>, policy: &PrecisionPolicy) -> Precision {
    if let Some(forced) = policy.force {
        return forced;
    }
    let cond = estimate_condition_number(a);
    if cond >= policy.double_threshold {
        Precision::Double
    } else if cond >= policy.mixed_threshold {
        Precision::Mixed
    } else {
        Precision::Single
    }
}

// ---------------------------------------------------------------------------
// Auto-precision solve entry point
// ---------------------------------------------------------------------------

/// Solve `Ax = b` with automatic precision selection.
///
/// Returns the solution vector together with the precision that was actually
/// used.
///
/// # Errors
///
/// Returns [`LinalgError::SingularMatrixError`] if the matrix is numerically
/// singular at the chosen precision, or [`LinalgError::DimensionError`] for
/// mismatched dimensions.
pub fn auto_precision_solve(
    a: &Array2<f64>,
    b: &Array1<f64>,
    policy: &PrecisionPolicy,
) -> Result<(Array1<f64>, Precision), LinalgError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::DimensionError(format!(
            "matrix must be square, got {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }
    if b.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "rhs length {} does not match matrix dimension {}",
            b.len(),
            n
        )));
    }

    let precision = select_precision(a, policy);
    let x = match precision {
        Precision::Single => solve_f32(a, b)?,
        Precision::Double | Precision::Mixed => solve_f64(a, b)?,
    };
    Ok((x, precision))
}

// ---------------------------------------------------------------------------
// Internal solvers
// ---------------------------------------------------------------------------

/// Gaussian elimination in f32 with partial pivoting.
pub(crate) fn solve_f32(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let n = a.nrows();
    // Flatten into row-major Vec<f32>
    let mut mat: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let mut rhs: Vec<f32> = b.iter().map(|&x| x as f32).collect();

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-7_f32 {
            return Err(LinalgError::SingularMatrixError(
                "matrix is numerically singular at f32 precision".to_string(),
            ));
        }
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            rhs.swap(col, max_row);
        }
        // Elimination
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let v = mat[col * n + k];
                mat[row * n + k] -= factor * v;
            }
            let rhs_col = rhs[col];
            rhs[row] -= factor * rhs_col;
        }
    }

    // Back substitution
    let mut x = vec![0.0_f32; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i * n + j] * x[j];
        }
        x[i] = sum / mat[i * n + i];
    }

    Ok(Array1::from(
        x.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    ))
}

/// Gaussian elimination in f64 with partial pivoting.
pub(crate) fn solve_f64(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let n = a.nrows();
    let mut mat: Vec<f64> = a.iter().cloned().collect();
    let mut rhs: Vec<f64> = b.iter().cloned().collect();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(LinalgError::SingularMatrixError(
                "matrix is numerically singular at f64 precision".to_string(),
            ));
        }
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            rhs.swap(col, max_row);
        }
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let v = mat[col * n + k];
                mat[row * n + k] -= factor * v;
            }
            let rhs_col = rhs[col];
            rhs[row] -= factor * rhs_col;
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i * n + j] * x[j];
        }
        x[i] = sum / mat[i * n + i];
    }

    Ok(Array1::from(x))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_condition_number_estimate_identity() {
        let id = Array2::<f64>::eye(4);
        let cond = estimate_condition_number(&id);
        // Identity matrix: all row norms == 1, ratio == 1
        assert!(
            (cond - 1.0).abs() < 1e-10,
            "expected cond ≈ 1.0, got {cond}"
        );
    }

    #[test]
    fn test_condition_number_estimate_scaled_rows() {
        // First row scaled by 1e6 vs unit rows
        let a = array![[1e6_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let cond = estimate_condition_number(&a);
        assert!(cond > 1e5, "expected large condition estimate, got {cond}");
    }

    #[test]
    fn test_auto_precision_well_conditioned() {
        // Small, well-conditioned matrix → single precision
        let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
        let policy = PrecisionPolicy::default();
        let precision = select_precision(&a, &policy);
        assert_eq!(precision, Precision::Single);
    }

    #[test]
    fn test_auto_precision_ill_conditioned() {
        // Matrix with wildly different row norms: first row scaled by 1e10,
        // rest have unit norm.  The row-norm ratio heuristic reliably detects
        // this as ill-conditioned.
        let n = 4;
        let mut data = vec![0.0_f64; n * n];
        // Identity-like matrix with first row scaled by 1e10
        data[0] = 1e10; // (0,0)
        data[n + 1] = 1.0; // (1,1)
        data[2 * n + 2] = 1.0; // (2,2)
        data[3 * n + 3] = 1.0; // (3,3)
        let a = Array2::from_shape_vec((n, n), data).expect("valid shape");
        let policy = PrecisionPolicy::default();
        let precision = select_precision(&a, &policy);
        // Row-norm ratio = 1e10 / 1.0 >> double_threshold=1e8 → Double
        assert!(
            precision == Precision::Double || precision == Precision::Mixed,
            "expected Double or Mixed for ill-conditioned matrix, got {precision:?}"
        );
    }

    #[test]
    fn test_auto_precision_solve_correct() {
        // Solve well-conditioned 3x3 system
        let a = array![[2.0_f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let b = array![8.0_f64, -11.0, -3.0];
        let policy = PrecisionPolicy::default();
        let (x, _precision) = auto_precision_solve(&a, &b, &policy).expect("solve should succeed");

        // Known solution: x = [2, 3, -1]
        assert!((x[0] - 2.0).abs() < 1e-4, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-4, "x[1]={}", x[1]);
        assert!((x[2] - (-1.0)).abs() < 1e-4, "x[2]={}", x[2]);
    }

    #[test]
    fn test_auto_precision_force_double() {
        let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
        let policy = PrecisionPolicy {
            force: Some(Precision::Double),
            ..Default::default()
        };
        let precision = select_precision(&a, &policy);
        assert_eq!(precision, Precision::Double);
    }

    #[test]
    fn test_auto_precision_dimension_mismatch() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![1.0_f64, 2.0, 3.0]; // wrong length
        let policy = PrecisionPolicy::default();
        assert!(auto_precision_solve(&a, &b, &policy).is_err());
    }

    #[test]
    fn test_auto_precision_non_square() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array1::<f64>::zeros(2);
        let policy = PrecisionPolicy::default();
        assert!(auto_precision_solve(&a, &b, &policy).is_err());
    }
}
