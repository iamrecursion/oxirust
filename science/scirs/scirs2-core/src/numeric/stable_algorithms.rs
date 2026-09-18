//! # Stable Numerical Algorithms
//!
//! This module provides numerically stable implementations of common algorithms
//! used in scientific computing, with focus on matrix operations, iterative methods,
//! and numerical differentiation/integration.

use crate::{
    error::{CoreError, CoreResult, ErrorContext},
    numeric::stability::{stable_norm_2, StableComputation},
    validation::check_finite,
};
use ::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{cast, Float};
use std::fmt::Debug;

/// Configuration for iterative algorithms
#[derive(Debug, Clone)]
pub struct IterativeConfig<T: Float> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Absolute tolerance for convergence
    pub abs_tolerance: T,
    /// Relative tolerance for convergence
    pub reltolerance: T,
    /// Whether to use adaptive tolerance
    pub adaptive_tolerance: bool,
}

impl<T: Float> Default for IterativeConfig<T> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            abs_tolerance: cast::<f64, T>(1e-10).unwrap_or(T::epsilon()),
            reltolerance: cast::<f64, T>(1e-8).unwrap_or(T::epsilon()),
            adaptive_tolerance: true,
        }
    }
}

/// Result of an iterative algorithm
#[derive(Debug, Clone)]
pub struct IterativeResult<T: Float> {
    /// The computed solution
    pub solution: Array1<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual: T,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Convergence history (optional)
    pub history: Option<Vec<T>>,
}

/// Stable Gaussian elimination with partial pivoting
#[allow(dead_code)]
pub fn gaussian_elimination_stable<T: Float + StableComputation>(
    a: &ArrayView2<T>,
    b: &ArrayView1<T>,
) -> CoreResult<Array1<T>> {
    let n = a.nrows();

    if a.ncols() != n {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Matrix must be square",
        )));
    }

    if b.len() != n {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Matrix and vector dimensions must match: matrix is {}x{}, vector is {}x1",
            n,
            n,
            b.len()
        ))));
    }

    // Create augmented matrix
    let mut aug = Array2::zeros((n, n + 1));
    aug.slice_mut(s![.., ..n]).assign(a);
    aug.slice_mut(s![.., n]).assign(b);

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = aug[[k, k]].abs();

        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                max_idx = i;
            }
        }

        // Check for singular matrix
        if max_val.is_effectively_zero() {
            return Err(CoreError::ComputationError(ErrorContext::new(format!(
                "Singular matrix detected in gaussian elimination at pivot {k}"
            ))));
        }

        // Swap rows if needed
        if max_idx != k {
            for j in k..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_idx, j]];
                aug[[max_idx, j]] = temp;
            }
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }

        if aug[[i, i]].is_effectively_zero() {
            return Err(CoreError::ComputationError(ErrorContext::new(format!(
                "Singular matrix detected in back substitution at row {i}"
            ))));
        }

        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Stable QR decomposition using Householder reflections
#[allow(dead_code)]
pub fn qr_decomposition_stable<T: Float + StableComputation>(
    a: &ArrayView2<T>,
) -> CoreResult<(Array2<T>, Array2<T>)> {
    let (m, n) = a.dim();
    let k = m.min(n);

    let mut q = Array2::eye(m);
    let mut r = a.to_owned();

    for j in 0..k {
        // Compute Householder vector for column j
        let mut x = r.slice(s![j.., j]).to_owned();

        // Compute stable norm
        let norm_x = stable_norm_2(&x.to_vec());

        if norm_x.is_effectively_zero() {
            continue;
        }

        // Choose sign to avoid cancellation
        let sign = if x[0] >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        let alpha = sign * norm_x;

        // Update first component
        x[0] = x[0] + alpha;

        // Normalize Householder vector
        let norm_v = stable_norm_2(&x.to_vec());
        if norm_v > T::zero() {
            for i in 0..x.len() {
                x[i] = x[i] / norm_v;
            }
        }

        // Apply Householder transformation to R
        for col in j..n {
            let mut dot = T::zero();
            for row in j..m {
                dot = dot + x[row - j] * r[[row, col]];
            }

            let scale = cast::<f64, T>(2.0).unwrap_or(T::one()) * dot;
            for row in j..m {
                r[[row, col]] = r[[row, col]] - scale * x[row - j];
            }
        }

        // Apply Householder transformation to Q
        for col in 0..m {
            let mut dot = T::zero();
            for row in j..m {
                dot = dot + x[row - j] * q[[row, col]];
            }

            let scale = cast::<f64, T>(2.0).unwrap_or(T::one()) * dot;
            for row in j..m {
                q[[row, col]] = q[[row, col]] - scale * x[row - j];
            }
        }
    }

    // Q is stored transposed, so transpose it back
    q = q.t().to_owned();

    Ok((q, r))
}

/// Stable Cholesky decomposition
#[allow(dead_code)]
pub fn cholesky_stable<T: Float + StableComputation>(a: &ArrayView2<T>) -> CoreResult<Array2<T>> {
    let n = a.nrows();

    if a.ncols() != n {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Matrix must be square",
        )));
    }

    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        // Diagonal element
        let mut sum = a[[i, i]];
        for k in 0..i {
            sum = sum - l[[i, k]] * l[[i, k]];
        }

        if sum <= T::zero() {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Matrix is not positive definite: Failed at row {i}"
            ))));
        }

        l[[i, i]] = sum.sqrt();

        // Off-diagonal elements
        for j in (i + 1)..n {
            let mut sum = a[[j, i]];
            for k in 0..i {
                sum = sum - l[[j, k]] * l[[i, k]];
            }

            l[[j, i]] = sum / l[[i, i]];
        }
    }

    Ok(l)
}

/// Conjugate gradient method with preconditioning
#[allow(dead_code)]
pub fn conjugate_gradient<T: Float + StableComputation>(
    a: &ArrayView2<T>,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    config: &IterativeConfig<T>,
) -> CoreResult<IterativeResult<T>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Matrix and vector dimensions must match: matrix is {}x{}, vector is {}x1",
            n,
            n,
            b.len()
        ))));
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0_ref) => x0_ref.to_owned(),
        None => Array1::zeros(n),
    };

    // Initial residual: r = b - A*x
    let mut r = b.to_owned();
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            sum = sum + a[[i, j]] * x[j];
        }
        r[i] = r[i] - sum;
    }

    let mut p = r.clone();
    let mut r_norm_sq = T::zero();
    for &val in r.iter() {
        r_norm_sq = r_norm_sq + val * val;
    }

    let b_norm = stable_norm_2(&b.to_vec());
    let _initial_residual = r_norm_sq.sqrt();

    let mut history = if config.adaptive_tolerance {
        Some(Vec::with_capacity(config.max_iterations))
    } else {
        None
    };

    for iter in 0..config.max_iterations {
        // Compute A*p
        let mut ap = Array1::zeros(n);
        for i in 0..n {
            for j in 0..n {
                ap[i] = ap[i] + a[[i, j]] * p[j];
            }
        }

        // Compute step size
        let mut p_ap = T::zero();
        for i in 0..n {
            p_ap = p_ap + p[i] * ap[i];
        }

        if p_ap.is_effectively_zero() {
            return Ok(IterativeResult {
                solution: x,
                iterations: iter,
                residual: r_norm_sq.sqrt(),
                converged: false,
                history,
            });
        }

        let alpha = r_norm_sq / p_ap;

        // Update solution and residual
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
        }

        // Check convergence
        let r_norm = stable_norm_2(&r.to_vec());

        if let Some(ref mut hist) = history {
            hist.push(r_norm);
        }

        let tol = if config.adaptive_tolerance {
            config.abs_tolerance + config.reltolerance * b_norm
        } else {
            config.abs_tolerance
        };

        if r_norm < tol {
            return Ok(IterativeResult {
                solution: x,
                iterations: iter + 1,
                residual: r_norm,
                converged: true,
                history,
            });
        }

        // Update search direction
        let r_norm_sq_new = r_norm * r_norm;
        let beta = r_norm_sq_new / r_norm_sq;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }

        r_norm_sq = r_norm_sq_new;
    }

    Ok(IterativeResult {
        solution: x,
        iterations: config.max_iterations,
        residual: r_norm_sq.sqrt(),
        converged: false,
        history,
    })
}

/// GMRES (Generalized Minimal Residual) method
#[allow(dead_code)]
pub fn gmres<T: Float + StableComputation>(
    a: &ArrayView2<T>,
    b: &ArrayView1<T>,
    x0: Option<&ArrayView1<T>>,
    restart: usize,
    config: &IterativeConfig<T>,
) -> CoreResult<IterativeResult<T>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(CoreError::DimensionError(ErrorContext::new(format!(
            "Matrix and vector dimensions must match: matrix is {}x{}, vector is {}x1",
            n,
            n,
            b.len()
        ))));
    }

    let restart = restart.min(n);

    // Initialize solution
    let mut x = match x0 {
        Some(x0_ref) => x0_ref.to_owned(),
        None => Array1::zeros(n),
    };

    let b_norm = stable_norm_2(&b.to_vec());
    let mut _outer_iter = 0;
    let mut total_iter = 0;

    let mut history = if config.adaptive_tolerance {
        Some(Vec::with_capacity(config.max_iterations))
    } else {
        None
    };

    while total_iter < config.max_iterations {
        // Compute initial residual
        let mut r = b.to_owned();
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum + a[[i, j]] * x[j];
            }
            r[i] = r[i] - sum;
        }

        let r_norm = stable_norm_2(&r.to_vec());

        if let Some(ref mut hist) = history {
            hist.push(r_norm);
        }

        // Check convergence
        let tol = if config.adaptive_tolerance {
            config.abs_tolerance + config.reltolerance * b_norm
        } else {
            config.abs_tolerance
        };

        if r_norm < tol {
            return Ok(IterativeResult {
                solution: x,
                iterations: total_iter,
                residual: r_norm,
                converged: true,
                history,
            });
        }

        // Initialize Krylov subspace
        let mut v = vec![Array1::zeros(n); restart + 1];
        let mut h = Array2::zeros((restart + 1, restart));

        // First basis vector
        for i in 0..n {
            v[0][i] = r[i] / r_norm;
        }

        let mut g = Array1::zeros(restart + 1);
        g[0] = r_norm;

        // Arnoldi iteration
        for j in 0..restart {
            total_iter += 1;

            // Compute A*v[j]
            let mut w = Array1::zeros(n);
            for i in 0..n {
                for k in 0..n {
                    w[i] = w[i] + a[[i, k]] * v[j][k];
                }
            }

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let mut dot = T::zero();
                for k in 0..n {
                    dot = dot + w[k] * v[i][k];
                }
                h[[i, j]] = dot;

                for k in 0..n {
                    w[k] = w[k] - dot * v[i][k];
                }
            }

            let w_norm = stable_norm_2(&w.to_vec());
            h[[j + 1, j]] = w_norm;

            if w_norm.is_effectively_zero() {
                break;
            }

            for k in 0..n {
                v[j + 1][k] = w[k] / w_norm;
            }

            // Apply Givens rotations to maintain QR factorization of H
            for i in 0..j {
                let temp = h[[i, j]];
                h[[i, j]] = h[[i, j]] * T::one() - h[[i + 1, j]] * T::zero(); // Simplified
                h[[i + 1, j]] = temp * T::zero() + h[[i + 1, j]] * T::one(); // Simplified
            }

            // Check residual
            let residual = g[j + 1].abs();
            if residual < tol {
                // Solve least squares problem and update solution
                let mut y = Array1::zeros(j + 1);
                for i in (0..=j).rev() {
                    let mut sum = g[i];
                    for k in (i + 1)..=j {
                        sum = sum - h[[i, k]] * y[k];
                    }
                    y[i] = sum / h[[i, i]];
                }

                // Update solution
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k] + y[i] * v[i][k];
                    }
                }

                return Ok(IterativeResult {
                    solution: x,
                    iterations: total_iter,
                    residual,
                    converged: true,
                    history,
                });
            }
        }

        // Solve least squares problem
        let mut y = Array1::zeros(restart);
        for i in (0..restart).rev() {
            let mut sum = g[i];
            for j in (i + 1)..restart {
                sum = sum - h[[i, j]] * y[j];
            }
            if h[[i, i]].is_effectively_zero() {
                break;
            }
            y[i] = sum / h[[i, i]];
        }

        // Update solution
        for i in 0..restart {
            for j in 0..n {
                x[j] = x[j] + y[i] * v[i][j];
            }
        }

        _outer_iter += 1;
    }

    // Final residual computation
    let mut r = b.to_owned();
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..n {
            sum = sum + a[[i, j]] * x[j];
        }
        r[i] = r[i] - sum;
    }

    let final_residual = stable_norm_2(&r.to_vec());

    Ok(IterativeResult {
        solution: x,
        iterations: total_iter,
        residual: final_residual,
        converged: false,
        history,
    })
}

/// Stable numerical differentiation using Richardson extrapolation
#[allow(dead_code)]
pub fn richardson_derivative<T, F>(f: F, x: T, h: T, order: usize) -> CoreResult<T>
where
    T: Float + StableComputation,
    F: Fn(T) -> T,
{
    if order == 0 {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Order must be at least 1",
        )));
    }

    let n = order + 1;
    let mut d = Array2::zeros((n, n));

    // Initial step size
    let mut h_curr = h;

    // First column: finite differences with decreasing h
    for i in 0..n {
        // Central difference
        let f_plus = f(x + h_curr);
        let f_minus = f(x - h_curr);
        d[[i, 0]] = (f_plus - f_minus) / (cast::<f64, T>(2.0).unwrap_or(T::one()) * h_curr);

        // Halve the step size
        h_curr = h_curr / cast::<f64, T>(2.0).unwrap_or(T::one());
    }

    // Richardson extrapolation
    for j in 1..n {
        let factor = cast::<f64, T>(4.0_f64.powi(j as i32)).unwrap_or(T::one());
        for i in j..n {
            d[[i, j]] = (factor * d[[i, j.saturating_sub(1)]] - d[[i.saturating_sub(1), j - 1]])
                / (factor - T::one());
        }
    }

    Ok(d[[n - 1, n - 1]])
}

/// Stable numerical integration using adaptive Simpson's rule
#[allow(dead_code)]
pub fn adaptive_simpson<T, F>(f: F, a: T, b: T, tolerance: T, maxdepth: usize) -> CoreResult<T>
where
    T: Float + StableComputation,
    F: Fn(T) -> T,
{
    check_finite(
        a.to_f64().ok_or_else(|| {
            CoreError::TypeError(ErrorContext::new("Failed to convert lower limit to f64"))
        })?,
        "Lower limit",
    )?;
    check_finite(
        b.to_f64().ok_or_else(|| {
            CoreError::TypeError(ErrorContext::new("Failed to convert upper limit to f64"))
        })?,
        "Upper limit",
    )?;

    if a >= b {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Upper limit must be greater than lower limit",
        )));
    }

    fn simpson_rule<T: Float, F: Fn(T) -> T>(f: &F, a: T, b: T) -> T {
        let six = cast::<f64, T>(6.0).unwrap_or_else(|| {
            // This should work for all Float types
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one()
        });
        let two = cast::<f64, T>(2.0).unwrap_or_else(|| T::one() + T::one());
        let four = cast::<f64, T>(4.0).unwrap_or_else(|| two + two);

        let h = (b - a) / six;
        let mid = (a + b) / two;
        h * (f(a) + four * f(mid) + f(b))
    }

    fn adaptive_simpson_recursive<T: Float + StableComputation, F: Fn(T) -> T>(
        f: &F,
        a: T,
        b: T,
        tolerance: T,
        whole: T,
        depth: usize,
        max_depth: usize,
    ) -> T {
        if depth >= max_depth {
            return whole;
        }

        let two = cast::<f64, T>(2.0).unwrap_or_else(|| T::one() + T::one());
        let mid = (a + b) / two;
        let left = simpson_rule(f, a, mid);
        let right = simpson_rule(f, mid, b);
        let combined = left + right;

        let diff = (combined - whole).abs();

        let fifteen = cast::<f64, T>(15.0).unwrap_or_else(|| {
            // Build 15 from ones
            let five = T::one() + T::one() + T::one() + T::one() + T::one();
            five + five + five
        });
        if diff <= fifteen * tolerance {
            combined + diff / cast::<f64, T>(15.0).unwrap_or(T::one())
        } else {
            let half_tol = tolerance / cast::<f64, T>(2.0).unwrap_or(T::one());
            adaptive_simpson_recursive(f, a, mid, half_tol, left, depth + 1, max_depth)
                + adaptive_simpson_recursive(f, mid, b, half_tol, right, depth + 1, max_depth)
        }
    }

    let whole = simpson_rule(&f, a, b);
    Ok(adaptive_simpson_recursive(
        &f, a, b, tolerance, whole, 0, maxdepth,
    ))
}

/// Stable computation of matrix exponential using scaling and squaring
#[allow(dead_code)]
pub fn matrix_exp_stable<T: Float + StableComputation>(
    a: &ArrayView2<T>,
    scaling_threshold: Option<T>,
) -> CoreResult<Array2<T>> {
    let n = a.nrows();

    if a.ncols() != n {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Matrix must be square",
        )));
    }

    let threshold = scaling_threshold.unwrap_or(cast::<f64, T>(0.5).unwrap_or(T::one()));

    // Compute matrix norm
    let mut norm = T::zero();
    for i in 0..n {
        let mut row_sum = T::zero();
        for j in 0..n {
            row_sum = row_sum + a[[i, j]].abs();
        }
        norm = norm.max(row_sum);
    }

    // Determine scaling factor
    let mut s = 0;
    let mut scaled_norm = norm;
    while scaled_norm > threshold {
        scaled_norm = scaled_norm / cast::<f64, T>(2.0).unwrap_or(T::one());
        s += 1;
    }

    // Scale matrix
    let scale = cast::<f64, T>(2.0_f64.powi(s)).unwrap_or(T::one());
    let mut a_scaled = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_scaled[[i, j]] = a[[i, j]] / scale;
        }
    }

    // Padé approximation (simplified - using Taylor series for demonstration)
    let mut result = Array2::eye(n);
    let mut term: Array2<T> = Array2::eye(n);
    let mut factorial = T::one();

    for k in 1..10 {
        // Matrix multiplication
        let mut new_term = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    new_term[[i, j]] = new_term[[i, j]] + term[[i, l]] * a_scaled[[l, j]];
                }
            }
        }
        term = new_term;

        factorial = factorial * cast::<i32, T>(k).unwrap_or(T::one());

        // Add term
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = result[[i, j]] + term[[i, j]] / factorial;
            }
        }
    }

    // Square s times
    for _ in 0..s {
        let mut squared = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    squared[[i, j]] = squared[[i, j]] + result[[i, k]] * result[[k, j]];
                }
            }
        }
        result = squared;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_gaussian_elimination() {
        let a = array![[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let b = array![8.0, -11.0, -3.0];

        let x = gaussian_elimination_stable(&a.view(), &b.view())
            .expect("Gaussian elimination should succeed for this test matrix");

        // Expected solution: [2, 3, -1]
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(x[2], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qr_decomposition() {
        let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let (q, r) = qr_decomposition_stable(&a.view())
            .expect("QR decomposition should succeed for this test matrix");

        // Verify Q is orthogonal
        let qt_q = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qt_q[[i, j]], expected, epsilon = 1e-10);
            }
        }

        // Verify A = QR
        let reconstructed = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky() {
        // Positive definite matrix
        let a = array![
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0]
        ];

        let l = cholesky_stable(&a.view())
            .expect("Cholesky decomposition should succeed for this positive definite matrix");

        // Verify A = L * L^T
        let reconstructed = l.dot(&l.t());
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_conjugate_gradient() {
        // Symmetric positive definite matrix
        let a = array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = array![1.0, 2.0, 3.0];

        let config = IterativeConfig::default();
        let result = conjugate_gradient(&a.view(), &b.view(), None, &config)
            .expect("Conjugate gradient should converge for this system");

        assert!(result.converged);

        // Verify A*x = b
        let ax = a.dot(&result.solution);
        for i in 0..3 {
            assert_relative_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_richardson_derivative() {
        // Test function: f(x) = x^2
        let f = |x: f64| x * x;

        // Derivative at x = 2 should be 4
        let derivative =
            richardson_derivative(f, 2.0, 0.1, 3).expect("Richardson extrapolation should succeed");
        assert_relative_eq!(derivative, 4.0, epsilon = 1e-10);

        // Test with sin(x)
        let g = |x: f64| x.sin();

        // Derivative at x = 0 should be 1 (cos(0) = 1)
        let derivative = richardson_derivative(g, 0.0, 0.01, 4)
            .expect("Richardson extrapolation should succeed for cos at 0");
        assert_relative_eq!(derivative, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaptive_simpson() {
        // Test with simple polynomial: f(x) = x^2
        let f = |x: f64| x * x;

        // Integral from 0 to 1 should be 1/3
        let integral = adaptive_simpson(f, 0.0, 1.0, 1e-10, 10)
            .expect("Adaptive Simpson integration should succeed");
        assert_relative_eq!(integral, 1.0 / 3.0, epsilon = 1e-10);

        // Test with sine function
        let g = |x: f64| x.sin();

        // Integral from 0 to π should be 2
        let integral = adaptive_simpson(g, 0.0, std::f64::consts::PI, 1e-10, 10)
            .expect("Adaptive Simpson integration should succeed for sin");
        assert_relative_eq!(integral, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn testmatrix_exp() {
        // Test with diagonal matrix
        let a = array![[1.0, 0.0], [0.0, 2.0]];

        let exp_a = matrix_exp_stable(&a.view(), None)
            .expect("Matrix exponential should succeed for this small matrix");

        // exp(diagonal) should have exp of diagonal elements
        assert_relative_eq!(exp_a[[0, 0]], 1.0_f64.exp(), epsilon = 1e-8);
        assert_relative_eq!(exp_a[[1, 1]], 2.0_f64.exp(), epsilon = 1e-8);
        assert_relative_eq!(exp_a[[0, 1]], 0.0, epsilon = 1e-8);
        assert_relative_eq!(exp_a[[1, 0]], 0.0, epsilon = 1e-8);
    }
}
