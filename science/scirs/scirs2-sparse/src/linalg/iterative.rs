#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

use crate::error::{SparseError, SparseResult};
use crate::linalg::interface::LinearOperator;
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::iter::Sum;

/// Result of an iterative solver
#[derive(Debug, Clone)]
pub struct IterationResult<F> {
    pub x: Vec<F>,
    pub iterations: usize,
    pub residual_norm: F,
    pub converged: bool,
    pub message: String,
}

/// Options for conjugate gradient solver
pub struct CGOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for CGOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            preconditioner: None,
        }
    }
}

/// Conjugate gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is symmetric positive definite
#[allow(dead_code)]
pub fn cg<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: CGOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for CG solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::sparse_zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Apply preconditioner if provided
    let mut z = if let Some(m) = &options.preconditioner {
        m.matvec(&r)?
    } else {
        r.clone()
    };

    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    // Check for convergence
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute Ap
        let ap = a.matvec(&p)?;

        // Compute alpha = (r,z) / (p,Ap)
        let pap = dot(&p, &ap);
        if pap <= F::sparse_zero() {
            return Ok(IterationResult {
                x,
                iterations,
                residual_norm: norm2(&r),
                converged: false,
                message: "Matrix not positive definite (p^T*A*p <= 0)".to_string(),
            });
        }
        let alpha = rz_old / pap;

        // Update solution: x = x + alpha*p
        for i in 0..n {
            x[i] += alpha * p[i];
        }

        // Update residual: r = r - alpha*Ap
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        // Check for convergence
        let rnorm = norm2(&r);
        if rnorm <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply preconditioner
        z = if let Some(m) = &options.preconditioner {
            m.matvec(&r)?
        } else {
            r.clone()
        };

        // Compute beta = (r_{i+1},z_{i+1}) / (r_i,z_i)
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;

        // Update direction: p = z + beta*p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }

        rz_old = rz_new;
        iterations += 1;
    }

    Ok(IterationResult {
        x,
        iterations,
        residual_norm: norm2(&r),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for BiCG solver
pub struct BiCGOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub left_preconditioner: Option<Box<dyn LinearOperator<F>>>,
    pub right_preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for BiCGOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            left_preconditioner: None,
            right_preconditioner: None,
        }
    }
}

/// Biconjugate Gradient solver
///
/// Solves Ax = b where A is non-symmetric.
#[allow(dead_code)]
pub fn bicg<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: BiCGOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for BiCG solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::sparse_zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Initialize r_star (shadow residual) = r
    let mut r_star = r.clone();

    // Apply preconditioners to initial residuals
    let mut z = if let Some(m1) = &options.left_preconditioner {
        m1.matvec(&r)?
    } else {
        r.clone()
    };

    let mut z_star = if let Some(m2) = &options.right_preconditioner {
        m2.matvec(&r_star)?
    } else {
        r_star.clone()
    };

    let mut p = z.clone();
    let mut p_star = z_star.clone();

    let mut rho_old = dot(&r_star, &z);

    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute q = A*p and q_star = A^T*p_star
        let mut q = a.matvec(&p)?;
        if let Some(m2) = &options.right_preconditioner {
            q = m2.matvec(&q)?;
        }

        let mut q_star = a.rmatvec(&p_star)?;
        if let Some(m1) = &options.left_preconditioner {
            q_star = m1.matvec(&q_star)?;
        }

        // Compute alpha = rho_old / (p_star, q)
        let alpha_den = dot(&p_star, &q);
        if alpha_den.abs()
            < F::epsilon() * F::from(10).expect("Failed to convert constant to float")
        {
            return Ok(IterationResult {
                x,
                iterations,
                residual_norm: norm2(&r),
                converged: false,
                message: "BiCG breakdown: (p_star, q) ≈ 0".to_string(),
            });
        }
        let alpha = rho_old / alpha_den;

        // Update solution and residuals
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            r_star[i] -= alpha * q_star[i];
        }

        // Check for convergence - compute residual norm BEFORE the next iteration
        let rnorm = norm2(&r);
        if rnorm <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply preconditioners
        z = if let Some(m1) = &options.left_preconditioner {
            m1.matvec(&r)?
        } else {
            r.clone()
        };

        z_star = if let Some(m2) = &options.right_preconditioner {
            m2.matvec(&r_star)?
        } else {
            r_star.clone()
        };

        // Compute new rho
        let rho = dot(&r_star, &z);
        if rho.abs() < F::epsilon() * F::from(10).expect("Failed to convert constant to float") {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: false,
                message: "BiCG breakdown: rho ≈ 0".to_string(),
            });
        }

        // Compute beta = rho / rho_old
        let beta = rho / rho_old;

        // Update search directions
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
            p_star[i] = z_star[i] + beta * p_star[i];
        }

        rho_old = rho;
        iterations += 1;
    }

    Ok(IterationResult {
        x,
        iterations,
        residual_norm: norm2(&r),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for BiCGSTAB solver
pub struct BiCGSTABOptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub left_preconditioner: Option<Box<dyn LinearOperator<F>>>,
    pub right_preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for BiCGSTABOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            left_preconditioner: None,
            right_preconditioner: None,
        }
    }
}

/// Result from BiCGSTAB solver
pub type BiCGSTABResult<F> = IterationResult<F>;

/// BiConjugate Gradient Stabilized method
///
/// An improved version of BiCG that avoids the irregular convergence patterns
/// and has better numerical stability. Works for general non-symmetric systems.
#[allow(dead_code)]
pub fn bicgstab<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: BiCGSTABOptions<F>,
) -> SparseResult<BiCGSTABResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for BiCGSTAB solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::sparse_zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Check if initial guess is solution
    let mut rnorm = norm2(&r);
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    if rnorm <= tolerance {
        return Ok(BiCGSTABResult {
            x,
            iterations: 0,
            residual_norm: rnorm,
            converged: true,
            message: "Converged with initial guess".to_string(),
        });
    }

    // Choose shadow residual (r_hat) as r0
    let r_hat = r.clone();

    let mut v = vec![F::sparse_zero(); n];
    let mut p = vec![F::sparse_zero(); n];
    let mut y = vec![F::sparse_zero(); n];
    let mut s = vec![F::sparse_zero(); n];
    let mut t: Vec<F>;

    let mut rho_old = F::sparse_one();
    let mut alpha = F::sparse_zero();
    let mut omega = F::sparse_one();

    // Main iteration loop
    let mut iterations = 0;
    while iterations < options.max_iter {
        // Compute rho = (r_hat, r)
        let rho = dot(&r_hat, &r);

        // Check for breakdown
        if rho.abs() < F::epsilon() * F::from(10).expect("Failed to convert constant to float") {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: rho ≈ 0".to_string(),
            });
        }

        // Compute beta and update p
        let beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply left preconditioner if provided
        let p_tilde = match &options.left_preconditioner {
            Some(m1) => m1.matvec(&p)?,
            None => p.clone(),
        };

        // Compute v = A * p_tilde
        v = a.matvec(&p_tilde)?;

        // Apply right preconditioner if provided
        if let Some(m2) = &options.right_preconditioner {
            v = m2.matvec(&v)?;
        }

        // Compute alpha = rho / (r_hat, v)
        let den = dot(&r_hat, &v);
        if den.abs() < F::epsilon() * F::from(10).expect("Failed to convert constant to float") {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: (r_hat, v) ≈ 0".to_string(),
            });
        }
        alpha = rho / den;

        // Check if alpha is reasonable
        if !alpha.is_finite() {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: alpha is not finite".to_string(),
            });
        }

        // Update solution and residual: s = r - alpha * v
        for i in 0..n {
            y[i] = x[i] + alpha * p_tilde[i];
            s[i] = r[i] - alpha * v[i];
        }

        // Check convergence
        let snorm = norm2(&s);
        if snorm <= tolerance {
            // Final update: x = y
            x = y;

            // Apply right preconditioner to final solution if provided
            if let Some(m2) = &options.right_preconditioner {
                x = m2.matvec(&x)?;
            }

            return Ok(BiCGSTABResult {
                x,
                iterations: iterations + 1,
                residual_norm: snorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        // Apply left preconditioner to s if provided
        let s_tilde = match &options.left_preconditioner {
            Some(m1) => m1.matvec(&s)?,
            None => s.clone(),
        };

        // Compute t = A * s_tilde
        t = a.matvec(&s_tilde)?;

        // Apply right preconditioner if provided
        if let Some(m2) = &options.right_preconditioner {
            t = m2.matvec(&t)?;
        }

        // Compute omega = (t, s) / (t, t)
        let ts = dot(&t, &s);
        let tt = dot(&t, &t);

        if tt < F::epsilon() * F::from(10).expect("Failed to convert constant to float") {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: (t, t) ≈ 0".to_string(),
            });
        }

        omega = ts / tt;

        // Check if omega is reasonable
        if !omega.is_finite()
            || omega.abs()
                < F::epsilon() * F::from(10).expect("Failed to convert constant to float")
        {
            return Ok(BiCGSTABResult {
                x,
                iterations,
                residual_norm: rnorm,
                converged: false,
                message: "BiCGSTAB breakdown: omega is not finite or too small".to_string(),
            });
        }

        // Update solution: x = y + omega * s_tilde
        for i in 0..n {
            x[i] = y[i] + omega * s_tilde[i];
            r[i] = s[i] - omega * t[i];
        }

        // Apply right preconditioner to final solution if provided
        if let Some(m2) = &options.right_preconditioner {
            x = m2.matvec(&x)?;
        }

        rnorm = norm2(&r);

        // Check for convergence
        if rnorm <= tolerance {
            return Ok(BiCGSTABResult {
                x,
                iterations: iterations + 1,
                residual_norm: rnorm,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        rho_old = rho;
        iterations += 1;
    }

    Ok(BiCGSTABResult {
        x,
        iterations,
        residual_norm: rnorm,
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Options for GMRES solver
pub struct GMRESOptions<F> {
    pub max_iter: usize,
    pub restart: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for GMRESOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            preconditioner: None,
        }
    }
}

/// Generalized Minimal Residual Method
///
/// Solves Ax = b for general non-symmetric systems. GMRES is particularly
/// robust but requires more memory than other methods due to the need to
/// store the Krylov basis vectors.
#[allow(dead_code)]
pub fn gmres<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: GMRESOptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for GMRES solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;
    let restart = options.restart.min(n);

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::sparse_zero(); n],
    };

    // Compute initial residual: r = b - A*x
    let ax = a.matvec(&x)?;
    let mut r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    // Apply preconditioner if provided
    if let Some(m) = &options.preconditioner {
        r = m.matvec(&r)?;
    }

    let mut rnorm = norm2(&r);
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    let mut outer_iterations = 0;

    // Outer iteration loop (restarts)
    while outer_iterations < options.max_iter && rnorm > tolerance {
        // Initialize Krylov subspace
        let mut v = vec![vec![F::sparse_zero(); n]; restart + 1];
        let mut h = vec![vec![F::sparse_zero(); restart]; restart + 1];
        let mut cs = vec![F::sparse_zero(); restart]; // Cosines for Givens rotations
        let mut sn = vec![F::sparse_zero(); restart]; // Sines for Givens rotations
        let mut s = vec![F::sparse_zero(); restart + 1]; // RHS for triangular system

        // Set up initial vector
        v[0] = r.iter().map(|&ri| ri / rnorm).collect();
        s[0] = rnorm;

        // Arnoldi iteration
        let mut inner_iter = 0;
        while inner_iter < restart && inner_iter + outer_iterations < options.max_iter {
            // Compute w = A * v[j]
            let mut w = a.matvec(&v[inner_iter])?;

            // Apply preconditioner if provided
            if let Some(m) = &options.preconditioner {
                w = m.matvec(&w)?;
            }

            // Orthogonalize against previous vectors
            for i in 0..=inner_iter {
                h[i][inner_iter] = dot(&v[i], &w);
                for (k, w_elem) in w.iter_mut().enumerate().take(n) {
                    *w_elem -= h[i][inner_iter] * v[i][k];
                }
            }

            h[inner_iter + 1][inner_iter] = norm2(&w);

            // Check for breakdown
            if h[inner_iter + 1][inner_iter]
                < F::epsilon() * F::from(10).expect("Failed to convert constant to float")
            {
                break;
            }

            // Normalize w and store in v[j+1]
            v[inner_iter + 1] = w
                .iter()
                .map(|&wi| wi / h[inner_iter + 1][inner_iter])
                .collect();

            // Apply previous Givens rotations
            for i in 0..inner_iter {
                let temp = cs[i] * h[i][inner_iter] + sn[i] * h[i + 1][inner_iter];
                h[i + 1][inner_iter] = -sn[i] * h[i][inner_iter] + cs[i] * h[i + 1][inner_iter];
                h[i][inner_iter] = temp;
            }

            // Compute new Givens rotation
            let rho = (h[inner_iter][inner_iter] * h[inner_iter][inner_iter]
                + h[inner_iter + 1][inner_iter] * h[inner_iter + 1][inner_iter])
                .sqrt();
            cs[inner_iter] = h[inner_iter][inner_iter] / rho;
            sn[inner_iter] = h[inner_iter + 1][inner_iter] / rho;

            // Apply new Givens rotation
            h[inner_iter][inner_iter] = rho;
            h[inner_iter + 1][inner_iter] = F::sparse_zero();

            let temp = cs[inner_iter] * s[inner_iter] + sn[inner_iter] * s[inner_iter + 1];
            s[inner_iter + 1] =
                -sn[inner_iter] * s[inner_iter] + cs[inner_iter] * s[inner_iter + 1];
            s[inner_iter] = temp;

            inner_iter += 1;

            // Check for convergence
            let residual = s[inner_iter].abs();
            if residual <= tolerance {
                break;
            }
        }

        // Solve the upper triangular system
        let mut y = vec![F::sparse_zero(); inner_iter];
        for i in (0..inner_iter).rev() {
            y[i] = s[i];
            for j in i + 1..inner_iter {
                let y_j = y[j];
                y[i] -= h[i][j] * y_j;
            }
            y[i] /= h[i][i];
        }

        // Update solution
        for i in 0..inner_iter {
            for (j, x_val) in x.iter_mut().enumerate().take(n) {
                *x_val += y[i] * v[i][j];
            }
        }

        // Compute new residual
        let ax = a.matvec(&x)?;
        r = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

        // Apply preconditioner if provided
        if let Some(m) = &options.preconditioner {
            r = m.matvec(&r)?;
        }

        rnorm = norm2(&r);
        outer_iterations += inner_iter;

        if rnorm <= tolerance {
            break;
        }
    }

    Ok(IterationResult {
        x,
        iterations: outer_iterations,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
        message: if rnorm <= tolerance {
            "Converged".to_string()
        } else {
            "Maximum iterations reached".to_string()
        },
    })
}

/// Trait for iterative solvers
pub trait IterativeSolver<F: Float> {
    /// Solve the linear system Ax = b
    fn solve(&self, a: &dyn LinearOperator<F>, b: &[F]) -> SparseResult<IterationResult<F>>;
}

// Helper functions

/// Compute the dot product of two vectors
pub(crate) fn dot<F: Float + Sum>(x: &[F], y: &[F]) -> F {
    x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum()
}

/// Compute the 2-norm of a vector
pub(crate) fn norm2<F: Float + Sum>(x: &[F]) -> F {
    dot(x, x).sqrt()
}

/// Options for LSQR solver
pub struct LSQROptions<F> {
    #[allow(dead_code)]
    pub max_iter: usize,
    #[allow(dead_code)]
    pub rtol: F,
    #[allow(dead_code)]
    pub atol: F,
    #[allow(dead_code)]
    pub btol: F,
    #[allow(dead_code)]
    pub x0: Option<Vec<F>>,
}

impl<F: Float> Default for LSQROptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            btol: F::from(1e-8).expect("Failed to convert constant to float"),
            x0: None,
        }
    }
}

/// Options for LSMR solver
#[allow(dead_code)]
pub struct LSMROptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub btol: F,
    pub x0: Option<Vec<F>>,
    pub damp: F,
}

impl<F: Float + SparseElement> Default for LSMROptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            btol: F::from(1e-8).expect("Failed to convert constant to float"),
            x0: None,
            damp: F::sparse_zero(),
        }
    }
}

/// Options for GCROT solver
#[allow(dead_code)]
pub struct GCROTOptions<F> {
    pub max_iter: usize,
    pub restart: usize,
    pub truncate: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for GCROTOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            truncate: 2,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            preconditioner: None,
        }
    }
}

/// Options for TFQMR solver
#[allow(dead_code)]
pub struct TFQMROptions<F> {
    pub max_iter: usize,
    pub rtol: F,
    pub atol: F,
    pub x0: Option<Vec<F>>,
    pub preconditioner: Option<Box<dyn LinearOperator<F>>>,
}

impl<F: Float> Default for TFQMROptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: F::from(1e-8).expect("Failed to convert constant to float"),
            atol: F::from(1e-12).expect("Failed to convert constant to float"),
            x0: None,
            preconditioner: None,
        }
    }
}

/// Stable implementation of Givens rotation
///
/// Computes (c, s, r) such that [ c  s] [a] = [r]
///                                [-s  c] [b]   [0]
fn sym_ortho<F: Float + SparseElement>(a: F, b: F) -> (F, F, F) {
    use scirs2_core::numeric::One;

    let zero = F::sparse_zero();
    let one = <F as One>::one();

    if b == zero {
        return (if a >= zero { one } else { -one }, zero, a.abs());
    } else if a == zero {
        return (zero, if b >= zero { one } else { -one }, b.abs());
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s_sign = if b >= zero { one } else { -one };
        let s = s_sign / (one + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c_sign = if a >= zero { one } else { -one };
        let c = c_sign / (one + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

/// Least Squares QR (LSQR) solver
///
/// Solves the least squares problem min ||Ax - b||_2 or the system Ax = b
/// using the LSQR algorithm. Suitable for large sparse least squares problems.
///
/// Implementation follows the SciPy reference based on:
/// C. C. Paige and M. A. Saunders (1982), "LSQR: An algorithm for sparse linear
/// equations and sparse least squares", ACM TOMS 8(1), 43-71.
///
/// Uses stable Givens rotations for numerical stability.
#[allow(dead_code)]
pub fn lsqr<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: LSQROptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (m, n) = a.shape();
    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![F::sparse_zero(); n],
    };

    // Set up the first vectors u and v for the bidiagonalization.
    // These satisfy  beta*u = b - A@x,  alfa*v = A'@u.
    let mut u: Vec<F> = b.to_vec();
    let bnorm = norm2(b);

    // Initialize from x0 if provided
    if x.iter().any(|&xi| xi != F::sparse_zero()) {
        let ax = a.matvec(&x)?;
        for i in 0..u.len() {
            u[i] -= ax[i];
        }
    }

    let mut beta = norm2(&u);
    if beta == F::sparse_zero() {
        return Ok(IterationResult {
            x,
            iterations: 0,
            residual_norm: F::sparse_zero(),
            converged: true,
            message: "Zero initial residual".to_string(),
        });
    }

    // Normalize u: u = u/beta
    for elem in &mut u {
        *elem /= beta;
    }

    // v = A'*u
    let mut v = a.rmatvec(&u)?;
    let mut alfa = norm2(&v);

    // Normalize v: v = v/alfa
    if alfa > F::sparse_zero() {
        for elem in &mut v {
            *elem /= alfa;
        }
    }

    let mut w = v.clone();

    // Initialize variables (matching SciPy naming)
    let mut rhobar = alfa;
    let mut phibar = beta;

    let tolerance = F::max(options.atol, options.rtol * bnorm);

    for iterations in 0..options.max_iter {
        // Perform the next step of the bidiagonalization to obtain the
        // next  beta, u, alfa, v. These satisfy the relations
        //     beta*u  =  A@v   -  alfa*u,
        //     alfa*v  =  A'@u  -  beta*v.

        // u = A.matvec(v) - alfa * u
        let av = a.matvec(&v)?;
        for i in 0..u.len() {
            u[i] = av[i] - alfa * u[i];
        }
        beta = norm2(&u);

        if beta > F::sparse_zero() {
            // u = (1/beta) * u
            for elem in &mut u {
                *elem /= beta;
            }

            // v = A.rmatvec(u) - beta * v
            let atu = a.rmatvec(&u)?;
            for i in 0..v.len() {
                v[i] = atu[i] - beta * v[i];
            }
            alfa = norm2(&v);

            if alfa > F::sparse_zero() {
                // v = (1/alfa) * v
                for elem in &mut v {
                    *elem /= alfa;
                }
            }
        }

        // Use a plane rotation to eliminate the subdiagonal element (beta)
        // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        let (cs, sn, rho) = sym_ortho(rhobar, beta);

        let theta = sn * alfa;
        rhobar = -cs * alfa;
        let phi = cs * phibar;
        phibar = sn * phibar;

        // Update x and w.
        let t1 = phi / rho;
        let t2 = -theta / rho;

        for i in 0..x.len() {
            x[i] += t1 * w[i];
        }
        for i in 0..w.len() {
            w[i] = v[i] + t2 * w[i];
        }

        // Check convergence
        let residual_norm = phibar.abs();
        if residual_norm <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: iterations + 1,
                residual_norm,
                converged: true,
                message: "Converged".to_string(),
            });
        }
    }

    Ok(IterationResult {
        x,
        iterations: options.max_iter,
        residual_norm: phibar.abs(),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Least Squares Minimal Residual (LSMR) solver
///
/// An enhanced version of LSQR with better numerical properties.
/// Implementation follows SciPy reference based on:
/// D. C.-L. Fong and M. A. Saunders (2011), "LSMR: An iterative algorithm
/// for sparse least-squares problems", SIAM J. Sci. Comput., 33(5), 2950-2971.
#[allow(dead_code)]
pub fn lsmr<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: LSMROptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    use scirs2_core::numeric::One;

    let (m, n) = a.shape();
    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    let zero = F::sparse_zero();
    let one = <F as One>::one();

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![zero; n],
    };

    // Set up the first vectors u and v for the bidiagonalization.
    let mut u: Vec<F> = b.to_vec();
    let normb = norm2(b);

    // Initialize from x0 if provided
    if x.iter().any(|&xi| xi != zero) {
        let ax = a.matvec(&x)?;
        for i in 0..u.len() {
            u[i] -= ax[i];
        }
    }

    let mut beta = norm2(&u);
    if beta == zero {
        return Ok(IterationResult {
            x,
            iterations: 0,
            residual_norm: zero,
            converged: true,
            message: "Zero initial residual".to_string(),
        });
    }

    // u = u / beta
    for elem in &mut u {
        *elem /= beta;
    }

    // v = A' * u
    let mut v = a.rmatvec(&u)?;
    let mut alpha = norm2(&v);

    // v = v / alpha
    if alpha > zero {
        for elem in &mut v {
            *elem /= alpha;
        }
    }

    // Initialize variables for 1st iteration.
    let mut alphabar = alpha;
    let mut zetabar = alpha * beta;
    let mut rho = one;
    let mut rhobar = one;
    let mut cbar = one;
    let mut sbar = zero;

    let mut h = v.clone();
    let mut hbar: Vec<F> = vec![zero; n];

    let tolerance = F::max(options.atol, options.rtol * normb);

    for itn in 0..options.max_iter {
        // Perform the next step of the bidiagonalization.
        // u = A * v - alpha * u
        let av = a.matvec(&v)?;
        for i in 0..u.len() {
            u[i] = av[i] - alpha * u[i];
        }
        beta = norm2(&u);

        if beta > zero {
            // u = u / beta
            for elem in &mut u {
                *elem /= beta;
            }

            // v = A' * u - beta * v
            let atu = a.rmatvec(&u)?;
            for i in 0..v.len() {
                v[i] = atu[i] - beta * v[i];
            }
            alpha = norm2(&v);

            if alpha > zero {
                for elem in &mut v {
                    *elem /= alpha;
                }
            }
        }

        // Use a plane rotation (Q_i) to turn B_i to R_i.
        let rhoold = rho;
        let (c, s, rho_new) = sym_ortho(alphabar, beta);
        rho = rho_new;
        let thetanew = s * alpha;
        alphabar = c * alpha;

        // Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar.
        let rhobarold = rhobar;
        let zetaold = zetabar / (rhoold * rhobarold);
        let thetabar = sbar * rho;
        let rhotemp = cbar * rho;
        let (cbar_new, sbar_new, rhobar_new) = sym_ortho(rhotemp, thetanew);
        cbar = cbar_new;
        sbar = sbar_new;
        rhobar = rhobar_new;
        let zeta = cbar * zetabar;
        zetabar = -sbar * zetabar;

        // Update h, hbar, x.
        let factor = thetabar * rho / (rhoold * rhobarold);
        for i in 0..n {
            hbar[i] = h[i] - factor * hbar[i];
        }
        let update_factor = zeta / (rho * rhobar);
        for i in 0..n {
            x[i] += update_factor * hbar[i];
        }
        let h_factor = thetanew / rho;
        for i in 0..n {
            h[i] = v[i] - h_factor * h[i];
        }

        // Check convergence based on zetabar (estimates ||A'r||)
        let normar = zetabar.abs();
        if normar <= tolerance {
            return Ok(IterationResult {
                x,
                iterations: itn + 1,
                residual_norm: normar,
                converged: true,
                message: "Converged".to_string(),
            });
        }
    }

    Ok(IterationResult {
        x,
        iterations: options.max_iter,
        residual_norm: zetabar.abs(),
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Transpose-Free Quasi-Minimal Residual (TFQMR) solver
///
/// A transpose-free variant of QMR that avoids the need for A^T.
/// Implementation follows SciPy's tfqmr based on R.W. Freund (1993).
///
/// Reference:
/// R. W. Freund, "A Transpose-Free Quasi-Minimal Residual Algorithm for
/// Non-Hermitian Linear Systems", SIAM J. Sci. Comput., 14(2), 470-482, 1993.
#[allow(dead_code)]
pub fn tfqmr<F>(
    a: &dyn LinearOperator<F>,
    b: &[F],
    options: TFQMROptions<F>,
) -> SparseResult<IterationResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + 'static,
{
    let (rows, cols) = a.shape();
    if rows != cols {
        return Err(SparseError::ValueError(
            "Matrix must be square for TFQMR solver".to_string(),
        ));
    }
    if b.len() != rows {
        return Err(SparseError::DimensionMismatch {
            expected: rows,
            found: b.len(),
        });
    }

    let n = rows;
    let one = F::sparse_one();
    let zero = F::sparse_zero();

    // Initialize solution
    let mut x: Vec<F> = match &options.x0 {
        Some(x0) => {
            if x0.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: x0.len(),
                });
            }
            x0.clone()
        }
        None => vec![zero; n],
    };

    // Compute initial residual r = b - A*x
    let ax = a.matvec(&x)?;
    let r: Vec<F> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();

    let r0norm = norm2(&r);
    let bnorm = norm2(b);
    let tolerance = F::max(options.atol, options.rtol * bnorm);

    if r0norm <= tolerance || r0norm == zero {
        return Ok(IterationResult {
            x,
            iterations: 0,
            residual_norm: r0norm,
            converged: true,
            message: "Converged with initial guess".to_string(),
        });
    }

    // Initialize vectors following SciPy's notation
    let mut u = r.clone();
    let mut w = r.clone();
    let rstar = r.clone(); // Shadow residual (r_tilde in some notations)

    // v = M^{-1} * A * r (with preconditioner)
    let ar = a.matvec(&r)?;
    let mut v = match &options.preconditioner {
        Some(m) => m.matvec(&ar)?,
        None => ar,
    };
    let mut uhat = v.clone();

    let mut d: Vec<F> = vec![zero; n];
    let mut theta = zero;
    let mut eta = zero;

    // rho = <rstar, r> (always real since rstar == r initially)
    let mut rho = dot(&rstar, &r);
    let mut rho_last = rho;
    let mut tau = r0norm;

    for iter in 0..options.max_iter {
        let even = iter % 2 == 0;

        // On even iterations, compute alpha and uNext
        let mut alpha = zero;
        let mut u_next: Vec<F> = vec![zero; n];

        if even {
            let vtrstar = dot(&rstar, &v);
            if vtrstar == zero {
                return Ok(IterationResult {
                    x,
                    iterations: iter,
                    residual_norm: tau,
                    converged: false,
                    message: "TFQMR breakdown: v'*rstar = 0".to_string(),
                });
            }
            alpha = rho / vtrstar;

            // uNext = u - alpha * v
            for i in 0..n {
                u_next[i] = u[i] - alpha * v[i];
            }
        }

        // w = w - alpha * uhat (every iteration)
        // Need alpha from either current even step or previous even step
        let alpha_used = if even {
            alpha
        } else {
            // On odd iterations, alpha comes from the previous even iteration
            // We need to save alpha across iterations
            rho / dot(&rstar, &v)
        };

        for i in 0..n {
            w[i] -= alpha_used * uhat[i];
        }

        // d = u + (theta^2 / alpha) * eta * d
        let theta_sq_over_alpha =
            if alpha_used.abs() > F::from(1e-300).expect("Failed to convert constant to float") {
                theta * theta / alpha_used
            } else {
                zero
            };
        for i in 0..n {
            d[i] = u[i] + theta_sq_over_alpha * eta * d[i];
        }

        // theta = ||w|| / tau
        theta = norm2(&w) / tau;

        // c = 1 / sqrt(1 + theta^2)
        let c = one / (one + theta * theta).sqrt();

        // tau = tau * theta * c
        tau = tau * theta * c;

        // eta = c^2 * alpha
        eta = c * c * alpha_used;

        // z = M^{-1} * d (apply preconditioner to d)
        let z = match &options.preconditioner {
            Some(m) => m.matvec(&d)?,
            None => d.clone(),
        };

        // x = x + eta * z
        for i in 0..n {
            x[i] += eta * z[i];
        }

        // Convergence criterion: tau * sqrt(iter+1) < tolerance
        let iter_f = F::from(iter + 1).expect("Failed to convert to float");
        if tau * iter_f.sqrt() < tolerance {
            return Ok(IterationResult {
                x,
                iterations: iter + 1,
                residual_norm: tau,
                converged: true,
                message: "Converged".to_string(),
            });
        }

        if !even {
            // Odd iteration updates
            // rho = <rstar, w>
            rho = dot(&rstar, &w);

            if rho == zero {
                return Ok(IterationResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: tau,
                    converged: false,
                    message: "TFQMR breakdown: rho = 0".to_string(),
                });
            }

            let beta = rho / rho_last;

            // u = w + beta * u
            for i in 0..n {
                u[i] = w[i] + beta * u[i];
            }

            // v = beta * uhat + beta^2 * v
            for i in 0..n {
                v[i] = beta * uhat[i] + beta * beta * v[i];
            }

            // uhat = M^{-1} * A * u
            let au = a.matvec(&u)?;
            uhat = match &options.preconditioner {
                Some(m) => m.matvec(&au)?,
                None => au,
            };

            // v = v + uhat
            for i in 0..n {
                v[i] += uhat[i];
            }
        } else {
            // Even iteration updates
            // uhat = M^{-1} * A * uNext
            let au_next = a.matvec(&u_next)?;
            uhat = match &options.preconditioner {
                Some(m) => m.matvec(&au_next)?,
                None => au_next,
            };

            // u = uNext
            u = u_next;

            // rho_last = rho
            rho_last = rho;
        }
    }

    Ok(IterationResult {
        x,
        iterations: options.max_iter,
        residual_norm: tau,
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::linalg::interface::AsLinearOperator;

    #[test]
    fn test_cg_identity() {
        // Test CG on identity matrix: I * x = b => x = b
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = CGOptions::default();
        let result = cg(&identity, &b, options).expect("Operation failed");

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-3);
        }
    }

    #[test]
    fn test_cg_diagonal() {
        // Test CG on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = CGOptions::default();
        let result = cg(&diag, &b, options).expect("Operation failed");

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-3);
        }
    }

    #[test]
    fn test_cg_sparse_matrix() {
        // Test CG on a sparse positive definite matrix
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 3.0];
        let options = CGOptions::default();
        let result = cg(op.as_ref(), &b, options).expect("Operation failed");

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).expect("Operation failed");
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn test_bicgstab_identity() {
        // Test BiCGSTAB on identity matrix: I * x = b => x = b
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(&identity, &b, options).expect("Operation failed");

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-3);
        }
    }

    #[test]
    fn test_bicgstab_diagonal() {
        // Test BiCGSTAB on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(&diag, &b, options).expect("Operation failed");

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-3);
        }
    }

    #[test]
    fn test_bicgstab_non_symmetric() {
        // Test BiCGSTAB on a non-symmetric matrix
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![4.0, -1.0, -2.0, -1.0, 4.0, -1.0, 0.0, -1.0, 3.0];
        let shape = (3, 3);

        let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 1.0];
        let options = BiCGSTABOptions::default();
        let result = bicgstab(op.as_ref(), &b, options).expect("Operation failed");

        assert!(result.converged);
        // Verify solution by checking Ax = b
        let ax = op.matvec(&result.x).expect("Operation failed");
        for (axi, bi) in ax.iter().zip(&b) {
            assert!((axi - bi).abs() < 1e-9);
        }
    }

    #[test]
    fn test_lsqr_identity() {
        // Test LSQR on identity matrix
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = LSQROptions::default();
        let result = lsqr(&identity, &b, options).expect("Operation failed");

        println!("Identity test - Expected: {:?}, Got: {:?}", b, result.x);
        println!(
            "Converged: {}, Iterations: {}",
            result.converged, result.iterations
        );

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lsqr_diagonal() {
        // Test LSQR on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = LSQROptions::default();
        let result = lsqr(&diag, &b, options).expect("Operation failed");

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lsmr_identity() {
        // Test LSMR on identity matrix
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = LSMROptions::default();
        let result = lsmr(&identity, &b, options).expect("Operation failed");

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lsmr_diagonal() {
        // Test LSMR on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = LSMROptions::default();
        let result = lsmr(&diag, &b, options).expect("Operation failed");

        assert!(result.converged);
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-3);
        }
    }

    #[test]
    fn test_tfqmr_identity() {
        // Test TFQMR on identity matrix
        let identity = crate::linalg::interface::IdentityOperator::<f64>::new(3);
        let b = vec![1.0, 2.0, 3.0];
        let options = TFQMROptions::default();
        let result = tfqmr(&identity, &b, options).expect("Operation failed");

        assert!(result.converged);
        for (xi, bi) in result.x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-3);
        }
    }

    #[test]
    fn test_tfqmr_diagonal() {
        // Test TFQMR on diagonal matrix
        let diag = crate::linalg::interface::DiagonalOperator::new(vec![2.0, 3.0, 4.0]);
        let b = vec![2.0, 6.0, 12.0];
        let options = TFQMROptions::default();
        let result = tfqmr(&diag, &b, options).expect("Operation failed");

        assert!(result.converged, "TFQMR did not converge");
        let expected = vec![1.0, 2.0, 3.0];
        for (xi, ei) in result.x.iter().zip(&expected) {
            assert!((xi - ei).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lsqr_least_squares() {
        // Test LSQR on overdetermined system (m > n)
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = (4, 2);

        let matrix = CsrMatrix::new(data, rows, cols, shape).expect("Operation failed");
        let op = matrix.as_linear_operator();

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let options = LSQROptions::default();
        let result = lsqr(op.as_ref(), &b, options).expect("Operation failed");

        // LSQR should converge for overdetermined systems
        assert!(result.converged || result.residual_norm < 1e-6);
    }

    #[test]
    fn test_solver_options_defaults() {
        let lsqr_opts = LSQROptions::<f64>::default();
        assert_eq!(lsqr_opts.max_iter, 1000);
        assert!(lsqr_opts.x0.is_none());

        let lsmr_opts = LSMROptions::<f64>::default();
        assert_eq!(lsmr_opts.max_iter, 1000);
        assert_eq!(lsmr_opts.damp, 0.0);
        assert!(lsmr_opts.x0.is_none());

        let tfqmr_opts = TFQMROptions::<f64>::default();
        assert_eq!(tfqmr_opts.max_iter, 1000);
        assert!(tfqmr_opts.x0.is_none());
        assert!(tfqmr_opts.preconditioner.is_none());
    }
}
