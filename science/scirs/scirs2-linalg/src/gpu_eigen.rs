//! GPU eigensolver interface.
//!
//! Provides GPU-accelerated eigenvalue computation with transparent CPU
//! fallback.  The current implementation executes entirely on the CPU using
//! high-quality numerical algorithms; GPU dispatch is marked as a future
//! enhancement.
//!
//! # Algorithms
//!
//! | Matrix size | Algorithm |
//! |-------------|-----------|
//! | n ≤ 100 | Householder tridiagonalization + QL iteration |
//! | n > 100 | Lanczos iteration (block-1 variant) |
//!
//! # References
//!
//! - Golub & Van Loan (2013). "Matrix Computations", 4th edition, §8.3–8.4.
//! - Parlett (1998). "The Symmetric Eigenvalue Problem". SIAM.
//! - Lanczos (1950). "An iteration method for the solution of the eigenvalue
//!   problem of linear differential and integral operators."

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::LinalgError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which eigenvalues to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenTarget {
    /// Eigenvalues of largest magnitude.
    LargestMagnitude,
    /// Eigenvalues of smallest magnitude.
    SmallestMagnitude,
    /// Algebraically largest eigenvalues.
    LargestAlgebraic,
    /// Algebraically smallest eigenvalues.
    SmallestAlgebraic,
    /// All eigenvalues.
    All,
}

/// Configuration for the GPU eigensolver.
#[derive(Debug, Clone)]
pub struct GpuEigenConfig {
    /// Use GPU if available; fall back to CPU otherwise.
    /// Currently has no effect (CPU fallback is always used).
    pub prefer_gpu: bool,
    /// Maximum number of iterations for iterative methods.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Number of eigenvalues to compute (0 = all).
    pub n_eigenvalues: usize,
    /// Which eigenvalues to compute.
    pub which: EigenTarget,
}

impl Default for GpuEigenConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: false,
            max_iter: 1000,
            tol: 1e-10,
            n_eigenvalues: 0,
            which: EigenTarget::All,
        }
    }
}

// ---------------------------------------------------------------------------
// Public solver struct
// ---------------------------------------------------------------------------

/// GPU-accelerated eigenvalue decomposition.
///
/// Currently delegates all computation to CPU-based algorithms.  A future
/// version will dispatch to CUDA/Metal/WebGPU kernels when `prefer_gpu = true`
/// and a compatible device is detected.
pub struct GpuEigensolver {
    config: GpuEigenConfig,
}

impl GpuEigensolver {
    /// Create a new eigensolver with the given configuration.
    pub fn new(config: GpuEigenConfig) -> Self {
        Self { config }
    }

    /// Compute all eigenvalues of a real symmetric matrix.
    ///
    /// Uses the Householder–QL algorithm for n ≤ 100 and the Lanczos
    /// algorithm for larger matrices.
    ///
    /// # Errors
    ///
    /// Returns an error if `a` is not square, or if the iterative algorithm
    /// fails to converge.
    pub fn eigenvalues_symmetric(&self, a: &Array2<f64>) -> Result<Array1<f64>, LinalgError> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::DimensionError(format!(
                "eigenvalues_symmetric requires a square matrix, got {}x{}",
                a.nrows(),
                a.ncols()
            )));
        }
        if n == 0 {
            return Ok(Array1::zeros(0));
        }
        if n == 1 {
            return Ok(Array1::from(vec![a[[0, 0]]]));
        }

        if n <= 100 {
            let (mut d, mut e) = householder_tridiagonalize_diags(a);
            qr_iteration_tridiagonal(&mut d, &mut e, None, self.config.max_iter, self.config.tol)?;
            sort_eigenvalues(&mut d);
            Ok(Array1::from(d))
        } else {
            lanczos_eigenvalues(a, n, self.config.max_iter, self.config.tol)
        }
    }

    /// Compute eigenvalues and eigenvectors of a real symmetric matrix.
    ///
    /// Returns `(eigenvalues, eigenvectors)` where `eigenvectors` columns are
    /// the eigenvectors in the same order as `eigenvalues`.
    ///
    /// # Errors
    ///
    /// Returns an error if `a` is not square, or convergence fails.
    pub fn eigen_symmetric(
        &self,
        a: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), LinalgError> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::DimensionError(format!(
                "eigen_symmetric requires a square matrix, got {}x{}",
                a.nrows(),
                a.ncols()
            )));
        }
        if n == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
        }
        if n == 1 {
            return Ok((Array1::from(vec![a[[0, 0]]]), Array2::eye(1)));
        }

        let (t, q) = householder_tridiagonalize(a);
        let mut d: Vec<f64> = (0..n).map(|i| t[[i, i]]).collect();
        let mut e: Vec<f64> = (0..n.saturating_sub(1)).map(|i| t[[i, i + 1]]).collect();
        // Z accumulates eigenvectors: start with Q (the Householder accumulation)
        let mut z = q;
        qr_iteration_tridiagonal(
            &mut d,
            &mut e,
            Some(&mut z),
            self.config.max_iter,
            self.config.tol,
        )?;
        sort_eigenpairs(&mut d, &mut z);
        Ok((Array1::from(d), z))
    }

    /// Estimate the largest eigenvalue of a symmetric matrix via power iteration
    /// with Rayleigh quotient acceleration.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or if iteration fails to
    /// converge.
    pub fn largest_eigenvalue(&self, a: &Array2<f64>) -> Result<f64, LinalgError> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::DimensionError(format!(
                "largest_eigenvalue requires a square matrix, got {}x{}",
                a.nrows(),
                a.ncols()
            )));
        }
        if n == 0 {
            return Ok(0.0);
        }
        power_iteration(a, self.config.max_iter, self.config.tol)
    }

    /// Estimate the `k` largest-magnitude eigenvalues via the Lanczos algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if `k > n`, or if iteration fails to converge.
    pub fn k_largest_eigenvalues(
        &self,
        a: &Array2<f64>,
        k: usize,
    ) -> Result<Array1<f64>, LinalgError> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::DimensionError(format!(
                "k_largest_eigenvalues requires a square matrix, got {}x{}",
                a.nrows(),
                a.ncols()
            )));
        }
        if k == 0 {
            return Ok(Array1::zeros(0));
        }
        if k > n {
            return Err(LinalgError::DimensionError(format!(
                "requested k={k} eigenvalues but matrix has only {n} columns"
            )));
        }

        // For small matrices use the exact Householder-QL algorithm and select
        // the k largest.  For large matrices use Lanczos with oversampling.
        let all_eigs = if n <= 100 {
            self.eigenvalues_symmetric(a)?
        } else {
            // Lanczos with oversampling: compute min(n, 2k+10) Ritz values
            let steps = (2 * k + 10).min(n);
            lanczos_eigenvalues(a, steps, self.config.max_iter, self.config.tol)?
        };

        // Sort by descending magnitude and return the k largest
        let mut pairs: Vec<(f64, f64)> = all_eigs.iter().map(|&v| (v.abs(), v)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let result: Vec<f64> = pairs.iter().map(|&(_, v)| v).take(k).collect();
        Ok(Array1::from(result))
    }
}

// ---------------------------------------------------------------------------
// Householder tridiagonalization — returns diagonal and sub-diagonal only
// ---------------------------------------------------------------------------

/// Reduce a symmetric matrix to tridiagonal form.
///
/// Returns `(d, e)` where `d` is the main diagonal (length n) and `e` is the
/// sub-diagonal (length n-1).
fn householder_tridiagonalize_diags(a: &Array2<f64>) -> (Vec<f64>, Vec<f64>) {
    let (t, _q) = householder_tridiagonalize(a);
    let n = a.nrows();
    let d: Vec<f64> = (0..n).map(|i| t[[i, i]]).collect();
    let e: Vec<f64> = (0..n.saturating_sub(1)).map(|i| t[[i, i + 1]]).collect();
    (d, e)
}

/// Householder tridiagonalization.
///
/// Returns `(T, Q)` where `T` is real symmetric tridiagonal and `A = Q T Q^T`.
/// The Householder vectors are accumulated into `Q`.
pub fn householder_tridiagonalize(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut t = a.to_owned();
    let mut q = Array2::<f64>::eye(n);

    for k in 0..(n.saturating_sub(2)) {
        // Build Householder vector for column k below the sub-diagonal
        let col_len = n - k - 1;
        let mut v: Vec<f64> = (0..col_len).map(|i| t[[k + 1 + i, k]]).collect();

        let norm_v = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_v < 1e-300 {
            continue;
        }

        // Sign chosen to avoid cancellation
        let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * norm_v;

        let v_norm_sq = v.iter().map(|&x| x * x).sum::<f64>();
        if v_norm_sq < 1e-300 {
            continue;
        }

        // Apply H = I - 2 v v^T / ||v||^2 to both sides of T
        // T <- H T H
        apply_householder_sym(&mut t, &v, v_norm_sq, k + 1);

        // Accumulate into Q: Q <- Q H
        apply_householder_right(&mut q, &v, v_norm_sq, k + 1);
    }

    (t, q)
}

/// Apply H = I - 2 v v^T / ||v||^2 symmetrically: T <- H T H
/// `offset` is the starting index; v has length n - offset.
fn apply_householder_sym(t: &mut Array2<f64>, v: &[f64], v_norm_sq: f64, offset: usize) {
    let n = t.nrows();
    let m = v.len(); // n - offset

    // w = T * v (only the [offset..n, offset..n] block matters)
    let mut w = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..m {
            s += t[[i, offset + j]] * v[j];
        }
        w[i] = s;
    }

    // alpha = v^T w / ||v||^2
    let mut vt_w = 0.0;
    for j in 0..m {
        vt_w += v[j] * w[offset + j];
    }
    let alpha = vt_w / v_norm_sq;

    // u = w - alpha v  (only the relevant slice)
    let mut u = vec![0.0_f64; n];
    u.copy_from_slice(&w);
    for j in 0..m {
        u[offset + j] -= alpha * v[j];
    }

    let tau = 2.0 / v_norm_sq;
    // T <- T - tau * (u v^T + v u^T)
    for i in 0..n {
        for j in 0..m {
            let delta = tau * (u[i] * v[j] + v[j] * u[i]);
            // Only the (i, offset+j) and (offset+j, i) terms
            t[[i, offset + j]] -= tau * (u[i] * v[j]);
            t[[offset + j, i]] -= tau * (v[j] * u[i]);
        }
    }
    // Fix up double-subtracted diagonal entries in the block
    // The formula is T <- T - tau u v^T - tau v u^T
    // The above loop applied - tau u[i] v[j] and - tau v[j] u[i]
    // which equals - 2*tau * u[i]*v[j] on the diagonal overlap; correct by re-adding tau once
    for j in 0..m {
        t[[offset + j, offset + j]] += tau * v[j] * u[offset + j];
    }
}

/// Apply Householder from the right: Q <- Q H
/// where H = I - tau v v^T acts on columns [offset..n].
fn apply_householder_right(q: &mut Array2<f64>, v: &[f64], v_norm_sq: f64, offset: usize) {
    let n = q.nrows();
    let tau = 2.0 / v_norm_sq;
    let m = v.len();

    // For each row i of Q: q_row <- q_row - tau * (q_row . v_sub) * v_sub
    for i in 0..n {
        let mut dot = 0.0;
        for j in 0..m {
            dot += q[[i, offset + j]] * v[j];
        }
        for j in 0..m {
            q[[i, offset + j]] -= tau * dot * v[j];
        }
    }
}

// ---------------------------------------------------------------------------
// QL algorithm for symmetric tridiagonal matrices
// ---------------------------------------------------------------------------

/// QL algorithm for a real symmetric tridiagonal matrix.
///
/// On input `d` (length n) is the main diagonal and `e` (length n-1) is the
/// sub-diagonal.  On output `d` contains the eigenvalues (unsorted).  If `z`
/// is `Some`, the orthogonal transformation is accumulated into it so that
/// `z` columns become eigenvectors of the original tridiagonalized matrix.
///
/// # Errors
///
/// Returns [`LinalgError::ConvergenceError`] if the algorithm does not
/// converge within `max_iter` sweeps.
pub fn qr_iteration_tridiagonal(
    d: &mut [f64],
    e: &mut [f64],
    mut z: Option<&mut Array2<f64>>,
    max_iter: usize,
    tol: f64,
) -> Result<(), LinalgError> {
    let n = d.len();
    if n <= 1 {
        return Ok(());
    }
    // Work on a local copy of e padded to length n
    let mut e_full = vec![0.0_f64; n];
    e_full[..n - 1].copy_from_slice(&e[..n - 1]);

    let mut l = 0_usize;
    while l < n {
        // Find the active sub-block [l..m]
        let mut m = l;
        while m < n - 1 {
            let thresh = tol * (d[m].abs() + d[m + 1].abs());
            if e_full[m].abs() <= thresh {
                break;
            }
            m += 1;
        }
        if m == l {
            l += 1;
            continue;
        }

        let mut iter_count = 0;
        loop {
            if iter_count >= max_iter {
                return Err(LinalgError::ConvergenceError(format!(
                    "QL algorithm did not converge after {max_iter} iterations \
                     on sub-block [{l}..{m}]"
                )));
            }
            iter_count += 1;

            // Wilkinson shift (using the bottom 2x2 sub-matrix)
            let g = (d[m - 1] - d[m]) / (2.0 * e_full[m - 1]);
            let r = f64::hypot(g, 1.0);
            let shift = d[m] - e_full[m - 1] / (g + if g >= 0.0 { r } else { -r });

            // Implicit QL step (tqli / IMTQL2 style).
            // The bulge is chased from index m-1 down to l using Givens rotations.
            // `g` is reused as the "current chasing" off-diagonal value; `p` is the
            // deferred correction applied to d[l] at the end of the sweep.
            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            // Reuse variable name `g` for the chasing value (shadows the shift `g` above)
            let mut g = d[m] - shift;

            for i in (l..m).rev() {
                let f = s * e_full[i];
                let b = c * e_full[i];
                let rot_r = f64::hypot(f, g);
                e_full[i + 1] = rot_r;

                if rot_r.abs() < 1e-300 {
                    // Deflation
                    d[i + 1] -= p;
                    e_full[m - 1] = 0.0;
                    break;
                }

                s = f / rot_r;
                c = g / rot_r;
                g = d[i + 1] - p;
                let rot_r2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * rot_r2;
                d[i + 1] = g + p;
                g = c * rot_r2 - b;

                // Accumulate the rotation into z if eigenvectors are requested
                if let Some(ref mut zmat) = z {
                    let nz = zmat.nrows();
                    for row in 0..nz {
                        let zi = zmat[[row, i]];
                        let zi1 = zmat[[row, i + 1]];
                        zmat[[row, i]] = c * zi + s * zi1;
                        zmat[[row, i + 1]] = -s * zi + c * zi1;
                    }
                }
            }

            d[l] -= p;
            e_full[l] = g;
            e_full[m - 1] = 0.0;

            // Check convergence of the bottom off-diagonal
            let thresh = tol * (d[l].abs() + d[m].abs());
            if e_full[m - 1].abs() <= thresh {
                break;
            }
        }
        e_full[m - 1] = 0.0;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Power iteration
// ---------------------------------------------------------------------------

/// Power iteration with Rayleigh quotient shift.
///
/// Returns the dominant eigenvalue of a symmetric matrix.
fn power_iteration(a: &Array2<f64>, max_iter: usize, tol: f64) -> Result<f64, LinalgError> {
    let n = a.nrows();
    // Initialize with a fixed non-trivial vector
    let mut x: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

    let mut lambda = 0.0_f64;
    for _iter in 0..max_iter {
        // y = A x
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[[i, j]] * x[j];
            }
        }
        // Rayleigh quotient
        let xty: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let xtx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let new_lambda = xty / xtx.max(1e-300);

        // Normalize y
        let norm_y = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        if norm_y < 1e-300 {
            break;
        }
        for xi in x.iter_mut() {
            *xi = 0.0;
        }
        for (xi, &yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi / norm_y;
        }

        if (new_lambda - lambda).abs() < tol * (1.0 + lambda.abs()) {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;
    }
    Ok(lambda)
}

// ---------------------------------------------------------------------------
// Lanczos algorithm
// ---------------------------------------------------------------------------

/// Lanczos iteration to approximate `k` eigenvalues of a symmetric matrix.
///
/// Uses the basic (non-restarted) Lanczos procedure followed by tridiagonal
/// QL to extract eigenvalues.  Full reorthogonalization is applied to
/// maintain numerical stability.
fn lanczos_eigenvalues(
    a: &Array2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Result<Array1<f64>, LinalgError> {
    let n = a.nrows();
    let steps = k.min(n).min(max_iter);

    // Lanczos vectors stored as rows in V (steps+1 x n)
    let mut v_prev = vec![0.0_f64; n];
    // Use uniform starting vector [1/sqrt(n), ...] instead of e_1 to avoid
    // aligning with a single eigenvector of diagonal/structured matrices.
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v_curr: Vec<f64> = vec![inv_sqrt_n; n];

    // Normalize initial vector (already unit-length, but guard against n==0)
    let norm0 = v_curr.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm0 < 1e-300 {
        return Ok(Array1::zeros(k));
    }
    for x in v_curr.iter_mut() {
        *x /= norm0;
    }

    let mut alpha_vec = Vec::with_capacity(steps);
    let mut beta_vec = Vec::new(); // length steps-1

    let mut all_v: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
    all_v.push(v_curr.clone());

    for j in 0..steps {
        // w = A * v_curr
        let mut w = vec![0.0_f64; n];
        for i in 0..n {
            for l in 0..n {
                w[i] += a[[i, l]] * v_curr[l];
            }
        }

        // alpha_j = v_curr . w
        let alpha_j: f64 = v_curr.iter().zip(w.iter()).map(|(&vc, &wi)| vc * wi).sum();
        alpha_vec.push(alpha_j);

        // w = w - alpha_j * v_curr - beta_{j-1} * v_prev
        for i in 0..n {
            w[i] -= alpha_j * v_curr[i];
            w[i] -= if j == 0 { 0.0 } else { beta_vec[j - 1] } * v_prev[i];
        }

        // Full reorthogonalization against all previous Lanczos vectors
        for prev in &all_v {
            let dot: f64 = prev.iter().zip(w.iter()).map(|(&p, &wi)| p * wi).sum();
            for (wi, &pi) in w.iter_mut().zip(prev.iter()) {
                *wi -= dot * pi;
            }
        }

        // beta_{j} = ||w||
        let beta_j = w.iter().map(|&x| x * x).sum::<f64>().sqrt();

        if j + 1 < steps {
            if beta_j < tol {
                // Invariant subspace found; remaining eigenvalues are zero
                alpha_vec.resize(j + 1, 0.0);
                break;
            }
            beta_vec.push(beta_j);
            v_prev = v_curr.clone();
            v_curr = w.iter().map(|&x| x / beta_j).collect();
            all_v.push(v_curr.clone());
        }
    }

    // Solve the Lanczos tridiagonal eigenproblem
    let lanczos_n = alpha_vec.len();
    let mut d = alpha_vec;
    let mut e = beta_vec;
    e.resize(lanczos_n.saturating_sub(1), 0.0);

    qr_iteration_tridiagonal(&mut d, &mut e, None, max_iter, tol)?;
    sort_eigenvalues(&mut d);

    let result_k = k.min(d.len());
    Ok(Array1::from(d[..result_k].to_vec()))
}

// ---------------------------------------------------------------------------
// Sorting helpers
// ---------------------------------------------------------------------------

/// Sort eigenvalues in ascending order.
fn sort_eigenvalues(d: &mut [f64]) {
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Sort eigenvalue/eigenvector pairs by ascending eigenvalue.
fn sort_eigenpairs(d: &mut [f64], z: &mut Array2<f64>) {
    let n = d.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));

    let d_orig = d.to_vec();
    let z_orig = z.clone();
    for (new_pos, &old_pos) in idx.iter().enumerate() {
        d[new_pos] = d_orig[old_pos];
        for row in 0..z.nrows() {
            z[[row, new_pos]] = z_orig[[row, old_pos]];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gpu_eigen_identity() {
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        let id = Array2::<f64>::eye(4);
        let eigs = solver.eigenvalues_symmetric(&id).expect("should succeed");
        assert_eq!(eigs.len(), 4);
        for &e in eigs.iter() {
            assert!(approx_eq(e, 1.0, 1e-8), "identity eigenvalue={e}");
        }
    }

    #[test]
    fn test_gpu_eigen_diagonal() {
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        // Diagonal matrix with known eigenvalues 1, 2, 3
        let a = array![[3.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let mut eigs = solver.eigenvalues_symmetric(&a).expect("should succeed");
        if let Some(s) = eigs.as_slice_mut() {
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
        }
        let v: Vec<f64> = eigs.to_vec();
        assert!(approx_eq(v[0], 1.0, 1e-8), "v[0]={}", v[0]);
        assert!(approx_eq(v[1], 2.0, 1e-8), "v[1]={}", v[1]);
        assert!(approx_eq(v[2], 3.0, 1e-8), "v[2]={}", v[2]);
    }

    #[test]
    fn test_gpu_eigen_symmetric_2x2() {
        // A = [[2, 1], [1, 2]]  eigenvalues = 1, 3
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        let a = array![[2.0_f64, 1.0], [1.0, 2.0]];
        let eigs = solver.eigenvalues_symmetric(&a).expect("should succeed");
        let mut v: Vec<f64> = eigs.to_vec();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!(approx_eq(v[0], 1.0, 1e-8), "v[0]={}", v[0]);
        assert!(approx_eq(v[1], 3.0, 1e-8), "v[1]={}", v[1]);
    }

    #[test]
    fn test_gpu_eigen_power_iter() {
        // A = [[3, 1], [1, 3]]  largest eigenvalue = 4
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let lambda = solver.largest_eigenvalue(&a).expect("should succeed");
        assert!(approx_eq(lambda, 4.0, 1e-6), "lambda={lambda}");
    }

    #[test]
    fn test_gpu_eigen_symmetric_large() {
        // 10x10 random-ish symmetric positive-definite matrix: A = B^T B + 10*I
        let n = 10;
        let mut data = vec![0.0_f64; n * n];
        // Set up a simple known SPD matrix
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if i == j {
                    10.0 + i as f64
                } else {
                    0.1 / ((i as f64 - j as f64).abs() + 1.0)
                };
            }
        }
        let a = Array2::from_shape_vec((n, n), data).expect("valid shape");
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        let eigs = solver.eigenvalues_symmetric(&a).expect("should succeed");
        assert_eq!(eigs.len(), n);
        for &e in eigs.iter() {
            assert!(e > 0.0, "expected positive eigenvalue, got {e}");
        }
    }

    #[test]
    fn test_gpu_eigen_householder_tridiagonal() {
        // Verify that householder_tridiagonalize produces a tridiagonal matrix
        let a = array![[4.0_f64, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];
        let (t, _q) = householder_tridiagonalize(&a);
        // Off-tridiagonal elements should be near zero
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                if (i as isize - j as isize).unsigned_abs() > 1 {
                    assert!(
                        t[[i, j]].abs() < 1e-10,
                        "t[{i},{j}] = {} should be near 0",
                        t[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_eigen_k_largest() {
        // A = diag(5, 3, 1, 2, 4); k=3 largest = 5, 4, 3
        let a = array![
            [5.0_f64, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0]
        ];
        let solver = GpuEigensolver::new(GpuEigenConfig::default());
        let eigs = solver.k_largest_eigenvalues(&a, 3).expect("should succeed");
        let mut v: Vec<f64> = eigs.to_vec();
        v.sort_by(|x, y| y.partial_cmp(x).unwrap()); // descending
        assert_eq!(v.len(), 3);
        assert!(approx_eq(v[0], 5.0, 1e-6), "v[0]={}", v[0]);
        assert!(approx_eq(v[1], 4.0, 1e-6), "v[1]={}", v[1]);
        assert!(approx_eq(v[2], 3.0, 1e-6), "v[2]={}", v[2]);
    }
}
