//! Tensor decomposition algorithms for the SciRS2 backend.
//!
//! Provides production-quality implementations of:
//! - **Truncated SVD** via deflation/power iteration (no BLAS/LAPACK dependency)
//! - **Mode-n unfolding / folding** for tensor-matrix conversions
//! - **Tucker-1** single-mode compression
//! - **CP / PARAFAC** via Alternating Least Squares (ALS) for 3-mode tensors
//! - **HOSVD** (Higher-Order SVD) multilinear compression
//!
//! All operations are pure Rust; no C or Fortran linkage is required.

use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during tensor decomposition.
#[derive(Debug, Clone)]
pub enum DecompositionError {
    /// Input shape is incompatible with the requested operation.
    ShapeError(String),
    /// Iterative algorithm did not converge within the allowed iterations.
    ConvergenceFailure { iterations: usize, residual: f64 },
    /// A matrix is numerically singular and cannot be inverted.
    SingularMatrix,
    /// Requested rank exceeds the maximum feasible rank for the dimension.
    InvalidRank { rank: usize, max_rank: usize },
    /// Operation applied to an empty tensor (at least one dim is 0).
    EmptyTensor,
    /// Operation requires a matrix (2-D array) but received an n-D tensor.
    NonMatrixInput { ndim: usize },
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeError(msg) => write!(f, "Shape error: {msg}"),
            Self::ConvergenceFailure {
                iterations,
                residual,
            } => write!(
                f,
                "Convergence failure after {iterations} iterations (residual={residual:.3e})"
            ),
            Self::SingularMatrix => write!(f, "Matrix is numerically singular"),
            Self::InvalidRank { rank, max_rank } => {
                write!(f, "Invalid rank {rank}: must be in 1..={max_rank}")
            }
            Self::EmptyTensor => write!(f, "Tensor has at least one zero-length dimension"),
            Self::NonMatrixInput { ndim } => {
                write!(f, "Expected a 2-D matrix, got a {ndim}-D tensor")
            }
        }
    }
}

impl std::error::Error for DecompositionError {}

// ---------------------------------------------------------------------------
// Truncated SVD
// ---------------------------------------------------------------------------

/// Result of a truncated SVD decomposition on a 2-D matrix.
#[derive(Debug, Clone)]
pub struct TruncatedSvd {
    /// Left singular vectors `[m, k]`.
    pub u: Array2<f64>,
    /// Singular values `[k]`, in descending order.
    pub s: Array1<f64>,
    /// Right singular vectors (transposed) `[k, n]`.
    pub vt: Array2<f64>,
    /// Number of singular components retained.
    pub rank: usize,
    /// Fraction of total variance explained: `Σ s[:k]² / Σ s_full²`.
    pub explained_variance_ratio: f64,
}

impl TruncatedSvd {
    /// Reconstruct the approximate matrix `U @ diag(s) @ Vt`.
    pub fn reconstruct(&self) -> Array2<f64> {
        let m = self.u.nrows();
        let n = self.vt.ncols();
        let mut result = Array2::<f64>::zeros((m, n));
        for i in 0..self.rank {
            let u_col = self.u.column(i);
            let vt_row = self.vt.row(i);
            let s_i = self.s[i];
            for r in 0..m {
                for c in 0..n {
                    result[[r, c]] += s_i * u_col[r] * vt_row[c];
                }
            }
        }
        result
    }

    /// Frobenius norm of `original - reconstructed`.
    pub fn reconstruction_error(&self, original: &Array2<f64>) -> f64 {
        let approx = self.reconstruct();
        let diff = original - &approx;
        frobenius_norm_2d(&diff)
    }
}

/// Compute truncated SVD using the deflation / power-iteration method.
///
/// The algorithm iterates over components one by one:
/// 1. Start with a random unit vector `v ∈ ℝⁿ`.
/// 2. Power-iterate: `v ← Mᵀ(Mv) / ‖Mᵀ(Mv)‖` for `n_iter` steps.
/// 3. Recover `u = Mv / ‖Mv‖`, `σ = uᵀMv`.
/// 4. Deflate: `M ← M − σ · uv ᵀ`.
///
/// # Parameters
/// - `matrix`: Input `[m, n]` matrix.
/// - `k`: Number of singular triplets to compute.
/// - `n_iter`: Power iterations per component (higher = more accurate; default 20).
/// - `tol`: Convergence tolerance for power iteration.
pub fn truncated_svd(
    matrix: &Array2<f64>,
    k: usize,
    n_iter: usize,
    tol: f64,
) -> Result<TruncatedSvd, DecompositionError> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    if m == 0 || n == 0 {
        return Err(DecompositionError::EmptyTensor);
    }
    let max_rank = m.min(n);
    if k == 0 || k > max_rank {
        return Err(DecompositionError::InvalidRank { rank: k, max_rank });
    }

    // Compute full Frobenius norm for explained variance denominator
    let total_sq: f64 = matrix.iter().map(|x| x * x).sum();

    let mut u_vecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut s_vals: Vec<f64> = Vec::with_capacity(k);
    let mut v_vecs: Vec<Vec<f64>> = Vec::with_capacity(k);

    // Working copy — we deflate this in-place
    let mut work = matrix.to_owned();

    for _comp in 0..k {
        // Initialise v with a deterministic pseudo-random vector
        let mut v = init_vector(n, _comp);
        normalize_vec(&mut v);

        let mut prev_sigma = f64::INFINITY;

        for _iter in 0..n_iter {
            // u = M v
            let u = mat_vec_mul(&work, &v);
            // v_new = Mᵀ u
            let mut v_new = mat_t_vec_mul(&work, &u);
            normalize_vec(&mut v_new);

            // convergence check via change in v
            let diff: f64 = v_new
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            v = v_new;

            let sigma_est = {
                let u_tmp = mat_vec_mul(&work, &v);
                dot_product(&u_tmp, &u_tmp).sqrt()
            };
            if (sigma_est - prev_sigma).abs() < tol && diff < tol {
                break;
            }
            prev_sigma = sigma_est;
        }

        // Final extraction
        let mut u = mat_vec_mul(&work, &v);
        let sigma = vec_norm(&u);
        if sigma < 1e-14 {
            // All remaining singular values are negligible
            u = vec![0.0; m];
        } else {
            u.iter_mut().for_each(|x| *x /= sigma);
        }

        // Deflate: work -= sigma * outer(u, v)
        for r in 0..m {
            for c in 0..n {
                work[[r, c]] -= sigma * u[r] * v[c];
            }
        }

        u_vecs.push(u);
        s_vals.push(sigma);
        v_vecs.push(v);
    }

    // Build result arrays
    let u_arr = Array2::from_shape_fn((m, k), |(r, c)| u_vecs[c][r]);
    let s_arr = Array1::from_vec(s_vals.clone());
    let vt_arr = Array2::from_shape_fn((k, n), |(r, c)| v_vecs[r][c]);

    let captured_sq: f64 = s_vals.iter().map(|s| s * s).sum();
    let explained_variance_ratio = if total_sq < 1e-30 {
        1.0
    } else {
        (captured_sq / total_sq).min(1.0)
    };

    Ok(TruncatedSvd {
        u: u_arr,
        s: s_arr,
        vt: vt_arr,
        rank: k,
        explained_variance_ratio,
    })
}

// ---------------------------------------------------------------------------
// Tensor unfolding / folding
// ---------------------------------------------------------------------------

/// Mode-n unfolding of a tensor: reshape to 2-D matrix where mode `n` becomes
/// the rows and all other modes are flattened column-wise.
///
/// Axis permutation: `[mode, 0, 1, …, mode-1, mode+1, …, ndim-1]`.
pub fn unfold(tensor: &ArrayD<f64>, mode: usize) -> Result<Array2<f64>, DecompositionError> {
    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(DecompositionError::ShapeError(
            "Cannot unfold a 0-D scalar tensor".into(),
        ));
    }
    if mode >= ndim {
        return Err(DecompositionError::ShapeError(format!(
            "mode {mode} out of range for {ndim}-D tensor"
        )));
    }
    if tensor.shape().contains(&0) {
        return Err(DecompositionError::EmptyTensor);
    }

    let shape = tensor.shape();
    let n_rows = shape[mode];
    let n_cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_i, &d)| d)
        .product();

    // Build permutation: [mode, 0, 1, …, mode-1, mode+1, …]
    let mut perm: Vec<usize> = Vec::with_capacity(ndim);
    perm.push(mode);
    for i in 0..ndim {
        if i != mode {
            perm.push(i);
        }
    }

    let permuted = tensor.view().permuted_axes(perm);
    // Collect in permuted order
    let data: Vec<f64> = permuted.iter().copied().collect();

    Array2::from_shape_vec((n_rows, n_cols), data)
        .map_err(|e| DecompositionError::ShapeError(e.to_string()))
}

/// Inverse of [`unfold`]: fold a 2-D matrix back to a tensor of the given shape.
///
/// The `mode` axis corresponds to the rows; all other axes are columns in the
/// same iteration order as [`unfold`].
pub fn fold(
    matrix: &Array2<f64>,
    mode: usize,
    shape: &[usize],
) -> Result<ArrayD<f64>, DecompositionError> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(DecompositionError::ShapeError(
            "Cannot fold to a 0-D tensor".into(),
        ));
    }
    if mode >= ndim {
        return Err(DecompositionError::ShapeError(format!(
            "mode {mode} out of range for {ndim}-D shape"
        )));
    }

    let n_rows = shape[mode];
    let n_cols: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != mode)
        .map(|(_i, &d)| d)
        .product();

    if matrix.nrows() != n_rows || matrix.ncols() != n_cols {
        return Err(DecompositionError::ShapeError(format!(
            "matrix shape {}×{} does not match expected {}×{} for mode={mode}, shape={shape:?}",
            matrix.nrows(),
            matrix.ncols(),
            n_rows,
            n_cols
        )));
    }

    // Build permuted shape: [mode, others...]
    let mut perm_shape: Vec<usize> = Vec::with_capacity(ndim);
    perm_shape.push(shape[mode]);
    for (i, &d) in shape.iter().enumerate() {
        if i != mode {
            perm_shape.push(d);
        }
    }

    let data: Vec<f64> = matrix.iter().copied().collect();
    let permuted = ArrayD::from_shape_vec(IxDyn(&perm_shape), data)
        .map_err(|e| DecompositionError::ShapeError(e.to_string()))?;

    // Inverse permutation: perm was [mode, 0..mode-1, mode+1..], inverse maps back
    // perm[0] = mode, perm[1] = 0, perm[2] = 1, ...
    // original axis i → permuted axis:
    //   mode → 0, j<mode → j+1, j>mode → j
    let mut inv_perm = vec![0usize; ndim];
    inv_perm[mode] = 0;
    let mut pos = 1usize;
    for (i, slot) in inv_perm.iter_mut().enumerate().take(ndim) {
        if i != mode {
            *slot = pos;
            pos += 1;
        }
    }

    let result = permuted.view().permuted_axes(inv_perm).to_owned();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tucker-1
// ---------------------------------------------------------------------------

/// Result of a Tucker-1 (single-mode) tensor decomposition.
#[derive(Debug, Clone)]
pub struct Tucker1Result {
    /// Compressed core tensor (same shape as original, except `shape[mode] = rank`).
    pub core: ArrayD<f64>,
    /// Factor matrix `[original_dim, rank]`.
    pub factor: Array2<f64>,
    /// The mode that was compressed.
    pub mode: usize,
    /// Rank used for this mode.
    pub rank: usize,
    /// `original_elements / core_elements`.
    pub compression_ratio: f64,
}

impl Tucker1Result {
    /// Reconstruct original tensor: core ×_mode factorᵀ  (Tucker product along mode).
    ///
    /// Equivalent to: unfold(core, mode) → factor @ unfolded → fold back.
    pub fn reconstruct(&self) -> ArrayD<f64> {
        // unfold core along mode → [rank, rest]
        let core_unfolded = match unfold(&self.core, self.mode) {
            Ok(m) => m,
            Err(_) => return self.core.clone(),
        };
        // factor: [orig_dim, rank], factor @ core_unfolded: [orig_dim, rest]
        let reconstructed_mat = self.factor.dot(&core_unfolded);

        // shape: replace mode dim with orig_dim
        let mut orig_shape: Vec<usize> = self.core.shape().to_vec();
        orig_shape[self.mode] = self.factor.nrows();

        match fold(&reconstructed_mat, self.mode, &orig_shape) {
            Ok(t) => t,
            Err(_) => ArrayD::zeros(IxDyn(&orig_shape)),
        }
    }

    /// Frobenius norm of `original - reconstruct()`.
    pub fn reconstruction_error(&self, original: &ArrayD<f64>) -> f64 {
        let approx = self.reconstruct();
        let diff = original - &approx;
        frobenius_norm_nd(&diff)
    }
}

/// Tucker-1 decomposition: compress tensor along a single mode using truncated SVD.
///
/// Steps:
/// 1. Unfold the tensor along `mode` → matrix `X_(mode)` of shape `[dim, rest]`.
/// 2. Compute truncated SVD of rank `rank`.
/// 3. `factor = U` (`[dim, rank]`).
/// 4. Core unfolding = `Uᵀ @ X_(mode)` → fold back.
pub fn tucker1(
    tensor: &ArrayD<f64>,
    mode: usize,
    rank: usize,
) -> Result<Tucker1Result, DecompositionError> {
    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(DecompositionError::ShapeError("0-D tensor".into()));
    }
    if mode >= ndim {
        return Err(DecompositionError::ShapeError(format!(
            "mode {mode} out of range for {ndim}-D tensor"
        )));
    }

    let orig_dim = tensor.shape()[mode];
    if orig_dim == 0 {
        return Err(DecompositionError::EmptyTensor);
    }
    if rank == 0 || rank > orig_dim {
        return Err(DecompositionError::InvalidRank {
            rank,
            max_rank: orig_dim,
        });
    }

    let unfolded = unfold(tensor, mode)?; // [orig_dim, rest]
    let svd = truncated_svd(&unfolded, rank, 30, 1e-10)?; // U:[m,k], s:[k], Vt:[k,n]

    // factor = U  ([orig_dim, rank])
    let factor = svd.u.clone();

    // core_unfolded = Uᵀ @ unfolded = [rank, rest]
    // U is [orig_dim, rank], so Uᵀ is [rank, orig_dim]
    let u_t = svd.u.t().to_owned(); // [rank, orig_dim]
    let core_unfolded = u_t.dot(&unfolded); // [rank, rest]

    // Build core shape
    let mut core_shape: Vec<usize> = tensor.shape().to_vec();
    core_shape[mode] = rank;

    let core = fold(&core_unfolded, mode, &core_shape)?;

    let original_elements: usize = tensor.shape().iter().product();
    let core_elements: usize = core_shape.iter().product();
    let compression_ratio = if core_elements == 0 {
        1.0
    } else {
        original_elements as f64 / core_elements as f64
    };

    Ok(Tucker1Result {
        core,
        factor,
        mode,
        rank,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// CP / PARAFAC decomposition (ALS, 3-mode)
// ---------------------------------------------------------------------------

/// Result of a CP (PARAFAC) tensor decomposition.
#[derive(Debug, Clone)]
pub struct CpDecomposition {
    /// Factor matrices, one per mode, each `[dim_i, rank]`.
    pub factors: Vec<Array2<f64>>,
    /// Normalization weights per component `[rank]`.
    pub weights: Array1<f64>,
    /// Number of components.
    pub rank: usize,
    /// Number of modes (tensor order).
    pub num_modes: usize,
    /// Number of ALS iterations actually performed.
    pub iterations: usize,
    /// Residual ‖X - X̂‖_F at convergence.
    pub final_residual: f64,
    /// Whether ALS converged within tolerance.
    pub converged: bool,
}

impl CpDecomposition {
    /// Reconstruct the tensor from factor matrices and weights.
    ///
    /// For each component r, accumulates `weights[r] * outer(A[:,r], B[:,r], C[:,r], …)`.
    pub fn reconstruct(&self) -> ArrayD<f64> {
        if self.factors.is_empty() {
            return ArrayD::zeros(IxDyn(&[]));
        }
        let shape: Vec<usize> = self.factors.iter().map(|f| f.nrows()).collect();
        let total: usize = shape.iter().product();
        let ndim = self.num_modes;

        let mut data = vec![0.0f64; total];

        for r in 0..self.rank {
            let w = self.weights[r];
            // Iterate over all multi-indices via flat index
            for (flat, slot) in data.iter_mut().enumerate().take(total) {
                let mut strides = vec![0usize; ndim];
                let mut remaining = flat;
                for d in (0..ndim).rev() {
                    strides[d] = remaining % shape[d];
                    remaining /= shape[d];
                }
                let contrib: f64 = self
                    .factors
                    .iter()
                    .zip(strides.iter())
                    .map(|(f, &i)| f[[i, r]])
                    .product();
                *slot += w * contrib;
            }
        }

        ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap_or_else(|_| ArrayD::zeros(IxDyn(&shape)))
    }

    /// Frobenius reconstruction error ‖original - reconstructed‖_F.
    pub fn reconstruction_error(&self, original: &ArrayD<f64>) -> f64 {
        let approx = self.reconstruct();
        let diff = original - &approx;
        frobenius_norm_nd(&diff)
    }

    /// Fraction of variance explained: `1 - ‖residual‖² / ‖original‖²`.
    pub fn explained_variance(&self, original: &ArrayD<f64>) -> f64 {
        let original_sq: f64 = original.iter().map(|x| x * x).sum();
        if original_sq < 1e-30 {
            return 1.0;
        }
        let residual = self.reconstruction_error(original);
        let explained = 1.0 - (residual * residual) / original_sq;
        explained.clamp(0.0, 1.0)
    }
}

/// Khatri-Rao product (column-wise Kronecker product).
///
/// For `A: [m, r]` and `B: [n, r]` → output `[mn, r]`.
fn khatri_rao(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, r_a) = (a.nrows(), a.ncols());
    let (n, r_b) = (b.nrows(), b.ncols());
    debug_assert_eq!(
        r_a, r_b,
        "Khatri-Rao: both matrices must have same number of columns"
    );
    let r = r_a.min(r_b);
    let mut result = Array2::<f64>::zeros((m * n, r));
    for col in 0..r {
        for i in 0..m {
            for j in 0..n {
                result[[i * n + j, col]] = a[[i, col]] * b[[j, col]];
            }
        }
    }
    result
}

/// Element-wise (Hadamard) product of two 2-D arrays (same shape).
fn hadamard(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a * b
}

/// Gram matrix: `Aᵀ A` for `A: [m, r]` → `[r, r]`.
fn gram(a: &Array2<f64>) -> Array2<f64> {
    a.t().dot(a)
}

/// Pseudo-inverse of a small square matrix via direct inversion with regularisation.
///
/// For the small `[r, r]` matrices in CP-ALS this is sufficient and avoids
/// pulling in a full linear-algebra solver.
fn pinv_small(m: &Array2<f64>, lambda: f64) -> Result<Array2<f64>, DecompositionError> {
    let n = m.nrows();
    debug_assert_eq!(n, m.ncols());

    // Regularised matrix: M + λI
    let mut reg = m.to_owned();
    for i in 0..n {
        reg[[i, i]] += lambda;
    }

    // Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = reg.row(i).to_vec();
            // Append identity column
            for j in 0..n {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot
        let pivot = (col..n).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot = pivot.ok_or(DecompositionError::SingularMatrix)?;
        aug.swap(col, pivot);

        let diag = aug[col][col];
        if diag.abs() < 1e-15 {
            return Err(DecompositionError::SingularMatrix);
        }

        let inv_diag = 1.0 / diag;
        for elem in aug[col].iter_mut().take(2 * n) {
            *elem *= inv_diag;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                let col_row: Vec<f64> = aug[col][..2 * n].to_vec();
                for (j, &cv) in col_row.iter().enumerate().take(2 * n) {
                    aug[row][j] -= factor * cv;
                }
            }
        }
    }

    // Extract inverse from right half
    let inv = Array2::from_shape_fn((n, n), |(i, j)| aug[i][n + j]);
    Ok(inv)
}

/// CP (PARAFAC) decomposition of a 3-mode tensor via Alternating Least Squares.
///
/// # Algorithm (ALS)
/// Given tensor X ∈ ℝ^{I×J×K} and rank R:
/// 1. Initialise factor matrices A, B, C randomly (or via HOSVD init for better
///    convergence).
/// 2. Repeat:
///    - `A ← X_(0) (C ⊙ B) (CᵀC * BᵀB)⁺`
///    - `B ← X_(1) (C ⊙ A) (CᵀC * AᵀA)⁺`
///    - `C ← X_(2) (B ⊙ A) (BᵀB * AᵀA)⁺`
///    - Normalise columns, track weights.
/// 3. Check convergence via relative change in residual.
///
/// ⊙ denotes the Khatri-Rao (column-wise Kronecker) product.
pub fn cp_als(
    tensor: &ArrayD<f64>,
    rank: usize,
    max_iter: usize,
    tol: f64,
) -> Result<CpDecomposition, DecompositionError> {
    if tensor.ndim() != 3 {
        return Err(DecompositionError::ShapeError(format!(
            "cp_als currently supports only 3-mode tensors, got {}-mode",
            tensor.ndim()
        )));
    }
    let shape = tensor.shape();
    let (i_dim, j_dim, k_dim) = (shape[0], shape[1], shape[2]);
    if i_dim == 0 || j_dim == 0 || k_dim == 0 {
        return Err(DecompositionError::EmptyTensor);
    }
    let max_rank = i_dim.min(j_dim).min(k_dim);
    if rank == 0 || rank > max_rank {
        return Err(DecompositionError::InvalidRank { rank, max_rank });
    }

    // Precompute unfoldings (read-only, computed once)
    let x0 = unfold(tensor, 0)?; // [I, J*K]
    let x1 = unfold(tensor, 1)?; // [J, I*K]
    let x2 = unfold(tensor, 2)?; // [K, I*J]

    let tensor_norm_sq: f64 = tensor.iter().map(|x| x * x).sum();

    // Initialise factors via truncated SVD (HOSVD-style warm start)
    let mut a = init_factor_svd(&x0, rank)?; // [I, rank]
    let mut b = init_factor_svd(&x1, rank)?; // [J, rank]
    let mut c = init_factor_svd(&x2, rank)?; // [K, rank]

    let mut weights = Array1::<f64>::ones(rank);
    let mut prev_residual = f64::INFINITY;
    let mut converged = false;
    let mut iter = 0usize;
    let regularization = 1e-10;

    for _it in 0..max_iter {
        iter = _it + 1;

        // --- Update A ---
        // X_(0) shape [I, J*K]: J is the outer (slow) index, K is inner (fast).
        // The matching KR product is B ⊙ C: rows indexed as [j*K + k] → shape [J*K, R].
        {
            let kr_bc = khatri_rao(&b, &c); // [J*K, rank]
            let gram_prod = hadamard(&gram(&b), &gram(&c)); // [rank, rank]
            let gram_inv = pinv_small(&gram_prod, regularization)?;
            let rhs = kr_bc.dot(&gram_inv); // [J*K, rank]
            a = x0.dot(&rhs); // [I, rank]
        }

        // --- Update B ---
        // X_(1) shape [J, I*K]: I is outer, K is inner.
        // Matching KR product is A ⊙ C: rows [i*K + k] → shape [I*K, R].
        {
            let kr_ac = khatri_rao(&a, &c); // [I*K, rank]
            let gram_prod = hadamard(&gram(&a), &gram(&c));
            let gram_inv = pinv_small(&gram_prod, regularization)?;
            let rhs = kr_ac.dot(&gram_inv);
            b = x1.dot(&rhs); // [J, rank]
        }

        // --- Update C ---
        // X_(2) shape [K, I*J]: I is outer, J is inner.
        // Matching KR product is A ⊙ B: rows [i*J + j] → shape [I*J, R].
        {
            let kr_ab = khatri_rao(&a, &b); // [I*J, rank]
            let gram_prod = hadamard(&gram(&a), &gram(&b));
            let gram_inv = pinv_small(&gram_prod, regularization)?;
            let rhs = kr_ab.dot(&gram_inv);
            c = x2.dot(&rhs); // [K, rank]
        }

        // --- Normalise columns, store weights ---
        for r in 0..rank {
            let norm_a = col_norm(&a, r);
            let norm_b = col_norm(&b, r);
            let norm_c = col_norm(&c, r);
            let w = norm_a * norm_b * norm_c;
            weights[r] = w;
            if norm_a > 1e-14 {
                a.column_mut(r).mapv_inplace(|x| x / norm_a);
            }
            if norm_b > 1e-14 {
                b.column_mut(r).mapv_inplace(|x| x / norm_b);
            }
            if norm_c > 1e-14 {
                c.column_mut(r).mapv_inplace(|x| x / norm_c);
            }
        }

        // --- Convergence check ---
        // Efficient residual: ‖X‖² - 2 ⟨X, X̂⟩ + ‖X̂‖²
        // For simplicity we use the reconstruction-based residual every iteration.
        // For large tensors this can be made cheaper; correctness first.
        let approx_norm_sq = compute_cp_norm_sq(&a, &b, &c, &weights, rank);
        let inner_xhat = compute_inner_x_xhat(&x0, &a, &b, &c, &weights, rank);
        let residual_sq = (tensor_norm_sq - 2.0 * inner_xhat + approx_norm_sq).max(0.0);
        let residual = residual_sq.sqrt();

        let rel_change = if prev_residual.is_finite() && prev_residual > 1e-30 {
            (prev_residual - residual).abs() / prev_residual
        } else {
            f64::INFINITY
        };

        if rel_change < tol && _it > 0 {
            converged = true;
            prev_residual = residual;
            break;
        }
        prev_residual = residual;
    }

    Ok(CpDecomposition {
        factors: vec![a, b, c],
        weights,
        rank,
        num_modes: 3,
        iterations: iter,
        final_residual: prev_residual,
        converged,
    })
}

// ---------------------------------------------------------------------------
// HOSVD
// ---------------------------------------------------------------------------

/// Result of a Higher-Order SVD (HOSVD) decomposition.
#[derive(Debug, Clone)]
pub struct HosvdResult {
    /// Compressed core tensor with shape `ranks`.
    pub core: ArrayD<f64>,
    /// Factor matrices, one per mode, each `[dim_i, rank_i]`.
    pub factors: Vec<Array2<f64>>,
    /// Target ranks per mode.
    pub ranks: Vec<usize>,
    /// `original_elements / core_elements`.
    pub compression_ratio: f64,
}

impl HosvdResult {
    /// Reconstruct the original tensor via successive Tucker products along each mode.
    pub fn reconstruct(&self) -> ArrayD<f64> {
        let mut current = self.core.clone();
        for (mode, factor) in self.factors.iter().enumerate() {
            current = match tucker_product(&current, factor, mode) {
                Ok(t) => t,
                Err(_) => return self.core.clone(),
            };
        }
        current
    }

    /// Frobenius reconstruction error.
    pub fn reconstruction_error(&self, original: &ArrayD<f64>) -> f64 {
        let approx = self.reconstruct();
        let diff = original - &approx;
        frobenius_norm_nd(&diff)
    }
}

/// Higher-Order SVD: compute truncated SVD along each mode independently.
///
/// For each mode n:
///   1. Unfold tensor along mode n → `X_(n)`.
///   2. Compute `truncated_svd(X_(n), ranks[n])` → `U_n`.
///
/// Core = X ×_1 U_1ᵀ ×_2 U_2ᵀ … ×_N U_Nᵀ (multi-mode product).
pub fn hosvd(tensor: &ArrayD<f64>, ranks: &[usize]) -> Result<HosvdResult, DecompositionError> {
    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(DecompositionError::ShapeError("0-D tensor".into()));
    }
    if ranks.len() != ndim {
        return Err(DecompositionError::ShapeError(format!(
            "ranks length {} must match tensor ndim {}",
            ranks.len(),
            ndim
        )));
    }
    if tensor.shape().contains(&0) {
        return Err(DecompositionError::EmptyTensor);
    }
    for (mode, &r) in ranks.iter().enumerate() {
        let dim = tensor.shape()[mode];
        if r == 0 || r > dim {
            return Err(DecompositionError::InvalidRank {
                rank: r,
                max_rank: dim,
            });
        }
    }

    let mut factors: Vec<Array2<f64>> = Vec::with_capacity(ndim);

    for (mode, &rank_m) in ranks.iter().enumerate().take(ndim) {
        let unfolded = unfold(tensor, mode)?;
        let svd = truncated_svd(&unfolded, rank_m, 30, 1e-10)?;
        factors.push(svd.u); // [dim_mode, rank_mode]
    }

    // Compute core = X ×_0 U_0ᵀ ×_1 U_1ᵀ … ×_(N-1) U_(N-1)ᵀ
    // We do this mode by mode: multiply the current tensor by Uᵀ along each mode.
    let mut core = tensor.to_owned();
    for (mode, factor) in factors.iter().enumerate() {
        // unfold along mode, multiply by Uᵀ ([rank, dim] @ [dim, rest]) → fold back
        let unfolded_core = unfold(&core, mode)?; // [dim_mode, rest]
        let u_t = factor.t().to_owned(); // [rank_mode, dim_mode]
        let compressed = u_t.dot(&unfolded_core); // [rank_mode, rest]

        let mut new_shape: Vec<usize> = core.shape().to_vec();
        new_shape[mode] = ranks[mode];

        core = fold(&compressed, mode, &new_shape)?;
    }

    let original_elements: usize = tensor.shape().iter().product();
    let core_elements: usize = ranks.iter().product();
    let compression_ratio = if core_elements == 0 {
        1.0
    } else {
        original_elements as f64 / core_elements as f64
    };

    Ok(HosvdResult {
        core,
        factors,
        ranks: ranks.to_vec(),
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Tucker mode product: tensor ×_mode factor ([dim, rank] → expand mode).
///
/// Equivalent to: unfold(tensor, mode) → factor @ unfolded → fold.
fn tucker_product(
    tensor: &ArrayD<f64>,
    factor: &Array2<f64>,
    mode: usize,
) -> Result<ArrayD<f64>, DecompositionError> {
    let unfolded = unfold(tensor, mode)?; // [old_dim, rest]
    let result_mat = factor.dot(&unfolded); // [new_dim, rest]

    let mut new_shape: Vec<usize> = tensor.shape().to_vec();
    new_shape[mode] = factor.nrows();

    fold(&result_mat, mode, &new_shape)
}

/// Initialise a factor matrix from the left singular vectors of `matrix`.
fn init_factor_svd(matrix: &Array2<f64>, rank: usize) -> Result<Array2<f64>, DecompositionError> {
    let effective_rank = rank.min(matrix.nrows().min(matrix.ncols()));
    if effective_rank == 0 {
        return Err(DecompositionError::EmptyTensor);
    }
    let svd = truncated_svd(matrix, effective_rank, 20, 1e-10)?;
    // Pad with random-ish columns if rank > effective_rank
    if effective_rank == rank {
        return Ok(svd.u);
    }
    let m = matrix.nrows();
    let mut factor = Array2::<f64>::zeros((m, rank));
    for c in 0..effective_rank {
        for r in 0..m {
            factor[[r, c]] = svd.u[[r, c]];
        }
    }
    // Fill remaining columns with pseudo-random unit vectors
    for c in effective_rank..rank {
        let mut v = init_vector(m, c);
        normalize_vec(&mut v);
        for r in 0..m {
            factor[[r, c]] = v[r];
        }
    }
    Ok(factor)
}

/// Deterministic pseudo-random unit vector of length `n`, seeded by `seed`.
fn init_vector(n: usize, seed: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = i
                .wrapping_mul(6364136223846793005usize)
                .wrapping_add(seed.wrapping_mul(1442695040888963407usize))
                as u64;
            (x as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

fn normalize_vec(v: &mut [f64]) {
    let norm = vec_norm(v);
    if norm > 1e-14 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

fn vec_norm(v: &[f64]) -> f64 {
    dot_product(v, v).sqrt()
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product: `A @ v` where `A: [m, n]`, `v: [n]` → `[m]`.
fn mat_vec_mul(a: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = vec![0.0f64; m];
    for i in 0..m {
        let mut s = 0.0;
        for j in 0..n {
            s += a[[i, j]] * v[j];
        }
        result[i] = s;
    }
    result
}

/// Matrix-transpose-vector product: `Aᵀ @ v` where `A: [m, n]`, `v: [m]` → `[n]`.
fn mat_t_vec_mul(a: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..m {
            s += a[[i, j]] * v[i];
        }
        result[j] = s;
    }
    result
}

fn frobenius_norm_2d(m: &Array2<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn frobenius_norm_nd(t: &ArrayD<f64>) -> f64 {
    t.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn col_norm(a: &Array2<f64>, col: usize) -> f64 {
    a.column(col).iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// ‖X̂‖²_F computed directly from factor matrices (Gram approach).
///
/// ‖X̂‖² = (AᵀA * BᵀB * CᵀC) ⊙ wwᵀ summed over all entries.
fn compute_cp_norm_sq(
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
    weights: &Array1<f64>,
    rank: usize,
) -> f64 {
    let ga = gram(a); // [rank, rank]
    let gb = gram(b);
    let gc = gram(c);

    let mut norm_sq = 0.0f64;
    for r1 in 0..rank {
        for r2 in 0..rank {
            let val = ga[[r1, r2]] * gb[[r1, r2]] * gc[[r1, r2]] * weights[r1] * weights[r2];
            norm_sq += val;
        }
    }
    norm_sq
}

/// Inner product ⟨X, X̂⟩ using the mode-0 unfolding.
///
/// X_(0) has shape [I, J*K] with J-outer/K-inner ordering.
/// The matching KR product is B ⊙ C (B outer, C inner) → [J*K, rank].
/// ⟨X, X̂⟩ = Σ_r w_r · (X_(0) (b_r ⊗ c_r)) · a_r
fn compute_inner_x_xhat(
    x0: &Array2<f64>, // [I, J*K]
    a: &Array2<f64>,  // [I, rank]
    b: &Array2<f64>,  // [J, rank]
    c: &Array2<f64>,  // [K, rank]
    weights: &Array1<f64>,
    rank: usize,
) -> f64 {
    let kr = khatri_rao(b, c); // [J*K, rank] — B outer (slow), C inner (fast)
                               // x0 @ kr → [I, rank]
    let mttkrp = x0.dot(&kr); // [I, rank]
                              // inner = sum_r weights[r] * (A[:,r] · mttkrp[:,r])
    let mut inner = 0.0f64;
    for r in 0..rank {
        let dot: f64 = a
            .column(r)
            .iter()
            .zip(mttkrp.column(r).iter())
            .map(|(x, y)| x * y)
            .sum();
        inner += weights[r] * dot;
    }
    inner
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::IxDyn;

    const TOL: f64 = 1e-6;
    const ALS_TOL: f64 = 1e-5;

    // Helper: build a random-ish ArrayD from a seed
    fn make_tensor(shape: &[usize], seed: usize) -> ArrayD<f64> {
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let x = i
                    .wrapping_mul(6364136223846793005usize)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407usize))
                    as u64;
                (x as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("shape ok")
    }

    fn make_matrix(rows: usize, cols: usize, seed: usize) -> Array2<f64> {
        let n = rows * cols;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let x = i
                    .wrapping_mul(6364136223846793005usize)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407usize))
                    as u64;
                (x as f64 / u64::MAX as f64) * 2.0 - 1.0
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data).expect("shape ok")
    }

    // --- Truncated SVD tests ---

    #[test]
    fn test_truncated_svd_rank1() {
        // Rank-1 matrix: outer product of two vectors
        let u_true: Vec<f64> = (0..5).map(|i| (i + 1) as f64).collect();
        let v_true: Vec<f64> = (0..4).map(|i| (i + 1) as f64).collect();
        let mut mat = Array2::<f64>::zeros((5, 4));
        for i in 0..5 {
            for j in 0..4 {
                mat[[i, j]] = u_true[i] * v_true[j];
            }
        }
        let svd = truncated_svd(&mat, 1, 40, 1e-12).expect("svd ok");
        let recon = svd.reconstruct();
        let err = (mat - recon).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-6, "rank-1 reconstruction error too large: {err}");
    }

    #[test]
    fn test_truncated_svd_reconstruction_error_decreases_with_rank() {
        let mat = make_matrix(8, 6, 42);
        let err1 = truncated_svd(&mat, 1, 30, 1e-12)
            .expect("ok")
            .reconstruction_error(&mat);
        let err3 = truncated_svd(&mat, 3, 30, 1e-12)
            .expect("ok")
            .reconstruction_error(&mat);
        let err6 = truncated_svd(&mat, 6, 30, 1e-12)
            .expect("ok")
            .reconstruction_error(&mat);
        assert!(
            err3 <= err1 + 1e-8,
            "rank-3 err ({err3}) should be <= rank-1 err ({err1})"
        );
        assert!(
            err6 <= err3 + 1e-8,
            "rank-6 err ({err6}) should be <= rank-3 err ({err3})"
        );
    }

    #[test]
    fn test_truncated_svd_singular_values_descending() {
        let mat = make_matrix(7, 5, 99);
        let svd = truncated_svd(&mat, 4, 30, 1e-12).expect("ok");
        for i in 0..svd.rank - 1 {
            assert!(
                svd.s[i] >= svd.s[i + 1] - 1e-8,
                "singular values not descending: s[{i}]={} < s[{}]={}",
                svd.s[i],
                i + 1,
                svd.s[i + 1]
            );
        }
    }

    #[test]
    fn test_truncated_svd_explained_variance_full_rank() {
        let mat = make_matrix(4, 4, 7);
        let max_rank = 4;
        let svd = truncated_svd(&mat, max_rank, 50, 1e-14).expect("ok");
        assert!(
            svd.explained_variance_ratio > 0.99,
            "full-rank EVR should be ≈1, got {}",
            svd.explained_variance_ratio
        );
    }

    #[test]
    fn test_truncated_svd_invalid_rank() {
        let mat = make_matrix(3, 4, 1);
        let result = truncated_svd(&mat, 0, 10, 1e-10);
        assert!(result.is_err(), "rank=0 should fail");
        let result2 = truncated_svd(&mat, 5, 10, 1e-10);
        assert!(result2.is_err(), "rank > min(m,n) should fail");
    }

    // --- Unfold / fold tests ---

    #[test]
    fn test_unfold_mode0_shape() {
        let tensor = make_tensor(&[3, 4, 5], 1);
        let mat = unfold(&tensor, 0).expect("ok");
        assert_eq!(mat.nrows(), 3, "mode-0 rows should be dim-0");
        assert_eq!(mat.ncols(), 4 * 5, "mode-0 cols should be dim-1 * dim-2");
    }

    #[test]
    fn test_unfold_mode1_shape() {
        let tensor = make_tensor(&[3, 4, 5], 2);
        let mat = unfold(&tensor, 1).expect("ok");
        assert_eq!(mat.nrows(), 4, "mode-1 rows should be dim-1");
        assert_eq!(mat.ncols(), 3 * 5, "mode-1 cols should be dim-0 * dim-2");
    }

    #[test]
    fn test_fold_roundtrip() {
        let original = make_tensor(&[3, 4, 5], 3);
        for mode in 0..3usize {
            let mat = unfold(&original, mode).expect("unfold ok");
            let recovered = fold(&mat, mode, &[3, 4, 5]).expect("fold ok");
            let err = (&original - &recovered)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(err < TOL, "fold(unfold(x, {mode})) != x, error={err}");
        }
    }

    // --- Tucker-1 tests ---

    #[test]
    fn test_tucker1_compression_ratio() {
        let tensor = make_tensor(&[10, 8, 6], 5);
        let result = tucker1(&tensor, 0, 3).expect("ok");
        assert!(
            result.compression_ratio > 1.0,
            "compressed core should be smaller than original, ratio={}",
            result.compression_ratio
        );
        assert_eq!(
            result.core.shape()[0],
            3,
            "core mode-0 dim should equal rank"
        );
    }

    #[test]
    fn test_tucker1_reconstruction_error_small() {
        // Build a tensor that is exactly rank-2 along mode 0
        let _basis: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let factor_true =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.3, 0.7])
                .expect("ok");
        // core: shape [2, 3, 3]
        let core_true = make_tensor(&[2, 3, 3], 11);
        // Reconstruct tensor via Tucker-product
        let core_unfolded = unfold(&core_true, 0).expect("ok"); // [2, 9]
        let t_unfolded = factor_true.dot(&core_unfolded); // [4, 9]
        let tensor = fold(&t_unfolded, 0, &[4, 3, 3]).expect("ok");

        let result = tucker1(&tensor, 0, 2).expect("ok");
        let err = result.reconstruction_error(&tensor);
        assert!(
            err < 1e-5,
            "Tucker-1 error too large for rank-2 tensor: {err}"
        );
    }

    #[test]
    fn test_tucker1_invalid_rank() {
        let tensor = make_tensor(&[3, 4, 5], 6);
        let result = tucker1(&tensor, 0, 10); // rank > dim-0 = 3
        assert!(result.is_err(), "rank > dim should return error");
    }

    #[test]
    fn test_tucker1_reconstruct_shape() {
        let tensor = make_tensor(&[5, 4, 3], 8);
        let result = tucker1(&tensor, 1, 2).expect("ok");
        let recon = result.reconstruct();
        assert_eq!(
            recon.shape(),
            tensor.shape(),
            "reconstructed shape must match original"
        );
    }

    // --- CP-ALS tests ---

    #[test]
    fn test_cp_als_3mode() {
        let tensor = make_tensor(&[4, 3, 3], 20);
        let result = cp_als(&tensor, 2, 200, ALS_TOL).expect("ok");
        assert_eq!(result.num_modes, 3);
        assert_eq!(result.rank, 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_cp_als_reconstruction_error() {
        // Build a rank-2 tensor exactly
        let a = make_matrix(4, 2, 30);
        let b = make_matrix(3, 2, 31);
        let c = make_matrix(3, 2, 32);
        // Reconstruct exactly
        let mut data = vec![0.0f64; 4 * 3 * 3];
        for i in 0..4 {
            for j in 0..3 {
                for k in 0..3 {
                    let v: f64 = (0..2).map(|r| a[[i, r]] * b[[j, r]] * c[[k, r]]).sum();
                    data[i * 9 + j * 3 + k] = v;
                }
            }
        }
        let tensor = ArrayD::from_shape_vec(IxDyn(&[4, 3, 3]), data).expect("ok");
        let result = cp_als(&tensor, 2, 300, 1e-8).expect("ok");
        let err = result.reconstruction_error(&tensor);
        // Allow some numerical tolerance
        assert!(
            err < 0.5,
            "CP-ALS reconstruction error too large for rank-2 tensor: {err}"
        );
    }

    #[test]
    fn test_cp_als_factors_shape() {
        let tensor = make_tensor(&[5, 4, 3], 40);
        let result = cp_als(&tensor, 2, 100, ALS_TOL).expect("ok");
        assert_eq!(result.factors.len(), 3);
        assert_eq!(result.factors[0].shape(), &[5, 2]);
        assert_eq!(result.factors[1].shape(), &[4, 2]);
        assert_eq!(result.factors[2].shape(), &[3, 2]);
    }

    #[test]
    fn test_cp_als_weights_positive() {
        let tensor = make_tensor(&[4, 3, 3], 50);
        let result = cp_als(&tensor, 2, 100, ALS_TOL).expect("ok");
        for (i, &w) in result.weights.iter().enumerate() {
            assert!(w >= 0.0, "weight[{i}] should be non-negative, got {w}");
        }
    }

    // --- HOSVD tests ---

    #[test]
    fn test_hosvd_shape() {
        let tensor = make_tensor(&[6, 5, 4], 60);
        let ranks = vec![3, 2, 2];
        let result = hosvd(&tensor, &ranks).expect("ok");
        assert_eq!(result.core.shape(), &[3usize, 2, 2][..]);
    }

    #[test]
    fn test_hosvd_factors_shape() {
        let tensor = make_tensor(&[6, 5, 4], 61);
        let ranks = vec![3, 2, 2];
        let result = hosvd(&tensor, &ranks).expect("ok");
        assert_eq!(result.factors.len(), 3);
        assert_eq!(result.factors[0].shape(), &[6, 3]);
        assert_eq!(result.factors[1].shape(), &[5, 2]);
        assert_eq!(result.factors[2].shape(), &[4, 2]);
    }

    #[test]
    fn test_hosvd_reconstruction_error_small() {
        // Full-rank HOSVD should reconstruct perfectly
        let tensor = make_tensor(&[3, 3, 3], 62);
        let ranks = vec![3, 3, 3];
        let result = hosvd(&tensor, &ranks).expect("ok");
        let err = result.reconstruction_error(&tensor);
        assert!(err < 1e-5, "Full-rank HOSVD error should be near 0: {err}");
    }

    // --- Error display tests ---

    #[test]
    fn test_decomposition_error_display() {
        let errors = vec![
            DecompositionError::ShapeError("bad shape".into()),
            DecompositionError::ConvergenceFailure {
                iterations: 100,
                residual: 1e-3,
            },
            DecompositionError::SingularMatrix,
            DecompositionError::InvalidRank {
                rank: 10,
                max_rank: 5,
            },
            DecompositionError::EmptyTensor,
            DecompositionError::NonMatrixInput { ndim: 3 },
        ];
        for e in &errors {
            let s = format!("{e}");
            assert!(!s.is_empty(), "Display for {e:?} should not be empty");
        }
    }

    // --- CP explained variance ---

    #[test]
    fn test_cp_decomp_explained_variance() {
        let tensor = make_tensor(&[4, 3, 3], 70);
        let result = cp_als(&tensor, 2, 100, ALS_TOL).expect("ok");
        let ev = result.explained_variance(&tensor);
        assert!(
            (0.0..=1.0).contains(&ev),
            "explained variance must be in [0,1], got {ev}"
        );
    }
}
