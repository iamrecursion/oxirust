//! # Kernel Alignment
//!
//! Implements Kernel Target Alignment (KTA), Centered Kernel Alignment (CKA),
//! HSIC (Hilbert-Schmidt Independence Criterion), and optimization routines for
//! selecting kernel hyperparameters based on alignment with a target kernel.
//!
//! ## Overview
//!
//! Kernel alignment metrics quantify how well a kernel matrix captures the
//! structure of the learning problem. Given labels, one constructs an "ideal"
//! target kernel where `T[i,j] = +1` if `labels[i] == labels[j]` and `-1`
//! otherwise. A high alignment between `K` and `T` indicates that the kernel
//! maps similar-class points close together.
//!
//! ### Kernel Target Alignment (KTA)
//!
//! ```text
//! KTA(K, T) = <K, T>_F / (||K||_F * ||T||_F)
//! ```
//!
//! ### Centered Kernel Alignment (CKA)
//!
//! CKA applies double-centering before computing alignment, making it invariant
//! to isotropic scaling and constant shifts:
//!
//! ```text
//! CKA(K1, K2) = HSIC(K1, K2) / sqrt(HSIC(K1,K1) * HSIC(K2,K2))
//! ```
//!
//! where `HSIC(K, L) = (1/n^2) * <H*K*H, H*L*H>_F`.
//!
//! ## References
//!
//! - Cortes, C., Mohri, M., & Rostamizadeh, A. (2012). Algorithms for learning
//!   kernels based on centered alignment. JMLR.
//! - Kornblith, S., et al. (2019). Similarity of neural network representations
//!   revisited. ICML.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during kernel alignment computations.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignmentError {
    /// The supplied data does not form a square matrix.
    NonSquareMatrix,
    /// The two matrices have incompatible sizes.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Received dimension.
        got: usize,
    },
    /// A numerical issue was encountered (e.g. zero-norm matrix).
    NumericalError(String),
    /// The matrix is singular or near-singular.
    SingularMatrix,
}

impl fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquareMatrix => write!(f, "Matrix is not square"),
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "Dimension mismatch: expected {}×{}, got {}×{}",
                expected, expected, got, got
            ),
            Self::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            Self::SingularMatrix => write!(f, "Matrix is singular or near-singular"),
        }
    }
}

impl std::error::Error for AlignmentError {}

// ---------------------------------------------------------------------------
// KernelMatrix
// ---------------------------------------------------------------------------

/// A square kernel matrix (n×n, symmetric positive semi-definite).
///
/// All matrix operations are implemented from scratch over `Vec<Vec<f64>>`.
#[derive(Debug, Clone)]
pub struct KernelMatrix {
    data: Vec<Vec<f64>>,
    n: usize,
}

impl KernelMatrix {
    /// Construct from a row-major `Vec<Vec<f64>>`.
    ///
    /// Returns `Err(AlignmentError::NonSquareMatrix)` if any row has a length
    /// different from the number of rows.
    pub fn new(data: Vec<Vec<f64>>) -> Result<KernelMatrix, AlignmentError> {
        let n = data.len();
        for row in &data {
            if row.len() != n {
                return Err(AlignmentError::NonSquareMatrix);
            }
        }
        Ok(KernelMatrix { data, n })
    }

    /// Construct from a flat slice of length `n*n` in row-major order.
    pub fn from_flat(flat: &[f64], n: usize) -> Result<KernelMatrix, AlignmentError> {
        if flat.len() != n * n {
            return Err(AlignmentError::NonSquareMatrix);
        }
        let data = (0..n).map(|i| flat[i * n..(i + 1) * n].to_vec()).collect();
        Ok(KernelMatrix { data, n })
    }

    /// Construct the `n×n` identity kernel matrix.
    pub fn identity(n: usize) -> KernelMatrix {
        let mut data = vec![vec![0.0_f64; n]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            data[i][i] = 1.0;
        }
        KernelMatrix { data, n }
    }

    /// Construct the "ideal" label kernel:
    /// `K[i,j] = 1.0` if `labels[i] == labels[j]`, else `-1.0`.
    ///
    /// This is equivalent to the outer product of the label sign vector and is
    /// the target used in KTA for binary or multi-class classification.
    pub fn from_labels(labels: &[f64]) -> KernelMatrix {
        let n = labels.len();
        let mut data = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                // Use approximate equality to handle floating-point labels
                data[i][j] = if (labels[i] - labels[j]).abs() < 1e-10 {
                    1.0
                } else {
                    -1.0
                };
            }
        }
        KernelMatrix { data, n }
    }

    /// Return element `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    /// Return the dimension `n` (number of rows/columns).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Compute the trace: `sum_i K[i,i]`.
    pub fn trace(&self) -> f64 {
        (0..self.n).map(|i| self.data[i][i]).sum()
    }

    /// Compute `||K||_F^2 = sum_{i,j} K[i,j]^2`.
    pub fn frobenius_norm_sq(&self) -> f64 {
        self.data
            .iter()
            .flat_map(|row| row.iter())
            .map(|&v| v * v)
            .sum()
    }

    /// Compute `<K1, K2>_F = sum_{i,j} K1[i,j] * K2[i,j]`.
    ///
    /// Returns `Err(AlignmentError::DimensionMismatch)` if the matrices have
    /// different sizes.
    pub fn frobenius_inner(&self, other: &KernelMatrix) -> Result<f64, AlignmentError> {
        if self.n != other.n {
            return Err(AlignmentError::DimensionMismatch {
                expected: self.n,
                got: other.n,
            });
        }
        let mut sum = 0.0_f64;
        for i in 0..self.n {
            for j in 0..self.n {
                sum += self.data[i][j] * other.data[i][j];
            }
        }
        Ok(sum)
    }

    /// Double-center the kernel matrix: `K_c = H * K * H`
    /// where `H = I - (1/n) * 1*1^T` is the centering matrix.
    ///
    /// Equivalent to:
    /// ```text
    /// K_c[i,j] = K[i,j] - row_mean[i] - col_mean[j] + grand_mean
    /// ```
    pub fn center(&self) -> KernelMatrix {
        let n = self.n;
        let n_f = n as f64;

        // Row means
        let row_means: Vec<f64> = self
            .data
            .iter()
            .map(|row| row.iter().sum::<f64>() / n_f)
            .collect();

        // Column means
        let col_means: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| self.data[i][j]).sum::<f64>() / n_f)
            .collect();

        // Grand mean
        let grand_mean: f64 = row_means.iter().sum::<f64>() / n_f;

        let mut data = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                data[i][j] = self.data[i][j] - row_means[i] - col_means[j] + grand_mean;
            }
        }
        KernelMatrix { data, n }
    }

    /// Matrix multiply: `(self * other)[i,j] = sum_k self[i,k] * other[k,j]`.
    ///
    /// Used internally; not exposed as a primary public API.
    #[allow(dead_code)]
    fn matmul(&self, other: &KernelMatrix) -> Result<KernelMatrix, AlignmentError> {
        if self.n != other.n {
            return Err(AlignmentError::DimensionMismatch {
                expected: self.n,
                got: other.n,
            });
        }
        let n = self.n;
        let mut data = vec![vec![0.0_f64; n]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for k in 0..n {
                let aik = self.data[i][k];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    data[i][j] += aik * other.data[k][j];
                }
            }
        }
        Ok(KernelMatrix { data, n })
    }

    /// Compute `trace(self * other)` efficiently in O(n^2) without full matmul.
    ///
    /// `trace(A * B) = sum_{i,j} A[i,j] * B[j,i]`
    #[allow(dead_code)]
    fn trace_product(&self, other: &KernelMatrix) -> Result<f64, AlignmentError> {
        if self.n != other.n {
            return Err(AlignmentError::DimensionMismatch {
                expected: self.n,
                got: other.n,
            });
        }
        let n = self.n;
        let mut tr = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                tr += self.data[i][j] * other.data[j][i];
            }
        }
        Ok(tr)
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a pairwise kernel alignment computation.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The alignment score, normalised to `[-1, 1]`.
    pub score: f64,
    /// The raw Frobenius inner product `<K1, K2>_F` (or `<K1_c, K2_c>_F`).
    pub numerator: f64,
    /// `sqrt(||K1||_F^2 * ||K2||_F^2)`.
    pub denominator: f64,
    /// Number of samples `n`.
    pub n_samples: usize,
}

/// Comprehensive alignment statistics between a kernel `K` and a target kernel.
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    /// Kernel Target Alignment (uncentered).
    pub kta: f64,
    /// Centered Kernel Alignment.
    pub cka: f64,
    /// Biased HSIC estimate `(1/n^2) * <K_c, T_c>_F`.
    pub hsic: f64,
    /// Number of samples.
    pub n_samples: usize,
}

// ---------------------------------------------------------------------------
// Core alignment functions
// ---------------------------------------------------------------------------

/// Compute the **Kernel Target Alignment (KTA)** between kernel `k` and target
/// kernel `target`.
///
/// ```text
/// KTA(K, T) = <K, T>_F / (||K||_F * ||T||_F)
/// ```
///
/// The score lies in `[-1, 1]`; values near `1` indicate the kernel faithfully
/// encodes the label structure.
pub fn kernel_target_alignment(
    k: &KernelMatrix,
    target: &KernelMatrix,
) -> Result<AlignmentResult, AlignmentError> {
    if k.n() != target.n() {
        return Err(AlignmentError::DimensionMismatch {
            expected: k.n(),
            got: target.n(),
        });
    }

    let numerator = k.frobenius_inner(target)?;
    let norm_k_sq = k.frobenius_norm_sq();
    let norm_t_sq = target.frobenius_norm_sq();
    let denominator = (norm_k_sq * norm_t_sq).sqrt();

    if denominator < f64::EPSILON {
        return Err(AlignmentError::NumericalError(
            "One or both kernel matrices have zero Frobenius norm".to_string(),
        ));
    }

    Ok(AlignmentResult {
        score: numerator / denominator,
        numerator,
        denominator,
        n_samples: k.n(),
    })
}

/// Compute the **Centered Kernel Alignment (CKA)** between kernel matrices
/// `k1` and `k2`.
///
/// CKA applies double-centering (via the centering matrix `H`) before alignment,
/// making it invariant to isotropic scaling and mean shifts:
///
/// ```text
/// CKA(K1, K2) = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
/// ```
pub fn centered_kernel_alignment(
    k1: &KernelMatrix,
    k2: &KernelMatrix,
) -> Result<AlignmentResult, AlignmentError> {
    if k1.n() != k2.n() {
        return Err(AlignmentError::DimensionMismatch {
            expected: k1.n(),
            got: k2.n(),
        });
    }

    let k1_c = k1.center();
    let k2_c = k2.center();

    let n_sq = (k1.n() * k1.n()) as f64;

    let hsic_12 = k1_c.frobenius_inner(&k2_c)? / n_sq;
    let hsic_11 = k1_c.frobenius_norm_sq() / n_sq;
    let hsic_22 = k2_c.frobenius_norm_sq() / n_sq;

    let denominator_sq = hsic_11 * hsic_22;
    if denominator_sq < f64::EPSILON * f64::EPSILON {
        return Err(AlignmentError::NumericalError(
            "HSIC self-alignment is zero; cannot normalise CKA".to_string(),
        ));
    }

    let denominator = denominator_sq.sqrt();
    let score = hsic_12 / denominator;

    Ok(AlignmentResult {
        score,
        numerator: hsic_12,
        denominator,
        n_samples: k1.n(),
    })
}

/// Compute the **biased HSIC** (Hilbert-Schmidt Independence Criterion) estimate:
///
/// ```text
/// HSIC(K, L) = (1/n^2) * trace(K * H * L * H)
///            = (1/n^2) * <K_c, L_c>_F
/// ```
///
/// where `K_c = H*K*H` and `L_c = H*L*H` are the doubly-centred versions.
pub fn hsic(k: &KernelMatrix, l: &KernelMatrix) -> Result<f64, AlignmentError> {
    if k.n() != l.n() {
        return Err(AlignmentError::DimensionMismatch {
            expected: k.n(),
            got: l.n(),
        });
    }
    let n_sq = (k.n() * k.n()) as f64;
    let k_c = k.center();
    let l_c = l.center();
    let inner = k_c.frobenius_inner(&l_c)?;
    Ok(inner / n_sq)
}

/// Compute **all alignment metrics** in a single pass.
///
/// Returns [`AlignmentStats`] containing KTA, CKA, and HSIC values.
pub fn alignment_stats(
    k: &KernelMatrix,
    target: &KernelMatrix,
) -> Result<AlignmentStats, AlignmentError> {
    if k.n() != target.n() {
        return Err(AlignmentError::DimensionMismatch {
            expected: k.n(),
            got: target.n(),
        });
    }

    let kta_result = kernel_target_alignment(k, target)?;
    let cka_result = centered_kernel_alignment(k, target)?;
    let hsic_val = hsic(k, target)?;

    Ok(AlignmentStats {
        kta: kta_result.score,
        cka: cka_result.score,
        hsic: hsic_val,
        n_samples: k.n(),
    })
}

// ---------------------------------------------------------------------------
// Optimization types and routines
// ---------------------------------------------------------------------------

/// Configuration for alignment-based kernel hyperparameter search.
#[derive(Debug, Clone)]
pub struct AlignmentOptConfig {
    /// Maximum number of iterations (default: 50).
    pub max_iterations: usize,
    /// Step size for gradient ascent (default: 0.01).
    pub learning_rate: f64,
    /// Convergence threshold: stop when `|Δscore| < tolerance` (default: 1e-6).
    pub tolerance: f64,
    /// If `true`, use CKA; otherwise use KTA (default: `true`).
    pub use_cka: bool,
    /// Finite-difference step for gradient estimation (default: 1e-5).
    pub fd_step: f64,
}

impl Default for AlignmentOptConfig {
    fn default() -> Self {
        AlignmentOptConfig {
            max_iterations: 50,
            learning_rate: 0.01,
            tolerance: 1e-6,
            use_cka: true,
            fd_step: 1e-5,
        }
    }
}

/// Outcome of a kernel alignment optimisation run.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best alignment score found.
    pub best_score: f64,
    /// Kernel hyperparameters that achieved `best_score`.
    pub best_params: Vec<f64>,
    /// Alignment score recorded after each iteration / grid point.
    pub score_history: Vec<f64>,
    /// Whether the optimiser converged before `max_iterations`.
    pub converged: bool,
    /// Total number of iterations (or grid points evaluated).
    pub iterations: usize,
}

/// Evaluate the alignment score for a given parameter vector.
fn evaluate_alignment(
    kernel_fn: &dyn Fn(&[f64]) -> KernelMatrix,
    target: &KernelMatrix,
    params: &[f64],
    use_cka: bool,
) -> Result<f64, AlignmentError> {
    let k = kernel_fn(params);
    if use_cka {
        centered_kernel_alignment(&k, target).map(|r| r.score)
    } else {
        kernel_target_alignment(&k, target).map(|r| r.score)
    }
}

/// **Grid search** over a discrete set of kernel parameter vectors, returning
/// the one that maximises alignment with `target`.
///
/// # Arguments
///
/// * `kernel_fn` - A closure that maps a parameter vector to a [`KernelMatrix`].
/// * `target`    - The target kernel (e.g. built from labels via
///   [`KernelMatrix::from_labels`]).
/// * `params_grid` - The set of parameter vectors to evaluate.
/// * `config`    - Search configuration (determines CKA vs KTA).
///
/// # Returns
///
/// An [`OptimizationResult`] with `best_params` set to the grid vector
/// achieving the highest alignment.
pub fn grid_search_alignment(
    kernel_fn: &dyn Fn(&[f64]) -> KernelMatrix,
    target: &KernelMatrix,
    params_grid: &[Vec<f64>],
    config: &AlignmentOptConfig,
) -> Result<OptimizationResult, AlignmentError> {
    if params_grid.is_empty() {
        return Err(AlignmentError::NumericalError(
            "params_grid must not be empty".to_string(),
        ));
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_params = params_grid[0].clone();
    let mut score_history = Vec::with_capacity(params_grid.len());

    for params in params_grid {
        let score = evaluate_alignment(kernel_fn, target, params, config.use_cka)?;
        score_history.push(score);
        if score > best_score {
            best_score = score;
            best_params = params.clone();
        }
    }

    Ok(OptimizationResult {
        best_score,
        best_params,
        score_history,
        converged: true,
        iterations: params_grid.len(),
    })
}

/// **Gradient ascent** on the alignment score via finite differences.
///
/// For each parameter `θ_k`, the partial derivative is approximated as:
///
/// ```text
/// ∂A/∂θ_k ≈ (A(θ + ε*e_k) - A(θ - ε*e_k)) / (2ε)
/// ```
///
/// Parameters are updated as `θ ← θ + η * ∇A(θ)` until convergence or
/// `max_iterations` is reached.
///
/// # Arguments
///
/// * `kernel_fn`      - A closure mapping parameters to a [`KernelMatrix`].
/// * `target`         - Target kernel.
/// * `initial_params` - Starting parameter vector.
/// * `config`         - Optimisation configuration.
pub fn gradient_ascent_alignment(
    kernel_fn: &dyn Fn(&[f64]) -> KernelMatrix,
    target: &KernelMatrix,
    initial_params: &[f64],
    config: &AlignmentOptConfig,
) -> Result<OptimizationResult, AlignmentError> {
    if initial_params.is_empty() {
        return Err(AlignmentError::NumericalError(
            "initial_params must not be empty".to_string(),
        ));
    }

    let d = initial_params.len();
    let mut params = initial_params.to_vec();
    let mut score_history = Vec::with_capacity(config.max_iterations);
    let mut converged = false;

    let mut current_score = evaluate_alignment(kernel_fn, target, &params, config.use_cka)?;
    score_history.push(current_score);

    for _iter in 0..config.max_iterations {
        // Compute finite-difference gradient
        let mut grad = vec![0.0_f64; d];
        for k in 0..d {
            let mut params_fwd = params.clone();
            let mut params_bwd = params.clone();
            params_fwd[k] += config.fd_step;
            params_bwd[k] -= config.fd_step;

            let score_fwd = evaluate_alignment(kernel_fn, target, &params_fwd, config.use_cka)?;
            let score_bwd = evaluate_alignment(kernel_fn, target, &params_bwd, config.use_cka)?;
            grad[k] = (score_fwd - score_bwd) / (2.0 * config.fd_step);
        }

        // Gradient ascent step
        for k in 0..d {
            params[k] += config.learning_rate * grad[k];
        }

        let new_score = evaluate_alignment(kernel_fn, target, &params, config.use_cka)?;
        score_history.push(new_score);

        if (new_score - current_score).abs() < config.tolerance {
            converged = true;
            current_score = new_score;
            break;
        }
        current_score = new_score;
    }

    let iterations = score_history.len();
    Ok(OptimizationResult {
        best_score: current_score,
        best_params: params,
        score_history,
        converged,
        iterations,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a small RBF kernel matrix for a 1-D dataset
    fn rbf_kernel_matrix(data: &[f64], gamma: f64) -> KernelMatrix {
        let n = data.len();
        let mut mat = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let diff = data[i] - data[j];
                mat[i][j] = (-gamma * diff * diff).exp();
            }
        }
        KernelMatrix::new(mat).expect("valid kernel matrix")
    }

    // ---------------------------------------------------------------------------
    // KernelMatrix structural tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_identity_trace_equals_n() {
        for n in [1_usize, 3, 5, 10] {
            let id = KernelMatrix::identity(n);
            let tr = id.trace();
            assert!(
                (tr - n as f64).abs() < 1e-12,
                "identity trace should be {n}, got {tr}"
            );
        }
    }

    #[test]
    fn test_from_labels_correct_values() {
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let k = KernelMatrix::from_labels(&labels);
        assert_eq!(k.n(), 4);
        // Same-class pairs
        assert!((k.get(0, 1) - 1.0).abs() < 1e-12);
        assert!((k.get(2, 3) - 1.0).abs() < 1e-12);
        // Diagonal
        assert!((k.get(0, 0) - 1.0).abs() < 1e-12);
        // Cross-class pairs
        assert!((k.get(0, 2) + 1.0).abs() < 1e-12);
        assert!((k.get(1, 3) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_center_zero_row_column_sums() {
        // Use a non-trivial positive semidefinite matrix
        let data = vec![
            vec![4.0, 2.0, 1.0],
            vec![2.0, 3.0, 0.5],
            vec![1.0, 0.5, 2.0],
        ];
        let k = KernelMatrix::new(data).expect("valid");
        let k_c = k.center();
        let n = k_c.n();

        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| k_c.get(i, j)).sum();
            assert!(row_sum.abs() < 1e-10, "centered row {i} sum = {row_sum}");
            let col_sum: f64 = (0..n).map(|j| k_c.get(j, i)).sum();
            assert!(col_sum.abs() < 1e-10, "centered col {i} sum = {col_sum}");
        }
    }

    #[test]
    fn test_frobenius_inner_symmetric() {
        let data1 = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let data2 = vec![vec![1.0, 0.5], vec![0.5, 2.0]];
        let k1 = KernelMatrix::new(data1).expect("valid");
        let k2 = KernelMatrix::new(data2).expect("valid");

        let inner_12 = k1.frobenius_inner(&k2).expect("ok");
        let inner_21 = k2.frobenius_inner(&k1).expect("ok");
        assert!(
            (inner_12 - inner_21).abs() < 1e-12,
            "<K1,K2> = {inner_12}, <K2,K1> = {inner_21}"
        );
    }

    #[test]
    fn test_frobenius_norm_identity() {
        for n in [1_usize, 4, 9] {
            let id = KernelMatrix::identity(n);
            let norm_sq = id.frobenius_norm_sq();
            let norm = norm_sq.sqrt();
            let expected = (n as f64).sqrt();
            assert!(
                (norm - expected).abs() < 1e-12,
                "||I_n||_F should be sqrt({n}) = {expected}, got {norm}"
            );
        }
    }

    #[test]
    fn test_from_flat_validates_square() {
        // 2×2 from flat works
        let flat = vec![1.0, 0.0, 0.0, 1.0];
        assert!(KernelMatrix::from_flat(&flat, 2).is_ok());

        // 5 elements cannot form a square matrix
        let bad = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(matches!(
            KernelMatrix::from_flat(&bad, 2),
            Err(AlignmentError::NonSquareMatrix)
        ));
    }

    // ---------------------------------------------------------------------------
    // KTA tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_kta_identical_kernels_is_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = rbf_kernel_matrix(&data, 0.5);
        let result = kernel_target_alignment(&k, &k).expect("ok");
        assert!(
            (result.score - 1.0).abs() < 1e-10,
            "KTA of K with itself should be 1.0, got {}",
            result.score
        );
    }

    #[test]
    fn test_kta_with_label_target_positive() {
        // Two well-separated clusters → RBF with large gamma → high KTA
        let data = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = rbf_kernel_matrix(&data, 1.0);
        let target = KernelMatrix::from_labels(&labels);
        let result = kernel_target_alignment(&k, &target).expect("ok");
        assert!(
            result.score > 0.0,
            "KTA should be positive for clustered data, got {}",
            result.score
        );
    }

    #[test]
    fn test_kta_range_is_minus_one_to_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![0.0, 1.0, 0.0, 1.0];
        let k = rbf_kernel_matrix(&data, 1.0);
        let target = KernelMatrix::from_labels(&labels);
        let result = kernel_target_alignment(&k, &target).expect("ok");
        assert!(
            result.score >= -1.0 - 1e-9 && result.score <= 1.0 + 1e-9,
            "KTA score out of range: {}",
            result.score
        );
    }

    // ---------------------------------------------------------------------------
    // CKA tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_cka_identical_kernels_is_one() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = rbf_kernel_matrix(&data, 0.5);
        let result = centered_kernel_alignment(&k, &k).expect("ok");
        assert!(
            (result.score - 1.0).abs() < 1e-10,
            "CKA of K with itself should be 1.0, got {}",
            result.score
        );
    }

    #[test]
    fn test_cka_invariant_to_scaling() {
        let data = vec![0.5, 1.0, 2.0, 3.0, 4.0];
        let k = rbf_kernel_matrix(&data, 0.3);
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let target = KernelMatrix::from_labels(&labels);

        // Build 2*K
        let n = k.n();
        let scaled_data: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| 2.0 * k.get(i, j)).collect())
            .collect();
        let k_scaled = KernelMatrix::new(scaled_data).expect("valid");

        let cka_original = centered_kernel_alignment(&k, &target).expect("ok").score;
        let cka_scaled = centered_kernel_alignment(&k_scaled, &target)
            .expect("ok")
            .score;

        assert!(
            (cka_original - cka_scaled).abs() < 1e-10,
            "CKA should be invariant to scaling: {cka_original} vs {cka_scaled}"
        );
    }

    #[test]
    fn test_cka_invariant_to_mean_shift() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = rbf_kernel_matrix(&data, 0.2);
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let target = KernelMatrix::from_labels(&labels);

        // Shift K by a constant c → K' = K + c*11^T
        let n = k.n();
        let c = 3.0_f64;
        let shifted_data: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| k.get(i, j) + c).collect())
            .collect();
        let k_shifted = KernelMatrix::new(shifted_data).expect("valid");

        let cka_original = centered_kernel_alignment(&k, &target).expect("ok").score;
        let cka_shifted = centered_kernel_alignment(&k_shifted, &target)
            .expect("ok")
            .score;

        assert!(
            (cka_original - cka_shifted).abs() < 1e-9,
            "CKA should be invariant to constant mean shift: {cka_original} vs {cka_shifted}"
        );
    }

    // ---------------------------------------------------------------------------
    // HSIC tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hsic_identical_kernel_positive() {
        let data = vec![1.0, 3.0, 5.0, 7.0];
        let k = rbf_kernel_matrix(&data, 1.0);
        let val = hsic(&k, &k).expect("ok");
        assert!(val > 0.0, "HSIC(K,K) should be positive, got {val}");
    }

    #[test]
    fn test_hsic_near_independent_kernels() {
        // An identity kernel encodes no inter-sample similarity; pairing with
        // a label kernel that has off-diagonal structure yields a small HSIC.
        let n = 8;
        let identity = KernelMatrix::identity(n);

        // Constant kernel (all ones) is trivially uninformative after centering
        let data = vec![vec![1.0_f64; n]; n];
        let constant_k = KernelMatrix::new(data).expect("valid");

        let val = hsic(&identity, &constant_k).expect("ok");
        // After centering a constant matrix becomes all zeros → HSIC = 0
        assert!(
            val.abs() < 1e-12,
            "HSIC(I, 1*1^T) after centering should be ~0, got {val}"
        );
    }

    // ---------------------------------------------------------------------------
    // AlignmentStats test
    // ---------------------------------------------------------------------------

    #[test]
    fn test_alignment_stats_reports_all_metrics() {
        let data = vec![0.0, 0.5, 1.0, 5.0, 5.5, 6.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = rbf_kernel_matrix(&data, 2.0);
        let target = KernelMatrix::from_labels(&labels);

        let stats = alignment_stats(&k, &target).expect("ok");
        assert_eq!(stats.n_samples, 6);
        // KTA, CKA should be in [-1,1]
        assert!(stats.kta >= -1.0 - 1e-9 && stats.kta <= 1.0 + 1e-9);
        assert!(stats.cka >= -1.0 - 1e-9 && stats.cka <= 1.0 + 1e-9);
    }

    #[test]
    fn test_alignment_stats_perfect_alignment_near_one() {
        // Identical kernels should give KTA = CKA = 1.0
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let k = rbf_kernel_matrix(&data, 0.5);
        let stats = alignment_stats(&k, &k).expect("ok");
        assert!(
            (stats.kta - 1.0).abs() < 1e-10,
            "KTA should be 1.0, got {}",
            stats.kta
        );
        assert!(
            (stats.cka - 1.0).abs() < 1e-10,
            "CKA should be 1.0, got {}",
            stats.cka
        );
    }

    // ---------------------------------------------------------------------------
    // Optimisation tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_grid_search_finds_best_params() {
        let data = vec![0.0, 0.2, 0.4, 5.0, 5.2, 5.4];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let target = KernelMatrix::from_labels(&labels);

        // Grid of gamma values: larger gamma → tighter clusters → higher alignment
        let params_grid: Vec<Vec<f64>> =
            vec![vec![0.01], vec![0.1], vec![1.0], vec![5.0], vec![10.0]];

        let config = AlignmentOptConfig {
            use_cka: true,
            ..Default::default()
        };

        let kernel_fn = |params: &[f64]| rbf_kernel_matrix(&data, params[0]);

        let result = grid_search_alignment(&kernel_fn, &target, &params_grid, &config).expect("ok");

        assert_eq!(result.iterations, 5);
        assert_eq!(result.score_history.len(), 5);
        assert!(result.converged);

        // Verify best_score is actually the maximum in history
        let max_in_history = result
            .score_history
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (result.best_score - max_in_history).abs() < 1e-12,
            "best_score {} should equal max in history {}",
            result.best_score,
            max_in_history
        );
    }

    #[test]
    fn test_gradient_ascent_converges_toward_higher_alignment() {
        let data = vec![0.0, 0.3, 0.6, 4.0, 4.3, 4.6];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let target = KernelMatrix::from_labels(&labels);

        let kernel_fn = |params: &[f64]| rbf_kernel_matrix(&data, params[0].abs());

        let initial_params = vec![0.01_f64];
        let config = AlignmentOptConfig {
            max_iterations: 30,
            learning_rate: 0.05,
            tolerance: 1e-8,
            use_cka: true,
            fd_step: 1e-4,
        };

        let result =
            gradient_ascent_alignment(&kernel_fn, &target, &initial_params, &config).expect("ok");

        assert!(
            !result.score_history.is_empty(),
            "score_history must be non-empty"
        );
        // The final score should be >= the initial score
        let first_score = result.score_history[0];
        assert!(
            result.best_score >= first_score - 1e-6,
            "gradient ascent should not decrease alignment: final {} < initial {}",
            result.best_score,
            first_score
        );
    }

    #[test]
    fn test_score_history_non_decreasing_approximately() {
        // We run gradient ascent on a simple 1-parameter RBF and check that the
        // alignment trend is upward (allowing small oscillations due to FD noise).
        let data = vec![0.0, 0.5, 1.0, 6.0, 6.5, 7.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let target = KernelMatrix::from_labels(&labels);

        let kernel_fn = |params: &[f64]| rbf_kernel_matrix(&data, params[0].abs() + 1e-3);

        let config = AlignmentOptConfig {
            max_iterations: 20,
            learning_rate: 0.02,
            tolerance: 1e-9,
            use_cka: true,
            fd_step: 1e-4,
        };

        let result = gradient_ascent_alignment(&kernel_fn, &target, &[0.01], &config).expect("ok");

        // The final best score should not be catastrophically worse than the midpoint
        let n = result.score_history.len();
        if n >= 2 {
            let final_score = result.score_history[n - 1];
            let initial_score = result.score_history[0];
            // Allow a 5% relative tolerance (gradient ascent may oscillate slightly)
            assert!(
                final_score >= initial_score - 0.05 * initial_score.abs().max(1e-3),
                "score history should trend upward: initial={initial_score}, final={final_score}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Error handling tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_kta_dimension_mismatch_error() {
        let k1 = KernelMatrix::identity(3);
        let k2 = KernelMatrix::identity(4);
        let result = kernel_target_alignment(&k1, &k2);
        assert!(matches!(
            result,
            Err(AlignmentError::DimensionMismatch {
                expected: 3,
                got: 4
            })
        ));
    }

    #[test]
    fn test_cka_dimension_mismatch_error() {
        let k1 = KernelMatrix::identity(2);
        let k2 = KernelMatrix::identity(5);
        let result = centered_kernel_alignment(&k1, &k2);
        assert!(matches!(
            result,
            Err(AlignmentError::DimensionMismatch {
                expected: 2,
                got: 5
            })
        ));
    }

    #[test]
    fn test_hsic_dimension_mismatch_error() {
        let k1 = KernelMatrix::identity(3);
        let k2 = KernelMatrix::identity(6);
        let result = hsic(&k1, &k2);
        assert!(matches!(
            result,
            Err(AlignmentError::DimensionMismatch {
                expected: 3,
                got: 6
            })
        ));
    }

    #[test]
    fn test_alignment_stats_dimension_mismatch() {
        let k = KernelMatrix::identity(3);
        let target = KernelMatrix::identity(4);
        let result = alignment_stats(&k, &target);
        assert!(matches!(
            result,
            Err(AlignmentError::DimensionMismatch { .. })
        ));
    }

    // ---------------------------------------------------------------------------
    // Matrix operation correctness
    // ---------------------------------------------------------------------------

    #[test]
    fn test_matmul_identity_neutral() {
        let n = 4;
        let id = KernelMatrix::identity(n);
        let k = rbf_kernel_matrix(&[1.0, 2.0, 3.0, 4.0], 0.5);
        let product = k.matmul(&id).expect("ok");
        for i in 0..n {
            for j in 0..n {
                let diff = (product.get(i, j) - k.get(i, j)).abs();
                assert!(diff < 1e-12, "K*I should equal K at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_trace_product_vs_matmul_trace() {
        let k = rbf_kernel_matrix(&[0.0, 1.0, 2.0, 3.0], 0.4);
        let l = rbf_kernel_matrix(&[0.0, 1.0, 2.0, 3.0], 0.8);
        let via_trace_product = k.trace_product(&l).expect("ok");
        let via_matmul = k.matmul(&l).expect("ok").trace();
        assert!(
            (via_trace_product - via_matmul).abs() < 1e-10,
            "trace(K*L) via trace_product ({via_trace_product}) vs matmul ({via_matmul})"
        );
    }
}
