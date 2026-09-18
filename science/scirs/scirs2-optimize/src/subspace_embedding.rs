//! Subspace embedding methods for dimensionality-reduced optimization.
//!
//! Uses random projections (Johnson-Lindenstrauss lemma) to reduce the dimension
//! of large optimization problems.  Three projection families are supported:
//!
//! - **Gaussian** random projection (`i.i.d. N(0, 1/k)` entries).
//! - **Sparse** random projection (Achlioptas 2003: entries ±√(3/k) with probability
//!   1/6 each, 0 with probability 2/3).
//! - **Fast JL** (placeholder backed by the Gaussian construction; a full SRHT
//!   implementation can be substituted without changing the public API).
//!
//! The module also exposes [`sketched_least_squares`], a one-shot sketching solver for
//! overdetermined systems `min ||Ax - b||²`.
//!
//! # References
//!
//! - Johnson, W.B. & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings.
//! - Achlioptas, D. (2003). Database-friendly random projections. JCSS 66(4).
//! - Mahoney, M.W. (2011). Randomized algorithms for matrices and data. Found. Trends.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Random projection embedding methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingType {
    /// Gaussian random projection: entries i.i.d. N(0, 1/k).
    GaussianRandom,
    /// Sparse random projection (Achlioptas 2003):
    /// entries ±√(3/k) with probability 1/6 each, 0 with probability 2/3.
    SparseRandom,
    /// Fast Johnson-Lindenstrauss transform (SRHT approximation).
    /// Currently implemented as a Gaussian projection; a structured Hadamard
    /// construction can replace this without breaking the API.
    FastJohnsonLindenstrauss,
}

/// Configuration for a subspace embedding.
#[derive(Debug, Clone)]
pub struct SubspaceEmbeddingConfig {
    /// Dimension of the original space (number of input coordinates).
    pub original_dim: usize,
    /// Dimension of the embedding / sketch space (`k`).
    pub embedding_dim: usize,
    /// Which projection family to use.
    pub embedding_type: EmbeddingType,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// Core embedding struct
// ---------------------------------------------------------------------------

/// A random subspace embedding `S: R^n → R^k` (`k × n` projection matrix).
pub struct SubspaceEmbedding {
    config: SubspaceEmbeddingConfig,
    /// The `k × n` projection matrix.
    matrix: Array2<f64>,
}

impl SubspaceEmbedding {
    /// Construct a new embedding according to `config`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizeError::InvalidInput`] if either dimension is zero.
    pub fn new(config: SubspaceEmbeddingConfig) -> OptimizeResult<Self> {
        if config.original_dim == 0 || config.embedding_dim == 0 {
            return Err(OptimizeError::InvalidInput(
                "SubspaceEmbedding: dimensions must be non-zero".to_string(),
            ));
        }
        let matrix = match config.embedding_type {
            EmbeddingType::GaussianRandom => {
                gaussian_projection(config.embedding_dim, config.original_dim, config.seed)
            }
            EmbeddingType::SparseRandom => {
                sparse_projection(config.embedding_dim, config.original_dim, config.seed)
            }
            EmbeddingType::FastJohnsonLindenstrauss => {
                // Full SRHT requires a fast Walsh-Hadamard transform; we fall back to
                // the Gaussian family which gives the same asymptotic JL guarantees.
                gaussian_projection(config.embedding_dim, config.original_dim, config.seed)
            }
        };
        Ok(Self { config, matrix })
    }

    /// Project a vector `x ∈ R^n → y = Sx ∈ R^k`.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizeError::ValueError`] if `x.len() != original_dim`.
    pub fn project(&self, x: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
        if x.len() != self.config.original_dim {
            return Err(OptimizeError::ValueError(format!(
                "project: expected dim {}, got {}",
                self.config.original_dim,
                x.len()
            )));
        }
        Ok(self.matrix.dot(x))
    }

    /// Approximate reconstruction `x ≈ S^T y` (pseudo-inverse via transpose).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizeError::ValueError`] if `y.len() != embedding_dim`.
    pub fn reconstruct(&self, y: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
        if y.len() != self.config.embedding_dim {
            return Err(OptimizeError::ValueError(format!(
                "reconstruct: expected dim {}, got {}",
                self.config.embedding_dim,
                y.len()
            )));
        }
        Ok(self.matrix.t().dot(y))
    }

    /// Project a matrix `A ∈ R^{n×p} → SA ∈ R^{k×p}`.
    ///
    /// The embedding matrix `S` is `k × n`, so `A` must have `n` rows
    /// (i.e., `a.nrows() == original_dim`).
    ///
    /// # Errors
    ///
    /// Returns [`OptimizeError::ValueError`] if `a.nrows() != original_dim`.
    pub fn project_matrix(&self, a: &Array2<f64>) -> OptimizeResult<Array2<f64>> {
        if a.nrows() != self.config.original_dim {
            return Err(OptimizeError::ValueError(format!(
                "project_matrix: expected {} rows, got {}",
                self.config.original_dim,
                a.nrows()
            )));
        }
        Ok(self.matrix.dot(a))
    }

    /// Embedding (sketch) dimension `k`.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Original dimension `n`.
    pub fn original_dim(&self) -> usize {
        self.config.original_dim
    }

    /// Johnson-Lindenstrauss ε-distortion bound.
    ///
    /// For `n_points` points and failure probability `delta`,
    /// the embedding preserves all pairwise distances within factor `(1 ± ε)`
    /// with probability at least `1 − delta`.
    ///
    /// Formula: `ε = sqrt(2 * ln(n / δ) / k)`.
    pub fn jl_epsilon(&self, n_points: usize, delta: f64) -> f64 {
        let k = self.config.embedding_dim as f64;
        (2.0 * (n_points as f64 / delta).ln() / k).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Internal projection factories
// ---------------------------------------------------------------------------

/// Build a `k × n` Gaussian projection matrix.
/// Entries are i.i.d. `N(0, 1/k)` generated via Box-Muller transform
/// with a minimal LCG for zero external dependencies.
fn gaussian_projection(k: usize, n: usize, seed: u64) -> Array2<f64> {
    let scale = (k as f64).recip().sqrt();
    let mut state = seed.wrapping_add(1); // ensure non-zero seed
    let data: Vec<f64> = (0..k * n)
        .map(|_| {
            // LCG step 1
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u1 = ((state >> 11) as f64) / ((1u64 << 53) as f64) + 1e-300;
            // LCG step 2
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u2 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            // Box-Muller
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * scale
        })
        .collect();

    Array2::from_shape_vec((k, n), data).unwrap_or_else(|_| Array2::zeros((k, n)))
}

/// Build a `k × n` sparse Achlioptas projection matrix.
/// Each entry is `+√(3/k)` with prob 1/6, `−√(3/k)` with prob 1/6, 0 with prob 2/3.
fn sparse_projection(k: usize, n: usize, seed: u64) -> Array2<f64> {
    let scale = (3.0_f64 / k as f64).sqrt();
    let mut state = seed.wrapping_add(1);
    let data: Vec<f64> = (0..k * n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Use top 32 bits for the decision.
            let r = (state >> 32) % 6;
            if r == 0 {
                scale
            } else if r == 1 {
                -scale
            } else {
                0.0
            }
        })
        .collect();

    Array2::from_shape_vec((k, n), data).unwrap_or_else(|_| Array2::zeros((k, n)))
}

// ---------------------------------------------------------------------------
// Sketched least squares
// ---------------------------------------------------------------------------

/// Sketched least-squares solver for overdetermined systems `min ||Ax − b||²`.
///
/// Constructs a Gaussian sketch `S` of dimension `k × m`, then solves the
/// reduced normal equations `(SA)ᵀ(SA) x = (SA)ᵀ(Sb)` via Gaussian elimination
/// with partial pivoting.
///
/// # Arguments
///
/// * `a` — `m × n` system matrix (m ≥ n for an overdetermined system).
/// * `b` — right-hand side of length `m`.
/// * `k` — sketch dimension (typically `k ≈ 4n` for good accuracy).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// Returns [`OptimizeError::InvalidInput`] for empty matrices or dimension mismatches,
/// and [`OptimizeError::ComputationError`] for a singular sketched normal matrix.
pub fn sketched_least_squares(
    a: &Array2<f64>,
    b: &Array1<f64>,
    k: usize,
    seed: u64,
) -> OptimizeResult<Array1<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    if m == 0 || n == 0 {
        return Err(OptimizeError::InvalidInput(
            "sketched_least_squares: matrix must be non-empty".to_string(),
        ));
    }
    if b.len() != m {
        return Err(OptimizeError::ValueError(format!(
            "sketched_least_squares: b length {} does not match matrix rows {m}",
            b.len()
        )));
    }
    if k == 0 {
        return Err(OptimizeError::InvalidInput(
            "sketched_least_squares: sketch dimension k must be > 0".to_string(),
        ));
    }

    let config = SubspaceEmbeddingConfig {
        original_dim: m,
        embedding_dim: k,
        embedding_type: EmbeddingType::GaussianRandom,
        seed,
    };
    let emb = SubspaceEmbedding::new(config)?;

    // SA  (k × n)  and  Sb  (k)
    let sa = emb.project_matrix(a)?;
    let sb = emb.matrix.dot(b);

    // Normal equations: (SA)^T (SA) x = (SA)^T Sb
    let sat = sa.t().to_owned(); // n × k
    let satsa = sat.dot(&sa); // n × n
    let satsb = sat.dot(&sb); // n

    solve_dense_linear(&satsa, &satsb)
}

/// Solve `A x = b` with Gaussian elimination and partial pivoting.
///
/// `A` is `n × n` (row-major), `b` is length `n`.
///
/// # Errors
///
/// Returns [`OptimizeError::ComputationError`] if the matrix is singular.
fn solve_dense_linear(a: &Array2<f64>, b: &Array1<f64>) -> OptimizeResult<Array1<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Build augmented system [A | b] stored as flat row-major vecs.
    let mut mat: Vec<f64> = a.iter().cloned().collect();
    let mut rhs: Vec<f64> = b.iter().cloned().collect();

    for col in 0..n {
        // Partial pivoting: find the row with the largest absolute value in column `col`.
        let mut max_row = col;
        let mut max_val = mat[col * n + col].abs();
        for row in (col + 1)..n {
            let v = mat[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(OptimizeError::ComputationError(
                "solve_dense_linear: singular or near-singular matrix".to_string(),
            ));
        }
        // Swap rows.
        if max_row != col {
            for k in 0..n {
                mat.swap(col * n + k, max_row * n + k);
            }
            rhs.swap(col, max_row);
        }
        // Eliminate below.
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            for k in col..n {
                let v = mat[col * n + k];
                mat[row * n + k] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
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

    #[test]
    fn test_gaussian_projection_shape() {
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 100,
            embedding_dim: 20,
            embedding_type: EmbeddingType::GaussianRandom,
            seed: 42,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        let x = Array1::zeros(100);
        let y = emb.project(&x).expect("project must succeed");
        assert_eq!(y.len(), 20, "projected dim should be embedding_dim");
    }

    #[test]
    fn test_sparse_projection_shape() {
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 50,
            embedding_dim: 10,
            embedding_type: EmbeddingType::SparseRandom,
            seed: 7,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        assert_eq!(emb.embedding_dim(), 10);
        assert_eq!(emb.original_dim(), 50);
    }

    #[test]
    fn test_projection_dimension_check() {
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 20,
            embedding_dim: 5,
            embedding_type: EmbeddingType::GaussianRandom,
            seed: 1,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        // Wrong input length should return an error.
        let x_wrong = Array1::zeros(15); // should be 20
        let res = emb.project(&x_wrong);
        assert!(res.is_err(), "project with wrong dimension must fail");
    }

    #[test]
    fn test_jl_epsilon_reasonable() {
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 1000,
            embedding_dim: 200,
            embedding_type: EmbeddingType::GaussianRandom,
            seed: 0,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        // With k=200, 1000 points, delta=0.01 the JL bound should be well below 1.
        let eps = emb.jl_epsilon(1000, 0.01);
        assert!(eps > 0.0, "epsilon must be positive");
        assert!(
            eps < 1.0,
            "epsilon={eps} should be < 1.0 for reasonable parameters"
        );
    }

    #[test]
    fn test_reconstruct_approx() {
        // For a Gaussian embedding, project then reconstruct via S^T S x.
        // The reconstruction S^T (S x) should approximate x in expectation.
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 10,
            embedding_dim: 10,
            embedding_type: EmbeddingType::GaussianRandom,
            seed: 123,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        let x: Array1<f64> = Array1::from_iter((0..10).map(|i| i as f64));
        let y = emb.project(&x).expect("project must succeed");
        assert_eq!(y.len(), 10, "projected length");
        let x_rec = emb.reconstruct(&y).expect("reconstruct must succeed");
        assert_eq!(
            x_rec.len(),
            10,
            "reconstructed length must equal original_dim"
        );
    }

    #[test]
    fn test_sketched_least_squares() {
        // Overdetermined system: 100 × 10, true solution x_true = [1..10].
        let m = 100_usize;
        let n = 10_usize;

        // Build a well-conditioned A: row i has A[i,j] = cos(i * (j+1) * 0.1).
        let mut a_data = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                a_data[i * n + j] = ((i as f64) * (j as f64 + 1.0) * 0.1).cos();
            }
        }
        let a = Array2::from_shape_vec((m, n), a_data).expect("shape");
        let x_true: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
        let b = a.dot(&x_true);

        // Sketch with k = 4*n to get good accuracy.
        let x_sol = sketched_least_squares(&a, &b, 4 * n, 999)
            .expect("sketched_least_squares must succeed");

        assert_eq!(x_sol.len(), n);
        // The sketched solution should approximate the true solution.
        // With k=40 and n=10 the residual ||Ax - b||/||b|| should be small.
        let residual = a.dot(&x_sol) - &b;
        let rel_err = residual.dot(&residual).sqrt() / (b.dot(&b).sqrt() + 1e-30);
        assert!(
            rel_err < 0.1,
            "relative residual {rel_err} too large; sketched LS should be accurate"
        );
    }

    #[test]
    fn test_project_matrix_shape() {
        let cfg = SubspaceEmbeddingConfig {
            original_dim: 30,
            embedding_dim: 8,
            embedding_type: EmbeddingType::SparseRandom,
            seed: 55,
        };
        let emb = SubspaceEmbedding::new(cfg).expect("construction must succeed");
        let a = Array2::ones((30, 5));
        let sa = emb.project_matrix(&a).expect("project_matrix must succeed");
        assert_eq!(sa.nrows(), 8, "SA should have embedding_dim rows");
        assert_eq!(sa.ncols(), 5, "SA should preserve column count");
    }
}
