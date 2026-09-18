//! Truncated SVD via power iteration, implemented from scratch.
//!
//! This module implements a truncated (rank-k) SVD using the power iteration /
//! deflation strategy:
//!
//! 1. Start with a random right singular-vector candidate `v`.
//! 2. Alternately apply `u = A v` (normalize) and `v = Aᵀ u` (normalize).
//! 3. Compute `σ = uᵀ A v`.
//! 4. Deflate: `A ← A − σ u vᵀ`.
//! 5. Repeat for the next triplet.
//!
//! Complexity: O(rank · m · n) — appropriate for truncated SVD of large matrices.

use super::config::LowRankConfig;
use super::error::LowRankError;

// ---------------------------------------------------------------------------
// SvdResult
// ---------------------------------------------------------------------------

/// Result of a truncated SVD decomposition.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors, shape: \[rows × rank\]  (row-major flat vec)
    pub u: Vec<f64>,
    pub u_rows: usize,
    pub u_cols: usize, // = rank
    /// Singular values, length = rank
    pub singular_values: Vec<f64>,
    /// Right singular vectors (transposed), shape: \[rank × cols\]
    pub vt: Vec<f64>,
    pub vt_rows: usize, // = rank
    pub vt_cols: usize,
    /// Frobenius error of approximation (‖A − U Σ Vᵀ‖_F)
    pub frobenius_error: f64,
    /// Actual rank used (may be < requested if matrix has lower rank)
    pub rank_used: usize,
}

impl SvdResult {
    /// Reconstruct the approximated matrix as a flat row-major `Vec<f64>`.
    ///
    /// Returns a matrix of shape \[u_rows × vt_cols\].
    pub fn reconstruct(&self) -> Vec<f64> {
        let rows = self.u_rows;
        let cols = self.vt_cols;
        let rank = self.rank_used;
        let mut out = vec![0.0_f64; rows * cols];

        for k in 0..rank {
            let sigma = self.singular_values[k];
            for i in 0..rows {
                let u_ik = self.u[i * self.u_cols + k];
                for j in 0..cols {
                    let vt_kj = self.vt[k * self.vt_cols + j];
                    out[i * cols + j] += sigma * u_ik * vt_kj;
                }
            }
        }
        out
    }

    /// Compute the relative Frobenius error versus the original matrix.
    ///
    /// Returns `‖original − reconstructed‖_F / ‖original‖_F`.
    /// Returns `0.0` if the original norm is zero.
    pub fn relative_error(&self, original: &[f64]) -> f64 {
        let reconstructed = self.reconstruct();
        TruncatedSvd::relative_frobenius_error(original, &reconstructed)
    }

    /// Energy captured as the fraction of singular-value sum used.
    ///
    /// `energy = (Σ_used) / (Σ_all)`.  When all singular values are zero
    /// the function returns `1.0` (trivially exact).
    pub fn energy_fraction(&self) -> f64 {
        let total: f64 = self.singular_values.iter().sum();
        if total == 0.0 {
            return 1.0;
        }
        // We only stored the truncated singular values; energy fraction is
        // relative to *those* values only (since we do not have the full spectrum
        // from a truncated decomposition).
        let used: f64 = self.singular_values[..self.rank_used].iter().sum();
        used / total
    }
}

// ---------------------------------------------------------------------------
// TruncatedSvd
// ---------------------------------------------------------------------------

/// Performs truncated SVD via power iteration with deflation.
pub struct TruncatedSvd {
    config: LowRankConfig,
}

impl TruncatedSvd {
    /// Construct a new `TruncatedSvd` with the given configuration.
    pub fn new(config: LowRankConfig) -> Self {
        TruncatedSvd { config }
    }

    // ------------------------------------------------------------------
    // Public matrix utilities
    // ------------------------------------------------------------------

    /// Compute the Frobenius norm of a flat matrix.
    pub fn frobenius_norm(matrix: &[f64]) -> f64 {
        matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the relative Frobenius error between two matrices of equal size.
    ///
    /// Returns `‖original − reconstructed‖_F / ‖original‖_F`, or `0.0` when the
    /// original norm is zero.
    pub fn relative_frobenius_error(original: &[f64], reconstructed: &[f64]) -> f64 {
        let orig_norm = Self::frobenius_norm(original);
        if orig_norm == 0.0 {
            return 0.0;
        }
        let diff_norm_sq: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        diff_norm_sq.sqrt() / orig_norm
    }

    /// Error bound derived from the truncated singular values (Eckart–Young).
    ///
    /// `‖A − A_k‖_F = sqrt(Σ_{i>k} σ_i²)`.
    pub fn error_bound_from_singular_values(all_singular_values: &[f64], rank: usize) -> f64 {
        let tail: f64 = all_singular_values.iter().skip(rank).map(|s| s * s).sum();
        tail.sqrt()
    }

    // ------------------------------------------------------------------
    // Core decomposition
    // ------------------------------------------------------------------

    /// Perform truncated SVD on a row-major matrix of shape `[rows × cols]`.
    ///
    /// Returns an `SvdResult` whose `rank_used` may be smaller than
    /// `self.config.rank` when the matrix's numerical rank is lower.
    pub fn decompose(
        &self,
        matrix: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<SvdResult, LowRankError> {
        // --- Input validation ------------------------------------------------
        if matrix.len() != rows * cols {
            return Err(LowRankError::InvalidInput(format!(
                "matrix length {} does not match rows={} × cols={}",
                matrix.len(),
                rows,
                cols
            )));
        }
        if rows == 0 || cols == 0 {
            return Err(LowRankError::InvalidInput(
                "matrix must have non-zero rows and cols".to_string(),
            ));
        }

        let max_rank = rows.min(cols);
        let rank = self.config.rank.min(max_rank);
        if self.config.rank > max_rank {
            return Err(LowRankError::RankExceedsDimensions {
                rank: self.config.rank,
                rows,
                cols,
            });
        }

        // Working copy of the matrix for deflation
        let mut a = matrix.to_vec();

        let mut u_vecs: Vec<Vec<f64>> = Vec::with_capacity(rank);
        let mut sigmas: Vec<f64> = Vec::with_capacity(rank);
        let mut v_vecs: Vec<Vec<f64>> = Vec::with_capacity(rank);
        let mut rank_used = 0;

        let a_norm = Self::frobenius_norm(matrix);

        for k in 0..rank {
            // ---------------------------------------------------------------
            // Initialise right singular vector with a deterministic seed
            // that varies per component k (so that components do not collapse
            // to the same direction).
            // ---------------------------------------------------------------
            let mut v = vec![0.0_f64; cols];
            for (j, vj) in v.iter_mut().enumerate() {
                // Simple quasi-random seed: different phase per (k, j)
                let angle = std::f64::consts::TAU
                    * ((k * 1_000_003 + j * 1_000_033) % 100_000) as f64
                    / 100_000.0_f64;
                *vj = angle.cos();
            }
            normalize_vec(&mut v).map_err(|_| {
                LowRankError::NumericalInstability(format!(
                    "could not normalize initial v for component {}",
                    k
                ))
            })?;

            let mut prev_sigma = f64::INFINITY;
            let mut u = vec![0.0_f64; rows];
            let mut sigma = 0.0_f64;

            for iter in 0..self.config.max_iterations {
                // u = A v  (matrix-vector product)
                matvec(&a, rows, cols, &v, &mut u);
                if normalize_vec(&mut u).is_err() {
                    // Deflated component has zero norm → rank exhausted
                    break;
                }

                // v = Aᵀ u  (transpose matrix-vector product)
                matvec_t(&a, rows, cols, &u, &mut v);
                if normalize_vec(&mut v).is_err() {
                    break;
                }

                // σ = uᵀ A v
                let mut av = vec![0.0_f64; rows];
                matvec(&a, rows, cols, &v, &mut av);
                sigma = dot(&u, &av);

                // Check convergence
                if iter > 0
                    && (sigma - prev_sigma).abs() < self.config.tolerance * sigma.abs().max(1e-12)
                {
                    break;
                }

                if iter == self.config.max_iterations - 1 {
                    // Reached max iterations without convergence; proceed with
                    // current estimate rather than failing hard (soft approach
                    // consistent with randomized SVD literature).
                }

                prev_sigma = sigma;
            }

            // Singular value must be non-negative
            sigma = sigma.abs();

            // Stop if this component contributes negligibly (relative to original norm)
            let relative_contribution = if a_norm > 0.0 { sigma / a_norm } else { 0.0 };
            if relative_contribution < self.config.tolerance && k > 0 {
                break;
            }

            sigmas.push(sigma);
            u_vecs.push(u.clone());
            v_vecs.push(v.clone());
            rank_used += 1;

            // Deflation: A ← A − σ · u · vᵀ
            for i in 0..rows {
                for j in 0..cols {
                    a[i * cols + j] -= sigma * u[i] * v[j];
                }
            }
        }

        if rank_used == 0 {
            return Err(LowRankError::SvdFailed {
                iterations: self.config.max_iterations,
                reason: "no singular components found".to_string(),
            });
        }

        // Assemble U  [rows × rank_used]
        let mut u_flat = vec![0.0_f64; rows * rank_used];
        for (k, uk) in u_vecs.iter().enumerate() {
            for (i, &val) in uk.iter().enumerate() {
                u_flat[i * rank_used + k] = val;
            }
        }

        // Assemble Vᵀ  [rank_used × cols]
        let mut vt_flat = vec![0.0_f64; rank_used * cols];
        for (k, vk) in v_vecs.iter().enumerate() {
            for (j, &val) in vk.iter().enumerate() {
                vt_flat[k * cols + j] = val;
            }
        }

        // Compute Frobenius reconstruction error
        let result_proto = SvdResult {
            u: u_flat.clone(),
            u_rows: rows,
            u_cols: rank_used,
            singular_values: sigmas.clone(),
            vt: vt_flat.clone(),
            vt_rows: rank_used,
            vt_cols: cols,
            frobenius_error: 0.0,
            rank_used,
        };
        let reconstructed = result_proto.reconstruct();
        let frobenius_error = Self::relative_frobenius_error(matrix, &reconstructed);

        Ok(SvdResult {
            u: u_flat,
            u_rows: rows,
            u_cols: rank_used,
            singular_values: sigmas,
            vt: vt_flat,
            vt_rows: rank_used,
            vt_cols: cols,
            frobenius_error,
            rank_used,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute `out = A · x` where A is stored row-major as `[rows × cols]`.
fn matvec(a: &[f64], rows: usize, cols: usize, x: &[f64], out: &mut [f64]) {
    for (i, out_i) in out.iter_mut().enumerate().take(rows) {
        *out_i = 0.0;
        for j in 0..cols {
            *out_i += a[i * cols + j] * x[j];
        }
    }
}

/// Compute `out = Aᵀ · x` where A is stored row-major as `[rows × cols]`.
fn matvec_t(a: &[f64], rows: usize, cols: usize, x: &[f64], out: &mut [f64]) {
    for out_j in out.iter_mut().take(cols) {
        *out_j = 0.0;
    }
    for i in 0..rows {
        for j in 0..cols {
            out[j] += a[i * cols + j] * x[i];
        }
    }
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Normalize a vector in-place.  Returns `Err(())` if the norm is near zero.
fn normalize_vec(v: &mut [f64]) -> Result<(), ()> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-300 {
        return Err(());
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(rank: usize) -> LowRankConfig {
        LowRankConfig::new(rank)
            .with_tolerance(1e-9)
            .with_max_iterations(500)
    }

    #[test]
    fn test_svd_frobenius_norm() {
        // 2×2 matrix with known Frobenius norm:  [[1,2],[3,4]]  → sqrt(30)
        let m = vec![1.0_f64, 2.0, 3.0, 4.0];
        let norm = TruncatedSvd::frobenius_norm(&m);
        let expected = (1.0_f64 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(
            (norm - expected).abs() < 1e-12,
            "norm={norm} expected={expected}"
        );
    }

    #[test]
    fn test_svd_2x2_identity() {
        // The 2×2 identity has two singular values both equal to 1.
        let m = vec![1.0_f64, 0.0, 0.0, 1.0];
        let svd = TruncatedSvd::new(make_config(2));
        let result = svd.decompose(&m, 2, 2).expect("SVD should succeed");
        assert_eq!(result.rank_used, 2);
        let mut svs = result.singular_values.clone();
        svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for sv in &svs {
            assert!(
                (*sv - 1.0).abs() < 1e-5,
                "singular value {sv} expected ~1.0"
            );
        }
    }

    #[test]
    fn test_svd_rank1_matrix() {
        // A rank-1 matrix: outer product of [1,2,3,4] with [1,2,3,4]
        let v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut m = vec![0.0_f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                m[i * 4 + j] = v[i] * v[j];
            }
        }
        let svd = TruncatedSvd::new(make_config(1));
        let result = svd.decompose(&m, 4, 4).expect("SVD should succeed");
        assert_eq!(result.rank_used, 1);
        // Reconstruction should be near-exact for a rank-1 matrix
        let rel_err = result.relative_error(&m);
        assert!(
            rel_err < 1e-5,
            "relative reconstruction error {rel_err} should be near zero"
        );
    }

    #[test]
    fn test_svd_reconstruction_error() {
        // 5×5 random-ish matrix, rank=3; compute Frobenius error
        #[rustfmt::skip]
        let m: Vec<f64> = vec![
             4.0,  3.0,  2.0,  1.0,  0.5,
             2.0,  5.0,  1.0,  0.5,  3.0,
             1.0,  2.0,  6.0,  2.0,  1.0,
             0.5,  1.0,  2.0,  4.0,  2.0,
             0.25, 0.5,  1.0,  2.0,  3.0,
        ];
        let svd = TruncatedSvd::new(make_config(3));
        let result = svd.decompose(&m, 5, 5).expect("SVD should succeed");
        assert!(result.rank_used >= 1);
        // The frobenius_error field should be a valid non-negative number
        assert!(result.frobenius_error >= 0.0);
        assert!(result.frobenius_error <= 1.1); // relative error ≤ 110%
    }

    #[test]
    fn test_svd_rank_exceeds_dimensions_error() {
        let m = vec![1.0_f64, 2.0, 3.0, 4.0];
        let svd = TruncatedSvd::new(make_config(5)); // rank=5 > min(2,2)=2
        let err = svd.decompose(&m, 2, 2);
        assert!(
            matches!(err, Err(LowRankError::RankExceedsDimensions { .. })),
            "expected RankExceedsDimensions, got {:?}",
            err
        );
    }

    #[test]
    fn test_svd_result_reconstruct() {
        // Build a trivial rank-1 SvdResult manually and verify reconstruct()
        // U = [[1], [0]], σ = [2], Vᵀ = [[0, 1]]
        // Reconstruction: [[0, 2], [0, 0]]
        let result = SvdResult {
            u: vec![1.0, 0.0],
            u_rows: 2,
            u_cols: 1,
            singular_values: vec![2.0],
            vt: vec![0.0, 1.0],
            vt_rows: 1,
            vt_cols: 2,
            frobenius_error: 0.0,
            rank_used: 1,
        };
        let rec = result.reconstruct();
        assert_eq!(rec.len(), 4);
        assert!((rec[0] - 0.0).abs() < 1e-12);
        assert!((rec[1] - 2.0).abs() < 1e-12);
        assert!((rec[2] - 0.0).abs() < 1e-12);
        assert!((rec[3] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_svd_result_energy_fraction() {
        let result = SvdResult {
            u: vec![1.0, 0.0, 0.0, 1.0],
            u_rows: 2,
            u_cols: 2,
            singular_values: vec![4.0, 2.0],
            vt: vec![1.0, 0.0, 0.0, 1.0],
            vt_rows: 2,
            vt_cols: 2,
            frobenius_error: 0.0,
            rank_used: 2,
        };
        let ef = result.energy_fraction();
        // 4+2 / 4+2 = 1.0
        assert!((ef - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_svd_result_relative_error() {
        // rank-1 outer product: exact reconstruction should give ~0 error
        let v: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut original = vec![0.0_f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                original[i * 3 + j] = v[i] * v[j];
            }
        }
        let svd = TruncatedSvd::new(make_config(1));
        let result = svd.decompose(&original, 3, 3).expect("SVD ok");
        let rel_err = result.relative_error(&original);
        assert!(
            rel_err < 1e-4,
            "relative error {rel_err} should be near zero"
        );
    }
}
