//! PROSAC (PROgressive SAmple Consensus) for robust model estimation.
//!
//! PROSAC is an improvement over RANSAC that exploits a quality ordering of
//! the input correspondences. Instead of sampling uniformly from all matches,
//! PROSAC starts by drawing from the top-ranked matches and progressively
//! expands the sampling pool. This typically converges much faster than RANSAC
//! when the matches are sorted by descriptor distance or response strength.
//!
//! # Algorithm
//!
//! 1. Sort matches by quality (ascending descriptor distance).
//! 2. Maintain a growing subset size `n` (starting from `min_sample_size`).
//! 3. At each iteration, with high probability draw at least one sample from
//!    the `n`-th match (the "growth" step), and the rest from the top `n-1`.
//! 4. If the model from the current sample has more inliers than the best so
//!    far, update the best model.
//! 5. Adaptively update `n` based on the inlier ratio to expand the pool.
//!
//! # References
//!
//! - Chum, O. and Matas, J. "Matching with PROSAC - Progressive Sample Consensus"
//!   CVPR 2005.

use crate::features::MatchPair;
use crate::{AlignError, AlignResult};

/// Configuration for PROSAC.
#[derive(Debug, Clone)]
pub struct ProsacConfig {
    /// Distance threshold for counting inliers (in pixels).
    pub inlier_threshold: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Minimum number of inliers required for a valid model.
    pub min_inliers: usize,
    /// Confidence level (0.0 to 1.0). Higher values mean more iterations.
    pub confidence: f64,
    /// Initial subset size (must be >= min_sample_size).
    /// If `None`, starts at `min_sample_size`.
    pub initial_n: Option<usize>,
}

impl Default for ProsacConfig {
    fn default() -> Self {
        Self {
            inlier_threshold: 3.0,
            max_iterations: 2000,
            min_inliers: 8,
            confidence: 0.99,
            initial_n: None,
        }
    }
}

/// Model type that PROSAC can estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProsacModelType {
    /// Affine transform (minimum 3 points).
    Affine,
    /// Homography (minimum 4 points).
    Homography,
}

impl ProsacModelType {
    /// Minimum number of samples required to fit this model.
    #[must_use]
    pub fn min_samples(&self) -> usize {
        match self {
            Self::Affine => 3,
            Self::Homography => 4,
        }
    }
}

/// Result of a PROSAC estimation.
#[derive(Debug, Clone)]
pub struct ProsacResult {
    /// The estimated model parameters (as a flat vector).
    /// For Affine: [a, b, tx, c, d, ty] (6 elements).
    /// For Homography: [h00..h22] (9 elements, row-major 3x3).
    pub params: Vec<f64>,
    /// Inlier mask (true = inlier).
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub num_inliers: usize,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// PROSAC estimator.
pub struct ProsacEstimator {
    /// Configuration.
    pub config: ProsacConfig,
    /// Model type.
    pub model_type: ProsacModelType,
}

impl ProsacEstimator {
    /// Create a new PROSAC estimator.
    #[must_use]
    pub fn new(config: ProsacConfig, model_type: ProsacModelType) -> Self {
        Self { config, model_type }
    }

    /// Run PROSAC on the given matches.
    ///
    /// Matches should be pre-sorted by quality (ascending descriptor distance
    /// = best first). If they are not sorted, PROSAC degrades gracefully to
    /// RANSAC-like behaviour.
    ///
    /// # Errors
    ///
    /// Returns an error if there are insufficient matches.
    pub fn estimate(&self, matches: &[MatchPair]) -> AlignResult<ProsacResult> {
        let min_s = self.model_type.min_samples();

        if matches.len() < min_s {
            return Err(AlignError::InsufficientData(format!(
                "Need at least {min_s} matches, got {}",
                matches.len()
            )));
        }

        let total = matches.len();
        let mut n = self.config.initial_n.unwrap_or(min_s).max(min_s).min(total);
        let mut best_inliers = 0usize;
        let mut best_mask: Vec<bool> = vec![false; total];
        let mut best_params: Vec<f64> = Vec::new();
        let mut best_iter = 0;

        // Deterministic PRNG seed
        let mut rng_state = 0x1234_5678_u64;

        // Growth function: how many iterations to spend at subset size n
        // before expanding to n+1.  We use the simplified formula from the
        // PROSAC paper:  T(n) = T(N) * C(n, m) / C(N, m)
        // where m = min_samples, N = total.
        // For practical purposes we approximate with a linear growth schedule.
        let mut t_n = 1.0_f64; // iterations at current n
        let mut t_n_prime = 0.0_f64; // fractional accumulator

        for iter in 0..self.config.max_iterations {
            // Progressive sampling: decide whether to include the n-th point
            t_n_prime += 1.0;

            if t_n_prime >= t_n && n < total {
                n += 1;
                // Update T(n) using the ratio formula
                let ratio = if n > min_s {
                    (n - min_s) as f64 / n as f64
                } else {
                    1.0
                };
                t_n *= 1.0 + ratio;
                t_n_prime = 0.0;
            }

            // Sample min_s points from the top n matches
            let sample = self.sample_from_top_n(matches, n, min_s, &mut rng_state);

            // Fit model
            let params = match self.model_type {
                ProsacModelType::Affine => self.fit_affine(&sample),
                ProsacModelType::Homography => self.fit_homography(&sample),
            };

            let params = match params {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Count inliers
            let (mask, count) = self.count_inliers(matches, &params);

            if count > best_inliers {
                best_inliers = count;
                best_mask = mask;
                best_params = params;
                best_iter = iter;

                // Adaptive termination: if we have enough inliers, check if
                // we can stop early based on the confidence level.
                if best_inliers >= self.config.min_inliers {
                    let inlier_ratio = best_inliers as f64 / total as f64;
                    let expected_iters =
                        adaptive_max_iterations(inlier_ratio, min_s, self.config.confidence);
                    if iter as f64 >= expected_iters {
                        break;
                    }
                }
            }
        }

        if best_inliers < self.config.min_inliers {
            return Err(AlignError::NoSolution(format!(
                "Insufficient inliers: {best_inliers} < {}",
                self.config.min_inliers
            )));
        }

        // Refine with all inliers
        let inlier_matches: Vec<&MatchPair> = matches
            .iter()
            .zip(&best_mask)
            .filter(|(_, &is_inlier)| is_inlier)
            .map(|(m, _)| m)
            .collect();

        let refined_params = match self.model_type {
            ProsacModelType::Affine => {
                let pairs: Vec<MatchPair> = inlier_matches.iter().map(|m| (*m).clone()).collect();
                self.fit_affine(&pairs).unwrap_or(best_params.clone())
            }
            ProsacModelType::Homography => {
                let pairs: Vec<MatchPair> = inlier_matches.iter().map(|m| (*m).clone()).collect();
                self.fit_homography(&pairs).unwrap_or(best_params.clone())
            }
        };

        // Recount inliers with refined model
        let (final_mask, final_count) = self.count_inliers(matches, &refined_params);

        Ok(ProsacResult {
            params: refined_params,
            inlier_mask: final_mask,
            num_inliers: final_count,
            iterations: best_iter + 1,
        })
    }

    // -- Sampling -------------------------------------------------------------

    fn sample_from_top_n(
        &self,
        matches: &[MatchPair],
        n: usize,
        count: usize,
        rng: &mut u64,
    ) -> Vec<MatchPair> {
        let pool_size = n.min(matches.len());
        let mut indices = Vec::with_capacity(count);

        while indices.len() < count {
            let idx = lcg_next(rng) as usize % pool_size;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        indices.iter().map(|&i| matches[i].clone()).collect()
    }

    // -- Model fitting --------------------------------------------------------

    fn fit_affine(&self, matches: &[MatchPair]) -> AlignResult<Vec<f64>> {
        if matches.len() < 3 {
            return Err(AlignError::InsufficientData(
                "Need >= 3 points for affine".to_string(),
            ));
        }

        // Least squares: [x y 1 0 0 0] [a]   [x']
        //                 [0 0 0 x y 1] [b] = [y']
        //                                      [tx]
        //                                      [c]
        //                                      [d]
        //                                      [ty]
        let n = matches.len();
        let _rows = n * 2;

        // Build ATA (6x6) and ATb (6x1) incrementally
        let mut ata = [0.0_f64; 36];
        let mut atb = [0.0_f64; 6];

        for m in matches {
            let x = m.point1.x;
            let y = m.point1.y;
            let xp = m.point2.x;
            let yp = m.point2.y;

            // Row 1: [x, y, 1, 0, 0, 0] -> xp
            let r1 = [x, y, 1.0, 0.0, 0.0, 0.0];
            // Row 2: [0, 0, 0, x, y, 1] -> yp
            let r2 = [0.0, 0.0, 0.0, x, y, 1.0];

            for i in 0..6 {
                for j in 0..6 {
                    ata[i * 6 + j] += r1[i] * r1[j] + r2[i] * r2[j];
                }
                atb[i] += r1[i] * xp + r2[i] * yp;
            }
        }

        // Solve 6x6 system using Cramer's/Gaussian elimination
        let solution = solve_6x6(&ata, &atb)?;

        Ok(solution.to_vec())
    }

    fn fit_homography(&self, matches: &[MatchPair]) -> AlignResult<Vec<f64>> {
        if matches.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need >= 4 points for homography".to_string(),
            ));
        }

        // Use DLT: build 2n x 9 matrix and find null space via SVD-like approach.
        // For simplicity, we use the iterative power method for the smallest
        // singular value.

        // Normalize points for numerical stability
        let (norm1, t1) = normalize_points(matches, true);
        let (norm2, t2) = normalize_points(matches, false);

        let n = matches.len();
        // Build ATA (9x9) = sum of a_i * a_i^T
        let mut ata = [0.0_f64; 81];

        for i in 0..n {
            let x = norm1[i].0;
            let y = norm1[i].1;
            let xp = norm2[i].0;
            let yp = norm2[i].1;

            let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp];
            let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp];

            for a in 0..9 {
                for b in 0..9 {
                    ata[a * 9 + b] += r1[a] * r1[b] + r2[a] * r2[b];
                }
            }
        }

        // Find eigenvector of ATA with smallest eigenvalue using inverse iteration
        let h_norm = find_smallest_eigenvector_9x9(&ata)?;

        // Denormalize: H = T2_inv * H_norm * T1
        let h = denormalize_homography(&h_norm, &t1, &t2);

        // Normalize so h[8] = 1
        if h[8].abs() < 1e-12 {
            return Err(AlignError::NumericalError(
                "Degenerate homography".to_string(),
            ));
        }

        let scale = h[8];
        Ok(h.iter().map(|&v| v / scale).collect())
    }

    // -- Inlier counting ------------------------------------------------------

    fn count_inliers(&self, matches: &[MatchPair], params: &[f64]) -> (Vec<bool>, usize) {
        let threshold_sq = self.config.inlier_threshold * self.config.inlier_threshold;
        let mut mask = vec![false; matches.len()];
        let mut count = 0usize;

        for (i, m) in matches.iter().enumerate() {
            let projected = self.project_point(m.point1.x, m.point1.y, params);
            let dx = projected.0 - m.point2.x;
            let dy = projected.1 - m.point2.y;
            let err_sq = dx * dx + dy * dy;

            if err_sq < threshold_sq {
                mask[i] = true;
                count += 1;
            }
        }

        (mask, count)
    }

    fn project_point(&self, x: f64, y: f64, params: &[f64]) -> (f64, f64) {
        match self.model_type {
            ProsacModelType::Affine => {
                if params.len() < 6 {
                    return (x, y);
                }
                let xp = params[0] * x + params[1] * y + params[2];
                let yp = params[3] * x + params[4] * y + params[5];
                (xp, yp)
            }
            ProsacModelType::Homography => {
                if params.len() < 9 {
                    return (x, y);
                }
                let w = params[6] * x + params[7] * y + params[8];
                if w.abs() < 1e-12 {
                    return (x, y);
                }
                let xp = (params[0] * x + params[1] * y + params[2]) / w;
                let yp = (params[3] * x + params[4] * y + params[5]) / w;
                (xp, yp)
            }
        }
    }
}

// -- Adaptive iteration count -------------------------------------------------

pub(crate) fn adaptive_max_iterations(
    inlier_ratio: f64,
    min_samples: usize,
    confidence: f64,
) -> f64 {
    if inlier_ratio <= 0.0 || inlier_ratio >= 1.0 {
        return 1.0;
    }

    let num = (1.0 - confidence).ln();
    let denom = (1.0 - inlier_ratio.powi(min_samples as i32)).ln();

    if denom.abs() < 1e-15 {
        return 1.0;
    }

    num / denom
}

// -- LCG PRNG -----------------------------------------------------------------

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

// -- Linear algebra helpers ---------------------------------------------------

/// Solve a 6x6 linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_6x6(ata: &[f64; 36], atb: &[f64; 6]) -> AlignResult<[f64; 6]> {
    let mut a = *ata;
    let mut b = *atb;

    // Forward elimination
    for col in 0..6 {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = a[col * 6 + col].abs();
        for row in (col + 1)..6 {
            let val = a[row * 6 + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(AlignError::NumericalError(
                "Singular matrix in 6x6 solve".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..6 {
                a.swap(col * 6 + j, max_row * 6 + j);
            }
            b.swap(col, max_row);
        }

        // Eliminate below
        let pivot = a[col * 6 + col];
        for row in (col + 1)..6 {
            let factor = a[row * 6 + col] / pivot;
            for j in col..6 {
                a[row * 6 + j] -= factor * a[col * 6 + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = [0.0_f64; 6];
    for col in (0..6).rev() {
        let mut sum = b[col];
        for j in (col + 1)..6 {
            sum -= a[col * 6 + j] * x[j];
        }
        x[col] = sum / a[col * 6 + col];
    }

    Ok(x)
}

/// Normalize 2D points: translate centroid to origin, scale so avg distance = sqrt(2).
fn normalize_points(matches: &[MatchPair], use_first: bool) -> (Vec<(f64, f64)>, [f64; 9]) {
    let pts: Vec<(f64, f64)> = if use_first {
        matches.iter().map(|m| (m.point1.x, m.point1.y)).collect()
    } else {
        matches.iter().map(|m| (m.point2.x, m.point2.y)).collect()
    };

    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.1).sum::<f64>() / n;

    let avg_dist: f64 = pts
        .iter()
        .map(|p| ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let s = if avg_dist > 1e-10 {
        std::f64::consts::SQRT_2 / avg_dist
    } else {
        1.0
    };

    let normalized: Vec<(f64, f64)> = pts
        .iter()
        .map(|p| ((p.0 - cx) * s, (p.1 - cy) * s))
        .collect();

    // T = [s 0 -s*cx; 0 s -s*cy; 0 0 1]
    let t = [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0];

    (normalized, t)
}

/// Denormalize homography: H = T2_inv * Hn * T1
fn denormalize_homography(h_norm: &[f64; 9], t1: &[f64; 9], t2: &[f64; 9]) -> [f64; 9] {
    // T2_inv
    let s2 = t2[0];
    let tx2 = t2[2];
    let ty2 = t2[5];

    let t2_inv = if s2.abs() > 1e-15 {
        let inv_s = 1.0 / s2;
        [
            inv_s,
            0.0,
            -tx2 * inv_s,
            0.0,
            inv_s,
            -ty2 * inv_s,
            0.0,
            0.0,
            1.0,
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    };

    let tmp = mat3_mul(&t2_inv, h_norm);
    mat3_mul(&tmp, t1)
}

fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

/// Find the eigenvector corresponding to the smallest eigenvalue of a 9x9
/// symmetric positive semi-definite matrix, using inverse iteration with
/// a shift near zero.
fn find_smallest_eigenvector_9x9(ata: &[f64; 81]) -> AlignResult<[f64; 9]> {
    // We use 50 iterations of inverse power iteration with a small shift.
    let shift = 1e-8;
    let mut a_shifted = *ata;
    for i in 0..9 {
        a_shifted[i * 9 + i] += shift;
    }

    // Start with a uniform vector
    let mut v = [1.0_f64 / 3.0; 9];

    for _iter in 0..50 {
        // Solve (ATA + shift*I) * w = v
        let w = solve_9x9_gauss(&a_shifted, &v)?;

        // Normalize
        let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return Err(AlignError::NumericalError(
                "Eigenvector iteration diverged".to_string(),
            ));
        }
        for i in 0..9 {
            v[i] = w[i] / norm;
        }
    }

    Ok(v)
}

/// Solve a 9x9 linear system using Gaussian elimination.
fn solve_9x9_gauss(a: &[f64; 81], b: &[f64; 9]) -> AlignResult<[f64; 9]> {
    let mut mat = *a;
    let mut rhs = *b;

    for col in 0..9 {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = mat[col * 9 + col].abs();
        for row in (col + 1)..9 {
            let val = mat[row * 9 + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(AlignError::NumericalError(
                "Singular matrix in 9x9 solve".to_string(),
            ));
        }

        if max_row != col {
            for j in 0..9 {
                mat.swap(col * 9 + j, max_row * 9 + j);
            }
            rhs.swap(col, max_row);
        }

        let pivot = mat[col * 9 + col];
        for row in (col + 1)..9 {
            let factor = mat[row * 9 + col] / pivot;
            for j in col..9 {
                mat[row * 9 + j] -= factor * mat[col * 9 + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut x = [0.0_f64; 9];
    for col in (0..9).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..9 {
            sum -= mat[col * 9 + j] * x[j];
        }
        x[col] = sum / mat[col * 9 + col];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point2D;

    fn make_affine_matches(
        a: f64,
        b: f64,
        tx: f64,
        c: f64,
        d: f64,
        ty: f64,
        n: usize,
    ) -> Vec<MatchPair> {
        (0..n)
            .map(|i| {
                let x = (i as f64 * 17.0) % 100.0;
                let y = (i as f64 * 31.0) % 100.0;
                let xp = a * x + b * y + tx;
                let yp = c * x + d * y + ty;
                MatchPair::new(i, i, i as u32, Point2D::new(x, y), Point2D::new(xp, yp))
            })
            .collect()
    }

    fn make_identity_matches(n: usize) -> Vec<MatchPair> {
        make_affine_matches(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, n)
    }

    // -- ProsacConfig ---------------------------------------------------------

    #[test]
    fn test_prosac_config_default() {
        let config = ProsacConfig::default();
        assert_eq!(config.max_iterations, 2000);
        assert_eq!(config.min_inliers, 8);
    }

    #[test]
    fn test_model_type_min_samples() {
        assert_eq!(ProsacModelType::Affine.min_samples(), 3);
        assert_eq!(ProsacModelType::Homography.min_samples(), 4);
    }

    // -- Affine estimation ----------------------------------------------------

    #[test]
    fn test_prosac_affine_identity() {
        let matches = make_identity_matches(20);

        let estimator = ProsacEstimator::new(
            ProsacConfig {
                min_inliers: 5,
                ..ProsacConfig::default()
            },
            ProsacModelType::Affine,
        );

        let result = estimator.estimate(&matches).expect("should succeed");
        assert!(result.num_inliers >= 5);
        assert_eq!(result.params.len(), 6);

        // Check that parameters are close to identity [1,0,0, 0,1,0]
        assert!(
            (result.params[0] - 1.0).abs() < 0.1,
            "a={}",
            result.params[0]
        );
        assert!((result.params[1]).abs() < 0.1, "b={}", result.params[1]);
        assert!((result.params[2]).abs() < 1.0, "tx={}", result.params[2]);
        assert!((result.params[3]).abs() < 0.1, "c={}", result.params[3]);
        assert!(
            (result.params[4] - 1.0).abs() < 0.1,
            "d={}",
            result.params[4]
        );
        assert!((result.params[5]).abs() < 1.0, "ty={}", result.params[5]);
    }

    #[test]
    fn test_prosac_affine_translation() {
        let matches = make_affine_matches(1.0, 0.0, 10.0, 0.0, 1.0, -5.0, 20);

        let estimator = ProsacEstimator::new(
            ProsacConfig {
                min_inliers: 5,
                ..ProsacConfig::default()
            },
            ProsacModelType::Affine,
        );

        let result = estimator.estimate(&matches).expect("should succeed");
        assert!(
            (result.params[2] - 10.0).abs() < 1.0,
            "tx={}",
            result.params[2]
        );
        assert!(
            (result.params[5] + 5.0).abs() < 1.0,
            "ty={}",
            result.params[5]
        );
    }

    #[test]
    fn test_prosac_affine_with_outliers() {
        let mut matches = make_affine_matches(1.0, 0.0, 5.0, 0.0, 1.0, 3.0, 30);

        // Add some outliers
        for i in 0..5 {
            matches.push(MatchPair::new(
                30 + i,
                30 + i,
                100,
                Point2D::new(i as f64 * 10.0, i as f64 * 10.0),
                Point2D::new(999.0, 999.0),
            ));
        }

        let estimator = ProsacEstimator::new(
            ProsacConfig {
                min_inliers: 5,
                ..ProsacConfig::default()
            },
            ProsacModelType::Affine,
        );

        let result = estimator.estimate(&matches).expect("should succeed");
        // Should reject outliers
        assert!(result.num_inliers >= 20);
    }

    // -- Homography estimation ------------------------------------------------

    #[test]
    fn test_prosac_homography_identity() {
        let matches = make_identity_matches(20);

        let estimator = ProsacEstimator::new(
            ProsacConfig {
                min_inliers: 5,
                ..ProsacConfig::default()
            },
            ProsacModelType::Homography,
        );

        let result = estimator.estimate(&matches).expect("should succeed");
        assert!(result.num_inliers >= 5);
        assert_eq!(result.params.len(), 9);

        // H should be close to identity
        assert!(
            (result.params[0] - 1.0).abs() < 0.2,
            "h00={}",
            result.params[0]
        );
        assert!(
            (result.params[4] - 1.0).abs() < 0.2,
            "h11={}",
            result.params[4]
        );
        assert!(
            (result.params[8] - 1.0).abs() < 0.2,
            "h22={}",
            result.params[8]
        );
    }

    #[test]
    fn test_prosac_insufficient_matches() {
        let matches = vec![MatchPair::new(
            0,
            0,
            0,
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        )];

        let estimator = ProsacEstimator::new(ProsacConfig::default(), ProsacModelType::Homography);
        let result = estimator.estimate(&matches);
        assert!(result.is_err());
    }

    // -- Adaptive iteration count ---------------------------------------------

    #[test]
    fn test_adaptive_max_iterations() {
        let iters = adaptive_max_iterations(0.5, 4, 0.99);
        assert!(iters > 0.0 && iters < 10_000.0, "iters={iters}");
    }

    #[test]
    fn test_adaptive_max_iterations_high_inlier_ratio() {
        let iters = adaptive_max_iterations(0.9, 4, 0.99);
        // High inlier ratio should need few iterations
        assert!(iters < 100.0, "iters={iters}");
    }

    #[test]
    fn test_adaptive_max_iterations_edge_cases() {
        assert_eq!(adaptive_max_iterations(0.0, 4, 0.99), 1.0);
        assert_eq!(adaptive_max_iterations(1.0, 4, 0.99), 1.0);
    }

    // -- LCG PRNG -------------------------------------------------------------

    #[test]
    fn test_lcg_deterministic() {
        let mut state1 = 42u64;
        let mut state2 = 42u64;
        assert_eq!(lcg_next(&mut state1), lcg_next(&mut state2));
    }

    #[test]
    fn test_lcg_different_seeds() {
        let mut s1 = 1u64;
        let mut s2 = 2u64;
        assert_ne!(lcg_next(&mut s1), lcg_next(&mut s2));
    }

    // -- Linear algebra -------------------------------------------------------

    #[test]
    fn test_solve_6x6_identity_system() {
        // Identity matrix * x = b => x = b
        let mut ata = [0.0_f64; 36];
        for i in 0..6 {
            ata[i * 6 + i] = 1.0;
        }
        let atb = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = solve_6x6(&ata, &atb).expect("should succeed");
        for i in 0..6 {
            assert!((x[i] - atb[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_solve_6x6_singular() {
        let ata = [0.0_f64; 36]; // all zeros = singular
        let atb = [1.0; 6];
        assert!(solve_6x6(&ata, &atb).is_err());
    }

    #[test]
    fn test_mat3_mul_identity() {
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let c = mat3_mul(&id, &a);
        for i in 0..9 {
            assert!((c[i] - a[i]).abs() < 1e-10);
        }
    }

    // -- Normalization --------------------------------------------------------

    #[test]
    fn test_normalize_points_centered() {
        let matches = vec![
            MatchPair::new(0, 0, 0, Point2D::new(-1.0, -1.0), Point2D::new(0.0, 0.0)),
            MatchPair::new(1, 1, 0, Point2D::new(1.0, 1.0), Point2D::new(0.0, 0.0)),
        ];
        let (norm, _t) = normalize_points(&matches, true);
        // Centroid should be at origin
        let cx = norm.iter().map(|p| p.0).sum::<f64>() / norm.len() as f64;
        let cy = norm.iter().map(|p| p.1).sum::<f64>() / norm.len() as f64;
        assert!(cx.abs() < 1e-10);
        assert!(cy.abs() < 1e-10);
    }
}
