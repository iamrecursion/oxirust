//! Parallel RANSAC with early termination.
//!
//! Uses rayon to run multiple RANSAC iterations in parallel, collecting
//! results and terminating early when the inlier ratio exceeds a threshold.
//! This significantly speeds up RANSAC on multi-core systems.
//!
//! # Algorithm
//!
//! 1. Divide the total iterations into parallel batches.
//! 2. Each batch independently samples, fits models, and counts inliers.
//! 3. After each batch, check the best inlier ratio. If it exceeds the
//!    early termination threshold, stop.
//! 4. Return the best model found.

use crate::features::MatchPair;
use crate::{prosac, AlignError, AlignResult};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Configuration for parallel RANSAC.
#[derive(Debug, Clone)]
pub struct ParallelRansacConfig {
    /// Distance threshold for inliers (pixels).
    pub threshold: f64,
    /// Maximum total iterations (split across threads).
    pub max_iterations: usize,
    /// Minimum number of inliers required for a valid model.
    pub min_inliers: usize,
    /// Early termination inlier ratio (0.0 to 1.0).
    /// If the best model has inlier_count / total >= this ratio, stop early.
    pub early_termination_ratio: f64,
    /// Number of iterations per parallel batch.
    pub batch_size: usize,
    /// RANSAC confidence level (0.0 to 1.0) for adaptive iteration bound.
    ///
    /// Higher values require more iterations to guarantee the bound.
    /// Default: 0.99.
    pub confidence: f64,
}

impl Default for ParallelRansacConfig {
    fn default() -> Self {
        Self {
            threshold: 3.0,
            max_iterations: 2000,
            min_inliers: 8,
            early_termination_ratio: 0.8,
            batch_size: 100,
            confidence: 0.99,
        }
    }
}

/// Model type for parallel RANSAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelModelType {
    /// Affine transform (3 point minimum).
    Affine,
    /// Homography (4 point minimum).
    Homography,
}

impl ParallelModelType {
    /// Minimum sample size for this model type.
    #[must_use]
    pub fn min_samples(&self) -> usize {
        match self {
            Self::Affine => 3,
            Self::Homography => 4,
        }
    }
}

/// Result of parallel RANSAC.
#[derive(Debug, Clone)]
pub struct ParallelRansacResult {
    /// The estimated model parameters.
    pub params: Vec<f64>,
    /// Inlier mask (true = inlier).
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub num_inliers: usize,
    /// Total iterations performed.
    pub iterations: usize,
    /// Whether early termination was triggered.
    pub early_terminated: bool,
}

/// Parallel RANSAC estimator.
pub struct ParallelRansac {
    /// Configuration.
    pub config: ParallelRansacConfig,
    /// Model type.
    pub model_type: ParallelModelType,
}

impl ParallelRansac {
    /// Create a new parallel RANSAC estimator.
    #[must_use]
    pub fn new(config: ParallelRansacConfig, model_type: ParallelModelType) -> Self {
        Self { config, model_type }
    }

    /// Run parallel RANSAC on the given matches.
    ///
    /// # Errors
    ///
    /// Returns an error if there are insufficient matches.
    pub fn estimate(&self, matches: &[MatchPair]) -> AlignResult<ParallelRansacResult> {
        let min_s = self.model_type.min_samples();
        if matches.len() < min_s {
            return Err(AlignError::InsufficientData(format!(
                "Need at least {min_s} matches, got {}",
                matches.len()
            )));
        }

        let total = matches.len();
        let threshold_sq = self.config.threshold * self.config.threshold;
        let terminated = AtomicBool::new(false);
        let global_best_inliers = AtomicUsize::new(0);

        let num_batches = self.config.max_iterations.div_ceil(self.config.batch_size);
        let mut overall_best: Option<(Vec<f64>, Vec<bool>, usize)> = None;
        let mut total_iterations = 0usize;
        let mut early_terminated = false;

        for batch_idx in 0..num_batches {
            if terminated.load(Ordering::Relaxed) {
                early_terminated = true;
                break;
            }

            let remaining = self.config.max_iterations - total_iterations;
            let this_batch = remaining.min(self.config.batch_size);

            // Run iterations in parallel
            let batch_results: Vec<Option<(Vec<f64>, Vec<bool>, usize)>> = (0..this_batch)
                .into_par_iter()
                .map(|iter_in_batch| {
                    if terminated.load(Ordering::Relaxed) {
                        return None;
                    }

                    // Deterministic seed per iteration
                    let global_iter = batch_idx * self.config.batch_size + iter_in_batch;
                    let mut rng = 0x1234_5678_u64 ^ (global_iter as u64 * 2654435761);

                    // Sample
                    let sample = self.sample(matches, min_s, &mut rng);

                    // Fit model
                    let params = match self.model_type {
                        ParallelModelType::Affine => self.fit_affine(&sample),
                        ParallelModelType::Homography => self.fit_homography(&sample),
                    };

                    let params = match params {
                        Ok(p) => p,
                        Err(_) => return None,
                    };

                    // Count inliers
                    let mut mask = vec![false; total];
                    let mut count = 0usize;
                    for (i, m) in matches.iter().enumerate() {
                        let (px, py) = self.project(m.point1.x, m.point1.y, &params);
                        let dx = px - m.point2.x;
                        let dy = py - m.point2.y;
                        if dx * dx + dy * dy < threshold_sq {
                            mask[i] = true;
                            count += 1;
                        }
                    }

                    // Update global best
                    let prev_best = global_best_inliers.load(Ordering::Relaxed);
                    if count > prev_best {
                        global_best_inliers.fetch_max(count, Ordering::Relaxed);
                    }

                    // Check early termination
                    let ratio = count as f64 / total as f64;
                    if ratio >= self.config.early_termination_ratio
                        && count >= self.config.min_inliers
                    {
                        terminated.store(true, Ordering::Relaxed);
                    }

                    Some((params, mask, count))
                })
                .collect();

            total_iterations += this_batch;

            // Find best in this batch
            for result in batch_results.into_iter().flatten() {
                let (params, mask, count) = result;
                let is_better = match &overall_best {
                    Some((_, _, best_count)) => count > *best_count,
                    None => true,
                };
                if is_better {
                    overall_best = Some((params, mask, count));
                }
            }

            // Check if early termination was triggered
            if terminated.load(Ordering::Relaxed) {
                early_terminated = true;
                break;
            }

            // Adaptive iteration bound: recompute max iterations needed
            // from the current best inlier ratio after each batch.
            let current_best = global_best_inliers.load(Ordering::Relaxed);
            if current_best > 0 {
                let inlier_ratio = current_best as f64 / total.max(1) as f64;

                // If essentially all matches are inliers, we can stop immediately.
                if inlier_ratio >= 1.0 - 1e-9 {
                    early_terminated = true;
                    break;
                }

                let n_adaptive = prosac::adaptive_max_iterations(
                    inlier_ratio.max(1e-6),
                    min_s,
                    self.config.confidence,
                )
                .ceil() as usize;

                if total_iterations >= n_adaptive {
                    early_terminated = true;
                    break;
                }
            }
        }

        let (best_params, best_mask, best_count) = overall_best.ok_or_else(|| {
            AlignError::NoSolution("No valid model found in parallel RANSAC".to_string())
        })?;

        if best_count < self.config.min_inliers {
            return Err(AlignError::NoSolution(format!(
                "Insufficient inliers: {best_count} < {}",
                self.config.min_inliers
            )));
        }

        // Refine with all inliers
        let inlier_matches: Vec<MatchPair> = matches
            .iter()
            .zip(&best_mask)
            .filter(|(_, &is_inlier)| is_inlier)
            .map(|(m, _)| m.clone())
            .collect();

        let refined_params = match self.model_type {
            ParallelModelType::Affine => self
                .fit_affine(&inlier_matches)
                .unwrap_or_else(|_| best_params.clone()),
            ParallelModelType::Homography => self
                .fit_homography(&inlier_matches)
                .unwrap_or_else(|_| best_params.clone()),
        };

        // Recount inliers with refined model
        let mut final_mask = vec![false; total];
        let mut final_count = 0usize;
        for (i, m) in matches.iter().enumerate() {
            let (px, py) = self.project(m.point1.x, m.point1.y, &refined_params);
            let dx = px - m.point2.x;
            let dy = py - m.point2.y;
            if dx * dx + dy * dy < threshold_sq {
                final_mask[i] = true;
                final_count += 1;
            }
        }

        Ok(ParallelRansacResult {
            params: refined_params,
            inlier_mask: final_mask,
            num_inliers: final_count,
            iterations: total_iterations,
            early_terminated,
        })
    }

    // -- Sampling --------------------------------------------------------------

    fn sample(&self, matches: &[MatchPair], count: usize, rng: &mut u64) -> Vec<MatchPair> {
        let pool = matches.len();
        let mut indices = Vec::with_capacity(count);

        while indices.len() < count {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (*rng >> 33) as usize % pool;
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

        let mut ata = [0.0_f64; 36];
        let mut atb = [0.0_f64; 6];

        for m in matches {
            let x = m.point1.x;
            let y = m.point1.y;
            let xp = m.point2.x;
            let yp = m.point2.y;

            let r1 = [x, y, 1.0, 0.0, 0.0, 0.0];
            let r2 = [0.0, 0.0, 0.0, x, y, 1.0];

            for i in 0..6 {
                for j in 0..6 {
                    ata[i * 6 + j] += r1[i] * r1[j] + r2[i] * r2[j];
                }
                atb[i] += r1[i] * xp + r2[i] * yp;
            }
        }

        let solution = solve_6x6(&ata, &atb)?;
        Ok(solution.to_vec())
    }

    fn fit_homography(&self, matches: &[MatchPair]) -> AlignResult<Vec<f64>> {
        if matches.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need >= 4 points for homography".to_string(),
            ));
        }

        let mut ata = [0.0_f64; 81];

        for m in matches {
            let x = m.point1.x;
            let y = m.point1.y;
            let xp = m.point2.x;
            let yp = m.point2.y;

            let r1 = [-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp];
            let r2 = [0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp];

            for a in 0..9 {
                for b in 0..9 {
                    ata[a * 9 + b] += r1[a] * r1[b] + r2[a] * r2[b];
                }
            }
        }

        let h = find_smallest_eigenvector_9x9(&ata)?;

        if h[8].abs() < 1e-12 {
            return Err(AlignError::NumericalError(
                "Degenerate homography".to_string(),
            ));
        }

        let scale = h[8];
        Ok(h.iter().map(|&v| v / scale).collect())
    }

    fn project(&self, x: f64, y: f64, params: &[f64]) -> (f64, f64) {
        match self.model_type {
            ParallelModelType::Affine => {
                if params.len() < 6 {
                    return (x, y);
                }
                (
                    params[0] * x + params[1] * y + params[2],
                    params[3] * x + params[4] * y + params[5],
                )
            }
            ParallelModelType::Homography => {
                if params.len() < 9 {
                    return (x, y);
                }
                let w = params[6] * x + params[7] * y + params[8];
                if w.abs() < 1e-12 {
                    return (x, y);
                }
                (
                    (params[0] * x + params[1] * y + params[2]) / w,
                    (params[3] * x + params[4] * y + params[5]) / w,
                )
            }
        }
    }
}

// -- Linear algebra helpers (duplicated from prosac to keep modules independent) --

fn solve_6x6(ata: &[f64; 36], atb: &[f64; 6]) -> AlignResult<[f64; 6]> {
    let mut a = *ata;
    let mut b = *atb;

    for col in 0..6 {
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
        if max_row != col {
            for j in 0..6 {
                a.swap(col * 6 + j, max_row * 6 + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * 6 + col];
        for row in (col + 1)..6 {
            let factor = a[row * 6 + col] / pivot;
            for j in col..6 {
                a[row * 6 + j] -= factor * a[col * 6 + j];
            }
            b[row] -= factor * b[col];
        }
    }

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

fn find_smallest_eigenvector_9x9(ata: &[f64; 81]) -> AlignResult<[f64; 9]> {
    let shift = 1e-8;
    let mut a_shifted = *ata;
    for i in 0..9 {
        a_shifted[i * 9 + i] += shift;
    }

    let mut v = [1.0_f64 / 3.0; 9];

    for _ in 0..50 {
        let w = solve_9x9_gauss(&a_shifted, &v)?;
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

fn solve_9x9_gauss(a: &[f64; 81], b: &[f64; 9]) -> AlignResult<[f64; 9]> {
    let mut mat = *a;
    let mut rhs = *b;

    for col in 0..9 {
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

    fn make_affine_matches(tx: f64, ty: f64, n: usize) -> Vec<MatchPair> {
        (0..n)
            .map(|i| {
                let x = (i as f64 * 17.0) % 100.0;
                let y = (i as f64 * 31.0) % 100.0;
                MatchPair::new(
                    i,
                    i,
                    i as u32,
                    Point2D::new(x, y),
                    Point2D::new(x + tx, y + ty),
                )
            })
            .collect()
    }

    fn make_identity_matches(n: usize) -> Vec<MatchPair> {
        make_affine_matches(0.0, 0.0, n)
    }

    #[test]
    fn test_config_default() {
        let config = ParallelRansacConfig::default();
        assert_eq!(config.max_iterations, 2000);
        assert_eq!(config.batch_size, 100);
        assert!((config.early_termination_ratio - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_model_type_min_samples() {
        assert_eq!(ParallelModelType::Affine.min_samples(), 3);
        assert_eq!(ParallelModelType::Homography.min_samples(), 4);
    }

    #[test]
    fn test_insufficient_matches() {
        let ransac =
            ParallelRansac::new(ParallelRansacConfig::default(), ParallelModelType::Affine);
        let matches = vec![MatchPair::new(
            0,
            0,
            0,
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        )];
        assert!(ransac.estimate(&matches).is_err());
    }

    #[test]
    fn test_affine_identity() {
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                min_inliers: 5,
                ..ParallelRansacConfig::default()
            },
            ParallelModelType::Affine,
        );

        let matches = make_identity_matches(20);
        let result = ransac.estimate(&matches).expect("should succeed");

        assert!(result.num_inliers >= 5);
        assert_eq!(result.params.len(), 6);
        assert!(
            (result.params[0] - 1.0).abs() < 0.2,
            "a={}",
            result.params[0]
        );
        assert!(
            (result.params[4] - 1.0).abs() < 0.2,
            "d={}",
            result.params[4]
        );
    }

    #[test]
    fn test_affine_translation() {
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                min_inliers: 5,
                ..ParallelRansacConfig::default()
            },
            ParallelModelType::Affine,
        );

        let matches = make_affine_matches(10.0, -5.0, 20);
        let result = ransac.estimate(&matches).expect("should succeed");
        assert!(
            (result.params[2] - 10.0).abs() < 2.0,
            "tx={}",
            result.params[2]
        );
        assert!(
            (result.params[5] + 5.0).abs() < 2.0,
            "ty={}",
            result.params[5]
        );
    }

    #[test]
    fn test_homography_identity() {
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                min_inliers: 5,
                ..ParallelRansacConfig::default()
            },
            ParallelModelType::Homography,
        );

        let matches = make_identity_matches(20);
        let result = ransac.estimate(&matches).expect("should succeed");
        assert!(result.num_inliers >= 5);
        assert_eq!(result.params.len(), 9);
    }

    #[test]
    fn test_affine_with_outliers() {
        let mut matches = make_affine_matches(5.0, 3.0, 30);

        // Add outliers
        for i in 0..5 {
            matches.push(MatchPair::new(
                30 + i,
                30 + i,
                100,
                Point2D::new(i as f64 * 10.0, i as f64 * 10.0),
                Point2D::new(999.0, 999.0),
            ));
        }

        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                min_inliers: 5,
                ..ParallelRansacConfig::default()
            },
            ParallelModelType::Affine,
        );

        let result = ransac.estimate(&matches).expect("should succeed");
        assert!(result.num_inliers >= 20, "inliers={}", result.num_inliers);
    }

    #[test]
    fn test_early_termination() {
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                min_inliers: 5,
                early_termination_ratio: 0.5,
                batch_size: 10,
                max_iterations: 1000,
                ..ParallelRansacConfig::default()
            },
            ParallelModelType::Affine,
        );

        let matches = make_identity_matches(20);
        let result = ransac.estimate(&matches).expect("should succeed");

        // With all-inlier data and 0.5 threshold, should terminate early
        assert!(
            result.iterations <= 1000,
            "iterations={}",
            result.iterations
        );
    }

    #[test]
    fn test_result_fields() {
        let result = ParallelRansacResult {
            params: vec![1.0; 6],
            inlier_mask: vec![true; 10],
            num_inliers: 10,
            iterations: 50,
            early_terminated: true,
        };
        assert!(result.early_terminated);
        assert_eq!(result.iterations, 50);
    }

    // -- Adaptive iteration bound tests ------------------------------------------

    /// LCG step — advances the PRNG state and returns a value in [0, 1).
    fn lcg_f64(rng: &mut u64) -> f64 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*rng >> 11) as f64) / ((u64::MAX >> 11) as f64)
    }

    /// Build matches for a known affine transform [tx, ty] with a given
    /// inlier fraction. Outliers have pseudo-random destination points that
    /// do not correspond to any coherent affine, preventing the outlier majority
    /// from forming a "better" model than the true inlier cluster.
    fn make_matches_with_outliers(
        tx: f64,
        ty: f64,
        n: usize,
        inlier_fraction: f64,
    ) -> Vec<MatchPair> {
        let n_inliers = ((n as f64 * inlier_fraction).round() as usize).max(1);
        let mut rng: u64 = 0xDEAD_BEEF_1234_5678;
        let mut matches = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f64 * 17.3) % 120.0;
            let y = (i as f64 * 29.7) % 120.0;
            let (dest_x, dest_y) = if i < n_inliers {
                (x + tx, y + ty)
            } else {
                // Random destination not consistent with any single affine,
                // so no large outlier subset forms a coherent model.
                let rx = lcg_f64(&mut rng) * 800.0;
                let ry = lcg_f64(&mut rng) * 800.0;
                (rx, ry)
            };
            matches.push(MatchPair::new(
                i,
                i,
                i as u32,
                Point2D::new(x, y),
                Point2D::new(dest_x, dest_y),
            ));
        }
        matches
    }

    /// 99% inlier ratio → adaptive bound should be very small; RANSAC terminates
    /// well before `max_iterations`.
    #[test]
    fn test_adaptive_bound_reduces_iterations_high_inlier() {
        let max_iters = 500;
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                threshold: 5.0,
                max_iterations: max_iters,
                min_inliers: 5,
                early_termination_ratio: 1.1, // disable ratio-based early term
                batch_size: 50,
                confidence: 0.99,
            },
            ParallelModelType::Affine,
        );

        // 200 matches with 99% inliers on a (tx=7, ty=3) affine
        let matches = make_matches_with_outliers(7.0, 3.0, 200, 0.99);
        let result = ransac.estimate(&matches).expect("should succeed");

        assert!(
            result.iterations < max_iters,
            "Expected adaptive termination; iterations={} max={}",
            result.iterations,
            max_iters
        );
        // Model accuracy: tx ≈ 7, ty ≈ 3 within 0.1px
        assert!(
            (result.params[2] - 7.0).abs() < 0.1,
            "tx error: params[2]={}",
            result.params[2]
        );
        assert!(
            (result.params[5] - 3.0).abs() < 0.1,
            "ty error: params[5]={}",
            result.params[5]
        );
    }

    /// 10% inlier ratio → adaptive bound is ~4600 iterations, far above
    /// `max_iterations=200`, so RANSAC must run to the max.
    #[test]
    fn test_adaptive_bound_runs_to_max_low_inlier() {
        let max_iters = 200;
        // Use large threshold so the sparse inlier cluster is genuinely found,
        // but with only 10% ratio the adaptive N stays huge (>>max_iters).
        let ransac = ParallelRansac::new(
            ParallelRansacConfig {
                threshold: 3.0,
                max_iterations: max_iters,
                min_inliers: 3,
                early_termination_ratio: 1.1, // disable ratio-based early term
                batch_size: 50,
                confidence: 0.99,
            },
            ParallelModelType::Affine,
        );

        // 200 matches with only 10% inliers on (tx=2, ty=2) — within threshold=3.0
        // Outliers are offset by 200px (well outside threshold).
        let matches = make_matches_with_outliers(2.0, 2.0, 200, 0.10);
        // For 10% inlier ratio with min_samples=3:
        // n_adaptive = ceil(log(0.01)/log(1 - 0.1^3)) ≈ ceil(4605) = 4605 >> 200
        // So RANSAC should run all the way to max_iterations.
        let result = ransac.estimate(&matches).expect("estimate ok");

        assert_eq!(
            result.iterations, max_iters,
            "Expected full run to max; iterations={} max={}",
            result.iterations, max_iters
        );
    }

    /// Lower confidence → fewer iterations needed. Same 80%-inlier data,
    /// confidence=0.95 should use strictly fewer iterations than confidence=0.999.
    #[test]
    fn test_adaptive_confidence_sensitivity() {
        let max_iters = 500;

        let make_ransac = |confidence: f64| {
            ParallelRansac::new(
                ParallelRansacConfig {
                    threshold: 5.0,
                    max_iterations: max_iters,
                    min_inliers: 5,
                    early_termination_ratio: 1.1, // disable ratio-based early term
                    batch_size: 10,
                    confidence,
                },
                ParallelModelType::Affine,
            )
        };

        let matches = make_matches_with_outliers(4.0, 2.0, 200, 0.80);

        let result_low = make_ransac(0.95).estimate(&matches).expect("low conf ok");
        let result_high = make_ransac(0.999).estimate(&matches).expect("high conf ok");

        assert!(
            result_low.iterations <= result_high.iterations,
            "confidence=0.95 should use <= iterations than confidence=0.999; \
             low={} high={}",
            result_low.iterations,
            result_high.iterations
        );
    }

    /// Model accuracy must be unchanged when adaptive bound is active vs when it
    /// is effectively disabled (max_iterations very large).
    #[test]
    fn test_model_accuracy_unchanged_with_adaptive() {
        let tx = 12.0_f64;
        let ty = -7.0_f64;
        let matches = make_matches_with_outliers(tx, ty, 200, 0.95);

        let base_ransac = ParallelRansac::new(
            ParallelRansacConfig {
                threshold: 5.0,
                max_iterations: 2000,
                min_inliers: 5,
                early_termination_ratio: 1.1, // disable ratio-based early term
                batch_size: 100,
                confidence: 0.9999, // very high confidence → adaptive bound near max
            },
            ParallelModelType::Affine,
        );

        let adaptive_ransac = ParallelRansac::new(
            ParallelRansacConfig {
                threshold: 5.0,
                max_iterations: 2000,
                min_inliers: 5,
                early_termination_ratio: 1.1,
                batch_size: 100,
                confidence: 0.99,
            },
            ParallelModelType::Affine,
        );

        let base_result = base_ransac.estimate(&matches).expect("base ok");
        let adaptive_result = adaptive_ransac.estimate(&matches).expect("adaptive ok");

        // Both should recover the affine translation within 0.5px tolerance.
        for (label, params) in [
            ("base", &base_result.params),
            ("adaptive", &adaptive_result.params),
        ] {
            assert!(
                (params[2] - tx).abs() < 0.5,
                "{label}: tx error, params[2]={}",
                params[2]
            );
            assert!(
                (params[5] - ty).abs() < 0.5,
                "{label}: ty error, params[5]={}",
                params[5]
            );
        }
    }
}
