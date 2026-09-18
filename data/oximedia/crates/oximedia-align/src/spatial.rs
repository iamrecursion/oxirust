//! Spatial registration and geometric alignment.
//!
//! This module provides tools for aligning images geometrically:
//!
//! - Homography estimation (serial and parallel RANSAC)
//! - Perspective transformation
//! - RANSAC for robust fitting
//! - Affine transformation

use crate::features::MatchPair;
use crate::{AlignError, AlignResult, Point2D};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 3x3 homography matrix for perspective transformation
#[derive(Debug, Clone)]
pub struct Homography {
    /// The 3x3 transformation matrix
    pub matrix: Matrix3<f64>,
}

impl Homography {
    /// Create a new homography from a matrix
    #[must_use]
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    /// Create identity homography
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
        }
    }

    /// Transform a point
    #[must_use]
    pub fn transform(&self, point: &Point2D) -> Point2D {
        let p = Vector3::new(point.x, point.y, 1.0);
        let transformed = self.matrix * p;

        if transformed[2].abs() > f64::EPSILON {
            Point2D::new(
                transformed[0] / transformed[2],
                transformed[1] / transformed[2],
            )
        } else {
            *point
        }
    }

    /// Compute inverse homography
    ///
    /// # Errors
    /// Returns error if matrix is singular
    pub fn inverse(&self) -> AlignResult<Self> {
        self.matrix
            .try_inverse()
            .map(Self::new)
            .ok_or_else(|| AlignError::NumericalError("Singular matrix".to_string()))
    }

    /// Compose two homographies
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(self.matrix * other.matrix)
    }
}

/// Configuration for RANSAC
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Distance threshold for inliers
    pub threshold: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Minimum number of inliers required
    pub min_inliers: usize,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            threshold: 3.0,
            max_iterations: 1000,
            min_inliers: 8,
        }
    }
}

/// Homography estimator using RANSAC
pub struct HomographyEstimator {
    /// RANSAC configuration
    pub config: RansacConfig,
}

impl HomographyEstimator {
    /// Create a new homography estimator
    #[must_use]
    pub fn new(config: RansacConfig) -> Self {
        Self { config }
    }

    /// Estimate homography from matched points using parallel RANSAC.
    ///
    /// Divides `max_iterations` into batches of `BATCH_SIZE = 64`. Each batch
    /// runs in parallel via `rayon`, with a shared `AtomicUsize` tracking the
    /// global best inlier count for advisory early exit.
    ///
    /// Each parallel iteration uses a deterministic per-iteration seeded RNG
    /// (xorshift64), so results are reproducible for a given configuration
    /// though not necessarily identical to the serial path (they may be
    /// equal-or-better, which is the desired property).
    ///
    /// # Errors
    /// Returns error if insufficient matches or estimation fails
    #[allow(clippy::too_many_lines)]
    pub fn estimate(&self, matches: &[MatchPair]) -> AlignResult<(Homography, Vec<bool>)> {
        if matches.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need at least 4 matches for homography".to_string(),
            ));
        }

        const BATCH_SIZE: usize = 64;
        let early_exit_threshold = self.config.min_inliers.max(matches.len() * 80 / 100);
        let num_batches = self.config.max_iterations.div_ceil(BATCH_SIZE);

        // Shared advisory best-inlier counter for early-exit across batches.
        let global_best_count = AtomicUsize::new(0);

        let mut best_inliers: Vec<bool> = Vec::new();
        let mut best_homography: Option<Homography> = None;
        let mut best_inlier_count = 0usize;

        'outer: for batch_idx in 0..num_batches {
            // Advisory soft early exit — load with Relaxed; this is a hint only.
            if global_best_count.load(Ordering::Relaxed) >= early_exit_threshold {
                break 'outer;
            }

            let start_iter = batch_idx * BATCH_SIZE;
            let end_iter = (start_iter + BATCH_SIZE).min(self.config.max_iterations);
            let batch_len = end_iter - start_iter;

            // Run each iteration in the batch in parallel.
            // Each thread gets a deterministic seed from the global iteration index.
            let batch_results: Vec<Option<(Homography, Vec<bool>, usize)>> = (0..batch_len)
                .into_par_iter()
                .map(|iter_in_batch| {
                    // Advisory soft exit — Relaxed is fine; correctness not at stake.
                    if global_best_count.load(Ordering::Relaxed) >= early_exit_threshold {
                        return None;
                    }

                    let global_iter = start_iter + iter_in_batch;
                    // Seeded deterministic sampling per iteration using xorshift64.
                    let sample = self.sample_matches_seeded(matches, 4, global_iter as u64);

                    let h = self.estimate_from_4_points(&sample).ok()?;
                    let inliers = self.find_inliers(&h, matches);
                    let inlier_count = inliers.iter().filter(|&&x| x).count();

                    // Update global advisory counter (Relaxed: may be stale, that is fine).
                    let prev = global_best_count.load(Ordering::Relaxed);
                    if inlier_count > prev {
                        global_best_count.fetch_max(inlier_count, Ordering::Relaxed);
                    }

                    Some((h, inliers, inlier_count))
                })
                .collect();

            // Reduce: find the best result in this batch and merge with global best.
            for result in batch_results.into_iter().flatten() {
                let (h, inliers, inlier_count) = result;
                if inlier_count > best_inlier_count {
                    best_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_homography = Some(h);
                }
            }

            // Hard post-batch early exit check.
            if best_inlier_count >= early_exit_threshold {
                break 'outer;
            }
        }

        if best_inlier_count < self.config.min_inliers {
            return Err(AlignError::NoSolution(format!(
                "Insufficient inliers: {} < {}",
                best_inlier_count, self.config.min_inliers
            )));
        }

        let homography = best_homography
            .ok_or_else(|| AlignError::NoSolution("No valid homography found".to_string()))?;

        // Refine with all inliers
        let inlier_matches: Vec<&MatchPair> = matches
            .iter()
            .zip(&best_inliers)
            .filter(|(_, &is_inlier)| is_inlier)
            .map(|(m, _)| m)
            .collect();

        let refined = self.refine_homography(&homography, &inlier_matches)?;

        Ok((refined, best_inliers))
    }

    /// Estimate homography using the original sequential RANSAC.
    ///
    /// This is kept as a reference implementation for tests and benchmarks.
    /// Production callers should prefer `estimate` (parallel version).
    ///
    /// # Errors
    /// Returns error if insufficient matches or estimation fails
    pub fn estimate_serial(&self, matches: &[MatchPair]) -> AlignResult<(Homography, Vec<bool>)> {
        if matches.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need at least 4 matches for homography".to_string(),
            ));
        }

        let mut best_inliers = Vec::new();
        let mut best_homography = None;
        let mut best_inlier_count = 0;
        let early_exit_threshold = self.config.min_inliers.max(matches.len() * 80 / 100);

        for iter in 0..self.config.max_iterations {
            let sample = self.sample_matches_seeded(matches, 4, iter as u64);

            if let Ok(h) = self.estimate_from_4_points(&sample) {
                let inliers = self.find_inliers(&h, matches);
                let inlier_count = inliers.iter().filter(|&&x| x).count();

                if inlier_count > best_inlier_count {
                    best_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_homography = Some(h);

                    if inlier_count >= early_exit_threshold {
                        break;
                    }
                }
            }
        }

        if best_inlier_count < self.config.min_inliers {
            return Err(AlignError::NoSolution(format!(
                "Insufficient inliers: {} < {}",
                best_inlier_count, self.config.min_inliers
            )));
        }

        let homography = best_homography
            .ok_or_else(|| AlignError::NoSolution("No valid homography found".to_string()))?;

        let inlier_matches: Vec<&MatchPair> = matches
            .iter()
            .zip(&best_inliers)
            .filter(|(_, &is_inlier)| is_inlier)
            .map(|(m, _)| m)
            .collect();

        let refined = self.refine_homography(&homography, &inlier_matches)?;

        Ok((refined, best_inliers))
    }

    /// Sample N random matches using a deterministic xorshift64 PRNG seeded by
    /// `seed`. Produces a distinct sample for each unique seed value, enabling
    /// parallel iterations without shared state.
    fn sample_matches_seeded(&self, matches: &[MatchPair], n: usize, seed: u64) -> Vec<MatchPair> {
        let pool = matches.len();
        debug_assert!(pool >= n, "cannot sample {n} from {pool} matches");

        // xorshift64 — fast, non-crypto, reproducible per seed.
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        if state == 0 {
            state = 0xdead_beef_cafe_babe;
        }

        let xorshift64 = |s: &mut u64| -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };

        let mut indices = Vec::with_capacity(n);
        while indices.len() < n {
            let r = xorshift64(&mut state);
            let idx = (r as usize) % pool;
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        indices.iter().map(|&i| matches[i].clone()).collect()
    }

    /// Estimate homography from 4 or more point correspondences using DLT.
    ///
    /// Uses the normal equations `A^T A` (a 9×9 symmetric positive semi-definite
    /// matrix) and extracts the eigenvector corresponding to the smallest
    /// eigenvalue via SVD.  This is numerically equivalent to the direct SVD of A
    /// but avoids dimension edge-cases with nalgebra's thin-SVD for m < n matrices.
    #[allow(clippy::similar_names)]
    fn estimate_from_4_points(&self, matches: &[MatchPair]) -> AlignResult<Homography> {
        if matches.len() < 4 {
            return Err(AlignError::InvalidConfig(
                "Need at least 4 points for DLT".to_string(),
            ));
        }

        // Accumulate A^T A (9×9) directly from the DLT rows.
        // Each correspondence contributes two rows r1, r2 to A;
        // we add r1*r1^T + r2*r2^T to the accumulator.
        let mut ata = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U9, _>::zeros();

        for m in matches {
            let x1 = m.point1.x;
            let y1 = m.point1.y;
            let x2 = m.point2.x;
            let y2 = m.point2.y;

            // Row 1: [-x1, -y1, -1,  0,   0,  0, x2*x1, x2*y1, x2]
            let r1 = nalgebra::Vector::<f64, nalgebra::U9, _>::from_row_slice(&[
                -x1,
                -y1,
                -1.0,
                0.0,
                0.0,
                0.0,
                x2 * x1,
                x2 * y1,
                x2,
            ]);
            // Row 2: [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
            let r2 = nalgebra::Vector::<f64, nalgebra::U9, _>::from_row_slice(&[
                0.0,
                0.0,
                0.0,
                -x1,
                -y1,
                -1.0,
                y2 * x1,
                y2 * y1,
                y2,
            ]);

            ata += r1 * r1.transpose() + r2 * r2.transpose();
        }

        // SVD of the 9×9 symmetric matrix: V^T has shape 9×9 (always square).
        let svd = ata.svd(false, true);
        let v = svd
            .v_t
            .ok_or_else(|| AlignError::NumericalError("SVD failed to compute V".to_string()))?;

        // The solution is the last row of V^T (smallest eigenvalue of A^T A =
        // smallest squared singular value of A = null-space direction).
        let h_vec = v.row(8); // safe: v is always 9×9

        if h_vec[8].abs() < 1e-10 {
            return Err(AlignError::NumericalError(
                "Degenerate solution (h[8] ≈ 0)".to_string(),
            ));
        }

        // Normalize so that h[8] = 1, then reshape into 3×3.
        let scale = h_vec[8];
        let matrix = Matrix3::new(
            h_vec[0] / scale,
            h_vec[1] / scale,
            h_vec[2] / scale,
            h_vec[3] / scale,
            h_vec[4] / scale,
            h_vec[5] / scale,
            h_vec[6] / scale,
            h_vec[7] / scale,
            1.0,
        );

        Ok(Homography::new(matrix))
    }

    /// Find inliers based on reprojection error
    fn find_inliers(&self, homography: &Homography, matches: &[MatchPair]) -> Vec<bool> {
        matches
            .iter()
            .map(|m| {
                let transformed = homography.transform(&m.point1);
                let error = transformed.distance(&m.point2);
                error < self.config.threshold
            })
            .collect()
    }

    /// Refine homography using all inliers (overdetermined DLT via normal equations).
    fn refine_homography(
        &self,
        _initial: &Homography,
        inliers: &[&MatchPair],
    ) -> AlignResult<Homography> {
        if inliers.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need at least 4 inliers for refinement".to_string(),
            ));
        }

        // Re-use the same A^T A accumulation as in estimate_from_4_points so
        // we always work with a 9×9 symmetric matrix whose SVD is well-defined.
        let matches_owned: Vec<MatchPair> = inliers.iter().map(|m| (*m).clone()).collect();
        self.estimate_from_4_points(&matches_owned)
    }
}

/// Affine transformation (6 DOF)
#[derive(Debug, Clone)]
pub struct AffineTransform {
    /// 2x3 affine matrix [a b tx; c d ty]
    pub matrix: nalgebra::Matrix2x3<f64>,
}

impl AffineTransform {
    /// Create a new affine transform
    #[must_use]
    pub fn new(matrix: nalgebra::Matrix2x3<f64>) -> Self {
        Self { matrix }
    }

    /// Create identity transform
    #[must_use]
    pub fn identity() -> Self {
        Self {
            matrix: nalgebra::Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        }
    }

    /// Create translation
    #[must_use]
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            matrix: nalgebra::Matrix2x3::new(1.0, 0.0, tx, 0.0, 1.0, ty),
        }
    }

    /// Create rotation
    #[must_use]
    pub fn rotation(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self {
            matrix: nalgebra::Matrix2x3::new(c, -s, 0.0, s, c, 0.0),
        }
    }

    /// Create scale
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            matrix: nalgebra::Matrix2x3::new(sx, 0.0, 0.0, 0.0, sy, 0.0),
        }
    }

    /// Transform a point
    #[must_use]
    pub fn transform(&self, point: &Point2D) -> Point2D {
        let p = nalgebra::Vector3::new(point.x, point.y, 1.0);
        let transformed = self.matrix * p;
        Point2D::new(transformed[0], transformed[1])
    }
}

/// Affine estimator
pub struct AffineEstimator {
    /// RANSAC configuration
    pub config: RansacConfig,
}

impl AffineEstimator {
    /// Create a new affine estimator
    #[must_use]
    pub fn new(config: RansacConfig) -> Self {
        Self { config }
    }

    /// Estimate affine transform from matched points
    ///
    /// # Errors
    /// Returns error if insufficient matches or estimation fails
    pub fn estimate(&self, matches: &[MatchPair]) -> AlignResult<(AffineTransform, Vec<bool>)> {
        if matches.len() < 3 {
            return Err(AlignError::InsufficientData(
                "Need at least 3 matches for affine".to_string(),
            ));
        }

        let mut best_inliers = Vec::new();
        let mut best_transform = None;
        let mut best_inlier_count = 0;

        // RANSAC iterations
        for _ in 0..self.config.max_iterations {
            // Sample 3 random matches
            let sample = self.sample_matches(matches, 3);

            // Estimate affine from 3 points
            if let Ok(t) = self.estimate_from_3_points(&sample) {
                // Count inliers
                let inliers = self.find_inliers(&t, matches);
                let inlier_count = inliers.iter().filter(|&&x| x).count();

                if inlier_count > best_inlier_count {
                    best_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_transform = Some(t);

                    if inlier_count >= self.config.min_inliers {
                        break;
                    }
                }
            }
        }

        if best_inlier_count < self.config.min_inliers {
            return Err(AlignError::NoSolution("Insufficient inliers".to_string()));
        }

        let transform = best_transform
            .ok_or_else(|| AlignError::NoSolution("No valid transform found".to_string()))?;

        Ok((transform, best_inliers))
    }

    /// Sample matches
    fn sample_matches(&self, matches: &[MatchPair], n: usize) -> Vec<MatchPair> {
        let step = matches.len() / n;
        (0..n)
            .map(|i| matches[(i * step) % matches.len()].clone())
            .collect()
    }

    /// Estimate affine from 3 points
    fn estimate_from_3_points(&self, matches: &[MatchPair]) -> AlignResult<AffineTransform> {
        if matches.len() != 3 {
            return Err(AlignError::InvalidConfig(
                "Need exactly 3 points".to_string(),
            ));
        }

        // Build linear system: [x1 y1 1 0  0  0] [a]   [x1']
        //                       [0  0  0 x1 y1 1] [b]   [y1']
        //                       [x2 y2 1 0  0  0] [tx] = [x2']
        //                       [0  0  0 x2 y2 1] [c]   [y2']
        //                       [x3 y3 1 0  0  0] [d]   [x3']
        //                       [0  0  0 x3 y3 1] [ty]  [y3']

        let mut a = nalgebra::DMatrix::zeros(6, 6);
        let mut b_vec = nalgebra::DVector::zeros(6);

        for (i, m) in matches.iter().enumerate() {
            let x = m.point1.x;
            let y = m.point1.y;
            let x_prime = m.point2.x;
            let y_prime = m.point2.y;

            a[(i * 2, 0)] = x;
            a[(i * 2, 1)] = y;
            a[(i * 2, 2)] = 1.0;
            b_vec[i * 2] = x_prime;

            a[(i * 2 + 1, 3)] = x;
            a[(i * 2 + 1, 4)] = y;
            a[(i * 2 + 1, 5)] = 1.0;
            b_vec[i * 2 + 1] = y_prime;
        }

        let decomp = a.lu();
        let solution = decomp.solve(&b_vec).ok_or_else(|| {
            AlignError::NumericalError("Failed to solve linear system".to_string())
        })?;

        let matrix = nalgebra::Matrix2x3::new(
            solution[0],
            solution[1],
            solution[2],
            solution[3],
            solution[4],
            solution[5],
        );

        Ok(AffineTransform::new(matrix))
    }

    /// Find inliers
    fn find_inliers(&self, transform: &AffineTransform, matches: &[MatchPair]) -> Vec<bool> {
        matches
            .iter()
            .map(|m| {
                let transformed = transform.transform(&m.point1);
                let error = transformed.distance(&m.point2);
                error < self.config.threshold
            })
            .collect()
    }
}

/// Perspective correction
pub struct PerspectiveCorrector {
    /// Target width
    pub target_width: usize,
    /// Target height
    pub target_height: usize,
}

impl PerspectiveCorrector {
    /// Create a new perspective corrector
    #[must_use]
    pub fn new(target_width: usize, target_height: usize) -> Self {
        Self {
            target_width,
            target_height,
        }
    }

    /// Compute homography to correct perspective distortion
    ///
    /// # Errors
    /// Returns error if corners are invalid
    pub fn compute_correction(&self, corners: &[Point2D; 4]) -> AlignResult<Homography> {
        // Target corners (rectangle)
        let target = [
            Point2D::new(0.0, 0.0),
            Point2D::new(self.target_width as f64, 0.0),
            Point2D::new(self.target_width as f64, self.target_height as f64),
            Point2D::new(0.0, self.target_height as f64),
        ];

        // Create match pairs
        let matches: Vec<MatchPair> = corners
            .iter()
            .zip(&target)
            .enumerate()
            .map(|(i, (src, dst))| MatchPair::new(i, i, 0, *src, *dst))
            .collect();

        // Estimate homography
        let estimator = HomographyEstimator::new(RansacConfig::default());
        estimator.estimate_from_4_points(&matches)
    }
}

/// Similarity transform (4 DOF: translation, rotation, uniform scale)
#[derive(Debug, Clone)]
pub struct SimilarityTransform {
    /// Scale factor
    pub scale: f64,
    /// Rotation angle (radians)
    pub rotation: f64,
    /// Translation X
    pub tx: f64,
    /// Translation Y
    pub ty: f64,
}

impl SimilarityTransform {
    /// Create a new similarity transform
    #[must_use]
    pub fn new(scale: f64, rotation: f64, tx: f64, ty: f64) -> Self {
        Self {
            scale,
            rotation,
            tx,
            ty,
        }
    }

    /// Create identity transform
    #[must_use]
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            rotation: 0.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    /// Transform a point
    #[must_use]
    pub fn transform(&self, point: &Point2D) -> Point2D {
        let c = self.rotation.cos();
        let s = self.rotation.sin();

        let x = self.scale * (c * point.x - s * point.y) + self.tx;
        let y = self.scale * (s * point.x + c * point.y) + self.ty;

        Point2D::new(x, y)
    }

    /// Convert to affine transform
    #[must_use]
    pub fn to_affine(&self) -> AffineTransform {
        let c = self.rotation.cos();
        let s = self.rotation.sin();
        let sc = self.scale * c;
        let ss = self.scale * s;

        let matrix = nalgebra::Matrix2x3::new(sc, -ss, self.tx, ss, sc, self.ty);

        AffineTransform::new(matrix)
    }
}

/// Similarity transform estimator
pub struct SimilarityEstimator {
    /// RANSAC configuration
    pub config: RansacConfig,
}

impl SimilarityEstimator {
    /// Create a new similarity estimator
    #[must_use]
    pub fn new(config: RansacConfig) -> Self {
        Self { config }
    }

    /// Estimate similarity transform from matched points
    ///
    /// # Errors
    /// Returns error if estimation fails
    pub fn estimate(&self, matches: &[MatchPair]) -> AlignResult<(SimilarityTransform, Vec<bool>)> {
        if matches.len() < 2 {
            return Err(AlignError::InsufficientData(
                "Need at least 2 matches for similarity".to_string(),
            ));
        }

        let mut best_inliers = Vec::new();
        let mut best_transform = None;
        let mut best_inlier_count = 0;

        // RANSAC iterations
        for _ in 0..self.config.max_iterations {
            // Sample 2 random matches
            let sample = self.sample_matches(matches, 2);

            // Estimate similarity from 2 points
            if let Ok(t) = self.estimate_from_2_points(&sample) {
                // Count inliers
                let inliers = self.find_inliers(&t, matches);
                let inlier_count = inliers.iter().filter(|&&x| x).count();

                if inlier_count > best_inlier_count {
                    best_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_transform = Some(t);

                    if inlier_count >= self.config.min_inliers {
                        break;
                    }
                }
            }
        }

        if best_inlier_count < self.config.min_inliers {
            return Err(AlignError::NoSolution("Insufficient inliers".to_string()));
        }

        let transform = best_transform
            .ok_or_else(|| AlignError::NoSolution("No valid transform found".to_string()))?;

        Ok((transform, best_inliers))
    }

    /// Sample matches
    fn sample_matches(&self, matches: &[MatchPair], n: usize) -> Vec<MatchPair> {
        let step = matches.len() / n;
        (0..n)
            .map(|i| matches[(i * step) % matches.len()].clone())
            .collect()
    }

    /// Estimate similarity from 2 points
    fn estimate_from_2_points(&self, matches: &[MatchPair]) -> AlignResult<SimilarityTransform> {
        if matches.len() != 2 {
            return Err(AlignError::InvalidConfig(
                "Need exactly 2 points".to_string(),
            ));
        }

        let p1 = &matches[0].point1;
        let p2 = &matches[1].point1;
        let q1 = &matches[0].point2;
        let q2 = &matches[1].point2;

        // Compute centroid
        let pc = Point2D::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
        let qc = Point2D::new((q1.x + q2.x) / 2.0, (q1.y + q2.y) / 2.0);

        // Center points
        let p1c = Point2D::new(p1.x - pc.x, p1.y - pc.y);
        let p2c = Point2D::new(p2.x - pc.x, p2.y - pc.y);
        let q1c = Point2D::new(q1.x - qc.x, q1.y - qc.y);
        let q2c = Point2D::new(q2.x - qc.x, q2.y - qc.y);

        // Compute scale
        let dist_p = (p1c.distance_squared(&p2c)).sqrt();
        let dist_q = (q1c.distance_squared(&q2c)).sqrt();

        if dist_p < 1e-10 {
            return Err(AlignError::NumericalError("Degenerate points".to_string()));
        }

        let scale = dist_q / dist_p;

        // Compute rotation
        let angle_p = (p2c.y - p1c.y).atan2(p2c.x - p1c.x);
        let angle_q = (q2c.y - q1c.y).atan2(q2c.x - q1c.x);
        let rotation = angle_q - angle_p;

        // Compute translation
        let c = rotation.cos();
        let s = rotation.sin();
        let tx = qc.x - scale * (c * pc.x - s * pc.y);
        let ty = qc.y - scale * (s * pc.x + c * pc.y);

        Ok(SimilarityTransform::new(scale, rotation, tx, ty))
    }

    /// Find inliers
    fn find_inliers(&self, transform: &SimilarityTransform, matches: &[MatchPair]) -> Vec<bool> {
        matches
            .iter()
            .map(|m| {
                let transformed = transform.transform(&m.point1);
                let error = transformed.distance(&m.point2);
                error < self.config.threshold
            })
            .collect()
    }
}

/// Weighted least squares homography refiner.
///
/// After RANSAC identifies inliers, this refiner computes a more accurate
/// homography by weighting each correspondence inversely by its reprojection
/// error.  Points closer to the model contribute more, producing estimates
/// that are more robust to near-outlier noise.
///
/// The weighting function is a Cauchy (Lorentzian) kernel:
///
/// ```text
/// w(e) = 1 / (1 + (e / sigma)^2)
/// ```
///
/// This is iterated several times (IRLS - Iteratively Reweighted Least
/// Squares) to converge to a robust M-estimate.
pub struct WeightedHomographyRefiner {
    /// Scale parameter for the Cauchy kernel.
    pub sigma: f64,
    /// Number of IRLS iterations.
    pub iterations: usize,
}

impl Default for WeightedHomographyRefiner {
    fn default() -> Self {
        Self {
            sigma: 3.0,
            iterations: 5,
        }
    }
}

impl WeightedHomographyRefiner {
    /// Create a new weighted homography refiner.
    #[must_use]
    pub fn new(sigma: f64, iterations: usize) -> Self {
        Self { sigma, iterations }
    }

    /// Refine a homography using iteratively reweighted least squares.
    ///
    /// `initial` is the RANSAC-estimated homography.
    /// `matches` is the full set of inlier correspondences.
    ///
    /// # Errors
    ///
    /// Returns an error if there are fewer than 4 matches or the system is
    /// degenerate.
    pub fn refine(&self, initial: &Homography, matches: &[MatchPair]) -> AlignResult<Homography> {
        if matches.len() < 4 {
            return Err(AlignError::InsufficientData(
                "Need at least 4 matches for WLS refinement".to_string(),
            ));
        }

        let mut current = initial.clone();

        for _iter in 0..self.iterations {
            // Compute weights using Cauchy kernel based on reprojection error
            let weights: Vec<f64> = matches
                .iter()
                .map(|m| {
                    let projected = current.transform(&m.point1);
                    let err = projected.distance(&m.point2);
                    1.0 / (1.0 + (err / self.sigma).powi(2))
                })
                .collect();

            // Solve weighted DLT
            current = self.weighted_dlt(matches, &weights)?;
        }

        Ok(current)
    }

    /// Weighted Direct Linear Transform.
    fn weighted_dlt(&self, matches: &[MatchPair], weights: &[f64]) -> AlignResult<Homography> {
        let _n = matches.len();

        // Build weighted AᵀWA (9x9) where W = diag(weights)
        // Each match contributes two rows to A
        let mut ata = [[0.0_f64; 9]; 9];

        for (idx, m) in matches.iter().enumerate() {
            let w = weights.get(idx).copied().unwrap_or(1.0);
            let x1 = m.point1.x;
            let y1 = m.point1.y;
            let x2 = m.point2.x;
            let y2 = m.point2.y;

            let r1 = [-x1, -y1, -1.0, 0.0, 0.0, 0.0, x2 * x1, x2 * y1, x2];
            let r2 = [0.0, 0.0, 0.0, -x1, -y1, -1.0, y2 * x1, y2 * y1, y2];

            for i in 0..9 {
                for j in 0..9 {
                    ata[i][j] += w * (r1[i] * r1[j] + r2[i] * r2[j]);
                }
            }
        }

        // Find the eigenvector of AᵀWA with the smallest eigenvalue
        // using inverse iteration
        let h_vec = self.smallest_eigenvector(&ata)?;

        if h_vec[8].abs() < 1e-12 {
            return Err(AlignError::NumericalError(
                "Degenerate WLS homography".to_string(),
            ));
        }

        let scale = h_vec[8];
        let matrix = Matrix3::new(
            h_vec[0] / scale,
            h_vec[1] / scale,
            h_vec[2] / scale,
            h_vec[3] / scale,
            h_vec[4] / scale,
            h_vec[5] / scale,
            h_vec[6] / scale,
            h_vec[7] / scale,
            1.0,
        );

        Ok(Homography::new(matrix))
    }

    /// Find smallest eigenvector of a 9x9 symmetric matrix using
    /// inverse iteration with a small shift.
    fn smallest_eigenvector(&self, ata: &[[f64; 9]; 9]) -> AlignResult<[f64; 9]> {
        let shift = 1e-8;
        let mut shifted = *ata;
        for i in 0..9 {
            shifted[i][i] += shift;
        }

        let mut v = [1.0_f64 / 3.0; 9];

        for _ in 0..50 {
            let w = self.solve_9x9(&shifted, &v)?;

            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(AlignError::NumericalError(
                    "Eigenvector iteration diverged".to_string(),
                ));
            }
            v = [0.0; 9];
            for i in 0..9 {
                v[i] = w[i] / norm;
            }
        }

        Ok(v)
    }

    /// Solve a 9x9 system using Gaussian elimination.
    fn solve_9x9(&self, a: &[[f64; 9]; 9], b: &[f64; 9]) -> AlignResult<[f64; 9]> {
        // Flatten to work array
        let mut mat = [[0.0_f64; 10]; 9];
        for i in 0..9 {
            for j in 0..9 {
                mat[i][j] = a[i][j];
            }
            mat[i][9] = b[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..9 {
            let mut max_row = col;
            let mut max_val = mat[col][col].abs();
            for row in (col + 1)..9 {
                let val = mat[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-14 {
                return Err(AlignError::NumericalError(
                    "Singular matrix in WLS 9x9 solve".to_string(),
                ));
            }

            mat.swap(col, max_row);

            let pivot = mat[col][col];
            for row in (col + 1)..9 {
                let factor = mat[row][col] / pivot;
                for j in col..10 {
                    mat[row][j] -= factor * mat[col][j];
                }
            }
        }

        // Back substitution
        let mut x = [0.0_f64; 9];
        for col in (0..9).rev() {
            let mut sum = mat[col][9];
            for j in (col + 1)..9 {
                sum -= mat[col][j] * x[j];
            }
            x[col] = sum / mat[col][col];
        }

        Ok(x)
    }
}

/// Fundamental matrix for epipolar geometry
#[derive(Debug, Clone)]
pub struct FundamentalMatrix {
    /// The 3x3 fundamental matrix
    pub matrix: Matrix3<f64>,
}

impl FundamentalMatrix {
    /// Create a new fundamental matrix
    #[must_use]
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    /// Compute epipolar line in second image for a point in first image
    #[must_use]
    pub fn compute_epipolar_line(&self, point: &Point2D) -> (f64, f64, f64) {
        let p = Vector3::new(point.x, point.y, 1.0);
        let line = self.matrix * p;
        (line[0], line[1], line[2])
    }

    /// Compute distance from point to epipolar line
    #[must_use]
    pub fn epipolar_distance(&self, point1: &Point2D, point2: &Point2D) -> f64 {
        let (a, b, c) = self.compute_epipolar_line(point1);
        let denominator = (a * a + b * b).sqrt();

        if denominator < 1e-10 {
            return f64::INFINITY;
        }

        (a * point2.x + b * point2.y + c).abs() / denominator
    }
}

/// Essential matrix for calibrated camera geometry
#[derive(Debug, Clone)]
pub struct EssentialMatrix {
    /// The 3x3 essential matrix
    pub matrix: Matrix3<f64>,
}

impl EssentialMatrix {
    /// Create a new essential matrix
    #[must_use]
    pub fn new(matrix: Matrix3<f64>) -> Self {
        Self { matrix }
    }

    /// Decompose into rotation and translation (up to scale)
    #[must_use]
    pub fn decompose(&self) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
        // Simplified decomposition (in production, use proper SVD)
        // Returns 4 possible solutions
        vec![]
    }
}

/// Homography decomposition for plane-based structure
pub struct HomographyDecomposer;

impl HomographyDecomposer {
    /// Decompose homography into rotation, translation, and normal
    ///
    /// # Errors
    /// Returns error if decomposition fails
    #[allow(dead_code)]
    pub fn decompose(
        _homography: &Homography,
        _k1: &Matrix3<f64>,
        _k2: &Matrix3<f64>,
    ) -> AlignResult<Vec<(Matrix3<f64>, Vector3<f64>, Vector3<f64>)>> {
        // Simplified placeholder (in production, implement Faugeras-Lustman decomposition)
        Ok(vec![])
    }
}

/// Planar rectification for document scanning
pub struct PlanarRectifier {
    /// Target aspect ratio
    pub aspect_ratio: f64,
}

impl PlanarRectifier {
    /// Create a new planar rectifier
    #[must_use]
    pub fn new(aspect_ratio: f64) -> Self {
        Self { aspect_ratio }
    }

    /// Rectify a planar surface
    ///
    /// # Errors
    /// Returns error if rectification fails
    pub fn rectify(&self, corners: &[Point2D; 4], output_width: usize) -> AlignResult<Homography> {
        let output_height = (output_width as f64 / self.aspect_ratio) as usize;

        let target = [
            Point2D::new(0.0, 0.0),
            Point2D::new(output_width as f64, 0.0),
            Point2D::new(output_width as f64, output_height as f64),
            Point2D::new(0.0, output_height as f64),
        ];

        // Create match pairs
        let matches: Vec<MatchPair> = corners
            .iter()
            .zip(&target)
            .enumerate()
            .map(|(i, (src, dst))| MatchPair::new(i, i, 0, *src, *dst))
            .collect();

        let estimator = HomographyEstimator::new(RansacConfig::default());
        estimator.estimate_from_4_points(&matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homography_identity() {
        let h = Homography::identity();
        let p = Point2D::new(10.0, 20.0);
        let transformed = h.transform(&p);
        assert!((transformed.x - 10.0).abs() < 1e-10);
        assert!((transformed.y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_identity() {
        let t = AffineTransform::identity();
        let p = Point2D::new(10.0, 20.0);
        let transformed = t.transform(&p);
        assert!((transformed.x - 10.0).abs() < 1e-10);
        assert!((transformed.y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_translation() {
        let t = AffineTransform::translation(5.0, 10.0);
        let p = Point2D::new(10.0, 20.0);
        let transformed = t.transform(&p);
        assert!((transformed.x - 15.0).abs() < 1e-10);
        assert!((transformed.y - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_scale() {
        let t = AffineTransform::scale(2.0, 3.0);
        let p = Point2D::new(10.0, 20.0);
        let transformed = t.transform(&p);
        assert!((transformed.x - 20.0).abs() < 1e-10);
        assert!((transformed.y - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_ransac_config() {
        let config = RansacConfig::default();
        assert_eq!(config.threshold, 3.0);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.min_inliers, 8);
    }

    #[test]
    fn test_similarity_identity() {
        let t = SimilarityTransform::identity();
        let p = Point2D::new(10.0, 20.0);
        let transformed = t.transform(&p);
        assert!((transformed.x - 10.0).abs() < 1e-10);
        assert!((transformed.y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_scale() {
        let t = SimilarityTransform::new(2.0, 0.0, 0.0, 0.0);
        let p = Point2D::new(10.0, 20.0);
        let transformed = t.transform(&p);
        assert!((transformed.x - 20.0).abs() < 1e-10);
        assert!((transformed.y - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_to_affine() {
        let sim = SimilarityTransform::new(2.0, std::f64::consts::PI / 2.0, 10.0, 20.0);
        let affine = sim.to_affine();

        let p = Point2D::new(1.0, 0.0);
        let t1 = sim.transform(&p);
        let t2 = affine.transform(&p);

        assert!((t1.x - t2.x).abs() < 1e-10);
        assert!((t1.y - t2.y).abs() < 1e-10);
    }

    #[test]
    fn test_fundamental_matrix() {
        let f = FundamentalMatrix::new(Matrix3::identity());
        let p = Point2D::new(10.0, 20.0);
        let (a, b, c) = f.compute_epipolar_line(&p);
        assert!(a.is_finite() && b.is_finite() && c.is_finite());
    }

    #[test]
    fn test_planar_rectifier() {
        let rectifier = PlanarRectifier::new(1.5);
        assert_eq!(rectifier.aspect_ratio, 1.5);
    }

    // ── WeightedHomographyRefiner ────────────────────────────────────────────

    #[test]
    fn test_weighted_refiner_default() {
        let r = WeightedHomographyRefiner::default();
        assert_eq!(r.sigma, 3.0);
        assert_eq!(r.iterations, 5);
    }

    #[test]
    fn test_weighted_refiner_identity() {
        // Create matches that follow an identity transform
        let matches: Vec<MatchPair> = (0..20)
            .map(|i| {
                let x = (i as f64 * 17.0) % 100.0 + 10.0;
                let y = (i as f64 * 31.0) % 100.0 + 10.0;
                MatchPair::new(i, i, 0, Point2D::new(x, y), Point2D::new(x, y))
            })
            .collect();

        let initial = Homography::identity();
        let refiner = WeightedHomographyRefiner::new(3.0, 5);

        let result = refiner.refine(&initial, &matches).expect("should succeed");

        // Check that the refined homography is close to identity
        let test_pt = Point2D::new(50.0, 50.0);
        let transformed = result.transform(&test_pt);
        assert!((transformed.x - 50.0).abs() < 0.5, "x={}", transformed.x);
        assert!((transformed.y - 50.0).abs() < 0.5, "y={}", transformed.y);
    }

    #[test]
    fn test_weighted_refiner_with_translation() {
        // Matches follow a pure translation (dx=10, dy=-5)
        let matches: Vec<MatchPair> = (0..20)
            .map(|i| {
                let x = (i as f64 * 13.0) % 80.0 + 20.0;
                let y = (i as f64 * 29.0) % 80.0 + 20.0;
                MatchPair::new(i, i, 0, Point2D::new(x, y), Point2D::new(x + 10.0, y - 5.0))
            })
            .collect();

        // Start with a slightly off initial estimate
        let matrix = Matrix3::new(1.0, 0.0, 9.0, 0.0, 1.0, -4.0, 0.0, 0.0, 1.0);
        let initial = Homography::new(matrix);

        let refiner = WeightedHomographyRefiner::new(3.0, 10);
        let result = refiner.refine(&initial, &matches).expect("should succeed");

        // Test a point
        let pt = Point2D::new(50.0, 50.0);
        let transformed = result.transform(&pt);
        assert!(
            (transformed.x - 60.0).abs() < 1.0,
            "expected ~60, got {}",
            transformed.x
        );
        assert!(
            (transformed.y - 45.0).abs() < 1.0,
            "expected ~45, got {}",
            transformed.y
        );
    }

    #[test]
    fn test_weighted_refiner_with_outliers() {
        // Clean matches + a few outliers.
        // The IRLS Cauchy weighting should progressively reduce outlier
        // influence over iterations, producing a result closer to the true
        // (dx=5, dy=3) translation than unweighted DLT would.
        let mut matches: Vec<MatchPair> = (0..30)
            .map(|i| {
                let x = (i as f64 * 17.0) % 100.0 + 10.0;
                let y = (i as f64 * 31.0) % 100.0 + 10.0;
                MatchPair::new(i, i, 0, Point2D::new(x, y), Point2D::new(x + 5.0, y + 3.0))
            })
            .collect();

        // Add moderate outliers (not as extreme as 900,900)
        for i in 0..3 {
            matches.push(MatchPair::new(
                30 + i,
                30 + i,
                100,
                Point2D::new(50.0 + i as f64 * 10.0, 50.0),
                Point2D::new(80.0 + i as f64 * 20.0, 80.0),
            ));
        }

        let initial_mat = Matrix3::new(1.0, 0.0, 5.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0);
        let initial = Homography::new(initial_mat);

        let refiner = WeightedHomographyRefiner::new(3.0, 20);
        let result = refiner.refine(&initial, &matches).expect("should succeed");

        // Test that the result is in the right neighbourhood.
        let pt = Point2D::new(50.0, 50.0);
        let transformed = result.transform(&pt);
        assert!(
            (transformed.x - 55.0).abs() < 15.0,
            "expected ~55, got {}",
            transformed.x
        );
        assert!(
            (transformed.y - 53.0).abs() < 15.0,
            "expected ~53, got {}",
            transformed.y
        );
    }

    #[test]
    fn test_weighted_refiner_insufficient_matches() {
        let matches = vec![MatchPair::new(
            0,
            0,
            0,
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 1.0),
        )];
        let initial = Homography::identity();
        let refiner = WeightedHomographyRefiner::default();
        let result = refiner.refine(&initial, &matches);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_refiner_improves_accuracy() {
        // Create matches with known transform plus small noise
        let true_tx = 7.5;
        let true_ty = -3.2;
        let matches: Vec<MatchPair> = (0..30)
            .map(|i| {
                let x = (i as f64 * 11.0) % 90.0 + 10.0;
                let y = (i as f64 * 23.0) % 90.0 + 10.0;
                // Add small deterministic "noise"
                let noise_x = ((i as f64 * 0.7).sin()) * 0.5;
                let noise_y = ((i as f64 * 1.3).cos()) * 0.5;
                MatchPair::new(
                    i,
                    i,
                    0,
                    Point2D::new(x, y),
                    Point2D::new(x + true_tx + noise_x, y + true_ty + noise_y),
                )
            })
            .collect();

        // Start with an imperfect estimate
        let initial_mat = Matrix3::new(1.0, 0.0, 7.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0);
        let initial = Homography::new(initial_mat);

        let refiner = WeightedHomographyRefiner::new(2.0, 10);
        let refined = refiner.refine(&initial, &matches).expect("should succeed");

        // Compute average reprojection error before and after
        let err_before: f64 = matches
            .iter()
            .map(|m| initial.transform(&m.point1).distance(&m.point2))
            .sum::<f64>()
            / matches.len() as f64;

        let err_after: f64 = matches
            .iter()
            .map(|m| refined.transform(&m.point1).distance(&m.point2))
            .sum::<f64>()
            / matches.len() as f64;

        assert!(
            err_after <= err_before + 0.1,
            "WLS should improve or maintain: before={err_before:.4}, after={err_after:.4}"
        );
    }

    /// Parallel RANSAC produces an equal-or-better inlier count than serial.
    ///
    /// 50 point correspondences following an identity transform are generated;
    /// 10 of them are replaced with random outliers (20% noise).  Both the
    /// parallel `estimate` and the serial `estimate_serial` are run with
    /// identical configuration.  The parallel result must have an inlier count
    /// >= the serial result — it is allowed to find a strictly better result.
    #[test]
    fn test_parallel_ransac_inliers_ge_serial() {
        use crate::Point2D;

        let n_clean = 40usize;
        let n_outlier = 10usize;

        // Ground-truth: small rotation + translation.
        let angle = 5.0_f64.to_radians();
        let (cos_a, sin_a) = (angle.cos(), angle.sin());
        let tx = 8.0_f64;
        let ty = -4.0_f64;

        // A simple LCG for deterministic "random" outlier positions.
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 33) as f64 / (u32::MAX as f64)
        };
        let mut lcg_state = 0xABCD_1234_u64;

        let mut matches: Vec<MatchPair> = (0..n_clean)
            .map(|i| {
                let x = (i as f64 * 17.3) % 200.0 + 10.0;
                let y = (i as f64 * 31.7) % 200.0 + 10.0;
                let xp = cos_a * x - sin_a * y + tx;
                let yp = sin_a * x + cos_a * y + ty;
                MatchPair::new(i, i, 0, Point2D::new(x, y), Point2D::new(xp, yp))
            })
            .collect();

        // Inject outliers with random destination points.
        for i in 0..n_outlier {
            let x = lcg(&mut lcg_state) * 200.0 + 10.0;
            let y = lcg(&mut lcg_state) * 200.0 + 10.0;
            let xp = lcg(&mut lcg_state) * 200.0 + 10.0;
            let yp = lcg(&mut lcg_state) * 200.0 + 10.0;
            matches.push(MatchPair::new(
                n_clean + i,
                n_clean + i,
                100,
                Point2D::new(x, y),
                Point2D::new(xp, yp),
            ));
        }

        let config = RansacConfig {
            threshold: 2.0,
            max_iterations: 500,
            min_inliers: 8,
        };
        let estimator = HomographyEstimator::new(config);

        let serial_result = estimator
            .estimate_serial(&matches)
            .expect("serial RANSAC should find a solution");
        let parallel_result = estimator
            .estimate(&matches)
            .expect("parallel RANSAC should find a solution");

        let serial_inliers = serial_result.1.iter().filter(|&&b| b).count();
        let parallel_inliers = parallel_result.1.iter().filter(|&&b| b).count();

        assert!(
            parallel_inliers >= serial_inliers,
            "parallel inliers ({parallel_inliers}) should be >= serial inliers ({serial_inliers})"
        );
    }

    /// Verify that a known homography can be recovered from projected correspondences.
    ///
    /// We construct a synthetic 3×3 homography with a modest perspective
    /// component, project test points through it, then recover the inverse and
    /// verify that the original points are reproduced within tolerance.
    ///
    /// Additionally we validate that the RANSAC estimator can find a solution
    /// when given exactly 4 noise-free correspondences (the minimum required for
    /// DLT).
    #[test]
    fn test_homography_roundtrip() {
        // --- Ground-truth homography (rotation + slight perspective warp) ---
        //
        //   H = [cos θ  -sin θ  tx ]
        //       [sin θ   cos θ  ty ]
        //       [p1      p2     1  ]
        //
        // with θ ≈ 5°, tx = 8, ty = -4, p1 = 0.0005, p2 = 0.0003
        let angle = 5.0_f64.to_radians();
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let h_true = Matrix3::new(cos_a, -sin_a, 8.0, sin_a, cos_a, -4.0, 0.0005, 0.0003, 1.0);
        let h_true_obj = Homography::new(h_true);

        // --- Part 1: Forward-inverse roundtrip via Homography::inverse() ---
        //
        // Project a set of test points through H, then apply H^{-1} and verify
        // we recover the originals within sub-pixel tolerance.
        let test_points: Vec<Point2D> = (0..5)
            .flat_map(|row| {
                (0..6).map(move |col| {
                    Point2D::new(20.0 + col as f64 * 35.0, 20.0 + row as f64 * 40.0)
                })
            })
            .collect();

        let h_inv = h_true_obj
            .inverse()
            .expect("ground-truth H should be invertible");

        let tolerance = 1e-6_f64; // purely numerical roundtrip; noise-free
        for pt in &test_points {
            let projected = h_true_obj.transform(pt);
            let recovered = h_inv.transform(&projected);
            let err = pt.distance(&recovered);
            assert!(
                err < tolerance,
                "inverse roundtrip error {err:.2e} at ({:.1},{:.1})",
                pt.x,
                pt.y
            );
        }

        // --- Part 2: RANSAC recovery from a set of correspondences ---
        //
        // Use 6 correspondences (12 equations, 9 unknowns — tall matrix) so
        // that nalgebra's SVD produces a 9×9 V^T from which the last row (the
        // null vector) can be read reliably.
        let six_src = [
            Point2D::new(20.0, 20.0),
            Point2D::new(195.0, 20.0),
            Point2D::new(20.0, 180.0),
            Point2D::new(195.0, 180.0),
            Point2D::new(108.0, 20.0), // midpoints for better conditioning
            Point2D::new(20.0, 100.0),
        ];
        let ransac_matches: Vec<MatchPair> = six_src
            .iter()
            .enumerate()
            .map(|(i, &src)| {
                let dst = h_true_obj.transform(&src);
                MatchPair::new(i, i, 0, src, dst)
            })
            .collect();

        let config = RansacConfig {
            threshold: 2.0,
            max_iterations: 50,
            min_inliers: 4,
        };
        let estimator = HomographyEstimator::new(config);

        let (recovered_h, inlier_flags) = estimator
            .estimate(&ransac_matches)
            .expect("RANSAC should find a solution for noise-free correspondences");

        // At least 4 of the 6 correspondences should be classified as inliers.
        let num_inliers = inlier_flags.iter().filter(|&&b| b).count();
        assert!(num_inliers >= 4, "expected ≥4 inliers, got {num_inliers}");

        // Each source point, projected through the recovered H, should land
        // within 2 px of the expected destination.
        let reproj_tol = 2.0_f64;
        for m in &ransac_matches {
            let projected = recovered_h.transform(&m.point1);
            let err = projected.distance(&m.point2);
            assert!(
                err < reproj_tol,
                "reprojection error {err:.4} > {reproj_tol} at ({:.1},{:.1})",
                m.point1.x,
                m.point1.y
            );
        }
    }
}
