//! Iterative Closest Point (ICP) algorithm for point cloud / contour alignment.
//!
//! Provides a centroid-based translation estimation approach to align a source
//! point set to a target point set.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A 2D point with single-precision coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// Horizontal coordinate.
    pub x: f32,
    /// Vertical coordinate.
    pub y: f32,
}

impl Point2D {
    /// Create a new 2D point.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance from this point to `other`.
    #[must_use]
    pub fn distance_to(&self, other: &Point2D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Return a new point translated by `(dx, dy)`.
    #[must_use]
    pub fn translate(&self, dx: f32, dy: f32) -> Point2D {
        Point2D {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
}

/// Find the closest point in `candidates` to `query`.
///
/// # Panics
///
/// Panics if `candidates` is empty.
#[must_use]
pub fn find_closest_point<'a>(query: &Point2D, candidates: &'a [Point2D]) -> &'a Point2D {
    candidates
        .iter()
        .min_by(|a, b| {
            query
                .distance_to(a)
                .partial_cmp(&query.distance_to(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("candidates must not be empty")
}

/// Compute the centroid (mean position) of a slice of points.
///
/// Returns `Point2D { x: 0.0, y: 0.0 }` if the slice is empty.
#[must_use]
pub fn compute_centroid(points: &[Point2D]) -> Point2D {
    if points.is_empty() {
        return Point2D::new(0.0, 0.0);
    }
    let n = points.len() as f32;
    let sum_x: f32 = points.iter().map(|p| p.x).sum();
    let sum_y: f32 = points.iter().map(|p| p.y).sum();
    Point2D::new(sum_x / n, sum_y / n)
}

/// Configuration for the ICP algorithm.
#[derive(Debug, Clone, Copy)]
pub struct IcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: u32,
    /// Stop iterating once the RMSE improvement falls below this threshold.
    pub convergence_threshold: f32,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-4,
        }
    }
}

/// Result of an ICP alignment run.
#[derive(Debug, Clone, Copy)]
pub struct IcpResult {
    /// Estimated translation `(tx, ty)` to align source onto target.
    pub translation: (f32, f32),
    /// Root-mean-square error after alignment.
    pub rmse: f32,
    /// Number of iterations performed.
    pub iterations: u32,
    /// Whether the algorithm converged within the allowed iterations.
    pub converged: bool,
}

impl IcpResult {
    /// Return `true` if the result is "good": converged and RMSE below 1.0.
    #[must_use]
    pub fn is_good(&self) -> bool {
        self.converged && self.rmse < 1.0
    }
}

/// Align `source` onto `target` using a centroid-based ICP approach.
///
/// At each iteration the algorithm:
/// 1. Finds the closest target point for every source point.
/// 2. Computes the centroid of matched source/target pairs.
/// 3. Updates the translation estimate.
/// 4. Checks for convergence.
///
/// Returns [`IcpResult`] with the best-found translation and quality metrics.
#[must_use]
pub fn icp_align(source: &[Point2D], target: &[Point2D], config: &IcpConfig) -> IcpResult {
    if source.is_empty() || target.is_empty() {
        return IcpResult {
            translation: (0.0, 0.0),
            rmse: f32::MAX,
            iterations: 0,
            converged: false,
        };
    }

    let mut tx = 0.0_f32;
    let mut ty = 0.0_f32;
    let mut prev_rmse = f32::MAX;
    let mut converged = false;
    let mut final_rmse = f32::MAX;
    let mut final_iter = 0_u32;

    for iter in 0..config.max_iterations {
        // Translate source by current estimate
        let translated: Vec<Point2D> = source.iter().map(|p| p.translate(tx, ty)).collect();

        // For each translated source point find its closest target point
        let mut sum_sq = 0.0_f32;
        let mut src_centroid_x = 0.0_f32;
        let mut src_centroid_y = 0.0_f32;
        let mut tgt_centroid_x = 0.0_f32;
        let mut tgt_centroid_y = 0.0_f32;

        for sp in &translated {
            let closest = find_closest_point(sp, target);
            let dx = closest.x - sp.x;
            let dy = closest.y - sp.y;
            sum_sq += dx * dx + dy * dy;
            src_centroid_x += sp.x;
            src_centroid_y += sp.y;
            tgt_centroid_x += closest.x;
            tgt_centroid_y += closest.y;
        }

        let n = translated.len() as f32;
        let rmse = (sum_sq / n).sqrt();

        // Update translation by the centroid difference
        tx += (tgt_centroid_x - src_centroid_x) / n;
        ty += (tgt_centroid_y - src_centroid_y) / n;

        final_rmse = rmse;
        final_iter = iter + 1;

        // Check convergence
        let improvement = (prev_rmse - rmse).abs();
        if improvement < config.convergence_threshold {
            converged = true;
            break;
        }
        prev_rmse = rmse;
    }

    IcpResult {
        translation: (tx, ty),
        rmse: final_rmse,
        iterations: final_iter,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance_to_self() {
        let p = Point2D::new(3.0, 4.0);
        assert_eq!(p.distance_to(&p), 0.0);
    }

    #[test]
    fn test_point2d_distance_to_pythagorean() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_point2d_translate() {
        let p = Point2D::new(1.0, 2.0);
        let q = p.translate(3.0, -1.0);
        assert!((q.x - 4.0).abs() < 1e-6);
        assert!((q.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_closest_point_single_candidate() {
        let query = Point2D::new(0.0, 0.0);
        let candidates = vec![Point2D::new(1.0, 1.0)];
        let closest = find_closest_point(&query, &candidates);
        assert_eq!(closest.x, 1.0);
        assert_eq!(closest.y, 1.0);
    }

    #[test]
    fn test_find_closest_point_multiple() {
        let query = Point2D::new(0.0, 0.0);
        let candidates = vec![
            Point2D::new(10.0, 10.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(5.0, 5.0),
        ];
        let closest = find_closest_point(&query, &candidates);
        assert!((closest.x - 1.0).abs() < 1e-6);
        assert!((closest.y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_centroid_empty() {
        let c = compute_centroid(&[]);
        assert_eq!(c.x, 0.0);
        assert_eq!(c.y, 0.0);
    }

    #[test]
    fn test_compute_centroid_single() {
        let pts = vec![Point2D::new(4.0, -2.0)];
        let c = compute_centroid(&pts);
        assert!((c.x - 4.0).abs() < 1e-6);
        assert!((c.y + 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_centroid_multiple() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 3.0),
        ];
        let c = compute_centroid(&pts);
        assert!((c.x - 1.0).abs() < 1e-5);
        assert!((c.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_icp_config_default() {
        let cfg = IcpConfig::default();
        assert_eq!(cfg.max_iterations, 50);
        assert!(cfg.convergence_threshold > 0.0);
    }

    #[test]
    fn test_icp_result_is_good() {
        let r = IcpResult {
            translation: (0.0, 0.0),
            rmse: 0.5,
            iterations: 3,
            converged: true,
        };
        assert!(r.is_good());
    }

    #[test]
    fn test_icp_result_not_good_high_rmse() {
        let r = IcpResult {
            translation: (0.0, 0.0),
            rmse: 5.0,
            iterations: 50,
            converged: true,
        };
        assert!(!r.is_good());
    }

    #[test]
    fn test_icp_result_not_good_not_converged() {
        let r = IcpResult {
            translation: (0.0, 0.0),
            rmse: 0.1,
            iterations: 50,
            converged: false,
        };
        assert!(!r.is_good());
    }

    #[test]
    fn test_icp_align_empty_source() {
        let target = vec![Point2D::new(1.0, 1.0)];
        let cfg = IcpConfig::default();
        let result = icp_align(&[], &target, &cfg);
        assert!(!result.converged);
    }

    #[test]
    fn test_icp_align_identical_sets() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let cfg = IcpConfig::default();
        let result = icp_align(&pts, &pts, &cfg);
        // When source == target, translation should be near zero
        assert!(result.translation.0.abs() < 0.1);
        assert!(result.translation.1.abs() < 0.1);
    }

    #[test]
    fn test_icp_align_pure_x_translation() {
        // Use a small translation (< half the point spacing) so nearest-neighbour
        // assignments are correct from the first iteration.
        let source = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(0.0, 3.0),
            Point2D::new(3.0, 3.0),
        ];
        let target = vec![
            Point2D::new(1.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(1.0, 3.0),
            Point2D::new(4.0, 3.0),
        ];
        let cfg = IcpConfig {
            max_iterations: 100,
            convergence_threshold: 1e-6,
        };
        let result = icp_align(&source, &target, &cfg);
        // The estimated tx should be approximately 1.0
        assert!((result.translation.0 - 1.0).abs() < 0.1);
        assert!(result.translation.1.abs() < 0.1);
    }
}
