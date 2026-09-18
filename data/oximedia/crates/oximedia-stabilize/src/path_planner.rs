#![allow(dead_code)]
//! Camera path planning and trajectory optimization for video stabilization.
//!
//! This module provides algorithms to plan optimal camera motion paths that
//! balance smoothness with content preservation. It supports constraint-based
//! planning that keeps subjects in frame while removing unwanted shake.

use std::collections::VecDeque;

/// Strategy for path planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStrategy {
    /// Minimize total path length (smoothest result).
    MinLength,
    /// Minimize acceleration changes (jerk minimization).
    MinJerk,
    /// Keep region of interest centered.
    RoiCenter,
    /// Blend between smoothness and original path.
    Hybrid,
}

/// A 2D point used in path planning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point.
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Compute Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Linear interpolation toward another point.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// A single waypoint along the planned path.
#[derive(Debug, Clone)]
pub struct Waypoint {
    /// Position at this waypoint.
    pub position: Point2D,
    /// Velocity (pixels per frame).
    pub velocity: Point2D,
    /// Frame index.
    pub frame_index: usize,
    /// Confidence score (0.0–1.0).
    pub confidence: f64,
}

/// Constraint that restricts the planned path.
#[derive(Debug, Clone)]
pub struct PathConstraint {
    /// Frame index where the constraint applies.
    pub frame_index: usize,
    /// Maximum allowed deviation from the original position (pixels).
    pub max_deviation: f64,
    /// Region of interest center that should stay in frame.
    pub roi_center: Option<Point2D>,
}

/// Per-axis motion stabilization constraints for user-defined lock/allow policy.
///
/// These constraints let the user specify which motion axes should be smoothed
/// and which should be passed through without correction.  For example, a
/// broadcaster may want to lock pan (X-axis) while allowing intentional tilt
/// (Y-axis) adjustments to remain visible in the output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionAxisConstraints {
    /// When `true`, horizontal (pan) motion is fully locked to the smoothed
    /// trajectory — no lateral jitter is passed through.
    /// When `false`, the X component is smoothed normally.
    pub lock_pan: bool,
    /// When `true`, vertical (tilt) motion is fully locked.
    /// When `false`, the Y component is smoothed normally.
    pub lock_tilt: bool,
    /// Blending factor applied to the *unlocked* axis (0.0 = original,
    /// 1.0 = fully smoothed).  Locked axes always use 1.0.
    pub unlock_blend: f64,
    /// Maximum allowed correction on the pan axis (pixels).
    /// Ignored when `lock_pan` is true (full lock always applies).
    pub pan_max_correction: f64,
    /// Maximum allowed correction on the tilt axis (pixels).
    pub tilt_max_correction: f64,
}

impl Default for MotionAxisConstraints {
    fn default() -> Self {
        Self {
            lock_pan: false,
            lock_tilt: false,
            unlock_blend: 0.8,
            pan_max_correction: f64::MAX,
            tilt_max_correction: f64::MAX,
        }
    }
}

impl MotionAxisConstraints {
    /// Create constraints with both axes freely smoothed (no lock).
    #[must_use]
    pub fn unconstrained() -> Self {
        Self::default()
    }

    /// Create constraints that lock the pan (X) axis while allowing tilt (Y).
    #[must_use]
    pub fn lock_pan_only() -> Self {
        Self {
            lock_pan: true,
            lock_tilt: false,
            ..Self::default()
        }
    }

    /// Create constraints that lock the tilt (Y) axis while allowing pan (X).
    #[must_use]
    pub fn lock_tilt_only() -> Self {
        Self {
            lock_pan: false,
            lock_tilt: true,
            ..Self::default()
        }
    }

    /// Lock both axes (full 2-D lock).
    #[must_use]
    pub fn lock_both() -> Self {
        Self {
            lock_pan: true,
            lock_tilt: true,
            ..Self::default()
        }
    }

    /// Set the blending factor for unlocked axes.
    #[must_use]
    pub fn with_unlock_blend(mut self, blend: f64) -> Self {
        self.unlock_blend = blend.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum pan correction in pixels.
    #[must_use]
    pub fn with_pan_max_correction(mut self, px: f64) -> Self {
        self.pan_max_correction = px.max(0.0);
        self
    }

    /// Set the maximum tilt correction in pixels.
    #[must_use]
    pub fn with_tilt_max_correction(mut self, px: f64) -> Self {
        self.tilt_max_correction = px.max(0.0);
        self
    }

    /// Apply these axis constraints to a pair of (original, smoothed) positions
    /// and return the constrained smoothed position.
    ///
    /// The returned point is what will be used as the stabilized output position
    /// for this frame.
    #[must_use]
    pub fn apply(&self, original: Point2D, smoothed: Point2D) -> Point2D {
        let x = if self.lock_pan {
            // Full lock: use the smoothed X directly, but still clamp correction
            let raw_correction = smoothed.x - original.x;
            let clamped = raw_correction.clamp(-self.pan_max_correction, self.pan_max_correction);
            original.x + clamped
        } else {
            // Partial blend on X
            let blended = original.x + (smoothed.x - original.x) * self.unlock_blend;
            let correction = blended - original.x;
            let clamped = correction.clamp(-self.pan_max_correction, self.pan_max_correction);
            original.x + clamped
        };

        let y = if self.lock_tilt {
            let raw_correction = smoothed.y - original.y;
            let clamped = raw_correction.clamp(-self.tilt_max_correction, self.tilt_max_correction);
            original.y + clamped
        } else {
            let blended = original.y + (smoothed.y - original.y) * self.unlock_blend;
            let correction = blended - original.y;
            let clamped = correction.clamp(-self.tilt_max_correction, self.tilt_max_correction);
            original.y + clamped
        };

        Point2D::new(x, y)
    }
}

/// Result of path planning.
#[derive(Debug, Clone)]
pub struct PlannedPath {
    /// Ordered waypoints describing the smoothed camera path.
    pub waypoints: Vec<Waypoint>,
    /// Total path length in pixels.
    pub total_length: f64,
    /// Average smoothness score (lower is smoother).
    pub smoothness_score: f64,
    /// Maximum deviation from the original path.
    pub max_deviation: f64,
}

/// Camera path planner that computes optimal stabilization trajectories.
#[derive(Debug)]
pub struct PathPlanner {
    /// Planning strategy.
    strategy: PlanStrategy,
    /// Smoothing window size.
    window_size: usize,
    /// Maximum allowed deviation from original path (pixels).
    max_deviation: f64,
    /// Blend factor for hybrid mode (0.0 = original, 1.0 = fully smoothed).
    blend_factor: f64,
}

impl PathPlanner {
    /// Create a new path planner with the given strategy.
    #[must_use]
    pub fn new(strategy: PlanStrategy) -> Self {
        Self {
            strategy,
            window_size: 30,
            max_deviation: 50.0,
            blend_factor: 0.8,
        }
    }

    /// Set the smoothing window size.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(3);
        self
    }

    /// Set the maximum allowed deviation.
    #[must_use]
    pub fn with_max_deviation(mut self, deviation: f64) -> Self {
        self.max_deviation = deviation.max(0.0);
        self
    }

    /// Set the blend factor for hybrid mode.
    #[must_use]
    pub fn with_blend_factor(mut self, factor: f64) -> Self {
        self.blend_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Plan an optimal path from the given raw camera positions.
    #[must_use]
    pub fn plan(&self, positions: &[Point2D]) -> PlannedPath {
        if positions.is_empty() {
            return PlannedPath {
                waypoints: Vec::new(),
                total_length: 0.0,
                smoothness_score: 0.0,
                max_deviation: 0.0,
            };
        }
        if positions.len() == 1 {
            return PlannedPath {
                waypoints: vec![Waypoint {
                    position: positions[0],
                    velocity: Point2D::new(0.0, 0.0),
                    frame_index: 0,
                    confidence: 1.0,
                }],
                total_length: 0.0,
                smoothness_score: 0.0,
                max_deviation: 0.0,
            };
        }

        let smoothed = match self.strategy {
            PlanStrategy::MinLength => self.smooth_min_length(positions),
            PlanStrategy::MinJerk => self.smooth_min_jerk(positions),
            PlanStrategy::RoiCenter => self.smooth_roi_center(positions),
            PlanStrategy::Hybrid => self.smooth_hybrid(positions),
        };

        self.build_planned_path(positions, &smoothed)
    }

    /// Plan a path with per-axis motion constraints (lock pan / allow tilt, etc.)
    ///
    /// This is the primary entry point for user-defined stabilization policies.
    /// The `axis` constraints control how much freedom the stabilized output has
    /// on each axis independently from the global `max_deviation` clamping.
    ///
    /// # Example — lock horizontal pan while allowing tilt
    ///
    /// ```
    /// use oximedia_stabilize::path_planner::{
    ///     PathPlanner, PlanStrategy, Point2D, MotionAxisConstraints,
    /// };
    ///
    /// let planner = PathPlanner::new(PlanStrategy::MinLength);
    /// let positions: Vec<Point2D> = (0..20)
    ///     .map(|i| Point2D::new(i as f64 * 5.0 + (i as f64 * 0.3).sin() * 8.0, 100.0))
    ///     .collect();
    /// let constraints = MotionAxisConstraints::lock_pan_only();
    /// let path = planner.plan_with_axis_constraints(&positions, constraints);
    /// assert_eq!(path.waypoints.len(), 20);
    /// ```
    #[must_use]
    pub fn plan_with_axis_constraints(
        &self,
        positions: &[Point2D],
        axis: MotionAxisConstraints,
    ) -> PlannedPath {
        if positions.is_empty() {
            return PlannedPath {
                waypoints: Vec::new(),
                total_length: 0.0,
                smoothness_score: 0.0,
                max_deviation: 0.0,
            };
        }
        if positions.len() == 1 {
            return PlannedPath {
                waypoints: vec![Waypoint {
                    position: positions[0],
                    velocity: Point2D::new(0.0, 0.0),
                    frame_index: 0,
                    confidence: 1.0,
                }],
                total_length: 0.0,
                smoothness_score: 0.0,
                max_deviation: 0.0,
            };
        }

        // Smooth each axis using the base strategy
        let smoothed_unconstrained = match self.strategy {
            PlanStrategy::MinLength => self.smooth_min_length(positions),
            PlanStrategy::MinJerk => self.smooth_min_jerk(positions),
            PlanStrategy::RoiCenter => self.smooth_roi_center(positions),
            PlanStrategy::Hybrid => self.smooth_hybrid(positions),
        };

        // Apply per-axis constraints to each smoothed position
        let constrained: Vec<Point2D> = positions
            .iter()
            .zip(smoothed_unconstrained.iter())
            .map(|(&orig, &sm)| axis.apply(orig, sm))
            .collect();

        self.build_planned_path(positions, &constrained)
    }

    /// Plan a path with constraints.
    #[must_use]
    pub fn plan_with_constraints(
        &self,
        positions: &[Point2D],
        constraints: &[PathConstraint],
    ) -> PlannedPath {
        if positions.is_empty() {
            return PlannedPath {
                waypoints: Vec::new(),
                total_length: 0.0,
                smoothness_score: 0.0,
                max_deviation: 0.0,
            };
        }

        let mut smoothed = match self.strategy {
            PlanStrategy::MinLength => self.smooth_min_length(positions),
            PlanStrategy::MinJerk => self.smooth_min_jerk(positions),
            PlanStrategy::RoiCenter => self.smooth_roi_center(positions),
            PlanStrategy::Hybrid => self.smooth_hybrid(positions),
        };

        // Apply constraints
        for constraint in constraints {
            if constraint.frame_index < smoothed.len() {
                let orig = positions[constraint.frame_index];
                let dev = smoothed[constraint.frame_index].distance_to(&orig);
                if dev > constraint.max_deviation {
                    let ratio = constraint.max_deviation / dev;
                    smoothed[constraint.frame_index] =
                        orig.lerp(&smoothed[constraint.frame_index], ratio);
                }
                if let Some(roi) = &constraint.roi_center {
                    let offset_x = roi.x - smoothed[constraint.frame_index].x;
                    let offset_y = roi.y - smoothed[constraint.frame_index].y;
                    smoothed[constraint.frame_index].x += offset_x * 0.3;
                    smoothed[constraint.frame_index].y += offset_y * 0.3;
                }
            }
        }

        self.build_planned_path(positions, &smoothed)
    }

    /// Moving-average smoothing (min-length strategy).
    fn smooth_min_length(&self, positions: &[Point2D]) -> Vec<Point2D> {
        let mut result = Vec::with_capacity(positions.len());
        let half = self.window_size / 2;
        for i in 0..positions.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(positions.len());
            let count = (end - start) as f64;
            let sum_x: f64 = positions[start..end].iter().map(|p| p.x).sum();
            let sum_y: f64 = positions[start..end].iter().map(|p| p.y).sum();
            result.push(Point2D::new(sum_x / count, sum_y / count));
        }
        self.clamp_deviation(positions, &result)
    }

    /// Jerk-minimizing smoothing using a double-pass Gaussian-like filter.
    fn smooth_min_jerk(&self, positions: &[Point2D]) -> Vec<Point2D> {
        // Forward pass
        let mut forward = Vec::with_capacity(positions.len());
        let alpha = 2.0 / (self.window_size as f64 + 1.0);
        let mut sx = positions[0].x;
        let mut sy = positions[0].y;
        for p in positions {
            sx = alpha * p.x + (1.0 - alpha) * sx;
            sy = alpha * p.y + (1.0 - alpha) * sy;
            forward.push(Point2D::new(sx, sy));
        }
        // Backward pass
        let mut backward = vec![Point2D::new(0.0, 0.0); positions.len()];
        sx = positions.last().map_or(0.0, |p| p.x);
        sy = positions.last().map_or(0.0, |p| p.y);
        for i in (0..positions.len()).rev() {
            sx = alpha * positions[i].x + (1.0 - alpha) * sx;
            sy = alpha * positions[i].y + (1.0 - alpha) * sy;
            backward[i] = Point2D::new(sx, sy);
        }
        // Average forward and backward
        let averaged: Vec<Point2D> = forward
            .iter()
            .zip(backward.iter())
            .map(|(f, b)| Point2D::new((f.x + b.x) * 0.5, (f.y + b.y) * 0.5))
            .collect();
        self.clamp_deviation(positions, &averaged)
    }

    /// ROI-centered smoothing that biases path toward centers.
    fn smooth_roi_center(&self, positions: &[Point2D]) -> Vec<Point2D> {
        // Use a simple centroid-biased moving average
        let centroid_x: f64 = positions.iter().map(|p| p.x).sum::<f64>() / positions.len() as f64;
        let centroid_y: f64 = positions.iter().map(|p| p.y).sum::<f64>() / positions.len() as f64;
        let centroid = Point2D::new(centroid_x, centroid_y);

        let smoothed: Vec<Point2D> = positions.iter().map(|p| p.lerp(&centroid, 0.3)).collect();
        self.clamp_deviation(positions, &smoothed)
    }

    /// Hybrid smoothing that blends original and smoothed paths.
    fn smooth_hybrid(&self, positions: &[Point2D]) -> Vec<Point2D> {
        let smoothed = self.smooth_min_length(positions);
        positions
            .iter()
            .zip(smoothed.iter())
            .map(|(orig, sm)| orig.lerp(sm, self.blend_factor))
            .collect()
    }

    /// Clamp smoothed positions so they don't deviate beyond `max_deviation`.
    fn clamp_deviation(&self, originals: &[Point2D], smoothed: &[Point2D]) -> Vec<Point2D> {
        originals
            .iter()
            .zip(smoothed.iter())
            .map(|(orig, sm)| {
                let dev = orig.distance_to(sm);
                if dev > self.max_deviation {
                    orig.lerp(sm, self.max_deviation / dev)
                } else {
                    *sm
                }
            })
            .collect()
    }

    /// Build the final `PlannedPath` from original and smoothed positions.
    fn build_planned_path(&self, originals: &[Point2D], smoothed: &[Point2D]) -> PlannedPath {
        let mut waypoints = Vec::with_capacity(smoothed.len());
        let mut total_length = 0.0;
        let mut max_dev = 0.0_f64;

        for (i, pos) in smoothed.iter().enumerate() {
            let velocity = if i + 1 < smoothed.len() {
                Point2D::new(smoothed[i + 1].x - pos.x, smoothed[i + 1].y - pos.y)
            } else {
                Point2D::new(0.0, 0.0)
            };

            if i > 0 {
                total_length += smoothed[i - 1].distance_to(pos);
            }

            let dev = originals[i].distance_to(pos);
            max_dev = max_dev.max(dev);

            waypoints.push(Waypoint {
                position: *pos,
                velocity,
                frame_index: i,
                confidence: 1.0 - (dev / self.max_deviation).min(1.0),
            });
        }

        // Compute smoothness as average acceleration magnitude
        let smoothness_score = compute_smoothness(&waypoints);

        PlannedPath {
            waypoints,
            total_length,
            smoothness_score,
            max_deviation: max_dev,
        }
    }
}

/// Compute a smoothness score from waypoint velocities (lower = smoother).
fn compute_smoothness(waypoints: &[Waypoint]) -> f64 {
    if waypoints.len() < 3 {
        return 0.0;
    }
    let mut total_accel = 0.0;
    for i in 1..waypoints.len() {
        let ax = waypoints[i].velocity.x - waypoints[i - 1].velocity.x;
        let ay = waypoints[i].velocity.y - waypoints[i - 1].velocity.y;
        total_accel += (ax * ax + ay * ay).sqrt();
    }
    total_accel / (waypoints.len() - 1) as f64
}

/// A ring-buffer based real-time path smoother for live stabilization.
#[derive(Debug)]
pub struct RealtimeSmoother {
    /// Internal ring buffer of recent positions.
    buffer: VecDeque<Point2D>,
    /// Maximum buffer capacity.
    capacity: usize,
    /// Exponential smoothing alpha.
    alpha: f64,
    /// Last smoothed position.
    last_smoothed: Option<Point2D>,
}

impl RealtimeSmoother {
    /// Create a new real-time smoother with the given window capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            alpha: 0.15,
            last_smoothed: None,
        }
    }

    /// Set the exponential smoothing alpha (0.0 = very smooth, 1.0 = no smoothing).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Push a new raw position and get the smoothed output.
    pub fn push(&mut self, position: Point2D) -> Point2D {
        self.buffer.push_back(position);
        if self.buffer.len() > self.capacity {
            self.buffer.pop_front();
        }

        let smoothed = match self.last_smoothed {
            Some(prev) => Point2D::new(
                self.alpha * position.x + (1.0 - self.alpha) * prev.x,
                self.alpha * position.y + (1.0 - self.alpha) * prev.y,
            ),
            None => position,
        };
        self.last_smoothed = Some(smoothed);
        smoothed
    }

    /// Reset the smoother state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.last_smoothed = None;
    }

    /// Get the number of samples currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_lerp() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-10);
        assert!((mid.y - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_lerp_clamp() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 10.0);
        let over = a.lerp(&b, 2.0);
        assert!((over.x - 10.0).abs() < 1e-10);
        let under = a.lerp(&b, -1.0);
        assert!((under.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_empty() {
        let planner = PathPlanner::new(PlanStrategy::MinLength);
        let path = planner.plan(&[]);
        assert!(path.waypoints.is_empty());
        assert!((path.total_length - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_single_point() {
        let planner = PathPlanner::new(PlanStrategy::MinLength);
        let path = planner.plan(&[Point2D::new(5.0, 10.0)]);
        assert_eq!(path.waypoints.len(), 1);
        assert!((path.waypoints[0].position.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_plan_min_length() {
        let planner = PathPlanner::new(PlanStrategy::MinLength).with_window_size(5);
        let positions: Vec<Point2D> = (0..20)
            .map(|i| {
                let noise = if i % 2 == 0 { 5.0 } else { -5.0 };
                Point2D::new(i as f64 * 10.0 + noise, 100.0 + noise)
            })
            .collect();
        let path = planner.plan(&positions);
        assert_eq!(path.waypoints.len(), 20);
        assert!(path.smoothness_score < 20.0);
    }

    #[test]
    fn test_plan_min_jerk() {
        let planner = PathPlanner::new(PlanStrategy::MinJerk).with_window_size(10);
        let positions: Vec<Point2D> = (0..30)
            .map(|i| Point2D::new(i as f64 * 5.0, (i as f64 * 0.3).sin() * 20.0))
            .collect();
        let path = planner.plan(&positions);
        assert_eq!(path.waypoints.len(), 30);
        assert!(path.total_length > 0.0);
    }

    #[test]
    fn test_plan_roi_center() {
        let planner = PathPlanner::new(PlanStrategy::RoiCenter);
        let positions: Vec<Point2D> = (0..15)
            .map(|i| Point2D::new(100.0 + i as f64, 200.0 - i as f64))
            .collect();
        let path = planner.plan(&positions);
        assert_eq!(path.waypoints.len(), 15);
    }

    #[test]
    fn test_plan_hybrid() {
        let planner = PathPlanner::new(PlanStrategy::Hybrid).with_blend_factor(0.5);
        let positions: Vec<Point2D> = (0..10)
            .map(|i| Point2D::new(i as f64 * 10.0, i as f64 * 10.0))
            .collect();
        let path = planner.plan(&positions);
        assert_eq!(path.waypoints.len(), 10);
    }

    #[test]
    fn test_plan_with_constraints() {
        let planner = PathPlanner::new(PlanStrategy::MinLength).with_max_deviation(10.0);
        let positions: Vec<Point2D> = (0..20)
            .map(|i| Point2D::new(i as f64 * 5.0, 50.0))
            .collect();
        let constraints = vec![PathConstraint {
            frame_index: 10,
            max_deviation: 3.0,
            roi_center: Some(Point2D::new(50.0, 50.0)),
        }];
        let path = planner.plan_with_constraints(&positions, &constraints);
        assert_eq!(path.waypoints.len(), 20);
    }

    #[test]
    fn test_max_deviation_clamped() {
        let planner = PathPlanner::new(PlanStrategy::MinLength)
            .with_window_size(50)
            .with_max_deviation(2.0);
        let positions: Vec<Point2D> = (0..50)
            .map(|i| {
                let noise = if i % 2 == 0 { 30.0 } else { -30.0 };
                Point2D::new(i as f64, noise)
            })
            .collect();
        let path = planner.plan(&positions);
        assert!(path.max_deviation <= 2.0 + 1e-9);
    }

    #[test]
    fn test_realtime_smoother_basic() {
        let mut smoother = RealtimeSmoother::new(10).with_alpha(0.5);
        let p1 = smoother.push(Point2D::new(0.0, 0.0));
        assert!((p1.x - 0.0).abs() < 1e-10);
        let p2 = smoother.push(Point2D::new(10.0, 10.0));
        assert!((p2.x - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_realtime_smoother_reset() {
        let mut smoother = RealtimeSmoother::new(5);
        smoother.push(Point2D::new(1.0, 1.0));
        smoother.push(Point2D::new(2.0, 2.0));
        assert_eq!(smoother.len(), 2);
        smoother.reset();
        assert!(smoother.is_empty());
    }

    #[test]
    fn test_realtime_smoother_capacity() {
        let mut smoother = RealtimeSmoother::new(3);
        for i in 0..10 {
            smoother.push(Point2D::new(i as f64, 0.0));
        }
        assert_eq!(smoother.len(), 3);
    }

    // ── MotionAxisConstraints ────────────────────────────────────────

    #[test]
    fn test_axis_constraints_unconstrained_identity() {
        let c = MotionAxisConstraints::unconstrained().with_unlock_blend(0.0);
        let orig = Point2D::new(10.0, 20.0);
        let smoothed = Point2D::new(15.0, 25.0);
        let result = c.apply(orig, smoothed);
        // blend = 0 → no change
        assert!((result.x - 10.0).abs() < 1e-9);
        assert!((result.y - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_full_blend_unconstrained() {
        let c = MotionAxisConstraints::unconstrained().with_unlock_blend(1.0);
        let orig = Point2D::new(10.0, 20.0);
        let smoothed = Point2D::new(15.0, 25.0);
        let result = c.apply(orig, smoothed);
        // blend = 1 → follow smoothed fully
        assert!((result.x - 15.0).abs() < 1e-9);
        assert!((result.y - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_lock_pan_x_follows_smooth() {
        let c = MotionAxisConstraints::lock_pan_only();
        let orig = Point2D::new(0.0, 0.0);
        let smoothed = Point2D::new(8.0, 5.0);
        let result = c.apply(orig, smoothed);
        // X (pan) locked → correction applied in full
        assert!((result.x - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_lock_pan_y_partial() {
        let c = MotionAxisConstraints::lock_pan_only().with_unlock_blend(0.5);
        let orig = Point2D::new(0.0, 0.0);
        let smoothed = Point2D::new(8.0, 10.0);
        let result = c.apply(orig, smoothed);
        // Y (tilt) unlocked with blend=0.5 → 5.0
        assert!((result.y - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_lock_tilt_y_follows_smooth() {
        let c = MotionAxisConstraints::lock_tilt_only();
        let orig = Point2D::new(0.0, 0.0);
        let smoothed = Point2D::new(6.0, 12.0);
        let result = c.apply(orig, smoothed);
        assert!((result.y - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_pan_max_correction_clamps() {
        let c = MotionAxisConstraints::lock_pan_only().with_pan_max_correction(3.0);
        let orig = Point2D::new(0.0, 0.0);
        let smoothed = Point2D::new(10.0, 0.0);
        let result = c.apply(orig, smoothed);
        // correction clamped to 3.0
        assert!((result.x - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_axis_constraints_lock_both() {
        let c = MotionAxisConstraints::lock_both();
        let orig = Point2D::new(5.0, 7.0);
        let smoothed = Point2D::new(10.0, 14.0);
        let result = c.apply(orig, smoothed);
        // Both axes locked → follow smoothed
        assert!((result.x - 10.0).abs() < 1e-9);
        assert!((result.y - 14.0).abs() < 1e-9);
    }

    #[test]
    fn test_plan_with_axis_constraints_lock_pan_length() {
        let planner = PathPlanner::new(PlanStrategy::MinLength).with_window_size(5);
        let positions: Vec<Point2D> = (0..20)
            .map(|i| {
                let noise = if i % 2 == 0 { 5.0 } else { -5.0 };
                Point2D::new(i as f64 * 10.0 + noise, 100.0 + noise)
            })
            .collect();
        let constraints = MotionAxisConstraints::lock_pan_only();
        let path = planner.plan_with_axis_constraints(&positions, constraints);
        assert_eq!(path.waypoints.len(), 20);
    }

    #[test]
    fn test_plan_with_axis_constraints_empty() {
        let planner = PathPlanner::new(PlanStrategy::MinLength);
        let path = planner.plan_with_axis_constraints(&[], MotionAxisConstraints::default());
        assert!(path.waypoints.is_empty());
    }

    #[test]
    fn test_plan_with_axis_constraints_single() {
        let planner = PathPlanner::new(PlanStrategy::MinLength);
        let path = planner.plan_with_axis_constraints(
            &[Point2D::new(3.0, 7.0)],
            MotionAxisConstraints::lock_pan_only(),
        );
        assert_eq!(path.waypoints.len(), 1);
    }

    #[test]
    fn test_plan_axis_constraints_lock_pan_x_smoother() {
        // With lock_pan, X corrections should track smoothed trajectory closely
        let planner = PathPlanner::new(PlanStrategy::MinLength).with_window_size(9);
        let positions: Vec<Point2D> = (0..30)
            .map(|i| {
                let noise = if i % 2 == 0 { 10.0 } else { -10.0 };
                Point2D::new(noise, i as f64) // pure noise in X, linear in Y
            })
            .collect();
        let constraints = MotionAxisConstraints::lock_pan_only();
        let path = planner.plan_with_axis_constraints(&positions, constraints);
        // The smoothed X should vary less than original
        let x_var_orig: f64 = positions
            .windows(2)
            .map(|w| (w[1].x - w[0].x).abs())
            .sum::<f64>()
            / positions.len() as f64;
        let x_var_smooth: f64 = path
            .waypoints
            .windows(2)
            .map(|w| (w[1].position.x - w[0].position.x).abs())
            .sum::<f64>()
            / path.waypoints.len() as f64;
        assert!(
            x_var_smooth <= x_var_orig + 1e-6,
            "X should be smoother with lock_pan"
        );
    }

    #[test]
    fn test_compute_smoothness_short() {
        let waypoints = vec![Waypoint {
            position: Point2D::new(0.0, 0.0),
            velocity: Point2D::new(1.0, 0.0),
            frame_index: 0,
            confidence: 1.0,
        }];
        assert!((compute_smoothness(&waypoints) - 0.0).abs() < 1e-10);
    }
}
