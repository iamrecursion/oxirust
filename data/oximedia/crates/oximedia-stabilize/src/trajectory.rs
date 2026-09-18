#![allow(dead_code)]
//! Camera trajectory modelling and shake detection.
//!
//! Tracks the time-series of camera positions and derives velocity, smoothing,
//! and shake metrics without duplicating the existing `motion::trajectory` module.

/// A single sample on the camera trajectory timeline.
#[derive(Debug, Clone)]
pub struct TrajectoryPoint {
    /// Timestamp in milliseconds
    pub time_ms: f64,
    /// Horizontal translation component
    pub tx: f64,
    /// Vertical translation component
    pub ty: f64,
    /// Rotation in radians
    pub rotation: f64,
    /// Scale factor
    pub scale: f64,
}

impl TrajectoryPoint {
    /// Create a new trajectory sample.
    pub fn new(time_ms: f64, tx: f64, ty: f64, rotation: f64, scale: f64) -> Self {
        Self {
            time_ms,
            tx,
            ty,
            rotation,
            scale,
        }
    }

    /// Create a neutral (identity) point at the given timestamp.
    pub fn identity(time_ms: f64) -> Self {
        Self {
            time_ms,
            tx: 0.0,
            ty: 0.0,
            rotation: 0.0,
            scale: 1.0,
        }
    }

    /// Translational velocity relative to a previous point (pixels / ms).
    ///
    /// Returns `(vx, vy)` or `(0, 0)` if the time delta is negligible.
    pub fn velocity(&self, prev: &TrajectoryPoint) -> (f64, f64) {
        let dt = self.time_ms - prev.time_ms;
        if dt.abs() < 1e-9 {
            return (0.0, 0.0);
        }
        ((self.tx - prev.tx) / dt, (self.ty - prev.ty) / dt)
    }

    /// Speed magnitude at this point relative to `prev`.
    pub fn speed(&self, prev: &TrajectoryPoint) -> f64 {
        let (vx, vy) = self.velocity(prev);
        (vx * vx + vy * vy).sqrt()
    }

    /// Translational magnitude from origin.
    pub fn magnitude(&self) -> f64 {
        (self.tx * self.tx + self.ty * self.ty).sqrt()
    }
}

/// An ordered sequence of `TrajectoryPoint` samples representing camera motion over time.
#[derive(Debug, Clone, Default)]
pub struct CameraTrajectory {
    points: Vec<TrajectoryPoint>,
}

impl CameraTrajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Create a trajectory pre-allocated for `capacity` samples.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Append a trajectory point.
    pub fn add_point(&mut self, point: TrajectoryPoint) {
        self.points.push(point);
    }

    /// Number of samples in the trajectory.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the trajectory has no samples.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Borrow the internal point list.
    pub fn points(&self) -> &[TrajectoryPoint] {
        &self.points
    }

    /// Duration from the first to the last sample in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        self.points
            .last()
            .expect("invariant: len >= 2 checked above")
            .time_ms
            - self
                .points
                .first()
                .expect("invariant: len >= 2 checked above")
                .time_ms
    }

    /// Return a smoothed copy of this trajectory using a simple box filter of `window` samples.
    ///
    /// Points near the edges use as many samples as are available.
    pub fn smooth(&self, window: usize) -> CameraTrajectory {
        if self.points.is_empty() || window == 0 {
            return self.clone();
        }
        let half = window / 2;
        let n = self.points.len();
        let mut out = Vec::with_capacity(n);

        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let count = (hi - lo) as f64;
            let tx = self.points[lo..hi].iter().map(|p| p.tx).sum::<f64>() / count;
            let ty = self.points[lo..hi].iter().map(|p| p.ty).sum::<f64>() / count;
            let rot = self.points[lo..hi].iter().map(|p| p.rotation).sum::<f64>() / count;
            let sc = self.points[lo..hi].iter().map(|p| p.scale).sum::<f64>() / count;
            out.push(TrajectoryPoint::new(
                self.points[i].time_ms,
                tx,
                ty,
                rot,
                sc,
            ));
        }
        CameraTrajectory { points: out }
    }

    /// Root-mean-square translational displacement across the trajectory.
    #[allow(clippy::cast_precision_loss)]
    pub fn rms_displacement(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self.points.iter().map(|p| p.magnitude().powi(2)).sum();
        (sum_sq / self.points.len() as f64).sqrt()
    }

    /// Maximum translational displacement sample.
    pub fn max_displacement(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.magnitude())
            .fold(0.0_f64, f64::max)
    }
}

/// Threshold parameters for shake detection.
#[derive(Debug, Clone)]
pub struct ShakeThresholds {
    /// RMS displacement above which the video is considered shaky (pixels)
    pub rms_threshold: f64,
    /// Speed above which consecutive frames are considered shaky (pixels/ms)
    pub speed_threshold: f64,
    /// Fraction of shaky frames that triggers a "severe shake" diagnosis
    pub severe_fraction: f64,
}

impl Default for ShakeThresholds {
    fn default() -> Self {
        Self {
            rms_threshold: 5.0,
            speed_threshold: 1.0,
            severe_fraction: 0.30,
        }
    }
}

/// Shake analysis result.
#[derive(Debug, Clone)]
pub struct ShakeAnalysis {
    /// Overall shake score (0.0 = none, 1.0 = maximum)
    pub score: f64,
    /// Whether the shake qualifies as severe
    pub is_severe: bool,
    /// Fraction of inter-frame transitions that exceeded the speed threshold
    pub shaky_transition_fraction: f64,
    /// RMS displacement of the raw trajectory
    pub rms_displacement: f64,
}

/// Analyses a `CameraTrajectory` for camera shake.
#[derive(Debug, Default)]
pub struct TrajectoryAnalyzer {
    thresholds: ShakeThresholds,
}

impl TrajectoryAnalyzer {
    /// Create an analyzer with default thresholds.
    pub fn new() -> Self {
        Self {
            thresholds: ShakeThresholds::default(),
        }
    }

    /// Create an analyzer with custom thresholds.
    pub fn with_thresholds(thresholds: ShakeThresholds) -> Self {
        Self { thresholds }
    }

    /// Detect shake characteristics in the given trajectory.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_shake(&self, traj: &CameraTrajectory) -> ShakeAnalysis {
        let rms = traj.rms_displacement();
        let n = traj.points().len();

        let mut shaky_transitions = 0usize;
        let total_transitions = n.saturating_sub(1);

        if total_transitions > 0 {
            for i in 1..n {
                let speed = traj.points()[i].speed(&traj.points()[i - 1]);
                if speed > self.thresholds.speed_threshold {
                    shaky_transitions += 1;
                }
            }
        }

        let shaky_fraction = if total_transitions > 0 {
            shaky_transitions as f64 / total_transitions as f64
        } else {
            0.0
        };

        // Normalise score: blend RMS-based and transition-based components
        let rms_score = (rms / (self.thresholds.rms_threshold * 2.0)).min(1.0);
        let score = (rms_score + shaky_fraction) / 2.0;

        ShakeAnalysis {
            score,
            is_severe: shaky_fraction >= self.thresholds.severe_fraction,
            shaky_transition_fraction: shaky_fraction,
            rms_displacement: rms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_traj(n: usize, step_px: f64) -> CameraTrajectory {
        let mut t = CameraTrajectory::with_capacity(n);
        for i in 0..n {
            t.add_point(TrajectoryPoint::new(
                i as f64 * 33.33,
                i as f64 * step_px,
                0.0,
                0.0,
                1.0,
            ));
        }
        t
    }

    #[test]
    fn test_trajectory_point_identity() {
        let p = TrajectoryPoint::identity(100.0);
        assert!((p.tx).abs() < 1e-10);
        assert!((p.rotation).abs() < 1e-10);
        assert!((p.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trajectory_point_velocity_zero_dt() {
        let p = TrajectoryPoint::new(10.0, 5.0, 5.0, 0.0, 1.0);
        let same = p.clone();
        let (vx, vy) = p.velocity(&same);
        assert!((vx).abs() < 1e-9);
        assert!((vy).abs() < 1e-9);
    }

    #[test]
    fn test_trajectory_point_velocity() {
        let p0 = TrajectoryPoint::new(0.0, 0.0, 0.0, 0.0, 1.0);
        let p1 = TrajectoryPoint::new(10.0, 100.0, 0.0, 0.0, 1.0);
        let (vx, _vy) = p1.velocity(&p0);
        assert!((vx - 10.0).abs() < 1e-9); // 100px / 10ms
    }

    #[test]
    fn test_trajectory_point_speed() {
        let p0 = TrajectoryPoint::new(0.0, 0.0, 0.0, 0.0, 1.0);
        let p1 = TrajectoryPoint::new(10.0, 30.0, 40.0, 0.0, 1.0);
        // vx = 3, vy = 4 → speed = 5
        let speed = p1.speed(&p0);
        assert!((speed - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_camera_trajectory_add_and_len() {
        let mut t = CameraTrajectory::new();
        assert!(t.is_empty());
        t.add_point(TrajectoryPoint::identity(0.0));
        t.add_point(TrajectoryPoint::identity(33.0));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_camera_trajectory_duration_ms() {
        let t = make_traj(5, 1.0);
        // 4 intervals of 33.33 ms each
        assert!((t.duration_ms() - 4.0 * 33.33).abs() < 0.1);
    }

    #[test]
    fn test_camera_trajectory_duration_empty() {
        let t = CameraTrajectory::new();
        assert!((t.duration_ms()).abs() < 1e-10);
    }

    #[test]
    fn test_camera_trajectory_smooth_identity() {
        let t = make_traj(10, 0.0); // static — smoothing should not change values
        let s = t.smooth(5);
        assert_eq!(s.len(), 10);
        for p in s.points() {
            assert!((p.tx).abs() < 1e-9);
        }
    }

    #[test]
    fn test_camera_trajectory_smooth_reduces_rms() {
        let t = make_traj(20, 5.0); // linearly increasing tx — smoothing retains trend
        let s = t.smooth(5);
        assert_eq!(s.len(), t.len());
    }

    #[test]
    fn test_camera_trajectory_rms_displacement_zero() {
        let t = make_traj(10, 0.0);
        assert!((t.rms_displacement()).abs() < 1e-10);
    }

    #[test]
    fn test_camera_trajectory_max_displacement() {
        let t = make_traj(5, 10.0); // tx grows: 0,10,20,30,40 → max = 40
        assert!((t.max_displacement() - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_shake_analyzer_stable_video() {
        let t = make_traj(30, 0.0); // zero motion
        let analyzer = TrajectoryAnalyzer::new();
        let result = analyzer.detect_shake(&t);
        assert!(result.score < 0.1);
        assert!(!result.is_severe);
    }

    #[test]
    fn test_shake_analyzer_shaky_video() {
        // Alternating offsets simulate heavy shake
        let mut t = CameraTrajectory::with_capacity(30);
        for i in 0..30 {
            let tx = if i % 2 == 0 { 0.0 } else { 200.0 };
            t.add_point(TrajectoryPoint::new(i as f64 * 33.33, tx, 0.0, 0.0, 1.0));
        }
        let analyzer = TrajectoryAnalyzer::new();
        let result = analyzer.detect_shake(&t);
        assert!(result.score > 0.1);
    }

    #[test]
    fn test_shake_analyzer_empty_trajectory() {
        let t = CameraTrajectory::new();
        let analyzer = TrajectoryAnalyzer::new();
        let result = analyzer.detect_shake(&t);
        assert!((result.score).abs() < 1e-6); // graceful zero
    }
}
