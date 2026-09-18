#![allow(dead_code)]
//! Tracking session management for virtual production camera systems.
//!
//! Groups tracking data points into logical sessions, supporting
//! multi-target scenarios (cameras, props, talent markers).

/// The type of entity being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackingTarget {
    /// A production camera.
    Camera,
    /// A talent/actor body marker.
    Talent,
    /// A tracked prop or object.
    Prop,
    /// A reference / calibration target.
    Reference,
}

impl std::fmt::Display for TrackingTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackingTarget::Camera => write!(f, "Camera"),
            TrackingTarget::Talent => write!(f, "Talent"),
            TrackingTarget::Prop => write!(f, "Prop"),
            TrackingTarget::Reference => write!(f, "Reference"),
        }
    }
}

impl TrackingTarget {
    /// Returns true if the target affects the LED background rendering.
    #[must_use]
    pub fn affects_background(&self) -> bool {
        matches!(self, TrackingTarget::Camera | TrackingTarget::Reference)
    }
}

/// A 3-D position in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position3D {
    /// Create a new position.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero / origin position.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Euclidean distance to another position.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// A 3-D rotation expressed as Euler angles in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation3D {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

impl Rotation3D {
    /// Create a new rotation.
    #[must_use]
    pub fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        Self { pitch, yaw, roll }
    }

    /// Zero rotation (identity).
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// A single tracking sample with position, rotation, and timestamp.
#[derive(Debug, Clone)]
pub struct TrackingPoint {
    /// Target type for this sample.
    pub target: TrackingTarget,
    /// 3-D position in metres.
    pub position: Position3D,
    /// Orientation.
    pub rotation: Rotation3D,
    /// Timestamp in microseconds since session start.
    pub timestamp_us: u64,
    /// Tracking quality in [0.0, 1.0].
    pub quality: f32,
}

impl TrackingPoint {
    /// Create a new tracking point.
    #[must_use]
    pub fn new(
        target: TrackingTarget,
        position: Position3D,
        rotation: Rotation3D,
        timestamp_us: u64,
        quality: f32,
    ) -> Self {
        Self {
            target,
            position,
            rotation,
            timestamp_us,
            quality: quality.clamp(0.0, 1.0),
        }
    }

    /// Returns true if the quality meets the given minimum threshold.
    #[must_use]
    pub fn is_quality_ok(&self, min_quality: f32) -> bool {
        self.quality >= min_quality
    }
}

/// State of a tracking session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingSessionState {
    /// Session is warming up / calibrating.
    Calibrating,
    /// Session is live and delivering data.
    Tracking,
    /// Session has lost lock and is recovering.
    LostLock,
    /// Session is stopped.
    Stopped,
}

impl std::fmt::Display for TrackingSessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackingSessionState::Calibrating => write!(f, "Calibrating"),
            TrackingSessionState::Tracking => write!(f, "Tracking"),
            TrackingSessionState::LostLock => write!(f, "LostLock"),
            TrackingSessionState::Stopped => write!(f, "Stopped"),
        }
    }
}

/// A live tracking session that accumulates tracking points.
#[derive(Debug)]
pub struct TrackingDataStream {
    /// Human-readable name for this stream.
    pub name: String,
    /// Type of target being tracked.
    pub target_type: TrackingTarget,
    /// Current state.
    pub state: TrackingSessionState,
    /// Accumulated tracking points.
    points: Vec<TrackingPoint>,
    /// Maximum number of points to retain (ring buffer semantics).
    pub max_points: usize,
}

impl TrackingDataStream {
    /// Create a new tracking stream in `Calibrating` state.
    pub fn new(name: impl Into<String>, target_type: TrackingTarget, max_points: usize) -> Self {
        Self {
            name: name.into(),
            target_type,
            state: TrackingSessionState::Calibrating,
            points: Vec::new(),
            max_points,
        }
    }

    /// Begin active tracking.
    pub fn start_tracking(&mut self) {
        if self.state == TrackingSessionState::Calibrating
            || self.state == TrackingSessionState::LostLock
        {
            self.state = TrackingSessionState::Tracking;
        }
    }

    /// Signal a lock loss.
    pub fn lose_lock(&mut self) {
        if self.state == TrackingSessionState::Tracking {
            self.state = TrackingSessionState::LostLock;
        }
    }

    /// Stop the stream.
    pub fn stop(&mut self) {
        self.state = TrackingSessionState::Stopped;
    }

    /// Ingest a new tracking point.  Only accepted when in `Tracking` state.
    /// When `max_points` is exceeded, the oldest point is dropped.
    pub fn push(&mut self, point: TrackingPoint) -> bool {
        if self.state != TrackingSessionState::Tracking {
            return false;
        }
        if self.points.len() >= self.max_points {
            self.points.remove(0);
        }
        self.points.push(point);
        true
    }

    /// Return the most recent tracking point, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&TrackingPoint> {
        self.points.last()
    }

    /// Number of points currently stored.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Average quality of all stored points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_quality(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.points.iter().map(|p| p.quality).sum();
        sum / self.points.len() as f32
    }

    /// Returns all points whose quality is at or above `min_quality`.
    #[must_use]
    pub fn high_quality_points(&self, min_quality: f32) -> Vec<&TrackingPoint> {
        self.points
            .iter()
            .filter(|p| p.is_quality_ok(min_quality))
            .collect()
    }

    /// Returns true if the stream is live.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.state == TrackingSessionState::Tracking
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(ts: u64, quality: f32) -> TrackingPoint {
        TrackingPoint::new(
            TrackingTarget::Camera,
            Position3D::new(1.0, 2.0, 0.5),
            Rotation3D::zero(),
            ts,
            quality,
        )
    }

    fn make_stream(name: &str) -> TrackingDataStream {
        TrackingDataStream::new(name, TrackingTarget::Camera, 10)
    }

    #[test]
    fn test_tracking_target_display() {
        assert_eq!(TrackingTarget::Camera.to_string(), "Camera");
        assert_eq!(TrackingTarget::Talent.to_string(), "Talent");
        assert_eq!(TrackingTarget::Prop.to_string(), "Prop");
        assert_eq!(TrackingTarget::Reference.to_string(), "Reference");
    }

    #[test]
    fn test_tracking_target_affects_background() {
        assert!(TrackingTarget::Camera.affects_background());
        assert!(TrackingTarget::Reference.affects_background());
        assert!(!TrackingTarget::Talent.affects_background());
        assert!(!TrackingTarget::Prop.affects_background());
    }

    #[test]
    fn test_position3d_distance() {
        let a = Position3D::new(0.0, 0.0, 0.0);
        let b = Position3D::new(3.0, 4.0, 0.0);
        let d = a.distance_to(&b);
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_position3d_zero() {
        let p = Position3D::zero();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
        assert_eq!(p.z, 0.0);
    }

    #[test]
    fn test_tracking_point_quality_clamp() {
        let p = make_point(0, 1.5); // quality > 1.0, should clamp
        assert_eq!(p.quality, 1.0);
        let p2 = make_point(0, -0.5); // quality < 0.0, should clamp
        assert_eq!(p2.quality, 0.0);
    }

    #[test]
    fn test_tracking_point_quality_ok() {
        let p = make_point(0, 0.8);
        assert!(p.is_quality_ok(0.5));
        assert!(!p.is_quality_ok(0.9));
    }

    #[test]
    fn test_tracking_session_state_display() {
        assert_eq!(TrackingSessionState::Tracking.to_string(), "Tracking");
        assert_eq!(TrackingSessionState::LostLock.to_string(), "LostLock");
        assert_eq!(TrackingSessionState::Stopped.to_string(), "Stopped");
    }

    #[test]
    fn test_stream_start_tracking() {
        let mut s = make_stream("cam1");
        assert_eq!(s.state, TrackingSessionState::Calibrating);
        s.start_tracking();
        assert_eq!(s.state, TrackingSessionState::Tracking);
        assert!(s.is_live());
    }

    #[test]
    fn test_stream_push_only_when_tracking() {
        let mut s = make_stream("cam2");
        assert!(!s.push(make_point(0, 0.9))); // still calibrating
        s.start_tracking();
        assert!(s.push(make_point(1, 0.9)));
        assert_eq!(s.point_count(), 1);
    }

    #[test]
    fn test_stream_ring_buffer_evicts_oldest() {
        let mut s = TrackingDataStream::new("cam3", TrackingTarget::Camera, 3);
        s.start_tracking();
        s.push(make_point(1, 0.9));
        s.push(make_point(2, 0.8));
        s.push(make_point(3, 0.7));
        s.push(make_point(4, 0.6)); // should evict ts=1
        assert_eq!(s.point_count(), 3);
        assert_eq!(s.latest().expect("should succeed in test").timestamp_us, 4);
    }

    #[test]
    fn test_stream_average_quality() {
        let mut s = make_stream("cam4");
        s.start_tracking();
        s.push(make_point(0, 0.6));
        s.push(make_point(1, 0.8));
        let avg = s.average_quality();
        assert!((avg - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_stream_high_quality_points() {
        let mut s = make_stream("cam5");
        s.start_tracking();
        s.push(make_point(0, 0.5));
        s.push(make_point(1, 0.9));
        s.push(make_point(2, 0.3));
        let hq = s.high_quality_points(0.7);
        assert_eq!(hq.len(), 1);
        assert_eq!(hq[0].timestamp_us, 1);
    }

    #[test]
    fn test_stream_lose_lock_and_recover() {
        let mut s = make_stream("cam6");
        s.start_tracking();
        s.lose_lock();
        assert_eq!(s.state, TrackingSessionState::LostLock);
        assert!(!s.is_live());
        s.start_tracking(); // recover
        assert_eq!(s.state, TrackingSessionState::Tracking);
    }

    #[test]
    fn test_stream_stop() {
        let mut s = make_stream("cam7");
        s.start_tracking();
        s.stop();
        assert_eq!(s.state, TrackingSessionState::Stopped);
        // Cannot push after stop
        assert!(!s.push(make_point(100, 1.0)));
    }

    #[test]
    fn test_stream_average_quality_empty() {
        let s = make_stream("cam8");
        assert_eq!(s.average_quality(), 0.0);
    }
}
