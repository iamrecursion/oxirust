#![allow(dead_code)]
//! Camera rig management for virtual production.
//!
//! Provides abstractions for multi-camera rigs used on LED-wall stages,
//! including boom, dolly, crane, and Steadicam configurations.
//! Each rig tracks its own coordinate space and can report
//! position/orientation deltas per frame.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of physical camera rig.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RigKind {
    /// Fixed tripod mount.
    Tripod,
    /// Dolly on rails.
    Dolly,
    /// Jib / crane arm.
    Crane,
    /// Steadicam / gimbal operator.
    Steadicam,
    /// Technocrane with encoded axes.
    Technocrane,
    /// Free-roaming handheld.
    Handheld,
}

/// 3-D position in stage coordinates (metres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// X coordinate (metres).
    pub x: f64,
    /// Y coordinate (metres).
    pub y: f64,
    /// Z coordinate (metres).
    pub z: f64,
}

impl Position {
    /// Create a new position.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another position.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Linear interpolation between two positions.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Euler-angle orientation (degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    /// Pan (yaw) in degrees.
    pub pan: f64,
    /// Tilt (pitch) in degrees.
    pub tilt: f64,
    /// Roll in degrees.
    pub roll: f64,
}

impl Orientation {
    /// Create a new orientation.
    #[must_use]
    pub fn new(pan: f64, tilt: f64, roll: f64) -> Self {
        Self { pan, tilt, roll }
    }

    /// Normalize all angles to [−180, 180).
    #[must_use]
    pub fn normalized(&self) -> Self {
        fn norm(deg: f64) -> f64 {
            let mut d = deg % 360.0;
            if d >= 180.0 {
                d -= 360.0;
            }
            if d < -180.0 {
                d += 360.0;
            }
            d
        }
        Self {
            pan: norm(self.pan),
            tilt: norm(self.tilt),
            roll: norm(self.roll),
        }
    }

    /// Check whether the orientation is within a tolerance of another.
    #[must_use]
    pub fn is_close_to(&self, other: &Self, tolerance_deg: f64) -> bool {
        let a = self.normalized();
        let b = other.normalized();
        (a.pan - b.pan).abs() < tolerance_deg
            && (a.tilt - b.tilt).abs() < tolerance_deg
            && (a.roll - b.roll).abs() < tolerance_deg
    }
}

impl Default for Orientation {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// A single frame of rig telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct RigFrame {
    /// Frame number.
    pub frame: u64,
    /// Absolute position.
    pub position: Position,
    /// Absolute orientation.
    pub orientation: Orientation,
    /// Lens focal length in mm (if available).
    pub focal_length_mm: Option<f64>,
    /// Lens focus distance in metres (if available).
    pub focus_distance_m: Option<f64>,
}

/// Limits that constrain a rig's movement envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct RigLimits {
    /// Minimum position bound.
    pub min_position: Position,
    /// Maximum position bound.
    pub max_position: Position,
    /// Maximum pan speed (degrees / second).
    pub max_pan_speed: f64,
    /// Maximum tilt speed (degrees / second).
    pub max_tilt_speed: f64,
    /// Maximum linear speed (metres / second).
    pub max_linear_speed: f64,
}

impl Default for RigLimits {
    fn default() -> Self {
        Self {
            min_position: Position::new(-10.0, 0.0, -10.0),
            max_position: Position::new(10.0, 5.0, 10.0),
            max_pan_speed: 180.0,
            max_tilt_speed: 90.0,
            max_linear_speed: 3.0,
        }
    }
}

impl RigLimits {
    /// Check whether a position lies inside the allowed envelope.
    #[must_use]
    pub fn contains(&self, pos: &Position) -> bool {
        pos.x >= self.min_position.x
            && pos.x <= self.max_position.x
            && pos.y >= self.min_position.y
            && pos.y <= self.max_position.y
            && pos.z >= self.min_position.z
            && pos.z <= self.max_position.z
    }

    /// Volume of the movement envelope (cubic metres).
    #[must_use]
    pub fn volume(&self) -> f64 {
        let dx = (self.max_position.x - self.min_position.x).abs();
        let dy = (self.max_position.y - self.min_position.y).abs();
        let dz = (self.max_position.z - self.min_position.z).abs();
        dx * dy * dz
    }
}

/// Represents a single camera rig on the virtual-production stage.
#[derive(Debug, Clone)]
pub struct CameraRig {
    /// Unique rig identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Kind of rig.
    pub kind: RigKind,
    /// Movement limits.
    pub limits: RigLimits,
    /// Telemetry history (latest frames).
    history: Vec<RigFrame>,
    /// Maximum history depth.
    max_history: usize,
}

impl CameraRig {
    /// Create a new camera rig.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: RigKind) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind,
            limits: RigLimits::default(),
            history: Vec::new(),
            max_history: 300,
        }
    }

    /// Override the movement limits.
    #[must_use]
    pub fn with_limits(mut self, limits: RigLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the maximum telemetry history depth.
    #[must_use]
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Record a new frame of telemetry.
    pub fn record_frame(&mut self, frame: RigFrame) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(frame);
    }

    /// Get the latest telemetry frame.
    #[must_use]
    pub fn latest_frame(&self) -> Option<&RigFrame> {
        self.history.last()
    }

    /// Number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.history.len()
    }

    /// Total distance travelled (metres) across all recorded frames.
    #[must_use]
    pub fn total_distance(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        self.history
            .windows(2)
            .map(|w| w[0].position.distance_to(&w[1].position))
            .sum()
    }

    /// Check whether the rig has exceeded its movement envelope in any frame.
    #[must_use]
    pub fn has_out_of_bounds_frames(&self) -> bool {
        self.history
            .iter()
            .any(|f| !self.limits.contains(&f.position))
    }

    /// Clear telemetry history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

/// Manages a collection of camera rigs on stage.
#[derive(Debug, Clone)]
pub struct RigManager {
    /// All rigs keyed by ID.
    rigs: HashMap<String, CameraRig>,
}

impl RigManager {
    /// Create an empty rig manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rigs: HashMap::new(),
        }
    }

    /// Add a rig. Returns `false` if the ID already exists.
    pub fn add_rig(&mut self, rig: CameraRig) -> bool {
        if self.rigs.contains_key(&rig.id) {
            return false;
        }
        self.rigs.insert(rig.id.clone(), rig);
        true
    }

    /// Remove a rig by ID. Returns the removed rig if present.
    pub fn remove_rig(&mut self, id: &str) -> Option<CameraRig> {
        self.rigs.remove(id)
    }

    /// Get a rig by ID.
    #[must_use]
    pub fn get_rig(&self, id: &str) -> Option<&CameraRig> {
        self.rigs.get(id)
    }

    /// Number of rigs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.rigs.len()
    }

    /// Iterate over all rigs.
    pub fn rigs(&self) -> impl Iterator<Item = &CameraRig> {
        self.rigs.values()
    }

    /// Find the rig that is closest to a given position (latest frame).
    #[must_use]
    pub fn closest_to(&self, pos: &Position) -> Option<&CameraRig> {
        self.rigs
            .values()
            .filter_map(|r| r.latest_frame().map(|f| (r, f.position.distance_to(pos))))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(r, _)| r)
    }
}

impl Default for RigManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame(frame: u64, x: f64, y: f64, z: f64) -> RigFrame {
        RigFrame {
            frame,
            position: Position::new(x, y, z),
            orientation: Orientation::default(),
            focal_length_mm: Some(35.0),
            focus_distance_m: Some(3.0),
        }
    }

    // -- Position --

    #[test]
    fn test_position_distance() {
        let a = Position::new(0.0, 0.0, 0.0);
        let b = Position::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_lerp_midpoint() {
        let a = Position::new(0.0, 0.0, 0.0);
        let b = Position::new(10.0, 10.0, 10.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-9);
        assert!((mid.y - 5.0).abs() < 1e-9);
        assert!((mid.z - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_position_lerp_clamped() {
        let a = Position::new(0.0, 0.0, 0.0);
        let b = Position::new(10.0, 0.0, 0.0);
        let over = a.lerp(&b, 2.0);
        assert!((over.x - 10.0).abs() < 1e-9); // clamped to 1.0
    }

    // -- Orientation --

    #[test]
    fn test_orientation_normalize() {
        let o = Orientation::new(370.0, -200.0, 540.0);
        let n = o.normalized();
        assert!((n.pan - 10.0).abs() < 1e-9);
        assert!((n.tilt - 160.0).abs() < 1e-9);
        assert!((n.roll - 180.0).abs() < 1e-6 || (n.roll + 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_orientation_is_close() {
        let a = Orientation::new(10.0, 20.0, 0.0);
        let b = Orientation::new(10.5, 20.3, 0.1);
        assert!(a.is_close_to(&b, 1.0));
        assert!(!a.is_close_to(&b, 0.1));
    }

    // -- RigLimits --

    #[test]
    fn test_limits_contains() {
        let lim = RigLimits::default();
        assert!(lim.contains(&Position::new(0.0, 1.0, 0.0)));
        assert!(!lim.contains(&Position::new(100.0, 0.0, 0.0)));
    }

    #[test]
    fn test_limits_volume() {
        let lim = RigLimits {
            min_position: Position::new(0.0, 0.0, 0.0),
            max_position: Position::new(2.0, 3.0, 4.0),
            ..Default::default()
        };
        assert!((lim.volume() - 24.0).abs() < 1e-9);
    }

    // -- CameraRig --

    #[test]
    fn test_rig_record_and_latest() {
        let mut rig = CameraRig::new("rig1", "Main", RigKind::Dolly);
        assert!(rig.latest_frame().is_none());
        rig.record_frame(sample_frame(0, 0.0, 0.0, 0.0));
        assert_eq!(rig.latest_frame().expect("should succeed in test").frame, 0);
        rig.record_frame(sample_frame(1, 1.0, 0.0, 0.0));
        assert_eq!(rig.latest_frame().expect("should succeed in test").frame, 1);
    }

    #[test]
    fn test_rig_history_limit() {
        let mut rig = CameraRig::new("r", "R", RigKind::Tripod).with_max_history(3);
        for i in 0..5 {
            rig.record_frame(sample_frame(i, 0.0, 0.0, 0.0));
        }
        assert_eq!(rig.frame_count(), 3);
        assert_eq!(rig.history[0].frame, 2);
    }

    #[test]
    fn test_rig_total_distance() {
        let mut rig = CameraRig::new("r", "R", RigKind::Dolly);
        rig.record_frame(sample_frame(0, 0.0, 0.0, 0.0));
        rig.record_frame(sample_frame(1, 3.0, 4.0, 0.0));
        assert!((rig.total_distance() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_rig_out_of_bounds() {
        let mut rig = CameraRig::new("r", "R", RigKind::Crane);
        rig.record_frame(sample_frame(0, 0.0, 0.0, 0.0));
        assert!(!rig.has_out_of_bounds_frames());
        rig.record_frame(sample_frame(1, 999.0, 0.0, 0.0));
        assert!(rig.has_out_of_bounds_frames());
    }

    #[test]
    fn test_rig_clear_history() {
        let mut rig = CameraRig::new("r", "R", RigKind::Handheld);
        rig.record_frame(sample_frame(0, 0.0, 0.0, 0.0));
        rig.clear_history();
        assert_eq!(rig.frame_count(), 0);
    }

    // -- RigManager --

    #[test]
    fn test_manager_add_remove() {
        let mut mgr = RigManager::new();
        let rig = CameraRig::new("a", "A", RigKind::Tripod);
        assert!(mgr.add_rig(rig));
        assert_eq!(mgr.count(), 1);
        assert!(!mgr.add_rig(CameraRig::new("a", "A2", RigKind::Dolly))); // dup
        assert!(mgr.remove_rig("a").is_some());
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_manager_closest_to() {
        let mut mgr = RigManager::new();
        let mut r1 = CameraRig::new("a", "A", RigKind::Tripod);
        r1.record_frame(sample_frame(0, 0.0, 0.0, 0.0));
        let mut r2 = CameraRig::new("b", "B", RigKind::Dolly);
        r2.record_frame(sample_frame(0, 10.0, 0.0, 0.0));
        mgr.add_rig(r1);
        mgr.add_rig(r2);

        let target = Position::new(1.0, 0.0, 0.0);
        let closest = mgr.closest_to(&target).expect("should succeed in test");
        assert_eq!(closest.id, "a");
    }
}
