#![allow(dead_code)]
//! Feature tracking for video stabilization.
//!
//! Tracks individual image features across consecutive frames to build
//! motion trajectories used by the stabilization pipeline.

/// Current state of a tracked feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    /// Feature is actively being tracked.
    Active,
    /// Feature was lost (moved out of frame or occluded).
    Lost,
    /// Feature was explicitly stopped by the tracker.
    Stopped,
    /// Feature was just detected and not yet confirmed.
    Tentative,
}

impl TrackState {
    /// Whether the feature is still usable for motion estimation.
    #[must_use]
    pub fn is_alive(self) -> bool {
        matches!(self, Self::Active | Self::Tentative)
    }
}

/// A single tracked feature across multiple frames.
#[derive(Debug, Clone)]
pub struct TrackedFeature {
    /// Unique identifier.
    pub id: u64,
    /// Current x position.
    pub x: f64,
    /// Current y position.
    pub y: f64,
    /// Tracking state.
    pub state: TrackState,
    /// History of positions `(x, y)` per frame (most recent last).
    pub history: Vec<(f64, f64)>,
    /// Quality / response strength of the feature.
    pub quality: f64,
    /// Frame index when the feature was first detected.
    pub first_frame: usize,
    /// Frame index of the most recent observation.
    pub last_frame: usize,
}

impl TrackedFeature {
    /// Create a new tracked feature.
    #[must_use]
    pub fn new(id: u64, x: f64, y: f64, quality: f64, frame: usize) -> Self {
        Self {
            id,
            x,
            y,
            state: TrackState::Tentative,
            history: vec![(x, y)],
            quality,
            first_frame: frame,
            last_frame: frame,
        }
    }

    /// Update position with a new observation.
    pub fn update(&mut self, x: f64, y: f64, frame: usize) {
        self.x = x;
        self.y = y;
        self.last_frame = frame;
        self.history.push((x, y));
        if self.state == TrackState::Tentative && self.history.len() >= 3 {
            self.state = TrackState::Active;
        }
    }

    /// Mark the feature as lost.
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    /// Number of frames this feature has been tracked.
    #[must_use]
    pub fn track_length(&self) -> usize {
        self.history.len()
    }

    /// Displacement from the previous frame, or `(0, 0)` if only one observation.
    #[must_use]
    pub fn last_displacement(&self) -> (f64, f64) {
        if self.history.len() < 2 {
            return (0.0, 0.0);
        }
        let n = self.history.len();
        let (px, py) = self.history[n - 2];
        let (cx, cy) = self.history[n - 1];
        (cx - px, cy - py)
    }

    /// Total path length traversed.
    #[must_use]
    pub fn path_length(&self) -> f64 {
        self.history
            .windows(2)
            .map(|w| {
                let dx = w[1].0 - w[0].0;
                let dy = w[1].1 - w[0].1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }
}

/// Configuration for the feature tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Maximum number of features to track simultaneously.
    pub max_features: usize,
    /// Minimum quality threshold for feature detection.
    pub quality_threshold: f64,
    /// Maximum displacement per frame (outlier rejection).
    pub max_displacement: f64,
    /// Number of consecutive misses before marking lost.
    pub max_miss_count: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_features: 500,
            quality_threshold: 0.01,
            max_displacement: 50.0,
            max_miss_count: 3,
        }
    }
}

/// Feature tracker that maintains a set of tracked features across frames.
#[derive(Debug)]
pub struct FeatureTracker {
    config: TrackerConfig,
    features: Vec<TrackedFeature>,
    next_id: u64,
    current_frame: usize,
}

impl FeatureTracker {
    /// Create a new tracker with the given configuration.
    #[must_use]
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            features: Vec::new(),
            next_id: 0,
            current_frame: 0,
        }
    }

    /// Create a tracker with default settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TrackerConfig::default())
    }

    /// Number of currently alive features.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.features.iter().filter(|f| f.state.is_alive()).count()
    }

    /// Total features (including lost).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.features.len()
    }

    /// Add a newly detected feature.
    pub fn add_feature(&mut self, x: f64, y: f64, quality: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let feature = TrackedFeature::new(id, x, y, quality, self.current_frame);
        self.features.push(feature);
        id
    }

    /// Update a feature by id with a new position. Returns false if not found or dead.
    pub fn update_feature(&mut self, id: u64, x: f64, y: f64) -> bool {
        if let Some(f) = self.features.iter_mut().find(|f| f.id == id) {
            if !f.state.is_alive() {
                return false;
            }
            let (dx, dy) = (x - f.x, y - f.y);
            let disp = (dx * dx + dy * dy).sqrt();
            if disp > self.config.max_displacement {
                f.mark_lost();
                return false;
            }
            f.update(x, y, self.current_frame);
            true
        } else {
            false
        }
    }

    /// Advance to the next frame. Features not updated will accumulate misses.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
        for f in &mut self.features {
            if f.state.is_alive()
                && f.last_frame
                    < self
                        .current_frame
                        .saturating_sub(self.config.max_miss_count)
            {
                f.mark_lost();
            }
        }
    }

    /// Prune lost features from the internal list.
    pub fn prune_lost(&mut self) {
        self.features.retain(|f| f.state.is_alive());
    }

    /// Get all active features.
    #[must_use]
    pub fn active_features(&self) -> Vec<&TrackedFeature> {
        self.features
            .iter()
            .filter(|f| f.state.is_alive())
            .collect()
    }

    /// Average track length of active features.
    #[must_use]
    pub fn avg_track_length(&self) -> f64 {
        let active: Vec<_> = self.active_features();
        if active.is_empty() {
            return 0.0;
        }
        let sum: usize = active.iter().map(|f| f.track_length()).sum();
        sum as f64 / active.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_state_is_alive() {
        assert!(TrackState::Active.is_alive());
        assert!(TrackState::Tentative.is_alive());
        assert!(!TrackState::Lost.is_alive());
        assert!(!TrackState::Stopped.is_alive());
    }

    #[test]
    fn test_tracked_feature_new() {
        let f = TrackedFeature::new(0, 10.0, 20.0, 0.9, 0);
        assert_eq!(f.id, 0);
        assert_eq!(f.state, TrackState::Tentative);
        assert_eq!(f.track_length(), 1);
    }

    #[test]
    fn test_tracked_feature_update_promotes() {
        let mut f = TrackedFeature::new(0, 0.0, 0.0, 1.0, 0);
        assert_eq!(f.state, TrackState::Tentative);
        f.update(1.0, 1.0, 1);
        assert_eq!(f.state, TrackState::Tentative); // need 3 observations
        f.update(2.0, 2.0, 2);
        assert_eq!(f.state, TrackState::Active);
    }

    #[test]
    fn test_last_displacement() {
        let mut f = TrackedFeature::new(0, 0.0, 0.0, 1.0, 0);
        let (dx, _dy) = f.last_displacement();
        assert!((dx).abs() < f64::EPSILON);
        f.update(3.0, 4.0, 1);
        let (dx, dy) = f.last_displacement();
        assert!((dx - 3.0).abs() < 1e-9);
        assert!((dy - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_length() {
        let mut f = TrackedFeature::new(0, 0.0, 0.0, 1.0, 0);
        f.update(3.0, 4.0, 1); // dist = 5
        f.update(3.0, 4.0, 2); // dist = 0
        assert!((f.path_length() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_mark_lost() {
        let mut f = TrackedFeature::new(0, 0.0, 0.0, 1.0, 0);
        f.mark_lost();
        assert_eq!(f.state, TrackState::Lost);
        assert!(!f.state.is_alive());
    }

    #[test]
    fn test_tracker_add_feature() {
        let mut tracker = FeatureTracker::with_defaults();
        let id = tracker.add_feature(10.0, 20.0, 0.5);
        assert_eq!(id, 0);
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.total_count(), 1);
    }

    #[test]
    fn test_tracker_update_feature() {
        let mut tracker = FeatureTracker::with_defaults();
        let id = tracker.add_feature(0.0, 0.0, 1.0);
        assert!(tracker.update_feature(id, 1.0, 1.0));
    }

    #[test]
    fn test_tracker_update_nonexistent() {
        let mut tracker = FeatureTracker::with_defaults();
        assert!(!tracker.update_feature(999, 0.0, 0.0));
    }

    #[test]
    fn test_tracker_outlier_rejection() {
        let mut tracker = FeatureTracker::new(TrackerConfig {
            max_displacement: 5.0,
            ..TrackerConfig::default()
        });
        let id = tracker.add_feature(0.0, 0.0, 1.0);
        // Move way beyond max_displacement
        assert!(!tracker.update_feature(id, 100.0, 100.0));
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_tracker_advance_frame_prunes() {
        let mut tracker = FeatureTracker::new(TrackerConfig {
            max_miss_count: 1,
            ..TrackerConfig::default()
        });
        tracker.add_feature(0.0, 0.0, 1.0);
        tracker.advance_frame();
        tracker.advance_frame();
        tracker.advance_frame();
        // After enough misses the feature should be lost
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn test_tracker_prune_lost() {
        let mut tracker = FeatureTracker::with_defaults();
        let _id = tracker.add_feature(0.0, 0.0, 1.0);
        tracker.features[0].mark_lost();
        assert_eq!(tracker.total_count(), 1);
        tracker.prune_lost();
        assert_eq!(tracker.total_count(), 0);
    }

    #[test]
    fn test_avg_track_length() {
        let mut tracker = FeatureTracker::with_defaults();
        let id1 = tracker.add_feature(0.0, 0.0, 1.0);
        tracker.update_feature(id1, 1.0, 1.0);
        let _id2 = tracker.add_feature(5.0, 5.0, 1.0);
        // id1 has length 2, id2 has length 1 => avg = 1.5
        assert!((tracker.avg_track_length() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_default_config() {
        let cfg = TrackerConfig::default();
        assert_eq!(cfg.max_features, 500);
        assert!((cfg.quality_threshold - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_active_features_list() {
        let mut tracker = FeatureTracker::with_defaults();
        tracker.add_feature(1.0, 2.0, 0.5);
        tracker.add_feature(3.0, 4.0, 0.8);
        tracker.features[0].mark_lost();
        let active = tracker.active_features();
        assert_eq!(active.len(), 1);
        assert!((active[0].x - 3.0).abs() < f64::EPSILON);
    }
}
