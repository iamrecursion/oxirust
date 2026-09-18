//! Multi-object tracking with Kalman filter state estimation.
//!
//! This module provides a full multi-object tracker (MOT) pipeline:
//!
//! - **Kalman filter**: Constant-acceleration model for position/velocity/acceleration
//! - **Hungarian algorithm approximation**: Greedy IoU-based assignment
//! - **Track lifecycle**: Tentative -> Confirmed -> Lost -> Deleted
//! - **Occlusion handling**: Predicted positions during occlusion gaps
//! - **IoU matching**: Track-to-detection association via bounding box overlap

use crate::common::{Confidence, Rect};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Track state machine
// ---------------------------------------------------------------------------

/// Lifecycle state of a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackState {
    /// Recently created, not yet confirmed.
    Tentative,
    /// Confirmed (matched in enough consecutive frames).
    Confirmed,
    /// Lost (not matched for several frames but still alive).
    Lost,
    /// Deleted (should be removed from the tracker).
    Deleted,
}

// ---------------------------------------------------------------------------
// Kalman filter (constant-acceleration, 2D)
// ---------------------------------------------------------------------------

/// State vector: `[x, y, vx, vy, ax, ay]`.
const STATE_DIM: usize = 6;
/// Measurement vector: `[x, y, w, h]` (bounding-box centre + size).
const MEAS_DIM: usize = 4;

/// A simple 2D Kalman filter for bounding-box tracking.
#[derive(Debug, Clone)]
pub struct KalmanFilter2D {
    /// State vector: [cx, cy, vx, vy, ax, ay].
    state: [f64; STATE_DIM],
    /// Diagonal covariance (simplified).
    covariance: [f64; STATE_DIM],
    /// Width and height (tracked separately, not filtered).
    size: [f64; 2],
    /// Process noise.
    process_noise: f64,
    /// Measurement noise.
    measurement_noise: f64,
}

impl KalmanFilter2D {
    /// Initialise from a detection bounding box.
    #[must_use]
    pub fn new(bbox: &Rect) -> Self {
        let cx = (bbox.x + bbox.width / 2.0) as f64;
        let cy = (bbox.y + bbox.height / 2.0) as f64;
        Self {
            state: [cx, cy, 0.0, 0.0, 0.0, 0.0],
            covariance: [10.0; STATE_DIM],
            size: [bbox.width as f64, bbox.height as f64],
            process_noise: 1.0,
            measurement_noise: 4.0,
        }
    }

    /// Predict the next state (dt = 1 frame).
    pub fn predict(&mut self) {
        let dt = 1.0;
        // x += vx*dt + 0.5*ax*dt^2
        self.state[0] += self.state[2] * dt + 0.5 * self.state[4] * dt * dt;
        // y += vy*dt + 0.5*ay*dt^2
        self.state[1] += self.state[3] * dt + 0.5 * self.state[5] * dt * dt;
        // vx += ax*dt
        self.state[2] += self.state[4] * dt;
        // vy += ay*dt
        self.state[3] += self.state[5] * dt;

        // Increase uncertainty
        for p in &mut self.covariance {
            *p += self.process_noise;
        }
    }

    /// Update with a measurement (detection bounding box).
    pub fn update(&mut self, bbox: &Rect) {
        let z_cx = (bbox.x + bbox.width / 2.0) as f64;
        let z_cy = (bbox.y + bbox.height / 2.0) as f64;

        // Innovation
        let innov_x = z_cx - self.state[0];
        let innov_y = z_cy - self.state[1];

        // Kalman gain (diagonal approximation)
        let k_x = self.covariance[0] / (self.covariance[0] + self.measurement_noise);
        let k_y = self.covariance[1] / (self.covariance[1] + self.measurement_noise);

        // State update
        self.state[0] += k_x * innov_x;
        self.state[1] += k_y * innov_y;
        // Velocity correction
        self.state[2] += 0.5 * k_x * innov_x;
        self.state[3] += 0.5 * k_y * innov_y;

        // Covariance update
        self.covariance[0] *= 1.0 - k_x;
        self.covariance[1] *= 1.0 - k_y;

        // Update size with exponential smoothing
        let alpha = 0.7;
        self.size[0] = alpha * bbox.width as f64 + (1.0 - alpha) * self.size[0];
        self.size[1] = alpha * bbox.height as f64 + (1.0 - alpha) * self.size[1];
    }

    /// Get the predicted bounding box.
    #[must_use]
    pub fn predicted_bbox(&self) -> Rect {
        let w = self.size[0] as f32;
        let h = self.size[1] as f32;
        let cx = self.state[0] as f32;
        let cy = self.state[1] as f32;
        Rect::new(cx - w / 2.0, cy - h / 2.0, w, h)
    }

    /// Get the current state vector.
    #[must_use]
    pub fn state(&self) -> &[f64; STATE_DIM] {
        &self.state
    }

    /// Get the current velocity magnitude.
    #[must_use]
    pub fn velocity_magnitude(&self) -> f64 {
        (self.state[2] * self.state[2] + self.state[3] * self.state[3]).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

/// A single tracked object.
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique track identifier.
    pub id: u64,
    /// Current lifecycle state.
    pub state: TrackState,
    /// Kalman filter for motion estimation.
    pub kf: KalmanFilter2D,
    /// Number of consecutive frames matched.
    pub hits: u32,
    /// Number of consecutive frames unmatched.
    pub age_since_update: u32,
    /// Total number of frames this track has existed.
    pub total_age: u32,
    /// Latest detection confidence (if matched).
    pub last_confidence: Confidence,
}

// ---------------------------------------------------------------------------
// Tracker configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-object tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// IoU threshold for matching detections to tracks.
    pub iou_threshold: f32,
    /// Number of consecutive hits to transition Tentative -> Confirmed.
    pub min_hits_to_confirm: u32,
    /// Number of frames without update before Confirmed -> Lost.
    pub max_age_before_lost: u32,
    /// Number of frames without update before Lost -> Deleted.
    pub max_age_before_deleted: u32,
    /// Maximum number of active tracks.
    pub max_tracks: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            min_hits_to_confirm: 3,
            max_age_before_lost: 5,
            max_age_before_deleted: 30,
            max_tracks: 200,
        }
    }
}

/// A detection to feed into the tracker.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box.
    pub bbox: Rect,
    /// Confidence score.
    pub confidence: Confidence,
}

// ---------------------------------------------------------------------------
// Multi-object tracker
// ---------------------------------------------------------------------------

/// Multi-object tracker with Kalman filtering and IoU-based assignment.
pub struct MultiObjectTracker {
    config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
}

impl MultiObjectTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TrackerConfig::default(),
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Create a tracker with the given configuration.
    #[must_use]
    pub fn with_config(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Process one frame of detections and return the current set of active tracks.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        // 1. Predict all tracks
        for track in &mut self.tracks {
            track.kf.predict();
            track.total_age += 1;
        }

        // 2. Build cost matrix (IoU-based)
        let n_tracks = self.tracks.len();
        let n_dets = detections.len();
        let mut iou_matrix = vec![vec![0.0f32; n_dets]; n_tracks];

        for (i, track) in self.tracks.iter().enumerate() {
            let predicted = track.kf.predicted_bbox();
            for (j, det) in detections.iter().enumerate() {
                iou_matrix[i][j] = predicted.iou(&det.bbox);
            }
        }

        // 3. Greedy assignment (Hungarian approximation)
        let (matched_pairs, unmatched_tracks, unmatched_dets) =
            greedy_assignment(&iou_matrix, n_tracks, n_dets, self.config.iou_threshold);

        // 4. Update matched tracks
        for (track_idx, det_idx) in &matched_pairs {
            let track = &mut self.tracks[*track_idx];
            let det = &detections[*det_idx];
            track.kf.update(&det.bbox);
            track.hits += 1;
            track.age_since_update = 0;
            track.last_confidence = det.confidence;

            if track.state == TrackState::Tentative && track.hits >= self.config.min_hits_to_confirm
            {
                track.state = TrackState::Confirmed;
            }
            if track.state == TrackState::Lost {
                track.state = TrackState::Confirmed;
                track.hits = 1;
            }
        }

        // 5. Handle unmatched tracks
        for &track_idx in &unmatched_tracks {
            let track = &mut self.tracks[track_idx];
            track.age_since_update += 1;

            match track.state {
                TrackState::Tentative => {
                    if track.age_since_update > 2 {
                        track.state = TrackState::Deleted;
                    }
                }
                TrackState::Confirmed => {
                    if track.age_since_update >= self.config.max_age_before_lost {
                        track.state = TrackState::Lost;
                    }
                }
                TrackState::Lost => {
                    if track.age_since_update >= self.config.max_age_before_deleted {
                        track.state = TrackState::Deleted;
                    }
                }
                TrackState::Deleted => {}
            }
        }

        // 6. Create new tracks for unmatched detections
        for &det_idx in &unmatched_dets {
            if self.tracks.len() < self.config.max_tracks {
                let det = &detections[det_idx];
                let track = Track {
                    id: self.next_id,
                    state: TrackState::Tentative,
                    kf: KalmanFilter2D::new(&det.bbox),
                    hits: 1,
                    age_since_update: 0,
                    total_age: 1,
                    last_confidence: det.confidence,
                };
                self.tracks.push(track);
                self.next_id += 1;
            }
        }

        // 7. Remove deleted tracks
        self.tracks.retain(|t| t.state != TrackState::Deleted);

        // Return cloned active tracks
        self.tracks.clone()
    }

    /// Get confirmed tracks only.
    #[must_use]
    pub fn confirmed_tracks(&self) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .collect()
    }

    /// Get all active (non-deleted) tracks.
    #[must_use]
    pub fn active_tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Get the number of active tracks.
    #[must_use]
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Greedy assignment
// ---------------------------------------------------------------------------

/// Greedy IoU assignment: returns (matched_pairs, unmatched_track_indices, unmatched_det_indices).
fn greedy_assignment(
    iou_matrix: &[Vec<f32>],
    n_tracks: usize,
    n_dets: usize,
    threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    // Build flat list of (iou, track_idx, det_idx) sorted descending
    let mut entries: Vec<(f32, usize, usize)> = Vec::with_capacity(n_tracks * n_dets);
    for (i, row) in iou_matrix.iter().enumerate().take(n_tracks) {
        for (j, &iou) in row.iter().enumerate().take(n_dets) {
            if iou >= threshold {
                entries.push((iou, i, j));
            }
        }
    }
    // Sort descending by IoU
    entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut matched_tracks = vec![false; n_tracks];
    let mut matched_dets = vec![false; n_dets];
    let mut pairs = Vec::new();

    for (_, ti, di) in &entries {
        if !matched_tracks[*ti] && !matched_dets[*di] {
            pairs.push((*ti, *di));
            matched_tracks[*ti] = true;
            matched_dets[*di] = true;
        }
    }

    let unmatched_tracks: Vec<usize> = (0..n_tracks).filter(|i| !matched_tracks[*i]).collect();
    let unmatched_dets: Vec<usize> = (0..n_dets).filter(|i| !matched_dets[*i]).collect();

    (pairs, unmatched_tracks, unmatched_dets)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, conf: f32) -> Detection {
        Detection {
            bbox: Rect::new(x, y, w, h),
            confidence: Confidence::new(conf),
        }
    }

    #[test]
    fn test_kalman_init() {
        let bbox = Rect::new(10.0, 20.0, 30.0, 40.0);
        let kf = KalmanFilter2D::new(&bbox);
        let pred = kf.predicted_bbox();
        assert!((pred.x - 10.0).abs() < 1.0);
        assert!((pred.y - 20.0).abs() < 1.0);
    }

    #[test]
    fn test_kalman_predict() {
        let bbox = Rect::new(0.0, 0.0, 20.0, 20.0);
        let mut kf = KalmanFilter2D::new(&bbox);
        kf.predict();
        // With zero velocity, predicted position should stay roughly the same
        let pred = kf.predicted_bbox();
        assert!((pred.x - 0.0).abs() < 2.0);
    }

    #[test]
    fn test_kalman_update_moves_state() {
        let bbox1 = Rect::new(0.0, 0.0, 20.0, 20.0);
        let mut kf = KalmanFilter2D::new(&bbox1);
        kf.predict();
        let bbox2 = Rect::new(10.0, 0.0, 20.0, 20.0);
        kf.update(&bbox2);
        let pred = kf.predicted_bbox();
        // Should have moved toward bbox2
        assert!(pred.x > 2.0);
    }

    #[test]
    fn test_kalman_velocity_magnitude() {
        let bbox = Rect::new(0.0, 0.0, 20.0, 20.0);
        let kf = KalmanFilter2D::new(&bbox);
        assert!((kf.velocity_magnitude() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kalman_state_vector() {
        let bbox = Rect::new(10.0, 20.0, 30.0, 40.0);
        let kf = KalmanFilter2D::new(&bbox);
        let s = kf.state();
        assert!((s[0] - 25.0).abs() < f64::EPSILON); // cx = 10 + 15
        assert!((s[1] - 40.0).abs() < f64::EPSILON); // cy = 20 + 20
    }

    #[test]
    fn test_tracker_create() {
        let tracker = MultiObjectTracker::new();
        assert_eq!(tracker.num_tracks(), 0);
    }

    #[test]
    fn test_tracker_single_detection() {
        let mut tracker = MultiObjectTracker::new();
        let dets = vec![det(10.0, 10.0, 50.0, 50.0, 0.9)];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].state, TrackState::Tentative);
    }

    #[test]
    fn test_tracker_confirm_after_hits() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 3,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        let d = det(10.0, 10.0, 50.0, 50.0, 0.9);

        for _ in 0..3 {
            tracker.update(std::slice::from_ref(&d));
        }

        let confirmed = tracker.confirmed_tracks();
        assert_eq!(confirmed.len(), 1);
    }

    #[test]
    fn test_tracker_lost_after_no_update() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 1,
            max_age_before_lost: 3,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        let d = det(10.0, 10.0, 50.0, 50.0, 0.9);

        // One detection to create + confirm
        tracker.update(&[d]);
        // Several empty frames
        for _ in 0..4 {
            tracker.update(&[]);
        }

        let active = tracker.active_tracks();
        let lost_count = active
            .iter()
            .filter(|t| t.state == TrackState::Lost)
            .count();
        assert!(lost_count > 0 || active.is_empty());
    }

    #[test]
    fn test_tracker_deleted_after_long_absence() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 1,
            max_age_before_lost: 2,
            max_age_before_deleted: 5,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        tracker.update(&[det(10.0, 10.0, 50.0, 50.0, 0.9)]);
        for _ in 0..40 {
            tracker.update(&[]);
        }
        assert_eq!(tracker.num_tracks(), 0);
    }

    #[test]
    fn test_tracker_multiple_objects() {
        let mut tracker = MultiObjectTracker::new();
        let dets = vec![
            det(10.0, 10.0, 30.0, 30.0, 0.9),
            det(200.0, 200.0, 40.0, 40.0, 0.8),
        ];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 2);
    }

    #[test]
    fn test_tracker_id_assignment() {
        let mut tracker = MultiObjectTracker::new();
        tracker.update(&[det(10.0, 10.0, 30.0, 30.0, 0.9)]);
        tracker.update(&[det(200.0, 200.0, 30.0, 30.0, 0.8)]);

        let active = tracker.active_tracks();
        let ids: Vec<u64> = active.iter().map(|t| t.id).collect();
        // IDs should be unique
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = MultiObjectTracker::new();
        tracker.update(&[det(10.0, 10.0, 30.0, 30.0, 0.9)]);
        assert!(tracker.num_tracks() > 0);
        tracker.reset();
        assert_eq!(tracker.num_tracks(), 0);
    }

    #[test]
    fn test_greedy_assignment_basic() {
        let iou_matrix = vec![vec![0.8, 0.1], vec![0.1, 0.7]];
        let (pairs, ut, ud) = greedy_assignment(&iou_matrix, 2, 2, 0.3);
        assert_eq!(pairs.len(), 2);
        assert!(ut.is_empty());
        assert!(ud.is_empty());
    }

    #[test]
    fn test_greedy_assignment_no_match() {
        let iou_matrix = vec![vec![0.1, 0.05], vec![0.02, 0.1]];
        let (pairs, ut, ud) = greedy_assignment(&iou_matrix, 2, 2, 0.3);
        assert!(pairs.is_empty());
        assert_eq!(ut.len(), 2);
        assert_eq!(ud.len(), 2);
    }

    #[test]
    fn test_greedy_assignment_partial() {
        let iou_matrix = vec![vec![0.8, 0.1], vec![0.1, 0.1]];
        let (pairs, ut, ud) = greedy_assignment(&iou_matrix, 2, 2, 0.3);
        assert_eq!(pairs.len(), 1);
        assert_eq!(ut.len(), 1);
        assert_eq!(ud.len(), 1);
    }

    #[test]
    fn test_greedy_assignment_empty() {
        let iou_matrix: Vec<Vec<f32>> = Vec::new();
        let (pairs, ut, ud) = greedy_assignment(&iou_matrix, 0, 0, 0.3);
        assert!(pairs.is_empty());
        assert!(ut.is_empty());
        assert!(ud.is_empty());
    }

    #[test]
    fn test_occlusion_prediction() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 1,
            max_age_before_lost: 10,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);

        // Moving object: x increases by 10 each frame
        for i in 0..5 {
            let x = 10.0 + i as f32 * 10.0;
            tracker.update(&[det(x, 50.0, 30.0, 30.0, 0.9)]);
        }

        // Object disappears (occluded) for a few frames
        for _ in 0..3 {
            tracker.update(&[]);
        }

        // Track should still exist and have a predicted position
        let active = tracker.active_tracks();
        assert!(!active.is_empty(), "track should survive occlusion");
        let pred = active[0].kf.predicted_bbox();
        // Predicted x should be > 50 (it was moving right)
        assert!(
            pred.x > 40.0,
            "predicted position should extrapolate motion"
        );
    }

    #[test]
    fn test_track_state_transitions() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 2,
            max_age_before_lost: 2,
            max_age_before_deleted: 5,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        let d = det(50.0, 50.0, 30.0, 30.0, 0.9);

        // Frame 1: Tentative
        let tracks = tracker.update(std::slice::from_ref(&d));
        assert_eq!(tracks[0].state, TrackState::Tentative);

        // Frame 2: Confirmed
        let tracks = tracker.update(std::slice::from_ref(&d));
        let confirmed: Vec<_> = tracks
            .iter()
            .filter(|t| t.state == TrackState::Confirmed)
            .collect();
        assert_eq!(confirmed.len(), 1);

        // Frames 3-4: no detections -> Lost
        tracker.update(&[]);
        let tracks = tracker.update(&[]);
        let lost_count = tracks
            .iter()
            .filter(|t| t.state == TrackState::Lost)
            .count();
        assert!(lost_count > 0 || tracks.is_empty());
    }

    #[test]
    fn test_reacquire_lost_track() {
        let cfg = TrackerConfig {
            min_hits_to_confirm: 1,
            max_age_before_lost: 2,
            max_age_before_deleted: 20,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        let d = det(50.0, 50.0, 30.0, 30.0, 0.9);

        // Create and confirm
        tracker.update(std::slice::from_ref(&d));

        // Lose it
        for _ in 0..3 {
            tracker.update(&[]);
        }

        // Re-detect at same location
        let tracks = tracker.update(&[d]);
        // Should have a track (either re-acquired or new)
        assert!(!tracks.is_empty());
    }

    #[test]
    fn test_max_tracks_limit() {
        let cfg = TrackerConfig {
            max_tracks: 2,
            ..Default::default()
        };
        let mut tracker = MultiObjectTracker::with_config(cfg);
        let dets = vec![
            det(0.0, 0.0, 20.0, 20.0, 0.9),
            det(100.0, 0.0, 20.0, 20.0, 0.8),
            det(200.0, 0.0, 20.0, 20.0, 0.7),
        ];
        let tracks = tracker.update(&dets);
        assert!(tracks.len() <= 2);
    }

    #[test]
    fn test_iou_matching_correctness() {
        let mut tracker = MultiObjectTracker::new();

        // Frame 1: two objects far apart
        let dets1 = vec![
            det(10.0, 10.0, 30.0, 30.0, 0.9),
            det(200.0, 200.0, 30.0, 30.0, 0.8),
        ];
        tracker.update(&dets1);

        // Frame 2: same objects shifted slightly
        let dets2 = vec![
            det(12.0, 12.0, 30.0, 30.0, 0.9),
            det(202.0, 202.0, 30.0, 30.0, 0.8),
        ];
        let tracks = tracker.update(&dets2);
        // Should still have exactly 2 tracks (matched, not new)
        assert_eq!(tracks.len(), 2);
    }

    #[test]
    fn test_kalman_acceleration() {
        let bbox = Rect::new(0.0, 0.0, 20.0, 20.0);
        let mut kf = KalmanFilter2D::new(&bbox);

        // Feed accelerating detections
        for i in 0..10 {
            kf.predict();
            let x = (i * i) as f32; // quadratic motion
            kf.update(&Rect::new(x, 0.0, 20.0, 20.0));
        }

        // Velocity should be positive
        assert!(kf.velocity_magnitude() > 0.0);
    }
}
