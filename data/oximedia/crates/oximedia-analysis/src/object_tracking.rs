//! Multi-object tracking across video frames.
//!
//! This module provides:
//! - **`IoU` computation** - Intersection over Union for bounding box overlap
//! - **Kalman tracking** - Linear motion prediction and update
//! - **Hungarian assignment** - Greedy cost-matrix assignment
//! - **Multi-object tracker** - High-level tracker with trajectory storage

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────
// TrackedObject
// ─────────────────────────────────────────────────────────────

/// A tracked object with its current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedObject {
    /// Unique object identifier
    pub id: u32,
    /// Bounding box `(x, y, width, height)` in pixels
    pub bbox: (u32, u32, u32, u32),
    /// Estimated velocity `(vx, vy)` in pixels/frame
    pub velocity: (f32, f32),
    /// Detection confidence (0.0–1.0)
    pub confidence: f32,
    /// Object class label
    pub class: String,
    /// Number of consecutive frames this object has been tracked
    pub age_frames: u32,
}

impl TrackedObject {
    /// Returns the centre of the bounding box.
    #[must_use]
    pub fn center(&self) -> (u32, u32) {
        (self.bbox.0 + self.bbox.2 / 2, self.bbox.1 + self.bbox.3 / 2)
    }

    /// Returns the area of the bounding box.
    #[must_use]
    pub fn area(&self) -> u32 {
        self.bbox.2 * self.bbox.3
    }
}

// ─────────────────────────────────────────────────────────────
// IoU
// ─────────────────────────────────────────────────────────────

/// Intersection over Union utilities.
pub struct IoU;

impl IoU {
    /// Compute the intersection-over-union of two bounding boxes.
    ///
    /// Boxes are `(x, y, width, height)`.
    #[must_use]
    pub fn compute(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> f32 {
        let ax1 = a.0;
        let ay1 = a.1;
        let ax2 = a.0 + a.2;
        let ay2 = a.1 + a.3;

        let bx1 = b.0;
        let by1 = b.1;
        let bx2 = b.0 + b.2;
        let by2 = b.1 + b.3;

        let ix1 = ax1.max(bx1);
        let iy1 = ay1.max(by1);
        let ix2 = ax2.min(bx2);
        let iy2 = ay2.min(by2);

        if ix2 <= ix1 || iy2 <= iy1 {
            return 0.0;
        }

        let inter = ((ix2 - ix1) * (iy2 - iy1)) as f32;
        let area_a = (a.2 * a.3) as f32;
        let area_b = (b.2 * b.3) as f32;
        let union = area_a + area_b - inter;
        if union <= 0.0 {
            return 0.0;
        }
        inter / union
    }
}

// ─────────────────────────────────────────────────────────────
// KalmanTracker
// ─────────────────────────────────────────────────────────────

/// A simple Kalman-like tracker for a single object.
///
/// State vector: `[x, y, w, h, vx, vy]`
#[derive(Debug, Clone)]
pub struct KalmanTracker {
    /// x coordinate
    x: f32,
    /// y coordinate
    y: f32,
    /// width
    w: f32,
    /// height
    h: f32,
    /// x velocity
    vx: f32,
    /// y velocity
    vy: f32,
    /// Number of frames without an update
    missed: u32,
}

impl KalmanTracker {
    /// Create a new tracker initialised to the given detection.
    #[must_use]
    pub fn new(det: (u32, u32, u32, u32)) -> Self {
        Self {
            x: det.0 as f32,
            y: det.1 as f32,
            w: det.2 as f32,
            h: det.3 as f32,
            vx: 0.0,
            vy: 0.0,
            missed: 0,
        }
    }

    /// Predict the next bounding box using linear motion.
    ///
    /// Returns the predicted `(x, y, w, h)` (clamped to non-negative).
    pub fn predict(&mut self) -> (u32, u32, u32, u32) {
        self.x += self.vx;
        self.y += self.vy;
        self.missed += 1;

        let x = self.x.max(0.0) as u32;
        let y = self.y.max(0.0) as u32;
        let w = self.w.max(0.0) as u32;
        let h = self.h.max(0.0) as u32;
        (x, y, w, h)
    }

    /// Update the tracker state with a new detection, blending prediction
    /// and measurement.
    pub fn update(&mut self, detection: (u32, u32, u32, u32)) {
        const ALPHA: f32 = 0.6; // blend factor towards measurement

        let new_x = detection.0 as f32;
        let new_y = detection.1 as f32;
        let new_w = detection.2 as f32;
        let new_h = detection.3 as f32;

        // Update velocity estimate
        self.vx = ALPHA * (new_x - self.x) + (1.0 - ALPHA) * self.vx;
        self.vy = ALPHA * (new_y - self.y) + (1.0 - ALPHA) * self.vy;

        // Update position / size with blending
        self.x = ALPHA * new_x + (1.0 - ALPHA) * self.x;
        self.y = ALPHA * new_y + (1.0 - ALPHA) * self.y;
        self.w = ALPHA * new_w + (1.0 - ALPHA) * self.w;
        self.h = ALPHA * new_h + (1.0 - ALPHA) * self.h;

        self.missed = 0;
    }

    /// Returns the current bounding box.
    #[must_use]
    pub fn bbox(&self) -> (u32, u32, u32, u32) {
        (
            self.x.max(0.0) as u32,
            self.y.max(0.0) as u32,
            self.w.max(0.0) as u32,
            self.h.max(0.0) as u32,
        )
    }

    /// Returns the current velocity.
    #[must_use]
    pub fn velocity(&self) -> (f32, f32) {
        (self.vx, self.vy)
    }

    /// Returns the number of frames this tracker has been without a match.
    #[must_use]
    pub fn missed_frames(&self) -> u32 {
        self.missed
    }
}

// ─────────────────────────────────────────────────────────────
// HungarianAssignment (greedy)
// ─────────────────────────────────────────────────────────────

/// Greedy assignment that approximates the Hungarian algorithm.
///
/// Finds row→column assignments that minimise total cost.
pub struct HungarianAssignment;

impl HungarianAssignment {
    /// Compute greedy assignments from a cost matrix.
    ///
    /// Returns a list of `(row, col)` pairs sorted by ascending cost.
    /// Each row and each column appears at most once.
    ///
    /// * `cost_matrix` – rows = detections, columns = trackers (or vice-versa)
    #[must_use]
    pub fn assign(cost_matrix: &[Vec<f32>]) -> Vec<(usize, usize)> {
        if cost_matrix.is_empty() {
            return Vec::new();
        }
        let rows = cost_matrix.len();
        let cols = if rows > 0 { cost_matrix[0].len() } else { 0 };
        if cols == 0 {
            return Vec::new();
        }

        // Collect all (cost, row, col) and sort by cost ascending
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
        for (r, row) in cost_matrix.iter().enumerate() {
            for (c, &cost) in row.iter().enumerate() {
                candidates.push((cost, r, c));
            }
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut assigned_rows = vec![false; rows];
        let mut assigned_cols = vec![false; cols];
        let mut assignments = Vec::new();

        for (_, r, c) in candidates {
            if !assigned_rows[r] && !assigned_cols[c] {
                assignments.push((r, c));
                assigned_rows[r] = true;
                assigned_cols[c] = true;
            }
        }

        assignments
    }
}

// ─────────────────────────────────────────────────────────────
// TrackletStore (ring-buffer trajectories)
// ─────────────────────────────────────────────────────────────

/// Stores per-object trajectory data using a fixed-capacity ring buffer.
pub struct TrackletStore {
    /// Maximum trajectory length per object
    capacity: usize,
    /// Map from object ID to centre-point history
    data: HashMap<u32, Vec<(u32, u32)>>,
}

impl TrackletStore {
    /// Create a new store with the given ring-buffer capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            data: HashMap::new(),
        }
    }

    /// Push a new centre point for the given object ID.
    pub fn push(&mut self, id: u32, center: (u32, u32)) {
        let buf = self.data.entry(id).or_default();
        if buf.len() >= self.capacity {
            buf.remove(0); // drop oldest
        }
        buf.push(center);
    }

    /// Returns the stored trajectory (centre points) for the given object ID.
    #[must_use]
    pub fn trajectory(&self, id: u32) -> Vec<(u32, u32)> {
        self.data.get(&id).cloned().unwrap_or_default()
    }

    /// Returns all tracked object IDs.
    #[must_use]
    pub fn tracked_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.data.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Returns the number of distinct objects being tracked.
    #[must_use]
    pub fn count(&self) -> usize {
        self.data.len()
    }
}

// ─────────────────────────────────────────────────────────────
// MultiObjectTracker
// ─────────────────────────────────────────────────────────────

/// High-level multi-object tracker.
///
/// Maintains a set of `KalmanTracker` instances and assigns new detections
/// using IoU-based greedy matching.
pub struct MultiObjectTracker {
    /// Active Kalman trackers, keyed by object ID
    trackers: HashMap<u32, (KalmanTracker, String, u32)>, // (tracker, class, age)
    /// Next available object ID
    next_id: u32,
    /// `IoU` threshold for matching
    iou_threshold: f32,
    /// Max frames without update before removing a tracker
    max_missed: u32,
    /// Trajectory store
    trajectories: TrackletStore,
}

impl MultiObjectTracker {
    /// Create a new tracker with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
            next_id: 0,
            iou_threshold: 0.30,
            max_missed: 5,
            trajectories: TrackletStore::new(60),
        }
    }

    /// Update the tracker with a new set of detections.
    ///
    /// * `detections` – `(x, y, w, h, class)` tuples
    ///
    /// Returns the set of currently tracked objects.
    pub fn update(&mut self, detections: &[(u32, u32, u32, u32, String)]) -> Vec<TrackedObject> {
        // Predict all existing trackers
        let mut tracker_ids: Vec<u32> = self.trackers.keys().copied().collect();
        let mut predicted: HashMap<u32, (u32, u32, u32, u32)> = HashMap::new();
        for &id in &tracker_ids {
            if let Some((tracker, _, _)) = self.trackers.get_mut(&id) {
                let bbox = tracker.predict();
                predicted.insert(id, bbox);
            }
        }

        // Build cost matrix: rows = detections, cols = existing trackers
        let n_det = detections.len();
        let n_trk = tracker_ids.len();

        let assignments = if n_det > 0 && n_trk > 0 {
            let mut cost_matrix: Vec<Vec<f32>> = Vec::with_capacity(n_det);
            for det in detections {
                let det_box = (det.0, det.1, det.2, det.3);
                let row: Vec<f32> = tracker_ids
                    .iter()
                    .map(|&tid| {
                        let trk_box = predicted[&tid];
                        // Cost = 1 - IoU (lower is better)
                        1.0 - IoU::compute(det_box, trk_box)
                    })
                    .collect();
                cost_matrix.push(row);
            }
            HungarianAssignment::assign(&cost_matrix)
        } else {
            Vec::new()
        };

        let mut matched_det = vec![false; n_det];
        let mut matched_trk = vec![false; n_trk];

        // Update matched trackers
        for (det_idx, trk_idx) in &assignments {
            let det = &detections[*det_idx];
            let tid = tracker_ids[*trk_idx];
            let cost = 1.0 - IoU::compute((det.0, det.1, det.2, det.3), predicted[&tid]);
            if cost <= (1.0 - self.iou_threshold) {
                if let Some((tracker, _, age)) = self.trackers.get_mut(&tid) {
                    tracker.update((det.0, det.1, det.2, det.3));
                    *age += 1;
                    let cx = det.0 + det.2 / 2;
                    let cy = det.1 + det.3 / 2;
                    self.trajectories.push(tid, (cx, cy));
                }
                matched_det[*det_idx] = true;
                matched_trk[*trk_idx] = true;
            }
        }

        // Create new trackers for unmatched detections
        for (i, det) in detections.iter().enumerate() {
            if !matched_det[i] {
                let id = self.next_id;
                self.next_id += 1;
                let mut tracker = KalmanTracker::new((det.0, det.1, det.2, det.3));
                tracker.missed = 0;
                self.trackers.insert(id, (tracker, det.4.clone(), 1));
                let cx = det.0 + det.2 / 2;
                let cy = det.1 + det.3 / 2;
                self.trajectories.push(id, (cx, cy));
            }
        }

        // Remove stale trackers
        for (i, &tid) in tracker_ids.iter().enumerate() {
            if !matched_trk[i] {
                let missed = self
                    .trackers
                    .get(&tid)
                    .map_or(0, |(t, _, _)| t.missed_frames());
                if missed > self.max_missed {
                    self.trackers.remove(&tid);
                }
            }
        }
        tracker_ids.retain(|id| self.trackers.contains_key(id));

        // Collect active tracked objects
        self.trackers
            .iter()
            .map(|(&id, (tracker, class, age))| {
                let bbox = tracker.bbox();
                let vel = tracker.velocity();
                TrackedObject {
                    id,
                    bbox,
                    velocity: vel,
                    confidence: 1.0
                        - (tracker.missed_frames() as f32 / (self.max_missed + 1) as f32),
                    class: class.clone(),
                    age_frames: *age,
                }
            })
            .collect()
    }

    /// Returns the trajectory (centre points) for the given object ID.
    #[must_use]
    pub fn trajectory(&self, id: u32) -> Vec<(u32, u32)> {
        self.trajectories.trajectory(id)
    }

    /// Returns the number of currently active tracks.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.trackers.len()
    }
}

impl Default for MultiObjectTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── IoU ──────────────────────────────────────────────────

    #[test]
    fn test_iou_identical_boxes() {
        let b = (10u32, 10u32, 50u32, 50u32);
        assert!((IoU::compute(b, b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = (0u32, 0u32, 10u32, 10u32);
        let b = (20u32, 20u32, 10u32, 10u32);
        assert_eq!(IoU::compute(a, b), 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = (0u32, 0u32, 20u32, 20u32); // 400px
        let b = (10u32, 0u32, 20u32, 20u32); // 400px, overlap 10×20=200
        let iou = IoU::compute(a, b);
        // intersection=200, union=400+400-200=600 → 200/600 ≈ 0.333
        assert!((iou - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_contained_box() {
        let outer = (0u32, 0u32, 100u32, 100u32);
        let inner = (25u32, 25u32, 50u32, 50u32);
        let iou = IoU::compute(outer, inner);
        // intersection=2500, union=10000 → 0.25
        assert!((iou - 0.25).abs() < 1e-5);
    }

    // ── KalmanTracker ─────────────────────────────────────────

    #[test]
    fn test_kalman_initial_state() {
        let det = (100u32, 200u32, 50u32, 60u32);
        let tracker = KalmanTracker::new(det);
        assert_eq!(tracker.bbox(), det);
        assert_eq!(tracker.velocity(), (0.0, 0.0));
    }

    #[test]
    fn test_kalman_predict_static() {
        let mut tracker = KalmanTracker::new((50, 50, 30, 30));
        let pred = tracker.predict();
        // No velocity → should stay ~same position
        assert_eq!(pred.0, 50);
        assert_eq!(pred.1, 50);
    }

    #[test]
    fn test_kalman_update_moves_towards_detection() {
        let mut tracker = KalmanTracker::new((50, 50, 30, 30));
        tracker.update((100, 100, 30, 30));
        let b = tracker.bbox();
        assert!(b.0 > 50); // moved towards 100
        assert!(b.1 > 50);
    }

    #[test]
    fn test_kalman_velocity_after_update() {
        let mut tracker = KalmanTracker::new((0, 0, 10, 10));
        tracker.update((10, 0, 10, 10));
        let (vx, _vy) = tracker.velocity();
        assert!(vx > 0.0); // positive x velocity
    }

    #[test]
    fn test_kalman_missed_frames() {
        let mut tracker = KalmanTracker::new((0, 0, 10, 10));
        tracker.predict();
        tracker.predict();
        assert_eq!(tracker.missed_frames(), 2);
        tracker.update((5, 5, 10, 10));
        assert_eq!(tracker.missed_frames(), 0);
    }

    // ── HungarianAssignment ───────────────────────────────────

    #[test]
    fn test_hungarian_empty() {
        assert!(HungarianAssignment::assign(&[]).is_empty());
    }

    #[test]
    fn test_hungarian_single() {
        let matrix = vec![vec![0.3f32]];
        let assignments = HungarianAssignment::assign(&matrix);
        assert_eq!(assignments, vec![(0, 0)]);
    }

    #[test]
    fn test_hungarian_no_duplicates() {
        let matrix = vec![
            vec![0.1f32, 0.9, 0.5],
            vec![0.9, 0.2, 0.5],
            vec![0.5, 0.5, 0.1],
        ];
        let assignments = HungarianAssignment::assign(&matrix);
        // No row or column should appear twice
        let rows: Vec<usize> = assignments.iter().map(|&(r, _)| r).collect();
        let cols: Vec<usize> = assignments.iter().map(|&(_, c)| c).collect();
        let unique_rows: std::collections::HashSet<_> = rows.iter().collect();
        let unique_cols: std::collections::HashSet<_> = cols.iter().collect();
        assert_eq!(rows.len(), unique_rows.len());
        assert_eq!(cols.len(), unique_cols.len());
    }

    #[test]
    fn test_hungarian_selects_min_cost() {
        // Optimal: (0,1) cost=0.1, (1,0) cost=0.2 → total=0.3
        let matrix = vec![vec![0.9f32, 0.1], vec![0.2, 0.8]];
        let assignments = HungarianAssignment::assign(&matrix);
        assert!(assignments.contains(&(0, 1)));
        assert!(assignments.contains(&(1, 0)));
    }

    // ── TrackletStore ─────────────────────────────────────────

    #[test]
    fn test_tracklet_store_empty_trajectory() {
        let store = TrackletStore::new(10);
        assert!(store.trajectory(99).is_empty());
    }

    #[test]
    fn test_tracklet_store_push_and_retrieve() {
        let mut store = TrackletStore::new(10);
        store.push(1, (10, 20));
        store.push(1, (15, 25));
        let traj = store.trajectory(1);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj[0], (10, 20));
    }

    #[test]
    fn test_tracklet_store_ring_buffer() {
        let mut store = TrackletStore::new(3);
        for i in 0..5u32 {
            store.push(1, (i, 0));
        }
        let traj = store.trajectory(1);
        assert_eq!(traj.len(), 3); // capped at 3
        assert_eq!(traj[0], (2, 0)); // oldest retained = i=2
    }

    #[test]
    fn test_tracklet_store_count() {
        let mut store = TrackletStore::new(10);
        store.push(1, (0, 0));
        store.push(2, (0, 0));
        assert_eq!(store.count(), 2);
    }

    // ── MultiObjectTracker ────────────────────────────────────

    #[test]
    fn test_mot_new_detections_create_tracks() {
        let mut mot = MultiObjectTracker::new();
        let dets = vec![(10u32, 10u32, 30u32, 30u32, "person".to_string())];
        let tracked = mot.update(&dets);
        assert_eq!(tracked.len(), 1);
        assert_eq!(tracked[0].class, "person");
    }

    #[test]
    fn test_mot_consistent_id_across_frames() {
        let mut mot = MultiObjectTracker::new();
        let dets = vec![(10u32, 10u32, 30u32, 30u32, "car".to_string())];
        let t1 = mot.update(&dets);
        let id1 = t1[0].id;

        // Slightly moved detection → should match
        let dets2 = vec![(12u32, 11u32, 30u32, 30u32, "car".to_string())];
        let t2 = mot.update(&dets2);
        let id2 = t2.iter().find(|o| o.class == "car").map(|o| o.id);
        assert_eq!(Some(id1), id2);
    }

    #[test]
    fn test_mot_no_detections() {
        let mut mot = MultiObjectTracker::new();
        let tracked = mot.update(&[]);
        assert_eq!(tracked.len(), 0);
    }

    #[test]
    fn test_mot_trajectory() {
        let mut mot = MultiObjectTracker::new();
        let dets = vec![(50u32, 50u32, 20u32, 20u32, "ball".to_string())];
        let t1 = mot.update(&dets);
        let id = t1[0].id;

        let dets2 = vec![(55u32, 55u32, 20u32, 20u32, "ball".to_string())];
        mot.update(&dets2);

        let traj = mot.trajectory(id);
        assert!(!traj.is_empty());
    }
}
