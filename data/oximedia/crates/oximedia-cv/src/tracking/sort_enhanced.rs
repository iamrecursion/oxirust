//! Enhanced SORT (Simple Online Real-time Tracking).
//!
//! Bewley et al. 2016 — pure Rust implementation with self-contained Kalman filter.
//!
//! This module provides a complete multi-object tracker that combines:
//! - Kalman filter prediction (constant-velocity model in [cx, cy, ar, h] space)
//! - Hungarian algorithm data association using IoU cost matrix
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::sort_enhanced::{BBox, SortTrackerV2};
//!
//! let mut tracker = SortTrackerV2::new(1, 3, 0.3);
//! let detections = vec![BBox { x1: 10.0, y1: 20.0, x2: 60.0, y2: 80.0 }];
//! let tracks = tracker.update(&detections);
//! ```

// ── constants ──────────────────────────────────────────────────────────────────
const STATE_DIM: usize = 8; // [cx, cy, ar, h, vcx, vcy, var, vh]
const MEAS_DIM: usize = 4; // [cx, cy, ar, h]

// ── BBox ───────────────────────────────────────────────────────────────────────

/// Axis-aligned bounding box in (x1, y1, x2, y2) format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
}

impl BBox {
    /// Create a BBox, clamping so that x1 <= x2, y1 <= y2.
    #[must_use]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
        }
    }

    /// Box area (always >= 0).
    #[must_use]
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    /// Center (cx, cy).
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Width.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Height.
    #[must_use]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Convert to [cx, cy, aspect_ratio, height] representation.
    ///
    /// `aspect_ratio = width / height`, guarded against division by zero.
    #[must_use]
    pub fn to_xyah(&self) -> [f32; 4] {
        let (cx, cy) = self.center();
        let w = self.width();
        let h = self.height().max(1e-6);
        [cx, cy, w / h, h]
    }

    /// Intersection over Union with `other`.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);

        let inter_w = (ix2 - ix1).max(0.0);
        let inter_h = (iy2 - iy1).max(0.0);
        let inter = inter_w * inter_h;

        if inter == 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - inter;
        if union <= 0.0 {
            return 0.0;
        }
        inter / union
    }

    /// Reconstruct from [cx, cy, ar, h] measurement vector.
    #[must_use]
    fn from_xyah(xyah: &[f32; 4]) -> Self {
        let cx = xyah[0];
        let cy = xyah[1];
        let ar = xyah[2].max(1e-6);
        let h = xyah[3].max(1e-6);
        let w = ar * h;
        Self {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
        }
    }
}

// ── Inline 8×8 Kalman filter (f32) ────────────────────────────────────────────
//
// State  x  = [cx, cy, ar, h, vcx, vcy, var, vh]
// Meas   z  = [cx, cy, ar, h]
// Model  x' = F x,  F is constant-velocity (dt=1)
// Obs    z  = H x,  H picks first 4 components

/// Add two flat-layout matrices of size `n*n`.
fn mat_add_n<const N: usize>(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = a[i] + b[i];
    }
    out
}

/// Multiply two flat-layout square matrices of size `S x S`.
fn mat_mul_sq<const S: usize, const SS: usize>(a: &[f32; SS], b: &[f32; SS]) -> [f32; SS] {
    // SS must equal S*S — enforced by callers.
    let mut out = [0.0f32; SS];
    for i in 0..S {
        for j in 0..S {
            let mut sum = 0.0f32;
            for k in 0..S {
                sum += a[i * S + k] * b[k * S + j];
            }
            out[i * S + j] = sum;
        }
    }
    out
}

/// Multiply `A` (R×C) by `B` (C×K) → (R×K).
fn mat_mul_rck<const R: usize, const C: usize, const K: usize>(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; R * K];
    for i in 0..R {
        for j in 0..K {
            let mut sum = 0.0f32;
            for k in 0..C {
                sum += a[i * C + k] * b[k * K + j];
            }
            out[i * K + j] = sum;
        }
    }
    out
}

/// Transpose a flat `R×C` matrix → `C×R`.
fn mat_t<const R: usize, const C: usize>(a: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; C * R];
    for i in 0..R {
        for j in 0..C {
            out[j * R + i] = a[i * C + j];
        }
    }
    out
}

/// Matrix-vector product `A` (R×C) · `x` (C) → (R).
fn mat_vec<const R: usize, const C: usize>(a: &[f32], x: &[f32; C]) -> [f32; R] {
    let mut out = [0.0f32; R];
    for i in 0..R {
        let mut s = 0.0f32;
        for j in 0..C {
            s += a[i * C + j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Invert a small square matrix via Gauss-Jordan.  Returns None if singular.
fn mat_inv_small(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut aug = vec![0.0f32; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        // find pivot
        let mut max_r = i;
        let mut max_v = aug[i * 2 * n + i].abs();
        for k in (i + 1)..n {
            let v = aug[k * 2 * n + i].abs();
            if v > max_v {
                max_v = v;
                max_r = k;
            }
        }
        if max_v < 1e-9 {
            return None; // singular
        }
        if max_r != i {
            for j in 0..2 * n {
                aug.swap(i * 2 * n + j, max_r * 2 * n + j);
            }
        }
        let pivot = aug[i * 2 * n + i];
        for j in 0..2 * n {
            aug[i * 2 * n + j] /= pivot;
        }
        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..2 * n {
                    let v = aug[i * 2 * n + j] * factor;
                    aug[k * 2 * n + j] -= v;
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

// ── State transition F (8×8, dt=1) ────────────────────────────────────────────
//
//  | 1 0 0 0 1 0 0 0 |
//  | 0 1 0 0 0 1 0 0 |
//  | 0 0 1 0 0 0 1 0 |
//  | 0 0 0 1 0 0 0 1 |
//  | 0 0 0 0 1 0 0 0 |
//  | 0 0 0 0 0 1 0 0 |
//  | 0 0 0 0 0 0 1 0 |
//  | 0 0 0 0 0 0 0 1 |
fn state_transition() -> [f32; STATE_DIM * STATE_DIM] {
    let mut f = [0.0f32; STATE_DIM * STATE_DIM];
    for i in 0..STATE_DIM {
        f[i * STATE_DIM + i] = 1.0; // identity
    }
    // velocity coupling: position += velocity (dt=1)
    for i in 0..MEAS_DIM {
        f[i * STATE_DIM + i + MEAS_DIM] = 1.0;
    }
    f
}

// ── Measurement matrix H (4×8) ─────────────────────────────────────────────────
fn meas_matrix() -> [f32; MEAS_DIM * STATE_DIM] {
    let mut h = [0.0f32; MEAS_DIM * STATE_DIM];
    for i in 0..MEAS_DIM {
        h[i * STATE_DIM + i] = 1.0;
    }
    h
}

// Process noise Q (8×8 diagonal)
fn process_noise_q() -> [f32; STATE_DIM * STATE_DIM] {
    let mut q = [0.0f32; STATE_DIM * STATE_DIM];
    // position noise
    let pos_noise: [f32; 4] = [1.0, 1.0, 0.01, 1.0];
    // velocity noise
    let vel_noise: [f32; 4] = [0.01, 0.01, 1e-5, 0.01];
    for i in 0..MEAS_DIM {
        q[i * STATE_DIM + i] = pos_noise[i];
        q[(i + MEAS_DIM) * STATE_DIM + (i + MEAS_DIM)] = vel_noise[i];
    }
    q
}

// Measurement noise R (4×4 diagonal)
fn meas_noise_r() -> [f32; MEAS_DIM * MEAS_DIM] {
    let mut r = [0.0f32; MEAS_DIM * MEAS_DIM];
    let diag: [f32; 4] = [1.0, 1.0, 0.01, 1.0];
    for i in 0..MEAS_DIM {
        r[i * MEAS_DIM + i] = diag[i];
    }
    r
}

// ── KalmanTrack ────────────────────────────────────────────────────────────────

/// Single-object Kalman filter track.
///
/// State vector: `[cx, cy, ar, h, vcx, vcy, var, vh]`
#[derive(Debug, Clone)]
pub struct KalmanTrack {
    /// Unique track identifier.
    pub track_id: u32,
    /// Total frames this track has existed.
    pub age: u32,
    /// Total detection updates received.
    pub hits: u32,
    /// Frames elapsed since last detection update.
    pub time_since_update: u32,
    /// Kalman state vector [cx, cy, ar, h, vcx, vcy, var, vh].
    state: [f32; STATE_DIM],
    /// Kalman error covariance (STATE_DIM × STATE_DIM, row-major).
    covariance: [f32; STATE_DIM * STATE_DIM],
}

impl KalmanTrack {
    /// Create a new track from an initial detection.
    #[must_use]
    pub fn new(bbox: BBox, track_id: u32) -> Self {
        let xyah = bbox.to_xyah();
        let mut state = [0.0f32; STATE_DIM];
        state[0] = xyah[0];
        state[1] = xyah[1];
        state[2] = xyah[2];
        state[3] = xyah[3];
        // velocities start at zero

        // Initial covariance — large uncertainty on velocity
        let mut cov = [0.0f32; STATE_DIM * STATE_DIM];
        let pos_var: [f32; 4] = [10.0, 10.0, 0.01, 10.0];
        let vel_var: [f32; 4] = [100.0, 100.0, 0.0001, 100.0];
        for i in 0..MEAS_DIM {
            cov[i * STATE_DIM + i] = pos_var[i];
            cov[(i + MEAS_DIM) * STATE_DIM + (i + MEAS_DIM)] = vel_var[i];
        }

        Self {
            track_id,
            age: 1,
            hits: 1,
            time_since_update: 0,
            state,
            covariance: cov,
        }
    }

    /// Kalman predict step: advance state and covariance.
    ///
    /// Returns the predicted bounding box.
    pub fn predict(&mut self) -> BBox {
        let f = state_transition();
        let q = process_noise_q();

        // x = F x
        let new_state = mat_vec::<STATE_DIM, STATE_DIM>(&f, &self.state);
        self.state = new_state;

        // P = F P F' + Q
        let fp = mat_mul_sq::<STATE_DIM, { STATE_DIM * STATE_DIM }>(&f, &self.covariance);
        let ft = {
            let ft_v = mat_t::<STATE_DIM, STATE_DIM>(&f);
            let mut arr = [0.0f32; STATE_DIM * STATE_DIM];
            arr.copy_from_slice(&ft_v);
            arr
        };
        let fpft = mat_mul_sq::<STATE_DIM, { STATE_DIM * STATE_DIM }>(&fp, &ft);
        self.covariance = mat_add_n::<{ STATE_DIM * STATE_DIM }>(&fpft, &q);

        self.age += 1;
        self.time_since_update += 1;
        self.bbox()
    }

    /// Kalman update step with a new detection.
    pub fn update(&mut self, bbox: BBox) {
        let z = bbox.to_xyah();
        let h = meas_matrix();
        let r = meas_noise_r();

        // Innovation: y = z - H x
        let hx = mat_vec::<MEAS_DIM, STATE_DIM>(&h, &self.state);
        let mut innov = [0.0f32; MEAS_DIM];
        for i in 0..MEAS_DIM {
            innov[i] = z[i] - hx[i];
        }

        // S = H P H' + R
        let hp = mat_mul_rck::<MEAS_DIM, STATE_DIM, STATE_DIM>(&h, &self.covariance);
        let ht = mat_t::<MEAS_DIM, STATE_DIM>(&h);
        let hpht = mat_mul_rck::<MEAS_DIM, STATE_DIM, MEAS_DIM>(&hp, &ht);
        let mut s = [0.0f32; MEAS_DIM * MEAS_DIM];
        for i in 0..(MEAS_DIM * MEAS_DIM) {
            s[i] = hpht[i] + r[i];
        }

        // K = P H' inv(S)
        let pht = mat_mul_rck::<STATE_DIM, STATE_DIM, MEAS_DIM>(&self.covariance, &ht);
        let s_inv = match mat_inv_small(&s, MEAS_DIM) {
            Some(inv) => inv,
            None => {
                // Fallback: just accept the measurement directly
                self.state[0] = z[0];
                self.state[1] = z[1];
                self.state[2] = z[2];
                self.state[3] = z[3];
                self.hits += 1;
                self.time_since_update = 0;
                return;
            }
        };
        let k = mat_mul_rck::<STATE_DIM, MEAS_DIM, MEAS_DIM>(&pht, &s_inv);

        // x = x + K y
        for i in 0..STATE_DIM {
            let mut sum = 0.0f32;
            for j in 0..MEAS_DIM {
                sum += k[i * MEAS_DIM + j] * innov[j];
            }
            self.state[i] += sum;
        }

        // P = (I - K H) P
        let kh = mat_mul_rck::<STATE_DIM, MEAS_DIM, STATE_DIM>(&k, &h);
        let mut i_kh = [0.0f32; STATE_DIM * STATE_DIM];
        for i in 0..STATE_DIM {
            i_kh[i * STATE_DIM + i] = 1.0;
        }
        for i in 0..(STATE_DIM * STATE_DIM) {
            i_kh[i] -= kh[i];
        }
        let new_p_v = mat_mul_rck::<STATE_DIM, STATE_DIM, STATE_DIM>(&i_kh, &self.covariance);
        self.covariance.copy_from_slice(&new_p_v);

        self.hits += 1;
        self.time_since_update = 0;
    }

    /// Current predicted bounding box from state vector.
    #[must_use]
    pub fn bbox(&self) -> BBox {
        let xyah = [self.state[0], self.state[1], self.state[2], self.state[3]];
        BBox::from_xyah(&xyah)
    }

    /// Track is confirmed once it has received at least `min_hits` detections.
    #[must_use]
    pub fn is_confirmed(&self, min_hits: u32) -> bool {
        self.hits >= min_hits
    }

    /// Track should be deleted when it has not been updated for `max_age` frames.
    #[must_use]
    pub fn is_dead(&self, max_age: u32) -> bool {
        self.time_since_update > max_age
    }
}

// ── TrackedObject ──────────────────────────────────────────────────────────────

/// Output of one tracking frame for a single object.
#[derive(Debug, Clone)]
pub struct TrackedObject {
    /// Track identifier (stable across frames).
    pub track_id: u32,
    /// Estimated bounding box.
    pub bbox: BBox,
    /// Tracking confidence in [0, 1].
    pub confidence: f32,
    /// Number of frames since track was created.
    pub age: u32,
    /// Whether the track has been confirmed by enough detections.
    pub is_confirmed: bool,
}

// ── SortTrackerV2 ──────────────────────────────────────────────────────────────

/// SORT multi-object tracker (Bewley et al. 2016).
///
/// Combines Kalman filter prediction with Hungarian-algorithm assignment.
#[derive(Debug, Clone)]
pub struct SortTrackerV2 {
    tracks: Vec<KalmanTrack>,
    next_id: u32,
    /// Frames without detection before a track is deleted.
    pub max_age: u32,
    /// Detection hits required before a track is reported.
    pub min_hits: u32,
    /// Minimum IoU to consider a detection–track pair compatible.
    pub iou_threshold: f32,
}

impl SortTrackerV2 {
    /// Create a tracker with explicit hyper-parameters.
    ///
    /// | parameter | typical default |
    /// |-----------|----------------|
    /// | `max_age` | 1              |
    /// | `min_hits`| 3              |
    /// | `iou_threshold` | 0.3     |
    #[must_use]
    pub fn new(max_age: u32, min_hits: u32, iou_threshold: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            min_hits,
            iou_threshold,
        }
    }

    /// Create a tracker with SORT paper defaults.
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(1, 3, 0.3)
    }

    /// Update tracker with new detections for the current frame.
    ///
    /// Returns all tracks that are currently active and confirmed.
    pub fn update(&mut self, detections: &[BBox]) -> Vec<TrackedObject> {
        // Step 1: predict all existing tracks
        let mut predicted_bboxes: Vec<BBox> = self.tracks.iter_mut().map(|t| t.predict()).collect();

        // Step 2: associate detections to tracks
        let (matched, unmatched_tracks, unmatched_dets) =
            self.associate(&predicted_bboxes, detections);

        // Step 3: update matched tracks
        for (track_idx, det_idx) in &matched {
            self.tracks[*track_idx].update(detections[*det_idx]);
        }

        // Step 4: mark unmatched tracks (time_since_update was already incremented in predict)
        // nothing extra needed — predict() already incremented time_since_update

        // Step 5: create new tracks for unmatched detections
        for &det_idx in &unmatched_dets {
            let track = KalmanTrack::new(detections[det_idx], self.next_id);
            self.next_id += 1;
            self.tracks.push(track);
        }

        // Step 6: remove dead tracks
        let max_age = self.max_age;
        self.tracks.retain(|t| !t.is_dead(max_age));

        // Step 7: collect output
        let min_hits = self.min_hits;
        self.tracks
            .iter()
            .filter(|t| t.is_confirmed(min_hits) || t.time_since_update == 0)
            .map(|t| TrackedObject {
                track_id: t.track_id,
                bbox: t.bbox(),
                confidence: 1.0 / (1.0 + t.time_since_update as f32),
                age: t.age,
                is_confirmed: t.is_confirmed(min_hits),
            })
            .collect()
    }

    /// Hungarian-algorithm data association.
    ///
    /// Returns `(matched_pairs, unmatched_track_indices, unmatched_det_indices)`.
    fn associate(
        &self,
        predicted: &[BBox],
        detections: &[BBox],
    ) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        if predicted.is_empty() {
            let unmatched_dets = (0..detections.len()).collect();
            return (Vec::new(), Vec::new(), unmatched_dets);
        }
        if detections.is_empty() {
            let unmatched_tracks = (0..predicted.len()).collect();
            return (Vec::new(), unmatched_tracks, Vec::new());
        }

        // Build cost matrix (cost = 1 − IoU)
        let n_t = predicted.len();
        let n_d = detections.len();
        let mut cost_f64 = vec![vec![1.0f64; n_d]; n_t];
        for (i, pb) in predicted.iter().enumerate() {
            for (j, db) in detections.iter().enumerate() {
                let iou = pb.iou(db);
                cost_f64[i][j] = 1.0 - iou as f64;
            }
        }

        // Reuse crate's Hungarian algorithm
        let assignments = crate::tracking::assignment::hungarian_algorithm(&cost_f64);
        let filtered = crate::tracking::assignment::filter_assignments_by_cost(
            &assignments,
            &cost_f64,
            1.0 - self.iou_threshold as f64,
        );

        let mut matched = Vec::new();
        let mut unmatched_tracks = Vec::new();
        let mut det_used = vec![false; n_d];

        for (t_idx, assignment) in filtered.iter().enumerate() {
            if let Some(d_idx) = assignment {
                matched.push((t_idx, *d_idx));
                det_used[*d_idx] = true;
            } else {
                unmatched_tracks.push(t_idx);
            }
        }

        let unmatched_dets: Vec<usize> = (0..n_d).filter(|&i| !det_used[i]).collect();
        (matched, unmatched_tracks, unmatched_dets)
    }

    /// References to all active `KalmanTrack` objects.
    #[must_use]
    pub fn active_tracks(&self) -> Vec<&KalmanTrack> {
        self.tracks.iter().collect()
    }

    /// Number of currently maintained tracks (confirmed + tentative).
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Reset tracker to empty state.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }
}

impl Default for SortTrackerV2 {
    fn default() -> Self {
        Self::default_params()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BBox tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_area_zero_for_degenerate() {
        let b = BBox::new(5.0, 5.0, 5.0, 5.0);
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn test_bbox_area_positive() {
        let b = BBox::new(0.0, 0.0, 4.0, 3.0);
        assert!((b.area() - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_center() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        let (cx, cy) = b.center();
        assert!((cx - 5.0).abs() < 1e-5);
        assert!((cy - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_iou_identical() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = BBox::new(0.0, 0.0, 5.0, 5.0);
        let b = BBox::new(10.0, 10.0, 15.0, 15.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_bbox_iou_partial_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        let iou = a.iou(&b);
        // intersection = 5*5=25, union = 100+100-25=175
        let expected = 25.0 / 175.0;
        assert!((iou - expected).abs() < 1e-4);
    }

    #[test]
    fn test_bbox_to_xyah_roundtrip() {
        let b = BBox::new(10.0, 20.0, 50.0, 80.0);
        let xyah = b.to_xyah();
        let b2 = BBox::from_xyah(&xyah);
        assert!((b.x1 - b2.x1).abs() < 1e-3);
        assert!((b.y1 - b2.y1).abs() < 1e-3);
        assert!((b.x2 - b2.x2).abs() < 1e-3);
        assert!((b.y2 - b2.y2).abs() < 1e-3);
    }

    #[test]
    fn test_bbox_new_clamps_order() {
        let b = BBox::new(10.0, 10.0, 5.0, 5.0);
        assert!(b.x1 <= b.x2);
        assert!(b.y1 <= b.y2);
    }

    // ── KalmanTrack tests ──────────────────────────────────────────────────────

    #[test]
    fn test_kalman_track_new_bbox() {
        let bbox = BBox::new(100.0, 100.0, 200.0, 200.0);
        let track = KalmanTrack::new(bbox, 1);
        let estimated = track.bbox();
        // Center should be close to (150, 150)
        let (cx, cy) = estimated.center();
        assert!((cx - 150.0).abs() < 1.0);
        assert!((cy - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_kalman_track_predict_increments_age() {
        let bbox = BBox::new(0.0, 0.0, 50.0, 50.0);
        let mut track = KalmanTrack::new(bbox, 1);
        let age_before = track.age;
        track.predict();
        assert_eq!(track.age, age_before + 1);
    }

    #[test]
    fn test_kalman_track_update_resets_time_since_update() {
        let bbox = BBox::new(0.0, 0.0, 50.0, 50.0);
        let mut track = KalmanTrack::new(bbox, 1);
        track.predict(); // time_since_update becomes 1
        assert_eq!(track.time_since_update, 1);
        track.update(bbox); // should reset to 0
        assert_eq!(track.time_since_update, 0);
    }

    #[test]
    fn test_kalman_track_is_dead() {
        let bbox = BBox::new(0.0, 0.0, 50.0, 50.0);
        let mut track = KalmanTrack::new(bbox, 42);
        assert!(!track.is_dead(1));
        track.predict(); // time_since_update = 1
        assert!(!track.is_dead(1)); // 1 > 1 is false
        track.predict(); // time_since_update = 2
        assert!(track.is_dead(1)); // 2 > 1 is true
    }

    #[test]
    fn test_kalman_track_is_confirmed() {
        let bbox = BBox::new(0.0, 0.0, 50.0, 50.0);
        let mut track = KalmanTrack::new(bbox, 1); // hits=1
        assert!(!track.is_confirmed(3));
        track.update(bbox); // hits=2
        assert!(!track.is_confirmed(3));
        track.update(bbox); // hits=3
        assert!(track.is_confirmed(3));
    }

    // ── SortTrackerV2 tests ────────────────────────────────────────────────────

    #[test]
    fn test_sort_tracker_empty_detections() {
        let mut tracker = SortTrackerV2::new(1, 1, 0.3);
        let tracks = tracker.update(&[]);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_sort_tracker_single_detection() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        let dets = vec![BBox::new(0.0, 0.0, 100.0, 100.0)];
        let tracks = tracker.update(&dets);
        // min_hits=1 → track is confirmed immediately
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].track_id, 1);
    }

    #[test]
    fn test_sort_tracker_consistent_id_across_frames() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        let bbox = BBox::new(100.0, 100.0, 150.0, 150.0);
        // First frame
        let t1 = tracker.update(&[bbox]);
        assert_eq!(t1.len(), 1);
        let id = t1[0].track_id;
        // Second frame — same position
        let t2 = tracker.update(&[bbox]);
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].track_id, id);
    }

    #[test]
    fn test_sort_tracker_new_id_for_non_overlapping() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        let bbox1 = BBox::new(0.0, 0.0, 50.0, 50.0);
        tracker.update(&[bbox1]);
        // Second frame: detection far away — should spawn new track
        let bbox2 = BBox::new(500.0, 500.0, 550.0, 550.0);
        let tracks = tracker.update(&[bbox2]);
        // id 1 may be dropped or still alive; id 2 definitely spawned
        let ids: Vec<u32> = tracks.iter().map(|t| t.track_id).collect();
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_sort_tracker_dead_track_removed() {
        let mut tracker = SortTrackerV2::new(1, 1, 0.3); // max_age=1
        let bbox = BBox::new(0.0, 0.0, 50.0, 50.0);
        tracker.update(&[bbox]);
        // No detections for 2 frames → track should die
        tracker.update(&[]);
        tracker.update(&[]);
        let tracks = tracker.update(&[]);
        assert!(tracks.is_empty());
    }

    #[test]
    fn test_sort_tracker_track_count() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        let dets = vec![
            BBox::new(0.0, 0.0, 50.0, 50.0),
            BBox::new(200.0, 200.0, 250.0, 250.0),
        ];
        tracker.update(&dets);
        assert_eq!(tracker.track_count(), 2);
    }

    #[test]
    fn test_sort_tracker_reset() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        tracker.update(&[BBox::new(0.0, 0.0, 50.0, 50.0)]);
        tracker.reset();
        assert_eq!(tracker.track_count(), 0);
    }

    #[test]
    fn test_sort_tracker_active_tracks() {
        let mut tracker = SortTrackerV2::new(5, 1, 0.3);
        tracker.update(&[BBox::new(0.0, 0.0, 100.0, 100.0)]);
        assert_eq!(tracker.active_tracks().len(), 1);
    }
}
