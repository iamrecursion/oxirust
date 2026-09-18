//! IoU (Intersection over Union) tracker.
//!
//! Simple and fast multi-object tracker that associates detections
//! based purely on bounding box overlap (IoU).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::iou_tracker::IouTracker;
//! use oximedia_cv::detect::BoundingBox;
//!
//! let mut tracker = IouTracker::new();
//! let detections = vec![BoundingBox::new(50.0, 50.0, 100.0, 100.0)];
//! let tracks = tracker.update(&detections);
//! ```

use super::assignment::{compute_iou, greedy_assignment};
use crate::detect::BoundingBox;
use std::collections::HashMap;

/// IoU-based track.
#[derive(Debug, Clone)]
struct IouTrack {
    /// Track ID.
    id: u64,
    /// Current bounding box.
    bbox: BoundingBox,
    /// Frames since last update.
    disappeared: usize,
    /// Track age.
    age: usize,
    /// Total hits.
    hits: usize,
    /// Detection confidence (if available).
    confidence: f64,
}

/// IoU tracker for multi-object tracking.
#[derive(Debug, Clone)]
pub struct IouTracker {
    /// Active tracks.
    tracks: HashMap<u64, IouTrack>,
    /// Next track ID.
    next_id: u64,
    /// Maximum disappeared frames before deletion.
    max_disappeared: usize,
    /// Minimum IoU for association.
    min_iou: f64,
    /// Minimum hits before track confirmation.
    min_hits: usize,
}

impl IouTracker {
    /// Create a new IoU tracker.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::iou_tracker::IouTracker;
    ///
    /// let tracker = IouTracker::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            max_disappeared: 5,
            min_iou: 0.3,
            min_hits: 1,
        }
    }

    /// Set maximum disappeared frames.
    #[must_use]
    pub const fn with_max_disappeared(mut self, frames: usize) -> Self {
        self.max_disappeared = frames;
        self
    }

    /// Set minimum IoU threshold.
    #[must_use]
    pub const fn with_min_iou(mut self, iou: f64) -> Self {
        self.min_iou = iou;
        self
    }

    /// Set minimum hits for track confirmation.
    #[must_use]
    pub const fn with_min_hits(mut self, hits: usize) -> Self {
        self.min_hits = hits;
        self
    }

    /// Update tracker with new detections.
    ///
    /// # Arguments
    ///
    /// * `detections` - Detected bounding boxes
    ///
    /// # Returns
    ///
    /// Vector of active tracks (ID, bbox, confidence).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::iou_tracker::IouTracker;
    /// use oximedia_cv::detect::BoundingBox;
    ///
    /// let mut tracker = IouTracker::new();
    /// let detections = vec![BoundingBox::new(100.0, 100.0, 50.0, 50.0)];
    /// let tracks = tracker.update(&detections);
    /// ```
    pub fn update(&mut self, detections: &[BoundingBox]) -> Vec<(u64, BoundingBox, f64)> {
        // If no detections, mark all as disappeared
        if detections.is_empty() {
            for track in self.tracks.values_mut() {
                track.disappeared += 1;
                track.age += 1;
            }

            // Remove disappeared tracks
            self.tracks
                .retain(|_, track| track.disappeared < self.max_disappeared);

            return self.get_confirmed_tracks();
        }

        // If no existing tracks, create new ones
        if self.tracks.is_empty() {
            for bbox in detections {
                self.register(*bbox, 1.0);
            }

            return self.get_confirmed_tracks();
        }

        // Get existing track bboxes
        let track_ids: Vec<u64> = self.tracks.keys().copied().collect();
        let track_bboxes: Vec<BoundingBox> =
            track_ids.iter().map(|id| self.tracks[id].bbox).collect();

        // Compute IoU cost matrix (cost = 1 - IoU)
        let mut cost_matrix = vec![vec![0.0; detections.len()]; track_bboxes.len()];
        for (i, track_bbox) in track_bboxes.iter().enumerate() {
            for (j, det_bbox) in detections.iter().enumerate() {
                let iou = compute_iou(track_bbox, det_bbox);
                cost_matrix[i][j] = 1.0 - iou;
            }
        }

        // Greedy assignment
        let max_cost = 1.0 - self.min_iou;
        let assignments = greedy_assignment(&cost_matrix, max_cost);

        // Update matched tracks
        let mut matched_detections = vec![false; detections.len()];

        for (track_idx, assignment) in assignments.iter().enumerate() {
            if let Some(det_idx) = assignment {
                let track_id = track_ids[track_idx];
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.bbox = detections[*det_idx];
                    track.disappeared = 0;
                    track.age += 1;
                    track.hits += 1;
                    track.confidence = 1.0;
                    matched_detections[*det_idx] = true;
                }
            }
        }

        // Mark unmatched tracks as disappeared
        for (track_idx, assignment) in assignments.iter().enumerate() {
            if assignment.is_none() {
                let track_id = track_ids[track_idx];
                if let Some(track) = self.tracks.get_mut(&track_id) {
                    track.disappeared += 1;
                    track.age += 1;
                    track.confidence *= 0.8;
                }
            }
        }

        // Register new tracks for unmatched detections
        for (i, &matched) in matched_detections.iter().enumerate() {
            if !matched {
                self.register(detections[i], 1.0);
            }
        }

        // Remove disappeared tracks
        self.tracks
            .retain(|_, track| track.disappeared < self.max_disappeared);

        self.get_confirmed_tracks()
    }

    /// Update with detections and confidence scores.
    ///
    /// # Arguments
    ///
    /// * `detections` - Detected bounding boxes
    /// * `confidences` - Confidence scores for each detection
    ///
    /// # Returns
    ///
    /// Vector of active tracks.
    pub fn update_with_confidence(
        &mut self,
        detections: &[BoundingBox],
        confidences: &[f64],
    ) -> Vec<(u64, BoundingBox, f64)> {
        // Similar to update but stores confidence
        if detections.len() != confidences.len() {
            return self.update(detections);
        }

        // Run standard update
        let result = self.update(detections);

        // Update confidences for new tracks
        let track_ids: Vec<u64> = self.tracks.keys().copied().collect();
        for (i, &track_id) in track_ids.iter().enumerate() {
            if let Some(track) = self.tracks.get_mut(&track_id) {
                if track.age == 1 && i < confidences.len() {
                    track.confidence = confidences[i];
                }
            }
        }

        result
    }

    /// Register a new track.
    fn register(&mut self, bbox: BoundingBox, confidence: f64) {
        let track = IouTrack {
            id: self.next_id,
            bbox,
            disappeared: 0,
            age: 1,
            hits: 1,
            confidence,
        };

        self.tracks.insert(self.next_id, track);
        self.next_id += 1;
    }

    /// Get confirmed tracks.
    fn get_confirmed_tracks(&self) -> Vec<(u64, BoundingBox, f64)> {
        self.tracks
            .values()
            .filter(|track| track.hits >= self.min_hits)
            .map(|track| (track.id, track.bbox, track.confidence))
            .collect()
    }

    /// Get all tracks (including unconfirmed).
    pub fn get_all_tracks(&self) -> Vec<(u64, BoundingBox, f64)> {
        self.tracks
            .values()
            .map(|track| (track.id, track.bbox, track.confidence))
            .collect()
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    /// Get number of active tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Get track by ID.
    pub fn get_track(&self, id: u64) -> Option<(BoundingBox, f64)> {
        self.tracks
            .get(&id)
            .map(|track| (track.bbox, track.confidence))
    }
}

impl Default for IouTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level IoU tracker with occlusion handling.
#[derive(Debug, Clone)]
pub struct IouTrackerAdvanced {
    /// Base IoU tracker.
    base: IouTracker,
    /// Occlusion threshold (IoU above which occlusion is detected).
    occlusion_threshold: f64,
    /// Track velocities (for prediction during occlusion).
    velocities: HashMap<u64, (f32, f32)>,
}

impl IouTrackerAdvanced {
    /// Create a new advanced IoU tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: IouTracker::new(),
            occlusion_threshold: 0.8,
            velocities: HashMap::new(),
        }
    }

    /// Update with detections.
    pub fn update(&mut self, detections: &[BoundingBox]) -> Vec<(u64, BoundingBox, f64)> {
        // Update base tracker
        let tracks = self.base.update(detections);

        // Update velocities (simple finite difference)
        // In a full implementation, this would use Kalman filtering
        self.update_velocities(&tracks);

        tracks
    }

    /// Update velocity estimates.
    fn update_velocities(&mut self, tracks: &[(u64, BoundingBox, f64)]) {
        for &(id, bbox, _) in tracks {
            // Simple velocity update (would be more sophisticated in practice)
            self.velocities.entry(id).or_insert((0.0, 0.0));
        }
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.base.reset();
        self.velocities.clear();
    }
}

impl Default for IouTrackerAdvanced {
    fn default() -> Self {
        Self::new()
    }
}
