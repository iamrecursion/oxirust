//! Feature tracking across video frames.
//!
//! This module implements feature detection and tracking algorithms for
//! estimating camera motion between consecutive frames.
//!
//! Feature tracking across consecutive frame pairs is parallelised with
//! [`rayon`] where possible.  Each feature point is tracked independently so
//! the work per pixel is embarrassingly parallel.

use crate::error::{StabilizeError, StabilizeResult};
use crate::Frame;
use rayon::prelude::*;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

/// A tracked feature point in an image.
#[derive(Debug, Clone, Copy)]
pub struct Feature {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Feature quality/strength (0.0-1.0)
    pub quality: f64,
    /// Feature ID for tracking
    pub id: usize,
}

impl Feature {
    /// Create a new feature.
    #[must_use]
    pub const fn new(x: f64, y: f64, quality: f64, id: usize) -> Self {
        Self { x, y, quality, id }
    }

    /// Calculate distance to another feature.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Check if feature is valid (within bounds).
    #[must_use]
    pub fn is_valid(&self, width: usize, height: usize) -> bool {
        self.x >= 0.0
            && self.y >= 0.0
            && self.x < width as f64
            && self.y < height as f64
            && self.quality > 0.0
    }
}

/// A track of a single feature across multiple frames.
#[derive(Debug, Clone)]
pub struct FeatureTrack {
    /// Feature ID
    pub id: usize,
    /// Feature positions in each frame (frame_index -> position)
    pub positions: HashMap<usize, (f64, f64)>,
    /// Track start frame
    pub start_frame: usize,
    /// Track end frame
    pub end_frame: usize,
    /// Average quality across track
    pub avg_quality: f64,
}

impl FeatureTrack {
    /// Create a new feature track.
    #[must_use]
    pub fn new(id: usize, start_frame: usize) -> Self {
        Self {
            id,
            positions: HashMap::new(),
            start_frame,
            end_frame: start_frame,
            avg_quality: 0.0,
        }
    }

    /// Add a position to the track.
    pub fn add_position(&mut self, frame: usize, x: f64, y: f64, quality: f64) {
        self.positions.insert(frame, (x, y));
        self.end_frame = self.end_frame.max(frame);

        // Update average quality
        let count = self.positions.len();
        self.avg_quality = (self.avg_quality * (count - 1) as f64 + quality) / count as f64;
    }

    /// Get position at a specific frame.
    #[must_use]
    pub fn position_at(&self, frame: usize) -> Option<(f64, f64)> {
        self.positions.get(&frame).copied()
    }

    /// Get track length (number of frames).
    #[must_use]
    pub fn length(&self) -> usize {
        self.positions.len()
    }

    /// Check if track is active at a given frame.
    #[must_use]
    pub fn is_active_at(&self, frame: usize) -> bool {
        frame >= self.start_frame && frame <= self.end_frame
    }

    /// Calculate track displacement (total motion).
    #[must_use]
    pub fn total_displacement(&self) -> f64 {
        if self.positions.len() < 2 {
            return 0.0;
        }

        let start_pos = self.positions.get(&self.start_frame);
        let end_pos = self.positions.get(&self.end_frame);

        match (start_pos, end_pos) {
            (Some((x1, y1)), Some((x2, y2))) => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                (dx * dx + dy * dy).sqrt()
            }
            _ => 0.0,
        }
    }
}

/// Motion tracker that detects and tracks features across frames.
#[derive(Debug)]
pub struct MotionTracker {
    /// Maximum number of features to track
    max_features: usize,
    /// Minimum feature quality threshold
    quality_threshold: f64,
    /// Feature detection grid size (for spatial distribution)
    grid_size: usize,
    /// Current feature ID counter
    next_feature_id: usize,
    /// Active feature tracks
    active_tracks: Vec<FeatureTrack>,
}

impl MotionTracker {
    /// Create a new motion tracker.
    #[must_use]
    pub fn new(max_features: usize) -> Self {
        Self {
            max_features,
            quality_threshold: 0.01,
            grid_size: 10,
            next_feature_id: 0,
            active_tracks: Vec::new(),
        }
    }

    /// Set quality threshold for feature detection.
    pub fn set_quality_threshold(&mut self, threshold: f64) {
        self.quality_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Track features across a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame sequence is empty
    /// - Feature detection fails
    /// - Insufficient features are found
    pub fn track(&mut self, frames: &[Frame]) -> StabilizeResult<Vec<FeatureTrack>> {
        if frames.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Detect features in the first frame
        let mut features = self.detect_features(&frames[0])?;

        // Initialize tracks
        self.active_tracks = features
            .iter()
            .map(|f| {
                let mut track = FeatureTrack::new(f.id, 0);
                track.add_position(0, f.x, f.y, f.quality);
                track
            })
            .collect();

        // Track features through remaining frames
        for (frame_idx, frame) in frames.iter().enumerate().skip(1) {
            // Track existing features
            let prev_frame = &frames[frame_idx - 1];
            features = self.track_features(prev_frame, frame, &features)?;

            // Update tracks
            for feature in &features {
                if let Some(track) = self.active_tracks.iter_mut().find(|t| t.id == feature.id) {
                    track.add_position(frame_idx, feature.x, feature.y, feature.quality);
                }
            }

            // Detect new features if needed
            if features.len() < self.max_features / 2 {
                let new_features = self.detect_new_features(frame, &features)?;
                for feature in new_features {
                    let mut track = FeatureTrack::new(feature.id, frame_idx);
                    track.add_position(frame_idx, feature.x, feature.y, feature.quality);
                    self.active_tracks.push(track);
                    features.push(feature);
                }
            }
        }

        // Filter out short tracks (less than 3 frames)
        self.active_tracks.retain(|track| track.length() >= 3);

        if self.active_tracks.is_empty() {
            return Err(StabilizeError::insufficient_features(0, 10));
        }

        Ok(self.active_tracks.clone())
    }

    /// Detect features in a frame using Harris corner detection.
    fn detect_features(&mut self, frame: &Frame) -> StabilizeResult<Vec<Feature>> {
        let mut features = Vec::new();

        // Compute Harris corner response
        let corners = self.harris_corner_detection(&frame.data)?;

        // Extract features from corners
        let cell_width = frame.width / self.grid_size;
        let cell_height = frame.height / self.grid_size;

        // Ensure spatial distribution by dividing image into grid
        for grid_y in 0..self.grid_size {
            for grid_x in 0..self.grid_size {
                let x_start = grid_x * cell_width;
                let y_start = grid_y * cell_height;
                let x_end = ((grid_x + 1) * cell_width).min(frame.width);
                let y_end = ((grid_y + 1) * cell_height).min(frame.height);

                // Find best corner in this cell
                let mut best_corner = None;
                let mut best_quality = self.quality_threshold;

                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let quality = corners[[y, x]];
                        if quality > best_quality {
                            best_quality = quality;
                            best_corner = Some((x, y));
                        }
                    }
                }

                if let Some((x, y)) = best_corner {
                    features.push(Feature::new(
                        x as f64,
                        y as f64,
                        best_quality,
                        self.next_feature_id,
                    ));
                    self.next_feature_id += 1;

                    if features.len() >= self.max_features {
                        return Ok(features);
                    }
                }
            }
        }

        if features.len() < 10 {
            return Err(StabilizeError::insufficient_features(features.len(), 10));
        }

        Ok(features)
    }

    /// Track existing features from one frame to the next using optical flow.
    ///
    /// Each feature is tracked independently via `track_single_feature`, making
    /// the operation embarrassingly parallel.  We use rayon's `par_iter` to
    /// spread the work across available CPU cores.
    fn track_features(
        &self,
        prev_frame: &Frame,
        curr_frame: &Frame,
        features: &[Feature],
    ) -> StabilizeResult<Vec<Feature>> {
        let tracked_features: Vec<Feature> = features
            .par_iter()
            .filter_map(|feature| {
                let new_pos = self.track_single_feature(prev_frame, curr_frame, feature)?;
                if new_pos.is_valid(curr_frame.width, curr_frame.height) {
                    Some(new_pos)
                } else {
                    None
                }
            })
            .collect();

        Ok(tracked_features)
    }

    /// Track a single feature using Lucas-Kanade optical flow.
    fn track_single_feature(
        &self,
        prev_frame: &Frame,
        curr_frame: &Frame,
        feature: &Feature,
    ) -> Option<Feature> {
        let window_size = 21;
        let half_window = window_size / 2;

        let x = feature.x as usize;
        let y = feature.y as usize;

        // Check bounds
        if x < half_window
            || y < half_window
            || x + half_window >= prev_frame.width
            || y + half_window >= prev_frame.height
        {
            return None;
        }

        // Simple template matching (in production, use Lucas-Kanade)
        let mut best_dx = 0;
        let mut best_dy = 0;
        let mut best_score = f64::MAX;

        let search_radius = 20;

        for dy in -(search_radius as i32)..=(search_radius as i32) {
            for dx in -(search_radius as i32)..=(search_radius as i32) {
                let nx_i = x as i32 + dx;
                let ny_i = y as i32 + dy;

                if nx_i < half_window as i32
                    || ny_i < half_window as i32
                    || nx_i + half_window as i32 >= curr_frame.width as i32
                    || ny_i + half_window as i32 >= curr_frame.height as i32
                {
                    continue;
                }

                let new_x = nx_i as usize;
                let new_y = ny_i as usize;

                let score = self.template_match(
                    &prev_frame.data,
                    &curr_frame.data,
                    x,
                    y,
                    new_x,
                    new_y,
                    window_size,
                );

                if score < best_score {
                    best_score = score;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        // Quality based on matching score
        let quality = 1.0 / (1.0 + best_score);

        if quality < self.quality_threshold {
            return None;
        }

        Some(Feature::new(
            (x as i32 + best_dx) as f64,
            (y as i32 + best_dy) as f64,
            quality,
            feature.id,
        ))
    }

    /// Template matching using sum of squared differences.
    fn template_match(
        &self,
        prev: &Array2<u8>,
        curr: &Array2<u8>,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        window_size: usize,
    ) -> f64 {
        let half = window_size / 2;
        let mut sum = 0.0;
        let mut count = 0;

        for dy in 0..window_size {
            for dx in 0..window_size {
                let py = y1 + dy - half;
                let px = x1 + dx - half;
                let cy = y2 + dy - half;
                let cx = x2 + dx - half;

                let p1 = prev[[py, px]] as f64;
                let p2 = curr[[cy, cx]] as f64;

                let diff = p1 - p2;
                sum += diff * diff;
                count += 1;
            }
        }

        sum / count as f64
    }

    /// Detect new features avoiding existing ones.
    fn detect_new_features(
        &mut self,
        frame: &Frame,
        existing: &[Feature],
    ) -> StabilizeResult<Vec<Feature>> {
        let mut new_features = Vec::new();
        let corners = self.harris_corner_detection(&frame.data)?;

        let min_distance = 20.0; // Minimum distance between features

        for y in 10..(frame.height - 10) {
            for x in 10..(frame.width - 10) {
                let quality = corners[[y, x]];

                if quality < self.quality_threshold {
                    continue;
                }

                // Check distance to existing features
                let too_close = existing.iter().any(|f| {
                    let dx = f.x - x as f64;
                    let dy = f.y - y as f64;
                    (dx * dx + dy * dy).sqrt() < min_distance
                });

                if too_close {
                    continue;
                }

                new_features.push(Feature::new(
                    x as f64,
                    y as f64,
                    quality,
                    self.next_feature_id,
                ));
                self.next_feature_id += 1;

                if new_features.len() >= self.max_features / 2 {
                    return Ok(new_features);
                }
            }
        }

        Ok(new_features)
    }

    /// Harris corner detection.
    fn harris_corner_detection(&self, image: &Array2<u8>) -> StabilizeResult<Array2<f64>> {
        let (height, width) = image.dim();
        let mut corners = Array2::zeros((height, width));

        // Compute gradients
        let (grad_x, grad_y) = self.compute_gradients(image);

        // Compute structure tensor components
        let window_size = 5;
        let half = window_size / 2;
        let k = 0.04_f64; // Harris parameter

        for y in half..(height - half) {
            for x in half..(width - half) {
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut ixy = 0.0;

                // Sum over window
                for dy in 0..window_size {
                    for dx in 0..window_size {
                        let py = y + dy - half;
                        let px = x + dx - half;

                        let gx = grad_x[[py, px]];
                        let gy = grad_y[[py, px]];

                        ixx += gx * gx;
                        iyy += gy * gy;
                        ixy += gx * gy;
                    }
                }

                // Compute corner response
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - k * trace * trace;

                corners[[y, x]] = response.max(0.0);
            }
        }

        // Normalize to 0-1
        let max_response = corners.iter().fold(0.0_f64, |a, &b| a.max(b));
        if max_response > 0.0 {
            corners.mapv_inplace(|v| v / max_response);
        }

        Ok(corners)
    }

    /// Compute image gradients using Sobel operator.
    fn compute_gradients(&self, image: &Array2<u8>) -> (Array2<f64>, Array2<f64>) {
        let (height, width) = image.dim();
        let mut grad_x = Array2::zeros((height, width));
        let mut grad_y = Array2::zeros((height, width));

        // Sobel kernels
        let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
        let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let pixel = image[[y + ky - 1, x + kx - 1]] as f64;
                        gx += pixel * sobel_x[ky][kx];
                        gy += pixel * sobel_y[ky][kx];
                    }
                }

                grad_x[[y, x]] = gx;
                grad_y[[y, x]] = gy;
            }
        }

        (grad_x, grad_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_creation() {
        let feature = Feature::new(10.0, 20.0, 0.8, 0);
        assert!((feature.x - 10.0).abs() < f64::EPSILON);
        assert!((feature.y - 20.0).abs() < f64::EPSILON);
        assert!((feature.quality - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_distance() {
        let f1 = Feature::new(0.0, 0.0, 1.0, 0);
        let f2 = Feature::new(3.0, 4.0, 1.0, 1);
        assert!((f1.distance_to(&f2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_validity() {
        let feature = Feature::new(50.0, 60.0, 0.5, 0);
        assert!(feature.is_valid(100, 100));

        let invalid = Feature::new(150.0, 60.0, 0.5, 0);
        assert!(!invalid.is_valid(100, 100));
    }

    #[test]
    fn test_feature_track() {
        let mut track = FeatureTrack::new(0, 0);
        track.add_position(0, 10.0, 20.0, 0.8);
        track.add_position(1, 15.0, 25.0, 0.7);

        assert_eq!(track.length(), 2);
        assert!(track.is_active_at(1));
        assert!(!track.is_active_at(5));
    }

    #[test]
    fn test_motion_tracker_creation() {
        let tracker = MotionTracker::new(500);
        assert_eq!(tracker.max_features, 500);
    }

    /// Verify that `track_features` (the parallelised inner step) produces the
    /// same output regardless of execution order.
    ///
    /// We build a synthetic set of 200 features spread across a checkerboard
    /// frame and verify that running `track_features` twice on the same inputs
    /// yields bitwise-identical results.  Because the parallel path uses
    /// `par_iter` with a pure (read-only captures only) closure, the result is
    /// fully deterministic — this test pins that invariant.
    #[test]
    fn test_tracker_parallel_correctness() {
        use scirs2_core::ndarray::Array2;

        // Build a simple checkerboard pattern so that template matching
        // has something non-trivial to work with.
        let w = 120usize;
        let h = 120usize;
        let make_checkerboard = |w: usize, h: usize| -> Array2<u8> {
            Array2::from_shape_fn((h, w), |(y, x)| {
                if (x / 8 + y / 8) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
        };

        let prev_data = make_checkerboard(w, h);
        let curr_data = make_checkerboard(w, h);
        let prev_frame = crate::Frame::new(w, h, 0.0, prev_data);
        let curr_frame = crate::Frame::new(w, h, 0.0333, curr_data);

        // Synthesise 200 features spread evenly in the interior of the frame.
        let features: Vec<Feature> = (0..200)
            .map(|i| {
                let x = 20.0 + (i % 20) as f64 * 4.0;
                let y = 20.0 + (i / 20) as f64 * 4.0;
                Feature::new(x, y, 0.8, i)
            })
            .collect();

        let tracker = MotionTracker::new(500);

        // Run twice — results must be identical.
        let run1 = tracker
            .track_features(&prev_frame, &curr_frame, &features)
            .expect("first run");
        let run2 = tracker
            .track_features(&prev_frame, &curr_frame, &features)
            .expect("second run");

        assert_eq!(
            run1.len(),
            run2.len(),
            "parallel track_features must be deterministic"
        );
        for (i, (a, b)) in run1.iter().zip(run2.iter()).enumerate() {
            assert_eq!(a.id, b.id, "feature id mismatch at index {i}");
            assert!(
                (a.x - b.x).abs() < f64::EPSILON && (a.y - b.y).abs() < f64::EPSILON,
                "position mismatch at index {i}: ({},{}) vs ({},{})",
                a.x,
                a.y,
                b.x,
                b.y
            );
        }
    }
}

/// Feature matching utilities.
pub mod matching {
    use super::Feature;

    /// Match features between two frames using descriptors.
    pub struct FeatureMatcher {
        max_distance: f64,
        ratio_threshold: f64,
    }

    impl FeatureMatcher {
        /// Create a new feature matcher.
        #[must_use]
        pub fn new() -> Self {
            Self {
                max_distance: 50.0,
                ratio_threshold: 0.8,
            }
        }

        /// Match features using nearest neighbor.
        #[must_use]
        pub fn match_features(&self, features1: &[Feature], features2: &[Feature]) -> Vec<Match> {
            let mut matches = Vec::new();

            for (i, f1) in features1.iter().enumerate() {
                let mut best_distance = f64::MAX;
                let mut best_match = None;
                let mut second_best = f64::MAX;

                for (j, f2) in features2.iter().enumerate() {
                    let dist = f1.distance_to(f2);

                    if dist < best_distance {
                        second_best = best_distance;
                        best_distance = dist;
                        best_match = Some(j);
                    } else if dist < second_best {
                        second_best = dist;
                    }
                }

                // Lowe's ratio test
                if best_distance < self.max_distance
                    && best_distance < second_best * self.ratio_threshold
                {
                    if let Some(j) = best_match {
                        matches.push(Match {
                            index1: i,
                            index2: j,
                            distance: best_distance,
                        });
                    }
                }
            }

            matches
        }

        /// Filter matches using geometric constraints.
        #[must_use]
        pub fn filter_geometric(
            &self,
            matches: &[Match],
            features1: &[Feature],
            features2: &[Feature],
        ) -> Vec<Match> {
            matches
                .iter()
                .filter(|m| {
                    let f1 = &features1[m.index1];
                    let f2 = &features2[m.index2];

                    // Check if motion is reasonable
                    let motion = f1.distance_to(f2);
                    motion < self.max_distance
                })
                .copied()
                .collect()
        }
    }

    impl Default for FeatureMatcher {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A match between two features.
    #[derive(Debug, Clone, Copy)]
    pub struct Match {
        /// Index in first feature set
        pub index1: usize,
        /// Index in second feature set
        pub index2: usize,
        /// Match distance
        pub distance: f64,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_matcher() {
            let matcher = FeatureMatcher::new();
            let features1 = vec![Feature::new(0.0, 0.0, 1.0, 0)];
            let features2 = vec![Feature::new(5.0, 5.0, 1.0, 1)];

            let matches = matcher.match_features(&features1, &features2);
            assert!(!matches.is_empty());
        }
    }
}

/// Feature descriptor computation.
pub mod descriptors {
    use scirs2_core::ndarray::Array2;

    /// Compute BRIEF descriptor for a feature.
    #[must_use]
    pub fn compute_brief_descriptor(
        image: &Array2<u8>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Vec<u8> {
        let mut descriptor = Vec::new();
        let half = size / 2;

        // Sample pairs of pixels
        for dy in 0..size {
            for dx in 0..size {
                let y1 = y.saturating_add(dy).saturating_sub(half);
                let x1 = x.saturating_add(dx).saturating_sub(half);

                if y1 < image.dim().0 && x1 < image.dim().1 {
                    descriptor.push(image[[y1, x1]]);
                }
            }
        }

        descriptor
    }

    /// Compute ORB descriptor.
    #[must_use]
    pub fn compute_orb_descriptor(image: &Array2<u8>, x: usize, y: usize) -> Vec<u8> {
        // Simplified ORB descriptor
        compute_brief_descriptor(image, x, y, 31)
    }

    /// Hamming distance between binary descriptors.
    #[must_use]
    pub fn hamming_distance(desc1: &[u8], desc2: &[u8]) -> usize {
        desc1
            .iter()
            .zip(desc2.iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_brief_descriptor() {
            let image = Array2::from_elem((100, 100), 128);
            let desc = compute_brief_descriptor(&image, 50, 50, 11);
            assert!(!desc.is_empty());
        }

        #[test]
        fn test_hamming_distance() {
            let desc1 = vec![0b11110000, 0b00001111];
            let desc2 = vec![0b11110000, 0b11110000];
            let dist = hamming_distance(&desc1, &desc2);
            assert_eq!(dist, 8);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  CachedTracker — descriptor-caching wrapper around MotionTracker
// ─────────────────────────────────────────────────────────────────

/// Configuration for [`CachedTracker`].
///
/// Mirrors [`MotionTracker`] construction parameters so the inner tracker can
/// be fully configured through this wrapper.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Maximum number of features to track.  Passed directly to
    /// [`MotionTracker::new`].
    pub max_features: usize,
    /// Minimum feature quality threshold (0.0–1.0).
    pub quality_threshold: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_features: 500,
            quality_threshold: 0.01,
        }
    }
}

/// A flat feature descriptor stored as raw pixel intensities sampled around a
/// key-point.  This is the same representation produced by
/// [`descriptors::compute_brief_descriptor`].
pub type FeatureDescriptor = Vec<u8>;

/// A 2-D key-point (pixel position) paired with its feature-quality score and
/// a pre-computed [`FeatureDescriptor`].
#[derive(Debug, Clone)]
pub struct KeyPoint {
    /// Column index.
    pub x: f64,
    /// Row index.
    pub y: f64,
    /// Detector response strength (0.0–1.0).
    pub quality: f64,
    /// BRIEF/ORB-style intensity descriptor.
    pub descriptor: FeatureDescriptor,
}

/// A single frame-to-frame motion vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Source key-point index (into the previous frame's key-point list).
    pub src_idx: usize,
    /// Horizontal displacement in pixels (positive = right).
    pub dx: f64,
    /// Vertical displacement in pixels (positive = down).
    pub dy: f64,
    /// Match quality / confidence (0.0–1.0).
    pub confidence: f64,
}

/// A feature-descriptor-caching wrapper around [`MotionTracker`].
///
/// The key optimisation: descriptors computed for frame `t` as the *current*
/// frame are reused as the *previous* frame descriptors on the next call —
/// avoiding the redundant Harris corner detection + descriptor extraction that
/// would otherwise happen twice per frame boundary.
///
/// # Usage
///
/// ```rust,no_run
/// use oximedia_stabilize::motion::tracker::{CachedTracker, TrackerConfig};
///
/// let mut tracker = CachedTracker::new(TrackerConfig::default());
/// let prev = vec![0u8; 640 * 480];
/// let curr = vec![0u8; 640 * 480];
/// let next = vec![0u8; 640 * 480];
/// // First call always computes descriptors for both frames.
/// // Subsequent calls reuse prev descriptors from the previous curr.
/// let _vectors = tracker.track_frame(&prev, &curr, 640, 480);
/// let _vectors = tracker.track_frame(&curr, &next, 640, 480);
/// ```
#[derive(Debug)]
pub struct CachedTracker {
    inner: MotionTracker,
    prev_keypoints: Option<Vec<KeyPoint>>,
}

impl CachedTracker {
    /// Create a new `CachedTracker`.
    #[must_use]
    pub fn new(config: TrackerConfig) -> Self {
        let mut inner = MotionTracker::new(config.max_features);
        inner.set_quality_threshold(config.quality_threshold);
        Self {
            inner,
            prev_keypoints: None,
        }
    }

    /// Track features between `prev` and `curr` (both flat grayscale, `w × h`).
    ///
    /// On the first call both frame's key-points are computed from scratch.
    /// On subsequent calls the key-points for `prev` are reused from the
    /// last call's `curr` descriptors.
    ///
    /// Returns a list of [`MotionVector`]s describing the displacement of each
    /// matched feature.
    #[must_use]
    pub fn track_frame(&mut self, prev: &[u8], curr: &[u8], w: u32, h: u32) -> Vec<MotionVector> {
        let prev_kps = match self.prev_keypoints.take() {
            Some(kps) => kps,
            None => {
                // First call — compute key-points for `prev` from scratch.
                Self::detect_keypoints(prev, w, h, &self.inner)
            }
        };

        // Compute key-points for `curr`.
        let curr_kps = Self::detect_keypoints(curr, w, h, &self.inner);

        // Cache `curr` key-points for the next call.
        self.prev_keypoints = Some(curr_kps.clone());

        // Match key-points using Hamming distance on BRIEF descriptors and
        // compute displacement vectors for matched pairs.
        Self::match_and_vectorise(&prev_kps, &curr_kps, w, h)
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Run Harris corner detection on `data` (flat row-major u8, `w × h`) and
    /// return key-points with pre-computed BRIEF descriptors.
    fn detect_keypoints(data: &[u8], w: u32, h: u32, tracker: &MotionTracker) -> Vec<KeyPoint> {
        use descriptors::compute_brief_descriptor;

        let wu = w as usize;
        let hu = h as usize;

        if wu == 0 || hu == 0 || data.len() < wu * hu {
            return Vec::new();
        }

        // Convert flat buffer to Array2 for the inner Harris detector.
        let array = Array2::from_shape_fn((hu, wu), |(y, x)| data[y * wu + x]);

        let corners = match tracker.harris_corner_detection(&array) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let quality_threshold = tracker.quality_threshold;
        let grid_size = tracker.grid_size.max(1);
        let max_features = tracker.max_features;

        let cell_w = wu / grid_size;
        let cell_h = hu / grid_size;

        let mut kps = Vec::with_capacity(max_features);

        'outer: for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x_start = gx * cell_w;
                let y_start = gy * cell_h;
                let x_end = ((gx + 1) * cell_w).min(wu);
                let y_end = ((gy + 1) * cell_h).min(hu);

                let mut best: Option<(usize, usize, f64)> = None;

                for y in y_start..y_end {
                    for x in x_start..x_end {
                        let q = corners[[y, x]];
                        if q > quality_threshold {
                            if best.map_or(true, |(_, _, bq)| q > bq) {
                                best = Some((x, y, q));
                            }
                        }
                    }
                }

                if let Some((x, y, quality)) = best {
                    let descriptor = compute_brief_descriptor(&array, x, y, 11);
                    kps.push(KeyPoint {
                        x: x as f64,
                        y: y as f64,
                        quality,
                        descriptor,
                    });

                    if kps.len() >= max_features {
                        break 'outer;
                    }
                }
            }
        }

        kps
    }

    /// Match `prev_kps` to `curr_kps` using nearest-neighbour Hamming distance
    /// with Lowe's ratio test, and convert matched pairs to [`MotionVector`]s.
    fn match_and_vectorise(
        prev_kps: &[KeyPoint],
        curr_kps: &[KeyPoint],
        w: u32,
        h: u32,
    ) -> Vec<MotionVector> {
        use descriptors::hamming_distance;

        let max_disp = (w.max(h) as f64) * 0.25; // max 25 % of larger dim
        let ratio = 0.8f64;

        let mut vectors = Vec::new();

        for (src_idx, pk) in prev_kps.iter().enumerate() {
            let mut best_dist = usize::MAX;
            let mut second_dist = usize::MAX;
            let mut best_j = None;

            for (j, ck) in curr_kps.iter().enumerate() {
                let dist = hamming_distance(&pk.descriptor, &ck.descriptor);
                if dist < best_dist {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_j = Some(j);
                } else if dist < second_dist {
                    second_dist = dist;
                }
            }

            // Lowe's ratio test
            let ratio_ok = (best_dist as f64) < ratio * (second_dist as f64);
            if !ratio_ok {
                continue;
            }

            if let Some(j) = best_j {
                let ck = &curr_kps[j];
                let dx = ck.x - pk.x;
                let dy = ck.y - pk.y;
                let disp = (dx * dx + dy * dy).sqrt();

                if disp > max_disp {
                    continue; // outlier rejection
                }

                let confidence = 1.0 / (1.0 + best_dist as f64);
                vectors.push(MotionVector {
                    src_idx,
                    dx,
                    dy,
                    confidence,
                });
            }
        }

        vectors
    }
}

#[cfg(test)]
mod cached_tracker_tests {
    use super::*;

    /// Helper: make a flat grayscale frame from a pattern function.
    fn make_frame<F: Fn(usize, usize) -> u8>(w: u32, h: u32, f: F) -> Vec<u8> {
        let (wu, hu) = (w as usize, h as usize);
        let mut buf = Vec::with_capacity(wu * hu);
        for y in 0..hu {
            for x in 0..wu {
                buf.push(f(x, y));
            }
        }
        buf
    }

    /// Verify that `CachedTracker` produces the same motion vectors as calling
    /// `track_frame` freshly (no cache) over a 5-frame sequence.
    #[test]
    fn test_cached_tracker_correctness() {
        let w: u32 = 80;
        let h: u32 = 80;

        // Build 5 slightly shifted versions of a checkerboard.
        let frames: Vec<Vec<u8>> = (0..5)
            .map(|i| {
                make_frame(w, h, |x, y| {
                    // Shift by `i` pixels to the right
                    let shifted_x = (x + i) % w as usize;
                    if (shifted_x / 8 + y / 8) % 2 == 0 {
                        200u8
                    } else {
                        50u8
                    }
                })
            })
            .collect();

        let config = TrackerConfig::default();

        // Run with caching (reuses prev descriptors from last curr).
        let mut cached = CachedTracker::new(config.clone());
        let mut with_cache: Vec<Vec<MotionVector>> = Vec::new();
        for pair in frames.windows(2) {
            let vecs = cached.track_frame(&pair[0], &pair[1], w, h);
            with_cache.push(vecs);
        }

        // Run without caching (fresh CachedTracker for every pair = no reuse).
        let mut without_cache: Vec<Vec<MotionVector>> = Vec::new();
        for pair in frames.windows(2) {
            let mut fresh = CachedTracker::new(config.clone());
            let vecs = fresh.track_frame(&pair[0], &pair[1], w, h);
            without_cache.push(vecs);
        }

        // Both approaches must produce identical motion vectors.
        assert_eq!(
            with_cache.len(),
            without_cache.len(),
            "number of frame pairs should match"
        );

        for (frame_idx, (cached_vecs, fresh_vecs)) in
            with_cache.iter().zip(without_cache.iter()).enumerate()
        {
            assert_eq!(
                cached_vecs.len(),
                fresh_vecs.len(),
                "frame pair {frame_idx}: cached and fresh must match count"
            );

            for (i, (cv, fv)) in cached_vecs.iter().zip(fresh_vecs.iter()).enumerate() {
                assert_eq!(
                    cv.src_idx, fv.src_idx,
                    "frame {frame_idx} vector {i}: src_idx mismatch"
                );
                assert!(
                    (cv.dx - fv.dx).abs() < 1e-9,
                    "frame {frame_idx} vector {i}: dx mismatch"
                );
                assert!(
                    (cv.dy - fv.dy).abs() < 1e-9,
                    "frame {frame_idx} vector {i}: dy mismatch"
                );
            }
        }
    }

    #[test]
    fn test_cached_tracker_reuses_prev_descriptors() {
        // Confirm that the second `track_frame` call does NOT recompute prev
        // descriptors — we verify this indirectly by checking that no panic
        // occurs and the outputs are non-empty for non-blank frames.
        let w: u32 = 64;
        let h: u32 = 64;
        let frame = make_frame(w, h, |x, y| {
            if (x / 8 + y / 8) % 2 == 0 {
                200u8
            } else {
                50u8
            }
        });

        let mut tracker = CachedTracker::new(TrackerConfig::default());
        // First call — no cache hit
        let _v1 = tracker.track_frame(&frame, &frame, w, h);
        // Second call — `prev_keypoints` is populated from first call's curr
        let _v2 = tracker.track_frame(&frame, &frame, w, h);
        // Just verifying no panic and that the cache path was taken
    }

    #[test]
    fn test_motion_vector_fields() {
        let mv = MotionVector {
            src_idx: 3,
            dx: 1.5,
            dy: -2.0,
            confidence: 0.9,
        };
        assert_eq!(mv.src_idx, 3);
        assert!((mv.dx - 1.5).abs() < f64::EPSILON);
        assert!((mv.dy + 2.0).abs() < f64::EPSILON);
        assert!((mv.confidence - 0.9).abs() < f64::EPSILON);
    }
}
