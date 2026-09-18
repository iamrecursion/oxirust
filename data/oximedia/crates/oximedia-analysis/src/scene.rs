//! Scene and shot boundary detection.
//!
//! This module implements multiple algorithms for detecting scene changes:
//! - Histogram difference for hard cuts
//! - Edge change ratio for gradual transitions (fades, dissolves)
//! - Motion compensation for camera movement
//!
//! # Algorithms
//!
//! ## Histogram Difference
//!
//! Compares the luminance histogram between consecutive frames. A large
//! difference indicates a hard cut.
//!
//! ## Edge Change Ratio (ECR)
//!
//! Detects changes in edge pixels between frames. Useful for detecting
//! gradual transitions like dissolves and fades.
//!
//! ## Motion Compensation
//!
//! Reduces false positives from camera pans and zooms by estimating
//! global motion.

use crate::{AnalysisError, AnalysisResult};
use serde::{Deserialize, Serialize};

/// Scene information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Starting frame number
    pub start_frame: usize,
    /// Ending frame number (exclusive)
    pub end_frame: usize,
    /// Scene change confidence (0.0-1.0)
    pub confidence: f64,
    /// Type of scene change
    pub change_type: SceneChangeType,
}

/// Type of scene change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneChangeType {
    /// Hard cut
    Cut,
    /// Gradual fade
    Fade,
    /// Dissolve transition (temporal cross-fade detected via inter-frame alpha blending)
    Dissolve,
    /// Wipe transition (spatial boundary sweeps across the frame)
    Wipe,
    /// Unknown/other
    Unknown,
}

/// Scene detector.
pub struct SceneDetector {
    threshold: f64,
    prev_histogram: Option<Histogram>,
    prev_edges: Option<EdgeMap>,
    prev_frame: Option<Vec<u8>>,
    prev_width: usize,
    prev_height: usize,
    scenes: Vec<Scene>,
    last_scene_frame: usize,
    /// Rolling window of histogram differences for dissolve/wipe tracking.
    transition_window: std::collections::VecDeque<f64>,
    /// Accumulated transition gradient magnitudes for wipe detection.
    wipe_gradients: std::collections::VecDeque<f64>,
}

impl SceneDetector {
    /// Create a new scene detector with the given threshold.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            prev_histogram: None,
            prev_edges: None,
            prev_frame: None,
            prev_width: 0,
            prev_height: 0,
            scenes: Vec::new(),
            last_scene_frame: 0,
            transition_window: std::collections::VecDeque::with_capacity(16),
            wipe_gradients: std::collections::VecDeque::with_capacity(16),
        }
    }

    /// Process a frame.
    pub fn process_frame(
        &mut self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        frame_number: usize,
    ) -> AnalysisResult<()> {
        if y_plane.len() != width * height {
            return Err(AnalysisError::InvalidInput(
                "Y plane size mismatch".to_string(),
            ));
        }

        // Compute histogram for current frame
        let histogram = compute_histogram(y_plane);

        // Compute edges for current frame
        let edges = compute_edges(y_plane, width, height);

        if let Some(prev_hist) = &self.prev_histogram {
            // Compute histogram difference
            let hist_diff = histogram_difference(prev_hist, &histogram);

            // Compute edge change ratio
            let ecr = if let Some(prev_edges) = &self.prev_edges {
                edge_change_ratio(prev_edges, &edges)
            } else {
                0.0
            };

            // Compute wipe score using spatial boundary analysis
            let wipe_score = if let Some(prev_frame) = &self.prev_frame {
                if self.prev_width == width && self.prev_height == height {
                    compute_wipe_score(prev_frame, y_plane, width, height)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Maintain rolling windows for gradual transition detection
            const TRANSITION_WINDOW: usize = 8;
            if self.transition_window.len() >= TRANSITION_WINDOW {
                self.transition_window.pop_front();
            }
            self.transition_window.push_back(hist_diff);

            if self.wipe_gradients.len() >= TRANSITION_WINDOW {
                self.wipe_gradients.pop_front();
            }
            self.wipe_gradients.push_back(wipe_score);

            // Determine if this is a scene change
            let (is_scene_change, change_type) =
                self.detect_scene_change(hist_diff, ecr, wipe_score);

            if is_scene_change {
                // Record the scene
                self.scenes.push(Scene {
                    start_frame: self.last_scene_frame,
                    end_frame: frame_number,
                    confidence: hist_diff.max(ecr).max(wipe_score),
                    change_type,
                });
                self.last_scene_frame = frame_number;
            }
        }

        self.prev_histogram = Some(histogram);
        self.prev_edges = Some(edges);
        self.prev_frame = Some(y_plane.to_vec());
        self.prev_width = width;
        self.prev_height = height;

        Ok(())
    }

    /// Detect if a scene change occurred based on metrics.
    fn detect_scene_change(
        &self,
        hist_diff: f64,
        ecr: f64,
        wipe_score: f64,
    ) -> (bool, SceneChangeType) {
        // Hard cut detection (high histogram difference)
        if hist_diff > self.threshold {
            return (true, SceneChangeType::Cut);
        }

        // Wipe detection: high spatial gradient motion (boundary sweeps across frame)
        // combined with moderate histogram change (two different scenes visible simultaneously)
        if wipe_score > self.threshold * 0.6 && hist_diff > self.threshold * 0.15 {
            return (true, SceneChangeType::Wipe);
        }

        // Dissolve detection: sustained moderate histogram change over multiple frames
        // (cross-fade mixes both scenes) without sharp hard-cut signature.
        // Requires at least 3 frames in the window showing progressive change.
        if self.transition_window.len() >= 3 {
            let sustained_change = self
                .transition_window
                .iter()
                .filter(|&&d| d > self.threshold * 0.15 && d < self.threshold * 0.9)
                .count();
            let window_len = self.transition_window.len();
            if sustained_change >= window_len.saturating_sub(1)
                && ecr > self.threshold * 0.4
                && hist_diff > self.threshold * 0.2
            {
                return (true, SceneChangeType::Dissolve);
            }
        }

        // Gradual transition detection (high edge change ratio, moderate histogram change)
        if ecr > self.threshold * 0.7 && hist_diff > self.threshold * 0.3 {
            return (true, SceneChangeType::Dissolve);
        }

        // Fade detection (moderate edge change, low histogram change)
        if ecr > self.threshold * 0.5 && hist_diff < self.threshold * 0.2 {
            return (true, SceneChangeType::Fade);
        }

        (false, SceneChangeType::Unknown)
    }

    /// Finalize and return detected scenes.
    pub fn finalize(self) -> Vec<Scene> {
        self.scenes
    }
}

/// Luminance histogram (256 bins).
type Histogram = [usize; 256];

/// Compute luminance histogram.
fn compute_histogram(y_plane: &[u8]) -> Histogram {
    let mut histogram = [0; 256];
    for &pixel in y_plane {
        histogram[pixel as usize] += 1;
    }
    histogram
}

/// Compute histogram difference (normalized).
fn histogram_difference(h1: &Histogram, h2: &Histogram) -> f64 {
    let total: usize = h1.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let diff: usize = h1.iter().zip(h2.iter()).map(|(a, b)| a.abs_diff(*b)).sum();

    diff as f64 / (2.0 * total as f64)
}

/// Edge map (binary edge detection).
struct EdgeMap {
    edges: Vec<bool>,
    #[allow(dead_code)]
    width: usize,
    #[allow(dead_code)]
    height: usize,
}

/// Compute edge map using Sobel operator.
fn compute_edges(y_plane: &[u8], width: usize, height: usize) -> EdgeMap {
    let mut edges = vec![false; width * height];
    let threshold = 30;

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;

            // Sobel X kernel
            let gx = (i32::from(y_plane[(y - 1) * width + (x + 1)])
                + 2 * i32::from(y_plane[y * width + (x + 1)])
                + i32::from(y_plane[(y + 1) * width + (x + 1)]))
                - (i32::from(y_plane[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(y_plane[y * width + (x - 1)])
                    + i32::from(y_plane[(y + 1) * width + (x - 1)]));

            // Sobel Y kernel
            let gy = (i32::from(y_plane[(y + 1) * width + (x - 1)])
                + 2 * i32::from(y_plane[(y + 1) * width + x])
                + i32::from(y_plane[(y + 1) * width + (x + 1)]))
                - (i32::from(y_plane[(y - 1) * width + (x - 1)])
                    + 2 * i32::from(y_plane[(y - 1) * width + x])
                    + i32::from(y_plane[(y - 1) * width + (x + 1)]));

            // Gradient magnitude
            let magnitude = f64::from(gx * gx + gy * gy).sqrt();
            edges[idx] = magnitude > f64::from(threshold);
        }
    }

    EdgeMap {
        edges,
        width,
        height,
    }
}

/// Compute wipe transition score.
///
/// A wipe is characterized by a linear (or near-linear) boundary that sweeps
/// across the frame, revealing a new scene.  We detect this by:
///
/// 1. Computing a per-column and per-row difference profile between consecutive frames.
/// 2. Fitting a simple step-function model to both profiles.
/// 3. If either profile has a clear step boundary (high difference on one side,
///    low on the other), we score it proportionally to the boundary sharpness.
fn compute_wipe_score(prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
    if prev.len() != width * height || curr.len() != width * height {
        return 0.0;
    }
    if width < 4 || height < 4 {
        return 0.0;
    }

    // Per-column mean absolute difference
    let mut col_diff = vec![0.0f64; width];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let d = (i32::from(curr[idx]) - i32::from(prev[idx])).unsigned_abs() as f64;
            col_diff[x] += d;
        }
    }
    for v in &mut col_diff {
        *v /= height as f64;
    }

    // Per-row mean absolute difference
    let mut row_diff = vec![0.0f64; height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let d = (i32::from(curr[idx]) - i32::from(prev[idx])).unsigned_abs() as f64;
            row_diff[y] += d;
        }
    }
    for v in &mut row_diff {
        *v /= width as f64;
    }

    // Compute step score for a profile: best single-cut position that maximises
    // |mean_left - mean_right|.  Returns a value in [0, 1].
    let step_score = |profile: &[f64]| -> f64 {
        let n = profile.len();
        if n < 4 {
            return 0.0;
        }
        let total: f64 = profile.iter().sum::<f64>() / n as f64;
        if total < 1.0 {
            return 0.0; // ignore near-static frames
        }
        let mut best: f64 = 0.0;
        // slide the cut position across the profile
        let mut left_sum = 0.0f64;
        for k in 1..n {
            left_sum += profile[k - 1];
            let left_mean = left_sum / k as f64;
            let right_mean = (profile.iter().sum::<f64>() - left_sum) / (n - k) as f64;
            let score = (left_mean - right_mean).abs() / (total + 1.0);
            if score > best {
                best = score;
            }
        }
        best.min(1.0)
    };

    let col_score = step_score(&col_diff);
    let row_score = step_score(&row_diff);

    // A wipe has a strong step in either direction; take the maximum.
    col_score.max(row_score)
}

/// Compute edge change ratio.
fn edge_change_ratio(e1: &EdgeMap, e2: &EdgeMap) -> f64 {
    if e1.edges.len() != e2.edges.len() {
        return 0.0;
    }

    let edge_count1: usize = e1.edges.iter().filter(|&&e| e).count();
    let edge_count2: usize = e2.edges.iter().filter(|&&e| e).count();

    if edge_count1 == 0 && edge_count2 == 0 {
        return 0.0;
    }

    let max_edges = edge_count1.max(edge_count2);
    if max_edges == 0 {
        return 0.0;
    }

    // Count pixels that changed edge status
    let changed: usize = e1
        .edges
        .iter()
        .zip(e2.edges.iter())
        .filter(|(a, b)| a != b)
        .count();

    changed as f64 / max_edges as f64
}

// ---------------------------------------------------------------------------
// Content-Complexity Adaptive Threshold
// ---------------------------------------------------------------------------

/// Configuration for the adaptive scene detection threshold.
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdConfig {
    /// Base threshold applied when no motion history is available.
    pub base_threshold: f64,
    /// Window size (frames) used to compute the running motion statistics.
    pub window_size: usize,
    /// Minimum allowed threshold (clamped lower-bound).
    pub min_threshold: f64,
    /// Maximum allowed threshold (clamped upper-bound).
    pub max_threshold: f64,
    /// Sensitivity multiplier: higher values make the threshold more
    /// responsive to motion changes.
    pub sensitivity: f64,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.30,
            window_size: 30,
            min_threshold: 0.10,
            max_threshold: 0.70,
            sensitivity: 1.0,
        }
    }
}

/// Adaptive scene detection threshold that adjusts dynamically based on
/// observed motion statistics.
///
/// In high-motion content (sports, action) the local pixel variance is large,
/// so the histogram difference between consecutive frames is naturally higher
/// even without a scene change. `ContentComplexityAdaptiveThreshold` raises
/// the detection threshold in proportion to the recent motion level, thereby
/// reducing false positives in high-motion segments while remaining sensitive
/// in static or low-motion content.
///
/// # Algorithm
///
/// 1. A sliding window of `window_size` per-frame motion scores is maintained.
/// 2. The *mean* and *standard deviation* of the window are computed.
/// 3. The adaptive threshold is:
///    ```text
///    T_adaptive = T_base + sensitivity * (mean_motion + 0.5 * stddev_motion)
///    ```
///    clamped to `[min_threshold, max_threshold]`.
pub struct ContentComplexityAdaptiveThreshold {
    config: AdaptiveThresholdConfig,
    /// Circular buffer of recent motion scores.
    motion_window: std::collections::VecDeque<f64>,
    /// Currently active threshold.
    current_threshold: f64,
}

impl ContentComplexityAdaptiveThreshold {
    /// Create a new adaptive threshold with the given configuration.
    #[must_use]
    pub fn new(config: AdaptiveThresholdConfig) -> Self {
        let base = config.base_threshold;
        Self {
            config,
            motion_window: std::collections::VecDeque::new(),
            current_threshold: base,
        }
    }

    /// Create an adaptive threshold with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(AdaptiveThresholdConfig::default())
    }

    /// Record a new per-frame motion score and update the threshold.
    ///
    /// `motion_score` should be in the `[0.0, 1.0]` range; a value of `0.0`
    /// represents a completely static frame and `1.0` a maximally changed frame.
    pub fn update(&mut self, motion_score: f64) {
        if self.motion_window.len() >= self.config.window_size {
            self.motion_window.pop_front();
        }
        self.motion_window.push_back(motion_score.clamp(0.0, 1.0));
        self.recompute();
    }

    /// Return the currently active adaptive threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Determine whether a given histogram difference constitutes a scene change
    /// using the current adaptive threshold.
    #[must_use]
    pub fn is_scene_change(&self, hist_diff: f64) -> bool {
        hist_diff > self.current_threshold
    }

    /// Recompute the threshold from the current window statistics.
    fn recompute(&mut self) {
        if self.motion_window.is_empty() {
            self.current_threshold = self.config.base_threshold;
            return;
        }

        let n = self.motion_window.len() as f64;
        let mean: f64 = self.motion_window.iter().sum::<f64>() / n;
        let variance: f64 = self
            .motion_window
            .iter()
            .map(|&m| (m - mean) * (m - mean))
            .sum::<f64>()
            / n;
        let stddev = variance.sqrt();

        let adaptive = self.config.base_threshold + self.config.sensitivity * (mean + 0.5 * stddev);
        self.current_threshold =
            adaptive.clamp(self.config.min_threshold, self.config.max_threshold);
    }

    /// Number of frames currently in the motion history window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.motion_window.len()
    }

    /// Reset the motion history and restore the base threshold.
    pub fn reset(&mut self) {
        self.motion_window.clear();
        self.current_threshold = self.config.base_threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_same_frame() {
        let frame = vec![128u8; 64 * 64];
        let h1 = compute_histogram(&frame);
        let h2 = compute_histogram(&frame);
        let diff = histogram_difference(&h1, &h2);
        assert!((diff - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_different_frames() {
        let frame1 = vec![0u8; 64 * 64];
        let frame2 = vec![255u8; 64 * 64];
        let h1 = compute_histogram(&frame1);
        let h2 = compute_histogram(&frame2);
        let diff = histogram_difference(&h1, &h2);
        assert!(diff > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_edge_detection() {
        // Create a simple edge pattern
        let mut frame = vec![0u8; 100 * 100];
        for y in 0..100 {
            for x in 50..100 {
                frame[y * 100 + x] = 255;
            }
        }
        let edges = compute_edges(&frame, 100, 100);
        // Should detect vertical edge around x=50
        assert!(edges.edges.iter().filter(|&&e| e).count() > 0);
    }

    #[test]
    fn test_scene_detector() {
        let mut detector = SceneDetector::new(0.3);

        // Process a few identical frames
        let frame1 = vec![100u8; 64 * 64];
        for i in 0..5 {
            detector
                .process_frame(&frame1, 64, 64, i)
                .expect("frame processing should succeed");
        }

        // Process a different frame (scene cut)
        let frame2 = vec![200u8; 64 * 64];
        detector
            .process_frame(&frame2, 64, 64, 5)
            .expect("frame processing should succeed");

        // Process more identical frames
        for i in 6..10 {
            detector
                .process_frame(&frame2, 64, 64, i)
                .expect("frame processing should succeed");
        }

        let scenes = detector.finalize();
        assert!(!scenes.is_empty());
    }

    #[test]
    fn test_invalid_input() {
        let mut detector = SceneDetector::new(0.3);
        let frame = vec![0u8; 100]; // Too small
        let result = detector.process_frame(&frame, 1920, 1080, 0);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Dissolve / Wipe detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wipe_score_horizontal() {
        let width = 64usize;
        let height = 64usize;
        // Previous frame: all dark
        let prev = vec![30u8; width * height];
        // Current frame: right half is bright (hard wipe at x=32)
        let mut curr = vec![30u8; width * height];
        for y in 0..height {
            for x in 32..width {
                curr[y * width + x] = 200;
            }
        }
        let score = compute_wipe_score(&prev, &curr, width, height);
        // Should detect a clear horizontal wipe boundary
        assert!(score > 0.3, "wipe_score={score}");
    }

    #[test]
    fn test_wipe_score_uniform_change_is_low() {
        let width = 64usize;
        let height = 64usize;
        // Both frames differ uniformly (like a hard cut — no spatial boundary)
        let prev = vec![30u8; width * height];
        let curr = vec![200u8; width * height];
        let score = compute_wipe_score(&prev, &curr, width, height);
        // Uniform change → step score is near zero (no clear boundary)
        assert!(score < 0.3, "wipe_score={score}");
    }

    #[test]
    fn test_detector_detects_wipe_transition() {
        let width = 80usize;
        let height = 60usize;
        let mut detector = SceneDetector::new(0.3);

        // Scene A: dark frame
        let scene_a = vec![40u8; width * height];
        for i in 0..5 {
            detector
                .process_frame(&scene_a, width, height, i)
                .expect("should process");
        }

        // Wipe frames: left portion gradually replaced by bright scene
        for step in 1usize..=8 {
            let boundary = step * width / 8;
            let mut frame = vec![40u8; width * height];
            for y in 0..height {
                for x in boundary..width {
                    frame[y * width + x] = 200;
                }
            }
            detector
                .process_frame(&frame, width, height, 5 + step - 1)
                .expect("should process");
        }

        // Scene B: all bright
        let scene_b = vec![200u8; width * height];
        for i in 0..5 {
            detector
                .process_frame(&scene_b, width, height, 13 + i)
                .expect("should process");
        }

        let scenes = detector.finalize();
        // Should detect at least one scene change during the wipe
        assert!(!scenes.is_empty(), "Should detect wipe transition");
    }

    // -----------------------------------------------------------------------
    // ContentComplexityAdaptiveThreshold tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_threshold_default() {
        let at = ContentComplexityAdaptiveThreshold::default_config();
        // Before any updates, the threshold should be the base.
        assert!((at.threshold() - 0.30).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_threshold_high_motion_raises_threshold() {
        let mut at = ContentComplexityAdaptiveThreshold::default_config();
        // Feed high-motion frames.
        for _ in 0..30 {
            at.update(0.9);
        }
        // Threshold should have risen above the base.
        assert!(at.threshold() > 0.30, "threshold={}", at.threshold());
        assert!(at.threshold() <= 0.70);
    }

    #[test]
    fn test_adaptive_threshold_low_motion_stays_near_base() {
        let mut at = ContentComplexityAdaptiveThreshold::default_config();
        // Feed low-motion frames.
        for _ in 0..30 {
            at.update(0.01);
        }
        // Threshold should remain close to base (slightly above due to small motion).
        assert!(at.threshold() >= 0.10);
        assert!(at.threshold() < 0.40, "threshold={}", at.threshold());
    }

    #[test]
    fn test_adaptive_threshold_clamped_to_min_max() {
        let config = AdaptiveThresholdConfig {
            base_threshold: 0.30,
            min_threshold: 0.10,
            max_threshold: 0.70,
            window_size: 5,
            sensitivity: 10.0, // extreme sensitivity to force clamp
        };
        let mut at = ContentComplexityAdaptiveThreshold::new(config);
        for _ in 0..5 {
            at.update(1.0);
        }
        assert!(at.threshold() <= 0.70, "threshold={}", at.threshold());
    }

    #[test]
    fn test_adaptive_threshold_is_scene_change() {
        let at = ContentComplexityAdaptiveThreshold::default_config();
        // With base threshold 0.30, a diff of 0.5 should be a scene change.
        assert!(at.is_scene_change(0.5));
        // A diff of 0.1 should not be.
        assert!(!at.is_scene_change(0.1));
    }

    #[test]
    fn test_adaptive_threshold_window_size_respected() {
        let config = AdaptiveThresholdConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut at = ContentComplexityAdaptiveThreshold::new(config);
        for _ in 0..10 {
            at.update(0.5);
        }
        // Window should never exceed the configured size.
        assert_eq!(at.window_len(), 5);
    }

    #[test]
    fn test_adaptive_threshold_reset() {
        let mut at = ContentComplexityAdaptiveThreshold::default_config();
        for _ in 0..10 {
            at.update(0.9);
        }
        let elevated = at.threshold();
        at.reset();
        // After reset, threshold should be back to base.
        assert!((at.threshold() - 0.30).abs() < 1e-9);
        assert!(elevated > at.threshold());
        assert_eq!(at.window_len(), 0);
    }
}
