#![allow(dead_code)]
//! Automatic scene detection with smart analysis for automated editing.
//!
//! Provides configurable scene-boundary detection algorithms that combine
//! histogram difference, edge-change ratio, and content-adaptive thresholds
//! to produce high-quality scene cut lists.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Detection algorithm choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionAlgorithm {
    /// Simple threshold on histogram difference.
    HistogramDiff,
    /// Edge-change ratio analysis.
    EdgeChange,
    /// Content-adaptive threshold combining multiple signals.
    ContentAdaptive,
    /// Two-pass approach: coarse then fine.
    TwoPass,
}

/// Configuration for automatic scene detection.
#[derive(Debug, Clone)]
pub struct SceneDetectConfig {
    /// Algorithm to use.
    pub algorithm: DetectionAlgorithm,
    /// Threshold for histogram-diff based detection (0.0 – 1.0).
    pub histogram_threshold: f64,
    /// Threshold for edge-change detection (0.0 – 1.0).
    pub edge_threshold: f64,
    /// Minimum scene length in frames.
    pub min_scene_frames: usize,
    /// Window size for adaptive threshold computation.
    pub adaptive_window: usize,
    /// Multiplier applied to the adaptive mean to form the threshold.
    pub adaptive_multiplier: f64,
    /// Whether to merge very short scenes into neighbours.
    pub merge_short_scenes: bool,
}

impl Default for SceneDetectConfig {
    fn default() -> Self {
        Self {
            algorithm: DetectionAlgorithm::ContentAdaptive,
            histogram_threshold: 0.35,
            edge_threshold: 0.30,
            min_scene_frames: 10,
            adaptive_window: 30,
            adaptive_multiplier: 2.5,
            merge_short_scenes: true,
        }
    }
}

impl SceneDetectConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set algorithm.
    pub fn with_algorithm(mut self, algo: DetectionAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Builder: set histogram threshold.
    pub fn with_histogram_threshold(mut self, t: f64) -> Self {
        self.histogram_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Builder: set edge threshold.
    pub fn with_edge_threshold(mut self, t: f64) -> Self {
        self.edge_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Builder: set minimum scene length.
    pub fn with_min_scene_frames(mut self, n: usize) -> Self {
        self.min_scene_frames = n.max(1);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.histogram_threshold < 0.0 || self.histogram_threshold > 1.0 {
            return Err("histogram_threshold out of range".into());
        }
        if self.edge_threshold < 0.0 || self.edge_threshold > 1.0 {
            return Err("edge_threshold out of range".into());
        }
        if self.adaptive_window == 0 {
            return Err("adaptive_window must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Scene boundary representation
// ---------------------------------------------------------------------------

/// Type of scene boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Hard cut.
    Cut,
    /// Dissolve / cross-fade.
    Dissolve,
    /// Fade to/from black.
    Fade,
    /// Wipe transition.
    Wipe,
}

/// A detected scene boundary.
#[derive(Debug, Clone)]
pub struct SceneBoundary {
    /// Frame index of the boundary.
    pub frame_index: usize,
    /// Confidence score (0.0 – 1.0).
    pub confidence: f64,
    /// Detected boundary type.
    pub boundary_type: BoundaryType,
    /// Metric value that triggered detection.
    pub metric_value: f64,
}

/// A detected scene (range of frames).
#[derive(Debug, Clone)]
pub struct DetectedScene {
    /// Start frame (inclusive).
    pub start_frame: usize,
    /// End frame (exclusive).
    pub end_frame: usize,
    /// Average metric value inside this scene.
    pub avg_metric: f64,
    /// Peak metric value inside this scene.
    pub peak_metric: f64,
}

impl DetectedScene {
    /// Duration in frames.
    pub fn length(&self) -> usize {
        self.end_frame.saturating_sub(self.start_frame)
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Automatic scene detector.
#[derive(Debug)]
pub struct SceneDetector {
    /// Configuration.
    config: SceneDetectConfig,
}

impl SceneDetector {
    /// Create a new detector with the given config.
    pub fn new(config: SceneDetectConfig) -> Self {
        Self { config }
    }

    /// Detect scene boundaries from a sequence of per-frame metric values.
    ///
    /// `metrics` is a slice of f64 in `[0, 1]` – one value per inter-frame
    /// difference (so `metrics.len() == frame_count - 1`).
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_boundaries(&self, metrics: &[f64]) -> Vec<SceneBoundary> {
        match self.config.algorithm {
            DetectionAlgorithm::HistogramDiff => {
                self.detect_fixed_threshold(metrics, self.config.histogram_threshold)
            }
            DetectionAlgorithm::EdgeChange => {
                self.detect_fixed_threshold(metrics, self.config.edge_threshold)
            }
            DetectionAlgorithm::ContentAdaptive => self.detect_adaptive(metrics),
            DetectionAlgorithm::TwoPass => self.detect_two_pass(metrics),
        }
    }

    /// Fixed-threshold detection.
    fn detect_fixed_threshold(&self, metrics: &[f64], threshold: f64) -> Vec<SceneBoundary> {
        let mut boundaries = Vec::new();
        let mut last_boundary: Option<usize> = None;

        for (i, &val) in metrics.iter().enumerate() {
            if val >= threshold {
                if let Some(last) = last_boundary {
                    if i - last < self.config.min_scene_frames {
                        continue;
                    }
                }
                boundaries.push(SceneBoundary {
                    frame_index: i + 1, // boundary is *at* the new frame
                    confidence: (val / threshold).min(1.0),
                    boundary_type: BoundaryType::Cut,
                    metric_value: val,
                });
                last_boundary = Some(i);
            }
        }
        boundaries
    }

    /// Content-adaptive detection using a sliding window.
    #[allow(clippy::cast_precision_loss)]
    fn detect_adaptive(&self, metrics: &[f64]) -> Vec<SceneBoundary> {
        let win = self.config.adaptive_window;
        let mult = self.config.adaptive_multiplier;
        let mut window: VecDeque<f64> = VecDeque::with_capacity(win);
        let mut boundaries = Vec::new();
        let mut last_boundary: Option<usize> = None;

        for (i, &val) in metrics.iter().enumerate() {
            window.push_back(val);
            if window.len() > win {
                window.pop_front();
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let threshold = (mean * mult).min(1.0);

            if val >= threshold && val > 0.05 {
                if let Some(last) = last_boundary {
                    if i - last < self.config.min_scene_frames {
                        continue;
                    }
                }
                let confidence = if threshold > 0.0 {
                    (val / threshold).min(1.0)
                } else {
                    1.0
                };
                boundaries.push(SceneBoundary {
                    frame_index: i + 1,
                    confidence,
                    boundary_type: Self::classify_boundary(val),
                    metric_value: val,
                });
                last_boundary = Some(i);
            }
        }
        boundaries
    }

    /// Two-pass detection: first coarse, then refine.
    fn detect_two_pass(&self, metrics: &[f64]) -> Vec<SceneBoundary> {
        // Pass 1: coarse with a high threshold
        let coarse_threshold = self.config.histogram_threshold * 1.5;
        let coarse = self.detect_fixed_threshold(metrics, coarse_threshold.min(1.0));

        // Pass 2: lower threshold in the neighbourhood of coarse boundaries
        let neighbourhood = self.config.min_scene_frames;
        let fine_threshold = self.config.histogram_threshold * 0.8;
        let mut refined = Vec::new();

        for b in &coarse {
            let start = b.frame_index.saturating_sub(neighbourhood);
            let end = (b.frame_index + neighbourhood).min(metrics.len());
            let mut best_idx = b.frame_index;
            let mut best_val = b.metric_value;
            for idx in start..end {
                if metrics[idx] > best_val {
                    best_val = metrics[idx];
                    best_idx = idx + 1;
                }
            }
            if best_val >= fine_threshold {
                refined.push(SceneBoundary {
                    frame_index: best_idx,
                    confidence: (best_val / fine_threshold).min(1.0),
                    boundary_type: Self::classify_boundary(best_val),
                    metric_value: best_val,
                });
            }
        }
        refined
    }

    /// Classify a boundary type based on metric magnitude.
    fn classify_boundary(metric: f64) -> BoundaryType {
        if metric > 0.8 {
            BoundaryType::Cut
        } else if metric > 0.5 {
            BoundaryType::Dissolve
        } else {
            BoundaryType::Fade
        }
    }

    /// Convert boundaries into scene ranges.
    pub fn boundaries_to_scenes(
        &self,
        boundaries: &[SceneBoundary],
        total_frames: usize,
        metrics: &[f64],
    ) -> Vec<DetectedScene> {
        let mut scenes = Vec::new();
        let mut prev = 0_usize;

        for b in boundaries {
            let start = prev;
            let end = b.frame_index;
            if end > start {
                let slice = &metrics[start..end.min(metrics.len())];
                let (avg, peak) = Self::slice_stats(slice);
                scenes.push(DetectedScene {
                    start_frame: start,
                    end_frame: end,
                    avg_metric: avg,
                    peak_metric: peak,
                });
            }
            prev = end;
        }
        // Trailing scene
        if prev < total_frames {
            let slice = &metrics[prev..metrics.len()];
            let (avg, peak) = Self::slice_stats(slice);
            scenes.push(DetectedScene {
                start_frame: prev,
                end_frame: total_frames,
                avg_metric: avg,
                peak_metric: peak,
            });
        }
        scenes
    }

    /// Compute mean and max of a slice.
    #[allow(clippy::cast_precision_loss)]
    fn slice_stats(slice: &[f64]) -> (f64, f64) {
        if slice.is_empty() {
            return (0.0, 0.0);
        }
        let sum: f64 = slice.iter().sum();
        let peak = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (sum / slice.len() as f64, peak)
    }

    /// Merge short scenes that are below the minimum length.
    pub fn merge_short_scenes(&self, scenes: &[DetectedScene]) -> Vec<DetectedScene> {
        if scenes.is_empty() {
            return Vec::new();
        }
        let min_len = self.config.min_scene_frames;
        let mut merged: Vec<DetectedScene> = Vec::new();

        for scene in scenes {
            if let Some(last) = merged.last_mut() {
                if scene.length() < min_len {
                    // Merge into previous
                    last.end_frame = scene.end_frame;
                    last.avg_metric = (last.avg_metric + scene.avg_metric) / 2.0;
                    if scene.peak_metric > last.peak_metric {
                        last.peak_metric = scene.peak_metric;
                    }
                    continue;
                }
            }
            merged.push(scene.clone());
        }
        merged
    }

    /// Reference to the config.
    pub fn config(&self) -> &SceneDetectConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics_with_cuts(len: usize, cut_positions: &[usize], cut_val: f64) -> Vec<f64> {
        let mut m = vec![0.05; len];
        for &pos in cut_positions {
            if pos < len {
                m[pos] = cut_val;
            }
        }
        m
    }

    #[test]
    fn test_config_default() {
        let cfg = SceneDetectConfig::default();
        assert_eq!(cfg.algorithm, DetectionAlgorithm::ContentAdaptive);
        assert!(cfg.histogram_threshold > 0.0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::HistogramDiff)
            .with_histogram_threshold(0.5)
            .with_edge_threshold(0.4)
            .with_min_scene_frames(20);
        assert_eq!(cfg.algorithm, DetectionAlgorithm::HistogramDiff);
        assert!((cfg.histogram_threshold - 0.5).abs() < 1e-9);
        assert_eq!(cfg.min_scene_frames, 20);
    }

    #[test]
    fn test_config_threshold_clamping() {
        let cfg = SceneDetectConfig::new()
            .with_histogram_threshold(2.0)
            .with_edge_threshold(-1.0);
        assert!((cfg.histogram_threshold - 1.0).abs() < 1e-9);
        assert!((cfg.edge_threshold - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_validate_bad_window() {
        let mut cfg = SceneDetectConfig::default();
        cfg.adaptive_window = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_fixed_threshold_detection() {
        let metrics = make_metrics_with_cuts(100, &[25, 50, 75], 0.8);
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::HistogramDiff)
            .with_histogram_threshold(0.35)
            .with_min_scene_frames(5);
        let det = SceneDetector::new(cfg);
        let boundaries = det.detect_boundaries(&metrics);
        assert_eq!(boundaries.len(), 3);
        assert_eq!(boundaries[0].frame_index, 26);
    }

    #[test]
    fn test_min_scene_frames_suppression() {
        // Two cuts only 3 frames apart – second should be suppressed
        let metrics = make_metrics_with_cuts(100, &[20, 23], 0.8);
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::HistogramDiff)
            .with_histogram_threshold(0.35)
            .with_min_scene_frames(10);
        let det = SceneDetector::new(cfg);
        let boundaries = det.detect_boundaries(&metrics);
        assert_eq!(boundaries.len(), 1);
    }

    #[test]
    fn test_adaptive_detection() {
        let metrics = make_metrics_with_cuts(200, &[50, 120, 180], 0.9);
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::ContentAdaptive)
            .with_min_scene_frames(5);
        let det = SceneDetector::new(cfg);
        let boundaries = det.detect_boundaries(&metrics);
        assert!(boundaries.len() >= 2); // adaptive may merge some
    }

    #[test]
    fn test_two_pass_detection() {
        let metrics = make_metrics_with_cuts(150, &[40, 90], 0.85);
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::TwoPass)
            .with_histogram_threshold(0.35)
            .with_min_scene_frames(5);
        let det = SceneDetector::new(cfg);
        let boundaries = det.detect_boundaries(&metrics);
        assert!(!boundaries.is_empty());
    }

    #[test]
    fn test_boundary_classification() {
        assert_eq!(SceneDetector::classify_boundary(0.9), BoundaryType::Cut);
        assert_eq!(
            SceneDetector::classify_boundary(0.6),
            BoundaryType::Dissolve
        );
        assert_eq!(SceneDetector::classify_boundary(0.3), BoundaryType::Fade);
    }

    #[test]
    fn test_boundaries_to_scenes() {
        let metrics = make_metrics_with_cuts(100, &[30, 60], 0.8);
        let cfg = SceneDetectConfig::new()
            .with_algorithm(DetectionAlgorithm::HistogramDiff)
            .with_histogram_threshold(0.35)
            .with_min_scene_frames(5);
        let det = SceneDetector::new(cfg);
        let boundaries = det.detect_boundaries(&metrics);
        let scenes = det.boundaries_to_scenes(&boundaries, 100, &metrics);
        assert!(scenes.len() >= 2);
        assert_eq!(scenes[0].start_frame, 0);
    }

    #[test]
    fn test_merge_short_scenes() {
        let scenes = vec![
            DetectedScene {
                start_frame: 0,
                end_frame: 50,
                avg_metric: 0.05,
                peak_metric: 0.1,
            },
            DetectedScene {
                start_frame: 50,
                end_frame: 53,
                avg_metric: 0.04,
                peak_metric: 0.06,
            },
            DetectedScene {
                start_frame: 53,
                end_frame: 100,
                avg_metric: 0.05,
                peak_metric: 0.1,
            },
        ];
        let cfg = SceneDetectConfig::new().with_min_scene_frames(10);
        let det = SceneDetector::new(cfg);
        let merged = det.merge_short_scenes(&scenes);
        // The 3-frame scene should be merged
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].end_frame, 53);
    }

    #[test]
    fn test_detected_scene_length() {
        let s = DetectedScene {
            start_frame: 10,
            end_frame: 40,
            avg_metric: 0.0,
            peak_metric: 0.0,
        };
        assert_eq!(s.length(), 30);
    }

    #[test]
    fn test_empty_metrics() {
        let det = SceneDetector::new(SceneDetectConfig::default());
        let boundaries = det.detect_boundaries(&[]);
        assert!(boundaries.is_empty());
    }
}
