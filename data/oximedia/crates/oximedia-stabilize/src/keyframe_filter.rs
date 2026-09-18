#![allow(dead_code)]
//! Keyframe selection and filtering for video stabilization.
//!
//! This module provides algorithms for selecting keyframes from a video sequence
//! and filtering them to produce optimal stabilization anchors. Keyframes are
//! reference frames that have high quality, low blur, and good feature
//! distribution, making them ideal anchors for inter-frame motion estimation.
//!
//! # Features
//!
//! - **Sharpness scoring**: Laplacian-based sharpness estimation
//! - **Coverage scoring**: Spatial feature distribution analysis
//! - **Temporal spacing**: Minimum/maximum keyframe interval enforcement
//! - **Adaptive selection**: Scene-change aware keyframe placement
//! - **Budget-constrained selection**: Select at most N keyframes

use std::collections::VecDeque;

/// Strategy used to select keyframes from a video sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeStrategy {
    /// Select keyframes at fixed intervals.
    FixedInterval,
    /// Select keyframes based on quality scores.
    QualityBased,
    /// Select keyframes at detected scene changes.
    SceneChange,
    /// Adaptive selection combining quality and scene analysis.
    Adaptive,
}

/// Configuration for keyframe filtering.
#[derive(Debug, Clone)]
pub struct KeyframeConfig {
    /// Strategy for keyframe selection.
    pub strategy: KeyframeStrategy,
    /// Minimum interval between keyframes (in frames).
    pub min_interval: usize,
    /// Maximum interval between keyframes (in frames).
    pub max_interval: usize,
    /// Sharpness weight in composite score (0.0..=1.0).
    pub sharpness_weight: f64,
    /// Coverage weight in composite score (0.0..=1.0).
    pub coverage_weight: f64,
    /// Scene change threshold (0.0..=1.0).
    pub scene_change_threshold: f64,
    /// Maximum number of keyframes to select (0 = unlimited).
    pub max_keyframes: usize,
}

impl Default for KeyframeConfig {
    fn default() -> Self {
        Self {
            strategy: KeyframeStrategy::Adaptive,
            min_interval: 5,
            max_interval: 60,
            sharpness_weight: 0.6,
            coverage_weight: 0.4,
            scene_change_threshold: 0.35,
            max_keyframes: 0,
        }
    }
}

impl KeyframeConfig {
    /// Create a new keyframe configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the selection strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: KeyframeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the minimum interval between keyframes.
    #[must_use]
    pub const fn with_min_interval(mut self, interval: usize) -> Self {
        self.min_interval = interval;
        self
    }

    /// Set the maximum interval between keyframes.
    #[must_use]
    pub const fn with_max_interval(mut self, interval: usize) -> Self {
        self.max_interval = interval;
        self
    }

    /// Set the maximum number of keyframes.
    #[must_use]
    pub const fn with_max_keyframes(mut self, max: usize) -> Self {
        self.max_keyframes = max;
        self
    }
}

/// A single keyframe with associated quality metrics.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Frame index in the original sequence.
    pub frame_index: usize,
    /// Sharpness score (higher = sharper).
    pub sharpness: f64,
    /// Feature coverage score (higher = better distribution).
    pub coverage: f64,
    /// Combined composite score.
    pub composite_score: f64,
    /// Whether this keyframe sits at a scene boundary.
    pub is_scene_change: bool,
}

impl Keyframe {
    /// Create a new keyframe entry.
    #[must_use]
    pub fn new(frame_index: usize, sharpness: f64, coverage: f64) -> Self {
        Self {
            frame_index,
            sharpness,
            coverage,
            composite_score: 0.0,
            is_scene_change: false,
        }
    }

    /// Compute the composite score using the given weights.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn with_composite(mut self, sharpness_w: f64, coverage_w: f64) -> Self {
        self.composite_score = self.sharpness * sharpness_w + self.coverage * coverage_w;
        self
    }

    /// Mark this keyframe as a scene change.
    #[must_use]
    pub const fn with_scene_change(mut self, is_scene: bool) -> Self {
        self.is_scene_change = is_scene;
        self
    }
}

/// Per-frame analysis metrics used for keyframe scoring.
#[derive(Debug, Clone)]
pub struct FrameMetrics {
    /// Frame index.
    pub index: usize,
    /// Sharpness score computed via Laplacian variance.
    pub sharpness: f64,
    /// Feature spatial coverage score.
    pub coverage: f64,
    /// Inter-frame difference (for scene change detection).
    pub frame_diff: f64,
}

/// Keyframe filter that selects optimal keyframes from a sequence.
#[derive(Debug)]
pub struct KeyframeFilter {
    /// Filter configuration.
    config: KeyframeConfig,
    /// History of recent frame metrics for sliding window analysis.
    history: VecDeque<FrameMetrics>,
}

impl KeyframeFilter {
    /// Create a new keyframe filter with the given configuration.
    #[must_use]
    pub fn new(config: KeyframeConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
        }
    }

    /// Create a keyframe filter with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(KeyframeConfig::default())
    }

    /// Estimate sharpness of pixel data using Laplacian variance approximation.
    ///
    /// The data is treated as a row-major grayscale image.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_sharpness(data: &[u8], width: usize, height: usize) -> f64 {
        if width < 3 || height < 3 || data.len() < width * height {
            return 0.0;
        }
        let mut sum = 0.0_f64;
        let mut count = 0u64;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center = f64::from(data[y * width + x]);
                let top = f64::from(data[(y - 1) * width + x]);
                let bottom = f64::from(data[(y + 1) * width + x]);
                let left = f64::from(data[y * width + (x - 1)]);
                let right = f64::from(data[y * width + (x + 1)]);
                let laplacian = (top + bottom + left + right) - 4.0 * center;
                sum += laplacian * laplacian;
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        sum / count as f64
    }

    /// Estimate feature spatial coverage by dividing the frame into a grid.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_coverage(data: &[u8], width: usize, height: usize, grid_size: usize) -> f64 {
        if width == 0 || height == 0 || grid_size == 0 || data.len() < width * height {
            return 0.0;
        }
        let cols = (width + grid_size - 1) / grid_size;
        let rows = (height + grid_size - 1) / grid_size;
        let total_cells = cols * rows;
        if total_cells == 0 {
            return 0.0;
        }
        let mut active_cells = 0usize;
        let variance_threshold = 100.0_f64;
        for gy in 0..rows {
            for gx in 0..cols {
                let x0 = gx * grid_size;
                let y0 = gy * grid_size;
                let x1 = (x0 + grid_size).min(width);
                let y1 = (y0 + grid_size).min(height);
                let mut sum = 0.0_f64;
                let mut sum_sq = 0.0_f64;
                let mut n = 0u64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let v = f64::from(data[y * width + x]);
                        sum += v;
                        sum_sq += v * v;
                        n += 1;
                    }
                }
                if n > 1 {
                    let mean = sum / n as f64;
                    let var = (sum_sq / n as f64) - mean * mean;
                    if var > variance_threshold {
                        active_cells += 1;
                    }
                }
            }
        }
        active_cells as f64 / total_cells as f64
    }

    /// Compute inter-frame difference as mean absolute difference.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn frame_difference(a: &[u8], b: &[u8]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 1.0;
        }
        let sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&va, &vb)| f64::from(va.abs_diff(vb)))
            .sum();
        sum / (a.len() as f64 * 255.0)
    }

    /// Select keyframes from pre-computed frame metrics using the configured strategy.
    #[must_use]
    pub fn select(&self, metrics: &[FrameMetrics]) -> Vec<Keyframe> {
        if metrics.is_empty() {
            return Vec::new();
        }
        match self.config.strategy {
            KeyframeStrategy::FixedInterval => self.select_fixed_interval(metrics),
            KeyframeStrategy::QualityBased => self.select_quality_based(metrics),
            KeyframeStrategy::SceneChange => self.select_scene_change(metrics),
            KeyframeStrategy::Adaptive => self.select_adaptive(metrics),
        }
    }

    /// Fixed interval selection.
    fn select_fixed_interval(&self, metrics: &[FrameMetrics]) -> Vec<Keyframe> {
        let interval = self.config.max_interval.max(1);
        let mut keyframes = Vec::new();
        let mut i = 0;
        while i < metrics.len() {
            let m = &metrics[i];
            let kf = Keyframe::new(m.index, m.sharpness, m.coverage)
                .with_composite(self.config.sharpness_weight, self.config.coverage_weight);
            keyframes.push(kf);
            i += interval;
        }
        self.apply_budget(keyframes)
    }

    /// Quality-based selection: pick the best frame in each window.
    fn select_quality_based(&self, metrics: &[FrameMetrics]) -> Vec<Keyframe> {
        let window = self.config.max_interval.max(1);
        let mut keyframes = Vec::new();
        let mut pos = 0;
        while pos < metrics.len() {
            let end = (pos + window).min(metrics.len());
            let best = metrics[pos..end]
                .iter()
                .max_by(|a, b| {
                    let sa = a.sharpness * self.config.sharpness_weight
                        + a.coverage * self.config.coverage_weight;
                    let sb = b.sharpness * self.config.sharpness_weight
                        + b.coverage * self.config.coverage_weight;
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("invariant: window slice is non-empty");
            let kf = Keyframe::new(best.index, best.sharpness, best.coverage)
                .with_composite(self.config.sharpness_weight, self.config.coverage_weight);
            keyframes.push(kf);
            pos = end;
        }
        self.apply_budget(keyframes)
    }

    /// Scene-change based selection.
    fn select_scene_change(&self, metrics: &[FrameMetrics]) -> Vec<Keyframe> {
        let mut keyframes = Vec::new();
        // Always include first frame
        let first = &metrics[0];
        keyframes.push(
            Keyframe::new(first.index, first.sharpness, first.coverage)
                .with_composite(self.config.sharpness_weight, self.config.coverage_weight)
                .with_scene_change(true),
        );
        let mut last_kf_idx = 0;
        for (i, m) in metrics.iter().enumerate().skip(1) {
            let is_scene = m.frame_diff > self.config.scene_change_threshold;
            let gap = i - last_kf_idx;
            if is_scene && gap >= self.config.min_interval {
                keyframes.push(
                    Keyframe::new(m.index, m.sharpness, m.coverage)
                        .with_composite(self.config.sharpness_weight, self.config.coverage_weight)
                        .with_scene_change(true),
                );
                last_kf_idx = i;
            } else if gap >= self.config.max_interval {
                keyframes.push(
                    Keyframe::new(m.index, m.sharpness, m.coverage)
                        .with_composite(self.config.sharpness_weight, self.config.coverage_weight),
                );
                last_kf_idx = i;
            }
        }
        self.apply_budget(keyframes)
    }

    /// Adaptive selection combining quality and scene change.
    fn select_adaptive(&self, metrics: &[FrameMetrics]) -> Vec<Keyframe> {
        let mut keyframes = Vec::new();
        let first = &metrics[0];
        keyframes.push(
            Keyframe::new(first.index, first.sharpness, first.coverage)
                .with_composite(self.config.sharpness_weight, self.config.coverage_weight),
        );
        let mut last_kf_idx = 0;
        for (i, m) in metrics.iter().enumerate().skip(1) {
            let gap = i - last_kf_idx;
            let is_scene = m.frame_diff > self.config.scene_change_threshold;
            if is_scene && gap >= self.config.min_interval {
                keyframes.push(
                    Keyframe::new(m.index, m.sharpness, m.coverage)
                        .with_composite(self.config.sharpness_weight, self.config.coverage_weight)
                        .with_scene_change(true),
                );
                last_kf_idx = i;
            } else if gap >= self.config.max_interval {
                // Pick best in window
                let start = last_kf_idx + 1;
                let end = (i + 1).min(metrics.len());
                if let Some(best) = metrics[start..end].iter().max_by(|a, b| {
                    let sa = a.sharpness * self.config.sharpness_weight
                        + a.coverage * self.config.coverage_weight;
                    let sb = b.sharpness * self.config.sharpness_weight
                        + b.coverage * self.config.coverage_weight;
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    keyframes.push(
                        Keyframe::new(best.index, best.sharpness, best.coverage).with_composite(
                            self.config.sharpness_weight,
                            self.config.coverage_weight,
                        ),
                    );
                    last_kf_idx = i;
                }
            }
        }
        self.apply_budget(keyframes)
    }

    /// Apply the maximum keyframe budget by keeping highest-scoring ones.
    fn apply_budget(&self, mut keyframes: Vec<Keyframe>) -> Vec<Keyframe> {
        if self.config.max_keyframes > 0 && keyframes.len() > self.config.max_keyframes {
            keyframes.sort_by(|a, b| {
                b.composite_score
                    .partial_cmp(&a.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            keyframes.truncate(self.config.max_keyframes);
            keyframes.sort_by_key(|kf| kf.frame_index);
        }
        keyframes
    }

    /// Add a frame metric to the internal history buffer.
    pub fn push_metric(&mut self, metric: FrameMetrics) {
        self.history.push_back(metric);
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    /// Get the number of buffered metrics.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub const fn config(&self) -> &KeyframeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(n: usize) -> Vec<FrameMetrics> {
        (0..n)
            .map(|i| FrameMetrics {
                index: i,
                sharpness: (i as f64 * 0.1).sin().abs(),
                coverage: 0.5 + (i as f64 * 0.05).cos() * 0.3,
                frame_diff: if i % 20 == 0 && i > 0 { 0.5 } else { 0.05 },
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = KeyframeConfig::default();
        assert_eq!(cfg.strategy, KeyframeStrategy::Adaptive);
        assert_eq!(cfg.min_interval, 5);
        assert_eq!(cfg.max_interval, 60);
    }

    #[test]
    fn test_config_builder() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::FixedInterval)
            .with_min_interval(10)
            .with_max_interval(30)
            .with_max_keyframes(5);
        assert_eq!(cfg.strategy, KeyframeStrategy::FixedInterval);
        assert_eq!(cfg.min_interval, 10);
        assert_eq!(cfg.max_interval, 30);
        assert_eq!(cfg.max_keyframes, 5);
    }

    #[test]
    fn test_keyframe_creation() {
        let kf = Keyframe::new(42, 0.8, 0.6)
            .with_composite(0.6, 0.4)
            .with_scene_change(true);
        assert_eq!(kf.frame_index, 42);
        assert!(kf.is_scene_change);
        let expected = 0.8 * 0.6 + 0.6 * 0.4;
        assert!((kf.composite_score - expected).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_interval_selection() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::FixedInterval)
            .with_max_interval(10);
        let filter = KeyframeFilter::new(cfg);
        let metrics = make_metrics(50);
        let kfs = filter.select(&metrics);
        assert!(!kfs.is_empty());
        assert_eq!(kfs[0].frame_index, 0);
        // Should have roughly 50/10 = 5 keyframes
        assert_eq!(kfs.len(), 5);
    }

    #[test]
    fn test_quality_based_selection() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::QualityBased)
            .with_max_interval(10);
        let filter = KeyframeFilter::new(cfg);
        let metrics = make_metrics(30);
        let kfs = filter.select(&metrics);
        assert!(!kfs.is_empty());
        // 30 / 10 = 3 windows
        assert_eq!(kfs.len(), 3);
    }

    #[test]
    fn test_scene_change_selection() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::SceneChange)
            .with_min_interval(3)
            .with_max_interval(100);
        let filter = KeyframeFilter::new(cfg);
        let metrics = make_metrics(60);
        let kfs = filter.select(&metrics);
        // Should have first frame + scene changes at 20, 40
        assert!(kfs.len() >= 3);
        assert!(kfs.iter().any(|k| k.is_scene_change));
    }

    #[test]
    fn test_adaptive_selection() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::Adaptive)
            .with_min_interval(3)
            .with_max_interval(25);
        let filter = KeyframeFilter::new(cfg);
        let metrics = make_metrics(100);
        let kfs = filter.select(&metrics);
        assert!(!kfs.is_empty());
        // Verify ordering
        for pair in kfs.windows(2) {
            assert!(pair[0].frame_index < pair[1].frame_index);
        }
    }

    #[test]
    fn test_budget_constraint() {
        let cfg = KeyframeConfig::new()
            .with_strategy(KeyframeStrategy::FixedInterval)
            .with_max_interval(5)
            .with_max_keyframes(3);
        let filter = KeyframeFilter::new(cfg);
        let metrics = make_metrics(50);
        let kfs = filter.select(&metrics);
        assert!(kfs.len() <= 3);
    }

    #[test]
    fn test_empty_metrics() {
        let filter = KeyframeFilter::with_defaults();
        let kfs = filter.select(&[]);
        assert!(kfs.is_empty());
    }

    #[test]
    fn test_estimate_sharpness() {
        // Create a simple 5x5 image with edges
        let mut data = vec![128u8; 25];
        data[12] = 255; // center bright pixel
        let sharpness = KeyframeFilter::estimate_sharpness(&data, 5, 5);
        assert!(sharpness > 0.0);
    }

    #[test]
    fn test_estimate_sharpness_uniform() {
        let data = vec![128u8; 100];
        let sharpness = KeyframeFilter::estimate_sharpness(&data, 10, 10);
        assert!((sharpness - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_coverage() {
        let mut data = vec![0u8; 400];
        // Add variance in some cells
        for i in 0..100 {
            data[i] = (i % 256) as u8;
        }
        let coverage = KeyframeFilter::estimate_coverage(&data, 20, 20, 10);
        assert!(coverage >= 0.0 && coverage <= 1.0);
    }

    #[test]
    fn test_frame_difference() {
        let a = vec![100u8; 100];
        let b = vec![100u8; 100];
        let diff = KeyframeFilter::frame_difference(&a, &b);
        assert!((diff - 0.0).abs() < 1e-10);

        let c = vec![200u8; 100];
        let diff2 = KeyframeFilter::frame_difference(&a, &c);
        assert!(diff2 > 0.0);
    }

    #[test]
    fn test_push_metric_and_history() {
        let mut filter = KeyframeFilter::with_defaults();
        for i in 0..5 {
            filter.push_metric(FrameMetrics {
                index: i,
                sharpness: 0.5,
                coverage: 0.5,
                frame_diff: 0.01,
            });
        }
        assert_eq!(filter.history_len(), 5);
    }
}
