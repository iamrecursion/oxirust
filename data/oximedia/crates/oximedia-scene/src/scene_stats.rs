#![allow(dead_code)]
//! Per-scene metrics and aggregate statistics across a sequence of scenes.

/// Metrics describing the visual and temporal properties of a single scene.
#[derive(Debug, Clone)]
pub struct SceneMetrics {
    /// Duration of the scene in frames.
    pub duration_frames: u64,
    /// Mean inter-frame motion magnitude normalised to [0.0, 1.0].
    pub mean_motion: f32,
    /// Mean luminance of the scene normalised to [0.0, 1.0].
    pub mean_luminance: f32,
    /// Mean colour saturation normalised to [0.0, 1.0].
    pub mean_saturation: f32,
}

impl SceneMetrics {
    /// Create a new `SceneMetrics`.
    #[must_use]
    pub fn new(
        duration_frames: u64,
        mean_motion: f32,
        mean_luminance: f32,
        mean_saturation: f32,
    ) -> Self {
        Self {
            duration_frames,
            mean_motion: mean_motion.clamp(0.0, 1.0),
            mean_luminance: mean_luminance.clamp(0.0, 1.0),
            mean_saturation: mean_saturation.clamp(0.0, 1.0),
        }
    }

    /// Heuristic motion intensity category.
    ///
    /// - `< 0.2` → "low"
    /// - `< 0.5` → "medium"
    /// - `>= 0.5` → "high"
    #[must_use]
    pub fn motion_intensity(&self) -> &'static str {
        if self.mean_motion < 0.2 {
            "low"
        } else if self.mean_motion < 0.5 {
            "medium"
        } else {
            "high"
        }
    }
}

// ---------------------------------------------------------------------------

/// Accumulated statistics over a sequence of scenes.
#[derive(Debug, Clone, Default)]
pub struct SceneStatistics {
    scenes: Vec<SceneMetrics>,
}

impl SceneStatistics {
    /// Create an empty `SceneStatistics`.
    #[must_use]
    pub fn new() -> Self {
        Self { scenes: Vec::new() }
    }

    /// Add a scene's metrics to the statistics.
    pub fn update(&mut self, metrics: SceneMetrics) {
        self.scenes.push(metrics);
    }

    /// Number of scenes accumulated.
    #[must_use]
    pub fn count(&self) -> usize {
        self.scenes.len()
    }

    /// Average scene duration in frames, or `None` if no scenes have been added.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg_duration_frames(&self) -> Option<f64> {
        if self.scenes.is_empty() {
            return None;
        }
        let total: u64 = self.scenes.iter().map(|s| s.duration_frames).sum();
        Some(total as f64 / self.scenes.len() as f64)
    }

    /// The scene with the most frames, or `None` if empty.
    #[must_use]
    pub fn longest_scene(&self) -> Option<&SceneMetrics> {
        self.scenes.iter().max_by_key(|s| s.duration_frames)
    }

    /// The scene with the fewest frames, or `None` if empty.
    #[must_use]
    pub fn shortest_scene(&self) -> Option<&SceneMetrics> {
        self.scenes.iter().min_by_key(|s| s.duration_frames)
    }

    /// Mean motion across all scenes, or `None` if empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_motion(&self) -> Option<f32> {
        if self.scenes.is_empty() {
            return None;
        }
        let sum: f32 = self.scenes.iter().map(|s| s.mean_motion).sum();
        Some(sum / self.scenes.len() as f32)
    }

    /// Mean luminance across all scenes, or `None` if empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_luminance(&self) -> Option<f32> {
        if self.scenes.is_empty() {
            return None;
        }
        let sum: f32 = self.scenes.iter().map(|s| s.mean_luminance).sum();
        Some(sum / self.scenes.len() as f32)
    }

    /// Clear all accumulated data.
    pub fn clear(&mut self) {
        self.scenes.clear();
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics(duration: u64, motion: f32) -> SceneMetrics {
        SceneMetrics::new(duration, motion, 0.5, 0.5)
    }

    // --- SceneMetrics ---

    #[test]
    fn test_motion_intensity_low() {
        let m = make_metrics(24, 0.1);
        assert_eq!(m.motion_intensity(), "low");
    }

    #[test]
    fn test_motion_intensity_medium() {
        let m = make_metrics(24, 0.35);
        assert_eq!(m.motion_intensity(), "medium");
    }

    #[test]
    fn test_motion_intensity_high() {
        let m = make_metrics(24, 0.8);
        assert_eq!(m.motion_intensity(), "high");
    }

    #[test]
    fn test_metrics_clamps_values() {
        let m = SceneMetrics::new(10, 2.0, -0.5, 1.5);
        assert!((m.mean_motion - 1.0).abs() < f32::EPSILON);
        assert!((m.mean_luminance - 0.0).abs() < f32::EPSILON);
        assert!((m.mean_saturation - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_metrics_boundary_motion() {
        let m = make_metrics(100, 0.2); // exactly 0.2 → medium
        assert_eq!(m.motion_intensity(), "medium");
    }

    // --- SceneStatistics ---

    #[test]
    fn test_stats_empty() {
        let stats = SceneStatistics::new();
        assert_eq!(stats.count(), 0);
        assert!(stats.avg_duration_frames().is_none());
        assert!(stats.longest_scene().is_none());
        assert!(stats.shortest_scene().is_none());
        assert!(stats.mean_motion().is_none());
    }

    #[test]
    fn test_stats_update_and_count() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(30, 0.1));
        stats.update(make_metrics(60, 0.6));
        assert_eq!(stats.count(), 2);
    }

    #[test]
    fn test_avg_duration_single() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(48, 0.0));
        let avg = stats.avg_duration_frames().expect("should succeed in test");
        assert!((avg - 48.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_duration_multiple() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(20, 0.0));
        stats.update(make_metrics(40, 0.0));
        stats.update(make_metrics(60, 0.0));
        let avg = stats.avg_duration_frames().expect("should succeed in test");
        assert!((avg - 40.0).abs() < 1e-9, "avg={avg}");
    }

    #[test]
    fn test_longest_scene() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(10, 0.0));
        stats.update(make_metrics(100, 0.0));
        stats.update(make_metrics(50, 0.0));
        assert_eq!(
            stats
                .longest_scene()
                .expect("should succeed in test")
                .duration_frames,
            100
        );
    }

    #[test]
    fn test_shortest_scene() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(10, 0.0));
        stats.update(make_metrics(100, 0.0));
        stats.update(make_metrics(50, 0.0));
        assert_eq!(
            stats
                .shortest_scene()
                .expect("should succeed in test")
                .duration_frames,
            10
        );
    }

    #[test]
    fn test_mean_motion() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(24, 0.2));
        stats.update(make_metrics(24, 0.4));
        let m = stats.mean_motion().expect("should succeed in test");
        assert!((m - 0.3).abs() < 1e-5, "mean_motion={m}");
    }

    #[test]
    fn test_clear() {
        let mut stats = SceneStatistics::new();
        stats.update(make_metrics(30, 0.5));
        stats.clear();
        assert_eq!(stats.count(), 0);
        assert!(stats.avg_duration_frames().is_none());
    }

    #[test]
    fn test_mean_luminance() {
        let mut stats = SceneStatistics::new();
        stats.update(SceneMetrics::new(24, 0.3, 0.2, 0.5));
        stats.update(SceneMetrics::new(24, 0.3, 0.6, 0.5));
        let lum = stats.mean_luminance().expect("should succeed in test");
        assert!((lum - 0.4).abs() < 1e-5, "mean_luminance={lum}");
    }
}
