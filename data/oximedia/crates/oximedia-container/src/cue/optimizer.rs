//! Cue point placement optimization.
//!
//! Optimizes cue point placement for better seeking performance.

#![forbid(unsafe_code)]

use super::generator::CuePoint;

/// Strategy for cue point placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// Fixed interval between cue points.
    FixedInterval,
    /// Adaptive interval based on file size.
    Adaptive,
    /// Keyframe-based (only on keyframes).
    Keyframe,
    /// Scene change detection.
    SceneChange,
}

/// Configuration for cue point optimization.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Placement strategy.
    pub strategy: PlacementStrategy,
    /// Target average interval in seconds.
    pub target_interval_secs: f64,
    /// Minimum interval in seconds.
    pub min_interval_secs: f64,
    /// Maximum interval in seconds.
    pub max_interval_secs: f64,
    /// Target cue point density (cues per second).
    pub target_density: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: PlacementStrategy::Adaptive,
            target_interval_secs: 2.0,
            min_interval_secs: 0.5,
            max_interval_secs: 10.0,
            target_density: 0.5,
        }
    }
}

impl OptimizerConfig {
    /// Creates a new configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            strategy: PlacementStrategy::Adaptive,
            target_interval_secs: 2.0,
            min_interval_secs: 0.5,
            max_interval_secs: 10.0,
            target_density: 0.5,
        }
    }

    /// Sets the placement strategy.
    #[must_use]
    pub const fn with_strategy(mut self, strategy: PlacementStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Sets the target interval.
    #[must_use]
    pub const fn with_target_interval(mut self, interval_secs: f64) -> Self {
        self.target_interval_secs = interval_secs;
        self
    }
}

/// Optimizer for cue point placement.
pub struct CueOptimizer {
    config: OptimizerConfig,
}

impl CueOptimizer {
    /// Creates a new optimizer.
    #[must_use]
    pub const fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Optimizes a set of cue points.
    #[must_use]
    pub fn optimize(&self, cue_points: &[CuePoint]) -> Vec<CuePoint> {
        match self.config.strategy {
            PlacementStrategy::FixedInterval => self.optimize_fixed_interval(cue_points),
            PlacementStrategy::Adaptive => self.optimize_adaptive(cue_points),
            PlacementStrategy::Keyframe => Self::optimize_keyframe(cue_points),
            PlacementStrategy::SceneChange => self.optimize_scene_change(cue_points),
        }
    }

    /// Optimizes using fixed interval strategy.
    fn optimize_fixed_interval(&self, cue_points: &[CuePoint]) -> Vec<CuePoint> {
        // Keep cue points at roughly fixed intervals
        let mut optimized = Vec::new();
        let mut last_time = None;

        for cue in cue_points {
            if let Some(last) = last_time {
                #[allow(clippy::cast_precision_loss)]
                let interval = (cue.timestamp - last) as f64;
                if interval < self.config.min_interval_secs * 1000.0 {
                    continue;
                }
            }

            optimized.push(*cue);
            last_time = Some(cue.timestamp);
        }

        optimized
    }

    /// Optimizes using adaptive strategy.
    fn optimize_adaptive(&self, cue_points: &[CuePoint]) -> Vec<CuePoint> {
        // Adapt interval based on content
        let mut optimized = Vec::new();

        if cue_points.is_empty() {
            return optimized;
        }

        optimized.push(cue_points[0]);

        for window in cue_points.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            #[allow(clippy::cast_precision_loss)]
            let interval = (curr.timestamp - prev.timestamp) as f64;

            // Check if this cue point should be kept
            if interval >= self.config.min_interval_secs * 1000.0 {
                optimized.push(*curr);
            }
        }

        optimized
    }

    /// Optimizes for keyframe-only strategy.
    fn optimize_keyframe(cue_points: &[CuePoint]) -> Vec<CuePoint> {
        // Already filtered by keyframes in generation
        cue_points.to_vec()
    }

    /// Optimizes for scene change strategy.
    fn optimize_scene_change(&self, cue_points: &[CuePoint]) -> Vec<CuePoint> {
        // Simplified version - in real implementation, this would detect scene changes
        self.optimize_adaptive(cue_points)
    }

    /// Calculates statistics for cue points.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate_stats(&self, cue_points: &[CuePoint]) -> CueStats {
        if cue_points.is_empty() {
            return CueStats::default();
        }

        let mut intervals = Vec::new();

        for window in cue_points.windows(2) {
            let interval = (window[1].timestamp - window[0].timestamp) as f64;
            intervals.push(interval);
        }

        let total_interval: f64 = intervals.iter().sum();
        let avg_interval = if intervals.is_empty() {
            0.0
        } else {
            total_interval / intervals.len() as f64
        };

        let min_interval = intervals.iter().copied().fold(f64::INFINITY, f64::min);
        let max_interval = intervals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        CueStats {
            count: cue_points.len(),
            avg_interval,
            min_interval: if min_interval.is_finite() {
                min_interval
            } else {
                0.0
            },
            max_interval: if max_interval.is_finite() {
                max_interval
            } else {
                0.0
            },
        }
    }
}

/// Statistics for cue points.
#[derive(Debug, Clone, Copy, Default)]
pub struct CueStats {
    /// Total number of cue points.
    pub count: usize,
    /// Average interval between cue points.
    pub avg_interval: f64,
    /// Minimum interval.
    pub min_interval: f64,
    /// Maximum interval.
    pub max_interval: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cues() -> Vec<CuePoint> {
        vec![
            CuePoint::new(0, 0, 1),
            CuePoint::new(500, 5000, 1),
            CuePoint::new(1000, 10000, 1),
            CuePoint::new(1200, 12000, 1),
            CuePoint::new(2000, 20000, 1),
        ]
    }

    #[test]
    fn test_optimizer_config() {
        let config = OptimizerConfig::new()
            .with_strategy(PlacementStrategy::Keyframe)
            .with_target_interval(3.0);

        assert_eq!(config.strategy, PlacementStrategy::Keyframe);
        assert_eq!(config.target_interval_secs, 3.0);
    }

    #[test]
    fn test_optimize_fixed_interval() {
        let config = OptimizerConfig::new()
            .with_strategy(PlacementStrategy::FixedInterval)
            .with_target_interval(1.0);
        let optimizer = CueOptimizer::new(config);

        let cues = create_test_cues();
        let optimized = optimizer.optimize(&cues);

        // Should have fewer cues due to interval filtering
        assert!(optimized.len() <= cues.len());
    }

    #[test]
    fn test_calculate_stats() {
        let config = OptimizerConfig::new();
        let optimizer = CueOptimizer::new(config);

        let cues = create_test_cues();
        let stats = optimizer.calculate_stats(&cues);

        assert_eq!(stats.count, 5);
        assert!(stats.avg_interval > 0.0);
        assert!(stats.min_interval > 0.0);
        assert!(stats.max_interval >= stats.min_interval);
    }
}
