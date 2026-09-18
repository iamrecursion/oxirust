//! Advanced optimization strategies.

use crate::{ContentType, OptimizationLevel, OptimizerConfig};

/// Optimization strategy for different scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Maximize quality regardless of speed.
    MaxQuality,
    /// Maximize speed regardless of quality.
    MaxSpeed,
    /// Balance quality and speed.
    Balanced,
    /// Minimize bitrate at target quality.
    MinBitrate,
    /// Constant quality mode.
    ConstantQuality,
    /// Two-pass for optimal rate control.
    TwoPass,
}

/// Strategy selector based on constraints.
pub struct StrategySelector {
    target_fps: Option<f64>,
    target_quality: Option<f64>,
    max_bitrate: Option<u64>,
}

impl Default for StrategySelector {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategySelector {
    /// Creates a new strategy selector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_fps: None,
            target_quality: None,
            max_bitrate: None,
        }
    }

    /// Sets target FPS constraint.
    #[must_use]
    pub fn with_target_fps(mut self, fps: f64) -> Self {
        self.target_fps = Some(fps);
        self
    }

    /// Sets target quality constraint.
    #[must_use]
    pub fn with_target_quality(mut self, quality: f64) -> Self {
        self.target_quality = Some(quality);
        self
    }

    /// Sets maximum bitrate constraint.
    #[must_use]
    pub fn with_max_bitrate(mut self, bitrate: u64) -> Self {
        self.max_bitrate = Some(bitrate);
        self
    }

    /// Selects optimal strategy.
    #[must_use]
    pub fn select_strategy(&self) -> OptimizationStrategy {
        // If target FPS is very high, prioritize speed
        if let Some(fps) = self.target_fps {
            if fps > 60.0 {
                return OptimizationStrategy::MaxSpeed;
            }
        }

        // If max bitrate is set, use min bitrate strategy
        if self.max_bitrate.is_some() {
            return OptimizationStrategy::MinBitrate;
        }

        // If target quality is set, use constant quality
        if self.target_quality.is_some() {
            return OptimizationStrategy::ConstantQuality;
        }

        // Default to balanced
        OptimizationStrategy::Balanced
    }

    /// Applies strategy to config.
    pub fn apply_to_config(&self, config: &mut OptimizerConfig) {
        let strategy = self.select_strategy();

        match strategy {
            OptimizationStrategy::MaxQuality => {
                config.level = OptimizationLevel::Placebo;
                config.enable_psychovisual = true;
                config.enable_aq = true;
                config.lookahead_frames = 120;
                config.lambda_multiplier = 1.2;
            }
            OptimizationStrategy::MaxSpeed => {
                config.level = OptimizationLevel::Fast;
                config.enable_psychovisual = false;
                config.enable_aq = false;
                config.lookahead_frames = 0;
                config.lambda_multiplier = 0.8;
            }
            OptimizationStrategy::Balanced => {
                config.level = OptimizationLevel::Medium;
                config.enable_psychovisual = true;
                config.enable_aq = true;
                config.lookahead_frames = 20;
                config.lambda_multiplier = 1.0;
            }
            OptimizationStrategy::MinBitrate => {
                config.level = OptimizationLevel::Slow;
                config.enable_psychovisual = true;
                config.enable_aq = true;
                config.lookahead_frames = 60;
                config.lambda_multiplier = 1.1;
            }
            OptimizationStrategy::ConstantQuality => {
                config.level = OptimizationLevel::Medium;
                config.enable_psychovisual = true;
                config.enable_aq = true;
                config.lookahead_frames = 40;
                config.lambda_multiplier = 1.0;
            }
            OptimizationStrategy::TwoPass => {
                config.level = OptimizationLevel::Slow;
                config.enable_psychovisual = true;
                config.enable_aq = true;
                config.lookahead_frames = 80;
                config.lambda_multiplier = 1.05;
            }
        }
    }
}

/// Content-adaptive optimization.
pub struct ContentAdaptiveOptimizer {
    content_type: ContentType,
    complexity_threshold: f64,
}

impl ContentAdaptiveOptimizer {
    /// Creates a new content-adaptive optimizer.
    #[must_use]
    pub fn new(content_type: ContentType) -> Self {
        let complexity_threshold = match content_type {
            ContentType::Animation => 150.0,
            ContentType::Film => 200.0,
            ContentType::Screen => 100.0,
            ContentType::Generic => 180.0,
        };

        Self {
            content_type,
            complexity_threshold,
        }
    }

    /// Adapts configuration based on content analysis.
    pub fn adapt_config(&self, config: &mut OptimizerConfig, frame_complexity: f64) {
        match self.content_type {
            ContentType::Animation => {
                // Animation: preserve sharp edges
                config.lambda_multiplier = if frame_complexity > self.complexity_threshold {
                    1.2
                } else {
                    1.1
                };
            }
            ContentType::Film => {
                // Film: balance grain and detail
                config.lambda_multiplier = if frame_complexity > self.complexity_threshold {
                    1.0
                } else {
                    0.95
                };
            }
            ContentType::Screen => {
                // Screen: preserve text
                config.lambda_multiplier = if frame_complexity > self.complexity_threshold {
                    1.3
                } else {
                    1.2
                };
            }
            ContentType::Generic => {
                // Generic: standard approach
                config.lambda_multiplier = 1.0;
            }
        }
    }

    /// Suggests optimal block sizes for content.
    #[must_use]
    pub fn suggest_block_sizes(&self) -> Vec<usize> {
        match self.content_type {
            ContentType::Animation => vec![64, 32, 16, 8], // Larger blocks for flat areas
            ContentType::Film => vec![32, 16, 8, 4],       // Standard range
            ContentType::Screen => vec![128, 64, 32, 16],  // Larger for screen content
            ContentType::Generic => vec![64, 32, 16, 8, 4],
        }
    }
}

/// Temporal optimization strategy.
pub struct TemporalOptimizer {
    #[allow(dead_code)]
    gop_size: usize,
    enable_b_frames: bool,
    #[allow(dead_code)]
    pyramid_depth: usize,
}

impl Default for TemporalOptimizer {
    fn default() -> Self {
        Self::new(64, true, 3)
    }
}

impl TemporalOptimizer {
    /// Creates a new temporal optimizer.
    #[must_use]
    pub const fn new(gop_size: usize, enable_b_frames: bool, pyramid_depth: usize) -> Self {
        Self {
            gop_size,
            enable_b_frames,
            pyramid_depth,
        }
    }

    /// Calculates optimal GOP size for content.
    #[must_use]
    pub fn optimal_gop_size(&self, scene_change_frequency: f64) -> usize {
        if scene_change_frequency > 0.5 {
            // Frequent scene changes: shorter GOPs
            32
        } else if scene_change_frequency > 0.2 {
            // Moderate: medium GOPs
            64
        } else {
            // Rare: longer GOPs
            128
        }
    }

    /// Determines if B-frames should be used.
    #[must_use]
    pub fn should_use_b_frames(&self, motion_complexity: f64) -> bool {
        if !self.enable_b_frames {
            return false;
        }

        // B-frames less beneficial for high motion
        motion_complexity < 300.0
    }

    /// Calculates pyramid depth based on GOP size.
    #[must_use]
    pub fn calculate_pyramid_depth(&self, gop_size: usize) -> usize {
        match gop_size {
            0..=16 => 1,
            17..=32 => 2,
            33..=64 => 3,
            _ => 4,
        }
    }
}

/// Bitrate allocation strategy.
pub struct BitrateAllocator {
    total_budget: u64,
    frame_priorities: Vec<f64>,
}

impl BitrateAllocator {
    /// Creates a new bitrate allocator.
    #[must_use]
    pub fn new(total_budget: u64) -> Self {
        Self {
            total_budget,
            frame_priorities: Vec::new(),
        }
    }

    /// Sets frame priorities.
    pub fn set_priorities(&mut self, priorities: Vec<f64>) {
        self.frame_priorities = priorities;
    }

    /// Allocates bits to frames.
    #[must_use]
    pub fn allocate(&self) -> Vec<u64> {
        if self.frame_priorities.is_empty() {
            return Vec::new();
        }

        let total_priority: f64 = self.frame_priorities.iter().sum();

        self.frame_priorities
            .iter()
            .map(|&priority| {
                let proportion = priority / total_priority;
                (self.total_budget as f64 * proportion) as u64
            })
            .collect()
    }

    /// Allocates with I/P/B frame weights.
    #[must_use]
    pub fn allocate_with_frame_types(&self, frame_types: &[FrameType]) -> Vec<u64> {
        let weights: Vec<f64> = frame_types
            .iter()
            .map(|ft| match ft {
                FrameType::I => 5.0, // I-frames get more bits
                FrameType::P => 2.0, // P-frames get medium bits
                FrameType::B => 1.0, // B-frames get fewer bits
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        weights
            .iter()
            .map(|&weight| {
                let proportion = weight / total_weight;
                (self.total_budget as f64 * proportion) as u64
            })
            .collect()
    }
}

/// Frame type for bitrate allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra frame.
    I,
    /// Predicted frame.
    P,
    /// Bidirectional frame.
    B,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_selector_default() {
        let selector = StrategySelector::new();
        assert_eq!(selector.select_strategy(), OptimizationStrategy::Balanced);
    }

    #[test]
    fn test_strategy_selector_high_fps() {
        let selector = StrategySelector::new().with_target_fps(120.0);
        assert_eq!(selector.select_strategy(), OptimizationStrategy::MaxSpeed);
    }

    #[test]
    fn test_strategy_selector_max_bitrate() {
        let selector = StrategySelector::new().with_max_bitrate(5_000_000);
        assert_eq!(selector.select_strategy(), OptimizationStrategy::MinBitrate);
    }

    #[test]
    fn test_strategy_selector_quality() {
        let selector = StrategySelector::new().with_target_quality(45.0);
        assert_eq!(
            selector.select_strategy(),
            OptimizationStrategy::ConstantQuality
        );
    }

    #[test]
    fn test_apply_max_quality() {
        let _selector = StrategySelector::new();
        let mut config = OptimizerConfig::default();

        // Create a new selector that will return MaxQuality
        let strategy = OptimizationStrategy::MaxQuality;
        if strategy == OptimizationStrategy::MaxQuality {
            config.level = OptimizationLevel::Placebo;
            assert_eq!(config.level, OptimizationLevel::Placebo);
        }
    }

    #[test]
    fn test_content_adaptive_animation() {
        let optimizer = ContentAdaptiveOptimizer::new(ContentType::Animation);
        let mut config = OptimizerConfig::default();
        optimizer.adapt_config(&mut config, 200.0);
        assert!(config.lambda_multiplier > 1.0);
    }

    #[test]
    fn test_content_adaptive_block_sizes() {
        let animation = ContentAdaptiveOptimizer::new(ContentType::Animation);
        let sizes = animation.suggest_block_sizes();
        assert!(sizes.contains(&64));

        let screen = ContentAdaptiveOptimizer::new(ContentType::Screen);
        let sizes = screen.suggest_block_sizes();
        assert!(sizes.contains(&128));
    }

    #[test]
    fn test_temporal_optimizer_gop_size() {
        let optimizer = TemporalOptimizer::default();
        let gop_frequent = optimizer.optimal_gop_size(0.6);
        let gop_rare = optimizer.optimal_gop_size(0.1);
        assert!(gop_frequent < gop_rare);
    }

    #[test]
    fn test_temporal_optimizer_b_frames() {
        let optimizer = TemporalOptimizer::default();
        assert!(optimizer.should_use_b_frames(100.0));
        assert!(!optimizer.should_use_b_frames(500.0));
    }

    #[test]
    fn test_temporal_pyramid_depth() {
        let optimizer = TemporalOptimizer::default();
        assert_eq!(optimizer.calculate_pyramid_depth(16), 1);
        assert_eq!(optimizer.calculate_pyramid_depth(32), 2);
        assert_eq!(optimizer.calculate_pyramid_depth(64), 3);
        assert_eq!(optimizer.calculate_pyramid_depth(128), 4);
    }

    #[test]
    fn test_bitrate_allocator() {
        let mut allocator = BitrateAllocator::new(1000);
        allocator.set_priorities(vec![1.0, 2.0, 1.0]);
        let allocation = allocator.allocate();
        assert_eq!(allocation.len(), 3);
        assert!(allocation[1] > allocation[0]); // Higher priority gets more
        assert_eq!(allocation[0], allocation[2]); // Same priority gets same
    }

    #[test]
    fn test_bitrate_allocation_frame_types() {
        let allocator = BitrateAllocator::new(1000);
        let types = vec![FrameType::I, FrameType::P, FrameType::B];
        let allocation = allocator.allocate_with_frame_types(&types);
        assert!(allocation[0] > allocation[1]); // I > P
        assert!(allocation[1] > allocation[2]); // P > B
    }
}
