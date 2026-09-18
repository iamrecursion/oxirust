//! Two-pass encoding optimization.
//!
//! Provides:
//! - First-pass analysis: complexity per frame / scene
//! - Bitrate allocation based on complexity scores
//! - VBV (Video Buffer Verifier) compliance checking
//! - Second-pass bitrate plan generation

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Complexity category of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum FrameComplexity {
    /// Very simple frame (flat, static).
    Low,
    /// Typical frame.
    Medium,
    /// High-motion or high-detail frame.
    High,
    /// Scene change or I-frame.
    SceneChange,
}

/// Per-frame statistics from the first pass.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FirstPassFrame {
    /// Frame index.
    pub index: usize,
    /// Frame type: 'I', 'P', or 'B'.
    pub frame_type: char,
    /// Complexity score (0.0–1.0).
    pub complexity: f64,
    /// Intra cost estimate.
    pub intra_cost: f64,
    /// Inter cost estimate.
    pub inter_cost: f64,
    /// Number of non-zero coefficients (proxy for texture detail).
    pub nnz: usize,
}

impl FirstPassFrame {
    /// Determine complexity category from the score.
    #[must_use]
    pub fn complexity_category(&self) -> FrameComplexity {
        if self.frame_type == 'I' && self.complexity > 0.7 {
            FrameComplexity::SceneChange
        } else if self.complexity < 0.25 {
            FrameComplexity::Low
        } else if self.complexity < 0.65 {
            FrameComplexity::Medium
        } else {
            FrameComplexity::High
        }
    }
}

/// VBV (Video Buffer Verifier) parameters.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VbvParams {
    /// VBV buffer size in bits.
    pub buffer_size: u64,
    /// Maximum instantaneous bitrate in bits/s.
    pub max_bitrate: u64,
    /// Initial buffer fullness (0.0–1.0).
    pub initial_fill: f64,
}

impl Default for VbvParams {
    fn default() -> Self {
        Self {
            buffer_size: 4_000_000, // 4 Mbit
            max_bitrate: 8_000_000, // 8 Mbit/s
            initial_fill: 0.9,
        }
    }
}

/// Configuration for two-pass encoding.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TwoPassConfig {
    /// Target average bitrate in bits/s.
    pub target_bitrate: u64,
    /// Frame rate (fps).
    pub fps: f64,
    /// VBV constraints.
    pub vbv: VbvParams,
    /// Complexity weight: how much to vary bitrate based on complexity (0.0–1.0).
    pub complexity_weight: f64,
    /// Minimum per-frame bitrate as a fraction of target.
    pub min_bitrate_fraction: f64,
    /// Maximum per-frame bitrate as a fraction of target.
    pub max_bitrate_fraction: f64,
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self {
            target_bitrate: 2_000_000,
            fps: 24.0,
            vbv: VbvParams::default(),
            complexity_weight: 0.7,
            min_bitrate_fraction: 0.2,
            max_bitrate_fraction: 3.0,
        }
    }
}

/// Per-frame bitrate allocation from the second pass.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FrameBitAllocation {
    /// Frame index.
    pub frame_index: usize,
    /// Allocated bits for this frame.
    pub bits: u64,
    /// Estimated VBV buffer level after this frame (bits).
    pub vbv_level: u64,
    /// Whether this frame is VBV-compliant.
    pub vbv_compliant: bool,
}

/// Two-pass encoder optimizer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TwoPassOptimizer {
    config: TwoPassConfig,
}

impl TwoPassOptimizer {
    /// Create a new two-pass optimizer.
    #[must_use]
    pub fn new(config: TwoPassConfig) -> Self {
        Self { config }
    }

    /// Simulate first-pass analysis given raw complexity scores.
    ///
    /// `complexity_scores` is one value per frame (0.0–1.0).
    /// Frame types are assigned: I at index 0 and every `gop_size`, P otherwise.
    #[must_use]
    pub fn first_pass(&self, complexity_scores: &[f64], gop_size: usize) -> Vec<FirstPassFrame> {
        complexity_scores
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let frame_type = if i % gop_size == 0 { 'I' } else { 'P' };
                let intra_cost = c * 100.0 + 10.0;
                let inter_cost = if frame_type == 'P' {
                    c * 50.0 + 5.0
                } else {
                    intra_cost
                };
                FirstPassFrame {
                    index: i,
                    frame_type,
                    complexity: c,
                    intra_cost,
                    inter_cost,
                    nnz: (c * 1024.0) as usize,
                }
            })
            .collect()
    }

    /// Allocate bits per frame based on first-pass stats.
    ///
    /// Returns a `FrameBitAllocation` for each frame.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn allocate_bits(&self, frames: &[FirstPassFrame]) -> Vec<FrameBitAllocation> {
        if frames.is_empty() {
            return Vec::new();
        }

        // Compute target bits per frame.
        let bits_per_frame = self.config.target_bitrate as f64 / self.config.fps;

        // Compute complexity-weighted allocations.
        let total_complexity: f64 = frames.iter().map(|f| f.complexity).sum();
        let avg_complexity = if total_complexity > 0.0 {
            total_complexity / frames.len() as f64
        } else {
            1.0
        };

        let min_bits = (bits_per_frame * self.config.min_bitrate_fraction) as u64;
        let max_bits = (bits_per_frame * self.config.max_bitrate_fraction) as u64;

        // VBV simulation
        let mut vbv_level =
            (self.config.vbv.buffer_size as f64 * self.config.vbv.initial_fill) as u64;
        let bits_drained_per_frame = (self.config.vbv.max_bitrate as f64 / self.config.fps) as u64;

        frames
            .iter()
            .map(|f| {
                let weight = if avg_complexity > 0.0 {
                    1.0 + self.config.complexity_weight * (f.complexity / avg_complexity - 1.0)
                } else {
                    1.0
                };
                let allocated = ((bits_per_frame * weight) as u64).clamp(min_bits, max_bits);

                // VBV: add bits, drain according to max_bitrate
                let new_level = vbv_level
                    .saturating_add(bits_drained_per_frame)
                    .saturating_sub(allocated);
                let vbv_compliant = new_level <= self.config.vbv.buffer_size;
                vbv_level = new_level.min(self.config.vbv.buffer_size);

                FrameBitAllocation {
                    frame_index: f.index,
                    bits: allocated,
                    vbv_level,
                    vbv_compliant,
                }
            })
            .collect()
    }

    /// Check overall VBV compliance of a bit allocation plan.
    #[must_use]
    pub fn check_vbv_compliance(allocations: &[FrameBitAllocation]) -> bool {
        allocations.iter().all(|a| a.vbv_compliant)
    }

    /// Compute the average allocated bitrate from a plan.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_bitrate(allocations: &[FrameBitAllocation], fps: f64) -> f64 {
        if allocations.is_empty() {
            return 0.0;
        }
        let total_bits: u64 = allocations.iter().map(|a| a.bits).sum();
        total_bits as f64 / allocations.len() as f64 * fps
    }

    /// Returns the config.
    #[must_use]
    pub fn config(&self) -> &TwoPassConfig {
        &self.config
    }
}

/// Estimate scene-change count from first-pass frames.
#[must_use]
pub fn count_scene_changes(frames: &[FirstPassFrame]) -> usize {
    frames
        .iter()
        .filter(|f| f.complexity_category() == FrameComplexity::SceneChange)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores(n: usize, value: f64) -> Vec<f64> {
        vec![value; n]
    }

    #[test]
    fn test_first_pass_frame_count() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let frames = opt.first_pass(&make_scores(30, 0.5), 12);
        assert_eq!(frames.len(), 30);
    }

    #[test]
    fn test_first_pass_i_frames_at_gop_boundaries() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let frames = opt.first_pass(&make_scores(24, 0.5), 8);
        assert_eq!(frames[0].frame_type, 'I');
        assert_eq!(frames[8].frame_type, 'I');
        assert_eq!(frames[16].frame_type, 'I');
        assert_eq!(frames[4].frame_type, 'P');
    }

    #[test]
    fn test_first_pass_complexity_stored() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let scores = vec![0.1, 0.5, 0.9];
        let frames = opt.first_pass(&scores, 10);
        assert!((frames[0].complexity - 0.1).abs() < 1e-9);
        assert!((frames[1].complexity - 0.5).abs() < 1e-9);
        assert!((frames[2].complexity - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_complexity_category_low() {
        let f = FirstPassFrame {
            index: 0,
            frame_type: 'P',
            complexity: 0.1,
            intra_cost: 10.0,
            inter_cost: 5.0,
            nnz: 100,
        };
        assert_eq!(f.complexity_category(), FrameComplexity::Low);
    }

    #[test]
    fn test_complexity_category_medium() {
        let f = FirstPassFrame {
            index: 0,
            frame_type: 'P',
            complexity: 0.5,
            intra_cost: 50.0,
            inter_cost: 25.0,
            nnz: 512,
        };
        assert_eq!(f.complexity_category(), FrameComplexity::Medium);
    }

    #[test]
    fn test_complexity_category_high() {
        let f = FirstPassFrame {
            index: 0,
            frame_type: 'P',
            complexity: 0.8,
            intra_cost: 90.0,
            inter_cost: 45.0,
            nnz: 800,
        };
        assert_eq!(f.complexity_category(), FrameComplexity::High);
    }

    #[test]
    fn test_complexity_category_scene_change() {
        let f = FirstPassFrame {
            index: 0,
            frame_type: 'I',
            complexity: 0.9,
            intra_cost: 100.0,
            inter_cost: 100.0,
            nnz: 1000,
        };
        assert_eq!(f.complexity_category(), FrameComplexity::SceneChange);
    }

    #[test]
    fn test_allocate_bits_count() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let frames = opt.first_pass(&make_scores(24, 0.5), 8);
        let allocs = opt.allocate_bits(&frames);
        assert_eq!(allocs.len(), 24);
    }

    #[test]
    fn test_allocate_bits_empty() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let allocs = opt.allocate_bits(&[]);
        assert!(allocs.is_empty());
    }

    #[test]
    fn test_allocate_bits_min_max_respected() {
        let config = TwoPassConfig {
            target_bitrate: 2_000_000,
            fps: 24.0,
            min_bitrate_fraction: 0.1,
            max_bitrate_fraction: 5.0,
            ..Default::default()
        };
        let opt = TwoPassOptimizer::new(config.clone());
        let frames = opt.first_pass(&make_scores(60, 0.5), 12);
        let allocs = opt.allocate_bits(&frames);
        let bits_per_frame = config.target_bitrate as f64 / config.fps;
        let min = (bits_per_frame * config.min_bitrate_fraction) as u64;
        let max = (bits_per_frame * config.max_bitrate_fraction) as u64;
        for a in &allocs {
            assert!(a.bits >= min && a.bits <= max);
        }
    }

    #[test]
    fn test_average_bitrate_non_zero() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let frames = opt.first_pass(&make_scores(24, 0.5), 8);
        let allocs = opt.allocate_bits(&frames);
        let avg = TwoPassOptimizer::average_bitrate(&allocs, 24.0);
        assert!(avg > 0.0);
    }

    #[test]
    fn test_average_bitrate_empty() {
        let avg = TwoPassOptimizer::average_bitrate(&[], 24.0);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_vbv_compliance_check() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let frames = opt.first_pass(&make_scores(24, 0.4), 8);
        let allocs = opt.allocate_bits(&frames);
        // Just check it returns a bool without panic
        let _ = TwoPassOptimizer::check_vbv_compliance(&allocs);
    }

    #[test]
    fn test_count_scene_changes() {
        let opt = TwoPassOptimizer::new(TwoPassConfig::default());
        let mut scores = vec![0.5f64; 20];
        scores[0] = 0.95; // I-frame scene change
        scores[8] = 0.95; // I-frame scene change
        let frames = opt.first_pass(&scores, 8);
        let sc = count_scene_changes(&frames);
        assert!(sc >= 2);
    }

    #[test]
    fn test_two_pass_config_default() {
        let config = TwoPassConfig::default();
        assert_eq!(config.target_bitrate, 2_000_000);
        assert!((config.fps - 24.0).abs() < 1e-9);
        assert!((config.complexity_weight - 0.7).abs() < 1e-9);
    }
}
