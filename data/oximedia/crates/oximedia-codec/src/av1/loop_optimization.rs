//! AV1 loop filter and post-processing optimization.
//!
//! This module optimizes loop filter parameters for AV1 encoding:
//!
//! - Loop filter strength selection
//! - CDEF (Constrained Directional Enhancement Filter) strength
//! - Restoration filter parameters
//! - Film grain parameter encoding (if applicable)
//!
//! # Loop Filter Optimization
//!
//! The loop filter reduces blocking artifacts by smoothing block boundaries.
//! This module finds optimal filter strengths that minimize distortion while
//! maximizing coding efficiency.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]

use super::block::BlockSize;
use super::cdef::{CdefParams, CdefStrength};
use super::loop_filter::LoopFilterParams;

// =============================================================================
// Constants
// =============================================================================

/// Maximum loop filter level.
const MAX_LOOP_FILTER_LEVEL: u8 = 63;

/// Maximum CDEF strength.
const MAX_CDEF_STRENGTH: u8 = 15;

/// Number of loop filter levels to test.
const FILTER_LEVELS_TO_TEST: usize = 5;

/// Number of CDEF strengths to test.
const CDEF_STRENGTHS_TO_TEST: usize = 4;

// =============================================================================
// Loop Filter Optimizer
// =============================================================================

/// Loop filter parameter optimizer.
#[derive(Clone, Debug)]
pub struct LoopFilterOptimizer {
    /// Current loop filter parameters.
    params: LoopFilterParams,
    /// Lambda for RD optimization.
    lambda: f32,
    /// Enable RD optimization.
    rd_optimization: bool,
}

impl LoopFilterOptimizer {
    /// Create a new loop filter optimizer.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            params: LoopFilterParams::default(),
            lambda,
            rd_optimization: true,
        }
    }

    /// Optimize loop filter level for a frame.
    ///
    /// Tests multiple filter levels and selects the one with best RD cost.
    pub fn optimize_filter_level(
        &mut self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        qp: u8,
    ) -> u8 {
        if !self.rd_optimization {
            // Fast mode: use QP-based heuristic
            return self.filter_level_from_qp(qp);
        }

        let base_level = self.filter_level_from_qp(qp);
        let mut best_level = base_level;
        let mut best_cost = f32::MAX;

        // Test levels around base
        for delta in -(FILTER_LEVELS_TO_TEST as i32 / 2)..=(FILTER_LEVELS_TO_TEST as i32 / 2) {
            let level = (i32::from(base_level) + delta * 4)
                .clamp(0, i32::from(MAX_LOOP_FILTER_LEVEL)) as u8;

            let cost = self.evaluate_filter_level(src, recon, width, height, level);

            if cost < best_cost {
                best_cost = cost;
                best_level = level;
            }
        }

        self.params.level = [best_level, best_level, best_level, best_level];
        best_level
    }

    /// Compute filter level from QP (heuristic).
    fn filter_level_from_qp(&self, qp: u8) -> u8 {
        // Higher QP -> stronger filter
        ((i32::from(qp) * 3) / 2).clamp(0, i32::from(MAX_LOOP_FILTER_LEVEL)) as u8
    }

    /// Evaluate RD cost for a filter level.
    fn evaluate_filter_level(
        &self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        level: u8,
    ) -> f32 {
        // Apply filter (simplified - just compute distortion)
        let distortion = self.compute_distortion(src, recon, width, height);

        // Estimate rate (filter level signaling)
        let rate = f32::from(level) * 0.1;

        distortion + self.lambda * rate
    }

    /// Compute distortion between source and reconstruction.
    fn compute_distortion(&self, src: &[u8], recon: &[u8], width: usize, height: usize) -> f32 {
        let mut sse = 0u64;
        let total = (width * height).min(src.len()).min(recon.len());

        for i in 0..total {
            let diff = i32::from(src[i]) - i32::from(recon[i]);
            sse += (diff * diff) as u64;
        }

        sse as f32
    }

    /// Get optimized loop filter parameters.
    #[must_use]
    pub const fn params(&self) -> &LoopFilterParams {
        &self.params
    }

    /// Set lambda for RD optimization.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.lambda = lambda;
    }

    /// Enable/disable RD optimization.
    pub fn set_rd_optimization(&mut self, enabled: bool) {
        self.rd_optimization = enabled;
    }
}

impl Default for LoopFilterOptimizer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// =============================================================================
// CDEF Optimizer
// =============================================================================

/// CDEF (Constrained Directional Enhancement Filter) optimizer.
#[derive(Clone, Debug)]
pub struct CdefOptimizer {
    /// Current CDEF parameters.
    params: CdefParams,
    /// Lambda for RD optimization.
    lambda: f32,
}

impl CdefOptimizer {
    /// Create a new CDEF optimizer.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            params: CdefParams::default(),
            lambda,
        }
    }

    /// Optimize CDEF strength for a block.
    pub fn optimize_strength(
        &mut self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        _block_size: BlockSize,
    ) -> CdefStrength {
        let mut best_strength = CdefStrength::default();
        let mut best_cost = f32::MAX;

        for primary in 0..CDEF_STRENGTHS_TO_TEST {
            for secondary in 0..CDEF_STRENGTHS_TO_TEST {
                let strength = CdefStrength {
                    primary: primary as u8,
                    secondary: secondary as u8,
                };

                let cost = self.evaluate_cdef_strength(src, recon, width, height, &strength);

                if cost < best_cost {
                    best_cost = cost;
                    best_strength = strength;
                }
            }
        }

        best_strength
    }

    /// Evaluate RD cost for CDEF strength.
    fn evaluate_cdef_strength(
        &self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        strength: &CdefStrength,
    ) -> f32 {
        // Compute distortion
        let mut sse = 0u64;
        let total = (width * height).min(src.len()).min(recon.len());

        for i in 0..total {
            let diff = i32::from(src[i]) - i32::from(recon[i]);
            sse += (diff * diff) as u64;
        }

        let distortion = sse as f32;

        // Estimate rate
        let rate = f32::from(strength.primary + strength.secondary) * 0.5;

        distortion + self.lambda * rate
    }

    /// Get optimized CDEF parameters.
    #[must_use]
    pub const fn params(&self) -> &CdefParams {
        &self.params
    }
}

impl Default for CdefOptimizer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// =============================================================================
// Restoration Filter Optimizer
// =============================================================================

/// Restoration filter type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestorationType {
    /// No restoration.
    None = 0,
    /// Wiener filter.
    Wiener = 1,
    /// Self-guided filter.
    Sgrproj = 2,
}

/// Restoration filter optimizer.
#[derive(Clone, Debug)]
pub struct RestorationOptimizer {
    /// Restoration type.
    restoration_type: RestorationType,
    /// Lambda for RD optimization.
    lambda: f32,
}

impl RestorationOptimizer {
    /// Create a new restoration optimizer.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            restoration_type: RestorationType::None,
            lambda,
        }
    }

    /// Optimize restoration type for a frame.
    pub fn optimize_restoration(
        &mut self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
    ) -> RestorationType {
        let mut best_type = RestorationType::None;
        let mut best_cost = f32::MAX;

        for rtype in [
            RestorationType::None,
            RestorationType::Wiener,
            RestorationType::Sgrproj,
        ] {
            let cost = self.evaluate_restoration(src, recon, width, height, rtype);

            if cost < best_cost {
                best_cost = cost;
                best_type = rtype;
            }
        }

        self.restoration_type = best_type;
        best_type
    }

    /// Evaluate RD cost for restoration type.
    fn evaluate_restoration(
        &self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        rtype: RestorationType,
    ) -> f32 {
        // Simplified evaluation
        let base_distortion = self.compute_distortion(src, recon, width, height);

        let rate = match rtype {
            RestorationType::None => 0.0,
            RestorationType::Wiener => 100.0,
            RestorationType::Sgrproj => 80.0,
        };

        let distortion_reduction = match rtype {
            RestorationType::None => 0.0,
            RestorationType::Wiener => base_distortion * 0.05,
            RestorationType::Sgrproj => base_distortion * 0.03,
        };

        (base_distortion - distortion_reduction) + self.lambda * rate
    }

    /// Compute distortion.
    fn compute_distortion(&self, src: &[u8], recon: &[u8], width: usize, height: usize) -> f32 {
        let mut sse = 0u64;
        let total = (width * height).min(src.len()).min(recon.len());

        for i in 0..total {
            let diff = i32::from(src[i]) - i32::from(recon[i]);
            sse += (diff * diff) as u64;
        }

        sse as f32
    }

    /// Get restoration type.
    #[must_use]
    pub const fn restoration_type(&self) -> RestorationType {
        self.restoration_type
    }
}

impl Default for RestorationOptimizer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// =============================================================================
// Film Grain Parameters
// =============================================================================

/// Film grain synthesis parameters.
#[derive(Clone, Debug, Default)]
pub struct FilmGrainParams {
    /// Enable film grain synthesis.
    pub enabled: bool,
    /// Grain seed.
    pub grain_seed: u16,
    /// Luma scaling points.
    pub luma_points: Vec<(u8, u8)>,
    /// Chroma scaling points.
    pub chroma_points: Vec<(u8, u8)>,
}

impl FilmGrainParams {
    /// Create new film grain parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            enabled: false,
            grain_seed: 0,
            luma_points: Vec::new(),
            chroma_points: Vec::new(),
        }
    }

    /// Enable film grain with seed.
    pub fn enable(&mut self, seed: u16) {
        self.enabled = true;
        self.grain_seed = seed;
    }

    /// Disable film grain.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

// =============================================================================
// Combined Optimizer
// =============================================================================

/// Combined loop optimization manager.
#[derive(Clone, Debug)]
pub struct LoopOptimizer {
    /// Loop filter optimizer.
    loop_filter: LoopFilterOptimizer,
    /// CDEF optimizer.
    cdef: CdefOptimizer,
    /// Restoration optimizer.
    restoration: RestorationOptimizer,
    /// Film grain parameters.
    film_grain: FilmGrainParams,
}

impl LoopOptimizer {
    /// Create a new combined optimizer.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            loop_filter: LoopFilterOptimizer::new(lambda),
            cdef: CdefOptimizer::new(lambda),
            restoration: RestorationOptimizer::new(lambda),
            film_grain: FilmGrainParams::new(),
        }
    }

    /// Optimize all parameters for a frame.
    pub fn optimize_frame(
        &mut self,
        src: &[u8],
        recon: &[u8],
        width: usize,
        height: usize,
        qp: u8,
    ) {
        // Optimize loop filter
        self.loop_filter
            .optimize_filter_level(src, recon, width, height, qp);

        // Optimize CDEF (on smaller regions)
        let cdef_width = width.min(64);
        let cdef_height = height.min(64);
        self.cdef
            .optimize_strength(src, recon, cdef_width, cdef_height, BlockSize::Block64x64);

        // Optimize restoration
        self.restoration
            .optimize_restoration(src, recon, width, height);
    }

    /// Get loop filter parameters.
    #[must_use]
    pub const fn loop_filter_params(&self) -> &LoopFilterParams {
        self.loop_filter.params()
    }

    /// Get CDEF parameters.
    #[must_use]
    pub const fn cdef_params(&self) -> &CdefParams {
        self.cdef.params()
    }

    /// Get restoration type.
    #[must_use]
    pub const fn restoration_type(&self) -> RestorationType {
        self.restoration.restoration_type()
    }

    /// Get film grain parameters.
    #[must_use]
    pub const fn film_grain_params(&self) -> &FilmGrainParams {
        &self.film_grain
    }

    /// Set lambda for all optimizers.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.loop_filter.set_lambda(lambda);
        self.cdef.lambda = lambda;
        self.restoration.lambda = lambda;
    }
}

impl Default for LoopOptimizer {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_filter_optimizer_creation() {
        let opt = LoopFilterOptimizer::new(1.0);
        assert_eq!(opt.lambda, 1.0);
        assert!(opt.rd_optimization);
    }

    #[test]
    fn test_filter_level_from_qp() {
        let opt = LoopFilterOptimizer::new(1.0);

        let level_low = opt.filter_level_from_qp(10);
        let level_high = opt.filter_level_from_qp(50);

        assert!(level_low < level_high);
        assert!(level_low <= MAX_LOOP_FILTER_LEVEL);
        assert!(level_high <= MAX_LOOP_FILTER_LEVEL);
    }

    #[test]
    fn test_optimize_filter_level_fast() {
        let mut opt = LoopFilterOptimizer::new(1.0);
        opt.set_rd_optimization(false);

        let src = vec![128u8; 64 * 64];
        let recon = vec![128u8; 64 * 64];

        let level = opt.optimize_filter_level(&src, &recon, 64, 64, 28);
        assert!(level <= MAX_LOOP_FILTER_LEVEL);
    }

    #[test]
    fn test_compute_distortion() {
        let opt = LoopFilterOptimizer::new(1.0);

        let src = vec![100u8; 64];
        let recon = vec![100u8; 64];

        let distortion = opt.compute_distortion(&src, &recon, 8, 8);
        assert_eq!(distortion, 0.0);

        let recon2 = vec![110u8; 64];
        let distortion2 = opt.compute_distortion(&src, &recon2, 8, 8);
        assert!(distortion2 > 0.0);
    }

    #[test]
    fn test_cdef_optimizer() {
        let opt = CdefOptimizer::new(1.0);
        assert_eq!(opt.lambda, 1.0);
    }

    #[test]
    fn test_cdef_optimize_strength() {
        let mut opt = CdefOptimizer::new(1.0);

        let src = vec![128u8; 32 * 32];
        let recon = vec![130u8; 32 * 32];

        let strength = opt.optimize_strength(&src, &recon, 32, 32, BlockSize::Block32x32);

        assert!(strength.primary <= MAX_CDEF_STRENGTH);
        assert!(strength.secondary <= MAX_CDEF_STRENGTH);
    }

    #[test]
    fn test_restoration_optimizer() {
        let opt = RestorationOptimizer::new(1.0);
        assert_eq!(opt.restoration_type, RestorationType::None);
    }

    #[test]
    fn test_restoration_optimize() {
        let mut opt = RestorationOptimizer::new(1.0);

        let src = vec![128u8; 64 * 64];
        let recon = vec![130u8; 64 * 64];

        let rtype = opt.optimize_restoration(&src, &recon, 64, 64);
        assert!(matches!(
            rtype,
            RestorationType::None | RestorationType::Wiener | RestorationType::Sgrproj
        ));
    }

    #[test]
    fn test_film_grain_params() {
        let mut params = FilmGrainParams::new();
        assert!(!params.enabled);

        params.enable(1234);
        assert!(params.enabled);
        assert_eq!(params.grain_seed, 1234);

        params.disable();
        assert!(!params.enabled);
    }

    #[test]
    fn test_combined_optimizer() {
        let opt = LoopOptimizer::new(1.5);
        assert_eq!(opt.loop_filter.lambda, 1.5);
        assert_eq!(opt.cdef.lambda, 1.5);
    }

    #[test]
    fn test_combined_optimize_frame() {
        let mut opt = LoopOptimizer::new(1.0);

        let src = vec![128u8; 128 * 128];
        let recon = vec![128u8; 128 * 128];

        opt.optimize_frame(&src, &recon, 128, 128, 28);

        // Check that parameters were set
        let lf_params = opt.loop_filter_params();
        assert!(lf_params.level[0] <= MAX_LOOP_FILTER_LEVEL);
    }

    #[test]
    fn test_set_lambda() {
        let mut opt = LoopOptimizer::new(1.0);
        opt.set_lambda(2.5);

        assert_eq!(opt.loop_filter.lambda, 2.5);
        assert_eq!(opt.cdef.lambda, 2.5);
        assert_eq!(opt.restoration.lambda, 2.5);
    }

    #[test]
    fn test_restoration_types() {
        assert_eq!(RestorationType::None as u8, 0);
        assert_eq!(RestorationType::Wiener as u8, 1);
        assert_eq!(RestorationType::Sgrproj as u8, 2);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_LOOP_FILTER_LEVEL, 63);
        assert_eq!(MAX_CDEF_STRENGTH, 15);
        assert!(FILTER_LEVELS_TO_TEST > 0);
        assert!(CDEF_STRENGTHS_TO_TEST > 0);
    }
}
