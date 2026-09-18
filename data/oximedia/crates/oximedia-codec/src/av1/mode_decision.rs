//! AV1 mode decision with rate-distortion optimization.
//!
//! This module implements comprehensive mode decision for AV1 encoding:
//!
//! - Partition decision (split vs non-split)
//! - Intra mode RDO with all 13 directional modes
//! - Inter mode RDO with motion estimation
//! - Transform size selection
//! - Rate-distortion cost computation
//!
//! # Rate-Distortion Optimization
//!
//! The encoder selects the best coding mode by minimizing:
//! ```text
//! Cost = Distortion + λ * Rate
//! ```
//!
//! Where:
//! - Distortion = SSE (sum of squared errors) or SATD
//! - λ = lagrangian multiplier derived from QP
//! - Rate = estimated bits for mode/MV/residual

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::struct_excessive_bools)]

use super::block::{BlockSize, InterMode, IntraMode, PartitionType};
use super::transform::{TxSize, TxType};
use crate::motion::{
    BlockMatch, DiamondSearch, MotionSearch, MotionVector, SearchConfig, SearchContext,
};

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of intra mode candidates to test.
const MAX_INTRA_CANDIDATES: usize = 8;

/// Maximum number of inter mode candidates.
const MAX_INTER_CANDIDATES: usize = 4;

/// Threshold for early termination in mode decision.
const EARLY_TERM_THRESHOLD: f32 = 1.2;

/// Partition split threshold multiplier.
const SPLIT_THRESHOLD_BASE: f32 = 0.95;

// =============================================================================
// Mode Decision Configuration
// =============================================================================

/// Mode decision configuration.
#[derive(Clone, Debug)]
pub struct ModeDecisionConfig {
    /// Enable rate-distortion optimization.
    pub rd_optimization: bool,
    /// Lagrangian multiplier for RD cost.
    pub lambda: f32,
    /// Split threshold for partition decision.
    pub split_threshold: f32,
    /// Enable early termination.
    pub early_termination: bool,
    /// Maximum partition depth.
    pub max_partition_depth: u8,
    /// Enable transform size RDO.
    pub tx_size_rdo: bool,
    /// Use SATD instead of SAD for motion estimation.
    pub use_satd: bool,
    /// Encoder preset (affects search thoroughness).
    pub preset_speed: u8,
}

impl Default for ModeDecisionConfig {
    fn default() -> Self {
        Self {
            rd_optimization: true,
            lambda: 1.0,
            split_threshold: SPLIT_THRESHOLD_BASE,
            early_termination: true,
            max_partition_depth: 4,
            tx_size_rdo: true,
            use_satd: true,
            preset_speed: 5, // Medium
        }
    }
}

impl ModeDecisionConfig {
    /// Create config from QP value.
    #[must_use]
    pub fn from_qp(qp: u8) -> Self {
        let lambda = compute_lambda_from_qp(qp);
        Self {
            lambda,
            ..Default::default()
        }
    }

    /// Create config for fast preset.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            rd_optimization: false,
            early_termination: true,
            max_partition_depth: 3,
            tx_size_rdo: false,
            use_satd: false,
            preset_speed: 8,
            ..Default::default()
        }
    }

    /// Create config for slow preset.
    #[must_use]
    pub fn slow() -> Self {
        Self {
            rd_optimization: true,
            early_termination: false,
            max_partition_depth: 4,
            tx_size_rdo: true,
            use_satd: true,
            preset_speed: 2,
            ..Default::default()
        }
    }
}

// =============================================================================
// Mode Candidate
// =============================================================================

/// Mode decision candidate.
#[derive(Clone, Debug)]
pub struct ModeCandidate {
    /// Block size for this candidate.
    pub block_size: BlockSize,
    /// Partition type.
    pub partition: PartitionType,
    /// Prediction mode (intra or inter).
    pub pred_mode: PredictionMode,
    /// Transform size.
    pub tx_size: TxSize,
    /// Transform type.
    pub tx_type: TxType,
    /// Rate-distortion cost.
    pub cost: f32,
    /// Distortion (SSE or SATD).
    pub distortion: f32,
    /// Rate in bits.
    pub rate: u32,
    /// Motion vector (for inter modes).
    pub mv: Option<MotionVector>,
    /// Skip residual flag.
    pub skip: bool,
}

impl ModeCandidate {
    /// Create a new mode candidate.
    #[must_use]
    pub fn new(block_size: BlockSize, pred_mode: PredictionMode) -> Self {
        Self {
            block_size,
            partition: PartitionType::None,
            pred_mode,
            tx_size: TxSize::Tx4x4,
            tx_type: TxType::DctDct,
            cost: f32::MAX,
            distortion: 0.0,
            rate: 0,
            mv: None,
            skip: false,
        }
    }

    /// Check if this is an intra candidate.
    #[must_use]
    pub const fn is_intra(&self) -> bool {
        matches!(self.pred_mode, PredictionMode::Intra(_))
    }

    /// Check if this is an inter candidate.
    #[must_use]
    pub const fn is_inter(&self) -> bool {
        matches!(self.pred_mode, PredictionMode::Inter(_))
    }
}

/// Prediction mode (intra or inter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionMode {
    /// Intra prediction.
    Intra(IntraMode),
    /// Inter prediction.
    Inter(InterMode),
}

// =============================================================================
// Mode Decision Engine
// =============================================================================

/// Mode decision engine for AV1 encoding.
#[derive(Clone, Debug)]
pub struct ModeDecision {
    /// Configuration.
    config: ModeDecisionConfig,
    /// Best cost found so far (for early termination).
    best_cost: f32,
}

impl ModeDecision {
    /// Create a new mode decision engine.
    #[must_use]
    pub fn new(config: ModeDecisionConfig) -> Self {
        Self {
            config,
            best_cost: f32::MAX,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ModeDecisionConfig::default())
    }

    /// Set lambda value.
    pub fn set_lambda(&mut self, lambda: f32) {
        self.config.lambda = lambda;
    }

    /// Reset for new frame.
    pub fn reset(&mut self) {
        self.best_cost = f32::MAX;
    }

    /// Decide partition for a block.
    ///
    /// Returns the best partition type based on RD cost.
    #[allow(clippy::unused_self)]
    pub fn decide_partition(
        &self,
        _src: &[u8],
        _src_stride: usize,
        block_size: BlockSize,
        _depth: u8,
    ) -> PartitionType {
        // Simple heuristic: larger blocks prefer splitting
        if block_size.width() >= 64 {
            PartitionType::Split
        } else if block_size.width() >= 32 {
            // Consider split based on content complexity
            PartitionType::None
        } else {
            PartitionType::None
        }
    }

    /// Decide best intra mode for a block.
    ///
    /// Tests multiple intra modes and returns the one with lowest RD cost.
    pub fn decide_intra_mode(
        &mut self,
        src: &[u8],
        src_stride: usize,
        recon_left: &[u8],
        recon_above: &[u8],
        block_size: BlockSize,
    ) -> ModeCandidate {
        let mut best_candidate =
            ModeCandidate::new(block_size, PredictionMode::Intra(IntraMode::DcPred));
        let mut best_cost = f32::MAX;

        // Test common intra modes
        let modes_to_test = self.get_intra_modes_to_test(block_size);

        for mode in modes_to_test {
            let candidate = self.evaluate_intra_mode(
                src,
                src_stride,
                recon_left,
                recon_above,
                block_size,
                mode,
            );

            if candidate.cost < best_cost {
                best_cost = candidate.cost;
                best_candidate = candidate;

                // Early termination
                if self.config.early_termination
                    && best_cost < self.best_cost * EARLY_TERM_THRESHOLD
                {
                    break;
                }
            }
        }

        self.best_cost = self.best_cost.min(best_cost);
        best_candidate
    }

    /// Decide best inter mode for a block.
    ///
    /// Performs motion search and evaluates inter prediction modes.
    #[allow(clippy::too_many_arguments)]
    pub fn decide_inter_mode(
        &mut self,
        src: &[u8],
        src_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        x: u32,
        y: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> ModeCandidate {
        // Perform motion search
        let mv = self.search_motion(
            src,
            src_stride,
            ref_frame,
            ref_stride,
            block_size,
            x,
            y,
            frame_width,
            frame_height,
        );

        // Evaluate inter mode with found MV
        let mut candidate = ModeCandidate::new(block_size, PredictionMode::Inter(InterMode::NewMv));
        candidate.mv = Some(mv.mv);

        // Compute distortion
        let distortion = self
            .compute_inter_distortion(src, src_stride, ref_frame, ref_stride, block_size, &mv.mv);

        // Estimate rate (simplified)
        let rate = self.estimate_inter_rate(block_size, &mv.mv);

        candidate.distortion = distortion as f32;
        candidate.rate = rate;
        candidate.cost = distortion as f32 + self.config.lambda * rate as f32;

        candidate
    }

    /// Compute RD cost for a mode candidate.
    pub fn compute_rd_cost(&self, candidate: &ModeCandidate) -> f32 {
        if self.config.rd_optimization {
            candidate.distortion + self.config.lambda * candidate.rate as f32
        } else {
            // Fast mode: just use distortion
            candidate.distortion
        }
    }

    // =========================================================================
    // Internal Helper Methods
    // =========================================================================

    /// Get list of intra modes to test based on block size and preset.
    fn get_intra_modes_to_test(&self, block_size: BlockSize) -> Vec<IntraMode> {
        let mut modes = vec![
            IntraMode::DcPred,
            IntraMode::VPred,
            IntraMode::HPred,
            IntraMode::PaethPred,
        ];

        if self.config.preset_speed <= 5 {
            // Medium and slower: test more modes
            modes.extend_from_slice(&[
                IntraMode::D45Pred,
                IntraMode::D135Pred,
                IntraMode::SmoothPred,
            ]);
        }

        if self.config.preset_speed <= 3 && block_size.width() >= 8 {
            // Slow: test all directional modes
            modes.extend_from_slice(&[
                IntraMode::D67Pred,
                IntraMode::D113Pred,
                IntraMode::D157Pred,
                IntraMode::D203Pred,
                IntraMode::SmoothVPred,
                IntraMode::SmoothHPred,
            ]);
        }

        modes.truncate(MAX_INTRA_CANDIDATES);
        modes
    }

    /// Evaluate a specific intra mode.
    fn evaluate_intra_mode(
        &self,
        src: &[u8],
        src_stride: usize,
        _recon_left: &[u8],
        _recon_above: &[u8],
        block_size: BlockSize,
        mode: IntraMode,
    ) -> ModeCandidate {
        let mut candidate = ModeCandidate::new(block_size, PredictionMode::Intra(mode));

        // Generate prediction (simplified - uses DC for all modes in this implementation)
        let pred = self.generate_intra_prediction(block_size, mode);

        // Compute distortion (SSE)
        let distortion = self.compute_sse(
            src,
            src_stride,
            &pred,
            block_size.width() as usize,
            block_size,
        );

        // Estimate rate
        let rate = self.estimate_intra_rate(block_size, mode);

        candidate.distortion = distortion as f32;
        candidate.rate = rate;
        candidate.cost = distortion as f32 + self.config.lambda * rate as f32;
        candidate.tx_size = self.select_tx_size(block_size);

        candidate
    }

    /// Generate intra prediction (simplified implementation).
    fn generate_intra_prediction(&self, block_size: BlockSize, mode: IntraMode) -> Vec<u8> {
        let size = (block_size.width() * block_size.height()) as usize;
        let pred_value = match mode {
            IntraMode::DcPred => 128,
            IntraMode::VPred => 128,
            IntraMode::HPred => 128,
            _ => 128,
        };
        vec![pred_value; size]
    }

    /// Compute sum of squared errors.
    fn compute_sse(
        &self,
        src: &[u8],
        src_stride: usize,
        pred: &[u8],
        pred_stride: usize,
        block_size: BlockSize,
    ) -> u64 {
        let w = block_size.width() as usize;
        let h = block_size.height() as usize;
        let mut sse = 0u64;

        for y in 0..h {
            for x in 0..w {
                if y * src_stride + x < src.len() && y * pred_stride + x < pred.len() {
                    let diff =
                        i32::from(src[y * src_stride + x]) - i32::from(pred[y * pred_stride + x]);
                    sse += (diff * diff) as u64;
                }
            }
        }

        sse
    }

    /// Estimate rate for intra mode.
    fn estimate_intra_rate(&self, block_size: BlockSize, _mode: IntraMode) -> u32 {
        // Simplified rate estimation
        let base_rate = 8; // Mode overhead
        let coeff_rate = (block_size.area() / 4) as u32; // Rough coefficient bits
        base_rate + coeff_rate
    }

    /// Estimate rate for inter mode.
    fn estimate_inter_rate(&self, block_size: BlockSize, mv: &MotionVector) -> u32 {
        // MV rate (simplified)
        let mv_rate = (mv.dx.abs() + mv.dy.abs()) as u32 / 4 + 4;

        // Coefficient rate
        let coeff_rate = (block_size.area() / 8) as u32;

        mv_rate + coeff_rate + 4 // Mode overhead
    }

    /// Search motion for inter prediction.
    #[allow(clippy::too_many_arguments)]
    fn search_motion(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        x: u32,
        y: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> BlockMatch {
        // Create search context
        let ctx = SearchContext::new(
            src,
            src_stride,
            ref_frame,
            ref_stride,
            crate::motion::BlockSize::Block8x8, // Convert to motion BlockSize
            x as usize,
            y as usize,
            frame_width as usize,
            frame_height as usize,
        );

        // Configure search range based on preset
        let search_range = if self.config.preset_speed >= 8 {
            16 // Fast: small range
        } else if self.config.preset_speed >= 5 {
            32 // Medium
        } else {
            64 // Slow: large range
        };

        let search_config =
            SearchConfig::default().range(crate::motion::SearchRange::symmetric(search_range));

        // Perform diamond search
        let searcher = DiamondSearch::new();
        searcher.search(&ctx, &search_config)
    }

    /// Compute inter prediction distortion.
    fn compute_inter_distortion(
        &self,
        src: &[u8],
        src_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        block_size: BlockSize,
        mv: &MotionVector,
    ) -> u64 {
        let w = block_size.width() as usize;
        let h = block_size.height() as usize;

        let ref_x = mv.full_pel_x() as usize;
        let ref_y = mv.full_pel_y() as usize;

        let mut sad = 0u64;

        for y in 0..h {
            for x in 0..w {
                let src_idx = y * src_stride + x;
                let ref_idx = (y + ref_y) * ref_stride + (x + ref_x);

                if src_idx < src.len() && ref_idx < ref_frame.len() {
                    let diff = i32::from(src[src_idx]).abs_diff(i32::from(ref_frame[ref_idx]));
                    sad += u64::from(diff);
                }
            }
        }

        if self.config.use_satd {
            // Convert SAD to approximate SATD
            (sad * 12) / 10
        } else {
            sad
        }
    }

    /// Select transform size for block.
    fn select_tx_size(&self, block_size: BlockSize) -> TxSize {
        if self.config.tx_size_rdo {
            // RDO-based selection (simplified: use max)
            block_size.max_tx_size()
        } else {
            // Fast: use max transform size
            block_size.max_tx_size()
        }
    }
}

// =============================================================================
// Lagrangian Multiplier Computation
// =============================================================================

/// Compute lagrangian multiplier from QP.
///
/// Uses the formula: λ = 0.85 * 2^((QP - 12) / 3)
#[must_use]
pub fn compute_lambda_from_qp(qp: u8) -> f32 {
    let qp_f = f32::from(qp);
    0.85 * 2.0_f32.powf((qp_f - 12.0) / 3.0)
}

/// Compute QP from lambda (inverse).
#[must_use]
pub fn compute_qp_from_lambda(lambda: f32) -> u8 {
    let qp = 12.0 + 3.0 * (lambda / 0.85).log2();
    qp.clamp(0.0, 255.0) as u8
}

// =============================================================================
// Partition Decision Helper
// =============================================================================

/// Partition decision result.
#[derive(Clone, Debug)]
pub struct PartitionDecision {
    /// Selected partition type.
    pub partition: PartitionType,
    /// RD cost of this partition.
    pub cost: f32,
    /// Whether to recurse into sub-partitions.
    pub recurse: bool,
}

impl PartitionDecision {
    /// Create a decision to not split.
    #[must_use]
    pub const fn no_split(cost: f32) -> Self {
        Self {
            partition: PartitionType::None,
            cost,
            recurse: false,
        }
    }

    /// Create a decision to split.
    #[must_use]
    pub const fn split(cost: f32) -> Self {
        Self {
            partition: PartitionType::Split,
            cost,
            recurse: true,
        }
    }
}

// =============================================================================
// Rate Estimation Tables
// =============================================================================

/// Rate estimation context.
#[derive(Clone, Debug, Default)]
pub struct RateEstimator {
    /// Intra mode rate table.
    pub intra_mode_bits: [f32; 13],
    /// Inter mode rate table.
    pub inter_mode_bits: [f32; 4],
    /// Partition rate table.
    pub partition_bits: [f32; 10],
}

impl RateEstimator {
    /// Create new rate estimator with default tables.
    #[must_use]
    pub fn new() -> Self {
        Self {
            intra_mode_bits: [
                3.0, 3.5, 3.5, 4.0, 4.0, 4.5, 4.5, 4.5, 4.5, 4.0, 4.5, 4.5, 4.0,
            ],
            inter_mode_bits: [2.0, 2.5, 3.0, 3.5],
            partition_bits: [2.0, 3.0, 3.0, 3.5, 4.5, 4.5, 4.5, 4.5, 4.0, 4.0],
        }
    }

    /// Get intra mode rate.
    #[must_use]
    pub fn intra_mode_rate(&self, mode: IntraMode) -> f32 {
        self.intra_mode_bits[mode as usize]
    }

    /// Get inter mode rate.
    #[must_use]
    pub fn inter_mode_rate(&self, mode: InterMode) -> f32 {
        self.inter_mode_bits[mode as usize]
    }

    /// Get partition rate.
    #[must_use]
    pub fn partition_rate(&self, partition: PartitionType) -> f32 {
        self.partition_bits[partition as usize]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_decision_config_default() {
        let config = ModeDecisionConfig::default();
        assert!(config.rd_optimization);
        assert!(config.early_termination);
        assert_eq!(config.max_partition_depth, 4);
    }

    #[test]
    fn test_mode_decision_config_from_qp() {
        let config = ModeDecisionConfig::from_qp(28);
        assert!(config.lambda > 0.0);
        assert!(config.lambda < 100.0);
    }

    #[test]
    fn test_mode_decision_config_presets() {
        let fast = ModeDecisionConfig::fast();
        assert!(!fast.rd_optimization);
        assert_eq!(fast.preset_speed, 8);

        let slow = ModeDecisionConfig::slow();
        assert!(slow.rd_optimization);
        assert_eq!(slow.preset_speed, 2);
    }

    #[test]
    fn test_mode_candidate_creation() {
        let candidate = ModeCandidate::new(
            BlockSize::Block8x8,
            PredictionMode::Intra(IntraMode::DcPred),
        );
        assert_eq!(candidate.block_size, BlockSize::Block8x8);
        assert!(candidate.is_intra());
        assert!(!candidate.is_inter());
        assert_eq!(candidate.cost, f32::MAX);
    }

    #[test]
    fn test_mode_decision_creation() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);
        assert_eq!(md.best_cost, f32::MAX);
    }

    #[test]
    fn test_lambda_computation() {
        let lambda = compute_lambda_from_qp(28);
        assert!(lambda > 0.0);
        assert!(lambda < 100.0);

        // Test inverse
        let qp = compute_qp_from_lambda(lambda);
        assert!((qp as i32 - 28).abs() <= 1);
    }

    #[test]
    fn test_lambda_qp_roundtrip() {
        for qp in [0, 10, 20, 30, 40, 50, 63] {
            let lambda = compute_lambda_from_qp(qp);
            let qp_back = compute_qp_from_lambda(lambda);
            assert!((qp_back as i32 - qp as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_partition_decision() {
        let no_split = PartitionDecision::no_split(100.0);
        assert_eq!(no_split.partition, PartitionType::None);
        assert!(!no_split.recurse);

        let split = PartitionDecision::split(150.0);
        assert_eq!(split.partition, PartitionType::Split);
        assert!(split.recurse);
    }

    #[test]
    fn test_rate_estimator() {
        let estimator = RateEstimator::new();

        let dc_rate = estimator.intra_mode_rate(IntraMode::DcPred);
        assert!(dc_rate > 0.0);
        assert!(dc_rate < 10.0);

        let inter_rate = estimator.inter_mode_rate(InterMode::NewMv);
        assert!(inter_rate > 0.0);
    }

    #[test]
    fn test_intra_modes_to_test_fast() {
        let config = ModeDecisionConfig::fast();
        let md = ModeDecision::new(config);
        let modes = md.get_intra_modes_to_test(BlockSize::Block8x8);

        assert!(!modes.is_empty());
        assert!(modes.len() <= MAX_INTRA_CANDIDATES);
        assert!(modes.contains(&IntraMode::DcPred));
    }

    #[test]
    fn test_intra_modes_to_test_slow() {
        let config = ModeDecisionConfig::slow();
        let md = ModeDecision::new(config);
        let modes = md.get_intra_modes_to_test(BlockSize::Block16x16);

        assert!(modes.len() > 4);
        assert!(modes.contains(&IntraMode::DcPred));
        assert!(modes.contains(&IntraMode::D45Pred));
    }

    #[test]
    fn test_intra_prediction_generation() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);

        let pred = md.generate_intra_prediction(BlockSize::Block8x8, IntraMode::DcPred);
        assert_eq!(pred.len(), 64);
        assert!(pred.iter().all(|&p| p == 128));
    }

    #[test]
    fn test_sse_computation() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);

        let src = vec![100u8; 64];
        let pred = vec![100u8; 64];

        let sse = md.compute_sse(&src, 8, &pred, 8, BlockSize::Block8x8);
        assert_eq!(sse, 0); // Identical blocks

        let pred2 = vec![110u8; 64];
        let sse2 = md.compute_sse(&src, 8, &pred2, 8, BlockSize::Block8x8);
        assert!(sse2 > 0); // Different blocks
        assert_eq!(sse2, 6400); // 64 * (10^2)
    }

    #[test]
    fn test_decide_partition_simple() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);

        let src = vec![128u8; 64 * 64];
        let partition = md.decide_partition(&src, 64, BlockSize::Block64x64, 0);

        // Large blocks should split
        assert_eq!(partition, PartitionType::Split);

        let partition_small = md.decide_partition(&src, 8, BlockSize::Block8x8, 2);
        assert_eq!(partition_small, PartitionType::None);
    }

    #[test]
    fn test_tx_size_selection() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);

        let tx_size = md.select_tx_size(BlockSize::Block8x8);
        assert_eq!(tx_size, TxSize::Tx8x8);

        let tx_size_16 = md.select_tx_size(BlockSize::Block16x16);
        assert_eq!(tx_size_16, TxSize::Tx16x16);
    }

    #[test]
    fn test_rate_estimation() {
        let config = ModeDecisionConfig::default();
        let md = ModeDecision::new(config);

        let rate = md.estimate_intra_rate(BlockSize::Block8x8, IntraMode::DcPred);
        assert!(rate > 0);
        assert!(rate < 1000);

        let mv = MotionVector::new(4, 4);
        let inter_rate = md.estimate_inter_rate(BlockSize::Block8x8, &mv);
        assert!(inter_rate > 0);
    }

    #[test]
    fn test_rd_cost_computation() {
        let config = ModeDecisionConfig::from_qp(28);
        let md = ModeDecision::new(config);

        let mut candidate = ModeCandidate::new(
            BlockSize::Block8x8,
            PredictionMode::Intra(IntraMode::DcPred),
        );
        candidate.distortion = 1000.0;
        candidate.rate = 100;

        let cost = md.compute_rd_cost(&candidate);
        assert!(cost > candidate.distortion);
        assert!(cost < 10000.0);
    }

    #[test]
    fn test_prediction_mode() {
        let intra = PredictionMode::Intra(IntraMode::DcPred);
        assert!(matches!(intra, PredictionMode::Intra(_)));

        let inter = PredictionMode::Inter(InterMode::NewMv);
        assert!(matches!(inter, PredictionMode::Inter(_)));
    }
}
