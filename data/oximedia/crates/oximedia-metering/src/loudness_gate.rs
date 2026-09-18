//! Loudness gating for integrated loudness measurement (ITU-R BS.1770-4).
//!
//! The standard defines two gating stages applied to 400 ms overlapping blocks:
//!
//! 1. **Absolute gate**: Blocks below −70 LKFS are excluded.
//! 2. **Relative gate**: After computing the ungated average, blocks below
//!    (ungated − 10 LU) are excluded.
//!
//! The integrated loudness is then the mean-square energy of the surviving blocks,
//! converted to LKFS.

#![allow(dead_code)]

/// State of the loudness gate for a single measurement block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// Block passes the gate (included in integrated loudness).
    Active,
    /// Block is below the absolute gate threshold (excluded).
    BelowAbsolute,
    /// Block is below the relative gate threshold (excluded).
    BelowRelative,
}

impl GateState {
    /// Returns `true` when the gate is passing (block is included).
    pub fn is_active(&self) -> bool {
        *self == Self::Active
    }
}

/// Absolute and relative gate thresholds.
#[derive(Clone, Copy, Debug)]
pub struct GatingThreshold {
    /// Absolute gate threshold in LKFS (default −70 LKFS).
    pub absolute_lkfs: f64,
    /// Relative gate offset in LU below the ungated loudness (default −10 LU).
    pub relative_offset_lu: f64,
}

impl Default for GatingThreshold {
    fn default() -> Self {
        Self {
            absolute_lkfs: -70.0,
            relative_offset_lu: -10.0,
        }
    }
}

impl GatingThreshold {
    /// Create a new gating threshold with the given values.
    pub fn new(absolute_lkfs: f64, relative_offset_lu: f64) -> Self {
        Self {
            absolute_lkfs,
            relative_offset_lu,
        }
    }

    /// Returns `true` when `block_lkfs` passes the absolute gate.
    pub fn is_above_gate(&self, block_lkfs: f64) -> bool {
        block_lkfs > self.absolute_lkfs
    }

    /// Returns `true` when `block_lkfs` passes the relative gate given `ungated_lkfs`.
    pub fn is_above_relative(&self, block_lkfs: f64, ungated_lkfs: f64) -> bool {
        block_lkfs > ungated_lkfs + self.relative_offset_lu
    }
}

/// A single processed loudness block.
#[derive(Clone, Copy, Debug)]
pub struct LoudnessBlock {
    /// Mean-square signal energy of this block (linear, K-weighted).
    pub mean_square: f64,
    /// Loudness of this block in LKFS.
    pub lkfs: f64,
    /// Gate state determined after gating.
    pub gate_state: GateState,
}

impl LoudnessBlock {
    /// Create a block from mean-square energy and threshold information.
    pub fn new(mean_square: f64) -> Self {
        let lkfs = if mean_square > 1e-10 {
            -0.691 + 10.0 * mean_square.log10()
        } else {
            f64::NEG_INFINITY
        };
        Self {
            mean_square,
            lkfs,
            gate_state: GateState::Active,
        }
    }
}

/// Processes audio blocks and applies the two-stage loudness gate.
#[derive(Clone, Debug)]
pub struct LoudnessGate {
    threshold: GatingThreshold,
    blocks: Vec<LoudnessBlock>,
}

impl LoudnessGate {
    /// Create a new [`LoudnessGate`] with the provided threshold.
    pub fn new(threshold: GatingThreshold) -> Self {
        Self {
            threshold,
            blocks: Vec::new(),
        }
    }

    /// Create a gate using ITU-R BS.1770-4 default thresholds.
    pub fn default_itu() -> Self {
        Self::new(GatingThreshold::default())
    }

    /// Submit a block's mean-square energy.
    ///
    /// The gate state is initially set to [`GateState::BelowAbsolute`] or
    /// [`GateState::Active`] based on the absolute threshold only;
    /// call [`apply_relative_gate`](Self::apply_relative_gate) afterwards to
    /// finalize the integrated measurement.
    pub fn process_block(&mut self, mean_square: f64) {
        let mut block = LoudnessBlock::new(mean_square);
        if !self.threshold.is_above_gate(block.lkfs) {
            block.gate_state = GateState::BelowAbsolute;
        }
        self.blocks.push(block);
    }

    /// Apply the relative gate using the current ungated mean.
    ///
    /// Must be called after all blocks have been submitted via [`process_block`](Self::process_block).
    pub fn apply_relative_gate(&mut self) {
        let ungated = self.ungated_loudness();
        if !ungated.is_finite() {
            return;
        }
        for block in &mut self.blocks {
            if block.gate_state == GateState::Active
                && !self.threshold.is_above_relative(block.lkfs, ungated)
            {
                block.gate_state = GateState::BelowRelative;
            }
        }
    }

    /// Returns `true` when the most recently submitted block would be excluded by the absolute gate.
    pub fn is_gated(&self) -> bool {
        self.blocks
            .last()
            .map_or(true, |b| b.gate_state != GateState::Active)
    }

    /// Ungated loudness (absolute gate only) in LKFS.
    fn ungated_loudness(&self) -> f64 {
        let sum: f64 = self
            .blocks
            .iter()
            .filter(|b| b.gate_state == GateState::Active)
            .map(|b| b.mean_square)
            .sum();
        let count = self
            .blocks
            .iter()
            .filter(|b| b.gate_state == GateState::Active)
            .count();
        if count == 0 || sum <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -0.691 + 10.0 * (sum / count as f64).log10()
    }

    /// Number of blocks submitted.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Number of blocks that passed both gates.
    pub fn active_block_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.gate_state == GateState::Active)
            .count()
    }

    /// Access all blocks.
    pub fn blocks(&self) -> &[LoudnessBlock] {
        &self.blocks
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.blocks.clear();
    }
}

/// Summary result of a complete gated loudness measurement.
#[derive(Clone, Debug)]
pub struct GatedMeasurement {
    /// Integrated loudness in LKFS after two-stage gating.
    pub integrated_lkfs: f64,
    /// Total blocks submitted.
    pub total_blocks: usize,
    /// Blocks excluded by the absolute gate.
    pub absolute_gated_blocks: usize,
    /// Blocks excluded by the relative gate.
    pub relative_gated_blocks: usize,
}

impl GatedMeasurement {
    /// Build a [`GatedMeasurement`] from a finalized [`LoudnessGate`].
    pub fn from_gate(gate: &LoudnessGate) -> Self {
        // Apply relative gate if not yet done (or re-compute).
        let ungated = gate.ungated_loudness();
        let relative_threshold = ungated + gate.threshold.relative_offset_lu;

        let mut sum_active = 0.0_f64;
        let mut count_active = 0_usize;
        let mut abs_gated = 0_usize;
        let mut rel_gated = 0_usize;

        for block in &gate.blocks {
            match block.gate_state {
                GateState::Active => {
                    if block.lkfs > relative_threshold {
                        sum_active += block.mean_square;
                        count_active += 1;
                    } else {
                        rel_gated += 1;
                    }
                }
                GateState::BelowAbsolute => abs_gated += 1,
                GateState::BelowRelative => rel_gated += 1,
            }
        }

        let integrated_lkfs = if count_active > 0 && sum_active > 0.0 {
            -0.691 + 10.0 * (sum_active / count_active as f64).log10()
        } else {
            f64::NEG_INFINITY
        };

        Self {
            integrated_lkfs,
            total_blocks: gate.blocks.len(),
            absolute_gated_blocks: abs_gated,
            relative_gated_blocks: rel_gated,
        }
    }

    /// Returns the integrated loudness in LKFS.
    pub fn integrated_lkfs(&self) -> f64 {
        self.integrated_lkfs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_state_is_active_true() {
        assert!(GateState::Active.is_active());
    }

    #[test]
    fn test_gate_state_is_active_false() {
        assert!(!GateState::BelowAbsolute.is_active());
        assert!(!GateState::BelowRelative.is_active());
    }

    #[test]
    fn test_gating_threshold_default() {
        let t = GatingThreshold::default();
        assert_eq!(t.absolute_lkfs, -70.0);
        assert_eq!(t.relative_offset_lu, -10.0);
    }

    #[test]
    fn test_is_above_gate_passes() {
        let t = GatingThreshold::default();
        assert!(t.is_above_gate(-23.0));
    }

    #[test]
    fn test_is_above_gate_fails() {
        let t = GatingThreshold::default();
        assert!(!t.is_above_gate(-80.0));
    }

    #[test]
    fn test_is_above_relative_passes() {
        let t = GatingThreshold::default();
        // ungated = -23, threshold = -33; block at -25 passes.
        assert!(t.is_above_relative(-25.0, -23.0));
    }

    #[test]
    fn test_is_above_relative_fails() {
        let t = GatingThreshold::default();
        // ungated = -23, threshold = -33; block at -40 fails.
        assert!(!t.is_above_relative(-40.0, -23.0));
    }

    #[test]
    fn test_loudness_block_lkfs_positive() {
        let block = LoudnessBlock::new(0.01);
        assert!(block.lkfs.is_finite());
    }

    #[test]
    fn test_loudness_block_zero_energy() {
        let block = LoudnessBlock::new(0.0);
        assert!(block.lkfs.is_infinite());
    }

    #[test]
    fn test_process_block_below_absolute() {
        let mut gate = LoudnessGate::default_itu();
        // Extremely small energy -> well below -70 LKFS
        gate.process_block(1e-20);
        assert!(gate.is_gated());
    }

    #[test]
    fn test_process_block_above_absolute() {
        let mut gate = LoudnessGate::default_itu();
        // 0.01 mean-square -> ~-21 LKFS, above -70
        gate.process_block(0.01);
        assert!(!gate.is_gated());
    }

    #[test]
    fn test_block_count() {
        let mut gate = LoudnessGate::default_itu();
        gate.process_block(0.01);
        gate.process_block(0.02);
        gate.process_block(1e-20); // gated
        assert_eq!(gate.block_count(), 3);
    }

    #[test]
    fn test_active_block_count() {
        let mut gate = LoudnessGate::default_itu();
        gate.process_block(0.01);
        gate.process_block(0.02);
        gate.process_block(1e-20); // gated
                                   // 2 of 3 should be active before relative gating
        assert_eq!(gate.active_block_count(), 2);
    }

    #[test]
    fn test_reset_clears_blocks() {
        let mut gate = LoudnessGate::default_itu();
        gate.process_block(0.01);
        gate.reset();
        assert_eq!(gate.block_count(), 0);
    }

    #[test]
    fn test_gated_measurement_finite_with_loud_blocks() {
        let mut gate = LoudnessGate::default_itu();
        // Submit 10 blocks with reasonable energy.
        for _ in 0..10 {
            gate.process_block(0.01); // ~-21 LKFS
        }
        gate.apply_relative_gate();
        let m = GatedMeasurement::from_gate(&gate);
        assert!(m.integrated_lkfs().is_finite());
    }

    #[test]
    fn test_gated_measurement_infinite_when_all_gated() {
        let mut gate = LoudnessGate::default_itu();
        gate.process_block(1e-20);
        gate.process_block(1e-20);
        gate.apply_relative_gate();
        let m = GatedMeasurement::from_gate(&gate);
        assert!(m.integrated_lkfs().is_infinite());
    }
}
