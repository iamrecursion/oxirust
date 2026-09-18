//! Gating algorithm for integrated loudness measurement.
//!
//! Implements ITU-R BS.1771 two-stage gating:
//! 1. Absolute gate at -70 LKFS (removes silence)
//! 2. Relative gate at -10 LU below ungated mean (removes quiet passages)

/// Absolute gate threshold in LKFS.
const ABSOLUTE_GATE: f64 = -70.0;

/// Relative gate offset in LU.
const RELATIVE_GATE_OFFSET: f64 = -10.0;

/// Block size for gating (400ms).
const GATING_BLOCK_MS: f64 = 400.0;

/// Block overlap for gating (75%).
const GATING_OVERLAP: f64 = 0.75;

/// Gating measurement block.
#[derive(Clone, Copy, Debug)]
struct GatingBlock {
    /// Block loudness in LKFS.
    loudness: f64,
    /// Block mean-square power.
    power: f64,
    /// Block timestamp in seconds.
    timestamp: f64,
}

impl GatingBlock {
    fn new(loudness: f64, power: f64, timestamp: f64) -> Self {
        Self {
            loudness,
            power,
            timestamp,
        }
    }
}

/// Gating processor for integrated loudness.
///
/// Implements the ITU-R BS.1771 two-stage gating algorithm to calculate
/// program loudness by excluding silent and very quiet passages.
pub struct GatingProcessor {
    sample_rate: f64,
    channels: usize,

    // Block processing
    block_size: usize,
    block_accumulator: Vec<f64>,
    block_sample_count: usize,
    channel_weights: Vec<f64>,

    // Gating blocks
    blocks: Vec<GatingBlock>,

    // Timing
    timestamp: f64,
    total_samples: usize,
}

impl GatingProcessor {
    /// Create a new gating processor.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let block_size = (sample_rate * GATING_BLOCK_MS / 1000.0) as usize;
        let channel_weights = Self::calculate_channel_weights(channels);

        Self {
            sample_rate,
            channels,
            block_size,
            block_accumulator: vec![0.0; channels],
            block_sample_count: 0,
            channel_weights,
            blocks: Vec::new(),
            timestamp: 0.0,
            total_samples: 0,
        }
    }

    /// Process interleaved K-weighted audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved K-weighted samples
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;

        for frame in 0..frames {
            // Accumulate mean-square per channel with weighting
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                if idx < samples.len() {
                    let sample = samples[idx];
                    self.block_accumulator[ch] += sample * sample * self.channel_weights[ch];
                }
            }

            self.block_sample_count += 1;
            self.total_samples += 1;

            // Complete block if full
            if self.block_sample_count >= self.block_size {
                self.complete_block();
            }

            self.timestamp = self.total_samples as f64 / self.sample_rate;
        }
    }

    /// Complete a gating block.
    fn complete_block(&mut self) {
        if self.block_sample_count == 0 {
            return;
        }

        // Calculate block mean-square
        let mut block_ms = 0.0;
        for &channel_ms in &self.block_accumulator {
            block_ms += channel_ms / self.block_sample_count as f64;
        }

        // Normalize by total channel weight
        let total_weight: f64 = self.channel_weights.iter().sum();
        if total_weight > 0.0 {
            block_ms /= total_weight;
        }

        // Convert to LKFS
        let loudness = if block_ms > 0.0 {
            -0.691 + 10.0 * block_ms.log10()
        } else {
            f64::NEG_INFINITY
        };

        // Store block
        self.blocks
            .push(GatingBlock::new(loudness, block_ms, self.timestamp));

        // Reset accumulator
        self.block_accumulator.fill(0.0);
        self.block_sample_count = 0;
    }

    /// Calculate integrated loudness using two-stage gating.
    ///
    /// # Returns
    ///
    /// Gated integrated loudness in LUFS
    pub fn integrated_loudness(&self) -> f64 {
        if self.blocks.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Stage 1: Absolute gate (-70 LKFS)
        let absolute_gated: Vec<&GatingBlock> = self
            .blocks
            .iter()
            .filter(|block| block.loudness >= ABSOLUTE_GATE)
            .collect();

        if absolute_gated.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate ungated loudness (mean of absolute-gated blocks)
        let ungated_sum: f64 = absolute_gated.iter().map(|block| block.power).sum();
        let ungated_mean = ungated_sum / absolute_gated.len() as f64;
        let ungated_loudness = if ungated_mean > 0.0 {
            -0.691 + 10.0 * ungated_mean.log10()
        } else {
            f64::NEG_INFINITY
        };

        // Stage 2: Relative gate (-10 LU below ungated)
        let relative_gate = ungated_loudness + RELATIVE_GATE_OFFSET;
        let relative_gated: Vec<&GatingBlock> = absolute_gated
            .into_iter()
            .filter(|block| block.loudness >= relative_gate)
            .collect();

        if relative_gated.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate gated loudness
        let gated_sum: f64 = relative_gated.iter().map(|block| block.power).sum();
        let gated_mean = gated_sum / relative_gated.len() as f64;

        if gated_mean > 0.0 {
            -0.691 + 10.0 * gated_mean.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get blocks above absolute gate for LRA calculation.
    ///
    /// Returns loudness values (in LKFS) of all blocks above the absolute gate.
    pub fn get_blocks_for_lra(&self) -> Vec<f64> {
        self.blocks
            .iter()
            .filter(|block| block.loudness >= ABSOLUTE_GATE)
            .map(|block| block.loudness)
            .collect()
    }

    /// Get all blocks (for advanced analysis).
    pub fn get_all_blocks(&self) -> Vec<(f64, f64)> {
        self.blocks
            .iter()
            .map(|block| (block.loudness, block.timestamp))
            .collect()
    }

    /// Get number of blocks processed.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Calculate ITU-R BS.1770-4 channel weights.
    fn calculate_channel_weights(channels: usize) -> Vec<f64> {
        match channels {
            1 => vec![1.0],
            2 => vec![1.0, 1.0],
            5 => vec![1.0, 1.0, 1.0, 1.41, 1.41],      // 5.0
            6 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41], // 5.1
            8 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41], // 7.1
            _ => vec![1.0; channels],
        }
    }

    /// Reset the gating processor.
    pub fn reset(&mut self) {
        self.block_accumulator.fill(0.0);
        self.block_sample_count = 0;
        self.blocks.clear();
        self.timestamp = 0.0;
        self.total_samples = 0;
    }
}

/// Result of gating calculation.
#[derive(Clone, Debug)]
pub struct GatingResult {
    /// Integrated loudness in LUFS.
    pub integrated_lufs: f64,
    /// Number of blocks processed.
    pub total_blocks: usize,
    /// Blocks above absolute gate.
    pub absolute_gated_count: usize,
    /// Blocks above relative gate.
    pub relative_gated_count: usize,
    /// Ungated loudness (before relative gate).
    pub ungated_lufs: f64,
    /// Relative gate threshold used.
    pub relative_gate_threshold: f64,
}

impl GatingResult {
    /// Check if measurement is valid.
    pub fn is_valid(&self) -> bool {
        self.integrated_lufs.is_finite() && self.relative_gated_count > 0
    }

    /// Get percentage of blocks gated.
    pub fn gated_percentage(&self) -> f64 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        (1.0 - (self.relative_gated_count as f64 / self.total_blocks as f64)) * 100.0
    }
}

/// Calculate detailed gating result with statistics.
///
/// # Arguments
///
/// * `processor` - Gating processor
///
/// # Returns
///
/// Detailed gating result
pub fn calculate_gating_result(processor: &GatingProcessor) -> GatingResult {
    let integrated = processor.integrated_loudness();
    let total_blocks = processor.block_count();

    let blocks = processor.blocks.clone();

    let absolute_gated: Vec<&GatingBlock> = blocks
        .iter()
        .filter(|block| block.loudness >= ABSOLUTE_GATE)
        .collect();

    let absolute_gated_count = absolute_gated.len();

    let (ungated_lufs, relative_gate_threshold, relative_gated_count) = if absolute_gated.is_empty()
    {
        (f64::NEG_INFINITY, ABSOLUTE_GATE, 0)
    } else {
        let ungated_sum: f64 = absolute_gated.iter().map(|block| block.power).sum();
        let ungated_mean = ungated_sum / absolute_gated.len() as f64;
        let ungated = if ungated_mean > 0.0 {
            -0.691 + 10.0 * ungated_mean.log10()
        } else {
            f64::NEG_INFINITY
        };

        let rel_gate = ungated + RELATIVE_GATE_OFFSET;
        let rel_count = absolute_gated
            .iter()
            .filter(|block| block.loudness >= rel_gate)
            .count();

        (ungated, rel_gate, rel_count)
    };

    GatingResult {
        integrated_lufs: integrated,
        total_blocks,
        absolute_gated_count,
        relative_gated_count,
        ungated_lufs,
        relative_gate_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_processor_creates() {
        let processor = GatingProcessor::new(48000.0, 2);
        assert_eq!(processor.block_count(), 0);
    }

    #[test]
    fn test_gating_constants() {
        assert_eq!(ABSOLUTE_GATE, -70.0);
        assert_eq!(RELATIVE_GATE_OFFSET, -10.0);
    }

    #[test]
    fn test_empty_processor_returns_neg_infinity() {
        let processor = GatingProcessor::new(48000.0, 2);
        assert_eq!(processor.integrated_loudness(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_gating_result_validity() {
        let result = GatingResult {
            integrated_lufs: -23.0,
            total_blocks: 100,
            absolute_gated_count: 90,
            relative_gated_count: 80,
            ungated_lufs: -22.0,
            relative_gate_threshold: -32.0,
        };
        assert!(result.is_valid());
    }

    #[test]
    fn test_gated_percentage() {
        let result = GatingResult {
            integrated_lufs: -23.0,
            total_blocks: 100,
            absolute_gated_count: 90,
            relative_gated_count: 50,
            ungated_lufs: -22.0,
            relative_gate_threshold: -32.0,
        };
        assert_eq!(result.gated_percentage(), 50.0);
    }
}
