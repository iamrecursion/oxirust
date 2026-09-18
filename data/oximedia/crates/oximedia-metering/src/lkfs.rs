//! LKFS/LUFS loudness calculation.
//!
//! Implements momentary and short-term loudness measurement according to
//! ITU-R BS.1770-4 specification.
//!
//! LKFS (Loudness, K-weighted, relative to Full Scale) and LUFS (Loudness Units
//! relative to Full Scale) are identical measurements with different terminology.

use std::collections::VecDeque;

/// Momentary loudness window duration (400ms).
const MOMENTARY_WINDOW_MS: f64 = 400.0;

/// Short-term loudness window duration (3000ms).
const SHORT_TERM_WINDOW_MS: f64 = 3000.0;

/// Block size for measurements (100ms).
const BLOCK_SIZE_MS: f64 = 100.0;

/// Overlap percentage (75%).
const OVERLAP_PERCENTAGE: f64 = 0.75;

/// LUFS value with associated metadata.
#[derive(Clone, Copy, Debug)]
pub struct LufsValue {
    /// Loudness value in LUFS.
    pub lufs: f64,
    /// Timestamp in seconds.
    pub timestamp: f64,
}

impl LufsValue {
    /// Create a new LUFS value.
    pub fn new(lufs: f64, timestamp: f64) -> Self {
        Self { lufs, timestamp }
    }

    /// Check if valid (not infinite).
    pub fn is_valid(&self) -> bool {
        self.lufs.is_finite()
    }
}

/// LKFS/LUFS calculator for momentary and short-term loudness.
///
/// Processes K-weighted audio to calculate loudness values according to
/// ITU-R BS.1770-4.
pub struct LkfsCalculator {
    sample_rate: f64,
    channels: usize,

    // Block processing
    block_size: usize,
    block_accumulator: Vec<f64>,
    block_sample_count: usize,

    // Momentary loudness (400ms)
    momentary_blocks: VecDeque<f64>,
    momentary_capacity: usize,
    momentary_loudness: f64,
    max_momentary: f64,

    // Short-term loudness (3s)
    short_term_blocks: VecDeque<f64>,
    short_term_capacity: usize,
    short_term_loudness: f64,
    max_short_term: f64,

    // Channel weights (ITU-R BS.1770-4)
    channel_weights: Vec<f64>,

    // Timing
    timestamp: f64,
    total_samples: usize,
}

impl LkfsCalculator {
    /// Create a new LKFS calculator.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let block_size = (sample_rate * BLOCK_SIZE_MS / 1000.0) as usize;

        let momentary_blocks_count = (MOMENTARY_WINDOW_MS / BLOCK_SIZE_MS) as usize;
        let short_term_blocks_count = (SHORT_TERM_WINDOW_MS / BLOCK_SIZE_MS) as usize;

        let channel_weights = Self::calculate_channel_weights(channels);

        Self {
            sample_rate,
            channels,
            block_size,
            block_accumulator: vec![0.0; channels],
            block_sample_count: 0,
            momentary_blocks: VecDeque::with_capacity(momentary_blocks_count),
            momentary_capacity: momentary_blocks_count,
            momentary_loudness: f64::NEG_INFINITY,
            max_momentary: f64::NEG_INFINITY,
            short_term_blocks: VecDeque::with_capacity(short_term_blocks_count),
            short_term_capacity: short_term_blocks_count,
            short_term_loudness: f64::NEG_INFINITY,
            max_short_term: f64::NEG_INFINITY,
            channel_weights,
            timestamp: 0.0,
            total_samples: 0,
        }
    }

    /// Process interleaved K-weighted audio samples.
    ///
    /// Input samples should already be K-weighted filtered.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved K-weighted samples
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let frames = samples.len() / self.channels;

        for frame in 0..frames {
            // Accumulate mean-square per channel
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                if idx < samples.len() {
                    let sample = samples[idx];
                    self.block_accumulator[ch] += sample * sample * self.channel_weights[ch];
                }
            }

            self.block_sample_count += 1;
            self.total_samples += 1;

            // Check if block is complete
            if self.block_sample_count >= self.block_size {
                self.complete_block();
            }

            self.timestamp = self.total_samples as f64 / self.sample_rate;
        }
    }

    /// Complete a measurement block.
    fn complete_block(&mut self) {
        if self.block_sample_count == 0 {
            return;
        }

        // Calculate block mean-square (averaged across channels)
        let mut block_ms = 0.0;
        for &channel_ms in &self.block_accumulator {
            block_ms += channel_ms / self.block_sample_count as f64;
        }

        // Normalize by total channel weight
        let total_weight: f64 = self.channel_weights.iter().sum();
        if total_weight > 0.0 {
            block_ms /= total_weight;
        }

        // Add to momentary window
        self.momentary_blocks.push_back(block_ms);
        if self.momentary_blocks.len() > self.momentary_capacity {
            self.momentary_blocks.pop_front();
        }

        // Add to short-term window
        self.short_term_blocks.push_back(block_ms);
        if self.short_term_blocks.len() > self.short_term_capacity {
            self.short_term_blocks.pop_front();
        }

        // Update loudness values
        if self.momentary_blocks.len() == self.momentary_capacity {
            self.momentary_loudness = Self::calculate_loudness(&self.momentary_blocks);
            self.max_momentary = self.max_momentary.max(self.momentary_loudness);
        }

        if self.short_term_blocks.len() == self.short_term_capacity {
            self.short_term_loudness = Self::calculate_loudness(&self.short_term_blocks);
            self.max_short_term = self.max_short_term.max(self.short_term_loudness);
        }

        // Reset block accumulator
        self.block_accumulator.fill(0.0);
        self.block_sample_count = 0;
    }

    /// Calculate loudness from block mean-squares.
    ///
    /// LUFS = -0.691 + 10 * log10(mean-square)
    fn calculate_loudness(blocks: &VecDeque<f64>) -> f64 {
        if blocks.is_empty() {
            return f64::NEG_INFINITY;
        }

        let mean_ms: f64 = blocks.iter().sum::<f64>() / blocks.len() as f64;

        if mean_ms > 0.0 {
            -0.691 + 10.0 * mean_ms.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Calculate ITU-R BS.1770-4 channel weights.
    fn calculate_channel_weights(channels: usize) -> Vec<f64> {
        match channels {
            1 => vec![1.0],
            2 => vec![1.0, 1.0],
            5 => vec![1.0, 1.0, 1.0, 1.41, 1.41], // 5.0: L, R, C, Ls, Rs
            6 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41], // 5.1: L, R, C, LFE, Ls, Rs
            8 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41], // 7.1
            _ => vec![1.0; channels],
        }
    }

    /// Get current momentary loudness (400ms).
    pub fn momentary_loudness(&self) -> f64 {
        self.momentary_loudness
    }

    /// Get current short-term loudness (3s).
    pub fn short_term_loudness(&self) -> f64 {
        self.short_term_loudness
    }

    /// Get maximum momentary loudness seen.
    pub fn max_momentary(&self) -> f64 {
        self.max_momentary
    }

    /// Get maximum short-term loudness seen.
    pub fn max_short_term(&self) -> f64 {
        self.max_short_term
    }

    /// Get all momentary blocks for gating/LRA calculation.
    pub fn get_momentary_blocks(&self) -> Vec<f64> {
        self.momentary_blocks.iter().copied().collect()
    }

    /// Reset the calculator.
    pub fn reset(&mut self) {
        self.block_accumulator.fill(0.0);
        self.block_sample_count = 0;
        self.momentary_blocks.clear();
        self.short_term_blocks.clear();
        self.momentary_loudness = f64::NEG_INFINITY;
        self.short_term_loudness = f64::NEG_INFINITY;
        self.max_momentary = f64::NEG_INFINITY;
        self.max_short_term = f64::NEG_INFINITY;
        self.timestamp = 0.0;
        self.total_samples = 0;
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Convert power to LUFS.
///
/// # Arguments
///
/// * `power` - Mean-square power value
///
/// # Returns
///
/// Loudness in LUFS
pub fn power_to_lufs(power: f64) -> f64 {
    if power > 0.0 {
        -0.691 + 10.0 * power.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Convert LUFS to power (mean-square).
///
/// # Arguments
///
/// * `lufs` - Loudness in LUFS
///
/// # Returns
///
/// Mean-square power value
pub fn lufs_to_power(lufs: f64) -> f64 {
    if lufs.is_finite() {
        10.0_f64.powf((lufs + 0.691) / 10.0)
    } else {
        0.0
    }
}

/// Convert dB to linear scale.
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear to dB.
pub fn linear_to_db(linear: f64) -> f64 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        f64::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lkfs_calculator_creates() {
        let calc = LkfsCalculator::new(48000.0, 2);
        assert_eq!(calc.sample_rate(), 48000.0);
        assert_eq!(calc.channels(), 2);
    }

    #[test]
    fn test_power_to_lufs_conversion() {
        let power = 0.1;
        let lufs = power_to_lufs(power);
        assert!(lufs.is_finite());
        assert!(lufs < 0.0); // Should be negative dB
    }

    #[test]
    fn test_lufs_power_roundtrip() {
        let original_power = 0.05;
        let lufs = power_to_lufs(original_power);
        let recovered_power = lufs_to_power(lufs);
        assert!((original_power - recovered_power).abs() < 1e-10);
    }

    #[test]
    fn test_db_linear_conversion() {
        let db = -6.0;
        let linear = db_to_linear(db);
        let recovered_db = linear_to_db(linear);
        assert!((db - recovered_db).abs() < 1e-10);
    }

    #[test]
    fn test_channel_weights_stereo() {
        let weights = LkfsCalculator::calculate_channel_weights(2);
        assert_eq!(weights, vec![1.0, 1.0]);
    }

    #[test]
    fn test_channel_weights_51() {
        let weights = LkfsCalculator::calculate_channel_weights(6);
        assert_eq!(weights[3], 0.0); // LFE should be 0
        assert_eq!(weights[4], 1.41); // Surround should be 1.41
    }
}
