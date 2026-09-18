//! Gating algorithm for integrated loudness measurement.
//!
//! Implements the two-stage gating algorithm specified in ITU-R BS.1770-4
//! and EBU R128 to measure integrated loudness.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]

/// Absolute gate threshold in LUFS.
///
/// Blocks below this level are excluded from loudness measurement.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;

/// Relative gate offset in LU (Loudness Units).
///
/// The relative gate is 10 LU below the ungated loudness.
const RELATIVE_GATE_OFFSET_LU: f64 = 10.0;

/// Gating processor for integrated loudness measurement.
///
/// The gating algorithm works in two stages:
/// 1. Absolute gate: Exclude blocks below -70 LUFS
/// 2. Relative gate: Exclude blocks below (ungated_loudness - 10 LU)
///
/// This prevents silence and very quiet passages from skewing the measurement.
#[derive(Clone, Debug)]
pub struct GatingProcessor {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Number of audio channels.
    channels: usize,
    /// Channel weights for summing (per ITU-R BS.1770-4).
    channel_weights: Vec<f64>,
}

impl GatingProcessor {
    /// Create a new gating processor.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        let channel_weights = Self::calculate_channel_weights(channels);

        Self {
            sample_rate,
            channels,
            channel_weights,
        }
    }

    /// Calculate channel weights according to ITU-R BS.1770-4.
    ///
    /// For stereo and 5.1:
    /// - Left, Right, Center: weight 1.0
    /// - Left surround, Right surround: weight 1.41 (+3 dB)
    /// - LFE: weight 0.0 (not included)
    fn calculate_channel_weights(channels: usize) -> Vec<f64> {
        match channels {
            1 => vec![1.0],                                        // Mono
            2 => vec![1.0, 1.0],                                   // Stereo: L, R
            3 => vec![1.0, 1.0, 0.0],                              // 2.1: L, R, LFE (ignore LFE)
            5 => vec![1.0, 1.0, 1.0, 1.41, 1.41],                  // 5.0: L, R, C, Ls, Rs
            6 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41],             // 5.1: L, R, C, LFE, Ls, Rs
            8 => vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.41, 1.41], // 7.1: L, R, C, LFE, Ls, Rs, Lb, Rb
            _ => vec![1.0; channels], // Default: all channels equal weight
        }
    }

    /// Calculate mean square power for a block of samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    ///
    /// # Returns
    ///
    /// Mean square power (linear scale)
    #[must_use]
    pub fn calculate_block_power(&self, samples: &[f64]) -> f64 {
        if samples.is_empty() || self.channels == 0 {
            return 0.0;
        }

        let frames = samples.len() / self.channels;
        if frames == 0 {
            return 0.0;
        }

        let mut power_sum = 0.0;
        let mut weight_sum = 0.0;

        for frame in 0..frames {
            for ch in 0..self.channels {
                let idx = frame * self.channels + ch;
                let weight = self.channel_weights.get(ch).copied().unwrap_or(1.0);

                if weight > 0.0 {
                    let sample = samples.get(idx).copied().unwrap_or(0.0);
                    power_sum += weight * sample * sample;
                    weight_sum += weight;
                }
            }
        }

        // Per ITU-R BS.1770-4: z_i = (1/T) Σ y_i² is the *per-channel* mean
        // square, and the gated loudness sums the weighted channel powers
        // (`-0.691 + 10·log10(Σ Gi·z_i)`). The channel weights `Gi` are already
        // folded into `power_sum`, so the block mean is `power_sum / frames`.
        // `weight_sum` here is ONLY the "any active channel" guard — it must NOT
        // divide the power again (doing so understates loudness by ~10·log10(N)).
        if weight_sum > 0.0 {
            power_sum / frames as f64
        } else {
            0.0
        }
    }

    /// Calculate mean square power for planar samples.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of per-channel sample buffers
    ///
    /// # Returns
    ///
    /// Mean square power (linear scale)
    #[must_use]
    pub fn calculate_block_power_planar(&self, channels: &[&[f64]]) -> f64 {
        if channels.is_empty() {
            return 0.0;
        }

        let frames = channels[0].len();
        if frames == 0 {
            return 0.0;
        }

        let mut power_sum = 0.0;
        let mut weight_sum = 0.0;

        for (ch_idx, samples) in channels.iter().enumerate() {
            let weight = self.channel_weights.get(ch_idx).copied().unwrap_or(1.0);

            if weight > 0.0 {
                for &sample in samples.iter().take(frames) {
                    power_sum += weight * sample * sample;
                }
                weight_sum += weight;
            }
        }

        // See `calculate_block_power`: the BS.1770 weighted channel-power sum
        // already has the weights `Gi` in `power_sum`; the block mean is
        // `power_sum / frames`. `weight_sum` is only the active-channel guard.
        if weight_sum > 0.0 {
            power_sum / frames as f64
        } else {
            0.0
        }
    }

    /// Convert mean square power to loudness in LUFS.
    ///
    /// # Arguments
    ///
    /// * `power` - Mean square power (linear scale)
    ///
    /// # Returns
    ///
    /// Loudness in LUFS (Loudness Units relative to Full Scale)
    #[must_use]
    pub fn power_to_lufs(power: f64) -> f64 {
        if power <= 0.0 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * power.log10()
        }
    }

    /// Convert LUFS to mean square power.
    ///
    /// # Arguments
    ///
    /// * `lufs` - Loudness in LUFS
    ///
    /// # Returns
    ///
    /// Mean square power (linear scale)
    #[must_use]
    pub fn lufs_to_power(lufs: f64) -> f64 {
        if lufs.is_infinite() && lufs.is_sign_negative() {
            0.0
        } else {
            10.0_f64.powf((lufs + 0.691) / 10.0)
        }
    }

    /// Apply two-stage gating to block powers and calculate integrated loudness.
    ///
    /// # Arguments
    ///
    /// * `block_powers` - Slice of block mean square powers
    ///
    /// # Returns
    ///
    /// Integrated loudness in LUFS
    #[must_use]
    pub fn calculate_gated_loudness(&self, block_powers: &[f64]) -> f64 {
        if block_powers.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Stage 1: Absolute gate (-70 LUFS)
        let absolute_threshold = Self::lufs_to_power(ABSOLUTE_GATE_LUFS);
        let mut gated_powers: Vec<f64> = block_powers
            .iter()
            .copied()
            .filter(|&p| p >= absolute_threshold)
            .collect();

        if gated_powers.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate ungated loudness (from absolute-gated blocks)
        let ungated_power: f64 = gated_powers.iter().sum::<f64>() / gated_powers.len() as f64;
        let ungated_lufs = Self::power_to_lufs(ungated_power);

        // Stage 2: Relative gate (ungated - 10 LU)
        let relative_threshold_lufs = ungated_lufs - RELATIVE_GATE_OFFSET_LU;
        let relative_threshold = Self::lufs_to_power(relative_threshold_lufs);

        gated_powers.retain(|&p| p >= relative_threshold);

        if gated_powers.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate final gated loudness
        let gated_power: f64 = gated_powers.iter().sum::<f64>() / gated_powers.len() as f64;
        Self::power_to_lufs(gated_power)
    }

    /// Calculate loudness range (LRA) from block powers.
    ///
    /// LRA is the difference between the 95th and 10th percentile
    /// of the block loudness distribution (after gating).
    ///
    /// # Arguments
    ///
    /// * `block_powers` - Slice of block mean square powers
    ///
    /// # Returns
    ///
    /// Loudness range in LU (Loudness Units)
    #[must_use]
    pub fn calculate_loudness_range(&self, block_powers: &[f64]) -> f64 {
        if block_powers.is_empty() {
            return 0.0;
        }

        // Apply absolute gate only for LRA
        let absolute_threshold = Self::lufs_to_power(ABSOLUTE_GATE_LUFS);
        let mut gated_lufs: Vec<f64> = block_powers
            .iter()
            .copied()
            .filter(|&p| p >= absolute_threshold)
            .map(Self::power_to_lufs)
            .collect();

        if gated_lufs.len() < 2 {
            return 0.0;
        }

        // Sort for percentile calculation
        gated_lufs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate 10th and 95th percentiles
        let low_idx = ((gated_lufs.len() - 1) as f64 * 0.10) as usize;
        let high_idx = ((gated_lufs.len() - 1) as f64 * 0.95) as usize;

        let low_percentile = gated_lufs[low_idx];
        let high_percentile = gated_lufs[high_idx];

        // LRA is the difference
        (high_percentile - low_percentile).max(0.0)
    }

    /// Get the absolute gate threshold in LUFS.
    #[must_use]
    pub fn absolute_gate_threshold() -> f64 {
        ABSOLUTE_GATE_LUFS
    }

    /// Get the relative gate offset in LU.
    #[must_use]
    pub fn relative_gate_offset() -> f64 {
        RELATIVE_GATE_OFFSET_LU
    }

    /// Get channel weights.
    #[must_use]
    pub fn channel_weights(&self) -> &[f64] {
        &self.channel_weights
    }
}

/// Block-based accumulator for gated loudness measurement.
///
/// Accumulates audio blocks and calculates integrated loudness with gating.
#[derive(Clone, Debug)]
pub struct BlockAccumulator {
    /// Gating processor.
    gating: GatingProcessor,
    /// Accumulated block powers.
    block_powers: Vec<f64>,
    /// Block size in samples (400ms for momentary, etc.).
    block_size: usize,
    /// Current block buffer.
    current_block: Vec<f64>,
    /// Number of samples in current block.
    current_count: usize,
}

impl BlockAccumulator {
    /// Create a new block accumulator.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `block_duration_ms` - Block duration in milliseconds
    #[must_use]
    pub fn new(sample_rate: f64, channels: usize, block_duration_ms: f64) -> Self {
        let gating = GatingProcessor::new(sample_rate, channels);
        let block_size = (sample_rate * block_duration_ms / 1000.0) as usize * channels;

        Self {
            gating,
            block_powers: Vec::new(),
            block_size,
            current_block: vec![0.0; block_size],
            current_count: 0,
        }
    }

    /// Add samples to the accumulator.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    pub fn add_samples(&mut self, samples: &[f64]) {
        let mut offset = 0;

        while offset < samples.len() {
            let space_left = self.block_size - self.current_count;
            let to_copy = (samples.len() - offset).min(space_left);

            self.current_block[self.current_count..self.current_count + to_copy]
                .copy_from_slice(&samples[offset..offset + to_copy]);

            self.current_count += to_copy;
            offset += to_copy;

            // Block complete?
            if self.current_count >= self.block_size {
                let power = self.gating.calculate_block_power(&self.current_block);
                self.block_powers.push(power);
                self.current_count = 0;
            }
        }
    }

    /// Get integrated loudness in LUFS.
    #[must_use]
    pub fn integrated_loudness(&self) -> f64 {
        self.gating.calculate_gated_loudness(&self.block_powers)
    }

    /// Get loudness range in LU.
    #[must_use]
    pub fn loudness_range(&self) -> f64 {
        self.gating.calculate_loudness_range(&self.block_powers)
    }

    /// Get number of blocks accumulated.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.block_powers.len()
    }

    /// Reset accumulator.
    pub fn reset(&mut self) {
        self.block_powers.clear();
        self.current_count = 0;
    }

    /// Get block powers (for analysis/debugging).
    #[must_use]
    pub fn block_powers(&self) -> &[f64] {
        &self.block_powers
    }
}
