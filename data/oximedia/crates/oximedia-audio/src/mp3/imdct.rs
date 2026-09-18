//! Inverse Modified Discrete Cosine Transform (IMDCT) for MP3.
//!
//! This module implements the IMDCT used in MP3 Layer III decoding.
//! Supports both long (36-point) and short (12-point) blocks with windowing.

#![allow(clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// IMDCT processor.
pub struct Imdct {
    /// Cosine tables for long blocks (36-point).
    cos_long: [[f32; 36]; 18],
    /// Cosine tables for short blocks (12-point).
    cos_short: [[f32; 12]; 6],
    /// Window coefficients for long blocks.
    window_long: [f32; 36],
    /// Window coefficients for short blocks.
    window_short: [f32; 12],
    /// Overlap buffer (previous block output).
    overlap: [[f32; 18]; 2],
}

impl Default for Imdct {
    fn default() -> Self {
        Self::new()
    }
}

impl Imdct {
    /// Create new IMDCT processor.
    #[must_use]
    pub fn new() -> Self {
        let mut imdct = Self {
            cos_long: [[0.0; 36]; 18],
            cos_short: [[0.0; 12]; 6],
            window_long: [0.0; 36],
            window_short: [0.0; 12],
            overlap: [[0.0; 18]; 2],
        };

        imdct.init_tables();
        imdct
    }

    /// Initialize cosine and window tables.
    fn init_tables(&mut self) {
        // Long block cosine table (36-point IMDCT)
        for i in 0..18 {
            for k in 0..36 {
                let angle = PI / 36.0 * (f32::from(k as u8) + 0.5) * f32::from((2 * i + 1) as u8);
                self.cos_long[i][k] = angle.cos();
            }
        }

        // Short block cosine table (12-point IMDCT)
        for i in 0..6 {
            for k in 0..12 {
                let angle = PI / 12.0 * (f32::from(k as u8) + 0.5) * f32::from((2 * i + 1) as u8);
                self.cos_short[i][k] = angle.cos();
            }
        }

        // Long block window (sine window)
        for i in 0..36 {
            let angle = PI / 36.0 * (f32::from(i as u8) + 0.5);
            self.window_long[i] = angle.sin();
        }

        // Short block window (sine window)
        for i in 0..12 {
            let angle = PI / 12.0 * (f32::from(i as u8) + 0.5);
            self.window_short[i] = angle.sin();
        }
    }

    /// Perform 36-point IMDCT for long blocks.
    pub fn imdct36(&mut self, input: &[f32], output: &mut [f32], channel: usize) {
        debug_assert!(input.len() >= 18);
        debug_assert!(output.len() >= 36);

        let mut tmp = [0.0f32; 36];

        // IMDCT transformation
        for k in 0..36 {
            let mut sum = 0.0f32;
            for i in 0..18 {
                sum += input[i] * self.cos_long[i][k];
            }
            tmp[k] = sum * self.window_long[k];
        }

        // Overlap-add with previous block
        for i in 0..18 {
            output[i] = tmp[i] + self.overlap[channel][i];
            self.overlap[channel][i] = tmp[i + 18];
        }
    }

    /// Perform 12-point IMDCT for short blocks.
    pub fn imdct12(&mut self, input: &[f32], output: &mut [f32], channel: usize, block: usize) {
        debug_assert!(input.len() >= 6);
        debug_assert!(output.len() >= 12);

        let mut tmp = [0.0f32; 12];

        // IMDCT transformation
        for k in 0..12 {
            let mut sum = 0.0f32;
            for i in 0..6 {
                sum += input[i] * self.cos_short[i][k];
            }
            tmp[k] = sum * self.window_short[k];
        }

        // Overlap-add (short blocks are windowed differently)
        let offset = block * 6;
        if offset < 18 {
            for i in 0..12.min(18 - offset) {
                if offset + i < 18 {
                    output[i] = tmp[i] + self.overlap[channel][offset + i];
                } else {
                    output[i] = tmp[i];
                }
            }

            // Update overlap for next block
            for i in 0..6.min(18usize.saturating_sub(offset)) {
                if offset + 6 + i < 18 {
                    self.overlap[channel][offset + 6 + i] = tmp[6 + i];
                }
            }
        }
    }

    /// Process mixed blocks (start/stop transitions).
    #[allow(clippy::too_many_arguments)]
    pub fn imdct_mixed(
        &mut self,
        input_long: &[f32],
        input_short: &[[f32; 6]; 3],
        output: &mut [f32],
        channel: usize,
        block_type: BlockType,
    ) {
        match block_type {
            BlockType::Long => {
                self.imdct36(input_long, output, channel);
            }
            BlockType::Short => {
                // Three short blocks
                for (i, short_input) in input_short.iter().enumerate() {
                    let mut short_out = [0.0f32; 12];
                    self.imdct12(short_input, &mut short_out, channel, i);

                    // Copy to output with proper offset
                    let offset = i * 6;
                    for (j, &sample) in short_out.iter().enumerate().take(12.min(36 - offset)) {
                        if offset + j < 36 {
                            output[offset + j] = sample;
                        }
                    }
                }
            }
            BlockType::Start => {
                // Long block with start window
                self.imdct36(input_long, output, channel);
            }
            BlockType::Stop => {
                // Long block with stop window
                self.imdct36(input_long, output, channel);
            }
        }
    }

    /// Reset overlap buffer.
    pub fn reset(&mut self) {
        self.overlap = [[0.0; 18]; 2];
    }

    /// Get overlap buffer for debugging.
    #[must_use]
    pub const fn overlap(&self) -> &[[f32; 18]; 2] {
        &self.overlap
    }
}

/// Block type for IMDCT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    /// Long block (normal).
    Long,
    /// Short block (three 12-point blocks).
    Short,
    /// Start block (transition to short).
    Start,
    /// Stop block (transition from short).
    Stop,
}

/// Window switching for transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSwitching {
    /// No window switching.
    None,
    /// Start window (long to short).
    Start,
    /// Short windows.
    Short,
    /// Stop window (short to long).
    Stop,
}

/// Apply window function to samples.
pub fn apply_window(samples: &mut [f32], block_type: BlockType) {
    match block_type {
        BlockType::Long => {
            // Apply long window (sine window)
            for (i, sample) in samples.iter_mut().enumerate().take(36) {
                let angle = PI / 36.0 * (i as f32 + 0.5);
                *sample *= angle.sin();
            }
        }
        BlockType::Short => {
            // Apply short window (sine window, repeated 3 times)
            for block in 0..3 {
                let offset = block * 12;
                for i in 0..12 {
                    if offset + i < samples.len() {
                        let angle = PI / 12.0 * (i as f32 + 0.5);
                        samples[offset + i] *= angle.sin();
                    }
                }
            }
        }
        BlockType::Start | BlockType::Stop => {
            // Apply transition windows (simplified)
            for (i, sample) in samples.iter_mut().enumerate().take(36) {
                let angle = PI / 36.0 * (i as f32 + 0.5);
                *sample *= angle.sin();
            }
        }
    }
}

/// Perform antialiasing filtering (for Layer III).
pub fn antialias(samples: &mut [f32], sb_limit: usize) {
    // Antialiasing coefficients
    const CS: [f32; 8] = [
        0.857_492_9,
        0.881_741_9,
        0.949_628_6,
        0.983_314_6,
        0.995_517_8,
        0.999_160_6,
        0.999_899_2,
        0.999_993_2,
    ];

    const CA: [f32; 8] = [
        -0.514_495_8,
        -0.471_731_9,
        -0.313_377_5,
        -0.181_913_2,
        -0.094_574_19,
        -0.040_965_58,
        -0.014_198_57,
        -0.003_699_97,
    ];

    // Apply antialiasing between subbands
    for sb in 1..sb_limit.min(32) {
        let offset = sb * 18;
        if offset >= 8 {
            for i in 0..8 {
                let idx1 = offset - 1 - i;
                let idx2 = offset + i;

                if idx1 < samples.len() && idx2 < samples.len() {
                    let s1 = samples[idx1];
                    let s2 = samples[idx2];

                    samples[idx1] = s1 * CS[i] - s2 * CA[i];
                    samples[idx2] = s2 * CS[i] + s1 * CA[i];
                }
            }
        }
    }
}
