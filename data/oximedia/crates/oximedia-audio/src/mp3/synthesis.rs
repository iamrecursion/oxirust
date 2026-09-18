//! Polyphase synthesis filterbank for MP3.
//!
//! This module implements the 32-band polyphase synthesis filterbank
//! used to reconstruct PCM samples from frequency domain coefficients.

use std::f32::consts::PI;

/// Number of subbands.
const SUBBANDS: usize = 32;

/// FIFO buffer size.
const FIFO_SIZE: usize = 1024;

/// Synthesis filterbank.
pub struct SynthesisFilter {
    /// FIFO buffer (V vector).
    fifo: [[f32; FIFO_SIZE]; 2],
    /// FIFO write offset.
    offset: usize,
    /// Synthesis window coefficients.
    window: [f32; 512],
    /// Cosine modulation matrix.
    cos_table: [[f32; 64]; 32],
}

impl Default for SynthesisFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthesisFilter {
    /// Create new synthesis filter.
    #[must_use]
    pub fn new() -> Self {
        let mut filter = Self {
            fifo: [[0.0; FIFO_SIZE]; 2],
            offset: 0,
            window: [0.0; 512],
            cos_table: [[0.0; 64]; 32],
        };

        filter.init_tables();
        filter
    }

    /// Initialize synthesis window and cosine tables.
    fn init_tables(&mut self) {
        // Initialize synthesis window (based on ISO/IEC 11172-3)
        for i in 0..512 {
            #[allow(clippy::cast_precision_loss)]
            let n = i as f32;
            let angle = PI / 512.0 * (n + 0.5);

            // Simplified window function
            self.window[i] = angle.sin();
        }

        // Initialize cosine modulation table
        for i in 0..32 {
            for k in 0..64 {
                #[allow(clippy::cast_precision_loss)]
                let angle = PI / 64.0 * f32::from((2 * k + 1) as u8) * f32::from((16 + i) as u8);
                self.cos_table[i][k] = angle.cos();
            }
        }
    }

    /// Synthesize 32 PCM samples from 32 subband samples.
    pub fn synthesize(&mut self, samples: &[f32], channel: usize, output: &mut [f32]) {
        debug_assert!(samples.len() >= SUBBANDS);
        debug_assert!(output.len() >= SUBBANDS);

        // Matrixing: compute V vector from subband samples
        let mut v = [0.0f32; 64];
        for i in 0..32 {
            let mut sum = 0.0f32;
            for k in 0..32 {
                sum += samples[k] * self.cos_table[k][i * 2];
            }
            v[i] = sum;
            v[i + 32] = sum;
        }

        // Insert V into FIFO
        let offset = self.offset;
        for (i, &val) in v.iter().enumerate() {
            self.fifo[channel][(offset + i) % FIFO_SIZE] = val;
        }

        // Build U vector by windowing
        let mut u = [0.0f32; 512];
        for i in 0..8 {
            for j in 0..32 {
                let idx = (offset + i * 64 + j) % FIFO_SIZE;
                u[i * 64 + j] = self.fifo[channel][idx] * self.window[i * 64 + j];
            }
            for j in 0..32 {
                let idx = (offset + i * 64 + j + 32) % FIFO_SIZE;
                u[i * 64 + j + 32] = self.fifo[channel][idx] * self.window[i * 64 + j + 32];
            }
        }

        // Compute output samples by summing U vectors
        for i in 0..32 {
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum += u[i + j * 32];
            }
            output[i] = sum;
        }

        // Update offset
        self.offset = (self.offset + 64) % FIFO_SIZE;
    }

    /// Reset filter state.
    pub fn reset(&mut self) {
        self.fifo = [[0.0; FIFO_SIZE]; 2];
        self.offset = 0;
    }

    /// Get FIFO offset.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }
}

/// Simplified polyphase synthesis for testing.
pub struct SimpleSynthesis {
    overlap: [[f32; 16]; 2],
}

impl Default for SimpleSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleSynthesis {
    /// Create new simple synthesis.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            overlap: [[0.0; 16]; 2],
        }
    }

    /// Synthesize with overlap-add.
    pub fn synthesize(&mut self, input: &[f32], channel: usize, output: &mut [f32]) {
        debug_assert!(input.len() >= 32);
        debug_assert!(output.len() >= 32);

        // Simple overlap-add synthesis
        for i in 0..16 {
            output[i] = input[i] + self.overlap[channel][i];
            self.overlap[channel][i] = input[i + 16];
        }

        for i in 16..32 {
            output[i] = input[i];
        }
    }

    /// Reset overlap buffer.
    pub fn reset(&mut self) {
        self.overlap = [[0.0; 16]; 2];
    }
}

/// DCT-based synthesis (alternative implementation).
#[allow(clippy::cast_precision_loss)]
pub fn dct_synthesis(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= 32);
    debug_assert!(output.len() >= 32);

    // Type-II DCT for synthesis
    for k in 0..32 {
        let mut sum = 0.0f32;

        for n in 0..32 {
            let angle = PI / 32.0 * f32::from((k + 1) as u8) * (f32::from(n as u8) + 0.5);
            sum += input[n] * angle.cos();
        }

        output[k] = sum * (2.0 / 32.0f32).sqrt();
    }
}

/// Apply post-synthesis filtering (deemphasis if needed).
pub fn apply_deemphasis(samples: &mut [f32], emphasis: bool, state: &mut f32) {
    if !emphasis {
        return;
    }

    // 50/15 microseconds deemphasis filter
    const COEF: f32 = 0.95;

    for sample in samples.iter_mut() {
        *sample += *state * COEF;
        *state = *sample;
    }
}

/// Normalize output samples to prevent clipping.
pub fn normalize_samples(samples: &mut [f32]) {
    // Find peak
    let mut peak = 0.0f32;
    for &sample in samples.iter() {
        peak = peak.max(sample.abs());
    }

    // Normalize if needed
    if peak > 1.0 {
        let scale = 1.0 / peak;
        for sample in samples.iter_mut() {
            *sample *= scale;
        }
    }
}

/// Interleave stereo samples (L, R, L, R, ...).
pub fn interleave_stereo(left: &[f32], right: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= left.len() + right.len());
    debug_assert!(left.len() == right.len());

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        output[i * 2] = l;
        output[i * 2 + 1] = r;
    }
}

/// Convert f32 samples to i16.
pub fn convert_to_i16(input: &[f32], output: &mut [i16]) {
    debug_assert!(input.len() == output.len());

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        // Clamp to [-1.0, 1.0] and scale to i16 range
        let clamped = inp.clamp(-1.0, 1.0);
        #[allow(clippy::cast_possible_truncation)]
        let scaled = (clamped * 32767.0) as i16;
        *out = scaled;
    }
}

/// Convert f32 samples to f32 (with normalization).
pub fn convert_to_f32(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() == output.len());

    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = inp.clamp(-1.0, 1.0);
    }
}
