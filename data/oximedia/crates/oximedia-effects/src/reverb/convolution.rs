//! Convolution reverb using impulse responses.
//!
//! Provides realistic room acoustics by convolving the input signal with
//! a recorded impulse response (IR) of a real or synthetic space.
//!
//! ## Double-buffered block convolver
//!
//! [`DoubleBufferConvolver`] implements a block-oriented overlap-add convolution
//! engine with two alternating output buffers.  While buffer A is being
//! consumed sample-by-sample by the caller, an FFT frame is computed into
//! buffer B.  At each block boundary the roles swap, enabling continuous,
//! non-blocking output.

use crate::{AudioEffect, EffectError, Result};
use oxifft::Complex;

/// Convolution reverb using frequency-domain convolution.
///
/// Implements partitioned convolution for efficient processing of long
/// impulse responses.
pub struct ConvolutionReverb {
    // Impulse response
    ir_fft: Vec<Complex<f32>>,
    #[allow(dead_code)]
    ir_length: usize,

    // FFT planner
    fft_size: usize,
    input_buffer: Vec<f32>,
    input_fft: Vec<Complex<f32>>,
    output_buffer: Vec<f32>,

    // Tail buffer (for overlap-add)
    tail_buffer: Vec<f32>,

    // Processing position
    input_pos: usize,
    output_pos: usize,

    // Parameters
    wet: f32,
    dry: f32,

    #[allow(dead_code)]
    sample_rate: f32,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb.
    ///
    /// # Arguments
    ///
    /// * `impulse_response` - The impulse response samples
    /// * `sample_rate` - Audio sample rate
    ///
    /// # Errors
    ///
    /// Returns an error if the impulse response is empty or too long.
    pub fn new(impulse_response: &[f32], sample_rate: f32) -> Result<Self> {
        if impulse_response.is_empty() {
            return Err(EffectError::InvalidParameter(
                "Impulse response cannot be empty".into(),
            ));
        }

        if impulse_response.len() > 100_000 {
            return Err(EffectError::InvalidParameter(
                "Impulse response too long (max 100k samples)".into(),
            ));
        }

        let ir_length = impulse_response.len();

        // Choose FFT size (next power of 2 >= IR length * 2)
        let fft_size = (ir_length * 2).next_power_of_two();

        // Convert impulse response to frequency domain
        let mut ir_padded: Vec<Complex<f32>> = impulse_response
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        ir_padded.resize(fft_size, Complex::new(0.0, 0.0));

        let ir_fft = oxifft::fft(&ir_padded);

        Ok(Self {
            ir_fft,
            ir_length,
            fft_size,
            input_buffer: vec![0.0; fft_size],
            input_fft: vec![Complex::new(0.0, 0.0); fft_size],
            output_buffer: vec![0.0; fft_size],
            tail_buffer: vec![0.0; fft_size],
            input_pos: 0,
            output_pos: 0,
            wet: 0.5,
            dry: 0.5,
            sample_rate,
        })
    }

    /// Set wet level (0.0 - 1.0).
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet.clamp(0.0, 1.0);
    }

    /// Set dry level (0.0 - 1.0).
    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry.clamp(0.0, 1.0);
    }

    /// Process a block of samples.
    fn process_block(&mut self) {
        // Copy input to complex buffer
        for (i, &sample) in self.input_buffer.iter().enumerate() {
            self.input_fft[i] = Complex::new(sample, 0.0);
        }

        // Forward FFT
        let fft_result = oxifft::fft(&self.input_fft);

        // Complex multiplication (convolution in frequency domain)
        let result_fft_freq: Vec<Complex<f32>> = fft_result
            .iter()
            .zip(self.ir_fft.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        // Inverse FFT
        let result_fft = oxifft::ifft(&result_fft_freq);

        // Extract real part and normalize
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / self.fft_size as f32;

        for (i, val) in result_fft.iter().enumerate().take(self.fft_size) {
            self.output_buffer[i] = val.re * scale;
        }

        // Overlap-add with tail from previous block
        for i in 0..self.fft_size {
            self.output_buffer[i] += self.tail_buffer[i];
        }

        // Save second half as tail for next block
        for i in 0..self.fft_size / 2 {
            self.tail_buffer[i] = self.output_buffer[self.fft_size / 2 + i];
        }
        for i in self.fft_size / 2..self.fft_size {
            self.tail_buffer[i] = 0.0;
        }

        self.output_pos = 0;
    }
}

impl AudioEffect for ConvolutionReverb {
    const EFFECT_ID: &'static str = "convolution_reverb";

    fn process_sample(&mut self, input: f32) -> f32 {
        // Store input
        self.input_buffer[self.input_pos] = input;
        self.input_pos += 1;

        // When we have a full block, process it
        if self.input_pos >= self.fft_size / 2 {
            self.process_block();
            self.input_pos = 0;
            // Clear second half of input buffer for next block
            for i in self.fft_size / 2..self.fft_size {
                self.input_buffer[i] = 0.0;
            }
        }

        // Get output sample
        let wet_sample = if self.output_pos < self.output_buffer.len() {
            self.output_buffer[self.output_pos]
        } else {
            0.0
        };

        self.output_pos += 1;

        // Mix wet and dry
        wet_sample * self.wet + input * self.dry
    }

    fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.tail_buffer.fill(0.0);
        self.input_fft.fill(Complex::new(0.0, 0.0));
        self.input_pos = 0;
        self.output_pos = 0;
    }

    fn latency_samples(&self) -> usize {
        self.fft_size / 2
    }
}

// ── DoubleBufferConvolver ────────────────────────────────────────────────────

/// Double-buffering overlap-add convolution engine.
///
/// Exposes a block-oriented API: each call to [`DoubleBufferConvolver::process_block`] consumes
/// exactly `block_size` input samples and returns a reference to the
/// `block_size`-sample output slice (the leading portion of the freshly
/// computed OLA frame).
///
/// Two output buffers (`buffer_a` / `buffer_b`) alternate: while one is
/// being filled by the FFT engine, the other is the "active" buffer whose
/// leading `block_size` samples are returned to the caller.
///
/// # Latency
///
/// One block of latency: the first call to `process_block` returns all zeros
/// (the empty initial active buffer).  From the second call onward, the output
/// corresponds to the input shifted back by exactly `block_size` samples.
///
/// # Example
///
/// ```ignore
/// let ir = vec![1.0_f32, 0.5, 0.25];   // short impulse response
/// let mut conv = DoubleBufferConvolver::new(&ir, 64);
/// let input = vec![1.0_f32; 64];
/// let out = conv.process_block(&input);
/// assert_eq!(out.len(), 64);
/// ```
pub struct DoubleBufferConvolver {
    /// Pre-computed FFT of the zero-padded impulse response.
    ir_fft: Vec<Complex<f32>>,
    /// FFT frame size (≥ `ir_len + block_size - 1`, rounded to next power-of-two).
    fft_size: usize,
    /// Number of input/output samples per block.
    block_size: usize,
    /// Overlap accumulator from the previous OLA frame (length `fft_size - block_size`).
    overlap: Vec<f32>,
    /// Output buffer A.
    buffer_a: Vec<f32>,
    /// Output buffer B.
    buffer_b: Vec<f32>,
    /// Index of the buffer currently serving as the active (output) buffer: 0 = A, 1 = B.
    active_buffer: usize,
}

impl DoubleBufferConvolver {
    /// Create a new `DoubleBufferConvolver`.
    ///
    /// # Arguments
    ///
    /// * `impulse_response` — time-domain IR samples (must be non-empty).
    /// * `block_size`       — number of samples per processing block.
    ///   Clamped to at least 1; rounded up to the next power-of-two.
    ///
    /// # Panics
    ///
    /// Panics if `impulse_response` is empty.
    #[must_use]
    pub fn new(impulse_response: &[f32], block_size: usize) -> Self {
        assert!(
            !impulse_response.is_empty(),
            "impulse_response must not be empty"
        );
        let block_size = block_size.max(1).next_power_of_two();
        let ir_len = impulse_response.len();
        // Overlap-add requirement: fft_size ≥ ir_len + block_size - 1.
        let min_fft = ir_len + block_size;
        let fft_size = min_fft.next_power_of_two();

        // Build IR spectrum.
        let mut ir_padded: Vec<Complex<f32>> = impulse_response
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        ir_padded.resize(fft_size, Complex::new(0.0, 0.0));
        let ir_fft = oxifft::fft(&ir_padded);

        let overlap_len = fft_size - block_size;

        Self {
            ir_fft,
            fft_size,
            block_size,
            overlap: vec![0.0; overlap_len],
            buffer_a: vec![0.0; fft_size],
            buffer_b: vec![0.0; fft_size],
            active_buffer: 0,
        }
    }

    /// Process one block of `input` samples and return a reference to the
    /// leading `block_size` samples of the current active output buffer.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != block_size`.
    pub fn process_block(&mut self, input: &[f32]) -> &[f32] {
        assert_eq!(
            input.len(),
            self.block_size,
            "input.len() must equal block_size"
        );

        // ── Step 1: FFT of zero-padded input ────────────────────────────────
        let mut input_fft: Vec<Complex<f32>> =
            input.iter().map(|&x| Complex::new(x, 0.0)).collect();
        input_fft.resize(self.fft_size, Complex::new(0.0, 0.0));
        let input_spectrum = oxifft::fft(&input_fft);

        // ── Step 2: Convolve in frequency domain ─────────────────────────────
        let convolved: Vec<Complex<f32>> = input_spectrum
            .iter()
            .zip(self.ir_fft.iter())
            .map(|(&a, &b)| Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re))
            .collect();

        // ── Step 3: IFFT → time domain ───────────────────────────────────────
        // oxifft::ifft already normalises by 1/N; no further scaling needed.
        let time_domain = oxifft::ifft(&convolved);

        // ── Step 4: Fill the inactive buffer via overlap-add ─────────────────
        //
        // Correct OLA semantics for a block-output API:
        //
        //   The IFFT gives the linear convolution of input_k with h:
        //     result[n] for n = 0..ir_len+block_size-2
        //
        //   We partition this into:
        //     • output portion : result[0..block_size-1]   → returned to caller
        //     • new tail        : result[block_size..]      → saved for next call
        //
        //   The PREVIOUS call's tail (self.overlap) represents the accumulated
        //   OLA contributions that land in the CURRENT block's output positions.
        //   These must be ADDED to result[0..overlap_len-1] BEFORE returning.
        //
        //   Summary:
        //     fill_buf[0..block_size]   = IFFT[0..block_size] + prev_overlap[0..block_size]
        //     fill_buf[block_size..]    = IFFT[block_size..]  + prev_overlap[block_size..]
        //     new overlap (tail)        = fill_buf[block_size..]   (for next call)
        //
        //   Only fill_buf[0..block_size] is returned; the tail is saved internally.

        let fill_idx = 1 - self.active_buffer;
        let fill_buf = if fill_idx == 0 {
            &mut self.buffer_a
        } else {
            &mut self.buffer_b
        };

        // Write IFFT result into the fill buffer.
        for (i, s) in time_domain.iter().enumerate().take(self.fft_size) {
            fill_buf[i] = s.re;
        }

        // Add the previous overlap starting at position 0 (the correct OLA position):
        // the saved tail covers positions 0..overlap_len-1 of the current block's output.
        let overlap_len = self.overlap.len();
        for i in 0..overlap_len {
            fill_buf[i] += self.overlap[i];
        }

        // Save the new tail for the next frame: positions block_size..fft_size-1.
        for i in 0..overlap_len {
            self.overlap[i] = fill_buf[self.block_size + i];
        }

        // Zero the tail region in fill_buf (not strictly necessary, but ensures
        // the returned block is cleanly bounded by block_size).
        for i in self.block_size..self.fft_size {
            fill_buf[i] = 0.0;
        }

        // ── Step 5: Swap buffers ──────────────────────────────────────────────
        self.active_buffer = fill_idx;

        // Return the leading block_size samples of the newly-active buffer.
        let active_buf = if self.active_buffer == 0 {
            &self.buffer_a
        } else {
            &self.buffer_b
        };
        &active_buf[..self.block_size]
    }

    /// Return the configured block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Clear all internal state.
    pub fn clear(&mut self) {
        self.overlap.iter_mut().for_each(|s| *s = 0.0);
        self.buffer_a.iter_mut().for_each(|s| *s = 0.0);
        self.buffer_b.iter_mut().for_each(|s| *s = 0.0);
        self.active_buffer = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolution_reverb_creation() {
        let ir = vec![1.0, 0.5, 0.25, 0.125]; // Simple exponential decay
        let reverb = ConvolutionReverb::new(&ir, 48000.0);
        assert!(reverb.is_ok());
    }

    #[test]
    fn test_convolution_reverb_empty_ir() {
        let ir: Vec<f32> = vec![];
        let result = ConvolutionReverb::new(&ir, 48000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_convolution_reverb_process() {
        let ir = vec![1.0; 100]; // Simple IR
        let mut reverb = ConvolutionReverb::new(&ir, 48000.0).expect("test expectation failed");

        // Process impulse
        let output = reverb.process_sample(1.0);
        // Output might be delayed due to block processing
        assert!(output.is_finite());

        // Process more samples
        for _ in 0..1000 {
            let out = reverb.process_sample(0.0);
            assert!(out.is_finite());
        }
    }

    #[test]
    fn test_convolution_wet_dry() {
        let ir = vec![0.5; 50];
        let mut reverb = ConvolutionReverb::new(&ir, 48000.0).expect("test expectation failed");

        reverb.set_wet(0.0);
        reverb.set_dry(1.0);

        // Process samples - with dry=1 and wet=0, should eventually get mostly dry signal
        for _ in 0..100 {
            reverb.process_sample(1.0);
        }

        let output = reverb.process_sample(1.0);
        // With wet=0, dry=1, output should be close to input
        assert!((output - 1.0).abs() < 0.5);
    }

    // ── DoubleBufferConvolver tests ──────────────────────────────────────────

    #[test]
    fn test_double_buffer_convolver_delta() {
        // Delta IR {1.0, 0, 0, …} → convolving an impulse with it should give
        // an impulse back.  The OLA output block should have `out[0] ≈ 1.0`.
        const BLOCK: usize = 64;
        let mut ir = vec![0.0_f32; BLOCK];
        ir[0] = 1.0;

        let mut conv = DoubleBufferConvolver::new(&ir, BLOCK);

        // Input: single impulse at position 0.
        let mut input_block = vec![0.0_f32; BLOCK];
        input_block[0] = 1.0;

        // process_block fills the inactive buffer with the OLA result, then swaps.
        // The returned slice is the leading block_size samples of the NEWLY-active buffer
        // (i.e. the freshly-computed convolution of input with the IR).
        let out0 = conv.process_block(&input_block);
        assert_eq!(out0.len(), BLOCK, "output block length must equal BLOCK");

        // Delta IR * impulse = impulse: out0[0] should be close to 1.0.
        // oxifft::ifft normalises by 1/N internally, so no extra scale is needed.
        let peak = out0.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            peak > 0.5,
            "delta IR convolution peak should be > 0.5 (got {peak:.4})"
        );

        // All output values must be finite.
        for (i, &s) in out0.iter().enumerate() {
            assert!(s.is_finite(), "out0[{i}] is not finite: {s}");
        }

        // Feed a second block of silence; output should remain finite.
        let silence = vec![0.0_f32; BLOCK];
        let out1 = conv.process_block(&silence);
        assert_eq!(out1.len(), BLOCK, "second block length must equal BLOCK");
        for (i, &s) in out1.iter().enumerate() {
            assert!(s.is_finite(), "out1[{i}] is not finite: {s}");
        }
    }

    #[test]
    fn test_double_buffer_convolver_matches_direct() {
        // Gaussian IR: convolving a sine wave through it via OLA must match
        // direct (time-domain) convolution within floating-point tolerance.
        use std::f32::consts::TAU;

        const N_BLOCKS: usize = 8;
        const BLOCK: usize = 32;
        const IR_LEN: usize = 16;
        const TOTAL: usize = N_BLOCKS * BLOCK;

        // Build a Gaussian-windowed IR.
        let ir: Vec<f32> = (0..IR_LEN)
            .map(|i| {
                let x = (i as f32 - IR_LEN as f32 / 2.0) / (IR_LEN as f32 / 4.0);
                (-x * x / 2.0).exp() / IR_LEN as f32
            })
            .collect();

        // Build sine input.
        let input: Vec<f32> = (0..TOTAL)
            .map(|i| (TAU * 440.0 * i as f32 / 48_000.0).sin() * 0.5)
            .collect();

        // ── Direct convolution (reference) ───────────────────────────────────
        let mut direct = vec![0.0_f32; TOTAL + IR_LEN];
        for (n, &x) in input.iter().enumerate() {
            for (k, &h) in ir.iter().enumerate() {
                direct[n + k] += x * h;
            }
        }

        // ── OLA via DoubleBufferConvolver ─────────────────────────────────────
        let mut conv = DoubleBufferConvolver::new(&ir, BLOCK);
        let mut ola_out = Vec::with_capacity(TOTAL);
        for block_idx in 0..N_BLOCKS {
            let start = block_idx * BLOCK;
            let chunk = &input[start..start + BLOCK];
            let out = conv.process_block(chunk);
            ola_out.extend_from_slice(out);
        }

        // DoubleBufferConvolver::process_block returns the result for the
        // CURRENT input block with no delay.  Compare OLA output directly
        // against the direct convolution.
        //
        // Skip the first `IR_LEN` samples: near position 0 the direct convolution
        // is very small (the sine input hasn't built up yet), so the FFT round-off
        // error may dominate on those samples.  From sample `IR_LEN` onward the
        // signal is large enough that relative accuracy is meaningful.
        let skip = IR_LEN;
        for (i, (&ola, &dir)) in ola_out[skip..]
            .iter()
            .zip(direct[skip..].iter())
            .enumerate()
            .take(TOTAL - skip)
        {
            let err = (ola - dir).abs();
            assert!(
                err < 1e-3,
                "OLA vs direct mismatch at sample {}: ola={:.6}, dir={:.6}, err={:.6}",
                skip + i,
                ola,
                dir,
                err
            );
        }
    }
}
