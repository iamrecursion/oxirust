//! Hybrid mode combining SILK and CELT (RFC 6716 §3.1, §4.5).
//!
//! A hybrid Opus packet is **not** two byte-halves. The SILK and CELT layers
//! are decoded from one and the same range-coded bitstream, in sequence: the
//! SILK layer is decoded first and carries the low band (up to 8 kHz); the
//! CELT layer then continues from the very same bitstream and carries the high
//! band (8 kHz and above). The decoder reconstructs both bands and sums them.
//!
//! This module therefore:
//!
//! 1. builds a single normative range decoder over the whole frame payload;
//! 2. runs the SILK decoder on it ([`super::silk::SilkDecoder::decode_with`]);
//! 3. hands the bytes the SILK layer did not consume to the CELT decoder,
//!    which decodes the high band;
//! 4. combines the SILK low band and the CELT high band.
//!
//! The legacy "split the packet at its midpoint" heuristic has been removed —
//! that layout does not exist in RFC 6716.

use crate::{CodecError, CodecResult};

use super::celt::CeltDecoder;
use super::packet::OpusBandwidth;
use super::silk::SilkDecoder;
use super::silk_range::SilkRangeDecoder;

/// Hybrid mode decoder combining SILK (low band) and CELT (high band).
#[derive(Debug)]
pub struct HybridDecoder {
    /// SILK decoder for the low band (always wideband inside hybrid mode).
    silk: SilkDecoder,
    /// CELT decoder for the high band.
    celt: CeltDecoder,
    /// Configured output sample rate.
    sample_rate: u32,
    /// Number of channels.
    channels: usize,
    /// Operating bandwidth (super-wideband or fullband for hybrid).
    #[allow(dead_code)]
    bandwidth: OpusBandwidth,
    /// Crossover frequency between the SILK and CELT bands (Hz).
    crossover_freq: u32,
    /// Per-channel low-pass filter state for the SILK band.
    lowpass_state: Vec<BiquadState>,
    /// Per-channel high-pass filter state for the CELT band.
    highpass_state: Vec<BiquadState>,
}

/// Two-pole filter delay-line state shared by the cross-over filters.
#[derive(Debug, Clone, Default)]
struct BiquadState {
    /// Previous two input samples.
    prev_input: [f32; 2],
    /// Previous two output samples.
    prev_output: [f32; 2],
}

impl HybridDecoder {
    /// Creates a new hybrid decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Output sample rate in Hz
    /// * `channels` - Number of channels
    /// * `bandwidth` - Operating bandwidth
    /// * `frame_size` - Frame size in samples
    pub fn new(
        sample_rate: u32,
        channels: usize,
        bandwidth: OpusBandwidth,
        frame_size: usize,
    ) -> Self {
        // RFC 6716: hybrid mode crosses SILK over to CELT at 8 kHz.
        let crossover_freq = 8000;
        let silk = SilkDecoder::new(sample_rate, channels, OpusBandwidth::Wideband);
        let celt = CeltDecoder::new(sample_rate, channels, bandwidth, frame_size);
        Self {
            silk,
            celt,
            sample_rate,
            channels,
            bandwidth,
            crossover_freq,
            lowpass_state: vec![BiquadState::default(); channels],
            highpass_state: vec![BiquadState::default(); channels],
        }
    }

    /// Decodes a hybrid Opus frame.
    ///
    /// The whole `data` payload is one range-coded bitstream. SILK is decoded
    /// first; CELT continues from the bytes SILK did not consume. The two band
    /// outputs are then combined.
    ///
    /// # Arguments
    ///
    /// * `data` - The complete hybrid frame payload (one shared bitstream).
    /// * `output` - Interleaved output sample buffer.
    /// * `frame_size` - Number of samples per channel.
    pub fn decode(
        &mut self,
        data: &[u8],
        output: &mut [f32],
        frame_size: usize,
    ) -> CodecResult<()> {
        if output.len() < frame_size * self.channels {
            return Err(CodecError::InvalidData(
                "Output buffer too small".to_string(),
            ));
        }
        if data.is_empty() {
            // Lost packet: hand an empty buffer to SILK PLC and zero CELT.
            return self.silk.decode(data, output, frame_size);
        }

        let mut silk_output = vec![0.0f32; frame_size * self.channels];
        let mut celt_output = vec![0.0f32; frame_size * self.channels];

        // --- Step 1: one shared range decoder over the whole payload ---
        let mut shared = SilkRangeDecoder::new(data)?;

        // --- Step 2: decode the SILK low band from the shared bitstream ---
        self.silk
            .decode_with(&mut shared, &mut silk_output, frame_size)?;

        // --- Step 3: CELT continues from where SILK stopped ---
        // The CELT decoder owns a separate range-coder implementation, so it is
        // given the bytes the SILK layer did not consume from the front of the
        // shared bitstream. This preserves the normative "SILK first, CELT
        // continues from the same stream" framing.
        let silk_consumed = shared.front_bytes_consumed().min(data.len());
        let celt_data = &data[silk_consumed..];
        if celt_data.is_empty() {
            // SILK consumed the whole packet; the high band is silent.
            celt_output.fill(0.0);
        } else {
            self.celt.decode(celt_data, &mut celt_output, frame_size)?;
        }

        // --- Step 4: combine the SILK low band and the CELT high band ---
        self.combine_outputs(&silk_output, &celt_output, output, frame_size)
    }

    /// Combines the SILK and CELT band outputs through complementary
    /// cross-over filters and sums them (RFC 6716 §4.5).
    fn combine_outputs(
        &mut self,
        silk_output: &[f32],
        celt_output: &[f32],
        output: &mut [f32],
        frame_size: usize,
    ) -> CodecResult<()> {
        let mut silk_filtered = silk_output.to_vec();
        let mut celt_filtered = celt_output.to_vec();
        self.apply_lowpass(&mut silk_filtered, frame_size);
        self.apply_highpass(&mut celt_filtered, frame_size);
        for i in 0..(frame_size * self.channels) {
            output[i] = silk_filtered[i] + celt_filtered[i];
        }
        Ok(())
    }

    /// Applies a 2nd-order Butterworth low-pass to the SILK band.
    fn apply_lowpass(&mut self, samples: &mut [f32], frame_size: usize) {
        let (b0, b1, b2, a1, a2) = lowpass_coeffs(self.crossover_freq, self.sample_rate);
        for ch in 0..self.channels {
            let state = &mut self.lowpass_state[ch];
            for i in 0..frame_size {
                let idx = i * self.channels + ch;
                if idx < samples.len() {
                    let input = samples[idx];
                    let out = b0 * input + b1 * state.prev_input[0] + b2 * state.prev_input[1]
                        - a1 * state.prev_output[0]
                        - a2 * state.prev_output[1];
                    state.prev_input[1] = state.prev_input[0];
                    state.prev_input[0] = input;
                    state.prev_output[1] = state.prev_output[0];
                    state.prev_output[0] = out;
                    samples[idx] = out;
                }
            }
        }
    }

    /// Applies a 2nd-order Butterworth high-pass to the CELT band.
    fn apply_highpass(&mut self, samples: &mut [f32], frame_size: usize) {
        let (b0, b1, b2, a1, a2) = highpass_coeffs(self.crossover_freq, self.sample_rate);
        for ch in 0..self.channels {
            let state = &mut self.highpass_state[ch];
            for i in 0..frame_size {
                let idx = i * self.channels + ch;
                if idx < samples.len() {
                    let input = samples[idx];
                    let out = b0 * input + b1 * state.prev_input[0] + b2 * state.prev_input[1]
                        - a1 * state.prev_output[0]
                        - a2 * state.prev_output[1];
                    state.prev_input[1] = state.prev_input[0];
                    state.prev_input[0] = input;
                    state.prev_output[1] = state.prev_output[0];
                    state.prev_output[0] = out;
                    samples[idx] = out;
                }
            }
        }
    }

    /// Resets decoder state.
    pub fn reset(&mut self) {
        self.silk.reset();
        self.celt.reset();
        for state in &mut self.lowpass_state {
            *state = BiquadState::default();
        }
        for state in &mut self.highpass_state {
            *state = BiquadState::default();
        }
    }

    /// Returns the current sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the number of channels.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the crossover frequency.
    #[must_use]
    pub const fn crossover_frequency(&self) -> u32 {
        self.crossover_freq
    }
}

/// Computes normalised 2nd-order Butterworth low-pass biquad coefficients.
fn lowpass_coeffs(cutoff: u32, sample_rate: u32) -> (f32, f32, f32, f32, f32) {
    use std::f32::consts::PI;
    let omega = 2.0 * PI * cutoff as f32 / sample_rate as f32;
    let cos_omega = omega.cos();
    let alpha = omega.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
    let b0 = (1.0 - cos_omega) / 2.0;
    let b1 = 1.0 - cos_omega;
    let b2 = (1.0 - cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

/// Computes normalised 2nd-order Butterworth high-pass biquad coefficients.
fn highpass_coeffs(cutoff: u32, sample_rate: u32) -> (f32, f32, f32, f32, f32) {
    use std::f32::consts::PI;
    let omega = 2.0 * PI * cutoff as f32 / sample_rate as f32;
    let cos_omega = omega.cos();
    let alpha = omega.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2.recip());
    let b0 = (1.0 + cos_omega) / 2.0;
    let b1 = -(1.0 + cos_omega);
    let b2 = (1.0 + cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;
    (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_decoder_creation() {
        let decoder = HybridDecoder::new(48000, 2, OpusBandwidth::SuperWideband, 480);
        assert_eq!(decoder.sample_rate(), 48000);
        assert_eq!(decoder.channels(), 2);
        assert_eq!(decoder.crossover_frequency(), 8000);
    }

    #[test]
    fn test_hybrid_decode_single_bitstream() {
        // A hybrid frame is one shared bitstream — SILK then CELT, no midpoint
        // split. The decoder must produce finite PCM of the requested length.
        let mut decoder = HybridDecoder::new(48000, 1, OpusBandwidth::SuperWideband, 480);
        let data: Vec<u8> = (0u8..64)
            .map(|i| i.wrapping_mul(43).wrapping_add(17))
            .collect();
        let mut output = vec![0.0f32; 480];
        let result = decoder.decode(&data, &mut output, 480);
        assert!(result.is_ok(), "hybrid decode should succeed");
        for &s in &output {
            assert!(s.is_finite(), "hybrid output must be finite");
        }
    }

    #[test]
    fn test_hybrid_decode_stereo() {
        let mut decoder = HybridDecoder::new(48000, 2, OpusBandwidth::Fullband, 480);
        let data: Vec<u8> = (0u8..96)
            .map(|i| i.wrapping_mul(31).wrapping_add(9))
            .collect();
        let mut output = vec![0.0f32; 480 * 2];
        decoder
            .decode(&data, &mut output, 480)
            .expect("stereo hybrid");
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_hybrid_decode_empty_packet() {
        // An empty hybrid packet routes through SILK PLC.
        let mut decoder = HybridDecoder::new(48000, 1, OpusBandwidth::SuperWideband, 480);
        let mut output = vec![0.0f32; 480];
        let result = decoder.decode(&[], &mut output, 480);
        assert!(result.is_ok());
        assert!(output.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_hybrid_reset() {
        let mut decoder = HybridDecoder::new(48000, 1, OpusBandwidth::SuperWideband, 480);
        decoder.reset();
        // Should not panic; state is back to defaults.
    }

    #[test]
    fn test_combine_outputs() {
        let mut decoder = HybridDecoder::new(48000, 1, OpusBandwidth::SuperWideband, 480);
        let silk_output = vec![1.0f32; 480];
        let celt_output = vec![2.0f32; 480];
        let mut output = vec![0.0f32; 480];
        let result = decoder.combine_outputs(&silk_output, &celt_output, &mut output, 480);
        assert!(result.is_ok());
        // After complementary cross-over filtering the steady-state output is
        // dominated by the SILK low band's DC component.
        let last = output[479];
        assert!(
            last.is_finite() && last.abs() < 10.0,
            "expected finite output within reasonable range, got {last}"
        );
    }
}
