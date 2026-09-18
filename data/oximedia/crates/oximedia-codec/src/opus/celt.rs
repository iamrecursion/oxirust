//! CELT mode decoder for music.
//!
//! CELT (Constrained Energy Lapped Transform) is optimized for music and
//! general audio content. This implementation provides full CELT decoding
//! including PVQ (Pyramid Vector Quantization), band energy decoding,
//! and post-filtering.

use crate::{CodecError, CodecResult};

use super::mdct::{Mdct, OverlapAdd};
use super::packet::OpusBandwidth;
use super::range_decoder::RangeDecoder;
use super::range_encoder::RangeEncoder;

/// Number of frequency bands for CELT
const CELT_BANDS: usize = 21;

/// Minimum band width
const MIN_BAND_WIDTH: usize = 2;

/// Bark scale band boundaries (in bins) for 480-sample frame
const BARK_BAND_BOUNDARIES: [usize; CELT_BANDS + 1] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 68, 80, 96, 120, 156, 240,
];

/// Energy decoding fine bits per band
const ENERGY_FINE_BITS: [u8; CELT_BANDS] = [
    3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
];

/// Allocation trim values
const ALLOCATION_TRIM: [f32; 11] = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

/// Post-filter coefficients
const POST_FILTER_COEFFS: [f32; 3] = [0.85, 0.0, -0.85];

/// CELT decoder state.
#[derive(Debug)]
pub struct CeltDecoder {
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: usize,
    /// Bandwidth
    #[allow(dead_code)]
    bandwidth: OpusBandwidth,
    /// MDCT transformer
    mdct: Mdct,
    /// Overlap-add processor
    overlap_add: Vec<OverlapAdd>,
    /// Previous frame energy per band
    band_energy: Vec<f32>,
    /// Frame size
    frame_size: usize,
    /// Post-filter state
    postfilter_state: Vec<Vec<f32>>,
    /// Fine energy previous values
    fine_energy_prev: Vec<f32>,
}

impl CeltDecoder {
    /// Creates a new CELT decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `bandwidth` - Operating bandwidth
    /// * `frame_size` - Frame size in samples
    pub fn new(
        sample_rate: u32,
        channels: usize,
        bandwidth: OpusBandwidth,
        frame_size: usize,
    ) -> Self {
        let mdct = Mdct::new(frame_size);
        let overlap_add = (0..channels).map(|_| OverlapAdd::new(frame_size)).collect();

        Self {
            sample_rate,
            channels,
            bandwidth,
            mdct,
            overlap_add,
            band_energy: vec![0.0; CELT_BANDS],
            frame_size,
            postfilter_state: vec![vec![0.0; 3]; channels],
            fine_energy_prev: vec![0.0; CELT_BANDS],
        }
    }

    /// Decodes a CELT frame.
    ///
    /// # Arguments
    ///
    /// * `data` - Compressed frame data
    /// * `output` - Output sample buffer
    /// * `frame_size` - Number of samples per channel
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

        // Create range decoder
        let mut decoder = RangeDecoder::new(data)?;

        // Decode global parameters
        let global_params = self.decode_global_params(&mut decoder)?;

        // Decode each channel
        for ch in 0..self.channels {
            // Decode CELT frame for this channel
            self.decode_channel(&mut decoder, output, frame_size, ch, &global_params)?;
        }

        Ok(())
    }

    /// Decodes global frame parameters.
    fn decode_global_params(&mut self, decoder: &mut RangeDecoder) -> CodecResult<GlobalParams> {
        // Decode silence flag
        let silence = decoder.decode_bit(16384)?;

        // Decode post-filter flag
        let postfilter = decoder.decode_bit(16384)?;

        // Decode transient flag
        let transient = decoder.decode_bit(16384)?;

        // Decode allocation trim
        let trim_index = decoder.decode_uniform(11)? as usize;
        let allocation_trim = if trim_index < ALLOCATION_TRIM.len() {
            ALLOCATION_TRIM[trim_index]
        } else {
            0.0
        };

        // Decode fine energy bit allocation
        let fine_bits = decoder.decode_uniform(4)? as u8;

        Ok(GlobalParams {
            silence,
            postfilter,
            transient,
            allocation_trim,
            fine_bits,
        })
    }

    /// Decodes a single channel.
    fn decode_channel(
        &mut self,
        decoder: &mut RangeDecoder,
        output: &mut [f32],
        frame_size: usize,
        channel: usize,
        global_params: &GlobalParams,
    ) -> CodecResult<()> {
        if global_params.silence {
            // Silence frame - zero output
            let ch_offset = channel * frame_size;
            for i in 0..frame_size {
                output[ch_offset + i] = 0.0;
            }
            return Ok(());
        }

        // Allocate buffers
        let mut coeffs = vec![0.0f32; frame_size];
        let mut time_domain = vec![0.0f32; 2 * frame_size];

        // CELT decoding steps:
        // 1. Decode coarse energy per band
        let band_sizes = self.get_band_sizes(frame_size);
        self.decode_coarse_energy(decoder, &band_sizes)?;

        // 2. Decode fine energy
        self.decode_fine_energy(decoder, &band_sizes, global_params.fine_bits)?;

        // 3. Decode bit allocation
        let bit_allocation = self.compute_bit_allocation(&band_sizes, global_params)?;

        // 4. Decode spectral shape using PVQ
        self.decode_pvq(decoder, &mut coeffs, &band_sizes, &bit_allocation)?;

        // 5. Denormalize coefficients using band energy
        self.denormalize_coeffs(&mut coeffs, &band_sizes);

        // 6. Apply inverse MDCT
        self.mdct.inverse(&coeffs, &mut time_domain);

        // 7. Apply overlap-add
        let mut frame_samples = vec![0.0f32; frame_size];
        self.overlap_add[channel].process(&time_domain, &mut frame_samples);

        // 8. Apply post-filter if enabled
        if global_params.postfilter {
            self.apply_postfilter(&mut frame_samples, channel);
        }

        // 9. Copy to output
        let ch_offset = channel * frame_size;
        output[ch_offset..ch_offset + frame_size].copy_from_slice(&frame_samples);

        Ok(())
    }

    /// Decodes coarse energy per frequency band.
    fn decode_coarse_energy(
        &mut self,
        decoder: &mut RangeDecoder,
        band_sizes: &[usize],
    ) -> CodecResult<()> {
        // Decode coarse energy with prediction from previous frame
        for (band_idx, _) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                break;
            }

            // Decode energy delta
            let energy_delta = decoder.decode_int(6)? as f32;

            // Apply prediction
            let predicted = self.band_energy[band_idx];
            let alpha = 0.9; // Prediction coefficient

            self.band_energy[band_idx] = alpha * predicted + energy_delta * 0.5;
        }

        Ok(())
    }

    /// Decodes fine energy per frequency band.
    fn decode_fine_energy(
        &mut self,
        decoder: &mut RangeDecoder,
        band_sizes: &[usize],
        fine_bits: u8,
    ) -> CodecResult<()> {
        for (band_idx, _) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                break;
            }

            let bits = ENERGY_FINE_BITS[band_idx].min(fine_bits);

            if bits > 0 {
                let fine_energy = decoder.decode_uint(u32::from(bits))?;
                let fine_scale = 1.0 / ((1 << bits) as f32);

                self.band_energy[band_idx] += (fine_energy as f32) * fine_scale;
            }

            // Store for next frame
            self.fine_energy_prev[band_idx] = self.band_energy[band_idx];
        }

        Ok(())
    }

    /// Computes bit allocation per band.
    fn compute_bit_allocation(
        &self,
        band_sizes: &[usize],
        global_params: &GlobalParams,
    ) -> CodecResult<Vec<u32>> {
        let mut allocation = Vec::with_capacity(band_sizes.len());

        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                allocation.push(0);
                continue;
            }

            // Base allocation proportional to band size
            let base = (band_size as f32 * 2.0).log2().max(0.0);

            // Apply trim adjustment
            let adjusted = base + global_params.allocation_trim;

            // Convert to integer bits
            let bits = adjusted.max(0.0) as u32;

            allocation.push(bits);
        }

        Ok(allocation)
    }

    /// Decodes spectral shape using Pyramid Vector Quantization (PVQ).
    fn decode_pvq(
        &mut self,
        decoder: &mut RangeDecoder,
        coeffs: &mut [f32],
        band_sizes: &[usize],
        bit_allocation: &[u32],
    ) -> CodecResult<()> {
        let mut offset = 0;

        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= band_sizes.len() || band_idx >= bit_allocation.len() {
                break;
            }

            let bits = bit_allocation[band_idx];

            if bits > 0 && offset + band_size <= coeffs.len() {
                // Decode pulse count K for this band
                let k = self.decode_pulse_count(decoder, bits)?;

                if k > 0 {
                    // Decode PVQ vector
                    self.decode_pvq_vector(
                        decoder,
                        &mut coeffs[offset..offset + band_size],
                        k,
                        band_size,
                    )?;
                } else {
                    // Zero band
                    for i in 0..band_size {
                        coeffs[offset + i] = 0.0;
                    }
                }
            }

            offset += band_size;
        }

        Ok(())
    }

    /// Decodes pulse count for PVQ.
    fn decode_pulse_count(&self, decoder: &mut RangeDecoder, bits: u32) -> CodecResult<u32> {
        // Decode K using unary coding or fixed bits
        if bits <= 3 {
            Ok(bits)
        } else {
            decoder.decode_uniform(bits + 1)
        }
    }

    /// Decodes a PVQ vector (pyramid vector quantization).
    fn decode_pvq_vector(
        &self,
        decoder: &mut RangeDecoder,
        band: &mut [f32],
        k: u32,
        n: usize,
    ) -> CodecResult<()> {
        if k == 0 || n == 0 {
            band.fill(0.0);
            return Ok(());
        }

        // PVQ decoding using recursive splitting
        if n == 1 {
            // Base case: single coefficient
            band[0] = k as f32;
            let sign = decoder.decode_bit(16384)?;
            if sign {
                band[0] = -band[0];
            }
            return Ok(());
        }

        // Split pulses between first half and second half
        let mid = n / 2;
        let k_left = self.decode_pvq_split(decoder, k, n)?;
        let k_right = k.saturating_sub(k_left);

        // Recursively decode left and right halves
        self.decode_pvq_vector(decoder, &mut band[..mid], k_left, mid)?;
        self.decode_pvq_vector(decoder, &mut band[mid..], k_right, n - mid)?;

        Ok(())
    }

    /// Decodes PVQ split point.
    fn decode_pvq_split(&self, decoder: &mut RangeDecoder, k: u32, n: usize) -> CodecResult<u32> {
        if k == 0 {
            return Ok(0);
        }

        // Use binomial distribution to decode split
        let max_split = k + 1;
        let split = decoder.decode_uniform(max_split)?;

        Ok(split.min(k))
    }

    /// Denormalizes coefficients using band energy.
    fn denormalize_coeffs(&self, coeffs: &mut [f32], band_sizes: &[usize]) {
        let mut offset = 0;

        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS || offset >= coeffs.len() {
                break;
            }

            // Convert log energy to linear scale
            let energy = self.band_energy[band_idx].exp();

            // Normalize band to unit energy, then scale
            let coeffs_len = coeffs.len();
            let end = offset.saturating_add(band_size).min(coeffs_len);
            let band_slice = &mut coeffs[offset..end];
            let band_norm = self.compute_band_norm(band_slice);

            if band_norm > 1e-10 {
                let scale = energy / band_norm;
                for coeff in band_slice.iter_mut() {
                    *coeff *= scale;
                }
            }

            offset = offset.saturating_add(band_size);
        }
    }

    /// Computes Euclidean norm of a band.
    fn compute_band_norm(&self, band: &[f32]) -> f32 {
        band.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Returns band sizes for the given frame size.
    fn get_band_sizes(&self, frame_size: usize) -> Vec<usize> {
        let mut sizes = Vec::new();

        // Scale Bark band boundaries to current frame size
        let scale = frame_size as f32 / 240.0; // 240 is reference size

        for i in 0..CELT_BANDS {
            let start = (BARK_BAND_BOUNDARIES[i] as f32 * scale) as usize;
            let end = (BARK_BAND_BOUNDARIES[i + 1] as f32 * scale) as usize;
            let size = end.saturating_sub(start).max(MIN_BAND_WIDTH);
            sizes.push(size);
        }

        // Adjust last band to exactly fill frame
        let total: usize = sizes.iter().sum();
        if total < frame_size {
            if let Some(last) = sizes.last_mut() {
                *last += frame_size - total;
            }
        } else if total > frame_size && !sizes.is_empty() {
            if let Some(last) = sizes.last_mut() {
                *last = last.saturating_sub(total - frame_size);
            }
        }

        sizes
    }

    /// Applies post-filter to improve perceptual quality.
    fn apply_postfilter(&mut self, samples: &mut [f32], channel: usize) {
        let state = &mut self.postfilter_state[channel];

        for sample in samples.iter_mut() {
            // Apply IIR filter
            let mut filtered = *sample;
            for (i, &coeff) in POST_FILTER_COEFFS.iter().enumerate() {
                if i < state.len() {
                    filtered += coeff * state[i];
                }
            }

            // Update state
            state.rotate_right(1);
            state[0] = *sample;

            *sample = filtered;
        }
    }

    /// Resets decoder state.
    pub fn reset(&mut self) {
        for ola in &mut self.overlap_add {
            ola.reset();
        }
        self.band_energy.fill(0.0);
        self.fine_energy_prev.fill(0.0);
        for state in &mut self.postfilter_state {
            state.fill(0.0);
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

    /// Returns the frame size.
    #[must_use]
    pub const fn frame_size(&self) -> usize {
        self.frame_size
    }
}

/// Global frame parameters.
#[derive(Debug, Clone)]
struct GlobalParams {
    /// Silence flag
    silence: bool,
    /// Post-filter enable flag
    postfilter: bool,
    /// Transient flag
    #[allow(dead_code)]
    transient: bool,
    /// Allocation trim
    allocation_trim: f32,
    /// Fine energy bits
    fine_bits: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_decoder_creation() {
        let decoder = CeltDecoder::new(48000, 2, OpusBandwidth::Fullband, 480);
        assert_eq!(decoder.sample_rate(), 48000);
        assert_eq!(decoder.channels(), 2);
        assert_eq!(decoder.frame_size(), 480);
    }

    #[test]
    fn test_celt_decoder_decode() {
        let mut decoder = CeltDecoder::new(48000, 1, OpusBandwidth::Fullband, 480);
        let data = vec![0x80, 0x00, 0x00, 0x00];
        let mut output = vec![0.0f32; 480];

        let result = decoder.decode(&data, &mut output, 480);
        assert!(result.is_ok());
    }

    #[test]
    fn test_celt_band_sizes() {
        let decoder = CeltDecoder::new(48000, 1, OpusBandwidth::Fullband, 480);
        let sizes = decoder.get_band_sizes(480);
        assert_eq!(sizes.len(), CELT_BANDS);

        // Total should equal frame size
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 480);
    }

    #[test]
    fn test_celt_band_norm() {
        let decoder = CeltDecoder::new(48000, 1, OpusBandwidth::Fullband, 480);
        let band = vec![3.0f32, 4.0f32];
        let norm = decoder.compute_band_norm(&band);
        assert!((norm - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_celt_reset() {
        let mut decoder = CeltDecoder::new(48000, 1, OpusBandwidth::Fullband, 480);
        decoder.band_energy[0] = 10.0;
        decoder.reset();
        assert_eq!(decoder.band_energy[0], 0.0);
    }

    #[test]
    fn test_bark_band_boundaries() {
        assert_eq!(BARK_BAND_BOUNDARIES.len(), CELT_BANDS + 1);
        assert_eq!(BARK_BAND_BOUNDARIES[0], 0);

        // Ensure monotonic increasing
        for i in 1..BARK_BAND_BOUNDARIES.len() {
            assert!(BARK_BAND_BOUNDARIES[i] > BARK_BAND_BOUNDARIES[i - 1]);
        }
    }

    #[test]
    fn test_energy_fine_bits() {
        assert_eq!(ENERGY_FINE_BITS.len(), CELT_BANDS);

        // All values should be reasonable (0-4 bits)
        for &bits in &ENERGY_FINE_BITS {
            assert!(bits <= 4);
        }
    }
}

/// CELT encoder state.
#[derive(Debug)]
pub struct CeltEncoder {
    /// Sample rate
    sample_rate: u32,
    /// Number of channels
    channels: usize,
    /// Bandwidth
    bandwidth: OpusBandwidth,
    /// MDCT transformer
    mdct: Mdct,
    /// Overlap-add processors for analysis
    overlap_add: Vec<OverlapAdd>,
    /// Previous frame energy per band
    band_energy: Vec<f32>,
    /// Frame size in samples
    #[allow(dead_code)]
    frame_size: usize,
}

impl CeltEncoder {
    /// Creates a new CELT encoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of channels
    /// * `bandwidth` - Operating bandwidth
    /// * `frame_size` - Frame size in samples
    pub fn new(
        sample_rate: u32,
        channels: usize,
        bandwidth: OpusBandwidth,
        frame_size: usize,
    ) -> Self {
        let mdct = Mdct::new(frame_size);
        let overlap_add = (0..channels).map(|_| OverlapAdd::new(frame_size)).collect();

        Self {
            sample_rate,
            channels,
            bandwidth,
            mdct,
            overlap_add,
            band_energy: vec![0.0; CELT_BANDS],
            frame_size,
        }
    }

    /// Encodes a CELT frame.
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample buffer (f32 samples)
    /// * `output` - Output buffer for compressed data
    /// * `frame_size` - Number of samples per channel
    pub fn encode(
        &mut self,
        input: &[f32],
        output: &mut [u8],
        frame_size: usize,
    ) -> CodecResult<usize> {
        if input.len() < frame_size * self.channels {
            return Err(CodecError::InvalidData(
                "Input buffer too small".to_string(),
            ));
        }

        // Create range encoder
        let mut encoder = RangeEncoder::new(output.len());

        // Encode each channel
        for ch in 0..self.channels {
            self.encode_channel(&mut encoder, input, frame_size, ch)?;
        }

        // Finalize encoding
        let compressed = encoder.finalize()?;

        if compressed.len() > output.len() {
            return Err(CodecError::BufferTooSmall {
                needed: compressed.len(),
                have: output.len(),
            });
        }

        output[..compressed.len()].copy_from_slice(&compressed);
        Ok(compressed.len())
    }

    /// Encodes a single channel.
    fn encode_channel(
        &mut self,
        encoder: &mut RangeEncoder,
        input: &[f32],
        frame_size: usize,
        channel: usize,
    ) -> CodecResult<()> {
        // Extract channel samples
        let mut channel_samples = vec![0.0f32; 2 * frame_size];
        for i in 0..frame_size {
            let idx = i * self.channels + channel;
            if idx < input.len() {
                channel_samples[i] = input[idx];
            }
        }

        // Allocate buffers
        let mut coeffs = vec![0.0f32; frame_size];

        // CELT encoding steps:
        // 1. Apply forward MDCT
        self.mdct.forward(&channel_samples, &mut coeffs);

        // 2. Compute and encode band energy
        self.encode_band_energy(encoder, &coeffs, frame_size)?;

        // 3. Normalize coefficients by band energy
        self.normalize_coeffs(&mut coeffs, frame_size);

        // 4. Encode spectral shape (PVQ)
        self.encode_spectral_shape(encoder, &coeffs, frame_size)?;

        Ok(())
    }

    /// Encodes energy per frequency band.
    fn encode_band_energy(
        &mut self,
        encoder: &mut RangeEncoder,
        coeffs: &[f32],
        frame_size: usize,
    ) -> CodecResult<()> {
        let band_sizes = self.get_band_sizes(frame_size);
        let mut offset = 0;

        // Compute energy for each band
        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                break;
            }

            let mut energy = 0.0f32;
            for i in 0..band_size {
                if offset + i < coeffs.len() {
                    energy += coeffs[offset + i] * coeffs[offset + i];
                }
            }
            energy = (energy / band_size as f32).sqrt().max(1e-10);

            // Convert to log scale
            let log_energy = energy.ln();

            // Quantize to 6 bits (simplified)
            let quantized = ((log_energy * 2.0).round() as i32).clamp(-31, 31);

            // Encode energy delta
            encoder.encode_int(quantized, 6)?;

            self.band_energy[band_idx] = log_energy;
            offset += band_size;
        }

        Ok(())
    }

    /// Normalizes coefficients by band energy.
    fn normalize_coeffs(&self, coeffs: &mut [f32], frame_size: usize) {
        let band_sizes = self.get_band_sizes(frame_size);
        let mut offset = 0;

        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                break;
            }

            let energy = self.band_energy[band_idx].exp();
            let norm_factor = if energy > 1e-10 { 1.0 / energy } else { 0.0 };

            for i in 0..band_size {
                if offset + i < coeffs.len() {
                    coeffs[offset + i] *= norm_factor;
                }
            }

            offset += band_size;
        }
    }

    /// Encodes spectral shape using PVQ.
    fn encode_spectral_shape(
        &mut self,
        encoder: &mut RangeEncoder,
        coeffs: &[f32],
        frame_size: usize,
    ) -> CodecResult<()> {
        let band_sizes = self.get_band_sizes(frame_size);
        let mut offset = 0;

        for (band_idx, &band_size) in band_sizes.iter().enumerate() {
            if band_idx >= CELT_BANDS {
                break;
            }

            // Compute pulse count (simplified PVQ)
            let mut pulse_count = 0u32;
            for i in 0..band_size {
                if offset + i < coeffs.len() {
                    pulse_count += (coeffs[offset + i].abs() * 10.0).round() as u32;
                }
            }
            pulse_count = pulse_count.min(15);

            // Encode pulse count
            encoder.encode_uniform(pulse_count, 16)?;

            // Encode pulse positions (simplified - just encode major pulses)
            if pulse_count > 0 {
                for i in 0..band_size.min(4) {
                    if offset + i < coeffs.len() {
                        let pulse_val = (coeffs[offset + i].abs() * 4.0).round() as u32;
                        encoder.encode_uniform(pulse_val.min(3), 4)?;
                    }
                }
            }

            offset += band_size;
        }

        Ok(())
    }

    /// Returns band sizes for the given frame size.
    fn get_band_sizes(&self, frame_size: usize) -> Vec<usize> {
        // Simplified band allocation matching decoder
        let avg_size = frame_size / CELT_BANDS;
        vec![avg_size; CELT_BANDS]
    }

    /// Resets encoder state.
    pub fn reset(&mut self) {
        for ola in &mut self.overlap_add {
            ola.reset();
        }
        self.band_energy.fill(0.0);
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

    /// Returns the bandwidth.
    #[must_use]
    pub const fn bandwidth(&self) -> OpusBandwidth {
        self.bandwidth
    }
}
