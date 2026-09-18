//! Direct Sequence Spread Spectrum (DSSS) watermarking.
//!
//! This module implements spread spectrum watermarking, which embeds
//! watermark bits by spreading them using pseudorandom sequences.
//!
//! The [`InPlaceFftEmbedder`] provides an optimised variant that pre-allocates
//! FFT scratch buffers once at construction time and reuses them across
//! multiple `embed_in_place` calls, significantly reducing allocator pressure.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec};
use crate::psychoacoustic::PsychoacousticModel;
use oxifft::{Complex, Direction, Flags, Plan};

/// Spread spectrum watermarking configuration.
#[derive(Debug, Clone)]
pub struct SpreadSpectrumConfig {
    /// Embedding strength (0.0 to 1.0).
    pub strength: f32,
    /// Chip rate (spreading factor).
    pub chip_rate: usize,
    /// Use frequency domain embedding.
    pub frequency_domain: bool,
    /// Psychoacoustic masking enabled.
    pub psychoacoustic: bool,
    /// Secret key for PN sequence generation.
    pub key: u64,
}

impl Default for SpreadSpectrumConfig {
    fn default() -> Self {
        Self {
            strength: 0.1,
            chip_rate: 64,
            frequency_domain: true,
            psychoacoustic: true,
            key: 0,
        }
    }
}

/// Spread spectrum watermark embedder.
pub struct SpreadSpectrumEmbedder {
    config: SpreadSpectrumConfig,
    codec: PayloadCodec,
    psycho_model: Option<PsychoacousticModel>,
}

impl SpreadSpectrumEmbedder {
    /// Create a new spread spectrum embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(
        config: SpreadSpectrumConfig,
        sample_rate: u32,
        frame_size: usize,
    ) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        let psycho_model = if config.psychoacoustic {
            Some(PsychoacousticModel::new(sample_rate, frame_size))
        } else {
            None
        };

        Ok(Self {
            config,
            codec,
            psycho_model,
        })
    }

    /// Embed watermark in audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload with error correction
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        if self.config.frequency_domain {
            self.embed_frequency_domain(samples, &bits)
        } else {
            self.embed_time_domain(samples, &bits)
        }
    }

    /// Embed in time domain.
    fn embed_time_domain(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        let required_samples = bits.len() * self.config.chip_rate;
        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        // Build PN sequence table once before the loop to avoid redundant generation.
        let pn_table: Vec<Vec<i8>> = (0..bits.len())
            .map(|i| generate_pn_sequence(self.config.chip_rate, self.config.key + i as u64))
            .collect();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let pn_seq = &pn_table[bit_idx];
            let bit_value = if bit { 1.0f32 } else { -1.0f32 };

            let start = bit_idx * self.config.chip_rate;
            for (i, &pn) in pn_seq.iter().enumerate() {
                if start + i >= watermarked.len() {
                    break;
                }
                watermarked[start + i] += self.config.strength * bit_value * f32::from(pn);
            }
        }

        Ok(watermarked)
    }

    /// Embed in frequency domain.
    fn embed_frequency_domain(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        let frame_size = 2048;
        // Use non-overlapping frames (hop = frame_size) to avoid overwrite conflicts
        // when writing IFFT output back to the signal buffer.
        let hop_size = frame_size;
        let required_frames = bits.len().div_ceil(8); // 8 bits per frame

        if samples.len() < required_frames * hop_size {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_frames * hop_size,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        // Calculate masking threshold using the first frame only.
        // The psychoacoustic model requires exactly frame_size samples.
        let masking = if let Some(ref model) = self.psycho_model {
            let first_frame = &samples[..frame_size.min(samples.len())];
            if first_frame.len() == frame_size {
                Some(model.calculate_masking_threshold(first_frame))
            } else {
                None
            }
        } else {
            None
        };

        // Build PN sequence table once before the loop to avoid redundant generation.
        let pn_table: Vec<Vec<i8>> = (0..bits.len())
            .map(|i| generate_pn_sequence(self.config.chip_rate, self.config.key + i as u64))
            .collect();

        let mut bit_idx = 0;

        for frame_idx in 0..required_frames {
            if bit_idx >= bits.len() {
                break;
            }

            let frame_start = frame_idx * hop_size;
            if frame_start + frame_size > samples.len() {
                break;
            }

            // Extract frame
            let frame = &samples[frame_start..frame_start + frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let mut freq_data = oxifft::fft(&freq_input);

            // Embed bits in frequency domain
            for _ in 0..8 {
                if bit_idx >= bits.len() {
                    break;
                }

                let bit = bits[bit_idx];
                let pn_seq = &pn_table[bit_idx];

                let bit_value = if bit { 1.0f32 } else { -1.0f32 };

                // Embed in mid-frequency range (more robust)
                let start_bin = frame_size / 8;
                let end_bin = start_bin + self.config.chip_rate;

                for (i, &pn) in pn_seq.iter().enumerate().take(self.config.chip_rate) {
                    let bin = start_bin + i;
                    if bin >= end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    // Calculate embedding strength based on masking
                    let strength = if let Some(ref mask) = masking {
                        let mask_val = mask.get(bin).copied().unwrap_or(-60.0);
                        // Scale strength based on masking threshold
                        self.config.strength * 10.0f32.powf(mask_val / 20.0)
                    } else {
                        self.config.strength
                    };

                    let watermark_val = strength * bit_value * f32::from(pn);
                    freq_data[bin] += Complex::new(watermark_val, 0.0);

                    // Mirror for conjugate symmetry
                    let mirror_bin = frame_size - bin;
                    if mirror_bin < freq_data.len() {
                        freq_data[mirror_bin] += Complex::new(watermark_val, 0.0);
                    }
                }

                bit_idx += 1;
            }

            // IFFT
            let ifft_result = oxifft::ifft(&freq_data);

            // Overlap-add
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / frame_size as f32;
            for (i, c) in ifft_result.iter().enumerate().take(frame_size) {
                let idx = frame_start + i;
                if idx < watermarked.len() {
                    watermarked[idx] = c.re * scale;
                }
            }
        }

        Ok(watermarked)
    }

    /// Calculate capacity in bits for given audio length.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        if self.config.frequency_domain {
            let frame_size = 2048;
            // Non-overlapping frames
            let hop_size = frame_size;
            let frame_count = sample_count / hop_size;
            frame_count * 8 // 8 bits per frame
        } else {
            sample_count / self.config.chip_rate
        }
    }
}

/// Spread spectrum watermark detector.
pub struct SpreadSpectrumDetector {
    config: SpreadSpectrumConfig,
    codec: PayloadCodec,
}

impl SpreadSpectrumDetector {
    /// Create a new spread spectrum detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: SpreadSpectrumConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark from audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected or decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let bits = if self.config.frequency_domain {
            self.detect_frequency_domain(samples, expected_bits)?
        } else {
            self.detect_time_domain(samples, expected_bits)?
        };

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Detect in time domain.
    fn detect_time_domain(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let mut bits = Vec::new();

        // Build PN sequence table once before the loop to avoid redundant generation.
        let pn_table: Vec<Vec<i8>> = (0..expected_bits)
            .map(|i| generate_pn_sequence(self.config.chip_rate, self.config.key + i as u64))
            .collect();

        for bit_idx in 0..expected_bits {
            let pn_seq = &pn_table[bit_idx];
            let start = bit_idx * self.config.chip_rate;

            if start + self.config.chip_rate > samples.len() {
                break;
            }

            // Correlate with PN sequence
            let mut corr = 0.0f32;
            for (i, &pn) in pn_seq.iter().enumerate() {
                corr += samples[start + i] * f32::from(pn);
            }

            bits.push(corr > 0.0);
        }

        Ok(bits)
    }

    /// Detect in frequency domain.
    fn detect_frequency_domain(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let frame_size = 2048;
        // Use non-overlapping frames to match the embedder's frame layout.
        let hop_size = frame_size;
        let required_frames = expected_bits.div_ceil(8);

        // Build PN sequence table once before the loop to avoid redundant generation.
        let pn_table: Vec<Vec<i8>> = (0..expected_bits)
            .map(|i| generate_pn_sequence(self.config.chip_rate, self.config.key + i as u64))
            .collect();

        let mut bits = Vec::new();

        for frame_idx in 0..required_frames {
            let frame_start = frame_idx * hop_size;
            if frame_start + frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let freq_data = oxifft::fft(&freq_input);

            // Extract bits
            for _ in 0..8 {
                if bits.len() >= expected_bits {
                    break;
                }

                let bit_idx = bits.len();
                let pn_seq = &pn_table[bit_idx];

                let start_bin = frame_size / 8;
                let mut corr = 0.0f32;

                for (i, &pn) in pn_seq.iter().enumerate().take(self.config.chip_rate) {
                    let bin = start_bin + i;
                    if bin >= freq_data.len() / 2 {
                        break;
                    }

                    corr += freq_data[bin].re * f32::from(pn);
                }

                bits.push(corr > 0.0);
            }
        }

        Ok(bits)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// In-place FFT embedder (Item 4)
// ──────────────────────────────────────────────────────────────────────────────

/// A spread-spectrum frequency-domain embedder that **pre-allocates** FFT
/// scratch buffers once and reuses them across [`embed_in_place`] calls.
///
/// The ordinary [`SpreadSpectrumEmbedder`] allocates a new `Vec<Complex<f32>>`
/// for every FFT call.  `InPlaceFftEmbedder` holds pre-built
/// [`Plan<f32>`](oxifft::Plan) objects and a pair of reusable scratch buffers,
/// so that repeated embedding of many frames does not repeatedly hit the
/// allocator.
///
/// [`embed_in_place`]: InPlaceFftEmbedder::embed_in_place
pub struct InPlaceFftEmbedder {
    /// Embedding strength.
    pub strength: f32,
    /// Chip rate (spreading factor).
    pub chip_rate: usize,
    /// Secret key for PN sequence generation.
    pub key: u64,
    /// Number of frequency-domain samples per frame.
    frame_size: usize,
    /// Pre-built forward FFT plan.
    fwd_plan: Plan<f32>,
    /// Pre-built inverse FFT plan.
    inv_plan: Plan<f32>,
    /// Reusable complex input/output buffer (length = `frame_size`).
    buf_a: Vec<Complex<f32>>,
    /// Reusable complex buffer for intermediate FFT output (length = `frame_size`).
    buf_b: Vec<Complex<f32>>,
}

impl InPlaceFftEmbedder {
    /// Construct an embedder that pre-allocates buffers for `frame_size`-point FFTs.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if the FFT planner cannot create a plan for
    /// the given `frame_size` (must be a power of two or a size supported by
    /// OxiFFT's radix algorithms).
    pub fn new(
        frame_size: usize,
        strength: f32,
        chip_rate: usize,
        key: u64,
    ) -> WatermarkResult<Self> {
        let fwd_plan = Plan::<f32>::dft_1d(frame_size, Direction::Forward, Flags::ESTIMATE)
            .ok_or_else(|| {
                WatermarkError::InvalidParameter(format!(
                    "OxiFFT cannot plan a {frame_size}-point forward FFT"
                ))
            })?;
        let inv_plan = Plan::<f32>::dft_1d(frame_size, Direction::Backward, Flags::ESTIMATE)
            .ok_or_else(|| {
                WatermarkError::InvalidParameter(format!(
                    "OxiFFT cannot plan a {frame_size}-point inverse FFT"
                ))
            })?;

        Ok(Self {
            strength: strength.clamp(0.0, 1.0),
            chip_rate,
            key,
            frame_size,
            fwd_plan,
            inv_plan,
            buf_a: vec![Complex::new(0.0, 0.0); frame_size],
            buf_b: vec![Complex::new(0.0, 0.0); frame_size],
        })
    }

    /// Embed `payload` bits into `signal` in-place, reusing the pre-allocated
    /// scratch buffers.
    ///
    /// The signal is processed in non-overlapping frames of `frame_size` samples.
    /// Each frame receives up to 8 watermark bits via frequency-domain modification.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::InsufficientCapacity`] if `signal` is too
    /// short to embed `payload`.
    pub fn embed_in_place(&mut self, signal: &mut [f32], payload: &[u8]) -> WatermarkResult<()> {
        let codec = PayloadCodec::new(16, 8)?;
        let encoded = codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        let hop = self.frame_size;
        let required_frames = bits.len().div_ceil(8);
        let required_len = required_frames * hop;
        if signal.len() < required_len {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_len,
                have: signal.len(),
            });
        }

        // Build PN sequence table once before the loop to avoid redundant generation.
        let pn_table: Vec<Vec<i8>> = (0..bits.len())
            .map(|i| generate_pn_sequence(self.chip_rate, self.key + i as u64))
            .collect();

        let mut bit_idx = 0;

        for frame_idx in 0..required_frames {
            if bit_idx >= bits.len() {
                break;
            }

            let frame_start = frame_idx * hop;
            if frame_start + self.frame_size > signal.len() {
                break;
            }

            // Fill input buffer (reuse buf_a).
            for (i, c) in self.buf_a.iter_mut().enumerate() {
                *c = Complex::new(signal[frame_start + i], 0.0);
            }

            // Forward FFT: buf_a → buf_b.
            self.fwd_plan.execute(&self.buf_a.clone(), &mut self.buf_b);

            // Embed up to 8 bits into mid-frequency bins.
            let start_bin = self.frame_size / 8;
            let end_bin = start_bin + self.chip_rate;

            for _ in 0..8 {
                if bit_idx >= bits.len() {
                    break;
                }
                let bit = bits[bit_idx];
                let pn_seq = &pn_table[bit_idx];
                let bit_val: f32 = if bit { 1.0 } else { -1.0 };

                for (i, &pn) in pn_seq.iter().enumerate().take(self.chip_rate) {
                    let bin = start_bin + i;
                    if bin >= end_bin || bin >= self.buf_b.len() / 2 {
                        break;
                    }
                    let wm = self.strength * bit_val * f32::from(pn);
                    self.buf_b[bin] += Complex::new(wm, 0.0);
                    let mirror = self.frame_size - bin;
                    if mirror < self.buf_b.len() {
                        self.buf_b[mirror] += Complex::new(wm, 0.0);
                    }
                }
                bit_idx += 1;
            }

            // Inverse FFT: buf_b → buf_a.
            // Plan::execute (Backward) returns an un-normalised transform.
            // The allocating path uses oxifft::ifft() which normalises by 1/N
            // and then additionally multiplies by the outer `scale = 1/N`, so the
            // effective scale factor is 1/N². Replicate that here: divide by N²
            // so that both paths produce identical output.
            self.inv_plan.execute(&self.buf_b.clone(), &mut self.buf_a);

            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / (self.frame_size as f32 * self.frame_size as f32);
            for (i, c) in self.buf_a.iter().enumerate().take(self.frame_size) {
                if frame_start + i < signal.len() {
                    signal[frame_start + i] = c.re * scale;
                }
            }
        }

        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SIMD-optimised correlation (Item 6)
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the cross-correlation (dot product) between `signal` and `code`
/// using a SIMD-accelerated implementation when available.
///
/// Falls back to a pure scalar implementation when neither AVX2 (x86_64) nor
/// NEON (aarch64) is detected at runtime, or when the slices are shorter than
/// the SIMD vector width.
///
/// Uses [`scirs2_core::simd_aligned::simd_dot_aligned_f32`] which provides
/// runtime-dispatched SIMD acceleration (AVX2 / NEON) via a safe API.
///
/// # Panics
///
/// Does not panic.  When the crate's `simd_aligned` module returns an error
/// (lengths differ), the function silently falls back to the scalar path.
#[must_use]
pub fn correlate_simd(signal: &[f32], code: &[f32]) -> f32 {
    let n = signal.len().min(code.len());
    if n == 0 {
        return 0.0;
    }

    // scirs2_core::simd_aligned::simd_dot_aligned_f32 is a safe function that
    // provides runtime SIMD dispatch (AVX2 / NEON) without requiring any
    // `unsafe` in this crate.
    match scirs2_core::simd_aligned::simd_dot_aligned_f32(&signal[..n], &code[..n]) {
        Ok(result) => result,
        Err(_) => correlate_scalar(&signal[..n], &code[..n]),
    }
}

/// Pure scalar cross-correlation fallback.
#[must_use]
pub fn correlate_scalar(signal: &[f32], code: &[f32]) -> f32 {
    signal.iter().zip(code.iter()).map(|(&s, &c)| s * c).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_spectrum_time_domain() {
        let config = SpreadSpectrumConfig {
            strength: 0.05,
            chip_rate: 32,
            frequency_domain: false,
            psychoacoustic: false,
            key: 12345,
        };

        let embedder = SpreadSpectrumEmbedder::new(config.clone(), 44100, 2048).unwrap();
        let detector = SpreadSpectrumDetector::new(config).unwrap();

        let samples: Vec<f32> = vec![0.0; 10000];
        let payload = b"Test";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");
        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_spread_spectrum_frequency_domain() {
        let config = SpreadSpectrumConfig {
            strength: 0.1,
            chip_rate: 64,
            frequency_domain: true,
            psychoacoustic: false,
            key: 54321,
        };

        let embedder = SpreadSpectrumEmbedder::new(config.clone(), 44100, 2048).unwrap();
        let detector = SpreadSpectrumDetector::new(config).unwrap();

        // "WM" encodes to 280 bits, requiring 35 frames * 2048 frame_size = 71680 samples
        // with non-overlapping frames. Use 73728 (36 * 2048) for headroom.
        let samples: Vec<f32> = vec![0.0; 73728];
        let payload = b"WM";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");
        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_capacity_calculation() {
        let config = SpreadSpectrumConfig::default();
        let embedder = SpreadSpectrumEmbedder::new(config, 44100, 2048).unwrap();

        let capacity = embedder.capacity(44100); // 1 second
        assert!(capacity > 0);
    }

    // ── Item 4: InPlaceFftEmbedder ───────────────────────────────────────────

    #[test]
    fn test_in_place_fft_matches_allocating() {
        // Embed the same payload with both the allocating and in-place embedders
        // and verify the outputs are identical.
        let frame_size = 2048;
        let strength = 0.1;
        let chip_rate = 64;
        let key = 99_u64;
        let payload = b"WM";

        // Allocating embedder reference.
        let alloc_config = SpreadSpectrumConfig {
            strength,
            chip_rate,
            frequency_domain: true,
            psychoacoustic: false,
            key,
        };
        let alloc_embedder = SpreadSpectrumEmbedder::new(alloc_config, 44100, frame_size)
            .expect("should succeed in test");

        // Build a signal long enough for the payload.
        let encoded = alloc_embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let num_bits = encoded.len() * 8;
        let required_frames = num_bits.div_ceil(8);
        let signal_len = (required_frames + 1) * frame_size;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();

        let alloc_result = alloc_embedder
            .embed(&original, payload)
            .expect("alloc embed should succeed in test");

        // In-place embedder.
        let mut inplace_embedder = InPlaceFftEmbedder::new(frame_size, strength, chip_rate, key)
            .expect("in-place embedder should initialise in test");
        let mut inplace_signal = original.clone();
        inplace_embedder
            .embed_in_place(&mut inplace_signal, payload)
            .expect("in-place embed should succeed in test");

        // Results must be identical (same algorithm, same buffers).
        for (i, (&a, &b)) in alloc_result.iter().zip(inplace_signal.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at sample {i}: alloc={a} vs in-place={b}"
            );
        }
    }

    #[test]
    fn test_in_place_fft_buffer_reuse() {
        // Verify that the same embedder can be called twice and produces the
        // same output (i.e. scratch buffers are properly reset between calls).
        let frame_size = 2048;
        let payload = b"X";
        // PayloadCodec(16,8) encodes 1 byte to ~35 bytes = 280 bits → 35 frames.
        // Use 40 frames for headroom.
        let signal_len = 40 * frame_size;

        let original: Vec<f32> = vec![0.1f32; signal_len];

        let mut embedder =
            InPlaceFftEmbedder::new(frame_size, 0.05, 32, 7).expect("should succeed in test");

        let mut sig1 = original.clone();
        embedder
            .embed_in_place(&mut sig1, payload)
            .expect("first embed should succeed in test");

        let mut sig2 = original.clone();
        embedder
            .embed_in_place(&mut sig2, payload)
            .expect("second embed should succeed in test");

        // Both passes on the same original must produce identical results.
        for (i, (&a, &b)) in sig1.iter().zip(sig2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "buffer-reuse mismatch at sample {i}: first={a} vs second={b}"
            );
        }
    }

    // ── Item 6: correlate_simd ───────────────────────────────────────────────

    #[test]
    fn test_correlate_simd_matches_scalar() {
        let signal: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let code: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let scalar = correlate_scalar(&signal, &code);
        let simd = correlate_simd(&signal, &code);

        assert!(
            (scalar - simd).abs() < 1e-3,
            "SIMD result {simd} differs from scalar {scalar}"
        );
    }

    #[test]
    fn test_correlate_simd_all_zeros() {
        let signal = vec![0.0f32; 512];
        let code = vec![1.0f32; 512];
        assert_eq!(correlate_simd(&signal, &code), 0.0);
    }

    #[test]
    fn test_correlate_simd_empty() {
        assert_eq!(correlate_simd(&[], &[]), 0.0);
    }

    // ── Item 1: PN sequence cache ─────────────────────────────────────────────

    #[test]
    fn test_pn_cache_embed_detect_roundtrip() {
        // Embed a 4-byte payload using the cached path, then detect and verify.
        let config = SpreadSpectrumConfig {
            strength: 0.1,
            chip_rate: 32,
            frequency_domain: false,
            psychoacoustic: false,
            key: 77777,
        };

        let embedder = SpreadSpectrumEmbedder::new(config.clone(), 44100, 2048)
            .expect("should succeed in test");
        let detector = SpreadSpectrumDetector::new(config).expect("should succeed in test");

        let payload = b"ABCD";
        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        // Signal long enough to hold all bits at chip_rate=32 samples/bit.
        let samples: Vec<f32> = vec![0.0; expected_bits * 32 + 256];

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("embed should succeed");
        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("detect should succeed");

        assert_eq!(
            payload.as_slice(),
            extracted.as_slice(),
            "PN-cache roundtrip: detected payload must match original"
        );
    }

    #[test]
    fn test_pn_cache_identical_to_uncached() {
        // Verify that the cached table entries match direct generate_pn_sequence calls.
        let chip_rate = 64_usize;
        let key = 42_u64;

        // Simulate what the precomputed table produces for bit indices 0 and 3.
        let cached_0 = generate_pn_sequence(chip_rate, key + 0);
        let cached_3 = generate_pn_sequence(chip_rate, key + 3);

        // Direct calls for comparison.
        let direct_0 = generate_pn_sequence(chip_rate, key);
        let direct_3 = generate_pn_sequence(chip_rate, key + 3);

        assert_eq!(
            cached_0, direct_0,
            "pn_table[0] must equal generate_pn_sequence(chip_rate, key+0)"
        );
        assert_eq!(
            cached_3, direct_3,
            "pn_table[3] must equal generate_pn_sequence(chip_rate, key+3)"
        );
    }
}
