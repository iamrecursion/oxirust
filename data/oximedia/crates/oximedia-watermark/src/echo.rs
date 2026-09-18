//! Echo hiding watermarking.
//!
//! This module implements echo-based watermarking techniques:
//! - Single echo
//! - Double echo (binary encoding)
//! - Triple echo (ternary encoding)

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};
use oxifft::Complex;

/// Kernel length threshold above which the FFT overlap-add path is used
/// instead of the direct time-domain convolution.
const OVERLAP_ADD_THRESHOLD: usize = 256;

/// Echo hiding configuration.
#[derive(Debug, Clone)]
pub struct EchoConfig {
    /// Delay for bit 0 (in samples).
    pub delay_0: usize,
    /// Delay for bit 1 (in samples).
    pub delay_1: usize,
    /// Echo amplitude (0.0 to 1.0).
    pub amplitude: f32,
    /// Decay rate for echo.
    pub decay: f32,
    /// Kernel size for echo detection.
    pub kernel_size: usize,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            delay_0: 50,  // ~1.1 ms at 44.1kHz
            delay_1: 100, // ~2.3 ms at 44.1kHz
            amplitude: 0.5,
            decay: 0.8,
            kernel_size: 512,
        }
    }
}

/// Echo hiding watermark embedder.
pub struct EchoEmbedder {
    config: EchoConfig,
    codec: PayloadCodec,
}

impl EchoEmbedder {
    /// Create a new echo embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: EchoConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark using echo hiding.
    ///
    /// For `kernel_size > OVERLAP_ADD_THRESHOLD` (default 512 > 256), uses an
    /// FFT overlap-add convolution for O(n log n) complexity.  Shorter kernels
    /// use the direct time-domain path.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        // Check capacity
        let required_samples = bits.len() * self.config.kernel_size;
        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        // Embed each bit
        for (bit_idx, &bit) in bits.iter().enumerate() {
            let delay = if bit {
                self.config.delay_1
            } else {
                self.config.delay_0
            };

            let start = bit_idx * self.config.kernel_size;
            let end = (start + self.config.kernel_size).min(watermarked.len());
            let block_len = end - start;

            if self.config.kernel_size > OVERLAP_ADD_THRESHOLD {
                // FFT overlap-add path.
                // IR: unit sample + decayed tap at delay + doubly-decayed tap at 2*delay.
                let ir_len = delay * 2 + 1;
                let fft_size = (block_len + ir_len - 1).next_power_of_two();

                // Build impulse response in freq domain (H).
                let mut ir_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
                // Direct signal: identity (gain 1 at sample 0 is implicit via source copy;
                // the IR represents ONLY the echo additions, not the signal itself).
                let amp1 = self.config.amplitude * self.config.decay;
                let amp2 = self.config.amplitude * self.config.decay.powi(2);
                if delay < fft_size {
                    ir_buf[delay] = Complex::new(amp1, 0.0);
                }
                if delay * 2 < fft_size {
                    ir_buf[delay * 2] = Complex::new(amp2, 0.0);
                }
                let h = oxifft::fft(&ir_buf);

                // Zero-pad input block.
                let mut x_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
                for (j, &s) in samples[start..end].iter().enumerate() {
                    x_buf[j] = Complex::new(s, 0.0);
                }
                let x = oxifft::fft(&x_buf);

                // Multiply in frequency domain.
                let y_freq: Vec<Complex<f32>> =
                    x.iter().zip(h.iter()).map(|(&xi, &hi)| xi * hi).collect();

                // IFFT: oxifft::ifft already normalises by 1/N, so no further scaling
                // is needed — the result is the linear convolution of the block with the IR.
                let y_time = oxifft::ifft(&y_freq);

                // Overlap-add: accumulate the echo contribution into the output.
                // Output length = block_len + ir_len - 1, clamped to the signal boundary.
                let out_len = (block_len + ir_len - 1).min(watermarked.len() - start);
                for j in 0..out_len {
                    watermarked[start + j] += y_time[j].re;
                }
            } else {
                // Direct time-domain path for short kernels.
                for i in start..end {
                    let echo_idx = i + delay;
                    if echo_idx < end && echo_idx < watermarked.len() {
                        let echo_amplitude = self.config.amplitude * self.config.decay;
                        watermarked[echo_idx] += samples[i] * echo_amplitude;
                    }

                    // Multiple echoes for more robustness
                    let echo_idx2 = i + delay * 2;
                    if echo_idx2 < end && echo_idx2 < watermarked.len() {
                        let echo_amplitude2 = self.config.amplitude * self.config.decay.powi(2);
                        watermarked[echo_idx2] += samples[i] * echo_amplitude2;
                    }
                }
            }
        }

        Ok(watermarked)
    }

    /// Embed a single block using the direct time-domain echo path.
    ///
    /// Used internally for comparison tests and short-kernel fallback.
    #[cfg(test)]
    pub(crate) fn embed_direct_block(
        &self,
        samples: &[f32],
        block_start: usize,
        block_len: usize,
        delay: usize,
        out: &mut [f32],
    ) {
        let end = block_start + block_len;
        for offset in 0..block_len {
            let i = block_start + offset;
            let echo_idx = i + delay;
            if echo_idx < end && echo_idx < out.len() {
                out[echo_idx] += samples[i] * self.config.amplitude * self.config.decay;
            }
            let echo_idx2 = i + delay * 2;
            if echo_idx2 < end && echo_idx2 < out.len() {
                out[echo_idx2] += samples[i] * self.config.amplitude * self.config.decay.powi(2);
            }
        }
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        sample_count / self.config.kernel_size
    }
}

/// Echo hiding watermark detector.
pub struct EchoDetector {
    config: EchoConfig,
    codec: PayloadCodec,
}

impl EchoDetector {
    /// Create a new echo detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: EchoConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected or decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let mut bits = Vec::new();

        for bit_idx in 0..expected_bits {
            let start = bit_idx * self.config.kernel_size;
            let end = (start + self.config.kernel_size).min(samples.len());

            if end <= start {
                break;
            }

            let segment = &samples[start..end];

            // Calculate autocorrelation at both delays
            let corr_0 = self.autocorrelation(segment, self.config.delay_0);
            let corr_1 = self.autocorrelation(segment, self.config.delay_1);

            // Choose delay with higher correlation
            bits.push(corr_1 > corr_0);
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Calculate autocorrelation at given delay.
    fn autocorrelation(&self, samples: &[f32], delay: usize) -> f32 {
        if samples.len() <= delay {
            return 0.0;
        }

        let mut sum = 0.0f32;
        let mut energy = 0.0f32;

        for i in 0..(samples.len() - delay) {
            sum += samples[i] * samples[i + delay];
            energy += samples[i] * samples[i];
        }

        if energy > 1e-10 {
            sum / energy
        } else {
            0.0
        }
    }

    /// Calculate cepstrum for echo detection.
    #[cfg(test)]
    pub(crate) fn cepstrum(&self, samples: &[f32]) -> Vec<f32> {
        let fft_size = samples.len().next_power_of_two();

        // FFT
        let freq_input: Vec<Complex<f32>> = samples
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(fft_size)
            .collect();

        let fft_result = oxifft::fft(&freq_input);

        // Log magnitude
        let log_mag: Vec<Complex<f32>> = fft_result
            .iter()
            .map(|c| {
                let mag = c.norm().max(1e-10);
                Complex::new(mag.ln(), 0.0)
            })
            .collect();

        // IFFT
        let ifft_result = oxifft::ifft(&log_mag);

        // Return real part
        #[allow(clippy::cast_precision_loss)]
        ifft_result.iter().map(|c| c.re / fft_size as f32).collect()
    }
}

/// Triple echo watermarking for ternary encoding.
pub struct TripleEchoEmbedder {
    delay_0: usize,
    delay_1: usize,
    delay_2: usize,
    amplitude: f32,
}

impl TripleEchoEmbedder {
    /// Create a new triple echo embedder.
    #[must_use]
    pub fn new(delay_0: usize, delay_1: usize, delay_2: usize, amplitude: f32) -> Self {
        Self {
            delay_0,
            delay_1,
            delay_2,
            amplitude,
        }
    }

    /// Embed ternary symbol (0, 1, or 2).
    #[must_use]
    pub fn embed_symbol(&self, samples: &[f32], symbol: u8) -> Vec<f32> {
        let delay = match symbol {
            0 => self.delay_0,
            1 => self.delay_1,
            _ => self.delay_2,
        };

        let mut watermarked = samples.to_vec();

        for i in 0..samples.len() {
            let echo_idx = i + delay;
            if echo_idx < watermarked.len() {
                watermarked[echo_idx] += samples[i] * self.amplitude;
            }
        }

        watermarked
    }

    /// Detect ternary symbol.
    #[must_use]
    pub fn detect_symbol(&self, samples: &[f32]) -> u8 {
        let corr_0 = self.autocorr(samples, self.delay_0);
        let corr_1 = self.autocorr(samples, self.delay_1);
        let corr_2 = self.autocorr(samples, self.delay_2);

        if corr_0 >= corr_1 && corr_0 >= corr_2 {
            0
        } else if corr_1 >= corr_2 {
            1
        } else {
            2
        }
    }

    /// Calculate autocorrelation.
    fn autocorr(&self, samples: &[f32], delay: usize) -> f32 {
        if samples.len() <= delay {
            return 0.0;
        }

        let mut sum = 0.0f32;
        for i in 0..(samples.len() - delay) {
            sum += samples[i] * samples[i + delay];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_embedding() {
        let config = EchoConfig::default();
        let embedder = EchoEmbedder::new(config.clone()).unwrap();
        let detector = EchoDetector::new(config.clone()).unwrap();

        // Payload "Echo Test" (9 bytes) encodes to ~280 bits with PayloadCodec(16,8).
        // Each bit needs kernel_size=512 samples, so we need at least 280*512=143360.
        // Use a signal with a single impulse at the start of each block: this gives
        // zero base autocorrelation at all non-zero lags, so only the embedded echo
        // creates correlation peaks for reliable detection.
        let kernel_size = config.kernel_size;
        let n_blocks = 300; // 300 * 512 = 153600 > 143360 needed
        let mut samples = vec![0.0f32; n_blocks * kernel_size];
        for block in 0..n_blocks {
            samples[block * kernel_size] = 1.0;
        }
        let payload = b"Echo Test";

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
    fn test_autocorrelation() {
        let config = EchoConfig::default();
        let detector = EchoDetector::new(config.clone()).unwrap();

        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                (i as f32 * 0.1).sin()
            })
            .collect();

        let corr = detector.autocorrelation(&samples, 10);
        assert!(corr.abs() <= 1.0);
    }

    #[test]
    fn test_triple_echo() {
        let embedder = TripleEchoEmbedder::new(30, 60, 90, 0.5);

        // Use pseudo-random noise with a long signal (N=10000).  For white noise,
        // the raw autocorrelation at non-zero lags is O(sqrt(N)) while the echo
        // contribution is O(N * amplitude), so the SNR is O(amplitude * sqrt(N)).
        // With amplitude=0.5 and N=10000, SNR ≈ 50 → reliable detection.
        let mut rng = scirs2_core::random::Random::seed(42);
        let samples: Vec<f32> = (0..10000).map(|_| rng.random_f64() as f32 - 0.5).collect();

        for symbol in 0..3 {
            let watermarked = embedder.embed_symbol(&samples, symbol);
            let detected = embedder.detect_symbol(&watermarked);
            assert_eq!(symbol, detected);
        }
    }

    #[test]
    fn test_capacity() {
        let config = EchoConfig::default();
        let embedder = EchoEmbedder::new(config).unwrap();

        let capacity = embedder.capacity(44100); // 1 second at 44.1kHz
        assert!(capacity > 0);
    }

    #[test]
    fn test_cepstrum_length() {
        // Verify the cepstrum helper produces output of expected length and does
        // not panic on typical input.
        let config = EchoConfig::default();
        let detector = EchoDetector::new(config).expect("should succeed in test");
        let samples: Vec<f32> = (0..256)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                (i as f32 * 0.05).sin()
            })
            .collect();

        let cep = detector.cepstrum(&samples);
        // next_power_of_two(256) = 256, so output must have 256 elements.
        assert_eq!(cep.len(), 256, "cepstrum output length mismatch");
        // The cepstrum of a sinusoid should have most energy near its period.
        assert!(
            cep.iter().any(|&v| v.abs() > 0.0),
            "cepstrum should be non-zero"
        );
    }

    // ── Item 2: Echo FFT overlap-add ─────────────────────────────────────────

    #[test]
    fn test_echo_fft_conv_approx_direct() {
        // Verify that the FFT overlap-add path (kernel_size=512 > 256) and the
        // direct time-domain path produce the same echo taps on an impulse signal.
        // We compare both paths at the expected echo positions.

        let delay: usize = 30;
        let amplitude = 0.4_f32;
        let decay = 0.7_f32;

        // --- FFT path: build signal large enough to embed a 1-bit payload ---
        let config_fft = EchoConfig {
            delay_0: delay,
            delay_1: 60,
            amplitude,
            decay,
            kernel_size: 512, // > OVERLAP_ADD_THRESHOLD → FFT path
        };
        let embedder_fft = EchoEmbedder::new(config_fft.clone()).expect("should succeed in test");
        let codec = crate::payload::PayloadCodec::new(16, 8).expect("codec");
        let payload = b"A";
        let encoded = codec.encode(payload).expect("encode");
        let n_bits = encoded.len() * 8;
        // Signal: impulse at start of every block.
        let total_len = (n_bits + 1) * 512;
        let mut samples_fft = vec![0.0f32; total_len];
        for b in 0..=(n_bits) {
            samples_fft[b * 512] = 1.0;
        }
        let watermarked_fft = embedder_fft
            .embed(&samples_fft, payload)
            .expect("fft embed should succeed in test");

        // --- Direct path: short kernel so OVERLAP_ADD_THRESHOLD is not crossed ---
        let config_direct = EchoConfig {
            delay_0: delay,
            delay_1: 60,
            amplitude,
            decay,
            kernel_size: 128, // ≤ 256 → direct path
        };
        let embedder_direct =
            EchoEmbedder::new(config_direct.clone()).expect("should succeed in test");
        let total_direct = (n_bits + 1) * 128;
        let mut samples_direct = vec![0.0f32; total_direct];
        for b in 0..=(n_bits) {
            samples_direct[b * 128] = 1.0;
        }
        let watermarked_direct = embedder_direct
            .embed(&samples_direct, payload)
            .expect("direct embed should succeed in test");

        // Both paths should create an echo at offset `delay` from the first block
        // with value = amplitude * decay.
        let expected_echo = amplitude * decay;

        // FFT path: echo at block[0]+delay.
        let fft_echo = watermarked_fft[delay] - samples_fft[delay];
        assert!(
            (fft_echo - expected_echo).abs() < 1e-3,
            "FFT echo tap: got {fft_echo}, expected {expected_echo}"
        );

        // Direct path: echo at block[0]+delay.
        let direct_echo = watermarked_direct[delay] - samples_direct[delay];
        assert!(
            (direct_echo - expected_echo).abs() < 1e-3,
            "Direct echo tap: got {direct_echo}, expected {expected_echo}"
        );

        // Cross-compare: both should agree.
        assert!(
            (fft_echo - direct_echo).abs() < 1e-3,
            "FFT echo {fft_echo} vs direct echo {direct_echo} differ"
        );

        // Also exercise embed_direct_block directly for a single block and
        // verify it produces the expected echo contribution.
        let direct_block_len = 128_usize;
        let mut block_signal = vec![0.0f32; direct_block_len];
        block_signal[0] = 1.0;
        let mut direct_block_out = block_signal.clone();
        embedder_direct.embed_direct_block(
            &block_signal,
            0,
            direct_block_len,
            delay,
            &mut direct_block_out,
        );
        assert!(
            (direct_block_out[delay] - expected_echo).abs() < 1e-4,
            "embed_direct_block echo at {delay}: got {}, expected {expected_echo}",
            direct_block_out[delay]
        );
    }

    #[test]
    fn test_echo_fft_roundtrip_detect() {
        // Embed a 2-byte payload with the FFT overlap-add path (kernel_size=512),
        // then detect it and assert the correct payload is recovered.
        let config = EchoConfig {
            delay_0: 50,
            delay_1: 100,
            amplitude: 0.5,
            decay: 0.8,
            kernel_size: 512, // > OVERLAP_ADD_THRESHOLD → FFT path
        };
        let embedder = EchoEmbedder::new(config.clone()).expect("should succeed in test");
        let detector = EchoDetector::new(config.clone()).expect("should succeed in test");

        let payload = b"Hi";
        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;
        let n_blocks = expected_bits + 10;
        let kernel_size = config.kernel_size;

        // Use impulse signal so autocorrelation is dominated by the echo.
        let mut samples = vec![0.0f32; n_blocks * kernel_size];
        for block in 0..n_blocks {
            samples[block * kernel_size] = 1.0;
        }

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("embed should succeed in test");
        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("detect should succeed in test");

        assert_eq!(
            payload.as_slice(),
            extracted.as_slice(),
            "FFT overlap-add roundtrip: detected payload must match original"
        );
    }
}
