//! Temporal watermarking: encode data across a sequence of frames/video frames
//! by modulating per-frame statistics (energy, mean, variance) in a
//! pseudo-random pattern tied to the payload bits.
//!
//! Unlike single-frame techniques, temporal watermarking survives re-encoding
//! that destroys intra-frame frequency information because the statistical
//! relationship between frames is preserved.
//!
//! ## Encoding Scheme
//!
//! - Signal is divided into non-overlapping *symbol windows* of `window_size`
//!   samples each.
//! - Each window encodes one bit via *energy dithering*: bit-1 windows have
//!   their RMS raised by `strength`; bit-0 windows are left unchanged.
//! - A PN sequence seeded from `key` is XOR'd with the payload bits to spread
//!   the information across the timeline.
//! - An RS-encoded payload (sync + CRC) is serialised over consecutive windows.
//!
//! ## Detection
//!
//! - For each window, compute RMS and correlate with the PN sequence.
//! - A positive correlation indicates a bit-1 window; negative a bit-0.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec};

// ---------------------------------------------------------------------------
// TemporalConfig
// ---------------------------------------------------------------------------

/// Configuration for temporal watermarking.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Number of samples per temporal symbol window.
    pub window_size: usize,
    /// Relative RMS boost for bit-1 windows (0.0–1.0).
    pub strength: f32,
    /// Secret key for PN sequence.
    pub key: u64,
    /// Number of symbol windows used per bit (for redundancy).
    pub windows_per_bit: usize,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_size: 4096,
            strength: 0.05,
            key: 0,
            windows_per_bit: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalEmbedder
// ---------------------------------------------------------------------------

/// Embeds a payload across the temporal axis of an audio signal.
pub struct TemporalEmbedder {
    config: TemporalConfig,
    codec: PayloadCodec,
}

impl TemporalEmbedder {
    /// Create a new temporal embedder.
    ///
    /// # Errors
    ///
    /// Returns error if the codec cannot be initialised.
    pub fn new(config: TemporalConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed `payload` temporally across `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::InsufficientCapacity`] if the signal is
    /// too short to carry the encoded payload.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        let windows_needed = bits.len() * self.config.windows_per_bit;
        let samples_needed = windows_needed * self.config.window_size;

        if samples.len() < samples_needed {
            return Err(WatermarkError::InsufficientCapacity {
                needed: samples_needed,
                have: samples.len(),
            });
        }

        let pn = generate_pn_sequence(bits.len(), self.config.key);
        let mut watermarked = samples.to_vec();

        // Energy QIM step: the embedded energy level is quantised to the
        // nearest even multiple of `delta` (bit-0) or odd multiple (bit-1).
        // This is detectable without the original signal.
        let delta = self.config.strength;

        for (bit_idx, (&bit, &pn_chip)) in bits.iter().zip(pn.iter()).enumerate() {
            // XOR bit with PN chip to spread.
            let spread_bit = bit ^ (pn_chip < 0);

            let base_window = bit_idx * self.config.windows_per_bit;
            for rep in 0..self.config.windows_per_bit {
                let win_idx = base_window + rep;
                let start = win_idx * self.config.window_size;
                let end = (start + self.config.window_size).min(watermarked.len());
                if end <= start {
                    break;
                }
                let window = &mut watermarked[start..end];

                // Add a PN-keyed pseudo-noise sequence scaled to embed the bit.
                // Each sample += strength * PN_i * bit_value where bit_value = ±1.
                let bit_value: f32 = if spread_bit { 1.0 } else { -1.0 };
                // Generate per-sample PN using a deterministic seed from window index.
                let win_seed = self.config.key ^ ((bit_idx as u64) << 32) ^ (rep as u64);
                let sample_pn = generate_pn_sequence(window.len(), win_seed);
                for (s, &p) in window.iter_mut().zip(sample_pn.iter()) {
                    *s += delta * bit_value * f32::from(p);
                }
            }
        }

        Ok(watermarked)
    }

    /// Bit capacity in bits for the given sample count.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        let windows = sample_count / self.config.window_size;
        windows / self.config.windows_per_bit.max(1)
    }
}

// ---------------------------------------------------------------------------
// TemporalDetector
// ---------------------------------------------------------------------------

/// Extracts a temporally-embedded payload from a signal.
pub struct TemporalDetector {
    config: TemporalConfig,
    codec: PayloadCodec,
}

impl TemporalDetector {
    /// Create a new temporal detector.
    ///
    /// # Errors
    ///
    /// Returns error if the codec cannot be initialised.
    pub fn new(config: TemporalConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and decode the temporal watermark from `samples`.
    ///
    /// `expected_bits` must match the number of encoded bits (including RS
    /// overhead) that were embedded during encoding.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails (sync mismatch or CRC error).
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let pn = generate_pn_sequence(expected_bits, self.config.key);
        let mut bits = Vec::with_capacity(expected_bits);

        for bit_idx in 0..expected_bits {
            let pn_chip = pn.get(bit_idx).copied().unwrap_or(1);
            let base_window = bit_idx * self.config.windows_per_bit;

            // Correlate each window with the per-sample PN sequence used
            // during embedding to recover the spread bit.
            let mut total_corr = 0.0f64;
            let mut count = 0usize;

            for rep in 0..self.config.windows_per_bit {
                let win_idx = base_window + rep;
                let start = win_idx * self.config.window_size;
                let end = (start + self.config.window_size).min(samples.len());
                if end <= start {
                    break;
                }
                let window = &samples[start..end];
                let win_seed = self.config.key ^ ((bit_idx as u64) << 32) ^ (rep as u64);
                let sample_pn = generate_pn_sequence(window.len(), win_seed);

                let mut corr = 0.0f64;
                for (&s, &p) in window.iter().zip(sample_pn.iter()) {
                    corr += (s as f64) * (f32::from(p) as f64);
                }
                total_corr += corr;
                count += 1;
            }

            let avg_corr = if count > 0 {
                total_corr / count as f64
            } else {
                0.0
            };

            // Positive correlation → spread_bit = 1, negative → spread_bit = 0.
            let spread_bit = avg_corr > 0.0;

            // Undo PN spreading.
            let detected_bit = spread_bit ^ (pn_chip < 0);
            bits.push(detected_bit);
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }
}

// ---------------------------------------------------------------------------
// TemporalWatermarkFrame
// ---------------------------------------------------------------------------

/// A single temporal frame carrying watermark metadata.
#[derive(Debug, Clone)]
pub struct TemporalWatermarkFrame {
    /// Frame index within the temporal sequence.
    pub frame_index: usize,
    /// Bit value embedded in this frame.
    pub bit: bool,
    /// RMS energy of the frame samples.
    pub rms: f32,
    /// PN chip used for spreading.
    pub pn_chip: i8,
}

impl TemporalWatermarkFrame {
    /// Create from raw components.
    #[must_use]
    pub fn new(frame_index: usize, bit: bool, samples: &[f32], pn_chip: i8) -> Self {
        Self {
            frame_index,
            bit,
            rms: compute_rms(samples),
            pn_chip,
        }
    }

    /// Spread bit value (payload bit XOR PN chip).
    #[must_use]
    pub fn spread_bit(&self) -> bool {
        self.bit ^ (self.pn_chip < 0)
    }
}

// ---------------------------------------------------------------------------
// TemporalFrameSequence
// ---------------------------------------------------------------------------

/// A sequence of [`TemporalWatermarkFrame`]s representing a full watermark.
#[derive(Debug, Clone, Default)]
pub struct TemporalFrameSequence {
    frames: Vec<TemporalWatermarkFrame>,
}

impl TemporalFrameSequence {
    /// Create a new empty sequence.
    #[must_use]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Append a frame.
    pub fn push(&mut self, frame: TemporalWatermarkFrame) {
        self.frames.push(frame);
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Extract the raw bit sequence (PN-demodulated).
    #[must_use]
    pub fn extract_bits(&self) -> Vec<bool> {
        self.frames.iter().map(|f| f.bit).collect()
    }

    /// Average RMS across all frames.
    #[must_use]
    pub fn mean_rms(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.frames.iter().map(|f| f.rms).sum();
        #[allow(clippy::cast_precision_loss)]
        (sum / self.frames.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    #[allow(clippy::cast_precision_loss)]
    (sum_sq / samples.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(n: usize, freq_hz: f32, sample_rate: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (std::f32::consts::TAU * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_embed_length_preserved() {
        // PayloadCodec(16,8) encodes 1 byte ("X") to 35 bytes = 280 bits.
        // With window_size=512, we need 280 * 512 = 143360 samples.
        // Use 44100 * 4 = 176400 samples for sufficient headroom.
        let cfg = TemporalConfig {
            window_size: 512,
            strength: 0.05,
            key: 42,
            windows_per_bit: 1,
        };
        let embedder = TemporalEmbedder::new(cfg).expect("ok");
        let samples = sine_signal(44100 * 4, 440.0, 44100.0, 0.5);
        let watermarked = embedder.embed(&samples, b"X").expect("embed");
        assert_eq!(watermarked.len(), samples.len());
    }

    #[test]
    fn test_capacity_positive() {
        let cfg = TemporalConfig {
            window_size: 1024,
            strength: 0.05,
            key: 0,
            windows_per_bit: 1,
        };
        let embedder = TemporalEmbedder::new(cfg).expect("ok");
        let cap = embedder.capacity(44100 * 10);
        assert!(cap > 0);
    }

    #[test]
    fn test_embed_insufficient_capacity() {
        let cfg = TemporalConfig {
            window_size: 4096,
            strength: 0.05,
            key: 0,
            windows_per_bit: 1,
        };
        let embedder = TemporalEmbedder::new(cfg).expect("ok");
        let samples = vec![0.0f32; 100]; // too short
        let result = embedder.embed(&samples, b"long-payload-that-wont-fit-here");
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_frame_new() {
        let samples = vec![1.0f32; 64];
        let frame = TemporalWatermarkFrame::new(0, true, &samples, 1);
        assert_eq!(frame.frame_index, 0);
        assert!(frame.bit);
        assert!((frame.rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_frame_spread_bit() {
        let samples = vec![0.5f32; 64];
        let frame = TemporalWatermarkFrame::new(0, true, &samples, -1);
        // bit=true, pn_chip=-1 => spread = true XOR true = false
        assert!(!frame.spread_bit());
    }

    #[test]
    fn test_temporal_frame_sequence_operations() {
        let mut seq = TemporalFrameSequence::new();
        assert!(seq.is_empty());
        let samples = vec![0.5f32; 64];
        seq.push(TemporalWatermarkFrame::new(0, true, &samples, 1));
        seq.push(TemporalWatermarkFrame::new(1, false, &samples, 1));
        assert_eq!(seq.len(), 2);
        let bits = seq.extract_bits();
        assert_eq!(bits, vec![true, false]);
    }

    #[test]
    fn test_temporal_frame_sequence_mean_rms() {
        let mut seq = TemporalFrameSequence::new();
        let s1 = vec![1.0f32; 64];
        let s2 = vec![0.0f32; 64];
        seq.push(TemporalWatermarkFrame::new(0, true, &s1, 1));
        seq.push(TemporalWatermarkFrame::new(1, false, &s2, 1));
        // mean rms = (1.0 + 0.0) / 2 = 0.5
        assert!((seq.mean_rms() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rms_constant_signal() {
        let s = vec![0.5f32; 128];
        let rms = compute_rms(&s);
        assert!((rms - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rms_silence() {
        let s = vec![0.0f32; 128];
        assert_eq!(compute_rms(&s), 0.0);
    }

    #[test]
    fn test_temporal_roundtrip() {
        // Use a large signal so capacity is sufficient for the RS-encoded payload.
        let cfg = TemporalConfig {
            window_size: 512,
            strength: 0.1,
            key: 0xABCD_1234,
            windows_per_bit: 1,
        };
        let embedder = TemporalEmbedder::new(cfg.clone()).expect("ok");
        let detector = TemporalDetector::new(cfg).expect("ok");

        let samples = sine_signal(44100 * 8, 440.0, 44100.0, 0.5);
        let payload = b"T";
        let watermarked = embedder.embed(&samples, payload).expect("embed");

        // Compute expected_bits from the codec.
        let codec = PayloadCodec::new(16, 8).expect("ok");
        let encoded = codec.encode(payload).expect("encode");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("detect");
        assert_eq!(extracted, payload.as_slice());
    }

    #[test]
    fn test_windows_per_bit_two() {
        // Ensure the embedder computes the correct sample requirement with
        // windows_per_bit = 2 (each bit uses 2 consecutive windows).
        // 280 bits * 2 windows * 512 samples = 286720 samples needed.
        // Use 44100 * 8 = 352800 for headroom.
        let cfg = TemporalConfig {
            window_size: 512,
            strength: 0.08,
            key: 99,
            windows_per_bit: 2,
        };
        let embedder = TemporalEmbedder::new(cfg).expect("ok");
        let samples = sine_signal(44100 * 8, 880.0, 44100.0, 0.3);
        let watermarked = embedder.embed(&samples, b"W").expect("embed");
        assert_eq!(watermarked.len(), samples.len());
    }
}
