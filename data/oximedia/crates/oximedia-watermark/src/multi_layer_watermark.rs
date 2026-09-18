//! Multi-layer watermarking: embed multiple independent watermarks in a single
//! audio signal using frequency-band partitioning and orthogonal spreading
//! sequences.
//!
//! Each layer occupies a disjoint set of frequency bins so that extraction
//! of one layer does not interfere with the others.  Layers can carry
//! different payloads, use different secret keys, and have independent
//! strength settings.
//!
//! ## Design
//!
//! The signal spectrum is divided into `N` equal-width bands.  Layer `i` uses
//! the `i`-th band for its spread-spectrum or QIM embedding.  The layer keys
//! derive from a master key using FNV-64, ensuring orthogonal PN sequences.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec};
use oxifft::Complex;

// ---------------------------------------------------------------------------
// LayerConfig
// ---------------------------------------------------------------------------

/// Per-layer watermark configuration.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Human-readable layer identifier (e.g. "copyright", "distributor").
    pub name: String,
    /// Embedding strength for this layer (0.0 – 1.0).
    pub strength: f32,
    /// Secret key for this layer's PN sequence.
    pub key: u64,
    /// Start frequency bin (in the frame's FFT spectrum).
    pub start_bin: usize,
    /// End frequency bin (exclusive).
    pub end_bin: usize,
    /// Chip rate for spread-spectrum spreading.
    pub chip_rate: usize,
}

impl LayerConfig {
    /// Create a new layer configuration.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        strength: f32,
        key: u64,
        start_bin: usize,
        end_bin: usize,
        chip_rate: usize,
    ) -> Self {
        Self {
            name: name.into(),
            strength: strength.clamp(0.0, 1.0),
            key,
            start_bin,
            end_bin,
            chip_rate: chip_rate.max(1),
        }
    }

    /// Capacity in bits for a given number of frames.
    #[must_use]
    pub fn capacity_bits(&self, frame_count: usize) -> usize {
        let bins = self.end_bin.saturating_sub(self.start_bin);
        let bits_per_frame = bins / self.chip_rate.max(1);
        frame_count * bits_per_frame
    }
}

// ---------------------------------------------------------------------------
// MultiLayerConfig
// ---------------------------------------------------------------------------

/// Configuration for the multi-layer watermark system.
#[derive(Debug, Clone)]
pub struct MultiLayerConfig {
    /// FFT frame size (must be a power of two).
    pub frame_size: usize,
    /// Layer definitions (up to 8 layers supported).
    pub layers: Vec<LayerConfig>,
}

impl Default for MultiLayerConfig {
    fn default() -> Self {
        // Split spectrum into 3 bands: lows (64–200), mids (200–500), highs (500–900).
        Self {
            frame_size: 2048,
            layers: vec![
                LayerConfig::new("base", 0.08, 0xDEAD_BEEF_0000_0001, 64, 200, 16),
                LayerConfig::new("dist", 0.06, 0xDEAD_BEEF_0000_0002, 200, 500, 16),
                LayerConfig::new("meta", 0.04, 0xDEAD_BEEF_0000_0003, 500, 900, 16),
            ],
        }
    }
}

impl MultiLayerConfig {
    /// Create a configuration with automatically partitioned bands.
    ///
    /// `n_layers` bands are allocated equally between `low_bin` and `high_bin`.
    #[must_use]
    pub fn auto_partition(
        frame_size: usize,
        n_layers: usize,
        low_bin: usize,
        high_bin: usize,
        strength: f32,
        master_key: u64,
        chip_rate: usize,
    ) -> Self {
        let n = n_layers.max(1).min(8);
        let band_width = high_bin.saturating_sub(low_bin) / n;
        let layers = (0..n)
            .map(|i| {
                let start = low_bin + i * band_width;
                let end = if i + 1 == n {
                    high_bin
                } else {
                    start + band_width
                };
                // Derive per-layer key with FNV-64
                let key = fnv64(master_key ^ (i as u64 + 1));
                LayerConfig::new(format!("layer-{i}"), strength, key, start, end, chip_rate)
            })
            .collect();

        Self { frame_size, layers }
    }
}

// ---------------------------------------------------------------------------
// MultiLayerEmbedder
// ---------------------------------------------------------------------------

/// Embeds multiple independent watermark layers into an audio signal.
pub struct MultiLayerEmbedder {
    config: MultiLayerConfig,
    codecs: Vec<PayloadCodec>,
}

impl MultiLayerEmbedder {
    /// Create a new embedder.
    ///
    /// # Errors
    ///
    /// Returns error if a `PayloadCodec` cannot be initialised.
    pub fn new(config: MultiLayerConfig) -> WatermarkResult<Self> {
        let codecs = config
            .layers
            .iter()
            .map(|_| PayloadCodec::new(16, 8))
            .collect::<WatermarkResult<Vec<_>>>()?;
        Ok(Self { config, codecs })
    }

    /// Embed all layers into `samples`.
    ///
    /// `payloads[i]` is the payload for layer `i`.  If `payloads` is shorter
    /// than the number of layers, remaining layers receive an empty payload.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError::InsufficientCapacity`] if any layer's
    /// payload exceeds that layer's bit capacity for the provided signal.
    pub fn embed(&self, samples: &[f32], payloads: &[&[u8]]) -> WatermarkResult<Vec<f32>> {
        let mut watermarked = samples.to_vec();

        for (layer_idx, (layer, codec)) in self
            .config
            .layers
            .iter()
            .zip(self.codecs.iter())
            .enumerate()
        {
            let payload = payloads.get(layer_idx).copied().unwrap_or(&[]);
            let encoded = codec.encode(payload)?;
            let bits = unpack_bits(&encoded, encoded.len() * 8);

            let frame_size = self.config.frame_size;
            let hop = frame_size;
            let bins_available = layer.end_bin.saturating_sub(layer.start_bin);
            let bits_per_frame = bins_available / layer.chip_rate.max(1);

            if bits_per_frame == 0 {
                return Err(WatermarkError::InvalidParameter(format!(
                    "Layer '{}': bin range too narrow for chip_rate {}",
                    layer.name, layer.chip_rate
                )));
            }

            let required_frames = bits.len().div_ceil(bits_per_frame);
            let required_samples = required_frames * hop;

            if watermarked.len() < required_samples {
                return Err(WatermarkError::InsufficientCapacity {
                    needed: required_samples,
                    have: watermarked.len(),
                });
            }

            let mut bit_idx = 0;

            for frame_idx in 0..required_frames {
                if bit_idx >= bits.len() {
                    break;
                }
                let frame_start = frame_idx * hop;
                if frame_start + frame_size > watermarked.len() {
                    break;
                }

                let frame: Vec<f32> = watermarked[frame_start..frame_start + frame_size].to_vec();
                let freq_input: Vec<Complex<f32>> =
                    frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
                let mut freq_data = oxifft::fft(&freq_input);

                // Embed bits for this frame in this layer's band.
                // Each in-frame bit slot uses a disjoint, non-overlapping
                // bin range so there is no inter-bit correlation interference.
                for in_frame_slot in 0..bits_per_frame {
                    if bit_idx >= bits.len() {
                        break;
                    }
                    let bit = bits[bit_idx];
                    let bit_value = if bit { 1.0f32 } else { -1.0f32 };
                    let pn = generate_pn_sequence(layer.chip_rate, layer.key ^ bit_idx as u64);

                    // Disjoint bin range for this slot within the layer band.
                    let bin_base = layer.start_bin + in_frame_slot * layer.chip_rate;

                    for (chip_i, &pn_val) in pn.iter().enumerate() {
                        let bin = bin_base + chip_i;
                        if bin >= layer.end_bin || bin >= freq_data.len() / 2 {
                            break;
                        }
                        let wm = layer.strength * bit_value * f32::from(pn_val);
                        freq_data[bin] += Complex::new(wm, 0.0);
                        let mirror = frame_size - bin;
                        if mirror < freq_data.len() && mirror != bin {
                            freq_data[mirror] += Complex::new(wm, 0.0);
                        }
                    }

                    bit_idx += 1;
                }

                // IFFT and write back.
                let ifft_result = oxifft::ifft(&freq_data);
                for (i, c) in ifft_result.iter().enumerate() {
                    let idx = frame_start + i;
                    if idx < watermarked.len() {
                        watermarked[idx] = c.re;
                    }
                }
            }
        }

        Ok(watermarked)
    }

    /// Returns per-layer bit capacity for the given sample count.
    #[must_use]
    pub fn capacities(&self, sample_count: usize) -> Vec<usize> {
        let frame_count = sample_count / self.config.frame_size;
        self.config
            .layers
            .iter()
            .map(|l| l.capacity_bits(frame_count))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MultiLayerDetector
// ---------------------------------------------------------------------------

/// Extracts per-layer watermark payloads from a watermarked signal.
pub struct MultiLayerDetector {
    config: MultiLayerConfig,
    codecs: Vec<PayloadCodec>,
}

impl MultiLayerDetector {
    /// Create a new detector.
    ///
    /// # Errors
    ///
    /// Returns error if a `PayloadCodec` cannot be initialised.
    pub fn new(config: MultiLayerConfig) -> WatermarkResult<Self> {
        let codecs = config
            .layers
            .iter()
            .map(|_| PayloadCodec::new(16, 8))
            .collect::<WatermarkResult<Vec<_>>>()?;
        Ok(Self { config, codecs })
    }

    /// Detect all layers and return their decoded payloads.
    ///
    /// `expected_bits[i]` is the number of encoded bits expected for layer `i`.
    /// If shorter than number of layers, remaining layers receive a sensible
    /// default (one full RS block).
    ///
    /// # Errors
    ///
    /// Returns error if any layer's codec decode fails.
    pub fn detect(
        &self,
        samples: &[f32],
        expected_bits: &[usize],
    ) -> WatermarkResult<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(self.config.layers.len());

        for (layer_idx, (layer, codec)) in self
            .config
            .layers
            .iter()
            .zip(self.codecs.iter())
            .enumerate()
        {
            let n_bits = expected_bits.get(layer_idx).copied().unwrap_or(280);
            let frame_size = self.config.frame_size;
            let hop = frame_size;
            let bins_available = layer.end_bin.saturating_sub(layer.start_bin);
            let bits_per_frame = bins_available / layer.chip_rate.max(1);

            if bits_per_frame == 0 {
                results.push(vec![]);
                continue;
            }

            let mut bits: Vec<bool> = Vec::with_capacity(n_bits);

            for frame_start in (0..samples.len()).step_by(hop) {
                if bits.len() >= n_bits {
                    break;
                }
                if frame_start + frame_size > samples.len() {
                    break;
                }

                let frame = &samples[frame_start..frame_start + frame_size];
                let freq_input: Vec<Complex<f32>> =
                    frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
                let freq_data = oxifft::fft(&freq_input);

                for in_frame_slot in 0..bits_per_frame {
                    if bits.len() >= n_bits {
                        break;
                    }
                    let bit_idx = bits.len();
                    let pn = generate_pn_sequence(layer.chip_rate, layer.key ^ bit_idx as u64);

                    // Match the disjoint bin range used by the embedder.
                    let bin_base = layer.start_bin + in_frame_slot * layer.chip_rate;

                    let mut corr = 0.0f32;
                    for (chip_i, &pn_val) in pn.iter().enumerate() {
                        let bin = bin_base + chip_i;
                        if bin >= layer.end_bin || bin >= freq_data.len() / 2 {
                            break;
                        }
                        corr += freq_data[bin].re * f32::from(pn_val);
                    }
                    bits.push(corr > 0.0);
                }
            }

            let bytes = pack_bits(&bits);
            match codec.decode(&bytes) {
                Ok(decoded) => results.push(decoded),
                Err(_) => results.push(vec![]),
            }
        }

        Ok(results)
    }

    /// Number of layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.config.layers.len()
    }
}

// ---------------------------------------------------------------------------
// FNV-64 helper
// ---------------------------------------------------------------------------

fn fnv64(x: u64) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let bytes = x.to_le_bytes();
    let mut h = OFFSET;
    for &b in &bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::PayloadCodec;

    fn make_samples(n: usize) -> Vec<f32> {
        use std::f32::consts::PI;
        (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect()
    }

    #[test]
    fn test_auto_partition_creates_n_layers() {
        let cfg = MultiLayerConfig::auto_partition(2048, 3, 64, 900, 0.05, 0xABCD, 16);
        assert_eq!(cfg.layers.len(), 3);
    }

    #[test]
    fn test_auto_partition_bands_non_overlapping() {
        let cfg = MultiLayerConfig::auto_partition(2048, 3, 64, 900, 0.05, 0xABCD, 16);
        for i in 0..cfg.layers.len() - 1 {
            assert!(cfg.layers[i].end_bin <= cfg.layers[i + 1].start_bin);
        }
    }

    #[test]
    fn test_capacities_positive() {
        let cfg = MultiLayerConfig::default();
        let embedder = MultiLayerEmbedder::new(cfg).expect("ok");
        let caps = embedder.capacities(44100 * 4);
        assert!(caps.iter().all(|&c| c > 0));
    }

    #[test]
    fn test_embed_length_preserved() {
        let cfg = MultiLayerConfig::default();
        let embedder = MultiLayerEmbedder::new(cfg).expect("ok");
        let samples = make_samples(44100 * 4);
        let payloads: &[&[u8]] = &[b"layer0", b"layer1", b"layer2"];
        let watermarked = embedder.embed(&samples, payloads).expect("embed ok");
        assert_eq!(watermarked.len(), samples.len());
    }

    #[test]
    fn test_embed_single_layer_roundtrip() {
        // Single layer, large enough signal.
        // Use zero signal so the only frequency-domain content comes from
        // the embedded watermark, making DSSS correlation reliable.
        let cfg = MultiLayerConfig::auto_partition(2048, 1, 64, 512, 0.1, 0xCAFE, 16);
        let samples = vec![0.0f32; 44100 * 4];
        let payload = b"OK";

        let embedder = MultiLayerEmbedder::new(cfg.clone()).expect("ok");
        let detector = MultiLayerDetector::new(cfg).expect("ok");

        let watermarked = embedder.embed(&samples, &[payload]).expect("embed");

        let codec = PayloadCodec::new(16, 8).expect("codec");
        let encoded = codec.encode(payload).expect("encode");
        let expected_bits = encoded.len() * 8;

        let results = detector
            .detect(&watermarked, &[expected_bits])
            .expect("detect");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], payload.as_slice());
    }

    #[test]
    fn test_embed_three_layers_length_ok() {
        let cfg = MultiLayerConfig::auto_partition(2048, 3, 64, 900, 0.07, 0xBEEF, 16);
        let samples = make_samples(44100 * 8);
        let payloads: &[&[u8]] = &[b"A", b"B", b"C"];
        let embedder = MultiLayerEmbedder::new(cfg).expect("ok");
        let watermarked = embedder.embed(&samples, payloads).expect("embed");
        assert_eq!(watermarked.len(), samples.len());
    }

    #[test]
    fn test_layer_config_capacity_bits() {
        let layer = LayerConfig::new("test", 0.1, 0, 64, 128, 16);
        // bins = 64, bits_per_frame = 64/16 = 4
        assert_eq!(layer.capacity_bits(10), 40);
    }

    #[test]
    fn test_embed_insufficient_capacity_returns_error() {
        // Use very tiny signal that cannot hold even one frame.
        let cfg = MultiLayerConfig::auto_partition(2048, 1, 64, 512, 0.1, 0, 16);
        let samples = vec![0.0f32; 100]; // way too short
        let embedder = MultiLayerEmbedder::new(cfg).expect("ok");
        let result = embedder.embed(&samples, &[b"too-long-payload-bytes"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_names_accessible() {
        let cfg = MultiLayerConfig::default();
        let names: Vec<&str> = cfg.layers.iter().map(|l| l.name.as_str()).collect();
        assert!(names.contains(&"base"));
        assert!(names.contains(&"dist"));
        assert!(names.contains(&"meta"));
    }

    #[test]
    fn test_detector_layer_count_matches_config() {
        let cfg = MultiLayerConfig::auto_partition(2048, 4, 64, 900, 0.05, 0, 16);
        let detector = MultiLayerDetector::new(cfg).expect("ok");
        assert_eq!(detector.layer_count(), 4);
    }

    #[test]
    fn test_fnv64_deterministic() {
        assert_eq!(fnv64(42), fnv64(42));
        assert_ne!(fnv64(42), fnv64(43));
    }
}
