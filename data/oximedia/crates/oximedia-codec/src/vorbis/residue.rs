//! Vorbis residue coding (vector quantisation of spectral residuals).
//!
//! After the floor curve is subtracted, the spectral residuals are encoded
//! using one of three residue formats (0, 1, 2).  This module implements
//! the vector quantisation bookkeeping and the partition / classification
//! logic common to all three types.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

/// Residue format type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidueType {
    /// Type 0: un-interleaved channel vectors.
    Type0 = 0,
    /// Type 1: un-interleaved channel scalars.
    Type1 = 1,
    /// Type 2: interleaved across all channels.
    Type2 = 2,
}

impl Default for ResidueType {
    fn default() -> Self {
        Self::Type1
    }
}

/// Residue configuration header.
#[derive(Clone, Debug)]
pub struct ResidueConfig {
    /// Residue type (0, 1, or 2).
    pub residue_type: ResidueType,
    /// Begin bin index.
    pub begin: u32,
    /// End bin index.
    pub end: u32,
    /// Partition size.
    pub partition_size: u32,
    /// Number of classification stages.
    pub classifications: u8,
    /// Classbook number (Huffman VQ book index).
    pub classbook: u8,
}

impl Default for ResidueConfig {
    fn default() -> Self {
        Self {
            residue_type: ResidueType::Type1,
            begin: 0,
            end: 128,
            partition_size: 32,
            classifications: 4,
            classbook: 0,
        }
    }
}

/// Simplified residue vector quantiser.
///
/// Encodes residual spectral vectors using scalar quantisation (not a full
/// Vorbis VQ book) — sufficient for a reference implementation.
#[derive(Clone, Debug)]
pub struct ResidueEncoder {
    /// Encoder configuration.
    pub config: ResidueConfig,
    /// Quantisation step size.
    pub step: f64,
}

impl ResidueEncoder {
    /// Create a new residue encoder.
    #[must_use]
    pub fn new(config: ResidueConfig, step: f64) -> Self {
        Self {
            config,
            step: step.max(1e-6),
        }
    }

    /// Scalar-quantise a residue vector to i16 codes.
    #[must_use]
    pub fn quantise(&self, residue: &[f64]) -> Vec<i16> {
        residue
            .iter()
            .map(|&v| {
                let q = (v / self.step).round();
                q.clamp(i16::MIN as f64, i16::MAX as f64) as i16
            })
            .collect()
    }

    /// Dequantise i16 codes back to f64 residual.
    #[must_use]
    pub fn dequantise(&self, codes: &[i16]) -> Vec<f64> {
        codes.iter().map(|&c| f64::from(c) * self.step).collect()
    }

    /// Compute the maximum quantisation error for a round-tripped block.
    #[must_use]
    pub fn max_quant_error(&self, original: &[f64]) -> f64 {
        let quantised = self.quantise(original);
        let recovered = self.dequantise(&quantised);
        original
            .iter()
            .zip(recovered.iter())
            .map(|(&o, &r)| (o - r).abs())
            .fold(0.0_f64, f64::max)
    }
}

/// Residue decoder.
#[derive(Clone, Debug)]
pub struct ResidueDecoder {
    /// Decoder configuration.
    pub config: ResidueConfig,
    /// Quantisation step size (must match encoder).
    pub step: f64,
}

impl ResidueDecoder {
    /// Create a new residue decoder.
    #[must_use]
    pub fn new(config: ResidueConfig, step: f64) -> Self {
        Self {
            config,
            step: step.max(1e-6),
        }
    }

    /// Dequantise codes to f64 residual vector.
    #[must_use]
    pub fn decode(&self, codes: &[i16]) -> Vec<f64> {
        codes.iter().map(|&c| f64::from(c) * self.step).collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_type_default() {
        assert_eq!(ResidueType::default(), ResidueType::Type1);
    }

    #[test]
    fn test_residue_encoder_quantise_zero() {
        let enc = ResidueEncoder::new(ResidueConfig::default(), 0.5);
        let v = vec![0.0f64; 8];
        let q = enc.quantise(&v);
        assert!(q.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_residue_encoder_roundtrip() {
        let step = 0.25;
        let enc = ResidueEncoder::new(ResidueConfig::default(), step);
        let original = vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 2.0, -2.0];
        let q = enc.quantise(&original);
        let recovered = enc.dequantise(&q);
        for (&o, &r) in original.iter().zip(recovered.iter()) {
            assert!(
                (o - r).abs() <= step / 2.0 + 1e-9,
                "Error too large: orig={o}, rec={r}"
            );
        }
    }

    #[test]
    fn test_residue_max_quant_error_is_bounded() {
        let step = 1.0;
        let enc = ResidueEncoder::new(ResidueConfig::default(), step);
        let data = vec![3.7, -2.1, 0.4, -0.9];
        let err = enc.max_quant_error(&data);
        assert!(
            err <= step / 2.0 + 1e-9,
            "Max error {err} exceeds half step"
        );
    }

    #[test]
    fn test_residue_decoder_decode() {
        let step = 0.5;
        let dec = ResidueDecoder::new(ResidueConfig::default(), step);
        let codes: Vec<i16> = vec![2, -4, 0, 1];
        let out = dec.decode(&codes);
        assert!((out[0] - 1.0).abs() < 1e-9);
        assert!((out[1] - -2.0).abs() < 1e-9);
        assert!((out[2] - 0.0).abs() < 1e-9);
        assert!((out[3] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_residue_encoder_clamping() {
        let step = 1.0;
        let enc = ResidueEncoder::new(ResidueConfig::default(), step);
        let huge = vec![1e10f64, -1e10];
        let q = enc.quantise(&huge);
        // Should be clamped to i16 range
        assert_eq!(q[0], i16::MAX);
        assert_eq!(q[1], i16::MIN);
    }
}
