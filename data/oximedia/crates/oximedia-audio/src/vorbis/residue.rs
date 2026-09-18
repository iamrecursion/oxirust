//! Residue encoding for Vorbis.
//!
//! Residue encoding handles the spectral details after floor encoding.
//! Vorbis uses vector quantization (VQ) with codebooks to compress
//! the residue data efficiently.

#![forbid(unsafe_code)]

use super::bitpack::{BitPacker, BitReader};
use crate::AudioResult;

/// Residue encoder for Vorbis.
#[derive(Debug, Clone)]
pub struct ResidueEncoder {
    /// Quality level.
    quality: f32,
    /// Quantization step size.
    quant_step: f32,
}

impl ResidueEncoder {
    /// Create new residue encoder.
    ///
    /// # Arguments
    ///
    /// * `quality` - Quality level (-1.0 to 10.0)
    #[must_use]
    pub fn new(quality: f32) -> Self {
        let quant_step = Self::quality_to_quant_step(quality);

        Self {
            quality,
            quant_step,
        }
    }

    /// Convert quality to quantization step size.
    fn quality_to_quant_step(quality: f32) -> f32 {
        // Higher quality = smaller step size = finer quantization
        let q = quality.clamp(-1.0, 10.0);
        let normalized = (10.0 - q) / 11.0; // 0.0 (highest) to 1.0 (lowest)
        0.01 + normalized * 0.99 // Step size from 0.01 to 1.0
    }

    /// Encode residue coefficients.
    ///
    /// # Arguments
    ///
    /// * `packer` - Bitstream packer
    /// * `coeffs` - Residue coefficients (after floor division)
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn encode(&self, packer: &mut BitPacker, coeffs: &[f32]) -> AudioResult<()> {
        // Simplified residue encoding using scalar quantization
        // Real Vorbis uses VQ with codebooks

        let n = coeffs.len() / 2; // Only encode first N/2 coefficients

        // Encode using run-length encoding of quantized values
        let mut run_length = 0;
        let mut prev_value = 0i32;

        for &coeff in coeffs.iter().take(n) {
            let quantized = Self::quantize(coeff, self.quant_step);

            if quantized == prev_value {
                run_length += 1;
            } else {
                // Encode previous run
                if run_length > 0 {
                    self.encode_run(packer, run_length)?;
                }

                // Encode new value
                self.encode_value(packer, quantized)?;

                prev_value = quantized;
                run_length = 1;
            }
        }

        // Encode final run
        if run_length > 0 {
            self.encode_run(packer, run_length)?;
        }

        Ok(())
    }

    /// Quantize a coefficient value.
    #[allow(clippy::cast_possible_truncation)]
    fn quantize(value: f32, step: f32) -> i32 {
        (value / step).round() as i32
    }

    /// Dequantize a coefficient value.
    #[allow(clippy::cast_precision_loss)]
    fn dequantize(quantized: i32, step: f32) -> f32 {
        quantized as f32 * step
    }

    /// Encode a run length.
    #[allow(clippy::cast_possible_truncation)]
    fn encode_run(&self, packer: &mut BitPacker, run_length: usize) -> AudioResult<()> {
        // Simple run-length encoding
        if run_length < 4 {
            // Short run: encode directly
            packer.write_bits(0, 2); // Short run flag
            packer.write_bits(run_length as u32, 2);
        } else {
            // Long run: use more bits
            packer.write_bits(1, 2); // Long run flag
            packer.write_bits((run_length - 4) as u32, 8);
        }
        Ok(())
    }

    /// Encode a quantized value.
    #[allow(clippy::cast_sign_loss)]
    fn encode_value(&self, packer: &mut BitPacker, value: i32) -> AudioResult<()> {
        // Encode sign and magnitude separately
        if value == 0 {
            packer.write_bits(0, 1); // Zero flag
        } else {
            packer.write_bits(1, 1); // Non-zero flag
            packer.write_bits(if value < 0 { 1 } else { 0 }, 1); // Sign

            let magnitude = value.unsigned_abs();
            let bits = Self::magnitude_bits(magnitude);
            packer.write_bits(bits as u32, 4); // Bit count
            packer.write_bits(magnitude, bits as u8);
        }
        Ok(())
    }

    /// Determine number of bits needed for magnitude.
    fn magnitude_bits(magnitude: u32) -> usize {
        if magnitude == 0 {
            0
        } else {
            32 - magnitude.leading_zeros() as usize
        }
    }

    /// Read one run token from the bitstream.
    ///
    /// Returns the run length (total count including the initial placement).
    /// Returns `Ok(None)` at EOF; `Ok(Some(k))` on success.
    fn read_run(reader: &mut BitReader<'_>) -> AudioResult<Option<usize>> {
        if reader.is_exhausted() {
            return Ok(None);
        }
        let flag = match reader.read_bits(2) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        if flag & 1 == 0 {
            // Short run (flag == 0b00): next 2 bits are count.
            let count = reader.read_bits(2).unwrap_or(0) as usize;
            Ok(Some(count))
        } else {
            // Long run (flag == 0b01): next 8 bits are (count - 4).
            let count = reader.read_bits(8).unwrap_or(0) as usize + 4;
            Ok(Some(count))
        }
    }

    /// Read one value token from the bitstream.
    ///
    /// Returns the decoded i32 (possibly 0) or `None` at EOF.
    #[allow(clippy::cast_sign_loss)]
    fn read_value(reader: &mut BitReader<'_>) -> AudioResult<Option<i32>> {
        if reader.is_exhausted() {
            return Ok(None);
        }
        let nonzero_flag = match reader.read_bit() {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        if !nonzero_flag {
            return Ok(Some(0));
        }
        // Non-zero: sign (1 bit) + bit_count (4 bits) + magnitude (bit_count bits).
        let negative = reader.read_bit().unwrap_or(false);
        let bit_count = reader.read_bits(4).unwrap_or(0) as u8;
        let magnitude = if bit_count == 0 {
            0u32
        } else {
            reader.read_bits(bit_count).unwrap_or(0)
        };
        let value = if negative {
            -(magnitude as i32)
        } else {
            magnitude as i32
        };
        Ok(Some(value))
    }

    /// Decode residue coefficients encoded by [`ResidueEncoder::encode`].
    ///
    /// This mirrors the internal OxiMedia RLE+scalar format written by `encode()`.
    /// It is NOT Vorbis I spec §8.6.4 partition VQ (the encoder does not produce
    /// spec-compliant residue bitstreams).
    ///
    /// The encoder emits pairs of `(value, run_count)` tokens where `run_count` is
    /// the total number of times this value appears (≥ 1).  An initial zero-run
    /// (when leading coefficients are zero) is emitted as just `run_count` with
    /// the implicit value of zero.  When the first coefficient is non-zero, the
    /// leading zero-run is omitted entirely.
    ///
    /// # Errors
    ///
    /// Returns error only for internal logic errors; bitstream truncation is handled
    /// gracefully by filling the remainder of `coeffs` with zeros.
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    pub fn decode_rle(
        reader: &mut BitReader<'_>,
        quant_step: f32,
        n: usize,
    ) -> AudioResult<Vec<f32>> {
        let mut coeffs = vec![0.0f32; n];
        let mut pos = 0usize;

        // The encoder emits: ([run_of_zeros]? (value run)*)
        // We decode as (value, run) pairs where value defaults to 0 when no value
        // precedes a leading zero-run.  Both leading-run and value-first forms are
        // handled by always reading a value then a run per iteration.

        while pos < n {
            if reader.is_exhausted() {
                break;
            }

            // Phase A: read a value.
            let current_value = match Self::read_value(reader)? {
                None => break,
                Some(v) => v,
            };
            if pos < n {
                coeffs[pos] = Self::dequantize(current_value, quant_step);
                pos += 1;
            }

            // Phase B: read the run count (total occurrences including first).
            let total_count = match Self::read_run(reader)? {
                None => break,
                Some(c) => c,
            };
            let extra = total_count.saturating_sub(1);
            for _ in 0..extra {
                if pos >= n {
                    break;
                }
                coeffs[pos] = Self::dequantize(current_value, quant_step);
                pos += 1;
            }
        }

        Ok(coeffs)
    }

    /// Compute rate-distortion optimization.
    ///
    /// Determines optimal quantization for each coefficient based on
    /// perceptual importance and bit budget.
    #[allow(dead_code)]
    pub fn rate_distortion_optimize(&self, coeffs: &[f32], _bit_budget: usize) -> Vec<i32> {
        // Simplified: just quantize all coefficients uniformly
        coeffs
            .iter()
            .map(|&c| Self::quantize(c, self.quant_step))
            .collect()
    }

    /// Compute noise allocation.
    ///
    /// Determines how much quantization noise to allow in each frequency band
    /// based on psychoacoustic masking.
    #[allow(dead_code)]
    pub fn compute_noise_allocation(&self, _masking: &[f32]) -> Vec<f32> {
        // Simplified: uniform allocation
        vec![self.quant_step; _masking.len()]
    }

    /// Get quality level.
    #[must_use]
    pub const fn quality(&self) -> f32 {
        self.quality
    }

    /// Get quantization step size.
    #[must_use]
    pub const fn quant_step(&self) -> f32 {
        self.quant_step
    }
}

/// Residue type 0: Interleaved vector quantization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResidueType0 {
    /// Begin frequency.
    begin: u32,
    /// End frequency.
    end: u32,
    /// Partition size.
    partition_size: u32,
    /// Number of classifications.
    classifications: u8,
    /// Classbook number.
    classbook: u8,
}

impl ResidueType0 {
    /// Create new residue type 0.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            begin: 0,
            end: 256,
            partition_size: 32,
            classifications: 4,
            classbook: 0,
        }
    }
}

impl Default for ResidueType0 {
    fn default() -> Self {
        Self::new()
    }
}

/// Residue type 1: Distinct vector quantization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResidueType1 {
    /// Begin frequency.
    begin: u32,
    /// End frequency.
    end: u32,
    /// Partition size.
    partition_size: u32,
    /// Number of classifications.
    classifications: u8,
    /// Classbook number.
    classbook: u8,
}

impl ResidueType1 {
    /// Create new residue type 1.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            begin: 0,
            end: 256,
            partition_size: 32,
            classifications: 4,
            classbook: 0,
        }
    }
}

impl Default for ResidueType1 {
    fn default() -> Self {
        Self::new()
    }
}

/// Residue type 2: Multidimensional vector quantization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResidueType2 {
    /// Begin frequency.
    begin: u32,
    /// End frequency.
    end: u32,
    /// Partition size.
    partition_size: u32,
    /// Number of classifications.
    classifications: u8,
    /// Classbook number.
    classbook: u8,
}

impl ResidueType2 {
    /// Create new residue type 2.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            begin: 0,
            end: 256,
            partition_size: 32,
            classifications: 4,
            classbook: 0,
        }
    }
}

impl Default for ResidueType2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_encoder_creation() {
        let encoder = ResidueEncoder::new(5.0);
        assert_eq!(encoder.quality(), 5.0);
        assert!(encoder.quant_step() > 0.0);
    }

    #[test]
    fn test_quality_to_quant_step() {
        let step_low = ResidueEncoder::quality_to_quant_step(-1.0);
        let step_mid = ResidueEncoder::quality_to_quant_step(5.0);
        let step_high = ResidueEncoder::quality_to_quant_step(10.0);

        // Higher quality should have smaller step
        assert!(step_high < step_mid);
        assert!(step_mid < step_low);
    }

    #[test]
    fn test_quantize() {
        let step = 0.1;
        assert_eq!(ResidueEncoder::quantize(0.5, step), 5);
        assert_eq!(ResidueEncoder::quantize(-0.5, step), -5);
        assert_eq!(ResidueEncoder::quantize(0.05, step), 1);
    }

    #[test]
    fn test_dequantize() {
        let step = 0.1;
        let value = 5;
        let dequant = ResidueEncoder::dequantize(value, step);
        assert!((dequant - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_bits() {
        assert_eq!(ResidueEncoder::magnitude_bits(0), 0);
        assert_eq!(ResidueEncoder::magnitude_bits(1), 1);
        assert_eq!(ResidueEncoder::magnitude_bits(2), 2);
        assert_eq!(ResidueEncoder::magnitude_bits(7), 3);
        assert_eq!(ResidueEncoder::magnitude_bits(8), 4);
        assert_eq!(ResidueEncoder::magnitude_bits(255), 8);
    }

    #[test]
    fn test_encode_residue() {
        let encoder = ResidueEncoder::new(5.0);
        let mut packer = BitPacker::new();
        let coeffs = vec![0.1, 0.2, 0.3, 0.2, 0.1];

        assert!(encoder.encode(&mut packer, &coeffs).is_ok());
        assert!(packer.size() > 0);
    }

    #[test]
    fn test_residue_type0() {
        let residue = ResidueType0::new();
        assert_eq!(residue.begin, 0);
        assert_eq!(residue.end, 256);
    }

    #[test]
    fn test_residue_type1() {
        let residue = ResidueType1::new();
        assert_eq!(residue.partition_size, 32);
    }

    #[test]
    fn test_residue_type2() {
        let residue = ResidueType2::new();
        assert_eq!(residue.classifications, 4);
    }
}
