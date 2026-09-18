//! FLAC subframe decoding.
//!
//! Each FLAC subframe contains samples for one channel. Subframe types:
//!
//! - **CONSTANT** - All samples have the same value
//! - **VERBATIM** - Uncompressed samples
//! - **FIXED** - Fixed linear prediction (order 0-4)
//! - **LPC** - Linear predictive coding (order 1-32)
//!
//! # Subframe Structure
//!
//! - Zero padding bit (1 bit)
//! - Subframe type (6 bits)
//! - Wasted bits-per-sample flag (1 bit)
//! - Optional wasted bits-per-sample (unary coded)
//! - Subframe data

#![forbid(unsafe_code)]

use crate::AudioError;

/// Subframe type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubframeType {
    /// All samples have the same value.
    #[default]
    Constant,
    /// Uncompressed samples.
    Verbatim,
    /// Fixed linear prediction (order 0-4).
    Fixed(u8),
    /// Linear predictive coding (order 1-32).
    Lpc(u8),
}

impl SubframeType {
    /// Create from raw subframe type value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(SubframeType::Constant),
            1 => Some(SubframeType::Verbatim),
            // 2..=7 and 13..=31 are reserved
            8..=12 => Some(SubframeType::Fixed(value - 8)),
            32..=63 => Some(SubframeType::Lpc(value - 31)),
            _ => None,
        }
    }

    /// Get predictor order for this subframe type.
    #[must_use]
    pub fn order(self) -> u8 {
        match self {
            SubframeType::Constant | SubframeType::Verbatim => 0,
            SubframeType::Fixed(order) | SubframeType::Lpc(order) => order,
        }
    }

    /// Check if this type uses residual coding.
    #[must_use]
    pub fn has_residual(self) -> bool {
        matches!(self, SubframeType::Fixed(_) | SubframeType::Lpc(_))
    }

    /// Get number of warmup samples needed.
    #[must_use]
    pub fn warmup_count(self) -> usize {
        self.order() as usize
    }
}

/// Subframe header.
#[derive(Debug, Clone, Default)]
pub struct SubframeHeader {
    /// Subframe type.
    pub subframe_type: SubframeType,
    /// Number of wasted bits per sample.
    pub wasted_bits: u8,
    /// Effective bits per sample (original - wasted).
    pub effective_bps: u8,
}

impl SubframeHeader {
    /// Parse subframe header from bit reader.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(first_byte: u8, channel_bps: u8) -> Result<Self, AudioError> {
        // Zero padding bit must be 0
        if (first_byte & 0x80) != 0 {
            return Err(AudioError::InvalidData(
                "Subframe padding bit must be 0".into(),
            ));
        }

        // Subframe type (bits 1-6)
        let type_bits = (first_byte >> 1) & 0x3F;
        let subframe_type = SubframeType::from_value(type_bits).ok_or_else(|| {
            AudioError::InvalidData(format!("Invalid subframe type: {type_bits}"))
        })?;

        // Wasted bits flag (bit 0)
        let has_wasted = (first_byte & 0x01) != 0;
        let wasted_bits = if has_wasted {
            // Would need to read unary coded value from bit stream
            // For now, skeleton just returns 0
            0
        } else {
            0
        };

        let effective_bps = channel_bps.saturating_sub(wasted_bits);

        Ok(Self {
            subframe_type,
            wasted_bits,
            effective_bps,
        })
    }

    /// Check if wasted bits need to be read.
    #[must_use]
    pub fn has_wasted_bits(first_byte: u8) -> bool {
        (first_byte & 0x01) != 0
    }
}

/// Warmup samples for prediction.
#[derive(Debug, Clone, Default)]
pub struct WarmupSamples {
    /// Warmup sample values.
    pub samples: Vec<i32>,
    /// Number of bits per sample used.
    pub bits_per_sample: u8,
}

impl WarmupSamples {
    /// Create new warmup samples.
    #[must_use]
    pub fn new(count: usize, bps: u8) -> Self {
        Self {
            samples: Vec::with_capacity(count),
            bits_per_sample: bps,
        }
    }

    /// Add a warmup sample.
    pub fn push(&mut self, sample: i32) {
        self.samples.push(sample);
    }

    /// Get number of warmup samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get sample by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<i32> {
        self.samples.get(index).copied()
    }
}

/// LPC (Linear Predictive Coding) coefficients.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LpcCoefficients {
    /// Coefficient precision in bits.
    pub precision: u8,
    /// Right shift for prediction (`qlp_coeff_shift`).
    pub shift: i8,
    /// Quantized LP coefficients.
    pub coefficients: Vec<i32>,
}

impl LpcCoefficients {
    /// Create new LPC coefficients.
    #[must_use]
    pub fn new(order: usize) -> Self {
        Self {
            precision: 0,
            shift: 0,
            coefficients: Vec::with_capacity(order),
        }
    }

    /// Get order (number of coefficients).
    #[must_use]
    pub fn order(&self) -> usize {
        self.coefficients.len()
    }

    /// Predict next sample from previous samples.
    #[must_use]
    #[allow(dead_code, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn predict(&self, samples: &[i32]) -> i32 {
        if samples.len() < self.coefficients.len() {
            return 0;
        }

        let mut sum: i64 = 0;

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            let sample_idx = samples.len() - 1 - i;
            sum += i64::from(coeff) * i64::from(samples[sample_idx]);
        }

        (sum >> self.shift as u32) as i32
    }
}

/// Fixed prediction coefficients for each order.
pub mod fixed_coefficients {
    /// Order 0: no prediction.
    pub const ORDER_0: &[i32] = &[];
    /// Order 1: first difference.
    pub const ORDER_1: &[i32] = &[1];
    /// Order 2: second difference.
    pub const ORDER_2: &[i32] = &[2, -1];
    /// Order 3: third difference.
    pub const ORDER_3: &[i32] = &[3, -3, 1];
    /// Order 4: fourth difference.
    pub const ORDER_4: &[i32] = &[4, -6, 4, -1];

    /// Get coefficients for order.
    #[must_use]
    pub fn for_order(order: u8) -> &'static [i32] {
        match order {
            1 => ORDER_1,
            2 => ORDER_2,
            3 => ORDER_3,
            4 => ORDER_4,
            // 0 and other values have no coefficients
            _ => ORDER_0,
        }
    }
}

/// Residual data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualType {
    /// Rice coding with 4-bit parameter.
    #[default]
    Rice4,
    /// Rice coding with 5-bit parameter.
    Rice5,
}

impl ResidualType {
    /// Create from coding method value.
    #[must_use]
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0 => Some(ResidualType::Rice4),
            1 => Some(ResidualType::Rice5),
            _ => None,
        }
    }

    /// Get parameter bits for this type.
    #[must_use]
    pub fn parameter_bits(self) -> u8 {
        match self {
            ResidualType::Rice4 => 4,
            ResidualType::Rice5 => 5,
        }
    }

    /// Get escape code for this type.
    #[must_use]
    pub fn escape_code(self) -> u8 {
        match self {
            ResidualType::Rice4 => 0x0F,
            ResidualType::Rice5 => 0x1F,
        }
    }
}

/// FLAC subframe.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Subframe {
    /// Subframe header.
    pub header: SubframeHeader,
    /// Warmup samples (for prediction types).
    pub warmup: WarmupSamples,
    /// LPC coefficients (for LPC type).
    pub lpc: Option<LpcCoefficients>,
    /// Decoded samples.
    pub samples: Vec<i32>,
}

impl Subframe {
    /// Create a new empty subframe.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create subframe with header.
    #[must_use]
    pub fn with_header(header: SubframeHeader) -> Self {
        let warmup_count = header.subframe_type.warmup_count();
        Self {
            header,
            warmup: WarmupSamples::new(warmup_count, 0),
            lpc: None,
            samples: Vec::new(),
        }
    }

    /// Get subframe type.
    #[must_use]
    pub fn subframe_type(&self) -> SubframeType {
        self.header.subframe_type
    }

    /// Get predictor order.
    #[must_use]
    pub fn order(&self) -> u8 {
        self.header.subframe_type.order()
    }

    /// Decode samples using fixed prediction.
    #[allow(clippy::cast_possible_truncation)]
    pub fn decode_fixed(&mut self, residuals: &[i32]) {
        let coeffs = fixed_coefficients::for_order(self.order());

        // Copy warmup samples
        self.samples.clear();
        self.samples.extend_from_slice(&self.warmup.samples);

        // Decode residuals using fixed prediction
        for &residual in residuals {
            if coeffs.is_empty() {
                self.samples.push(residual);
            } else {
                let mut prediction: i64 = 0;
                for (i, &coeff) in coeffs.iter().enumerate() {
                    let sample_idx = self.samples.len() - 1 - i;
                    prediction += i64::from(coeff) * i64::from(self.samples[sample_idx]);
                }
                self.samples.push(prediction as i32 + residual);
            }
        }

        // Shift back if wasted bits
        if self.header.wasted_bits > 0 {
            let shift = u32::from(self.header.wasted_bits);
            for sample in &mut self.samples {
                *sample <<= shift;
            }
        }
    }

    /// Decode samples using LPC prediction.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn decode_lpc(&mut self, residuals: &[i32]) {
        let Some(lpc) = &self.lpc else { return };

        let shift = lpc.shift as u32;

        // Copy warmup samples
        self.samples.clear();
        self.samples.extend_from_slice(&self.warmup.samples);

        // Decode residuals using LPC prediction
        for &residual in residuals {
            let mut prediction: i64 = 0;
            for (i, &coeff) in lpc.coefficients.iter().enumerate() {
                let sample_idx = self.samples.len() - 1 - i;
                prediction += i64::from(coeff) * i64::from(self.samples[sample_idx]);
            }
            let predicted = (prediction >> shift) as i32;
            self.samples.push(predicted.wrapping_add(residual));
        }

        // Shift back if wasted bits
        if self.header.wasted_bits > 0 {
            let shift = u32::from(self.header.wasted_bits);
            for sample in &mut self.samples {
                *sample <<= shift;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subframe_type() {
        assert_eq!(SubframeType::from_value(0), Some(SubframeType::Constant));
        assert_eq!(SubframeType::from_value(1), Some(SubframeType::Verbatim));
        assert_eq!(SubframeType::from_value(8), Some(SubframeType::Fixed(0)));
        assert_eq!(SubframeType::from_value(12), Some(SubframeType::Fixed(4)));
        assert_eq!(SubframeType::from_value(32), Some(SubframeType::Lpc(1)));
        assert_eq!(SubframeType::from_value(63), Some(SubframeType::Lpc(32)));
        assert_eq!(SubframeType::from_value(2), None); // Reserved
    }

    #[test]
    fn test_subframe_type_order() {
        assert_eq!(SubframeType::Constant.order(), 0);
        assert_eq!(SubframeType::Verbatim.order(), 0);
        assert_eq!(SubframeType::Fixed(3).order(), 3);
        assert_eq!(SubframeType::Lpc(12).order(), 12);
    }

    #[test]
    fn test_subframe_type_has_residual() {
        assert!(!SubframeType::Constant.has_residual());
        assert!(!SubframeType::Verbatim.has_residual());
        assert!(SubframeType::Fixed(2).has_residual());
        assert!(SubframeType::Lpc(8).has_residual());
    }

    #[test]
    fn test_warmup_samples() {
        let mut warmup = WarmupSamples::new(4, 16);
        assert!(warmup.is_empty());

        warmup.push(100);
        warmup.push(200);
        assert_eq!(warmup.len(), 2);
        assert_eq!(warmup.get(0), Some(100));
        assert_eq!(warmup.get(1), Some(200));
        assert_eq!(warmup.get(2), None);
    }

    #[test]
    fn test_lpc_coefficients() {
        let mut lpc = LpcCoefficients::new(4);
        lpc.coefficients = vec![1, 2, 3, 4];
        lpc.shift = 0;
        assert_eq!(lpc.order(), 4);
    }

    #[test]
    fn test_fixed_coefficients() {
        assert!(fixed_coefficients::for_order(0).is_empty());
        assert_eq!(fixed_coefficients::for_order(1), &[1]);
        assert_eq!(fixed_coefficients::for_order(2), &[2, -1]);
        assert_eq!(fixed_coefficients::for_order(3), &[3, -3, 1]);
        assert_eq!(fixed_coefficients::for_order(4), &[4, -6, 4, -1]);
    }

    #[test]
    fn test_residual_type() {
        assert_eq!(ResidualType::from_value(0), Some(ResidualType::Rice4));
        assert_eq!(ResidualType::from_value(1), Some(ResidualType::Rice5));
        assert_eq!(ResidualType::from_value(2), None);

        assert_eq!(ResidualType::Rice4.parameter_bits(), 4);
        assert_eq!(ResidualType::Rice5.parameter_bits(), 5);
        assert_eq!(ResidualType::Rice4.escape_code(), 0x0F);
        assert_eq!(ResidualType::Rice5.escape_code(), 0x1F);
    }

    #[test]
    fn test_subframe() {
        let header = SubframeHeader {
            subframe_type: SubframeType::Fixed(2),
            wasted_bits: 0,
            effective_bps: 16,
        };
        let subframe = Subframe::with_header(header);
        assert_eq!(subframe.order(), 2);
        assert_eq!(subframe.subframe_type(), SubframeType::Fixed(2));
    }

    #[test]
    fn test_subframe_decode_fixed() {
        let header = SubframeHeader {
            subframe_type: SubframeType::Fixed(1),
            wasted_bits: 0,
            effective_bps: 16,
        };
        let mut subframe = Subframe::with_header(header);
        subframe.warmup.samples = vec![100];

        // First difference: each sample = previous + residual
        let residuals = vec![10, 20, 30];
        subframe.decode_fixed(&residuals);

        assert_eq!(subframe.samples, vec![100, 110, 130, 160]);
    }
}
