//! Forensic watermark detection for DRM.
//!
//! Provides the complementary detection side to `watermark_embed.rs`:
//! - Frame-level watermark extraction from pixel data
//! - Statistical correlation detector (spread-spectrum approach)
//! - Multi-frame accumulation for robust detection under noise
//! - Payload recovery from extracted watermark bits
//! - Confidence scoring and detection thresholds

#![allow(clippy::cast_precision_loss)]

use crate::watermark_embed::{WatermarkPayload, WatermarkStrength};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Watermark detection errors.
#[derive(Error, Debug)]
pub enum WatermarkDetectError {
    #[error("no watermark detected (confidence {confidence:.4} below threshold {threshold:.4})")]
    NotDetected { confidence: f64, threshold: f64 },

    #[error("insufficient data: need at least {needed} samples, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("accumulator overflow: too many frames accumulated ({0})")]
    AccumulatorOverflow(usize),
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

/// Result of a watermark detection attempt.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether a watermark was detected.
    pub detected: bool,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f64,
    /// Recovered payload (if detected).
    pub payload: Option<WatermarkPayload>,
    /// Estimated watermark strength.
    pub estimated_strength: Option<WatermarkStrength>,
    /// Number of frames used in detection.
    pub frames_analyzed: u32,
    /// Raw correlation values per bit position (64 bits).
    pub bit_correlations: Vec<f64>,
}

impl DetectionResult {
    /// Return `true` if confidence exceeds the given threshold.
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

// ---------------------------------------------------------------------------
// Detection configuration
// ---------------------------------------------------------------------------

/// Configuration for watermark detection.
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Minimum confidence threshold to consider a watermark "detected".
    pub confidence_threshold: f64,
    /// Number of payload bits to extract (must match embed config; default 64).
    pub payload_bits: usize,
    /// Spread-spectrum chip rate: how many pixels per payload bit.
    pub chips_per_bit: usize,
    /// Maximum number of frames to accumulate.
    pub max_accumulation_frames: usize,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.6,
            payload_bits: 64,
            chips_per_bit: 128,
            max_accumulation_frames: 1000,
        }
    }
}

impl DetectionConfig {
    /// Configuration tuned for detecting invisible watermarks (needs more frames).
    pub fn for_invisible() -> Self {
        Self {
            confidence_threshold: 0.5,
            payload_bits: 64,
            chips_per_bit: 256,
            max_accumulation_frames: 2000,
        }
    }

    /// Configuration tuned for strong watermarks (fast single-frame detection).
    pub fn for_strong() -> Self {
        Self {
            confidence_threshold: 0.7,
            payload_bits: 64,
            chips_per_bit: 64,
            max_accumulation_frames: 100,
        }
    }

    /// Total pixels required per frame for detection.
    pub fn required_pixels(&self) -> usize {
        self.payload_bits * self.chips_per_bit
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random chip sequence (must match embedder)
// ---------------------------------------------------------------------------

/// Generate a deterministic PN (pseudo-noise) chip sequence for a given bit
/// position, using a simple LCG seeded by `bit_index`.  This must be
/// identical to the embedder's chip sequence.
fn generate_chip_sequence(bit_index: usize, length: usize) -> Vec<i8> {
    let mut chips = Vec::with_capacity(length);
    // LCG parameters (same as embedder would use)
    let mut state: u64 = 0x5DEE_CE66_D_u64.wrapping_add(bit_index as u64);
    for _ in 0..length {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Map to +1 / -1
        if (state >> 33) & 1 == 0 {
            chips.push(1);
        } else {
            chips.push(-1);
        }
    }
    chips
}

// ---------------------------------------------------------------------------
// Single-frame detector
// ---------------------------------------------------------------------------

/// Extract watermark bit correlations from a single grayscale frame.
///
/// `pixels` is a flat row-major grayscale buffer (one byte per pixel).
/// Returns one correlation value per payload bit.
pub fn detect_single_frame(
    pixels: &[u8],
    config: &DetectionConfig,
) -> Result<Vec<f64>, WatermarkDetectError> {
    let required = config.required_pixels();
    if pixels.len() < required {
        return Err(WatermarkDetectError::InsufficientData {
            needed: required,
            got: pixels.len(),
        });
    }

    let mut correlations = Vec::with_capacity(config.payload_bits);

    for bit_idx in 0..config.payload_bits {
        let chips = generate_chip_sequence(bit_idx, config.chips_per_bit);
        let offset = bit_idx * config.chips_per_bit;

        let mut sum: f64 = 0.0;
        for (j, &chip) in chips.iter().enumerate() {
            let pixel = pixels[offset + j] as f64 - 128.0; // center around 0
            sum += pixel * chip as f64;
        }
        correlations.push(sum / config.chips_per_bit as f64);
    }

    Ok(correlations)
}

// ---------------------------------------------------------------------------
// Multi-frame accumulator
// ---------------------------------------------------------------------------

/// Accumulates watermark evidence across multiple frames for robust detection.
#[derive(Debug, Clone)]
pub struct FrameAccumulator {
    config: DetectionConfig,
    /// Running sum of per-bit correlations.
    accumulated: Vec<f64>,
    /// Number of frames added.
    frame_count: u32,
}

impl FrameAccumulator {
    /// Create a new accumulator.
    pub fn new(config: DetectionConfig) -> Self {
        let bits = config.payload_bits;
        Self {
            config,
            accumulated: vec![0.0; bits],
            frame_count: 0,
        }
    }

    /// Add a frame's pixel data to the accumulator.
    pub fn add_frame(&mut self, pixels: &[u8]) -> Result<(), WatermarkDetectError> {
        if self.frame_count as usize >= self.config.max_accumulation_frames {
            return Err(WatermarkDetectError::AccumulatorOverflow(
                self.config.max_accumulation_frames,
            ));
        }

        let correlations = detect_single_frame(pixels, &self.config)?;
        for (acc, corr) in self.accumulated.iter_mut().zip(correlations.iter()) {
            *acc += corr;
        }
        self.frame_count += 1;
        Ok(())
    }

    /// Number of frames accumulated so far.
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.accumulated.iter_mut().for_each(|v| *v = 0.0);
        self.frame_count = 0;
    }

    /// Attempt detection using all accumulated evidence.
    pub fn detect(&self) -> DetectionResult {
        if self.frame_count == 0 {
            return DetectionResult {
                detected: false,
                confidence: 0.0,
                payload: None,
                estimated_strength: None,
                frames_analyzed: 0,
                bit_correlations: vec![0.0; self.config.payload_bits],
            };
        }

        let n = self.frame_count as f64;
        let averaged: Vec<f64> = self.accumulated.iter().map(|&s| s / n).collect();

        // Decode bits: positive correlation -> 1, negative -> 0
        let mut bits: u64 = 0;
        for (i, &corr) in averaged
            .iter()
            .enumerate()
            .take(64.min(self.config.payload_bits))
        {
            if corr > 0.0 {
                bits |= 1u64 << (63 - i);
            }
        }

        // Compute confidence as the mean absolute correlation normalized
        let mean_abs: f64 = averaged.iter().map(|c| c.abs()).sum::<f64>() / averaged.len() as f64;
        // Normalize to [0, 1] range using a sigmoid-like mapping
        let confidence = (mean_abs / (mean_abs + 1.0)).min(1.0);

        let detected = confidence >= self.config.confidence_threshold;

        let estimated_strength = estimate_strength(mean_abs);

        DetectionResult {
            detected,
            confidence,
            payload: if detected {
                Some(WatermarkPayload::decode(bits))
            } else {
                None
            },
            estimated_strength: if detected {
                Some(estimated_strength)
            } else {
                None
            },
            frames_analyzed: self.frame_count,
            bit_correlations: averaged,
        }
    }
}

/// Estimate watermark strength from mean absolute correlation.
fn estimate_strength(mean_abs: f64) -> WatermarkStrength {
    if mean_abs < 0.5 {
        WatermarkStrength::Invisible
    } else if mean_abs < 2.0 {
        WatermarkStrength::Light
    } else if mean_abs < 5.0 {
        WatermarkStrength::Medium
    } else {
        WatermarkStrength::Strong
    }
}

// ---------------------------------------------------------------------------
// Convenience: detect from a single frame with threshold check
// ---------------------------------------------------------------------------

/// Run full detection on a single frame and return a `DetectionResult`.
pub fn detect_watermark(
    pixels: &[u8],
    config: &DetectionConfig,
) -> Result<DetectionResult, WatermarkDetectError> {
    let mut accumulator = FrameAccumulator::new(config.clone());
    accumulator.add_frame(pixels)?;
    Ok(accumulator.detect())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> DetectionConfig {
        DetectionConfig {
            confidence_threshold: 0.3,
            payload_bits: 8,
            chips_per_bit: 16,
            max_accumulation_frames: 100,
        }
    }

    #[test]
    fn test_chip_sequence_deterministic() {
        let s1 = generate_chip_sequence(0, 32);
        let s2 = generate_chip_sequence(0, 32);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_chip_sequence_different_bits() {
        let s0 = generate_chip_sequence(0, 32);
        let s1 = generate_chip_sequence(1, 32);
        assert_ne!(s0, s1);
    }

    #[test]
    fn test_chip_sequence_only_plus_minus_one() {
        let seq = generate_chip_sequence(42, 1024);
        for &chip in &seq {
            assert!(chip == 1 || chip == -1);
        }
    }

    #[test]
    fn test_detect_single_frame_insufficient_data() {
        let config = make_config();
        let pixels = vec![128u8; 10]; // way too short
        let result = detect_single_frame(&pixels, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_single_frame_returns_correlations() {
        let config = make_config();
        let pixels = vec![128u8; config.required_pixels()];
        let correlations = detect_single_frame(&pixels, &config).expect("detection should succeed");
        assert_eq!(correlations.len(), config.payload_bits);
    }

    #[test]
    fn test_accumulator_frame_count() {
        let config = make_config();
        let mut acc = FrameAccumulator::new(config.clone());
        assert_eq!(acc.frame_count(), 0);

        let pixels = vec![128u8; config.required_pixels()];
        acc.add_frame(&pixels).expect("add_frame should succeed");
        assert_eq!(acc.frame_count(), 1);

        acc.add_frame(&pixels).expect("add_frame should succeed");
        assert_eq!(acc.frame_count(), 2);
    }

    #[test]
    fn test_accumulator_reset() {
        let config = make_config();
        let mut acc = FrameAccumulator::new(config.clone());
        let pixels = vec![128u8; config.required_pixels()];
        acc.add_frame(&pixels).expect("add_frame should succeed");
        acc.reset();
        assert_eq!(acc.frame_count(), 0);
    }

    #[test]
    fn test_accumulator_overflow() {
        let config = DetectionConfig {
            max_accumulation_frames: 2,
            ..make_config()
        };
        let mut acc = FrameAccumulator::new(config.clone());
        let pixels = vec![128u8; config.required_pixels()];
        acc.add_frame(&pixels).expect("add_frame should succeed");
        acc.add_frame(&pixels).expect("add_frame should succeed");
        let result = acc.add_frame(&pixels);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_empty_accumulator() {
        let config = make_config();
        let acc = FrameAccumulator::new(config);
        let result = acc.detect();
        assert!(!result.detected);
        assert_eq!(result.frames_analyzed, 0);
        assert!(result.payload.is_none());
    }

    #[test]
    fn test_detect_neutral_frame() {
        // All 128 pixels -> centered -> near-zero correlations
        let config = make_config();
        let mut acc = FrameAccumulator::new(config.clone());
        let pixels = vec![128u8; config.required_pixels()];
        acc.add_frame(&pixels).expect("add_frame should succeed");
        let result = acc.detect();
        // With uniform data, correlation should be very low
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_detection_result_meets_threshold() {
        let result = DetectionResult {
            detected: true,
            confidence: 0.8,
            payload: None,
            estimated_strength: None,
            frames_analyzed: 1,
            bit_correlations: vec![],
        };
        assert!(result.meets_threshold(0.7));
        assert!(!result.meets_threshold(0.9));
    }

    #[test]
    fn test_estimate_strength_invisible() {
        assert_eq!(estimate_strength(0.1), WatermarkStrength::Invisible);
    }

    #[test]
    fn test_estimate_strength_light() {
        assert_eq!(estimate_strength(1.0), WatermarkStrength::Light);
    }

    #[test]
    fn test_estimate_strength_medium() {
        assert_eq!(estimate_strength(3.0), WatermarkStrength::Medium);
    }

    #[test]
    fn test_estimate_strength_strong() {
        assert_eq!(estimate_strength(10.0), WatermarkStrength::Strong);
    }

    #[test]
    fn test_detect_watermark_convenience_function() {
        let config = make_config();
        let pixels = vec![128u8; config.required_pixels()];
        let result = detect_watermark(&pixels, &config).expect("detection should succeed");
        assert_eq!(result.frames_analyzed, 1);
    }

    #[test]
    fn test_detection_config_required_pixels() {
        let config = DetectionConfig {
            payload_bits: 64,
            chips_per_bit: 128,
            ..Default::default()
        };
        assert_eq!(config.required_pixels(), 64 * 128);
    }

    #[test]
    fn test_detection_config_for_invisible() {
        let config = DetectionConfig::for_invisible();
        assert_eq!(config.chips_per_bit, 256);
        assert!(config.confidence_threshold < 0.6);
    }

    #[test]
    fn test_detection_config_for_strong() {
        let config = DetectionConfig::for_strong();
        assert_eq!(config.chips_per_bit, 64);
        assert!(config.confidence_threshold > 0.6);
    }

    #[test]
    fn test_embed_then_detect_simulated() {
        // Simulate watermark embedding: for bit=1, add positive chips;
        // for bit=0, add negative chips.
        let config = make_config();
        let required = config.required_pixels();
        let mut pixels = vec![128u8; required];

        // Embed payload bits: 0b10101010 = 0xAA
        let payload_bits: [bool; 8] = [true, false, true, false, true, false, true, false];
        let amplitude = 20.0_f64;

        for (bit_idx, &is_one) in payload_bits.iter().enumerate() {
            let chips = generate_chip_sequence(bit_idx, config.chips_per_bit);
            let offset = bit_idx * config.chips_per_bit;
            for (j, &chip) in chips.iter().enumerate() {
                let delta = if is_one {
                    (amplitude * chip as f64) as i16
                } else {
                    (-amplitude * chip as f64) as i16
                };
                let val = (pixels[offset + j] as i16 + delta).clamp(0, 255) as u8;
                pixels[offset + j] = val;
            }
        }

        // Detect
        let correlations = detect_single_frame(&pixels, &config).expect("detection should succeed");

        // Verify each bit has the correct sign
        for (bit_idx, &corr) in correlations.iter().enumerate() {
            let expected_positive = payload_bits[bit_idx];
            if expected_positive {
                assert!(
                    corr > 0.0,
                    "bit {bit_idx}: expected positive correlation, got {corr}"
                );
            } else {
                assert!(
                    corr < 0.0,
                    "bit {bit_idx}: expected negative correlation, got {corr}"
                );
            }
        }
    }

    #[test]
    fn test_multi_frame_accumulation_improves_confidence() {
        let config = make_config();
        let required = config.required_pixels();

        // Create a weakly watermarked frame
        let amplitude = 3.0_f64; // very weak
        let mut frame = vec![128u8; required];
        let payload_bits = [true; 8]; // all ones

        for (bit_idx, &is_one) in payload_bits.iter().enumerate() {
            let chips = generate_chip_sequence(bit_idx, config.chips_per_bit);
            let offset = bit_idx * config.chips_per_bit;
            for (j, &chip) in chips.iter().enumerate() {
                let delta = if is_one {
                    (amplitude * chip as f64) as i16
                } else {
                    (-amplitude * chip as f64) as i16
                };
                frame[offset + j] = (128i16 + delta).clamp(0, 255) as u8;
            }
        }

        // Single frame
        let mut acc1 = FrameAccumulator::new(config.clone());
        acc1.add_frame(&frame).expect("add_frame should succeed");
        let r1 = acc1.detect();

        // Multiple frames
        let mut acc5 = FrameAccumulator::new(config.clone());
        for _ in 0..5 {
            acc5.add_frame(&frame).expect("add_frame should succeed");
        }
        let r5 = acc5.detect();

        // Multi-frame confidence should be >= single (same signal repeated)
        assert!(
            r5.confidence >= r1.confidence,
            "5-frame confidence ({}) should be >= 1-frame ({})",
            r5.confidence,
            r1.confidence,
        );
    }

    #[test]
    fn test_error_display() {
        let e = WatermarkDetectError::NotDetected {
            confidence: 0.3,
            threshold: 0.6,
        };
        let msg = format!("{e}");
        assert!(msg.contains("0.3"));
        assert!(msg.contains("0.6"));

        let e2 = WatermarkDetectError::InsufficientData {
            needed: 100,
            got: 50,
        };
        assert!(format!("{e2}").contains("100"));
    }
}
