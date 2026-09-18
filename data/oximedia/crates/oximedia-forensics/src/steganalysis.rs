//! Steganography analysis for detecting hidden data in media files.
//!
//! Provides LSB (Least Significant Bit) analysis, statistical anomaly detection,
//! and entropy mapping to identify steganographic content.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};

/// Result of LSB analysis on an image channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LsbAnalysisResult {
    /// Channel analyzed (R, G, B, or Y)
    pub channel: String,
    /// Fraction of LSBs that are 1 (expected ~0.5 for natural images)
    pub lsb_one_ratio: f64,
    /// Chi-square statistic (high values indicate non-random patterns)
    pub chi_square: f64,
    /// Whether steganography is suspected in this channel
    pub suspected: bool,
    /// Number of pixels analyzed
    pub pixel_count: usize,
}

impl LsbAnalysisResult {
    /// Create a new LSB analysis result
    #[must_use]
    pub fn new(channel: impl Into<String>, pixel_count: usize) -> Self {
        Self {
            channel: channel.into(),
            lsb_one_ratio: 0.0,
            chi_square: 0.0,
            suspected: false,
            pixel_count,
        }
    }

    /// Compute ratio from raw pixel values
    pub fn compute_from_pixels(&mut self, pixels: &[u8]) {
        if pixels.is_empty() {
            return;
        }
        let ones: usize = pixels.iter().map(|&p| (p & 1) as usize).sum();
        self.lsb_one_ratio = ones as f64 / pixels.len() as f64;

        // Chi-square test: expected 50% ones
        let expected = pixels.len() as f64 / 2.0;
        let observed_ones = ones as f64;
        let observed_zeros = pixels.len() as f64 - observed_ones;
        self.chi_square = (observed_ones - expected).powi(2) / expected
            + (observed_zeros - expected).powi(2) / expected;

        // Threshold: chi-square > 3.84 (p<0.05) suggests non-random
        self.suspected = self.chi_square > 3.84 || (self.lsb_one_ratio - 0.5).abs() > 0.05;
    }
}

/// Entropy map of an image (divides image into blocks and computes entropy per block)
#[derive(Debug, Clone)]
pub struct EntropyMap {
    /// Entropy values per block (row-major)
    pub blocks: Vec<f64>,
    /// Number of block columns
    pub cols: usize,
    /// Number of block rows
    pub rows: usize,
    /// Block size in pixels
    pub block_size: usize,
}

impl EntropyMap {
    /// Create a new entropy map from pixel data
    #[must_use]
    pub fn compute(pixels: &[u8], width: usize, height: usize, block_size: usize) -> Self {
        let block_size = block_size.max(1);
        let cols = (width + block_size - 1) / block_size;
        let rows = (height + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(rows * cols);

        for br in 0..rows {
            for bc in 0..cols {
                let mut hist = [0u32; 256];
                let mut count = 0usize;

                for dy in 0..block_size {
                    let y = br * block_size + dy;
                    if y >= height {
                        break;
                    }
                    for dx in 0..block_size {
                        let x = bc * block_size + dx;
                        if x >= width {
                            break;
                        }
                        let idx = y * width + x;
                        if idx < pixels.len() {
                            hist[pixels[idx] as usize] += 1;
                            count += 1;
                        }
                    }
                }

                blocks.push(shannon_entropy(&hist, count));
            }
        }

        Self {
            blocks,
            cols,
            rows,
            block_size,
        }
    }

    /// Mean entropy across all blocks
    #[must_use]
    pub fn mean_entropy(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        self.blocks.iter().sum::<f64>() / self.blocks.len() as f64
    }

    /// Standard deviation of entropy across blocks
    #[must_use]
    pub fn entropy_std_dev(&self) -> f64 {
        if self.blocks.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_entropy();
        let variance = self.blocks.iter().map(|&e| (e - mean).powi(2)).sum::<f64>()
            / (self.blocks.len() - 1) as f64;
        variance.sqrt()
    }

    /// Blocks with anomalously high entropy (possible hidden data)
    #[must_use]
    pub fn anomalous_blocks(&self, threshold_sigma: f64) -> Vec<usize> {
        let mean = self.mean_entropy();
        let std = self.entropy_std_dev();
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > mean + threshold_sigma * std)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Compute Shannon entropy from a histogram
fn shannon_entropy(hist: &[u32; 256], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    hist.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.log2()
        })
        .sum()
}

/// Statistical anomaly detector for detecting non-natural pixel distributions
#[derive(Debug, Clone)]
pub struct StatisticalAnomalyDetector {
    /// Sensitivity (0.0 = least sensitive, 1.0 = most sensitive)
    pub sensitivity: f64,
}

impl StatisticalAnomalyDetector {
    /// Create a new detector
    #[must_use]
    pub fn new(sensitivity: f64) -> Self {
        Self {
            sensitivity: sensitivity.clamp(0.0, 1.0),
        }
    }

    /// Analyze byte pair histogram for RS (Regular-Singular) steganalysis
    pub fn rs_analysis(&self, pixels: &[u8]) -> RsAnalysisResult {
        let n = pixels.len();
        if n < 4 {
            return RsAnalysisResult::default();
        }

        let mut regular_pos = 0usize;
        let mut singular_pos = 0usize;
        let mut regular_neg = 0usize;
        let mut singular_neg = 0usize;
        let mut total_groups = 0usize;

        // Process pairs of pixels
        for chunk in pixels.chunks(2) {
            if chunk.len() < 2 {
                break;
            }
            let (a, b) = (chunk[0] as i32, chunk[1] as i32);
            let diff = (a - b).abs();

            // After flipping LSB of b
            let b_flipped = (chunk[1] ^ 1) as i32;
            let diff_flipped = (a - b_flipped).abs();

            if diff_flipped < diff {
                regular_pos += 1;
            } else if diff_flipped > diff {
                singular_pos += 1;
            }

            // After flipping all bits of b (negative flip)
            let b_neg = if chunk[1] == 0 {
                255i32
            } else {
                chunk[1] as i32 - 1
            };
            let diff_neg = (a - b_neg).abs();

            if diff_neg < diff {
                regular_neg += 1;
            } else if diff_neg > diff {
                singular_neg += 1;
            }

            total_groups += 1;
        }

        let total_f = total_groups as f64;
        RsAnalysisResult {
            rm: regular_pos as f64 / total_f,
            sm: singular_pos as f64 / total_f,
            rm_neg: regular_neg as f64 / total_f,
            sm_neg: singular_neg as f64 / total_f,
            estimated_payload: self.estimate_payload_rs(
                regular_pos as f64 / total_f,
                singular_pos as f64 / total_f,
                regular_neg as f64 / total_f,
                singular_neg as f64 / total_f,
            ),
            total_groups,
        }
    }

    /// Estimate hidden payload size using RS statistics
    fn estimate_payload_rs(&self, rm: f64, sm: f64, rm_neg: f64, sm_neg: f64) -> f64 {
        // Simplified estimation: high RS asymmetry suggests payload
        let asymmetry = ((rm - rm_neg).abs() + (sm - sm_neg).abs()) / 2.0;
        (asymmetry * 2.0 * self.sensitivity).clamp(0.0, 1.0)
    }

    /// Detect statistical anomalies in pixel values
    #[must_use]
    pub fn detect_anomalies(&self, pixels: &[u8]) -> Vec<AnomalyRegion> {
        let mut regions = Vec::new();
        let block_size = 64;
        let threshold = 3.84 * (1.0 - self.sensitivity * 0.5);

        let mut hist = [0u32; 256];
        for &p in pixels {
            hist[p as usize] += 1;
        }
        let global_entropy = shannon_entropy(&hist, pixels.len());

        for (block_idx, chunk) in pixels.chunks(block_size).enumerate() {
            let mut block_hist = [0u32; 256];
            for &p in chunk {
                block_hist[p as usize] += 1;
            }
            let block_entropy = shannon_entropy(&block_hist, chunk.len());

            // Anomaly: block entropy significantly higher than global
            if block_entropy > global_entropy + threshold * 0.1 {
                regions.push(AnomalyRegion {
                    block_index: block_idx,
                    entropy: block_entropy,
                    severity: ((block_entropy - global_entropy) / global_entropy).clamp(0.0, 1.0),
                });
            }
        }

        regions
    }
}

impl Default for StatisticalAnomalyDetector {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// Result of RS (Regular-Singular) steganalysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RsAnalysisResult {
    /// Regular groups (positive mask)
    pub rm: f64,
    /// Singular groups (positive mask)
    pub sm: f64,
    /// Regular groups (negative mask)
    pub rm_neg: f64,
    /// Singular groups (negative mask)
    pub sm_neg: f64,
    /// Estimated payload fraction (0.0 = no hidden data, 1.0 = fully embedded)
    pub estimated_payload: f64,
    /// Total groups analyzed
    pub total_groups: usize,
}

impl RsAnalysisResult {
    /// Returns true if steganography is likely present
    #[must_use]
    pub fn stego_detected(&self, threshold: f64) -> bool {
        self.estimated_payload > threshold
    }
}

/// A region of the image with anomalous entropy
#[derive(Debug, Clone)]
pub struct AnomalyRegion {
    /// Block index in the pixel stream
    pub block_index: usize,
    /// Entropy value of this block
    pub entropy: f64,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
}

/// Complete steganalysis report for an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteganalysisReport {
    /// LSB analysis results per channel
    pub lsb_results: Vec<LsbAnalysisResult>,
    /// RS analysis result
    #[serde(skip)]
    pub rs_result: Option<RsAnalysisResult>,
    /// Overall steganography detection confidence (0.0 to 1.0)
    pub stego_confidence: f64,
    /// Whether steganography is detected
    pub stego_detected: bool,
}

impl SteganalysisReport {
    /// Create a new report
    #[must_use]
    pub fn new() -> Self {
        Self {
            lsb_results: Vec::new(),
            rs_result: None,
            stego_confidence: 0.0,
            stego_detected: false,
        }
    }

    /// Compute overall confidence from component results
    pub fn compute_confidence(&mut self) {
        let mut scores = Vec::new();

        // LSB evidence
        let lsb_suspected = self.lsb_results.iter().filter(|r| r.suspected).count();
        if !self.lsb_results.is_empty() {
            scores.push(lsb_suspected as f64 / self.lsb_results.len() as f64);
        }

        // RS evidence
        if let Some(ref rs) = self.rs_result {
            scores.push(rs.estimated_payload);
        }

        self.stego_confidence = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        self.stego_detected = self.stego_confidence > 0.3;
    }
}

impl Default for SteganalysisReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsb_analysis_result_creation() {
        let result = LsbAnalysisResult::new("R", 1000);
        assert_eq!(result.channel, "R");
        assert_eq!(result.pixel_count, 1000);
        assert!(!result.suspected);
    }

    #[test]
    fn test_lsb_analysis_natural_image() {
        // Natural image: roughly 50% LSBs are 1
        let pixels: Vec<u8> = (0..1000u16).map(|i| (i % 256) as u8).collect();
        let mut result = LsbAnalysisResult::new("Y", pixels.len());
        result.compute_from_pixels(&pixels);
        // Even distribution, chi-square should be small
        assert!(result.lsb_one_ratio > 0.0);
    }

    #[test]
    fn test_lsb_analysis_all_zeros() {
        // All zeros: all LSBs are 0, ratio=0, high chi-square
        let pixels = vec![0u8; 100];
        let mut result = LsbAnalysisResult::new("G", pixels.len());
        result.compute_from_pixels(&pixels);
        assert_eq!(result.lsb_one_ratio, 0.0);
        assert!(result.suspected); // Highly non-random
    }

    #[test]
    fn test_lsb_analysis_all_ones() {
        // All 0xFF: all LSBs are 1
        let pixels = vec![0xFFu8; 200];
        let mut result = LsbAnalysisResult::new("B", pixels.len());
        result.compute_from_pixels(&pixels);
        assert!((result.lsb_one_ratio - 1.0).abs() < f64::EPSILON);
        assert!(result.suspected);
    }

    #[test]
    fn test_lsb_analysis_empty() {
        let mut result = LsbAnalysisResult::new("R", 0);
        result.compute_from_pixels(&[]);
        assert_eq!(result.lsb_one_ratio, 0.0);
    }

    #[test]
    fn test_entropy_map_compute() {
        let pixels: Vec<u8> = (0..256u16).map(|i| i as u8).collect();
        let map = EntropyMap::compute(&pixels, 16, 16, 4);
        assert_eq!(map.rows, 4);
        assert_eq!(map.cols, 4);
        assert_eq!(map.blocks.len(), 16);
    }

    #[test]
    fn test_entropy_map_mean() {
        let pixels: Vec<u8> = (0..256u16).map(|i| i as u8).collect();
        let map = EntropyMap::compute(&pixels, 16, 16, 4);
        let mean = map.mean_entropy();
        assert!(mean > 0.0);
        assert!(mean <= 8.0); // Max entropy for 8-bit values
    }

    #[test]
    fn test_entropy_map_uniform_is_high_entropy() {
        // Uniform distribution = maximum entropy
        let pixels: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let map = EntropyMap::compute(&pixels, 32, 32, 32);
        assert!(map.mean_entropy() > 7.0); // Near max entropy
    }

    #[test]
    fn test_entropy_map_constant_is_zero_entropy() {
        // Constant image = zero entropy
        let pixels = vec![128u8; 256];
        let map = EntropyMap::compute(&pixels, 16, 16, 16);
        assert!(map.mean_entropy() < 0.001);
    }

    #[test]
    fn test_shannon_entropy_uniform() {
        let mut hist = [0u32; 256];
        for i in 0..256 {
            hist[i] = 1;
        }
        let entropy = shannon_entropy(&hist, 256);
        assert!((entropy - 8.0).abs() < 0.001); // log2(256) = 8
    }

    #[test]
    fn test_shannon_entropy_zero_total() {
        let hist = [0u32; 256];
        assert_eq!(shannon_entropy(&hist, 0), 0.0);
    }

    #[test]
    fn test_rs_analysis_natural() {
        let pixels: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        let detector = StatisticalAnomalyDetector::new(0.5);
        let result = detector.rs_analysis(&pixels);
        assert!(result.total_groups > 0);
        // Natural image should have low estimated payload
        assert!(result.estimated_payload < 0.8);
    }

    #[test]
    fn test_rs_analysis_all_lsb_set() {
        // Setting all LSBs to 1 is a strong stego signal
        let pixels: Vec<u8> = (0..=255u8).map(|p| p | 1).collect();
        let detector = StatisticalAnomalyDetector::new(0.9);
        let result = detector.rs_analysis(&pixels);
        assert!(result.total_groups > 0);
    }

    #[test]
    fn test_stego_detected_threshold() {
        let result = RsAnalysisResult {
            rm: 0.6,
            sm: 0.3,
            rm_neg: 0.4,
            sm_neg: 0.2,
            estimated_payload: 0.5,
            total_groups: 100,
        };
        assert!(result.stego_detected(0.4));
        assert!(!result.stego_detected(0.6));
    }

    #[test]
    fn test_steganalysis_report_creation() {
        let report = SteganalysisReport::new();
        assert!(!report.stego_detected);
        assert_eq!(report.stego_confidence, 0.0);
    }

    #[test]
    fn test_steganalysis_report_compute_confidence() {
        let mut report = SteganalysisReport::new();
        let mut lsb = LsbAnalysisResult::new("R", 100);
        lsb.suspected = true;
        report.lsb_results.push(lsb);
        let lsb2 = LsbAnalysisResult::new("G", 100);
        report.lsb_results.push(lsb2);

        report.compute_confidence();
        // 1/2 channels suspected = 0.5 confidence
        assert!((report.stego_confidence - 0.5).abs() < f64::EPSILON);
        assert!(report.stego_detected);
    }

    #[test]
    fn test_anomaly_detector_default() {
        let detector = StatisticalAnomalyDetector::default();
        assert!((detector.sensitivity - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_anomaly_detector_clamps_sensitivity() {
        let detector = StatisticalAnomalyDetector::new(1.5);
        assert!((detector.sensitivity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lsb_detects_stego() {
        // Embed LSB steganography: set ALL LSBs to 1.
        // This yields lsb_one_ratio=1.0, deviation 0.5 > threshold 0.05 → suspected=true.
        let mut pixels = vec![128u8; 4096];
        for p in pixels.iter_mut() {
            *p |= 1; // force LSB to 1
        }
        let mut result = LsbAnalysisResult::new("luma", 4096);
        result.compute_from_pixels(&pixels);
        assert!(
            result.suspected,
            "all-LSB-set pattern should be detected as suspected \
             (chi_square={}, lsb_one_ratio={})",
            result.chi_square, result.lsb_one_ratio
        );
    }

    #[test]
    fn test_lsb_clean_not_suspected() {
        // Balanced alternating-LSB data: half pixels 128 (LSB=0), half 129 (LSB=1).
        // lsb_one_ratio=0.5, chi_square≈0 → suspected=false.
        let pixels: Vec<u8> = (0..4096u32).map(|i| 128 + (i % 2) as u8).collect();
        let mut result = LsbAnalysisResult::new("luma", 4096);
        result.compute_from_pixels(&pixels);
        assert!(
            !result.suspected,
            "balanced 50/50 LSB data should not be suspected \
             (chi_square={}, lsb_one_ratio={})",
            result.chi_square, result.lsb_one_ratio
        );
    }

    #[test]
    fn test_rs_analysis_stego_detected() {
        // Construct pairs (small_odd, large_even) that maximise RS asymmetry.
        // For pair (a=9, b=200) where b is even and b > a:
        //   b_flip = 201, diff_flip=192 > diff=191 → singular_pos
        //   b_neg  = 199, diff_neg =190 < diff=191 → regular_neg
        // This yields sm=1, rm_neg=1, rm=0, sm_neg=0 → asymmetry=1.0 → payload=1.0.
        let pixels: Vec<u8> = (0..4096u32)
            .map(|i| if i % 2 == 0 { 9u8 } else { 200u8 })
            .collect();
        let result = StatisticalAnomalyDetector::new(0.5).rs_analysis(&pixels);
        assert!(
            result.stego_detected(0.0),
            "RS analysis must detect high-asymmetry stego signal \
             (estimated_payload={}, rm={:.3}, sm={:.3}, rm_neg={:.3}, sm_neg={:.3})",
            result.estimated_payload,
            result.rm,
            result.sm,
            result.rm_neg,
            result.sm_neg,
        );
    }
}
