#![allow(dead_code)]
//! Frequency domain forensic analysis for image tampering detection.
//!
//! This module performs forensic analysis in the frequency domain (DCT/DFT) to detect
//! artifacts from JPEG re-compression, copy-paste operations, and other manipulations.
//!
//! # Features
//!
//! - **DCT coefficient analysis** for detecting double JPEG compression
//! - **Frequency spectrum anomaly detection** for identifying spliced regions
//! - **Periodic pattern detection** for finding upscaling or resampling artifacts
//! - **Block artifact grid analysis** for detecting misaligned JPEG grids
//! - **Power spectrum analysis** for source identification

use rayon::prelude::*;
use std::f64::consts::PI;

/// A block of DCT coefficients (8x8).
#[derive(Debug, Clone)]
pub struct DctBlock {
    /// The 64 DCT coefficients in zigzag order.
    pub coefficients: [f64; 64],
    /// Block position (x, y) in the image grid.
    pub position: (u32, u32),
}

impl DctBlock {
    /// Create a new DCT block from coefficients.
    #[must_use]
    pub fn new(coefficients: [f64; 64], position: (u32, u32)) -> Self {
        Self {
            coefficients,
            position,
        }
    }

    /// Create a zero-valued DCT block.
    #[must_use]
    pub fn zero(position: (u32, u32)) -> Self {
        Self {
            coefficients: [0.0; 64],
            position,
        }
    }

    /// Get the DC coefficient (index 0).
    #[must_use]
    pub fn dc(&self) -> f64 {
        self.coefficients[0]
    }

    /// Get the energy (sum of squared coefficients excluding DC).
    #[must_use]
    pub fn ac_energy(&self) -> f64 {
        self.coefficients[1..].iter().map(|c| c * c).sum()
    }

    /// Count the number of zero coefficients.
    #[must_use]
    pub fn zero_count(&self) -> usize {
        self.coefficients.iter().filter(|c| c.abs() < 1e-10).count()
    }

    /// Compute the ratio of zero to non-zero coefficients.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sparsity(&self) -> f64 {
        self.zero_count() as f64 / 64.0
    }
}

/// Result of DCT coefficient histogram analysis for one frequency index.
#[derive(Debug, Clone)]
pub struct DctHistogram {
    /// Frequency index (0-63).
    pub freq_index: usize,
    /// Histogram bin counts.
    pub bins: Vec<u64>,
    /// Bin width.
    pub bin_width: f64,
    /// Minimum coefficient value.
    pub min_value: f64,
    /// Maximum coefficient value.
    pub max_value: f64,
}

impl DctHistogram {
    /// Create a new DCT histogram from coefficient values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_values(freq_index: usize, values: &[f64], num_bins: usize) -> Self {
        if values.is_empty() || num_bins == 0 {
            return Self {
                freq_index,
                bins: vec![0; num_bins.max(1)],
                bin_width: 1.0,
                min_value: 0.0,
                max_value: 0.0,
            };
        }

        let min_val = values.iter().cloned().fold(f64::MAX, f64::min);
        let max_val = values.iter().cloned().fold(f64::MIN, f64::max);
        let range = max_val - min_val;
        let bin_width = if range > 1e-10 {
            range / num_bins as f64
        } else {
            1.0
        };

        let mut bins = vec![0u64; num_bins];
        for &v in values {
            let idx = if range > 1e-10 {
                ((v - min_val) / bin_width).floor() as usize
            } else {
                0
            };
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }

        Self {
            freq_index,
            bins,
            bin_width,
            min_value: min_val,
            max_value: max_val,
        }
    }

    /// Detect periodic gaps in the histogram (sign of double compression).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_periodic_gaps(&self) -> f64 {
        if self.bins.len() < 4 {
            return 0.0;
        }

        // Look for alternating high-low pattern
        let mut transitions = 0u32;
        let avg: f64 = self.bins.iter().map(|&b| b as f64).sum::<f64>() / self.bins.len() as f64;

        if avg < 1.0 {
            return 0.0;
        }

        let mut prev_above = self.bins[0] as f64 > avg;
        for &bin in self.bins.iter().skip(1) {
            let above = bin as f64 > avg;
            if above != prev_above {
                transitions += 1;
            }
            prev_above = above;
        }

        // High transition count relative to bins suggests periodicity
        let expected = self.bins.len() as f64 / 2.0;
        if expected < 1.0 {
            return 0.0;
        }
        (transitions as f64 / expected).min(1.0)
    }
}

/// A detected frequency-domain anomaly.
#[derive(Debug, Clone)]
pub struct FrequencyAnomaly {
    /// Type of anomaly.
    pub anomaly_type: FrequencyAnomalyType,
    /// Location in the image (block x, block y) or None for global.
    pub location: Option<(u32, u32)>,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Description.
    pub description: String,
}

/// Types of frequency-domain anomalies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyAnomalyType {
    /// Double JPEG compression detected.
    DoubleCompression,
    /// Misaligned block grid detected.
    GridMisalignment,
    /// Resampling artifacts (periodic peaks in spectrum).
    ResamplingArtifact,
    /// Unusual spectral energy distribution.
    SpectralAnomaly,
    /// Localized frequency inconsistency.
    LocalInconsistency,
}

/// Configuration for frequency forensic analysis.
#[derive(Debug, Clone)]
pub struct FrequencyForensicsConfig {
    /// Number of histogram bins for DCT analysis.
    pub num_histogram_bins: usize,
    /// Periodic gap detection threshold (0 to 1).
    pub periodicity_threshold: f64,
    /// Minimum confidence to report an anomaly.
    pub min_confidence: f64,
    /// Block artifact grid analysis threshold.
    pub grid_threshold: f64,
    /// Spectral energy ratio threshold.
    pub spectral_threshold: f64,
}

impl Default for FrequencyForensicsConfig {
    fn default() -> Self {
        Self {
            num_histogram_bins: 64,
            periodicity_threshold: 0.6,
            min_confidence: 0.5,
            grid_threshold: 0.3,
            spectral_threshold: 2.0,
        }
    }
}

/// Result of frequency-domain forensic analysis.
#[derive(Debug, Clone)]
pub struct FrequencyForensicsReport {
    /// Total blocks analyzed.
    pub total_blocks: u64,
    /// Detected anomalies.
    pub anomalies: Vec<FrequencyAnomaly>,
    /// Average block sparsity (0 to 1).
    pub avg_sparsity: f64,
    /// Double compression likelihood (0 to 1).
    pub double_compression_likelihood: f64,
    /// Overall tampering score (0 to 1).
    pub tampering_score: f64,
}

/// Perform a 1D DCT-II on the given input.
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn dct_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let n_f64 = n as f64;

    (0..n)
        .map(|k| {
            let sum: f64 = input
                .iter()
                .enumerate()
                .map(|(i, &x)| x * (PI * (2.0 * i as f64 + 1.0) * k as f64 / (2.0 * n_f64)).cos())
                .sum();

            let scale = if k == 0 {
                (1.0 / n_f64).sqrt()
            } else {
                (2.0 / n_f64).sqrt()
            };

            sum * scale
        })
        .collect()
}

/// Compute the 2D DCT of an 8x8 block (given as 64 f64 values in row-major order).
#[must_use]
pub fn dct_2d_8x8(block: &[f64; 64]) -> [f64; 64] {
    let mut temp = [0.0f64; 64];

    // DCT on rows
    for r in 0..8 {
        let row: Vec<f64> = (0..8).map(|c| block[r * 8 + c]).collect();
        let dct_row = dct_1d(&row);
        for c in 0..8 {
            temp[r * 8 + c] = dct_row[c];
        }
    }

    // DCT on columns
    let mut result = [0.0f64; 64];
    for c in 0..8 {
        let col: Vec<f64> = (0..8).map(|r| temp[r * 8 + c]).collect();
        let dct_col = dct_1d(&col);
        for r in 0..8 {
            result[r * 8 + c] = dct_col[r];
        }
    }

    result
}

/// Frequency-domain forensic analyzer.
#[derive(Debug, Clone)]
pub struct FrequencyForensicAnalyzer {
    /// Configuration.
    config: FrequencyForensicsConfig,
}

impl FrequencyForensicAnalyzer {
    /// Create a new frequency forensic analyzer.
    #[must_use]
    pub fn new(config: FrequencyForensicsConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: FrequencyForensicsConfig::default(),
        }
    }

    /// Analyze a set of DCT blocks for forensic anomalies.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_blocks(&self, blocks: &[DctBlock]) -> FrequencyForensicsReport {
        let mut anomalies = Vec::new();

        if blocks.is_empty() {
            return FrequencyForensicsReport {
                total_blocks: 0,
                anomalies,
                avg_sparsity: 0.0,
                double_compression_likelihood: 0.0,
                tampering_score: 0.0,
            };
        }

        // Compute average sparsity
        let avg_sparsity = blocks.iter().map(|b| b.sparsity()).sum::<f64>() / blocks.len() as f64;

        // Analyze DCT coefficient histograms for double compression
        let double_comp = self.detect_double_compression(blocks);

        if double_comp > self.config.periodicity_threshold {
            anomalies.push(FrequencyAnomaly {
                anomaly_type: FrequencyAnomalyType::DoubleCompression,
                location: None,
                confidence: double_comp,
                description: format!(
                    "Double JPEG compression detected (confidence={:.2})",
                    double_comp
                ),
            });
        }

        // Analyze block grid alignment
        let grid_anomalies = self.detect_grid_misalignment(blocks);
        anomalies.extend(grid_anomalies);

        // Analyze spectral energy distribution
        let spectral_anomalies = self.detect_spectral_anomalies(blocks);
        anomalies.extend(spectral_anomalies);

        // Filter by confidence
        anomalies.retain(|a| a.confidence >= self.config.min_confidence);

        let tampering_score = if anomalies.is_empty() {
            0.0
        } else {
            anomalies.iter().map(|a| a.confidence).sum::<f64>() / anomalies.len() as f64
        };

        FrequencyForensicsReport {
            total_blocks: blocks.len() as u64,
            anomalies,
            avg_sparsity,
            double_compression_likelihood: double_comp,
            tampering_score,
        }
    }

    /// Analyze grayscale image data directly.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_image(&self, data: &[u8], width: u32, height: u32) -> FrequencyForensicsReport {
        if data.is_empty() || width < 8 || height < 8 {
            return FrequencyForensicsReport {
                total_blocks: 0,
                anomalies: Vec::new(),
                avg_sparsity: 0.0,
                double_compression_likelihood: 0.0,
                tampering_score: 0.0,
            };
        }

        let block_cols = width / 8;
        let block_rows = height / 8;
        let mut blocks = Vec::new();

        for by in 0..block_rows {
            for bx in 0..block_cols {
                let mut pixel_block = [0.0f64; 64];
                for r in 0..8 {
                    for c in 0..8 {
                        let x = bx * 8 + c;
                        let y = by * 8 + r;
                        let idx = (y * width + x) as usize;
                        if idx < data.len() {
                            pixel_block[(r * 8 + c) as usize] = f64::from(data[idx]);
                        }
                    }
                }
                let coeffs = dct_2d_8x8(&pixel_block);
                blocks.push(DctBlock::new(coeffs, (bx, by)));
            }
        }

        self.analyze_blocks(&blocks)
    }

    /// Detect double JPEG compression from DCT coefficient histograms.
    ///
    /// Each frequency index is independent so we analyse them in parallel using
    /// rayon.  The periodicity scores are then averaged to produce the final
    /// confidence value.  The result is identical to the serial version because
    /// histogram construction and gap detection have no shared mutable state.
    #[allow(clippy::cast_precision_loss)]
    fn detect_double_compression(&self, blocks: &[DctBlock]) -> f64 {
        if blocks.is_empty() {
            return 0.0;
        }

        // Analyze a few AC frequencies (indices 1, 2, 3) for periodicity
        let freq_indices = [1usize, 2, 3, 8, 9, 16];
        let num_histogram_bins = self.config.num_histogram_bins;

        // Compute periodicity for each frequency index in parallel.
        let periodicities: Vec<f64> = freq_indices
            .par_iter()
            .filter(|&&fi| fi < 64)
            .map(|&fi| {
                let values: Vec<f64> = blocks.iter().map(|b| b.coefficients[fi]).collect();
                let hist = DctHistogram::from_values(fi, &values, num_histogram_bins);
                hist.detect_periodic_gaps()
            })
            .collect();

        if periodicities.is_empty() {
            return 0.0;
        }

        periodicities.iter().sum::<f64>() / periodicities.len() as f64
    }

    /// Detect JPEG block grid misalignment.
    #[allow(clippy::cast_precision_loss)]
    fn detect_grid_misalignment(&self, blocks: &[DctBlock]) -> Vec<FrequencyAnomaly> {
        let mut anomalies = Vec::new();

        if blocks.len() < 4 {
            return anomalies;
        }

        // Compare AC energy of neighboring blocks — large differences suggest misaligned grids
        let energies: Vec<(u32, u32, f64)> = blocks
            .iter()
            .map(|b| (b.position.0, b.position.1, b.ac_energy()))
            .collect();

        let avg_energy = energies.iter().map(|e| e.2).sum::<f64>() / energies.len() as f64;
        if avg_energy < 1e-10 {
            return anomalies;
        }

        for &(x, y, energy) in &energies {
            let ratio = energy / avg_energy;
            if ratio > self.config.spectral_threshold {
                anomalies.push(FrequencyAnomaly {
                    anomaly_type: FrequencyAnomalyType::GridMisalignment,
                    location: Some((x, y)),
                    confidence: ((ratio - 1.0) / self.config.spectral_threshold).min(1.0),
                    description: format!(
                        "Block ({}, {}) has {:.1}x average AC energy",
                        x, y, ratio
                    ),
                });
            }
        }

        anomalies
    }

    /// Detect spectral energy anomalies.
    #[allow(clippy::cast_precision_loss)]
    fn detect_spectral_anomalies(&self, blocks: &[DctBlock]) -> Vec<FrequencyAnomaly> {
        let mut anomalies = Vec::new();

        if blocks.len() < 4 {
            return anomalies;
        }

        // Check for blocks with anomalous sparsity
        let sparsities: Vec<f64> = blocks.iter().map(|b| b.sparsity()).collect();
        let avg_sparsity = sparsities.iter().sum::<f64>() / sparsities.len() as f64;
        let std_sparsity = (sparsities
            .iter()
            .map(|s| (s - avg_sparsity).powi(2))
            .sum::<f64>()
            / sparsities.len() as f64)
            .sqrt();

        if std_sparsity < 1e-10 {
            return anomalies;
        }

        for (i, &s) in sparsities.iter().enumerate() {
            let z_score = (s - avg_sparsity).abs() / std_sparsity;
            if z_score > 3.0 {
                anomalies.push(FrequencyAnomaly {
                    anomaly_type: FrequencyAnomalyType::SpectralAnomaly,
                    location: Some(blocks[i].position),
                    confidence: (z_score / 6.0).min(1.0),
                    description: format!(
                        "Block ({}, {}) has anomalous sparsity {:.2} (z={:.1})",
                        blocks[i].position.0, blocks[i].position.1, s, z_score
                    ),
                });
            }
        }

        anomalies
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FrequencyForensicsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_block_creation() {
        let block = DctBlock::zero((0, 0));
        assert!((block.dc()).abs() < f64::EPSILON);
        assert!((block.ac_energy()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dct_block_sparsity() {
        let block = DctBlock::zero((0, 0));
        assert!((block.sparsity() - 1.0).abs() < f64::EPSILON);

        let mut coeffs = [0.0; 64];
        for (i, c) in coeffs.iter_mut().enumerate() {
            *c = i as f64;
        }
        let block2 = DctBlock::new(coeffs, (0, 0));
        // Only coeff[0] = 0.0, so sparsity = 1/64
        assert!((block2.sparsity() - 1.0 / 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_dct_block_ac_energy() {
        let mut coeffs = [0.0; 64];
        coeffs[0] = 100.0; // DC
        coeffs[1] = 3.0;
        coeffs[2] = 4.0;
        let block = DctBlock::new(coeffs, (0, 0));
        assert!((block.ac_energy() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_dct_block_zero_count() {
        let mut coeffs = [0.0; 64];
        coeffs[0] = 1.0;
        coeffs[1] = 2.0;
        let block = DctBlock::new(coeffs, (0, 0));
        assert_eq!(block.zero_count(), 62);
    }

    #[test]
    fn test_dct_1d_constant_input() {
        let input = vec![1.0; 8];
        let output = dct_1d(&input);
        // DC component should be sqrt(8) * 1/sqrt(8) = ~2.828
        assert!(output[0] > 0.0);
        // AC components should be ~0 for constant input
        for c in output.iter().skip(1) {
            assert!(c.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dct_2d_constant_block() {
        let block = [128.0; 64];
        let coeffs = dct_2d_8x8(&block);
        // DC should be non-zero
        assert!(coeffs[0] > 0.0);
        // All AC should be ~0
        for c in coeffs.iter().skip(1) {
            assert!(c.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dct_histogram_from_values() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let hist = DctHistogram::from_values(0, &values, 4);
        assert_eq!(hist.bins.len(), 4);
        assert!((hist.min_value).abs() < f64::EPSILON);
        assert!((hist.max_value - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dct_histogram_empty() {
        let hist = DctHistogram::from_values(0, &[], 4);
        assert_eq!(hist.bins.len(), 4);
    }

    #[test]
    fn test_config_default() {
        let config = FrequencyForensicsConfig::default();
        assert_eq!(config.num_histogram_bins, 64);
        assert!((config.periodicity_threshold - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_empty_blocks() {
        let analyzer = FrequencyForensicAnalyzer::with_defaults();
        let report = analyzer.analyze_blocks(&[]);
        assert_eq!(report.total_blocks, 0);
        assert!(report.anomalies.is_empty());
    }

    #[test]
    fn test_analyze_uniform_blocks() {
        let analyzer = FrequencyForensicAnalyzer::with_defaults();
        let blocks: Vec<DctBlock> = (0..16).map(|i| DctBlock::zero((i % 4, i / 4))).collect();
        let report = analyzer.analyze_blocks(&blocks);
        assert_eq!(report.total_blocks, 16);
        assert!((report.avg_sparsity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_image_too_small() {
        let analyzer = FrequencyForensicAnalyzer::with_defaults();
        let data = vec![128u8; 4 * 4];
        let report = analyzer.analyze_image(&data, 4, 4);
        assert_eq!(report.total_blocks, 0);
    }

    #[test]
    fn test_analyze_image_uniform() {
        let analyzer = FrequencyForensicAnalyzer::with_defaults();
        let data = vec![128u8; 64 * 64];
        let report = analyzer.analyze_image(&data, 64, 64);
        assert!(report.total_blocks > 0);
        // Uniform image should have very high sparsity
        assert!(report.avg_sparsity > 0.9);
    }
}
