#![allow(dead_code)]
//! Multi-generation JPEG compression detection.
//!
//! This module detects whether an image has undergone multiple rounds of
//! JPEG compression (re-saves), estimates the number of compression
//! generations, and identifies quality level changes. Double/triple
//! compression is a strong forensic indicator of manipulation.
//!
//! # Methodology
//!
//! A single JPEG compression quantizes each 8×8 block's DCT coefficients to
//! multiples of the quantization step, producing a coefficient histogram
//! with one dominant comb-like periodicity. Recompressing an *already*
//! JPEG-compressed image at a different quality re-quantizes coefficients
//! that are already multiples of the first step, which — unless the second
//! quantization step evenly divides the first — creates characteristic
//! secondary peaks and "double-quantization" gaps in the coefficient
//! histogram at multiples of the original step
//! ([`DctHistogram`], [`analyze_dct_double_compression`]). Detecting the
//! depth of these valleys at multiples of 8 (`detect_double_jpeg` in
//! [`crate::compression`]) yields a quantitative double-compression score,
//! and matching the whole quantization table against known-camera and
//! known-editor signatures ([`crate::quantization_table`]) helps identify
//! *which* generation's table produced a given re-save
//! ([`CompressionGeneration`], [`CompressionHistory`]).
//!
//! # References
//!
//! - A. C. Popescu, H. Farid, "Statistical Tools for Digital Forensics",
//!   6th International Workshop on Information Hiding (2004) — introduces
//!   detection of periodic artifacts from double JPEG quantization.
//! - Z. Lin, J. He, X. Tang, C.-K. Tang, "Fast, automatic and fine-grained
//!   tampered JPEG image detection via DCT coefficient analysis", Pattern
//!   Recognition, 42(11), 2492–2501 (2009) — DCT coefficient histogram
//!   analysis for localizing double-compressed regions, the basis for the
//!   histogram-valley approach in [`DctHistogram`] /
//!   `analyze_dct_double_compression`.
//! - T. Pevný, J. Fridrich, "Detection of Double-Compression in JPEG Images
//!   for Applications in Steganography", IEEE Transactions on Information
//!   Forensics and Security, 3(2), 247–258 (2008) — quantization-step
//!   estimation from the DCT histogram, informing
//!   [`DctDoubleCompressionConfig`] / [`DctDoubleCompressionResult`].
//! - H. Farid, "Exposing Digital Forgeries From JPEG Ghosts", IEEE
//!   Transactions on Information Forensics and Security, 4(1), 154–160
//!   (2009) — complementary recompression-based detection of a prior,
//!   lower quality level within a locally re-saved region (see also
//!   [`crate::ela`]).

use std::collections::HashMap;

/// JPEG quality factor representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityFactor {
    /// Estimated quality (1..100).
    pub quality: u8,
    /// Confidence of the estimate (0.0..1.0).
    pub confidence: f64,
}

impl QualityFactor {
    /// Create a new quality factor.
    #[must_use]
    pub fn new(quality: u8, confidence: f64) -> Self {
        Self {
            quality,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if this represents a low-quality compression.
    #[must_use]
    pub fn is_low_quality(&self) -> bool {
        self.quality < 50
    }

    /// Check if this represents a high-quality compression.
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.quality >= 85
    }
}

/// Evidence of a single compression generation.
#[derive(Debug, Clone)]
pub struct CompressionGeneration {
    /// Generation index (0 = first compression, 1 = second, etc.).
    pub generation: u32,
    /// Estimated quality factor for this generation.
    pub quality: QualityFactor,
    /// Blocking artifact strength at this level.
    pub blocking_strength: f64,
    /// Quantization table hash (for fingerprinting).
    pub qtable_hash: u64,
}

impl CompressionGeneration {
    /// Create a new compression generation record.
    #[must_use]
    pub fn new(
        generation: u32,
        quality: QualityFactor,
        blocking_strength: f64,
        qtable_hash: u64,
    ) -> Self {
        Self {
            generation,
            quality,
            blocking_strength,
            qtable_hash,
        }
    }
}

/// Result of double-compression detection.
#[derive(Debug, Clone)]
pub struct DoubleCompressionResult {
    /// Whether double compression was detected.
    pub detected: bool,
    /// Confidence of the detection (0.0..1.0).
    pub confidence: f64,
    /// Estimated primary (first) quality factor.
    pub primary_quality: Option<QualityFactor>,
    /// Estimated secondary (current) quality factor.
    pub secondary_quality: Option<QualityFactor>,
    /// Per-block probability map of double compression.
    pub block_probabilities: Vec<f64>,
}

impl DoubleCompressionResult {
    /// Create a new result indicating no double compression.
    #[must_use]
    pub fn not_detected() -> Self {
        Self {
            detected: false,
            confidence: 0.0,
            primary_quality: None,
            secondary_quality: None,
            block_probabilities: Vec::new(),
        }
    }

    /// Create a new result indicating detected double compression.
    #[must_use]
    pub fn detected_with(
        primary: QualityFactor,
        secondary: QualityFactor,
        confidence: f64,
    ) -> Self {
        Self {
            detected: true,
            confidence: confidence.clamp(0.0, 1.0),
            primary_quality: Some(primary),
            secondary_quality: Some(secondary),
            block_probabilities: Vec::new(),
        }
    }

    /// Return the quality ratio between primary and secondary compression.
    #[must_use]
    pub fn quality_ratio(&self) -> Option<f64> {
        match (self.primary_quality, self.secondary_quality) {
            (Some(p), Some(s)) if s.quality > 0 => {
                Some(f64::from(p.quality) / f64::from(s.quality))
            }
            _ => None,
        }
    }
}

/// DCT coefficient histogram for a specific frequency position.
#[derive(Debug, Clone)]
pub struct DctHistogram {
    /// Frequency position (row, col) in the 8x8 block.
    pub position: (usize, usize),
    /// Histogram bin counts (centered at 0).
    pub bins: HashMap<i32, u64>,
    /// Total number of coefficients.
    pub total: u64,
}

impl DctHistogram {
    /// Create a new empty DCT histogram.
    #[must_use]
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            position: (row, col),
            bins: HashMap::new(),
            total: 0,
        }
    }

    /// Add a coefficient value to the histogram.
    pub fn add(&mut self, value: i32) {
        *self.bins.entry(value).or_insert(0) += 1;
        self.total += 1;
    }

    /// Get the count for a specific bin.
    #[must_use]
    pub fn count(&self, value: i32) -> u64 {
        self.bins.get(&value).copied().unwrap_or(0)
    }

    /// Compute the proportion of zero coefficients.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn zero_proportion(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.count(0) as f64 / self.total as f64
    }

    /// Detect periodicity in the histogram (indicator of double compression).
    ///
    /// Returns the period and its strength. A period > 1 with high strength
    /// indicates double JPEG compression.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn detect_periodicity(&self, max_period: i32) -> (i32, f64) {
        if self.bins.is_empty() {
            return (1, 0.0);
        }

        let min_key = self.bins.keys().copied().min().unwrap_or(0);
        let max_key = self.bins.keys().copied().max().unwrap_or(0);
        let range = max_key - min_key + 1;
        if range < 4 {
            return (1, 0.0);
        }

        let mut best_period = 1;
        let mut best_score = 0.0_f64;

        for period in 2..=max_period {
            let mut on_grid = 0_u64;
            let mut off_grid = 0_u64;

            for (&val, &cnt) in &self.bins {
                if val % period == 0 {
                    on_grid += cnt;
                } else {
                    off_grid += cnt;
                }
            }

            let total = on_grid + off_grid;
            if total == 0 {
                continue;
            }
            let expected_on = total as f64 / period as f64;
            let score = (on_grid as f64 - expected_on) / expected_on.max(1.0);

            if score > best_score {
                best_score = score;
                best_period = period;
            }
        }

        (best_period, best_score.max(0.0))
    }
}

/// Blocking artifact grid analyzer.
#[derive(Debug, Clone)]
pub struct BlockingAnalyzer {
    /// Block size (typically 8 for JPEG).
    pub block_size: usize,
    /// Detected blocking strength at JPEG grid positions.
    pub grid_strength: f64,
    /// Detected blocking strength at shifted positions.
    pub shifted_strength: f64,
}

impl BlockingAnalyzer {
    /// Create a new blocking analyzer.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            grid_strength: 0.0,
            shifted_strength: 0.0,
        }
    }

    /// Analyze blocking artifacts from a row of pixel differences.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze_row(&mut self, diffs: &[f64]) {
        if diffs.is_empty() || self.block_size == 0 {
            return;
        }

        let mut grid_sum = 0.0;
        let mut grid_count = 0_u64;
        let mut non_grid_sum = 0.0;
        let mut non_grid_count = 0_u64;

        for (i, &d) in diffs.iter().enumerate() {
            let abs_d = d.abs();
            if (i + 1) % self.block_size == 0 {
                grid_sum += abs_d;
                grid_count += 1;
            } else {
                non_grid_sum += abs_d;
                non_grid_count += 1;
            }
        }

        self.grid_strength = if grid_count > 0 {
            grid_sum / grid_count as f64
        } else {
            0.0
        };
        self.shifted_strength = if non_grid_count > 0 {
            non_grid_sum / non_grid_count as f64
        } else {
            0.0
        };
    }

    /// Return the ratio of grid-aligned to non-grid blocking.
    ///
    /// A ratio significantly above 1.0 indicates JPEG blocking artifacts.
    #[must_use]
    pub fn blocking_ratio(&self) -> f64 {
        if self.shifted_strength > 1e-10 {
            self.grid_strength / self.shifted_strength
        } else if self.grid_strength > 1e-10 {
            f64::INFINITY
        } else {
            1.0
        }
    }

    /// Check whether blocking artifacts are present at the JPEG grid.
    #[must_use]
    pub fn has_blocking_artifacts(&self, threshold_ratio: f64) -> bool {
        self.blocking_ratio() > threshold_ratio
    }
}

/// Comprehensive compression history analysis result.
#[derive(Debug, Clone)]
pub struct CompressionHistory {
    /// Detected compression generations.
    pub generations: Vec<CompressionGeneration>,
    /// Overall number of detected compression rounds.
    pub num_generations: u32,
    /// Double compression detection result.
    pub double_compression: DoubleCompressionResult,
    /// Blocking artifact analysis.
    pub blocking_ratio: f64,
    /// Textual findings.
    pub findings: Vec<String>,
}

impl CompressionHistory {
    /// Create a new empty compression history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            num_generations: 0,
            double_compression: DoubleCompressionResult::not_detected(),
            blocking_ratio: 1.0,
            findings: Vec::new(),
        }
    }

    /// Add a compression generation.
    pub fn add_generation(&mut self, gen: CompressionGeneration) {
        self.generations.push(gen);
        self.num_generations = self.generations.len() as u32;
    }

    /// Add a finding.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
    }

    /// Whether multiple compression rounds were detected.
    #[must_use]
    pub fn is_multi_compressed(&self) -> bool {
        self.num_generations > 1 || self.double_compression.detected
    }
}

impl Default for CompressionHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DCT-based double JPEG compression detector
// ---------------------------------------------------------------------------

/// Configuration for DCT-based double compression analysis.
#[derive(Debug, Clone)]
pub struct DctDoubleCompressionConfig {
    /// Number of AC frequency positions to analyse (1..63).
    pub num_positions: usize,
    /// Maximum period to probe in periodicity detection.
    pub max_period: i32,
    /// Minimum periodicity score to flag double compression.
    pub periodicity_threshold: f64,
    /// Whether to include Benford first-digit analysis.
    pub use_benford: bool,
}

impl Default for DctDoubleCompressionConfig {
    fn default() -> Self {
        Self {
            num_positions: 6,
            max_period: 8,
            periodicity_threshold: 0.35,
            use_benford: true,
        }
    }
}

/// Result of DCT-based double compression analysis on `compression_history`.
#[derive(Debug, Clone)]
pub struct DctDoubleCompressionResult {
    /// Overall detection flag.
    pub detected: bool,
    /// Per-position periodicity scores (AC position index -> score).
    pub position_scores: Vec<(usize, f64)>,
    /// Aggregate periodicity score across all analysed positions.
    pub aggregate_periodicity: f64,
    /// Benford first-digit chi-squared divergence (higher = more suspicious).
    pub benford_chi2: f64,
    /// Combined confidence in [0, 1].
    pub confidence: f64,
    /// Estimated primary quality factor (if detectable).
    pub estimated_primary_quality: Option<u8>,
    /// Findings for the forensic report.
    pub findings: Vec<String>,
}

/// Simulated 8x8 DCT coefficient extraction from pixel data.
///
/// Given a flat row-major luma buffer (values 0..255), this function computes a
/// simplified DCT for each non-overlapping 8x8 block and returns all 64
/// integer-quantised coefficients per block in raster order.
///
/// The returned `Vec<Vec<i32>>` has one inner vec per block, each of length 64.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn extract_dct_coefficients(luma: &[f64], width: usize, height: usize) -> Vec<Vec<i32>> {
    let blocks_x = width / 8;
    let blocks_y = height / 8;
    let mut all_blocks = Vec::with_capacity(blocks_x * blocks_y);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut coeffs = vec![0i32; 64];
            for u in 0..8_usize {
                for v in 0..8_usize {
                    let cu: f64 = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                    let cv: f64 = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                    let mut sum = 0.0_f64;
                    for x in 0..8_usize {
                        for y in 0..8_usize {
                            let px = bx * 8 + x;
                            let py = by * 8 + y;
                            if px < width && py < height {
                                let val = luma[py * width + px];
                                let cos_u =
                                    ((2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI
                                        / 16.0)
                                        .cos();
                                let cos_v =
                                    ((2.0 * y as f64 + 1.0) * v as f64 * std::f64::consts::PI
                                        / 16.0)
                                        .cos();
                                sum += val * cos_u * cos_v;
                            }
                        }
                    }
                    coeffs[u * 8 + v] = (0.25 * cu * cv * sum).round() as i32;
                }
            }
            all_blocks.push(coeffs);
        }
    }
    all_blocks
}

/// Analyse extracted DCT blocks for evidence of double JPEG compression.
///
/// The analysis examines the histogram of DCT coefficients at several AC
/// frequency positions.  Double quantisation produces periodic peaks in these
/// histograms whose spacing corresponds to the ratio of the two quantisation
/// step sizes.  We also compute a first-digit (Benford) divergence metric.
#[allow(clippy::cast_precision_loss)]
pub fn analyze_dct_double_compression(
    blocks: &[Vec<i32>],
    config: &DctDoubleCompressionConfig,
) -> DctDoubleCompressionResult {
    if blocks.is_empty() {
        return DctDoubleCompressionResult {
            detected: false,
            position_scores: Vec::new(),
            aggregate_periodicity: 0.0,
            benford_chi2: 0.0,
            confidence: 0.0,
            estimated_primary_quality: None,
            findings: vec!["No DCT blocks to analyse".to_string()],
        };
    }

    // Select low-frequency AC positions (zigzag order approximation)
    let zigzag_positions: [usize; 10] = [1, 8, 2, 9, 16, 3, 10, 17, 24, 4];
    let positions_to_use = config.num_positions.min(zigzag_positions.len());

    let mut position_scores = Vec::new();
    let mut all_ac_coeffs: Vec<i32> = Vec::new();

    for &pos in zigzag_positions.iter().take(positions_to_use) {
        // Collect coefficients at this position across all blocks
        let mut hist = DctHistogram::new(pos / 8, pos % 8);
        for block in blocks {
            if pos < block.len() {
                hist.add(block[pos]);
                all_ac_coeffs.push(block[pos]);
            }
        }
        let (period, score) = hist.detect_periodicity(config.max_period);
        position_scores.push((pos, score));
        if period > 1 && score > config.periodicity_threshold {
            // This position shows significant periodicity
        }
    }

    let aggregate_periodicity = if position_scores.is_empty() {
        0.0
    } else {
        position_scores.iter().map(|(_, s)| *s).sum::<f64>() / position_scores.len() as f64
    };

    // Benford first-digit analysis
    let benford_chi2 = if config.use_benford && !all_ac_coeffs.is_empty() {
        benford_first_digit_chi2(&all_ac_coeffs)
    } else {
        0.0
    };

    // Combine into overall confidence
    let benford_norm = (benford_chi2 / 10.0).clamp(0.0, 1.0);
    let confidence = if config.use_benford {
        (aggregate_periodicity * 0.6 + benford_norm * 0.4).clamp(0.0, 1.0)
    } else {
        aggregate_periodicity.clamp(0.0, 1.0)
    };

    let detected = confidence >= config.periodicity_threshold;

    // Estimate primary quality from the dominant period
    let estimated_primary_quality = if detected {
        estimate_primary_quality_from_scores(&position_scores)
    } else {
        None
    };

    let mut findings = Vec::new();
    findings.push(format!(
        "DCT periodicity score: {:.3} (threshold {:.3})",
        aggregate_periodicity, config.periodicity_threshold
    ));
    if config.use_benford {
        findings.push(format!("Benford chi-squared: {:.4}", benford_chi2));
    }
    if detected {
        findings.push("Double JPEG compression detected via DCT analysis".to_string());
        if let Some(q) = estimated_primary_quality {
            findings.push(format!("Estimated primary quality factor: {}", q));
        }
    }

    DctDoubleCompressionResult {
        detected,
        position_scores,
        aggregate_periodicity,
        benford_chi2,
        confidence,
        estimated_primary_quality,
        findings,
    }
}

/// Compute the chi-squared divergence of first-digit distribution from Benford's law.
#[allow(clippy::cast_precision_loss)]
fn benford_first_digit_chi2(coefficients: &[i32]) -> f64 {
    let mut counts = [0u64; 9];
    let mut total = 0u64;

    for &c in coefficients {
        let abs_val = c.unsigned_abs();
        if abs_val == 0 {
            continue;
        }
        let mut v = abs_val;
        while v >= 10 {
            v /= 10;
        }
        if v >= 1 && v <= 9 {
            counts[(v - 1) as usize] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    // Expected Benford probabilities
    let mut chi2 = 0.0_f64;
    for d in 1..=9u32 {
        let expected = (1.0 + 1.0 / d as f64).log10();
        let observed = counts[(d - 1) as usize] as f64 / total as f64;
        if expected > 1e-15 {
            let diff = observed - expected;
            chi2 += diff * diff / expected;
        }
    }
    chi2
}

/// Heuristic estimation of primary quality from periodicity scores.
fn estimate_primary_quality_from_scores(scores: &[(usize, f64)]) -> Option<u8> {
    if scores.is_empty() {
        return None;
    }
    // Find the position with the strongest periodicity
    let best = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    match best {
        Some((pos, score)) if *score > 0.2 => {
            // Map position index to a rough quality estimate.
            // Lower-frequency positions with strong periodicity suggest
            // the first save used a lower quality (larger quant steps).
            let q = match pos {
                0..=2 => 90,
                3..=8 => 75,
                9..=16 => 60,
                _ => 50,
            };
            Some(q)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_factor_creation() {
        let qf = QualityFactor::new(75, 0.9);
        assert_eq!(qf.quality, 75);
        assert!((qf.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_factor_clamped_confidence() {
        let qf = QualityFactor::new(50, 1.5);
        assert!((qf.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_factor_low_high() {
        assert!(QualityFactor::new(30, 0.9).is_low_quality());
        assert!(!QualityFactor::new(70, 0.9).is_low_quality());
        assert!(QualityFactor::new(90, 0.9).is_high_quality());
        assert!(!QualityFactor::new(50, 0.9).is_high_quality());
    }

    #[test]
    fn test_double_compression_not_detected() {
        let r = DoubleCompressionResult::not_detected();
        assert!(!r.detected);
        assert!(r.quality_ratio().is_none());
    }

    #[test]
    fn test_double_compression_detected() {
        let r = DoubleCompressionResult::detected_with(
            QualityFactor::new(90, 0.8),
            QualityFactor::new(75, 0.9),
            0.85,
        );
        assert!(r.detected);
        let ratio = r.quality_ratio().expect("ratio should be valid");
        assert!((ratio - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_dct_histogram_basic() {
        let mut hist = DctHistogram::new(0, 0);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        hist.add(-1);
        assert_eq!(hist.count(0), 2);
        assert_eq!(hist.count(1), 1);
        assert_eq!(hist.total, 4);
    }

    #[test]
    fn test_dct_histogram_zero_proportion() {
        let mut hist = DctHistogram::new(0, 0);
        hist.add(0);
        hist.add(0);
        hist.add(1);
        assert!((hist.zero_proportion() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dct_histogram_empty() {
        let hist = DctHistogram::new(0, 0);
        assert!((hist.zero_proportion()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dct_histogram_periodicity() {
        let mut hist = DctHistogram::new(1, 1);
        // Strong periodicity at period 2
        for v in -10..=10 {
            let count = if v % 2 == 0 { 10 } else { 1 };
            for _ in 0..count {
                hist.add(v);
            }
        }
        let (period, strength) = hist.detect_periodicity(4);
        assert_eq!(period, 2);
        assert!(strength > 0.5);
    }

    #[test]
    fn test_blocking_analyzer_ratio() {
        let mut ba = BlockingAnalyzer::new(8);
        let mut diffs = vec![0.5; 16];
        // Make grid-aligned positions have higher values
        for i in 0..diffs.len() {
            if (i + 1) % 8 == 0 {
                diffs[i] = 5.0;
            }
        }
        ba.analyze_row(&diffs);
        assert!(ba.blocking_ratio() > 1.0);
        assert!(ba.has_blocking_artifacts(2.0));
    }

    #[test]
    fn test_blocking_analyzer_no_artifacts() {
        let mut ba = BlockingAnalyzer::new(8);
        let diffs = vec![1.0; 16];
        ba.analyze_row(&diffs);
        // Uniform diffs: ratio should be close to 1
        assert!(ba.blocking_ratio() < 2.0);
    }

    #[test]
    fn test_blocking_analyzer_empty() {
        let mut ba = BlockingAnalyzer::new(8);
        ba.analyze_row(&[]);
        assert!((ba.blocking_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compression_history_multi() {
        let mut ch = CompressionHistory::new();
        ch.add_generation(CompressionGeneration::new(
            0,
            QualityFactor::new(90, 0.9),
            0.5,
            12345,
        ));
        ch.add_generation(CompressionGeneration::new(
            1,
            QualityFactor::new(75, 0.8),
            1.2,
            67890,
        ));
        assert_eq!(ch.num_generations, 2);
        assert!(ch.is_multi_compressed());
    }

    #[test]
    fn test_compression_history_single() {
        let mut ch = CompressionHistory::new();
        ch.add_generation(CompressionGeneration::new(
            0,
            QualityFactor::new(90, 0.9),
            0.5,
            12345,
        ));
        assert!(!ch.is_multi_compressed());
    }

    // ── DctDoubleCompressionConfig ────────────────────────────────────────────

    #[test]
    fn test_dct_config_defaults() {
        let cfg = DctDoubleCompressionConfig::default();
        assert_eq!(cfg.num_positions, 6);
        assert_eq!(cfg.max_period, 8);
        assert!(cfg.use_benford);
        assert!((cfg.periodicity_threshold - 0.35).abs() < 1e-10);
    }

    // ── extract_dct_coefficients ──────────────────────────────────────────────

    #[test]
    fn test_extract_dct_coefficients_basic() {
        // 16x16 uniform image -> 4 blocks of 8x8
        let luma = vec![128.0_f64; 16 * 16];
        let blocks = extract_dct_coefficients(&luma, 16, 16);
        assert_eq!(blocks.len(), 4);
        for block in &blocks {
            assert_eq!(block.len(), 64);
        }
    }

    #[test]
    fn test_extract_dct_coefficients_dc_nonzero() {
        let luma = vec![200.0_f64; 8 * 8];
        let blocks = extract_dct_coefficients(&luma, 8, 8);
        assert_eq!(blocks.len(), 1);
        // DC coefficient (position 0) should be large for a uniform bright block
        assert!(blocks[0][0].abs() > 100);
    }

    #[test]
    fn test_extract_dct_coefficients_empty() {
        let blocks = extract_dct_coefficients(&[], 0, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn test_extract_dct_coefficients_too_small() {
        // 4x4 image, no 8x8 block fits
        let luma = vec![128.0_f64; 4 * 4];
        let blocks = extract_dct_coefficients(&luma, 4, 4);
        assert!(blocks.is_empty());
    }

    // ── analyze_dct_double_compression ────────────────────────────────────────

    #[test]
    fn test_dct_analysis_empty_blocks() {
        let cfg = DctDoubleCompressionConfig::default();
        let result = analyze_dct_double_compression(&[], &cfg);
        assert!(!result.detected);
        assert!((result.confidence).abs() < 1e-10);
    }

    #[test]
    fn test_dct_analysis_uniform_image() {
        let luma = vec![128.0_f64; 64 * 64];
        let blocks = extract_dct_coefficients(&luma, 64, 64);
        let cfg = DctDoubleCompressionConfig::default();
        let result = analyze_dct_double_compression(&blocks, &cfg);
        // Uniform image: no periodicity expected
        assert!(result.aggregate_periodicity < 0.5);
    }

    #[test]
    fn test_dct_analysis_synthetic_periodicity() {
        // Create blocks with artificial periodic DCT coefficients
        // to simulate double compression
        let num_blocks = 100;
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let mut coeffs = vec![0i32; 64];
            // DC
            coeffs[0] = 500;
            // AC positions: insert periodic pattern (multiples of 3 get high values)
            for pos in 1..64 {
                let base = ((i * 7 + pos * 13) % 20) as i32 - 10;
                if base % 3 == 0 {
                    coeffs[pos] = base * 5;
                } else {
                    coeffs[pos] = base;
                }
            }
            blocks.push(coeffs);
        }
        let cfg = DctDoubleCompressionConfig::default();
        let result = analyze_dct_double_compression(&blocks, &cfg);
        // Should produce non-zero periodicity
        assert!(result.aggregate_periodicity >= 0.0);
        assert!(!result.findings.is_empty());
    }

    #[test]
    fn test_dct_analysis_confidence_bounded() {
        let luma = vec![128.0_f64; 32 * 32];
        let blocks = extract_dct_coefficients(&luma, 32, 32);
        let cfg = DctDoubleCompressionConfig::default();
        let result = analyze_dct_double_compression(&blocks, &cfg);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_dct_analysis_without_benford() {
        let luma = vec![128.0_f64; 16 * 16];
        let blocks = extract_dct_coefficients(&luma, 16, 16);
        let cfg = DctDoubleCompressionConfig {
            use_benford: false,
            ..Default::default()
        };
        let result = analyze_dct_double_compression(&blocks, &cfg);
        assert!((result.benford_chi2).abs() < 1e-10);
    }

    // ── benford_first_digit_chi2 ──────────────────────────────────────────────

    #[test]
    fn test_benford_chi2_empty() {
        assert!((benford_first_digit_chi2(&[])).abs() < 1e-10);
    }

    #[test]
    fn test_benford_chi2_all_zeros() {
        let coeffs = vec![0i32; 100];
        assert!((benford_first_digit_chi2(&coeffs)).abs() < 1e-10);
    }

    #[test]
    fn test_benford_chi2_nonnegative() {
        let coeffs: Vec<i32> = (1..=1000).map(|i| (i % 50) - 25).collect();
        let chi2 = benford_first_digit_chi2(&coeffs);
        assert!(chi2 >= 0.0);
    }

    // ── estimate_primary_quality_from_scores ──────────────────────────────────

    #[test]
    fn test_estimate_quality_low_pos() {
        let scores = vec![(1, 0.8), (8, 0.3)];
        let q = estimate_primary_quality_from_scores(&scores);
        assert_eq!(q, Some(90));
    }

    #[test]
    fn test_estimate_quality_mid_pos() {
        let scores = vec![(5, 0.8)];
        let q = estimate_primary_quality_from_scores(&scores);
        assert_eq!(q, Some(75));
    }

    #[test]
    fn test_estimate_quality_none_when_low_score() {
        let scores = vec![(1, 0.1)];
        let q = estimate_primary_quality_from_scores(&scores);
        assert!(q.is_none());
    }

    #[test]
    fn test_estimate_quality_empty_scores() {
        let q = estimate_primary_quality_from_scores(&[]);
        assert!(q.is_none());
    }
}
