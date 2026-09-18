//! Cross-codec comparison tools for analyzing benchmark results.

use crate::bd_rate::{BdPoint, BdRateCalculator};
use crate::CodecBenchmarkResult;
use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};

/// Compute the Bjontegaard Delta Rate (BD-Rate) between two rate-distortion curves.
///
/// Each element of the input slices is `(bitrate_kbps, psnr_db)`.
/// A **negative** result means the test codec achieves the same quality at lower bitrate.
/// Returns `0.0` when the curves are identical or computation fails (e.g., insufficient points).
///
/// # Reference
/// G. Bjontegaard, ITU-T SG16 VCEG-M33, Austin TX, Apr. 2001.
#[must_use]
pub fn compute_bd_rate(rd_points_a: &[(f64, f64)], rd_points_b: &[(f64, f64)]) -> f64 {
    let ref_pts: Vec<BdPoint> = rd_points_a
        .iter()
        .map(|&(bitrate, quality)| BdPoint::new(bitrate, quality))
        .collect();
    let tst_pts: Vec<BdPoint> = rd_points_b
        .iter()
        .map(|&(bitrate, quality)| BdPoint::new(bitrate, quality))
        .collect();

    match BdRateCalculator::compute(&ref_pts, &tst_pts) {
        Ok(result) => result.bd_rate,
        Err(_) => 0.0,
    }
}

/// Result of comparing two codecs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Speed ratio (codec_a / codec_b)
    pub encoding_speed_ratio: f64,
    /// Decoding speed ratio
    pub decoding_speed_ratio: f64,
    /// Quality difference (PSNR)
    pub psnr_difference: Option<f64>,
    /// SSIM difference
    pub ssim_difference: Option<f64>,
    /// File size ratio
    pub file_size_ratio: f64,
    /// Efficiency score (quality per bit)
    pub efficiency_score: Option<f64>,
}

/// Codec comparison utilities.
pub struct CodecComparison;

impl CodecComparison {
    /// Compare two codec results.
    #[must_use]
    pub fn compare(
        codec_a: Vec<&CodecBenchmarkResult>,
        codec_b: Vec<&CodecBenchmarkResult>,
    ) -> ComparisonResult {
        let avg_a = Self::average_results(&codec_a);
        let avg_b = Self::average_results(&codec_b);

        let encoding_speed_ratio = avg_a.encoding_fps / avg_b.encoding_fps;
        let decoding_speed_ratio = avg_a.decoding_fps / avg_b.decoding_fps;
        let file_size_ratio = avg_a.file_size as f64 / avg_b.file_size as f64;

        let psnr_difference = match (avg_a.psnr, avg_b.psnr) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };

        let ssim_difference = match (avg_a.ssim, avg_b.ssim) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };

        let efficiency_score = if let (Some(psnr_a), Some(psnr_b)) = (avg_a.psnr, avg_b.psnr) {
            let eff_a = psnr_a / (avg_a.file_size as f64 / 1_000_000.0);
            let eff_b = psnr_b / (avg_b.file_size as f64 / 1_000_000.0);
            Some(eff_a / eff_b)
        } else {
            None
        };

        ComparisonResult {
            encoding_speed_ratio,
            decoding_speed_ratio,
            psnr_difference,
            ssim_difference,
            file_size_ratio,
            efficiency_score,
        }
    }

    fn average_results(results: &[&CodecBenchmarkResult]) -> AverageMetrics {
        let mut total_encoding_fps = 0.0;
        let mut total_decoding_fps = 0.0;
        let mut total_file_size = 0u64;
        let mut total_psnr = 0.0;
        let mut total_ssim = 0.0;
        let mut psnr_count = 0;
        let mut ssim_count = 0;
        let mut count = 0;

        for result in results {
            for seq in &result.sequence_results {
                total_encoding_fps += seq.encoding_fps;
                total_decoding_fps += seq.decoding_fps;
                total_file_size += seq.file_size_bytes;

                if let Some(psnr) = seq.metrics.psnr {
                    total_psnr += psnr;
                    psnr_count += 1;
                }

                if let Some(ssim) = seq.metrics.ssim {
                    total_ssim += ssim;
                    ssim_count += 1;
                }

                count += 1;
            }
        }

        if count == 0 {
            return AverageMetrics::default();
        }

        AverageMetrics {
            encoding_fps: total_encoding_fps / count as f64,
            decoding_fps: total_decoding_fps / count as f64,
            file_size: total_file_size / count as u64,
            psnr: if psnr_count > 0 {
                Some(total_psnr / psnr_count as f64)
            } else {
                None
            },
            ssim: if ssim_count > 0 {
                Some(total_ssim / ssim_count as f64)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Default)]
struct AverageMetrics {
    encoding_fps: f64,
    decoding_fps: f64,
    file_size: u64,
    psnr: Option<f64>,
    ssim: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodecBenchmarkResult, QualityMetrics, SequenceResult, Statistics};
    use oximedia_core::types::CodecId;
    use std::time::Duration;

    fn make_sequence_result(
        name: &str,
        encoding_fps: f64,
        decoding_fps: f64,
        file_size: u64,
        psnr: Option<f64>,
    ) -> SequenceResult {
        SequenceResult {
            sequence_name: name.to_string(),
            frames_processed: 100,
            encoding_fps,
            decoding_fps,
            file_size_bytes: file_size,
            metrics: QualityMetrics {
                psnr,
                ..Default::default()
            },
            encoding_duration: Duration::from_secs(1),
            decoding_duration: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_comparison_result() {
        let result_a = CodecBenchmarkResult {
            codec_id: CodecId::Av1,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            sequence_results: vec![make_sequence_result(
                "seq1",
                30.0,
                60.0,
                1_000_000,
                Some(40.0),
            )],
            statistics: Statistics::default(),
        };

        let result_b = CodecBenchmarkResult {
            codec_id: CodecId::Vp9,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            sequence_results: vec![make_sequence_result(
                "seq1",
                60.0,
                120.0,
                1_200_000,
                Some(38.0),
            )],
            statistics: Statistics::default(),
        };

        let comparison = CodecComparison::compare(vec![&result_a], vec![&result_b]);

        assert_eq!(comparison.encoding_speed_ratio, 0.5);
        assert_eq!(comparison.decoding_speed_ratio, 0.5);
        assert_eq!(comparison.psnr_difference, Some(2.0));
    }

    #[test]
    fn test_bd_rate_equal_curves() {
        // Two identical RD curves must yield BD-Rate ≈ 0.
        let pts = [
            (500.0_f64, 32.0_f64),
            (1000.0, 35.0),
            (2000.0, 38.0),
            (4000.0, 41.0),
        ];
        let bd = compute_bd_rate(&pts, &pts);
        assert!(
            bd.abs() < 1e-6,
            "BD-Rate for identical curves should be ~0, got {bd:.8}"
        );
    }
}

/// Detailed codec comparison with statistical analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedComparison {
    /// Basic comparison result
    pub basic: ComparisonResult,
    /// Speed comparison details
    pub speed: SpeedComparison,
    /// Quality comparison details
    pub quality: QualityComparison,
    /// Efficiency comparison details
    pub efficiency: EfficiencyComparison,
}

/// Speed comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedComparison {
    /// Encoding speed statistics
    pub encoding: SpeedStats,
    /// Decoding speed statistics
    pub decoding: SpeedStats,
}

/// Speed statistics for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedStats {
    /// Mean FPS for codec A
    pub mean_a: f64,
    /// Mean FPS for codec B
    pub mean_b: f64,
    /// Median FPS for codec A
    pub median_a: f64,
    /// Median FPS for codec B
    pub median_b: f64,
    /// Standard deviation for codec A
    pub std_dev_a: f64,
    /// Standard deviation for codec B
    pub std_dev_b: f64,
    /// Speed ratio (A/B)
    pub ratio: f64,
    /// Statistical significance p-value
    pub p_value: Option<f64>,
}

/// Quality comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityComparison {
    /// PSNR comparison
    pub psnr: Option<QualityStats>,
    /// SSIM comparison
    pub ssim: Option<QualityStats>,
    /// VMAF comparison
    pub vmaf: Option<QualityStats>,
}

/// Quality statistics for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    /// Mean quality for codec A
    pub mean_a: f64,
    /// Mean quality for codec B
    pub mean_b: f64,
    /// Quality difference
    pub difference: f64,
    /// Relative improvement percentage
    pub relative_improvement: f64,
}

/// Efficiency comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyComparison {
    /// Bits per pixel
    pub bpp: BppComparison,
    /// Quality per bit
    pub quality_per_bit: Option<f64>,
    /// Compression ratio
    pub compression_ratio: CompressionRatio,
}

/// Bits per pixel comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BppComparison {
    /// BPP for codec A
    pub bpp_a: f64,
    /// BPP for codec B
    pub bpp_b: f64,
    /// BPP ratio
    pub ratio: f64,
}

/// Compression ratio comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionRatio {
    /// Compression ratio for codec A
    pub ratio_a: f64,
    /// Compression ratio for codec B
    pub ratio_b: f64,
    /// Relative efficiency
    pub relative_efficiency: f64,
}

/// Preset comparison for analyzing different encoding presets.
#[derive(Debug, Clone)]
pub struct PresetComparison {
    codec_id: CodecId,
    presets: Vec<String>,
}

impl PresetComparison {
    /// Create a new preset comparison.
    #[must_use]
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            presets: Vec::new(),
        }
    }

    /// Add a preset to compare.
    pub fn add_preset(&mut self, preset: impl Into<String>) {
        self.presets.push(preset.into());
    }

    /// Compare presets.
    #[must_use]
    pub fn compare(&self, _results: &[&CodecBenchmarkResult]) -> PresetComparisonResult {
        // Placeholder for preset comparison
        PresetComparisonResult {
            codec_id: self.codec_id,
            fastest_preset: String::new(),
            highest_quality_preset: String::new(),
            best_balanced_preset: String::new(),
        }
    }
}

/// Result of preset comparison.
#[derive(Debug, Clone)]
pub struct PresetComparisonResult {
    /// Codec being compared
    pub codec_id: CodecId,
    /// Fastest encoding preset
    pub fastest_preset: String,
    /// Highest quality preset
    pub highest_quality_preset: String,
    /// Best balanced preset (speed vs quality)
    pub best_balanced_preset: String,
}

/// Rate-distortion curve comparison.
#[derive(Debug, Clone)]
pub struct RdCurveComparison {
    points_a: Vec<RdPoint>,
    points_b: Vec<RdPoint>,
}

/// Rate-distortion point.
#[derive(Debug, Clone, Copy)]
pub struct RdPoint {
    /// Bitrate in kbps
    pub bitrate: f64,
    /// Quality metric (PSNR or VMAF)
    pub quality: f64,
}

impl RdCurveComparison {
    /// Create a new RD curve comparison.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points_a: Vec::new(),
            points_b: Vec::new(),
        }
    }

    /// Add a point for codec A.
    pub fn add_point_a(&mut self, bitrate: f64, quality: f64) {
        self.points_a.push(RdPoint { bitrate, quality });
    }

    /// Add a point for codec B.
    pub fn add_point_b(&mut self, bitrate: f64, quality: f64) {
        self.points_b.push(RdPoint { bitrate, quality });
    }

    /// Calculate BD-Rate (Bjøntegaard Delta Rate).
    #[must_use]
    pub fn calculate_bd_rate(&self) -> Option<f64> {
        if self.points_a.len() < 2 || self.points_b.len() < 2 {
            return None;
        }

        // Simplified BD-Rate calculation (placeholder)
        // Real implementation would use polynomial fitting
        Some(0.0)
    }

    /// Calculate BD-PSNR (Bjøntegaard Delta PSNR).
    #[must_use]
    pub fn calculate_bd_psnr(&self) -> Option<f64> {
        if self.points_a.len() < 2 || self.points_b.len() < 2 {
            return None;
        }

        // Simplified BD-PSNR calculation (placeholder)
        Some(0.0)
    }
}

impl Default for RdCurveComparison {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec ranking based on multiple criteria.
#[derive(Debug, Clone)]
pub struct CodecRanking {
    rankings: Vec<RankingEntry>,
}

/// Ranking entry for a codec.
#[derive(Debug, Clone)]
pub struct RankingEntry {
    /// Codec identifier
    pub codec_id: CodecId,
    /// Overall score (0-100)
    pub overall_score: f64,
    /// Speed score (0-100)
    pub speed_score: f64,
    /// Quality score (0-100)
    pub quality_score: f64,
    /// Efficiency score (0-100)
    pub efficiency_score: f64,
}

impl CodecRanking {
    /// Create a new codec ranking.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rankings: Vec::new(),
        }
    }

    /// Add a codec to the ranking.
    pub fn add(&mut self, entry: RankingEntry) {
        self.rankings.push(entry);
    }

    /// Sort rankings by overall score.
    pub fn sort_by_overall(&mut self) {
        self.rankings.sort_by(|a, b| {
            b.overall_score
                .partial_cmp(&a.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the top codec.
    #[must_use]
    pub fn top_codec(&self) -> Option<&RankingEntry> {
        self.rankings.first()
    }

    /// Get rankings.
    #[must_use]
    pub fn rankings(&self) -> &[RankingEntry] {
        &self.rankings
    }
}

impl Default for CodecRanking {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_speed_stats() {
        let stats = SpeedStats {
            mean_a: 30.0,
            mean_b: 60.0,
            median_a: 29.0,
            median_b: 59.0,
            std_dev_a: 2.0,
            std_dev_b: 3.0,
            ratio: 0.5,
            p_value: Some(0.01),
        };

        assert_eq!(stats.ratio, 0.5);
    }

    #[test]
    fn test_quality_stats() {
        let stats = QualityStats {
            mean_a: 38.0,
            mean_b: 40.0,
            difference: 2.0,
            relative_improvement: 5.26,
        };

        assert_eq!(stats.difference, 2.0);
    }

    #[test]
    fn test_rd_curve() {
        let mut curve = RdCurveComparison::new();
        curve.add_point_a(1000.0, 35.0);
        curve.add_point_a(2000.0, 38.0);
        curve.add_point_b(1000.0, 34.0);
        curve.add_point_b(2000.0, 37.5);

        assert_eq!(curve.points_a.len(), 2);
        assert_eq!(curve.points_b.len(), 2);
    }

    #[test]
    fn test_codec_ranking() {
        let mut ranking = CodecRanking::new();

        ranking.add(RankingEntry {
            codec_id: CodecId::Av1,
            overall_score: 85.0,
            speed_score: 70.0,
            quality_score: 95.0,
            efficiency_score: 90.0,
        });

        ranking.add(RankingEntry {
            codec_id: CodecId::Vp9,
            overall_score: 80.0,
            speed_score: 75.0,
            quality_score: 90.0,
            efficiency_score: 75.0,
        });

        ranking.sort_by_overall();

        let top = ranking.top_codec().expect("top should be valid");
        assert_eq!(top.codec_id, CodecId::Av1);
        assert_eq!(top.overall_score, 85.0);
    }

    #[test]
    fn test_preset_comparison() {
        let mut comp = PresetComparison::new(CodecId::Av1);
        comp.add_preset("fast");
        comp.add_preset("medium");
        comp.add_preset("slow");

        assert_eq!(comp.presets.len(), 3);
    }

    #[test]
    fn test_bpp_comparison() {
        let bpp = BppComparison {
            bpp_a: 0.5,
            bpp_b: 0.6,
            ratio: 0.833,
        };

        assert_eq!(bpp.bpp_a, 0.5);
        assert_eq!(bpp.bpp_b, 0.6);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = CompressionRatio {
            ratio_a: 50.0,
            ratio_b: 40.0,
            relative_efficiency: 1.25,
        };

        assert_eq!(ratio.ratio_a, 50.0);
        assert_eq!(ratio.relative_efficiency, 1.25);
    }
}

// ─── 4-point Vandermonde Bjøntegaard Delta (BD-Rate / BD-PSNR) ───────────────

/// A single rate–distortion sample for the 4-point BD calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BdRdPoint {
    /// Bitrate in kbps (must be positive).
    pub rate: f64,
    /// PSNR in dB (or any distortion metric, higher = better).
    pub psnr: f64,
}

impl BdRdPoint {
    /// Construct a new `BdRdPoint`.
    #[must_use]
    pub fn new(rate: f64, psnr: f64) -> Self {
        Self { rate, psnr }
    }
}

/// Errors produced by the 4-point BD-Rate / BD-PSNR functions.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BdDeltaError {
    /// Both curves must have exactly 4 points.
    #[error("expected exactly 4 RD points, got anchor={anchor} test={test}")]
    WrongPointCount {
        /// Number of points in the anchor curve.
        anchor: usize,
        /// Number of points in the test curve.
        test: usize,
    },
    /// The Vandermonde system is rank-deficient (collinear points).
    #[error("Vandermonde system is ill-conditioned: collinear RD points")]
    IllConditioned,
    /// The PSNR ranges of the two curves do not overlap.
    #[error("PSNR ranges of the two curves do not overlap")]
    NoOverlap,
    /// A rate value was non-positive (logarithm undefined).
    #[error("non-positive rate value {rate}: log is undefined")]
    NonPositiveRate {
        /// The invalid rate value.
        rate: f64,
    },
}

// ── Gaussian elimination (in-place, 4×5 augmented matrix) ─────────────────────

/// Solve a 4×4 linear system via Gaussian elimination with partial pivoting.
///
/// `mat` is a 4×5 augmented matrix [A | b] stored row-major.
/// Returns the solution vector, or `BdDeltaError::IllConditioned` if the system
/// is rank-deficient (pivot magnitude < 1e-10).
fn solve_4x4(mat: &mut [[f64; 5]; 4]) -> Result<[f64; 4], BdDeltaError> {
    const EPS: f64 = 1e-10;

    for col in 0..4 {
        // Partial pivoting: find the row with the largest absolute value in this column.
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..4 {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < EPS {
            return Err(BdDeltaError::IllConditioned);
        }

        mat.swap(col, max_row);

        let pivot = mat[col][col];
        for row in (col + 1)..4 {
            let factor = mat[row][col] / pivot;
            for k in col..5 {
                let val = mat[col][k];
                mat[row][k] -= factor * val;
            }
        }
    }

    // Back-substitution
    let mut x = [0.0_f64; 4];
    for i in (0..4).rev() {
        x[i] = mat[i][4];
        for j in (i + 1)..4 {
            let xj = x[j];
            x[i] -= mat[i][j] * xj;
        }
        x[i] /= mat[i][i];
    }

    Ok(x)
}

// ── Cubic polynomial fitting ───────────────────────────────────────────────────

/// Fit a cubic polynomial  p(x) = c0 + c1·x + c2·x² + c3·x³  through exactly
/// 4 (x, y) data points using a Vandermonde system solved by Gaussian elimination.
///
/// Returns coefficients `[c0, c1, c2, c3]`.
fn fit_cubic(xs: &[f64; 4], ys: &[f64; 4]) -> Result<[f64; 4], BdDeltaError> {
    // Augmented Vandermonde matrix [1, x, x², x³ | y] for each of 4 rows.
    let mut mat = [[0.0_f64; 5]; 4];
    for i in 0..4 {
        let x = xs[i];
        mat[i][0] = 1.0;
        mat[i][1] = x;
        mat[i][2] = x * x;
        mat[i][3] = x * x * x;
        mat[i][4] = ys[i];
    }
    solve_4x4(&mut mat)
}

/// Evaluate the analytic integral of  c0 + c1·x + c2·x² + c3·x³  over [lo, hi].
fn integrate_cubic(coeffs: &[f64; 4], lo: f64, hi: f64) -> f64 {
    // ∫(lo→hi) (c0 + c1·x + c2·x² + c3·x³) dx
    //  = [c0·x + c1/2·x² + c2/3·x³ + c3/4·x⁴](lo→hi)
    let antideriv = |t: f64| {
        coeffs[0] * t
            + coeffs[1] / 2.0 * t * t
            + coeffs[2] / 3.0 * t * t * t
            + coeffs[3] / 4.0 * t * t * t * t
    };
    antideriv(hi) - antideriv(lo)
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Compute the Bjøntegaard Delta Rate (BD-Rate) between an anchor and a test curve.
///
/// BD-Rate is the average bitrate savings (negative = test is better) of the test
/// codec relative to the anchor, integrated over the PSNR range common to both curves.
/// A result of −10 % means the test codec requires 10 % less bitrate at the same quality.
///
/// Both slices must contain **exactly 4 points** sorted by increasing bitrate.
/// Points must have positive rate values (log is taken internally).
///
/// Returns `Ok(percent_savings)` where savings < 0 is better.
pub fn bd_rate(anchor: &[BdRdPoint; 4], test: &[BdRdPoint; 4]) -> Result<f64, BdDeltaError> {
    // Validate positive rates
    for p in anchor.iter().chain(test.iter()) {
        if p.rate <= 0.0 {
            return Err(BdDeltaError::NonPositiveRate { rate: p.rate });
        }
    }

    // Work in log-rate space
    let log_anchor_rates: [f64; 4] = [
        anchor[0].rate.ln(),
        anchor[1].rate.ln(),
        anchor[2].rate.ln(),
        anchor[3].rate.ln(),
    ];
    let anchor_psnrs: [f64; 4] = [
        anchor[0].psnr,
        anchor[1].psnr,
        anchor[2].psnr,
        anchor[3].psnr,
    ];

    let log_test_rates: [f64; 4] = [
        test[0].rate.ln(),
        test[1].rate.ln(),
        test[2].rate.ln(),
        test[3].rate.ln(),
    ];
    let test_psnrs: [f64; 4] = [test[0].psnr, test[1].psnr, test[2].psnr, test[3].psnr];

    // Fit cubic log-rate = f(PSNR)  (PSNR on x-axis, log-rate on y-axis)
    let coeffs_anchor = fit_cubic(&anchor_psnrs, &log_anchor_rates)?;
    let coeffs_test = fit_cubic(&test_psnrs, &log_test_rates)?;

    // Overlapping PSNR range
    let psnr_lo = anchor_psnrs[0]
        .max(anchor_psnrs[3].min(anchor_psnrs[0])) // min of anchor
        .max(test_psnrs[0].min(test_psnrs[3])); // min of test
    let psnr_hi = anchor_psnrs[0]
        .min(anchor_psnrs[3].max(anchor_psnrs[0])) // max of anchor
        .min(test_psnrs[0].max(test_psnrs[3])); // max of test

    // Simpler: real min/max across all four
    let anchor_psnr_min = anchor_psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let anchor_psnr_max = anchor_psnrs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let test_psnr_min = test_psnrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let test_psnr_max = test_psnrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let overlap_lo = anchor_psnr_min.max(test_psnr_min);
    let overlap_hi = anchor_psnr_max.min(test_psnr_max);

    // Suppress the earlier incorrect calculation and use the correct overlap
    let _ = (psnr_lo, psnr_hi);

    if overlap_hi <= overlap_lo {
        return Err(BdDeltaError::NoOverlap);
    }

    // Average of each cubic in the overlapping range
    let range = overlap_hi - overlap_lo;
    let avg_anchor = integrate_cubic(&coeffs_anchor, overlap_lo, overlap_hi) / range;
    let avg_test = integrate_cubic(&coeffs_test, overlap_lo, overlap_hi) / range;

    // BD-Rate = (exp(avg_test − avg_anchor) − 1) × 100 %
    let bd = (avg_test - avg_anchor).exp() - 1.0;
    Ok(bd * 100.0)
}

/// Compute the Bjøntegaard Delta PSNR (BD-PSNR) between an anchor and a test curve.
///
/// BD-PSNR is the average PSNR gain (positive = test is better) of the test codec
/// relative to the anchor, integrated over the log-bitrate range common to both curves.
///
/// Both slices must contain **exactly 4 points** sorted by increasing bitrate.
pub fn bd_psnr(anchor: &[BdRdPoint; 4], test: &[BdRdPoint; 4]) -> Result<f64, BdDeltaError> {
    for p in anchor.iter().chain(test.iter()) {
        if p.rate <= 0.0 {
            return Err(BdDeltaError::NonPositiveRate { rate: p.rate });
        }
    }

    // Work in log-rate on x-axis, PSNR on y-axis (opposite of bd_rate)
    let log_anchor_rates: [f64; 4] = [
        anchor[0].rate.ln(),
        anchor[1].rate.ln(),
        anchor[2].rate.ln(),
        anchor[3].rate.ln(),
    ];
    let anchor_psnrs: [f64; 4] = [
        anchor[0].psnr,
        anchor[1].psnr,
        anchor[2].psnr,
        anchor[3].psnr,
    ];

    let log_test_rates: [f64; 4] = [
        test[0].rate.ln(),
        test[1].rate.ln(),
        test[2].rate.ln(),
        test[3].rate.ln(),
    ];
    let test_psnrs: [f64; 4] = [test[0].psnr, test[1].psnr, test[2].psnr, test[3].psnr];

    // Fit cubic PSNR = f(log-rate)
    let coeffs_anchor = fit_cubic(&log_anchor_rates, &anchor_psnrs)?;
    let coeffs_test = fit_cubic(&log_test_rates, &test_psnrs)?;

    // Overlapping log-rate range
    let anchor_min = log_anchor_rates
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let anchor_max = log_anchor_rates
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let test_min = log_test_rates.iter().cloned().fold(f64::INFINITY, f64::min);
    let test_max = log_test_rates
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let overlap_lo = anchor_min.max(test_min);
    let overlap_hi = anchor_max.min(test_max);

    if overlap_hi <= overlap_lo {
        return Err(BdDeltaError::NoOverlap);
    }

    let range = overlap_hi - overlap_lo;
    let avg_anchor = integrate_cubic(&coeffs_anchor, overlap_lo, overlap_hi) / range;
    let avg_test = integrate_cubic(&coeffs_test, overlap_lo, overlap_hi) / range;

    Ok(avg_test - avg_anchor)
}
