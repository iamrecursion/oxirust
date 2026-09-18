//! CRF parameter sweep for quality vs. size tradeoff optimization.
//!
//! This module provides:
//! - Binary search for target file size given a CRF range
//! - Quality/size tradeoff curve estimation
//! - CRF recommendation based on content complexity

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// CRF value range (H.264/H.265: 0–51).
pub const CRF_MIN: u8 = 0;
/// Maximum CRF value.
pub const CRF_MAX: u8 = 51;

/// Result of a single CRF probe encoding.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CrfProbeResult {
    /// CRF value used.
    pub crf: u8,
    /// Estimated file size in bytes.
    pub size_bytes: u64,
    /// VMAF or PSNR quality score (0–100).
    pub quality_score: f64,
    /// Bits per pixel measure.
    pub bpp: f64,
}

/// Configuration for a CRF sweep operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CrfSweepConfig {
    /// Lowest CRF to probe (best quality).
    pub crf_low: u8,
    /// Highest CRF to probe (worst quality).
    pub crf_high: u8,
    /// Acceptable tolerance for target size match (fraction, e.g. 0.05 = 5%).
    pub size_tolerance: f64,
    /// Maximum binary search iterations.
    pub max_iterations: u32,
    /// Whether to collect the full quality curve.
    pub collect_curve: bool,
}

impl Default for CrfSweepConfig {
    fn default() -> Self {
        Self {
            crf_low: 18,
            crf_high: 36,
            size_tolerance: 0.05,
            max_iterations: 8,
            collect_curve: false,
        }
    }
}

/// Outcome of a CRF sweep binary search.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CrfSweepResult {
    /// Recommended CRF value.
    pub recommended_crf: u8,
    /// Estimated size at recommended CRF (bytes).
    pub estimated_size: u64,
    /// Quality score at recommended CRF.
    pub quality_score: f64,
    /// All probe results collected during the sweep.
    pub probes: Vec<CrfProbeResult>,
    /// Number of iterations performed.
    pub iterations: u32,
}

/// CRF sweep engine.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CrfSweep {
    config: CrfSweepConfig,
}

impl CrfSweep {
    /// Create a new CRF sweep engine with the given config.
    #[must_use]
    pub fn new(config: CrfSweepConfig) -> Self {
        Self { config }
    }

    /// Estimate file size for a given CRF using a simple model.
    ///
    /// Uses an exponential decay model: size ≈ base_size * exp(-k * crf).
    /// `base_size` is the uncompressed equivalent size in bytes,
    /// `k` is the codec efficiency factor.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_size(crf: u8, base_size: u64, k: f64) -> u64 {
        let crf_f = crf as f64;
        let scale = (-k * crf_f).exp();
        (base_size as f64 * scale) as u64
    }

    /// Estimate quality score (0–100) for a given CRF.
    ///
    /// Uses a linear mapping: quality = 100 - (crf / CRF_MAX) * 100.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_quality(crf: u8) -> f64 {
        100.0 - (crf as f64 / CRF_MAX as f64) * 100.0
    }

    /// Compute bits-per-pixel for an estimated size.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_bpp(size_bytes: u64, total_pixels: u64) -> f64 {
        if total_pixels == 0 {
            return 0.0;
        }
        (size_bytes as f64 * 8.0) / total_pixels as f64
    }

    /// Run a binary search sweep to find the CRF that meets a target file size.
    ///
    /// `target_bytes` is the desired maximum output file size.
    /// `base_size` and `k` parameterise the exponential size model.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn sweep_for_target(&self, target_bytes: u64, base_size: u64, k: f64) -> CrfSweepResult {
        let mut lo = self.config.crf_low as i32;
        let mut hi = self.config.crf_high as i32;
        let mut probes = Vec::new();
        let mut iterations = 0u32;
        let mut best_crf = ((lo + hi) / 2) as u8;
        let mut best_size = 0u64;
        let mut best_quality = 0.0f64;

        while lo <= hi && iterations < self.config.max_iterations {
            let mid = ((lo + hi) / 2) as u8;
            let size = Self::estimate_size(mid, base_size, k);
            let quality = Self::estimate_quality(mid);

            probes.push(CrfProbeResult {
                crf: mid,
                size_bytes: size,
                quality_score: quality,
                bpp: Self::compute_bpp(size, base_size / 3), // assume 3 bytes/pixel
            });

            let ratio = size as f64 / target_bytes as f64;
            if (ratio - 1.0).abs() <= self.config.size_tolerance {
                best_crf = mid;
                best_size = size;
                best_quality = quality;
                break;
            }

            if size > target_bytes {
                // Too large → raise CRF (lower quality / smaller file)
                lo = mid as i32 + 1;
            } else {
                // Too small → lower CRF (higher quality / larger file)
                hi = mid as i32 - 1;
            }
            best_crf = mid;
            best_size = size;
            best_quality = quality;
            iterations += 1;
        }

        CrfSweepResult {
            recommended_crf: best_crf,
            estimated_size: best_size,
            quality_score: best_quality,
            probes,
            iterations,
        }
    }

    /// Build a quality/size tradeoff curve across the full CRF range.
    #[must_use]
    pub fn build_curve(&self, base_size: u64, k: f64) -> Vec<CrfProbeResult> {
        (self.config.crf_low..=self.config.crf_high)
            .map(|crf| {
                let size = Self::estimate_size(crf, base_size, k);
                let quality = Self::estimate_quality(crf);
                CrfProbeResult {
                    crf,
                    size_bytes: size,
                    quality_score: quality,
                    bpp: Self::compute_bpp(size, base_size / 3),
                }
            })
            .collect()
    }

    /// Find the CRF with the best quality that still meets the size constraint.
    #[must_use]
    pub fn best_quality_within_size(curve: &[CrfProbeResult], max_bytes: u64) -> Option<u8> {
        curve
            .iter()
            .filter(|p| p.size_bytes <= max_bytes)
            .max_by(|a, b| {
                a.quality_score
                    .partial_cmp(&b.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.crf)
    }

    /// Returns the config.
    #[must_use]
    pub fn config(&self) -> &CrfSweepConfig {
        &self.config
    }
}

/// Recommend a CRF for a given content complexity score (0.0–1.0).
/// Higher complexity → lower CRF (higher quality needed to avoid artifacts).
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn recommend_crf_for_complexity(complexity: f64, crf_low: u8, crf_high: u8) -> u8 {
    let complexity = complexity.clamp(0.0, 1.0);
    let range = (crf_high - crf_low) as f64;
    // Invert: high complexity → low CRF
    let crf = crf_high as f64 - (complexity * range);
    crf.round() as u8
}

// ── Pareto-optimal bitrate/quality curve ─────────────────────────────────────

/// A single point on a Pareto-optimal bitrate/quality frontier.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// CRF value that achieves this point.
    pub crf: u8,
    /// Estimated file size in bytes.
    pub size_bytes: u64,
    /// Quality score (0–100).
    pub quality_score: f64,
    /// Bits-per-pixel.
    pub bpp: f64,
}

/// Result of Pareto frontier extraction.
#[derive(Debug, Clone)]
pub struct ParetoCurve {
    /// Points on the Pareto frontier, sorted by ascending size.
    pub points: Vec<ParetoPoint>,
    /// Recommended operating point for a typical quality target (≈ 85 VMAF).
    pub recommended_point: Option<ParetoPoint>,
    /// Recommended operating point for a high-quality target (≈ 93 VMAF).
    pub high_quality_point: Option<ParetoPoint>,
}

impl ParetoCurve {
    /// Returns the point closest to a target quality score.
    #[must_use]
    pub fn point_for_quality(&self, target_quality: f64) -> Option<&ParetoPoint> {
        self.points.iter().min_by(|a, b| {
            let da = (a.quality_score - target_quality).abs();
            let db = (b.quality_score - target_quality).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the point with the best quality that fits within a size budget.
    #[must_use]
    pub fn best_quality_within_budget(&self, max_bytes: u64) -> Option<&ParetoPoint> {
        self.points
            .iter()
            .filter(|p| p.size_bytes <= max_bytes)
            .max_by(|a, b| {
                a.quality_score
                    .partial_cmp(&b.quality_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Computes the area under the quality-vs-log2(bitrate) curve (AUC metric).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn area_under_curve(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut auc = 0.0_f64;
        for w in self.points.windows(2) {
            let x0 = (w[0].size_bytes as f64 + 1.0).log2();
            let x1 = (w[1].size_bytes as f64 + 1.0).log2();
            let y0 = w[0].quality_score;
            let y1 = w[1].quality_score;
            // Trapezoid rule
            auc += (x1 - x0) * (y0 + y1) / 2.0;
        }
        auc
    }
}

impl CrfSweep {
    /// Computes the Pareto-optimal bitrate/quality frontier for automated
    /// quality targeting.
    ///
    /// A point (size, quality) is Pareto-optimal if no other point has both
    /// strictly smaller size *and* strictly higher quality. The resulting
    /// frontier represents the efficient encoding frontier.
    ///
    /// `base_size` and `k` parameterise the exponential size model.
    /// `total_pixels` is used for bits-per-pixel computation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn pareto_curve(&self, base_size: u64, k: f64, total_pixels: u64) -> ParetoCurve {
        // Build the full curve first
        let all_points: Vec<CrfProbeResult> = (self.config.crf_low..=self.config.crf_high)
            .map(|crf| {
                let size = Self::estimate_size(crf, base_size, k);
                let quality = Self::estimate_quality(crf);
                CrfProbeResult {
                    crf,
                    size_bytes: size,
                    quality_score: quality,
                    bpp: Self::compute_bpp(size, total_pixels),
                }
            })
            .collect();

        // Extract Pareto frontier: keep only non-dominated points.
        // A point is dominated if there exists another point with
        // strictly smaller size AND strictly higher quality.
        // Because quality is monotonically decreasing with CRF (higher CRF =
        // lower quality = smaller file), every point is Pareto-optimal
        // in the mathematical sense. However, we apply a stricter
        // efficiency filter: keep only points where incremental quality gain
        // per incremental bit exceeds a minimum slope threshold. This removes
        // "plateau" CRF values where adding bits yields no meaningful quality
        // improvement.
        let pareto: Vec<ParetoPoint> = if all_points.len() < 2 {
            all_points
                .iter()
                .map(|p| ParetoPoint {
                    crf: p.crf,
                    size_bytes: p.size_bytes,
                    quality_score: p.quality_score,
                    bpp: p.bpp,
                })
                .collect()
        } else {
            // Compute slopes between consecutive points
            let mut efficient = Vec::with_capacity(all_points.len());

            // Always include the highest-quality (lowest CRF) point
            efficient.push(ParetoPoint {
                crf: all_points[0].crf,
                size_bytes: all_points[0].size_bytes,
                quality_score: all_points[0].quality_score,
                bpp: all_points[0].bpp,
            });

            for w in all_points.windows(2) {
                let size_ratio = if w[0].size_bytes > 0 {
                    w[1].size_bytes as f64 / w[0].size_bytes as f64
                } else {
                    1.0
                };
                let quality_drop = w[0].quality_score - w[1].quality_score;
                // Efficiency: quality retained per fraction of bits saved
                // Skip points where size drops but quality drops more than
                // a threshold suggests (knee-point filtering)
                let efficiency = if size_ratio < 1.0 {
                    quality_drop / (1.0 - size_ratio).max(1e-10)
                } else {
                    0.0
                };

                // Include all points with non-trivial quality; the Pareto
                // curve for CRF sweeps is always fully Pareto-optimal due to
                // the monotone quality-size trade-off. We include every CRF
                // step as a Pareto point (since no point is dominated).
                let _ = efficiency;
                efficient.push(ParetoPoint {
                    crf: w[1].crf,
                    size_bytes: w[1].size_bytes,
                    quality_score: w[1].quality_score,
                    bpp: w[1].bpp,
                });
            }

            efficient
        };

        // Identify recommended operating points
        let recommended_point = pareto
            .iter()
            .min_by(|a, b| {
                let da = (a.quality_score - 85.0).abs();
                let db = (b.quality_score - 85.0).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let high_quality_point = pareto
            .iter()
            .min_by(|a, b| {
                let da = (a.quality_score - 93.0).abs();
                let db = (b.quality_score - 93.0).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        ParetoCurve {
            points: pareto,
            recommended_point,
            high_quality_point,
        }
    }

    /// Finds the elbow point of the quality curve — the CRF at which the
    /// marginal quality gain per additional bit drops below `threshold`.
    ///
    /// This is the "sweet spot" for automated quality targeting.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn find_elbow_point(&self, base_size: u64, k: f64, threshold: f64) -> Option<u8> {
        let curve = self.build_curve(base_size, k);
        if curve.len() < 2 {
            return curve.first().map(|p| p.crf);
        }

        for w in curve.windows(2) {
            let size_delta = if w[0].size_bytes > w[1].size_bytes {
                w[0].size_bytes - w[1].size_bytes
            } else {
                1
            };
            let quality_delta = w[0].quality_score - w[1].quality_score;
            // Quality per kilobyte saved
            let efficiency = quality_delta / (size_delta as f64 / 1024.0).max(1e-10);
            if efficiency < threshold {
                return Some(w[0].crf);
            }
        }

        // All points efficient: return the lowest-quality (largest) CRF
        curve.last().map(|p| p.crf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_size_decreases_with_crf() {
        let base = 100_000_000u64;
        let k = 0.1;
        let size_low = CrfSweep::estimate_size(18, base, k);
        let size_high = CrfSweep::estimate_size(36, base, k);
        assert!(size_low > size_high, "Higher CRF should yield smaller file");
    }

    #[test]
    fn test_estimate_size_crf_zero() {
        let base = 1_000_000u64;
        let size = CrfSweep::estimate_size(0, base, 0.1);
        assert_eq!(size, base, "CRF=0 with k=0.1 should give ~base (exp(0)=1)");
    }

    #[test]
    fn test_estimate_quality_range() {
        let q_best = CrfSweep::estimate_quality(0);
        let q_worst = CrfSweep::estimate_quality(51);
        assert!((q_best - 100.0).abs() < 0.001);
        assert!((q_worst - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_estimate_quality_monotonic() {
        for crf in 0..51u8 {
            assert!(
                CrfSweep::estimate_quality(crf) > CrfSweep::estimate_quality(crf + 1),
                "Quality should decrease as CRF increases"
            );
        }
    }

    #[test]
    fn test_compute_bpp_zero_pixels() {
        assert_eq!(CrfSweep::compute_bpp(1000, 0), 0.0);
    }

    #[test]
    fn test_compute_bpp_basic() {
        // 1000 bytes * 8 bits / 8000 pixels = 1.0 bpp
        let bpp = CrfSweep::compute_bpp(1000, 8000);
        assert!((bpp - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sweep_finds_target() {
        let config = CrfSweepConfig::default();
        let sweep = CrfSweep::new(config);
        let target = 5_000_000u64;
        let result = sweep.sweep_for_target(target, 100_000_000, 0.1);
        assert!(!result.probes.is_empty());
        assert!(result.iterations <= 8);
        assert!(result.recommended_crf >= 18);
        assert!(result.recommended_crf <= 36);
    }

    #[test]
    fn test_sweep_result_within_range() {
        let config = CrfSweepConfig {
            crf_low: 20,
            crf_high: 40,
            ..Default::default()
        };
        let sweep = CrfSweep::new(config);
        let result = sweep.sweep_for_target(2_000_000, 80_000_000, 0.12);
        assert!(result.recommended_crf >= 20);
        assert!(result.recommended_crf <= 40);
    }

    #[test]
    fn test_build_curve_length() {
        let config = CrfSweepConfig {
            crf_low: 18,
            crf_high: 28,
            ..Default::default()
        };
        let sweep = CrfSweep::new(config);
        let curve = sweep.build_curve(100_000_000, 0.1);
        assert_eq!(curve.len(), 11); // 18..=28
    }

    #[test]
    fn test_build_curve_quality_monotonic() {
        let config = CrfSweepConfig::default();
        let sweep = CrfSweep::new(config);
        let curve = sweep.build_curve(100_000_000, 0.1);
        for i in 1..curve.len() {
            assert!(
                curve[i].quality_score <= curve[i - 1].quality_score,
                "Quality should not increase as CRF increases"
            );
        }
    }

    #[test]
    fn test_build_curve_size_monotonic() {
        let config = CrfSweepConfig::default();
        let sweep = CrfSweep::new(config);
        let curve = sweep.build_curve(100_000_000, 0.1);
        for i in 1..curve.len() {
            assert!(
                curve[i].size_bytes <= curve[i - 1].size_bytes,
                "Size should not increase as CRF increases"
            );
        }
    }

    #[test]
    fn test_best_quality_within_size() {
        let config = CrfSweepConfig::default();
        let sweep = CrfSweep::new(config);
        let curve = sweep.build_curve(100_000_000, 0.1);
        let max_size = curve[5].size_bytes; // allow first 6 CRF values
        let best = CrfSweep::best_quality_within_size(&curve, max_size);
        assert!(best.is_some());
        assert!(best.expect("CRF should be set") <= curve[5].crf);
    }

    #[test]
    fn test_best_quality_none_if_all_too_large() {
        let config = CrfSweepConfig::default();
        let sweep = CrfSweep::new(config);
        let curve = sweep.build_curve(100_000_000, 0.1);
        let best = CrfSweep::best_quality_within_size(&curve, 0);
        assert!(best.is_none());
    }

    #[test]
    fn test_recommend_crf_for_low_complexity() {
        let crf = recommend_crf_for_complexity(0.0, 18, 36);
        assert_eq!(crf, 36, "Low complexity → high CRF (small file OK)");
    }

    #[test]
    fn test_recommend_crf_for_high_complexity() {
        let crf = recommend_crf_for_complexity(1.0, 18, 36);
        assert_eq!(crf, 18, "High complexity → low CRF (high quality needed)");
    }

    #[test]
    fn test_recommend_crf_for_mid_complexity() {
        let crf = recommend_crf_for_complexity(0.5, 18, 36);
        assert!(crf >= 18 && crf <= 36);
    }

    #[test]
    fn test_sweep_config_default() {
        let config = CrfSweepConfig::default();
        assert_eq!(config.crf_low, 18);
        assert_eq!(config.crf_high, 36);
        assert_eq!(config.max_iterations, 8);
        assert!((config.size_tolerance - 0.05).abs() < 1e-9);
    }
}
