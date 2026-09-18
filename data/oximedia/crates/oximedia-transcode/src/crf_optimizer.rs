//! CRF/quality optimizer for finding optimal encoding parameters.
//!
//! This module provides binary search over CRF space to find the
//! optimal encoding quality that meets a given target within bitrate constraints.

/// Quality target specification.
#[derive(Debug, Clone)]
pub struct QualityTarget {
    /// Minimum acceptable PSNR in decibels.
    pub min_psnr_db: f32,
    /// Minimum acceptable SSIM (0.0–1.0).
    pub min_ssim: f32,
    /// Maximum allowed bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
}

impl QualityTarget {
    /// Creates a new quality target.
    #[must_use]
    pub fn new(min_psnr_db: f32, min_ssim: f32, max_bitrate_kbps: u32) -> Self {
        Self {
            min_psnr_db,
            min_ssim,
            max_bitrate_kbps,
        }
    }
}

/// CRF range definition for a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrfRange {
    /// Minimum CRF value (best quality).
    pub min_crf: u8,
    /// Maximum CRF value (worst quality).
    pub max_crf: u8,
}

impl CrfRange {
    /// Creates a new CRF range.
    #[must_use]
    pub fn new(min_crf: u8, max_crf: u8) -> Self {
        Self { min_crf, max_crf }
    }

    /// Returns the CRF range for H.264 (17–51).
    #[must_use]
    pub fn h264_range() -> Self {
        Self {
            min_crf: 17,
            max_crf: 51,
        }
    }

    /// Returns the CRF range for AV1 (0–63).
    #[must_use]
    pub fn av1_range() -> Self {
        Self {
            min_crf: 0,
            max_crf: 63,
        }
    }

    /// Returns the midpoint CRF value.
    #[must_use]
    pub fn midpoint(&self) -> u8 {
        self.min_crf + (self.max_crf - self.min_crf) / 2
    }

    /// Returns the number of CRF values in this range.
    #[must_use]
    pub fn span(&self) -> u8 {
        self.max_crf - self.min_crf
    }
}

/// Result of CRF optimization.
#[derive(Debug, Clone)]
pub struct CrfOptimizerResult {
    /// The optimal CRF value found.
    pub optimal_crf: u8,
    /// Estimated bitrate at the optimal CRF.
    pub estimated_bitrate_kbps: u32,
    /// Estimated PSNR at the optimal CRF.
    pub estimated_psnr: f32,
}

/// CRF optimizer using binary search over the CRF space.
#[derive(Debug, Clone, Default)]
pub struct CrfOptimizer;

impl CrfOptimizer {
    /// Creates a new `CrfOptimizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Finds the optimal CRF for the given quality target using binary search.
    ///
    /// The bitrate model is: `bitrate = complexity * base * 2^((28 - crf) / 6)`.
    /// Higher CRF → lower quality → lower bitrate.
    /// Searches for the lowest CRF (best quality) that keeps bitrate ≤ `max_bitrate_kbps`.
    #[must_use]
    pub fn find_optimal(
        target: &QualityTarget,
        crf_range: CrfRange,
        content_complexity: f32,
    ) -> CrfOptimizerResult {
        // Binary search: we want the highest CRF (lowest bitrate) that
        // still produces a bitrate ≤ max_bitrate_kbps.
        // However, we also want the best quality that meets bitrate.
        // Strategy: find lowest CRF whose estimated bitrate ≤ max_bitrate_kbps.
        let mut lo = crf_range.min_crf;
        let mut hi = crf_range.max_crf;
        let mut best_crf = crf_range.max_crf;

        // We want highest CRF (worst quality) that stays within budget
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            let bitrate = BitrateModel::predict(mid, content_complexity, "h264");
            if bitrate <= target.max_bitrate_kbps {
                best_crf = mid;
                // Try going lower CRF (higher quality) if budget allows
                if mid == 0 {
                    break;
                }
                hi = mid.saturating_sub(1);
            } else {
                lo = mid.saturating_add(1);
                if lo > hi {
                    break;
                }
            }
        }

        let estimated_bitrate_kbps = BitrateModel::predict(best_crf, content_complexity, "h264");
        let estimated_psnr = Self::estimate_psnr(best_crf, content_complexity);

        CrfOptimizerResult {
            optimal_crf: best_crf,
            estimated_bitrate_kbps,
            estimated_psnr,
        }
    }

    /// Estimates PSNR for a given CRF.
    ///
    /// Simple heuristic: lower CRF → higher PSNR. CRF=0 → ~50dB, CRF=51 → ~30dB.
    #[must_use]
    pub fn estimate_psnr(crf: u8, _complexity: f32) -> f32 {
        50.0 - (f32::from(crf) * 20.0 / 51.0)
    }
}

/// Bitrate model for predicting bitrate from CRF and content complexity.
#[derive(Debug, Clone, Default)]
pub struct BitrateModel;

impl BitrateModel {
    /// Predicts bitrate in kbps for a given CRF, content complexity, and codec.
    ///
    /// Model: `bitrate = complexity * base * 2^((crf - 28) / 6)`
    ///
    /// Codec-specific base bitrates (kbps):
    /// - h264: 2000
    /// - vp9: 1500 (typically more efficient)
    /// - av1: 1200 (most efficient)
    /// - hevc / h265: 1400
    /// - others: 2000
    #[must_use]
    pub fn predict(crf: u8, complexity: f32, codec: &str) -> u32 {
        let base_kbps = match codec {
            "h264" | "libx264" => 2000.0_f32,
            "vp9" | "libvpx-vp9" => 1500.0,
            "av1" | "libaom-av1" | "libsvtav1" => 1200.0,
            "hevc" | "h265" | "libx265" => 1400.0,
            _ => 2000.0,
        };

        // Higher CRF → lower quality → lower bitrate: negate the exponent
        let exponent = (28.0 - f32::from(crf)) / 6.0;
        let scale = 2.0_f32.powf(exponent);
        let bitrate = complexity * base_kbps * scale;

        // Clamp to reasonable range [10, 100_000] kbps
        bitrate.clamp(10.0, 100_000.0).round() as u32
    }

    /// Estimates the CRF needed to achieve a target bitrate.
    #[must_use]
    pub fn crf_for_bitrate(target_kbps: u32, complexity: f32, codec: &str) -> u8 {
        let base_kbps = match codec {
            "h264" | "libx264" => 2000.0_f32,
            "vp9" | "libvpx-vp9" => 1500.0,
            "av1" | "libaom-av1" | "libsvtav1" => 1200.0,
            "hevc" | "h265" | "libx265" => 1400.0,
            _ => 2000.0,
        };

        if complexity <= 0.0 || base_kbps <= 0.0 {
            return 28;
        }

        // Solve: target = complexity * base * 2^((28-crf)/6)
        // → (28-crf)/6 = log2(target / (complexity * base))
        // → crf = 28 - 6 * log2(target / (complexity * base))
        let ratio = target_kbps as f32 / (complexity * base_kbps);
        if ratio <= 0.0 {
            return 51;
        }
        let crf_f = 28.0 - 6.0 * ratio.log2();
        (crf_f.round() as i32).clamp(0, 63) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_target_new() {
        let t = QualityTarget::new(35.0, 0.95, 4000);
        assert_eq!(t.min_psnr_db, 35.0);
        assert_eq!(t.min_ssim, 0.95);
        assert_eq!(t.max_bitrate_kbps, 4000);
    }

    #[test]
    fn test_crf_range_h264() {
        let r = CrfRange::h264_range();
        assert_eq!(r.min_crf, 17);
        assert_eq!(r.max_crf, 51);
    }

    #[test]
    fn test_crf_range_av1() {
        let r = CrfRange::av1_range();
        assert_eq!(r.min_crf, 0);
        assert_eq!(r.max_crf, 63);
    }

    #[test]
    fn test_crf_range_midpoint() {
        let r = CrfRange::new(0, 63);
        assert_eq!(r.midpoint(), 31);
    }

    #[test]
    fn test_crf_range_span() {
        let r = CrfRange::h264_range();
        assert_eq!(r.span(), 34);
    }

    #[test]
    fn test_bitrate_model_predict_h264() {
        // At CRF 28 with complexity 1.0, bitrate should equal base (2000 kbps)
        let b = BitrateModel::predict(28, 1.0, "h264");
        assert_eq!(b, 2000);
    }

    #[test]
    fn test_bitrate_model_predict_higher_crf_lower_bitrate() {
        let b_low = BitrateModel::predict(20, 1.0, "h264");
        let b_high = BitrateModel::predict(35, 1.0, "h264");
        assert!(b_low > b_high, "Lower CRF should produce higher bitrate");
    }

    #[test]
    fn test_bitrate_model_predict_av1_lower_than_h264() {
        let h264 = BitrateModel::predict(28, 1.0, "h264");
        let av1 = BitrateModel::predict(28, 1.0, "av1");
        assert!(av1 < h264, "AV1 should have lower base bitrate");
    }

    #[test]
    fn test_bitrate_model_complexity_scaling() {
        let b1 = BitrateModel::predict(28, 1.0, "h264");
        let b2 = BitrateModel::predict(28, 2.0, "h264");
        assert_eq!(b2, b1 * 2);
    }

    #[test]
    fn test_crf_optimizer_finds_within_budget() {
        let target = QualityTarget::new(30.0, 0.9, 5000);
        let crf_range = CrfRange::h264_range();
        let result = CrfOptimizer::find_optimal(&target, crf_range, 1.0);
        assert!(
            result.estimated_bitrate_kbps <= 5000,
            "Bitrate {} should be <= 5000",
            result.estimated_bitrate_kbps
        );
        assert!(result.optimal_crf >= crf_range.min_crf);
        assert!(result.optimal_crf <= crf_range.max_crf);
    }

    #[test]
    fn test_crf_optimizer_result_fields() {
        let target = QualityTarget::new(30.0, 0.9, 4000);
        let result = CrfOptimizer::find_optimal(&target, CrfRange::h264_range(), 1.0);
        assert!(result.estimated_psnr > 0.0);
        assert!(result.estimated_bitrate_kbps > 0);
    }

    #[test]
    fn test_estimate_psnr_decreases_with_crf() {
        let psnr_low = CrfOptimizer::estimate_psnr(17, 1.0);
        let psnr_high = CrfOptimizer::estimate_psnr(51, 1.0);
        assert!(psnr_low > psnr_high);
    }
}
