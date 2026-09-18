//! A/B quality comparison: rank multiple encodes by perceptual quality.
//!
//! Given a reference sequence and two or more distorted sequences, the
//! [`AbComparator`] computes a set of quality metrics for every candidate
//! and produces a ranked report so that callers can select the best encode.
//!
//! # Design goals
//!
//! * Pure-Rust, no external processes required.
//! * Parallel per-candidate assessment via `rayon`.
//! * Composite scoring with configurable metric weights.
//! * Rich ranking report including per-metric detail and confidence intervals.

use crate::{
    confidence::{ConfidenceCalculator, ConfidenceInterval, ConfidenceLevel},
    Frame, MetricType, QualityAssessor,
};
use oximedia_core::OxiResult;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─── Metric weight configuration ──────────────────────────────────────────────

/// Relative weights assigned to individual metrics when computing the
/// composite perceptual score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    /// Weight for SSIM (0.0–1.0).
    pub ssim: f64,
    /// Weight for PSNR (0.0–1.0).
    pub psnr: f64,
    /// Weight for VMAF (0.0–1.0).
    pub vmaf: f64,
    /// Weight for MS-SSIM (0.0–1.0).
    pub ms_ssim: f64,
}

impl MetricWeights {
    /// Weights tuned for perceptual quality: VMAF > SSIM > MS-SSIM > PSNR.
    #[must_use]
    pub fn perceptual() -> Self {
        Self {
            ssim: 0.20,
            psnr: 0.10,
            vmaf: 0.50,
            ms_ssim: 0.20,
        }
    }

    /// Uniform weights (all metrics equal).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            ssim: 0.25,
            psnr: 0.25,
            vmaf: 0.25,
            ms_ssim: 0.25,
        }
    }

    /// PSNR-centric weights (matches legacy broadcast workflows).
    #[must_use]
    pub fn psnr_centric() -> Self {
        Self {
            ssim: 0.15,
            psnr: 0.60,
            vmaf: 0.15,
            ms_ssim: 0.10,
        }
    }

    /// Returns a normalised copy so that all weights sum to 1.0.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let total = self.ssim + self.psnr + self.vmaf + self.ms_ssim;
        if total <= 0.0 {
            return Self::uniform();
        }
        Self {
            ssim: self.ssim / total,
            psnr: self.psnr / total,
            vmaf: self.vmaf / total,
            ms_ssim: self.ms_ssim / total,
        }
    }
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self::perceptual()
    }
}

// ─── Per-candidate results ────────────────────────────────────────────────────

/// Raw aggregate metric values for one candidate encode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMetrics {
    /// Label for this encode (e.g. `"crf18"`, `"crf28"`, `"bitrate_2m"`).
    pub label: String,
    /// Mean PSNR across all frames (dB).
    pub psnr_mean: f64,
    /// Mean SSIM across all frames \[0, 1\].
    pub ssim_mean: f64,
    /// Mean VMAF score across all frames \[0, 100\].
    pub vmaf_mean: f64,
    /// Mean MS-SSIM across all frames \[0, 1\].
    pub ms_ssim_mean: f64,
    /// 95% confidence interval for SSIM (if sufficient frames).
    pub ssim_ci: Option<ConfidenceInterval>,
    /// 95% confidence interval for PSNR (if sufficient frames).
    pub psnr_ci: Option<ConfidenceInterval>,
    /// Weighted composite perceptual quality score in \[0, 100\].
    pub composite_score: f64,
}

/// Ranked comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbComparisonReport {
    /// Candidates sorted from best to worst by `composite_score`.
    pub ranked: Vec<CandidateMetrics>,
    /// Metric weights used to compute composite scores.
    pub weights: MetricWeights,
    /// Total number of reference frames assessed.
    pub frame_count: usize,
}

impl AbComparisonReport {
    /// Returns the best-ranked candidate, or `None` if the report is empty.
    #[must_use]
    pub fn winner(&self) -> Option<&CandidateMetrics> {
        self.ranked.first()
    }

    /// Returns the worst-ranked candidate.
    #[must_use]
    pub fn loser(&self) -> Option<&CandidateMetrics> {
        self.ranked.last()
    }

    /// Returns a human-readable ranking summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "A/B Quality Report — {} candidates, {} frames",
            self.ranked.len(),
            self.frame_count
        ));
        lines.push(format!(
            "Weights: SSIM={:.2} PSNR={:.2} VMAF={:.2} MS-SSIM={:.2}",
            self.weights.ssim, self.weights.psnr, self.weights.vmaf, self.weights.ms_ssim
        ));
        lines.push(String::new());
        for (rank, c) in self.ranked.iter().enumerate() {
            lines.push(format!(
                "#{} {}: composite={:.2}  VMAF={:.2}  SSIM={:.4}  PSNR={:.2}  MS-SSIM={:.4}",
                rank + 1,
                c.label,
                c.composite_score,
                c.vmaf_mean,
                c.ssim_mean,
                c.psnr_mean,
                c.ms_ssim_mean
            ));
        }
        lines.join("\n")
    }
}

// ─── Comparator ───────────────────────────────────────────────────────────────

/// A/B quality comparator.
pub struct AbComparator {
    weights: MetricWeights,
    assessor: QualityAssessor,
    ci_calculator: ConfidenceCalculator,
}

impl AbComparator {
    /// Creates a new comparator with default perceptual weights.
    #[must_use]
    pub fn new() -> Self {
        Self::with_weights(MetricWeights::perceptual())
    }

    /// Creates a comparator with the specified metric weights.
    #[must_use]
    pub fn with_weights(weights: MetricWeights) -> Self {
        Self {
            weights: weights.normalized(),
            assessor: QualityAssessor::new(),
            ci_calculator: ConfidenceCalculator::new(ConfidenceLevel::Ninety5),
        }
    }

    /// Compares multiple candidate encodes against the same reference sequence.
    ///
    /// Each element of `candidates` is `(label, frames)`.  All candidate frame
    /// sequences must have the same length as `reference_frames`.
    ///
    /// # Errors
    ///
    /// Returns an error if any candidate has a different frame count from the
    /// reference, or if metric calculation fails.
    pub fn compare<'a>(
        &self,
        reference_frames: &[Frame],
        candidates: &[(&'a str, Vec<Frame>)],
    ) -> OxiResult<AbComparisonReport> {
        if candidates.is_empty() {
            return Ok(AbComparisonReport {
                ranked: Vec::new(),
                weights: self.weights.clone(),
                frame_count: reference_frames.len(),
            });
        }

        // Validate all candidates have matching frame counts
        for (label, frames) in candidates {
            if frames.len() != reference_frames.len() {
                return Err(oximedia_core::OxiError::InvalidData(format!(
                    "Candidate '{}' has {} frames but reference has {}",
                    label,
                    frames.len(),
                    reference_frames.len()
                )));
            }
        }

        // Assess each candidate in parallel
        let candidate_metrics: Vec<OxiResult<CandidateMetrics>> = candidates
            .par_iter()
            .map(|(label, dist_frames)| self.assess_candidate(label, reference_frames, dist_frames))
            .collect();

        let mut ranked: Vec<CandidateMetrics> = candidate_metrics
            .into_iter()
            .collect::<OxiResult<Vec<_>>>()?;

        // Sort descending by composite score
        ranked.sort_by(|a, b| {
            b.composite_score
                .partial_cmp(&a.composite_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(AbComparisonReport {
            ranked,
            weights: self.weights.clone(),
            frame_count: reference_frames.len(),
        })
    }

    fn assess_candidate(
        &self,
        label: &str,
        reference_frames: &[Frame],
        distorted_frames: &[Frame],
    ) -> OxiResult<CandidateMetrics> {
        let mut psnr_values: Vec<f64> = Vec::with_capacity(reference_frames.len());
        let mut ssim_values: Vec<f64> = Vec::with_capacity(reference_frames.len());
        let mut vmaf_values: Vec<f64> = Vec::with_capacity(reference_frames.len());
        let mut ms_ssim_values: Vec<f64> = Vec::with_capacity(reference_frames.len());

        for (ref_frame, dist_frame) in reference_frames.iter().zip(distorted_frames.iter()) {
            let psnr = self
                .assessor
                .assess(ref_frame, dist_frame, MetricType::Psnr)?;
            psnr_values.push(psnr.score);

            let ssim = self
                .assessor
                .assess(ref_frame, dist_frame, MetricType::Ssim)?;
            ssim_values.push(ssim.score);

            let vmaf = self
                .assessor
                .assess(ref_frame, dist_frame, MetricType::Vmaf)?;
            vmaf_values.push(vmaf.score);

            // MS-SSIM requires multi-scale downsampling; fall back to SSIM when
            // the frame is too small to support all scale levels.
            let ms_ssim = self
                .assessor
                .assess(ref_frame, dist_frame, MetricType::MsSsim)
                .unwrap_or_else(|_| {
                    self.assessor
                        .assess(ref_frame, dist_frame, MetricType::Ssim)
                        .unwrap_or_else(|_| crate::QualityScore::new(MetricType::MsSsim, 0.0))
                });
            ms_ssim_values.push(ms_ssim.score);
        }

        let n = psnr_values.len() as f64;
        let psnr_mean = psnr_values.iter().sum::<f64>() / n;
        let ssim_mean = ssim_values.iter().sum::<f64>() / n;
        let vmaf_mean = vmaf_values.iter().sum::<f64>() / n;
        let ms_ssim_mean = ms_ssim_values.iter().sum::<f64>() / n;

        // Compute confidence intervals
        let ssim_ci = self.ci_calculator.compute(&ssim_values);
        let psnr_ci = self.ci_calculator.compute(&psnr_values);

        // Composite score: normalise PSNR to [0,1] range (assume max 60 dB) for weighting
        let w = &self.weights;
        let psnr_norm = (psnr_mean / 60.0).clamp(0.0, 1.0);
        let composite_score = (w.ssim * ssim_mean
            + w.psnr * psnr_norm
            + w.vmaf * (vmaf_mean / 100.0)
            + w.ms_ssim * ms_ssim_mean)
            * 100.0;

        Ok(CandidateMetrics {
            label: label.to_string(),
            psnr_mean,
            ssim_mean,
            vmaf_mean,
            ms_ssim_mean,
            ssim_ci,
            psnr_ci,
            composite_score,
        })
    }
}

impl Default for AbComparator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_frame(width: usize, height: usize, y: u8) -> Frame {
        let mut f =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("should succeed in test");
        f.planes[0].fill(y);
        f.planes[1].fill(128);
        f.planes[2].fill(128);
        f
    }

    fn make_frames(count: usize, width: usize, height: usize, y: u8) -> Vec<Frame> {
        (0..count).map(|_| make_frame(width, height, y)).collect()
    }

    #[test]
    fn test_ab_compare_two_candidates() {
        let comparator = AbComparator::new();
        let reference = make_frames(4, 32, 32, 128);
        let good_dist = make_frames(4, 32, 32, 130); // close to reference
        let bad_dist = make_frames(4, 32, 32, 50); // far from reference

        let candidates: Vec<(&str, Vec<Frame>)> = vec![("good", good_dist), ("bad", bad_dist)];
        let report = comparator
            .compare(&reference, &candidates)
            .expect("should succeed");

        assert_eq!(report.ranked.len(), 2);
        assert_eq!(report.ranked[0].label, "good");
        assert_eq!(report.ranked[1].label, "bad");
        assert!(
            report.ranked[0].composite_score > report.ranked[1].composite_score,
            "winner should have higher composite score"
        );
    }

    #[test]
    fn test_ab_winner_loser() {
        let comparator = AbComparator::new();
        let reference = make_frames(4, 32, 32, 128);
        let c1 = make_frames(4, 32, 32, 128); // identical
        let c2 = make_frames(4, 32, 32, 200); // different

        let candidates = vec![("identical", c1), ("different", c2)];
        let report = comparator
            .compare(&reference, &candidates)
            .expect("should succeed");

        assert!(report.winner().is_some());
        assert!(report.loser().is_some());
        assert_ne!(
            report.winner().map(|c| &c.label),
            report.loser().map(|c| &c.label)
        );
    }

    #[test]
    fn test_ab_empty_candidates() {
        let comparator = AbComparator::new();
        let reference = make_frames(4, 32, 32, 128);
        let candidates: Vec<(&str, Vec<Frame>)> = vec![];
        let report = comparator
            .compare(&reference, &candidates)
            .expect("should succeed");
        assert!(report.ranked.is_empty());
        assert!(report.winner().is_none());
    }

    #[test]
    fn test_ab_mismatched_frame_count_errors() {
        let comparator = AbComparator::new();
        let reference = make_frames(5, 32, 32, 128);
        let wrong_count = make_frames(3, 32, 32, 128);
        let candidates = vec![("wrong", wrong_count)];
        assert!(comparator.compare(&reference, &candidates).is_err());
    }

    #[test]
    fn test_metric_weights_normalized_sums_to_one() {
        let w = MetricWeights::perceptual().normalized();
        let total = w.ssim + w.psnr + w.vmaf + w.ms_ssim;
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_weights_zero_total_falls_back_to_uniform() {
        let w = MetricWeights {
            ssim: 0.0,
            psnr: 0.0,
            vmaf: 0.0,
            ms_ssim: 0.0,
        }
        .normalized();
        let total = w.ssim + w.psnr + w.vmaf + w.ms_ssim;
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ab_report_summary_nonempty() {
        let comparator = AbComparator::new();
        let reference = make_frames(2, 32, 32, 128);
        let c1 = make_frames(2, 32, 32, 128);
        let candidates = vec![("encode_a", c1)];
        let report = comparator
            .compare(&reference, &candidates)
            .expect("should succeed");
        let summary = report.summary();
        assert!(summary.contains("encode_a"));
        assert!(summary.contains("#1"));
    }

    #[test]
    fn test_composite_score_range() {
        let comparator = AbComparator::new();
        let reference = make_frames(4, 32, 32, 128);
        let candidate = make_frames(4, 32, 32, 140);
        let candidates = vec![("c", candidate)];
        let report = comparator
            .compare(&reference, &candidates)
            .expect("should succeed");
        for c in &report.ranked {
            assert!(c.composite_score >= 0.0, "composite must be non-negative");
            assert!(
                c.composite_score <= 105.0,
                "composite should be near [0,100]"
            );
        }
    }
}
