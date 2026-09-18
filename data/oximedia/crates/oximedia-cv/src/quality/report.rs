//! Quality assessment reporting and analysis.
//!
//! This module provides comprehensive quality reports that combine
//! multiple metrics and provide actionable insights.

use std::fmt::Write as FmtWrite;

use super::{PsnrResult, QualityMetrics, SsimResult, TemporalInfo, VmafResult};
use crate::error::{CvError, CvResult};
use oximedia_codec::VideoFrame;

/// Comprehensive quality report for video assessment.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Overall quality score (0-100).
    pub overall_score: f64,

    /// Quality level classification.
    pub quality_level: QualityLevel,

    /// Individual metric results.
    pub metrics: QualityMetrics,

    /// Detailed analysis.
    pub analysis: QualityAnalysis,

    /// Recommendations for improvement.
    pub recommendations: Vec<String>,

    /// Frame count analyzed.
    pub frame_count: usize,
}

impl QualityReport {
    /// Generate a comprehensive quality report.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn generate(reference: &VideoFrame, distorted: &VideoFrame) -> CvResult<Self> {
        let metrics = super::calculate_metrics(reference, distorted)?;

        let overall_score = metrics.overall_score();
        let quality_level = QualityLevel::from_score(overall_score);

        let analysis = QualityAnalysis::from_metrics(&metrics);
        let recommendations = generate_recommendations(&metrics, &analysis);

        Ok(Self {
            overall_score,
            quality_level,
            metrics,
            analysis,
            recommendations,
            frame_count: 1,
        })
    }

    /// Generate report for a sequence of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if sequences are incompatible.
    pub fn generate_sequence(
        reference_frames: &[VideoFrame],
        distorted_frames: &[VideoFrame],
    ) -> CvResult<Self> {
        if reference_frames.len() != distorted_frames.len() {
            return Err(CvError::invalid_parameter(
                "frame_count",
                format!("{} vs {}", reference_frames.len(), distorted_frames.len()),
            ));
        }

        if reference_frames.is_empty() {
            return Err(CvError::insufficient_data(1, 0));
        }

        let mut all_metrics = Vec::new();

        for (ref_frame, dist_frame) in reference_frames.iter().zip(distorted_frames.iter()) {
            let metrics = super::calculate_metrics(ref_frame, dist_frame)?;
            all_metrics.push(metrics);
        }

        let avg_metrics = average_metrics(&all_metrics);
        let overall_score = avg_metrics.overall_score();
        let quality_level = QualityLevel::from_score(overall_score);

        let analysis = QualityAnalysis::from_metrics_sequence(&all_metrics);
        let recommendations = generate_recommendations(&avg_metrics, &analysis);

        Ok(Self {
            overall_score,
            quality_level,
            metrics: avg_metrics,
            analysis,
            recommendations,
            frame_count: reference_frames.len(),
        })
    }

    /// Get a summary string of the report.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Quality: {} ({:.1}/100)\nPSNR: {:.2} dB | SSIM: {:.4} | VMAF: {:.2}\nFrames: {}",
            self.quality_level.description(),
            self.overall_score,
            self.metrics.psnr,
            self.metrics.ssim,
            self.metrics.vmaf,
            self.frame_count,
        )
    }

    /// Get detailed report as formatted string.
    #[must_use]
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Video Quality Assessment Report ===\n\n");

        let _ = writeln!(report, "Overall Score: {:.2}/100", self.overall_score);
        let _ = writeln!(
            report,
            "Quality Level: {}",
            self.quality_level.description()
        );
        let _ = writeln!(report, "Frames Analyzed: {}\n", self.frame_count);

        report.push_str("--- Objective Metrics ---\n");
        let _ = writeln!(report, "PSNR: {:.2} dB", self.metrics.psnr);
        let _ = writeln!(report, "SSIM: {:.4}", self.metrics.ssim);
        let _ = writeln!(report, "VMAF: {:.2}", self.metrics.vmaf);
        let _ = writeln!(report, "MS-SSIM: {:.4}", self.metrics.ms_ssim);
        let _ = writeln!(report, "PSNR-HVS: {:.2} dB", self.metrics.psnr_hvs);
        let _ = writeln!(report, "CIEDE2000: {:.2}\n", self.metrics.ciede2000);

        if !self.metrics.psnr_planes.is_empty() {
            report.push_str("--- Per-Plane PSNR ---\n");
            for (i, psnr) in self.metrics.psnr_planes.iter().enumerate() {
                let plane_name = match i {
                    0 => "Y (Luma)",
                    1 => "U (Cb)",
                    2 => "V (Cr)",
                    _ => "Plane",
                };
                let _ = writeln!(report, "{plane_name}: {psnr:.2} dB");
            }
            report.push('\n');
        }

        report.push_str("--- Quality Analysis ---\n");
        let _ = writeln!(report, "Spatial Quality: {}", self.analysis.spatial_quality);
        let _ = writeln!(
            report,
            "Temporal Quality: {}",
            self.analysis.temporal_quality
        );
        let _ = writeln!(report, "Color Fidelity: {}", self.analysis.color_fidelity);
        let _ = writeln!(
            report,
            "Structural Preservation: {}",
            self.analysis.structural_preservation
        );
        let _ = writeln!(
            report,
            "Perceptual Quality: {}\n",
            self.analysis.perceptual_quality
        );

        if !self.analysis.detected_issues.is_empty() {
            report.push_str("--- Detected Issues ---\n");
            for issue in &self.analysis.detected_issues {
                let _ = writeln!(report, "- {issue}");
            }
            report.push('\n');
        }

        if !self.recommendations.is_empty() {
            report.push_str("--- Recommendations ---\n");
            for rec in &self.recommendations {
                let _ = writeln!(report, "- {rec}");
            }
        }

        report
    }
}

/// Quality level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    /// Excellent quality (90-100).
    Excellent,
    /// Very Good quality (80-90).
    VeryGood,
    /// Good quality (70-80).
    Good,
    /// Fair quality (60-70).
    Fair,
    /// Poor quality (50-60).
    Poor,
    /// Very Poor quality (< 50).
    VeryPoor,
}

impl QualityLevel {
    /// Convert score to quality level.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 90.0 {
            Self::Excellent
        } else if score >= 80.0 {
            Self::VeryGood
        } else if score >= 70.0 {
            Self::Good
        } else if score >= 60.0 {
            Self::Fair
        } else if score >= 50.0 {
            Self::Poor
        } else {
            Self::VeryPoor
        }
    }

    /// Get descriptive string.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::VeryGood => "Very Good",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::VeryPoor => "Very Poor",
        }
    }
}

/// Detailed quality analysis results.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityAnalysis {
    /// Spatial quality assessment (0-100).
    pub spatial_quality: f64,

    /// Temporal quality assessment (0-100).
    pub temporal_quality: f64,

    /// Color fidelity score (0-100).
    pub color_fidelity: f64,

    /// Structural preservation score (0-100).
    pub structural_preservation: f64,

    /// Perceptual quality score (0-100).
    pub perceptual_quality: f64,

    /// Detected quality issues.
    pub detected_issues: Vec<String>,

    /// Strengths of the encode.
    pub strengths: Vec<String>,

    /// Weaknesses of the encode.
    pub weaknesses: Vec<String>,
}

impl QualityAnalysis {
    /// Analyze metrics for a single frame.
    #[must_use]
    pub fn from_metrics(metrics: &QualityMetrics) -> Self {
        let spatial_quality = calculate_spatial_quality(metrics);
        let temporal_quality = (metrics.temporal_info / 100.0).clamp(0.0, 1.0) * 100.0;
        let color_fidelity = calculate_color_fidelity(metrics);
        let structural_preservation = metrics.ssim * 100.0;
        let perceptual_quality = metrics.vmaf;

        let detected_issues = detect_issues(metrics);
        let strengths = identify_strengths(metrics);
        let weaknesses = identify_weaknesses(metrics);

        Self {
            spatial_quality,
            temporal_quality,
            color_fidelity,
            structural_preservation,
            perceptual_quality,
            detected_issues,
            strengths,
            weaknesses,
        }
    }

    /// Analyze metrics across a sequence.
    #[must_use]
    pub fn from_metrics_sequence(metrics_seq: &[QualityMetrics]) -> Self {
        if metrics_seq.is_empty() {
            return Self {
                spatial_quality: 0.0,
                temporal_quality: 0.0,
                color_fidelity: 0.0,
                structural_preservation: 0.0,
                perceptual_quality: 0.0,
                detected_issues: Vec::new(),
                strengths: Vec::new(),
                weaknesses: Vec::new(),
            };
        }

        let avg = average_metrics(metrics_seq);
        Self::from_metrics(&avg)
    }
}

/// Calculate spatial quality score.
fn calculate_spatial_quality(metrics: &QualityMetrics) -> f64 {
    // Combine PSNR and SSIM for spatial quality
    let psnr_score = ((metrics.psnr - 20.0) / 30.0).clamp(0.0, 1.0) * 100.0;
    let ssim_score = metrics.ssim * 100.0;

    (0.4 * psnr_score + 0.6 * ssim_score).clamp(0.0, 100.0)
}

/// Calculate color fidelity score.
fn calculate_color_fidelity(metrics: &QualityMetrics) -> f64 {
    // Lower CIEDE2000 is better (less color difference)
    // Normalize to 0-100 scale
    let color_score = (1.0 - (metrics.ciede2000 / 50.0).min(1.0)) * 100.0;

    // Also consider chroma PSNR if available
    if metrics.psnr_planes.len() >= 3 {
        let chroma_psnr = (metrics.psnr_planes[1] + metrics.psnr_planes[2]) / 2.0;
        let chroma_score = ((chroma_psnr - 20.0) / 30.0).clamp(0.0, 1.0) * 100.0;
        (0.5 * color_score + 0.5 * chroma_score).clamp(0.0, 100.0)
    } else {
        color_score
    }
}

/// Detect quality issues from metrics.
fn detect_issues(metrics: &QualityMetrics) -> Vec<String> {
    let mut issues = Vec::new();

    if metrics.psnr < 30.0 {
        issues.push("Low PSNR indicating significant pixel-level distortion".to_string());
    }

    if metrics.ssim < 0.9 {
        issues.push("Low SSIM indicating structural degradation".to_string());
    }

    if metrics.vmaf < 70.0 {
        issues.push("Low VMAF indicating poor perceptual quality".to_string());
    }

    if metrics.ciede2000 > 10.0 {
        issues.push("High color difference detected".to_string());
    }

    // Check for plane imbalance
    if metrics.psnr_planes.len() >= 3 {
        let y_psnr = metrics.psnr_planes[0];
        let u_psnr = metrics.psnr_planes[1];
        let v_psnr = metrics.psnr_planes[2];

        if (y_psnr - u_psnr).abs() > 10.0 || (y_psnr - v_psnr).abs() > 10.0 {
            issues.push("Imbalanced quality between luma and chroma planes".to_string());
        }
    }

    // Check MS-SSIM vs SSIM
    if metrics.ms_ssim < metrics.ssim - 0.1 {
        issues.push("Multi-scale quality degradation detected".to_string());
    }

    issues
}

/// Identify strengths in the encoding.
fn identify_strengths(metrics: &QualityMetrics) -> Vec<String> {
    let mut strengths = Vec::new();

    if metrics.psnr > 40.0 {
        strengths.push("Excellent pixel-level fidelity".to_string());
    }

    if metrics.ssim > 0.95 {
        strengths.push("Strong structural preservation".to_string());
    }

    if metrics.vmaf > 85.0 {
        strengths.push("High perceptual quality".to_string());
    }

    if metrics.ms_ssim > 0.95 {
        strengths.push("Consistent quality across scales".to_string());
    }

    if metrics.ciede2000 < 2.0 {
        strengths.push("Excellent color fidelity".to_string());
    }

    strengths
}

/// Identify weaknesses in the encoding.
fn identify_weaknesses(metrics: &QualityMetrics) -> Vec<String> {
    let mut weaknesses = Vec::new();

    if metrics.psnr < 35.0 && metrics.psnr >= 30.0 {
        weaknesses.push("Moderate pixel-level distortion".to_string());
    }

    if metrics.ssim < 0.95 && metrics.ssim >= 0.9 {
        weaknesses.push("Moderate structural distortion".to_string());
    }

    if metrics.vmaf < 80.0 && metrics.vmaf >= 70.0 {
        weaknesses.push("Moderate perceptual quality".to_string());
    }

    if metrics.psnr_hvs < metrics.psnr - 5.0 {
        weaknesses.push("Distortion in perceptually important areas".to_string());
    }

    weaknesses
}

/// Generate recommendations based on metrics and analysis.
fn generate_recommendations(metrics: &QualityMetrics, analysis: &QualityAnalysis) -> Vec<String> {
    let mut recommendations = Vec::new();

    if metrics.psnr < 35.0 {
        recommendations
            .push("Consider increasing bitrate or using higher quality preset".to_string());
    }

    if metrics.ssim < 0.9 {
        recommendations
            .push("Structure is degraded - consider using psychovisual optimizations".to_string());
    }

    if metrics.vmaf < 75.0 {
        recommendations.push(
            "Perceptual quality is low - increase bitrate or adjust rate control".to_string(),
        );
    }

    if analysis.color_fidelity < 80.0 {
        recommendations.push(
            "Color fidelity issues - check color space conversion and chroma subsampling"
                .to_string(),
        );
    }

    // Check for plane-specific issues
    if metrics.psnr_planes.len() >= 3 {
        let y_psnr = metrics.psnr_planes[0];
        let chroma_psnr = (metrics.psnr_planes[1] + metrics.psnr_planes[2]) / 2.0;

        if chroma_psnr < y_psnr - 8.0 {
            recommendations.push("Chroma quality is significantly lower - consider 4:4:4 subsampling or higher chroma QP offset".to_string());
        }
    }

    if metrics.ms_ssim < metrics.ssim - 0.05 {
        recommendations.push(
            "Quality varies across scales - optimize for multi-resolution encoding".to_string(),
        );
    }

    if recommendations.is_empty() {
        recommendations
            .push("Quality metrics are good - current settings are appropriate".to_string());
    }

    recommendations
}

/// Average metrics across multiple frames.
fn average_metrics(metrics_seq: &[QualityMetrics]) -> QualityMetrics {
    let count = metrics_seq.len() as f64;

    let psnr = metrics_seq.iter().map(|m| m.psnr).sum::<f64>() / count;
    let ssim = metrics_seq.iter().map(|m| m.ssim).sum::<f64>() / count;
    let vmaf = metrics_seq.iter().map(|m| m.vmaf).sum::<f64>() / count;
    let ms_ssim = metrics_seq.iter().map(|m| m.ms_ssim).sum::<f64>() / count;
    let psnr_hvs = metrics_seq.iter().map(|m| m.psnr_hvs).sum::<f64>() / count;
    let ciede2000 = metrics_seq.iter().map(|m| m.ciede2000).sum::<f64>() / count;
    let temporal_info = metrics_seq.iter().map(|m| m.temporal_info).sum::<f64>() / count;

    // Average per-plane metrics
    let plane_count = metrics_seq[0].psnr_planes.len();
    let mut psnr_planes = vec![0.0; plane_count];
    let mut ssim_planes = vec![0.0; plane_count];

    for metrics in metrics_seq {
        for (i, &val) in metrics.psnr_planes.iter().enumerate() {
            if i < psnr_planes.len() {
                psnr_planes[i] += val;
            }
        }
        for (i, &val) in metrics.ssim_planes.iter().enumerate() {
            if i < ssim_planes.len() {
                ssim_planes[i] += val;
            }
        }
    }

    for val in &mut psnr_planes {
        *val /= count;
    }
    for val in &mut ssim_planes {
        *val /= count;
    }

    QualityMetrics {
        psnr,
        ssim,
        vmaf,
        psnr_planes,
        ssim_planes,
        ms_ssim,
        psnr_hvs,
        ciede2000,
        temporal_info,
    }
}

/// Quality comparison between two distorted versions.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityComparison {
    /// Metrics for first distorted version.
    pub metrics_a: QualityMetrics,

    /// Metrics for second distorted version.
    pub metrics_b: QualityMetrics,

    /// Which version has better quality (true = A, false = B).
    pub winner: bool,

    /// Quality difference score.
    pub difference: f64,

    /// Comparison summary.
    pub summary: String,
}

impl QualityComparison {
    /// Compare two distorted versions against the same reference.
    ///
    /// # Errors
    ///
    /// Returns an error if comparison fails.
    pub fn compare(
        reference: &VideoFrame,
        distorted_a: &VideoFrame,
        distorted_b: &VideoFrame,
    ) -> CvResult<Self> {
        let metrics_a = super::calculate_metrics(reference, distorted_a)?;
        let metrics_b = super::calculate_metrics(reference, distorted_b)?;

        let score_a = metrics_a.overall_score();
        let score_b = metrics_b.overall_score();

        let winner = score_a > score_b;
        let difference = (score_a - score_b).abs();

        let summary = generate_comparison_summary(&metrics_a, &metrics_b, winner, difference);

        Ok(Self {
            metrics_a,
            metrics_b,
            winner,
            difference,
            summary,
        })
    }
}

/// Generate comparison summary.
fn generate_comparison_summary(
    metrics_a: &QualityMetrics,
    metrics_b: &QualityMetrics,
    winner: bool,
    difference: f64,
) -> String {
    let (winner_name, winner_metrics, loser_metrics) = if winner {
        ("Version A", metrics_a, metrics_b)
    } else {
        ("Version B", metrics_b, metrics_a)
    };

    let mut summary =
        format!("{winner_name} has better quality (difference: {difference:.1} points)\n\n");

    summary.push_str("Metric Comparison:\n");
    let _ = writeln!(
        summary,
        "PSNR: {:.2} vs {:.2} dB",
        winner_metrics.psnr, loser_metrics.psnr
    );
    let _ = writeln!(
        summary,
        "SSIM: {:.4} vs {:.4}",
        winner_metrics.ssim, loser_metrics.ssim
    );
    let _ = writeln!(
        summary,
        "VMAF: {:.2} vs {:.2}",
        winner_metrics.vmaf, loser_metrics.vmaf
    );

    summary
}
