#![allow(dead_code)]
//! Transcode report generation for post-conversion analysis.
//!
//! This module generates detailed reports comparing input and output media
//! files, tracking quality metrics, bitrate efficiency, encoding statistics,
//! and providing actionable recommendations for future conversions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Quality metric with a name, value, and acceptable range.
#[derive(Debug, Clone)]
pub struct QualityMetric {
    /// Name of the metric (e.g., "PSNR", "SSIM", "VMAF").
    pub name: String,
    /// Computed value.
    pub value: f64,
    /// Minimum acceptable value (pass threshold).
    pub pass_threshold: f64,
    /// Whether this metric passed the threshold.
    pub passed: bool,
    /// Unit of measurement (e.g., "dB", "", "score").
    pub unit: String,
}

impl QualityMetric {
    /// Create a new quality metric and evaluate it against the threshold.
    #[must_use]
    pub fn new(name: &str, value: f64, pass_threshold: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            pass_threshold,
            passed: value >= pass_threshold,
            unit: unit.to_string(),
        }
    }

    /// Format the metric as a human-readable string.
    #[must_use]
    pub fn format_display(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        format!(
            "{}: {:.2} {} [threshold: {:.2} {}] -> {}",
            self.name, self.value, self.unit, self.pass_threshold, self.unit, status,
        )
    }
}

/// Bitrate statistics for a media stream.
#[derive(Debug, Clone)]
pub struct BitrateStats {
    /// Average bitrate in bits per second.
    pub avg_bitrate_bps: u64,
    /// Maximum instantaneous bitrate in bps.
    pub max_bitrate_bps: u64,
    /// Minimum instantaneous bitrate in bps.
    pub min_bitrate_bps: u64,
    /// Standard deviation of bitrate.
    pub std_dev_bps: f64,
    /// Bitrate efficiency (quality per bit, higher = better).
    pub efficiency_score: f64,
}

impl BitrateStats {
    /// Compute the bitrate variability ratio (max / avg).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn variability_ratio(&self) -> f64 {
        if self.avg_bitrate_bps == 0 {
            0.0
        } else {
            self.max_bitrate_bps as f64 / self.avg_bitrate_bps as f64
        }
    }

    /// Whether the bitrate is relatively stable (variability < 2.0).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.variability_ratio() < 2.0
    }
}

/// Encoding performance statistics.
#[derive(Debug, Clone)]
pub struct EncodingStats {
    /// Total encoding wall-clock time.
    pub wall_time: Duration,
    /// Frames per second during encoding.
    pub fps: f64,
    /// Speed ratio relative to real-time (> 1 = faster than real-time).
    pub speed_ratio: f64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Number of passes performed.
    pub pass_count: u32,
    /// Total frames encoded.
    pub total_frames: u64,
}

impl EncodingStats {
    /// Compute estimated time to encode a given duration at the same speed.
    #[must_use]
    pub fn estimate_encode_time(&self, content_duration_s: f64) -> Duration {
        if self.speed_ratio <= 0.0 {
            return Duration::from_secs(0);
        }
        Duration::from_secs_f64(content_duration_s / self.speed_ratio)
    }
}

/// A recommendation for improving future conversions.
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Severity of the recommendation.
    pub severity: RecommendationSeverity,
    /// Category (e.g., "quality", "performance", "size").
    pub category: String,
    /// Human-readable suggestion.
    pub message: String,
}

/// Severity level for recommendations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationSeverity {
    /// Nice-to-have improvement.
    Suggestion,
    /// Noticeable improvement opportunity.
    Advisory,
    /// Significant issue that should be addressed.
    Important,
    /// Critical issue affecting quality or compliance.
    Critical,
}

impl std::fmt::Display for RecommendationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Suggestion => write!(f, "SUGGESTION"),
            Self::Advisory => write!(f, "ADVISORY"),
            Self::Important => write!(f, "IMPORTANT"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Full transcode report comparing input and output.
#[derive(Debug, Clone)]
pub struct TranscodeReport {
    /// Report identifier.
    pub report_id: String,
    /// Input file path.
    pub input_path: PathBuf,
    /// Output file path.
    pub output_path: PathBuf,
    /// Input file size in bytes.
    pub input_size_bytes: u64,
    /// Output file size in bytes.
    pub output_size_bytes: u64,
    /// Input duration in seconds.
    pub input_duration_s: f64,
    /// Output duration in seconds.
    pub output_duration_s: f64,
    /// Quality metrics.
    pub quality_metrics: Vec<QualityMetric>,
    /// Video stream bitrate stats.
    pub video_bitrate: Option<BitrateStats>,
    /// Audio stream bitrate stats.
    pub audio_bitrate: Option<BitrateStats>,
    /// Encoding performance stats.
    pub encoding_stats: Option<EncodingStats>,
    /// Recommendations.
    pub recommendations: Vec<Recommendation>,
    /// Custom metadata / notes.
    pub notes: HashMap<String, String>,
}

impl TranscodeReport {
    /// Create a new empty report with the given paths.
    #[must_use]
    pub fn new(report_id: &str, input: PathBuf, output: PathBuf) -> Self {
        Self {
            report_id: report_id.to_string(),
            input_path: input,
            output_path: output,
            input_size_bytes: 0,
            output_size_bytes: 0,
            input_duration_s: 0.0,
            output_duration_s: 0.0,
            quality_metrics: Vec::new(),
            video_bitrate: None,
            audio_bitrate: None,
            encoding_stats: None,
            recommendations: Vec::new(),
            notes: HashMap::new(),
        }
    }

    /// Compression ratio (input / output size).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self) -> f64 {
        if self.output_size_bytes == 0 {
            0.0
        } else {
            self.input_size_bytes as f64 / self.output_size_bytes as f64
        }
    }

    /// Size reduction percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn size_reduction_pct(&self) -> f64 {
        if self.input_size_bytes == 0 {
            0.0
        } else {
            let diff = self.input_size_bytes.saturating_sub(self.output_size_bytes);
            (diff as f64 / self.input_size_bytes as f64) * 100.0
        }
    }

    /// Duration drift between input and output.
    #[must_use]
    pub fn duration_drift_s(&self) -> f64 {
        (self.output_duration_s - self.input_duration_s).abs()
    }

    /// Whether all quality metrics passed.
    #[must_use]
    pub fn all_quality_passed(&self) -> bool {
        self.quality_metrics.iter().all(|m| m.passed)
    }

    /// Count of failed quality metrics.
    #[must_use]
    pub fn failed_metric_count(&self) -> usize {
        self.quality_metrics.iter().filter(|m| !m.passed).count()
    }

    /// Add a quality metric.
    pub fn add_metric(&mut self, metric: QualityMetric) {
        self.quality_metrics.push(metric);
    }

    /// Add a recommendation.
    pub fn add_recommendation(
        &mut self,
        severity: RecommendationSeverity,
        category: &str,
        message: &str,
    ) {
        self.recommendations.push(Recommendation {
            severity,
            category: category.to_string(),
            message: message.to_string(),
        });
    }

    /// Generate recommendations based on collected metrics.
    pub fn generate_recommendations(&mut self) {
        // Check duration drift
        if self.duration_drift_s() > 0.1 {
            self.add_recommendation(
                RecommendationSeverity::Important,
                "sync",
                &format!(
                    "Duration drift of {:.3}s detected between input and output",
                    self.duration_drift_s()
                ),
            );
        }

        // Check compression ratio
        if self.compression_ratio() < 1.0 && self.output_size_bytes > 0 {
            self.add_recommendation(
                RecommendationSeverity::Advisory,
                "size",
                "Output file is larger than input. Consider using a more efficient codec or lower bitrate.",
            );
        }

        // Check quality metrics
        let quality_recs: Vec<Recommendation> = self
            .quality_metrics
            .iter()
            .filter(|metric| !metric.passed)
            .map(|metric| Recommendation {
                severity: RecommendationSeverity::Critical,
                category: "quality".to_string(),
                message: format!(
                    "Quality metric '{}' failed: {:.2} < {:.2} {}",
                    metric.name, metric.value, metric.pass_threshold, metric.unit,
                ),
            })
            .collect();
        self.recommendations.extend(quality_recs);

        // Check bitrate stability
        if let Some(ref vb) = self.video_bitrate {
            if !vb.is_stable() {
                self.add_recommendation(
                    RecommendationSeverity::Suggestion,
                    "performance",
                    "Video bitrate is highly variable. Consider VBV/HRD constraints for streaming.",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_metric_pass() {
        let m = QualityMetric::new("PSNR", 40.0, 35.0, "dB");
        assert!(m.passed);
        assert!(m.format_display().contains("PASS"));
    }

    #[test]
    fn test_quality_metric_fail() {
        let m = QualityMetric::new("SSIM", 0.85, 0.90, "");
        assert!(!m.passed);
        assert!(m.format_display().contains("FAIL"));
    }

    #[test]
    fn test_bitrate_variability() {
        let stats = BitrateStats {
            avg_bitrate_bps: 5_000_000,
            max_bitrate_bps: 8_000_000,
            min_bitrate_bps: 2_000_000,
            std_dev_bps: 1_000_000.0,
            efficiency_score: 0.8,
        };
        let ratio = stats.variability_ratio();
        assert!((ratio - 1.6).abs() < 0.01);
        assert!(stats.is_stable());
    }

    #[test]
    fn test_bitrate_unstable() {
        let stats = BitrateStats {
            avg_bitrate_bps: 5_000_000,
            max_bitrate_bps: 15_000_000,
            min_bitrate_bps: 500_000,
            std_dev_bps: 5_000_000.0,
            efficiency_score: 0.5,
        };
        assert!(!stats.is_stable());
    }

    #[test]
    fn test_bitrate_zero_avg() {
        let stats = BitrateStats {
            avg_bitrate_bps: 0,
            max_bitrate_bps: 0,
            min_bitrate_bps: 0,
            std_dev_bps: 0.0,
            efficiency_score: 0.0,
        };
        assert!((stats.variability_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_encoding_stats_estimate() {
        let stats = EncodingStats {
            wall_time: Duration::from_secs(10),
            fps: 60.0,
            speed_ratio: 2.0,
            peak_memory_bytes: 1_000_000,
            pass_count: 1,
            total_frames: 600,
        };
        let est = stats.estimate_encode_time(120.0);
        assert!((est.as_secs_f64() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_encoding_stats_zero_speed() {
        let stats = EncodingStats {
            wall_time: Duration::from_secs(0),
            fps: 0.0,
            speed_ratio: 0.0,
            peak_memory_bytes: 0,
            pass_count: 0,
            total_frames: 0,
        };
        assert_eq!(stats.estimate_encode_time(100.0), Duration::from_secs(0));
    }

    #[test]
    fn test_report_compression_ratio() {
        let mut r = TranscodeReport::new("r1", PathBuf::from("in"), PathBuf::from("out"));
        r.input_size_bytes = 1_000_000;
        r.output_size_bytes = 400_000;
        assert!((r.compression_ratio() - 2.5).abs() < 0.01);
        assert!((r.size_reduction_pct() - 60.0).abs() < 0.01);
    }

    #[test]
    fn test_report_zero_output() {
        let r = TranscodeReport::new("r2", PathBuf::from("in"), PathBuf::from("out"));
        assert!((r.compression_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_duration_drift() {
        let mut r = TranscodeReport::new("r3", PathBuf::from("in"), PathBuf::from("out"));
        r.input_duration_s = 60.0;
        r.output_duration_s = 60.05;
        assert!((r.duration_drift_s() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_all_quality_passed() {
        let mut r = TranscodeReport::new("r4", PathBuf::from("in"), PathBuf::from("out"));
        r.add_metric(QualityMetric::new("PSNR", 40.0, 35.0, "dB"));
        r.add_metric(QualityMetric::new("SSIM", 0.95, 0.90, ""));
        assert!(r.all_quality_passed());
        assert_eq!(r.failed_metric_count(), 0);
    }

    #[test]
    fn test_failed_quality() {
        let mut r = TranscodeReport::new("r5", PathBuf::from("in"), PathBuf::from("out"));
        r.add_metric(QualityMetric::new("PSNR", 30.0, 35.0, "dB"));
        assert!(!r.all_quality_passed());
        assert_eq!(r.failed_metric_count(), 1);
    }

    #[test]
    fn test_generate_recommendations() {
        let mut r = TranscodeReport::new("r6", PathBuf::from("in"), PathBuf::from("out"));
        r.input_size_bytes = 100;
        r.output_size_bytes = 200;
        r.input_duration_s = 10.0;
        r.output_duration_s = 10.5;
        r.add_metric(QualityMetric::new("VMAF", 50.0, 70.0, "score"));
        r.generate_recommendations();
        assert!(
            r.recommendations.len() >= 2,
            "Should have recommendations for drift and quality"
        );
    }

    #[test]
    fn test_recommendation_severity_display() {
        assert_eq!(
            format!("{}", RecommendationSeverity::Suggestion),
            "SUGGESTION"
        );
        assert_eq!(format!("{}", RecommendationSeverity::Critical), "CRITICAL");
    }
}
