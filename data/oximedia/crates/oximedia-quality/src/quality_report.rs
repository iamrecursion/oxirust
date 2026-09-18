#![allow(dead_code)]
//! Quality assessment report generation.
//!
//! Produces structured reports summarizing quality metrics across an entire
//! video file or a batch of files, including per-frame breakdowns,
//! aggregate statistics, pass/fail verdicts, and human-readable summaries.

use std::collections::HashMap;

/// Severity level for quality issues found during assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational — no action required
    Info,
    /// Minor issue — may be acceptable
    Warning,
    /// Significant quality problem
    Error,
    /// Critical — content fails acceptance criteria
    Critical,
}

/// A single quality issue found during assessment.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Severity of the issue
    pub severity: Severity,
    /// Short description of the issue
    pub message: String,
    /// Frame number where the issue was detected (if applicable)
    pub frame: Option<usize>,
    /// Metric that triggered the issue
    pub metric: Option<String>,
    /// Measured value
    pub measured_value: Option<f64>,
    /// Threshold that was violated
    pub threshold: Option<f64>,
}

impl QualityIssue {
    /// Creates a new quality issue.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            frame: None,
            metric: None,
            measured_value: None,
            threshold: None,
        }
    }

    /// Sets the frame number.
    #[must_use]
    pub fn with_frame(mut self, frame: usize) -> Self {
        self.frame = Some(frame);
        self
    }

    /// Sets the metric name.
    pub fn with_metric(mut self, metric: impl Into<String>) -> Self {
        self.metric = Some(metric.into());
        self
    }

    /// Sets the measured value and threshold.
    #[must_use]
    pub fn with_values(mut self, measured: f64, threshold: f64) -> Self {
        self.measured_value = Some(measured);
        self.threshold = Some(threshold);
        self
    }
}

/// Per-frame quality measurement.
#[derive(Debug, Clone)]
pub struct FrameMetric {
    /// Frame index
    pub frame: usize,
    /// Metric name
    pub metric: String,
    /// Score value
    pub value: f64,
}

/// Aggregate statistics for a metric across all frames.
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Metric name
    pub name: String,
    /// Mean value
    pub mean: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Standard deviation
    pub stddev: f64,
    /// Median value
    pub median: f64,
    /// 5th percentile
    pub percentile_5: f64,
    /// 95th percentile
    pub percentile_95: f64,
    /// Number of samples
    pub count: usize,
}

/// Computes a `MetricSummary` from a slice of values.
#[allow(clippy::cast_precision_loss)]
pub fn summarize_values(name: &str, values: &[f64]) -> MetricSummary {
    if values.is_empty() {
        return MetricSummary {
            name: name.to_string(),
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
            median: 0.0,
            percentile_5: 0.0,
            percentile_95: 0.0,
            count: 0,
        };
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let p5_idx = ((sorted.len() as f64) * 0.05) as usize;
    let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
    let percentile_5 = sorted[p5_idx.min(sorted.len() - 1)];
    let percentile_95 = sorted[p95_idx.min(sorted.len() - 1)];

    MetricSummary {
        name: name.to_string(),
        mean,
        min,
        max,
        stddev,
        median,
        percentile_5,
        percentile_95,
        count: values.len(),
    }
}

/// Overall pass/fail verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// All metrics within acceptable thresholds
    Pass,
    /// Some warnings but no critical failures
    ConditionalPass,
    /// One or more metrics failed
    Fail,
}

/// A complete quality assessment report.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Report title / filename
    pub title: String,
    /// Per-frame metrics
    pub frame_metrics: Vec<FrameMetric>,
    /// Aggregate summaries per metric
    pub summaries: HashMap<String, MetricSummary>,
    /// Issues found during assessment
    pub issues: Vec<QualityIssue>,
    /// Overall verdict
    pub verdict: Verdict,
    /// Total number of frames assessed
    pub total_frames: usize,
    /// Duration in seconds
    pub duration_secs: f64,
}

impl QualityReport {
    /// Creates a new empty report.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            frame_metrics: Vec::new(),
            summaries: HashMap::new(),
            issues: Vec::new(),
            verdict: Verdict::Pass,
            total_frames: 0,
            duration_secs: 0.0,
        }
    }

    /// Adds a per-frame metric measurement.
    pub fn add_frame_metric(&mut self, frame: usize, metric: impl Into<String>, value: f64) {
        self.frame_metrics.push(FrameMetric {
            frame,
            metric: metric.into(),
            value,
        });
    }

    /// Adds an issue to the report.
    pub fn add_issue(&mut self, issue: QualityIssue) {
        self.issues.push(issue);
    }

    /// Computes summaries from the collected frame metrics.
    pub fn compute_summaries(&mut self) {
        let mut grouped: HashMap<String, Vec<f64>> = HashMap::new();
        for fm in &self.frame_metrics {
            grouped.entry(fm.metric.clone()).or_default().push(fm.value);
        }
        for (name, values) in &grouped {
            self.summaries
                .insert(name.clone(), summarize_values(name, values));
        }
    }

    /// Determines the verdict based on the collected issues.
    pub fn determine_verdict(&mut self) {
        let max_severity = self.issues.iter().map(|i| i.severity).max();
        self.verdict = match max_severity {
            Some(Severity::Critical | Severity::Error) => Verdict::Fail,
            Some(Severity::Warning) => Verdict::ConditionalPass,
            _ => Verdict::Pass,
        };
    }

    /// Returns a count of issues per severity level.
    #[must_use]
    pub fn issue_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for issue in &self.issues {
            let key = format!("{:?}", issue.severity);
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Generates a human-readable text summary of the report.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_text_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== Quality Report: {} ===", self.title));
        lines.push(format!(
            "Frames: {} | Duration: {:.2}s",
            self.total_frames, self.duration_secs
        ));
        lines.push(format!("Verdict: {:?}", self.verdict));
        lines.push(String::new());

        if !self.summaries.is_empty() {
            lines.push("--- Metric Summaries ---".to_string());
            for (name, s) in &self.summaries {
                lines.push(format!(
                    "{name}: mean={:.3} min={:.3} max={:.3} stddev={:.3} (n={})",
                    s.mean, s.min, s.max, s.stddev, s.count
                ));
            }
            lines.push(String::new());
        }

        if !self.issues.is_empty() {
            lines.push("--- Issues ---".to_string());
            for issue in &self.issues {
                let frame_str = issue
                    .frame
                    .map(|f| format!(" @frame {f}"))
                    .unwrap_or_default();
                lines.push(format!(
                    "[{:?}]{frame_str} {}",
                    issue.severity, issue.message
                ));
            }
        }

        lines.join("\n")
    }

    /// Returns the worst-case frame for a given metric (lowest value).
    #[must_use]
    pub fn worst_frame(&self, metric: &str) -> Option<&FrameMetric> {
        self.frame_metrics
            .iter()
            .filter(|fm| fm.metric == metric)
            .min_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Returns the best-case frame for a given metric (highest value).
    #[must_use]
    pub fn best_frame(&self, metric: &str) -> Option<&FrameMetric> {
        self.frame_metrics
            .iter()
            .filter(|fm| fm.metric == metric)
            .max_by(|a, b| {
                a.value
                    .partial_cmp(&b.value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Exports per-frame metrics to CSV format.
    ///
    /// The output has one row per frame-metric pair, with columns:
    /// `frame,metric,value`
    ///
    /// If `include_summaries` is `true`, a second section is appended after
    /// a blank line with columns:
    /// `metric,mean,min,max,stddev,median,p5,p95,count`
    #[must_use]
    pub fn to_csv(&self, include_summaries: bool) -> String {
        let mut csv = String::new();

        // Per-frame section
        csv.push_str("frame,metric,value\n");
        for fm in &self.frame_metrics {
            csv.push_str(&format!("{},{},{:.6}\n", fm.frame, fm.metric, fm.value));
        }

        // Optional summaries section
        if include_summaries && !self.summaries.is_empty() {
            csv.push('\n');
            csv.push_str("metric,mean,min,max,stddev,median,p5,p95,count\n");
            let mut metric_names: Vec<&String> = self.summaries.keys().collect();
            metric_names.sort();
            for name in metric_names {
                if let Some(s) = self.summaries.get(name) {
                    csv.push_str(&format!(
                        "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                        s.name,
                        s.mean,
                        s.min,
                        s.max,
                        s.stddev,
                        s.median,
                        s.percentile_5,
                        s.percentile_95,
                        s.count
                    ));
                }
            }
        }

        csv
    }

    /// Exports the report to CSV and writes it to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_csv(
        &self,
        path: &std::path::Path,
        include_summaries: bool,
    ) -> std::io::Result<()> {
        let csv = self.to_csv(include_summaries);
        std::fs::write(path, csv)
    }

    /// Exports only per-frame data for a specific metric to CSV.
    ///
    /// Columns: `frame,value`
    #[must_use]
    pub fn metric_to_csv(&self, metric: &str) -> String {
        let mut csv = String::new();
        csv.push_str("frame,value\n");
        for fm in &self.frame_metrics {
            if fm.metric == metric {
                csv.push_str(&format!("{},{:.6}\n", fm.frame, fm.value));
            }
        }
        csv
    }

    /// Exports issues to CSV format.
    ///
    /// Columns: `severity,frame,metric,measured,threshold,message`
    #[must_use]
    pub fn issues_to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("severity,frame,metric,measured,threshold,message\n");
        for issue in &self.issues {
            let frame_str = issue.frame.map_or_else(String::new, |f| f.to_string());
            let metric_str = issue.metric.as_deref().unwrap_or("");
            let measured_str = issue
                .measured_value
                .map_or_else(String::new, |v| format!("{v:.6}"));
            let threshold_str = issue
                .threshold
                .map_or_else(String::new, |v| format!("{v:.6}"));
            // Escape commas and quotes in message
            let escaped_message = if issue.message.contains(',') || issue.message.contains('"') {
                format!("\"{}\"", issue.message.replace('"', "\"\""))
            } else {
                issue.message.clone()
            };
            csv.push_str(&format!(
                "{:?},{},{},{},{},{}\n",
                issue.severity, frame_str, metric_str, measured_str, threshold_str, escaped_message
            ));
        }
        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_values_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = summarize_values("test", &values);
        assert!((s.mean - 3.0).abs() < 0.001);
        assert!((s.min - 1.0).abs() < 0.001);
        assert!((s.max - 5.0).abs() < 0.001);
        assert_eq!(s.count, 5);
    }

    #[test]
    fn test_summarize_values_empty() {
        let s = summarize_values("empty", &[]);
        assert_eq!(s.count, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_summarize_median_odd() {
        let values = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let s = summarize_values("m", &values);
        assert!((s.median - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_summarize_median_even() {
        let values = vec![1.0, 3.0, 5.0, 7.0];
        let s = summarize_values("m", &values);
        assert!((s.median - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_quality_issue_builder() {
        let issue = QualityIssue::new(Severity::Warning, "Low PSNR")
            .with_frame(42)
            .with_metric("PSNR")
            .with_values(28.5, 30.0);
        assert_eq!(issue.severity, Severity::Warning);
        assert_eq!(issue.frame, Some(42));
        assert_eq!(issue.metric.as_deref(), Some("PSNR"));
    }

    #[test]
    fn test_report_add_frame_metrics() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(1, "PSNR", 33.0);
        assert_eq!(report.frame_metrics.len(), 2);
    }

    #[test]
    fn test_report_compute_summaries() {
        let mut report = QualityReport::new("test.mp4");
        for i in 0..10 {
            report.add_frame_metric(i, "SSIM", 0.9 + (i as f64) * 0.01);
        }
        report.compute_summaries();
        assert!(report.summaries.contains_key("SSIM"));
        let s = &report.summaries["SSIM"];
        assert_eq!(s.count, 10);
    }

    #[test]
    fn test_report_verdict_pass() {
        let mut report = QualityReport::new("clean.mp4");
        report.determine_verdict();
        assert_eq!(report.verdict, Verdict::Pass);
    }

    #[test]
    fn test_report_verdict_fail() {
        let mut report = QualityReport::new("bad.mp4");
        report.add_issue(QualityIssue::new(
            Severity::Critical,
            "Black frame detected",
        ));
        report.determine_verdict();
        assert_eq!(report.verdict, Verdict::Fail);
    }

    #[test]
    fn test_report_verdict_conditional() {
        let mut report = QualityReport::new("warn.mp4");
        report.add_issue(QualityIssue::new(Severity::Warning, "Minor banding"));
        report.determine_verdict();
        assert_eq!(report.verdict, Verdict::ConditionalPass);
    }

    #[test]
    fn test_report_issue_counts() {
        let mut report = QualityReport::new("multi.mp4");
        report.add_issue(QualityIssue::new(Severity::Warning, "w1"));
        report.add_issue(QualityIssue::new(Severity::Warning, "w2"));
        report.add_issue(QualityIssue::new(Severity::Error, "e1"));
        let counts = report.issue_counts();
        assert_eq!(counts.get("Warning"), Some(&2));
        assert_eq!(counts.get("Error"), Some(&1));
    }

    #[test]
    fn test_report_worst_best_frame() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(1, "PSNR", 28.0);
        report.add_frame_metric(2, "PSNR", 40.0);
        let worst = report.worst_frame("PSNR").expect("should succeed in test");
        assert_eq!(worst.frame, 1);
        let best = report.best_frame("PSNR").expect("should succeed in test");
        assert_eq!(best.frame, 2);
    }

    #[test]
    fn test_report_text_summary() {
        let mut report = QualityReport::new("example.mp4");
        report.total_frames = 100;
        report.duration_secs = 4.0;
        let text = report.to_text_summary();
        assert!(text.contains("example.mp4"));
        assert!(text.contains("Frames: 100"));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    // ── CSV Export Tests ──────────────────────────────────────────────────

    #[test]
    fn test_to_csv_per_frame_only() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(1, "PSNR", 33.5);
        report.add_frame_metric(0, "SSIM", 0.95);
        let csv = report.to_csv(false);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "frame,metric,value");
        assert_eq!(lines.len(), 4); // header + 3 data rows
        assert!(lines[1].starts_with("0,PSNR,"));
        assert!(lines[2].starts_with("1,PSNR,"));
        assert!(lines[3].starts_with("0,SSIM,"));
    }

    #[test]
    fn test_to_csv_with_summaries() {
        let mut report = QualityReport::new("test.mp4");
        for i in 0..5 {
            report.add_frame_metric(i, "PSNR", 30.0 + i as f64);
        }
        report.compute_summaries();
        let csv = report.to_csv(true);
        assert!(csv.contains("metric,mean,min,max"));
        assert!(csv.contains("PSNR,"));
    }

    #[test]
    fn test_to_csv_empty_report() {
        let report = QualityReport::new("empty.mp4");
        let csv = report.to_csv(false);
        assert_eq!(csv, "frame,metric,value\n");
    }

    #[test]
    fn test_metric_to_csv() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(1, "PSNR", 33.0);
        report.add_frame_metric(0, "SSIM", 0.95);
        let csv = report.metric_to_csv("PSNR");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "frame,value");
        assert_eq!(lines.len(), 3); // header + 2 PSNR rows
    }

    #[test]
    fn test_metric_to_csv_missing_metric() {
        let report = QualityReport::new("test.mp4");
        let csv = report.metric_to_csv("VMAF");
        assert_eq!(csv, "frame,value\n");
    }

    #[test]
    fn test_issues_to_csv() {
        let mut report = QualityReport::new("test.mp4");
        report.add_issue(
            QualityIssue::new(Severity::Warning, "Low PSNR")
                .with_frame(5)
                .with_metric("PSNR")
                .with_values(28.0, 30.0),
        );
        report.add_issue(QualityIssue::new(Severity::Error, "Black frame").with_frame(10));
        let csv = report.issues_to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "severity,frame,metric,measured,threshold,message");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("Warning"));
        assert!(lines[1].contains("Low PSNR"));
        assert!(lines[2].contains("Error"));
    }

    #[test]
    fn test_issues_to_csv_escapes_commas() {
        let mut report = QualityReport::new("test.mp4");
        report.add_issue(QualityIssue::new(
            Severity::Info,
            "Value is 1,234 which is fine",
        ));
        let csv = report.issues_to_csv();
        // Message with comma should be quoted
        assert!(csv.contains("\"Value is 1,234 which is fine\""));
    }

    #[test]
    fn test_export_csv_writes_file() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(1, "PSNR", 33.0);
        report.compute_summaries();

        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_quality_test_export.csv");
        report
            .export_csv(&path, true)
            .expect("should write CSV file");
        let contents = std::fs::read_to_string(&path).expect("should read CSV file");
        assert!(contents.contains("frame,metric,value"));
        assert!(contents.contains("PSNR"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_to_csv_summaries_sorted() {
        let mut report = QualityReport::new("test.mp4");
        report.add_frame_metric(0, "SSIM", 0.95);
        report.add_frame_metric(0, "PSNR", 35.0);
        report.add_frame_metric(0, "VMAF", 80.0);
        report.compute_summaries();
        let csv = report.to_csv(true);
        // Summaries should be sorted alphabetically
        let summary_section: &str = csv.split("\n\n").last().unwrap_or("");
        let lines: Vec<&str> = summary_section.lines().collect();
        // Skip header line, first metric should be PSNR (alphabetically before SSIM)
        if lines.len() >= 3 {
            assert!(lines[1].starts_with("PSNR"));
            assert!(lines[2].starts_with("SSIM"));
            assert!(lines[3].starts_with("VMAF"));
        }
    }
}
