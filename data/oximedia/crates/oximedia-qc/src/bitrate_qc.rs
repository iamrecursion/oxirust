#![allow(dead_code)]
//! Bitrate distribution quality control analysis.
//!
//! Validates CBR/VBR bitrate compliance, detects scenes with insufficient or
//! excessive bitrate, analyzes bitrate consistency, and reports statistical
//! distribution metrics across the media file.

/// Bitrate mode of the encoded content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitrateMode {
    /// Constant bitrate — each segment should be close to the target.
    Cbr,
    /// Variable bitrate — bitrate fluctuates within bounds.
    Vbr,
    /// Average bitrate — overall average should match the target.
    Abr,
    /// Constant quality factor (CRF/CQP).
    ConstantQuality,
}

impl std::fmt::Display for BitrateMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cbr => write!(f, "CBR"),
            Self::Vbr => write!(f, "VBR"),
            Self::Abr => write!(f, "ABR"),
            Self::ConstantQuality => write!(f, "CQ"),
        }
    }
}

/// A bitrate sample for a segment of the file.
#[derive(Debug, Clone)]
pub struct BitrateSample {
    /// Start time of the segment in seconds.
    pub start_secs: f64,
    /// Duration of the segment in seconds.
    pub duration_secs: f64,
    /// Measured bitrate of this segment in kbps.
    pub bitrate_kbps: f64,
    /// Whether this segment contains a scene change.
    pub is_scene_change: bool,
}

impl BitrateSample {
    /// Creates a new bitrate sample.
    #[must_use]
    pub fn new(start_secs: f64, duration_secs: f64, bitrate_kbps: f64) -> Self {
        Self {
            start_secs,
            duration_secs,
            bitrate_kbps,
            is_scene_change: false,
        }
    }

    /// Marks this sample as containing a scene change.
    #[must_use]
    pub fn with_scene_change(mut self, is_sc: bool) -> Self {
        self.is_scene_change = is_sc;
        self
    }

    /// Returns the end time of this segment in seconds.
    #[must_use]
    pub fn end_secs(&self) -> f64 {
        self.start_secs + self.duration_secs
    }
}

/// Severity level for bitrate QC findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitrateSeverity {
    /// Informational.
    Info,
    /// Warning — borderline compliance.
    Warning,
    /// Error — out of specification.
    Error,
}

impl std::fmt::Display for BitrateSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// A single bitrate QC finding.
#[derive(Debug, Clone)]
pub struct BitrateFinding {
    /// Severity of the finding.
    pub severity: BitrateSeverity,
    /// Short code for the check.
    pub code: String,
    /// Human-readable description.
    pub message: String,
}

impl BitrateFinding {
    /// Creates a new bitrate finding.
    #[must_use]
    pub fn new(severity: BitrateSeverity, code: &str, message: &str) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message: message.to_string(),
        }
    }

    /// Returns whether this finding indicates a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.severity == BitrateSeverity::Error
    }
}

/// Statistical summary of bitrate distribution.
#[derive(Debug, Clone)]
pub struct BitrateStats {
    /// Minimum bitrate in kbps.
    pub min_kbps: f64,
    /// Maximum bitrate in kbps.
    pub max_kbps: f64,
    /// Mean bitrate in kbps.
    pub mean_kbps: f64,
    /// Median bitrate in kbps.
    pub median_kbps: f64,
    /// Standard deviation in kbps.
    pub std_dev_kbps: f64,
    /// Coefficient of variation (std_dev / mean).
    pub cv: f64,
    /// 5th percentile in kbps.
    pub p5_kbps: f64,
    /// 95th percentile in kbps.
    pub p95_kbps: f64,
    /// Number of samples analyzed.
    pub sample_count: usize,
}

impl BitrateStats {
    /// Computes bitrate statistics from a set of samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_samples(samples: &[BitrateSample]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut values: Vec<f64> = samples.iter().map(|s| s.bitrate_kbps).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / n as f64;

        let median = if n % 2 == 0 {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        } else {
            values[n / 2]
        };

        let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let cv = if mean.abs() > f64::EPSILON {
            std_dev / mean
        } else {
            0.0
        };

        #[allow(clippy::similar_names)]
        let p5_idx = ((n as f64) * 0.05).floor() as usize;
        let p95_idx = ((n as f64) * 0.95).ceil() as usize;
        let p5 = values[p5_idx.min(n - 1)];
        let p95 = values[p95_idx.min(n - 1)];

        Some(Self {
            min_kbps: values[0],
            max_kbps: values[n - 1],
            mean_kbps: mean,
            median_kbps: median,
            std_dev_kbps: std_dev,
            cv,
            p5_kbps: p5,
            p95_kbps: p95,
            sample_count: n,
        })
    }
}

/// Result of a bitrate QC analysis.
#[derive(Debug, Clone)]
pub struct BitrateQcReport {
    /// Whether the overall check passed.
    pub passed: bool,
    /// All findings.
    pub findings: Vec<BitrateFinding>,
    /// Statistical summary.
    pub stats: Option<BitrateStats>,
    /// Detected bitrate mode.
    pub detected_mode: Option<BitrateMode>,
}

impl BitrateQcReport {
    /// Creates a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            passed: true,
            findings: Vec::new(),
            stats: None,
            detected_mode: None,
        }
    }

    /// Adds a finding and updates pass/fail status.
    pub fn add_finding(&mut self, finding: BitrateFinding) {
        if finding.is_failure() {
            self.passed = false;
        }
        self.findings.push(finding);
    }

    /// Returns only the error findings.
    #[must_use]
    pub fn errors(&self) -> Vec<&BitrateFinding> {
        self.findings.iter().filter(|f| f.is_failure()).collect()
    }

    /// Returns total finding count.
    #[must_use]
    pub fn finding_count(&self) -> usize {
        self.findings.len()
    }
}

impl Default for BitrateQcReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for bitrate QC checks.
#[derive(Debug, Clone)]
pub struct BitrateQcConfig {
    /// Expected bitrate mode.
    pub expected_mode: Option<BitrateMode>,
    /// Target bitrate in kbps (for CBR/ABR checks).
    pub target_kbps: Option<f64>,
    /// Minimum allowed bitrate in kbps.
    pub min_kbps: f64,
    /// Maximum allowed bitrate in kbps.
    pub max_kbps: f64,
    /// CBR tolerance as a fraction (e.g. 0.10 = +/- 10%).
    pub cbr_tolerance: f64,
    /// ABR tolerance as a fraction for the overall average.
    pub abr_tolerance: f64,
    /// Maximum coefficient of variation for CBR content (e.g. 0.15).
    pub max_cv_for_cbr: f64,
    /// Whether to allow scene-change bitrate spikes.
    pub allow_scene_change_spikes: bool,
}

impl BitrateQcConfig {
    /// Creates a new bitrate QC configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            expected_mode: None,
            target_kbps: None,
            min_kbps: 100.0,
            max_kbps: 100_000.0,
            cbr_tolerance: 0.10,
            abr_tolerance: 0.05,
            max_cv_for_cbr: 0.15,
            allow_scene_change_spikes: true,
        }
    }

    /// Sets the expected bitrate mode.
    #[must_use]
    pub fn with_expected_mode(mut self, mode: BitrateMode) -> Self {
        self.expected_mode = Some(mode);
        self
    }

    /// Sets the target bitrate.
    #[must_use]
    pub fn with_target_kbps(mut self, kbps: f64) -> Self {
        self.target_kbps = Some(kbps);
        self
    }

    /// Sets the minimum allowed bitrate.
    #[must_use]
    pub fn with_min_kbps(mut self, kbps: f64) -> Self {
        self.min_kbps = kbps;
        self
    }

    /// Sets the maximum allowed bitrate.
    #[must_use]
    pub fn with_max_kbps(mut self, kbps: f64) -> Self {
        self.max_kbps = kbps;
        self
    }

    /// Sets the CBR tolerance.
    #[must_use]
    pub fn with_cbr_tolerance(mut self, tolerance: f64) -> Self {
        self.cbr_tolerance = tolerance;
        self
    }
}

impl Default for BitrateQcConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Bitrate quality control checker.
///
/// Analyzes bitrate distribution, validates against CBR/VBR/ABR targets,
/// detects under/over-bitrate segments, and reports statistical summaries.
#[derive(Debug, Clone)]
pub struct BitrateQcChecker {
    /// Configuration for the checker.
    config: BitrateQcConfig,
}

impl BitrateQcChecker {
    /// Creates a new bitrate QC checker with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BitrateQcConfig::new(),
        }
    }

    /// Creates a new bitrate QC checker with the given configuration.
    #[must_use]
    pub fn with_config(config: BitrateQcConfig) -> Self {
        Self { config }
    }

    /// Analyzes a sequence of bitrate samples.
    #[must_use]
    pub fn analyze(&self, samples: &[BitrateSample]) -> BitrateQcReport {
        let mut report = BitrateQcReport::new();

        if samples.is_empty() {
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Info,
                "BR-001",
                "No bitrate samples available",
            ));
            return report;
        }

        // Compute statistics
        let stats = BitrateStats::from_samples(samples);
        report.stats = stats.clone();

        if let Some(ref stats) = stats {
            // Detect mode
            let detected = self.detect_mode(stats);
            report.detected_mode = Some(detected);

            // Check overall bounds
            self.check_bounds(samples, &mut report);

            // Check mode-specific compliance
            if let Some(target) = self.config.target_kbps {
                match self.config.expected_mode.unwrap_or(detected) {
                    BitrateMode::Cbr => self.check_cbr(stats, target, &mut report),
                    BitrateMode::Abr => self.check_abr(stats, target, &mut report),
                    BitrateMode::Vbr | BitrateMode::ConstantQuality => {
                        self.check_vbr(stats, &mut report);
                    }
                }
            }

            // Report statistics
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Info,
                "BR-100",
                &format!(
                    "Bitrate stats: mean={:.0} kbps, median={:.0} kbps, stddev={:.0} kbps, CV={:.3}",
                    stats.mean_kbps, stats.median_kbps, stats.std_dev_kbps, stats.cv
                ),
            ));
        }

        report
    }

    /// Detects the bitrate mode based on the coefficient of variation.
    #[must_use]
    fn detect_mode(&self, stats: &BitrateStats) -> BitrateMode {
        if stats.cv < 0.05 {
            BitrateMode::Cbr
        } else if stats.cv < 0.20 {
            BitrateMode::Abr
        } else {
            BitrateMode::Vbr
        }
    }

    /// Checks that all samples are within absolute bounds.
    fn check_bounds(&self, samples: &[BitrateSample], report: &mut BitrateQcReport) {
        for sample in samples {
            if self.config.allow_scene_change_spikes && sample.is_scene_change {
                continue;
            }

            if sample.bitrate_kbps < self.config.min_kbps {
                report.add_finding(BitrateFinding::new(
                    BitrateSeverity::Error,
                    "BR-010",
                    &format!(
                        "Bitrate {:.0} kbps at {:.2}s is below minimum {:.0} kbps",
                        sample.bitrate_kbps, sample.start_secs, self.config.min_kbps
                    ),
                ));
            }

            if sample.bitrate_kbps > self.config.max_kbps {
                report.add_finding(BitrateFinding::new(
                    BitrateSeverity::Error,
                    "BR-011",
                    &format!(
                        "Bitrate {:.0} kbps at {:.2}s exceeds maximum {:.0} kbps",
                        sample.bitrate_kbps, sample.start_secs, self.config.max_kbps
                    ),
                ));
            }
        }
    }

    /// Checks CBR compliance — coefficient of variation and per-sample deviation.
    fn check_cbr(&self, stats: &BitrateStats, target: f64, report: &mut BitrateQcReport) {
        // CV check
        if stats.cv > self.config.max_cv_for_cbr {
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Error,
                "BR-020",
                &format!(
                    "CBR: coefficient of variation {:.3} exceeds max {:.3}",
                    stats.cv, self.config.max_cv_for_cbr
                ),
            ));
        }

        // Mean vs target
        let deviation = ((stats.mean_kbps - target) / target).abs();
        if deviation > self.config.cbr_tolerance {
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Error,
                "BR-021",
                &format!(
                    "CBR: mean bitrate {:.0} kbps deviates {:.1}% from target {:.0} kbps (tolerance {:.1}%)",
                    stats.mean_kbps,
                    deviation * 100.0,
                    target,
                    self.config.cbr_tolerance * 100.0
                ),
            ));
        }

        // Range check
        let range_ratio = (stats.max_kbps - stats.min_kbps) / target;
        if range_ratio > self.config.cbr_tolerance * 3.0 {
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Warning,
                "BR-022",
                &format!(
                    "CBR: bitrate range {:.0}-{:.0} kbps is wide for target {:.0} kbps",
                    stats.min_kbps, stats.max_kbps, target
                ),
            ));
        }
    }

    /// Checks ABR compliance — overall average must match target.
    fn check_abr(&self, stats: &BitrateStats, target: f64, report: &mut BitrateQcReport) {
        let deviation = ((stats.mean_kbps - target) / target).abs();
        if deviation > self.config.abr_tolerance {
            report.add_finding(BitrateFinding::new(
                BitrateSeverity::Error,
                "BR-030",
                &format!(
                    "ABR: mean bitrate {:.0} kbps deviates {:.1}% from target {:.0} kbps (tolerance {:.1}%)",
                    stats.mean_kbps,
                    deviation * 100.0,
                    target,
                    self.config.abr_tolerance * 100.0
                ),
            ));
        }
    }

    /// Checks VBR content for extreme variation.
    fn check_vbr(&self, stats: &BitrateStats, report: &mut BitrateQcReport) {
        // VBR is expected to vary, but extreme ratios are suspicious
        if stats.min_kbps > 0.0 {
            let ratio = stats.max_kbps / stats.min_kbps;
            if ratio > 50.0 {
                report.add_finding(BitrateFinding::new(
                    BitrateSeverity::Warning,
                    "BR-040",
                    &format!(
                        "VBR: extreme bitrate ratio {:.1}x (min={:.0}, max={:.0} kbps)",
                        ratio, stats.min_kbps, stats.max_kbps
                    ),
                ));
            }
        }
    }
}

impl Default for BitrateQcChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_samples(kbps: f64, count: usize) -> Vec<BitrateSample> {
        (0..count)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let start = i as f64;
                BitrateSample::new(start, 1.0, kbps)
            })
            .collect()
    }

    #[test]
    fn test_empty_samples() {
        let checker = BitrateQcChecker::new();
        let report = checker.analyze(&[]);
        assert!(report.passed);
        assert!(report.findings.iter().any(|f| f.code == "BR-001"));
    }

    #[test]
    fn test_uniform_cbr_detected() {
        let checker = BitrateQcChecker::new();
        let samples = uniform_samples(5000.0, 20);
        let report = checker.analyze(&samples);
        assert_eq!(report.detected_mode, Some(BitrateMode::Cbr));
    }

    #[test]
    fn test_stats_computation() {
        let samples = vec![
            BitrateSample::new(0.0, 1.0, 100.0),
            BitrateSample::new(1.0, 1.0, 200.0),
            BitrateSample::new(2.0, 1.0, 300.0),
        ];
        let stats = BitrateStats::from_samples(&samples).expect("should succeed in test");
        assert!((stats.mean_kbps - 200.0).abs() < f64::EPSILON);
        assert!((stats.median_kbps - 200.0).abs() < f64::EPSILON);
        assert!((stats.min_kbps - 100.0).abs() < f64::EPSILON);
        assert!((stats.max_kbps - 300.0).abs() < f64::EPSILON);
        assert_eq!(stats.sample_count, 3);
    }

    #[test]
    fn test_stats_empty() {
        let result = BitrateStats::from_samples(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_below_min_bitrate() {
        let config = BitrateQcConfig::new().with_min_kbps(500.0);
        let checker = BitrateQcChecker::with_config(config);
        let samples = vec![BitrateSample::new(0.0, 1.0, 200.0)];
        let report = checker.analyze(&samples);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "BR-010"));
    }

    #[test]
    fn test_above_max_bitrate() {
        let config = BitrateQcConfig::new().with_max_kbps(10_000.0);
        let checker = BitrateQcChecker::with_config(config);
        let samples = vec![BitrateSample::new(0.0, 1.0, 15_000.0)];
        let report = checker.analyze(&samples);
        assert!(!report.passed);
        assert!(report.findings.iter().any(|f| f.code == "BR-011"));
    }

    #[test]
    fn test_cbr_compliance_pass() {
        let config = BitrateQcConfig::new()
            .with_expected_mode(BitrateMode::Cbr)
            .with_target_kbps(5000.0);
        let checker = BitrateQcChecker::with_config(config);
        let samples = uniform_samples(5000.0, 20);
        let report = checker.analyze(&samples);
        assert!(report.passed);
    }

    #[test]
    fn test_cbr_mean_deviation() {
        let config = BitrateQcConfig::new()
            .with_expected_mode(BitrateMode::Cbr)
            .with_target_kbps(5000.0)
            .with_cbr_tolerance(0.05);
        let checker = BitrateQcChecker::with_config(config);
        // Mean is 6000, target is 5000 => 20% deviation
        let samples = uniform_samples(6000.0, 20);
        let report = checker.analyze(&samples);
        assert!(report.findings.iter().any(|f| f.code == "BR-021"));
    }

    #[test]
    fn test_abr_compliance_pass() {
        let config = BitrateQcConfig::new()
            .with_expected_mode(BitrateMode::Abr)
            .with_target_kbps(5000.0);
        let checker = BitrateQcChecker::with_config(config);
        let samples = uniform_samples(5000.0, 10);
        let report = checker.analyze(&samples);
        assert!(!report.findings.iter().any(|f| f.code == "BR-030"));
    }

    #[test]
    fn test_abr_deviation_error() {
        let config = BitrateQcConfig::new()
            .with_expected_mode(BitrateMode::Abr)
            .with_target_kbps(5000.0);
        let checker = BitrateQcChecker::with_config(config);
        // Mean will be 8000, target 5000 => 60% deviation
        let samples = uniform_samples(8000.0, 10);
        let report = checker.analyze(&samples);
        assert!(report.findings.iter().any(|f| f.code == "BR-030"));
    }

    #[test]
    fn test_scene_change_spike_allowed() {
        let config = BitrateQcConfig::new().with_max_kbps(10_000.0);
        let checker = BitrateQcChecker::with_config(config);
        let samples = vec![
            BitrateSample::new(0.0, 1.0, 5000.0),
            BitrateSample::new(1.0, 1.0, 15_000.0).with_scene_change(true),
        ];
        let report = checker.analyze(&samples);
        // The scene change spike should be ignored for bounds check
        assert!(!report.findings.iter().any(|f| f.code == "BR-011"));
    }

    #[test]
    fn test_bitrate_mode_display() {
        assert_eq!(BitrateMode::Cbr.to_string(), "CBR");
        assert_eq!(BitrateMode::Vbr.to_string(), "VBR");
        assert_eq!(BitrateMode::Abr.to_string(), "ABR");
        assert_eq!(BitrateMode::ConstantQuality.to_string(), "CQ");
    }

    #[test]
    fn test_sample_end_secs() {
        let sample = BitrateSample::new(10.0, 2.5, 5000.0);
        assert!((sample.end_secs() - 12.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(BitrateSeverity::Info.to_string(), "INFO");
        assert_eq!(BitrateSeverity::Warning.to_string(), "WARNING");
        assert_eq!(BitrateSeverity::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_finding_is_failure() {
        let info = BitrateFinding::new(BitrateSeverity::Info, "I", "info");
        let warn = BitrateFinding::new(BitrateSeverity::Warning, "W", "warn");
        let err = BitrateFinding::new(BitrateSeverity::Error, "E", "err");
        assert!(!info.is_failure());
        assert!(!warn.is_failure());
        assert!(err.is_failure());
    }

    #[test]
    fn test_report_error_count() {
        let mut report = BitrateQcReport::new();
        report.add_finding(BitrateFinding::new(BitrateSeverity::Info, "I", "info"));
        report.add_finding(BitrateFinding::new(BitrateSeverity::Error, "E1", "err1"));
        report.add_finding(BitrateFinding::new(BitrateSeverity::Error, "E2", "err2"));
        assert_eq!(report.errors().len(), 2);
        assert_eq!(report.finding_count(), 3);
        assert!(!report.passed);
    }

    #[test]
    fn test_config_builder() {
        let config = BitrateQcConfig::new()
            .with_expected_mode(BitrateMode::Vbr)
            .with_target_kbps(8000.0)
            .with_min_kbps(200.0)
            .with_max_kbps(50_000.0)
            .with_cbr_tolerance(0.15);
        assert_eq!(config.expected_mode, Some(BitrateMode::Vbr));
        assert!(
            (config.target_kbps.expect("should succeed in test") - 8000.0).abs() < f64::EPSILON
        );
        assert!((config.min_kbps - 200.0).abs() < f64::EPSILON);
        assert!((config.max_kbps - 50_000.0).abs() < f64::EPSILON);
        assert!((config.cbr_tolerance - 0.15).abs() < f64::EPSILON);
    }
}
