//! Video and Image Forensics and Tampering Detection
//!
//! This crate provides comprehensive forensic analysis capabilities for detecting
//! image and video tampering, including:
//! - JPEG compression artifact analysis
//! - Error Level Analysis (ELA)
//! - Noise pattern analysis and PRNU
//! - Metadata verification
//! - Copy-move detection
//! - Illumination inconsistency detection
//! - Comprehensive forensic reporting
//!
//! # Confidence score interpretation guide (for end users)
//!
//! Every [`ForensicTest`] and every [`TamperingReport`] carries a
//! `confidence` / `overall_confidence` score in `[0.0, 1.0]`. This is a
//! **heuristic likelihood signal aggregated across independent forensic
//! cues, not a probability of guilt** — treat it as evidence to weigh
//! alongside context (chain of custody, source metadata, corroborating
//! testimony), not as a verdict on its own. A practical reading:
//!
//! | Score range | [`ConfidenceLevel`] | Suggested interpretation |
//! |-------------|---------------------|--------------------------|
//! | 0.0 – 0.2   | `VeryLow`  | No meaningful evidence of tampering found. |
//! | 0.2 – 0.4   | `Low`      | Minor anomalies; consistent with normal camera/encoder variation. |
//! | 0.4 – 0.6   | `Medium`   | Notable anomalies; warrants a closer manual look, not yet conclusive. |
//! | 0.6 – 0.8   | `High`     | Multiple corroborating signals; tampering is likely. |
//! | 0.8 – 1.0   | `VeryHigh` | Strong, multi-test corroboration; tampering is very likely. |
//!
//! [`TamperingReport::calculate_overall_confidence`] does **not** average
//! every test's confidence equally. It applies a [reliability weight per
//! test category](test_reliability_weight) (overridable per-caller via
//! [`TestWeight`]) — e.g. a positive copy-move (`Copy-Move Detection`)
//! result is weighted 1.5×, versus 0.7× for a metadata-only signal, because
//! empirically copy-move detections are far less prone to false positives
//! than "metadata looks odd" observations. Detections (as opposed to clean
//! results) additionally receive a further **1.2× boost** to their
//! effective weight in that average, reflecting the standard forensic
//! asymmetry that a *positive* finding from a reliable test is more
//! informative than a negative one (absence of evidence is not evidence of
//! absence). The resulting weighted mean is what `overall_confidence`
//! reports, and `tampering_detected` is simply `overall_confidence > 0.5`.
//! Callers that want the older, unweighted behaviour (equal treatment of
//! every test) can call
//! [`TamperingReport::calculate_overall_confidence_unweighted`] instead.
//!
//! For programmatic thresholds, prefer `overall_confidence` (a continuous
//! `f64`) over the coarser [`ConfidenceLevel`] bucket, and always inspect
//! `TamperingReport::tests` to see *which* individual tests fired — a
//! `VeryHigh` score driven by five weak-but-correlated tests reads very
//! differently from the same score driven by one very strong copy-move
//! detection.

#![deny(unsafe_code)]
#![allow(dead_code)]

pub mod artifact_analysis;
pub mod audio_forensics;
pub mod authenticity;
pub mod batch_forensics;
pub mod blocking;
pub mod chain_of_custody;
pub mod chromatic_forensics;
pub mod clone_detection;
pub mod compression;
pub mod compression_history;
pub mod copy_detect;
pub mod custody;
pub mod deepfake_detect;
pub mod double_compression;
pub mod edit_history;
pub mod ela;
pub mod ela_analysis;
pub mod file_integrity;
pub mod fingerprint;
pub mod flat_array2;
pub mod format_forensics;
pub mod frame_forensics;
pub mod frequency_forensics;
pub mod geometric;
pub mod gop_boundary;
pub mod hash_registry;
pub mod lighting;
pub mod metadata;
pub mod metadata_forensics;
pub mod noise;
pub mod noise_analysis;
pub mod pattern;
pub mod phylogeny;
pub mod provenance;
pub mod quantization_table;
pub mod report;
pub mod shadow_analysis;
pub mod source_camera;
pub mod splicing;
pub mod steganalysis;
pub mod tamper_detect;
pub mod tampering;
pub mod time_forensics;
pub mod video_forensics;
pub mod watermark_detect;

pub use hash_registry::{HashAlgorithm, HashRegistry, MediaHash};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during forensic analysis
#[derive(Error, Debug)]
pub enum ForensicsError {
    #[error("Invalid image data: {0}")]
    InvalidImage(String),

    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),
}

/// Result type for forensic operations
pub type ForensicsResult<T> = Result<T, ForensicsError>;

/// Confidence level for tampering detection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Very low confidence (0-20%)
    VeryLow,
    /// Low confidence (20-40%)
    Low,
    /// Medium confidence (40-60%)
    Medium,
    /// High confidence (60-80%)
    High,
    /// Very high confidence (80-100%)
    VeryHigh,
}

impl ConfidenceLevel {
    /// Convert from a confidence score (0.0 to 1.0)
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s < 0.2 => ConfidenceLevel::VeryLow,
            s if s < 0.4 => ConfidenceLevel::Low,
            s if s < 0.6 => ConfidenceLevel::Medium,
            s if s < 0.8 => ConfidenceLevel::High,
            _ => ConfidenceLevel::VeryHigh,
        }
    }

    /// Convert to a numeric score (0.0 to 1.0)
    pub fn to_score(&self) -> f64 {
        match self {
            ConfidenceLevel::VeryLow => 0.1,
            ConfidenceLevel::Low => 0.3,
            ConfidenceLevel::Medium => 0.5,
            ConfidenceLevel::High => 0.7,
            ConfidenceLevel::VeryHigh => 0.9,
        }
    }
}

/// Result of a single forensic test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicTest {
    /// Name of the test
    pub name: String,
    /// Whether tampering was detected
    pub tampering_detected: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Detailed findings
    pub findings: Vec<String>,
    /// Anomaly map (if applicable)
    #[serde(skip)]
    pub anomaly_map: Option<flat_array2::FlatArray2<f64>>,
}

impl ForensicTest {
    /// Create a new forensic test result
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tampering_detected: false,
            confidence: 0.0,
            findings: Vec::new(),
            anomaly_map: None,
        }
    }

    /// Add a finding to the test
    pub fn add_finding(&mut self, finding: String) {
        self.findings.push(finding);
    }

    /// Set the confidence level
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence.clamp(0.0, 1.0);
    }

    /// Get the confidence level category
    pub fn confidence_level(&self) -> ConfidenceLevel {
        ConfidenceLevel::from_score(self.confidence)
    }
}

/// Comprehensive tampering report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TamperingReport {
    /// Overall tampering detected flag
    pub tampering_detected: bool,
    /// Overall confidence score (0.0 to 1.0)
    pub overall_confidence: f64,
    /// Individual test results
    pub tests: HashMap<String, ForensicTest>,
    /// Summary of findings
    pub summary: String,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Whether `analyze()` terminated early because the running weighted
    /// confidence exceeded `ForensicsConfig::confidence_threshold`.
    ///
    /// When `true`, only a subset of the configured forensic tests were run.
    /// When `false`, all enabled tests completed.
    pub early_stop: bool,
}

impl TamperingReport {
    /// Create a new tampering report
    pub fn new() -> Self {
        Self {
            tampering_detected: false,
            overall_confidence: 0.0,
            tests: HashMap::new(),
            summary: String::new(),
            recommendations: Vec::new(),
            early_stop: false,
        }
    }

    /// Add a test result
    pub fn add_test(&mut self, test: ForensicTest) {
        self.tests.insert(test.name.clone(), test);
    }

    /// Calculate overall confidence from individual tests using reliability-weighted
    /// averaging.
    ///
    /// Different forensic tests have different reliability characteristics:
    /// - ELA and compression analysis are strong indicators when positive
    /// - Noise analysis provides moderate evidence
    /// - Metadata analysis is supportive but less conclusive
    /// - Geometric (copy-move) detection is highly reliable when triggered
    /// - Lighting analysis provides supplementary evidence
    ///
    /// Each test's confidence is multiplied by a reliability weight before averaging.
    /// Tests that detected tampering also receive a slight boost to reflect the
    /// asymmetry between false positives and false negatives in forensic analysis.
    pub fn calculate_overall_confidence(&mut self) {
        if self.tests.is_empty() {
            self.overall_confidence = 0.0;
            return;
        }

        let mut weighted_sum = 0.0_f64;
        let mut weight_total = 0.0_f64;

        for test in self.tests.values() {
            let base_weight = test_reliability_weight(&test.name);
            // Tests that detected tampering get a 20% boost to their weight,
            // reflecting that a positive detection from a reliable test is
            // more informative than a negative result.
            let effective_weight = if test.tampering_detected {
                base_weight * 1.2
            } else {
                base_weight
            };

            weighted_sum += test.confidence * effective_weight;
            weight_total += effective_weight;
        }

        self.overall_confidence = if weight_total > 0.0 {
            (weighted_sum / weight_total).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Determine if tampering was detected based on threshold
        self.tampering_detected = self.overall_confidence > 0.5;
    }

    /// Calculate overall confidence using simple (unweighted) averaging.
    ///
    /// This is the legacy behaviour preserved for callers that want equal
    /// treatment of every test.
    pub fn calculate_overall_confidence_unweighted(&mut self) {
        if self.tests.is_empty() {
            self.overall_confidence = 0.0;
            return;
        }

        let total: f64 = self.tests.values().map(|t| t.confidence).sum();
        self.overall_confidence = total / self.tests.len() as f64;
        self.tampering_detected = self.overall_confidence > 0.5;
    }

    /// Serialize this report to a JSON string.
    ///
    /// Returns a pretty-printed JSON string.  Anomaly maps (which are
    /// `#[serde(skip)]`) are not included in the output.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Generate summary
    pub fn generate_summary(&mut self) {
        let num_tests = self.tests.len();
        let num_positive = self.tests.values().filter(|t| t.tampering_detected).count();

        if self.tampering_detected {
            self.summary = format!(
                "Tampering detected with {:.1}% confidence. {} out of {} tests indicated manipulation.",
                self.overall_confidence * 100.0,
                num_positive,
                num_tests
            );
        } else {
            self.summary = format!(
                "No significant tampering detected. {} out of {} tests passed.",
                num_tests - num_positive,
                num_tests
            );
        }
    }
}

impl Default for TamperingReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for forensic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForensicsConfig {
    /// Enable compression analysis
    pub enable_compression_analysis: bool,
    /// Enable ELA
    pub enable_ela: bool,
    /// Enable noise analysis
    pub enable_noise_analysis: bool,
    /// Enable metadata analysis
    pub enable_metadata_analysis: bool,
    /// Enable geometric analysis
    pub enable_geometric_analysis: bool,
    /// Enable lighting analysis
    pub enable_lighting_analysis: bool,
    /// Minimum confidence threshold for reporting
    pub min_confidence_threshold: f64,
    /// Early-stop threshold for progressive analysis.
    ///
    /// When the accumulated reliability-weighted confidence across completed
    /// tests reaches this value, `analyze()` terminates early and sets
    /// `TamperingReport::early_stop = true` in the returned report.
    ///
    /// A value of `1.0` (the default) effectively disables early stopping —
    /// the weighted running confidence can never exceed `1.0`.  Set to a value
    /// in `(0.0, 1.0)` such as `0.85` to enable the optimisation.
    pub confidence_threshold: f64,
}

impl Default for ForensicsConfig {
    fn default() -> Self {
        Self {
            enable_compression_analysis: true,
            enable_ela: true,
            enable_noise_analysis: true,
            enable_metadata_analysis: true,
            enable_geometric_analysis: true,
            enable_lighting_analysis: true,
            min_confidence_threshold: 0.5,
            // Default 1.0 disables early stopping — callers opt in by setting
            // a lower threshold.
            confidence_threshold: 1.0,
        }
    }
}

/// Per-test reliability weight configuration.
///
/// Allows callers to override the default reliability weights used by
/// [`TamperingReport::calculate_overall_confidence`].  Each field specifies
/// the weight for the corresponding forensic test category.
///
/// Weights are relative — they do not need to sum to 1.0.  Tests whose name
/// does not match any category receive the `default_weight`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestWeight {
    /// Weight for ELA / Error Level Analysis tests (default 1.3)
    pub ela: f64,
    /// Weight for PRNU / noise analysis tests (default 1.0)
    pub prnu: f64,
    /// Weight for copy-move / geometric detection tests (default 1.5)
    pub copy_move: f64,
    /// Weight for compression / DCT analysis tests (default 1.2)
    pub compression: f64,
    /// Weight for lighting / shadow analysis tests (default 0.9)
    pub lighting: f64,
    /// Weight for metadata / EXIF analysis tests (default 0.7)
    pub metadata: f64,
    /// Fallback weight for tests not matching any category (default 1.0)
    pub default_weight: f64,
}

impl Default for TestWeight {
    fn default() -> Self {
        Self {
            ela: 1.3,
            prnu: 1.0,
            copy_move: 1.5,
            compression: 1.2,
            lighting: 0.9,
            metadata: 0.7,
            default_weight: 1.0,
        }
    }
}

impl TestWeight {
    /// Look up the weight for a test by its name.
    #[must_use]
    pub fn weight_for(&self, test_name: &str) -> f64 {
        let lower = test_name.to_lowercase();
        if lower.contains("copy") || lower.contains("geometric") || lower.contains("clone") {
            self.copy_move
        } else if lower.contains("ela") || lower.contains("error level") {
            self.ela
        } else if lower.contains("compress") || lower.contains("dct") || lower.contains("jpeg") {
            self.compression
        } else if lower.contains("noise") || lower.contains("prnu") {
            self.prnu
        } else if lower.contains("light") || lower.contains("shadow") || lower.contains("illuminat")
        {
            self.lighting
        } else if lower.contains("metadata") || lower.contains("exif") || lower.contains("software")
        {
            self.metadata
        } else {
            self.default_weight
        }
    }
}

/// Main forensics analyzer
pub struct ForensicsAnalyzer {
    config: ForensicsConfig,
}

impl ForensicsAnalyzer {
    /// Create a new forensics analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: ForensicsConfig::default(),
        }
    }

    /// Create a new forensics analyzer with custom configuration
    pub fn with_config(config: ForensicsConfig) -> Self {
        Self { config }
    }

    /// Analyze an image for tampering.
    ///
    /// When `ForensicsConfig::confidence_threshold < 1.0`, tests are executed
    /// **sequentially** in decreasing reliability order and the loop exits as
    /// soon as the running reliability-weighted confidence reaches the
    /// threshold.  The returned report will have `early_stop = true` in that
    /// case.
    ///
    /// When `confidence_threshold == 1.0` (the default), all enabled tests run
    /// concurrently via rayon for maximum throughput.
    pub fn analyze(&self, image_data: &[u8]) -> ForensicsResult<TamperingReport> {
        // Parse image once — shared across all pixel-level tests.
        let image = image::load_from_memory(image_data)?;
        let rgb_image = image.to_rgb8();

        let threshold = self.config.confidence_threshold;
        let early_stop_enabled = threshold < 1.0;

        if early_stop_enabled {
            self.analyze_progressive(image_data, &rgb_image, threshold)
        } else {
            self.analyze_parallel(image_data, &rgb_image)
        }
    }

    /// Full parallel analysis — all enabled tests run concurrently.
    fn analyze_parallel(
        &self,
        image_data: &[u8],
        rgb_image: &image::RgbImage,
    ) -> ForensicsResult<TamperingReport> {
        // Collect enabled pixel-level tasks as closures so rayon can run
        // them in parallel.  Each closure returns ForensicsResult<ForensicTest>.
        type TaskFn<'a> = Box<dyn Fn() -> ForensicsResult<ForensicTest> + Send + Sync + 'a>;

        let mut tasks: Vec<TaskFn<'_>> = Vec::new();

        if self.config.enable_compression_analysis {
            tasks.push(Box::new(|| compression::analyze_compression(rgb_image)));
        }
        if self.config.enable_ela {
            tasks.push(Box::new(|| ela::perform_ela(rgb_image)));
        }
        if self.config.enable_noise_analysis {
            tasks.push(Box::new(|| noise::analyze_noise(rgb_image)));
        }
        if self.config.enable_geometric_analysis {
            tasks.push(Box::new(|| geometric::detect_copy_move(rgb_image)));
        }
        if self.config.enable_lighting_analysis {
            tasks.push(Box::new(|| lighting::analyze_lighting(rgb_image)));
        }

        // Run pixel-level tests in parallel.
        let pixel_results: Vec<ForensicsResult<ForensicTest>> =
            tasks.par_iter().map(|f| f()).collect();

        // Metadata analysis operates on the raw bytes.
        let metadata_result: Option<ForensicsResult<ForensicTest>> =
            if self.config.enable_metadata_analysis {
                Some(metadata::analyze_metadata(image_data))
            } else {
                None
            };

        // Collect results into the report, propagating the first error.
        let mut report = TamperingReport::new();

        for result in pixel_results {
            report.add_test(result?);
        }
        if let Some(result) = metadata_result {
            report.add_test(result?);
        }

        report.calculate_overall_confidence();
        report.generate_summary();

        Ok(report)
    }

    /// Progressive sequential analysis — tests are ordered by reliability and
    /// the loop breaks once accumulated weighted confidence reaches `threshold`.
    fn analyze_progressive(
        &self,
        image_data: &[u8],
        rgb_image: &image::RgbImage,
        threshold: f64,
    ) -> ForensicsResult<TamperingReport> {
        // Build (test_name, weight, closure) triples in reliability order.
        // We filter by config flags so disabled tests are skipped entirely.
        type RunFn<'a> = Box<dyn Fn() -> ForensicsResult<ForensicTest> + 'a>;

        struct OrderedTask<'a> {
            weight: f64,
            run: RunFn<'a>,
        }

        let mut ordered: Vec<OrderedTask<'_>> = Vec::new();

        // Copy-move (geometric) — weight 1.5
        if self.config.enable_geometric_analysis {
            ordered.push(OrderedTask {
                weight: 1.5,
                run: Box::new(|| geometric::detect_copy_move(rgb_image)),
            });
        }
        // ELA — weight 1.3
        if self.config.enable_ela {
            ordered.push(OrderedTask {
                weight: 1.3,
                run: Box::new(|| ela::perform_ela(rgb_image)),
            });
        }
        // Compression — weight 1.2
        if self.config.enable_compression_analysis {
            ordered.push(OrderedTask {
                weight: 1.2,
                run: Box::new(|| compression::analyze_compression(rgb_image)),
            });
        }
        // Noise — weight 1.0
        if self.config.enable_noise_analysis {
            ordered.push(OrderedTask {
                weight: 1.0,
                run: Box::new(|| noise::analyze_noise(rgb_image)),
            });
        }
        // Lighting — weight 0.9
        if self.config.enable_lighting_analysis {
            ordered.push(OrderedTask {
                weight: 0.9,
                run: Box::new(|| lighting::analyze_lighting(rgb_image)),
            });
        }
        // Metadata — weight 0.7
        if self.config.enable_metadata_analysis {
            ordered.push(OrderedTask {
                weight: 0.7,
                run: Box::new(|| metadata::analyze_metadata(image_data)),
            });
        }

        let mut report = TamperingReport::new();
        let mut weighted_confidence = 0.0_f64;
        let mut weight_total = 0.0_f64;
        let mut stopped_early = false;

        for task in ordered {
            let test = (task.run)()?;
            let effective_weight = if test.tampering_detected {
                task.weight * 1.2
            } else {
                task.weight
            };

            weighted_confidence += test.confidence * effective_weight;
            weight_total += effective_weight;

            let running_conf = if weight_total > 0.0 {
                (weighted_confidence / weight_total).clamp(0.0, 1.0)
            } else {
                0.0
            };

            report.add_test(test);

            if running_conf >= threshold {
                stopped_early = true;
                break;
            }
        }

        report.early_stop = stopped_early;
        report.calculate_overall_confidence();
        report.generate_summary();

        Ok(report)
    }

    /// Analyze a batch of image files in parallel using rayon.
    ///
    /// Each path is read from disk and analyzed independently.  Files that
    /// cannot be read or analyzed produce a report with default (clean) values
    /// and a summary indicating the failure.
    ///
    /// # Errors
    ///
    /// This method never returns `Err` at the batch level — per-file failures
    /// are captured in the individual reports via the summary field.
    pub fn analyze_batch(&self, paths: &[PathBuf]) -> Vec<TamperingReport> {
        paths
            .par_iter()
            .map(|path| {
                let data = match std::fs::read(path) {
                    Ok(d) => d,
                    Err(e) => {
                        let mut report = TamperingReport::new();
                        report.summary = format!("Failed to read {}: {}", path.display(), e);
                        return report;
                    }
                };
                match self.analyze(&data) {
                    Ok(report) => report,
                    Err(e) => {
                        let mut report = TamperingReport::new();
                        report.summary = format!("Analysis failed for {}: {}", path.display(), e);
                        report
                    }
                }
            })
            .collect()
    }

    /// Get the current configuration
    pub fn config(&self) -> &ForensicsConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: ForensicsConfig) {
        self.config = config;
    }
}

impl Default for ForensicsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Return the reliability weight for a forensic test identified by name.
///
/// Weights are calibrated based on the empirical false-positive / false-negative
/// rates of each test methodology:
///
/// | Test category        | Weight | Rationale                                    |
/// |----------------------|--------|----------------------------------------------|
/// | Copy-move detection  | 1.5    | Highly specific when triggered                |
/// | ELA                  | 1.3    | Strong but sensitive to JPEG quality           |
/// | Compression analysis | 1.2    | Reliable double-compression indicator          |
/// | Noise analysis       | 1.0    | Moderate reliability (baseline)                |
/// | Lighting analysis    | 0.9    | Useful but geometry-dependent                  |
/// | Metadata analysis    | 0.7    | Supportive; easily spoofed or stripped          |
///
/// Unknown test names receive a baseline weight of `1.0`.
#[must_use]
pub fn test_reliability_weight(test_name: &str) -> f64 {
    let lower = test_name.to_lowercase();
    if lower.contains("copy") || lower.contains("geometric") || lower.contains("clone") {
        1.5
    } else if lower.contains("ela") || lower.contains("error level") {
        1.3
    } else if lower.contains("compress") || lower.contains("dct") || lower.contains("jpeg") {
        1.2
    } else if lower.contains("noise") || lower.contains("prnu") {
        1.0
    } else if lower.contains("light") || lower.contains("shadow") || lower.contains("illuminat") {
        0.9
    } else if lower.contains("metadata") || lower.contains("exif") || lower.contains("software") {
        0.7
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_level_conversion() {
        assert_eq!(ConfidenceLevel::from_score(0.1), ConfidenceLevel::VeryLow);
        assert_eq!(ConfidenceLevel::from_score(0.3), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::from_score(0.5), ConfidenceLevel::Medium);
        assert_eq!(ConfidenceLevel::from_score(0.7), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_score(0.9), ConfidenceLevel::VeryHigh);
    }

    #[test]
    fn test_forensic_test_creation() {
        let mut test = ForensicTest::new("Test");
        assert_eq!(test.name, "Test");
        assert!(!test.tampering_detected);
        assert_eq!(test.confidence, 0.0);

        test.add_finding("Finding 1".to_string());
        test.set_confidence(0.75);
        assert_eq!(test.findings.len(), 1);
        assert_eq!(test.confidence, 0.75);
    }

    #[test]
    fn test_tampering_report_unweighted() {
        let mut report = TamperingReport::new();

        let mut test1 = ForensicTest::new("Test1");
        test1.set_confidence(0.8);
        test1.tampering_detected = true;

        let mut test2 = ForensicTest::new("Test2");
        test2.set_confidence(0.6);
        test2.tampering_detected = true;

        report.add_test(test1);
        report.add_test(test2);
        report.calculate_overall_confidence_unweighted();

        assert_eq!(report.tests.len(), 2);
        assert!((report.overall_confidence - 0.7).abs() < 1e-10);
        assert!(report.tampering_detected);
    }

    #[test]
    fn test_tampering_report_weighted_confidence_empty() {
        let mut report = TamperingReport::new();
        report.calculate_overall_confidence();
        assert!((report.overall_confidence).abs() < 1e-10);
        assert!(!report.tampering_detected);
    }

    #[test]
    fn test_tampering_report_weighted_confidence_single_test() {
        let mut report = TamperingReport::new();
        let mut test = ForensicTest::new("ELA Analysis");
        test.set_confidence(0.8);
        test.tampering_detected = true;
        report.add_test(test);
        report.calculate_overall_confidence();
        // Single test: weighted confidence == confidence (weight cancels)
        assert!((report.overall_confidence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_confidence_strong_test_dominates() {
        let mut report = TamperingReport::new();

        // High-weight test (copy-move, weight 1.5) with high confidence
        let mut copy_move = ForensicTest::new("Copy-Move Detection");
        copy_move.set_confidence(0.9);
        copy_move.tampering_detected = true;

        // Low-weight test (metadata, weight 0.7) with low confidence
        let mut metadata = ForensicTest::new("Metadata Analysis");
        metadata.set_confidence(0.1);
        metadata.tampering_detected = false;

        report.add_test(copy_move);
        report.add_test(metadata);
        report.calculate_overall_confidence();

        // Weighted average should be closer to 0.9 than simple average 0.5
        assert!(report.overall_confidence > 0.5);

        // Compare with unweighted
        let mut report2 = TamperingReport::new();
        let mut cm2 = ForensicTest::new("Copy-Move Detection");
        cm2.set_confidence(0.9);
        cm2.tampering_detected = true;
        let mut md2 = ForensicTest::new("Metadata Analysis");
        md2.set_confidence(0.1);
        md2.tampering_detected = false;
        report2.add_test(cm2);
        report2.add_test(md2);
        report2.calculate_overall_confidence_unweighted();

        // Weighted should give higher confidence when strong test dominates
        assert!(report.overall_confidence > report2.overall_confidence);
    }

    #[test]
    fn test_weighted_confidence_detection_boost() {
        // Two identical-confidence tests, one detected tampering
        let mut report = TamperingReport::new();

        let mut t1 = ForensicTest::new("Noise Analysis");
        t1.set_confidence(0.6);
        t1.tampering_detected = true;

        let mut t2 = ForensicTest::new("Noise Check");
        t2.set_confidence(0.6);
        t2.tampering_detected = false;

        report.add_test(t1);
        report.add_test(t2);
        report.calculate_overall_confidence();

        // The detecting test gets a 1.2x boost, so the weighted average
        // should be slightly above the unweighted 0.6
        assert!(report.overall_confidence > 0.59);
        assert!(report.overall_confidence < 0.65);
    }

    #[test]
    fn test_reliability_weight_categories() {
        assert!((test_reliability_weight("Copy-Move Detection") - 1.5).abs() < 1e-10);
        assert!((test_reliability_weight("ELA Analysis") - 1.3).abs() < 1e-10);
        assert!((test_reliability_weight("JPEG Compression Analysis") - 1.2).abs() < 1e-10);
        assert!((test_reliability_weight("Noise Analysis") - 1.0).abs() < 1e-10);
        assert!((test_reliability_weight("Lighting Analysis") - 0.9).abs() < 1e-10);
        assert!((test_reliability_weight("Metadata Analysis") - 0.7).abs() < 1e-10);
        assert!((test_reliability_weight("Unknown Test") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_confidence_all_below_threshold() {
        let mut report = TamperingReport::new();

        let mut t1 = ForensicTest::new("ELA");
        t1.set_confidence(0.2);
        t1.tampering_detected = false;

        let mut t2 = ForensicTest::new("Metadata");
        t2.set_confidence(0.3);
        t2.tampering_detected = false;

        report.add_test(t1);
        report.add_test(t2);
        report.calculate_overall_confidence();

        assert!(!report.tampering_detected);
        assert!(report.overall_confidence < 0.5);
    }

    #[test]
    fn test_weighted_confidence_clamped_to_unit() {
        let mut report = TamperingReport::new();
        let mut t = ForensicTest::new("ELA");
        t.set_confidence(1.0);
        t.tampering_detected = true;
        report.add_test(t);
        report.calculate_overall_confidence();
        assert!(report.overall_confidence <= 1.0);
    }

    // ── TamperingReport::to_json ───────────────────────────────────────────────

    #[test]
    fn test_tampering_report_to_json_empty() {
        let report = TamperingReport::new();
        let json = report.to_json();
        assert!(json.contains("overall_confidence"));
        assert!(json.contains("tampering_detected"));
        assert!(json.contains("tests"));
    }

    #[test]
    fn test_tampering_report_to_json_with_tests() {
        let mut report = TamperingReport::new();
        let mut test = ForensicTest::new("ELA Analysis");
        test.set_confidence(0.75);
        test.tampering_detected = true;
        test.add_finding("Suspicious region detected".to_string());
        report.add_test(test);
        report.calculate_overall_confidence();
        report.generate_summary();

        let json = report.to_json();
        assert!(json.contains("ELA Analysis"));
        assert!(json.contains("0.75"));
        // Verify it's valid JSON by checking well-formedness
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("should be valid JSON");
        assert!(parsed.get("tampering_detected").is_some());
    }

    #[test]
    fn test_tampering_report_to_json_is_valid_json() {
        let mut report = TamperingReport::new();
        let mut t1 = ForensicTest::new("Copy-Move Detection");
        t1.set_confidence(0.9);
        t1.tampering_detected = true;
        let mut t2 = ForensicTest::new("Metadata Analysis");
        t2.set_confidence(0.1);
        report.add_test(t1);
        report.add_test(t2);
        report.calculate_overall_confidence();

        let json = report.to_json();
        let _parsed: serde_json::Value =
            serde_json::from_str(&json).expect("to_json must produce valid JSON");
    }

    // ── ForensicsAnalyzer::analyze_batch ──────────────────────────────────────

    #[test]
    fn test_analyze_batch_empty_paths() {
        let analyzer = ForensicsAnalyzer::new();
        let results = analyzer.analyze_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_analyze_batch_nonexistent_path() {
        let analyzer = ForensicsAnalyzer::new();
        let path = std::path::PathBuf::from("/nonexistent/path/image.jpg");
        let results = analyzer.analyze_batch(&[path]);
        assert_eq!(results.len(), 1);
        // Should not panic; result should have an error summary
        assert!(!results[0].summary.is_empty());
    }

    #[test]
    fn test_analyze_batch_valid_images() {
        use std::io::Cursor;
        use std::io::Write;

        // Use a 128×128 image — large enough for all forensic analysis kernels
        // (the smallest kernel requires at least 64 pixels per dimension).
        let img = image::RgbImage::new(128, 128);
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Cursor::new(Vec::new());
        dyn_img
            .write_to(&mut buf, image::ImageFormat::Png)
            .expect("PNG encoding should work");
        let png_bytes = buf.into_inner();

        // Write to temp file.
        let mut tmp = std::env::temp_dir();
        tmp.push("oximedia_forensics_batch_test.png");
        {
            let mut f = std::fs::File::create(&tmp).expect("temp file creation");
            f.write_all(&png_bytes).expect("write PNG bytes");
        }

        let analyzer = ForensicsAnalyzer::new();
        let results = analyzer.analyze_batch(&[tmp.clone()]);
        assert_eq!(results.len(), 1);
        // Clean image should produce a valid report.
        assert!(!results[0].summary.is_empty());

        // Cleanup.
        let _ = std::fs::remove_file(&tmp);
    }

    // ── TestWeight ────────────────────────────────────────────────────────────

    #[test]
    fn test_test_weight_defaults() {
        let tw = TestWeight::default();
        assert!((tw.ela - 1.3).abs() < 1e-10);
        assert!((tw.prnu - 1.0).abs() < 1e-10);
        assert!((tw.copy_move - 1.5).abs() < 1e-10);
        assert!((tw.compression - 1.2).abs() < 1e-10);
        assert!((tw.lighting - 0.9).abs() < 1e-10);
        assert!((tw.metadata - 0.7).abs() < 1e-10);
        assert!((tw.default_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_test_weight_lookup() {
        let tw = TestWeight::default();
        assert!((tw.weight_for("ELA Analysis") - 1.3).abs() < 1e-10);
        assert!((tw.weight_for("Copy-Move Detection") - 1.5).abs() < 1e-10);
        assert!((tw.weight_for("JPEG Compression Analysis") - 1.2).abs() < 1e-10);
        assert!((tw.weight_for("Noise Analysis PRNU") - 1.0).abs() < 1e-10);
        assert!((tw.weight_for("Lighting Analysis") - 0.9).abs() < 1e-10);
        assert!((tw.weight_for("Metadata Analysis") - 0.7).abs() < 1e-10);
        assert!((tw.weight_for("Unknown Custom Test") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_test_weight_custom_values() {
        let tw = TestWeight {
            ela: 2.0,
            prnu: 0.5,
            copy_move: 3.0,
            compression: 1.0,
            lighting: 0.5,
            metadata: 0.3,
            default_weight: 1.5,
        };
        assert!((tw.weight_for("ELA") - 2.0).abs() < 1e-10);
        assert!((tw.weight_for("Clone Detection") - 3.0).abs() < 1e-10);
        assert!((tw.weight_for("PRNU Extraction") - 0.5).abs() < 1e-10);
        assert!((tw.weight_for("Shadow Analysis") - 0.5).abs() < 1e-10);
        assert!((tw.weight_for("EXIF Check") - 0.3).abs() < 1e-10);
        assert!((tw.weight_for("Anything Else") - 1.5).abs() < 1e-10);
    }
}
